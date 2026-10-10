#include "q8q8_dispatch.h"

#include <limits.h>
#include <string.h>

enum {
    Q8Q8_QK = 32,
    Q8Q8_TM = 16,
    Q8Q8_MAX_M_PER_JOB = 4320,
    Q8Q8_MAX_N = 9728,
    Q8Q8_MAX_K = 9728,
};

_Static_assert(sizeof(struct q8q8_block_q8_0) == 34,
               "GGML Q8_0 block must be exactly 34 bytes");

static int is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static int checked_add(size_t left, size_t right, size_t *result) {
    if (left > SIZE_MAX - right) return Q8Q8_RT_ERR_ARGUMENT;
    *result = left + right;
    return Q8Q8_RT_OK;
}

static int checked_mul(size_t left, size_t right, size_t *result) {
    if (left != 0 && right > SIZE_MAX / left) return Q8Q8_RT_ERR_ARGUMENT;
    *result = left * right;
    return Q8Q8_RT_OK;
}

static int append_aligned(size_t *used, size_t bytes, size_t alignment) {
    const size_t mask = alignment - 1;
    size_t aligned;
    if (*used > SIZE_MAX - mask) return Q8Q8_RT_ERR_ARGUMENT;
    aligned = (*used + mask) & ~mask;
    return checked_add(aligned, bytes, used);
}

static int validate_workload(const struct q8q8_workload_descriptor *workload) {
    if (workload == NULL || workload->operation != Q8Q8_OP_MATMUL_NT ||
            workload->weight_layout != Q8Q8_LAYOUT_GGML_Q8_0 ||
            workload->activation_layout != Q8Q8_LAYOUT_GGML_Q8_0 ||
            workload->output_layout != Q8Q8_LAYOUT_ROW_MAJOR_F32 ||
            workload->m == 0 || workload->n == 0 || workload->k == 0 ||
            workload->n > Q8Q8_MAX_N || workload->k > Q8Q8_MAX_K ||
            workload->n % 4u != 0 || workload->k % Q8Q8_QK != 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    return Q8Q8_RT_OK;
}

static int calculate_sizes(uint32_t m, uint32_t n, uint32_t k,
                           size_t alignment,
                           struct q8q8_dispatch_plan *sizes) {
    size_t weight_blocks;
    size_t activation_blocks;
    size_t used = 0;
    if (checked_mul(n, k / Q8Q8_QK, &weight_blocks) != Q8Q8_RT_OK ||
            checked_mul(m, k / Q8Q8_QK, &activation_blocks) != Q8Q8_RT_OK ||
            checked_mul(weight_blocks, Q8Q8_QK,
                        &sizes->weights_q_bytes) != Q8Q8_RT_OK ||
            checked_mul(weight_blocks, sizeof(uint16_t),
                        &sizes->weights_scale_bytes) != Q8Q8_RT_OK ||
            checked_mul(activation_blocks, Q8Q8_QK,
                        &sizes->activations_q_bytes) != Q8Q8_RT_OK ||
            checked_mul(activation_blocks, sizeof(uint16_t),
                        &sizes->activations_scale_bytes) != Q8Q8_RT_OK ||
            checked_mul(m, n, &sizes->output_bytes) != Q8Q8_RT_OK ||
            checked_mul(sizes->output_bytes, sizeof(float),
                        &sizes->output_bytes) != Q8Q8_RT_OK ||
            append_aligned(&used, sizes->weights_q_bytes, alignment) != Q8Q8_RT_OK ||
            append_aligned(&used, sizes->weights_scale_bytes, alignment) != Q8Q8_RT_OK ||
            append_aligned(&used, sizes->activations_q_bytes, alignment) != Q8Q8_RT_OK ||
            append_aligned(&used, sizes->activations_scale_bytes, alignment) != Q8Q8_RT_OK ||
            append_aligned(&used, sizes->output_bytes, alignment) != Q8Q8_RT_OK) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    sizes->arena_bytes = used;
    return Q8Q8_RT_OK;
}

int q8q8_plan_workload(const struct q8q8_workload_descriptor *workload,
                       size_t arena_capacity, size_t alignment,
                       struct q8q8_dispatch_plan *plan) {
    struct q8q8_dispatch_plan candidate;
    uint32_t low = 1;
    uint32_t high;
    uint32_t best = 0;
    if (plan == NULL || validate_workload(workload) != Q8Q8_RT_OK ||
            !is_power_of_two(alignment)) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    high = workload->m < Q8Q8_MAX_M_PER_JOB
            ? workload->m : Q8Q8_MAX_M_PER_JOB;
    while (low <= high) {
        const uint32_t middle = low + (high - low) / 2;
        memset(&candidate, 0, sizeof(candidate));
        if (calculate_sizes(middle, workload->n, workload->k, alignment,
                            &candidate) == Q8Q8_RT_OK &&
                candidate.arena_bytes <= arena_capacity) {
            best = middle;
            low = middle + 1;
        } else {
            high = middle - 1;
        }
    }
    if (best == 0) return Q8Q8_RT_ERR_ARGUMENT;
    if (best >= Q8Q8_TM && best < workload->m) {
        best -= best % Q8Q8_TM;
    }
    memset(plan, 0, sizeof(*plan));
    plan->workload = *workload;
    plan->chunk_m = best;
    plan->chunk_count = (workload->m + best - 1u) / best;
    plan->alignment = alignment;
    if (calculate_sizes(best, workload->n, workload->k, alignment, plan) !=
            Q8Q8_RT_OK || plan->arena_bytes > arena_capacity) {
        memset(plan, 0, sizeof(*plan));
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    return Q8Q8_RT_OK;
}

int q8q8_allocate_plan(struct q8q8_arena *arena,
                       const struct q8q8_dispatch_plan *plan,
                       struct q8q8_dispatch_buffers *buffers) {
    if (arena == NULL || plan == NULL || buffers == NULL ||
            plan->chunk_m == 0 || plan->alignment == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    q8q8_arena_reset(arena);
    if (q8q8_arena_alloc(arena, plan->weights_q_bytes, plan->alignment,
                         &buffers->weights_q) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, plan->weights_scale_bytes, plan->alignment,
                             &buffers->weights_scale) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, plan->activations_q_bytes, plan->alignment,
                             &buffers->activations_q) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, plan->activations_scale_bytes,
                             plan->alignment,
                             &buffers->activations_scale) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, plan->output_bytes, plan->alignment,
                             &buffers->output) != Q8Q8_RT_OK) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    return Q8Q8_RT_OK;
}

int q8q8_pack_ggml_q8_0(const struct q8q8_block_q8_0 *source,
                        uint32_t rows, uint32_t k,
                        const struct q8q8_buffer *quantized,
                        const struct q8q8_buffer *scales) {
    size_t block_count;
    size_t block;
    if (source == NULL || quantized == NULL || scales == NULL || rows == 0 ||
            k == 0 || k % Q8Q8_QK != 0 ||
            checked_mul(rows, k / Q8Q8_QK, &block_count) != Q8Q8_RT_OK ||
            quantized->size < block_count * Q8Q8_QK ||
            scales->size < block_count * sizeof(uint16_t)) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    for (block = 0; block < block_count; ++block) {
        if (q8q8_buffer_store(quantized, block * Q8Q8_QK,
                              source[block].values, Q8Q8_QK) != Q8Q8_RT_OK ||
                q8q8_buffer_store(scales, block * sizeof(uint16_t),
                                  &source[block].scale_fp16,
                                  sizeof(uint16_t)) != Q8Q8_RT_OK) {
            return Q8Q8_RT_ERR_ARGUMENT;
        }
    }
    return Q8Q8_RT_OK;
}

int q8q8_plan_job(const struct q8q8_dispatch_plan *plan,
                  const struct q8q8_dispatch_buffers *buffers,
                  uint32_t chunk_index, struct q8q8_job *job,
                  uint32_t *row_offset) {
    uint64_t first_row;
    uint32_t rows;
    if (plan == NULL || buffers == NULL || job == NULL || row_offset == NULL ||
            plan->chunk_m == 0 || chunk_index >= plan->chunk_count) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    first_row = (uint64_t)chunk_index * plan->chunk_m;
    rows = plan->workload.m - (uint32_t)first_row;
    if (rows > plan->chunk_m) rows = plan->chunk_m;
    memset(job, 0, sizeof(*job));
    job->weights_q = buffers->weights_q.phys;
    job->weights_scale = buffers->weights_scale.phys;
    job->activations_q = buffers->activations_q.phys;
    job->activations_scale = buffers->activations_scale.phys;
    job->output = buffers->output.phys;
    job->m = rows;
    job->n = plan->workload.n;
    job->k = plan->workload.k;
    *row_offset = (uint32_t)first_row;
    return q8q8_validate_job(job);
}

int q8q8_execute_plan(struct q8q8_device *device,
                      const struct q8q8_dispatch_plan *plan,
                      const struct q8q8_dispatch_buffers *buffers,
                      const struct q8q8_block_q8_0 *activations,
                      float *output, uint64_t timeout_us_per_chunk,
                      struct q8q8_dispatch_stats *stats) {
    const size_t blocks_per_row = plan != NULL
            ? plan->workload.k / Q8Q8_QK : 0;
    uint32_t chunk;
    struct q8q8_dispatch_stats completed = {0, 0, 0};
    if (device == NULL || plan == NULL || buffers == NULL ||
            activations == NULL || output == NULL || timeout_us_per_chunk == 0 ||
            plan->chunk_m == 0 || plan->chunk_count == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    for (chunk = 0; chunk < plan->chunk_count; ++chunk) {
        struct q8q8_job job;
        struct q8q8_result result;
        uint32_t row_offset;
        size_t output_values;
        int status = q8q8_plan_job(plan, buffers, chunk, &job, &row_offset);
        if (status != Q8Q8_RT_OK) return status;
        status = q8q8_pack_ggml_q8_0(
                activations + (size_t)row_offset * blocks_per_row,
                job.m, job.k, &buffers->activations_q,
                &buffers->activations_scale);
        if (status != Q8Q8_RT_OK) return status;
        status = q8q8_run(device, &job, timeout_us_per_chunk, &result);
        if (status != Q8Q8_RT_OK) return status;
        if (checked_mul(job.m, job.n, &output_values) != Q8Q8_RT_OK ||
                checked_mul(output_values, sizeof(float), &output_values) !=
                        Q8Q8_RT_OK ||
                q8q8_buffer_load(output + (size_t)row_offset * job.n,
                                 &buffers->output, 0, output_values) !=
                        Q8Q8_RT_OK) {
            return Q8Q8_RT_ERR_ARGUMENT;
        }
        ++completed.chunks_completed;
        completed.rows_completed += job.m;
        completed.hardware_elapsed_us += result.elapsed_us;
    }
    if (stats != NULL) *stats = completed;
    return Q8Q8_RT_OK;
}
