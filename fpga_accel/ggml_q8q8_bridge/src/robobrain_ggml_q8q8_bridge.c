#include "robobrain_ggml_q8q8_bridge.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#include <float.h>
#endif

enum {
    RB_QK8_0 = 32,
    RB_Q8_0_BYTES = 34,
};

static int dimensions_fit_u32(const struct rb_ggml_mul_mat_view *operation) {
    return operation->src0_weights.ne[0] <= UINT32_MAX &&
           operation->src0_weights.ne[1] <= UINT32_MAX &&
           operation->src1_activations.ne[1] <= UINT32_MAX;
}

static int is_2d(const struct rb_ggml_tensor_view *tensor) {
    return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static int is_contiguous_q8_0(const struct rb_ggml_tensor_view *tensor) {
    size_t row;
    if (tensor->ne[0] == 0 || tensor->ne[0] % RB_QK8_0 != 0 ||
            tensor->ne[0] > SIZE_MAX / RB_Q8_0_BYTES) {
        return 0;
    }
    row = (size_t)(tensor->ne[0] / RB_QK8_0) * RB_Q8_0_BYTES;
    return tensor->nb[0] == RB_Q8_0_BYTES && tensor->nb[1] == row &&
           tensor->ne[1] <= SIZE_MAX / row &&
           tensor->nb[2] == row * (size_t)tensor->ne[1] &&
           tensor->nb[3] == tensor->nb[2];
}

static int is_contiguous_f32(const struct rb_ggml_tensor_view *tensor) {
    size_t row;
    if (tensor->ne[0] == 0 || tensor->ne[0] > SIZE_MAX / sizeof(float)) {
        return 0;
    }
    row = (size_t)tensor->ne[0] * sizeof(float);
    return tensor->nb[0] == sizeof(float) && tensor->nb[1] == row &&
           tensor->ne[1] <= SIZE_MAX / row &&
           tensor->nb[2] == row * (size_t)tensor->ne[1] &&
           tensor->nb[3] == tensor->nb[2];
}

static void reject(struct rb_ggml_q8q8_report *report,
                   enum rb_ggml_q8q8_reason reason) {
    memset(report, 0, sizeof(*report));
    report->reason = reason;
}

int rb_ggml_q8q8_classify(const struct rb_ggml_mul_mat_view *operation,
                          size_t arena_capacity, size_t alignment,
                          struct rb_ggml_q8q8_report *report) {
    struct q8q8_workload_descriptor workload;
    int status;
    if (report == NULL) return -1;
    if (operation == NULL) {
        reject(report, RB_GGML_Q8Q8_NULL_ARGUMENT);
        return -1;
    }
    if (operation->src0_weights.data == NULL ||
            operation->src1_activations.data == NULL ||
            operation->dst_f32.data == NULL) {
        reject(report, RB_GGML_Q8Q8_NULL_DATA);
        return 0;
    }
    if (operation->src0_weights.type != RB_GGML_TYPE_Q8_0) {
        reject(report, RB_GGML_Q8Q8_WEIGHT_TYPE);
        return 0;
    }
    if (operation->src1_activations.type != RB_GGML_TYPE_Q8_0 &&
            operation->src1_activations.type != RB_GGML_TYPE_F32) {
        reject(report, RB_GGML_Q8Q8_ACTIVATION_TYPE);
        return 0;
    }
    if (operation->dst_f32.type != RB_GGML_TYPE_F32) {
        reject(report, RB_GGML_Q8Q8_OUTPUT_TYPE);
        return 0;
    }
    if (!is_2d(&operation->src0_weights) ||
            !is_2d(&operation->src1_activations) ||
            !is_2d(&operation->dst_f32)) {
        reject(report, RB_GGML_Q8Q8_BATCH_OR_BROADCAST);
        return 0;
    }
    if (!dimensions_fit_u32(operation) ||
            operation->src0_weights.ne[0] == 0 ||
            operation->src0_weights.ne[1] == 0 ||
            operation->src1_activations.ne[1] == 0 ||
            operation->src1_activations.ne[0] !=
                    operation->src0_weights.ne[0] ||
            operation->dst_f32.ne[0] != operation->src0_weights.ne[1] ||
            operation->dst_f32.ne[1] !=
                    operation->src1_activations.ne[1]) {
        reject(report, RB_GGML_Q8Q8_SHAPE_MISMATCH);
        return 0;
    }
    if (!is_contiguous_q8_0(&operation->src0_weights)) {
        reject(report, RB_GGML_Q8Q8_WEIGHT_LAYOUT);
        return 0;
    }
    if ((operation->src1_activations.type == RB_GGML_TYPE_Q8_0 &&
             !is_contiguous_q8_0(&operation->src1_activations)) ||
            (operation->src1_activations.type == RB_GGML_TYPE_F32 &&
             !is_contiguous_f32(&operation->src1_activations))) {
        reject(report, RB_GGML_Q8Q8_ACTIVATION_LAYOUT);
        return 0;
    }
    if (!is_contiguous_f32(&operation->dst_f32)) {
        reject(report, RB_GGML_Q8Q8_OUTPUT_LAYOUT);
        return 0;
    }
    memset(report, 0, sizeof(*report));
    report->m = (uint32_t)operation->src1_activations.ne[1];
    report->n = (uint32_t)operation->src0_weights.ne[1];
    report->k = (uint32_t)operation->src0_weights.ne[0];
    report->activation_quantization_required =
            operation->src1_activations.type == RB_GGML_TYPE_F32;
    workload.operation = Q8Q8_OP_MATMUL_NT;
    workload.weight_layout = Q8Q8_LAYOUT_GGML_Q8_0;
    workload.activation_layout = Q8Q8_LAYOUT_GGML_Q8_0;
    workload.output_layout = Q8Q8_LAYOUT_ROW_MAJOR_F32;
    workload.m = report->m;
    workload.n = report->n;
    workload.k = report->k;
    status = q8q8_plan_workload(&workload, arena_capacity, alignment,
                                &report->plan);
    if (status != Q8Q8_RT_OK) {
        report->reason = (report->n > 9728 || report->k > 9728 ||
                          report->n % 4 != 0 || report->k % 32 != 0)
                ? RB_GGML_Q8Q8_HARDWARE_LIMIT
                : RB_GGML_Q8Q8_ARENA_CAPACITY;
        return 0;
    }
    report->reason = RB_GGML_Q8Q8_ELIGIBLE;
    return 1;
}

/* IEEE binary32 to binary16, round-to-nearest-even. */
static uint16_t fp32_to_fp16(float value) {
    uint32_t bits;
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000u;
    exponent = (bits >> 23) & 0xffu;
    mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    if (exponent > 142u) return (uint16_t)(sign | 0x7c00u);
    if (exponent < 113u) {
        uint32_t shift;
        uint32_t rounded;
        if (exponent < 103u) return (uint16_t)sign;
        mantissa |= 0x800000u;
        shift = 125u - exponent;
        rounded = mantissa >> shift;
        if ((mantissa & ((1u << shift) - 1u)) > (1u << (shift - 1u)) ||
                ((mantissa & ((1u << shift) - 1u)) ==
                 (1u << (shift - 1u)) && (rounded & 1u))) {
            ++rounded;
        }
        return (uint16_t)(sign | rounded);
    }
    exponent -= 112u;
    mantissa += 0xfffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31u) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | (exponent << 10) | (mantissa >> 13));
}

int rb_ggml_quantize_f32_q8_0(const float *source, uint32_t rows, uint32_t k,
                              size_t source_row_stride,
                              struct q8q8_block_q8_0 *destination,
                              size_t destination_bytes) {
    size_t blocks_per_row;
    size_t required;
    uint32_t row;
    if (source == NULL || destination == NULL || rows == 0 || k == 0 ||
            k % RB_QK8_0 != 0 ||
            source_row_stride < (size_t)k * sizeof(float)) return -1;
    blocks_per_row = k / RB_QK8_0;
    if (rows > SIZE_MAX / blocks_per_row ||
            (size_t)rows * blocks_per_row >
                    SIZE_MAX / sizeof(*destination)) return -1;
    required = (size_t)rows * blocks_per_row * sizeof(*destination);
    if (destination_bytes < required) return -1;
    for (row = 0; row < rows; ++row) {
        const float *row_data = (const float *)((const uint8_t *)source +
                                                (size_t)row * source_row_stride);
        size_t block;
        for (block = 0; block < blocks_per_row; ++block) {
            struct q8q8_block_q8_0 *out =
                    &destination[(size_t)row * blocks_per_row + block];
            float amax = 0.0f;
            float scale;
            float inverse;
            size_t lane;
            for (lane = 0; lane < RB_QK8_0; ++lane) {
                const float value = row_data[block * RB_QK8_0 + lane];
                float magnitude;
                if (!isfinite(value)) return -2;
                magnitude = fabsf(value);
                if (magnitude > amax) amax = magnitude;
            }
            scale = amax / 127.0f;
            inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
            out->scale_fp16 = fp32_to_fp16(scale);
            for (lane = 0; lane < RB_QK8_0; ++lane) {
                long quantized = lroundf(row_data[block * RB_QK8_0 + lane] *
                                         inverse);
                if (quantized > 127) quantized = 127;
                if (quantized < -127) quantized = -127;
                out->values[lane] = (int8_t)quantized;
            }
        }
    }
    return 0;
}

struct quantize_worker {
    const float *source;
    uint32_t first_row;
    uint32_t row_count;
    uint32_t k;
    size_t source_row_stride;
    struct q8q8_block_q8_0 *destination;
    int status;
};

#if defined(__aarch64__) && defined(__ARM_NEON)
static int quantize_block_fast(const float *source,
                               struct q8q8_block_q8_0 *destination) {
    float32x4_t values[8];
    float32x4_t absolute[8];
    float32x4_t maxima[4];
    const float32x4_t finite_limit = vdupq_n_f32(FLT_MAX);
    uint32_t index;
    float amax;
    float scale;
    float inverse;
    for (index = 0; index < 8; ++index) {
        values[index] = vld1q_f32(source + 4 * index);
        absolute[index] = vabsq_f32(values[index]);
        if (vminvq_u32(vcleq_f32(absolute[index], finite_limit)) == 0) {
            return -2;
        }
    }
    for (index = 0; index < 4; ++index) {
        maxima[index] = vmaxq_f32(absolute[2 * index],
                                   absolute[2 * index + 1]);
    }
    maxima[0] = vmaxq_f32(maxima[0], maxima[1]);
    maxima[2] = vmaxq_f32(maxima[2], maxima[3]);
    amax = vmaxvq_f32(vmaxq_f32(maxima[0], maxima[2]));
    scale = amax / 127.0f;
    inverse = scale != 0.0f ? 1.0f / scale : 0.0f;
    destination->scale_fp16 = fp32_to_fp16(scale);
    for (index = 0; index < 8; index += 4) {
        const int32x4_t q0 = vcvtaq_s32_f32(vmulq_n_f32(values[index], inverse));
        const int32x4_t q1 = vcvtaq_s32_f32(vmulq_n_f32(values[index + 1], inverse));
        const int32x4_t q2 = vcvtaq_s32_f32(vmulq_n_f32(values[index + 2], inverse));
        const int32x4_t q3 = vcvtaq_s32_f32(vmulq_n_f32(values[index + 3], inverse));
        const int16x8_t q01 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t q23 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        const int8x16_t packed = vcombine_s8(vqmovn_s16(q01), vqmovn_s16(q23));
        vst1q_s8(destination->values + 4 * index, packed);
    }
    return 0;
}
#endif

static int quantize_rows_fast(struct quantize_worker *worker) {
    const size_t blocks_per_row = worker->k / RB_QK8_0;
    uint32_t local_row;
    for (local_row = 0; local_row < worker->row_count; ++local_row) {
        const uint32_t row = worker->first_row + local_row;
        const float *row_data = (const float *)((const uint8_t *)worker->source +
                                                (size_t)row * worker->source_row_stride);
#if defined(__aarch64__) && defined(__ARM_NEON)
        size_t block;
        for (block = 0; block < blocks_per_row; ++block) {
            const int status = quantize_block_fast(
                    row_data + block * RB_QK8_0,
                    &worker->destination[(size_t)row * blocks_per_row + block]);
            if (status != 0) return status;
        }
#else
        const int status = rb_ggml_quantize_f32_q8_0(
                row_data, 1, worker->k, worker->source_row_stride,
                &worker->destination[(size_t)row * blocks_per_row],
                blocks_per_row * sizeof(*worker->destination));
        if (status != 0) return status;
#endif
    }
    return 0;
}

static void *quantize_worker_entry(void *opaque) {
    struct quantize_worker *worker = opaque;
    worker->status = quantize_rows_fast(worker);
    return NULL;
}

int rb_ggml_quantize_f32_q8_0_fast(
        const float *source, uint32_t rows, uint32_t k,
        size_t source_row_stride, struct q8q8_block_q8_0 *destination,
        size_t destination_bytes, uint32_t threads) {
    pthread_t handles[63];
    struct quantize_worker workers[64];
    size_t blocks_per_row;
    size_t required;
    uint32_t started = 0;
    uint32_t worker_count;
    uint32_t index;
    int result = 0;
    if (source == NULL || destination == NULL || rows == 0 || k == 0 ||
            k % RB_QK8_0 != 0 || threads == 0 || threads > 64 ||
            source_row_stride < (size_t)k * sizeof(float)) return -1;
    blocks_per_row = k / RB_QK8_0;
    if (rows > SIZE_MAX / blocks_per_row ||
            (size_t)rows * blocks_per_row > SIZE_MAX / sizeof(*destination)) {
        return -1;
    }
    required = (size_t)rows * blocks_per_row * sizeof(*destination);
    if (destination_bytes < required) return -1;
    worker_count = threads < rows ? threads : rows;
    memset(workers, 0, sizeof(workers));
    for (index = 0; index < worker_count; ++index) {
        const uint32_t first = (uint32_t)(((uint64_t)rows * index) /
                                          worker_count);
        const uint32_t end = (uint32_t)(((uint64_t)rows * (index + 1)) /
                                        worker_count);
        workers[index].source = source;
        workers[index].first_row = first;
        workers[index].row_count = end - first;
        workers[index].k = k;
        workers[index].source_row_stride = source_row_stride;
        workers[index].destination = destination;
    }
    for (index = 0; index + 1 < worker_count; ++index) {
        if (pthread_create(&handles[index], NULL, quantize_worker_entry,
                           &workers[index]) != 0) {
            result = -1;
            break;
        }
        ++started;
    }
    if (result == 0) quantize_worker_entry(&workers[worker_count - 1]);
    for (index = 0; index < started; ++index) pthread_join(handles[index], NULL);
    if (result != 0) return result;
    for (index = 0; index < worker_count; ++index) {
        if (workers[index].status != 0) return workers[index].status;
    }
    return 0;
}

struct pack_worker {
    const struct q8q8_block_q8_0 *source;
    volatile uint8_t *quantized;
    volatile uint16_t *scales;
    size_t first_block;
    size_t block_count;
};

static void *pack_worker_entry(void *opaque) {
    const struct pack_worker *worker = opaque;
    size_t local;
    for (local = 0; local < worker->block_count; ++local) {
        const size_t block = worker->first_block + local;
#if defined(__aarch64__) && defined(__ARM_NEON)
        const int8x16_t low = vld1q_s8(worker->source[block].values);
        const int8x16_t high = vld1q_s8(worker->source[block].values + 16);
        vst1q_s8((int8_t *)(worker->quantized + block * RB_QK8_0), low);
        vst1q_s8((int8_t *)(worker->quantized + block * RB_QK8_0 + 16), high);
#else
        size_t lane;
        for (lane = 0; lane < RB_QK8_0; ++lane) {
            worker->quantized[block * RB_QK8_0 + lane] =
                    (uint8_t)worker->source[block].values[lane];
        }
#endif
        worker->scales[block] = worker->source[block].scale_fp16;
    }
    return NULL;
}

int rb_ggml_pack_q8_0_split_fast(
        const struct q8q8_block_q8_0 *source, uint32_t rows, uint32_t k,
        const struct q8q8_buffer *quantized,
        const struct q8q8_buffer *scales, uint32_t threads) {
    pthread_t handles[63];
    struct pack_worker workers[64];
    size_t blocks;
    uint32_t worker_count;
    uint32_t started = 0;
    uint32_t index;
    int result = 0;
    if (source == NULL || quantized == NULL || scales == NULL ||
            quantized->data == NULL || scales->data == NULL || rows == 0 ||
            k == 0 || k % RB_QK8_0 != 0 || threads == 0 || threads > 64 ||
            rows > SIZE_MAX / (k / RB_QK8_0)) return -1;
    blocks = (size_t)rows * (k / RB_QK8_0);
    if (blocks > SIZE_MAX / RB_QK8_0 ||
            blocks > SIZE_MAX / sizeof(uint16_t) ||
            quantized->size < blocks * RB_QK8_0 ||
            scales->size < blocks * sizeof(uint16_t)) return -1;
    worker_count = threads;
    if ((size_t)worker_count > blocks) worker_count = (uint32_t)blocks;
    memset(workers, 0, sizeof(workers));
    for (index = 0; index < worker_count; ++index) {
        const size_t first = blocks * index / worker_count;
        const size_t end = blocks * (index + 1) / worker_count;
        workers[index].source = source;
        workers[index].quantized = quantized->data;
        workers[index].scales = (volatile uint16_t *)scales->data;
        workers[index].first_block = first;
        workers[index].block_count = end - first;
    }
    for (index = 0; index + 1 < worker_count; ++index) {
        if (pthread_create(&handles[index], NULL, pack_worker_entry,
                           &workers[index]) != 0) {
            result = -1;
            break;
        }
        ++started;
    }
    if (result == 0) pack_worker_entry(&workers[worker_count - 1]);
    for (index = 0; index < started; ++index) pthread_join(handles[index], NULL);
    __sync_synchronize();
    return result;
}

struct copy_worker {
    uint8_t *destination;
    const volatile uint8_t *source;
    size_t first;
    size_t bytes;
};

static void *copy_worker_entry(void *opaque) {
    const struct copy_worker *worker = opaque;
    size_t index = 0;
#if defined(__aarch64__) && defined(__ARM_NEON)
    for (; index + 64 <= worker->bytes; index += 64) {
        const uint8x16_t v0 = vld1q_u8((const uint8_t *)(worker->source + index));
        const uint8x16_t v1 = vld1q_u8((const uint8_t *)(worker->source + index + 16));
        const uint8x16_t v2 = vld1q_u8((const uint8_t *)(worker->source + index + 32));
        const uint8x16_t v3 = vld1q_u8((const uint8_t *)(worker->source + index + 48));
        vst1q_u8(worker->destination + index, v0);
        vst1q_u8(worker->destination + index + 16, v1);
        vst1q_u8(worker->destination + index + 32, v2);
        vst1q_u8(worker->destination + index + 48, v3);
    }
#endif
    for (; index < worker->bytes; ++index) {
        worker->destination[index] = worker->source[index];
    }
    return NULL;
}

int rb_ggml_copy_from_uio_fast(
        void *destination, const struct q8q8_buffer *source,
        size_t source_offset, size_t bytes, uint32_t threads) {
    pthread_t handles[63];
    struct copy_worker workers[64];
    uint32_t worker_count;
    uint32_t started = 0;
    uint32_t index;
    int result = 0;
    if (destination == NULL || source == NULL || source->data == NULL ||
            bytes == 0 || threads == 0 || threads > 64 ||
            source_offset > source->size || bytes > source->size - source_offset) {
        return -1;
    }
    worker_count = threads;
    if ((size_t)worker_count > bytes / 64 && bytes >= 64) {
        worker_count = (uint32_t)(bytes / 64);
    }
    if (worker_count == 0) worker_count = 1;
    memset(workers, 0, sizeof(workers));
    for (index = 0; index < worker_count; ++index) {
        size_t first = bytes * index / worker_count;
        size_t end = bytes * (index + 1) / worker_count;
        if (index != 0) first &= ~(size_t)63;
        if (index + 1 != worker_count) end &= ~(size_t)63;
        workers[index].destination = (uint8_t *)destination + first;
        workers[index].source = source->data + source_offset + first;
        workers[index].first = first;
        workers[index].bytes = end - first;
    }
    for (index = 0; index + 1 < worker_count; ++index) {
        if (pthread_create(&handles[index], NULL, copy_worker_entry,
                           &workers[index]) != 0) {
            result = -1;
            break;
        }
        ++started;
    }
    if (result == 0) copy_worker_entry(&workers[worker_count - 1]);
    for (index = 0; index < started; ++index) pthread_join(handles[index], NULL);
    __sync_synchronize();
    return result;
}

int rb_ggml_q8q8_execute(const struct rb_ggml_mul_mat_view *operation,
                         struct q8q8_device *device,
                         struct q8q8_arena *device_arena,
                         struct q8q8_block_q8_0 *activation_workspace,
                         size_t activation_workspace_bytes,
                         uint64_t timeout_us_per_chunk,
                         struct q8q8_dispatch_stats *stats,
                         struct rb_ggml_q8q8_report *report) {
    struct q8q8_dispatch_buffers buffers;
    const struct q8q8_block_q8_0 *activations;
    int eligible;
    int status;
    if (operation == NULL || device == NULL || device_arena == NULL ||
            report == NULL) return Q8Q8_RT_ERR_ARGUMENT;
    eligible = rb_ggml_q8q8_classify(operation, device_arena->size,
                                     64, report);
    if (eligible != 1) return Q8Q8_RT_ERR_ARGUMENT;
    status = q8q8_allocate_plan(device_arena, &report->plan, &buffers);
    if (status != Q8Q8_RT_OK) return status;
    status = q8q8_pack_ggml_q8_0(
            (const struct q8q8_block_q8_0 *)operation->src0_weights.data,
            report->n, report->k, &buffers.weights_q,
            &buffers.weights_scale);
    if (status != Q8Q8_RT_OK) return status;
    if (report->activation_quantization_required) {
        if (activation_workspace == NULL) return Q8Q8_RT_ERR_ARGUMENT;
        status = rb_ggml_quantize_f32_q8_0(
                (const float *)operation->src1_activations.data,
                report->m, report->k, operation->src1_activations.nb[1],
                activation_workspace, activation_workspace_bytes);
        if (status == -2) {
            report->reason = RB_GGML_Q8Q8_NONFINITE_ACTIVATION;
            return Q8Q8_RT_ERR_ARGUMENT;
        }
        if (status != 0) return Q8Q8_RT_ERR_ARGUMENT;
        activations = activation_workspace;
    } else {
        activations = operation->src1_activations.data;
    }
    return q8q8_execute_plan(device, &report->plan, &buffers, activations,
                             (float *)operation->dst_f32.data,
                             timeout_us_per_chunk, stats);
}

int rb_ggml_q8q8_execute_fast(
        const struct rb_ggml_mul_mat_view *operation,
        struct q8q8_device *device, struct q8q8_arena *device_arena,
        struct q8q8_block_q8_0 *activation_workspace,
        size_t activation_workspace_bytes, uint64_t timeout_us_per_chunk,
        uint32_t threads, struct q8q8_dispatch_stats *stats,
        struct rb_ggml_q8q8_report *report) {
    struct q8q8_dispatch_buffers buffers;
    const struct q8q8_block_q8_0 *activations;
    struct q8q8_dispatch_stats completed = {0, 0, 0};
    size_t blocks_per_row;
    uint32_t chunk;
    int eligible;
    int status;
    if (operation == NULL || device == NULL || device_arena == NULL ||
            report == NULL || threads == 0 || timeout_us_per_chunk == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    eligible = rb_ggml_q8q8_classify(operation, device_arena->size, 64, report);
    if (eligible != 1) return Q8Q8_RT_ERR_ARGUMENT;
    status = q8q8_allocate_plan(device_arena, &report->plan, &buffers);
    if (status != Q8Q8_RT_OK) return status;
    if (rb_ggml_pack_q8_0_split_fast(
                operation->src0_weights.data, report->n, report->k,
                &buffers.weights_q, &buffers.weights_scale, threads) != 0) {
        return Q8Q8_RT_ERR_IO;
    }
    if (report->activation_quantization_required) {
        if (activation_workspace == NULL) return Q8Q8_RT_ERR_ARGUMENT;
        status = rb_ggml_quantize_f32_q8_0_fast(
                operation->src1_activations.data, report->m, report->k,
                operation->src1_activations.nb[1], activation_workspace,
                activation_workspace_bytes, threads);
        if (status == -2) {
            report->reason = RB_GGML_Q8Q8_NONFINITE_ACTIVATION;
            return Q8Q8_RT_ERR_ARGUMENT;
        }
        if (status != 0) return Q8Q8_RT_ERR_ARGUMENT;
        activations = activation_workspace;
    } else {
        activations = operation->src1_activations.data;
    }
    blocks_per_row = report->k / RB_QK8_0;
    for (chunk = 0; chunk < report->plan.chunk_count; ++chunk) {
        struct q8q8_job job;
        struct q8q8_result result;
        uint32_t row_offset;
        size_t output_bytes;
        status = q8q8_plan_job(&report->plan, &buffers, chunk, &job,
                               &row_offset);
        if (status != Q8Q8_RT_OK) return status;
        if (rb_ggml_pack_q8_0_split_fast(
                    activations + (size_t)row_offset * blocks_per_row,
                    job.m, job.k, &buffers.activations_q,
                    &buffers.activations_scale, threads) != 0) {
            return Q8Q8_RT_ERR_IO;
        }
        status = q8q8_run(device, &job, timeout_us_per_chunk, &result);
        if (status != Q8Q8_RT_OK) return status;
        if ((size_t)job.m > SIZE_MAX / job.n ||
                (size_t)job.m * job.n > SIZE_MAX / sizeof(float)) {
            return Q8Q8_RT_ERR_ARGUMENT;
        }
        output_bytes = (size_t)job.m * job.n * sizeof(float);
        if (rb_ggml_copy_from_uio_fast(
                    (float *)operation->dst_f32.data +
                            (size_t)row_offset * job.n,
                    &buffers.output, 0, output_bytes, threads) != 0) {
            return Q8Q8_RT_ERR_IO;
        }
        ++completed.chunks_completed;
        completed.rows_completed += job.m;
        completed.hardware_elapsed_us += result.elapsed_us;
    }
    if (stats != NULL) *stats = completed;
    return Q8Q8_RT_OK;
}

const char *rb_ggml_q8q8_reason_string(enum rb_ggml_q8q8_reason reason) {
    static const char *const names[] = {
        "eligible", "null_argument", "null_data", "weight_type",
        "activation_type", "output_type", "batch_or_broadcast",
        "shape_mismatch", "weight_layout", "activation_layout",
        "output_layout", "hardware_limit", "arena_capacity",
        "nonfinite_activation",
    };
    if ((unsigned)reason >= sizeof(names) / sizeof(names[0])) return "unknown";
    return names[reason];
}
