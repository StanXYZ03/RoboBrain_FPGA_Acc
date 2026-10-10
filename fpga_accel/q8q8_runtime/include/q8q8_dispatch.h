#ifndef ROBOBRAIN_Q8Q8_DISPATCH_H
#define ROBOBRAIN_Q8Q8_DISPATCH_H

#include "q8q8_arena.h"
#include "q8q8_runtime.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum q8q8_operation {
    /* C[M,N] = A[M,K] * B[N,K]^T. */
    Q8Q8_OP_MATMUL_NT = 1,
};

enum q8q8_tensor_layout {
    /* GGML block_q8_0: one FP16 scale followed by 32 signed bytes. */
    Q8Q8_LAYOUT_GGML_Q8_0 = 1,
    Q8Q8_LAYOUT_ROW_MAJOR_F32 = 2,
};

/* This descriptor deliberately contains only mathematical and storage
 * properties. General/Pointing/Grounding/Trajectory never enter the PL ABI. */
struct q8q8_workload_descriptor {
    enum q8q8_operation operation;
    enum q8q8_tensor_layout weight_layout;
    enum q8q8_tensor_layout activation_layout;
    enum q8q8_tensor_layout output_layout;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct q8q8_dispatch_plan {
    struct q8q8_workload_descriptor workload;
    uint32_t chunk_m;
    uint32_t chunk_count;
    size_t alignment;
    size_t weights_q_bytes;
    size_t weights_scale_bytes;
    size_t activations_q_bytes;
    size_t activations_scale_bytes;
    size_t output_bytes;
    size_t arena_bytes;
};

struct q8q8_dispatch_buffers {
    struct q8q8_buffer weights_q;
    struct q8q8_buffer weights_scale;
    struct q8q8_buffer activations_q;
    struct q8q8_buffer activations_scale;
    struct q8q8_buffer output;
};

struct q8q8_dispatch_stats {
    uint32_t chunks_completed;
    uint32_t rows_completed;
    uint64_t hardware_elapsed_us;
};

struct q8q8_block_q8_0 {
    uint16_t scale_fp16;
    int8_t values[32];
} __attribute__((packed));

int q8q8_plan_workload(const struct q8q8_workload_descriptor *workload,
                       size_t arena_capacity, size_t alignment,
                       struct q8q8_dispatch_plan *plan);
int q8q8_allocate_plan(struct q8q8_arena *arena,
                       const struct q8q8_dispatch_plan *plan,
                       struct q8q8_dispatch_buffers *buffers);
int q8q8_pack_ggml_q8_0(const struct q8q8_block_q8_0 *source,
                        uint32_t rows, uint32_t k,
                        const struct q8q8_buffer *quantized,
                        const struct q8q8_buffer *scales);
int q8q8_plan_job(const struct q8q8_dispatch_plan *plan,
                  const struct q8q8_dispatch_buffers *buffers,
                  uint32_t chunk_index, struct q8q8_job *job,
                  uint32_t *row_offset);
int q8q8_execute_plan(struct q8q8_device *device,
                      const struct q8q8_dispatch_plan *plan,
                      const struct q8q8_dispatch_buffers *buffers,
                      const struct q8q8_block_q8_0 *activations,
                      float *output, uint64_t timeout_us_per_chunk,
                      struct q8q8_dispatch_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
