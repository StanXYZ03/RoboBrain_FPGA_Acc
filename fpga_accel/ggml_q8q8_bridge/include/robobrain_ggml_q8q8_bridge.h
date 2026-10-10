#ifndef ROBOBRAIN_GGML_Q8Q8_BRIDGE_H
#define ROBOBRAIN_GGML_Q8Q8_BRIDGE_H

#include "q8q8_dispatch.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Project-owned, minimal GGML ABI view.  This intentionally does not depend on
 * private llama.cpp headers and therefore does not modify upstream sources. */
enum rb_ggml_scalar_type {
    RB_GGML_TYPE_F32 = 0,
    RB_GGML_TYPE_Q8_0 = 8,
};

struct rb_ggml_tensor_view {
    const void *data;
    enum rb_ggml_scalar_type type;
    uint64_t ne[4];
    size_t nb[4];
};

struct rb_ggml_mul_mat_view {
    /* GGML MUL_MAT semantics: dst = src0^T * src1. */
    struct rb_ggml_tensor_view src0_weights;
    struct rb_ggml_tensor_view src1_activations;
    struct rb_ggml_tensor_view dst_f32;
};

enum rb_ggml_q8q8_reason {
    RB_GGML_Q8Q8_ELIGIBLE = 0,
    RB_GGML_Q8Q8_NULL_ARGUMENT,
    RB_GGML_Q8Q8_NULL_DATA,
    RB_GGML_Q8Q8_WEIGHT_TYPE,
    RB_GGML_Q8Q8_ACTIVATION_TYPE,
    RB_GGML_Q8Q8_OUTPUT_TYPE,
    RB_GGML_Q8Q8_BATCH_OR_BROADCAST,
    RB_GGML_Q8Q8_SHAPE_MISMATCH,
    RB_GGML_Q8Q8_WEIGHT_LAYOUT,
    RB_GGML_Q8Q8_ACTIVATION_LAYOUT,
    RB_GGML_Q8Q8_OUTPUT_LAYOUT,
    RB_GGML_Q8Q8_HARDWARE_LIMIT,
    RB_GGML_Q8Q8_ARENA_CAPACITY,
    RB_GGML_Q8Q8_NONFINITE_ACTIVATION,
};

struct rb_ggml_q8q8_report {
    enum rb_ggml_q8q8_reason reason;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    int activation_quantization_required;
    struct q8q8_dispatch_plan plan;
};

/* Returns 1 when the operation can be sent to PL, 0 for a safe CPU fallback,
 * and -1 only for a malformed API call. */
int rb_ggml_q8q8_classify(const struct rb_ggml_mul_mat_view *operation,
                          size_t arena_capacity, size_t alignment,
                          struct rb_ggml_q8q8_report *report);

/* Bit-compatible storage contract with GGML block_q8_0.  Non-finite input is
 * rejected so the caller can take the established CPU path. */
int rb_ggml_quantize_f32_q8_0(const float *source, uint32_t rows, uint32_t k,
                              size_t source_row_stride,
                              struct q8q8_block_q8_0 *destination,
                              size_t destination_bytes);

/* AArch64 uses NEON and rows can be divided across caller-selected workers.
 * Other architectures retain the same exact scalar contract.  The reference
 * entry above remains available for differential and fallback testing. */
int rb_ggml_quantize_f32_q8_0_fast(
        const float *source, uint32_t rows, uint32_t k,
        size_t source_row_stride, struct q8q8_block_q8_0 *destination,
        size_t destination_bytes, uint32_t threads);

/* Split immutable GGML Q8_0 blocks into the payload/scale planes consumed by
 * the current PL ABI.  Destinations may be cacheable DDR or volatile UIO. */
int rb_ggml_pack_q8_0_split_fast(
        const struct q8q8_block_q8_0 *source, uint32_t rows, uint32_t k,
        const struct q8q8_buffer *quantized,
        const struct q8q8_buffer *scales, uint32_t threads);

/* Copy PL output from volatile/non-cached UIO into a normal GGML tensor. */
int rb_ggml_copy_from_uio_fast(
        void *destination, const struct q8q8_buffer *source,
        size_t source_offset, size_t bytes, uint32_t threads);

/* Executes only an already classified, eligible operation.  The caller owns
 * fallback policy and supplies a Q8_0 workspace when src1 is F32. */
int rb_ggml_q8q8_execute(const struct rb_ggml_mul_mat_view *operation,
                         struct q8q8_device *device,
                         struct q8q8_arena *device_arena,
                         struct q8q8_block_q8_0 *activation_workspace,
                         size_t activation_workspace_bytes,
                         uint64_t timeout_us_per_chunk,
                         struct q8q8_dispatch_stats *stats,
                         struct rb_ggml_q8q8_report *report);

/* Default-off integration candidate using the measured fast quantize, split,
 * and UIO readback primitives.  It still returns every op to PS memory and is
 * therefore a bring-up path, not the final PL-resident Block architecture. */
int rb_ggml_q8q8_execute_fast(
        const struct rb_ggml_mul_mat_view *operation,
        struct q8q8_device *device, struct q8q8_arena *device_arena,
        struct q8q8_block_q8_0 *activation_workspace,
        size_t activation_workspace_bytes, uint64_t timeout_us_per_chunk,
        uint32_t threads, struct q8q8_dispatch_stats *stats,
        struct rb_ggml_q8q8_report *report);

const char *rb_ggml_q8q8_reason_string(enum rb_ggml_q8q8_reason reason);

#ifdef __cplusplus
}
#endif

#endif
