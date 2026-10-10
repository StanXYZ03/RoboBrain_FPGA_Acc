#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"

#include "robobrain_ggml_q8q8_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static uint32_t state = 0x6d2b79f5u;

static float next_value(void) {
    state = state * 1664525u + 1013904223u;
    return ((float)(int32_t)(state >> 8) / 8388608.0f) * 19.0f;
}

int main(void) {
    enum { ROWS = 19, K = 256, BLOCKS = ROWS * K / 32 };
    float input[ROWS * K];
    block_q8_0 upstream[BLOCKS];
    struct q8q8_block_q8_0 project[BLOCKS];
    struct q8q8_block_q8_0 fast_single[BLOCKS];
    struct q8q8_block_q8_0 fast_parallel[BLOCKS];
    size_t i;
    for (i = 0; i < ROWS * K; ++i) input[i] = next_value();
    memset(input + K, 0, K * sizeof(float));
    input[2 * K] = 65504.0f;
    input[2 * K + 1] = -65504.0f;
    quantize_row_q8_0_ref(input, upstream, ROWS * K);
    if (rb_ggml_quantize_f32_q8_0(input, ROWS, K, K * sizeof(float),
                                  project, sizeof(project)) != 0) {
        fputs("UPSTREAM_Q8_0_QUANTIZER_FAIL project_status\n", stderr);
        return 1;
    }
    if (memcmp(upstream, project, sizeof(upstream)) != 0) {
        for (i = 0; i < BLOCKS; ++i) {
            if (memcmp(&upstream[i], &project[i], sizeof(upstream[i])) != 0) {
                fprintf(stderr,
                        "UPSTREAM_Q8_0_QUANTIZER_FAIL first_block=%zu "
                        "upstream_scale=0x%04x project_scale=0x%04x\n",
                        i, upstream[i].d, project[i].scale_fp16);
                break;
            }
        }
        return 1;
    }
    if (rb_ggml_quantize_f32_q8_0_fast(
                input, ROWS, K, K * sizeof(float), fast_single,
                sizeof(fast_single), 1) != 0 ||
            rb_ggml_quantize_f32_q8_0_fast(
                input, ROWS, K, K * sizeof(float), fast_parallel,
                sizeof(fast_parallel), 4) != 0 ||
            memcmp(upstream, fast_single, sizeof(upstream)) != 0 ||
            memcmp(upstream, fast_parallel, sizeof(upstream)) != 0) {
        fputs("UPSTREAM_Q8_0_QUANTIZER_FAIL fast_byte_difference\n", stderr);
        return 1;
    }
    printf("UPSTREAM_Q8_0_QUANTIZER_PASS rows=%d k=%d blocks=%d "
           "byte_equal=yes fast_threads=1,4 upstream_modified=no\n",
           ROWS, K, BLOCKS);
    return 0;
}
