#define _POSIX_C_SOURCE 200809L

#include "robobrain_ggml_q8q8_bridge.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    REPEATS = 3,
    ALIGNMENT = 64,
};

struct shape {
    const char *name;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

static const struct shape cases[] = {
    {"decoder_m1", 1, 2560, 2560},
    {"text_prefill_m49", 49, 2560, 2560},
    {"visual_tail_m56", 56, 9728, 2560},
    {"language_prefill_m512", 512, 9728, 2560},
    {"visual_prefill_m1080", 1080, 9728, 2560},
    {"vision_encoder_m4320", 4320, 4096, 1024},
};

static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static void *allocate(size_t bytes) {
    void *pointer = NULL;
    if (posix_memalign(&pointer, ALIGNMENT, bytes) != 0) return NULL;
    return pointer;
}

static float source_value(size_t index) {
    const int32_t centered = (int32_t)((index * 17u + 29u) % 257u) - 128;
    return (float)centered / 37.0f;
}

static void fill_f32(float *data, size_t count) {
    size_t index;
    for (index = 0; index < count; ++index) data[index] = source_value(index);
}

static void fill_q8(struct q8q8_block_q8_0 *data, size_t blocks) {
    size_t block;
    for (block = 0; block < blocks; ++block) {
        size_t lane;
        data[block].scale_fp16 = (uint16_t)(0x2000u + block % 0x1fffu);
        for (lane = 0; lane < 32; ++lane) {
            data[block].values[lane] =
                    (int8_t)(((block * 13u + lane * 7u) % 255u) - 127);
        }
    }
}

static uint64_t checksum_q8(const struct q8q8_block_q8_0 *data,
                            size_t blocks) {
    const uint8_t *bytes = (const uint8_t *)data;
    const size_t count = blocks * sizeof(*data);
    uint64_t checksum = UINT64_C(1469598103934665603);
    size_t index;
    for (index = 0; index < count; index += 4093) {
        checksum ^= bytes[index];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static uint64_t checksum_split(const uint8_t *quantized,
                               const uint16_t *scales, size_t blocks) {
    uint64_t checksum = UINT64_C(1469598103934665603);
    size_t block;
    for (block = 0; block < blocks; block += 127) {
        checksum ^= quantized[block * 32];
        checksum *= UINT64_C(1099511628211);
        checksum ^= scales[block];
        checksum *= UINT64_C(1099511628211);
    }
    return checksum;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static int verify_fast_quantization(const struct shape *shape) {
    const size_t values = (size_t)shape->m * shape->k;
    const size_t blocks = values / 32;
    const size_t source_bytes = values * sizeof(float);
    const size_t output_bytes = blocks * sizeof(struct q8q8_block_q8_0);
    float *source = allocate(source_bytes);
    struct q8q8_block_q8_0 *reference = allocate(output_bytes);
    struct q8q8_block_q8_0 *fast1 = allocate(output_bytes);
    struct q8q8_block_q8_0 *fast4 = allocate(output_bytes);
    int status = -1;
    if (source == NULL || reference == NULL || fast1 == NULL || fast4 == NULL) {
        goto done;
    }
    fill_f32(source, values);
    if (rb_ggml_quantize_f32_q8_0(
                source, shape->m, shape->k,
                (size_t)shape->k * sizeof(float), reference,
                output_bytes) != 0 ||
            rb_ggml_quantize_f32_q8_0_fast(
                source, shape->m, shape->k,
                (size_t)shape->k * sizeof(float), fast1,
                output_bytes, 1) != 0 ||
            rb_ggml_quantize_f32_q8_0_fast(
                source, shape->m, shape->k,
                (size_t)shape->k * sizeof(float), fast4,
                output_bytes, 4) != 0 ||
            memcmp(reference, fast1, output_bytes) != 0 ||
            memcmp(reference, fast4, output_bytes) != 0) {
        goto done;
    }
    printf("PREP_NUMERIC case=%s m=%u n=%u k=%u bytes=%zu "
           "reference_vs_fast1=byte_equal reference_vs_fast4=byte_equal "
           "result=PASS\n", shape->name, shape->m, shape->n, shape->k,
           output_bytes);
    status = 0;
done:
    free(fast4);
    free(fast1);
    free(reference);
    free(source);
    return status;
}

static int benchmark_quantize(const struct shape *shape, const char *operation,
                              uint32_t threads) {
    const size_t values = (size_t)shape->m * shape->k;
    const size_t blocks = values / 32;
    const size_t source_bytes = values * sizeof(float);
    const size_t output_bytes = blocks * sizeof(struct q8q8_block_q8_0);
    float *source = allocate(source_bytes);
    struct q8q8_block_q8_0 *output = allocate(output_bytes);
    uint64_t samples[REPEATS];
    uint64_t checksum = 0;
    unsigned repeat;
    if (source == NULL || output == NULL) {
        free(output);
        free(source);
        return -1;
    }
    fill_f32(source, values);
    memset(output, 0, output_bytes);
    for (repeat = 0; repeat < REPEATS; ++repeat) {
        const uint64_t begin = now_ns();
        const int status = threads == 0
                ? rb_ggml_quantize_f32_q8_0(
                          source, shape->m, shape->k,
                          (size_t)shape->k * sizeof(float), output, output_bytes)
                : rb_ggml_quantize_f32_q8_0_fast(
                          source, shape->m, shape->k,
                          (size_t)shape->k * sizeof(float), output,
                          output_bytes, threads);
        const uint64_t end = now_ns();
        if (status != 0 || begin == 0 || end < begin) {
            free(output);
            free(source);
            return -1;
        }
        samples[repeat] = end - begin;
        checksum ^= checksum_q8(output, blocks);
        printf("PREP_SAMPLE operation=%s case=%s repeat=%u "
               "m=%u n=%u k=%u elapsed_ns=%" PRIu64 "\n",
               operation, shape->name, repeat, shape->m, shape->n, shape->k,
               samples[repeat]);
    }
    qsort(samples, REPEATS, sizeof(samples[0]), compare_u64);
    printf("PREP_RESULT operation=%s case=%s m=%u n=%u "
           "k=%u input_bytes=%zu output_bytes=%zu median_ns=%" PRIu64
           " throughput_gbs=%.6f checksum=%" PRIu64 " result=PASS\n",
           operation, shape->name, shape->m, shape->n, shape->k, source_bytes,
           output_bytes, samples[REPEATS / 2],
           (double)source_bytes / (double)samples[REPEATS / 2], checksum);
    free(output);
    free(source);
    return 0;
}

static int benchmark_weight_pack(const struct shape *shape) {
    const size_t blocks = (size_t)shape->n * (shape->k / 32);
    const size_t source_bytes = blocks * sizeof(struct q8q8_block_q8_0);
    const size_t quantized_bytes = blocks * 32;
    const size_t scale_bytes = blocks * sizeof(uint16_t);
    struct q8q8_block_q8_0 *source = allocate(source_bytes);
    uint8_t *quantized = allocate(quantized_bytes);
    uint16_t *scales = allocate(scale_bytes);
    struct q8q8_buffer quantized_buffer;
    struct q8q8_buffer scale_buffer;
    uint64_t samples[REPEATS];
    uint64_t checksum = 0;
    unsigned repeat;
    if (source == NULL || quantized == NULL || scales == NULL) {
        free(scales);
        free(quantized);
        free(source);
        return -1;
    }
    fill_q8(source, blocks);
    memset(quantized, 0, quantized_bytes);
    memset(scales, 0, scale_bytes);
    quantized_buffer.data = quantized;
    quantized_buffer.phys = 64;
    quantized_buffer.offset = 0;
    quantized_buffer.size = quantized_bytes;
    scale_buffer.data = (volatile uint8_t *)scales;
    scale_buffer.phys = 64;
    scale_buffer.offset = 0;
    scale_buffer.size = scale_bytes;
    for (repeat = 0; repeat < REPEATS; ++repeat) {
        const uint64_t begin = now_ns();
        const int status = q8q8_pack_ggml_q8_0(
                source, shape->n, shape->k, &quantized_buffer, &scale_buffer);
        const uint64_t end = now_ns();
        if (status != Q8Q8_RT_OK || begin == 0 || end < begin) {
            free(scales);
            free(quantized);
            free(source);
            return -1;
        }
        samples[repeat] = end - begin;
        checksum ^= checksum_split(quantized, scales, blocks);
        printf("PREP_SAMPLE operation=weight_pack case=%s repeat=%u m=%u "
               "n=%u k=%u elapsed_ns=%" PRIu64 "\n",
               shape->name, repeat, shape->m, shape->n, shape->k,
               samples[repeat]);
    }
    qsort(samples, REPEATS, sizeof(samples[0]), compare_u64);
    printf("PREP_RESULT operation=weight_pack case=%s m=%u n=%u k=%u "
           "input_bytes=%zu output_bytes=%zu median_ns=%" PRIu64
           " throughput_gbs=%.6f checksum=%" PRIu64 " result=PASS\n",
           shape->name, shape->m, shape->n, shape->k, source_bytes,
           quantized_bytes + scale_bytes, samples[REPEATS / 2],
           (double)(source_bytes + quantized_bytes + scale_bytes) /
                   (double)samples[REPEATS / 2], checksum);
    free(scales);
    free(quantized);
    free(source);
    return 0;
}

int main(void) {
    size_t index;
#if defined(__aarch64__) && defined(__ARM_NEON)
    const char *implementation = "aarch64_neon";
#else
    const char *implementation = "scalar_fallback";
#endif
    printf("PREP_BENCH_BEGIN version=3 repeats=%u destination=cacheable_ddr "
           "fast_implementation=%s\n", REPEATS, implementation);
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (verify_fast_quantization(&cases[index]) != 0 ||
                benchmark_quantize(&cases[index], "activation_quantize_reference", 0) != 0 ||
                benchmark_quantize(&cases[index], "activation_quantize_fast1", 1) != 0 ||
                benchmark_quantize(&cases[index], "activation_quantize_fast4", 4) != 0 ||
                benchmark_weight_pack(&cases[index]) != 0) {
            printf("PREP_BENCH_FAIL case=%s\n", cases[index].name);
            return 1;
        }
    }
    printf("PREP_BENCH_PASS cases=%zu measurements=%zu numeric_cases=%zu\n",
           sizeof(cases) / sizeof(cases[0]),
           4 * sizeof(cases) / sizeof(cases[0]),
           sizeof(cases) / sizeof(cases[0]));
    return 0;
}
