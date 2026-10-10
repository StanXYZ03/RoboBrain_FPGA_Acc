#define _POSIX_C_SOURCE 200809L

#include "q8q8_uio.h"
#include "robobrain_ggml_q8q8_bridge.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { REPEATS = 3, ALIGNMENT = 64 };

struct weight_shape {
    const char *name;
    uint32_t n;
    uint32_t k;
};

static const struct weight_shape cases[] = {
    {"llm_square_2560", 2560, 2560},
    {"llm_ffn_9728", 9728, 2560},
    {"vision_fc_4096", 4096, 1024},
};

static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + value.tv_nsec;
}

static void *allocate(size_t bytes) {
    void *pointer = NULL;
    if (posix_memalign(&pointer, ALIGNMENT, bytes) != 0) return NULL;
    return pointer;
}

static size_t align_up(size_t value) {
    return (value + ALIGNMENT - 1) & ~(size_t)(ALIGNMENT - 1);
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static void fill_source(struct q8q8_block_q8_0 *source, size_t blocks) {
    size_t block;
    for (block = 0; block < blocks; ++block) {
        size_t lane;
        source[block].scale_fp16 = (uint16_t)(0x2000u + block % 0x1fffu);
        for (lane = 0; lane < 32; ++lane) {
            source[block].values[lane] =
                    (int8_t)(((block * 13u + lane * 7u) % 255u) - 127);
        }
    }
}

static int run_pack(const char *operation,
                    const struct q8q8_block_q8_0 *source,
                    const struct weight_shape *shape,
                    const struct q8q8_buffer *quantized,
                    const struct q8q8_buffer *scales, uint32_t threads,
                    uint64_t *median_ns) {
    uint64_t samples[REPEATS];
    unsigned repeat;
    for (repeat = 0; repeat < REPEATS; ++repeat) {
        const uint64_t begin = now_ns();
        const int status = threads == 0
                ? q8q8_pack_ggml_q8_0(source, shape->n, shape->k,
                                      quantized, scales)
                : rb_ggml_pack_q8_0_split_fast(source, shape->n, shape->k,
                                               quantized, scales, threads);
        const uint64_t end = now_ns();
        if (status != 0 || begin == 0 || end < begin) return -1;
        samples[repeat] = end - begin;
        printf("WEIGHT_PATH_SAMPLE destination=%s operation=%s case=%s "
               "repeat=%u n=%u k=%u elapsed_ns=%" PRIu64 "\n",
               quantized->phys == 64 ? "cacheable_ddr" : "uio_noncached",
               operation, shape->name, repeat, shape->n, shape->k,
               samples[repeat]);
    }
    qsort(samples, REPEATS, sizeof(samples[0]), compare_u64);
    *median_ns = samples[REPEATS / 2];
    return 0;
}

static int benchmark_case(const struct weight_shape *shape,
                          struct q8q8_uio *uio) {
    const size_t blocks = (size_t)shape->n * (shape->k / 32);
    const size_t source_bytes = blocks * sizeof(struct q8q8_block_q8_0);
    const size_t quantized_bytes = blocks * 32;
    const size_t scale_bytes = blocks * sizeof(uint16_t);
    struct q8q8_block_q8_0 *source = allocate(source_bytes);
    uint8_t *golden_q = allocate(quantized_bytes);
    uint16_t *golden_s = allocate(scale_bytes);
    uint8_t *target_q = allocate(quantized_bytes);
    uint16_t *target_s = allocate(scale_bytes);
    struct q8q8_buffer golden_q_buffer = {golden_q, 64, 0, quantized_bytes};
    struct q8q8_buffer golden_s_buffer = {
        (volatile uint8_t *)golden_s, 64, 0, scale_bytes
    };
    struct q8q8_buffer target_q_buffer = {target_q, 64, 0, quantized_bytes};
    struct q8q8_buffer target_s_buffer = {
        (volatile uint8_t *)target_s, 64, 0, scale_bytes
    };
    uint64_t reference_ns;
    uint64_t fast1_ns;
    uint64_t fast4_ns;
    int status = -1;
    if (source == NULL || golden_q == NULL || golden_s == NULL ||
            target_q == NULL || target_s == NULL) goto done;
    fill_source(source, blocks);
    if (run_pack("reference", source, shape, &golden_q_buffer,
                 &golden_s_buffer, 0, &reference_ns) != 0 ||
            run_pack("fast1", source, shape, &target_q_buffer,
                     &target_s_buffer, 1, &fast1_ns) != 0 ||
            memcmp(golden_q, target_q, quantized_bytes) != 0 ||
            memcmp(golden_s, target_s, scale_bytes) != 0 ||
            run_pack("fast4", source, shape, &target_q_buffer,
                     &target_s_buffer, 4, &fast4_ns) != 0 ||
            memcmp(golden_q, target_q, quantized_bytes) != 0 ||
            memcmp(golden_s, target_s, scale_bytes) != 0) goto done;
    printf("WEIGHT_PATH_RESULT destination=cacheable_ddr case=%s n=%u k=%u "
           "bytes=%zu reference_ns=%" PRIu64 " fast1_ns=%" PRIu64
           " fast4_ns=%" PRIu64 " fast1_speedup=%.6f fast4_speedup=%.6f "
           "byte_equal=yes result=PASS\n", shape->name, shape->n, shape->k,
           source_bytes, reference_ns, fast1_ns, fast4_ns,
           (double)reference_ns / fast1_ns, (double)reference_ns / fast4_ns);
    if (uio != NULL) {
        const size_t scale_offset = align_up(quantized_bytes);
        struct q8q8_buffer uio_q;
        struct q8q8_buffer uio_s;
        if (scale_offset > uio->arena_size ||
                scale_bytes > uio->arena_size - scale_offset) goto done;
        uio_q.data = uio->arena;
        uio_q.phys = uio->arena_phys;
        uio_q.offset = 0;
        uio_q.size = quantized_bytes;
        uio_s.data = (volatile uint8_t *)uio->arena + scale_offset;
        uio_s.phys = uio->arena_phys + scale_offset;
        uio_s.offset = scale_offset;
        uio_s.size = scale_bytes;
        if (run_pack("reference", source, shape, &uio_q, &uio_s, 0,
                     &reference_ns) != 0 ||
                memcmp(golden_q, (const void *)uio_q.data,
                       quantized_bytes) != 0 ||
                memcmp(golden_s, (const void *)uio_s.data, scale_bytes) != 0 ||
                run_pack("fast1", source, shape, &uio_q, &uio_s, 1,
                         &fast1_ns) != 0 ||
                memcmp(golden_q, (const void *)uio_q.data,
                       quantized_bytes) != 0 ||
                memcmp(golden_s, (const void *)uio_s.data, scale_bytes) != 0 ||
                run_pack("fast4", source, shape, &uio_q, &uio_s, 4,
                         &fast4_ns) != 0 ||
                memcmp(golden_q, (const void *)uio_q.data,
                       quantized_bytes) != 0 ||
                memcmp(golden_s, (const void *)uio_s.data, scale_bytes) != 0) {
            goto done;
        }
        printf("WEIGHT_PATH_RESULT destination=uio_noncached case=%s n=%u k=%u "
               "bytes=%zu reference_ns=%" PRIu64 " fast1_ns=%" PRIu64
               " fast4_ns=%" PRIu64 " fast1_speedup=%.6f fast4_speedup=%.6f "
               "byte_equal=yes result=PASS\n", shape->name, shape->n,
               shape->k, source_bytes, reference_ns, fast1_ns, fast4_ns,
               (double)reference_ns / fast1_ns,
               (double)reference_ns / fast4_ns);
    }
    status = 0;
done:
    free(target_s);
    free(target_q);
    free(golden_s);
    free(golden_q);
    free(source);
    return status;
}

int main(int argc, char **argv) {
    struct q8q8_uio uio;
    struct q8q8_uio *uio_pointer = NULL;
    int cache_only = argc == 2 && strcmp(argv[1], "--cache-only") == 0;
    size_t index;
    if (argc > 2 || (argc == 2 && !cache_only)) {
        fputs("usage: weight-path-benchmark [--cache-only]\n", stderr);
        return 2;
    }
    if (!cache_only) {
        if (q8q8_uio_open(&uio, "q8q8-gemm") != Q8Q8_RT_OK) {
            fputs("WEIGHT_PATH_FAIL stage=uio_open\n", stderr);
            return 1;
        }
        uio_pointer = &uio;
    }
    printf("WEIGHT_PATH_BEGIN version=1 repeats=%u destinations=%s "
           "pl_control_access=no sd_qspi_access=no\n", REPEATS,
           cache_only ? "cacheable_ddr" : "cacheable_ddr,uio_noncached");
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (benchmark_case(&cases[index], uio_pointer) != 0) {
            printf("WEIGHT_PATH_FAIL case=%s\n", cases[index].name);
            if (uio_pointer != NULL) q8q8_uio_close(uio_pointer);
            return 1;
        }
    }
    if (uio_pointer != NULL) q8q8_uio_close(uio_pointer);
    printf("WEIGHT_PATH_PASS cases=3 destinations=%u byte_equal=yes "
           "pl_control_access=no sd_qspi_written=no\n",
           cache_only ? 1u : 2u);
    return 0;
}
