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

struct output_shape { const char *name; uint32_t m; uint32_t n; };
static const struct output_shape cases[] = {
    {"language_m512", 512, 9728},
    {"visual_chunk_m976", 976, 9728},
    {"vision_chunk_m3584", 3584, 4096},
};

static uint64_t now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + value.tv_nsec;
}

static int compare_u64(const void *left, const void *right) {
    const uint64_t a = *(const uint64_t *)left;
    const uint64_t b = *(const uint64_t *)right;
    return (a > b) - (a < b);
}

static int measure(const char *operation, void *destination,
                   const struct q8q8_buffer *source, size_t bytes,
                   uint32_t threads, uint64_t *median) {
    uint64_t samples[REPEATS];
    unsigned repeat;
    for (repeat = 0; repeat < REPEATS; ++repeat) {
        const uint64_t begin = now_ns();
        const int status = threads == 0
                ? q8q8_buffer_load(destination, source, 0, bytes)
                : rb_ggml_copy_from_uio_fast(destination, source, 0, bytes,
                                             threads);
        const uint64_t end = now_ns();
        if (status != 0 || begin == 0 || end < begin) return -1;
        samples[repeat] = end - begin;
        printf("OUTPUT_PATH_SAMPLE operation=%s repeat=%u bytes=%zu "
               "elapsed_ns=%" PRIu64 "\n", operation, repeat, bytes,
               samples[repeat]);
    }
    qsort(samples, REPEATS, sizeof(samples[0]), compare_u64);
    *median = samples[REPEATS / 2];
    return 0;
}

int main(void) {
    struct q8q8_uio uio;
    size_t index;
    if (q8q8_uio_open(&uio, "q8q8-gemm") != Q8Q8_RT_OK) {
        fputs("OUTPUT_PATH_FAIL stage=uio_open\n", stderr);
        return 1;
    }
    printf("OUTPUT_PATH_BEGIN version=1 repeats=%u source=uio_noncached "
           "pl_control_access=no sd_qspi_access=no\n", REPEATS);
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        const size_t bytes = (size_t)cases[index].m * cases[index].n * sizeof(float);
        uint8_t *reference = malloc(bytes);
        uint8_t *fast = malloc(bytes);
        struct q8q8_buffer source = {
            uio.arena, uio.arena_phys, 0, bytes
        };
        uint64_t reference_ns;
        uint64_t fast1_ns;
        uint64_t fast4_ns;
        size_t byte;
        if (bytes > uio.arena_size || reference == NULL || fast == NULL) {
            free(fast);
            free(reference);
            q8q8_uio_close(&uio);
            return 1;
        }
        for (byte = 0; byte < bytes; ++byte)
            ((volatile uint8_t *)uio.arena)[byte] = (uint8_t)(byte * 17u + 23u);
        if (measure("reference", reference, &source, bytes, 0, &reference_ns) != 0 ||
                measure("fast1", fast, &source, bytes, 1, &fast1_ns) != 0 ||
                memcmp(reference, fast, bytes) != 0 ||
                measure("fast4", fast, &source, bytes, 4, &fast4_ns) != 0 ||
                memcmp(reference, fast, bytes) != 0) {
            printf("OUTPUT_PATH_FAIL case=%s\n", cases[index].name);
            free(fast);
            free(reference);
            q8q8_uio_close(&uio);
            return 1;
        }
        printf("OUTPUT_PATH_RESULT case=%s m=%u n=%u bytes=%zu "
               "reference_ns=%" PRIu64 " fast1_ns=%" PRIu64
               " fast4_ns=%" PRIu64 " fast1_speedup=%.6f "
               "fast4_speedup=%.6f byte_equal=yes result=PASS\n",
               cases[index].name, cases[index].m, cases[index].n, bytes,
               reference_ns, fast1_ns, fast4_ns,
               (double)reference_ns / fast1_ns,
               (double)reference_ns / fast4_ns);
        free(fast);
        free(reference);
    }
    q8q8_uio_close(&uio);
    puts("OUTPUT_PATH_PASS cases=3 byte_equal=yes pl_control_access=no sd_qspi_written=no");
    return 0;
}
