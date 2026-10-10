#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double) value.tv_sec + (double) value.tv_nsec * 1e-9;
}

static void report(const char * name, uint64_t bytes, double seconds, uint64_t checksum) {
    printf("DDR_BW operation=%s bytes=%" PRIu64 " seconds=%.6f GBps=%.3f checksum=%" PRIu64 "\n",
           name, bytes, seconds, (double) bytes / seconds / 1e9, checksum);
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? (size_t) strtoull(argv[1], NULL, 10) : 256;
    const size_t loops = argc > 2 ? (size_t) strtoull(argv[2], NULL, 10) : 8;
    const size_t bytes = mib * 1024 * 1024;
    uint64_t * a = NULL;
    uint64_t * b = NULL;
    if (mib == 0 || loops == 0 || posix_memalign((void **) &a, 64, bytes) != 0 ||
            posix_memalign((void **) &b, 64, bytes) != 0) {
        fprintf(stderr, "allocation failed: %s\n", strerror(errno));
        free(a);
        free(b);
        return 2;
    }

    memset(a, 0x5a, bytes);
    memset(b, 0xa5, bytes);

    double start = now_seconds();
    for (size_t loop = 0; loop < loops; ++loop) {
        memcpy(b, a, bytes);
        __asm__ volatile("" ::: "memory");
        a[loop % (bytes / sizeof(*a))] ^= (uint64_t) loop;
    }
    double elapsed = now_seconds() - start;
    report("copy", (uint64_t) bytes * loops * 2, elapsed, b[0]);

    uint64_t sum = 0;
    start = now_seconds();
    for (size_t loop = 0; loop < loops; ++loop) {
        for (size_t index = 0; index < bytes / sizeof(*a); ++index) {
            sum += a[index];
        }
        __asm__ volatile("" : "+r"(sum) :: "memory");
    }
    elapsed = now_seconds() - start;
    report("read", (uint64_t) bytes * loops, elapsed, sum);

    start = now_seconds();
    for (size_t loop = 0; loop < loops; ++loop) {
        memset(a, (int) loop, bytes);
        __asm__ volatile("" ::: "memory");
    }
    elapsed = now_seconds() - start;
    report("write", (uint64_t) bytes * loops, elapsed, a[0]);

    free(a);
    free(b);
    return 0;
}
