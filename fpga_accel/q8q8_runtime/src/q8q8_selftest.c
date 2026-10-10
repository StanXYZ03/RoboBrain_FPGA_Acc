#include "q8q8_smoke.h"
#include "q8q8_uio.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *name = "q8q8-gemm";
    uint64_t timeout_us = 5000000;
    struct q8q8_uio uio;
    struct q8q8_smoke_vector vector;
    struct q8q8_result result;
    float max_abs = 0.0f;
    int status;
    int index;

    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--uio-name") == 0 && index + 1 < argc) {
            name = argv[++index];
        } else if (strcmp(argv[index], "--timeout-ms") == 0 && index + 1 < argc) {
            char *end = NULL;
            unsigned long value;
            errno = 0;
            value = strtoul(argv[++index], &end, 10);
            if (errno != 0 || end == argv[index] || *end != '\0' || value == 0) {
                fprintf(stderr, "SELFTEST_FAIL invalid timeout\n");
                return 2;
            }
            timeout_us = (uint64_t)value * 1000u;
        } else {
            fprintf(stderr, "usage: %s [--uio-name NAME] [--timeout-ms N]\n",
                    argv[0]);
            return 2;
        }
    }

    printf("SELFTEST_STAGE name=uio_open state=begin\n");
    status = q8q8_uio_open(&uio, name);
    if (status != Q8Q8_RT_OK) {
        fprintf(stderr, "SELFTEST_FAIL stage=uio_open status=%d reason=%s\n",
                status, q8q8_runtime_status_string(status));
        return 1;
    }
    printf("SELFTEST_STAGE name=uio_open state=done control=0x%llx arena=0x%llx bytes=%zu\n",
           (unsigned long long)uio.control_phys,
           (unsigned long long)uio.arena_phys, uio.arena_size);
    if (uio.control_phys != UINT64_C(0xa0000000)) {
        fprintf(stderr, "SELFTEST_FAIL unexpected control address 0x%llx\n",
                (unsigned long long)uio.control_phys);
        q8q8_uio_close(&uio);
        return 1;
    }

    printf("SELFTEST_STAGE name=prepare state=begin\n");
    status = q8q8_smoke_prepare(uio.arena, uio.arena_size, uio.arena_phys,
                                &vector);
    if (status != Q8Q8_RT_OK) {
        fprintf(stderr, "SELFTEST_FAIL stage=prepare status=%d\n", status);
        q8q8_uio_close(&uio);
        return 1;
    }
    printf("SELFTEST_STAGE name=prepare state=done bytes=%zu\n",
           vector.bytes_used);
    printf("SELFTEST_STAGE name=accelerator state=begin m=%u n=%u k=%u\n",
           vector.job.m, vector.job.n, vector.job.k);
    status = q8q8_run(&uio.device, &vector.job, timeout_us, &result);
    printf("SELFTEST_STAGE name=accelerator state=done runtime_status=%d hardware_status=%d elapsed_us=%llu control=0x%08x\n",
           status, result.hardware_status,
           (unsigned long long)result.elapsed_us, result.final_control);
    if (status == Q8Q8_RT_OK) {
        status = q8q8_smoke_check(uio.arena, &vector, 1.0e-5f, &max_abs);
    }
    q8q8_uio_close(&uio);
    if (status != Q8Q8_RT_OK) {
        fprintf(stderr, "SELFTEST_FAIL stage=compare status=%d max_abs=%.9g\n",
                status, max_abs);
        return 1;
    }
    printf("SELFTEST_PASS m=%u n=%u k=%u outputs=%u max_abs=%.9g\n",
           vector.job.m, vector.job.n, vector.job.k,
           vector.job.m * vector.job.n, max_abs);
    return 0;
}
