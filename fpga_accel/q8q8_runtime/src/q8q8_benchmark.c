#define _POSIX_C_SOURCE 200809L
#include "q8q8_arena.h"
#include "q8q8_apm.h"
#include "q8q8_dispatch.h"
#include "q8q8_runtime.h"
#include "q8q8_uio.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    QK = 32,
    TM = 16,
    TN = 64,
    CLOCK_MHZ = 200,
    SAMPLE_CAPACITY = 16,
};

struct benchmark_case {
    const char *name;
    uint32_t total_m;
    uint32_t chunk_m;
    uint32_t n;
    uint32_t k;
};

struct benchmark_buffers {
    struct q8q8_buffer weights_q;
    struct q8q8_buffer weights_scale;
    struct q8q8_buffer activations_q;
    struct q8q8_buffer activations_scale;
    struct q8q8_buffer output;
};

static const struct benchmark_case cases[] = {
    /* Frozen one-launch shape used for PE8/PE16/DATAFLOW kernel comparison. */
    {"language_prefill", 512, 512, 9728, 2560},
    /* Full representative image-prompt prefill used for chunk-policy A/B. */
    {"visual_prefill", 1080, 512, 9728, 2560},
    {"vision_encoder", 4320, 512, 4096, 1024},
    {"decoder_gemv", 1, 1, 9728, 2560},
};

static uint64_t now_us(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * UINT64_C(1000000) +
            (uint64_t)now.tv_nsec / UINT64_C(1000);
}

static int checked_mul(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) return Q8Q8_RT_ERR_ARGUMENT;
    *result = a * b;
    return Q8Q8_RT_OK;
}

static int allocate_buffers(struct q8q8_arena *arena, uint32_t max_m,
                            uint32_t n, uint32_t k,
                            struct benchmark_buffers *buffers) {
    size_t k_blocks = k / QK;
    size_t weight_blocks;
    size_t activation_blocks;
    size_t output_values;
    size_t bytes;
    if (checked_mul(n, k_blocks, &weight_blocks) != Q8Q8_RT_OK ||
            checked_mul(max_m, k_blocks, &activation_blocks) != Q8Q8_RT_OK ||
            checked_mul(max_m, n, &output_values) != Q8Q8_RT_OK) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if (checked_mul(weight_blocks, 32, &bytes) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, bytes, 64, &buffers->weights_q) != Q8Q8_RT_OK ||
            checked_mul(weight_blocks, 2, &bytes) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, bytes, 64, &buffers->weights_scale) != Q8Q8_RT_OK ||
            checked_mul(activation_blocks, 32, &bytes) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, bytes, 64, &buffers->activations_q) != Q8Q8_RT_OK ||
            checked_mul(activation_blocks, 2, &bytes) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, bytes, 64, &buffers->activations_scale) != Q8Q8_RT_OK ||
            checked_mul(output_values, sizeof(float), &bytes) != Q8Q8_RT_OK ||
            q8q8_arena_alloc(arena, bytes, 64, &buffers->output) != Q8Q8_RT_OK) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    return Q8Q8_RT_OK;
}

static int8_t weight_value(uint32_t row, uint32_t column) {
    return (int8_t)((int)((row * 3u + column * 5u) % 7u) - 3);
}

static int8_t activation_value(uint32_t row, uint32_t column) {
    return (int8_t)((int)((row * 5u + column * 3u) % 5u) - 2);
}

static int prepare_weights(const struct q8q8_buffer *quant,
                           const struct q8q8_buffer *scale,
                           uint32_t n, uint32_t k) {
    size_t index = 0;
    uint32_t row;
    uint32_t column;
    const uint16_t one = UINT16_C(0x3c00);
    for (row = 0; row < n; ++row) {
        for (column = 0; column < k; ++column) {
            quant->data[index++] = (uint8_t)weight_value(row, column);
        }
    }
    for (index = 0; index < (size_t)n * (k / QK); ++index) {
        if (q8q8_buffer_store(scale, index * 2, &one, sizeof(one)) != Q8Q8_RT_OK) {
            return Q8Q8_RT_ERR_ARGUMENT;
        }
    }
    return Q8Q8_RT_OK;
}

static int prepare_activations(const struct q8q8_buffer *quant,
                               const struct q8q8_buffer *scale,
                               uint32_t global_m0, uint32_t m, uint32_t k) {
    size_t index = 0;
    uint32_t row;
    uint32_t column;
    const uint16_t one = UINT16_C(0x3c00);
    for (row = 0; row < m; ++row) {
        for (column = 0; column < k; ++column) {
            quant->data[index++] = (uint8_t)activation_value(global_m0 + row,
                                                               column);
        }
    }
    for (index = 0; index < (size_t)m * (k / QK); ++index) {
        if (q8q8_buffer_store(scale, index * 2, &one, sizeof(one)) != Q8Q8_RT_OK) {
            return Q8Q8_RT_ERR_ARGUMENT;
        }
    }
    return Q8Q8_RT_OK;
}

static float reference_value(uint32_t m, uint32_t n, uint32_t k) {
    int64_t sum = 0;
    uint32_t column;
    for (column = 0; column < k; ++column) {
        sum += (int)activation_value(m, column) *
                (int)weight_value(n, column);
    }
    return (float)sum;
}

static int check_samples(const struct q8q8_buffer *output, uint32_t global_m0,
                         uint32_t m, uint32_t n, uint32_t k,
                         float *max_abs_error, uint32_t *sample_count) {
    uint32_t rows[4];
    uint32_t columns[4];
    uint32_t row_count = 0;
    uint32_t column_count = 0;
    uint32_t ri;
    uint32_t ci;
    float maximum = 0.0f;
    rows[row_count++] = 0;
    if (m > 1) rows[row_count++] = m / 2;
    if (m > 2) rows[row_count++] = m - 1;
    columns[column_count++] = 0;
    if (n > 1) columns[column_count++] = 1;
    if (n > 2) columns[column_count++] = n / 2;
    if (n > 3) columns[column_count++] = n - 1;
    *sample_count = 0;
    for (ri = 0; ri < row_count; ++ri) {
        for (ci = 0; ci < column_count; ++ci) {
            const size_t index = (size_t)rows[ri] * n + columns[ci];
            const float expected = reference_value(global_m0 + rows[ri],
                                                   columns[ci], k);
            float actual;
            float error;
            if (*sample_count >= SAMPLE_CAPACITY ||
                    q8q8_buffer_load(&actual, output,
                                     index * sizeof(float), sizeof(actual)) !=
                            Q8Q8_RT_OK) {
                return Q8Q8_RT_ERR_ARGUMENT;
            }
            error = fabsf(actual - expected);
            if (!isfinite(actual) || error > 1.0e-4f) {
                fprintf(stderr,
                        "BENCH_MISMATCH m=%u n=%u got=%.9g expected=%.9g error=%.9g\n",
                        global_m0 + rows[ri], columns[ci], actual, expected,
                        error);
                return Q8Q8_RT_ERR_HARDWARE;
            }
            if (error > maximum) maximum = error;
            ++*sample_count;
        }
    }
    *max_abs_error = maximum;
    return Q8Q8_RT_OK;
}

static double component_model_us(uint32_t m, uint32_t n, uint32_t k) {
    const uint64_t m_tiles = (m + TM - 1u) / TM;
    const uint64_t n_tiles = (n + TN - 1u) / TN;
    const uint64_t k_blocks = k / QK;
    /* v2 csynth component latencies at II=1.  Summing the components is a
       conservative no-overlap model; wall time remains the decisive metric. */
    const uint64_t weight_cycles = UINT64_C(64) * k_blocks + 14;
    const uint64_t activation_cycles = UINT64_C(16) * k_blocks + 14;
    const uint64_t compute_cycles = UINT64_C(128) * k_blocks + 25;
    const uint64_t init_cycles = 1026;
    const uint64_t store_cycles = 1034;
    const uint64_t cycles = n_tiles * (weight_cycles + m_tiles *
            (activation_cycles + init_cycles + compute_cycles +
             store_cycles));
    return (double)cycles / CLOCK_MHZ;
}

static double benchmark_component_model_us(
        const struct benchmark_case *benchmark) {
    double total = 0.0;
    uint32_t global_m0;
    for (global_m0 = 0; global_m0 < benchmark->total_m;
         global_m0 += benchmark->chunk_m) {
        uint32_t m = benchmark->total_m - global_m0;
        if (m > benchmark->chunk_m) m = benchmark->chunk_m;
        total += component_model_us(m, benchmark->n, benchmark->k);
    }
    return total;
}

static void modeled_ddr_bytes(uint32_t m, uint32_t n, uint32_t k,
                              uint64_t *read_bytes,
                              uint64_t *write_bytes) {
    const uint64_t k_blocks = k / QK;
    const uint64_t n_tiles = (n + TN - 1u) / TN;
    const uint64_t weight_bytes = (uint64_t)n * k +
            (uint64_t)n * k_blocks * 2;
    const uint64_t activation_bytes = (uint64_t)m * k +
            (uint64_t)m * k_blocks * 2;
    *read_bytes = weight_bytes + n_tiles * activation_bytes;
    *write_bytes = (uint64_t)m * n * sizeof(float);
}

static const struct benchmark_case *find_case(const char *name) {
    size_t index;
    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        if (strcmp(cases[index].name, name) == 0) return &cases[index];
    }
    return NULL;
}

static int parse_positive(const char *text, uint32_t *value) {
    char *end = NULL;
    unsigned long parsed;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
            parsed > UINT32_MAX) return -1;
    *value = (uint32_t)parsed;
    return 0;
}

static int benchmark_self_test(void) {
    uint8_t storage[16384];
    struct q8q8_arena arena;
    struct benchmark_buffers buffers;
    const uint32_t m = 3;
    const uint32_t n = 5;
    const uint32_t k = 64;
    uint32_t row;
    uint32_t column;
    float max_abs_error = 0.0f;
    uint32_t samples = 0;
    if (q8q8_arena_init(&arena, storage, sizeof(storage),
                        UINT64_C(0xb7c000000)) != Q8Q8_RT_OK ||
            allocate_buffers(&arena, m, n, k, &buffers) != Q8Q8_RT_OK ||
            prepare_weights(&buffers.weights_q, &buffers.weights_scale,
                            n, k) != Q8Q8_RT_OK ||
            prepare_activations(&buffers.activations_q,
                                &buffers.activations_scale, 0, m, k) !=
                    Q8Q8_RT_OK) {
        fprintf(stderr, "BENCH_SELFTEST_FAIL stage=prepare\n");
        return 1;
    }
    for (row = 0; row < m; ++row) {
        for (column = 0; column < n; ++column) {
            const float value = reference_value(row, column, k);
            if (q8q8_buffer_store(&buffers.output,
                                  ((size_t)row * n + column) * sizeof(float),
                                  &value, sizeof(value)) != Q8Q8_RT_OK) {
                fprintf(stderr, "BENCH_SELFTEST_FAIL stage=golden_store\n");
                return 1;
            }
        }
    }
    if (check_samples(&buffers.output, 0, m, n, k, &max_abs_error,
                      &samples) != Q8Q8_RT_OK || max_abs_error != 0.0f ||
            samples != 12 || !(component_model_us(m, n, k) > 0.0)) {
        fprintf(stderr,
                "BENCH_SELFTEST_FAIL stage=compare samples=%u max_abs=%.9g\n",
                samples, max_abs_error);
        return 1;
    }
    printf("BENCH_SELFTEST_PASS m=%u n=%u k=%u samples=%u max_abs=%.9g used_bytes=%zu\n",
           m, n, k, samples, max_abs_error, arena.used);
    return 0;
}

static int run_benchmark(struct q8q8_uio *uio,
                         const struct benchmark_case *benchmark,
                         uint32_t repeats, uint64_t timeout_us,
                         struct q8q8_apm *apm) {
    struct q8q8_arena arena;
    struct benchmark_buffers buffers;
    uint64_t weight_prepare_begin;
    uint64_t weight_prepare_us;
    uint64_t total_kernel_us = 0;
    uint64_t total_prepare_us = 0;
    uint64_t total_read_bytes = 0;
    uint64_t total_write_bytes = 0;
    uint64_t total_apm_cycles = 0;
    uint64_t total_apm_counters[10] = {0};
    uint32_t repeat;
    int status;

    status = q8q8_arena_init(&arena, uio->arena, uio->arena_size,
                             uio->arena_phys);
    if (status != Q8Q8_RT_OK ||
            allocate_buffers(&arena, benchmark->chunk_m, benchmark->n,
                             benchmark->k, &buffers) != Q8Q8_RT_OK) {
        fprintf(stderr,
                "BENCH_FAIL stage=allocate case=%s arena_bytes=%zu requested_chunk_m=%u\n",
                benchmark->name, uio->arena_size, benchmark->chunk_m);
        return 1;
    }
    printf("BENCH_LAYOUT case=%s arena_bytes=%zu used_bytes=%zu wq_bytes=%zu ws_bytes=%zu xq_bytes=%zu xs_bytes=%zu output_bytes=%zu\n",
           benchmark->name, uio->arena_size, arena.used,
           buffers.weights_q.size, buffers.weights_scale.size,
           buffers.activations_q.size, buffers.activations_scale.size,
           buffers.output.size);

    weight_prepare_begin = now_us();
    status = prepare_weights(&buffers.weights_q, &buffers.weights_scale,
                             benchmark->n, benchmark->k);
    weight_prepare_us = now_us() - weight_prepare_begin;
    if (status != Q8Q8_RT_OK) return 1;
    printf("BENCH_STAGE case=%s name=weight_prepare state=done elapsed_us=%" PRIu64 "\n",
           benchmark->name, weight_prepare_us);

    for (repeat = 0; repeat < repeats; ++repeat) {
        uint32_t global_m0;
        uint64_t repeat_kernel_us = 0;
        uint64_t repeat_prepare_us = 0;
        uint64_t repeat_read_bytes = 0;
        uint64_t repeat_write_bytes = 0;
        uint32_t chunks = 0;
        for (global_m0 = 0; global_m0 < benchmark->total_m;
             global_m0 += benchmark->chunk_m) {
            struct q8q8_job job;
            struct q8q8_result result;
            uint32_t m = benchmark->total_m - global_m0;
            uint64_t begin;
            float max_abs_error;
            uint32_t samples;
            uint64_t chunk_read_bytes;
            uint64_t chunk_write_bytes;
            struct q8q8_apm_snapshot apm_snapshot;
            const struct q8q8_apm_metric *apm_metrics = NULL;
            unsigned int apm_index;
            if (m > benchmark->chunk_m) m = benchmark->chunk_m;
            begin = now_us();
            status = prepare_activations(&buffers.activations_q,
                                         &buffers.activations_scale,
                                         global_m0, m, benchmark->k);
            q8q8_buffer_fill(&buffers.output, UINT8_C(0xa5));
            repeat_prepare_us += now_us() - begin;
            if (status != Q8Q8_RT_OK) return 1;

            memset(&job, 0, sizeof(job));
            memset(&result, 0, sizeof(result));
            job.weights_q = buffers.weights_q.phys;
            job.weights_scale = buffers.weights_scale.phys;
            job.activations_q = buffers.activations_q.phys;
            job.activations_scale = buffers.activations_scale.phys;
            job.output = buffers.output.phys;
            job.m = m;
            job.n = benchmark->n;
            job.k = benchmark->k;
            printf("BENCH_STAGE case=%s repeat=%u chunk=%u name=accelerator state=begin global_m0=%u m=%u n=%u k=%u\n",
                   benchmark->name, repeat + 1, chunks + 1, global_m0,
                   m, benchmark->n, benchmark->k);
            if (apm != NULL) {
                apm_metrics = q8q8_apm_profile_metrics(apm->profile);
                if (q8q8_apm_start(apm) != 0) {
                    fprintf(stderr, "BENCH_FAIL case=%s stage=apm_start\n",
                            benchmark->name);
                    return 1;
                }
            }
            status = q8q8_run(&uio->device, &job, timeout_us, &result);
            if (apm != NULL && q8q8_apm_stop(apm, &apm_snapshot) != 0) {
                fprintf(stderr, "BENCH_FAIL case=%s stage=apm_stop\n",
                        benchmark->name);
                return 1;
            }
            printf("BENCH_STAGE case=%s repeat=%u chunk=%u name=accelerator state=done status=%d hardware_status=%d elapsed_us=%" PRIu64 " control=0x%08x\n",
                   benchmark->name, repeat + 1, chunks + 1, status,
                   result.hardware_status, result.elapsed_us,
                   result.final_control);
            if (status != Q8Q8_RT_OK) {
                fprintf(stderr, "BENCH_FAIL case=%s stage=accelerator reason=%s\n",
                        benchmark->name, q8q8_runtime_status_string(status));
                return 1;
            }
            if (apm != NULL) {
                const double apm_clock_mhz = result.elapsed_us == 0 ? 0.0 :
                        (double)apm_snapshot.global_cycles /
                        (double)result.elapsed_us;
                printf("APM_SAMPLE case=%s repeat=%u chunk=%u profile=%s global_cycles=%u overflow=0x%08x derived_clock_mhz=%.6f\n",
                       benchmark->name, repeat + 1, chunks + 1,
                       q8q8_apm_profile_name(apm->profile),
                       apm_snapshot.global_cycles,
                       apm_snapshot.overflow_status, apm_clock_mhz);
                if (apm_snapshot.overflow_status != 0) {
                    fprintf(stderr,
                            "BENCH_FAIL case=%s stage=apm_overflow status=0x%08x\n",
                            benchmark->name, apm_snapshot.overflow_status);
                    return 1;
                }
                total_apm_cycles += apm_snapshot.global_cycles;
                for (apm_index = 0; apm_index < 10; ++apm_index) {
                    printf("APM_COUNTER case=%s repeat=%u chunk=%u counter=%u slot=%u metric=%u name=%s value=%u\n",
                           benchmark->name, repeat + 1, chunks + 1,
                           apm_index, apm_metrics[apm_index].slot,
                           apm_metrics[apm_index].metric,
                           apm_metrics[apm_index].name,
                           apm_snapshot.counters[apm_index]);
                    if (apm_metrics[apm_index].metric == 15) {
                        if (apm_snapshot.counters[apm_index] >
                                total_apm_counters[apm_index]) {
                            total_apm_counters[apm_index] =
                                    apm_snapshot.counters[apm_index];
                        }
                    } else {
                        total_apm_counters[apm_index] +=
                                apm_snapshot.counters[apm_index];
                    }
                }
            }
            modeled_ddr_bytes(m, benchmark->n, benchmark->k,
                              &chunk_read_bytes, &chunk_write_bytes);
            repeat_read_bytes += chunk_read_bytes;
            repeat_write_bytes += chunk_write_bytes;
            printf("BENCH_COUNTER case=%s repeat=%u chunk=%u pl_cycles_200mhz=%" PRIu64 " modeled_ddr_read_bytes=%" PRIu64 " modeled_ddr_write_bytes=%" PRIu64 " modeled_bandwidth_gbs=%.6f weight_reuse_m_tiles=%u\n",
                   benchmark->name, repeat + 1, chunks + 1,
                   result.elapsed_us * CLOCK_MHZ, chunk_read_bytes,
                   chunk_write_bytes,
                   (double)(chunk_read_bytes + chunk_write_bytes) /
                           ((double)result.elapsed_us * 1000.0),
                   (m + TM - 1u) / TM);
            status = check_samples(&buffers.output, global_m0, m,
                                   benchmark->n, benchmark->k,
                                   &max_abs_error, &samples);
            if (status != Q8Q8_RT_OK) {
                fprintf(stderr,
                        "BENCH_FAIL case=%s stage=compare global_m0=%u\n",
                        benchmark->name, global_m0);
                return 1;
            }
            printf("BENCH_STAGE case=%s repeat=%u chunk=%u name=compare state=done samples=%u max_abs=%.9g\n",
                   benchmark->name, repeat + 1, chunks + 1, samples,
                   max_abs_error);
            repeat_kernel_us += result.elapsed_us;
            ++chunks;
        }
        total_kernel_us += repeat_kernel_us;
        total_prepare_us += repeat_prepare_us;
        total_read_bytes += repeat_read_bytes;
        total_write_bytes += repeat_write_bytes;
        printf("BENCH_REPEAT case=%s repeat=%u chunks=%u kernel_us=%" PRIu64 " prepare_us=%" PRIu64 "\n",
               benchmark->name, repeat + 1, chunks, repeat_kernel_us,
               repeat_prepare_us);
    }

    {
        const double operations = 2.0 * benchmark->total_m * benchmark->n *
                benchmark->k;
        const double average_kernel_us = (double)total_kernel_us / repeats;
        const double effective_gops = operations / (average_kernel_us * 1000.0);
        const double component_model = benchmark_component_model_us(benchmark);
        const double average_read_bytes = (double)total_read_bytes / repeats;
        const double average_write_bytes = (double)total_write_bytes / repeats;
        printf("BENCH_RESULT case=%s m=%u n=%u k=%u chunk_m=%u repeats=%u average_kernel_us=%.3f component_model_us=%.3f model_ratio=%.6f effective_gops=%.6f average_prepare_us=%.3f weight_prepare_us=%" PRIu64 " pl_cycles_200mhz=%.0f modeled_ddr_read_bytes=%.0f modeled_ddr_write_bytes=%.0f modeled_bandwidth_gbs=%.6f result=PASS\n",
               benchmark->name, benchmark->total_m, benchmark->n,
               benchmark->k, benchmark->chunk_m, repeats,
               average_kernel_us, component_model,
               average_kernel_us / component_model,
               effective_gops, (double)total_prepare_us / repeats,
               weight_prepare_us, average_kernel_us * CLOCK_MHZ,
               average_read_bytes, average_write_bytes,
                       (average_read_bytes + average_write_bytes) /
                       (average_kernel_us * 1000.0));
        if (apm != NULL) {
            const struct q8q8_apm_metric *metrics =
                    q8q8_apm_profile_metrics(apm->profile);
            unsigned int index;
            printf("APM_RESULT case=%s profile=%s repeats=%u global_cycles=%" PRIu64 " derived_clock_mhz=%.6f\n",
                   benchmark->name, q8q8_apm_profile_name(apm->profile),
                   repeats, total_apm_cycles,
                   total_kernel_us == 0 ? 0.0 :
                           (double)total_apm_cycles /
                           (double)total_kernel_us);
            for (index = 0; index < 10; ++index) {
                printf("APM_TOTAL case=%s profile=%s counter=%u slot=%u metric=%u name=%s value=%" PRIu64 " aggregation=%s\n",
                       benchmark->name,
                       q8q8_apm_profile_name(apm->profile), index,
                       metrics[index].slot, metrics[index].metric,
                       metrics[index].name, total_apm_counters[index],
                       metrics[index].metric == 15 ? "max" : "sum");
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *name = "q8q8-gemm";
    const char *case_name = "language_prefill";
    uint32_t repeats = 1;
    uint32_t timeout_ms = 120000;
    uint32_t requested_chunk_m = 0;
    int auto_chunk = 0;
    int apm_enabled = 0;
    enum q8q8_apm_profile apm_profile = Q8Q8_APM_PROFILE_TRAFFIC;
    const struct benchmark_case *selected_benchmark;
    struct benchmark_case benchmark;
    struct q8q8_uio uio;
    struct q8q8_apm apm;
    int status;
    int index;
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return benchmark_self_test();
    }
    for (index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--case") == 0 && index + 1 < argc) {
            case_name = argv[++index];
        } else if (strcmp(argv[index], "--repeat") == 0 && index + 1 < argc) {
            if (parse_positive(argv[++index], &repeats) != 0) return 2;
        } else if (strcmp(argv[index], "--timeout-ms") == 0 && index + 1 < argc) {
            if (parse_positive(argv[++index], &timeout_ms) != 0) return 2;
        } else if (strcmp(argv[index], "--chunk-m") == 0 && index + 1 < argc) {
            if (parse_positive(argv[++index], &requested_chunk_m) != 0) return 2;
        } else if (strcmp(argv[index], "--auto-chunk") == 0) {
            auto_chunk = 1;
        } else if (strcmp(argv[index], "--uio-name") == 0 && index + 1 < argc) {
            name = argv[++index];
        } else if (strcmp(argv[index], "--apm") == 0 && index + 1 < argc) {
            const char *profile = argv[++index];
            apm_enabled = 1;
            if (strcmp(profile, "traffic") == 0) {
                apm_profile = Q8Q8_APM_PROFILE_TRAFFIC;
            } else if (strcmp(profile, "latency") == 0) {
                apm_profile = Q8Q8_APM_PROFILE_LATENCY;
            } else {
                fprintf(stderr, "BENCH_FAIL invalid_apm_profile=%s\n", profile);
                return 2;
            }
        } else {
            fprintf(stderr,
                    "usage: %s [--case language_prefill|visual_prefill|vision_encoder|decoder_gemv] [--repeat N] [--timeout-ms N] [--chunk-m N|--auto-chunk] [--uio-name NAME] [--apm traffic|latency]\n",
                    argv[0]);
            return 2;
        }
    }
    if (auto_chunk && requested_chunk_m != 0) {
        fprintf(stderr, "BENCH_FAIL chunk_policy_conflict=1\n");
        return 2;
    }
    selected_benchmark = find_case(case_name);
    if (selected_benchmark == NULL) {
        fprintf(stderr, "BENCH_FAIL unknown_case=%s\n", case_name);
        return 2;
    }
    benchmark = *selected_benchmark;
    if (requested_chunk_m != 0) {
        if (requested_chunk_m > benchmark.total_m || requested_chunk_m > 4320) {
            fprintf(stderr, "BENCH_FAIL invalid_chunk_m=%u\n", requested_chunk_m);
            return 2;
        }
        benchmark.chunk_m = requested_chunk_m;
    }
    status = q8q8_uio_open(&uio, name);
    if (status != Q8Q8_RT_OK) {
        fprintf(stderr, "BENCH_FAIL stage=uio_open status=%d reason=%s\n",
                status, q8q8_runtime_status_string(status));
        return 1;
    }
    if (auto_chunk) {
        struct q8q8_workload_descriptor descriptor = {
            Q8Q8_OP_MATMUL_NT,
            Q8Q8_LAYOUT_GGML_Q8_0,
            Q8Q8_LAYOUT_GGML_Q8_0,
            Q8Q8_LAYOUT_ROW_MAJOR_F32,
            benchmark.total_m, benchmark.n, benchmark.k,
        };
        struct q8q8_dispatch_plan plan;
        status = q8q8_plan_workload(&descriptor, uio.arena_size, 64, &plan);
        if (status != Q8Q8_RT_OK) {
            fprintf(stderr,
                    "BENCH_FAIL stage=auto_chunk arena_bytes=%zu status=%d\n",
                    uio.arena_size, status);
            q8q8_uio_close(&uio);
            return 1;
        }
        benchmark.chunk_m = plan.chunk_m;
        printf("BENCH_CHUNK_POLICY mode=auto chunk_m=%u chunks=%u planned_bytes=%zu arena_bytes=%zu\n",
               plan.chunk_m, plan.chunk_count, plan.arena_bytes,
               uio.arena_size);
    } else {
        printf("BENCH_CHUNK_POLICY mode=%s chunk_m=%u\n",
               requested_chunk_m == 0 ? "frozen_default" : "manual",
               benchmark.chunk_m);
    }
    printf("BENCH_BEGIN case=%s total_m=%u chunk_m=%u n=%u k=%u repeats=%u\n",
           benchmark.name, benchmark.total_m, benchmark.chunk_m,
           benchmark.n, benchmark.k, repeats);
    memset(&apm, 0, sizeof(apm));
    apm.fd = -1;
    if (apm_enabled && (q8q8_apm_open_ddr(&apm) != 0 ||
            q8q8_apm_configure(&apm, apm_profile) != 0)) {
        fprintf(stderr, "BENCH_FAIL stage=apm_open_or_configure\n");
        q8q8_apm_close(&apm);
        q8q8_uio_close(&uio);
        return 1;
    }
    status = run_benchmark(&uio, &benchmark, repeats,
                           (uint64_t)timeout_ms * 1000u,
                           apm_enabled ? &apm : NULL);
    q8q8_apm_close(&apm);
    q8q8_uio_close(&uio);
    return status;
}
