#include "q8q8_dispatch.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "DISPATCH_TEST_FAIL line=%d check=%s\n", \
                __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static struct q8q8_workload_descriptor workload(uint32_t m, uint32_t n,
                                                 uint32_t k) {
    struct q8q8_workload_descriptor result = {
        Q8Q8_OP_MATMUL_NT,
        Q8Q8_LAYOUT_GGML_Q8_0,
        Q8Q8_LAYOUT_GGML_Q8_0,
        Q8Q8_LAYOUT_ROW_MAJOR_F32,
        m, n, k,
    };
    return result;
}

static int test_q8_0_split(void) {
    struct q8q8_block_q8_0 source[4];
    uint8_t quantized_storage[128];
    uint8_t scale_storage[8];
    struct q8q8_buffer quantized = {quantized_storage, 0x1000, 0,
                                    sizeof(quantized_storage)};
    struct q8q8_buffer scales = {scale_storage, 0x2000, 0,
                                sizeof(scale_storage)};
    size_t block;
    size_t value;
    for (block = 0; block < 4; ++block) {
        source[block].scale_fp16 = (uint16_t)(0x3000u + block * 17u);
        for (value = 0; value < 32; ++value) {
            source[block].values[value] = (int8_t)(block * 32 + value - 64);
        }
    }
    CHECK(q8q8_pack_ggml_q8_0(source, 2, 64, &quantized, &scales) == Q8Q8_RT_OK);
    for (block = 0; block < 4; ++block) {
        uint16_t scale;
        memcpy(&scale, &scale_storage[block * 2], sizeof(scale));
        CHECK(scale == source[block].scale_fp16);
        CHECK(memcmp(&quantized_storage[block * 32], source[block].values, 32) == 0);
    }
    puts("DISPATCH_CASE name=ggml_q8_0_split result=PASS blocks=4");
    return 0;
}

static int test_representative_plans(void) {
    const size_t arena_bytes = 64u * 1024u * 1024u;
    struct q8q8_workload_descriptor language = workload(1080, 9728, 2560);
    struct q8q8_workload_descriptor vision = workload(4320, 4096, 1024);
    struct q8q8_workload_descriptor decoder = workload(1, 9728, 2560);
    struct q8q8_dispatch_plan plan;

    CHECK(q8q8_plan_workload(&language, arena_bytes, 64, &plan) == Q8Q8_RT_OK);
    CHECK(plan.chunk_m >= 512 && plan.chunk_m < language.m);
    CHECK(plan.chunk_m % 16 == 0 && plan.chunk_count == 2);
    CHECK(plan.arena_bytes <= arena_bytes);
    printf("DISPATCH_PLAN name=visual_prefill m=%u n=%u k=%u chunk_m=%u chunks=%u bytes=%zu\n",
           language.m, language.n, language.k, plan.chunk_m,
           plan.chunk_count, plan.arena_bytes);

    CHECK(q8q8_plan_workload(&vision, arena_bytes, 64, &plan) == Q8Q8_RT_OK);
    CHECK(plan.chunk_m > 512 && plan.chunk_m < vision.m);
    CHECK(plan.chunk_m % 16 == 0 && plan.chunk_count == 2);
    CHECK(plan.arena_bytes <= arena_bytes);
    printf("DISPATCH_PLAN name=vision_encoder m=%u n=%u k=%u chunk_m=%u chunks=%u bytes=%zu\n",
           vision.m, vision.n, vision.k, plan.chunk_m,
           plan.chunk_count, plan.arena_bytes);

    CHECK(q8q8_plan_workload(&decoder, arena_bytes, 64, &plan) == Q8Q8_RT_OK);
    CHECK(plan.chunk_m == 1 && plan.chunk_count == 1);
    CHECK(plan.arena_bytes <= arena_bytes);
    printf("DISPATCH_PLAN name=decoder_gemv m=%u n=%u k=%u chunk_m=%u chunks=%u bytes=%zu\n",
           decoder.m, decoder.n, decoder.k, plan.chunk_m,
           plan.chunk_count, plan.arena_bytes);
    return 0;
}

static int test_allocation_and_jobs(void) {
    uint8_t storage[1024 * 1024];
    struct q8q8_arena arena;
    struct q8q8_workload_descriptor descriptor = workload(37, 256, 128);
    struct q8q8_dispatch_plan plan;
    struct q8q8_dispatch_buffers buffers;
    struct q8q8_job job;
    uint32_t row_offset;
    uint32_t index;
    CHECK(q8q8_plan_workload(&descriptor, sizeof(storage), 64, &plan) == Q8Q8_RT_OK);
    CHECK(q8q8_arena_init(&arena, storage, sizeof(storage), 0xB7C000000ULL) == Q8Q8_RT_OK);
    CHECK(q8q8_allocate_plan(&arena, &plan, &buffers) == Q8Q8_RT_OK);
    CHECK(arena.used == plan.arena_bytes);
    CHECK((buffers.weights_q.phys & 63u) == 0);
    CHECK((buffers.weights_scale.phys & 63u) == 0);
    CHECK((buffers.activations_q.phys & 63u) == 0);
    CHECK((buffers.activations_scale.phys & 63u) == 0);
    CHECK((buffers.output.phys & 63u) == 0);
    for (index = 0; index < plan.chunk_count; ++index) {
        CHECK(q8q8_plan_job(&plan, &buffers, index, &job, &row_offset) == Q8Q8_RT_OK);
        CHECK(row_offset == index * plan.chunk_m);
        CHECK(job.m <= plan.chunk_m && job.n == 256 && job.k == 128);
    }
    puts("DISPATCH_CASE name=allocation_and_jobs result=PASS");
    return 0;
}

struct mock_accelerator {
    uint32_t regs[0x100 / 4];
    uint8_t *arena;
    size_t arena_size;
    uint64_t phys_base;
    uint64_t now_us;
};

static uint64_t mock_reg64(const struct mock_accelerator *mock,
                           uint32_t offset) {
    return mock->regs[offset / 4] |
            ((uint64_t)mock->regs[offset / 4 + 1] << 32);
}

static void *mock_pointer(struct mock_accelerator *mock, uint64_t address,
                          size_t size) {
    uint64_t offset;
    if (address < mock->phys_base) return NULL;
    offset = address - mock->phys_base;
    if (offset > mock->arena_size || size > mock->arena_size - offset) return NULL;
    return mock->arena + offset;
}

static void mock_compute(struct mock_accelerator *mock) {
    const uint32_t m = mock->regs[Q8Q8_REG_M / 4];
    const uint32_t n = mock->regs[Q8Q8_REG_N / 4];
    const uint32_t k = mock->regs[Q8Q8_REG_K / 4];
    int8_t *weights = mock_pointer(mock, mock_reg64(mock, Q8Q8_REG_WEIGHTS_Q),
                                   (size_t)n * k);
    int8_t *activations = mock_pointer(mock,
            mock_reg64(mock, Q8Q8_REG_ACTIVATIONS_Q), (size_t)m * k);
    uint16_t *weight_scales = mock_pointer(mock,
            mock_reg64(mock, Q8Q8_REG_WEIGHTS_SCALE),
            (size_t)n * (k / 32) * sizeof(uint16_t));
    uint16_t *activation_scales = mock_pointer(mock,
            mock_reg64(mock, Q8Q8_REG_ACTIVATIONS_SCALE),
            (size_t)m * (k / 32) * sizeof(uint16_t));
    float *output = mock_pointer(mock, mock_reg64(mock, Q8Q8_REG_OUTPUT),
                                 (size_t)m * n * sizeof(float));
    uint32_t mi;
    uint32_t ni;
    if (weights == NULL || activations == NULL || weight_scales == NULL ||
            activation_scales == NULL || output == NULL || k != 32) {
        mock->regs[Q8Q8_REG_AP_RETURN / 4] = (uint32_t)-1;
    } else {
        for (mi = 0; mi < m; ++mi) {
            for (ni = 0; ni < n; ++ni) {
                int32_t sum = 0;
                uint32_t lane;
                for (lane = 0; lane < k; ++lane) {
                    sum += weights[(size_t)ni * k + lane] *
                            activations[(size_t)mi * k + lane];
                }
                if (weight_scales[ni] != 0x3c00 ||
                        activation_scales[mi] != 0x3c00) {
                    mock->regs[Q8Q8_REG_AP_RETURN / 4] = (uint32_t)-2;
                    return;
                }
                output[(size_t)mi * n + ni] = (float)sum;
            }
        }
        mock->regs[Q8Q8_REG_AP_RETURN / 4] = 0;
    }
    mock->now_us += 7;
    mock->regs[Q8Q8_REG_AP_CTRL / 4] =
            Q8Q8_CTRL_DONE | Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
}

static uint32_t mock_read32(void *opaque, uint32_t offset) {
    struct mock_accelerator *mock = opaque;
    uint32_t value = mock->regs[offset / 4];
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_DONE) != 0) {
        mock->regs[offset / 4] &= ~Q8Q8_CTRL_DONE;
    }
    return value;
}

static void mock_write32(void *opaque, uint32_t offset, uint32_t value) {
    struct mock_accelerator *mock = opaque;
    mock->regs[offset / 4] = value;
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_START) != 0) {
        mock_compute(mock);
    }
}

static uint64_t mock_now_us(void *opaque) {
    return ((struct mock_accelerator *)opaque)->now_us;
}

static void mock_sleep_us(void *opaque, uint32_t usec) {
    ((struct mock_accelerator *)opaque)->now_us += usec;
}

static int mock_sync(void *opaque, uint64_t address, size_t size,
                     enum q8q8_sync_direction direction) {
    (void)direction;
    return mock_pointer(opaque, address, size) == NULL ? -1 : 0;
}

static void mock_fence(void *opaque) { (void)opaque; }

static int test_end_to_end_dispatch(void) {
    uint8_t storage[352];
    const uint64_t phys_base = UINT64_C(0xB7C000000);
    struct q8q8_workload_descriptor descriptor = workload(3, 4, 32);
    struct q8q8_dispatch_plan plan;
    struct q8q8_dispatch_buffers buffers;
    struct q8q8_arena arena;
    struct q8q8_block_q8_0 weights[4];
    struct q8q8_block_q8_0 activations[3];
    float output[12];
    struct mock_accelerator mock;
    struct q8q8_device device;
    struct q8q8_dispatch_stats stats;
    const struct q8q8_io_ops ops = {
        mock_read32, mock_write32, mock_now_us, mock_sleep_us,
        mock_sync, mock_fence,
    };
    size_t row;
    size_t column;
    CHECK(q8q8_plan_workload(&descriptor, sizeof(storage), 64, &plan) == Q8Q8_RT_OK);
    CHECK(plan.chunk_m == 2 && plan.chunk_count == 2);
    CHECK(q8q8_arena_init(&arena, storage, sizeof(storage), phys_base) == Q8Q8_RT_OK);
    CHECK(q8q8_allocate_plan(&arena, &plan, &buffers) == Q8Q8_RT_OK);
    for (row = 0; row < 4; ++row) {
        weights[row].scale_fp16 = 0x3c00;
        for (column = 0; column < 32; ++column) {
            weights[row].values[column] = (int8_t)((row + column) % 5 - 2);
        }
    }
    for (row = 0; row < 3; ++row) {
        activations[row].scale_fp16 = 0x3c00;
        for (column = 0; column < 32; ++column) {
            activations[row].values[column] = (int8_t)((row * 2 + column) % 7 - 3);
        }
    }
    CHECK(q8q8_pack_ggml_q8_0(weights, 4, 32, &buffers.weights_q,
                              &buffers.weights_scale) == Q8Q8_RT_OK);
    memset(&mock, 0, sizeof(mock));
    mock.arena = storage;
    mock.arena_size = sizeof(storage);
    mock.phys_base = phys_base;
    mock.regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
    q8q8_device_init(&device, &ops, &mock);
    CHECK(q8q8_execute_plan(&device, &plan, &buffers, activations, output,
                            1000, &stats) == Q8Q8_RT_OK);
    CHECK(stats.chunks_completed == 2 && stats.rows_completed == 3 &&
          stats.hardware_elapsed_us == 14);
    for (row = 0; row < 3; ++row) {
        for (column = 0; column < 4; ++column) {
            int32_t expected = 0;
            size_t lane;
            for (lane = 0; lane < 32; ++lane) {
                expected += activations[row].values[lane] *
                        weights[column].values[lane];
            }
            CHECK(output[row * 4 + column] == (float)expected);
        }
    }
    puts("DISPATCH_CASE name=end_to_end_mock result=PASS chunks=2 rows=3 max_abs=0");
    return 0;
}

static int test_rejections(void) {
    struct q8q8_workload_descriptor descriptor = workload(32, 256, 128);
    struct q8q8_dispatch_plan plan;
    CHECK(q8q8_plan_workload(&descriptor, 128, 64, &plan) == Q8Q8_RT_ERR_ARGUMENT);
    descriptor.n = 255;
    CHECK(q8q8_plan_workload(&descriptor, 1u << 20, 64, &plan) == Q8Q8_RT_ERR_ARGUMENT);
    descriptor = workload(32, 256, 127);
    CHECK(q8q8_plan_workload(&descriptor, 1u << 20, 64, &plan) == Q8Q8_RT_ERR_ARGUMENT);
    descriptor = workload(32, 256, 128);
    descriptor.operation = (enum q8q8_operation)99;
    CHECK(q8q8_plan_workload(&descriptor, 1u << 20, 64, &plan) == Q8Q8_RT_ERR_ARGUMENT);
    puts("DISPATCH_CASE name=invalid_descriptors result=PASS rejected=4");
    return 0;
}

int main(void) {
    if (test_q8_0_split() != 0 || test_representative_plans() != 0 ||
            test_allocation_and_jobs() != 0 || test_end_to_end_dispatch() != 0 ||
            test_rejections() != 0) {
        return 1;
    }
    puts("DISPATCH_TEST_RESULT=PASS cases=5 task_fields=0");
    return 0;
}
