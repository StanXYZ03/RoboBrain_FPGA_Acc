#include "robobrain_ggml_q8q8_bridge.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "BRIDGE_TEST_FAIL line=%d check=%s\n", \
                __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static struct rb_ggml_tensor_view q8_view(const void *data, uint64_t k,
                                          uint64_t rows) {
    const size_t row = (size_t)(k / 32) * 34;
    struct rb_ggml_tensor_view view = {
        data, RB_GGML_TYPE_Q8_0, {k, rows, 1, 1},
        {34, row, row * (size_t)rows, row * (size_t)rows},
    };
    return view;
}

static struct rb_ggml_tensor_view f32_view(const void *data, uint64_t width,
                                           uint64_t rows) {
    const size_t row = (size_t)width * sizeof(float);
    struct rb_ggml_tensor_view view = {
        data, RB_GGML_TYPE_F32, {width, rows, 1, 1},
        {sizeof(float), row, row * (size_t)rows, row * (size_t)rows},
    };
    return view;
}

static struct rb_ggml_mul_mat_view operation(void *weights, void *activations,
                                              void *output, uint32_t m,
                                              uint32_t n, uint32_t k,
                                              int activations_are_f32) {
    struct rb_ggml_mul_mat_view op;
    op.src0_weights = q8_view(weights, k, n);
    op.src1_activations = activations_are_f32
            ? f32_view(activations, k, m) : q8_view(activations, k, m);
    op.dst_f32 = f32_view(output, n, m);
    return op;
}

static int test_semantics_and_plans(void) {
    uint8_t dummy = 0;
    struct rb_ggml_mul_mat_view op = operation(&dummy, &dummy, &dummy,
                                                1080, 9728, 2560, 0);
    struct rb_ggml_q8q8_report report;
    CHECK(rb_ggml_q8q8_classify(&op, 64u * 1024u * 1024u, 64,
                                &report) == 1);
    CHECK(report.reason == RB_GGML_Q8Q8_ELIGIBLE);
    CHECK(report.m == 1080 && report.n == 9728 && report.k == 2560);
    CHECK(report.activation_quantization_required == 0);
    CHECK(report.plan.chunk_count == 2 && report.plan.chunk_m >= 512);
    op = operation(&dummy, &dummy, &dummy, 512, 9728, 2560, 1);
    CHECK(rb_ggml_q8q8_classify(&op, 64u * 1024u * 1024u, 64,
                                &report) == 1);
    CHECK(report.activation_quantization_required == 1);
    puts("BRIDGE_CASE name=ggml_semantics_and_plans result=PASS "
         "mapping=src0_KN_src1_KM_dst_NM");
    return 0;
}

static int test_safe_fallbacks(void) {
    uint8_t dummy = 0;
    struct rb_ggml_mul_mat_view op = operation(&dummy, &dummy, &dummy,
                                                4, 8, 32, 0);
    struct rb_ggml_q8q8_report report;
    unsigned rejected = 0;
#define REJECT(expected) do { \
    CHECK(rb_ggml_q8q8_classify(&op, 1u << 20, 64, &report) == 0); \
    CHECK(report.reason == (expected)); \
    ++rejected; \
} while (0)
    op.src0_weights.type = RB_GGML_TYPE_F32;
    REJECT(RB_GGML_Q8Q8_WEIGHT_TYPE);
    op = operation(&dummy, &dummy, &dummy, 4, 8, 32, 0);
    op.src1_activations.ne[2] = 2;
    REJECT(RB_GGML_Q8Q8_BATCH_OR_BROADCAST);
    op = operation(&dummy, &dummy, &dummy, 4, 8, 32, 0);
    op.dst_f32.ne[0] = 4;
    REJECT(RB_GGML_Q8Q8_SHAPE_MISMATCH);
    op = operation(&dummy, &dummy, &dummy, 4, 8, 32, 0);
    op.src0_weights.nb[1] += 34;
    REJECT(RB_GGML_Q8Q8_WEIGHT_LAYOUT);
    op = operation(&dummy, &dummy, &dummy, 4, 8, 32, 1);
    op.src1_activations.nb[1] += sizeof(float);
    REJECT(RB_GGML_Q8Q8_ACTIVATION_LAYOUT);
    op = operation(&dummy, &dummy, &dummy, 4, 8, 32, 0);
    op.dst_f32.nb[1] += sizeof(float);
    REJECT(RB_GGML_Q8Q8_OUTPUT_LAYOUT);
    op = operation(&dummy, &dummy, &dummy, 4, 6, 32, 0);
    REJECT(RB_GGML_Q8Q8_HARDWARE_LIMIT);
    op = operation(&dummy, &dummy, &dummy, 512, 9728, 2560, 0);
    REJECT(RB_GGML_Q8Q8_ARENA_CAPACITY);
#undef REJECT
    CHECK(rejected == 8);
    printf("BRIDGE_CASE name=safe_fallbacks result=PASS rejected=%u\n",
           rejected);
    return 0;
}

static int test_reference_quantization(void) {
    float values[64] = {0};
    struct q8q8_block_q8_0 blocks[2];
    struct q8q8_block_q8_0 fast[2];
    size_t lane;
    values[0] = 127.0f;
    values[1] = -127.0f;
    values[2] = 63.5f;
    values[3] = -63.5f;
    CHECK(rb_ggml_quantize_f32_q8_0(values, 2, 32, 32 * sizeof(float),
                                    blocks, sizeof(blocks)) == 0);
    CHECK(blocks[0].scale_fp16 == 0x3c00);
    CHECK(blocks[0].values[0] == 127 && blocks[0].values[1] == -127);
    CHECK(blocks[0].values[2] == 64 && blocks[0].values[3] == -64);
    CHECK(blocks[1].scale_fp16 == 0);
    for (lane = 0; lane < 32; ++lane) CHECK(blocks[1].values[lane] == 0);
    CHECK(rb_ggml_quantize_f32_q8_0_fast(
              values, 2, 32, 32 * sizeof(float), fast, sizeof(fast), 4) == 0);
    CHECK(memcmp(blocks, fast, sizeof(blocks)) == 0);
    values[5] = NAN;
    CHECK(rb_ggml_quantize_f32_q8_0(values, 1, 32, 32 * sizeof(float),
                                    blocks, sizeof(blocks)) == -2);
    CHECK(rb_ggml_quantize_f32_q8_0_fast(
              values, 1, 32, 32 * sizeof(float), fast, sizeof(fast), 4) == -2);
    puts("BRIDGE_CASE name=reference_quantization result=PASS "
         "rounding=away_from_zero nonfinite=fallback");
    return 0;
}

static int test_fast_weight_pack(void) {
    struct q8q8_block_q8_0 source[6];
    uint8_t reference_q[192] = {0};
    uint16_t reference_s[6] = {0};
    uint8_t fast_q[192] = {0};
    uint16_t fast_s[6] = {0};
    struct q8q8_buffer rq = {reference_q, 64, 0, sizeof(reference_q)};
    struct q8q8_buffer rs = {(volatile uint8_t *)reference_s, 64, 0, sizeof(reference_s)};
    struct q8q8_buffer fq = {fast_q, 64, 0, sizeof(fast_q)};
    struct q8q8_buffer fs = {(volatile uint8_t *)fast_s, 64, 0, sizeof(fast_s)};
    size_t block;
    size_t lane;
    for (block = 0; block < 6; ++block) {
        source[block].scale_fp16 = (uint16_t)(0x3000 + block);
        for (lane = 0; lane < 32; ++lane)
            source[block].values[lane] = (int8_t)(block * 11 + lane - 63);
    }
    CHECK(q8q8_pack_ggml_q8_0(source, 3, 64, &rq, &rs) == Q8Q8_RT_OK);
    CHECK(rb_ggml_pack_q8_0_split_fast(source, 3, 64, &fq, &fs, 4) == 0);
    CHECK(memcmp(reference_q, fast_q, sizeof(fast_q)) == 0);
    CHECK(memcmp(reference_s, fast_s, sizeof(fast_s)) == 0);
    CHECK(rb_ggml_pack_q8_0_split_fast(source, 3, 64, &fq, &fs, 0) == -1);
    puts("BRIDGE_CASE name=fast_weight_pack result=PASS threads=4 byte_equal=yes");
    return 0;
}

static int test_fast_output_copy(void) {
    uint8_t source_data[1027];
    uint8_t reference[1027];
    uint8_t fast[1027];
    struct q8q8_buffer source = {source_data, 64, 0, sizeof(source_data)};
    size_t index;
    for (index = 0; index < sizeof(source_data); ++index)
        source_data[index] = (uint8_t)(index * 29u + 7u);
    CHECK(q8q8_buffer_load(reference, &source, 3, 1001) == Q8Q8_RT_OK);
    CHECK(rb_ggml_copy_from_uio_fast(fast, &source, 3, 1001, 4) == 0);
    CHECK(memcmp(reference, fast, 1001) == 0);
    CHECK(rb_ggml_copy_from_uio_fast(fast, &source, 30, 1001, 4) == -1);
    puts("BRIDGE_CASE name=fast_output_copy result=PASS threads=4 byte_equal=yes");
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
        mock->regs[Q8Q8_REG_AP_CTRL / 4] =
                Q8Q8_CTRL_DONE | Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
        return;
    }
    for (mi = 0; mi < m; ++mi) {
        for (ni = 0; ni < n; ++ni) {
            int32_t sum = 0;
            uint32_t lane;
            if (weight_scales[ni] != 0x3c00 ||
                    activation_scales[mi] != 0x3c00) {
                mock->regs[Q8Q8_REG_AP_RETURN / 4] = (uint32_t)-2;
                mock->regs[Q8Q8_REG_AP_CTRL / 4] =
                        Q8Q8_CTRL_DONE | Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
                return;
            }
            for (lane = 0; lane < k; ++lane) {
                sum += weights[(size_t)ni * k + lane] *
                        activations[(size_t)mi * k + lane];
            }
            output[(size_t)mi * n + ni] = (float)sum;
        }
    }
    mock->now_us += 11;
    mock->regs[Q8Q8_REG_AP_RETURN / 4] = 0;
    mock->regs[Q8Q8_REG_AP_CTRL / 4] =
            Q8Q8_CTRL_DONE | Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
}

static uint32_t mock_read32(void *opaque, uint32_t offset) {
    struct mock_accelerator *mock = opaque;
    uint32_t value = mock->regs[offset / 4];
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_DONE))
        mock->regs[offset / 4] &= ~Q8Q8_CTRL_DONE;
    return value;
}

static void mock_write32(void *opaque, uint32_t offset, uint32_t value) {
    struct mock_accelerator *mock = opaque;
    mock->regs[offset / 4] = value;
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_START))
        mock_compute(mock);
}

static uint64_t mock_now_us(void *opaque) {
    return ((struct mock_accelerator *)opaque)->now_us;
}
static void mock_sleep_us(void *opaque, uint32_t us) {
    ((struct mock_accelerator *)opaque)->now_us += us;
}
static int mock_sync(void *opaque, uint64_t address, size_t size,
                     enum q8q8_sync_direction direction) {
    (void)direction;
    return mock_pointer(opaque, address, size) == NULL ? -1 : 0;
}
static void mock_fence(void *opaque) { (void)opaque; }

static int test_end_to_end_bridge(void) {
    uint8_t storage[1024];
    struct q8q8_block_q8_0 weights[4];
    struct q8q8_block_q8_0 activations[3];
    float output[12] = {0};
    struct rb_ggml_mul_mat_view op = operation(weights, activations, output,
                                                3, 4, 32, 0);
    struct q8q8_arena arena;
    struct q8q8_device device;
    struct q8q8_dispatch_stats stats;
    struct rb_ggml_q8q8_report report;
    struct mock_accelerator mock;
    const struct q8q8_io_ops ops = {
        mock_read32, mock_write32, mock_now_us, mock_sleep_us,
        mock_sync, mock_fence,
    };
    size_t row;
    size_t lane;
    memset(&mock, 0, sizeof(mock));
    for (row = 0; row < 4; ++row) {
        weights[row].scale_fp16 = 0x3c00;
        for (lane = 0; lane < 32; ++lane)
            weights[row].values[lane] = (int8_t)((row + lane) % 5 - 2);
    }
    for (row = 0; row < 3; ++row) {
        activations[row].scale_fp16 = 0x3c00;
        for (lane = 0; lane < 32; ++lane)
            activations[row].values[lane] = (int8_t)((2 * row + lane) % 7 - 3);
    }
    mock.arena = storage;
    mock.arena_size = sizeof(storage);
    mock.phys_base = UINT64_C(0xb7c000000);
    mock.regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
    CHECK(q8q8_arena_init(&arena, storage, sizeof(storage),
                           mock.phys_base) == Q8Q8_RT_OK);
    q8q8_device_init(&device, &ops, &mock);
    CHECK(rb_ggml_q8q8_execute(&op, &device, &arena, NULL, 0, 1000,
                               &stats, &report) == Q8Q8_RT_OK);
    CHECK(stats.chunks_completed == 1 && stats.rows_completed == 3 &&
          stats.hardware_elapsed_us == 11);
    for (row = 0; row < 3; ++row) {
        size_t column;
        for (column = 0; column < 4; ++column) {
            int32_t expected = 0;
            for (lane = 0; lane < 32; ++lane)
                expected += activations[row].values[lane] *
                            weights[column].values[lane];
            CHECK(output[row * 4 + column] == (float)expected);
        }
    }
    memset(output, 0, sizeof(output));
    CHECK(rb_ggml_q8q8_execute_fast(&op, &device, &arena, NULL, 0, 1000,
                                    4, &stats, &report) == Q8Q8_RT_OK);
    CHECK(stats.chunks_completed == 1 && stats.rows_completed == 3 &&
          stats.hardware_elapsed_us == 11);
    for (row = 0; row < 3; ++row) {
        size_t column;
        for (column = 0; column < 4; ++column) {
            int32_t expected = 0;
            for (lane = 0; lane < 32; ++lane)
                expected += activations[row].values[lane] *
                            weights[column].values[lane];
            CHECK(output[row * 4 + column] == (float)expected);
        }
    }
    puts("BRIDGE_CASE name=end_to_end_mock result=PASS paths=reference,fast chunks=1 max_abs=0");
    return 0;
}

int main(void) {
    if (test_semantics_and_plans() || test_safe_fallbacks() ||
            test_reference_quantization() || test_fast_weight_pack() ||
            test_fast_output_copy() ||
            test_end_to_end_bridge()) return 1;
    puts("GGML_Q8Q8_BRIDGE_TEST_RESULT=PASS cases=6 upstream_modified=no "
         "board_accessed=no");
    return 0;
}
