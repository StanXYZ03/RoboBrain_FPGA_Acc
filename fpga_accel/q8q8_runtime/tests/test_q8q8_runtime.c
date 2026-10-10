#include "q8q8_runtime.h"
#include "q8q8_arena.h"
#include "q8q8_smoke.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct mock_device {
    uint32_t regs[0x100 / 4];
    uint8_t arena[4096];
    uint64_t phys_base;
    uint64_t now_us;
    int run_on_start;
    int sync_to_device;
    int sync_from_device;
    int writes;
};

static uint64_t reg64(struct mock_device *mock, uint32_t offset) {
    return mock->regs[offset / 4] |
            ((uint64_t)mock->regs[offset / 4 + 1] << 32);
}

static void *phys_to_virt(struct mock_device *mock, uint64_t address,
                          size_t size) {
    uint64_t offset;
    if (address < mock->phys_base) return NULL;
    offset = address - mock->phys_base;
    if (offset > sizeof(mock->arena) || size > sizeof(mock->arena) - offset) {
        return NULL;
    }
    return mock->arena + offset;
}

static void emulate_gemm(struct mock_device *mock) {
    uint32_t m = mock->regs[Q8Q8_REG_M / 4];
    uint32_t n = mock->regs[Q8Q8_REG_N / 4];
    uint32_t k = mock->regs[Q8Q8_REG_K / 4];
    int8_t *wq = phys_to_virt(mock, reg64(mock, Q8Q8_REG_WEIGHTS_Q),
                              (size_t)n * (k / 32) * 32);
    int8_t *xq = phys_to_virt(mock, reg64(mock, Q8Q8_REG_ACTIVATIONS_Q),
                              (size_t)m * (k / 32) * 32);
    float *output = phys_to_virt(mock, reg64(mock, Q8Q8_REG_OUTPUT),
                                 (size_t)m * n * sizeof(float));
    uint32_t mi;
    uint32_t ni;
    if (wq == NULL || xq == NULL || output == NULL || k != 32) {
        mock->regs[Q8Q8_REG_AP_RETURN / 4] = (uint32_t)-1;
    } else {
        for (mi = 0; mi < m; ++mi) {
            for (ni = 0; ni < n; ++ni) {
                int sum = 0;
                int lane;
                for (lane = 0; lane < 32; ++lane) {
                    sum += wq[ni * 32 + lane] * xq[mi * 32 + lane];
                }
                output[mi * n + ni] = (float)sum;
            }
        }
        mock->regs[Q8Q8_REG_AP_RETURN / 4] = 0;
    }
    mock->regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_DONE |
            Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
}

static uint32_t mock_read32(void *opaque, uint32_t offset) {
    struct mock_device *mock = opaque;
    uint32_t value = mock->regs[offset / 4];
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_DONE) != 0) {
        mock->regs[offset / 4] &= ~Q8Q8_CTRL_DONE;
    }
    return value;
}

static void mock_write32(void *opaque, uint32_t offset, uint32_t value) {
    struct mock_device *mock = opaque;
    mock->regs[offset / 4] = value;
    ++mock->writes;
    if (offset == Q8Q8_REG_AP_CTRL && (value & Q8Q8_CTRL_START) != 0 &&
            mock->run_on_start) {
        emulate_gemm(mock);
    }
}

static uint64_t mock_now_us(void *opaque) {
    return ((struct mock_device *)opaque)->now_us;
}

static void mock_sleep_us(void *opaque, uint32_t usec) {
    ((struct mock_device *)opaque)->now_us += usec;
}

static int mock_sync(void *opaque, uint64_t address, size_t size,
                     enum q8q8_sync_direction direction) {
    struct mock_device *mock = opaque;
    if (phys_to_virt(mock, address, size) == NULL) return -1;
    if (direction == Q8Q8_SYNC_TO_DEVICE) ++mock->sync_to_device;
    else ++mock->sync_from_device;
    return 0;
}

static void mock_fence(void *opaque) { (void)opaque; }

static const struct q8q8_io_ops mock_ops = {
    mock_read32, mock_write32, mock_now_us, mock_sleep_us,
    mock_sync, mock_fence
};

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "TEST_FAIL line=%d reason=%s\n", __LINE__, message); \
        return 1; \
    } \
} while (0)

int main(void) {
    struct mock_device mock;
    struct q8q8_device device;
    struct q8q8_smoke_vector vector;
    struct q8q8_result result;
    float max_abs = 0.0f;
    int status;

    {
        uint8_t storage[256];
        uint8_t source[5] = {1, 2, 3, 4, 5};
        uint8_t destination[5] = {0};
        struct q8q8_arena arena;
        struct q8q8_buffer first;
        struct q8q8_buffer second;
        CHECK(q8q8_arena_init(&arena, storage, sizeof(storage),
                              UINT64_C(0xb7c000000)) == Q8Q8_RT_OK,
              "arena init");
        CHECK(q8q8_arena_alloc(&arena, 33, 32, &first) == Q8Q8_RT_OK &&
              first.offset == 0 && first.phys == UINT64_C(0xb7c000000),
              "first arena allocation");
        CHECK(q8q8_arena_alloc(&arena, 5, 64, &second) == Q8Q8_RT_OK &&
              second.offset == 64 && second.phys == UINT64_C(0xb7c000040),
              "aligned arena allocation");
        CHECK(q8q8_buffer_store(&second, 0, source, sizeof(source)) ==
                      Q8Q8_RT_OK &&
              q8q8_buffer_load(destination, &second, 0,
                               sizeof(destination)) == Q8Q8_RT_OK &&
              memcmp(source, destination, sizeof(source)) == 0,
              "device-safe buffer copy");
        CHECK(q8q8_arena_alloc(&arena, sizeof(storage), 64, &first) ==
                      Q8Q8_RT_ERR_ARGUMENT,
              "arena overflow rejection");
        printf("TEST_CASE name=arena_allocator result=PASS used=%zu\n",
               arena.used);
    }

    memset(&mock, 0, sizeof(mock));
    mock.phys_base = UINT64_C(0x0000000b7c000000);
    mock.run_on_start = 1;
    mock.regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
    q8q8_device_init(&device, &mock_ops, &mock);
    CHECK(q8q8_smoke_prepare(mock.arena, sizeof(mock.arena), mock.phys_base,
                             &vector) == Q8Q8_RT_OK, "smoke prepare");
    status = q8q8_run(&device, &vector.job, 1000, &result);
    CHECK(status == Q8Q8_RT_OK, "valid job status");
    CHECK(result.hardware_status == 0, "hardware status");
    CHECK(q8q8_smoke_check(mock.arena, &vector, 0.0f, &max_abs) == Q8Q8_RT_OK,
          "numeric output");
    CHECK(max_abs == 0.0f, "exact output");
    CHECK(reg64(&mock, Q8Q8_REG_WEIGHTS_Q) == vector.job.weights_q,
          "64-bit weight address");
    CHECK(reg64(&mock, Q8Q8_REG_OUTPUT) == vector.job.output,
          "64-bit output address");
    CHECK(mock.sync_to_device == 5 && mock.sync_from_device == 1,
          "buffer synchronization sequence");
    CHECK(mock.writes == 14, "register write count");
    printf("TEST_CASE name=valid_high_address_numeric result=PASS writes=%d sync_to=%d sync_from=%d\n",
           mock.writes, mock.sync_to_device, mock.sync_from_device);

    vector.job.weights_q += 1;
    CHECK(q8q8_run(&device, &vector.job, 1000, &result) ==
          Q8Q8_RT_ERR_ALIGNMENT, "alignment rejection");
    vector.job.weights_q -= 1;
    vector.job.k = 33;
    CHECK(q8q8_run(&device, &vector.job, 1000, &result) ==
          Q8Q8_RT_ERR_ARGUMENT, "K alignment rejection");
    vector.job.k = 32;
    vector.job.n = 3;
    CHECK(q8q8_run(&device, &vector.job, 1000, &result) ==
          Q8Q8_RT_ERR_ARGUMENT, "N packing alignment rejection");
    vector.job.n = 4;
    printf("TEST_CASE name=invalid_job_rejected result=PASS\n");

    mock.run_on_start = 0;
    mock.regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
    CHECK(q8q8_run(&device, &vector.job, 50, &result) == Q8Q8_RT_ERR_TIMEOUT,
          "timeout detection");
    CHECK(device.faulted == 1, "fault latch");
    CHECK(q8q8_run(&device, &vector.job, 50, &result) == Q8Q8_RT_ERR_FAULTED,
          "fault prevents unsafe reuse");
    CHECK(q8q8_try_recover(&device) == Q8Q8_RT_ERR_FAULTED,
          "running core cannot soft recover");
    mock.regs[Q8Q8_REG_AP_CTRL / 4] = Q8Q8_CTRL_IDLE | Q8Q8_CTRL_READY;
    CHECK(q8q8_try_recover(&device) == Q8Q8_RT_OK && !device.faulted,
          "idle recovery");
    printf("TEST_CASE name=timeout_fault_latch result=PASS elapsed_us=%llu\n",
           (unsigned long long)result.elapsed_us);
    printf("TEST_RESULT=PASS cases=4\n");
    return 0;
}
