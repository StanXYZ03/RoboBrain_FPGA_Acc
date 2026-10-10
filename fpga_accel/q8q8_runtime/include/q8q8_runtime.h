#ifndef ROBOBRAIN_Q8Q8_RUNTIME_H
#define ROBOBRAIN_Q8Q8_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum q8q8_runtime_status {
    Q8Q8_RT_OK = 0,
    Q8Q8_RT_ERR_ARGUMENT = -1,
    Q8Q8_RT_ERR_ALIGNMENT = -2,
    Q8Q8_RT_ERR_BUSY = -3,
    Q8Q8_RT_ERR_TIMEOUT = -4,
    Q8Q8_RT_ERR_FAULTED = -5,
    Q8Q8_RT_ERR_HARDWARE = -6,
    Q8Q8_RT_ERR_IO = -7,
};

enum q8q8_sync_direction {
    Q8Q8_SYNC_TO_DEVICE = 0,
    Q8Q8_SYNC_FROM_DEVICE = 1,
};

struct q8q8_io_ops {
    uint32_t (*read32)(void *opaque, uint32_t offset);
    void (*write32)(void *opaque, uint32_t offset, uint32_t value);
    uint64_t (*now_us)(void *opaque);
    void (*sleep_us)(void *opaque, uint32_t usec);
    int (*sync)(void *opaque, uint64_t phys_addr, size_t size,
                enum q8q8_sync_direction direction);
    void (*fence)(void *opaque);
};

struct q8q8_device {
    const struct q8q8_io_ops *ops;
    void *opaque;
    int faulted;
};

struct q8q8_job {
    uint64_t weights_q;
    uint64_t weights_scale;
    uint64_t activations_q;
    uint64_t activations_scale;
    uint64_t output;
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct q8q8_result {
    int32_t hardware_status;
    uint64_t elapsed_us;
    uint32_t final_control;
};

enum {
    Q8Q8_REG_AP_CTRL = 0x00,
    Q8Q8_REG_AP_RETURN = 0x10,
    Q8Q8_REG_WEIGHTS_Q = 0x18,
    Q8Q8_REG_WEIGHTS_SCALE = 0x24,
    Q8Q8_REG_ACTIVATIONS_Q = 0x30,
    Q8Q8_REG_ACTIVATIONS_SCALE = 0x3c,
    Q8Q8_REG_OUTPUT = 0x48,
    Q8Q8_REG_M = 0x54,
    Q8Q8_REG_N = 0x5c,
    Q8Q8_REG_K = 0x64,
};

enum {
    Q8Q8_CTRL_START = 1u << 0,
    Q8Q8_CTRL_DONE = 1u << 1,
    Q8Q8_CTRL_IDLE = 1u << 2,
    Q8Q8_CTRL_READY = 1u << 3,
    Q8Q8_CTRL_AUTO_RESTART = 1u << 7,
};

void q8q8_device_init(struct q8q8_device *device,
                      const struct q8q8_io_ops *ops, void *opaque);
int q8q8_validate_job(const struct q8q8_job *job);
int q8q8_run(struct q8q8_device *device, const struct q8q8_job *job,
             uint64_t timeout_us, struct q8q8_result *result);
int q8q8_try_recover(struct q8q8_device *device);
const char *q8q8_runtime_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif
