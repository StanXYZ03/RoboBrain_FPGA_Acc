#define _POSIX_C_SOURCE 200809L
#include "q8q8_runtime.h"

#include <stdatomic.h>
#include <string.h>

enum {
    Q8Q8_QK = 32,
    Q8Q8_MAX_M = 4320,
    Q8Q8_MAX_N = 9728,
    Q8Q8_MAX_K = 9728,
};

static int checked_product(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    *result = a * b;
    return Q8Q8_RT_OK;
}

static void runtime_fence(struct q8q8_device *device) {
    atomic_thread_fence(memory_order_seq_cst);
    if (device->ops->fence != NULL) {
        device->ops->fence(device->opaque);
    }
    atomic_thread_fence(memory_order_seq_cst);
}

static void write64(struct q8q8_device *device, uint32_t offset,
                    uint64_t value) {
    device->ops->write32(device->opaque, offset, (uint32_t)value);
    device->ops->write32(device->opaque, offset + 4,
                         (uint32_t)(value >> 32));
}

static int sync_buffer(struct q8q8_device *device, uint64_t address,
                       size_t size, enum q8q8_sync_direction direction) {
    if (device->ops->sync == NULL) {
        return Q8Q8_RT_OK;
    }
    return device->ops->sync(device->opaque, address, size, direction) == 0
            ? Q8Q8_RT_OK : Q8Q8_RT_ERR_IO;
}

void q8q8_device_init(struct q8q8_device *device,
                      const struct q8q8_io_ops *ops, void *opaque) {
    if (device == NULL) {
        return;
    }
    device->ops = ops;
    device->opaque = opaque;
    device->faulted = 0;
}

int q8q8_validate_job(const struct q8q8_job *job) {
    if (job == NULL || job->m == 0 || job->n == 0 || job->k == 0 ||
            job->m > Q8Q8_MAX_M || job->n > Q8Q8_MAX_N ||
            job->k > Q8Q8_MAX_K) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if ((job->k % Q8Q8_QK) != 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if ((job->n % 4u) != 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if ((job->weights_q & 31u) != 0 ||
            (job->activations_q & 31u) != 0 ||
            (job->weights_scale & 1u) != 0 ||
            (job->activations_scale & 1u) != 0 ||
            (job->output & 3u) != 0) {
        return Q8Q8_RT_ERR_ALIGNMENT;
    }
    if (job->weights_q == 0 || job->weights_scale == 0 ||
            job->activations_q == 0 || job->activations_scale == 0 ||
            job->output == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    return Q8Q8_RT_OK;
}

int q8q8_run(struct q8q8_device *device, const struct q8q8_job *job,
             uint64_t timeout_us, struct q8q8_result *result) {
    size_t k_blocks;
    size_t weight_blocks;
    size_t activation_blocks;
    size_t output_values;
    uint32_t control;
    uint64_t started;
    int status;

    if (device == NULL || device->ops == NULL ||
            device->ops->read32 == NULL || device->ops->write32 == NULL ||
            device->ops->now_us == NULL || timeout_us == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if (device->faulted) {
        return Q8Q8_RT_ERR_FAULTED;
    }
    status = q8q8_validate_job(job);
    if (status != Q8Q8_RT_OK) {
        return status;
    }
    k_blocks = job->k / Q8Q8_QK;
    if (checked_product(job->n, k_blocks, &weight_blocks) != Q8Q8_RT_OK ||
            checked_product(job->m, k_blocks, &activation_blocks) != Q8Q8_RT_OK ||
            checked_product(job->m, job->n, &output_values) != Q8Q8_RT_OK) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }

    control = device->ops->read32(device->opaque, Q8Q8_REG_AP_CTRL);
    if ((control & Q8Q8_CTRL_START) != 0 ||
            (control & Q8Q8_CTRL_IDLE) == 0) {
        return Q8Q8_RT_ERR_BUSY;
    }

    status = sync_buffer(device, job->weights_q, weight_blocks * 32,
                         Q8Q8_SYNC_TO_DEVICE);
    if (status == Q8Q8_RT_OK) {
        status = sync_buffer(device, job->weights_scale, weight_blocks * 2,
                             Q8Q8_SYNC_TO_DEVICE);
    }
    if (status == Q8Q8_RT_OK) {
        status = sync_buffer(device, job->activations_q,
                             activation_blocks * 32, Q8Q8_SYNC_TO_DEVICE);
    }
    if (status == Q8Q8_RT_OK) {
        status = sync_buffer(device, job->activations_scale,
                             activation_blocks * 2, Q8Q8_SYNC_TO_DEVICE);
    }
    if (status == Q8Q8_RT_OK) {
        status = sync_buffer(device, job->output, output_values * sizeof(float),
                             Q8Q8_SYNC_TO_DEVICE);
    }
    if (status != Q8Q8_RT_OK) {
        return status;
    }

    write64(device, Q8Q8_REG_WEIGHTS_Q, job->weights_q);
    write64(device, Q8Q8_REG_WEIGHTS_SCALE, job->weights_scale);
    write64(device, Q8Q8_REG_ACTIVATIONS_Q, job->activations_q);
    write64(device, Q8Q8_REG_ACTIVATIONS_SCALE, job->activations_scale);
    write64(device, Q8Q8_REG_OUTPUT, job->output);
    device->ops->write32(device->opaque, Q8Q8_REG_M, job->m);
    device->ops->write32(device->opaque, Q8Q8_REG_N, job->n);
    device->ops->write32(device->opaque, Q8Q8_REG_K, job->k);
    runtime_fence(device);

    started = device->ops->now_us(device->opaque);
    device->ops->write32(device->opaque, Q8Q8_REG_AP_CTRL, Q8Q8_CTRL_START);
    for (;;) {
        runtime_fence(device);
        control = device->ops->read32(device->opaque, Q8Q8_REG_AP_CTRL);
        if ((control & Q8Q8_CTRL_DONE) != 0) {
            int32_t hardware_status = (int32_t)device->ops->read32(
                    device->opaque, Q8Q8_REG_AP_RETURN);
            runtime_fence(device);
            status = sync_buffer(device, job->output,
                                 output_values * sizeof(float),
                                 Q8Q8_SYNC_FROM_DEVICE);
            if (result != NULL) {
                result->hardware_status = hardware_status;
                result->elapsed_us = device->ops->now_us(device->opaque) - started;
                result->final_control = control;
            }
            if (status != Q8Q8_RT_OK) {
                return status;
            }
            return hardware_status == 0 ? Q8Q8_RT_OK : Q8Q8_RT_ERR_HARDWARE;
        }
        if (device->ops->now_us(device->opaque) - started >= timeout_us) {
            device->faulted = 1;
            if (result != NULL) {
                memset(result, 0, sizeof(*result));
                result->elapsed_us = device->ops->now_us(device->opaque) - started;
                result->final_control = control;
            }
            return Q8Q8_RT_ERR_TIMEOUT;
        }
        if (device->ops->sleep_us != NULL) {
            device->ops->sleep_us(device->opaque, 10);
        }
    }
}

int q8q8_try_recover(struct q8q8_device *device) {
    uint32_t control;
    if (device == NULL || device->ops == NULL || device->ops->read32 == NULL) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    if (!device->faulted) {
        return Q8Q8_RT_OK;
    }
    control = device->ops->read32(device->opaque, Q8Q8_REG_AP_CTRL);
    if ((control & Q8Q8_CTRL_IDLE) == 0 || (control & Q8Q8_CTRL_START) != 0) {
        return Q8Q8_RT_ERR_FAULTED;
    }
    device->faulted = 0;
    return Q8Q8_RT_OK;
}

const char *q8q8_runtime_status_string(int status) {
    switch (status) {
    case Q8Q8_RT_OK: return "ok";
    case Q8Q8_RT_ERR_ARGUMENT: return "invalid argument";
    case Q8Q8_RT_ERR_ALIGNMENT: return "misaligned physical address";
    case Q8Q8_RT_ERR_BUSY: return "accelerator busy";
    case Q8Q8_RT_ERR_TIMEOUT: return "accelerator timeout";
    case Q8Q8_RT_ERR_FAULTED: return "accelerator requires reset or idle recovery";
    case Q8Q8_RT_ERR_HARDWARE: return "accelerator returned an error";
    case Q8Q8_RT_ERR_IO: return "buffer synchronization failed";
    default: return "unknown runtime status";
    }
}
