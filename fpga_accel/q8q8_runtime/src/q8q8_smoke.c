#include "q8q8_smoke.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/* The UIO arena is exposed as non-cached device memory.  Optimized libc
   memset/memcpy implementations may issue vector stores or DC ZVA, neither
   of which is valid for an ARM device mapping.  Keep every arena access an
   explicit volatile byte transaction; ordinary stack objects may still use
   libc normally. */
static void arena_zero(volatile uint8_t *destination, size_t size) {
    size_t index;
    for (index = 0; index < size; ++index) destination[index] = 0;
}

static void arena_store(volatile uint8_t *destination, const void *source,
                        size_t size) {
    const uint8_t *source_bytes = (const uint8_t *)source;
    size_t index;
    for (index = 0; index < size; ++index) {
        destination[index] = source_bytes[index];
    }
}

static void arena_load(void *destination, const volatile uint8_t *source,
                       size_t size) {
    uint8_t *destination_bytes = (uint8_t *)destination;
    size_t index;
    for (index = 0; index < size; ++index) {
        destination_bytes[index] = source[index];
    }
}

int q8q8_smoke_prepare(void *arena, size_t arena_size, uint64_t phys_base,
                       struct q8q8_smoke_vector *vector) {
    volatile uint8_t *bytes = (volatile uint8_t *)arena;
    size_t wq_offset = 0;
    size_t ws_offset = align_up(wq_offset + 4 * 32, 64);
    size_t xq_offset = align_up(ws_offset + 4 * 2, 64);
    size_t xs_offset = align_up(xq_offset + 2 * 32, 64);
    size_t y_offset = align_up(xs_offset + 2 * 2, 64);
    size_t used = y_offset + 8 * sizeof(float);
    int row;
    int lane;

    if (arena == NULL || vector == NULL || arena_size < used ||
            (phys_base & 63u) != 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    arena_zero(bytes, used);
    for (lane = 0; lane < 32; ++lane) {
        bytes[wq_offset + lane] = (uint8_t)(int8_t)1;
        bytes[wq_offset + 32 + lane] = (uint8_t)(int8_t)
                ((lane & 1) ? 1 : -1);
        bytes[wq_offset + 64 + lane] = (uint8_t)(int8_t)(lane - 16);
        bytes[wq_offset + 96 + lane] = 0;
    }
    for (row = 0; row < 4; ++row) {
        uint16_t one = 0x3c00;
        arena_store(bytes + ws_offset + row * 2, &one, sizeof(one));
    }
    for (lane = 0; lane < 32; ++lane) {
        bytes[xq_offset + lane] = (uint8_t)(int8_t)2;
        bytes[xq_offset + 32 + lane] = (uint8_t)(int8_t)-3;
    }
    for (row = 0; row < 2; ++row) {
        uint16_t one = 0x3c00;
        arena_store(bytes + xs_offset + row * 2, &one, sizeof(one));
    }
    for (row = 0; row < 8; ++row) {
        float sentinel = -12345.0f;
        arena_store(bytes + y_offset + row * sizeof(float), &sentinel,
                    sizeof(sentinel));
    }

    memset(vector, 0, sizeof(*vector));
    vector->job.weights_q = phys_base + wq_offset;
    vector->job.weights_scale = phys_base + ws_offset;
    vector->job.activations_q = phys_base + xq_offset;
    vector->job.activations_scale = phys_base + xs_offset;
    vector->job.output = phys_base + y_offset;
    vector->job.m = 2;
    vector->job.n = 4;
    vector->job.k = 32;
    vector->expected[0] = 64.0f;
    vector->expected[1] = 0.0f;
    vector->expected[2] = -32.0f;
    vector->expected[3] = 0.0f;
    vector->expected[4] = -96.0f;
    vector->expected[5] = 0.0f;
    vector->expected[6] = 48.0f;
    vector->expected[7] = 0.0f;
    vector->output_offset = y_offset;
    vector->bytes_used = used;
    return Q8Q8_RT_OK;
}

int q8q8_smoke_check(const void *arena,
                     const struct q8q8_smoke_vector *vector,
                     float tolerance, float *max_abs_error) {
    const volatile uint8_t *bytes = (const volatile uint8_t *)arena;
    float maximum = 0.0f;
    int failures = 0;
    int index;
    if (arena == NULL || vector == NULL || tolerance < 0.0f) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    for (index = 0; index < 8; ++index) {
        float actual;
        float error;
        arena_load(&actual,
                   bytes + vector->output_offset + index * sizeof(float),
                   sizeof(actual));
        error = fabsf(actual - vector->expected[index]);
        if (error > maximum) {
            maximum = error;
        }
        if (!isfinite(actual) || error > tolerance) {
            ++failures;
        }
    }
    if (max_abs_error != NULL) {
        *max_abs_error = maximum;
    }
    return failures == 0 ? Q8Q8_RT_OK : Q8Q8_RT_ERR_HARDWARE;
}
