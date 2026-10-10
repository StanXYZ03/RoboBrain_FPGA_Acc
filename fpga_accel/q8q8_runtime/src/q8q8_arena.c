#include "q8q8_arena.h"

#include "q8q8_runtime.h"

#include <limits.h>

static int is_power_of_two(size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

static int align_up(size_t value, size_t alignment, size_t *result) {
    size_t mask;
    if (!is_power_of_two(alignment)) return Q8Q8_RT_ERR_ARGUMENT;
    mask = alignment - 1;
    if (value > SIZE_MAX - mask) return Q8Q8_RT_ERR_ARGUMENT;
    *result = (value + mask) & ~mask;
    return Q8Q8_RT_OK;
}

int q8q8_arena_init(struct q8q8_arena *arena, void *base, size_t size,
                    uint64_t phys_base) {
    if (arena == NULL || base == NULL || size == 0) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    arena->base = (volatile uint8_t *)base;
    arena->size = size;
    arena->used = 0;
    arena->phys_base = phys_base;
    return Q8Q8_RT_OK;
}

int q8q8_arena_alloc(struct q8q8_arena *arena, size_t size, size_t alignment,
                     struct q8q8_buffer *buffer) {
    size_t offset;
    if (arena == NULL || buffer == NULL || size == 0 ||
            align_up(arena->used, alignment, &offset) != Q8Q8_RT_OK ||
            offset > arena->size || size > arena->size - offset ||
            offset > UINT64_MAX - arena->phys_base) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    buffer->data = arena->base + offset;
    buffer->phys = arena->phys_base + offset;
    buffer->offset = offset;
    buffer->size = size;
    arena->used = offset + size;
    return Q8Q8_RT_OK;
}

void q8q8_arena_reset(struct q8q8_arena *arena) {
    if (arena != NULL) arena->used = 0;
}

void q8q8_buffer_fill(const struct q8q8_buffer *buffer, uint8_t value) {
    size_t index;
    if (buffer == NULL || buffer->data == NULL) return;
    for (index = 0; index < buffer->size; ++index) {
        buffer->data[index] = value;
    }
}

int q8q8_buffer_store(const struct q8q8_buffer *buffer, size_t offset,
                      const void *source, size_t size) {
    const uint8_t *bytes = (const uint8_t *)source;
    size_t index;
    if (buffer == NULL || buffer->data == NULL || source == NULL ||
            offset > buffer->size || size > buffer->size - offset) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    for (index = 0; index < size; ++index) {
        buffer->data[offset + index] = bytes[index];
    }
    return Q8Q8_RT_OK;
}

int q8q8_buffer_load(void *destination, const struct q8q8_buffer *buffer,
                     size_t offset, size_t size) {
    uint8_t *bytes = (uint8_t *)destination;
    size_t index;
    if (destination == NULL || buffer == NULL || buffer->data == NULL ||
            offset > buffer->size || size > buffer->size - offset) {
        return Q8Q8_RT_ERR_ARGUMENT;
    }
    for (index = 0; index < size; ++index) {
        bytes[index] = buffer->data[offset + index];
    }
    return Q8Q8_RT_OK;
}
