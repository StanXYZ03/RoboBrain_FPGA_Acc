#ifndef ROBOBRAIN_Q8Q8_ARENA_H
#define ROBOBRAIN_Q8Q8_ARENA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct q8q8_arena {
    volatile uint8_t *base;
    size_t size;
    size_t used;
    uint64_t phys_base;
};

struct q8q8_buffer {
    volatile uint8_t *data;
    uint64_t phys;
    size_t offset;
    size_t size;
};

int q8q8_arena_init(struct q8q8_arena *arena, void *base, size_t size,
                    uint64_t phys_base);
int q8q8_arena_alloc(struct q8q8_arena *arena, size_t size, size_t alignment,
                     struct q8q8_buffer *buffer);
void q8q8_arena_reset(struct q8q8_arena *arena);

/* The reserved UIO arena is mapped as non-cached device memory on ZynqMP.
 * These helpers deliberately issue byte transactions instead of libc vector
 * accesses or DC ZVA, which are not valid for that mapping. */
void q8q8_buffer_fill(const struct q8q8_buffer *buffer, uint8_t value);
int q8q8_buffer_store(const struct q8q8_buffer *buffer, size_t offset,
                      const void *source, size_t size);
int q8q8_buffer_load(void *destination, const struct q8q8_buffer *buffer,
                     size_t offset, size_t size);

#ifdef __cplusplus
}
#endif

#endif
