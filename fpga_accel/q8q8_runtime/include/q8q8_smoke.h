#ifndef ROBOBRAIN_Q8Q8_SMOKE_H
#define ROBOBRAIN_Q8Q8_SMOKE_H

#include "q8q8_runtime.h"

struct q8q8_smoke_vector {
    struct q8q8_job job;
    float expected[8];
    size_t output_offset;
    size_t bytes_used;
};

int q8q8_smoke_prepare(void *arena, size_t arena_size, uint64_t phys_base,
                       struct q8q8_smoke_vector *vector);
int q8q8_smoke_check(const void *arena,
                     const struct q8q8_smoke_vector *vector,
                     float tolerance, float *max_abs_error);

#endif
