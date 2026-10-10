#ifndef ROBOBRAIN_Q8Q8_UIO_H
#define ROBOBRAIN_Q8Q8_UIO_H

#include "q8q8_runtime.h"

struct q8q8_uio {
    int fd;
    unsigned int uio_number;
    void *control;
    size_t control_size;
    void *arena;
    size_t arena_size;
    uint64_t control_phys;
    uint64_t arena_phys;
    struct q8q8_device device;
};

int q8q8_uio_open(struct q8q8_uio *uio, const char *instance_name);
void q8q8_uio_close(struct q8q8_uio *uio);

#endif
