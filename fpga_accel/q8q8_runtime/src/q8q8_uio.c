#define _POSIX_C_SOURCE 200809L
#include "q8q8_uio.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int read_text(const char *path, char *buffer, size_t capacity) {
    FILE *file = fopen(path, "r");
    size_t length;
    if (file == NULL || fgets(buffer, (int)capacity, file) == NULL) {
        if (file != NULL) fclose(file);
        return -1;
    }
    fclose(file);
    length = strlen(buffer);
    if (length != 0 && buffer[length - 1] == '\n') {
        buffer[length - 1] = '\0';
    }
    return 0;
}

static int read_u64(const char *path, uint64_t *value) {
    char text[64];
    char *end = NULL;
    unsigned long long parsed;
    if (read_text(path, text, sizeof(text)) != 0) return -1;
    errno = 0;
    parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') return -1;
    *value = (uint64_t)parsed;
    return 0;
}

static int find_uio(const char *name, unsigned int *number) {
    DIR *directory = opendir("/sys/class/uio");
    struct dirent *entry;
    if (directory == NULL) return -1;
    while ((entry = readdir(directory)) != NULL) {
        char path[PATH_MAX];
        char actual[128];
        char *end = NULL;
        unsigned long parsed;
        if (strncmp(entry->d_name, "uio", 3) != 0) continue;
        parsed = strtoul(entry->d_name + 3, &end, 10);
        if (end == entry->d_name + 3 || *end != '\0') continue;
        snprintf(path, sizeof(path), "/sys/class/uio/%s/name", entry->d_name);
        if (read_text(path, actual, sizeof(actual)) == 0 &&
                strcmp(actual, name) == 0) {
            *number = (unsigned int)parsed;
            closedir(directory);
            return 0;
        }
    }
    closedir(directory);
    return -1;
}

static uint32_t uio_read32(void *opaque, uint32_t offset) {
    struct q8q8_uio *uio = (struct q8q8_uio *)opaque;
    volatile uint32_t *reg = (volatile uint32_t *)
            ((uint8_t *)uio->control + offset);
    return *reg;
}

static void uio_write32(void *opaque, uint32_t offset, uint32_t value) {
    struct q8q8_uio *uio = (struct q8q8_uio *)opaque;
    volatile uint32_t *reg = (volatile uint32_t *)
            ((uint8_t *)uio->control + offset);
    *reg = value;
}

static uint64_t uio_now_us(void *opaque) {
    struct timespec now;
    (void)opaque;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000u + (uint64_t)now.tv_nsec / 1000u;
}

static void uio_sleep_us(void *opaque, uint32_t usec) {
    struct timespec delay;
    (void)opaque;
    delay.tv_sec = usec / 1000000u;
    delay.tv_nsec = (long)(usec % 1000000u) * 1000L;
    while (nanosleep(&delay, &delay) != 0 && errno == EINTR) {}
}

static int uio_sync(void *opaque, uint64_t phys_addr, size_t size,
                    enum q8q8_sync_direction direction) {
    struct q8q8_uio *uio = (struct q8q8_uio *)opaque;
    uint64_t end;
    (void)direction;
    if (size > UINT64_MAX - phys_addr) return -1;
    end = phys_addr + size;
    if (phys_addr < uio->arena_phys ||
            end > uio->arena_phys + uio->arena_size) return -1;
    /* UIO_MEM_PHYS is mapped non-cached.  A full barrier is sufficient; no
       cache clean/invalidate syscall is required for this MVP arena. */
    __sync_synchronize();
    return 0;
}

static void uio_fence(void *opaque) {
    (void)opaque;
    __sync_synchronize();
}

static const struct q8q8_io_ops uio_ops = {
    uio_read32, uio_write32, uio_now_us, uio_sleep_us, uio_sync, uio_fence
};

int q8q8_uio_open(struct q8q8_uio *uio, const char *instance_name) {
    char path[PATH_MAX];
    uint64_t control_size;
    uint64_t arena_size;
    long page_size;
    if (uio == NULL || instance_name == NULL) return Q8Q8_RT_ERR_ARGUMENT;
    memset(uio, 0, sizeof(*uio));
    uio->fd = -1;
    if (find_uio(instance_name, &uio->uio_number) != 0) return Q8Q8_RT_ERR_IO;
    snprintf(path, sizeof(path), "/sys/class/uio/uio%u/maps/map0/addr",
             uio->uio_number);
    if (read_u64(path, &uio->control_phys) != 0) return Q8Q8_RT_ERR_IO;
    snprintf(path, sizeof(path), "/sys/class/uio/uio%u/maps/map0/size",
             uio->uio_number);
    if (read_u64(path, &control_size) != 0 || control_size < 0x100) {
        return Q8Q8_RT_ERR_IO;
    }
    snprintf(path, sizeof(path), "/sys/class/uio/uio%u/maps/map1/addr",
             uio->uio_number);
    if (read_u64(path, &uio->arena_phys) != 0) return Q8Q8_RT_ERR_IO;
    snprintf(path, sizeof(path), "/sys/class/uio/uio%u/maps/map1/size",
             uio->uio_number);
    if (read_u64(path, &arena_size) != 0 || arena_size < 4096 ||
            control_size > SIZE_MAX || arena_size > SIZE_MAX) {
        return Q8Q8_RT_ERR_IO;
    }
    uio->control_size = (size_t)control_size;
    uio->arena_size = (size_t)arena_size;
    snprintf(path, sizeof(path), "/dev/uio%u", uio->uio_number);
    uio->fd = open(path, O_RDWR | O_CLOEXEC);
    if (uio->fd < 0) return Q8Q8_RT_ERR_IO;
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) goto fail;
    uio->control = mmap(NULL, uio->control_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, uio->fd, 0);
    if (uio->control == MAP_FAILED) goto fail;
    uio->arena = mmap(NULL, uio->arena_size, PROT_READ | PROT_WRITE,
                      MAP_SHARED, uio->fd, page_size);
    if (uio->arena == MAP_FAILED) goto fail;
    q8q8_device_init(&uio->device, &uio_ops, uio);
    return Q8Q8_RT_OK;

fail:
    q8q8_uio_close(uio);
    return Q8Q8_RT_ERR_IO;
}

void q8q8_uio_close(struct q8q8_uio *uio) {
    if (uio == NULL) return;
    if (uio->arena != NULL && uio->arena != MAP_FAILED) {
        munmap(uio->arena, uio->arena_size);
    }
    if (uio->control != NULL && uio->control != MAP_FAILED) {
        munmap(uio->control, uio->control_size);
    }
    if (uio->fd >= 0) close(uio->fd);
    memset(uio, 0, sizeof(*uio));
    uio->fd = -1;
}
