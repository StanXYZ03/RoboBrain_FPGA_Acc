#define _POSIX_C_SOURCE 200809L
#include "q8q8_apm.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

enum {
    DDR_APM_PHYS = UINT32_C(0xfd0b0000),
    APM_GCC_LOW = 0x004,
    APM_ISR = 0x038,
    APM_MSR0 = 0x044,
    APM_MSR1 = 0x048,
    APM_MSR2 = 0x04c,
    APM_MC0 = 0x100,
    APM_CONTROL = 0x300,
    APM_METRIC_ENABLE = 1u << 0,
    APM_METRIC_RESET = 1u << 1,
    APM_GCC_ENABLE = 1u << 16,
    APM_GCC_RESET = 1u << 17,
    APM_COUNTER_OVERFLOW_MASK = 0x1ff8u,
};

/* ZynqMP DDR_APM slots 1/2 are the two striped CCI-to-DDR paths.  They
   observe the combined HPC0/HPC1 traffic and must not be attributed to an
   individual HLS bundle.  Slot 3 includes PL HP0, which carries AUX in the
   v2 block design. */
static const struct q8q8_apm_metric traffic_metrics[10] = {
    {1, 3, "cci_ddr1_read_bytes"},
    {1, 1, "cci_ddr1_read_transactions"},
    {2, 3, "cci_ddr2_read_bytes"},
    {2, 1, "cci_ddr2_read_transactions"},
    {3, 3, "hp0_ddr3_read_bytes"},
    {3, 1, "hp0_ddr3_read_transactions"},
    {3, 2, "hp0_ddr3_write_bytes"},
    {3, 0, "hp0_ddr3_write_transactions"},
    {1, 11, "cci_ddr1_read_bursts"},
    {3, 4, "hp0_ddr3_write_beats"},
};

static const struct q8q8_apm_metric latency_metrics[10] = {
    {1, 5, "cci_ddr1_total_read_latency"},
    {1, 8, "cci_ddr1_master_read_idle"},
    {1, 15, "cci_ddr1_max_read_latency"},
    {2, 5, "cci_ddr2_total_read_latency"},
    {2, 8, "cci_ddr2_master_read_idle"},
    {2, 15, "cci_ddr2_max_read_latency"},
    {3, 5, "hp0_ddr3_total_read_latency"},
    {3, 8, "hp0_ddr3_master_read_idle"},
    {3, 6, "hp0_ddr3_total_write_latency"},
    {3, 7, "hp0_ddr3_slave_write_idle"},
};

static int read_text(const char *path, char *buffer, size_t capacity) {
    FILE *file = fopen(path, "r");
    size_t length;
    if (file == NULL || fgets(buffer, (int)capacity, file) == NULL) {
        if (file != NULL) fclose(file);
        return -1;
    }
    fclose(file);
    length = strlen(buffer);
    if (length != 0 && buffer[length - 1] == '\n') buffer[length - 1] = '\0';
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

static int find_ddr_apm(unsigned int *number, size_t *map_size) {
    DIR *directory = opendir("/sys/class/uio");
    struct dirent *entry;
    if (directory == NULL) return -1;
    while ((entry = readdir(directory)) != NULL) {
        char path[PATH_MAX];
        char name[128];
        char *end = NULL;
        unsigned long parsed;
        uint64_t address;
        uint64_t size;
        if (strncmp(entry->d_name, "uio", 3) != 0) continue;
        parsed = strtoul(entry->d_name + 3, &end, 10);
        if (end == entry->d_name + 3 || *end != '\0') continue;
        snprintf(path, sizeof(path), "/sys/class/uio/%s/name", entry->d_name);
        if (read_text(path, name, sizeof(name)) != 0 ||
                strcmp(name, "axi-pmon") != 0) continue;
        snprintf(path, sizeof(path),
                 "/sys/class/uio/%s/maps/map0/addr", entry->d_name);
        if (read_u64(path, &address) != 0 || address != DDR_APM_PHYS) continue;
        snprintf(path, sizeof(path),
                 "/sys/class/uio/%s/maps/map0/size", entry->d_name);
        if (read_u64(path, &size) != 0 || size < 0x304 || size > SIZE_MAX) {
            closedir(directory);
            return -1;
        }
        *number = (unsigned int)parsed;
        *map_size = (size_t)size;
        closedir(directory);
        return 0;
    }
    closedir(directory);
    return -1;
}

static uint32_t reg_read(const struct q8q8_apm *apm, uint32_t offset) {
    return apm->regs[offset / sizeof(uint32_t)];
}

static void reg_write(struct q8q8_apm *apm, uint32_t offset, uint32_t value) {
    apm->regs[offset / sizeof(uint32_t)] = value;
    __sync_synchronize();
}

uint8_t q8q8_apm_selector_byte(uint8_t slot, uint8_t metric) {
    if (slot > 7 || metric > 31) return UINT8_C(0xff);
    return (uint8_t)((slot << 5) | metric);
}

const struct q8q8_apm_metric *q8q8_apm_profile_metrics(
        enum q8q8_apm_profile profile) {
    if (profile == Q8Q8_APM_PROFILE_TRAFFIC) return traffic_metrics;
    if (profile == Q8Q8_APM_PROFILE_LATENCY) return latency_metrics;
    return NULL;
}

const char *q8q8_apm_profile_name(enum q8q8_apm_profile profile) {
    if (profile == Q8Q8_APM_PROFILE_TRAFFIC) return "traffic";
    if (profile == Q8Q8_APM_PROFILE_LATENCY) return "latency";
    return "invalid";
}

int q8q8_apm_open_ddr(struct q8q8_apm *apm) {
    char path[64];
    if (apm == NULL) return -1;
    memset(apm, 0, sizeof(*apm));
    apm->fd = -1;
    if (find_ddr_apm(&apm->uio_number, &apm->regs_size) != 0) return -1;
    snprintf(path, sizeof(path), "/dev/uio%u", apm->uio_number);
    apm->fd = open(path, O_RDWR | O_CLOEXEC);
    if (apm->fd < 0) return -1;
    apm->regs = mmap(NULL, apm->regs_size, PROT_READ | PROT_WRITE,
                     MAP_SHARED, apm->fd, 0);
    if (apm->regs == MAP_FAILED) {
        apm->regs = NULL;
        q8q8_apm_close(apm);
        return -1;
    }
    return 0;
}

int q8q8_apm_configure(struct q8q8_apm *apm,
                       enum q8q8_apm_profile profile) {
    const struct q8q8_apm_metric *metrics = q8q8_apm_profile_metrics(profile);
    uint32_t selectors[3] = {0, 0, 0};
    uint32_t control;
    unsigned int counter;
    if (apm == NULL || apm->regs == NULL || metrics == NULL) return -1;
    control = reg_read(apm, APM_CONTROL);
    control &= ~(APM_METRIC_ENABLE | APM_GCC_ENABLE);
    reg_write(apm, APM_CONTROL, control);
    reg_write(apm, APM_CONTROL,
              control | APM_METRIC_RESET | APM_GCC_RESET);
    reg_write(apm, APM_CONTROL, control);
    for (counter = 0; counter < 10; ++counter) {
        const unsigned int word = counter / 4;
        const unsigned int shift = (counter % 4) * 8;
        selectors[word] |= (uint32_t)q8q8_apm_selector_byte(
                metrics[counter].slot, metrics[counter].metric) << shift;
    }
    reg_write(apm, APM_MSR0, selectors[0]);
    reg_write(apm, APM_MSR1, selectors[1]);
    reg_write(apm, APM_MSR2, selectors[2]);
    apm->profile = profile;
    return 0;
}

int q8q8_apm_start(struct q8q8_apm *apm) {
    uint32_t control;
    if (apm == NULL || apm->regs == NULL) return -1;
    control = reg_read(apm, APM_CONTROL);
    control &= ~(APM_METRIC_ENABLE | APM_GCC_ENABLE);
    reg_write(apm, APM_CONTROL,
              control | APM_METRIC_RESET | APM_GCC_RESET);
    reg_write(apm, APM_CONTROL, control);
    reg_write(apm, APM_ISR, APM_COUNTER_OVERFLOW_MASK | 1u);
    reg_write(apm, APM_CONTROL,
              control | APM_METRIC_ENABLE | APM_GCC_ENABLE);
    return 0;
}

int q8q8_apm_stop(struct q8q8_apm *apm,
                  struct q8q8_apm_snapshot *snapshot) {
    uint32_t control;
    unsigned int counter;
    if (apm == NULL || apm->regs == NULL || snapshot == NULL) return -1;
    control = reg_read(apm, APM_CONTROL);
    reg_write(apm, APM_CONTROL,
              control & ~(APM_METRIC_ENABLE | APM_GCC_ENABLE));
    snapshot->global_cycles = reg_read(apm, APM_GCC_LOW);
    snapshot->overflow_status = reg_read(apm, APM_ISR) &
            (APM_COUNTER_OVERFLOW_MASK | 1u);
    for (counter = 0; counter < 10; ++counter) {
        snapshot->counters[counter] = reg_read(apm,
                APM_MC0 + counter * 0x10u);
    }
    return 0;
}

void q8q8_apm_close(struct q8q8_apm *apm) {
    if (apm == NULL) return;
    if (apm->regs != NULL) munmap((void *)apm->regs, apm->regs_size);
    if (apm->fd >= 0) close(apm->fd);
    memset(apm, 0, sizeof(*apm));
    apm->fd = -1;
}
