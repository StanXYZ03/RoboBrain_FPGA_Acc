#ifndef ROBOBRAIN_Q8Q8_APM_H
#define ROBOBRAIN_Q8Q8_APM_H

#include <stddef.h>
#include <stdint.h>

enum q8q8_apm_profile {
    Q8Q8_APM_PROFILE_TRAFFIC = 0,
    Q8Q8_APM_PROFILE_LATENCY = 1,
};

struct q8q8_apm_metric {
    uint8_t slot;
    uint8_t metric;
    const char *name;
};

struct q8q8_apm_snapshot {
    uint32_t global_cycles;
    uint32_t overflow_status;
    uint32_t counters[10];
};

struct q8q8_apm {
    int fd;
    unsigned int uio_number;
    volatile uint32_t *regs;
    size_t regs_size;
    enum q8q8_apm_profile profile;
};

uint8_t q8q8_apm_selector_byte(uint8_t slot, uint8_t metric);
const struct q8q8_apm_metric *q8q8_apm_profile_metrics(
        enum q8q8_apm_profile profile);
const char *q8q8_apm_profile_name(enum q8q8_apm_profile profile);
int q8q8_apm_open_ddr(struct q8q8_apm *apm);
int q8q8_apm_configure(struct q8q8_apm *apm,
                       enum q8q8_apm_profile profile);
int q8q8_apm_start(struct q8q8_apm *apm);
int q8q8_apm_stop(struct q8q8_apm *apm,
                  struct q8q8_apm_snapshot *snapshot);
void q8q8_apm_close(struct q8q8_apm *apm);

#endif
