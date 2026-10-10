#include "q8q8_apm.h"

#include <stdio.h>
#include <string.h>

static int check_profile(enum q8q8_apm_profile profile) {
    const struct q8q8_apm_metric *metrics = q8q8_apm_profile_metrics(profile);
    unsigned int index;
    if (metrics == NULL) return -1;
    for (index = 0; index < 10; ++index) {
        const uint8_t encoded = q8q8_apm_selector_byte(
                metrics[index].slot, metrics[index].metric);
        if ((encoded >> 5) != metrics[index].slot ||
                (encoded & 31u) != metrics[index].metric ||
                metrics[index].name == NULL || metrics[index].name[0] == '\0') {
            return -1;
        }
    }
    return 0;
}

int main(void) {
    const struct q8q8_apm_metric *traffic = q8q8_apm_profile_metrics(
            Q8Q8_APM_PROFILE_TRAFFIC);
    if (q8q8_apm_selector_byte(1, 3) != 0x23 ||
            q8q8_apm_selector_byte(3, 2) != 0x62 ||
            q8q8_apm_selector_byte(8, 0) != 0xff ||
            strcmp(q8q8_apm_profile_name(Q8Q8_APM_PROFILE_TRAFFIC),
                   "traffic") != 0 ||
            strcmp(q8q8_apm_profile_name(Q8Q8_APM_PROFILE_LATENCY),
                   "latency") != 0 ||
            check_profile(Q8Q8_APM_PROFILE_TRAFFIC) != 0 ||
            check_profile(Q8Q8_APM_PROFILE_LATENCY) != 0 ||
            traffic[0].slot != 1 || traffic[2].slot != 2 ||
            traffic[4].slot != 3 || traffic[9].metric != 4) {
        fprintf(stderr, "APM_UNIT_FAIL\n");
        return 1;
    }
    printf("APM_UNIT_PASS profiles=2 counters_per_profile=10 slots=1,2,3\n");
    return 0;
}
