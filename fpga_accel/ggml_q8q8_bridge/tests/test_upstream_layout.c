#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "q8q8_dispatch.h"

#include <stddef.h>
#include <stdio.h>

_Static_assert(QK8_0 == 32, "upstream Q8_0 block length changed");
_Static_assert(sizeof(block_q8_0) == sizeof(struct q8q8_block_q8_0),
               "project and upstream Q8_0 sizes differ");
_Static_assert(offsetof(block_q8_0, d) ==
                       offsetof(struct q8q8_block_q8_0, scale_fp16),
               "project and upstream Q8_0 scale offsets differ");
_Static_assert(offsetof(block_q8_0, qs) ==
                       offsetof(struct q8q8_block_q8_0, values),
               "project and upstream Q8_0 payload offsets differ");

int main(void) {
    printf("UPSTREAM_Q8_0_LAYOUT_PASS qk=%d bytes=%zu scale_offset=%zu "
           "payload_offset=%zu upstream_modified=no\n",
           QK8_0, sizeof(block_q8_0), offsetof(block_q8_0, d),
           offsetof(block_q8_0, qs));
    return 0;
}
