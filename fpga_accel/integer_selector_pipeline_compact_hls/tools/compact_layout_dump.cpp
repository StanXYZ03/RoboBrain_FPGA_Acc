/*
 * compact_layout_dump.cpp - print the compact top's on-chip layout for the
 * Python layout tests.  It compiles the compact translation unit (which
 * re-uses the accepted upstream cores unmodified) and exposes the two layout
 * mirrors so they can be compared against the frozen Python authority.
 *
 *   compact_layout_dump out <budget>   -> output-arena offsets/sizes/total
 *   compact_layout_dump in  <G>        -> input-arena offsets/total
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/integer_selector_pipeline_compact.cpp"

int main(int argc, char** argv) {
    if (argc >= 3 && strcmp(argv[1], "out") == 0) {
        int budget = atoi(argv[2]);
        struct IspcLayout layout;
        int status = ispc_fpga_layout(budget, &layout);
        printf("status %d\n", status);
        for (int i = 0; i < ISPC_LAYOUT_NUM_SEGMENTS; i++) {
            printf("seg %d %d %d\n", i, layout.offsets[i], layout.sizes[i]);
        }
        printf("total %d\n", layout.total_bytes);
        return status == 0 ? 0 : 1;
    }
    if (argc >= 3 && strcmp(argv[1], "in") == 0) {
        int group_count = atoi(argv[2]);
        struct IspcInputLayout layout;
        int status = ispc_input_layout(group_count, &layout);
        printf("status %d\n", status);
        printf("tokens %d\n", layout.tokens_offset);
        printf("group_mean %d\n", layout.group_mean_offset);
        printf("query %d\n", layout.query_offset);
        printf("total %d\n", layout.total_bytes);
        return status == 0 ? 0 : 1;
    }
    fprintf(stderr, "usage: %s out <budget> | in <G>\n", argv[0]);
    return 2;
}
