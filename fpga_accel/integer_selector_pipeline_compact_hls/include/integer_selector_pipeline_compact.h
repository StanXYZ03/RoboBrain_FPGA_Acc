/*
 * integer_selector_pipeline_compact.h - compact AXI interface wrapper around
 * the accepted fused integer Selector pipeline.
 *
 * This header deliberately reuses the accepted scorer/selector headers (read
 * only, never modified) so the data types, the Q0.15 score width and every
 * status code are identical to ``integer_selector_pipeline``.  The only new
 * thing is the *memory envelope*:
 *
 *   input_arena  : ONE contiguous 128-bit arena
 *                     tokens_int8      @ 0                   G*4096 B
 *                     group_mean_int8  @ G*4096              G*1024 B
 *                     query_int8       @ G*4096 + G*1024     128 B
 *                  total G*5120 + 128 B (every offset 64 B aligned)
 *   wv_q         : the 128 KiB projection weight (8192 packed words)
 *   output_arena : the canonical fixed-budget 5-segment arena defined by
 *                  evaluation.hardware_regular_token_selector.fpga_transfer_layout
 *
 * Exactly three top-level ``m_axi`` bundles and exactly one AXI-Lite control
 * bundle.  ``score_q`` stays an on-chip ``uint16_t`` buffer[G] and is never a
 * port.  Small metadata (K <= 96) is kept in bounded on-chip arrays and packed
 * into the canonical output arena after selection; the token segment is written
 * straight into the output arena (no full input-arena cache).
 */
#ifndef INTEGER_SELECTOR_PIPELINE_COMPACT_H
#define INTEGER_SELECTOR_PIPELINE_COMPACT_H

#include <stdint.h>

#include "integer_selector_scorer.h"
#include "token_selector_hls.h"

/* Status codes: identical to integer_selector_pipeline (the union of the two
 * frozen upstream contracts).  The two "illegal group count" codes share 2. */
enum ispc_status {
    ISPC_STATUS_OK = 0,
    ISPC_STATUS_ERR_NULL = 1,
    ISPC_STATUS_ERR_GROUP_COUNT = 2,
    ISPC_STATUS_ERR_GRID = 3,
    ISPC_STATUS_ERR_BUDGET = 4,
    ISPC_STATUS_ERR_INSUFFICIENT = 5,
    ISPC_STATUS_ERR_CANDIDATE_POOL = 6,
    ISPC_STATUS_ERR_INTERNAL = 7,
    ISPC_STATUS_ERR_LOCAL_CAP = 8
};

#define ISPC_LAYOUT_NUM_SEGMENTS TS_NUM_SEGMENTS /* 5 */
/* 128-bit words in one group's tokens / group-mean. */
#define ISPC_TOKENS_WORDS_PER_GROUP TS_GROUP_WORDS          /* 256 */
#define ISPC_GM_WORDS_PER_GROUP IS_GM_WORDS_PER_GROUP       /* 64  */
#define ISPC_QUERY_WORDS IS_QUERY_WORDS                     /* 8   */
/* input arena words at the largest legal G (4096): 4096*320 + 8. */
#define ISPC_INPUT_ARENA_MAX_WORDS \
    (IS_MAX_GROUPS * (ISPC_TOKENS_WORDS_PER_GROUP + ISPC_GM_WORDS_PER_GROUP) + \
     ISPC_QUERY_WORDS)
/* output arena words at the largest legal budget (96): 393792 / 16. */
#define ISPC_OUTPUT_ARENA_MAX_WORDS (393792 / IS_WORD_BYTES)
/* wv_q words: 128 * 1024 / 16. */
#define ISPC_WV_WORDS IS_WV_WORDS

/* Canonical output-arena layout (byte offsets/sizes + total). */
struct IspcLayout {
    int offsets[ISPC_LAYOUT_NUM_SEGMENTS];
    int sizes[ISPC_LAYOUT_NUM_SEGMENTS];
    int total_bytes;
};

/* Frozen input-arena layout (byte offsets + total).  All offsets are already
 * 64-byte aligned for every legal integer G; no extra padding is introduced. */
struct IspcInputLayout {
    int tokens_offset;
    int group_mean_offset;
    int query_offset;
    int total_bytes;
};

/*
 * Input-arena layout: tokens @ 0 (G*4096 B), group means immediately after
 * (G*1024 B), query immediately after (128 B); total G*5120 + 128 B.
 */
static inline int ispc_input_layout(int group_count,
                                    struct IspcInputLayout* layout) {
    if (layout == 0 || !is_group_count_legal(group_count)) {
        return ISPC_STATUS_ERR_GROUP_COUNT;
    }
    layout->tokens_offset = 0;
    layout->group_mean_offset =
        group_count * (ISPC_TOKENS_WORDS_PER_GROUP * IS_WORD_BYTES);
    layout->query_offset =
        layout->group_mean_offset +
        group_count * (ISPC_GM_WORDS_PER_GROUP * IS_WORD_BYTES);
    layout->total_bytes = layout->query_offset + ISPC_QUERY_WORDS * IS_WORD_BYTES;
    return ISPC_STATUS_OK;
}

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Frozen layout of the output arena (a direct on-chip mirror of
 * evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)):
 * every segment start is independently 64 B aligned; sizes are
 * K*4096, K*2, K, K, K and the total is 64 B aligned.
 * Returns ISPC_STATUS_ERR_BUDGET for a budget that is not a positive multiple
 * of 16 <= 96 (the accepted selector guard).
 */
int ispc_fpga_layout(int budget, struct IspcLayout* layout);

/*
 * Compact fused top.
 *
 *   input_arena  : [G*320 + 8] 128-bit packed words (tokens | group means | query)
 *   wv_q         : [8192] 128-bit packed words (row-major 128x1024 INT8)
 *   output_arena : [ispc_fpga_layout(budget).total_bytes / 16] 128-bit words
 *                  with tokens at offset 0 and the 4 metadata segments at their
 *                  frozen 64 B aligned offsets; every gap and unused entry is
 *                  written as deterministic zero through the runtime budget.
 *   score_q      : NOT a port (local on-chip uint16_t buffer[G]).
 */
int integer_selector_pipeline_compact(
    const ts_word_t* input_arena,
    const is_word_t* wv_q,
    ts_word_t* output_arena,
    int group_count,
    int grid_h,
    int grid_w,
    int budget,
    int local_cap,
    float scale_in,
    float* scale_out,
    int* output_count,
    int* real_count,
    int* context_count);

#ifdef __cplusplus
}
#endif

#endif /* INTEGER_SELECTOR_PIPELINE_COMPACT_H */
