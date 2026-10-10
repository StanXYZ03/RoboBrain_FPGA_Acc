/*
 * integer_selector_pipeline.h - single HLS top that fuses the accepted
 * bit-exact integer Selector scorer and the accepted packed v4 token selector
 * on chip.
 *
 * Dataflow (one invocation, one image/prompt, one budget):
 *
 *   group_mean_q + wv_q + query_q
 *        -> [ is_scorer_core ]  -> score_local[G]   (on-chip BRAM only)
 *        -> [ token_selector_core_packed ] -> selected tokens + metadata
 *
 * The intermediate ``score_q[G]`` is a local on-chip buffer inside the top:
 * it is **never** a top-level ``m_axi`` port and is **never** written to DDR.
 * The top deliberately exposes *both* ``tokens_q`` and ``group_mean_q``, and
 * this round does not change the accepted group-mean quantisation contract.
 *
 * Reuse strategy (no second copy of the maths, no edit of the accepted dirs):
 * the two accepted translation units are compiled *into this wrapper* and the
 * existing cores are called directly:
 *
 *   * ``fpga_accel/integer_selector_scorer_hls/src/integer_selector_scorer.cpp``
 *     is included here because ``is_scorer_core<>`` lives in an anonymous
 *     namespace in that file (static visibility) and therefore cannot be
 *     linked from a separate translation unit.
 *   * ``fpga_accel/token_selector_hls/src/token_selector_hls.cpp`` is included
 *     here as well so the whole fused datapath is one translation unit.
 *
 * Both upstream files are reused byte-for-byte and never modified.  The
 * synthesized top is only aware of the pipeline top's own ``#pragma HLS
 * INTERFACE`` lines; the unreachable upstream tops (with their interface
 * pragmas) are pruned by the tool.
 *
 * Status codes are the union of the two frozen contracts (the two "illegal
 * group count" codes share the numeric value 2):
 *   0 OK, 1 NULL, 2 GROUP_COUNT/GROUP_LIMIT, 3 GRID, 4 BUDGET,
 *   5 INSUFFICIENT, 6 CANDIDATE_POOL, 7 INTERNAL, 8 LOCAL_CAP.
 */
#ifndef INTEGER_SELECTOR_PIPELINE_H
#define INTEGER_SELECTOR_PIPELINE_H

#include <stdint.h>

#include "integer_selector_scorer.h"
#include "token_selector_hls.h"

/* Combined pipeline status codes. */
enum isp_status {
    ISP_STATUS_OK = 0,
    ISP_STATUS_ERR_NULL = 1,
    ISP_STATUS_ERR_GROUP_COUNT = 2, /* == IS_STATUS_ERR_GROUP_COUNT
                                       == TS_STATUS_ERR_GROUP_LIMIT */
    ISP_STATUS_ERR_GRID = 3,
    ISP_STATUS_ERR_BUDGET = 4,
    ISP_STATUS_ERR_INSUFFICIENT = 5,
    ISP_STATUS_ERR_CANDIDATE_POOL = 6,
    ISP_STATUS_ERR_INTERNAL = 7,
    ISP_STATUS_ERR_LOCAL_CAP = 8
};

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fused HLS top.
 *
 *   tokens_q      : [G * 256] 128-bit packed INT8 words (4x1024 lanes/group)
 *   group_mean_q  : [G * 64]  128-bit packed INT8 words (1024 lanes/group)
 *   wv_q          : [8192]    128-bit packed INT8 words (128x1024)
 *   query_q       : [8]       128-bit packed INT8 words (128 lanes)
 *   out_tokens    : [budget * 256] 128-bit packed INT8 words
 *   out_group_index/source_type/region_id/valid : [budget] metadata
 *   score_q       : NOT a port - produced on chip
 *
 * ``scale_out``/``output_count``/``real_count``/``context_count`` are the
 * register pass-through outputs of the accepted packed selector.
 */
int integer_selector_pipeline(
    const ts_word_t* tokens_q,
    const is_word_t* group_mean_q,
    const is_word_t* wv_q,
    const is_word_t* query_q,
    ts_word_t* out_tokens,
    uint16_t* out_group_index,
    uint8_t* out_source_type,
    uint8_t* out_region_id,
    uint8_t* out_valid,
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

#ifndef __SYNTHESIS__
/*
 * Testbench-only twin: identical datapath plus the per-group on-chip
 * intermediate exposed for the byte-for-byte golden check.  Never synthesized,
 * never exported, never a port.
 */
int integer_selector_pipeline_debug(
    const ts_word_t* tokens_q,
    const is_word_t* group_mean_q,
    const is_word_t* wv_q,
    const is_word_t* query_q,
    ts_word_t* out_tokens,
    uint16_t* out_group_index,
    uint8_t* out_source_type,
    uint8_t* out_region_id,
    uint8_t* out_valid,
    uint16_t* score_q_out,
    int32_t* semantic_q16_out,
    uint16_t* l2_int_out,
    uint16_t* semantic_norm_q15_out,
    uint16_t* l2_norm_q15_out,
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
#endif

#ifdef __cplusplus
}
#endif

#endif /* INTEGER_SELECTOR_PIPELINE_H */
