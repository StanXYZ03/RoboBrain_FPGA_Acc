/*
 * integer_selector_pipeline.cpp - fused on-chip scorer + packed selector top.
 *
 * The accepted upstream translation units are compiled into this wrapper and
 * their cores are called directly, so there is exactly one copy of every
 * arithmetic rule:
 *
 *   * ``is_scorer_core<false>``            (integer_selector_scorer.cpp)
 *   * ``token_selector_core_packed``        (token_selector_hls.cpp)
 *
 * ``is_scorer_core`` is file-local (anonymous namespace) in the accepted scorer
 * translation unit, so it cannot be linked from here; including the accepted
 * .cpp (unmodified) is the minimal auditable exposure of the static core.  The
 * accepted selector core is a normal external symbol, but its translation unit
 * is included here as well so the whole datapath is a single synthesis unit
 * and HLS can inline both phases.
 *
 * The intermediate ``score_local[G]`` is declared directly in the top and
 * passed to both cores; it is not a port.  The only top-level data ports are
 * the accepted inputs (tokens_q, group_mean_q, wv_q, query_q) and the accepted
 * selector's 5-segment output (out_tokens + 4 metadata arrays).
 *
 * Synthesis note: the fused body is written *directly* in the top-level
 * ``extern "C"`` function.  Vitis HLS 2023.2 crashes in its ``HLSGen`` pass if
 * the two accepted cores are fused inside a helper function (or a C++ function
 * template) with the shared local score buffer; the flat top-level form is the
 * one the tool accepts.  The host-only debug twin is a separate function under
 * ``#ifndef __SYNTHESIS__`` and is never synthesized.
 */
#include "integer_selector_pipeline.h"

/* Reuse the accepted upstream sources byte-for-byte (never modified). */
#include "../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp"
#include "../../token_selector_hls/src/token_selector_hls.cpp"

extern "C" int integer_selector_pipeline(
    const ts_word_t* tokens_q, const is_word_t* group_mean_q,
    const is_word_t* wv_q, const is_word_t* query_q, ts_word_t* out_tokens,
    uint16_t* out_group_index, uint8_t* out_source_type, uint8_t* out_region_id,
    uint8_t* out_valid, int group_count, int grid_h, int grid_w, int budget,
    int local_cap, float scale_in, float* scale_out, int* output_count,
    int* real_count, int* context_count) {
    /* Accepted scorer inputs. */
#pragma HLS INTERFACE m_axi port = group_mean_q offset = slave bundle = gmem_in_gm depth = 262144 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = wv_q offset = slave bundle = gmem_in_wv depth = 8192 max_read_burst_length = 64 num_read_outstanding = 16
#pragma HLS INTERFACE m_axi port = query_q offset = slave bundle = gmem_in_query depth = 8 max_read_burst_length = 16 num_read_outstanding = 4
    /* Accepted selector inputs. */
#pragma HLS INTERFACE m_axi port = tokens_q offset = slave bundle = gmem_in_tokens depth = 1048576 max_read_burst_length = 64 num_read_outstanding = 16
    /* Accepted selector 5-segment output (tokens + 4 metadata arrays). */
#pragma HLS INTERFACE m_axi port = out_tokens offset = slave bundle = gmem_out_tokens depth = 1048576 max_write_burst_length = 64 num_write_outstanding = 16
#pragma HLS INTERFACE m_axi port = out_group_index offset = slave bundle = gmem_out_group depth = 4096 max_write_burst_length = 16 num_write_outstanding = 4
#pragma HLS INTERFACE m_axi port = out_source_type offset = slave bundle = gmem_out_source depth = 4096 max_write_burst_length = 16 num_write_outstanding = 4
#pragma HLS INTERFACE m_axi port = out_region_id offset = slave bundle = gmem_out_region depth = 4096 max_write_burst_length = 16 num_write_outstanding = 4
#pragma HLS INTERFACE m_axi port = out_valid offset = slave bundle = gmem_out_valid depth = 4096 max_write_burst_length = 16 num_write_outstanding = 4
    /* Scalar control/status.  NOTE: no score_q port exists. */
#pragma HLS INTERFACE s_axilite port = group_count bundle = control
#pragma HLS INTERFACE s_axilite port = grid_h bundle = control
#pragma HLS INTERFACE s_axilite port = grid_w bundle = control
#pragma HLS INTERFACE s_axilite port = budget bundle = control
#pragma HLS INTERFACE s_axilite port = local_cap bundle = control
#pragma HLS INTERFACE s_axilite port = scale_in bundle = control
#pragma HLS INTERFACE s_axilite port = scale_out bundle = control
#pragma HLS INTERFACE s_axilite port = output_count bundle = control
#pragma HLS INTERFACE s_axilite port = real_count bundle = control
#pragma HLS INTERFACE s_axilite port = context_count bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

    /*
     * On-chip only: the frozen scorer's score_q[G] lives here for the whole
     * invocation.  It is never a top-level port and never round-trips DDR.
     */
    uint16_t score_local[IS_MAX_GROUPS];

    /* Data-pointer guard (the frozen selector additionally guards metadata). */
    if (tokens_q == 0 || group_mean_q == 0 || wv_q == 0 || query_q == 0 ||
        out_tokens == 0 || out_group_index == 0 || out_source_type == 0 ||
        out_region_id == 0 || out_valid == 0) {
        return ISP_STATUS_ERR_NULL;
    }
    if (!is_group_count_legal(group_count)) {
        return ISP_STATUS_ERR_GROUP_COUNT;
    }

    /* Phase 1: accepted integer scorer, writing the on-chip score buffer. */
    int score_status = is_scorer_core<false>(
        group_mean_q, wv_q, query_q, score_local, 0, 0, 0, 0, group_count);
    if (score_status != IS_STATUS_OK) {
        return score_status;
    }

    /* Phase 2: accepted packed selector reads the on-chip score buffer. */
    struct TokenSelectorResult result;
    int select_status = token_selector_core_packed(
        tokens_q, score_local, group_count, grid_h, grid_w, budget, local_cap,
        scale_in, scale_out, out_tokens, out_group_index, out_source_type,
        out_region_id, out_valid, &result);

    if (select_status == TS_STATUS_OK) {
        if (output_count != 0) {
            *output_count = result.output_count;
        }
        if (real_count != 0) {
            *real_count = result.real_count;
        }
        if (context_count != 0) {
            *context_count = result.context_count;
        }
    } else {
        if (output_count != 0) {
            *output_count = 0;
        }
        if (real_count != 0) {
            *real_count = 0;
        }
        if (context_count != 0) {
            *context_count = 0;
        }
    }
    return select_status;
}

#ifndef __SYNTHESIS__
/*
 * Testbench-only twin: identical datapath plus the per-group on-chip
 * intermediate exposed for the byte-for-byte golden check.  Never synthesized,
 * never exported, never a port.
 */
extern "C" int integer_selector_pipeline_debug(
    const ts_word_t* tokens_q, const is_word_t* group_mean_q,
    const is_word_t* wv_q, const is_word_t* query_q, ts_word_t* out_tokens,
    uint16_t* out_group_index, uint8_t* out_source_type, uint8_t* out_region_id,
    uint8_t* out_valid, uint16_t* score_q_out, int32_t* semantic_q16_out,
    uint16_t* l2_int_out, uint16_t* semantic_norm_q15_out,
    uint16_t* l2_norm_q15_out, int group_count, int grid_h, int grid_w,
    int budget, int local_cap, float scale_in, float* scale_out,
    int* output_count, int* real_count, int* context_count) {
    if (score_q_out == 0 || semantic_q16_out == 0 || l2_int_out == 0 ||
        semantic_norm_q15_out == 0 || l2_norm_q15_out == 0) {
        return ISP_STATUS_ERR_NULL;
    }
    uint16_t score_local[IS_MAX_GROUPS];
    if (tokens_q == 0 || group_mean_q == 0 || wv_q == 0 || query_q == 0 ||
        out_tokens == 0 || out_group_index == 0 || out_source_type == 0 ||
        out_region_id == 0 || out_valid == 0) {
        return ISP_STATUS_ERR_NULL;
    }
    if (!is_group_count_legal(group_count)) {
        return ISP_STATUS_ERR_GROUP_COUNT;
    }
    int score_status = is_scorer_core<true>(
        group_mean_q, wv_q, query_q, score_local, semantic_q16_out, l2_int_out,
        semantic_norm_q15_out, l2_norm_q15_out, group_count);
    if (score_status != IS_STATUS_OK) {
        return score_status;
    }
    for (int g = 0; g < group_count; g++) {
        score_q_out[g] = score_local[g];
    }
    struct TokenSelectorResult result;
    int select_status = token_selector_core_packed(
        tokens_q, score_local, group_count, grid_h, grid_w, budget, local_cap,
        scale_in, scale_out, out_tokens, out_group_index, out_source_type,
        out_region_id, out_valid, &result);
    if (select_status == TS_STATUS_OK) {
        if (output_count != 0) {
            *output_count = result.output_count;
        }
        if (real_count != 0) {
            *real_count = result.real_count;
        }
        if (context_count != 0) {
            *context_count = result.context_count;
        }
    } else {
        if (output_count != 0) {
            *output_count = 0;
        }
        if (real_count != 0) {
            *real_count = 0;
        }
        if (context_count != 0) {
            *context_count = 0;
        }
    }
    return select_status;
}
#endif
