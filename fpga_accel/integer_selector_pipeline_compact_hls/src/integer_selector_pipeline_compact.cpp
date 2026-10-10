/*
 * integer_selector_pipeline_compact.cpp - compact AXI envelope around the
 * accepted fused integer Selector pipeline.
 *
 * Reuse strategy (no second copy of the maths, no edit of the accepted dirs):
 * the two accepted upstream translation units are #include'd here, byte for
 * byte, and their cores are called directly:
 *
 *   * ``is_scorer_core<false>``      (integer_selector_scorer.cpp)
 *   * ``token_selector_core_packed`` (token_selector_hls.cpp)
 *
 * ``is_scorer_core`` lives in an anonymous namespace in the accepted scorer
 * translation unit (static visibility), so it cannot be linked from a separate
 * translation unit; including the accepted .cpp unmodified is the minimal
 * auditable exposure.  The whole datapath is therefore one synthesis unit and
 * HLS can inline both phases, exactly as the accepted
 * ``integer_selector_pipeline`` does.
 *
 * Interface:
 *   * exactly three top-level ``m_axi`` bundles -- input_arena, wv_q,
 *     output_arena -- each 128-bit;
 *   * exactly one AXI-Lite control bundle.  NOTE: every ``m_axi`` port also
 *     gets an explicit ``s_axilite ... bundle=control`` pragma so the pointer
 *     offset registers are merged into ``control``.  Without that, Vitis HLS
 *     2023.2 silently creates a second ``control_r`` AXI-Lite bundle (observed
 *     on the accepted pipeline/scorer reports); the explicit binding is the
 *     same pattern the accepted ZCU102 q8q8 kernels use.
 *   * no ``score_q`` port: the intermediate is a local on-chip buffer[G].
 *
 * Memory envelope:
 *   * input_arena : tokens_int8 @ 0 (G*4096 B) | group_mean_int8 @ G*4096
 *                   (G*1024 B) | query_int8 @ G*5120 (128 B), total G*5120+128;
 *                   all offsets are already 64 B aligned for every legal G.
 *   * output_arena: the canonical 5-segment fixed-budget arena; tokens are
 *                   written straight to the arena start, the bounded K<=96
 *                   metadata is kept on chip and packed into the arena after
 *                   selection.  The arena is zeroed through the runtime budget
 *                   first, so every alignment gap and unused entry is
 *                   deterministic zero.
 *
 * Synthesis note (inherited from the accepted pipeline): the fused body is
 * written directly in the top-level ``extern "C"`` function.  Vitis HLS 2023.2
 * crashes in its HLSGen pass if the two cores are fused inside a helper
 * function/template with the shared local score buffer.
 */
#include "integer_selector_pipeline_compact.h"

/* Reuse the accepted upstream sources byte-for-byte (never modified). */
#include "../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp"
#include "../../token_selector_hls/src/token_selector_hls.cpp"

/* Compile-time proof that the compact status codes are the accepted ones. */
#include "../../integer_selector_pipeline_hls/include/integer_selector_pipeline.h"
static_assert(ISPC_STATUS_OK == ISP_STATUS_OK, "status OK");
static_assert(ISPC_STATUS_ERR_NULL == ISP_STATUS_ERR_NULL, "status NULL");
static_assert(ISPC_STATUS_ERR_GROUP_COUNT == ISP_STATUS_ERR_GROUP_COUNT,
              "status GROUP_COUNT");
static_assert(ISPC_STATUS_ERR_GRID == ISP_STATUS_ERR_GRID, "status GRID");
static_assert(ISPC_STATUS_ERR_BUDGET == ISP_STATUS_ERR_BUDGET, "status BUDGET");
static_assert(ISPC_STATUS_ERR_INSUFFICIENT == ISP_STATUS_ERR_INSUFFICIENT,
              "status INSUFFICIENT");
static_assert(ISPC_STATUS_ERR_CANDIDATE_POOL == ISP_STATUS_ERR_CANDIDATE_POOL,
              "status CANDIDATE_POOL");
static_assert(ISPC_STATUS_ERR_INTERNAL == ISP_STATUS_ERR_INTERNAL,
              "status INTERNAL");
static_assert(ISPC_STATUS_ERR_LOCAL_CAP == ISP_STATUS_ERR_LOCAL_CAP,
              "status LOCAL_CAP");

/* ------------------------------------------------------------------ */
/* Frozen output-arena layout (on-chip mirror of the Python authority) */
/* ------------------------------------------------------------------ */

static int ispc_align_up(int value) {
    return ((value + TS_LAYOUT_ALIGNMENT - 1) / TS_LAYOUT_ALIGNMENT) *
           TS_LAYOUT_ALIGNMENT;
}

int ispc_fpga_layout(int budget, struct IspcLayout* layout) {
    if (layout == 0) {
        return ISPC_STATUS_ERR_BUDGET;
    }
    if (!(budget > 0 && budget <= TS_MAX_BUDGET && (budget % 16) == 0)) {
        return ISPC_STATUS_ERR_BUDGET;
    }
    int sizes[ISPC_LAYOUT_NUM_SEGMENTS];
    sizes[0] = budget * TS_GROUP_ELEMENTS; /* tokens_int8      */
    sizes[1] = budget * 2;                 /* group_index u16 */
    sizes[2] = budget;                     /* source_type u8  */
    sizes[3] = budget;                     /* region_id u8    */
    sizes[4] = budget;                     /* valid u8        */
    int offset = 0;
    for (int i = 0; i < ISPC_LAYOUT_NUM_SEGMENTS; i++) {
        offset = ispc_align_up(offset);
        layout->offsets[i] = offset;
        layout->sizes[i] = sizes[i];
        offset += sizes[i];
    }
    layout->total_bytes = ispc_align_up(offset);
    return ISPC_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Bounded metadata packing (little-endian lanes, zero padded)         */
/* ------------------------------------------------------------------ */

static void ispc_put_u8_segment(ts_word_t* arena, int byte_offset,
                                const uint8_t* src, int count) {
    /* count <= TS_MAX_OUTPUT (96) -> words <= 6. */
    int words = (count + TS_PACKED_LANES - 1) / TS_PACKED_LANES;
    int base_word = byte_offset / TS_WORD_BYTES;
    for (int i = 0; i < words; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=6
        ts_word_t word = (ts_word_t)0;
        for (int lane = 0; lane < TS_PACKED_LANES; lane++) {
            int index = i * TS_PACKED_LANES + lane;
            int8_t value = (index < count) ? (int8_t)src[index] : (int8_t)0;
            word = ts_word_set_lane(word, lane, value);
        }
        arena[base_word + i] = word;
    }
}

static void ispc_put_u16_segment(ts_word_t* arena, int byte_offset,
                                 const uint16_t* src, int count) {
    /* count <= TS_MAX_OUTPUT (96) -> 2*count bytes -> words <= 12. */
    int byte_count = count * 2;
    int words = (byte_count + TS_PACKED_LANES - 1) / TS_PACKED_LANES;
    int base_word = byte_offset / TS_WORD_BYTES;
    for (int i = 0; i < words; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=12
        ts_word_t word = (ts_word_t)0;
        for (int lane = 0; lane < TS_PACKED_LANES; lane++) {
            int index = i * TS_PACKED_LANES + lane;
            int8_t value = (int8_t)0;
            if (index < byte_count) {
                uint16_t entry = src[index >> 1];
                value = (int8_t)(uint8_t)(((index & 1) == 0)
                                              ? (entry & 0xFF)
                                              : (entry >> 8));
            }
            word = ts_word_set_lane(word, lane, value);
        }
        arena[base_word + i] = word;
    }
}

/* ------------------------------------------------------------------ */
/* Compact top                                                         */
/* ------------------------------------------------------------------ */

extern "C" int integer_selector_pipeline_compact(
    const ts_word_t* input_arena, const is_word_t* wv_q,
    ts_word_t* output_arena, int group_count, int grid_h, int grid_w,
    int budget, int local_cap, float scale_in, float* scale_out,
    int* output_count, int* real_count, int* context_count) {
    /* Exactly three 128-bit m_axi bundles. */
#pragma HLS INTERFACE m_axi port=input_arena offset=slave bundle=input_arena depth=ISPC_INPUT_ARENA_MAX_WORDS max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=wv_q offset=slave bundle=wv_q depth=ISPC_WV_WORDS max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=output_arena offset=slave bundle=output_arena depth=ISPC_OUTPUT_ARENA_MAX_WORDS max_write_burst_length=64 num_write_outstanding=16
    /* Merge every m_axi offset register into the single AXI-Lite control
     * bundle; without this Vitis HLS creates a second "control_r" bundle. */
#pragma HLS INTERFACE s_axilite port=input_arena bundle=control
#pragma HLS INTERFACE s_axilite port=wv_q bundle=control
#pragma HLS INTERFACE s_axilite port=output_arena bundle=control
#pragma HLS INTERFACE s_axilite port=group_count bundle=control
#pragma HLS INTERFACE s_axilite port=grid_h bundle=control
#pragma HLS INTERFACE s_axilite port=grid_w bundle=control
#pragma HLS INTERFACE s_axilite port=budget bundle=control
#pragma HLS INTERFACE s_axilite port=local_cap bundle=control
#pragma HLS INTERFACE s_axilite port=scale_in bundle=control
#pragma HLS INTERFACE s_axilite port=scale_out bundle=control
#pragma HLS INTERFACE s_axilite port=output_count bundle=control
#pragma HLS INTERFACE s_axilite port=real_count bundle=control
#pragma HLS INTERFACE s_axilite port=context_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    /* On-chip only: no score_q port, no whole-arena input cache. */
    uint16_t score_local[IS_MAX_GROUPS];
    uint16_t meta_group_index[TS_MAX_OUTPUT];
    uint8_t meta_source_type[TS_MAX_OUTPUT];
    uint8_t meta_region_id[TS_MAX_OUTPUT];
    uint8_t meta_valid[TS_MAX_OUTPUT];

    /* Data-pointer guard. */
    if (input_arena == 0 || wv_q == 0 || output_arena == 0) {
        return ISPC_STATUS_ERR_NULL;
    }
    if (!is_group_count_legal(group_count)) {
        return ISPC_STATUS_ERR_GROUP_COUNT;
    }

    struct IspcLayout layout;
    if (ispc_fpga_layout(budget, &layout) != ISPC_STATUS_OK) {
        return ISPC_STATUS_ERR_BUDGET;
    }

    /* Deterministic zero through the runtime budget.  This zeroes the token
     * tail, every inter-segment alignment gap, the metadata of unused entries
     * and the arena tail. */
    int total_words = layout.total_bytes / IS_WORD_BYTES;
    for (int w = 0; w < total_words; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=4112 max=24612
        output_arena[w] = (ts_word_t)0;
    }

    /* One contiguous input arena -> the accepted core pointers. */
    const ts_word_t* tokens_q = input_arena;
    const is_word_t* group_mean_q = (const is_word_t*)(
        input_arena + (size_t)group_count * ISPC_TOKENS_WORDS_PER_GROUP);
    const is_word_t* query_q = (const is_word_t*)(
        input_arena +
        (size_t)group_count *
            (ISPC_TOKENS_WORDS_PER_GROUP + ISPC_GM_WORDS_PER_GROUP));

    /* Phase 1: accepted integer scorer -> on-chip score buffer. */
    int score_status = is_scorer_core<false>(group_mean_q, wv_q, query_q,
                                             score_local, 0, 0, 0, 0,
                                             group_count);
    if (score_status != IS_STATUS_OK) {
        return score_status;
    }

    /* Phase 2: accepted packed selector; tokens go straight to the arena,
     * metadata to bounded on-chip arrays. */
    struct TokenSelectorResult result;
    int select_status = token_selector_core_packed(
        tokens_q, score_local, group_count, grid_h, grid_w, budget, local_cap,
        scale_in, scale_out, output_arena, meta_group_index, meta_source_type,
        meta_region_id, meta_valid, &result);

    if (select_status != TS_STATUS_OK) {
        if (output_count != 0) {
            *output_count = 0;
        }
        if (real_count != 0) {
            *real_count = 0;
        }
        if (context_count != 0) {
            *context_count = 0;
        }
        return select_status;
    }

    int out_n = result.output_count;
    if (output_count != 0) {
        *output_count = out_n;
    }
    if (real_count != 0) {
        *real_count = result.real_count;
    }
    if (context_count != 0) {
        *context_count = result.context_count;
    }

    /* Pack the on-chip metadata into the canonical arena.  The zero pass above
     * already cleared every gap and the metadata of unused entries (identity
     * bypass writes only output_count <= budget entries). */
    ispc_put_u16_segment(output_arena, layout.offsets[1], meta_group_index,
                         out_n);
    ispc_put_u8_segment(output_arena, layout.offsets[2], meta_source_type,
                        out_n);
    ispc_put_u8_segment(output_arena, layout.offsets[3], meta_region_id,
                        out_n);
    ispc_put_u8_segment(output_arena, layout.offsets[4], meta_valid, out_n);

    return select_status;
}
