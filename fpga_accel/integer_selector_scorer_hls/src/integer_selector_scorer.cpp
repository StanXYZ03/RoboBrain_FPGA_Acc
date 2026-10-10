/*
 * integer_selector_scorer.cpp - bit-exact integer Selector scorer kernel.
 *
 * Only integer arithmetic is used.  The exact floor square root and the exact
 * signed-rational round-half-away-from-zero division are re-implemented with
 * host-compatible integer loops so g++, HLS C-simulation and HLS synthesis all
 * execute the same code and produce the same bytes as the frozen Python golden
 * ``evaluation/selector_integer_scorer_audit.py`` (whose primitives are
 * ``floor_sqrt_int64`` and ``round_half_away_div``).
 *
 * On-chip layout:
 *   wv_local[128][1024]   int8   one bank per output row (dim1 complete)
 *   gm_row[1024]          int8   current group row, cyclic-16 for the packed
 *                                high-throughput load
 *   sem_arr[4096]         int32  per-group semantic_q16 (two-pass min-max)
 *   l2_arr[4096]          uint16 per-group l2_int      (two-pass min-max)
 *
 * The weight tensor is cached once per invocation and then re-used for every
 * group, so the Wv AXI traffic is a one-time cost.
 */
#include "integer_selector_scorer.h"

/*
 * Frozen analytical width bounds (checked at compile time, mirrored in
 * quantization_contract.md and integer_width_contract.json).  ACTIVATION_MAX_ABS
 * = 128 and WEIGHT_MAX_ABS = 127 give PROJECTION_BOUND = 1024*128*127.
 */
static_assert(IS_EMBED_DIM * 128LL * 127LL <= 2147483647LL,
              "visual projection fits INT32");
static_assert(IS_QUERY_DIM * (IS_EMBED_DIM * 128LL * 127LL) * 128LL <=
                  9223372036854775807LL,
              "semantic dot fits INT64");
static_assert(IS_QUERY_DIM * (IS_EMBED_DIM * 128LL * 127LL) *
                      (IS_EMBED_DIM * 128LL * 127LL) <=
                  9223372036854775807LL,
              "semantic norm2 fits INT64");
static_assert((IS_QUERY_DIM * (IS_EMBED_DIM * 128LL * 127LL) * 128LL) * 65536LL <=
                  9223372036854775807LL,
              "dot*2^16 fits INT64");
static_assert(2LL * (IS_QUERY_DIM * 12LL) * 65536LL <= 2147483647LL,
              "semantic_q16 fits INT32");
static_assert(IS_EMBED_DIM * 128LL * 128LL <= 9223372036854775807LL,
              "l2 squared sum fits INT64");
static_assert(4096LL <= 65535LL, "l2_int fits UINT16");
static_assert(IS_SCORE_Q_MAX == 32767, "score_q Q0.15 max");

namespace {

/* Exact ``floor(sqrt(value))`` for ``value < 2**63`` (32 fixed iterations of
 * the bit-by-bit digit method; starting one power of four above the value is
 * equivalent to starting at the largest power of four <= value). */
uint64_t is_floor_sqrt_u64(uint64_t value) {
    if (value < 2) {
        return value;
    }
    uint64_t remainder = value;
    uint64_t result = 0;
    uint64_t bit = (uint64_t)1 << 62;
    for (int i = 0; i < 32; i++) {
#pragma HLS PIPELINE II=1
        if (remainder >= result + bit) {
            remainder -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result = result >> 1;
        }
        bit = bit >> 2;
    }
    return result;
}

/* Exact unsigned long division ``num / den`` for ``den != 0``.  The remainder
 * stays strictly below ``den <= 2**63`` before each shift, so ``rem << 1``
 * cannot wrap a uint64. */
uint64_t is_udiv_u64(uint64_t num, uint64_t den) {
    uint64_t quotient = 0;
    uint64_t remainder = 0;
    for (int i = 63; i >= 0; i--) {
#pragma HLS PIPELINE II=2
        remainder = (remainder << 1) | ((num >> i) & (uint64_t)1);
        if (remainder >= den) {
            remainder -= den;
            quotient |= (uint64_t)1 << i;
        }
    }
    return quotient;
}

/* Exact signed rational round-half-away-from-zero ``num / den`` with
 * ``den > 0``: magnitude = (2*|num| + den) / (2*den), then re-apply the sign.
 * The frozen Python golden is ``round_half_away_div``. */
int64_t is_round_half_away_div(int64_t num, int64_t den) {
    uint64_t magnitude_num =
        (num < 0) ? (uint64_t)(-(num + 1)) + 1u : (uint64_t)num;
    uint64_t numerator = 2u * magnitude_num + (uint64_t)den;
    uint64_t denominator = 2u * (uint64_t)den;
    int64_t magnitude = (int64_t)is_udiv_u64(numerator, denominator);
    return (num < 0) ? -magnitude : magnitude;
}

/* Exact min-max map to [0, 32767]; an all-equal vector maps to zero. */
uint16_t is_minmax_q15(int64_t value, int64_t low, int64_t high) {
    if (high <= low) {
        return (uint16_t)0;
    }
    int64_t numerator = (value - low) * (int64_t)IS_SCORE_Q_MAX;
    return (uint16_t)is_round_half_away_div(numerator, high - low);
}

/*
 * Shared kernel body.  ``kDebug`` is a compile-time constant: the synthesized
 * top instantiates ``<false>`` and the host testbench instantiates ``<true>``
 * through ``integer_selector_scorer_debug``.  No debug port is ever
 * synthesized.
 */
template <bool kDebug>
int is_scorer_core(const is_word_t* group_mean_q, const is_word_t* wv_q,
                   const is_word_t* query_q, uint16_t* score_q,
                   int32_t* dbg_semantic_q16, uint16_t* dbg_l2_int,
                   uint16_t* dbg_semantic_norm_q15,
                   uint16_t* dbg_l2_norm_q15, int group_count) {
    /* Cached per-tensor Wv: one BRAM bank per output row (128 x 1 KiB). */
    int8_t wv_local[IS_WV_ROWS][IS_EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=wv_local dim=1 complete
    /* Per-group intermediates for the two-pass min-max / output flow. */
    int32_t semantic_arr[IS_MAX_GROUPS];
    uint16_t l2_arr[IS_MAX_GROUPS];
    /* Current group row and the INT8 prompt query. */
    int8_t group_row[IS_EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=group_row cyclic factor=16 dim=1
    int8_t query_local[IS_QUERY_DIM];

    /* 1. Cache the accepted seed-20261003 Wv INT8 tensor once per invocation.
     *    The packed AXI word holds 16 little-endian lanes; the 16 bytes of one
     *    word go to the same bank at 16 consecutive addresses. */
    for (int w = 0; w < IS_WV_WORDS; w++) {
#pragma HLS LOOP_TRIPCOUNT min=8192 max=8192
        is_word_t word = wv_q[w];
        int row = w / IS_GM_WORDS_PER_GROUP;
        int base = (w % IS_GM_WORDS_PER_GROUP) * IS_PACKED_LANES;
        for (int lane = 0; lane < IS_PACKED_LANES; lane++) {
#pragma HLS UNROLL
            wv_local[row][base + lane] = (int8_t)is_word_lane(word, lane);
        }
    }

    /* 2. Cache the 128-lane query (8 packed words). */
    for (int w = 0; w < IS_QUERY_WORDS; w++) {
#pragma HLS PIPELINE II=1
        is_word_t word = query_q[w];
        for (int lane = 0; lane < IS_PACKED_LANES; lane++) {
#pragma HLS UNROLL
            query_local[w * IS_PACKED_LANES + lane] =
                (int8_t)is_word_lane(word, lane);
        }
    }

    int64_t semantic_low = 0;
    int64_t semantic_high = 0;
    int64_t l2_low = 0;
    int64_t l2_high = 0;

    /* 3. Per-group scoring pass. */
    for (int g = 0; g < group_count; g++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096

        /* 3a. Load the packed group-mean row into on-chip memory. */
        for (int w = 0; w < IS_GM_WORDS_PER_GROUP; w++) {
#pragma HLS PIPELINE II=1
            is_word_t word = group_mean_q[g * IS_GM_WORDS_PER_GROUP + w];
            for (int lane = 0; lane < IS_PACKED_LANES; lane++) {
#pragma HLS UNROLL
                group_row[w * IS_PACKED_LANES + lane] =
                    (int8_t)is_word_lane(word, lane);
            }
        }

        /* 3b. Exact INT32 visual projection and INT64 squared group L2.  The
         *     analytical bound is |p| <= 1024*128*127 = 16,646,144 < 2**31-1
         *     and sum_1024 group_mean_q^2 <= 16,777,216. */
        int32_t projection[IS_WV_ROWS];
#pragma HLS ARRAY_PARTITION variable=projection complete
        for (int o = 0; o < IS_WV_ROWS; o++) {
#pragma HLS UNROLL
            projection[o] = 0;
        }
        int64_t l2_squared = 0;
        for (int d = 0; d < IS_EMBED_DIM; d++) {
#pragma HLS PIPELINE II=1
            int32_t sample = (int32_t)group_row[d];
            for (int o = 0; o < IS_WV_ROWS; o++) {
#pragma HLS UNROLL
                projection[o] += sample * (int32_t)wv_local[o][d];
            }
            l2_squared += (int64_t)sample * (int64_t)sample;
        }

        /* 3c. Exact INT64 semantic dot and nonnegative INT64 norm2. */
        int64_t dot = 0;
        int64_t norm2 = 0;
        for (int o = 0; o < IS_WV_ROWS; o++) {
#pragma HLS PIPELINE II=1
            int32_t value = projection[o];
            dot += (int64_t)value * (int64_t)query_local[o];
            norm2 += (int64_t)value * (int64_t)value;
        }

        /* 3d. semantic_q16 = round_half_away((dot << 16)/floor_sqrt(norm2));
         *     the left shift is written as a multiplication of a signed value
         *     by 2**16, never as a signed left shift. */
        int32_t semantic_q16 = 0;
        if (norm2 != 0) {
            uint64_t root = is_floor_sqrt_u64((uint64_t)norm2);
            int64_t numerator =
                dot * (int64_t)((int64_t)1 << IS_SEMANTIC_FRACTIONAL_BITS);
            semantic_q16 =
                (int32_t)is_round_half_away_div(numerator, (int64_t)root);
        }

        /* 3e. l2_int = floor_sqrt(sum_1024 group_mean_q^2) <= 4096. */
        uint16_t l2_int = (uint16_t)is_floor_sqrt_u64((uint64_t)l2_squared);

        semantic_arr[g] = semantic_q16;
        l2_arr[g] = l2_int;
        if (kDebug) {
            dbg_semantic_q16[g] = semantic_q16;
            dbg_l2_int[g] = l2_int;
        }
        if (g == 0) {
            semantic_low = semantic_high = (int64_t)semantic_q16;
            l2_low = l2_high = (int64_t)l2_int;
        } else {
            if ((int64_t)semantic_q16 < semantic_low) {
                semantic_low = (int64_t)semantic_q16;
            }
            if ((int64_t)semantic_q16 > semantic_high) {
                semantic_high = (int64_t)semantic_q16;
            }
            if ((int64_t)l2_int < l2_low) {
                l2_low = (int64_t)l2_int;
            }
            if ((int64_t)l2_int > l2_high) {
                l2_high = (int64_t)l2_int;
            }
        }
    }

    /* 4. Second pass: independent exact min-max to Q0.15 then the lambda=0.5
     *    half-away combine, clamped to [0, 32767]. */
    for (int g = 0; g < group_count; g++) {
        uint16_t semantic_norm =
            is_minmax_q15((int64_t)semantic_arr[g], semantic_low, semantic_high);
        uint16_t l2_norm =
            is_minmax_q15((int64_t)l2_arr[g], l2_low, l2_high);
        int64_t combined = (int64_t)semantic_norm + (int64_t)l2_norm;
        int64_t score = is_round_half_away_div(combined, 2);
        if (score < 0) {
            score = 0;
        }
        if (score > IS_SCORE_Q_MAX) {
            score = IS_SCORE_Q_MAX;
        }
        score_q[g] = (uint16_t)score;
        if (kDebug) {
            dbg_semantic_norm_q15[g] = semantic_norm;
            dbg_l2_norm_q15[g] = l2_norm;
        }
    }

    return IS_STATUS_OK;
}

} /* namespace */

extern "C" int integer_selector_scorer(const is_word_t* group_mean_q,
                                       const is_word_t* wv_q,
                                       const is_word_t* query_q,
                                       uint16_t* score_q, int group_count) {
#pragma HLS INTERFACE m_axi port=group_mean_q offset=slave bundle=gmem_gm \
    depth=IS_MAX_GM_WORDS max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=wv_q offset=slave bundle=gmem_wv \
    depth=IS_WV_WORDS max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=query_q offset=slave bundle=gmem_query \
    depth=IS_QUERY_WORDS max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=score_q offset=slave bundle=gmem_score \
    depth=IS_MAX_GROUPS max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=group_count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (group_mean_q == 0 || wv_q == 0 || query_q == 0 || score_q == 0) {
        return IS_STATUS_ERR_NULL;
    }
    if (!is_group_count_legal(group_count)) {
        return IS_STATUS_ERR_GROUP_COUNT;
    }
    return is_scorer_core<false>(group_mean_q, wv_q, query_q, score_q, 0, 0, 0,
                                 0, group_count);
}

#ifndef __SYNTHESIS__
extern "C" int integer_selector_scorer_debug(
    const is_word_t* group_mean_q, const is_word_t* wv_q,
    const is_word_t* query_q, uint16_t* score_q, int32_t* semantic_q16_out,
    uint16_t* l2_int_out, uint16_t* semantic_norm_q15_out,
    uint16_t* l2_norm_q15_out, int group_count) {
    if (group_mean_q == 0 || wv_q == 0 || query_q == 0 || score_q == 0 ||
        semantic_q16_out == 0 || l2_int_out == 0 ||
        semantic_norm_q15_out == 0 || l2_norm_q15_out == 0) {
        return IS_STATUS_ERR_NULL;
    }
    if (!is_group_count_legal(group_count)) {
        return IS_STATUS_ERR_GROUP_COUNT;
    }
    return is_scorer_core<true>(group_mean_q, wv_q, query_q, score_q,
                                semantic_q16_out, l2_int_out,
                                semantic_norm_q15_out, l2_norm_q15_out,
                                group_count);
}

extern "C" uint64_t is_test_floor_sqrt_u64(uint64_t value) {
    return is_floor_sqrt_u64(value);
}

extern "C" int64_t is_test_round_half_away_div(int64_t numerator,
                                               int64_t denominator) {
    return is_round_half_away_div(numerator, denominator);
}

extern "C" uint16_t is_test_minmax_q15(int64_t value, int64_t low,
                                       int64_t high) {
    return is_minmax_q15(value, low, high);
}
#endif
