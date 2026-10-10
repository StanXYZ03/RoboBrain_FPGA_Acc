/*
 * token_selector_hls.h - frozen fixed-point token selector interface.
 *
 * The top level uses `m_axi` data ports plus an `s_axilite` control/status
 * bundle, but the whole translation unit is plain C++ so it also compiles and
 * runs under g++ (the pragmas are ignored by a host compiler).
 *
 * The selection maths and the integer pooling rules are bit-identical to
 * `evaluation/hardware_regular_selector_fixedpoint.py`.
 *
 * The weighted-centroid cross-multiplication uses only 64-bit intermediates.
 * `TS_MAX_GRID_DIM = 128` keeps the analytic worst case at
 * `distance <= 9.13e17 < 2**63-1` and `|accumulator| <= 2.06e9`, so no
 * 128-bit arithmetic is required anywhere in the kernel.
 */
#ifndef TOKEN_SELECTOR_HLS_H
#define TOKEN_SELECTOR_HLS_H

#include <stdint.h>

#define TS_MERGE_UNIT 4
#define TS_FEATURE_DIM 1024
#define TS_GROUP_ELEMENTS (TS_MERGE_UNIT * TS_FEATURE_DIM)
#define TS_REGION_GRID 3
#define TS_NUM_REGIONS (TS_REGION_GRID * TS_REGION_GRID)
#define TS_LOCAL_CANDIDATE_CAP 16
/* Frozen compile-time cap for the per-region bounded insertion.  The global
 * candidate pool is therefore at most TS_NUM_REGIONS * TS_MAX_LOCAL_CANDIDATE_CAP
 * = 9 * 16 = 144. */
#define TS_MAX_LOCAL_CANDIDATE_CAP 16
#define TS_MAX_GROUPS 4096
#define TS_MAX_GROUP_INDEX 65535
/* Proven int64-safe per-dimension grid bound (see selector overflow proof). */
#define TS_MAX_GRID_DIM 128
/* Frozen compile-time budget bound.  Runtime budgets stay positive multiples of
 * 16 and <= 96 (boundary fixtures 16/32/48/64/80/96; production 48/64/96), so
 * every output/selected/entry loop is statically sized. */
#define TS_MAX_BUDGET 96
/* Worst-case members of one 3x3 band with G <= 4096 and grid dims <= 128 is
 * <= 491; round up to a power-of-two compile-time bound. */
#define TS_MAX_REGION_MEMBERS 512
#define TS_MAX_CANDIDATE_POOL (TS_NUM_REGIONS * TS_MAX_LOCAL_CANDIDATE_CAP)
#define TS_MAX_OUTPUT TS_MAX_BUDGET
#define TS_SCORE_Q_MAX 32767
#define TS_LAYOUT_ALIGNMENT 64
#define TS_NUM_SEGMENTS 5

#define TS_SOURCE_REAL 0
#define TS_SOURCE_CONTEXT 1

/*
 * 128-bit packed token datapath (2026-09-27 v3).
 *
 * A packed word holds 16 consecutive little-endian int8 lanes: lane L occupies
 * bits [8L+7:8L].  On a little-endian host the in-memory byte order matches the
 * lane order, so the byte at address `group*4096 + word*16 + lane` is exactly
 * the original `tokens_int8` byte.  One group is 4096 B = 256 words.  The host
 * 64 B alignment and the total byte count of the 5-segment layout are
 * unchanged; only the transfer element width becomes 128 bit.
 */
#define TS_PACKED_LANES 16
#define TS_WORD_BYTES 16
#define TS_GROUP_WORDS (TS_GROUP_ELEMENTS / TS_PACKED_LANES) /* 256 */
#define TS_MAX_WORDS (TS_MAX_GROUPS * TS_GROUP_WORDS)        /* 1048576 */

#ifdef __SYNTHESIS__
#include <ap_int.h>
typedef ap_uint<128> ts_word_t;
#else
typedef unsigned __int128 ts_word_t;
#endif

/* Lane read: sign-extends lane L (little-endian, L = byte offset). */
#ifdef __SYNTHESIS__
static inline int ts_word_lane(const ts_word_t word, int lane) {
  return (int)(signed char)(word.range(lane * 8 + 7, lane * 8).to_uint());
}
#else
static inline int ts_word_lane(const ts_word_t word, int lane) {
  return (int)(signed char)(unsigned char)((word >> (lane * 8)) &
                                           (ts_word_t)0xFF);
}
#endif

/* Lane write: leaves every other lane untouched. */
static inline ts_word_t ts_word_set_lane(ts_word_t word, int lane, int8_t value) {
  ts_word_t mask = ((ts_word_t)0xFF) << (lane * 8);
  return (ts_word_t)((word & ~mask) |
                     (((ts_word_t)(uint8_t)value) << (lane * 8)));
}

/*
 * Byte-stream helpers.  Byte index B always maps to word B/16, lane B%16; a
 * short tail is zero padded.  These are host-only conveniences (the kernel
 * never packs/unpacks whole groups) but are written against `ts_word_t` so the
 * same mapping is testable under g++ and reproducible under HLS.
 */
static inline void ts_pack_int8_to_words(const int8_t* bytes, int byte_count,
                                         ts_word_t* words, int word_count) {
  for (int w = 0; w < word_count; w++) {
    ts_word_t word = (ts_word_t)0;
    for (int lane = 0; lane < TS_PACKED_LANES; lane++) {
      int index = w * TS_PACKED_LANES + lane;
      int8_t value = (index < byte_count) ? bytes[index] : (int8_t)0;
      word = ts_word_set_lane(word, lane, value);
    }
    words[w] = word;
  }
}

static inline void ts_unpack_words_to_int8(const ts_word_t* words,
                                           int word_count, int8_t* bytes,
                                           int byte_count) {
  for (int index = 0; index < byte_count; index++) {
    int w = index / TS_PACKED_LANES;
    int lane = index % TS_PACKED_LANES;
    bytes[index] =
        (w < word_count) ? (int8_t)ts_word_lane(words[w], lane) : (int8_t)0;
  }
}

/* Status codes (must match hardware_regular_selector_fixedpoint.py). */
enum ts_status {
  TS_STATUS_OK = 0,
  TS_STATUS_ERR_NULL = 1,
  TS_STATUS_ERR_GROUP_LIMIT = 2,
  TS_STATUS_ERR_GRID = 3,
  TS_STATUS_ERR_BUDGET = 4,
  TS_STATUS_ERR_INSUFFICIENT = 5,
  TS_STATUS_ERR_CANDIDATE_POOL = 6,
  TS_STATUS_ERR_INTERNAL = 7,
  TS_STATUS_ERR_LOCAL_CAP = 8
};

struct TokenSelectorResult {
  int status;
  int output_count;
  int real_count;
  int context_count;
};

struct TokenSelectorLayout {
  int offsets[TS_NUM_SEGMENTS];
  int sizes[TS_NUM_SEGMENTS];
  int total_bytes;
};

#ifdef __cplusplus
extern "C" {
#endif

/* Guard: 1 <= group_count <= 65535 (uint16 representative index). */
int token_selector_check_group_limit(int group_count);
/* Guard: 1 <= grid_h, grid_w <= TS_MAX_GRID_DIM (128). */
int token_selector_check_grid_dims(int grid_h, int grid_w);
/* Guard: positive multiple of 16; production budgets are 48/64/96. */
int token_selector_check_budget(int budget);
/* 64-byte aligned 5-segment transfer layout description. */
int token_selector_layout(int budget, struct TokenSelectorLayout* layout);

/*
 * Core selection without any AXI plumbing.  `out_tokens` must provide room for
 * max(group_count, budget) * TS_GROUP_ELEMENTS bytes.  Returns a status code and
 * fills `result`.
 */
int token_selector_core(
    const int8_t* tokens_q,
    const uint16_t* score_q,
    int group_count,
    int grid_h,
    int grid_w,
    int budget,
    int local_cap,
    float scale_in,
    float* scale_out,
    int8_t* out_tokens,
    uint16_t* out_group_index,
    uint8_t* out_source_type,
    uint8_t* out_region_id,
    uint8_t* out_valid,
    struct TokenSelectorResult* result);

/*
 * HLS top level: `m_axi` for the 5 output segments plus the 2 inputs, and
 * `s_axilite` for every scalar.  Returns the status; output_count/real_count/
 * context_count/scale_out are written through the s_axilite registers.
 */
int token_selector_top(
    const int8_t* tokens_q,
    const uint16_t* score_q,
    int8_t* out_tokens,
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

/*
 * Packed (128-bit word) twin of `token_selector_core`.  Selection maths,
 * metadata and context rounding are bit-identical to the int8 core; only the
 * token read path differs: each 128-bit word yields 16 lanes accumulated in
 * parallel, and real groups are copied as 256 words.
 */
int token_selector_core_packed(
    const ts_word_t* tokens_q,
    const uint16_t* score_q,
    int group_count,
    int grid_h,
    int grid_w,
    int budget,
    int local_cap,
    float scale_in,
    float* scale_out,
    ts_word_t* out_tokens,
    uint16_t* out_group_index,
    uint8_t* out_source_type,
    uint8_t* out_region_id,
    uint8_t* out_valid,
    struct TokenSelectorResult* result);

/*
 * Packed HLS top level: `tokens_q` and `out_tokens` are `ap_uint<128>` m_axi
 * ports (128-bit HPC data width); every metadata port and scalar is unchanged
 * from `token_selector_top`.
 */
int token_selector_top_packed(
    const ts_word_t* tokens_q,
    const uint16_t* score_q,
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

#ifdef __cplusplus
}
#endif

#endif /* TOKEN_SELECTOR_HLS_H */
