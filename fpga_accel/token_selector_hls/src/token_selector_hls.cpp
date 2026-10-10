/*
 * token_selector_hls.cpp - fixed-point hardware-regular token selector.
 *
 * Strict C++ (no Vitis headers) so the exact same code can be exercised by the
 * g++ C testbench and, later, by Vitis HLS.  The `#pragma HLS` lines are inert
 * for a host compiler.
 *
 * Bit-exactness is defined by `evaluation/hardware_regular_selector_fixedpoint.py`:
 *   * one real anchor per non-empty 3x3 region (highest score, lowest index),
 *   * one context group per region with >= 2 members (reserving >= 1 unselected),
 *   * real = K - C, bounded local candidates then global bounded Top-K,
 *   * context pooling with non-negative score_q weights (uniform if all zero),
 *     int64 accumulation and round-half-away-from-zero,
 *   * representative = nearest unselected group to the weighted centroid using
 *     integer cross-multiplication; ties pick the lower index,
 *   * globally unique representatives emitted in ascending order,
 *   * 64-bit-only arithmetic: `1 <= grid_h, grid_w <= TS_MAX_GRID_DIM (128)`
 *     makes the squared-distance cross-multiplication provably overflow-free.
 *
 * 2026-09-27 v2 redesign (no selection-maths change):
 *   * `TS_MAX_BUDGET = 96` and `local_cap <= 16` are compile-time bounds, so
 *     every output/selected/entry loop is statically sized.
 *   * The per-region "collect every member then full sort" step is replaced by
 *     a stable bounded insertion that keeps exactly the top `local_cap`
 *     candidates; the global pool is at most 9*16 = 144 and is selected with a
 *     bounded insertion sort (no O(G^2) pass over the 4096 groups).
 *   * The anchor scan caches the running best score instead of re-reading
 *     `score_q[best]` on every comparison.
 *   * Context remaining sets are streamed from `region_of`/`is_selected`, so no
 *     `region_members[9][4096]` array is materialised.
 *
 * 2026-09-28 v4 timing redesign of the *synthesized* packed kernel (no
 * selection-maths change; the v3 bounded-insertion kernel is retained only as
 * the `#ifndef __SYNTHESIS__` int8 reference used by the testbench):
 *   * The fully-partitioned `local_top[9][16]` stable-insertion bank and its
 *     15-way insertion mux are removed.  Each region's exact top-`local_cap`
 *     candidate set is produced by a sequential deterministic multi-pass
 *     max-select that scans the region once per rank, skipping the anchor and
 *     already-picked groups (higher score first, ties to the lower index).
 *   * The <= 144 global pool and the <= 96 selected / entry sorts are sequential
 *     multi-pass selects with a single comparator per pass (no 16-way or 144-way
 *     combinational insertion structures).
 *   * The weighted-centroid representative narrows explicitly to int32 (the
 *     frozen proof bounds |delta| <= 675721074) and binds each 32x32->64 square
 *     to a DSP with a >= 2-cycle registered pipelined latency.
 *   * The 128-bit packed context-pooling path (`ts_pool_context_packed`) is
 *     deliberately unchanged: it already met the 3.650 ns budget.
 */
#include "token_selector_hls.h"

#include <string.h>

/* ------------------------------------------------------------------ */
/* Small integer helpers                                              */
/* ------------------------------------------------------------------ */

static int64_t ts_round_half_away(int64_t numerator, int64_t denominator) {
  int64_t sign = (numerator < 0) ? -1 : 1;
  uint64_t magnitude = (uint64_t)((numerator < 0) ? -numerator : numerator);
  uint64_t den = (uint64_t)denominator;
  uint64_t quotient = (2u * magnitude + den) / (2u * den);
  return sign * (int64_t)quotient;
}

static void ts_region_bounds(int size, int bounds[TS_REGION_GRID][2]) {
  for (int r = 0; r < TS_REGION_GRID; r++) {
    bounds[r][0] = r * size / TS_REGION_GRID;
    bounds[r][1] = (r + 1) * size / TS_REGION_GRID;
  }
}

/* True iff `a` ranks strictly before `b`: higher score first, ties lower index.
 * Only the frozen int8 reference kernel uses it now; the packed kernel performs
 * the same comparison inline during its multi-pass max-select. */
#ifndef __SYNTHESIS__
static int ts_rank_before(int a, int b, const uint16_t* score_q) {
  int score_a = (int)score_q[a];
  int score_b = (int)score_q[b];
  if (score_a != score_b) {
    return score_a > score_b;
  }
  return a < b;
}
#endif

static int ts_entry_less(int r1, int s1, int g1, int r2, int s2, int g2) {
  if (r1 != r2) return r1 < r2;
  if (s1 != s2) return s1 < s2;
  return g1 < g2;
}

/* ------------------------------------------------------------------ */
/* Context pooling                                                    */
/* ------------------------------------------------------------------ */

static void ts_context_weights(
    const uint16_t* score_q,
    const int* remaining,
    int count,
    int64_t* weights,
    int64_t* total) {
  int64_t sum = 0;
  for (int i = 0; i < count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    weights[i] = (int64_t)score_q[remaining[i]];
    sum += weights[i];
  }
  if (sum == 0) {
    for (int i = 0; i < count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
      weights[i] = 1;
    }
    sum = (int64_t)count;
  }
  *total = sum;
}

/* Stream the unselected members of one region out of region_of/is_selected. */
static void ts_collect_remaining(
    const int* region_of,
    const int* is_selected,
    int group_count,
    int rid,
    int* remaining,
    int* out_count) {
  int count = 0;
  for (int group = 0; group < TS_MAX_GROUPS; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    if (group >= group_count) break;
    if (region_of[group] == rid && !is_selected[group]) {
      remaining[count] = group;
      count += 1;
    }
  }
  *out_count = count;
}

static int ts_context_representative(
    const uint16_t* score_q,
    const int* remaining,
    int count,
    int grid_w) {
  if (count <= 0) {
    return -1;
  }
  int64_t weights[TS_MAX_REGION_MEMBERS];
  int64_t total = 0;
  ts_context_weights(score_q, remaining, count, weights, &total);
  int64_t row_sum = 0;
  int64_t col_sum = 0;
  for (int i = 0; i < count; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    int row = remaining[i] / grid_w;
    int col = remaining[i] % grid_w;
    /* weight <= 32767 and row/col <= 127, so the 32-bit product is exact. */
    row_sum += (int64_t)((int)weights[i] * row);
    col_sum += (int64_t)((int)weights[i] * col);
  }
  /*
   * total <= 512*32767 < 2**24 and |row_sum|,|col_sum| <= total*127 < 2**31,
   * and the frozen overflow proof bounds the cross-multiplied centroid delta by
   * |delta| <= 675721074 < 2**30 (squared distance <= 9.13e17 < 2**63-1).
   * The centroid operands are therefore narrowed explicitly to int32 and the
   * two per-axis squares are 32x32->64 multiplies bound to a DSP with a
   * >= 2-cycle registered/pipelined latency.  The squared Euclidean distance
   * and the (distance, lower index) tie rule stay bit-for-bit identical.
   */
  int32_t total32 = (int32_t)total;
  int32_t row_sum32 = (int32_t)row_sum;
  int32_t col_sum32 = (int32_t)col_sum;
  int best_group = remaining[0];
  uint64_t best_distance = 0;
  int have_best = 0;
  for (int i = 0; i < count; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    int group = remaining[i];
    int32_t row = (int32_t)(remaining[i] / grid_w);
    int32_t col = (int32_t)(remaining[i] % grid_w);
    /* row,col <= 127 and total32 < 2**24, so row*total32 fits int32 and the
     * difference is the proven-bounded centroid delta. */
    int32_t delta_row = (int32_t)(row * total32) - row_sum32;
    int32_t delta_col = (int32_t)(col * total32) - col_sum32;
    int64_t delta_row_sq = (int64_t)delta_row * delta_row;
#pragma HLS BIND_OP variable=delta_row_sq op=mul impl=dsp latency=2
    int64_t delta_col_sq = (int64_t)delta_col * delta_col;
#pragma HLS BIND_OP variable=delta_col_sq op=mul impl=dsp latency=2
    uint64_t distance = (uint64_t)delta_row_sq + (uint64_t)delta_col_sq;
    if (!have_best || distance < best_distance ||
        (distance == best_distance && group < best_group)) {
      best_distance = distance;
      best_group = group;
      have_best = 1;
    }
  }
  return best_group;
}

/*
 * The int8 kernel (`ts_pool_context`, `token_selector_core`,
 * `token_selector_top`) is the frozen v3 bounded-insertion reference.  It is
 * kept only for the g++/ASan testbench, which diffs it byte-for-byte against the
 * new multi-pass packed kernel; it is never part of the synthesized design.
 */
#ifndef __SYNTHESIS__
static void ts_pool_context(
    const int8_t* tokens_q,
    const uint16_t* score_q,
    const int* remaining,
    int count,
    int8_t* out_tokens) {
  int64_t weights[TS_MAX_REGION_MEMBERS];
  int64_t total = 0;
  ts_context_weights(score_q, remaining, count, weights, &total);
  /*
   * Dense weighted sum into a cyclically-partitioned int64 accumulator: the
   * eight unrolled lanes touch eight different banks and read eight contiguous
   * token bytes per cycle, so the pipeline sustains ~8 MAC/cycle.  The integer
   * math is bit-identical (exact commutativity of the accumulation plus one
   * round-half-away division per output element).
   */
  int64_t accumulator[TS_GROUP_ELEMENTS];
#pragma HLS ARRAY_PARTITION variable=accumulator cyclic factor=8 dim=1
  for (int element = 0; element < TS_GROUP_ELEMENTS; element++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    accumulator[element] = 0;
  }
  for (int i = 0; i < count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    int64_t weight = weights[i];
    int base = remaining[i] * TS_GROUP_ELEMENTS;
    for (int element = 0; element < TS_GROUP_ELEMENTS; element += 8) {
#pragma HLS PIPELINE II=1
      accumulator[element + 0] += weight * (int64_t)tokens_q[base + element + 0];
      accumulator[element + 1] += weight * (int64_t)tokens_q[base + element + 1];
      accumulator[element + 2] += weight * (int64_t)tokens_q[base + element + 2];
      accumulator[element + 3] += weight * (int64_t)tokens_q[base + element + 3];
      accumulator[element + 4] += weight * (int64_t)tokens_q[base + element + 4];
      accumulator[element + 5] += weight * (int64_t)tokens_q[base + element + 5];
      accumulator[element + 6] += weight * (int64_t)tokens_q[base + element + 6];
      accumulator[element + 7] += weight * (int64_t)tokens_q[base + element + 7];
    }
  }
  for (int element = 0; element < TS_GROUP_ELEMENTS; element++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    out_tokens[element] =
        (int8_t)ts_round_half_away(accumulator[element], total);
  }
}
#endif /* !__SYNTHESIS__ : end of the int8 reference kernels */

/*
 * 128-bit context pooling: one word-wide load yields 16 contiguous int8 lanes
 * per cycle, all 16 lanes multiply-accumulate in parallel into a
 * cyclically-partitioned int64 accumulator.  The integer math is bit-identical
 * to `ts_pool_context` (exact integer accumulation plus one round-half-away
 * division per output element); only the fetch/loop structure changed, so no
 * per-group byte staging buffer and no whole-G BRAM cache is needed.
 */
static void ts_pool_context_packed(
    const ts_word_t* tokens_q,
    const uint16_t* score_q,
    const int* remaining,
    int count,
    ts_word_t* out_words) {
  int64_t weights[TS_MAX_REGION_MEMBERS];
  int64_t total = 0;
  ts_context_weights(score_q, remaining, count, weights, &total);
  int64_t accumulator[TS_GROUP_ELEMENTS];
#pragma HLS ARRAY_PARTITION variable=accumulator cyclic factor=16 dim=1
  for (int element = 0; element < TS_GROUP_ELEMENTS; element++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    accumulator[element] = 0;
  }
  for (int i = 0; i < count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
    /* score_q <= 32767 and |lane| <= 128, so the 32-bit product is exact. */
    int weight = (int)weights[i];
    int base = remaining[i] * TS_GROUP_WORDS;
    for (int w = 0; w < TS_GROUP_WORDS; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
      ts_word_t word = tokens_q[base + w];
      int element = w * TS_PACKED_LANES;
      /* 16 lanes fetched per 128-bit beat, accumulated in parallel.  The lane
       * loop is written out explicitly: Vitis HLS 2023.2 rejects an inner
       * `#pragma HLS UNROLL` inside a pipelined loop (HLS 214-274). */
      accumulator[element + 0] += (int64_t)(weight * ts_word_lane(word, 0));
      accumulator[element + 1] += (int64_t)(weight * ts_word_lane(word, 1));
      accumulator[element + 2] += (int64_t)(weight * ts_word_lane(word, 2));
      accumulator[element + 3] += (int64_t)(weight * ts_word_lane(word, 3));
      accumulator[element + 4] += (int64_t)(weight * ts_word_lane(word, 4));
      accumulator[element + 5] += (int64_t)(weight * ts_word_lane(word, 5));
      accumulator[element + 6] += (int64_t)(weight * ts_word_lane(word, 6));
      accumulator[element + 7] += (int64_t)(weight * ts_word_lane(word, 7));
      accumulator[element + 8] += (int64_t)(weight * ts_word_lane(word, 8));
      accumulator[element + 9] += (int64_t)(weight * ts_word_lane(word, 9));
      accumulator[element + 10] += (int64_t)(weight * ts_word_lane(word, 10));
      accumulator[element + 11] += (int64_t)(weight * ts_word_lane(word, 11));
      accumulator[element + 12] += (int64_t)(weight * ts_word_lane(word, 12));
      accumulator[element + 13] += (int64_t)(weight * ts_word_lane(word, 13));
      accumulator[element + 14] += (int64_t)(weight * ts_word_lane(word, 14));
      accumulator[element + 15] += (int64_t)(weight * ts_word_lane(word, 15));
    }
  }
  /*
   * Rounding uses ONE time-shared divider over the 4096 elements (never a
   * replicated divider per lane); the packed word is then assembled from the
   * rounded bytes.  This keeps the datapath resource-bounded while remaining
   * bit-identical to the int8 kernel.
   */
  int8_t rounded[TS_GROUP_ELEMENTS];
#pragma HLS ARRAY_PARTITION variable=rounded cyclic factor=16 dim=1
  for (int element = 0; element < TS_GROUP_ELEMENTS; element++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    rounded[element] =
        (int8_t)ts_round_half_away(accumulator[element], total);
  }
  for (int w = 0; w < TS_GROUP_WORDS; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
    int element = w * TS_PACKED_LANES;
    ts_word_t word = (ts_word_t)0;
    word = ts_word_set_lane(word, 0, rounded[element + 0]);
    word = ts_word_set_lane(word, 1, rounded[element + 1]);
    word = ts_word_set_lane(word, 2, rounded[element + 2]);
    word = ts_word_set_lane(word, 3, rounded[element + 3]);
    word = ts_word_set_lane(word, 4, rounded[element + 4]);
    word = ts_word_set_lane(word, 5, rounded[element + 5]);
    word = ts_word_set_lane(word, 6, rounded[element + 6]);
    word = ts_word_set_lane(word, 7, rounded[element + 7]);
    word = ts_word_set_lane(word, 8, rounded[element + 8]);
    word = ts_word_set_lane(word, 9, rounded[element + 9]);
    word = ts_word_set_lane(word, 10, rounded[element + 10]);
    word = ts_word_set_lane(word, 11, rounded[element + 11]);
    word = ts_word_set_lane(word, 12, rounded[element + 12]);
    word = ts_word_set_lane(word, 13, rounded[element + 13]);
    word = ts_word_set_lane(word, 14, rounded[element + 14]);
    word = ts_word_set_lane(word, 15, rounded[element + 15]);
    out_words[w] = word;
  }
}

/* ------------------------------------------------------------------ */
/* Guards and layout                                                  */
/* ------------------------------------------------------------------ */

int token_selector_check_group_limit(int group_count) {
  return (group_count >= 1 && group_count <= TS_MAX_GROUP_INDEX) ? 1 : 0;
}

int token_selector_check_grid_dims(int grid_h, int grid_w) {
  return (grid_h >= 1 && grid_h <= TS_MAX_GRID_DIM && grid_w >= 1 &&
          grid_w <= TS_MAX_GRID_DIM)
             ? 1
             : 0;
}

int token_selector_check_budget(int budget) {
  return (budget > 0 && budget <= TS_MAX_BUDGET && (budget % 16) == 0) ? 1 : 0;
}

static int ts_align_up(int value) {
  return ((value + TS_LAYOUT_ALIGNMENT - 1) / TS_LAYOUT_ALIGNMENT) *
         TS_LAYOUT_ALIGNMENT;
}

int token_selector_layout(int budget, struct TokenSelectorLayout* layout) {
  if (layout == 0 || !token_selector_check_budget(budget)) {
    return TS_STATUS_ERR_BUDGET;
  }
  int sizes[TS_NUM_SEGMENTS];
  sizes[0] = budget * TS_GROUP_ELEMENTS;
  sizes[1] = budget * 2;
  sizes[2] = budget;
  sizes[3] = budget;
  sizes[4] = budget;
  int offset = 0;
  for (int i = 0; i < TS_NUM_SEGMENTS; i++) {
    offset = ts_align_up(offset);
    layout->offsets[i] = offset;
    layout->sizes[i] = sizes[i];
    offset += sizes[i];
  }
  layout->total_bytes = ts_align_up(offset);
  return TS_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Core selection                                                     */
/* ------------------------------------------------------------------ */

#ifndef __SYNTHESIS__
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
    struct TokenSelectorResult* result) {
#pragma HLS INLINE
  if (result != 0) {
    result->status = TS_STATUS_OK;
    result->output_count = 0;
    result->real_count = 0;
    result->context_count = 0;
  } else {
    return TS_STATUS_ERR_NULL;
  }
  if (tokens_q == 0 || score_q == 0 || out_tokens == 0 ||
      out_group_index == 0 || out_source_type == 0 || out_region_id == 0 ||
      out_valid == 0) {
    result->status = TS_STATUS_ERR_NULL;
    return TS_STATUS_ERR_NULL;
  }
  if (scale_out != 0) {
    *scale_out = scale_in; /* register pass-through only */
  }

  if (group_count < 1 || group_count > TS_MAX_GROUP_INDEX ||
      group_count > TS_MAX_GROUPS) {
    result->status = TS_STATUS_ERR_GROUP_LIMIT;
    return TS_STATUS_ERR_GROUP_LIMIT;
  }
  if (!token_selector_check_grid_dims(grid_h, grid_w) || grid_h > group_count ||
      grid_w > group_count || grid_h * grid_w != group_count) {
    result->status = TS_STATUS_ERR_GRID;
    return TS_STATUS_ERR_GRID;
  }
  if (!token_selector_check_budget(budget)) {
    result->status = TS_STATUS_ERR_BUDGET;
    return TS_STATUS_ERR_BUDGET;
  }
  if (local_cap <= 0 || local_cap > TS_MAX_LOCAL_CANDIDATE_CAP) {
    result->status = TS_STATUS_ERR_LOCAL_CAP;
    return TS_STATUS_ERR_LOCAL_CAP;
  }

  int region_of[TS_MAX_GROUPS];
  int member_count[TS_NUM_REGIONS];
  int anchor[TS_NUM_REGIONS];
  int anchor_score[TS_NUM_REGIONS];
  int row_bounds[TS_REGION_GRID][2];
  int col_bounds[TS_REGION_GRID][2];
  int non_empty[TS_NUM_REGIONS];
  int context_region[TS_NUM_REGIONS];
  int is_context_region[TS_NUM_REGIONS];
  int local_top[TS_NUM_REGIONS][TS_LOCAL_CANDIDATE_CAP];
  int local_count[TS_NUM_REGIONS];
  int local_limit[TS_NUM_REGIONS];
  int pool[TS_MAX_CANDIDATE_POOL];
  int selected[TS_MAX_OUTPUT];
  int is_selected[TS_MAX_GROUPS];
  int entry_rep[TS_MAX_OUTPUT];
  int entry_source[TS_MAX_OUTPUT];
  int entry_region[TS_MAX_OUTPUT];
  int remaining[TS_MAX_REGION_MEMBERS];

#pragma HLS ARRAY_PARTITION variable=member_count complete
#pragma HLS ARRAY_PARTITION variable=anchor complete
#pragma HLS ARRAY_PARTITION variable=anchor_score complete
#pragma HLS ARRAY_PARTITION variable=non_empty complete
#pragma HLS ARRAY_PARTITION variable=context_region complete
#pragma HLS ARRAY_PARTITION variable=is_context_region complete
#pragma HLS ARRAY_PARTITION variable=local_top complete dim=0
#pragma HLS ARRAY_PARTITION variable=local_count complete
#pragma HLS ARRAY_PARTITION variable=local_limit complete
#pragma HLS ARRAY_PARTITION variable=pool complete
#pragma HLS ARRAY_PARTITION variable=entry_rep complete
#pragma HLS ARRAY_PARTITION variable=entry_source complete
#pragma HLS ARRAY_PARTITION variable=entry_region complete

  ts_region_bounds(grid_h, row_bounds);
  ts_region_bounds(grid_w, col_bounds);
  for (int r = 0; r < TS_NUM_REGIONS; r++) {
    member_count[r] = 0;
    anchor[r] = -1;
    anchor_score[r] = -1;
  }

  /*
   * Pass 1: classify every group into its 3x3 region and find the anchor with
   * a cached running best score (no repeated score_q[best] load).  Band lengths
   * are <= ceil(128/3) = 43.
   */
  for (int rr = 0; rr < TS_REGION_GRID; rr++) {
    for (int cc = 0; cc < TS_REGION_GRID; cc++) {
      int rid = rr * TS_REGION_GRID + cc;
      for (int row = row_bounds[rr][0]; row < row_bounds[rr][1]; row++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
        for (int col = col_bounds[cc][0]; col < col_bounds[cc][1]; col++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
          int group = row * grid_w + col;
          region_of[group] = rid;
          member_count[rid] += 1;
          int score = (int)score_q[group];
          if (anchor[rid] < 0 || score > anchor_score[rid] ||
              (score == anchor_score[rid] && group < anchor[rid])) {
            anchor[rid] = group;
            anchor_score[rid] = score;
          }
        }
      }
    }
  }

  /* Identity bypass: element-wise copy, C = 0.  group_count <= budget <= 96. */
  if (group_count <= budget) {
    for (int group = 0; group < TS_MAX_BUDGET; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=96
      if (group >= group_count) break;
      out_group_index[group] = (uint16_t)group;
      out_source_type[group] = (uint8_t)TS_SOURCE_REAL;
      out_region_id[group] = (uint8_t)region_of[group];
      out_valid[group] = 1;
      for (int byte = 0; byte < TS_GROUP_ELEMENTS; byte++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        out_tokens[(size_t)group * TS_GROUP_ELEMENTS + byte] =
            tokens_q[(size_t)group * TS_GROUP_ELEMENTS + byte];
      }
    }
    result->output_count = group_count;
    result->real_count = group_count;
    result->context_count = 0;
    result->status = TS_STATUS_OK;
    return TS_STATUS_OK;
  }

  int non_empty_count = 0;
  int context_count = 0;
  for (int r = 0; r < TS_NUM_REGIONS; r++) {
    is_context_region[r] = 0;
    if (member_count[r] > 0) {
      non_empty[non_empty_count] = r;
      non_empty_count += 1;
      if (member_count[r] >= 2) {
        is_context_region[r] = 1;
        context_region[context_count] = r;
        context_count += 1;
      }
    }
  }
  if (budget - context_count < non_empty_count) {
    result->status = TS_STATUS_ERR_INSUFFICIENT;
    return TS_STATUS_ERR_INSUFFICIENT;
  }
  int real_budget = budget - context_count;

  /*
   * Pass 2: for every region keep exactly the top `local_cap` non-anchor
   * candidates with a stable bounded insertion.  The resulting global pool is
   * at most 9 * 16 = 144 entries.
   */
  for (int r = 0; r < TS_NUM_REGIONS; r++) {
    local_count[r] = 0;
    int allowed = member_count[r] - 1;
    if (is_context_region[r]) {
      allowed -= 1;
    }
    if (allowed < 0) {
      allowed = 0;
    }
    int limit = local_cap;
    if (limit > allowed) {
      limit = allowed;
    }
    local_limit[r] = limit;
  }

  for (int group = 0; group < TS_MAX_GROUPS; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    if (group >= group_count) break;
    int rid = region_of[group];
    if (group == anchor[rid]) continue;
    int limit = local_limit[rid];
    if (limit <= 0) continue;
    int count = local_count[rid];
    if (count < limit) {
      int pos = count;
      while (pos > 0) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=15
        int previous = local_top[rid][pos - 1];
        if (ts_rank_before(previous, group, score_q)) break;
        local_top[rid][pos] = previous;
        pos -= 1;
      }
      local_top[rid][pos] = group;
      local_count[rid] = count + 1;
    } else {
      int worst = local_top[rid][limit - 1];
      if (ts_rank_before(group, worst, score_q)) {
        int pos = limit - 1;
        while (pos > 0) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=15
          int previous = local_top[rid][pos - 1];
          if (ts_rank_before(previous, group, score_q)) break;
          local_top[rid][pos] = previous;
          pos -= 1;
        }
        local_top[rid][pos] = group;
      }
    }
  }

  int pool_count = 0;
  for (int index = 0; index < non_empty_count; index++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = non_empty[index];
    int count = local_count[rid];
    for (int k = 0; k < count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=16
      pool[pool_count] = local_top[rid][k];
      pool_count += 1;
    }
  }

  /* Stable bounded insertion sort of the <= 144 pool (never O(G^2)). */
  for (int i = 1; i < pool_count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=143
    int key = pool[i];
    int j = i - 1;
    while (j >= 0) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=143
      if (ts_rank_before(pool[j], key, score_q)) break;
      pool[j + 1] = pool[j];
      j -= 1;
    }
    pool[j + 1] = key;
  }

  int details_needed = real_budget - non_empty_count;
  if (pool_count < details_needed) {
    result->status = TS_STATUS_ERR_CANDIDATE_POOL;
    return TS_STATUS_ERR_CANDIDATE_POOL;
  }

  int selected_count = 0;
  for (int index = 0; index < non_empty_count; index++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    selected[selected_count] = anchor[non_empty[index]];
    selected_count += 1;
  }
  for (int k = 0; k < details_needed; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    selected[selected_count] = pool[k];
    selected_count += 1;
  }
  for (int i = 1; i < selected_count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=95
    int key = selected[i];
    int j = i - 1;
    while (j >= 0 && selected[j] > key) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=95
      selected[j + 1] = selected[j];
      j -= 1;
    }
    selected[j + 1] = key;
  }
  for (int group = 0; group < TS_MAX_GROUPS; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    if (group >= group_count) break;
    is_selected[group] = 0;
  }
  for (int k = 0; k < selected_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    is_selected[selected[k]] = 1;
  }

  /* Build entry metadata (real groups first, then context groups). */
  int entry_count = 0;
  for (int k = 0; k < selected_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    int group = selected[k];
    entry_rep[entry_count] = group;
    entry_source[entry_count] = TS_SOURCE_REAL;
    entry_region[entry_count] = region_of[group];
    entry_count += 1;
  }
  for (int c = 0; c < context_count; c++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = context_region[c];
    int remaining_count = 0;
    ts_collect_remaining(region_of, is_selected, group_count, rid, remaining,
                         &remaining_count);
    if (remaining_count <= 0) {
      result->status = TS_STATUS_ERR_INTERNAL;
      return TS_STATUS_ERR_INTERNAL;
    }
    int representative =
        ts_context_representative(score_q, remaining, remaining_count, grid_w);
    entry_rep[entry_count] = representative;
    entry_source[entry_count] = TS_SOURCE_CONTEXT;
    entry_region[entry_count] = rid;
    entry_count += 1;
  }
  if (entry_count != budget) {
    result->status = TS_STATUS_ERR_INTERNAL;
    return TS_STATUS_ERR_INTERNAL;
  }

  /* Stable insertion sort by (representative, source_type, region_id). */
  for (int i = 1; i < entry_count; i++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=95
    int r = entry_rep[i];
    int s = entry_source[i];
    int g = entry_region[i];
    int j = i - 1;
    while (j >= 0 && ts_entry_less(r, s, g, entry_rep[j], entry_source[j],
                                   entry_region[j])) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=95
      entry_rep[j + 1] = entry_rep[j];
      entry_source[j + 1] = entry_source[j];
      entry_region[j + 1] = entry_region[j];
      j -= 1;
    }
    entry_rep[j + 1] = r;
    entry_source[j + 1] = s;
    entry_region[j + 1] = g;
  }

  /*
   * Materialise the sorted entries.  Metadata and real token copies run in a
   * loop bounded by the budget; context pooling is a *separate* loop bounded by
   * `context_count <= 9`, so the expensive pooling is not multiplied by the
   * full budget bound.
   */
  for (int e = 0; e < entry_count; e++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    if (entry_source[e] == TS_SOURCE_REAL) {
      memcpy(out_tokens + (size_t)e * TS_GROUP_ELEMENTS,
             tokens_q + (size_t)entry_rep[e] * TS_GROUP_ELEMENTS,
             TS_GROUP_ELEMENTS);
    }
    out_group_index[e] = (uint16_t)entry_rep[e];
    out_source_type[e] = (uint8_t)entry_source[e];
    out_region_id[e] = (uint8_t)entry_region[e];
    out_valid[e] = 1;
  }
  for (int c = 0; c < context_count; c++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = context_region[c];
    int slot = -1;
    for (int e = 0; e < entry_count; e++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
      if (entry_source[e] == TS_SOURCE_CONTEXT && entry_region[e] == rid) {
        slot = e;
        break;
      }
    }
    int remaining_count = 0;
    ts_collect_remaining(region_of, is_selected, group_count, rid, remaining,
                         &remaining_count);
    ts_pool_context(tokens_q, score_q, remaining, remaining_count,
                    out_tokens + (size_t)slot * TS_GROUP_ELEMENTS);
  }

  result->output_count = entry_count;
  result->context_count = context_count;
  result->real_count = entry_count - context_count;
  result->status = TS_STATUS_OK;
  return TS_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* HLS top level                                                      */
/* ------------------------------------------------------------------ */

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
    int* context_count) {
  /* Separate bundles: tokens vs each metadata segment, so the materialise
   * loop is no longer limited by a single gmem_out port. */
#pragma HLS INTERFACE m_axi port = tokens_q offset = slave bundle = gmem_in_tokens depth = 16777216
#pragma HLS INTERFACE m_axi port = score_q offset = slave bundle = gmem_in_score depth = 4096
#pragma HLS INTERFACE m_axi port = out_tokens offset = slave bundle = gmem_out_tokens depth = 16777216
#pragma HLS INTERFACE m_axi port = out_group_index offset = slave bundle = gmem_out_group depth = 4096
#pragma HLS INTERFACE m_axi port = out_source_type offset = slave bundle = gmem_out_source depth = 4096
#pragma HLS INTERFACE m_axi port = out_region_id offset = slave bundle = gmem_out_region depth = 4096
#pragma HLS INTERFACE m_axi port = out_valid offset = slave bundle = gmem_out_valid depth = 4096
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

  struct TokenSelectorResult result;
  int status = token_selector_core(
      tokens_q, score_q, group_count, grid_h, grid_w, budget, local_cap,
      scale_in, scale_out, out_tokens, out_group_index, out_source_type,
      out_region_id, out_valid, &result);
  if (output_count != 0) {
    *output_count = result.output_count;
  }
  if (real_count != 0) {
    *real_count = result.real_count;
  }
  if (context_count != 0) {
    *context_count = result.context_count;
  }
  return status;
}
#endif /* !__SYNTHESIS__ : end of the int8 top level */

/* ------------------------------------------------------------------ */
/* 128-bit packed core (selection maths identical to the int8 core)    */
/* ------------------------------------------------------------------ */

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
    struct TokenSelectorResult* result) {
#pragma HLS INLINE
  if (result != 0) {
    result->status = TS_STATUS_OK;
    result->output_count = 0;
    result->real_count = 0;
    result->context_count = 0;
  } else {
    return TS_STATUS_ERR_NULL;
  }
  if (tokens_q == 0 || score_q == 0 || out_tokens == 0 ||
      out_group_index == 0 || out_source_type == 0 || out_region_id == 0 ||
      out_valid == 0) {
    result->status = TS_STATUS_ERR_NULL;
    return TS_STATUS_ERR_NULL;
  }
  if (scale_out != 0) {
    *scale_out = scale_in; /* register pass-through only */
  }

  if (group_count < 1 || group_count > TS_MAX_GROUP_INDEX ||
      group_count > TS_MAX_GROUPS) {
    result->status = TS_STATUS_ERR_GROUP_LIMIT;
    return TS_STATUS_ERR_GROUP_LIMIT;
  }
  if (!token_selector_check_grid_dims(grid_h, grid_w) || grid_h > group_count ||
      grid_w > group_count || grid_h * grid_w != group_count) {
    result->status = TS_STATUS_ERR_GRID;
    return TS_STATUS_ERR_GRID;
  }
  if (!token_selector_check_budget(budget)) {
    result->status = TS_STATUS_ERR_BUDGET;
    return TS_STATUS_ERR_BUDGET;
  }
  if (local_cap <= 0 || local_cap > TS_MAX_LOCAL_CANDIDATE_CAP) {
    result->status = TS_STATUS_ERR_LOCAL_CAP;
    return TS_STATUS_ERR_LOCAL_CAP;
  }

  int region_of[TS_MAX_GROUPS];
  int member_count[TS_NUM_REGIONS];
  int anchor[TS_NUM_REGIONS];
  int anchor_score[TS_NUM_REGIONS];
  int row_bounds[TS_REGION_GRID][2];
  int col_bounds[TS_REGION_GRID][2];
  int non_empty[TS_NUM_REGIONS];
  int context_region[TS_NUM_REGIONS];
  int is_context_region[TS_NUM_REGIONS];
  /*
   * v4: no fully-partitioned local_top[9][16] bank and no insertion workspace.
   * Per-region candidates are produced by a sequential multi-pass max-select
   * straight into the bounded `pool` (single rank comparator).
   */
  int pool[TS_MAX_CANDIDATE_POOL];
  int pool_score[TS_MAX_CANDIDATE_POOL];
  int pool_picked[TS_MAX_CANDIDATE_POOL];
  int picked[TS_MAX_GROUPS];
  int selected[TS_MAX_OUTPUT];
  int selected_picked[TS_MAX_OUTPUT];
  int is_selected[TS_MAX_GROUPS];
  int entry_rep[TS_MAX_OUTPUT];
  int entry_source[TS_MAX_OUTPUT];
  int entry_region[TS_MAX_OUTPUT];
  int entry_picked[TS_MAX_OUTPUT];
  int entry_rep2[TS_MAX_OUTPUT];
  int entry_source2[TS_MAX_OUTPUT];
  int entry_region2[TS_MAX_OUTPUT];
  int remaining[TS_MAX_REGION_MEMBERS];

#pragma HLS ARRAY_PARTITION variable=member_count complete
#pragma HLS ARRAY_PARTITION variable=anchor complete
#pragma HLS ARRAY_PARTITION variable=anchor_score complete
#pragma HLS ARRAY_PARTITION variable=non_empty complete
#pragma HLS ARRAY_PARTITION variable=context_region complete
#pragma HLS ARRAY_PARTITION variable=is_context_region complete

  ts_region_bounds(grid_h, row_bounds);
  ts_region_bounds(grid_w, col_bounds);
  for (int r = 0; r < TS_NUM_REGIONS; r++) {
    member_count[r] = 0;
    anchor[r] = -1;
    anchor_score[r] = -1;
  }

  /* Pass 1: region classification + anchor (cached running best score). */
  for (int rr = 0; rr < TS_REGION_GRID; rr++) {
    for (int cc = 0; cc < TS_REGION_GRID; cc++) {
      int rid = rr * TS_REGION_GRID + cc;
      for (int row = row_bounds[rr][0]; row < row_bounds[rr][1]; row++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
        for (int col = col_bounds[cc][0]; col < col_bounds[cc][1]; col++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
          int group = row * grid_w + col;
          region_of[group] = rid;
          member_count[rid] += 1;
          int score = (int)score_q[group];
          if (anchor[rid] < 0 || score > anchor_score[rid] ||
              (score == anchor_score[rid] && group < anchor[rid])) {
            anchor[rid] = group;
            anchor_score[rid] = score;
          }
        }
      }
    }
  }

  /* Identity bypass: 256-word element-wise copy, C = 0. */
  if (group_count <= budget) {
    for (int group = 0; group < TS_MAX_BUDGET; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=96
      if (group >= group_count) break;
      out_group_index[group] = (uint16_t)group;
      out_source_type[group] = (uint8_t)TS_SOURCE_REAL;
      out_region_id[group] = (uint8_t)region_of[group];
      out_valid[group] = 1;
      for (int w = 0; w < TS_GROUP_WORDS; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        out_tokens[(size_t)group * TS_GROUP_WORDS + w] =
            tokens_q[(size_t)group * TS_GROUP_WORDS + w];
      }
    }
    result->output_count = group_count;
    result->real_count = group_count;
    result->context_count = 0;
    result->status = TS_STATUS_OK;
    return TS_STATUS_OK;
  }

  int non_empty_count = 0;
  int context_count = 0;
  for (int r = 0; r < TS_NUM_REGIONS; r++) {
    is_context_region[r] = 0;
    if (member_count[r] > 0) {
      non_empty[non_empty_count] = r;
      non_empty_count += 1;
      if (member_count[r] >= 2) {
        is_context_region[r] = 1;
        context_region[context_count] = r;
        context_count += 1;
      }
    }
  }
  if (budget - context_count < non_empty_count) {
    result->status = TS_STATUS_ERR_INSUFFICIENT;
    return TS_STATUS_ERR_INSUFFICIENT;
  }
  int real_budget = budget - context_count;

  /*
   * Pass 2: sequential deterministic multi-pass max-select.  Every region keeps
   * exactly its top `local_cap` non-anchor candidates by repeatedly scanning the
   * region and taking the best-ranked not-yet-picked group (higher score first,
   * ties to the lower index).  This yields the *same ordered candidate set* as
   * the frozen v3 stable bounded insertion, but it is a single-comparator
   * reduction: no fully-partitioned local_top[9][16] register bank and no 15-way
   * insertion mux.  Candidates are appended to the global pool in the same
   * non_empty-region order as v3 (each region best-first).
   */
  for (int group = 0; group < TS_MAX_GROUPS; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    if (group >= group_count) break;
    picked[group] = 0;
  }

  int pool_count = 0;
  for (int index = 0; index < non_empty_count; index++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = non_empty[index];
    int rr = rid / TS_REGION_GRID;
    int cc = rid % TS_REGION_GRID;
    int allowed = member_count[rid] - 1;
    if (is_context_region[rid]) {
      allowed -= 1;
    }
    if (allowed < 0) {
      allowed = 0;
    }
    int limit = local_cap;
    if (limit > allowed) {
      limit = allowed;
    }
    for (int rank = 0; rank < limit; rank++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=15
      int best = -1;
      int best_score = -1;
      for (int row = row_bounds[rr][0]; row < row_bounds[rr][1]; row++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
        for (int col = col_bounds[cc][0]; col < col_bounds[cc][1]; col++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=43
          int group = row * grid_w + col;
          if (group == anchor[rid] || picked[group]) continue;
          int score = (int)score_q[group];
          if (best < 0 || score > best_score ||
              (score == best_score && group < best)) {
            best = group;
            best_score = score;
          }
        }
      }
      if (best < 0) break;
      picked[best] = 1;
      pool[pool_count] = best;
      pool_score[pool_count] = best_score;
      pool_count += 1;
    }
  }

  int details_needed = real_budget - non_empty_count;
  if (pool_count < details_needed) {
    result->status = TS_STATUS_ERR_CANDIDATE_POOL;
    return TS_STATUS_ERR_CANDIDATE_POOL;
  }

  /*
   * Global details and the ascending index order use sequential deterministic
   * multi-pass select with a single comparator per pass.  Every array index is
   * either the loop counter or a scalar register, so there is no 16/144-way
   * combinational insertion mux and no data-dependent read port.
   */
  for (int k = 0; k < pool_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=143
    pool_picked[k] = 0;
  }
  int selected_count = 0;
  for (int index = 0; index < non_empty_count; index++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    selected[selected_count] = anchor[non_empty[index]];
    selected_count += 1;
  }
  for (int k = 0; k < details_needed; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    int best = -1;
    int best_score = -1;
    int best_group = -1;
    for (int j = 0; j < pool_count; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=143
      if (pool_picked[j]) continue;
      int score_j = pool_score[j];
      int group_j = pool[j];
      if (best < 0 || score_j > best_score ||
          (score_j == best_score && group_j < best_group)) {
        best = j;
        best_score = score_j;
        best_group = group_j;
      }
    }
    if (best < 0) break;
    pool_picked[best] = 1;
    selected[selected_count] = best_group;
    selected_count += 1;
  }

  for (int k = 0; k < selected_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    selected_picked[k] = 0;
  }
  int entry_count = 0;
  for (int out = 0; out < selected_count; out++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    int best = -1;
    int best_group = 0;
    for (int j = 0; j < selected_count; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
      if (selected_picked[j]) continue;
      int group_j = selected[j];
      if (best < 0 || group_j < best_group) {
        best = j;
        best_group = group_j;
      }
    }
    selected_picked[best] = 1;
    entry_rep[entry_count] = best_group;
    entry_source[entry_count] = TS_SOURCE_REAL;
    entry_region[entry_count] = region_of[best_group];
    entry_count += 1;
  }

  for (int group = 0; group < TS_MAX_GROUPS; group++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
    if (group >= group_count) break;
    is_selected[group] = 0;
  }
  for (int k = 0; k < entry_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    is_selected[entry_rep[k]] = 1;
  }

  /* Build context entries (appended after the ascending real entries). */
  for (int c = 0; c < context_count; c++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = context_region[c];
    int remaining_count = 0;
    ts_collect_remaining(region_of, is_selected, group_count, rid, remaining,
                         &remaining_count);
    if (remaining_count <= 0) {
      result->status = TS_STATUS_ERR_INTERNAL;
      return TS_STATUS_ERR_INTERNAL;
    }
    int representative =
        ts_context_representative(score_q, remaining, remaining_count, grid_w);
    entry_rep[entry_count] = representative;
    entry_source[entry_count] = TS_SOURCE_CONTEXT;
    entry_region[entry_count] = rid;
    entry_count += 1;
  }
  if (entry_count != budget) {
    result->status = TS_STATUS_ERR_INTERNAL;
    return TS_STATUS_ERR_INTERNAL;
  }

  /* Ascending (representative, source_type, region_id) into a second buffer. */
  for (int k = 0; k < entry_count; k++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    entry_picked[k] = 0;
  }
  for (int out = 0; out < entry_count; out++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    int best = -1;
    int best_rep = 0;
    int best_source = 0;
    int best_region = 0;
    for (int j = 0; j < entry_count; j++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
      if (entry_picked[j]) continue;
      int rep_j = entry_rep[j];
      int source_j = entry_source[j];
      int region_j = entry_region[j];
      if (best < 0 ||
          ts_entry_less(rep_j, source_j, region_j, best_rep, best_source,
                        best_region)) {
        best = j;
        best_rep = rep_j;
        best_source = source_j;
        best_region = region_j;
      }
    }
    entry_picked[best] = 1;
    entry_rep2[out] = best_rep;
    entry_source2[out] = best_source;
    entry_region2[out] = best_region;
  }
  for (int e = 0; e < entry_count; e++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    entry_rep[e] = entry_rep2[e];
    entry_source[e] = entry_source2[e];
    entry_region[e] = entry_region2[e];
  }

  /*
   * Materialise metadata + real token copies (256 words each), then pool the
   * context groups separately (bounded by context_count <= 9).
   */
  for (int e = 0; e < entry_count; e++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
    if (entry_source[e] == TS_SOURCE_REAL) {
      int src = entry_rep[e] * TS_GROUP_WORDS;
      int dst = e * TS_GROUP_WORDS;
      for (int w = 0; w < TS_GROUP_WORDS; w++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=256
        out_tokens[dst + w] = tokens_q[src + w];
      }
    }
    out_group_index[e] = (uint16_t)entry_rep[e];
    out_source_type[e] = (uint8_t)entry_source[e];
    out_region_id[e] = (uint8_t)entry_region[e];
    out_valid[e] = 1;
  }
  for (int c = 0; c < context_count; c++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
    int rid = context_region[c];
    int slot = -1;
    for (int e = 0; e < entry_count; e++) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=96
      if (entry_source[e] == TS_SOURCE_CONTEXT && entry_region[e] == rid) {
        slot = e;
        break;
      }
    }
    int remaining_count = 0;
    ts_collect_remaining(region_of, is_selected, group_count, rid, remaining,
                         &remaining_count);
    ts_pool_context_packed(tokens_q, score_q, remaining, remaining_count,
                           out_tokens + (size_t)slot * TS_GROUP_WORDS);
  }

  result->output_count = entry_count;
  result->context_count = context_count;
  result->real_count = entry_count - context_count;
  result->status = TS_STATUS_OK;
  return TS_STATUS_OK;
}

/* ------------------------------------------------------------------ */
/* Packed HLS top level                                               */
/* ------------------------------------------------------------------ */

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
    int* context_count) {
  /* Token ports are 128-bit (16 int8 lanes per beat); metadata bundles are
   * unchanged from the int8 top. */
#pragma HLS INTERFACE m_axi port = tokens_q offset = slave bundle = gmem_in_tokens depth = 1048576
#pragma HLS INTERFACE m_axi port = score_q offset = slave bundle = gmem_in_score depth = 4096
#pragma HLS INTERFACE m_axi port = out_tokens offset = slave bundle = gmem_out_tokens depth = 1048576
#pragma HLS INTERFACE m_axi port = out_group_index offset = slave bundle = gmem_out_group depth = 4096
#pragma HLS INTERFACE m_axi port = out_source_type offset = slave bundle = gmem_out_source depth = 4096
#pragma HLS INTERFACE m_axi port = out_region_id offset = slave bundle = gmem_out_region depth = 4096
#pragma HLS INTERFACE m_axi port = out_valid offset = slave bundle = gmem_out_valid depth = 4096
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

  struct TokenSelectorResult result;
  int status = token_selector_core_packed(
      tokens_q, score_q, group_count, grid_h, grid_w, budget, local_cap,
      scale_in, scale_out, out_tokens, out_group_index, out_source_type,
      out_region_id, out_valid, &result);
  if (output_count != 0) {
    *output_count = result.output_count;
  }
  if (real_count != 0) {
    *real_count = result.real_count;
  }
  if (context_count != 0) {
    *context_count = result.context_count;
  }
  return status;
}
