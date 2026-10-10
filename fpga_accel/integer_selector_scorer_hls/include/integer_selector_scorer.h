/*
 * integer_selector_scorer.h - frozen bit-exact integer Selector scorer.
 *
 * This is a standalone HLS kernel that reproduces, byte for byte, the
 * ``INTEGER_SCORER_Q16`` chain audited in
 * ``evaluation/selector_integer_scorer_audit.py``:
 *
 *   1. visual projection ``p[G,128] = group_mean_q @ Wv_q^T`` with exact
 *      INT32 accumulation;
 *   2. signed ``dot = sum_d p[d]*query_q[d]`` and nonnegative
 *      ``norm2 = sum_d p[d]^2`` both accumulated in INT64;
 *   3. ``semantic_q16 = (norm2==0) ? 0 :
 *          round_half_away((dot << 16) / floor_sqrt(norm2))`` using an exact
 *      signed-rational round-half-away-from-zero division;
 *   4. ``l2_int = floor_sqrt(sum_1024 group_mean_q^2)``;
 *   5. independent exact min-max normalisation of ``semantic_q16`` and
 *      ``l2_int`` to ``[0, 32767]`` (all-equal maps to zero);
 *   6. ``score_q = clamp(round_half_away((semantic_norm_q15 +
 *      l2_norm_q15)/2), 0, 32767)``.
 *
 * There is **no float or double anywhere in the datapath** and no signed shift
 * of a negative value (``dot`` is multiplied by ``2**16``, never left-shifted).
 *
 * Datapath widths (frozen analytical bounds, see quantization_contract.md):
 *   projection   INT32    |p|        <= 1024*128*127     = 16,646,144   < 2**31-1
 *   dot          INT64    |dot|      <= 128*16,646,144*128= 2.73e11      < 2**63-1
 *   norm2        INT64    norm2      <= 128*16,646,144^2  = 3.55e16      < 2**63-1
 *   numerator    INT64    2|dot|*2^16 <= 3.58e16                         < 2**63-1
 *   semantic_q16 INT32    |value|    <= 2*128*12*2^16     = 201,326,592   < 2**31-1
 *   l2_int       UINT16   value      <= floor(sqrt(1024*128^2)) = 4096
 *   score_q      UINT16   value      in [0, 32767]
 *
 * The whole translation unit is plain C++ so it also compiles and runs under
 * g++ (the pragmas are ignored by a host compiler).  Under synthesis the
 * 128-bit AXI word becomes ``ap_uint<128>``; everywhere else it is the host
 * ``unsigned __int128`` with the identical little-endian lane layout.
 */
#ifndef INTEGER_SELECTOR_SCORER_H
#define INTEGER_SELECTOR_SCORER_H

#include <stdint.h>

#define IS_EMBED_DIM 1024
#define IS_QUERY_DIM 128
#define IS_WV_ROWS 128
#define IS_MAX_GROUPS 4096
#define IS_MIN_GROUPS 1

/* 128-bit packed AXI word: 16 consecutive little-endian int8 lanes.  Lane L
 * occupies bits [8L+7:8L]; the byte at address ``row*1024 + word*16 + lane`` is
 * exactly the original row-major int8 byte. */
#define IS_PACKED_LANES 16
#define IS_WORD_BYTES 16
#define IS_GM_WORDS_PER_GROUP (IS_EMBED_DIM / IS_PACKED_LANES) /* 64  */
#define IS_MAX_GM_WORDS (IS_MAX_GROUPS * IS_GM_WORDS_PER_GROUP) /* 262144 */
#define IS_WV_WORDS (IS_WV_ROWS * IS_GM_WORDS_PER_GROUP) /* 8192 */
#define IS_QUERY_WORDS (IS_QUERY_DIM / IS_PACKED_LANES) /* 8 */

#define IS_SEMANTIC_FRACTIONAL_BITS 16
#define IS_SCORE_FRACTIONAL_BITS 15
#define IS_SCORE_Q_MAX 32767
/* int64 input to floor_sqrt: |dot|*2^16 <= 3.58e16 < 2**63-1. */
#define IS_NORM2_MAX 35468046088798208LL
#define IS_SHIFTED_NUMERATOR_MAX 17873661021126656LL

#define IS_MAX_GROUP_INDEX_WORDS (IS_MAX_GROUPS * IS_GM_WORDS_PER_GROUP)

#ifdef __SYNTHESIS__
#include <ap_int.h>
typedef ap_uint<128> is_word_t;
#else
typedef unsigned __int128 is_word_t;
#endif

/* Status codes (must match evaluation/selector_integer_scorer_audit.py). */
enum is_status {
    IS_STATUS_OK = 0,
    IS_STATUS_ERR_NULL = 1,
    IS_STATUS_ERR_GROUP_COUNT = 2
};

/* Lane read: sign-extends lane ``lane`` (little-endian, lane = byte offset). */
static inline int is_word_lane(const is_word_t word, int lane) {
#ifdef __SYNTHESIS__
    return (int)(signed char)(word.range(lane * 8 + 7, lane * 8).to_uint());
#else
    unsigned int raw =
        (unsigned int)((word >> (lane * 8)) & (is_word_t)0xFF);
    return raw < 128u ? (int)raw : (int)raw - 256;
#endif
}

/* Byte-stream packing helpers (host + HLS).  Byte B maps to word B/16, lane
 * B%16; a short tail is zero padded.  Is is_word_set_lane needed?  The kernel
 * only reads packed words, the testbench builds them with this pair. */
static inline is_word_t is_word_set_lane(is_word_t word, int lane, int value) {
    is_word_t mask = ((is_word_t)0xFF) << (lane * 8);
    return (is_word_t)((word & ~mask) |
                       (((is_word_t)(uint8_t)(int8_t)value) << (lane * 8)));
}

static inline void is_pack_int8_to_words(const int8_t* bytes, int byte_count,
                                         is_word_t* words, int word_count) {
    for (int w = 0; w < word_count; w++) {
        is_word_t word = (is_word_t)0;
        for (int lane = 0; lane < IS_PACKED_LANES; lane++) {
            int index = w * IS_PACKED_LANES + lane;
            int8_t value = (index < byte_count) ? bytes[index] : (int8_t)0;
            word = is_word_set_lane(word, lane, (int)value);
        }
        words[w] = word;
    }
}

static inline void is_unpack_words_to_int8(const is_word_t* words,
                                           int word_count, int8_t* bytes,
                                           int byte_count) {
    for (int index = 0; index < byte_count; index++) {
        int w = index / IS_PACKED_LANES;
        int lane = index % IS_PACKED_LANES;
        bytes[index] =
            (w < word_count) ? (int8_t)is_word_lane(words[w], lane) : (int8_t)0;
    }
}

/* Guard used by the testbench and by the HLS top: 1 <= group_count <= 4096. */
static inline int is_group_count_legal(int group_count) {
    return group_count >= IS_MIN_GROUPS && group_count <= IS_MAX_GROUPS;
}

#ifdef __cplusplus
extern "C" {
#endif

/*
 * HLS top level.  ``group_mean_q`` is ``group_count*64`` 128-bit words,
 * ``wv_q`` is 8192 words, ``query_q`` is 8 words and ``score_q`` is
 * ``group_count`` uint16 values.  Returns an ``is_status`` code.  The data
 * ports are independent m_axi bundles; every scalar/return is s_axilite.
 */
int integer_selector_scorer(const is_word_t* group_mean_q,
                            const is_word_t* wv_q,
                            const is_word_t* query_q,
                            uint16_t* score_q,
                            int group_count);

#ifndef __SYNTHESIS__
/*
 * Testbench-only twin of the top: identical arithmetic, and additionally
 * copies the per-group intermediates so a host test can check them against the
 * frozen Python golden.  Never synthesized, never exported, never a port.
 */
int integer_selector_scorer_debug(const is_word_t* group_mean_q,
                                  const is_word_t* wv_q,
                                  const is_word_t* query_q,
                                  uint16_t* score_q,
                                  int32_t* semantic_q16_out,
                                  uint16_t* l2_int_out,
                                  uint16_t* semantic_norm_q15_out,
                                  uint16_t* l2_norm_q15_out,
                                  int group_count);

/* Testbench-only accessors for the exact integer primitives.  They let the
 * host test compare the C++ floor-sqrt / round-half-away / min-max directly
 * with the frozen Python golden; they are never synthesized or exported. */
uint64_t is_test_floor_sqrt_u64(uint64_t value);
int64_t is_test_round_half_away_div(int64_t numerator, int64_t denominator);
uint16_t is_test_minmax_q15(int64_t value, int64_t low, int64_t high);
#endif

#ifdef __cplusplus
}
#endif

#endif /* INTEGER_SELECTOR_SCORER_H */
