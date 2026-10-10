#ifndef ROBOBRAIN_SELECTOR_QKV_ADAPTER_H
#define ROBOBRAIN_SELECTOR_QKV_ADAPTER_H

#include <ap_int.h>

// ---------------------------------------------------------------------------
// Size contract
// ---------------------------------------------------------------------------
// The adapter consumes the Selector segment-0 token payload and produces the
// GGML Q8_0 activation contract used by the existing Q8Q8 PE16 DATAFLOW GEMM.
// The bias-add kernel is an independent element-wise FP32 operator that adds
// the fused QKV bias to the Q8Q8 output; it is NOT an attention kernel.
constexpr int SQA_HIDDEN          = 1024;  // Selector block0 hidden width
constexpr int SQA_BLOCK           = 32;    // GGML Q8_0 block size
constexpr int SQA_BLOCKS_PER_ROW  = SQA_HIDDEN / SQA_BLOCK;          // 32
constexpr int SQA_INT8_LANES      = 16;    // int8 lanes per 128-bit word
constexpr int SQA_WORDS_PER_ROW   = SQA_HIDDEN / SQA_INT8_LANES;     // 64
constexpr int SQA_F32_LANES       = 4;     // fp32 lanes per 128-bit word
constexpr int SQA_PARAM_WORDS     = SQA_HIDDEN / SQA_F32_LANES;      // 256
constexpr int SQA_Q8_LANES        = 32;    // int8 lanes per 256-bit payload
// Adapter row contract: the Selector legal budgets are K = 48 / 64 / 96 only,
// so the adapter accepts exactly M = 4K = 192 / 256 / 384. Any other value
// (including 1, 191, 193, 385, 4320) is a dimension error. SQA_MAX_ROWS is the
// adapter-only upper bound and equals the largest legal M.
constexpr int SQA_LEGAL_ROWS_0    = 192;   // K = 48
constexpr int SQA_LEGAL_ROWS_1    = 256;   // K = 64
constexpr int SQA_LEGAL_ROWS_2    = 384;   // K = 96
constexpr int SQA_MAX_ROWS        = SQA_LEGAL_ROWS_2;  // adapter upper bound (384)
// qkv_bias_add stays a general element-wise operator; it keeps the historical
// Q8Q8 PE16 row gate. The formal qkv case this round is M = 192/256/384,
// N = 3072.
constexpr int SQA_BIAS_MAX_ROWS   = 4320;  // bias-add general row gate
constexpr int SQA_MAX_COLS        = 9728;  // Q8Q8 PE16 output column gate
constexpr int SQA_QKV_COLS        = 3072;  // fused QKV output width
constexpr int SQA_QKV_WORDS       = SQA_QKV_COLS / SQA_F32_LANES;    // 768
constexpr int SQA_PARAM_BYTES     = SQA_PARAM_WORDS * 16;            // 4096 B each
constexpr float SQA_EPSILON       = 1.0e-6f;

// Selector int8 clamp allows -128 (quantization_contract.md §1); GGML Q8_0
// clamps the *requantized* activation to [-127, 127] (ffn_vector.cpp quantize).
constexpr int SQA_SELECTOR_MIN = -128;
constexpr int SQA_SELECTOR_MAX = 127;
constexpr int SQA_Q8_0_MIN     = -127;
constexpr int SQA_Q8_0_MAX     = 127;

// 128-bit little-endian word: lane L occupies bits [8L+7:8L].
typedef ap_uint<128> sqa_token_word_t;  // 16 x int8 packed token lanes
typedef ap_uint<128> sqa_f32x4_t;       // 4 x fp32 packed lanes
typedef ap_uint<256> sqa_q8x32_t;       // 32 x int8 packed Q8_0 payload

enum sqa_status {
    SQA_OK             = 0,
    SQA_ERR_DIMENSION  = -1,
    SQA_ERR_SCALE      = -2,
    SQA_ERR_PARAMETER  = -3,
};

// Adapter row gate: exactly the three Selector budgets K = 48/64/96.
inline bool sqa_is_legal_adapter_rows(int rows) {
    return rows == SQA_LEGAL_ROWS_0 || rows == SQA_LEGAL_ROWS_1 ||
           rows == SQA_LEGAL_ROWS_2;
}

// Selector segment0 [rows,1024] int8 -> LayerNorm(norm1, eps=1e-6) -> Q8_0.
//
// Legal rows are exactly M = 192 / 256 / 384 (Selector K = 48 / 64 / 96); any
// other value returns SQA_ERR_DIMENSION.
//
// The LayerNorm is evaluated on the dequantized value x = q / global_scale with
// the algebraically equivalent fused form
//     LN(q/s) = ((q - mean_q) / sqrt(var_q + eps*s*s)) * gamma + beta
// where mean_q/var_q are the population statistics of the int8 lanes. This
// avoids materializing x and is exactly the LayerNorm semantics, not a scale
// broadcast. Output layout matches q8q8_gemm activations:
//     act_q[row * 32 + block]      : 256-bit payload (32 int8, little-endian)
//     act_scale[row * 32 + block]  : binary16 scale bits
extern "C" int selector_ln_q8_adapter(
        const sqa_token_word_t *tokens_q,
        const sqa_f32x4_t *norm_gamma,
        const sqa_f32x4_t *norm_beta,
        sqa_q8x32_t *act_q,
        ap_uint<16> *act_scale,
        float global_scale,
        int rows);

// Independent element-wise FP32 bias-add operator.
//   y_out[row, col] = y_in[row, col] + bias[col]
// Inputs/outputs use the Q8Q8 4 x fp32 / 128-bit word layout
//     word index = row * (cols / 4) + col / 4, lane = col % 4.
extern "C" int qkv_bias_add(
        const sqa_f32x4_t *y_in,
        const sqa_f32x4_t *bias,
        sqa_f32x4_t *y_out,
        int rows,
        int cols);

#endif
