#include "selector_qkv_adapter.h"

#include <cstdint>

// The Vitis HLS math library is only available inside the HLS toolchain. System
// g++ (the normal and --fsanitize builds) falls back to <cmath>. Both paths must
// agree bit-for-bit: sqrt is IEEE correctly rounded and roundf is
// round-half-away-from-zero in both libraries. HLS C-sim defines __HLS_CSIM__
// and synthesis defines __SYNTHESIS__.
#if defined(__HLS_CSIM__) || defined(__SYNTHESIS__) || defined(__VITIS_HLS__)
#  include <hls_math.h>
#  define SQA_HAS_HLS_MATH 1
#endif
#ifndef SQA_HAS_HLS_MATH
#  include <cmath>
#endif

namespace {

inline float sqa_sqrtf(float value) {
#ifdef SQA_HAS_HLS_MATH
    return hls::sqrtf(value);
#else
    return std::sqrt(value);
#endif
}

inline float sqa_fabsf(float value) {
#ifdef SQA_HAS_HLS_MATH
    return hls::fabsf(value);
#else
    return std::fabs(value);
#endif
}

inline float sqa_roundf(float value) {
#ifdef SQA_HAS_HLS_MATH
    return hls::roundf(value);
#else
    return std::round(value);
#endif
}

float bits_to_float(std::uint32_t bits) {
    union {
        std::uint32_t u;
        float f;
    } value;
    value.u = bits;
    return value.f;
}

std::uint32_t float_to_bits(float input) {
    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.f = input;
    return converter.u;
}

// Exact copy of ffn_vector.cpp::float_to_half_bits so the stored binary16
// scale is byte-identical to the existing Q8_0 requantization contract.
ap_uint<16> float_to_half_bits(float input) {
    const std::uint32_t bits = float_to_bits(input);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    std::uint32_t exponent = (bits >> 23) & 0xffu;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return ap_uint<16>(sign | (mantissa ? 0x7e00u : 0x7c00u));
    }
    if (exponent > 142u) return ap_uint<16>(sign | 0x7c00u);
    if (exponent < 113u) {
        if (exponent < 103u) return ap_uint<16>(sign);
        mantissa |= 0x800000u;
        const std::uint32_t shift = 125u - exponent;
        std::uint32_t rounded = mantissa >> shift;
        const std::uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const std::uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) {
            ++rounded;
        }
        return ap_uint<16>(sign | rounded);
    }
    exponent -= 112u;
    mantissa += 0xfffu + ((mantissa >> 13) & 1u);
    if (mantissa & 0x800000u) {
        mantissa = 0;
        ++exponent;
        if (exponent >= 31u) return ap_uint<16>(sign | 0x7c00u);
    }
    return ap_uint<16>(sign | (exponent << 10) | (mantissa >> 13));
}

float lane_f32(const sqa_f32x4_t &word, int index) {
#pragma HLS INLINE
    return bits_to_float(word.range(32 * index + 31, 32 * index).to_uint());
}

sqa_f32x4_t pack_f32(const float values[SQA_F32_LANES]) {
#pragma HLS INLINE
    sqa_f32x4_t word = 0;
    for (int index = 0; index < SQA_F32_LANES; ++index) {
#pragma HLS UNROLL
        word.range(32 * index + 31, 32 * index) = float_to_bits(values[index]);
    }
    return word;
}

inline int unpack_int8_lane(const sqa_token_word_t &word, int lane) {
#pragma HLS INLINE
    const ap_int<8> value = word.range(8 * lane + 7, 8 * lane);
    return value.to_int();
}

bool is_finite_positive(float value) {
    if (!(value > 0.0f)) return false;
    const std::uint32_t bits = float_to_bits(value);
    return ((bits >> 23) & 0xffu) != 0xffu;
}

// GGML Q8_0 requantization of one 32-lane block, identical to
// ffn_vector.cpp::quantize_block (d = amax/127, clamp [-127,127], half scale).
void quantize_block(const float values[SQA_BLOCK],
                    sqa_q8x32_t &packed,
                    ap_uint<16> &scale) {
#pragma HLS INLINE
    float amax = 0.0f;
    for (int index = 0; index < SQA_BLOCK; ++index) {
// Scheduling-only pragma: the loop-carried max-reduction path
// (fcmp -> select -> store on 'amax') does not fit 3.650 ns at II=1 in the
// 2023.2 fabric compare. II=2 gives the scheduler a second cycle; the
// reduction order, arithmetic and result are unchanged. This is at most a
// scheduling fix and does not change the Q8_0 quantization.
#pragma HLS PIPELINE II=2
        const float magnitude = sqa_fabsf(values[index]);
        if (magnitude > amax) amax = magnitude;
    }
    const float d = amax / 127.0f;
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    scale = float_to_half_bits(d);
    packed = 0;
    for (int index = 0; index < SQA_BLOCK; ++index) {
#pragma HLS PIPELINE II=1
        int quantized = static_cast<int>(sqa_roundf(values[index] * inverse));
        if (quantized > SQA_Q8_0_MAX) quantized = SQA_Q8_0_MAX;
        if (quantized < SQA_Q8_0_MIN) quantized = SQA_Q8_0_MIN;
        const ap_int<8> q8 = quantized;
        packed.range(8 * index + 7, 8 * index) = q8.range(7, 0);
    }
}

}  // namespace

extern "C" int selector_ln_q8_adapter(
        const sqa_token_word_t *tokens_q,
        const sqa_f32x4_t *norm_gamma,
        const sqa_f32x4_t *norm_beta,
        sqa_q8x32_t *act_q,
        ap_uint<16> *act_scale,
        float global_scale,
        int rows) {
#pragma HLS INTERFACE m_axi port=tokens_q offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=norm_gamma offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=norm_beta offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=act_q offset=slave bundle=gmem1 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=act_scale offset=slave bundle=gmem1 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=tokens_q bundle=control
#pragma HLS INTERFACE s_axilite port=norm_gamma bundle=control
#pragma HLS INTERFACE s_axilite port=norm_beta bundle=control
#pragma HLS INTERFACE s_axilite port=act_q bundle=control
#pragma HLS INTERFACE s_axilite port=act_scale bundle=control
#pragma HLS INTERFACE s_axilite port=global_scale bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (!sqa_is_legal_adapter_rows(rows)) {
        return SQA_ERR_DIMENSION;
    }
    if (!is_finite_positive(global_scale)) {
        return SQA_ERR_SCALE;
    }

    // eps * s * s, left-associative, matching the scalar dataflow.
    const float eps_sq = SQA_EPSILON * global_scale * global_scale;

    // norm_gamma / norm_beta are loop-invariant for the whole call. Read each
    // packed 128-bit word exactly once into an on-chip cache (256 words each,
    // 4 KiB each, 8 KiB total) bound to BRAM, and serve every row/block from
    // the cache. This changes only the storage/scheduling of the parameter
    // reads; the arithmetic, Q8_0 payload/scale and port layout are unchanged.
    sqa_f32x4_t gamma_cache[SQA_PARAM_WORDS];
    sqa_f32x4_t beta_cache[SQA_PARAM_WORDS];
#pragma HLS BIND_STORAGE variable=gamma_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=beta_cache type=ram_2p impl=bram
    for (int param_word = 0; param_word < SQA_PARAM_WORDS; ++param_word) {
#pragma HLS PIPELINE II=1
        gamma_cache[param_word] = norm_gamma[param_word];
        beta_cache[param_word] = norm_beta[param_word];
    }

    for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=SQA_LEGAL_ROWS_0 max=SQA_MAX_ROWS
        const int row_word_base = row * SQA_WORDS_PER_ROW;

        // Pass 1: integer statistics of the int8 lanes. Sums are exact in
        // int32 (|sum| <= 131072, sum_sq <= 1024 * 16384 = 2^24).
        int sum_q = 0;
        int sum_sq = 0;
        for (int word_index = 0; word_index < SQA_WORDS_PER_ROW; ++word_index) {
#pragma HLS PIPELINE II=1
            const sqa_token_word_t word = tokens_q[row_word_base + word_index];
            for (int lane = 0; lane < SQA_INT8_LANES; ++lane) {
#pragma HLS UNROLL
                const int q = unpack_int8_lane(word, lane);
                sum_q += q;
                sum_sq += q * q;
            }
        }
        const float mean = static_cast<float>(sum_q) / static_cast<float>(SQA_HIDDEN);
        const float mean_square = static_cast<float>(sum_sq) /
                static_cast<float>(SQA_HIDDEN);
        float variance = mean_square - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        const float denominator = variance + eps_sq;
        // Constant row with eps*s*s == 0 -> zero variance; the centered value
        // is exactly zero, so inverse_norm = 0 gives LN = beta, matching the
        // direct chain (x - mean_x == 0). NaN denominators also map to beta.
        const float inverse_norm =
                (denominator > 0.0f) ? 1.0f / sqa_sqrtf(denominator) : 0.0f;

        // Pass 2: normalize, affine, requantize per 32-lane block. Requires
        // cols % 32 == 0 and cols % 4 == 0.
        for (int block = 0; block < SQA_BLOCKS_PER_ROW; ++block) {
#pragma HLS LOOP_TRIPCOUNT min=SQA_BLOCKS_PER_ROW max=SQA_BLOCKS_PER_ROW
            float values[SQA_BLOCK];
#pragma HLS ARRAY_PARTITION variable=values cyclic factor=SQA_F32_LANES
            const int block_col = block * SQA_BLOCK;
            for (int word_in_block = 0; word_in_block < SQA_BLOCK / SQA_F32_LANES;
                 ++word_in_block) {
#pragma HLS PIPELINE II=1
                const int local_base = word_in_block * SQA_F32_LANES;
                const int col = block_col + local_base;
                const sqa_f32x4_t gamma_word = gamma_cache[col / SQA_F32_LANES];
                const sqa_f32x4_t beta_word = beta_cache[col / SQA_F32_LANES];
                for (int lane = 0; lane < SQA_F32_LANES; ++lane) {
#pragma HLS UNROLL
                    const int element = local_base + lane;
                    const int feature = block_col + element;
                    const int q = unpack_int8_lane(
                            tokens_q[row_word_base + feature / SQA_INT8_LANES],
                            feature % SQA_INT8_LANES);
                    const float centered = static_cast<float>(q) - mean;
                    const float normalized = (centered * inverse_norm) *
                            lane_f32(gamma_word, lane);
                    values[element] = normalized + lane_f32(beta_word, lane);
                }
            }
            sqa_q8x32_t packed;
            ap_uint<16> scale;
            quantize_block(values, packed, scale);
            const int output_index = row * SQA_BLOCKS_PER_ROW + block;
            act_q[output_index] = packed;
            act_scale[output_index] = scale;
        }
    }
    return SQA_OK;
}

extern "C" int qkv_bias_add(
        const sqa_f32x4_t *y_in,
        const sqa_f32x4_t *bias,
        sqa_f32x4_t *y_out,
        int rows,
        int cols) {
#pragma HLS INTERFACE m_axi port=y_in offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=y_out offset=slave bundle=gmem1 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=y_in bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=y_out bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (rows <= 0 || rows > SQA_BIAS_MAX_ROWS || cols <= 0 ||
            cols > SQA_MAX_COLS || (cols % SQA_F32_LANES) != 0) {
        return SQA_ERR_DIMENSION;
    }
    const int words_per_row = cols / SQA_F32_LANES;
    for (int row = 0; row < rows; ++row) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=SQA_BIAS_MAX_ROWS
        for (int word_index = 0; word_index < words_per_row; ++word_index) {
#pragma HLS PIPELINE II=1
            const sqa_f32x4_t input_word = y_in[row * words_per_row + word_index];
            const sqa_f32x4_t bias_word = bias[word_index];
            float values[SQA_F32_LANES];
#pragma HLS ARRAY_PARTITION variable=values complete
            for (int lane = 0; lane < SQA_F32_LANES; ++lane) {
#pragma HLS UNROLL
                values[lane] = lane_f32(input_word, lane) + lane_f32(bias_word, lane);
            }
            y_out[row * words_per_row + word_index] = pack_f32(values);
        }
    }
    return SQA_OK;
}
