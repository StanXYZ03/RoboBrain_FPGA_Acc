#include "ffn_vector.h"

#include <cstdint>
#include <hls_math.h>

namespace {

constexpr int PARAM_BANKS = FFNV_LANES;
constexpr int STAT_BANKS = 16;

const float SILU_LUT[65] = {
    -0.00268280104f, -0.00333681738f, -0.00414583978f, -0.00514511039f,
    -0.00637735836f, -0.00789419429f, -0.00975768467f, -0.0120420916f,
    -0.0148357389f, -0.0182429263f, -0.0223857574f, -0.0274056599f,
    -0.0334642546f, -0.0407430557f, -0.0494412418f, -0.0597704149f,
    -0.0719448398f, -0.0861651372f, -0.102592808f, -0.121312384f,
    -0.14227762f, -0.165238288f, -0.18964545f, -0.214536296f,
    -0.238405844f, -0.259082597f, -0.273638286f, -0.278375174f,
    -0.268941421f, -0.240615976f, -0.188770334f, -0.109455875f,
    0.0f, 0.140544125f, 0.311229666f, 0.509384024f,
    0.731058579f, 0.971624826f, 1.22636171f, 1.4909174f,
    1.76159416f, 2.0354637f, 2.31035455f, 2.58476171f,
    2.85772238f, 3.12868762f, 3.39740719f, 3.66383486f,
    3.92805516f, 4.19022959f, 4.45055876f, 4.70925694f,
    4.96653575f, 5.22259434f, 5.47761424f, 5.73175707f,
    5.98516426f, 6.23795791f, 6.49024232f, 6.74210581f,
    6.99362264f, 7.24485489f, 7.49585416f, 7.74666318f,
    7.9973172f,
};

const float GELU_LUT[65] = {
    -4.88498131e-15f, -3.5707548e-14f, -2.3939184e-13f, -1.51081925e-12f,
    -8.95866714e-12f, -4.9897711e-11f, -2.61039912e-10f, -1.2826646e-09f,
    -5.91952587e-09f, -2.56574917e-08f, -1.04442594e-07f, -3.99260427e-07f,
    -1.43325786e-06f, -4.8311454e-06f, -1.52895291e-05f, -4.54262345e-05f,
    -0.000126684967f, -0.00033156482f, -0.000814201777f, -0.00187533139f,
    -0.00404969409f, -0.0081943489f, -0.0155241633f, -0.0275050635f,
    -0.0455002639f, -0.0701035245f, -0.100210802f, -0.132062217f,
    -0.158655254f, -0.169970514f, -0.154268769f, -0.100323419f,
    0.0f, 0.149676581f, 0.345731231f, 0.580029486f,
    0.841344746f, 1.11793778f, 1.3997892f, 1.67989648f,
    1.95449974f, 2.22249494f, 2.48447584f, 2.74180565f,
    2.99595031f, 3.24812467f, 3.4991858f, 3.74966844f,
    3.99987332f, 4.24995457f, 4.49998471f, 4.74999517f,
    4.99999857f, 5.2499996f, 5.4999999f, 5.74999997f,
    5.99999999f, 6.25f, 6.5f, 6.75f,
    7.0f, 7.25f, 7.5f, 7.75f, 8.0f,
};

float bits_to_float(ap_uint<32> bits) {
#pragma HLS INLINE
    union {
        std::uint32_t u;
        float f;
    } value;
    value.u = bits.to_uint();
    return value.f;
}

ap_uint<32> float_to_bits(float input) {
#pragma HLS INLINE
    union {
        std::uint32_t u;
        float f;
    } value;
    value.f = input;
    return value.u;
}

ap_uint<16> float_to_half_bits(float input) {
#pragma HLS INLINE
    const std::uint32_t bits = float_to_bits(input).to_uint();
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

float lane(const ffnv_f32x4_t &word, int index) {
#pragma HLS INLINE
    return bits_to_float(word.range(32 * index + 31, 32 * index));
}

ffnv_f32x4_t pack4(const float values[FFNV_LANES]) {
#pragma HLS INLINE
    ffnv_f32x4_t word = 0;
    for (int index = 0; index < FFNV_LANES; ++index) {
#pragma HLS UNROLL
        word.range(32 * index + 31, 32 * index) = float_to_bits(values[index]);
    }
    return word;
}

float pwl_activation(float x, const float table[65]) {
#pragma HLS INLINE
    if (x <= -8.0f) return 0.0f;
    if (x >= 8.0f) return x;
    const float position = (x + 8.0f) * 4.0f;
    int segment = static_cast<int>(position);
    if (segment < 0) segment = 0;
    if (segment > 63) segment = 63;
    const float fraction = position - static_cast<float>(segment);
    const float left = table[segment];
    return left + (table[segment + 1] - left) * fraction;
}

float silu_pwl(float x) {
#pragma HLS INLINE
    return pwl_activation(x, SILU_LUT);
}

float gelu_pwl(float x) {
#pragma HLS INLINE
    return pwl_activation(x, GELU_LUT);
}

void load_parameters(const ffnv_f32x4_t *param0,
                     const ffnv_f32x4_t *param1,
                     float cache0[FFNV_MAX_COLS],
                     float cache1[FFNV_MAX_COLS],
                     int cols,
                     int flags) {
#pragma HLS INLINE off
    const int words = cols / FFNV_LANES;
    for (int word_index = 0; word_index < words; ++word_index) {
#pragma HLS PIPELINE II=2
        ffnv_f32x4_t word0 = 0;
        ffnv_f32x4_t word1 = 0;
        if (flags & FFNV_PARAM0_PRESENT) word0 = param0[word_index];
        if (flags & FFNV_PARAM1_PRESENT) word1 = param1[word_index];
        for (int item = 0; item < FFNV_LANES; ++item) {
#pragma HLS UNROLL
            const int col = word_index * FFNV_LANES + item;
            if (flags & FFNV_PARAM0_PRESENT) cache0[col] = lane(word0, item);
            if (flags & FFNV_PARAM1_PRESENT) cache1[col] = lane(word1, item);
        }
    }
}

void row_statistics(const ffnv_f32x4_t *src,
                    int row,
                    int cols,
                    bool center,
                    float &mean,
                    float &mean_square) {
#pragma HLS INLINE off
    float partial_sum[STAT_BANKS];
    float partial_square[STAT_BANKS];
#pragma HLS ARRAY_PARTITION variable=partial_sum complete
#pragma HLS ARRAY_PARTITION variable=partial_square complete
    for (int bank = 0; bank < STAT_BANKS; ++bank) {
#pragma HLS UNROLL
        partial_sum[bank] = 0.0f;
        partial_square[bank] = 0.0f;
    }
    const int words = cols / FFNV_LANES;
    const int base = row * words;
    for (int word_index = 0; word_index < words; ++word_index) {
#pragma HLS PIPELINE II=5
#pragma HLS DEPENDENCE variable=partial_sum inter false
#pragma HLS DEPENDENCE variable=partial_square inter false
        const ffnv_f32x4_t word = src[base + word_index];
        float sum4 = 0.0f;
        float square4 = 0.0f;
        for (int item = 0; item < FFNV_LANES; ++item) {
#pragma HLS UNROLL
            const float value = lane(word, item);
            sum4 += value;
            square4 += value * value;
        }
        const int bank = word_index & (STAT_BANKS - 1);
        partial_sum[bank] += sum4;
        partial_square[bank] += square4;
    }
    float sum = 0.0f;
    float square = 0.0f;
    for (int bank = 0; bank < STAT_BANKS; ++bank) {
#pragma HLS PIPELINE II=5
        sum += partial_sum[bank];
        square += partial_square[bank];
    }
    mean = center ? sum / static_cast<float>(cols) : 0.0f;
    mean_square = square / static_cast<float>(cols);
}

void quantize_block(const float values[FFNV_QK],
                    ffnv_q8x32_t &packed,
                    ap_uint<16> &scale) {
#pragma HLS INLINE off
    float amax = 0.0f;
    for (int index = 0; index < FFNV_QK; ++index) {
#pragma HLS PIPELINE II=1
        const float magnitude = hls::fabsf(values[index]);
        if (magnitude > amax) amax = magnitude;
    }
    const float d = amax / 127.0f;
    const float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    scale = float_to_half_bits(d);
    packed = 0;
    for (int index = 0; index < FFNV_QK; ++index) {
#pragma HLS PIPELINE II=1
        int quantized = static_cast<int>(hls::roundf(values[index] * inverse));
        if (quantized > 127) quantized = 127;
        if (quantized < -127) quantized = -127;
        const ap_int<8> q8 = quantized;
        packed.range(8 * index + 7, 8 * index) = q8.range(7, 0);
    }
}

void norm_quant(const ffnv_f32x4_t *src,
                const float gamma[FFNV_MAX_COLS],
                const float beta[FFNV_MAX_COLS],
                ffnv_q8x32_t *q_out,
                ap_uint<16> *scale_out,
                int rows,
                int cols,
                float epsilon,
                bool center,
                bool has_beta) {
#pragma HLS INLINE off
    const int words = cols / FFNV_LANES;
    const int blocks = cols / FFNV_QK;
    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        float mean_square = 0.0f;
        row_statistics(src, row, cols, center, mean, mean_square);
        float variance = center ? mean_square - mean * mean : mean_square;
        if (variance < 0.0f) variance = 0.0f;
        const float inverse_norm = 1.0f / hls::sqrtf(variance + epsilon);
        for (int block = 0; block < blocks; ++block) {
            float values[FFNV_QK];
#pragma HLS ARRAY_PARTITION variable=values cyclic factor=FFNV_LANES
            for (int word_in_block = 0; word_in_block < FFNV_QK / FFNV_LANES;
                 ++word_in_block) {
#pragma HLS PIPELINE II=1
                const int word_index = block * (FFNV_QK / FFNV_LANES) + word_in_block;
                const ffnv_f32x4_t word = src[row * words + word_index];
                for (int item = 0; item < FFNV_LANES; ++item) {
#pragma HLS UNROLL
                    const int local = word_in_block * FFNV_LANES + item;
                    const int col = block * FFNV_QK + local;
                    float value = (lane(word, item) - mean) * inverse_norm * gamma[col];
                    if (has_beta) value += beta[col];
                    values[local] = value;
                }
            }
            ffnv_q8x32_t packed;
            ap_uint<16> scale;
            quantize_block(values, packed, scale);
            const int output_index = row * blocks + block;
            q_out[output_index] = packed;
            scale_out[output_index] = scale;
        }
    }
}

void activation_quant(const ffnv_f32x4_t *src0,
                      const ffnv_f32x4_t *src1,
                      const float bias0[FFNV_MAX_COLS],
                      const float bias1[FFNV_MAX_COLS],
                      ffnv_q8x32_t *q_out,
                      ap_uint<16> *scale_out,
                      int rows,
                      int cols,
                      int mode,
                      int flags) {
#pragma HLS INLINE off
    const int words = cols / FFNV_LANES;
    const int blocks = cols / FFNV_QK;
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            float values[FFNV_QK];
#pragma HLS ARRAY_PARTITION variable=values cyclic factor=FFNV_LANES
            for (int word_in_block = 0; word_in_block < FFNV_QK / FFNV_LANES;
                 ++word_in_block) {
#pragma HLS PIPELINE II=1
                const int word_index = row * words +
                        block * (FFNV_QK / FFNV_LANES) + word_in_block;
                const ffnv_f32x4_t word0 = src0[word_index];
                ffnv_f32x4_t word1 = 0;
                if (mode == FFNV_SILU_MUL_QUANT) word1 = src1[word_index];
                for (int item = 0; item < FFNV_LANES; ++item) {
#pragma HLS UNROLL
                    const int local = word_in_block * FFNV_LANES + item;
                    const int col = block * FFNV_QK + local;
                    float first = lane(word0, item);
                    if (flags & FFNV_PARAM0_PRESENT) first += bias0[col];
                    if (mode == FFNV_SILU_MUL_QUANT) {
                        float second = lane(word1, item);
                        if (flags & FFNV_PARAM1_PRESENT) second += bias1[col];
                        values[local] = silu_pwl(first) * second;
                    } else {
                        values[local] = gelu_pwl(first);
                    }
                }
            }
            ffnv_q8x32_t packed;
            ap_uint<16> scale;
            quantize_block(values, packed, scale);
            const int output_index = row * blocks + block;
            q_out[output_index] = packed;
            scale_out[output_index] = scale;
        }
    }
}

void residual_add(const ffnv_f32x4_t *src0,
                  const ffnv_f32x4_t *src1,
                  const float bias[FFNV_MAX_COLS],
                  ffnv_f32x4_t *output,
                  int rows,
                  int cols,
                  bool has_bias) {
#pragma HLS INLINE off
    const int words = cols / FFNV_LANES;
    for (int row = 0; row < rows; ++row) {
        for (int word_index = 0; word_index < words; ++word_index) {
#pragma HLS PIPELINE II=1
            const ffnv_f32x4_t left = src0[row * words + word_index];
            const ffnv_f32x4_t right = src1[row * words + word_index];
            float values[FFNV_LANES];
#pragma HLS ARRAY_PARTITION variable=values complete
            for (int item = 0; item < FFNV_LANES; ++item) {
#pragma HLS UNROLL
                const int col = word_index * FFNV_LANES + item;
                values[item] = lane(left, item) + lane(right, item) +
                        (has_bias ? bias[col] : 0.0f);
            }
            output[row * words + word_index] = pack4(values);
        }
    }
}

}  // namespace

extern "C" int ffn_vector(
        const ffnv_f32x4_t *src0,
        const ffnv_f32x4_t *src1,
        const ffnv_f32x4_t *param0,
        const ffnv_f32x4_t *param1,
        ffnv_q8x32_t *q_out,
        ap_uint<16> *scale_out,
        ffnv_f32x4_t *f32_out,
        int rows,
        int cols,
        float epsilon,
        int mode,
        int flags) {
#pragma HLS INTERFACE m_axi port=src0 offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=param0 offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=param1 offset=slave bundle=gmem0 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=src1 offset=slave bundle=gmem1 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=q_out offset=slave bundle=gmem2 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=scale_out offset=slave bundle=gmem2 max_write_burst_length=64
#pragma HLS INTERFACE m_axi port=f32_out offset=slave bundle=gmem2 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=src0 bundle=control
#pragma HLS INTERFACE s_axilite port=src1 bundle=control
#pragma HLS INTERFACE s_axilite port=param0 bundle=control
#pragma HLS INTERFACE s_axilite port=param1 bundle=control
#pragma HLS INTERFACE s_axilite port=q_out bundle=control
#pragma HLS INTERFACE s_axilite port=scale_out bundle=control
#pragma HLS INTERFACE s_axilite port=f32_out bundle=control
#pragma HLS INTERFACE s_axilite port=rows bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=epsilon bundle=control
#pragma HLS INTERFACE s_axilite port=mode bundle=control
#pragma HLS INTERFACE s_axilite port=flags bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    if (rows <= 0 || rows > FFNV_MAX_ROWS || cols <= 0 ||
            cols > FFNV_MAX_COLS || (cols % FFNV_QK) != 0) {
        return FFNV_ERR_DIMENSION;
    }
    if (mode < FFNV_RMS_NORM_QUANT || mode > FFNV_RESIDUAL_ADD) {
        return FFNV_ERR_MODE;
    }
    if ((mode == FFNV_RMS_NORM_QUANT || mode == FFNV_LAYER_NORM_QUANT) &&
            (!(flags & FFNV_PARAM0_PRESENT) || epsilon <= 0.0f)) {
        return FFNV_ERR_PARAMETER;
    }
    if (mode == FFNV_LAYER_NORM_QUANT && !(flags & FFNV_PARAM1_PRESENT)) {
        return FFNV_ERR_PARAMETER;
    }

    float cache0[FFNV_MAX_COLS];
    float cache1[FFNV_MAX_COLS];
#pragma HLS ARRAY_PARTITION variable=cache0 cyclic factor=PARAM_BANKS
#pragma HLS ARRAY_PARTITION variable=cache1 cyclic factor=PARAM_BANKS
#pragma HLS BIND_STORAGE variable=cache0 type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=cache1 type=ram_2p impl=bram

    load_parameters(param0, param1, cache0, cache1, cols, flags);
    if (mode == FFNV_RMS_NORM_QUANT || mode == FFNV_LAYER_NORM_QUANT) {
        norm_quant(src0, cache0, cache1, q_out, scale_out,
                   rows, cols, epsilon, mode == FFNV_LAYER_NORM_QUANT,
                   flags & FFNV_PARAM1_PRESENT);
    } else if (mode == FFNV_SILU_MUL_QUANT || mode == FFNV_GELU_QUANT) {
        activation_quant(src0, src1, cache0, cache1, q_out, scale_out,
                         rows, cols, mode, flags);
    } else {
        residual_add(src0, src1, cache0, f32_out, rows, cols,
                     flags & FFNV_PARAM0_PRESENT);
    }
    return FFNV_OK;
}
