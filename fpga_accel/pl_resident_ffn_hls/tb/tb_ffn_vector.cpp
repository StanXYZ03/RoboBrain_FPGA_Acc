#include "ffn_vector.h"

#include <hls_half.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {

ap_uint<32> float_to_bits(float value) {
    union { std::uint32_t u; float f; } converter;
    converter.f = value;
    return converter.u;
}

float bits_to_float(ap_uint<32> value) {
    union { std::uint32_t u; float f; } converter;
    converter.u = value.to_uint();
    return converter.f;
}

float half_bits_to_float(ap_uint<16> bits) {
    half value;
    value.set_bits(static_cast<unsigned short>(bits.to_uint()));
    return static_cast<float>(value);
}

std::vector<ffnv_f32x4_t> pack(const std::vector<float> &input) {
    std::vector<ffnv_f32x4_t> output((input.size() + FFNV_LANES - 1) / FFNV_LANES);
    for (std::size_t word = 0; word < output.size(); ++word) {
        ffnv_f32x4_t packed = 0;
        for (int item = 0; item < FFNV_LANES; ++item) {
            const std::size_t index = word * FFNV_LANES + item;
            const float value = index < input.size() ? input[index] : 0.0f;
            packed.range(32 * item + 31, 32 * item) = float_to_bits(value);
        }
        output[word] = packed;
    }
    return output;
}

std::vector<float> unpack(const std::vector<ffnv_f32x4_t> &input,
                          std::size_t count) {
    std::vector<float> output(count);
    for (std::size_t index = 0; index < count; ++index) {
        const int item = static_cast<int>(index % FFNV_LANES);
        output[index] = bits_to_float(input[index / FFNV_LANES].range(
                32 * item + 31, 32 * item));
    }
    return output;
}

std::vector<float> dequantize(const std::vector<ffnv_q8x32_t> &q,
                              const std::vector<ap_uint<16> > &scale,
                              int rows,
                              int cols) {
    const int blocks = cols / FFNV_QK;
    std::vector<float> output(rows * cols);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks; ++block) {
            const int block_index = row * blocks + block;
            const float d = half_bits_to_float(scale[block_index]);
            for (int item = 0; item < FFNV_QK; ++item) {
                const ap_int<8> value = q[block_index].range(
                        8 * item + 7, 8 * item);
                output[row * cols + block * FFNV_QK + item] =
                        static_cast<float>(value.to_int()) * d;
            }
        }
    }
    return output;
}

bool check_error(const char *name,
                 const std::vector<float> &actual,
                 const std::vector<float> &expected,
                 float max_limit,
                 float rmse_limit) {
    float max_abs = 0.0f;
    double square_sum = 0.0;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float error = std::fabs(actual[index] - expected[index]);
        max_abs = std::max(max_abs, error);
        square_sum += static_cast<double>(error) * error;
    }
    const float rmse = static_cast<float>(std::sqrt(square_sum / expected.size()));
    std::printf("FFNV_CASE name=%s max_abs=%.9g rmse=%.9g result=%s\n",
                name, max_abs, rmse,
                max_abs <= max_limit && rmse <= rmse_limit ? "PASS" : "FAIL");
    return max_abs <= max_limit && rmse <= rmse_limit;
}

std::vector<float> make_input(int rows, int cols, int salt, float scale) {
    std::vector<float> result(rows * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int raw = ((row * 97 + col * 37 + salt * 17) % 257) - 128;
            result[row * cols + col] = static_cast<float>(raw) * scale;
        }
    }
    return result;
}

struct Q8Matrix {
    int outputs;
    int inputs;
    std::vector<std::int8_t> values;
    std::vector<float> scales;
};

Q8Matrix make_q8_matrix(int outputs, int inputs, int salt) {
    Q8Matrix matrix;
    matrix.outputs = outputs;
    matrix.inputs = inputs;
    matrix.values.resize(outputs * inputs);
    matrix.scales.resize(outputs * inputs / FFNV_QK);
    const int blocks = inputs / FFNV_QK;
    for (int output = 0; output < outputs; ++output) {
        for (int block = 0; block < blocks; ++block) {
            matrix.scales[output * blocks + block] =
                    0.001953125f * static_cast<float>(1 +
                    ((output * 3 + block + salt) % 4));
            for (int item = 0; item < FFNV_QK; ++item) {
                const int input = block * FFNV_QK + item;
                const int raw = (output * 19 + input * 11 + salt * 7) % 31;
                matrix.values[output * inputs + input] =
                        static_cast<std::int8_t>(raw - 15);
            }
        }
    }
    return matrix;
}

std::vector<float> q8_linear(const std::vector<ffnv_q8x32_t> &input,
                             const std::vector<ap_uint<16> > &input_scales,
                             int rows,
                             const Q8Matrix &weight) {
    const int blocks = weight.inputs / FFNV_QK;
    std::vector<float> output(rows * weight.outputs, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < weight.outputs; ++out) {
            float sum = 0.0f;
            for (int block = 0; block < blocks; ++block) {
                int integer_dot = 0;
                const int input_block = row * blocks + block;
                for (int item = 0; item < FFNV_QK; ++item) {
                    const ap_int<8> activation = input[input_block].range(
                            8 * item + 7, 8 * item);
                    const int input_col = block * FFNV_QK + item;
                    integer_dot += activation.to_int() *
                            static_cast<int>(weight.values[
                            out * weight.inputs + input_col]);
                }
                sum += static_cast<float>(integer_dot) *
                        half_bits_to_float(input_scales[input_block]) *
                        weight.scales[out * blocks + block];
            }
            output[row * weight.outputs + out] = sum;
        }
    }
    return output;
}

std::vector<float> float_linear(const std::vector<float> &input,
                                int rows,
                                const Q8Matrix &weight) {
    const int blocks = weight.inputs / FFNV_QK;
    std::vector<float> output(rows * weight.outputs, 0.0f);
    for (int row = 0; row < rows; ++row) {
        for (int out = 0; out < weight.outputs; ++out) {
            float sum = 0.0f;
            for (int input_col = 0; input_col < weight.inputs; ++input_col) {
                const int block = input_col / FFNV_QK;
                const float dequant_weight = static_cast<float>(weight.values[
                        out * weight.inputs + input_col]) *
                        weight.scales[out * blocks + block];
                sum += input[row * weight.inputs + input_col] * dequant_weight;
            }
            output[row * weight.outputs + out] = sum;
        }
    }
    return output;
}

std::vector<float> reference_norm(const std::vector<float> &input,
                                  const std::vector<float> &gamma,
                                  const std::vector<float> &beta,
                                  int rows,
                                  int cols,
                                  bool center,
                                  float epsilon) {
    std::vector<float> output(rows * cols);
    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        float mean_square = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float value = input[row * cols + col];
            mean += value;
            mean_square += value * value;
        }
        mean = center ? mean / cols : 0.0f;
        float variance = mean_square / cols - (center ? mean * mean : 0.0f);
        const float inverse = 1.0f /
                std::sqrt(std::max(variance, 0.0f) + epsilon);
        for (int col = 0; col < cols; ++col) {
            output[row * cols + col] =
                    (input[row * cols + col] - mean) * inverse * gamma[col] +
                    (center ? beta[col] : 0.0f);
        }
    }
    return output;
}

bool run_ffn_composition(bool vision) {
    const int rows = 2;
    const int hidden = 64;
    const int expanded = 64;
    const float epsilon = 1.0e-5f;
    const std::vector<float> input = make_input(rows, hidden,
            vision ? 23 : 17, 0.01171875f);
    std::vector<float> gamma(hidden);
    std::vector<float> beta(hidden);
    for (int col = 0; col < hidden; ++col) {
        gamma[col] = 0.875f + 0.00390625f * static_cast<float>(col % 17);
        beta[col] = vision ? 0.00390625f * static_cast<float>((col % 7) - 3) : 0.0f;
    }
    const Q8Matrix first_weight = make_q8_matrix(expanded, hidden,
                                                  vision ? 31 : 29);
    const Q8Matrix second_weight = make_q8_matrix(expanded, hidden, 37);
    const Q8Matrix down_weight = make_q8_matrix(hidden, expanded,
                                                vision ? 43 : 41);
    std::vector<float> activation_bias(expanded);
    std::vector<float> down_bias(hidden);
    for (int col = 0; col < expanded; ++col) {
        activation_bias[col] = vision ?
                0.00390625f * static_cast<float>((col % 5) - 2) : 0.0f;
    }
    for (int col = 0; col < hidden; ++col) {
        down_bias[col] = vision ?
                0.001953125f * static_cast<float>((col % 3) - 1) : 0.0f;
    }

    const auto input_words = pack(input);
    const auto gamma_words = pack(gamma);
    const auto beta_words = pack(beta);
    std::vector<ffnv_q8x32_t> norm_q(rows * hidden / FFNV_QK);
    std::vector<ap_uint<16> > norm_scales(norm_q.size());
    std::vector<ffnv_f32x4_t> unused_f32(rows * hidden / FFNV_LANES);
    const int norm_status = ffn_vector(input_words.data(), input_words.data(),
            gamma_words.data(), beta_words.data(), norm_q.data(),
            norm_scales.data(), unused_f32.data(), rows, hidden, epsilon,
            vision ? FFNV_LAYER_NORM_QUANT : FFNV_RMS_NORM_QUANT,
            FFNV_PARAM0_PRESENT | (vision ? FFNV_PARAM1_PRESENT : 0));
    if (norm_status != FFNV_OK) return false;

    std::vector<float> first = q8_linear(norm_q, norm_scales, rows, first_weight);
    std::vector<float> second = vision ? std::vector<float>(first.size(), 0.0f) :
            q8_linear(norm_q, norm_scales, rows, second_weight);
    const auto first_words = pack(first);
    const auto second_words = pack(second);
    const auto activation_bias_words = pack(activation_bias);
    std::vector<ffnv_q8x32_t> activation_q(rows * expanded / FFNV_QK);
    std::vector<ap_uint<16> > activation_scales(activation_q.size());
    const int activation_status = ffn_vector(first_words.data(),
            second_words.data(), activation_bias_words.data(),
            activation_bias_words.data(), activation_q.data(),
            activation_scales.data(), unused_f32.data(), rows, expanded,
            epsilon, vision ? FFNV_GELU_QUANT : FFNV_SILU_MUL_QUANT,
            vision ? FFNV_PARAM0_PRESENT : 0);
    if (activation_status != FFNV_OK) return false;

    const std::vector<float> down = q8_linear(activation_q, activation_scales,
                                               rows, down_weight);
    const auto down_words = pack(down);
    const auto down_bias_words = pack(down_bias);
    std::vector<ffnv_f32x4_t> output_words(rows * hidden / FFNV_LANES);
    std::vector<ffnv_q8x32_t> unused_q(rows * hidden / FFNV_QK);
    std::vector<ap_uint<16> > unused_scales(unused_q.size());
    const int residual_status = ffn_vector(input_words.data(), down_words.data(),
            down_bias_words.data(), down_bias_words.data(), unused_q.data(),
            unused_scales.data(), output_words.data(), rows, hidden, epsilon,
            FFNV_RESIDUAL_ADD, vision ? FFNV_PARAM0_PRESENT : 0);
    if (residual_status != FFNV_OK) return false;

    const std::vector<float> normalized = reference_norm(input, gamma, beta,
            rows, hidden, vision, epsilon);
    std::vector<float> reference_first = float_linear(normalized, rows,
                                                       first_weight);
    std::vector<float> reference_activated(rows * expanded);
    if (vision) {
        for (int index = 0; index < rows * expanded; ++index) {
            const float x = reference_first[index] +
                    activation_bias[index % expanded];
            reference_activated[index] = 0.5f * x *
                    (1.0f + std::erf(x / std::sqrt(2.0f)));
        }
    } else {
        const std::vector<float> reference_second = float_linear(
                normalized, rows, second_weight);
        for (int index = 0; index < rows * expanded; ++index) {
            const float x = reference_first[index];
            reference_activated[index] = x / (1.0f + std::exp(-x)) *
                    reference_second[index];
        }
    }
    std::vector<float> expected = float_linear(reference_activated, rows,
                                                down_weight);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < hidden; ++col) {
            const int index = row * hidden + col;
            expected[index] += input[index] + down_bias[col];
        }
    }
    return check_error(vision ? "vision_ffn_composition" :
                       "llm_ffn_composition", unpack(output_words, expected.size()),
                       expected, vision ? 0.04f : 0.04f,
                       vision ? 0.012f : 0.012f);
}

bool run_norm_case(int mode) {
    const int rows = 3;
    const int cols = 64;
    const float epsilon = 1.0e-5f;
    std::vector<float> input = make_input(rows, cols, 1, 0.015625f);
    std::vector<float> gamma(cols);
    std::vector<float> beta(cols);
    for (int col = 0; col < cols; ++col) {
        gamma[col] = 0.75f + 0.005f * static_cast<float>(col);
        beta[col] = 0.01f * static_cast<float>((col % 7) - 3);
    }
    const auto src = pack(input);
    const auto gamma_words = pack(gamma);
    const auto beta_words = pack(beta);
    std::vector<ffnv_q8x32_t> q(rows * cols / FFNV_QK);
    std::vector<ap_uint<16> > scales(q.size());
    std::vector<ffnv_f32x4_t> unused(rows * cols / FFNV_LANES);
    const int flags = FFNV_PARAM0_PRESENT |
            (mode == FFNV_LAYER_NORM_QUANT ? FFNV_PARAM1_PRESENT : 0);
    const int status = ffn_vector(src.data(), src.data(), gamma_words.data(),
            beta_words.data(), q.data(), scales.data(), unused.data(),
            rows, cols, epsilon, mode, flags);
    if (status != FFNV_OK) return false;

    std::vector<float> expected(rows * cols);
    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        float square = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float value = input[row * cols + col];
            mean += value;
            square += value * value;
        }
        mean = mode == FFNV_LAYER_NORM_QUANT ? mean / cols : 0.0f;
        float variance = square / cols;
        if (mode == FFNV_LAYER_NORM_QUANT) variance -= mean * mean;
        const float inverse = 1.0f / std::sqrt(std::max(variance, 0.0f) + epsilon);
        for (int col = 0; col < cols; ++col) {
            float value = (input[row * cols + col] - mean) * inverse * gamma[col];
            if (mode == FFNV_LAYER_NORM_QUANT) value += beta[col];
            expected[row * cols + col] = value;
        }
    }
    return check_error(mode == FFNV_RMS_NORM_QUANT ? "rms_norm_quant" :
                       "layer_norm_quant", dequantize(q, scales, rows, cols),
                       expected, 0.025f, 0.009f);
}

bool run_activation_case(int mode) {
    const int rows = 4;
    const int cols = 64;
    std::vector<float> first = make_input(rows, cols, 3, 0.046875f);
    std::vector<float> second = make_input(rows, cols, 7, 0.015625f);
    std::vector<float> bias0(cols);
    std::vector<float> bias1(cols);
    for (int col = 0; col < cols; ++col) {
        bias0[col] = 0.0078125f * static_cast<float>((col % 9) - 4);
        bias1[col] = 0.00390625f * static_cast<float>((col % 5) - 2);
    }
    const auto src0 = pack(first);
    const auto src1 = pack(second);
    const auto param0 = pack(bias0);
    const auto param1 = pack(bias1);
    std::vector<ffnv_q8x32_t> q(rows * cols / FFNV_QK);
    std::vector<ap_uint<16> > scales(q.size());
    std::vector<ffnv_f32x4_t> unused(rows * cols / FFNV_LANES);
    const int flags = FFNV_PARAM0_PRESENT |
            (mode == FFNV_SILU_MUL_QUANT ? FFNV_PARAM1_PRESENT : 0);
    const int status = ffn_vector(src0.data(), src1.data(), param0.data(),
            param1.data(), q.data(), scales.data(), unused.data(),
            rows, cols, 1.0e-5f, mode, flags);
    if (status != FFNV_OK) return false;

    std::vector<float> expected(rows * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int index = row * cols + col;
            const float x = first[index] + bias0[col];
            if (mode == FFNV_SILU_MUL_QUANT) {
                const float silu = x / (1.0f + std::exp(-x));
                expected[index] = silu * (second[index] + bias1[col]);
            } else {
                expected[index] = 0.5f * x *
                        (1.0f + std::erf(x / std::sqrt(2.0f)));
            }
        }
    }
    return check_error(mode == FFNV_SILU_MUL_QUANT ? "silu_mul_quant_pwl" :
                       "gelu_quant_pwl", dequantize(q, scales, rows, cols),
                       expected,
                       mode == FFNV_SILU_MUL_QUANT ? 0.075f : 0.04f,
                       mode == FFNV_SILU_MUL_QUANT ? 0.02f : 0.012f);
}

bool run_residual_case() {
    const int rows = 3;
    const int cols = 64;
    const std::vector<float> left = make_input(rows, cols, 11, 0.03125f);
    const std::vector<float> right = make_input(rows, cols, 13, 0.015625f);
    std::vector<float> bias(cols);
    for (int col = 0; col < cols; ++col) bias[col] = 0.125f * (col % 3);
    const auto left_words = pack(left);
    const auto right_words = pack(right);
    const auto bias_words = pack(bias);
    std::vector<ffnv_f32x4_t> output(rows * cols / FFNV_LANES);
    std::vector<ffnv_q8x32_t> unused_q(rows * cols / FFNV_QK);
    std::vector<ap_uint<16> > unused_scale(unused_q.size());
    const int status = ffn_vector(left_words.data(), right_words.data(),
            bias_words.data(), bias_words.data(), unused_q.data(),
            unused_scale.data(), output.data(), rows, cols, 1.0e-5f,
            FFNV_RESIDUAL_ADD, FFNV_PARAM0_PRESENT);
    if (status != FFNV_OK) return false;
    std::vector<float> expected(rows * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int index = row * cols + col;
            expected[index] = left[index] + right[index] + bias[col];
        }
    }
    return check_error("residual_add", unpack(output, expected.size()),
                       expected, 0.0f, 0.0f);
}

bool run_zero_case() {
    const int rows = 1;
    const int cols = 32;
    std::vector<float> zeros(cols, 0.0f);
    std::vector<float> ones(cols, 1.0f);
    const auto src = pack(zeros);
    const auto gamma = pack(ones);
    std::vector<ffnv_q8x32_t> q(1);
    std::vector<ap_uint<16> > scale(1);
    std::vector<ffnv_f32x4_t> unused(cols / FFNV_LANES);
    const int status = ffn_vector(src.data(), src.data(), gamma.data(),
            gamma.data(), q.data(), scale.data(), unused.data(), rows, cols,
            1.0e-5f, FFNV_RMS_NORM_QUANT, FFNV_PARAM0_PRESENT);
    const bool pass = status == FFNV_OK && q[0] == 0 && scale[0] == 0;
    std::printf("FFNV_CASE name=zero_variance status=%d result=%s\n",
                status, pass ? "PASS" : "FAIL");
    return pass;
}

bool run_negative_cases() {
    std::vector<ffnv_f32x4_t> f32(16);
    std::vector<ffnv_q8x32_t> q(2);
    std::vector<ap_uint<16> > scale(2);
    int pass = 0;
    pass += ffn_vector(f32.data(), f32.data(), f32.data(), f32.data(),
            q.data(), scale.data(), f32.data(), 1, 31, 1.0e-5f,
            FFNV_GELU_QUANT, 0) == FFNV_ERR_DIMENSION;
    pass += ffn_vector(f32.data(), f32.data(), f32.data(), f32.data(),
            q.data(), scale.data(), f32.data(), 1, 32, 1.0e-5f,
            99, 0) == FFNV_ERR_MODE;
    pass += ffn_vector(f32.data(), f32.data(), f32.data(), f32.data(),
            q.data(), scale.data(), f32.data(), 1, 32, 1.0e-5f,
            FFNV_LAYER_NORM_QUANT, FFNV_PARAM0_PRESENT) == FFNV_ERR_PARAMETER;
    const bool ok = pass == 3;
    std::printf("FFNV_CASE name=negative_contract checks=%d result=%s\n",
                pass, ok ? "PASS" : "FAIL");
    return ok;
}

#pragma pack(push, 1)
struct RealWeightHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t rows;
    std::uint32_t hidden;
    std::uint32_t expanded;
    std::uint32_t outputs;
    std::uint32_t token_count;
    float epsilon;
};
#pragma pack(pop)

bool read_exact(FILE *file, void *destination, std::size_t bytes) {
    return std::fread(destination, 1, bytes, file) == bytes;
}

bool read_q8_matrix(FILE *file, int outputs, int inputs, Q8Matrix &matrix) {
    if (inputs <= 0 || (inputs % FFNV_QK) != 0) return false;
    matrix.outputs = outputs;
    matrix.inputs = inputs;
    matrix.values.resize(static_cast<std::size_t>(outputs) * inputs);
    matrix.scales.resize(static_cast<std::size_t>(outputs) * inputs / FFNV_QK);
    const int blocks = inputs / FFNV_QK;
    for (int output = 0; output < outputs; ++output) {
        for (int block = 0; block < blocks; ++block) {
            std::uint16_t scale_bits = 0;
            std::int8_t payload[FFNV_QK];
            if (!read_exact(file, &scale_bits, sizeof(scale_bits)) ||
                    !read_exact(file, payload, sizeof(payload))) {
                return false;
            }
            matrix.scales[output * blocks + block] =
                    half_bits_to_float(ap_uint<16>(scale_bits));
            for (int item = 0; item < FFNV_QK; ++item) {
                matrix.values[output * inputs + block * FFNV_QK + item] =
                        payload[item];
            }
        }
    }
    return true;
}

bool run_real_weight_slice_case(const char *path) {
    FILE *file = std::fopen(path, "rb");
    if (!file) {
        std::printf("FFNV_CASE name=real_weight_proxy_slice error=open result=FAIL\n");
        return false;
    }
    RealWeightHeader header = {};
    const char expected_magic[8] = {'F', 'F', 'N', 'V', 'R', 'W', '0', '1'};
    bool valid = read_exact(file, &header, sizeof(header));
    valid = valid && std::equal(expected_magic, expected_magic + 8, header.magic);
    valid = valid && header.version == 1 && header.rows > 0 &&
            header.rows <= FFNV_MAX_ROWS && header.hidden > 0 &&
            header.hidden <= FFNV_MAX_COLS && header.expanded > 0 &&
            header.expanded <= FFNV_MAX_COLS && header.outputs > 0 &&
            header.outputs <= header.hidden && header.token_count == header.rows &&
            (header.hidden % FFNV_QK) == 0 &&
            (header.expanded % FFNV_QK) == 0 &&
            (header.outputs % FFNV_LANES) == 0 && header.epsilon > 0.0f;
    if (!valid) {
        std::fclose(file);
        std::printf("FFNV_CASE name=real_weight_proxy_slice error=header result=FAIL\n");
        return false;
    }
    const int rows = static_cast<int>(header.rows);
    const int hidden = static_cast<int>(header.hidden);
    const int expanded = static_cast<int>(header.expanded);
    const int outputs = static_cast<int>(header.outputs);
    std::vector<std::uint32_t> token_ids(header.token_count);
    std::vector<float> input(static_cast<std::size_t>(rows) * hidden);
    std::vector<float> gamma(hidden);
    valid = read_exact(file, token_ids.data(), token_ids.size() * sizeof(token_ids[0])) &&
            read_exact(file, input.data(), input.size() * sizeof(input[0])) &&
            read_exact(file, gamma.data(), gamma.size() * sizeof(gamma[0]));
    Q8Matrix gate;
    Q8Matrix up;
    Q8Matrix down;
    valid = valid && read_q8_matrix(file, expanded, hidden, gate) &&
            read_q8_matrix(file, expanded, hidden, up) &&
            read_q8_matrix(file, outputs, expanded, down);
    std::vector<float> reference(static_cast<std::size_t>(rows) * outputs);
    valid = valid && read_exact(file, reference.data(), reference.size() * sizeof(reference[0]));
    const int trailing = std::fgetc(file);
    std::fclose(file);
    if (!valid || trailing != EOF) {
        std::printf("FFNV_CASE name=real_weight_proxy_slice error=payload result=FAIL\n");
        return false;
    }

    const auto input_words = pack(input);
    const auto gamma_words = pack(gamma);
    std::vector<float> zeros(std::max(hidden, expanded), 0.0f);
    const auto zero_words = pack(zeros);
    std::vector<ffnv_q8x32_t> norm_q(static_cast<std::size_t>(rows) * hidden / FFNV_QK);
    std::vector<ap_uint<16> > norm_scales(norm_q.size());
    std::vector<ffnv_f32x4_t> unused_f32(static_cast<std::size_t>(rows) * hidden / FFNV_LANES);
    int status = ffn_vector(input_words.data(), input_words.data(),
            gamma_words.data(), zero_words.data(), norm_q.data(),
            norm_scales.data(), unused_f32.data(), rows, hidden,
            header.epsilon, FFNV_RMS_NORM_QUANT, FFNV_PARAM0_PRESENT);
    if (status != FFNV_OK) return false;

    const std::vector<float> gate_output = q8_linear(norm_q, norm_scales, rows, gate);
    const std::vector<float> up_output = q8_linear(norm_q, norm_scales, rows, up);
    const auto gate_words = pack(gate_output);
    const auto up_words = pack(up_output);
    std::vector<ffnv_q8x32_t> activation_q(
            static_cast<std::size_t>(rows) * expanded / FFNV_QK);
    std::vector<ap_uint<16> > activation_scales(activation_q.size());
    status = ffn_vector(gate_words.data(), up_words.data(), zero_words.data(),
            zero_words.data(), activation_q.data(), activation_scales.data(),
            unused_f32.data(), rows, expanded, header.epsilon,
            FFNV_SILU_MUL_QUANT, 0);
    if (status != FFNV_OK) return false;

    const std::vector<float> down_output = q8_linear(
            activation_q, activation_scales, rows, down);
    std::vector<float> residual(static_cast<std::size_t>(rows) * outputs);
    for (int row = 0; row < rows; ++row) {
        std::copy(input.begin() + static_cast<std::size_t>(row) * hidden,
                  input.begin() + static_cast<std::size_t>(row) * hidden + outputs,
                  residual.begin() + static_cast<std::size_t>(row) * outputs);
    }
    const auto residual_words = pack(residual);
    const auto down_words = pack(down_output);
    std::vector<ffnv_f32x4_t> output_words(
            static_cast<std::size_t>(rows) * outputs / FFNV_LANES);
    std::vector<ffnv_q8x32_t> unused_q(
            static_cast<std::size_t>(rows) * outputs / FFNV_QK);
    std::vector<ap_uint<16> > unused_scales(unused_q.size());
    status = ffn_vector(residual_words.data(), down_words.data(),
            zero_words.data(), zero_words.data(), unused_q.data(),
            unused_scales.data(), output_words.data(), rows, outputs,
            header.epsilon, FFNV_RESIDUAL_ADD, 0);
    if (status != FFNV_OK) return false;
    std::printf("FFNV_REAL_WEIGHT_METADATA rows=%d hidden=%d expanded=%d outputs=%d "
                "tokens=%u,%u,%u,%u\n", rows, hidden, expanded, outputs,
                token_ids.size() > 0 ? token_ids[0] : 0,
                token_ids.size() > 1 ? token_ids[1] : 0,
                token_ids.size() > 2 ? token_ids[2] : 0,
                token_ids.size() > 3 ? token_ids[3] : 0);
    return check_error("real_weight_proxy_slice",
                       unpack(output_words, reference.size()), reference,
                       0.003f, 0.001f);
}

}  // namespace

int main() {
    int passed = 0;
    passed += run_norm_case(FFNV_RMS_NORM_QUANT);
    passed += run_norm_case(FFNV_LAYER_NORM_QUANT);
    passed += run_activation_case(FFNV_SILU_MUL_QUANT);
    passed += run_activation_case(FFNV_GELU_QUANT);
    passed += run_residual_case();
    passed += run_ffn_composition(false);
    passed += run_ffn_composition(true);
    passed += run_zero_case();
    passed += run_negative_cases();
    int total = 9;
    const char *real_case = std::getenv("FFNV_REAL_CASE");
    if (real_case && real_case[0]) {
        ++total;
        passed += run_real_weight_slice_case(real_case);
    }
    std::printf("FFNV_TEST_SUMMARY passed=%d total=%d result=%s\n",
                passed, total, passed == total ? "PASS" : "FAIL");
    return passed == total ? 0 : 1;
}
