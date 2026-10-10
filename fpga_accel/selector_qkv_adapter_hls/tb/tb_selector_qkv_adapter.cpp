// C++ reference / testbench for the Selector -> LayerNorm-Q8_0 adapter and the
// independent QKV bias-add kernel.
//
// This file is host code. It is compiled both by system g++ (normal and
// ASan/UBSan) and by the Vitis HLS C-sim compiler. It contains:
//   * an independent direct dequant -> FP32 LayerNorm reference,
//   * a fused-domain (q = int8) reference of the kernel identity,
//   * deterministic synthetic cases for K=48/64/96 (M=192/256/384),
//   * extreme / constant / -128 / invalid-parameter / M-boundary cases
//     (the adapter accepts exactly M = 192/256/384; all other M are rejected),
//   * a frozen-fixture reader that byte-compares the kernel output with the
//     Python golden fixture when SQA_ADAPTER_FIXTURE / SQA_BIAS_FIXTURE are set.
//
// It never loads a model, uses a GPU, a board, synthesis, cosim or the network.

#include "selector_qkv_adapter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

const char *const ADAPTER_FIXTURE_ENV = "SQA_ADAPTER_FIXTURE";
const char *const BIAS_FIXTURE_ENV = "SQA_BIAS_FIXTURE";

std::uint32_t float_bits(float value) {
    union { std::uint32_t u; float f; } converter;
    converter.f = value;
    return converter.u;
}

float bits_float(std::uint32_t bits) {
    union { std::uint32_t u; float f; } converter;
    converter.u = bits;
    return converter.f;
}

float half_bits_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (bits >> 15) & 1u;
    const std::uint32_t exponent = (bits >> 10) & 0x1fu;
    const std::uint32_t mantissa = bits & 0x3ffu;
    double magnitude;
    if (exponent == 0u) {
        magnitude = std::ldexp(static_cast<double>(mantissa), -24);
    } else if (exponent == 31u) {
        magnitude = mantissa ? std::nan("")
                             : std::numeric_limits<double>::infinity();
    } else {
        magnitude = std::ldexp(static_cast<double>(1024u + mantissa),
                               static_cast<int>(exponent) - 25);
    }
    return static_cast<float>(sign ? -magnitude : magnitude);
}

sqa_token_word_t pack_token_word(const std::int8_t *values) {
    sqa_token_word_t word = 0;
    for (int lane = 0; lane < SQA_INT8_LANES; ++lane) {
        const ap_int<8> value = values[lane];
        word.range(8 * lane + 7, 8 * lane) = value.range(7, 0);
    }
    return word;
}

std::vector<sqa_token_word_t> pack_tokens(const std::vector<std::int8_t> &q,
                                          int rows) {
    std::vector<sqa_token_word_t> words(
            static_cast<std::size_t>(rows) * SQA_WORDS_PER_ROW);
    for (int row = 0; row < rows; ++row) {
        for (int word = 0; word < SQA_WORDS_PER_ROW; ++word) {
            words[row * SQA_WORDS_PER_ROW + word] = pack_token_word(
                    &q[static_cast<std::size_t>(row) * SQA_HIDDEN +
                       word * SQA_INT8_LANES]);
        }
    }
    return words;
}

sqa_f32x4_t pack_f32x4(const float *values) {
    sqa_f32x4_t word = 0;
    for (int lane = 0; lane < SQA_F32_LANES; ++lane) {
        word.range(32 * lane + 31, 32 * lane) = float_bits(values[lane]);
    }
    return word;
}

void unpack_f32x4(const sqa_f32x4_t &word, float *values) {
    for (int lane = 0; lane < SQA_F32_LANES; ++lane) {
        values[lane] = bits_float(word.range(32 * lane + 31, 32 * lane).to_uint());
    }
}

std::vector<sqa_f32x4_t> pack_f32_matrix(const std::vector<float> &values,
                                         int rows, int cols) {
    const int words = cols / SQA_F32_LANES;
    std::vector<sqa_f32x4_t> packed(static_cast<std::size_t>(rows) * words);
    for (int row = 0; row < rows; ++row) {
        for (int word = 0; word < words; ++word) {
            packed[static_cast<std::size_t>(row) * words + word] = pack_f32x4(
                    &values[static_cast<std::size_t>(row) * cols +
                            word * SQA_F32_LANES]);
        }
    }
    return packed;
}

std::vector<float> unpack_f32_matrix(const std::vector<sqa_f32x4_t> &packed,
                                     int rows, int cols) {
    const int words = cols / SQA_F32_LANES;
    std::vector<float> values(static_cast<std::size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int word = 0; word < words; ++word) {
            unpack_f32x4(packed[static_cast<std::size_t>(row) * words + word],
                         &values[static_cast<std::size_t>(row) * cols +
                                 word * SQA_F32_LANES]);
        }
    }
    return values;
}

std::vector<float> dequantize_q8(const std::vector<sqa_q8x32_t> &payload,
                                 const std::vector<ap_uint<16> > &scales,
                                 int rows) {
    std::vector<float> values(static_cast<std::size_t>(rows) * SQA_HIDDEN);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < SQA_BLOCKS_PER_ROW; ++block) {
            const int index = row * SQA_BLOCKS_PER_ROW + block;
            const float d = half_bits_to_float(
                    static_cast<std::uint16_t>(scales[index].to_uint()));
            for (int element = 0; element < SQA_BLOCK; ++element) {
                const ap_int<8> qv = payload[index].range(8 * element + 7,
                                                          8 * element);
                values[static_cast<std::size_t>(row) * SQA_HIDDEN +
                       block * SQA_BLOCK + element] =
                        static_cast<float>(qv.to_int()) * d;
            }
        }
    }
    return values;
}

// Independent direct dequant -> FP32 LayerNorm (population variance).
std::vector<float> direct_layernorm_fp32(const std::vector<std::int8_t> &q,
                                         float scale,
                                         const std::vector<float> &gamma,
                                         const std::vector<float> &beta) {
    const int rows = static_cast<int>(q.size()) / SQA_HIDDEN;
    std::vector<float> output(q.size());
    for (int row = 0; row < rows; ++row) {
        float mean = 0.0f;
        float mean_square = 0.0f;
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            const float x = static_cast<float>(q[row * SQA_HIDDEN + col]) / scale;
            mean += x;
            mean_square += x * x;
        }
        mean /= static_cast<float>(SQA_HIDDEN);
        mean_square /= static_cast<float>(SQA_HIDDEN);
        float variance = mean_square - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        const float inverse = 1.0f / std::sqrt(variance + SQA_EPSILON);
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            const float x = static_cast<float>(q[row * SQA_HIDDEN + col]) / scale;
            output[row * SQA_HIDDEN + col] =
                    (x - mean) * inverse * gamma[col] + beta[col];
        }
    }
    return output;
}

// Fused-domain reference of the algebraic identity implemented by the kernel.
std::vector<float> fused_layernorm_reference(const std::vector<std::int8_t> &q,
                                             float scale,
                                             const std::vector<float> &gamma,
                                             const std::vector<float> &beta) {
    const int rows = static_cast<int>(q.size()) / SQA_HIDDEN;
    const float eps_sq = SQA_EPSILON * scale * scale;
    std::vector<float> output(q.size());
    for (int row = 0; row < rows; ++row) {
        int sum_q = 0;
        int sum_sq = 0;
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            const int value = q[row * SQA_HIDDEN + col];
            sum_q += value;
            sum_sq += value * value;
        }
        const float mean = static_cast<float>(sum_q) /
                static_cast<float>(SQA_HIDDEN);
        const float mean_square = static_cast<float>(sum_sq) /
                static_cast<float>(SQA_HIDDEN);
        float variance = mean_square - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        const float denominator = variance + eps_sq;
        const float inverse = (denominator > 0.0f)
                ? 1.0f / std::sqrt(denominator) : 0.0f;
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            const float centered = static_cast<float>(q[row * SQA_HIDDEN + col]) -
                    mean;
            output[row * SQA_HIDDEN + col] =
                    (centered * inverse) * gamma[col] + beta[col];
        }
    }
    return output;
}

struct Metrics {
    double max_abs;
    double rmse;
    double relative_l2;
    double cosine;
};

Metrics compare(const std::vector<float> &actual,
                const std::vector<float> &reference) {
    Metrics metrics = {0.0, 0.0, 0.0, 0.0};
    double square_error = 0.0;
    double square_reference = 0.0;
    double dot = 0.0;
    double square_actual = 0.0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double a = actual[index];
        const double b = reference[index];
        const double error = std::fabs(a - b);
        metrics.max_abs = std::max(metrics.max_abs, error);
        square_error += error * error;
        square_reference += b * b;
        square_actual += a * a;
        dot += a * b;
    }
    const double count = static_cast<double>(reference.size());
    metrics.rmse = std::sqrt(square_error / count);
    metrics.relative_l2 = square_reference > 0.0
            ? std::sqrt(square_error / square_reference) : 0.0;
    const double norm = std::sqrt(square_actual) * std::sqrt(square_reference);
    metrics.cosine = norm > 0.0 ? dot / norm : 0.0;
    return metrics;
}

std::vector<std::int8_t> make_synthetic_q(int rows, int seed) {
    std::vector<std::int8_t> q(static_cast<std::size_t>(rows) * SQA_HIDDEN);
    std::uint32_t state = static_cast<std::uint32_t>(seed) | 1u;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            state = state * 1664525u + 1013904223u;
            const int noise = static_cast<int>((state >> 16) & 0xffu) - 128;
            const int low_frequency = ((col / 64 + row) % 5) - 2;
            int value = noise / 4 + low_frequency * 9;
            if (value > SQA_SELECTOR_MAX) value = SQA_SELECTOR_MAX;
            if (value < SQA_SELECTOR_MIN) value = SQA_SELECTOR_MIN;
            q[static_cast<std::size_t>(row) * SQA_HIDDEN + col] =
                    static_cast<std::int8_t>(value);
        }
    }
    return q;
}

std::vector<float> make_gamma() {
    std::vector<float> gamma(SQA_HIDDEN);
    for (int col = 0; col < SQA_HIDDEN; ++col) {
        gamma[col] = 0.75f + 0.00390625f * static_cast<float>(col % 17);
    }
    return gamma;
}

std::vector<float> make_beta() {
    std::vector<float> beta(SQA_HIDDEN);
    for (int col = 0; col < SQA_HIDDEN; ++col) {
        beta[col] = 0.00390625f * static_cast<float>((col % 7) - 3);
    }
    return beta;
}

bool read_file(const char *path, std::vector<std::uint8_t> &bytes) {
    FILE *file = std::fopen(path, "rb");
    if (!file) return false;
    if (std::fseek(file, 0, SEEK_END) != 0) { std::fclose(file); return false; }
    const long size = std::ftell(file);
    if (size < 0) { std::fclose(file); return false; }
    std::rewind(file);
    bytes.resize(static_cast<std::size_t>(size));
    const std::size_t read = bytes.empty()
            ? 0 : std::fread(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    return read == bytes.size();
}

struct ByteCursor {
    const std::uint8_t *data;
    std::size_t size;
    std::size_t offset;

    bool take(void *destination, std::size_t count) {
        if (offset + count > size) return false;
        std::memcpy(destination, data + offset, count);
        offset += count;
        return true;
    }
};

bool run_adapter_fixture(const char *path) {
    std::vector<std::uint8_t> bytes;
    if (!read_file(path, bytes)) {
        std::printf("SQA_FIXTURE name=adapter error=open result=FAIL\n");
        return false;
    }
    ByteCursor cursor = {bytes.data(), bytes.size(), 0};
    char magic[8] = {0};
    std::uint32_t version = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t block = 0;
    std::uint32_t blocks = 0;
    float scale = 0.0f;
    float epsilon = 0.0f;
    std::uint32_t reserved[2] = {0, 0};
    bool valid = cursor.take(magic, 8) &&
            cursor.take(&version, 4) && cursor.take(&rows, 4) &&
            cursor.take(&cols, 4) && cursor.take(&block, 4) &&
            cursor.take(&blocks, 4) && cursor.take(&scale, 4) &&
            cursor.take(&epsilon, 4) && cursor.take(reserved, 8);
    const char expected_magic[8] = {'S', 'Q', 'A', 'L', 'N', 'Q', '0', '1'};
    valid = valid && std::equal(expected_magic, expected_magic + 8, magic) &&
            version == 1 && sqa_is_legal_adapter_rows(static_cast<int>(rows)) &&
            cols == static_cast<std::uint32_t>(SQA_HIDDEN) &&
            block == static_cast<std::uint32_t>(SQA_BLOCK) &&
            blocks == static_cast<std::uint32_t>(SQA_BLOCKS_PER_ROW) &&
            epsilon > 0.0f;
    const int row_count = static_cast<int>(rows);
    std::vector<float> gamma(cols);
    std::vector<float> beta(cols);
    std::vector<std::uint8_t> input_bytes(
            static_cast<std::size_t>(row_count) * cols);
    std::vector<std::uint8_t> expected_payload(
            static_cast<std::size_t>(row_count) * blocks * block);
    std::vector<std::uint8_t> expected_scale_bits(
            static_cast<std::size_t>(row_count) * blocks * 2);
    valid = valid && cursor.take(gamma.data(), gamma.size() * 4) &&
            cursor.take(beta.data(), beta.size() * 4) &&
            cursor.take(input_bytes.data(), input_bytes.size()) &&
            cursor.take(expected_payload.data(), expected_payload.size()) &&
            cursor.take(expected_scale_bits.data(), expected_scale_bits.size()) &&
            cursor.offset == cursor.size;
    if (!valid) {
        std::printf("SQA_FIXTURE name=adapter error=format result=FAIL\n");
        return false;
    }

    std::vector<sqa_token_word_t> tokens(
            static_cast<std::size_t>(row_count) * SQA_WORDS_PER_ROW);
    for (std::size_t word = 0; word < tokens.size(); ++word) {
        sqa_token_word_t packed = 0;
        for (int lane = 0; lane < SQA_INT8_LANES; ++lane) {
            const ap_int<8> value = static_cast<std::int8_t>(
                    input_bytes[word * SQA_INT8_LANES + lane]);
            packed.range(8 * lane + 7, 8 * lane) = value.range(7, 0);
        }
        tokens[word] = packed;
    }
    std::vector<sqa_f32x4_t> gamma_words(SQA_PARAM_WORDS);
    std::vector<sqa_f32x4_t> beta_words(SQA_PARAM_WORDS);
    for (int word = 0; word < SQA_PARAM_WORDS; ++word) {
        gamma_words[word] = pack_f32x4(&gamma[word * SQA_F32_LANES]);
        beta_words[word] = pack_f32x4(&beta[word * SQA_F32_LANES]);
    }

    std::vector<sqa_q8x32_t> payload(
            static_cast<std::size_t>(row_count) * SQA_BLOCKS_PER_ROW);
    std::vector<ap_uint<16> > scales(payload.size());
    const int status = selector_ln_q8_adapter(tokens.data(), gamma_words.data(),
                                              beta_words.data(), payload.data(),
                                              scales.data(), scale, row_count);
    std::size_t payload_mismatch = 0;
    for (std::size_t index = 0; index < payload.size(); ++index) {
        for (int element = 0; element < SQA_BLOCK; ++element) {
            const std::uint8_t value = static_cast<std::uint8_t>(
                    payload[index].range(8 * element + 7, 8 * element).to_uint());
            if (value != expected_payload[index * SQA_BLOCK + element]) {
                ++payload_mismatch;
            }
        }
    }
    std::size_t scale_mismatch = 0;
    for (std::size_t index = 0; index < scales.size(); ++index) {
        const std::uint16_t value = static_cast<std::uint16_t>(
                scales[index].to_uint() & 0xffffu);
        const std::uint16_t expected = static_cast<std::uint16_t>(
                expected_scale_bits[2 * index] |
                (static_cast<std::uint16_t>(expected_scale_bits[2 * index + 1]) << 8));
        if (value != expected) ++scale_mismatch;
    }

    // Metrics vs the direct dequant -> FP32 LayerNorm chain.
    std::vector<std::int8_t> q_int8(input_bytes.size());
    for (std::size_t index = 0; index < input_bytes.size(); ++index) {
        q_int8[index] = static_cast<std::int8_t>(input_bytes[index]);
    }
    const std::vector<float> dequantized = dequantize_q8(payload, scales, row_count);
    const std::vector<float> direct = direct_layernorm_fp32(q_int8, scale, gamma, beta);
    const std::vector<float> fused = fused_layernorm_reference(q_int8, scale, gamma, beta);
    const Metrics quant_vs_direct = compare(dequantized, direct);
    const Metrics fused_vs_direct = compare(fused, direct);
    const bool ok = status == SQA_OK && payload_mismatch == 0 && scale_mismatch == 0;
    std::printf("SQA_FIXTURE name=adapter_real_block1 rows=%d cols=%u scale=%.9g "
                "payload_byte_mismatch=%zu scale_byte_mismatch=%zu "
                "fused_vs_direct[max_abs=%.9g relative_l2=%.9g cosine=%.9g] "
                "quant_vs_direct[max_abs=%.9g rmse=%.9g relative_l2=%.9g cosine=%.9g] "
                "result=%s\n",
                row_count, cols, static_cast<double>(scale),
                payload_mismatch, scale_mismatch,
                fused_vs_direct.max_abs, fused_vs_direct.relative_l2,
                fused_vs_direct.cosine, quant_vs_direct.max_abs,
                quant_vs_direct.rmse, quant_vs_direct.relative_l2,
                quant_vs_direct.cosine, ok ? "PASS" : "FAIL");
    return ok;
}

bool run_bias_fixture(const char *path) {
    std::vector<std::uint8_t> bytes;
    if (!read_file(path, bytes)) {
        std::printf("SQA_FIXTURE name=qkv_bias_add error=open result=FAIL\n");
        return false;
    }
    ByteCursor cursor = {bytes.data(), bytes.size(), 0};
    char magic[8] = {0};
    std::uint32_t version = 0;
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::uint32_t reserved[2] = {0, 0};
    bool valid = cursor.take(magic, 8) && cursor.take(&version, 4) &&
            cursor.take(&rows, 4) && cursor.take(&cols, 4) &&
            cursor.take(reserved, 8);
    const char expected_magic[8] = {'S', 'Q', 'A', 'B', 'I', 'A', '0', '1'};
    valid = valid && std::equal(expected_magic, expected_magic + 8, magic) &&
            version == 1 && rows > 0 &&
            rows <= static_cast<std::uint32_t>(SQA_BIAS_MAX_ROWS) &&
            cols > 0 && cols <= static_cast<std::uint32_t>(SQA_MAX_COLS) &&
            (cols % SQA_F32_LANES) == 0;
    const int row_count = static_cast<int>(rows);
    const int col_count = static_cast<int>(cols);
    const std::size_t elements = static_cast<std::size_t>(row_count) * col_count;
    std::vector<float> input(elements);
    std::vector<float> bias(col_count);
    std::vector<std::uint8_t> expected(elements * 4);
    valid = valid && cursor.take(input.data(), input.size() * 4) &&
            cursor.take(bias.data(), bias.size() * 4) &&
            cursor.take(expected.data(), expected.size()) &&
            cursor.offset == cursor.size;
    if (!valid) {
        std::printf("SQA_FIXTURE name=qkv_bias_add error=format result=FAIL\n");
        return false;
    }
    const std::vector<sqa_f32x4_t> input_words = pack_f32_matrix(input, row_count,
                                                                 col_count);
    const std::vector<sqa_f32x4_t> bias_words = pack_f32_matrix(bias, 1, col_count);
    std::vector<sqa_f32x4_t> output_words(
            static_cast<std::size_t>(row_count) * (col_count / SQA_F32_LANES));
    const int status = qkv_bias_add(input_words.data(), bias_words.data(),
                                    output_words.data(), row_count, col_count);
    std::size_t byte_mismatch = 0;
    for (int row = 0; row < row_count; ++row) {
        for (int word = 0; word < col_count / SQA_F32_LANES; ++word) {
            float values[SQA_F32_LANES];
            unpack_f32x4(output_words[static_cast<std::size_t>(row) *
                                     (col_count / SQA_F32_LANES) + word], values);
            for (int lane = 0; lane < SQA_F32_LANES; ++lane) {
                const std::size_t element =
                        (static_cast<std::size_t>(row) * col_count +
                         word * SQA_F32_LANES + lane) * 4;
                std::uint32_t expected_bits = 0;
                std::memcpy(&expected_bits, &expected[element], 4);
                if (float_bits(values[lane]) != expected_bits) ++byte_mismatch;
            }
        }
    }
    const bool ok = status == SQA_OK && byte_mismatch == 0;
    std::printf("SQA_FIXTURE name=qkv_bias_add_real rows=%d cols=%d "
                "byte_mismatch=%zu result=%s\n",
                row_count, col_count, byte_mismatch, ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace

int main() {
    const std::vector<float> gamma = make_gamma();
    const std::vector<float> beta = make_beta();
    std::vector<sqa_f32x4_t> gamma_words(SQA_PARAM_WORDS);
    std::vector<sqa_f32x4_t> beta_words(SQA_PARAM_WORDS);
    for (int word = 0; word < SQA_PARAM_WORDS; ++word) {
        gamma_words[word] = pack_f32x4(&gamma[word * SQA_F32_LANES]);
        beta_words[word] = pack_f32x4(&beta[word * SQA_F32_LANES]);
    }

    int passed = 0;
    int total = 0;

    struct SyntheticCase {
        const char *case_id;
        int rows;
        int seed;
        float scale;
    };
    const SyntheticCase synthetic_cases[] = {
        {"k48_m192_seed20260928", 192, 20260928, 32.0f},
        {"k48_m192_seed20260929", 192, 20260929, 31.0f},
        {"k64_m256_seed20260930", 256, 20260930, 33.5f},
        {"k64_m256_seed20261002", 256, 20261002, 29.5f},
        {"k96_m384_seed20261001", 384, 20261001, 30.25f},
        {"k96_m384_seed20261003", 384, 20261003, 31.75f},
    };
    for (const SyntheticCase &test : synthetic_cases) {
        ++total;
        const std::vector<std::int8_t> q = make_synthetic_q(test.rows, test.seed);
        const std::vector<sqa_token_word_t> tokens = pack_tokens(q, test.rows);
        std::vector<sqa_q8x32_t> payload(
                static_cast<std::size_t>(test.rows) * SQA_BLOCKS_PER_ROW);
        std::vector<ap_uint<16> > scales(payload.size());
        const int status = selector_ln_q8_adapter(
                tokens.data(), gamma_words.data(), beta_words.data(),
                payload.data(), scales.data(), test.scale, test.rows);
        const std::vector<float> dequantized = dequantize_q8(payload, scales,
                                                             test.rows);
        const std::vector<float> direct = direct_layernorm_fp32(q, test.scale,
                                                                gamma, beta);
        const std::vector<float> fused = fused_layernorm_reference(q, test.scale,
                                                                   gamma, beta);
        const Metrics fused_vs_direct = compare(fused, direct);
        const Metrics quant_vs_fused = compare(dequantized, fused);
        const Metrics quant_vs_direct = compare(dequantized, direct);
        const bool ok = status == SQA_OK &&
                fused_vs_direct.relative_l2 <= 1.0e-4 &&
                fused_vs_direct.cosine >= 0.99999 &&
                quant_vs_fused.relative_l2 <= 0.02;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=%s rows=%d status=%d "
                    "fused_vs_direct[max_abs=%.9g rmse=%.9g relative_l2=%.9g cosine=%.9g] "
                    "quant_vs_fused[max_abs=%.9g rmse=%.9g relative_l2=%.9g cosine=%.9g] "
                    "quant_vs_direct[max_abs=%.9g relative_l2=%.9g cosine=%.9g] result=%s\n",
                    test.case_id, test.rows, status,
                    fused_vs_direct.max_abs, fused_vs_direct.rmse,
                    fused_vs_direct.relative_l2, fused_vs_direct.cosine,
                    quant_vs_fused.max_abs, quant_vs_fused.rmse,
                    quant_vs_fused.relative_l2, quant_vs_fused.cosine,
                    quant_vs_direct.max_abs, quant_vs_direct.relative_l2,
                    quant_vs_direct.cosine, ok ? "PASS" : "FAIL");
    }

    // Constant rows: LN must reduce to beta (centered value is exactly zero).
    {
        const int constant_rows = SQA_LEGAL_ROWS_0;  // 192
        const char *names[] = {"constant_row_q7", "constant_row_qneg128"};
        const int values[] = {7, -128};
        for (int variant = 0; variant < 2; ++variant) {
            ++total;
            std::vector<std::int8_t> q(
                    static_cast<std::size_t>(constant_rows) * SQA_HIDDEN,
                    static_cast<std::int8_t>(values[variant]));
            const std::vector<sqa_token_word_t> tokens = pack_tokens(
                    q, constant_rows);
            std::vector<sqa_q8x32_t> payload(
                    static_cast<std::size_t>(constant_rows) * SQA_BLOCKS_PER_ROW);
            std::vector<ap_uint<16> > scales(payload.size());
            const int status = selector_ln_q8_adapter(
                    tokens.data(), gamma_words.data(), beta_words.data(),
                    payload.data(), scales.data(), 30.0f, constant_rows);
            const std::vector<float> dequantized = dequantize_q8(payload, scales,
                                                                 constant_rows);
            const std::vector<float> fused = fused_layernorm_reference(q, 30.0f,
                                                                       gamma, beta);
            const Metrics metrics = compare(dequantized, fused);
            bool exactly_beta = true;
            for (int row = 0; row < constant_rows && exactly_beta; ++row) {
                for (int col = 0; col < SQA_HIDDEN; ++col) {
                    if (fused[row * SQA_HIDDEN + col] != beta[col]) {
                        exactly_beta = false;
                        break;
                    }
                }
            }
            const bool ok = status == SQA_OK && metrics.relative_l2 <= 0.02 &&
                    exactly_beta;
            passed += ok ? 1 : 0;
            std::printf("SQA_CASE case=%s rows=%d input_q=%d status=%d "
                        "dequant_vs_beta[max_abs=%.9g relative_l2=%.9g] "
                        "fused_exactly_beta=%s result=%s\n",
                        names[variant], constant_rows, values[variant], status,
                        metrics.max_abs, metrics.relative_l2,
                        exactly_beta ? "true" : "false", ok ? "PASS" : "FAIL");
        }
    }

    // Mixed -128 / 127 row must not be silently clamped before LayerNorm.
    {
        ++total;
        const int mixed_rows = SQA_LEGAL_ROWS_0;  // 192
        std::vector<std::int8_t> q(
                static_cast<std::size_t>(mixed_rows) * SQA_HIDDEN, 0);
        for (int col = 0; col < SQA_HIDDEN; ++col) {
            q[col] = (col % 2 == 0) ? static_cast<std::int8_t>(-128)
                                    : static_cast<std::int8_t>(127);
        }
        for (int row = 1; row < mixed_rows; ++row) {
            for (int col = 0; col < SQA_HIDDEN; ++col) {
                q[static_cast<std::size_t>(row) * SQA_HIDDEN + col] =
                        static_cast<std::int8_t>((col % 5) - 2);
            }
        }
        int min_seen = 0;
        int max_seen = 0;
        for (std::size_t index = 0; index < q.size(); ++index) {
            min_seen = std::min(min_seen, static_cast<int>(q[index]));
            max_seen = std::max(max_seen, static_cast<int>(q[index]));
        }
        const std::vector<sqa_token_word_t> tokens = pack_tokens(q, mixed_rows);
        std::vector<sqa_q8x32_t> payload(
                static_cast<std::size_t>(mixed_rows) * SQA_BLOCKS_PER_ROW);
        std::vector<ap_uint<16> > scales(payload.size());
        const int status = selector_ln_q8_adapter(
                tokens.data(), gamma_words.data(), beta_words.data(),
                payload.data(), scales.data(), 25.0f, mixed_rows);
        const std::vector<float> dequantized = dequantize_q8(payload, scales,
                                                             mixed_rows);
        const std::vector<float> fused = fused_layernorm_reference(q, 25.0f,
                                                                    gamma, beta);
        const Metrics metrics = compare(dequantized, fused);
        const bool ok = status == SQA_OK && min_seen == -128 && max_seen == 127 &&
                metrics.relative_l2 <= 0.02;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=mixed_minus128_max127 status=%d min=%d max=%d "
                    "relative_l2=%.9g result=%s\n",
                    status, min_seen, max_seen, metrics.relative_l2,
                    ok ? "PASS" : "FAIL");
    }

    // Extreme global scales must stay finite and status-clean.
    {
        const float extremes[] = {1.0e-30f, 1.0e30f};
        for (float scale : extremes) {
            ++total;
            const std::vector<std::int8_t> q = make_synthetic_q(192, 77);
            const std::vector<sqa_token_word_t> tokens = pack_tokens(q, 192);
            std::vector<sqa_q8x32_t> payload(192 * SQA_BLOCKS_PER_ROW);
            std::vector<ap_uint<16> > scales(payload.size());
            const int status = selector_ln_q8_adapter(
                    tokens.data(), gamma_words.data(), beta_words.data(),
                    payload.data(), scales.data(), scale, 192);
            const std::vector<float> dequantized = dequantize_q8(payload, scales,
                                                                 192);
            bool finite = true;
            for (std::size_t index = 0; index < dequantized.size(); ++index) {
                if (!std::isfinite(dequantized[index])) { finite = false; break; }
            }
            const bool ok = status == SQA_OK && finite;
            passed += ok ? 1 : 0;
            std::printf("SQA_CASE case=extreme_scale_%.3g status=%d finite=%s result=%s\n",
                        static_cast<double>(scale), status,
                        finite ? "true" : "false", ok ? "PASS" : "FAIL");
        }
    }

    // Invalid scales (zero, negative, NaN, +inf) must be rejected.
    {
        const float invalid[] = {0.0f, -1.0f,
                                 std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::infinity()};
        int rejected = 0;
        const std::vector<std::int8_t> q = make_synthetic_q(192, 5);
        const std::vector<sqa_token_word_t> tokens = pack_tokens(q, 192);
        std::vector<sqa_q8x32_t> payload(192 * SQA_BLOCKS_PER_ROW);
        std::vector<ap_uint<16> > scales(payload.size());
        for (float scale : invalid) {
            const int status = selector_ln_q8_adapter(
                    tokens.data(), gamma_words.data(), beta_words.data(),
                    payload.data(), scales.data(), scale, 192);
            if (status == SQA_ERR_SCALE) ++rejected;
        }
        ++total;
        const bool ok = rejected == 4;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=invalid_scales rejected=%d expected=4 result=%s\n",
                    rejected, ok ? "PASS" : "FAIL");
    }

    // M boundaries: only the Selector legal budgets M = 192/256/384 are
    // accepted. 0 / negative / 1 / 191 / 193 / 385 / 4320 / >384 are rejected.
    {
        int rejected_dim = 0;
        int accepted_legal = 0;
        const std::vector<std::int8_t> q = make_synthetic_q(1, 9);
        const std::vector<sqa_token_word_t> one = pack_tokens(q, 1);
        std::vector<sqa_q8x32_t> payload(SQA_LEGAL_ROWS_2 * SQA_BLOCKS_PER_ROW);
        std::vector<ap_uint<16> > scales(payload.size());
        const int bad_rows[] = {0, -1, 1, 191, 193, 385, 4320,
                                SQA_LEGAL_ROWS_2 + 1};
        for (int rows : bad_rows) {
            if (selector_ln_q8_adapter(one.data(), gamma_words.data(),
                                       beta_words.data(), payload.data(),
                                       scales.data(), 10.0f, rows) ==
                    SQA_ERR_DIMENSION) {
                ++rejected_dim;
            }
        }
        const int legal_rows[] = {SQA_LEGAL_ROWS_0, SQA_LEGAL_ROWS_1,
                                  SQA_LEGAL_ROWS_2};
        for (int rows : legal_rows) {
            const std::vector<std::int8_t> qr = make_synthetic_q(rows, 3);
            const std::vector<sqa_token_word_t> tokens = pack_tokens(qr, rows);
            if (selector_ln_q8_adapter(tokens.data(), gamma_words.data(),
                                       beta_words.data(), payload.data(),
                                       scales.data(), 10.0f, rows) == SQA_OK) {
                ++accepted_legal;
            }
        }
        ++total;
        const bool ok = rejected_dim == 8 && accepted_legal == 3;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=adapter_rows_contract rejected=%d expected=8 "
                    "legal_accepted=%d expected=3 result=%s\n",
                    rejected_dim, accepted_legal, ok ? "PASS" : "FAIL");
    }

    // Independent bias-add kernel element-wise check. The formal qkv boundary
    // this round is M = 192/256/384, N = 3072; the operator itself stays
    // general (row gate 1..4320, cols multiple of 4).
    {
        const int formal_rows[] = {SQA_LEGAL_ROWS_0, SQA_LEGAL_ROWS_1,
                                   SQA_LEGAL_ROWS_2};
        int cases_ok = 0;
        int formal_total = 0;
        for (int rows : formal_rows) {
            const int cols = SQA_QKV_COLS;
            ++formal_total;
            std::vector<float> input(static_cast<std::size_t>(rows) * cols);
            std::vector<float> bias(cols);
            for (int col = 0; col < cols; ++col) {
                bias[col] = 0.001953125f * static_cast<float>((col % 9) - 4);
            }
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    input[static_cast<std::size_t>(row) * cols + col] =
                            0.03125f * static_cast<float>(
                                    ((row * 31 + col * 17) % 33) - 16);
                }
            }
            const std::vector<sqa_f32x4_t> input_words = pack_f32_matrix(input,
                                                                        rows, cols);
            const std::vector<sqa_f32x4_t> bias_words = pack_f32_matrix(bias, 1,
                                                                       cols);
            std::vector<sqa_f32x4_t> output_words(
                    static_cast<std::size_t>(rows) * (cols / SQA_F32_LANES));
            const int status = qkv_bias_add(input_words.data(), bias_words.data(),
                                            output_words.data(), rows, cols);
            const std::vector<float> output = unpack_f32_matrix(output_words,
                                                               rows, cols);
            bool byte_exact = true;
            for (int row = 0; row < rows && byte_exact; ++row) {
                for (int col = 0; col < cols; ++col) {
                    const float expected =
                            input[static_cast<std::size_t>(row) * cols + col] +
                            bias[col];
                    if (float_bits(expected) != float_bits(
                            output[static_cast<std::size_t>(row) * cols + col])) {
                        byte_exact = false;
                        break;
                    }
                }
            }
            cases_ok += (status == SQA_OK && byte_exact) ? 1 : 0;
        }
        ++total;
        passed += (cases_ok == formal_total) ? 1 : 0;
        std::printf("SQA_CASE case=bias_add_formal_m192_256_384_n3072 "
                    "exact_cases=%d expected=%d result=%s\n",
                    cases_ok, formal_total,
                    cases_ok == formal_total ? "PASS" : "FAIL");
    }

    // General (non-formal) bias-add dimensions still work.
    {
        const int cols = 64;
        const int rows = 5;
        std::vector<float> input(static_cast<std::size_t>(rows) * cols);
        std::vector<float> bias(cols);
        for (int col = 0; col < cols; ++col) {
            bias[col] = 0.001953125f * static_cast<float>((col % 9) - 4);
        }
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                input[static_cast<std::size_t>(row) * cols + col] =
                        0.03125f * static_cast<float>(
                                ((row * 31 + col * 17) % 33) - 16);
            }
        }
        const std::vector<sqa_f32x4_t> input_words = pack_f32_matrix(input,
                                                                    rows, cols);
        const std::vector<sqa_f32x4_t> bias_words = pack_f32_matrix(bias, 1, cols);
        std::vector<sqa_f32x4_t> output_words(
                static_cast<std::size_t>(rows) * (cols / SQA_F32_LANES));
        const int status = qkv_bias_add(input_words.data(), bias_words.data(),
                                        output_words.data(), rows, cols);
        const std::vector<float> output = unpack_f32_matrix(output_words,
                                                            rows, cols);
        bool byte_exact = true;
        for (int row = 0; row < rows && byte_exact; ++row) {
            for (int col = 0; col < cols; ++col) {
                const float expected =
                        input[static_cast<std::size_t>(row) * cols + col] +
                        bias[col];
                if (float_bits(expected) != float_bits(
                        output[static_cast<std::size_t>(row) * cols + col])) {
                    byte_exact = false;
                    break;
                }
            }
        }
        ++total;
        const bool ok = status == SQA_OK && byte_exact;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=bias_add_general_m5_n64 result=%s\n",
                    ok ? "PASS" : "FAIL");
    }

    // Invalid dimensions for bias-add (general row gate stays 1..4320).
    {
        std::vector<sqa_f32x4_t> words(64);
        const int checks[] = {
            qkv_bias_add(words.data(), words.data(), words.data(), 0, SQA_QKV_COLS),
            qkv_bias_add(words.data(), words.data(), words.data(), 1, 0),
            qkv_bias_add(words.data(), words.data(), words.data(), 1,
                         SQA_QKV_COLS + 1),
            qkv_bias_add(words.data(), words.data(), words.data(),
                         SQA_BIAS_MAX_ROWS + 1, 4),
        };
        int rejected = 0;
        for (int status : checks) {
            if (status == SQA_ERR_DIMENSION) ++rejected;
        }
        ++total;
        const bool ok = rejected == 4;
        passed += ok ? 1 : 0;
        std::printf("SQA_CASE case=bias_add_invalid_dims rejected=%d expected=4 result=%s\n",
                    rejected, ok ? "PASS" : "FAIL");
    }

    // Frozen real-block1 fixture byte comparison (only when env vars are set).
    const char *adapter_fixture = std::getenv(ADAPTER_FIXTURE_ENV);
    if (adapter_fixture && adapter_fixture[0]) {
        ++total;
        passed += run_adapter_fixture(adapter_fixture) ? 1 : 0;
    }
    const char *bias_fixture = std::getenv(BIAS_FIXTURE_ENV);
    if (bias_fixture && bias_fixture[0]) {
        ++total;
        passed += run_bias_fixture(bias_fixture) ? 1 : 0;
    }

    std::printf("SQA_TEST_SUMMARY passed=%d total=%d result=%s\n",
                passed, total, passed == total ? "PASS" : "FAIL");
    return passed == total ? 0 : 1;
}
