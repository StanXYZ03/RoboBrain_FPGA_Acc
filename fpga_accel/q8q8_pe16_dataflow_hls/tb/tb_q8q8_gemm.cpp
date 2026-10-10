#include "q8q8_gemm.h"

#include <hls_half.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

ap_uint<16> float_to_half_bits(float value) {
    half h = value;
    return ap_uint<16>(h.get_bits());
}

float half_bits_to_float(ap_uint<16> bits) {
    half h;
    h.set_bits(static_cast<unsigned short>(bits.to_uint()));
    return static_cast<float>(h);
}

float output_lane_to_float(const q8q8_output_word_t &word, int lane) {
    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.u = word.range(32 * lane + 31, 32 * lane).to_uint();
    return converter.f;
}

// Exact scalar algorithm used by llama.cpp quantize_row_q8_0_ref: the int8
// values use the full-precision d, while the stored block scale is FP16.
void quantize_activation_row(
        const float * src,
        int k,
        ap_uint<256> * dst_q,
        ap_uint<16> * dst_scale) {
    const int k_blocks = k / Q8Q8_QK;
    for (int kb = 0; kb < k_blocks; ++kb) {
        float amax = 0.0f;
        for (int lane = 0; lane < Q8Q8_QK; ++lane) {
            amax = std::max(amax, std::fabs(src[kb * Q8Q8_QK + lane]));
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        dst_scale[kb] = float_to_half_bits(d);
        ap_uint<256> packed = 0;
        for (int lane = 0; lane < Q8Q8_QK; ++lane) {
            const int q = static_cast<int>(std::round(
                    src[kb * Q8Q8_QK + lane] * id));
            const ap_int<8> q8 = q;
            packed.range(8 * lane + 7, 8 * lane) = q8.range(7, 0);
        }
        dst_q[kb] = packed;
    }
}

void make_inputs(
        int m,
        int n,
        int k,
        std::vector<ap_uint<256> > & weights_q,
        std::vector<ap_uint<16> > & weights_scale,
        std::vector<ap_uint<256> > & activations_q,
        std::vector<ap_uint<16> > & activations_scale) {
    const int k_blocks = k / Q8Q8_QK;
    weights_q.resize(std::max(n * k_blocks, Q8Q8_TB_WEIGHT_DEPTH));
    weights_scale.resize(std::max(n * k_blocks, Q8Q8_TB_WEIGHT_DEPTH));
    activations_q.resize(std::max(m * k_blocks, Q8Q8_TB_X_DEPTH));
    activations_scale.resize(std::max(m * k_blocks, Q8Q8_TB_X_DEPTH));

    for (int row = 0; row < n; ++row) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            const int block = row * k_blocks + kb;
            if ((row + 3 * kb) % 31 == 1) {
                weights_scale[block] = ap_uint<16>(0x0001);
            } else {
                const float scale = ((row + 3 * kb) % 13 == 0)
                        ? 0.0f
                        : 0.0025f * static_cast<float>(1 + ((row + kb) % 13));
                weights_scale[block] = float_to_half_bits(scale);
            }
            ap_uint<256> packed = 0;
            for (int lane = 0; lane < Q8Q8_QK; ++lane) {
                int q = ((row * 37 + kb * 19 + lane * 11) % 255) - 127;
                if ((lane % 17) == 0) {
                    q = lane == 0 ? -128 : 127;
                }
                const ap_int<8> q8 = q;
                packed.range(8 * lane + 7, 8 * lane) = q8.range(7, 0);
            }
            weights_q[block] = packed;
        }
    }

    std::vector<float> row(k);
    for (int mi = 0; mi < m; ++mi) {
        for (int col = 0; col < k; ++col) {
            row[col] = ((mi + col) % 97 == 0)
                    ? 0.0f
                    : 0.03125f * static_cast<float>(
                            ((mi * 29 + col * 7) % 41) - 20);
        }
        if (mi == m - 1 && (mi % 5) == 0) {
            std::fill(row.begin(), row.begin() + Q8Q8_QK, 0.0f);
        }
        quantize_activation_row(row.data(), k,
                &activations_q[mi * k_blocks],
                &activations_scale[mi * k_blocks]);
    }
}

void reference(
        const std::vector<ap_uint<256> > & weights_q,
        const std::vector<ap_uint<16> > & weights_scale,
        const std::vector<ap_uint<256> > & activations_q,
        const std::vector<ap_uint<16> > & activations_scale,
        std::vector<float> & y,
        int m,
        int n,
        int k) {
    const int k_blocks = k / Q8Q8_QK;
    y.assign(m * n, 0.0f);
    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            float sumf = 0.0f;
            for (int kb = 0; kb < k_blocks; ++kb) {
                const int wb = ni * k_blocks + kb;
                const int xb = mi * k_blocks + kb;
                int sumi = 0;
                for (int lane = 0; lane < Q8Q8_QK; ++lane) {
                    const ap_int<8> w = weights_q[wb].range(
                            8 * lane + 7, 8 * lane);
                    const ap_int<8> x = activations_q[xb].range(
                            8 * lane + 7, 8 * lane);
                    sumi += w.to_int() * x.to_int();
                }
                sumf += static_cast<float>(sumi) *
                        (half_bits_to_float(weights_scale[wb]) *
                         half_bits_to_float(activations_scale[xb]));
            }
            y[mi * n + ni] = sumf;
        }
    }
}

bool run_case(const char * name, int m, int n, int k) {
    std::vector<ap_uint<256> > weights_q;
    std::vector<ap_uint<16> > weights_scale;
    std::vector<ap_uint<256> > activations_q;
    std::vector<ap_uint<16> > activations_scale;
    std::vector<float> expected;
    std::vector<q8q8_output_word_t> actual(std::max(
            (m * n + Q8Q8_OUTPUT_LANES - 1) / Q8Q8_OUTPUT_LANES,
            Q8Q8_TB_Y_DEPTH));
    make_inputs(m, n, k, weights_q, weights_scale,
            activations_q, activations_scale);
    reference(weights_q, weights_scale, activations_q, activations_scale,
            expected, m, n, k);

    const int status = q8q8_gemm(weights_q.data(), weights_scale.data(),
            activations_q.data(), activations_scale.data(), actual.data(),
            m, n, k);
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int failures = status == Q8Q8_OK ? 0 : 1;
    for (int i = 0; i < m * n; ++i) {
        const float actual_value = output_lane_to_float(
                actual[i / Q8Q8_OUTPUT_LANES], i % Q8Q8_OUTPUT_LANES);
        const float abs_err = std::fabs(actual_value - expected[i]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(expected[i]));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > 1.0e-4f && rel_err > 1.0e-5f) {
            ++failures;
            if (failures <= 4) {
                std::printf("MISMATCH case=%s index=%d got=%.9g expected=%.9g\n",
                        name, i, actual_value, expected[i]);
            }
        }
    }
    std::printf("TB_CASE name=%s m=%d n=%d k=%d status=%d max_abs=%.9g "
                "max_rel=%.9g result=%s\n",
            name, m, n, k, status, max_abs, max_rel,
            failures == 0 ? "PASS" : "FAIL");
    return failures == 0;
}

bool run_invalid_case() {
    std::vector<ap_uint<256> > wq(Q8Q8_TB_WEIGHT_DEPTH);
    std::vector<ap_uint<16> > ws(Q8Q8_TB_WEIGHT_DEPTH);
    std::vector<ap_uint<256> > xq(Q8Q8_TB_X_DEPTH);
    std::vector<ap_uint<16> > xs(Q8Q8_TB_X_DEPTH);
    std::vector<q8q8_output_word_t> y(Q8Q8_TB_Y_DEPTH);
    const q8q8_output_word_t sentinel =
            ap_uint<128>("0x0123456789abcdeffedcba9876543210");
    y[0] = sentinel;
    const int bad_alignment = q8q8_gemm(wq.data(), ws.data(), xq.data(),
            xs.data(), y.data(), 1, 4, 33);
    const int bad_dimension = q8q8_gemm(wq.data(), ws.data(), xq.data(),
            xs.data(), y.data(), 0, 4, 32);
    const int bad_n_alignment = q8q8_gemm(wq.data(), ws.data(), xq.data(),
            xs.data(), y.data(), 1, 3, 32);
    const bool pass = bad_alignment == Q8Q8_ERR_K_ALIGNMENT &&
            bad_dimension == Q8Q8_ERR_DIMENSION &&
            bad_n_alignment == Q8Q8_ERR_N_ALIGNMENT && y[0] == sentinel;
    std::printf("TB_CASE name=invalid_input alignment_status=%d "
                "dimension_status=%d n_alignment_status=%d "
                "output_unchanged=%d result=%s\n",
            bad_alignment, bad_dimension, bad_n_alignment,
            y[0] == sentinel,
            pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    bool pass = true;
    if (argc > 1 && std::strcmp(argv[1], "--rtl-smoke") == 0) {
        pass &= run_case("rtl_small", 3, 4, 64);
        pass &= run_case("rtl_pe16_full_tile", 3, 64, 64);
        pass &= run_case("rtl_m_boundary", 17, 4, 32);
        pass &= run_case("rtl_n_boundary", 1, 68, 32);
        pass &= run_invalid_case();
        std::printf("TB_RESULT=%s mode=rtl_smoke\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    pass &= run_case("small_tile", 3, 4, 64);
    pass &= run_case("gemv", 1, 8, 96);
    pass &= run_case("pe16_full_tile", 3, 64, 64);
    pass &= run_case("tile_boundary", 17, 68, 32);
    pass &= run_case("dataflow_three_tiles", 33, 64, 64);
    pass &= run_case("dataflow_pingpong_reuse", 48, 128, 96);
    pass &= run_case("robobrain_k2560", 2, 8, 2560);
    pass &= run_case("vision_k1024", 2, 8, 1024);
    pass &= run_invalid_case();
    std::printf("TB_RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
