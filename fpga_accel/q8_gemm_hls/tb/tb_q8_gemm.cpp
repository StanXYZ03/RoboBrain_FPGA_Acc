#include "q8_gemm.h"

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

void make_inputs(
        int m,
        int n,
        int k,
        std::vector<ap_uint<256> > & weights_q,
        std::vector<ap_uint<16> > & scales,
        std::vector<float> & x) {
    const int k_blocks = k / Q8_GEMM_QK;
    weights_q.resize(std::max(n * k_blocks, Q8_GEMM_TB_WEIGHT_DEPTH));
    scales.resize(std::max(n * k_blocks, Q8_GEMM_TB_WEIGHT_DEPTH));
    x.resize(std::max(m * k, Q8_GEMM_TB_X_DEPTH));

    for (int row = 0; row < n; ++row) {
        for (int kb = 0; kb < k_blocks; ++kb) {
            const int block = row * k_blocks + kb;
            if ((row + 3 * kb) % 29 == 1) {
                scales[block] = ap_uint<16>(0x0001); // smallest subnormal
            } else {
                const float scale = ((row + 3 * kb) % 11 == 0)
                        ? 0.0f
                        : 0.0025f * static_cast<float>(1 + ((row + kb) % 13));
                scales[block] = float_to_half_bits(scale);
            }
            ap_uint<256> packed = 0;
            for (int lane = 0; lane < Q8_GEMM_QK; ++lane) {
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

    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < k; ++col) {
            x[row * k + col] =
                    0.03125f * static_cast<float>(((row * 29 + col * 7) % 41) - 20);
        }
    }
}

void reference(
        const std::vector<ap_uint<256> > & weights_q,
        const std::vector<ap_uint<16> > & scales,
        const std::vector<float> & x,
        std::vector<float> & y,
        int m,
        int n,
        int k) {
    const int k_blocks = k / Q8_GEMM_QK;
    y.assign(m * n, 0.0f);
    for (int mi = 0; mi < m; ++mi) {
        for (int ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (int kb = 0; kb < k_blocks; ++kb) {
                const int block = ni * k_blocks + kb;
                float product[Q8_GEMM_QK];
                for (int lane = 0; lane < Q8_GEMM_QK; ++lane) {
                    const ap_int<8> q = weights_q[block].range(
                            8 * lane + 7, 8 * lane);
                    product[lane] = static_cast<float>(q.to_int()) *
                            x[mi * k + kb * Q8_GEMM_QK + lane];
                }
                for (int width = Q8_GEMM_QK; width > 1; width /= 2) {
                    for (int i = 0; i < width / 2; ++i) {
                        product[i] = product[2 * i] + product[2 * i + 1];
                    }
                }
                acc += half_bits_to_float(scales[block]) * product[0];
            }
            y[mi * n + ni] = acc;
        }
    }
}

bool run_case(const char * name, int m, int n, int k) {
    std::vector<ap_uint<256> > weights_q;
    std::vector<ap_uint<16> > scales;
    std::vector<float> x;
    std::vector<float> expected;
    std::vector<float> actual(
            std::max(m * n, Q8_GEMM_TB_Y_DEPTH), -12345.0f);
    make_inputs(m, n, k, weights_q, scales, x);
    reference(weights_q, scales, x, expected, m, n, k);

    const int status = q8_gemm(
            weights_q.data(), scales.data(), x.data(), actual.data(), m, n, k);
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int failures = status == Q8_GEMM_OK ? 0 : 1;
    for (int i = 0; i < m * n; ++i) {
        const float abs_err = std::fabs(actual[i] - expected[i]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(expected[i]));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        if (abs_err > 1.0e-4f && rel_err > 1.0e-5f) {
            ++failures;
            if (failures <= 4) {
                std::printf("MISMATCH case=%s index=%d got=%.9g expected=%.9g\n",
                        name, i, actual[i], expected[i]);
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
    std::vector<ap_uint<256> > wq(Q8_GEMM_TB_WEIGHT_DEPTH);
    std::vector<ap_uint<16> > ws(Q8_GEMM_TB_WEIGHT_DEPTH);
    std::vector<float> x(Q8_GEMM_TB_X_DEPTH);
    std::vector<float> y(Q8_GEMM_TB_Y_DEPTH);
    y[0] = 17.0f;
    y[1] = 19.0f;
    const int bad_alignment = q8_gemm(
            wq.data(), ws.data(), x.data(), y.data(), 1, 1, 33);
    const int bad_dimension = q8_gemm(
            wq.data(), ws.data(), x.data(), y.data(), 0, 1, 32);
    const bool pass = bad_alignment == Q8_GEMM_ERR_K_ALIGNMENT &&
            bad_dimension == Q8_GEMM_ERR_DIMENSION &&
            y[0] == 17.0f && y[1] == 19.0f;
    std::printf("TB_CASE name=invalid_input alignment_status=%d dimension_status=%d "
                "output_unchanged=%d result=%s\n",
            bad_alignment, bad_dimension, y[0] == 17.0f && y[1] == 19.0f,
            pass ? "PASS" : "FAIL");
    return pass;
}

} // namespace

int main(int argc, char ** argv) {
    bool pass = true;
    if (argc > 1 && std::strcmp(argv[1], "--rtl-smoke") == 0) {
        pass &= run_case("rtl_small", 3, 5, 64);
        pass &= run_case("rtl_m_boundary", 17, 1, 32);
        pass &= run_case("rtl_n_boundary", 1, 65, 32);
        pass &= run_invalid_case();
        std::printf("TB_RESULT=%s mode=rtl_smoke\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    }

    pass &= run_case("small_tile", 3, 5, 64);
    pass &= run_case("gemv", 1, 7, 96);
    pass &= run_case("tile_boundary", 17, 65, 32);
    pass &= run_case("robobrain_k2560", 2, 5, 2560);
    pass &= run_case("vision_k1024", 2, 5, 1024);
    pass &= run_invalid_case();
    std::printf("TB_RESULT=%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
