#include "q8_gemm.h"

#include <cstdint>

namespace {

float q8_dot32(const ap_uint<256> packed_q, const float x[Q8_GEMM_QK]) {
#pragma HLS INLINE
    float product[Q8_GEMM_QK];
    float sum16[16];
    float sum8[8];
    float sum4[4];
    float sum2[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=sum16 complete
#pragma HLS ARRAY_PARTITION variable=sum8 complete
#pragma HLS ARRAY_PARTITION variable=sum4 complete
#pragma HLS ARRAY_PARTITION variable=sum2 complete

    // A balanced tree avoids the 32-add serial dependency of a scalar dot.
    for (int lane = 0; lane < Q8_GEMM_QK; ++lane) {
#pragma HLS UNROLL
        const ap_int<8> q = packed_q.range(8 * lane + 7, 8 * lane);
        product[lane] = static_cast<float>(q.to_int()) * x[lane];
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        sum16[i] = product[2 * i] + product[2 * i + 1];
    }
    for (int i = 0; i < 8; ++i) {
#pragma HLS UNROLL
        sum8[i] = sum16[2 * i] + sum16[2 * i + 1];
    }
    for (int i = 0; i < 4; ++i) {
#pragma HLS UNROLL
        sum4[i] = sum8[2 * i] + sum8[2 * i + 1];
    }
    for (int i = 0; i < 2; ++i) {
#pragma HLS UNROLL
        sum2[i] = sum4[2 * i] + sum4[2 * i + 1];
    }
    return sum2[0] + sum2[1];
}

float half_bits_to_float(const ap_uint<16> bits) {
#pragma HLS INLINE
    const ap_uint<1> sign = bits[15];
    const ap_uint<5> exponent = bits.range(14, 10);
    ap_uint<10> mantissa = bits.range(9, 0);
    ap_uint<32> fp32_bits = 0;
    fp32_bits[31] = sign;

    if (exponent == 0) {
        if (mantissa != 0) {
            // Normalize a binary16 subnormal.  A mantissa with bit 9 set is
            // 1.0 * 2^-15; every additional leading zero lowers the exponent.
            int unbiased = -15;
            while (mantissa[9] == 0) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=9
                mantissa <<= 1;
                --unbiased;
            }
            mantissa <<= 1;
            fp32_bits.range(30, 23) = unbiased + 127;
            fp32_bits.range(22, 13) = mantissa;
        }
    } else if (exponent == 31) {
        fp32_bits.range(30, 23) = 255;
        fp32_bits.range(22, 13) = mantissa;
    } else {
        fp32_bits.range(30, 23) = exponent.to_uint() + 112;
        fp32_bits.range(22, 13) = mantissa;
    }

    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.u = fp32_bits.to_uint();
    return converter.f;
}

} // namespace

extern "C" int q8_gemm(
        const ap_uint<256> * weights_q,
        const ap_uint<16>  * weights_scale,
        const float        * activations,
        float              * output,
        int m,
        int n,
        int k) {
#pragma HLS INTERFACE m_axi port=weights_q     offset=slave bundle=gmem_wq depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=weights_scale offset=slave bundle=gmem_ws depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=activations   offset=slave bundle=gmem_x  depth=8192 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output        offset=slave bundle=gmem_y  depth=2048 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=weights_q     bundle=control
#pragma HLS INTERFACE s_axilite port=weights_scale bundle=control
#pragma HLS INTERFACE s_axilite port=activations   bundle=control
#pragma HLS INTERFACE s_axilite port=output        bundle=control
#pragma HLS INTERFACE s_axilite port=m             bundle=control
#pragma HLS INTERFACE s_axilite port=n             bundle=control
#pragma HLS INTERFACE s_axilite port=k             bundle=control
#pragma HLS INTERFACE s_axilite port=return        bundle=control

    if (m <= 0 || n <= 0 || k <= 0 ||
            m > Q8_GEMM_MAX_M || n > Q8_GEMM_MAX_N || k > Q8_GEMM_MAX_K) {
        return Q8_GEMM_ERR_DIMENSION;
    }
    if ((k % Q8_GEMM_QK) != 0) {
        return Q8_GEMM_ERR_K_ALIGNMENT;
    }

    const int k_blocks = k / Q8_GEMM_QK;

    // C = X * W^T, with X[M,K] and W[N,K].  TM and TN provide reuse of
    // activation and weight blocks; PE_N controls physical dot-product lanes.
    for (int m0 = 0; m0 < m; m0 += Q8_GEMM_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=270 avg=32
        for (int n0 = 0; n0 < n; n0 += Q8_GEMM_TN) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=152 avg=64
            float accum[Q8_GEMM_TM][Q8_GEMM_TN];
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8_GEMM_PE_N dim=2

            for (int mi = 0; mi < Q8_GEMM_TM; ++mi) {
                for (int ni = 0; ni < Q8_GEMM_TN; ++ni) {
#pragma HLS PIPELINE II=1
                    accum[mi][ni] = 0.0f;
                }
            }

            for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=304 avg=80
                float x_tile[Q8_GEMM_TM][Q8_GEMM_QK];
#pragma HLS ARRAY_PARTITION variable=x_tile complete dim=2

                for (int mi = 0; mi < Q8_GEMM_TM; ++mi) {
                    for (int lane = 0; lane < Q8_GEMM_QK; ++lane) {
#pragma HLS PIPELINE II=1
                        const int gm = m0 + mi;
                        x_tile[mi][lane] = gm < m
                                ? activations[gm * k + kb * Q8_GEMM_QK + lane]
                                : 0.0f;
                    }
                }

                ap_uint<256> q_tile[Q8_GEMM_TN];
                float scale_tile[Q8_GEMM_TN];
#pragma HLS ARRAY_PARTITION variable=q_tile cyclic factor=Q8_GEMM_PE_N
#pragma HLS ARRAY_PARTITION variable=scale_tile cyclic factor=Q8_GEMM_PE_N

                // Cache a full N tile before compute.  This lets all 256
                // (M row, N PE-group) work items share one deep pipeline
                // instead of flushing the 65-stage FP tree for every group.
                for (int ni = 0; ni < Q8_GEMM_TN; ++ni) {
#pragma HLS PIPELINE II=1
                        const int gn = n0 + ni;
                        if (gn < n) {
                            const int block_index = gn * k_blocks + kb;
                            q_tile[ni] = weights_q[block_index];
                            scale_tile[ni] = half_bits_to_float(weights_scale[block_index]);
                        } else {
                            q_tile[ni] = 0;
                            scale_tile[ni] = 0.0f;
                        }
                }

                constexpr int N_GROUPS = Q8_GEMM_TN / Q8_GEMM_PE_N;
                for (int work = 0; work < Q8_GEMM_TM * N_GROUPS; ++work) {
#pragma HLS PIPELINE II=1
                        const int mi = work / N_GROUPS;
                        const int ng = (work % N_GROUPS) * Q8_GEMM_PE_N;
                        if (m0 + mi < m) {
                            for (int pe = 0; pe < Q8_GEMM_PE_N; ++pe) {
#pragma HLS UNROLL
                                const float dot = q8_dot32(q_tile[ng + pe], x_tile[mi]);
                                accum[mi][ng + pe] += scale_tile[ng + pe] * dot;
                            }
                        }
                }
            }

            for (int mi = 0; mi < Q8_GEMM_TM; ++mi) {
                for (int ni = 0; ni < Q8_GEMM_TN; ++ni) {
#pragma HLS PIPELINE II=1
                    if (m0 + mi < m && n0 + ni < n) {
                        output[(m0 + mi) * n + n0 + ni] = accum[mi][ni];
                    }
                }
            }
        }
    }

    return Q8_GEMM_OK;
}
