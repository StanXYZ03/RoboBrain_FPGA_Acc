#include "q8q8_gemm.h"

#include <cstdint>

namespace {

ap_int<21> dot_q8_0_block(
        const ap_uint<256> lhs,
        const ap_uint<256> rhs) {
#pragma HLS INLINE
    ap_int<16> product[32];
    ap_int<17> sum16[16];
    ap_int<18> sum8[8];
    ap_int<19> sum4[4];
    ap_int<20> sum2[2];
#pragma HLS ARRAY_PARTITION variable=product complete
#pragma HLS ARRAY_PARTITION variable=sum16 complete
#pragma HLS ARRAY_PARTITION variable=sum8 complete
#pragma HLS ARRAY_PARTITION variable=sum4 complete
#pragma HLS ARRAY_PARTITION variable=sum2 complete

    for (int lane = 0; lane < Q8Q8_QK; ++lane) {
#pragma HLS UNROLL
        const ap_int<8> a = lhs.range(8 * lane + 7, 8 * lane);
        const ap_int<8> b = rhs.range(8 * lane + 7, 8 * lane);
        product[lane] = a * b;
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

extern "C" int q8q8_gemm(
        const ap_uint<256> * weights_q,
        const ap_uint<16>  * weights_scale,
        const ap_uint<256> * activations_q,
        const ap_uint<16>  * activations_scale,
        float              * output,
        int m,
        int n,
        int k) {
#pragma HLS INTERFACE m_axi port=weights_q        offset=slave bundle=gmem_wq depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=weights_scale    offset=slave bundle=gmem_ws depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=activations_q    offset=slave bundle=gmem_xq depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=activations_scale offset=slave bundle=gmem_xs depth=512 max_read_burst_length=64
#pragma HLS INTERFACE m_axi port=output           offset=slave bundle=gmem_y depth=2048 max_write_burst_length=64
#pragma HLS INTERFACE s_axilite port=weights_q         bundle=control
#pragma HLS INTERFACE s_axilite port=weights_scale     bundle=control
#pragma HLS INTERFACE s_axilite port=activations_q     bundle=control
#pragma HLS INTERFACE s_axilite port=activations_scale bundle=control
#pragma HLS INTERFACE s_axilite port=output            bundle=control
#pragma HLS INTERFACE s_axilite port=m                 bundle=control
#pragma HLS INTERFACE s_axilite port=n                 bundle=control
#pragma HLS INTERFACE s_axilite port=k                 bundle=control
#pragma HLS INTERFACE s_axilite port=return            bundle=control

    if (m <= 0 || n <= 0 || k <= 0 ||
            m > Q8Q8_MAX_M || n > Q8Q8_MAX_N || k > Q8Q8_MAX_K) {
        return Q8Q8_ERR_DIMENSION;
    }
    if ((k % Q8Q8_QK) != 0) {
        return Q8Q8_ERR_K_ALIGNMENT;
    }

    const int k_blocks = k / Q8Q8_QK;

    for (int m0 = 0; m0 < m; m0 += Q8Q8_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=270 avg=32
        for (int n0 = 0; n0 < n; n0 += Q8Q8_TN) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=152 avg=64
            float accum[Q8Q8_TM][Q8Q8_TN];
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8Q8_PE_N dim=2

            for (int mi = 0; mi < Q8Q8_TM; ++mi) {
                for (int ni = 0; ni < Q8Q8_TN; ++ni) {
#pragma HLS PIPELINE II=1
                    accum[mi][ni] = 0.0f;
                }
            }

            for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=304 avg=80
                ap_uint<256> x_q_tile[Q8Q8_TM];
                float x_scale_tile[Q8Q8_TM];
#pragma HLS ARRAY_PARTITION variable=x_q_tile complete
#pragma HLS ARRAY_PARTITION variable=x_scale_tile complete

                for (int mi = 0; mi < Q8Q8_TM; ++mi) {
#pragma HLS PIPELINE II=1
                    const int gm = m0 + mi;
                    if (gm < m) {
                        const int block_index = gm * k_blocks + kb;
                        x_q_tile[mi] = activations_q[block_index];
                        x_scale_tile[mi] = half_bits_to_float(
                                activations_scale[block_index]);
                    } else {
                        x_q_tile[mi] = 0;
                        x_scale_tile[mi] = 0.0f;
                    }
                }

                ap_uint<256> w_q_tile[Q8Q8_TN];
                float w_scale_tile[Q8Q8_TN];
#pragma HLS ARRAY_PARTITION variable=w_q_tile cyclic factor=Q8Q8_PE_N
#pragma HLS ARRAY_PARTITION variable=w_scale_tile cyclic factor=Q8Q8_PE_N

                for (int ni = 0; ni < Q8Q8_TN; ++ni) {
#pragma HLS PIPELINE II=1
                    const int gn = n0 + ni;
                    if (gn < n) {
                        const int block_index = gn * k_blocks + kb;
                        w_q_tile[ni] = weights_q[block_index];
                        w_scale_tile[ni] = half_bits_to_float(
                                weights_scale[block_index]);
                    } else {
                        w_q_tile[ni] = 0;
                        w_scale_tile[ni] = 0.0f;
                    }
                }

                constexpr int N_GROUPS = Q8Q8_TN / Q8Q8_PE_N;
                for (int work = 0; work < Q8Q8_TM * N_GROUPS; ++work) {
#pragma HLS PIPELINE II=1
                    const int mi = work / N_GROUPS;
                    const int ng = (work % N_GROUPS) * Q8Q8_PE_N;
                    if (m0 + mi < m) {
                        for (int pe = 0; pe < Q8Q8_PE_N; ++pe) {
#pragma HLS UNROLL
                            const ap_int<21> dot = dot_q8_0_block(
                                    w_q_tile[ng + pe], x_q_tile[mi]);
                            const float block_scale =
                                    w_scale_tile[ng + pe] * x_scale_tile[mi];
                            accum[mi][ng + pe] +=
                                    static_cast<float>(dot.to_int()) * block_scale;
                        }
                    }
                }
            }

            for (int mi = 0; mi < Q8Q8_TM; ++mi) {
                for (int ni = 0; ni < Q8Q8_TN; ++ni) {
#pragma HLS PIPELINE II=1
                    if (m0 + mi < m && n0 + ni < n) {
                        output[(m0 + mi) * n + n0 + ni] = accum[mi][ni];
                    }
                }
            }
        }
    }

    return Q8Q8_OK;
}
