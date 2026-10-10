#include "q8q8_gemm.h"

#include <cstdint>

namespace {

ap_int<21> dot_q8_0_block(const ap_uint<256> lhs,
                          const ap_uint<256> rhs) {
#pragma HLS INLINE
    ap_int<16> product[Q8Q8_QK];
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
            ap_uint<4> shift;
            if (mantissa[9]) shift = 0;
            else if (mantissa[8]) shift = 1;
            else if (mantissa[7]) shift = 2;
            else if (mantissa[6]) shift = 3;
            else if (mantissa[5]) shift = 4;
            else if (mantissa[4]) shift = 5;
            else if (mantissa[3]) shift = 6;
            else if (mantissa[2]) shift = 7;
            else if (mantissa[1]) shift = 8;
            else shift = 9;
            mantissa <<= shift.to_uint() + 1;
            fp32_bits.range(30, 23) = 112 - shift.to_uint();
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

ap_uint<32> float_to_bits(const float value) {
#pragma HLS INLINE
    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.f = value;
    return converter.u;
}

void load_weight_stationary_tile(
        const ap_uint<256> *weights_q,
        const ap_uint<16> *weights_scale,
        ap_uint<256> weight_q_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS],
        float weight_scale_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS],
        int n0,
        int n,
        int k_blocks) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight_q_cache cyclic factor=Q8Q8_PE_N dim=1
#pragma HLS ARRAY_PARTITION variable=weight_scale_cache cyclic factor=Q8Q8_PE_N dim=1
    // ni -> kb is intentional: every valid row becomes one contiguous AXI
    // burst instead of K-block-major strided accesses across weight rows.
    for (int ni = 0; ni < Q8Q8_TN; ++ni) {
        for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            const int gn = n0 + ni;
            if (gn < n) {
                const int index = gn * k_blocks + kb;
                weight_q_cache[ni][kb] = weights_q[index];
                weight_scale_cache[ni][kb] =
                        half_bits_to_float(weights_scale[index]);
            } else {
                weight_q_cache[ni][kb] = 0;
                weight_scale_cache[ni][kb] = 0.0f;
            }
        }
    }
}

void load_activation_tile(
        const ap_uint<256> *activations_q,
        const ap_uint<16> *activations_scale,
        ap_uint<256> activation_q_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS],
        float activation_scale_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS],
        int m0,
        int m,
        int k_blocks) {
#pragma HLS INLINE off
    // The same row-major ordering creates contiguous activation bursts.
    for (int mi = 0; mi < Q8Q8_TM; ++mi) {
        for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_FLATTEN
            const int gm = m0 + mi;
            if (gm < m) {
                const int index = gm * k_blocks + kb;
                activation_q_cache[mi][kb] = activations_q[index];
                activation_scale_cache[mi][kb] =
                        half_bits_to_float(activations_scale[index]);
            } else {
                activation_q_cache[mi][kb] = 0;
                activation_scale_cache[mi][kb] = 0.0f;
            }
        }
    }
}

void compute_cached_tile(
        const ap_uint<256> weight_q_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS],
        const float weight_scale_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS],
        const ap_uint<256> activation_q_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS],
        const float activation_scale_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS],
        float accum[Q8Q8_TM][Q8Q8_TN],
        int m0,
        int m,
        int k_blocks) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=weight_q_cache cyclic factor=Q8Q8_PE_N dim=1
#pragma HLS ARRAY_PARTITION variable=weight_scale_cache cyclic factor=Q8Q8_PE_N dim=1
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8Q8_PE_N dim=2
    constexpr int N_GROUPS = Q8Q8_TN / Q8Q8_PE_N;
    for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=304 avg=80
        for (int work = 0; work < Q8Q8_TM * N_GROUPS; ++work) {
#pragma HLS PIPELINE II=1
            // Row-major interleaving keeps accesses to the same partitioned
            // accumulator bank far enough apart for the FP32 adder pipeline.
            const int ng = (work / Q8Q8_TM) * Q8Q8_PE_N;
            const int mi = work % Q8Q8_TM;
            if (m0 + mi < m) {
                const ap_uint<256> xq = activation_q_cache[mi][kb];
                const float xs = activation_scale_cache[mi][kb];
                for (int pe = 0; pe < Q8Q8_PE_N; ++pe) {
#pragma HLS UNROLL
                    const ap_int<21> dot = dot_q8_0_block(
                            weight_q_cache[ng + pe][kb], xq);
                    const float block_scale =
                            weight_scale_cache[ng + pe][kb] * xs;
                    accum[mi][ng + pe] +=
                            static_cast<float>(dot.to_int()) * block_scale;
                }
            }
        }
    }
}

void initialize_accumulator(float accum[Q8Q8_TM][Q8Q8_TN]) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8Q8_PE_N dim=2
    for (int mi = 0; mi < Q8Q8_TM; ++mi) {
        for (int ni = 0; ni < Q8Q8_TN; ++ni) {
#pragma HLS PIPELINE II=1
            accum[mi][ni] = 0.0f;
        }
    }
}

void store_output_tile(q8q8_output_word_t *output,
                       const float accum[Q8Q8_TM][Q8Q8_TN],
                       int m0, int n0, int m, int n) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8Q8_PE_N dim=2
    const int valid_rows = (m - m0 < Q8Q8_TM) ? m - m0 : Q8Q8_TM;
    const int valid_columns = (n - n0 < Q8Q8_TN) ? n - n0 : Q8Q8_TN;
    const int tile_words = valid_columns / Q8Q8_OUTPUT_LANES;
    const int row_words = n / Q8Q8_OUTPUT_LANES;
    for (int mi = 0; mi < valid_rows; ++mi) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=16
        const int output_base = (m0 + mi) * row_words +
                n0 / Q8Q8_OUTPUT_LANES;
        for (int word = 0; word < tile_words; ++word) {
#pragma HLS PIPELINE II=1
            q8q8_output_word_t packed = 0;
            for (int lane = 0; lane < Q8Q8_OUTPUT_LANES; ++lane) {
#pragma HLS UNROLL
                const int ni = word * Q8Q8_OUTPUT_LANES + lane;
                packed.range(32 * lane + 31, 32 * lane) =
                        float_to_bits(accum[mi][ni]);
            }
            output[output_base + word] = packed;
        }
    }
}

} // namespace

extern "C" int q8q8_gemm(
        const ap_uint<256> *weights_q,
        const ap_uint<16> *weights_scale,
        const ap_uint<256> *activations_q,
        const ap_uint<16> *activations_scale,
        q8q8_output_word_t *output,
        int m,
        int n,
        int k) {
#pragma HLS INTERFACE m_axi port=weights_q offset=slave bundle=gmem_wq depth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=weights_scale offset=slave bundle=gmem_ws depth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=activations_q offset=slave bundle=gmem_xq depth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=activations_scale offset=slave bundle=gmem_xs depth=512 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem_y depth=2048 max_write_burst_length=16 num_write_outstanding=4
#pragma HLS INTERFACE s_axilite port=weights_q bundle=control
#pragma HLS INTERFACE s_axilite port=weights_scale bundle=control
#pragma HLS INTERFACE s_axilite port=activations_q bundle=control
#pragma HLS INTERFACE s_axilite port=activations_scale bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=m bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (m <= 0 || n <= 0 || k <= 0 ||
            m > Q8Q8_MAX_M || n > Q8Q8_MAX_N || k > Q8Q8_MAX_K) {
        return Q8Q8_ERR_DIMENSION;
    }
    if ((k % Q8Q8_QK) != 0) return Q8Q8_ERR_K_ALIGNMENT;
    if ((n % Q8Q8_OUTPUT_LANES) != 0) return Q8Q8_ERR_N_ALIGNMENT;
    const int k_blocks = k / Q8Q8_QK;

    ap_uint<256> weight_q_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS];
    float weight_scale_cache[Q8Q8_TN][Q8Q8_MAX_K_BLOCKS];
    ap_uint<256> activation_q_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS];
    float activation_scale_cache[Q8Q8_TM][Q8Q8_MAX_K_BLOCKS];
#pragma HLS ARRAY_PARTITION variable=weight_q_cache cyclic factor=Q8Q8_PE_N dim=1
#pragma HLS ARRAY_PARTITION variable=weight_scale_cache cyclic factor=Q8Q8_PE_N dim=1
#pragma HLS BIND_STORAGE variable=weight_q_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=weight_scale_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=activation_q_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=activation_scale_cache type=ram_2p impl=bram

    // N outermost is the weight-stationary transformation. A complete 64xK
    // weight tile is loaded once and reused by every M tile.
    for (int n0 = 0; n0 < n; n0 += Q8Q8_TN) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=152 avg=64
        load_weight_stationary_tile(weights_q, weights_scale,
                                    weight_q_cache, weight_scale_cache,
                                    n0, n, k_blocks);
        for (int m0 = 0; m0 < m; m0 += Q8Q8_TM) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=270 avg=32
            float accum[Q8Q8_TM][Q8Q8_TN];
#pragma HLS ARRAY_PARTITION variable=accum cyclic factor=Q8Q8_PE_N dim=2
            load_activation_tile(activations_q, activations_scale,
                                 activation_q_cache, activation_scale_cache,
                                 m0, m, k_blocks);
            initialize_accumulator(accum);
            compute_cached_tile(weight_q_cache, weight_scale_cache,
                                activation_q_cache, activation_scale_cache,
                                accum, m0, m, k_blocks);
            store_output_tile(output, accum, m0, n0, m, n);
        }
    }
    return Q8Q8_OK;
}
