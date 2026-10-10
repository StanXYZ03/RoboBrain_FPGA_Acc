#include "q8q8_gemv_stream6.h"

#include <cstdint>
#include <hls_stream.h>

namespace {

ap_int<21> dot_q8_block(const ap_uint<256> lhs, const ap_uint<256> rhs) {
#pragma HLS INLINE
    ap_int<16> products[GEMV_QK];
    ap_int<17> sum16[16];
    ap_int<18> sum8[8];
    ap_int<19> sum4[4];
    ap_int<20> sum2[2];
#pragma HLS ARRAY_PARTITION variable=products complete
#pragma HLS ARRAY_PARTITION variable=sum16 complete
#pragma HLS ARRAY_PARTITION variable=sum8 complete
#pragma HLS ARRAY_PARTITION variable=sum4 complete
#pragma HLS ARRAY_PARTITION variable=sum2 complete
    for (int lane = 0; lane < GEMV_QK; ++lane) {
#pragma HLS UNROLL
        const ap_int<8> a = lhs.range(8 * lane + 7, 8 * lane);
        const ap_int<8> b = rhs.range(8 * lane + 7, 8 * lane);
        products[lane] = a * b;
    }
    for (int i = 0; i < 16; ++i) {
#pragma HLS UNROLL
        sum16[i] = products[2 * i] + products[2 * i + 1];
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

void load_activation(const ap_uint<256> *xq,
                     const ap_uint<16> *xs,
                     ap_uint<256> xq_cache[GEMV_MAX_K_BLOCKS],
                     float xs_cache[GEMV_MAX_K_BLOCKS],
                     int k_blocks) {
#pragma HLS INLINE off
    for (int kb = 0; kb < k_blocks; ++kb) {
#pragma HLS PIPELINE II=2
        xq_cache[kb] = xq[kb];
        xs_cache[kb] = half_bits_to_float(xs[kb]);
    }
}

void load_weight_records(const ap_uint<256> *w0,
                         const ap_uint<256> *w1,
                         const ap_uint<256> *w2,
                         const ap_uint<256> *w3,
                         const ap_uint<256> *w4,
                         const ap_uint<256> *w5,
                         hls::stream<ap_uint<256> > &s0,
                         hls::stream<ap_uint<256> > &s1,
                         hls::stream<ap_uint<256> > &s2,
                         hls::stream<ap_uint<256> > &s3,
                         hls::stream<ap_uint<256> > &s4,
                         hls::stream<ap_uint<256> > &s5,
                         int base,
                         int words) {
#pragma HLS INLINE off
    for (int word = 0; word < words; ++word) {
#pragma HLS PIPELINE II=1
        const int address = base + word;
        s0.write(w0[address]);
        s1.write(w1[address]);
        s2.write(w2[address]);
        s3.write(w3[address]);
        s4.write(w4[address]);
        s5.write(w5[address]);
    }
}

void compute_record_streams(hls::stream<ap_uint<256> > &s0,
                            hls::stream<ap_uint<256> > &s1,
                            hls::stream<ap_uint<256> > &s2,
                            hls::stream<ap_uint<256> > &s3,
                            hls::stream<ap_uint<256> > &s4,
                            hls::stream<ap_uint<256> > &s5,
                            const ap_uint<256> xq_cache[GEMV_MAX_K_BLOCKS],
                            const float xs_cache[GEMV_MAX_K_BLOCKS],
                            float accum[GEMV_GROUP_N],
                            int k_blocks) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accum complete
    float ring0[GEMV_SLOTS_PER_PORT] = {};
    float ring1[GEMV_SLOTS_PER_PORT] = {};
    float ring2[GEMV_SLOTS_PER_PORT] = {};
    float ring3[GEMV_SLOTS_PER_PORT] = {};
    float ring4[GEMV_SLOTS_PER_PORT] = {};
    float ring5[GEMV_SLOTS_PER_PORT] = {};
#pragma HLS ARRAY_PARTITION variable=ring0 complete
#pragma HLS ARRAY_PARTITION variable=ring1 complete
#pragma HLS ARRAY_PARTITION variable=ring2 complete
#pragma HLS ARRAY_PARTITION variable=ring3 complete
#pragma HLS ARRAY_PARTITION variable=ring4 complete
#pragma HLS ARRAY_PARTITION variable=ring5 complete
    ap_uint<256> h0 = 0;
    ap_uint<256> h1 = 0;
    ap_uint<256> h2 = 0;
    ap_uint<256> h3 = 0;
    ap_uint<256> h4 = 0;
    ap_uint<256> h5 = 0;
    const int phases = k_blocks * GEMV_RECORD_WORDS;
    for (int work = 0; work < phases; ++work) {
#pragma HLS PIPELINE II=1
        const int kb = work / GEMV_RECORD_WORDS;
        const int phase = work - kb * GEMV_RECORD_WORDS;
        const ap_uint<256> value0 = s0.read();
        const ap_uint<256> value1 = s1.read();
        const ap_uint<256> value2 = s2.read();
        const ap_uint<256> value3 = s3.read();
        const ap_uint<256> value4 = s4.read();
        const ap_uint<256> value5 = s5.read();
        if (phase == 0) {
            h0 = value0;
            h1 = value1;
            h2 = value2;
            h3 = value3;
            h4 = value4;
            h5 = value5;
        } else {
            const int slot = phase - 1;
            const ap_uint<256> activation = xq_cache[kb];
            const ap_uint<256> q0 = value0;
            const ap_uint<256> q1 = value1;
            const ap_uint<256> q2 = value2;
            const ap_uint<256> q3 = value3;
            const ap_uint<256> q4 = value4;
            const ap_uint<256> q5 = value5;
            const ap_uint<16> s0 = h0.range(16 * slot + 15, 16 * slot);
            const ap_uint<16> s1 = h1.range(16 * slot + 15, 16 * slot);
            const ap_uint<16> s2 = h2.range(16 * slot + 15, 16 * slot);
            const ap_uint<16> s3 = h3.range(16 * slot + 15, 16 * slot);
            const ap_uint<16> s4 = h4.range(16 * slot + 15, 16 * slot);
            const ap_uint<16> s5 = h5.range(16 * slot + 15, 16 * slot);
            const ap_int<21> d0 = dot_q8_block(q0, activation);
            const ap_int<21> d1 = dot_q8_block(q1, activation);
            const ap_int<21> d2 = dot_q8_block(q2, activation);
            const ap_int<21> d3 = dot_q8_block(q3, activation);
            const ap_int<21> d4 = dot_q8_block(q4, activation);
            const ap_int<21> d5 = dot_q8_block(q5, activation);
            const float activation_scale = xs_cache[kb];
            const float c0 = static_cast<float>(d0.to_int()) *
                    (half_bits_to_float(s0) * activation_scale);
            const float c1 = static_cast<float>(d1.to_int()) *
                    (half_bits_to_float(s1) * activation_scale);
            const float c2 = static_cast<float>(d2.to_int()) *
                    (half_bits_to_float(s2) * activation_scale);
            const float c3 = static_cast<float>(d3.to_int()) *
                    (half_bits_to_float(s3) * activation_scale);
            const float c4 = static_cast<float>(d4.to_int()) *
                    (half_bits_to_float(s4) * activation_scale);
            const float c5 = static_cast<float>(d5.to_int()) *
                    (half_bits_to_float(s5) * activation_scale);
            const float next0 = ring0[0] + c0;
            const float next1 = ring1[0] + c1;
            const float next2 = ring2[0] + c2;
            const float next3 = ring3[0] + c3;
            const float next4 = ring4[0] + c4;
            const float next5 = ring5[0] + c5;
            for (int shift = 0; shift < GEMV_SLOTS_PER_PORT - 1; ++shift) {
#pragma HLS UNROLL
                ring0[shift] = ring0[shift + 1];
                ring1[shift] = ring1[shift + 1];
                ring2[shift] = ring2[shift + 1];
                ring3[shift] = ring3[shift + 1];
                ring4[shift] = ring4[shift + 1];
                ring5[shift] = ring5[shift + 1];
            }
            ring0[GEMV_SLOTS_PER_PORT - 1] = next0;
            ring1[GEMV_SLOTS_PER_PORT - 1] = next1;
            ring2[GEMV_SLOTS_PER_PORT - 1] = next2;
            ring3[GEMV_SLOTS_PER_PORT - 1] = next3;
            ring4[GEMV_SLOTS_PER_PORT - 1] = next4;
            ring5[GEMV_SLOTS_PER_PORT - 1] = next5;
        }
    }
    for (int slot = 0; slot < GEMV_SLOTS_PER_PORT; ++slot) {
#pragma HLS UNROLL
        const int base = slot * GEMV_PORTS;
        accum[base + 0] = ring0[slot];
        accum[base + 1] = ring1[slot];
        accum[base + 2] = ring2[slot];
        accum[base + 3] = ring3[slot];
        accum[base + 4] = ring4[slot];
        accum[base + 5] = ring5[slot];
    }
}

void compute_group(const ap_uint<256> *w0,
                   const ap_uint<256> *w1,
                   const ap_uint<256> *w2,
                   const ap_uint<256> *w3,
                   const ap_uint<256> *w4,
                   const ap_uint<256> *w5,
                   const ap_uint<256> xq_cache[GEMV_MAX_K_BLOCKS],
                   const float xs_cache[GEMV_MAX_K_BLOCKS],
                   float accum[GEMV_GROUP_N],
                   int group,
                   int k_blocks) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION variable=accum complete
    hls::stream<ap_uint<256> > s0("weight_stream_0");
    hls::stream<ap_uint<256> > s1("weight_stream_1");
    hls::stream<ap_uint<256> > s2("weight_stream_2");
    hls::stream<ap_uint<256> > s3("weight_stream_3");
    hls::stream<ap_uint<256> > s4("weight_stream_4");
    hls::stream<ap_uint<256> > s5("weight_stream_5");
#pragma HLS STREAM variable=s0 depth=32
#pragma HLS STREAM variable=s1 depth=32
#pragma HLS STREAM variable=s2 depth=32
#pragma HLS STREAM variable=s3 depth=32
#pragma HLS STREAM variable=s4 depth=32
#pragma HLS STREAM variable=s5 depth=32
    const int words = k_blocks * GEMV_RECORD_WORDS;
    const int base = group * words;
    load_weight_records(w0, w1, w2, w3, w4, w5,
                        s0, s1, s2, s3, s4, s5, base, words);
    compute_record_streams(s0, s1, s2, s3, s4, s5,
                           xq_cache, xs_cache, accum, k_blocks);
}

void store_group(gemv_output_word_t *output,
                 const float accum[GEMV_GROUP_N],
                 int n0,
                 int n) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=accum complete
    int valid = n - n0;
    if (valid > GEMV_GROUP_N) valid = GEMV_GROUP_N;
    for (int word = 0; word < valid / GEMV_OUTPUT_LANES; ++word) {
#pragma HLS PIPELINE II=1
        gemv_output_word_t packed = 0;
        for (int lane = 0; lane < GEMV_OUTPUT_LANES; ++lane) {
#pragma HLS UNROLL
            packed.range(32 * lane + 31, 32 * lane) =
                    float_to_bits(accum[word * GEMV_OUTPUT_LANES + lane]);
        }
        output[n0 / GEMV_OUTPUT_LANES + word] = packed;
    }
}

}  // namespace

extern "C" int q8q8_gemv_stream6(
        const ap_uint<256> *w0,
        const ap_uint<256> *w1,
        const ap_uint<256> *w2,
        const ap_uint<256> *w3,
        const ap_uint<256> *w4,
        const ap_uint<256> *w5,
        const ap_uint<256> *xq,
        const ap_uint<16> *xs,
        gemv_output_word_t *output,
        int n,
        int k) {
#pragma HLS INTERFACE m_axi port=w0 offset=slave bundle=gmem_w0 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=w3 offset=slave bundle=gmem_w3 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=w4 offset=slave bundle=gmem_w4 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=w5 offset=slave bundle=gmem_w5 depth=4096 max_read_burst_length=16 num_read_outstanding=4
#pragma HLS INTERFACE m_axi port=xq offset=slave bundle=gmem_io depth=304 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=xs offset=slave bundle=gmem_io depth=304 max_read_burst_length=256 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem_io depth=37984 max_write_burst_length=256 num_write_outstanding=16
#pragma HLS INTERFACE s_axilite port=w0 bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=w3 bundle=control
#pragma HLS INTERFACE s_axilite port=w4 bundle=control
#pragma HLS INTERFACE s_axilite port=w5 bundle=control
#pragma HLS INTERFACE s_axilite port=xq bundle=control
#pragma HLS INTERFACE s_axilite port=xs bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=n bundle=control
#pragma HLS INTERFACE s_axilite port=k bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    if (n <= 0 || k <= 0 || n > GEMV_MAX_N || k > GEMV_MAX_K) {
        return GEMV_ERR_DIMENSION;
    }
    if (k % GEMV_QK != 0) return GEMV_ERR_K_ALIGNMENT;
    if (n % GEMV_OUTPUT_LANES != 0) return GEMV_ERR_N_ALIGNMENT;

    const int k_blocks = k / GEMV_QK;
    const int groups = (n + GEMV_GROUP_N - 1) / GEMV_GROUP_N;
    ap_uint<256> xq_cache[GEMV_MAX_K_BLOCKS];
    float xs_cache[GEMV_MAX_K_BLOCKS];
#pragma HLS BIND_STORAGE variable=xq_cache type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=xs_cache type=ram_2p impl=bram
    load_activation(xq, xs, xq_cache, xs_cache, k_blocks);
    for (int group = 0; group < groups; ++group) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3166 avg=203
        float accum[GEMV_GROUP_N];
#pragma HLS ARRAY_PARTITION variable=accum complete
        compute_group(w0, w1, w2, w3, w4, w5,
                      xq_cache, xs_cache, accum, group, k_blocks);
        store_group(output, accum, group * GEMV_GROUP_N, n);
    }
    return GEMV_OK;
}
