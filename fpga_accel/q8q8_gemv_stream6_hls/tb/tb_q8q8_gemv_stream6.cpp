#include "q8q8_gemv_stream6.h"

#include <hls_half.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

ap_uint<16> float_to_half_bits(float value) {
    half h = value;
    return ap_uint<16>(h.get_bits());
}

float half_bits_to_float(ap_uint<16> bits) {
    const std::uint16_t raw = static_cast<std::uint16_t>(bits.to_uint());
    const std::uint32_t sign = static_cast<std::uint32_t>(raw & 0x8000u) << 16;
    const std::uint32_t exponent = (raw >> 10) & 0x1fu;
    std::uint32_t mantissa = raw & 0x03ffu;
    std::uint32_t fp32 = sign;
    if (exponent == 0) {
        if (mantissa != 0) {
            int shift = 0;
            while ((mantissa & 0x0200u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa = (mantissa << 1) & 0x03ffu;
            fp32 |= static_cast<std::uint32_t>(112 - shift) << 23;
            fp32 |= mantissa << 13;
        }
    } else if (exponent == 31) {
        fp32 |= 0xffu << 23;
        fp32 |= mantissa << 13;
    } else {
        fp32 |= (exponent + 112u) << 23;
        fp32 |= mantissa << 13;
    }
    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.u = fp32;
    return converter.f;
}

float output_float(const gemv_output_word_t &word, int lane) {
    union {
        std::uint32_t u;
        float f;
    } converter;
    converter.u = word.range(32 * lane + 31, 32 * lane).to_uint();
    return converter.f;
}

void make_case(int n, int k,
               std::vector<ap_uint<256> > &row_q,
               std::vector<ap_uint<16> > &row_s,
               std::vector<ap_uint<256> > &xq,
               std::vector<ap_uint<16> > &xs) {
    const int blocks = k / GEMV_QK;
    row_q.resize(n * blocks);
    row_s.resize(n * blocks);
    xq.resize(blocks);
    xs.resize(blocks);
    for (int row = 0; row < n; ++row) {
        for (int kb = 0; kb < blocks; ++kb) {
            const int index = row * blocks + kb;
            row_s[index] = ((row + kb) % 37 == 0)
                    ? ap_uint<16>(0x0001)
                    : float_to_half_bits(0.001f * (1 + ((row + 3 * kb) % 23)));
            ap_uint<256> packed = 0;
            for (int lane = 0; lane < GEMV_QK; ++lane) {
                int value = ((row * 31 + kb * 17 + lane * 7) % 255) - 127;
                if (lane == 0) value = -128;
                const ap_int<8> q = value;
                packed.range(8 * lane + 7, 8 * lane) = q.range(7, 0);
            }
            row_q[index] = packed;
        }
    }
    for (int kb = 0; kb < blocks; ++kb) {
        xs[kb] = (kb % 19 == 0)
                ? ap_uint<16>(0x0001)
                : float_to_half_bits(0.002f * (1 + kb % 17));
        ap_uint<256> packed = 0;
        for (int lane = 0; lane < GEMV_QK; ++lane) {
            const ap_int<8> q = ((kb * 13 + lane * 11) % 255) - 127;
            packed.range(8 * lane + 7, 8 * lane) = q.range(7, 0);
        }
        xq[kb] = packed;
    }
}

void repack(const std::vector<ap_uint<256> > &row_q,
            const std::vector<ap_uint<16> > &row_s,
            int n,
            int k,
            std::array<std::vector<ap_uint<256> >, GEMV_PORTS> &streams) {
    const int blocks = k / GEMV_QK;
    const int groups = (n + GEMV_GROUP_N - 1) / GEMV_GROUP_N;
    const int words = groups * blocks * GEMV_RECORD_WORDS;
    for (int port = 0; port < GEMV_PORTS; ++port) {
        streams[port].assign(std::max(words, 4096), ap_uint<256>(0));
    }
    for (int group = 0; group < groups; ++group) {
        for (int kb = 0; kb < blocks; ++kb) {
            const int record = (group * blocks + kb) * GEMV_RECORD_WORDS;
            for (int port = 0; port < GEMV_PORTS; ++port) {
                ap_uint<256> header = 0;
                for (int slot = 0; slot < GEMV_SLOTS_PER_PORT; ++slot) {
                    const int local = slot * GEMV_PORTS + port;
                    const int row = group * GEMV_GROUP_N + local;
                    if (row < n) {
                        const int source = row * blocks + kb;
                        header.range(16 * slot + 15, 16 * slot) = row_s[source];
                        streams[port][record + 1 + slot] = row_q[source];
                    }
                }
                streams[port][record] = header;
            }
        }
    }
}

std::vector<float> reference(const std::vector<ap_uint<256> > &row_q,
                             const std::vector<ap_uint<16> > &row_s,
                             const std::vector<ap_uint<256> > &xq,
                             const std::vector<ap_uint<16> > &xs,
                             int n,
                             int k) {
    const int blocks = k / GEMV_QK;
    std::vector<float> result(n, 0.0f);
    for (int row = 0; row < n; ++row) {
        for (int kb = 0; kb < blocks; ++kb) {
            int dot = 0;
            for (int lane = 0; lane < GEMV_QK; ++lane) {
                const ap_int<8> w = row_q[row * blocks + kb].range(
                        8 * lane + 7, 8 * lane);
                const ap_int<8> x = xq[kb].range(8 * lane + 7, 8 * lane);
                dot += w.to_int() * x.to_int();
            }
            result[row] += static_cast<float>(dot) *
                    (half_bits_to_float(row_s[row * blocks + kb]) *
                     half_bits_to_float(xs[kb]));
        }
    }
    return result;
}

bool run_case(const char *name, int n, int k) {
    std::vector<ap_uint<256> > row_q;
    std::vector<ap_uint<16> > row_s;
    std::vector<ap_uint<256> > xq;
    std::vector<ap_uint<16> > xs;
    std::array<std::vector<ap_uint<256> >, GEMV_PORTS> streams;
    make_case(n, k, row_q, row_s, xq, xs);
    repack(row_q, row_s, n, k, streams);
    const std::vector<float> expected = reference(row_q, row_s, xq, xs, n, k);
    std::vector<gemv_output_word_t> actual(std::max(
            n / GEMV_OUTPUT_LANES, 37984), gemv_output_word_t(0));
    const int status = q8q8_gemv_stream6(
            streams[0].data(), streams[1].data(), streams[2].data(),
            streams[3].data(), streams[4].data(), streams[5].data(),
            xq.data(), xs.data(), actual.data(), n, k);
    int failures = status == GEMV_OK ? 0 : 1;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (int row = 0; row < n; ++row) {
        const float got = output_float(actual[row / 4], row % 4);
        const float error = std::fabs(got - expected[row]);
        const float relative = error / std::max(1.0f, std::fabs(expected[row]));
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, relative);
        if (error > 1.0e-4f && relative > 1.0e-5f) {
            ++failures;
            if (failures <= 4) {
                std::printf("GEMV6_MISMATCH case=%s row=%d got=%.9g expected=%.9g "
                            "abs=%.9g rel=%.9g\n",
                            name, row, got, expected[row], error, relative);
            }
        }
    }
    std::printf("GEMV6_CASE name=%s n=%d k=%d status=%d max_abs=%.9g "
                "max_rel=%.9g result=%s\n",
                name, n, k, status, max_abs, max_rel,
                failures ? "FAIL" : "PASS");
    return failures == 0;
}

bool invalid_case() {
    std::vector<ap_uint<256> > w(4096);
    std::vector<ap_uint<256> > xq(304);
    std::vector<ap_uint<16> > xs(304);
    std::vector<gemv_output_word_t> y(37984);
    y[0] = ap_uint<128>(1234);
    const int a = q8q8_gemv_stream6(w.data(), w.data(), w.data(), w.data(),
            w.data(), w.data(), xq.data(), xs.data(), y.data(), 4, 33);
    const int b = q8q8_gemv_stream6(w.data(), w.data(), w.data(), w.data(),
            w.data(), w.data(), xq.data(), xs.data(), y.data(), 3, 32);
    const int c = q8q8_gemv_stream6(w.data(), w.data(), w.data(), w.data(),
            w.data(), w.data(), xq.data(), xs.data(), y.data(), 0, 32);
    const bool pass = a == GEMV_ERR_K_ALIGNMENT &&
            b == GEMV_ERR_N_ALIGNMENT && c == GEMV_ERR_DIMENSION &&
            y[0] == ap_uint<128>(1234);
    std::printf("GEMV6_CASE name=invalid a=%d b=%d c=%d unchanged=%d result=%s\n",
                a, b, c, y[0] == ap_uint<128>(1234), pass ? "PASS" : "FAIL");
    return pass;
}

}  // namespace

int main() {
    bool pass = true;
    pass &= run_case("small_partial", 4, 32);
    pass &= run_case("one_group", 48, 64);
    pass &= run_case("group_boundary", 52, 96);
    pass &= run_case("multiple_groups", 100, 64);
    pass &= run_case("robobrain_k2560", 48, 2560);
    pass &= run_case("vision_k1024", 48, 1024);
    pass &= invalid_case();
    std::printf("GEMV6_RESULT=%s cases=7\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
