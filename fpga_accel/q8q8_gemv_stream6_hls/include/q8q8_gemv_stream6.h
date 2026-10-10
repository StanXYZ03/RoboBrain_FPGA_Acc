#ifndef ROBOBRAIN_Q8Q8_GEMV_STREAM6_H
#define ROBOBRAIN_Q8Q8_GEMV_STREAM6_H

#include <ap_int.h>

constexpr int GEMV_QK = 32;
constexpr int GEMV_PORTS = 6;
constexpr int GEMV_SLOTS_PER_PORT = 8;
constexpr int GEMV_GROUP_N = GEMV_PORTS * GEMV_SLOTS_PER_PORT;
constexpr int GEMV_RECORD_WORDS = 1 + GEMV_SLOTS_PER_PORT;
constexpr int GEMV_OUTPUT_LANES = 4;
constexpr int GEMV_MAX_N = 151936;
constexpr int GEMV_MAX_K = 9728;
constexpr int GEMV_MAX_K_BLOCKS = GEMV_MAX_K / GEMV_QK;

static_assert(GEMV_GROUP_N == 48, "frozen six-port group contract changed");
static_assert(GEMV_RECORD_WORDS == 9, "one header plus eight payloads expected");

enum q8q8_gemv_status {
    GEMV_OK = 0,
    GEMV_ERR_DIMENSION = -1,
    GEMV_ERR_K_ALIGNMENT = -2,
    GEMV_ERR_N_ALIGNMENT = -3,
};

typedef ap_uint<128> gemv_output_word_t;

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
        int k);

#endif
