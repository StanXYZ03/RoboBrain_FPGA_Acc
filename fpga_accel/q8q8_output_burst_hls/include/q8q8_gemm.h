#ifndef ROBOBRAIN_Q8Q8_GEMM_H
#define ROBOBRAIN_Q8Q8_GEMM_H

#include <ap_int.h>

constexpr int Q8Q8_QK           = 32;
constexpr int Q8Q8_TM           = 16;
constexpr int Q8Q8_TN           = 64;
constexpr int Q8Q8_PE_N         = 8;
constexpr int Q8Q8_OUTPUT_LANES = 4;
constexpr int Q8Q8_MAX_M        = 4320;
constexpr int Q8Q8_MAX_N        = 9728;
#ifndef Q8Q8_MAX_K_VALUE
#define Q8Q8_MAX_K_VALUE 9728
#endif
constexpr int Q8Q8_MAX_K        = Q8Q8_MAX_K_VALUE;
constexpr int Q8Q8_MAX_K_BLOCKS = Q8Q8_MAX_K / Q8Q8_QK;

constexpr int Q8Q8_TB_WEIGHT_DEPTH = 512;
constexpr int Q8Q8_TB_X_DEPTH      = 512;
constexpr int Q8Q8_TB_Y_DEPTH      = 2048;

enum q8q8_gemm_status {
    Q8Q8_OK              = 0,
    Q8Q8_ERR_DIMENSION   = -1,
    Q8Q8_ERR_K_ALIGNMENT = -2,
    Q8Q8_ERR_N_ALIGNMENT = -3,
};

typedef ap_uint<128> q8q8_output_word_t;

extern "C" int q8q8_gemm(
        const ap_uint<256> * weights_q,
        const ap_uint<16>  * weights_scale,
        const ap_uint<256> * activations_q,
        const ap_uint<16>  * activations_scale,
        q8q8_output_word_t * output,
        int m,
        int n,
        int k);

#endif
