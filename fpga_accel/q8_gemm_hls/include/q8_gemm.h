#ifndef ROBOBRAIN_Q8_GEMM_H
#define ROBOBRAIN_Q8_GEMM_H

#include <ap_int.h>

// GGML Q8_0 stores 32 signed int8 values and one IEEE-754 binary16 scale.
// The accelerator keeps these two fields in independent, naturally aligned
// AXI planes.  Repacking is bit-exact; it does not requantize the model.
constexpr int Q8_GEMM_QK       = 32;
constexpr int Q8_GEMM_TM       = 16;
constexpr int Q8_GEMM_TN       = 64;
constexpr int Q8_GEMM_PE_N     = 4;
constexpr int Q8_GEMM_MAX_M    = 4320;
constexpr int Q8_GEMM_MAX_N    = 9728;
constexpr int Q8_GEMM_MAX_K    = 9728;

// Interface depth is a co-simulation test-vector hint only.  It does not
// constrain the address range of the synthesized AXI masters.
constexpr int Q8_GEMM_TB_WEIGHT_DEPTH = 512;
constexpr int Q8_GEMM_TB_X_DEPTH      = 8192;
constexpr int Q8_GEMM_TB_Y_DEPTH      = 2048;

enum q8_gemm_status {
    Q8_GEMM_OK             = 0,
    Q8_GEMM_ERR_DIMENSION  = -1,
    Q8_GEMM_ERR_K_ALIGNMENT = -2,
};

extern "C" int q8_gemm(
        const ap_uint<256> * weights_q,
        const ap_uint<16>  * weights_scale,
        const float        * activations,
        float              * output,
        int m,
        int n,
        int k);

#endif
