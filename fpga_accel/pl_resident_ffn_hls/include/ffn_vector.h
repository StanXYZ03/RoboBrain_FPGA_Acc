#ifndef ROBOBRAIN_PL_RESIDENT_FFN_VECTOR_H
#define ROBOBRAIN_PL_RESIDENT_FFN_VECTOR_H

#include <ap_int.h>

constexpr int FFNV_LANES = 4;
constexpr int FFNV_QK = 32;
constexpr int FFNV_MAX_ROWS = 4320;
constexpr int FFNV_MAX_COLS = 9728;

typedef ap_uint<128> ffnv_f32x4_t;
typedef ap_uint<256> ffnv_q8x32_t;

enum ffnv_mode {
    FFNV_RMS_NORM_QUANT = 0,
    FFNV_LAYER_NORM_QUANT = 1,
    FFNV_SILU_MUL_QUANT = 2,
    FFNV_GELU_QUANT = 3,
    FFNV_RESIDUAL_ADD = 4,
};

enum ffnv_flags {
    FFNV_PARAM0_PRESENT = 1,
    FFNV_PARAM1_PRESENT = 2,
};

enum ffnv_status {
    FFNV_OK = 0,
    FFNV_ERR_DIMENSION = -1,
    FFNV_ERR_MODE = -2,
    FFNV_ERR_PARAMETER = -3,
};

extern "C" int ffn_vector(
        const ffnv_f32x4_t *src0,
        const ffnv_f32x4_t *src1,
        const ffnv_f32x4_t *param0,
        const ffnv_f32x4_t *param1,
        ffnv_q8x32_t *q_out,
        ap_uint<16> *scale_out,
        ffnv_f32x4_t *f32_out,
        int rows,
        int cols,
        float epsilon,
        int mode,
        int flags);

#endif
