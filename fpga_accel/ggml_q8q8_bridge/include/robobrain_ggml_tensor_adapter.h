#ifndef ROBOBRAIN_GGML_TENSOR_ADAPTER_H
#define ROBOBRAIN_GGML_TENSOR_ADAPTER_H

#include "ggml.h"
#include "robobrain_ggml_q8q8_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

enum rb_ggml_adapter_status {
    RB_GGML_ADAPTER_OK = 0,
    RB_GGML_ADAPTER_NOT_MUL_MAT = 1,
    RB_GGML_ADAPTER_MISSING_SOURCE = 2,
    RB_GGML_ADAPTER_VIEW_UNSUPPORTED = 3,
    RB_GGML_ADAPTER_TYPE_UNSUPPORTED = 4,
    RB_GGML_ADAPTER_NULL_ARGUMENT = -1,
};

/* Converts the public ggml_tensor representation into the stable project ABI.
 * It performs no dispatch and therefore is safe to call before fallback. */
int rb_ggml_tensor_to_q8q8_view(const struct ggml_tensor *dst,
                                struct rb_ggml_mul_mat_view *operation);

const char *rb_ggml_adapter_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif
