#include "robobrain_ggml_tensor_adapter.h"

#include <string.h>

static int translate_type(enum ggml_type source,
                          enum rb_ggml_scalar_type *destination) {
    if (source == GGML_TYPE_F32) {
        *destination = RB_GGML_TYPE_F32;
        return 0;
    }
    if (source == GGML_TYPE_Q8_0) {
        *destination = RB_GGML_TYPE_Q8_0;
        return 0;
    }
    return -1;
}

static int translate_tensor(const struct ggml_tensor *source,
                            struct rb_ggml_tensor_view *destination) {
    size_t dimension;
    if (translate_type(source->type, &destination->type) != 0) return -1;
    destination->data = source->data;
    for (dimension = 0; dimension < 4; ++dimension) {
        if (source->ne[dimension] < 0) return -1;
        destination->ne[dimension] = (uint64_t)source->ne[dimension];
        destination->nb[dimension] = source->nb[dimension];
    }
    return 0;
}

int rb_ggml_tensor_to_q8q8_view(const struct ggml_tensor *dst,
                                struct rb_ggml_mul_mat_view *operation) {
    const struct ggml_tensor *src0;
    const struct ggml_tensor *src1;
    if (dst == NULL || operation == NULL) return RB_GGML_ADAPTER_NULL_ARGUMENT;
    memset(operation, 0, sizeof(*operation));
    if (dst->op != GGML_OP_MUL_MAT) return RB_GGML_ADAPTER_NOT_MUL_MAT;
    src0 = dst->src[0];
    src1 = dst->src[1];
    if (src0 == NULL || src1 == NULL) return RB_GGML_ADAPTER_MISSING_SOURCE;
    if (src0->view_src != NULL || src1->view_src != NULL ||
            dst->view_src != NULL) return RB_GGML_ADAPTER_VIEW_UNSUPPORTED;
    if (translate_tensor(src0, &operation->src0_weights) != 0 ||
            translate_tensor(src1, &operation->src1_activations) != 0 ||
            translate_tensor(dst, &operation->dst_f32) != 0) {
        memset(operation, 0, sizeof(*operation));
        return RB_GGML_ADAPTER_TYPE_UNSUPPORTED;
    }
    return RB_GGML_ADAPTER_OK;
}

const char *rb_ggml_adapter_status_string(int status) {
    switch (status) {
        case RB_GGML_ADAPTER_OK: return "ok";
        case RB_GGML_ADAPTER_NOT_MUL_MAT: return "not_mul_mat";
        case RB_GGML_ADAPTER_MISSING_SOURCE: return "missing_source";
        case RB_GGML_ADAPTER_VIEW_UNSUPPORTED: return "view_unsupported";
        case RB_GGML_ADAPTER_TYPE_UNSUPPORTED: return "type_unsupported";
        case RB_GGML_ADAPTER_NULL_ARGUMENT: return "null_argument";
        default: return "unknown";
    }
}
