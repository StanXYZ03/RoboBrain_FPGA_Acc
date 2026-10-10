#include "robobrain_ggml_tensor_adapter.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "TENSOR_ADAPTER_FAIL line=%d check=%s\n", \
                __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static void set_shape(struct ggml_tensor *tensor, int64_t ne0, int64_t ne1,
                      size_t nb0, size_t nb1) {
    tensor->ne[0] = ne0;
    tensor->ne[1] = ne1;
    tensor->ne[2] = 1;
    tensor->ne[3] = 1;
    tensor->nb[0] = nb0;
    tensor->nb[1] = nb1;
    tensor->nb[2] = nb1 * (size_t)ne1;
    tensor->nb[3] = tensor->nb[2];
}

int main(void) {
    uint8_t weights_data = 0;
    float activations_data = 0;
    float output_data = 0;
    struct ggml_tensor weights;
    struct ggml_tensor activations;
    struct ggml_tensor output;
    struct rb_ggml_mul_mat_view operation;
    struct rb_ggml_q8q8_report report;
    memset(&weights, 0, sizeof(weights));
    memset(&activations, 0, sizeof(activations));
    memset(&output, 0, sizeof(output));
    weights.type = GGML_TYPE_Q8_0;
    weights.data = &weights_data;
    set_shape(&weights, 2560, 9728, 34, 2560 / 32 * 34);
    activations.type = GGML_TYPE_F32;
    activations.data = &activations_data;
    set_shape(&activations, 2560, 512, sizeof(float), 2560 * sizeof(float));
    output.type = GGML_TYPE_F32;
    output.data = &output_data;
    output.op = GGML_OP_MUL_MAT;
    output.src[0] = &weights;
    output.src[1] = &activations;
    set_shape(&output, 9728, 512, sizeof(float), 9728 * sizeof(float));
    CHECK(rb_ggml_tensor_to_q8q8_view(&output, &operation) ==
          RB_GGML_ADAPTER_OK);
    CHECK(rb_ggml_q8q8_classify(&operation, 64u * 1024u * 1024u, 64,
                                &report) == 1);
    CHECK(report.m == 512 && report.n == 9728 && report.k == 2560);
    CHECK(report.activation_quantization_required == 1);
    weights.view_src = &weights;
    CHECK(rb_ggml_tensor_to_q8q8_view(&output, &operation) ==
          RB_GGML_ADAPTER_VIEW_UNSUPPORTED);
    weights.view_src = NULL;
    output.op = GGML_OP_ADD;
    CHECK(rb_ggml_tensor_to_q8q8_view(&output, &operation) ==
          RB_GGML_ADAPTER_NOT_MUL_MAT);
    puts("GGML_TENSOR_ADAPTER_PASS mapping=public_ggml_tensor "
         "view_fallback=yes upstream_modified=no board_accessed=no");
    return 0;
}
