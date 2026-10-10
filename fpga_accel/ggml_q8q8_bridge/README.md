# GGML Q8_0 → ZCU102 PL bridge

This project-owned adapter translates the tensor contract of GGML `MUL_MAT`
into the existing Q8×Q8 PL runtime without changing `profiling_src/llama.cpp`.

Supported initial path:

- `src0`: contiguous GGML Q8_0 weights, logical shape `[K,N,1,1]`;
- `src1`: contiguous Q8_0 or FP32 activations, shape `[K,M,1,1]`;
- `dst`: contiguous FP32 output, shape `[N,M,1,1]`;
- mathematical operation: `Y[M,N] = X[M,K] * W[N,K]^T`;
- no batch, broadcast, transpose, view, or architecture-specific repack;
- unsupported calls return a reason and must use the normal CPU path.

FP32 activations are quantized into the upstream-compatible 34-byte Q8_0
block contract. Non-finite activations are rejected rather than sent to PL.
`robobrain_ggml_tensor_adapter` converts the public `ggml_tensor` structure to
this contract and rejects views before the PL classifier runs.

Run the host, sanitizer, upstream-layout, byte-for-byte upstream quantizer, and
AArch64 build gates:

```sh
make -C fpga_accel/ggml_q8q8_bridge all \
  ARTIFACTS=artifacts/ggml_bridge_20260830
```

Passing these tests proves format and dispatch compatibility only. It is not
evidence that llama.cpp has been integrated or that the board is faster.
