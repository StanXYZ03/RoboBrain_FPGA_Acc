# Q8_0 x Q8_0 integer GEMM/GEMV HLS prototype

This prototype reproduces the scalar GGML `ggml_vec_dot_q8_0_q8_0`
semantics used by the pinned llama.cpp CPU backend:

`sum += dot_i8x32(weight.qs, activation.qs) * weight.d * activation.d`

Both Q8_0 operands are stored as naturally aligned 256-bit quant planes and
16-bit FP16 scale planes. Activation blocks are produced with the exact
`quantize_row_q8_0_ref` rule in the testbench; an integrated streaming
quantizer is a later stage. The model weights are unchanged.

The core uses `TM=16`, `TN=64`, eight parallel N PEs, an integer balanced dot
tree, FP32 block-scale fusion and FP32 output accumulation. It is kept separate
from the earlier `q8_gemm_hls` FP32-activation prototype for direct resource
and latency comparison.

Run, in order:

```sh
make csim
make csynth
make cosim
```

No bitstream is generated or programmed by these targets.

## Verified result (2026-08-22)

- C simulation: PASS for six test groups.
- HLS synthesis: compute loop II=1; estimated 4.185 ns; 179 DSP,
  148 BRAM18K, 24,942 LUT and 30,238 FF.
- Verilog C/RTL co-simulation: PASS for all five RTL transactions.
- Minimum scheduled K-block latency: 378 cycles, versus 982 cycles in the
  FP32-activation prototype.

These are pre-place-and-route results, not board measurements. The activation
quantizer and load/compute double buffering are intentionally the next design
stage.
