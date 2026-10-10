# Q8_0 GEMM/GEMV HLS prototype

This is the first PL compute prototype selected by the measured ZCU102
operator profile.  It computes `Y[M,N] = X[M,K] * W[N,K]^T` for GGML Q8_0
weights and FP32 activations/accumulators.

## Architecture

- Q8_0 block: 32 signed int8 values plus one FP16 scale.
- Lossless accelerator layout: one 256-bit quant word and one 16-bit scale
  word per block.  This avoids unaligned 34-byte AXI records.
- Tiles: `TM=16`, `TN=64`, `K-block=32`.
- Four parallel output PEs; each PE uses a balanced 32-element dot tree.
- Runtime dimensions support both GEMV (`M=1`) and GEMM, including the
  RoboBrain dimensions `K=1024`, `2560`, `4096`, and `9728`.
- Four independent AXI masters for quant weights, scales, activations, and
  output.  AXI/DMA integration is intentionally deferred until the compute
  core passes simulation and synthesis.

The separated weight planes preserve the exact GGUF FP16 scales and int8
values.  They are a storage-layout conversion, not a new quantization.

## Verification gates

```sh
make csim
make csynth
make cosim
```

The C testbench covers small GEMM, GEMV, tile boundaries, real RoboBrain
`K=2560`, vision `K=1024`, zero-scale blocks, int8 endpoints, and invalid
dimensions. RTL co-simulation uses a shorter suite that still crosses both M
and N tile boundaries and checks AXI/error behavior. Full
`512 x 9728 x 2560` computation is not executed in C/RTL simulation because
it contains 12.75 billion MACs; its latency is derived from the synthesized
loop schedule, then must be measured on hardware.

No bitstream is generated or programmed by these scripts.

## Verified result (2026-08-22)

- C simulation: PASS.
- HLS synthesis: estimated 3.650 ns, 655 DSP, 80 BRAM18K, 58,902 LUT,
  and 81,202 FF on `xczu9eg-ffvb1156-2-e`.
- Verilog C/RTL co-simulation: PASS for all five RTL transactions.
- The continuous compute loop achieved II=1. These remain pre-place-and-route
  results and are not board measurements.

The detailed assumptions and representative RoboBrain latency estimates are in
`../../reports/2026-08-22-zcu102-q8-gemm-hls-prototype.md`.
