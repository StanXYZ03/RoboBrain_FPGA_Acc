# Q8_0 x Q8_0 DATAFLOW GEMM/GEMV HLS prototype

This version preserves the verified integer kernel semantics and overlaps
activation reads, weight reads, and tile computation with HLS DATAFLOW and
two-tile FIFOs. It is separate from `q8q8_gemm_hls`, which remains the
non-overlapped reference.

The arithmetic is the pinned llama.cpp/GGML Q8_0 path:

`sum += dot_i8x32(weight.qs, activation.qs) * weight.d * activation.d`

Both operands use 256-bit INT8 quant planes and 16-bit FP16 scale planes. The
core uses `TM=16`, `TN=64`, eight N-direction PEs, an integer balanced dot
tree, FP32 block-scale fusion, and FP32 accumulation. The model weights are
unchanged; the activation quantizer is not yet integrated into this IP.

Run in order:

```sh
make csim
make csynth
make cosim
```

`make cosim` includes a narrowly scoped Vivado 2023.2 workaround. On this
host, XSim completes elaboration but crashes when it automatically loads the
generated DATAFLOW GUI waveform. The script preserves that waveform as
`.disabled`, retries without wave loading (and with `-nosignalhandlers` if
needed), runs the unchanged RTL snapshot, and then executes HLS's generated C
post-check. `make cosim-native` is available
to reproduce the native-tool failure. No target creates a bitstream or
programs the board.

## Verified result (2026-08-22)

- C simulation: PASS for six groups, including tile boundaries, K=2560,
  K=1024, invalid dimensions, and FP16 subnormal scales.
- HLS extracted three DATAFLOW tasks: activation reader, weight reader, and
  compute.
- Both readers achieve II=1 after replacing variable-loop FP16 normalization
  with an equivalent fixed priority encoder.
- Compute ingest: 66 cycles; eight-PE MAC stage: 150 cycles at II=1; complete
  K-block compute: 220 cycles; DATAFLOW interval: 222 cycles.
- Estimated clock: 4.185 ns (238.95 MHz before place-and-route).
- HLS pre-place estimate: 179 DSP, 180 BRAM18K, 24,764 LUT, 31,000 FF.
- Actual out-of-context routed implementation at 200 MHz: 308 DSP, 43
  BRAM18K (21.5 tiles), 15,549 LUT, 20,784 FF; WNS +0.873 ns and WHS
  +0.039 ns.
- Verilog RTL simulation: 5/5 transactions completed; generated C post-check:
  PASS for every transaction.

At a conservative 200 MHz, the scheduled per-output-tile cost is
`2143 + 220 * (K/32)` cycles. Representative projections are 0.480 s for
512x9728x2560, 0.793 s for 4320x4096x1024, and 15.0 ms for
1x9728x2560. These are HLS schedule projections, not board measurements.
