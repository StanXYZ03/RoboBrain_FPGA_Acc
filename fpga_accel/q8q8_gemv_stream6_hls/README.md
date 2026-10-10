# Six-port Q8_0 Decoder GEMV baseline

This independent HLS branch targets batch-one Decoder weight streaming. It does
not replace the board-validated PE16 DATAFLOW GEMM.

The activation vector is cached once. Every group covers 48 output rows, striped
over six independent weight ports. One port record per K-block is nine 256-bit
words: one header carrying eight FP16 scales followed by eight Q8 payload blocks.
Thus a port transfers 288 bytes for 272 bytes of native Q8_0 content (94.44%
packing efficiency) using only contiguous accesses. The six ports are intended
to expose aggregate bandwidth; actual ZCU102 DDR throughput can only be claimed
after full-system routing and APM board measurement.

```bash
make -C fpga_accel/q8q8_gemv_stream6_hls csim
make -C fpga_accel/q8q8_gemv_stream6_hls csynth
make -C fpga_accel/q8q8_gemv_stream6_hls cosim
make -C fpga_accel/q8q8_gemv_stream6_hls summarize
make -C fpga_accel/q8q8_gemv_stream6_hls export
make -C fpga_accel/q8q8_gemv_stream6_hls export_impl
make -C fpga_accel/q8q8_gemv_stream6_hls summarize_impl
```

The weight reader and the arithmetic pipeline are separated with DATAFLOW FIFOs.
This permits each of the six contiguous 256-bit weight streams to infer an AXI
burst while the nine-phase compute loop (one scale header plus eight Q8 blocks)
maintains II=1. Passing C/RTL simulation and HLS synthesis is not a bitstream or
board performance result. No board action is part of these targets.
