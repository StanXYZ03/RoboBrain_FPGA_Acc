# PL-resident FFN unified vector engine

This independent HLS branch implements the vector operations required between
the board-validated PE16 DATAFLOW GEMMs. One mode-selected engine covers:

- RMSNorm or LayerNorm followed by GGML-compatible Q8_0 requantization;
- SiLU(gate) multiplied by up, followed by Q8_0 requantization;
- GELU followed by Q8_0 requantization;
- bias-aware residual addition to FP32.

The activation functions use a 65-entry, 0.25-step piecewise-linear table over
`[-8, 8]` with saturated asymptotes. The table is intentionally evaluated
against exact SiLU/GELU before any model-level accuracy claim. Its purpose is
to avoid synthesizing exp/tanh/erf datapaths while keeping intermediate FFN
tensors on the PL side; it is not yet a proven paper contribution.

The AXI datapath packs four FP32 lanes into 128 bits and writes each Q8_0 block
as one 256-bit payload plus one FP16 scale. Three AXI masters are exposed:
`gmem0` carries source 0 and the parameter preload, `gmem1` carries source 1,
and `gmem2` carries all results. This preserves concurrent dual-input reads and
output writes without paying for seven independent AXI adapters. Parameter
vectors are cached in partitioned BRAM because XCZU9EG does not provide
UltraRAM. The ABI is task-independent and contains no RoboBrain task name.
The two optional parameter vectors share `gmem0`, so their one-time preload is
explicitly scheduled at `II=2`; data processing still keeps source 0, source 1,
and output traffic on three independent masters.

Activation evaluation remains four lanes wide, while the Q8_0 block reduction
and rounding are intentionally single-lane. Requantization is not on the GEMM
critical path in the current block model, so this saves LUTs for the matrix
engine rather than maximizing an isolated vector microbenchmark.

The FP32 row-statistics accumulation has a five-cycle recurrence and therefore
uses an explicit `II=5` contract. This is a deliberate resource/latency trade:
for RoboBrain's real hidden sizes its contribution is in the millisecond range,
well below the Q8 GEMMs, and the synthesis report must not imply an unattainable
`II=1` constraint.

Run C simulation and synthesis:

```bash
make -C fpga_accel/pl_resident_ffn_hls csim
make -C fpga_accel/pl_resident_ffn_hls csynth
```

This branch does not program the board. It must pass C simulation, HLS
synthesis, RTL/full-system timing, a complete FFN software-model comparison,
and board measurement before it can be described as a PL-resident FFN result.

The subsequent real-weight proxy gate is recorded in
`artifacts/real_weight_proxy_20260831_v2/summary.json`. It uses all real layer-0
RoboBrain Q8 FFN weights and a compact real-weight slice in HLS C simulation.
Its token-embedding inputs are model-derived proxies, not captured runtime
hidden states, so runtime-activation validation remains open.
