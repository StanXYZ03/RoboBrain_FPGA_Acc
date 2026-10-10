# ZCU102 PL-resident FFN dual-IP full-system candidate

This independent Vivado branch integrates the routed PE16 DATAFLOW Q8 GEMM IP
and the routed unified FFN vector IP in one ZynqMP system. It does not modify
the board-validated DATAFLOW branch.

DDR topology at 200 MHz:

- HPC0: GEMM weight payload plus vector source 0 / parameter preload;
- HPC1: GEMM activation payload plus vector source 1;
- HP0: GEMM scales/output plus vector Q8/scale/FP32 output;
- HPM0: one control SmartConnect with independent GEMM and vector AXI-Lite
  address segments. The board-validated GEMM ABI remains at `0xA0000000`; the
  new vector controller is fixed at `0xA0010000`.

Run append-only gates:

```bash
make -C fpga_accel/pl_resident_ffn_zcu102_vivado validate
make -C fpga_accel/pl_resident_ffn_zcu102_vivado impl
```

`impl` stops after routed timing, route, DRC, methodology, power, and checkpoint
reports. It does not create a bitstream, reset/program the board, or touch SD/QSPI.

Frozen full-system result:

- 200 MHz route pass: WNS `+0.060 ns`, WHS `+0.009 ns`, unrouted nets `0`;
- 58,445 LUT, 88,156 FF, 483 BRAM tiles, 815 DSP;
- machine summary: `artifacts/pl_resident_ffn_system_summary.json`;
- no bitstream, board access, or SD/QSPI modification.
