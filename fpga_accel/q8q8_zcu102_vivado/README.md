# ZCU102 Q8×Q8 DATAFLOW system integration

This project integrates the verified `q8q8_dataflow` HLS IP with the ZCU102
processing system and PS DDR through AXI SmartConnect instances.

Memory traffic is split across three independent PS slave ports:

- packed Q8 weights → `S_AXI_HPC0_FPD`;
- packed Q8 activations → `S_AXI_HPC1_FPD`;
- weight scales, activation scales and output → `S_AXI_HP0_FPD`.

The A53 controls the accelerator through `M_AXI_HPM0_FPD` and AXI4-Lite.
The PS supplies a 99.990005 MHz reference and an MMCM generates a
199.98001 MHz clock for all accelerator-side logic. Separate targets stop
after validation, synthesis, or routed implementation. The explicit
`bitstream` target additionally exports a `.bit` and matching `.xsa`; no
target programs the board.

```bash
make validate
make synth
make impl
make bitstream
make fsbl
```

`make bitstream` reruns implementation from the packaged, co-simulated HLS IP
and refuses export when setup timing, hold timing, or routing gates fail.
After `make bitstream`, `make fsbl` builds an FSBL from that exact XSA without
rerunning implementation. The FSBL is only an offline artifact; it is not
downloaded by the build.

The block design uses the official ZCU102 board preset. The installed 16 GiB
DIMM still requires the already validated SPD-aware boot-time DDR configuration;
this system project does not replace that runtime initialization.

The 2026-08-22 routed implementation meets the 200 MHz timing constraint with
WNS `+0.439 ns`, WHS `+0.010 ns`, and zero unrouted nets. The complete system
uses 19,947 LUTs, 28,539 flip-flops, 21.5 BRAM tiles, and 308 DSP blocks.
Vivado's vectorless power estimate is 4.250 W and is not a board measurement.
