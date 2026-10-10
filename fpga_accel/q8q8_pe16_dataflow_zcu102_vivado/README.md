# ZCU102 Q8×Q8 PE16 DATAFLOW full-system candidate

This independent branch replaces only the board-validated PE16 HLS IP with
`robobrain.local:hls:q8q8_pe16_dataflow:1.0`. The ZCU102 PS, HPC0/HPC1/HP0
topology, 128-bit output path, AXI-Lite register layout, and 200 MHz clock are
kept compatible with the frozen PE16 runtime.

Timestamped validation and routed-implementation gates are:

```bash
./scripts/run_stage.sh validate
./scripts/run_stage.sh impl
./scripts/run_bitstream.sh
```

The `impl` stage stops after route, DRC, methodology, power, and checkpoint
generation. `run_bitstream.sh` only exports timestamped offline bitstream/XSA
artifacts from the passing routed run. Neither script accesses the board.
