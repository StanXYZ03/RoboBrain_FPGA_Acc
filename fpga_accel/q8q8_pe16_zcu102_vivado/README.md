# ZCU102 Q8×Q8 PE16 full-system candidate

This project replaces only the board-validated v3 HLS IP with
`robobrain.local:hls:q8q8_pe16:1.0`. The ZCU102 PS, HPC0/HPC1/HP0
topology, 128-bit output path, AXI-Lite register layout, and 200 MHz clock are
kept compatible with the v3 runtime.

The default research gate stops after routed implementation:

```bash
make validate
make impl
python3 scripts/summarize_impl.py
```

`make impl` does not generate a bitstream and does not program the board.
Bitstream generation and board microbenchmarking are separate gated stages.
