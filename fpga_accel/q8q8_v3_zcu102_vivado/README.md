# ZCU102 Q8×Q8 v3 output-burst system

This project integrates `q8q8_output_burst:3.0` with the ZCU102 PS and its
DDR through the same three-port topology used by the board-verified v2 system.
The only data-layout change inside the IP is that four adjacent FP32 outputs
are packed into one 128-bit AXI write beat; the DDR row-major output layout and
AXI4-Lite register offsets remain compatible with the runtime.

The targets validate, synthesize, route, or additionally export a bitstream.
None of them programs the board.

```bash
make validate
make synth
make impl
make bitstream
```

Board programming is permitted only after the C tests, directed RTL/XSim test,
block-design validation, and full 200 MHz system timing gates have passed.
