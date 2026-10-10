# Heterogeneous partition gate

This analysis freezes the RoboBrain language-model dimensions and tensor byte
counts extracted read-only from the verified Q8_0 GGUF. It computes payload-only
communication lower bounds for candidate Jetson/FPGA boundaries and bandwidth-only
Decoder token-rate ceilings. Link rates other than the prior ZCU102 DDR observation
are planning scenarios, not measurements.

```bash
make -C fpga_accel/heterogeneous_partition test
make -C fpga_accel/heterogeneous_partition analyze \
  OUTPUT_DIR=artifacts/partition_gate_YYYYMMDD
```

The output directory is exclusive-create and contains machine-readable JSON and
a Chinese Markdown report. The analysis neither loads CUDA nor accesses the board.
