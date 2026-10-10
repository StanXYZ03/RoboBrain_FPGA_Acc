# ZCU102 six-stream Decoder GEMV integration

This is the minimal full-system integration branch for the independent
six-stream Q8_0 Decoder GEMV. The six weight masters map one-to-one onto
HPC0, HPC1, HP0, HP1, HP2 and HP3 through width-converting SmartConnects.
The low-traffic activation/output master shares the HPC0 interconnect with
weight port 0. The accelerator and all PL-side AXI paths target 200 MHz.

```bash
make -C fpga_accel/q8q8_gemv_stream6_zcu102_vivado validate
make -C fpga_accel/q8q8_gemv_stream6_zcu102_vivado synth
make -C fpga_accel/q8q8_gemv_stream6_zcu102_vivado impl
```

These targets validate, synthesize and route the design only. They do not
write a bitstream, program the board, reset the board or modify the SD card.
