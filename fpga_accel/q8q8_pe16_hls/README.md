# Q8×Q8 PE_N=16 HLS research candidate

This directory is an independent branch of the board-validated
`q8q8_output_burst_hls` v3 kernel. It changes only output-channel
parallelism from 8 to 16 and adds a full-64-column numerical case.

The separate `q8q8_pe16_zcu102_vivado` system has met routed timing and
resource gates at 200 MHz. No PE16 bitstream or board result is claimed here,
and this kernel is not integrated into the RoboBrain Transformer runtime.

Run C simulation:

```bash
./scripts/run_stage.sh csim
```

Run C simulation and synthesis for XCZU9EG at 200 MHz:

```bash
./scripts/run_stage.sh csynth
python3 scripts/summarize_hls.py
```

Run the directed XSim check for the HLS-generated PE16 compute controller:

```bash
./scripts/run_compute_rtl_smoke.sh
python3 scripts/summarize_rtl_validation.py
```

The directed RTL check validates the interleaved controller, 16-bank
accumulator addressing, valid-row gating, and `ap_ctrl_hs` handshake.
Arithmetic submodules are zero-data behavioral stubs in this check, so
numerical correctness remains covered by C simulation. Full top-level HLS AXI
cosim is not claimed: Vitis HLS 2023.2 generated about 30 GiB of AXI adapter
Verilog and was stopped before an out-of-memory condition.

Outputs:

- `artifacts/vitis_hls_csynth.log`
- `artifacts/pe16_hls_summary.json`
- `artifacts/pe16_rtl_validation_summary.json`
- `../../reports/2026-08-27-zcu102-q8q8-pe16-hls.md`
- `../../reports/2026-08-27-zcu102-q8q8-pe16-rtl-validation.md`

The summary compares HLS pre-place estimates against PE8 and uses the frozen
v3 UART board results only for a first-order latency projection. Projection is
not a PE16 board measurement.
