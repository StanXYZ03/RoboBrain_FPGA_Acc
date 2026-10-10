# integer_selector_minimal_zynq_bd

Minimal Vivado 2023.2 ZCU102 block design for the accepted compact integer
Selector IP (`robobrain.local:hls:integer_selector_pipeline_compact:1.0`).

* `scripts/build_bd.tcl` – builds / validates / addresses / generates / wraps the BD.
* `scripts/synth_bd.tcl` – bounded synthesis of the wrapper (no implementation, no bitstream).
* `scripts/run_bd.sh` – single-job driver; all large files under `/tmp`, only bounded
  text evidence is copied into `artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd/`.
* `scripts/protected_sha.py` – before/after SHA-256 audit of the frozen Selector trees.
* `scripts/summarize_bd.py` – writes the JSON / Markdown reports and harness result.

Topology:

```
PS M_AXI_HPM0_FPD (128b AXI4) -- smartconnect_control -- Selector s_axi_control (AXI4-Lite 32b)
Selector m_axi_input_arena  (128b AXI4) --\
Selector m_axi_wv_q         (128b AXI4) --- smartconnect_ddr (3->1) -- PS S_AXI_HP0_FPD (128b AXI4, DDR)
Selector m_axi_output_arena (128b AXI4) --/
PS pl_clk0 (IOPLL, 200 MHz) -- fabric clock ; pl_resetn0 -- proc_sys_reset -- peripheral_aresetn
```

The three Selector masters are arbitrated by the SmartConnect and are **not** merged
at the IP boundary.  No DMA is inserted: the Selector masters read/write DDR directly.

Run:

```bash
bash fpga_accel/integer_selector_minimal_zynq_bd/scripts/run_bd.sh
```
