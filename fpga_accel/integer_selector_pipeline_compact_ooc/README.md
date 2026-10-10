# integer_selector_pipeline_compact_ooc

Isolated work area for exporting the **accepted** compact integer Selector
top (`integer_selector_pipeline_compact`) as a Vitis HLS 2023.2 IP catalog
component and running a Vivado 2023.2 **out-of-context** synthesis +
implementation on `xczu9eg-ffvb1156-2-e` with `ap_clk = 5.000 ns`.

This directory contains **scripts and notes only**.  The accepted source is
read-only; nothing here edits `integer_selector_pipeline_compact_hls/`,
the other protected HLS directories, `evaluation/`, or any prior report,
log, prompt, fixture or artifact.

## Flow

```
Vitis HLS 2023.2
  run_hls_export.tcl   csynth_design
                       export_design -rtl verilog -format ip_catalog -flow syn
                       (top integer_selector_pipeline_compact,
                        part xczu9eg-ffvb1156-2-e,
                        create_clock -period 5.0 -name default)

Vivado 2023.2 (OOC)
  run_vivado_ooc.tcl   read exported hdl/verilog RTL + constraints/*_ooc.xdc
                       synth_design -mode out_of_context
                       opt_design
                       place_design
                       phys_opt_design
                       route_design
                       (reports only; no write_bitstream, no board)
```

All HLS/Vivado projects, caches, checkpoints and the IP repo stay under the
private `/tmp` of the single driver process.  Only bounded text/JSON evidence
is copied into
`artifacts/token_selector/2026-10-03-integer-selector-compact-ip-ooc-route/`.

## Run

```bash
bash fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_ooc.sh
```

The driver must run as one process group because the sandbox gives every
command its own private `/tmp`; HLS and Vivado therefore share one job.

## Files

| file | purpose |
| --- | --- |
| `scripts/run_ooc.sh` | single-job driver: HLS export -> Vivado OOC -> evidence copy |
| `scripts/run_hls_export.tcl` | Vitis HLS `csynth_design` + `export_design -format ip_catalog -flow syn` |
| `scripts/run_vivado_ooc.tcl` | explicit OOC synth/opt/place/phys_opt/route + reports |
| `scripts/protected_sha.py` | protected-file SHA-256 snapshot/compare |
| `scripts/summarize_ooc.py` | parse archived evidence, evaluate gates, write JSON/MD/harness result |

## Decision gates

`INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS` requires all of: protected files
byte-identical; HLS csynth + IP export with no error/fatal; OOC synth, opt,
place and route successful; post-route WNS >= 0 and WHS >= 0 at 5.000 ns; no
unrouted nets; no DRC ERROR / CRITICAL WARNING; LUT/FF/BRAM/DSP/URAM each
below 80 %; generated 3x128-bit AXI4 master + 1 AXI4-Lite control interface
(no `control_r`, no `score_q` port); exact part and 5.000 ns clock; and no
bitstream.

## Limits

OOC only: no PS, no DDR, no AXI interconnect contention, no block design, no
bitstream, no board measurement, and no end-to-end or task-accuracy claim.
