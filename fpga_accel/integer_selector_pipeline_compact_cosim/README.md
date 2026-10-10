# `integer_selector_pipeline_compact_cosim`

Isolated, bounded Vitis HLS 2023.2 **C/RTL co-simulation** work area for the
already accepted compact integer Selector pipeline top
`integer_selector_pipeline_compact`.

This directory contains **only** the co-simulation harness.  It never edits the
accepted sources:

* design : `fpga_accel/integer_selector_pipeline_compact_hls/src/integer_selector_pipeline_compact.cpp`
* header : `fpga_accel/integer_selector_pipeline_compact_hls/include/integer_selector_pipeline_compact.h`
* fixtures: `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures/`

all read-only.

## What is proven

For exactly five representative accepted fixture cases (resolved verbatim from
the accepted `manifest.txt`), the generated RTL reproduces the frozen expected
`status`, `scale_out`, `output_count`, `real_count`, `context_count` and **every
byte** of the fixed output arena (tokens, metadata, alignment gaps, unused
budget entries, tail):

| slot | accepted case | G | grid | K |
|---|---|---|---|---|
| production non-identity G80/K48 | `synth_g80_K48_cap16` | 80 | 8x10 | 48 |
| real correct-prompt G260/K64 | `real_grounding_correct_K64_cap16` | 260 | 13x20 | 64 |
| production stress G400/K96 | `synth_g400_K96_cap16` | 400 | 20x20 | 96 |
| identity G80/K96 | `synth_g80_K96_cap16` | 80 | 8x10 | 96 |
| deterministic all-zero G128/K48 | `zero_g128_K48_cap16` | 128 | 8x16 | 48 |

## Layout

```
cases/rtl_cosim_cases.txt          five accepted manifest lines (verbatim)
tb/                                dedicated RTL-cosim testbench (no dlopen)
scripts/run_host_test.sh           focused host g++ + ASan/UBSan test
scripts/run_cosim.tcl              csynth_design + cosim_design (verilog/xsim/none)
scripts/run_cosim.sh               driver + 60-min-XSim watchdog + /tmp archiving
tools/validate_cases.py            resolves/validates the five slots
tools/protected_sha_audit.py       frozen protected-file SHA snapshot/verify
tools/build_report.py              JSON/Markdown/harness reports
```

## Reproduce

```bash
bash scripts/run_host_test.sh
bash scripts/run_cosim.sh
python3 tools/protected_sha_audit.py snapshot \
  --out ../../artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_after.json
python3 tools/protected_sha_audit.py verify \
  --before ../../artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_before.json \
  --after  ../../artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_after.json \
  --out    ../../artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_verify.json
python3 tools/build_report.py
```

The HLS project is created under `/tmp/robobrain_integer_selector_compact_cosim`
and is never copied into the repository; `scripts/run_cosim.sh` archives only
compact reports / transaction+performance summaries / metadata.

## Out of scope

No IP export, no Vivado implementation, no bitstream, no board, no GPU, no
network, no training.  Co-simulation proves RTL transaction behaviour only; it
does **not** claim routed timing, board latency, model end-to-end speed or task
accuracy.
