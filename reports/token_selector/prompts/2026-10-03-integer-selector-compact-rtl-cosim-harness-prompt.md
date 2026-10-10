# Harness task: bounded RTL co-simulation of the accepted compact selector IP

Work only in:
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

Return one concise JSON object only after completing the work. Do not narrate intermediate reasoning.

## Authority and immutability

- Read `CURRENT_STATUS.md` and:
  - `reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.{md,json}`
  - `artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/`
- Freeze every pre-existing file under:
  - `fpga_accel/integer_selector_scorer_hls/`
  - `fpga_accel/token_selector_hls/`
  - `fpga_accel/integer_selector_pipeline_hls/`
  - `fpga_accel/integer_selector_pipeline_compact_hls/`
  - `evaluation/`
  - the existing compact-interface reports/artifacts/logs
- Record SHA-256 before/after and do not modify these files. Add only new paths requested below plus one concise top entry in `CURRENT_STATUS.md` after evidence exists.
- Do not change the accepted arithmetic, layout, HLS top, pragmas, reports, or fixtures.

## Objective

Run a bounded Vitis HLS 2023.2 C/RTL co-simulation of the already accepted top:
`integer_selector_pipeline_compact`
for `xczu9eg-ffvb1156-2-e`, Verilog RTL, XSim.

This phase proves that the synthesized RTL transaction behavior matches the frozen expected arenas. It does not claim routed timing, bitstream, board latency, model end-to-end speed, or task accuracy.

## New isolated work area

Create only:

- `fpga_accel/integer_selector_pipeline_compact_cosim/`
- `artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/`
- `reports/token_selector/2026-10-03-integer-selector-compact-rtl-cosim.{md,json}`
- `reports/token_selector/logs/2026-10-03-integer-selector-compact-rtl-cosim-harness-result.json`

The cosim project must reference the accepted compact source/header and accepted fixture payloads read-only.

## Testbench and fixture subset

Create a dedicated RTL-cosim testbench in the new directory. Do not use `dlopen` during RTL simulation. Read the frozen expected full arena and scalar results directly from accepted fixture files and compare:

- return status;
- `scale_out`, `output_count`, `real_count`, `context_count`;
- every byte of the fixed output arena, including tokens, all metadata, alignment gaps, unused budget entries, and final tail;
- input/output canaries where the cosim wrapper permits reliable checking.

Use exactly this bounded representative subset, resolving the corresponding accepted case names from the existing manifest:

1. production non-identity G80/K48;
2. real correct-prompt G260/K64;
3. production stress G400/K96;
4. identity G80/K96;
5. one deterministic all-zero or all-equal/tie case at a supported budget.

If an exact listed case is absent, fail with evidence; do not silently substitute a weaker case.

## Runtime and storage constraints

- Use `cosim_design -rtl verilog -tool xsim`.
- Disable waveform/trace dumping (`trace_level none` or the exact supported equivalent).
- Do not request VCD/WDB/SAIF waveform export.
- Keep the HLS project in `/tmp`; copy only compact logs, reports, transaction/performance summaries, and essential generated metadata into the artifact directory.
- Do not copy the complete simulator work directory or generated RTL tree into the repository.
- Record artifact total bytes and the 20 largest files.
- Do not use GPU, network, training, package installation, Vivado implementation, IP export, bitstream, or board access in this phase.

## Verification and gates

Before cosim, run focused host compilation/test of the dedicated testbench against the same five cases. Then run C/RTL cosim.

Decision may be `INTEGER_SELECTOR_COMPACT_RTL_COSIM_PASS` only when all are true:

- all protected files are byte-for-byte unchanged;
- focused host test passes all five cases with zero scalar/arena mismatches;
- XSim C/RTL cosim completes normally with zero simulation errors;
- all five RTL transactions pass with zero scalar/arena mismatches;
- the generated RTL interface still has exactly three 128-bit `m_axi` bundles and one AXI-Lite `control`, with no `control_r` and no external score port;
- trace/waveform dumping is disabled and repository artifacts stay bounded;
- no HLS/cosim fatal error is present.

If G400/K96 makes XSim exceed 60 minutes after simulation actually starts, terminate only this task's process, preserve the log and mark `INTEGER_SELECTOR_COMPACT_RTL_COSIM_BLOCKED_TIMEOUT`; do not weaken or omit the stress case and do not call it PASS.

## Required reports

The JSON/Markdown reports and final JSON stdout must include:

- exact five case names and G/grid/K values;
- host and RTL transaction counts, checks, mismatches, status/scalars;
- cosim tool, RTL language, simulator, elapsed wall time and per-transaction latency/cycle figures when Vitis reports them;
- generated interface bundle names/counts and data widths;
- cosim warnings/errors/fatals summary;
- trace setting and artifact total/largest-file audit;
- protected-file SHA audit;
- exact commands and reproduction steps;
- limitations distinguishing cosim from routed/board/end-to-end evidence;
- one recommended next action: export the accepted IP and run ZCU102 out-of-context synthesis/implementation only after this gate passes.

Do not overwrite the prior compact-interface PASS report. Do not report PASS from C-sim alone.
