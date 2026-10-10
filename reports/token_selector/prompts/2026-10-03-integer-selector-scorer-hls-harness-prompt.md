# Harness task: integer Selector scorer HLS reference kernel

Implement, test and (only after tests pass) run controlled C synthesis for a
new standalone scorer kernel. Codex owns the architecture and gates. Keep the
final response terse and save evidence.

## Safety and scope

- Add files only under `fpga_accel/integer_selector_scorer_hls/`, its new
  report/artifact paths, tests if needed, and the final `CURRENT_STATUS.md`
  update. Do not modify prior Python audits, existing selector/HLS kernels,
  model/checkpoint/cache/manifest/config/report files.
- CPU/toolchain only. No GPU/network/4B/raw images/internal-test data.
- No cosim, export IP, implementation, bitstream or board access this round.
- Use Vitis HLS 2023.2, part `xczu9eg-ffvb1156-2-e`, 5.0 ns clock and default
  uncertainty. Do not relax timing or alter arithmetic to make reports pass.

## Exact top-level contract

Create top `integer_selector_scorer`:

- `group_mean_q`: row-major signed INT8 `[G,1024]`, packed 16 lanes per
  128-bit AXI word, little-endian lane bits.
- `wv_q`: row-major signed INT8 `[128,1024]`, same 128-bit packing. Cache once
  per invocation in on-chip memory; cyclic/banked layout may be used.
- `query_q`: signed INT8 `[128]`, packed as 8 x 128-bit words.
- `score_q`: uint16 `[G]`, Q0.15 range `[0,32767]`.
- scalar `group_count`, legal `1..4096`; dimensions 1024 and 128 are fixed.
- explicit status codes for invalid dimensions/nulls where host C++ permits.
- independent AXI bundles where useful and AXI-Lite control.

Arithmetic must be bit-identical to
`evaluation/selector_integer_scorer_audit.py`:

1. INT8xINT8 to INT32 visual projection.
2. INT64 semantic dot and norm2.
3. exact floor integer sqrt.
4. exact signed rational round-half-away for Q16 semantic ratio.
5. integer l2 sqrt, exact integer min-max to Q0.15, all-equal to zero.
6. lambda 0.5 combine by exact positive half-away rounding and clamp.

Never use float/double in DUT. Avoid signed-shift undefined behavior. Add static
assertions/comments for the frozen analytical bounds. Store semantic/l2 arrays
on chip for the two-pass min-max/output flow.

## Verification and fixtures

- Reuse accepted seed 20261003 Wv bundle by SHA and the grounding/pointing
  fixtures from `artifacts/token_selector/selector_integer_scorer_audit/`;
  convert them to compact C++-readable binaries with a documented header.
- Add deterministic synthetic cases: zero/all-equal, positive/negative signed
  division ties, G=1, illegal G=0/4097, and a G=400 stress fixture. Do not embed
  400 full random rows if a streaming deterministic generator plus expected
  hash/output is smaller.
- Python fixture generator must reuse the frozen integer golden, not copy its
  math. Testbench compares every `score_q` byte and status.
- Run Python fixture/unit checks, g++ normal test, ASan+UBSan, and Vitis HLS
  C-sim. All must pass before synthesis.

## Controlled synthesis and decision

Run only `csynth_design`. Archive small reports/XML/summary under
`artifacts/token_selector/2026-10-03-integer-selector-scorer-hls/`; keep HLS
project in `/tmp`.

Report:
- BRAM/URAM/DSP/LUT/FF and percentages;
- estimated period and effective timing budget/default uncertainty;
- bounded min/max latency if available;
- Wv load cycles, cycles/group, and derived G=80/260/400 latency at 200 MHz,
  clearly labelled as synthesis estimate, not measurement;
- whether output can connect directly to existing token selector `score_q`.

Decision `INTEGER_SELECTOR_SCORER_HLS_PASS` only if:

- all Python/g++/sanitizer/C-sim fixtures are byte-exact;
- csynth has no errors, timing meets the effective budget, every resource type
  is below 50% of ZCU102, and the report provides a finite defensible derived
  G=400 latency <=20 ms (<=4,000,000 cycles including Wv load).

Otherwise use `INTEGER_SELECTOR_SCORER_HLS_REDESIGN_REQUIRED` and identify the
actual bottleneck. Never claim board latency or complete end-to-end selector.

Write README, quantization/interface contract, register-map draft, Makefile,
scripts, source/header/testbench/tools, Chinese report
`reports/token_selector/2026-10-03-integer-selector-scorer-hls.md`, JSON summary,
four concise logs, and update `CURRENT_STATUS.md` only after verification.
