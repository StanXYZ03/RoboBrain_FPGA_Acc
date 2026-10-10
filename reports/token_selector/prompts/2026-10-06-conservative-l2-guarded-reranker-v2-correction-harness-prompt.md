# Harness correction task: pre-execution STOP v1 and preregister guarded reranker v2

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

This is CPU-only. Do not run GPU/model weights/network/training, do not install dependencies, do not edit original RoboBrain model/app, and do not overwrite any frozen v1 source/config/report/result. Preserve the complete v1 audit trail and mark it `PREEXECUTION_DESIGN_STOP` before any GPU run.

## Independently confirmed blocking defect

The preregistered v1 ranks swaps by:

`(semantic_q15[j] - L2_q15[j]) - (semantic_q15[i] - L2_q15[i])`

for selected `i` and same-region candidate `j`. This has the wrong sign for an L2-protecting mechanism. Counterexample:

- `semantic_i = semantic_j = 10000`
- `L2_i = 30000`, `L2_j = 1000`
- v1 gain = `29000 > 0`, so it accepts a candidate with no semantic improvement and severe L2 loss.

Do not patch v1 in place and do not run it. Add a concise v1 pre-execution STOP report JSON/MD that pins the old config/source hashes and records this exact counterexample. Update the top of `CURRENT_STATUS.md` without rewriting history.

## Single amended mechanism

Create `conservative_l2_guarded_rerank_v2` as an additive implementation. Reuse v1's validated baseline selection and layout rebuild helpers where safe, but keep v1 byte-identical.

For each group `g`:

- `semantic_q15[g]`: accepted integer semantic Q0.15;
- `l2_q15[g]`: corrected integer patch-L2 Q0.15;
- `utility_q16[g] = semantic_q15[g] + l2_q15[g]`, range `[0, 65534]`, stored in widened integer;
- for selected non-anchor `i` and same-region unselected candidate `j`:
  - `semantic_gain = semantic_q15[j] - semantic_q15[i]`;
  - `l2_loss = l2_q15[i] - l2_q15[j]`;
  - `legal_gain = semantic_gain - l2_loss = utility_q16[j] - utility_q16[i]`.

Candidate enumeration must rank unselected groups by `utility_q16` descending, lower original group index first, capped by the existing per-region `local_candidate_cap=16`. Enumerate same-region swaps, accept only `legal_gain > 0`, sort globally by `legal_gain` descending then `(added_group_id, removed_group_id, region_id)` ascending, apply at most `S=16` nonconflicting one-for-one swaps. Freeze corrected-L2 anchors, preserve per-region real counts, exact K96 layout and L2-weighted context pooling. No labels, answers, gradients, task/source IDs, new training, floating lambda, threshold sweep or parameter sweep.

Required invariant tests before GPU:

1. Flat semantic Q0.15 over all groups yields exactly zero swaps for valid corrected-L2 baseline selection.
2. The explicit counterexample above yields no swap.
3. Every applied swap satisfies `legal_gain == semantic_gain - l2_loss == utility_added - utility_removed > 0` in widened signed arithmetic.
4. Anchors, per-region selected-real counts, K96, current full-size `87 real + 9 context`, nonempty context pools and corrected-L2 pooling remain exact.
5. Zero-swap v2 remains byte/numerically equivalent to corrected integer-L2 selection/layout.
6. Deterministic ties follow `(added_id, removed_id, region_id)` after descending gain.
7. Explicitly verify the accepted semantic Q0.15 and corrected patch-L2 Q0.15 source files/checkpoint/rounding/saturation hashes; do not infer them from the older mean-of-patch float-L2 helper.

## Additive deliverables

Use clear v2 names; do not modify v1 files:

- `evaluation/conservative_l2_guarded_reranker.py`
- real-model v2 runner, reusing the already tested B1 integration if its scorer interface accepts a prebuilt `HardwareSelection`; otherwise add a minimal v2 integration without changing v1;
- fail-closed v2 full2638 development summarizer;
- focused tests under `evaluation/tests/`;
- `configs/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json`;
- Chinese v1 STOP report JSON/MD;
- Chinese v2 preregistration report JSON/MD;
- update `CURRENT_STATUS.md`.

The six-cell smoke remains contract-only. The single full2638 pass remains contaminated development triage only.

## Corrected decision gates

Use the exact pinned dense score from the frozen report: `0.8727298774444869`; calculate and store `0.9 * dense = 0.7854568897000382` rather than a rounded hand-entered approximation.

GO to fresh external validation requires all:

1. 2638 unique complete and parseable;
2. exact layout/bypass/leakage contracts;
3. official score `>= 0.7854568897000382`;
4. paired bootstrap 95% CI lower bound for v2 minus corrected integer-L2 `>= 0.0`, 5000 repeats, seed 20261006.

The prior prompt-fusion score `0.7773617325561999` is a descriptive strongest same-budget development comparator, not a redundant gate because the 90%-dense gate already implies it.

Passing means only `DEVELOPMENT_GO_EXTERNAL_VALIDATION`; failing means `DEVELOPMENT_STOP`. Never call it paper confirmation, official untouched validation, FPGA execution, acceleration or latency evidence.

## Verification

- py_compile all v2 files/tests;
- focused v2 tests;
- full `evaluation/tests` suite with exact count and real exit code;
- v1 source/config/report hashes byte-identical before/after;
- all existing frozen inputs byte-identical;
- v2 summarizer must fail closed with exit code 2 and write no formal report while v2 arm is absent;
- `gpu_runs=0`.

Return compact JSON only with `status`, `v1_stop`, `v2_contract`, `files_created`, `files_modified`, `v1_hashes_unchanged`, `pins_verified`, `focused_tests`, `full_tests`, `gpu_runs`, `prereg_status`, `next_host_commands`, `limitations`.
