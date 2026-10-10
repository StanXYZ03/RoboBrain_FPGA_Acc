# Harness task: layer-5 score-formula replay

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

Implement one narrow, causal diagnostic. Do not run the 4B model or use GPU/network. Keep final response under 25 lines and report only changed files, tests, hashes, and blockers.

## Question

On the frozen CV-Bench screening100 set, can the historical `shallow_r0250` index sets be reproduced by changing only the per-group score from the current control formula

`norm(mean(x_0, x_1, x_2, x_3))`

to the historical formula

`mean(norm(x_0), norm(x_1), norm(x_2), norm(x_3))`?

For all frozen reference samples K>=16, so both paths use a 3x3 spatial anchor grid. Selection must otherwise be identical: one highest-score anchor in each non-empty 3x3 region, then stable global descending score with lower group index as tie-break, exact per-sample K_C from the reference, output indices sorted ascending.

## Constraints

- Do not modify any existing integration, runner, JSONL, report, model, checkpoint, manifest, or evaluation protocol.
- Add only new files for this experiment.
- CPU/import safe. No dependency installation.
- Reuse existing validated loaders/helpers where practical, but do not call the historical end-to-end selector as the candidate implementation.
- Fail closed on manifest/reference/control mismatch, duplicate/missing/extra IDs, task mismatch, index range, K/G mismatch, hash mismatch, non-finite activations, and unexpected identity bypass.
- Real replay later will be run by Codex, not Harness.

## Deliverables

1. `evaluation/layer5_score_formula_replay.py`
   - pure helpers for both score formulas;
   - candidate selector implementing historical mean-of-patch-L2 plus the exact fixed 3x3 anchor/global-fill policy;
   - comparison helpers (exact match, Jaccard, index hashes);
   - a vision-only unified layer-5 replay path, reusing the frozen `_unified_forward` only if this does not alter it; no LLM decode.

2. `evaluation/run_hf_layer5_score_formula_replay.py`
   - offline local BF16+SDPA runner for frozen screening100;
   - load model/processor once, run only through the candidate layer-5 vision path;
   - per record: sample/task, G, K_C, grid, reference/candidate/control indices and hashes, candidate==reference, candidate-vs-reference and control-vs-reference Jaccard, both score summaries, synchronized stage timing and peak GPU memory;
   - safe resume with strict provenance validation;
   - default output under `artifacts/token_selector/2026-10-04-layer5-score-formula-replay/`.

3. `evaluation/summarize_layer5_score_formula_replay.py`
   - validate exactly 100 expected records and all provenance;
   - write JSON/CSV/Markdown under `reports/cv_bench/2026-10-04-layer5-score-formula-replay-screening100.*`;
   - primary decision only: 100/100 exact candidate/reference matches => `L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET`; otherwise `PARTIAL_OR_NOT_EXPLAINED`;
   - report exact count, mean/min Jaccard for candidate/reference and control/reference, mismatch IDs, G/K distributions, timing descriptive only.

4. Focused tests for formula arithmetic, cancellation case distinguishing formulas, exact tie behavior, fixed 3x3 selection equivalence to historical `structured_group_indices` across representative grids/K (including all reference shapes if cheap), validation/resume/summarizer fail-closed behavior.

5. A preregistration JSON in `configs/` pinning current source inputs, manifest/reference/control JSONL hashes, output path, exact decision rule, sample count 100, and diagnostic/non-deployable scope. Do not claim it immutable until Codex runs the experiment.

Run focused tests and `py_compile`. Do not run full unittest discovery. Do not update `CURRENT_STATUS.md`; Codex will do that after review.
