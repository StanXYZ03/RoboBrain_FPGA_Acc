# Harness task: B1 answer-margin Oracle challenge/control pilot24 v2

You are the implementation executor. Work only in:
`/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`

Keep the response compact. Implement, test, and report facts only. Do not propose a different research direction.

## Safety and scope

- CPU-only in this task: no CUDA/GPU/model-weight loading/network/training/dependency install/sudo/kill.
- Do not modify the original RoboBrain model, `app`, frozen historical configs/results, or the existing random pilot24 v1 files.
- New artifacts must use the distinct stem `2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24`.
- Local tokenizer/processor use is allowed only if needed and must be `local_files_only=True`, offline.
- Ground-truth and existing dense/L2 outputs are intentionally used for outcome-enriched diagnostic sampling. Mark this everywhere as label/outcome leakage, non-deployable, non-representative, and forbidden for official absolute accuracy claims.

## Fixed diagnostic design

Create a deterministic 24-sample manifest from `datasets/cv_bench_full2638.jsonl`, with 4 samples in each of these six source x task cells:

1. ADE20K x Count
2. ADE20K x Relation
3. COCO x Count
4. COCO x Relation
5. Omni3D x Depth
6. Omni3D x Distance

Within each cell select exactly:

- 2 challenge samples: dense BF16 correct AND corrected integer-L2 B1/K96 wrong.
- 2 regression-control samples: dense BF16 correct AND corrected integer-L2 B1/K96 correct.

Use these frozen inputs:

- dense: `evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl`
- corrected L2 control: `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl`
- official parsing/scoring helpers already used by the current repository.

Eligibility and deterministic ordering:

- Exclude sample `1`.
- Exclude every ID in `datasets/cv_bench_screening100.jsonl`.
- Exclude every ID in `datasets/2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl`.
- Exclude every ID in the existing random v1 manifest `datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl`.
- Require the real-tokenizer answer-branch contract and processor-derived `G > 96`.
- Rank eligible samples by `sha256(str(sample_id))`, then sample ID; take the first two in each outcome stratum/cell.
- Across the eight selected Omni3D samples, require Hypersim, SUNRGBD, and nuScenes all represented, with at least two samples from each subdataset. If the naive first-two rule does not satisfy this, implement a deterministic constrained selection minimizing the ordered SHA-rank tuple; document and test it. Fail closed if impossible.

## Oracle arm

Reuse the existing tested B1 answer-margin Oracle implementation and runner where safe. Do not duplicate model-forward code unnecessarily.

Frozen runtime contract:

- local RoboBrain2.5-4B, BF16/SDPA, greedy, `do_sample=False`, `max_new_tokens=16`;
- cutpoint B1 after vision block 0;
- K96, region_grid=3, local_candidate_cap=16;
- dense full-sequence official-correct-answer branch margin;
- per-group score `sum(abs(gradient * activation))`;
- inject as `combined_scores` through the existing `integer_scorer` callback and existing `select_hardware_regular_groups` implementation;
- preserve the same hardware rule, not necessarily the same L2-selected anchor indices: exact output=96, real=87, context=9 for every non-bypass pilot sample, 3x3 coverage/context pooling/original-position ordering, no simple `fixed_indices` substitute;
- model parameters frozen, no optimizer, parameter grads all `None`.

The new config must pin every input, source script, runner, summarizer, test, manifest, dense JSONL, and corrected-L2 JSONL SHA256. Raw records must retain the existing safe-resume/hash/timing/peak-memory/gradient/layout contracts.

## Decision-only summarizer

Create a fail-closed summarizer for this enriched diagnostic. It must not report a 24-sample result as official CV-Bench accuracy and must not use CI/significance language.

GO only if all hold:

- challenge repairs: at least 8 of the 12 samples that corrected L2 got wrong become correct under Oracle;
- control regressions: at most 1 of the 12 samples that corrected L2 got correct becomes wrong under Oracle;
- Oracle parse success: 24/24;
- exact K96 layout contract: 24/24;
- all hashes/config/arm/sample IDs and challenge/control labels verify.

Otherwise STOP. Report per-cell repair/regression counts and per-sample transitions. Dense is only the eligibility reference, not a third accuracy arm.

## Required new outputs

- deterministic selection script;
- new manifest JSONL;
- new preregistered config JSON;
- new fail-closed summarizer;
- only the minimum runner adaptation needed, preferably reuse the generic existing runner;
- focused CPU tests for selection, constrained Omni3D coverage, leakage declarations, resume/layout validation, summarizer GO/STOP and malformed inputs;
- Chinese preregistration report JSON + Markdown with status `PREREGISTERED_NOT_EXECUTED` and `gpu_runs=0`;
- update `CURRENT_STATUS.md` with evidence and the exact host GPU command, but do not execute it.

Run `py_compile`, focused tests, and the full `evaluation/tests` suite. Final stdout must be one compact JSON object with: `status`, `files_changed`, `sample_ids`, `cell_counts`, `outcome_counts`, `omni3d_subdataset_counts`, `focused_tests`, `full_tests`, `gpu_runs`, `pins_verified`, `next_command`.
