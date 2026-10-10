# Harness task: spatial-supervision redesign feasibility audit

You are the execution engineer. Codex is the decision maker. Implement, run, test, and report this bounded audit. Keep your final response concise and factual; do not propose unrelated research directions.

## Security and scope

- Project root: `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.
- Python: `/home/zhaoxiangyu/conda-envs/robobrain25/bin/python`.
- CPU-only, float64 metrics where practical. No GPU, no model load, no inference, no network, no package installation.
- Never read or upload raw images, model weights, API/SSH keys, proxy files, credentials, or private home-directory data.
- You may read project source, JSON/JSONL metadata, cached derived NPZ features, reports, and metrics. Do not send NPZ tensor payloads to the API; use them locally only.
- This is a zero-update audit: no optimizer, no training, no new Wt/Wv, no checkpoint.
- Do not modify `app/`, model directories, existing frozen configs/reports/artifacts, or prior result files.
- Use only the frozen TRAIN split: 318 records / 159 images. Do not resolve, hash, open, or score val/test NPZ files. Do not evaluate CV-Bench.
- Do not read image bytes. Ground-truth geometry comes only from `datasets/prompt_projection_clean_inductive_v1.jsonl`.

## Scientific motivation

The preregistered `prompt_mean_vector` paired retrain ended `NO_PROMOTION`. Existing full-vision max-cosine teacher pairs are non-identical but weakly task-specific: on train, same-image grounding/pointing teacher argmax is identical for about 88% of images. Before designing another trainable scorer, determine whether a parameter-free, task-specific spatial supervision target can provide a stronger and valid signal while remaining compatible with the existing fixed-budget/anchor selector.

This audit must not claim that a student can learn the target, that downstream accuracy improves, or that FPGA inference accelerates.

## Pre-registration requirement

Before computing any new per-sample or aggregate audit metric, write:

- `configs/spatial_teacher_feasibility.preregistered.json`
- `reports/token_selector/2026-10-03-spatial-teacher-feasibility-preregistration.json`

Both must pin the input paths and SHA256 values for:

- `configs/prompt_projection_distillation.json`
- `artifacts/token_selector/prompt_projection_features/index.jsonl`
- `datasets/prompt_projection_clean_inductive_v1.jsonl`
- `datasets/prompt_projection_clean_inductive_v1.lock.json`
- `reports/token_selector/2026-10-02-teacher-pair-signal-train-only.json`
- `configs/hardware_regular_token_selector.json`

Record the exact formulas, split rule, metrics, resampling seeds/counts, decision labels, and prohibitions below. Verify all pinned hashes before and after the run and report byte-for-byte immutability.

## Frozen spatial-target definitions

For every TRAIN sample, join cache/index and clean manifest by `sample_id`; hard-fail on missing/duplicate/mismatched task or image identity. Require `group_count == grid_h * grid_w` or stop with an explicit blocker.

Coordinate convention: manifest coordinates are normalized `[0,1000]`. A grid cell `(r,c)` covers `[1000*c/W, 1000*(c+1)/W] x [1000*r/H, 1000*(r+1)/H]` for grid `H x W`.

1. `grounding_area_target`: for each cell, compute exact intersection area with `bbox_1000`; divide by total intersection area so mass sums to 1. Clip bbox to `[0,1000]` only for numerical boundary tolerance; hard-fail invalid/zero-area boxes.
2. `pointing_bilinear_target`: map point `(x,y)` to continuous cell-center coordinates `u=x*W/1000-0.5`, `v=y*H/1000-0.5`; clamp to valid center-coordinate range; distribute mass to the up to four neighboring grid centers with separable bilinear weights; combine duplicate boundary indices and normalize to sum 1. No Gaussian and no bandwidth parameter.
3. Do not blend these targets with the semantic teacher in this audit. Do not tune formulas after seeing metrics.

Use the cached `teacher_soft_target` only as the existing semantic-teacher comparator. Reuse existing split, metric, JS/KL, L2 baseline, anchor, bootstrap/permutation utilities where applicable; do not create inconsistent duplicate implementations.

## Required measurements

All paired inference units are images, not two task records.

1. Integrity and geometry:
   - joined/missing/duplicate counts; task pair completeness;
   - bbox validity, point validity, point-inside-bbox fraction;
   - grid shape distribution and `group_count == H*W` fraction;
   - target finite/nonnegative/sum-to-one errors;
   - support size, entropy, effective support `exp(entropy)`, max mass, for each task.
2. Same-image task separation for both existing semantic teacher and new spatial teacher:
   - JS divergence in nats, symmetric KL, TV, argmax-different fraction;
   - top-K Jaccard/overlap/disagreement at K=48/64/96 with `K_eff=min(K,G)`;
   - paired per-image difference `spatial - semantic` for JS, TV, and top-K disagreement.
3. Statistics for those paired differences:
   - mean, median, std, 95% image-level bootstrap CI (2000 resamples), two-sided sign-flip p-value (10000 resamples), Cohen dz;
   - deterministic seed `20261003`; no multiple candidate search is being performed, so report raw p-values and do not invent Holm correction.
4. Semantic/spatial compatibility per task:
   - Spearman and full NDCG between each spatial target and cached semantic teacher target, summarized by task and overall;
   - explicitly state that low agreement means the objectives differ, not automatically that one is wrong.
5. Hardware-regular baseline relevance:
   - score cached `group_mean` with the existing L2 baseline;
   - use the existing deterministic 3x3 anchor rule if it can be reused exactly from project code; if no reusable exact function exists, report `NOT_COMPUTED` instead of inventing one;
   - for K=48/64/96, report spatial-target NDCG and retained target mass for L2 and available anchor baseline, by task and overall.
6. Provenance/isolation:
   - opened NPZ count and exact split role;
   - `val_test_npz_opened=[]` and a spy-loader/corrupted-val-test-NPZ test proving isolation;
   - input/source/output SHA256, runtime, peak RSS, no-GPU/no-model/no-network flags.

## Frozen decision labels

This is not a promotion gate. Emit exactly one:

- `SPATIAL_SUPERVISION_SIGNAL_PRESENT` only if all integrity checks pass and the 95% bootstrap CI lower bound of `spatial-semantic` is greater than zero for BOTH JS and TV. Top-K results remain descriptive and cannot override this rule.
- `SPATIAL_SUPERVISION_SIGNAL_NOT_ESTABLISHED` otherwise.
- `BLOCKED_DATA_CONTRACT` if joins, geometry, grid, or isolation contracts fail.

Even when signal is present, state that learnability and task accuracy remain untested.

## Implementation and outputs

Create:

- `evaluation/spatial_teacher_feasibility_audit.py`
- `evaluation/tests/test_spatial_teacher_feasibility_audit.py`
- `reports/token_selector/2026-10-03-spatial-teacher-feasibility.csv` with one row per TRAIN image and raw paired metrics
- `reports/token_selector/2026-10-03-spatial-teacher-feasibility.json`
- `reports/token_selector/2026-10-03-spatial-teacher-feasibility.md`
- `reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-run.log`
- `reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-focused-tests.log`
- `reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-full-tests.log`
- `reports/token_selector/logs/2026-10-03-spatial-teacher-feasibility-harness-result.json`

Tests must cover exact bbox-cell overlap, bilinear point mass including boundaries, normalization, deterministic resampling, join hard failures, grid mismatch hard failure, train-only access with corrupt val/test NPZ, decision logic, report schemas, and frozen-file immutability.

Run focused tests and then the full `evaluation/tests` suite. Update the top of `CURRENT_STATUS.md` only after completion, with decision, key values, evidence paths, test counts, limitations, and the exact next decision for Codex. Do not alter older status entries.

Final response must be a compact JSON object containing: `status`, `prereg_sha256`, `train_records`, `train_images`, `val_test_npz_opened`, `integrity`, `semantic_pair_signal`, `spatial_pair_signal`, `paired_differences`, `semantic_spatial_compatibility`, `l2_baseline`, `anchor_baseline`, `focused_tests`, `full_tests`, `runtime_seconds`, `peak_rss_mib`, `files_created`, `limitations`, `next_decision`. Do not include verbose reasoning.
