# Harness task: conservative corrected-L2 backbone + bounded integer task-residual reranker

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

This task is CPU-only design, implementation, tests, and preregistration. Do not run CUDA/model weights/network/training, do not install dependencies, do not modify the original RoboBrain model/app, do not alter any frozen prior manifest/config/raw output/report, and do not tune anything on the frozen 24-sample answer-margin pilot. Keep the direct absolute-gradient full-replacement teacher permanently `STOPPED`.

## Evidence and correction to preserve

- Dense BF16 official CV-Bench score: `0.8727298774`.
- Corrected integer-L2 B1 K96: `0.7668517265` (87.8682% of dense).
- Prior full-score prompt+corrected-L2 integer fusion: `0.777362` (89.0725% of dense), paired gain over corrected L2 `+0.010510`, but its 95% paired bootstrap CI `[-0.005430,+0.025699]` did not prove prompt gain.
- The full CV-Bench 2638 has already been observed repeatedly. It is development evidence only, never an untouched confirmation set.
- The exact compressed K96 layout is dynamic `87 real + 9 context` for the current full-size inputs. The 3x3 L2 anchors are a subset of the 87 real groups. `local_candidate_cap=16` is a per-region candidate bound; do not misdescribe it as a swap count.
- B1 means after vision block 0. Keep B1, K96, region_grid=3, existing corrected integer patch-L2 definition, accepted integer semantic Q0.15 chain, original-position ordering, context pooling and transfer metadata.

## Single mechanism to implement

Implement one new additive selector family only: `conservative_l2_residual_rerank_v1`.

1. Obtain the baseline real selection from the existing corrected integer-L2 K96 path, including one L2 anchor per nonempty 3x3 region and the existing dynamic context reservation.
2. Freeze every L2 anchor. Never replace an anchor.
3. Preserve each region's number of selected real groups. Therefore every replacement is a same-region one-for-one swap and cannot consume the region's reserved unselected context group.
4. Define signed task-residual score exactly as `semantic_norm_q15 - corrected_l2_norm_q15`, using the already accepted integer semantic chain and the corrected integer patch-L2 Q0.15 score. No labels, answer tokens, gradients, floating lambda, learned new weights, or task/source IDs.
5. Within each region, pair the currently selected non-anchor groups with unselected candidate groups to maximize strictly positive residual-score gain. Ties must be deterministic by lower original group index. Candidate enumeration must remain bounded by the existing per-region `local_candidate_cap=16`.
6. Apply at most `S=16` globally highest positive, nonconflicting, same-region swaps. `S=16` is one independent fixed hardware work quantum; it is not inferred from the stopped 24 samples and is not the same concept as local candidate cap. If fewer than 16 strictly positive legal swaps exist, perform fewer; K remains exactly 96 because each operation is one-for-one.
7. Rebuild the 87 real entries and 9 context entries with the existing layout contract. Context pooling weights remain corrected integer-L2 based, so prompt information affects only the bounded residual swaps. Emit deterministic audit metadata: anchors, baseline details, removed/added IDs, region, residual scores/gain, swap_count, exact real/context/output counts, and hashes.
8. Fail closed on any violation. Do not introduce a sweep of S, K, cut points, anchors, formulas, thresholds, or score mixtures.

If the existing accepted integer chain does not expose a corrected patch-L2 Q0.15 vector suitable for the subtraction, reuse the exact corrected integer-L2 implementation and document the conversion. Do not silently fall back to the old mean-of-patch L2 or float score.

## Required implementation and preregistration

Create new files rather than rewriting frozen code where feasible:

- `evaluation/conservative_l2_residual_reranker.py`
- focused tests under `evaluation/tests/`
- a real-model runner that reuses the existing corrected-L2 B1 runner/integration and supports `--smoke` plus resumable full2638 execution;
- a fail-closed summarizer;
- `configs/2026-10-05-conservative-l2-residual-reranker-full2638-development.preregistered.json`;
- Chinese preregistration report JSON/MD;
- update the top of `CURRENT_STATUS.md`.

Pin all source/config/manifest/checkpoint/weight/baseline hashes available at preregistration time. The new arm output does not yet exist and must be marked not executed. The runner must use the local model, BF16/SDPA, deterministic greedy decoding, `max_new_tokens=16`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, GPU synchronize around timed GPU segments, no model-load time in steady timing, and raw per-sample JSONL with safe resume.

The six-cell smoke is contract-only and may not decide quality. The only development quality run is one full2638 pass of the single frozen v1 arm. Reuse the frozen dense, corrected integer-L2, and prompt+corrected-L2 outputs as paired references after verifying sample IDs/configuration/hashes.

## Frozen development decision

The summarizer must distinguish `DEVELOPMENT_GO_EXTERNAL_VALIDATION` from paper confirmation. GO requires all of:

1. all 2638 unique samples complete and parseable;
2. every compressed sample obeys exact K96 and the frozen real/context/layout contract; legal `G<=K` identity bypass remains allowed and separately counted;
3. official CV-Bench score `>= 0.78545688966`, i.e. at least 90% of dense BF16;
4. point estimate no worse than the strongest prior same-budget development baseline `0.777362`;
5. paired bootstrap 95% CI lower bound for new minus corrected integer-L2 `>= 0.0`, 5000 repeats, seed 20261005;
6. no forbidden label/answer/task/source leakage.

Otherwise emit `DEVELOPMENT_STOP`. Passing is only permission to acquire/run a fresh untouched external spatial-reasoning benchmark; it is not an official confirmation, deployability, FPGA latency, or paper accuracy claim. Do not tune v1 after seeing full2638.

## Verification

- Run py_compile and focused CPU tests.
- Run the complete `evaluation/tests` suite and report the exact count.
- Verify all frozen prior inputs are byte-identical before/after.
- No GPU/model/network/training in this task; report `gpu_runs=0`.

Return compact JSON only with: `status`, `mechanism_contract`, `corrected_87_plus_9_interpretation`, `files_created`, `files_modified`, `pins_verified`, `focused_tests`, `full_tests`, `gpu_runs`, `prereg_status`, `next_host_commands`, `limitations`.
