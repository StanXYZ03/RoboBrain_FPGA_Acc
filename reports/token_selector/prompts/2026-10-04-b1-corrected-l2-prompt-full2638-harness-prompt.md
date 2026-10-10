# Harness task: preregister full2638 confirmation

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

The frozen screening100 experiment has completed with `CORRECTED_PROMPT_SELECTOR_SCREENING_PASS`: float corrected L2=0.733333, integer corrected L2=0.733333, Prompt+integer corrected L2=0.820833; prompt-minus-integer +0.0875, paired CI95 [+0.016667,+0.1625]. Do not tune anything from this result.

Implement only the full2638 promotion scaffold. Do not run GPU/model/network. Do not modify any existing source/config/report/raw output. Add new files only. Final response under 20 lines.

## Frozen full experiment

- Reuse `evaluation/run_hf_hardware_regular_b1_corrected_l2.py` unchanged.
- Manifest: `datasets/cv_bench_full2638.jsonl`.
- Three arms unchanged: `corrected_l2_float`, `corrected_l2_integer`, `prompt_plus_corrected_l2_integer`.
- B1, K96, region_grid=3, local_candidate_cap=16, BF16/SDPA, greedy, max_new_tokens=16, seed-20261003 Wt/Wv_q, no training/tuning.
- Output directory: `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/` with one JSONL per arm.
- Dense full2638 and old current-integer K96 full2638 are descriptive only.

## Deliverables

1. `configs/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.preregistered.json`, written before outputs, status pending/not immutable. Pin the screening promotion report/raw hashes, current runner/core/evaluator/config/checkpoint hashes, manifest/image aggregate if available, dense/current descriptive references, exact output paths, and decision rules.
2. `evaluation/summarize_b1_corrected_l2_prompt_ablation_full2638.py`: reuse/import validated helpers from the screening summarizer where safe; avoid copying 1300 lines. Strictly validate exactly 2638 unique IDs per arm, same full manifest IDs/task/source/grid, K96 or valid identity bypass, record hashes, checkpoint hashes, and arm contracts. Main metric only official `cv_bench_accuracy`; paired stratified bootstrap via existing helper, seed 20261004, repeats 5000. Write JSON/CSV/Chinese Markdown under `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.*`.
3. Focused tests for all three decision branches, exact 2638/count/ID/arm/hash/checkpoint/grid/K fail-closed behavior, relative-vs-absolute metric separation, and no-write-on-failure. Use small synthetic count overrides in tests.

## Decision rules (unchanged)

- integer−float quantization gate passes iff paired CI95 lower >= -0.02.
- prompt−integer-L2 gain confirmed iff paired CI95 lower > 0.
- Ordered outcomes: `CORRECTED_L2_INTEGERIZATION_FAIL`, `CORRECTED_PROMPT_SELECTOR_FULL2638_PASS`, otherwise `PROMPT_GAIN_NOT_CONFIRMED_FULL2638`.
- Dense/old-current references and all timing are descriptive only.

Run focused tests and py_compile only. Do not update CURRENT_STATUS.md.
