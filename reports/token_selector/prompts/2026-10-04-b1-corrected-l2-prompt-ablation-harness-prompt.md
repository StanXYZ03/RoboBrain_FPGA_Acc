# Harness task: B1 corrected-L2 / Prompt ablation

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

Implement the next preregistered screening100 experiment. Do not run GPU/model/network and do not modify existing source, frozen JSONL, reports, checkpoints, HLS, or configs. Add only new files. Final response under 25 lines: files, tests, hashes, blockers.

## Established result

The real-activation layer-5 replay proved 100/100 that the historical baseline is `mean(norm(each of 4 patches))`; current `group_l2_saliency`/integer scorer instead uses `norm(mean(4 patches))`. We must now evaluate Prompt conditioning against the corrected strong baseline, not the defective one.

## Frozen experimental design

All proposed arms use the same B1 cut (after vision block 0), K=96, region_grid=3, local_candidate_cap=16, existing `select_hardware_regular_groups`, context behavior/layout, BF16 model, SDPA, greedy decode, max_new_tokens=16, screening100 manifest, and frozen seed-20261003 Wt/Wv_q. No training/tuning.

Proposed arms:

1. `corrected_l2_float`: score per group = `mean(norm(x_i))` over 4 raw patch vectors, then existing float min-max normalization.
2. `corrected_l2_integer`: global-per-image symmetric INT8 quantize all raw patch vectors with the existing accepted activation quantizer; per patch compute INT64 sum of squares + integer floor sqrt; per group sum the four patch norms (equivalent in rank/min-max to their mean); existing exact Q0.15 integer min-max normalization. No Prompt term.
3. `prompt_plus_corrected_l2_integer`: preserve the existing accepted integer semantic chain exactly (group mean quantization, frozen Wv_q, frozen prompt query, semantic Q0.15); replace only old `norm(group_mean)` L2 Q0.15 by arm 2 corrected patch-wise L2 Q0.15; combine exactly with the existing lambda=0.5 round-half-away/clamp rule.

Frozen descriptive references, not new arms: existing B1 defective-L2 K96 screening100 JSONL and current integer K96 full2638 JSONL restricted to screening IDs.

## Deliverables

1. New CPU-safe module implementing bit-exact corrected patch-L2 integer chain and three scorer callbacks usable by the frozen B1 integration without modifying it. Reuse accepted quantization/rounding/minmax helpers; no duplicate math where an accepted helper exists. Emit intermediate arrays/bounds sufficient for audit.
2. New one-arm-at-a-time GPU runner with `--arm` restricted to the three names above; strict offline local model; safe resume/provenance; synchronized timing; raw JSONL. It must use the same B1 unified forward and run full greedy output for CV-Bench scoring. Each record stores all score/index hashes, selected indices, output text/token IDs, G/K/grid, context/real counts, timings and peak memory.
3. New fail-closed summarizer producing JSON/CSV/Chinese Markdown. Main metric only `evaluation.build_cv_bench_pruning_report.score(...)["cv_bench_accuracy"]`; Pointing/Grounding/NDCG/retained-mass forbidden as primary. Paired bootstrap must reuse existing `paired_bootstrap_delta`, stratified by `task::source_dataset`, seed 20261004, repeats 5000.
4. Preregistered config written before outputs. Pin all current input/source/checkpoint/manifest/reference hashes and three output paths. Status pending/not immutable until Codex runs.
5. Focused tests and py_compile only; no full discovery.

## Decision rules

- `integer_minus_float` = corrected_l2_integer minus corrected_l2_float. Quantization gate passes iff paired 95% CI lower bound >= -0.02.
- `prompt_minus_integer_l2` = prompt_plus_corrected_l2_integer minus corrected_l2_integer. Prompt gain is confirmed iff paired 95% CI lower bound > 0.
- Overall outcome:
  - both pass: `CORRECTED_PROMPT_SELECTOR_SCREENING_PASS`
  - quantization passes but Prompt gain not confirmed: `PROMPT_GAIN_NOT_CONFIRMED`
  - quantization fails: `CORRECTED_L2_INTEGERIZATION_FAIL`
- Dense, old defective L2, and old current integer are descriptive only and cannot override these rules.

## Required tests

- cancellation example distinguishing `mean(norm)` from `norm(mean)`;
- integer arithmetic, saturation, rounding, bounds, deterministic ties;
- corrected integer L2 selection vs corrected float selection on synthetic and cached representative activations;
- lambda=0.5 combination exactly matches existing accepted combine behavior when supplied the same semantic/L2 Q15 vectors;
- wrong/missing arm, IDs, K/G/grid, checkpoint hash, record hash, duplicate/partial/stale resume all fail closed;
- summarizer decision branches and no-write-on-failure.

Do not update `CURRENT_STATUS.md`; Codex does that after review.
