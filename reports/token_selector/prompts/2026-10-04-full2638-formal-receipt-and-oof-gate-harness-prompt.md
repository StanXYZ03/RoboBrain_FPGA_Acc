# Harness task: formal receipt plus out-of-fold G-gated Prompt diagnostic

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

This task has two ordered parts. Keep the response concise and do not use GPU, network, model loading, training, or dependency installation.

## Part A: finish the frozen full2638 formal execution record

Read only the existing preregistration, three raw arm JSONLs, formal JSON/CSV/MD report, screening promotion report, and relevant tests. Do not regenerate or edit any of those inputs.

Create:

- `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638-execution-receipt.json`

The receipt must record exact paths and current SHA-256 hashes for preregistration, source scripts, manifest, all three raw arms, and formal reports; exact commands and test counts available from current artifacts/status; formal scores, paired CIs, decision, GPU model, and the fact that raw runs were real A6000 BF16/SDPA inference while selector software timings are not FPGA claims. Verify all authoritative hashes before writing.

Insert a concise full2638 formal milestone into `CURRENT_STATUS.md` immediately after the current top heterogeneity entry and before the older screening100 entry. Do not alter older entries.

## Part B: out-of-fold G-gated Prompt diagnostic

Objective: test whether a deployment-observable, hardware-trivial gate based only on pre-selector group count `G = full_llm_visual_tokens` can choose between the already-frozen integer-L2 arm and Prompt+integer-L2 arm without dataset/task labels. This is exploratory model selection, not a new formal claim.

Inputs are frozen:

- `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl`
- `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/prompt_plus_corrected_l2_integer_k96_full2638.jsonl`
- `datasets/cv_bench_full2638.jsonl`
- `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json`

Rules:

1. Build deterministic 5-fold splits stratified by `task::source_dataset`, seed 20261004. Every sample appears in exactly one held-out fold.
2. Candidate policies are fixed before scoring: `never_prompt`, `always_prompt`, and `prompt_if_G_gt_{96,256,512,1024}`. No task/source/dataset identity may enter a policy.
3. In each fold, select exactly one policy using only the other four folds, maximizing official `cv_bench_accuracy`; ties prefer lower Prompt-use rate, then the fixed order above. Apply it unchanged to the held-out fold.
4. Pool held-out predictions across all five folds and compute official CV-Bench score. Compare OOF policy against `never_prompt`, `always_prompt`, and dense descriptive baseline. Use paired stratified bootstrap seed 20261004/repeats 5000 for OOF-minus-never and OOF-minus-always.
5. Report fold-selected policies, Prompt-use rate, W->R/R->W transitions, task/source/G-bin diagnostics, and whether selections are stable across folds.
6. State clearly that threshold discovery uses the same full2638 corpus under cross-validation and therefore still requires an external frozen dataset before a paper claim.
7. Reuse existing scorer/parser helpers. Fail closed on hash, sample ID, config, and record integrity mismatches. Do not modify any prior report/config/raw result.

Create only:

- `evaluation/analyze_full2638_oof_gated_prompt.py`
- `evaluation/tests/test_analyze_full2638_oof_gated_prompt.py`
- `reports/cv_bench/2026-10-04-full2638-oof-gated-prompt.json`
- `reports/cv_bench/2026-10-04-full2638-oof-gated-prompt.csv`
- `reports/cv_bench/2026-10-04-full2638-oof-gated-prompt.md`
- `reports/cv_bench/2026-10-04-full2638-oof-gated-prompt-execution-receipt.json`

Validation:

- Run focused tests plus the two full2638 summarizer/heterogeneity tests and CV-Bench scorer tests.
- Record exact test counts and all input/output hashes.
- Add a short new top entry to `CURRENT_STATUS.md` only after successful execution.
- Final response only: files, tests, the OOF score/deltas/CIs, fold policies, Prompt-use rate, and blocker.
