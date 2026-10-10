# Harness task: full2638 Prompt gain heterogeneity diagnostic

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

Objective: explain the already-frozen formal result `PROMPT_GAIN_NOT_CONFIRMED_FULL2638` without changing the experiment or inventing a new primary metric. This is a post-hoc diagnostic only.

Authoritative inputs:

- `reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json`
- `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl`
- `artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/prompt_plus_corrected_l2_integer_k96_full2638.jsonl`
- `datasets/cv_bench_full2638.jsonl`

Constraints:

1. CPU-only. No GPU, network, model loading, training, or dependency installation.
2. Do not modify authoritative inputs, existing configs, existing reports, or `configs/evaluation_protocol.json`.
3. Add only a diagnostic script, focused tests, JSON/CSV/Chinese Markdown report, a concise execution receipt, and a short new entry at the top of `CURRENT_STATUS.md`.
4. Use existing CV-Bench answer parsing/scoring helpers; do not reimplement scoring loosely.
5. Keep the formal conclusion unchanged. Explicitly label all subgroup analyses exploratory/post-hoc and not evidence for a new claim.
6. Keep output concise. Do not narrate implementation details in the final response.

Required analysis:

- Pair records exactly by sample ID and verify hashes/config fields remain consistent.
- For Prompt minus integer-L2, report exact counts and rates of wrong-to-right, right-to-wrong, unchanged-right, unchanged-wrong.
- Break those transition counts and paired accuracy differences down by task, source dataset, identity-bypass status, and pre-selector `real_group_count` bins: `<=96`, `97-256`, `257-512`, `513-1024`, `>1024`.
- For task and source strata, compute paired bootstrap 95% CIs with deterministic seed 20261004 and 5000 repeats. State sample size for every stratum. Do not apply these CIs as a formal gate.
- Report selection-index overlap for non-bypass samples: exact match rate and Jaccard mean/min, overall and by task.
- Identify whether aggregate Prompt gain is concentrated in a small subset and whether any task/source materially regresses. Use factual language only.
- Include dense accuracy by task/source only as descriptive context if it is already available through the existing scorer; do not turn it into a gate.

Create:

- `evaluation/analyze_full2638_prompt_gain_heterogeneity.py`
- `evaluation/tests/test_analyze_full2638_prompt_gain_heterogeneity.py`
- `reports/cv_bench/2026-10-04-full2638-prompt-gain-heterogeneity.json`
- `reports/cv_bench/2026-10-04-full2638-prompt-gain-heterogeneity.csv`
- `reports/cv_bench/2026-10-04-full2638-prompt-gain-heterogeneity.md`
- `reports/cv_bench/2026-10-04-full2638-prompt-gain-heterogeneity-execution-receipt.json`

Validation:

- Run focused tests and relevant existing summarizer/scorer tests.
- Run the new script once.
- Record input/output SHA-256 hashes and exact test counts in the receipt.
- Return only: created/modified files, test result, five most decision-relevant numerical findings, and any blocker.
