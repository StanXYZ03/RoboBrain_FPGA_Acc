# Harness task: audit the frozen v2 full2638 development result

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

This is a CPU-only, read-only audit and diagnostic over an already completed GPU run. Do not run the model/GPU/network/training, install dependencies, tune thresholds, alter v2, or overwrite any frozen source/config/raw/prior report. The new raw output is:

`artifacts/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638/conservative_l2_guarded_rerank_v2_k96_full2638.jsonl`

Observed facts that must be independently verified, not trusted:

- SHA256 `3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1`;
- 2638 unique records, 143 identity bypasses and 2495 compressed records;
- frozen summarizer decision `DEVELOPMENT_STOP`, solely because `score_at_least_90pct_dense=false`;
- dense `0.8727298774444869`, corrected integer-L2 `0.7668517264725796`, v2 `0.7837404943432142`;
- v2-minus-L2 `+0.016888767870634647`, stratified paired bootstrap 95% CI `[0.006157592750679508, 0.0271543129924543]`, 5000 repeats, seed 20261006;
- 90%-of-dense floor `0.7854568897000382`.

Independently audit, fail closed:

1. Raw SHA, 2638 unique manifest IDs, no missing/extra/duplicate rows, every `record_sha256`, model/config/checkpoint/source pins and all leakage flags.
2. Exact identity bypass contract for `G<=96`; every compressed row exactly K96 with 87 real + 9 context; anchors frozen; same-region one-for-one swaps; per-region real counts unchanged; nonempty context pools; swap gain identities and deterministic array hashes.
3. Recompute parsing and the official primary metric using the same frozen `score(...)["cv_bench_accuracy"]` implementation for dense, corrected integer-L2, prior prompt fusion and v2. Do not invent or substitute a metric.
4. Recompute the stratified paired bootstrap v2-minus-L2 with exactly 5000 repeats and seed 20261006. Verify the 90%-dense floor and exact absolute/relative shortfall.
5. Recompute all decision gates and require the decision to remain exactly `DEVELOPMENT_STOP` if any gate fails. Do not relax the preregistered 90% gate after seeing results.
6. Produce a label-aware postmortem strictly marked diagnostic/development-only: per task, source dataset and frozen original-G bins; v2-vs-L2 repairs/regressions/unchanged; v2-vs-dense repairs/regressions/agreement; parse failures; answer changes; swap-count and bypass breakdown. Report enough counts to locate the loss, but do not tune or select a v3 rule from these labels.
7. State explicitly that full2638 has been repeatedly observed, is not an untouched confirmation set, and provides no FPGA latency, board deployment, paper accuracy or external-validation claim.

Create only new audit artifacts plus a top status entry:

- `evaluation/audit_conservative_l2_guarded_rerank_v2_full2638.py`
- focused CPU tests under `evaluation/tests/`
- `reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-audit.{json,md}` (MD in Chinese)
- `reports/token_selector/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-postmortem.{json,md}` (MD in Chinese)
- update the top of `CURRENT_STATUS.md`

Run py_compile, focused tests and the full `evaluation/tests` suite, reporting exact counts and real exit codes. Recheck all frozen v1/v2 pins after creating the audit artifacts. This task has `gpu_runs=0`, `audited_gpu_runs=1`, `audited_samples=2638`.

Return compact JSON only with: `status`, `decision`, `raw_sha256`, `record_counts`, `scores`, `retained_dense_fraction`, `gate_shortfall`, `paired_bootstrap`, `contract_checks`, `diagnostic_breakdown`, `files_created`, `files_modified`, `focused_tests`, `full_tests`, `gpu_runs`, `audited_gpu_runs`, `next_allowed_action`, `limitations`.
