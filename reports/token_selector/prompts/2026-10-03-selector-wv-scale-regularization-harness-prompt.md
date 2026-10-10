# Harness task: Wv scale hardware-regularization audit

Implement and run exactly `configs/selector_wv_scale_regularization.preregistered.json`.
Codex owns the design and decision. Keep terminal/final output terse; put evidence
in files.

## Safety and isolation

- CPU-only, offline, validation-only, no training/tuning/GPU/network/4B/raw images.
- Never resolve/hash/open internal-test NPZ or parse internal-test reports.
- Verify all pins before and after; do not modify historical code/reports,
  checkpoints, caches, manifests, preregistrations, selector HLS, or prior audit.
- Reuse prior audit loaders, scorer, spatial targets, selection and aggregation.
  Add a new module/test only; do not copy a second implementation unnecessarily.

## Experiment

Create `evaluation/selector_wv_scale_regularization.py` and
`evaluation/tests/test_selector_wv_scale_regularization.py`.

Reference arm is the prior primary chain `PER_ROW_WV`: one symmetric INT8 scale
per Wv output row. Candidate `PER_TENSOR_WV` replaces only Wv quantization with
one scale for all Wv values:

`d=max(abs(Wv))/127; q=clamp(round_half_away(Wv/d),-127,127); Wv_hat=q*d`.

Everything else must remain identical: one global activation scale per image,
float Wt prompt_mean_vector followed by one INT8 query vector, exact INT32
Gx1024 by 1024x128 projection, float software-golden normalization/L2/min-max,
Q0.15 score and the existing 3x3 anchor selection. Float reference remains the
same frozen scorer. Evaluate correct and same-image-other-task prompts for all
76 val records, B-arm checkpoints seeds 20261003/04/05 and K=48/64/96.

For candidate vs per-row reference report: selected-set Jaccard, exact match,
replacement count, score Spearman, retained-mass delta and NDCG delta. Also
report candidate vs float mass/NDCG/Jaccard and K48 Pointing prompt-utility gap
delta. Aggregate per record by median across seeds, then 38 independent images
with 2000 percentile-bootstrap CI. Grounding paired-prompt utility is
descriptive only.

Export per-seed candidate Wv as row-major int8 bytes, one little-endian float32
scale, metadata/hashes, and two compact fixtures. Do not export Wt. Validate
INT32 bounds and round-trip fixtures.

## Frozen decision

`PER_TENSOR_WV_ACCEPTABLE_FOR_HLS` iff all are true:

- pins/isolation/INT32/fixture checks pass;
- candidate-vs-per-row mean selected-set Jaccard >=0.98 for every K;
- candidate-vs-per-row mean replacement count <=1.0 for every K;
- candidate-vs-per-row mean score Spearman >=0.995;
- candidate-vs-float retained-mass delta >=-0.01 for every K;
- candidate-vs-float selected-NDCG delta >=-0.01 for every K;
- K48 Pointing prompt-utility gap delta candidate-vs-float >=-0.01;
- every seed K48 candidate-vs-per-row mean Jaccard >=0.97.

Otherwise `PER_TENSOR_WV_NOT_ESTABLISHED`. Exact-set rate is descriptive and
cannot fail the decision. Do not claim fixed-point normalization, HLS latency,
end-to-end accuracy, or publication readiness.

Write JSON, Chinese Markdown and record CSV under
`reports/token_selector/2026-10-03-selector-wv-scale-regularization`, four logs
under `reports/token_selector/logs/`, and bundles under
`artifacts/token_selector/selector_wv_scale_regularization/<seed>/`. Run focused
and full evaluation test suites. Update `CURRENT_STATUS.md` only at the end.
