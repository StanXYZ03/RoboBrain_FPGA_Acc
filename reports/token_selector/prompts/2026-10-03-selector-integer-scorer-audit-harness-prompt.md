# Harness task: bit-exact integer Selector scorer audit

Implement and run the frozen experiment in
`configs/selector_integer_scorer_audit.preregistered.json`. Codex owns the
design and decision. Keep response concise; save evidence to files.

## Isolation

- CPU-only, offline, validation-only; no training/tuning/GPU/network/4B/raw images.
- Do not resolve/hash/open internal-test NPZ or parse internal-test reports.
- Verify pins before/after. Do not modify any prior code/report/config/checkpoint,
  cache, manifest or HLS source. Add a new module and tests only.
- Reuse the accepted per-tensor Wv quantization audit and prior loaders,
  quantizers, metrics, selection and aggregation.

## Frozen integer chain

Create `evaluation/selector_integer_scorer_audit.py` and
`evaluation/tests/test_selector_integer_scorer_audit.py`.

Reference `PER_TENSOR_FLOAT_NORM` is the accepted per-tensor-Wv chain from the
previous audit. Candidate `INTEGER_SCORER_Q16` uses exactly:

1. Existing global A8 `group_mean_q`, per-tensor Wv INT8, and per-vector INT8
   prompt query. Compute visual projection `p[G,128]` with exact INT32 sums.
2. For each group compute signed `dot=sum_d p[d]*query_q[d]` in INT64 and
   `norm2=sum_d p[d]^2` in unsigned/signed-safe INT64.
3. Because activation/Wv/query scales are positive common factors that cancel
   in cosine plus per-image min-max, omit those scales and query norm. Compute
   `semantic_q16 = round_half_away((dot << 16) / floor_sqrt(norm2))`; zero norm
   maps to zero. Division must use an exact signed rational round-half-away
   implementation, not float.
4. Compute `l2_int=floor_sqrt(sum_1024 group_mean_q^2)` in integer arithmetic.
5. Independently min-max normalize semantic_q16 and l2_int to [0,32767] with
   exact rational round-half-away; all-equal maps to zero.
6. With lambda=0.5, output `score_q=round_half_away((semantic_norm_q15 +
   l2_norm_q15)/2)`, clamped [0,32767]. Existing anchor selection consumes the
   uint16 values (or exact dequantization only for reuse of the selector).

No float is allowed inside candidate scoring after input quantization. Python
may use arbitrary-precision integers only to prove fixed hardware widths; also
implement explicit checked INT32/INT64 behavior and report observed/analytical
bounds. Freeze semantic fractional bits at 16; do not sweep/tune bit widths.

Evaluate 76 val records, B-arm seeds 20261003/04/05, correct/wrong prompts and
K=48/64/96. Candidate vs per-tensor reference: Jaccard, exact match,
replacement count, score Spearman. Candidate vs original float: retained-mass
and selected-NDCG deltas plus Jaccard. Report K48 Pointing prompt-utility gap;
Grounding is descriptive only. Aggregate per record by seed median, then 38
independent images with 2000 bootstrap CI.

Export per seed compact bit-exact fixtures containing group_mean_q, query_q,
Wv_q, projection checksum/bounds, semantic_q16, l2_int, normalized components,
score_q and selected indices. Do not duplicate Wv in each fixture; reference
the accepted bundle and pin its hash. Add a machine-readable integer width and
operation contract artifact under
`artifacts/token_selector/selector_integer_scorer_audit/`.

## Frozen decision

`INTEGER_SCORER_NUMERICS_PASS` iff all hold:

- pins/isolation/fixture checks pass and every analytical/observed value fits
  the declared hardware width;
- candidate-vs-per-tensor mean Jaccard >=0.98 for every K;
- mean replacement count <=1.0 for every K;
- mean score Spearman >=0.995;
- candidate-vs-float retained-mass delta >=-0.01 for every K;
- candidate-vs-float NDCG delta >=-0.01 for every K;
- K48 Pointing prompt-utility gap delta candidate-vs-float >=-0.01;
- each seed K48 candidate-vs-per-tensor mean Jaccard >=0.97.

Otherwise `INTEGER_SCORER_NUMERICS_NOT_ESTABLISHED`. Exact-set rate is
descriptive. Passing is permission to implement HLS, not evidence of synthesis,
latency, task accuracy or end-to-end acceleration.

Write flat report-stem files
`reports/token_selector/2026-10-03-selector-integer-scorer-audit.{json,md}` and
`...-records.csv`, four logs, artifacts, then run focused/full evaluation tests
and update `CURRENT_STATUS.md` only after completion.
