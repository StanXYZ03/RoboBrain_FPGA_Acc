# Harness task: Selector INT8 numerical audit

Work only in this repository. Implement and run the frozen experiment in
`configs/selector_int8_numerical_audit.preregistered.json`. Keep the response
short; write all evidence to files.

## Authority and boundaries

- Codex owns the experiment design and decision. Do not redesign gates, tune
  thresholds, add training, or recommend a different research direction.
- CPU-only, offline, validation-only. Do not use GPU/network, load the 4B
  model, read raw images, or resolve/hash/open any internal-test NPZ/report.
- Do not modify historical reports, checkpoints, preregistrations, source
  manifests, cached NPZ files, or existing selector/HLS sources.
- Verify every frozen SHA256 before loading arrays, and verify again after.
- Use only the three frozen B-arm best checkpoints and the already frozen val
  split. Do not inspect test metrics.
- Reuse existing split, loaders, spatial targets, scorer, combine, anchor
  selection and metrics. Do not create subtly different duplicate semantics.

## Required implementation

Create `evaluation/selector_int8_numerical_audit.py` and focused tests at
`evaluation/tests/test_selector_int8_numerical_audit.py`.

Implement two deterministic round-half-away-from-zero INT8 chains:

1. `SPLIT_WV_Q8_A8_Q8QUERY` (primary deployment arm): Jetson/host computes the
   frozen float `prompt_mean_vector` query using float Wt, then per-vector
   symmetric INT8 quantizes the final 128-D unit query. FPGA boundary receives
   this query plus visual data. Per image, quantize the complete Gx1024
   `group_mean` tensor with one global Selector scale using q=clamp(round(x*s),
   -128,127), s=128/amax, x_hat=q/s. Quantize each Wv output row symmetrically
   to [-127,127] with d_w=amax/127. Compute Gx128 projection with exact INT32
   accumulation, dequantize by (1/s)*d_w[row], row-L2 normalize, and dot the
   dequantized/re-normalized INT8 query. Compute L2 from x_hat, combine using
   frozen lambda=0.5, then quantize combined score to uint16 Q0.15 via
   clamp(round-half-away(score*32768),0,32767). Selection must consume the
   dequantized Q0.15 score, so it is exactly rank-equivalent to hardware
   `score_q` including deterministic index tie-breaks.
2. `FULL_WT_WV_Q8_A8` (descriptive control): additionally quantize Wt by output
   row and prompt embeddings per prompt tensor, use INT32 projection, then the
   same frozen project-row-normalize, mean, final-normalize query definition.
   This is not the proposed FPGA partition and cannot determine the decision.

Float reference must be the frozen B checkpoint scorer and float combined
score path. For each of 76 val records, 3 seeds and K=48/64/96, compare primary
quantized vs float reference using exact-set match, selected-set Jaccard,
retained spatial-target mass delta (quant-float), selected-set NDCG delta, and
score rank Spearman. Also compare correct prompt and same-image-other-task
prompt. Report Pointing prompt-utility retained-mass gap preservation at K48;
Grounding is descriptive only because its targets are nested and its paired
prompt is not a valid negative.

Export one hardware parameter bundle per seed under
`artifacts/token_selector/selector_int8_numerical_audit/<seed>/` containing
Wv int8 row-major bytes, float32 little-endian per-row dequant scales, metadata
with hashes/layout/packing/rounding, and small deterministic validation fixtures
for one grounding and one pointing record (no raw image). Do not export Wt in
the primary bundle. Include query int8, query scale, group_mean int8/global
scale, score_q and expected selected indices for K48/64/96. Keep artifacts
compact.

Validate analytical INT32 worst-case bounds for lengths 1024, 2560 and 128.
State explicitly that float L2 normalization/min-max remains a software golden;
this audit does not yet prove a complete fixed-point FPGA scorer.

## Frozen decision

Aggregate per record, then take the median across the 3 seeds for each record;
report means and percentile bootstrap 95% CIs over 38 independent images. Do
not use p-values for this engineering-equivalence audit.

Primary `INT8_NUMERICS_PASS` iff all hold for the primary arm:

- all pins/isolation checks pass and no INT32 overflow;
- for every K in {48,64,96}, mean selected-set Jaccard >= 0.95;
- for every K, exact-set match rate >= 0.75;
- for every K, mean retained-mass delta >= -0.01;
- for every K, mean selected-set NDCG delta >= -0.01;
- K48 Pointing prompt-utility retained-mass gap changes by no less than -0.01
  relative to float;
- all three seeds have K48 mean Jaccard >= 0.95.

Otherwise decision is `INT8_NUMERICS_NOT_ESTABLISHED`. Do not promote model
accuracy, FPGA latency, end-to-end acceleration, or publication claims.

## Outputs and verification

Write JSON, Chinese Markdown and record CSV under report stem
`reports/token_selector/2026-10-03-selector-int8-numerical-audit`, plus run,
focused-test, full-test and concise Harness-result logs. Run focused tests and
the full `evaluation/tests` suite. Update `CURRENT_STATUS.md` only after all
checks finish, preserving prior content. Return only decision, key metrics,
test counts, artifact paths, isolation status and limitations.
