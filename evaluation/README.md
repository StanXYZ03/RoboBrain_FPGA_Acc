# Spatial baseline gate

This directory implements the labeled gate required before FPGA accuracy claims.

## Scope

- `dac_spatial_eval_smoke.jsonl` contains 32 COCO val2017 images and two queries per image: one Pointing query and one Grounding query.
- `dac_spatial_eval_expanded.jsonl` contains another 96 images / 192 queries, with all 32 smoke images explicitly excluded. It is used for selector development evidence, not as a final embodied benchmark.
- Pointing ground truth is the center of a real COCO bounding-box annotation. This is an object-pointing smoke metric, not an affordance benchmark.
- Images are resized to a maximum side of 512 before every runtime sees them. Coordinates remain normalized to `[0,1000]`.
- The smoke set validates the pipeline only. A publication must replace or augment it with frozen, held-out spatial benchmarks and confidence intervals.
- The expanded-set selector reports use paired bootstrap intervals because every selector is evaluated on exactly the same samples.

## Deterministic model comparison

The required order is:

```text
Qwen3-VL-4B-Instruct BF16
RoboBrain2.5-4B BF16
RoboBrain2.5-4B Q8_0 GGUF
```

All runs use the same manifest, raster images, prompts, `temperature=0`, and task-specific output limit. Raw JSONL output is append-only and resumable.

## Metrics

- Pointing: parse rate, Euclidean error in normalized coordinates, image-diagonal NME, target-box-diagonal NME, PCK@0.1, PCK@0.2.
- Grounding: parse rate, IoU, Acc@0.5, center error.
- A parse failure is a metric failure; it is never silently dropped.
- Mean, median, and nearest-rank p95 generation latency are retained for audit; cross-runtime latency is not treated as a quantization comparison.

Run unit tests with:

```bash
make eval-test
```

The exact download, inference, and report commands are recorded in the generated experiment report rather than relying on shell history.
## Joint-deletion development gate

`extract_joint_deletion_teacher.py` measures teacher-forced coordinate-token NLL for
same-budget, within-region multi-group swaps around the layer-5 L2 baseline. It does
not use generated model output as ground truth.

Previously revealed manifests whose names contain `blind` or `validation` are rejected
before model loading. Raw JSONL is append-only and resumable. The initial two-task pilot,
repeatability result, limitations, and next scale gate are recorded in
`reports/2026-08-26-joint-deletion-coordinate-nll-pilot.md`.

The set-level loss in `joint_deletion.py` ranks complete retained sets by measured
coordinate NLL. It is a development objective only; model-output accuracy and a new
held-out test are still required before PL integration.

## Multi-cut-point geometric-oracle gate

`dac_cutpoint_oracle_dev.jsonl` contains 16 new COCO images / 32 requests and has
zero image overlap with every earlier smoke, expanded, calibration, validation,
blind, and routing manifest. It is development data, not a hidden test set.

`run_hf_cutpoint_oracle.py` evaluates ground-truth-box-aware group retention at
Patch Embedding, vision blocks 1 and 3, the first DeepStack extraction (zero-based
layer 5), and after the visual encoder/merger. This oracle is unattainable at
deployment and is only a fail-fast upper bound: if an early cut cannot retain
accuracy even with the target box, no selector RTL is started for that boundary.

`build_cutpoint_preflight.py` reports dominant MAC ratios and transfer
serialization estimates for both a one-way resident prefix and an external
GPU/NPU-to-FPGA-to-GPU/NPU round trip. `build_cutpoint_oracle_report.py` combines
the real outputs, task metrics, stage timings, MAC ratios, and communication gate.
The locked commands and hashes are in
`reports/2026-09-04-cutpoint-oracle.lock.json`.
