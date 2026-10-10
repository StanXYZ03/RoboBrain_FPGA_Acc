# Harness task: build a lifecycle-safe future evaluation protocol v2

Work in `/home/zhaoxiangyu/Documents/Codex/2026-07-20-home-zhaoxiangyu-robobrain2-5-xilinx-zcu102`.

This is CPU-only infrastructure for future mechanisms. Do not run GPU/model inference/network/training, install dependencies, or modify any frozen v1/v2 source/config/test/raw/prior report. In particular, do not change:

- `evaluation/build_cv_bench_pruning_report.py`;
- `evaluation/summarize_conservative_l2_guarded_rerank_full2638.py`;
- any v1/v2 preregistration config or pinned test;
- any existing JSONL result or `reports/cv_bench/*` report.

The completed v2 audit found two reusable protocol gaps:

1. the frozen scorer's `parse_success_rate` derives the parsed-answer key via `correct_key.replace("_correct", "_answer")`; when `correct_key="correct"`, this aliases the boolean correctness field and makes the parse gate appear 1.0 although honest failures are dense=40, corrected-L2=42, prompt=42, v2=43;
2. the frozen v2 preregistration pins code/checkpoints/data but not the local model weight shards, and its pre-execution tests assume the future output path remains absent after execution.

Create a small, reusable protocol v2 using new files only:

1. `evaluation/cv_bench_metric_v2.py`
   - reuse the frozen `parse_choice`/`answer_letter` behavior;
   - require explicit `correct_key` and explicit `parsed_answer_key` with distinct fields;
   - validate record count/required keys/types and fail closed;
   - compute the same frozen CV-Bench primary score weighting, plus honest parse success/count;
   - never silently infer a parsed-answer key from a correctness key.
2. `scripts/build_model_identity_manifest.py`
   - stream-hash model identity files without loading weights into RAM;
   - include every local safetensors shard, index/config, processor/tokenizer/chat-template files that exist;
   - record relative path, byte size and SHA256; deterministic sorted JSON; reject missing referenced shards;
   - default model path `/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B` but support CLI override/output.
3. Generate `artifacts/provenance/robobrain25-4b-model-identity.json` from the actual local model. Do not alter model files.
4. `configs/evaluation_protocol_v2.json`
   - future-only, not retroactive;
   - pin the new metric source and model identity manifest SHA;
   - define explicit lifecycle states `PREREGISTERED_NOT_EXECUTED`, `EXECUTED_UNAUDITED`, `AUDITED_STOP`, `AUDITED_GO_EXTERNAL_VALIDATION`;
   - require missing-output tests to use temporary/explicit paths, never the real default output;
   - require true parse gate from explicit parsed-answer fields;
   - distinguish development, untouched confirmation, software latency, FPGA estimate and board measurement.
5. Add focused tests for metric correctness/fail-closed behavior, model manifest determinism/index-shard coverage, and lifecycle-safe path handling. Tests must use temp paths and must not rewrite existing reports.
6. Produce Chinese reports:
   - `reports/token_selector/2026-10-06-future-evaluation-protocol-v2.{json,md}`
   - include the four-arm honest parse counts recomputed read-only from existing JSONLs;
   - state these findings do not retroactively change v2 primary scores or `DEVELOPMENT_STOP`.
7. Update the top of `CURRENT_STATUS.md`.

Run py_compile and only the new focused tests plus the existing corrected full2638 audit test module. Do not run the full `evaluation/tests` suite because five frozen pre-execution tests are state-stale and one rewrites the frozen development report when the arm exists. Recheck the 36 frozen v2 pins and verify the v2 raw SHA and frozen development `generated_at` remain unchanged.

Return compact JSON only with `status`, `files_created`, `files_modified`, `metric_v2_checks`, `honest_parse_counts`, `model_manifest`, `lifecycle_contract`, `focused_tests`, `frozen_pins`, `gpu_runs`, `limitations`, `next_allowed_action`.
