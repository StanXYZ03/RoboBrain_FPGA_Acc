#!/usr/bin/env python3
"""Generate the 2026-10-07 edge576 CV-Bench r2 three-arm preregistration.

CPU-only, read-only over every pinned input.  This script never runs a model: it
recomputes the SHA256 of the corrected auditor, the re-audit report, the r2
focused tests, the built edge576 manifest and summary, the frozen full manifest,
the model identity manifest, the future-only metric module, the frozen
comparison-arm outputs and the **frozen existing**
``conservative_l2_guarded_rerank_v2`` mechanism chain (core, integration, runner,
summariser, tests, projection npz, Wv int8, per-tensor scale and the integer
scoring / corrected-L2 core).  It then writes
``configs/2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json``
plus a Chinese r2 preregistration report.

This is the corrected successor of the superseded r1 preregistration.  r1 is
kept byte-identical as evidence of the pre-execution design defects and must
never be executed:

* r1 left the third arm ``mechanism_id=null`` (not a pre-registered candidate);
* r1 demanded "all 2638 present and parseable" instead of the honest contract
  "2638 unique complete outputs, parse failures counted as wrong and reported";
* the r1 auditor CLI accepted ``--model`` but never forwarded it to ``audit``.

r2 freezes the third arm to the **existing** ``conservative_l2_guarded_rerank_v2``
mechanism (no new mechanism is invented) and runs it, Dense and corrected integer
L2 under the identical ``G<=576`` edge576 runtime domain.  This is a
developmental re-test of v2 in a new edge runtime domain; it does not rewrite the
frozen 2026-10-06 native-resolution ``DEVELOPMENT_STOP``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json"
)
DEFAULT_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r2.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r2.md"
)

# --- frozen inputs (must not be modified by this round) -----------------------
EDGE_MANIFEST = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.jsonl"
EDGE_SUMMARY = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.summary.json"
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
MODEL_IDENTITY = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
METRIC_MODULE = ROOT / "evaluation" / "cv_bench_metric_v2.py"
BUILDER = ROOT / "evaluation" / "build_edge576_cv_bench_manifest.py"
AUDITOR = ROOT / "evaluation" / "audit_edge576_cv_bench_manifest.py"
BUILDER_TEST = ROOT / "evaluation" / "tests" / "test_build_edge576_cv_bench_manifest.py"
AUDITOR_TEST = ROOT / "evaluation" / "tests" / "test_audit_edge576_cv_bench_manifest.py"
PREREG_TEST_R2 = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r2.py"
AUDIT_R2_REPORT = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cv-bench-manifest-audit-r2.json"
)
CORRECTED_L2_RUNNER = ROOT / "evaluation" / "run_hf_hardware_regular_b1_corrected_l2.py"
CORRECTED_L2_INTEGER_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
DENSE_FULL_JSONL = (
    ROOT / "evaluation" / "results" / "cv_bench_l2_0875_full2638_20260916" / "shallow_r1000.jsonl"
)

# --- superseded r1 evidence (kept byte-identical; never re-generated) ---------
R1_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development.preregistered.json"
)
R1_REPORT_JSON = ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg.json"
R1_SUPERSEDED_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-r1-preexecution-superseded.json"
)

# --- existing frozen v2 mechanism chain (no new mechanism) --------------------
V2_CONFIG = ROOT / "configs" / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json"
V2_MECHANISM_CORE = ROOT / "evaluation" / "conservative_l2_guarded_reranker.py"
V2_INTEGRATION = ROOT / "evaluation" / "hardware_regular_b1_guarded_rerank.py"
V2_RUNNER = ROOT / "evaluation" / "run_hf_conservative_l2_guarded_rerank.py"
V2_SUMMARIZER = ROOT / "evaluation" / "summarize_conservative_l2_guarded_rerank_full2638.py"
V2_TEST_MECHANISM = ROOT / "evaluation" / "tests" / "test_conservative_l2_guarded_reranker.py"
V2_TEST_INTEGRATION = ROOT / "evaluation" / "tests" / "test_hardware_regular_b1_guarded_rerank.py"
V2_TEST_RUNNER = ROOT / "evaluation" / "tests" / "test_run_hf_conservative_l2_guarded_rerank.py"
V2_TEST_SUMMARIZER = ROOT / "evaluation" / "tests" / "test_summarize_conservative_l2_guarded_rerank_full2638.py"
V2_PROJECTION_NPZ = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "spatial_prompt_contrastive_trainval"
    / "B_SPATIAL_KL_CONTRASTIVE"
    / "20261003"
    / "best.npz"
)
V2_WV_INT8 = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "selector_wv_scale_regularization"
    / "20261003"
    / "Wv_int8_rowmajor.bin"
)
V2_WV_SCALE = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "selector_wv_scale_regularization"
    / "20261003"
    / "Wv_per_tensor_scale_f32le.bin"
)
V2_NATIVE_ARM_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
    / "conservative_l2_guarded_rerank_v2_k96_full2638.jsonl"
)

# --- integer scoring / corrected-L2 core --------------------------------------
INTEGER_TOKEN_SELECTOR = ROOT / "evaluation" / "hardware_regular_token_selector.py"
INTEGER_CORRECTED_L2 = ROOT / "evaluation" / "hardware_regular_b1_corrected_l2.py"
INTEGER_B1_INTEGRATION = ROOT / "evaluation" / "hardware_regular_b1_integration.py"
INTEGER_CURRENT_SELECTOR = ROOT / "evaluation" / "current_selector_integer.py"
INTEGER_INT8_AUDIT = ROOT / "evaluation" / "selector_int8_numerical_audit.py"
INTEGER_SCORER_AUDIT = ROOT / "evaluation" / "selector_integer_scorer_audit.py"
V1_LAYOUT_REBUILD_CORE = ROOT / "evaluation" / "conservative_l2_residual_reranker.py"
EVALUATOR = ROOT / "evaluation" / "build_cv_bench_pruning_report.py"
METRICS = ROOT / "evaluation" / "metrics.py"

MODEL_PATH = "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B"


class PreregistrationError(RuntimeError):
    """Raised when a required pin input is missing (fail closed)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise PreregistrationError(f"missing pinned input {path} (fail closed)")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "role": role}


def build_config() -> dict[str, Any]:
    summary = json.loads(EDGE_SUMMARY.read_text(encoding="utf-8"))
    config: dict[str, Any] = {
        "version": "2026-10-07-edge576-cvbench-three-arm-development-r2",
        "revision": "r2",
        "date": "2026-10-07",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "arm_status": "NOT_EXECUTED",
        "executed": False,
        "future_only": True,
        "retroactive": False,
        "gpu_runs": 0,
        "purpose": (
            "Corrected future-only edge deployment study superseding r1: under the "
            "G<=576 edge runtime domain, compare Dense@edge576, corrected-integer-"
            "L2@B1/K96 and the frozen existing conservative_l2_guarded_rerank_v2 "
            "candidate on byte-identical inputs."
        ),
        "measurement_class": "development",
        "development_retest_statement": (
            "This is a developmental re-test of the EXISTING, already frozen "
            "conservative_l2_guarded_rerank_v2 mechanism in a new edge576 runtime "
            "domain. No new mechanism is invented and no v2 hyper-parameter is "
            "tuned. The edge576 domain is a different input distribution from the "
            "native-resolution development run, so this is a new falsifiable test "
            "of the same frozen mechanism, not a re-scoring of the old result."
        ),
        "development_set_statement": (
            "CV-Bench full2638 has been repeatedly observed and is a development "
            "set only; passing the gates permits one fresh untouched external "
            "benchmark confirmation, never a publication or deployment claim."
        ),
        "non_retroactive_statement": (
            "The 2026-10-06 native-resolution conservative_l2_guarded_rerank_v2 "
            "DEVELOPMENT_STOP (output SHA "
            "3018f7593d31fd02854f3f63f992b6fc6293ba64c24185e2905877a71d88fdb1) is "
            "frozen and is NOT modified, re-scored or re-adjudicated here; "
            "native-resolution results stay read-only."
        ),
        "supersedes": {
            "path": str(R1_CONFIG.relative_to(ROOT)),
            "sha256": sha256_file(R1_CONFIG),
            "report_json": str(R1_REPORT_JSON.relative_to(ROOT)),
            "report_json_sha256": sha256_file(R1_REPORT_JSON),
            "status": "PREEXECUTION_SUPERSEDED",
            "never_executed": True,
            "must_not_be_run": True,
            "defects": [
                "A: the r1 third arm had mechanism_id=null and therefore was not a pre-registered candidate",
                "B: r1 success_gates.record_completeness required all 2638 samples to be parseable, which conflicts with the honest parse contract; Dense itself emits unparseable outputs",
                "C: the r1 auditor CLI accepted --model but never forwarded it to audit(), and verify_processor silently used DEFAULT_MODEL",
            ],
            "superseded_evidence": str(R1_SUPERSEDED_JSON.relative_to(ROOT)),
        },
        "edge_runtime_domain": {
            "keep_aspect_ratio": True,
            "min_pixels": 65536,
            "max_pixels": 589824,
            "budget_g": 576,
            "patch_size": 16,
            "merge_size": 2,
            "merge_unit": 4,
            "resize_factor": 32,
            "g_definition": (
                "full_llm_visual_tokens = (image_grid_thw[0]*image_grid_thw[1]*"
                "image_grid_thw[2]) // (spatial_merge_size**2)"
            ),
            "identity_bypass_g_le": 96,
            "compression_range": "96 < G <= 576",
            "high_resolution_policy": (
                "native dynamic resolution (up to G=3588) is retained only as a "
                "frozen stress test; it is not the edge promotion gate and its "
                "existing DEVELOPMENT_STOP is not rewritten."
            ),
        },
        "pins": {
            "edge576_manifest": pin(EDGE_MANIFEST, role="edge576_manifest"),
            "edge576_summary": pin(EDGE_SUMMARY, role="edge576_summary"),
            "source_full_manifest": pin(FULL_MANIFEST, role="frozen_full_manifest"),
            "builder": pin(BUILDER, role="edge576_builder"),
            "auditor": pin(AUDITOR, role="edge576_auditor_model_path_fixed"),
            "edge576_audit_r2_report": pin(AUDIT_R2_REPORT, role="edge576_reaudit_report"),
            "builder_test": pin(BUILDER_TEST, role="edge576_builder_test"),
            "auditor_test": pin(AUDITOR_TEST, role="edge576_auditor_test_model_path"),
            "prereg_test_r2": pin(PREREG_TEST_R2, role="edge576_prereg_r2_test"),
            "prereg_generator_r2": pin(
                Path(__file__).resolve(), role="edge576_prereg_r2_generator"
            ),
            "cv_bench_metric_v2": pin(METRIC_MODULE, role="future_metric_v2"),
            "model_identity_manifest": pin(MODEL_IDENTITY, role="model_identity"),
            "corrected_l2_runner": pin(CORRECTED_L2_RUNNER, role="corrected_l2_runner"),
            "corrected_l2_integer_jsonl": pin(
                CORRECTED_L2_INTEGER_JSONL, role="frozen_corrected_l2_control"
            ),
            "dense_full_jsonl": pin(DENSE_FULL_JSONL, role="frozen_native_dense_reference"),
            "v2_config": pin(V2_CONFIG, role="frozen_v2_preregistered_config"),
            "v2_mechanism_core": pin(V2_MECHANISM_CORE, role="frozen_v2_mechanism_core"),
            "v2_integration": pin(V2_INTEGRATION, role="frozen_v2_integration"),
            "v2_runner": pin(V2_RUNNER, role="frozen_v2_runner"),
            "v2_summarizer": pin(V2_SUMMARIZER, role="frozen_v2_summarizer"),
            "v2_test_mechanism": pin(V2_TEST_MECHANISM, role="frozen_v2_test_mechanism"),
            "v2_test_integration": pin(V2_TEST_INTEGRATION, role="frozen_v2_test_integration"),
            "v2_test_runner": pin(V2_TEST_RUNNER, role="frozen_v2_test_runner"),
            "v2_test_summarizer": pin(V2_TEST_SUMMARIZER, role="frozen_v2_test_summarizer"),
            "v2_projection_npz": pin(V2_PROJECTION_NPZ, role="v2_checkpoint_projection_npz"),
            "v2_wv_int8": pin(V2_WV_INT8, role="v2_checkpoint_wv_int8"),
            "v2_wv_per_tensor_scale": pin(V2_WV_SCALE, role="v2_checkpoint_wv_scale"),
            "v2_native_arm_jsonl": pin(V2_NATIVE_ARM_JSONL, role="frozen_native_v2_output"),
            "integer_token_selector_core": pin(
                INTEGER_TOKEN_SELECTOR, role="integer_scoring_token_selector_core"
            ),
            "integer_corrected_l2_core": pin(
                INTEGER_CORRECTED_L2, role="integer_corrected_l2_core"
            ),
            "integer_b1_integration_core": pin(
                INTEGER_B1_INTEGRATION, role="integer_b1_integration_core"
            ),
            "integer_current_selector_core": pin(
                INTEGER_CURRENT_SELECTOR, role="integer_current_selector_core"
            ),
            "integer_int8_numerical_audit": pin(
                INTEGER_INT8_AUDIT, role="integer_int8_numerical_audit"
            ),
            "integer_scorer_audit": pin(
                INTEGER_SCORER_AUDIT, role="integer_scorer_audit"
            ),
            "v1_layout_rebuild_core": pin(
                V1_LAYOUT_REBUILD_CORE, role="v1_layout_rebuild_core_reused_by_v2"
            ),
            "evaluator": pin(EVALUATOR, role="official_cv_bench_evaluator"),
            "metrics": pin(METRICS, role="metrics"),
            "r1_prereg_config": pin(R1_CONFIG, role="superseded_r1_config"),
            "r1_prereg_report_json": pin(R1_REPORT_JSON, role="superseded_r1_report"),
            "r1_preexecution_superseded_json": pin(
                R1_SUPERSEDED_JSON, role="r1_preexecution_superseded_evidence"
            ),
        },
        "model_identity": {
            "default_model_path": MODEL_PATH,
            "manifest_path": str(MODEL_IDENTITY.relative_to(ROOT)),
            "weights_loaded_in_this_round": False,
            "processor_audit_model_path": MODEL_PATH,
        },
        "manifest_contract": {
            "sample_count": summary["sample_count"],
            "unique_sample_id_count": summary["unique_sample_id_count"],
            "all_g_within_budget": summary["all_g_within_budget"],
            "g_min": summary["g_min"],
            "g_max": summary["g_max"],
            "g_histogram": summary["g_histogram"],
            "labels_must_match_frozen_full_manifest": True,
            "no_sample_may_be_deleted_for_size": True,
        },
        "shared_arm_contract": {
            "manifest": str(EDGE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha256_file(EDGE_MANIFEST),
            "prompt_source": (
                "the edge576 manifest `prompt` field, byte-identical for all three arms"
            ),
            "model": MODEL_PATH,
            "dtype": "bf16",
            "attention_implementation": "sdpa",
            "device": "cuda:0",
            "do_sample": False,
            "greedy": True,
            "max_new_tokens": 16,
            "processor": {
                "min_pixels": 65536,
                "max_pixels": 589824,
                "keep_aspect_ratio": True,
                "patch_size": 16,
                "merge_size": 2,
            },
            "grid_verification": (
                "every arm must assert that its processor-produced image_grid_thw "
                "equals the value recorded in the edge576 manifest for that "
                "sample, otherwise it fails closed; native-resolution pixel "
                "budgets are forbidden for these arms"
            ),
        },
        "arms": {
            "dense_edge576": {
                "role": "quality denominator (Dense@edge576)",
                "cutpoint": None,
                "selector_budget": None,
                "output_path": (
                    "artifacts/token_selector/2026-10-07-edge576-cvbench/"
                    "dense_edge576_full2638.jsonl"
                ),
                "output_sha256": None,
                "status": "NOT_EXECUTED",
                "edge576_processor_budget_required": True,
            },
            "corrected_l2_integer_b1_k96": {
                "role": "hardware-rule control (corrected integer L2 @ B1, K96)",
                "cutpoint": "B1_after_vision_block_0",
                "prune_after_layer": 0,
                "selector_budget": {
                    "K": 96,
                    "region_grid": 3,
                    "local_candidate_cap": 16,
                    "identity_bypass_g_le": 96,
                },
                "runner": str(CORRECTED_L2_RUNNER.relative_to(ROOT)),
                "output_path": (
                    "artifacts/token_selector/2026-10-07-edge576-cvbench/"
                    "corrected_l2_integer_k96_full2638.jsonl"
                ),
                "output_sha256": None,
                "status": "NOT_EXECUTED",
                "edge576_processor_budget_required": True,
            },
            "v2_edge576_b1_k96": {
                "role": (
                    "frozen pre-registered candidate (existing "
                    "conservative_l2_guarded_rerank_v2; no new mechanism)"
                ),
                "mechanism_id": "conservative_l2_guarded_rerank_v2",
                "mechanism_family": "conservative_l2_guarded_rerank",
                "is_new_mechanism": False,
                "status": "NOT_EXECUTED",
                "cutpoint": "B1_after_vision_block_0",
                "prune_after_layer": 0,
                "selector_budget": {
                    "K": 96,
                    "region_grid": 3,
                    "local_candidate_cap": 16,
                    "max_swaps_S": 16,
                    "identity_bypass_g_le": 96,
                },
                "runner": str(V2_RUNNER.relative_to(ROOT)),
                "summariser": str(V2_SUMMARIZER.relative_to(ROOT)),
                "output_path": (
                    "artifacts/token_selector/2026-10-07-edge576-cvbench/"
                    "v2_edge576_k96_full2638.jsonl"
                ),
                "output_sha256": None,
                "edge576_processor_budget_required": True,
                "reuses_native_resolution_v2_mechanism": True,
                "native_resolution_v2_output_sha256": sha256_file(V2_NATIVE_ARM_JSONL),
                "native_resolution_result_not_rewritten": True,
                "allowed_mechanism_scope": (
                    "the frozen conservative_l2_guarded_rerank_v2 mechanism only; "
                    "no new mechanism, no v2 re-tuning on edge576, no G/source/task "
                    "routing, no label/answer leakage"
                ),
            },
        },
        "frozen_candidate_mechanism": {
            "mechanism_id": "conservative_l2_guarded_rerank_v2",
            "source_config": str(V2_CONFIG.relative_to(ROOT)),
            "source_config_sha256": sha256_file(V2_CONFIG),
            "utility_formula": "semantic_q15 + corrected_l2_q15",
            "legal_gain_formula": "semantic_gain - l2_loss = utility_added - utility_removed",
            "context_pooling_basis": "corrected_integer_l2_norm_q15",
            "anchors_frozen": True,
            "every_replacement_is_same_region_one_for_one": True,
            "max_swaps_S": 16,
            "K_remains_exactly_96": True,
            "real_group_count": 87,
            "context_group_count": 9,
            "no_labels_or_answer_tokens_or_gradients": True,
        },
        "primary_metric": {
            "name": "cv_bench_accuracy",
            "module": str(METRIC_MODULE.relative_to(ROOT)),
            "module_sha256": sha256_file(METRIC_MODULE),
            "weighting": "((accuracy_ADE20K + accuracy_COCO) / 2 + accuracy_Omni3D) / 2",
            "correct_key": "correct",
            "parsed_answer_key": "parsed_answer",
            "keys_must_be_distinct": True,
            "never_infer_parsed_answer_key_from_correct_key": True,
            "parse_failures_count_as_wrong": True,
            "secondary_diagnostics_only": [
                "unweighted_sample_accuracy",
                "per_source_accuracy",
                "per_task_accuracy",
            ],
        },
        "success_gates": {
            "official_cv_bench_accuracy": {
                "metric": "cv_bench_accuracy",
                "module": str(METRIC_MODULE.relative_to(ROOT)),
                "weighting": "((accuracy_ADE20K + accuracy_COCO) / 2 + accuracy_Omni3D) / 2",
                "required": True,
            },
            "quality_retention_vs_same_input_dense_edge576": {
                "definition": (
                    "selector_cv_bench_accuracy / dense_edge576_cv_bench_accuracy"
                ),
                "threshold": 0.90,
                "comparison": ">=",
                "note": (
                    "must never compare an edge selector against a native-resolution "
                    "dense score"
                ),
            },
            "relative_to_corrected_integer_l2": {
                "paired_bootstrap": {
                    "strata": "task::source_dataset",
                    "repeats": 5000,
                    "seed": 20261007,
                    "confidence": 0.95,
                    "lower_bound": ">=0",
                },
                "point_estimate": ">=0",
            },
            "layout_contract": {
                "K": 96,
                "real_slots": 87,
                "context_slots": 9,
                "identity_bypass_g_le": 96,
                "applicable_if_layout_is_reused": True,
            },
            "record_completeness": {
                "definition": (
                    "exactly 2638 unique sample_id, each with one complete output "
                    "record; completeness is about record presence, never about "
                    "parse success"
                ),
                "sample_count": 2638,
                "unique_sample_id_count": 2638,
                "all_2638_unique_complete_outputs": True,
                "parsed_answer_may_be_null": True,
                "parse_failures_count_as_wrong": True,
                "requires_100_percent_parse": False,
                "must_report": [
                    "parse_success_count",
                    "parse_failure_count",
                    "parse_success_rate",
                    "parse_failure_rate",
                ],
            },
            "honest_parse_reporting": {
                "source_field": "parsed_answer_key only",
                "parsed_answer_may_be_null": True,
                "parse_failures_count_as_wrong": True,
                "must_report": [
                    "parse_success_count",
                    "parse_failure_count",
                    "parse_success_rate",
                    "parse_failure_rate",
                ],
                "requires_100_percent_parse": False,
                "aliasing_correct_key_forbidden": True,
                "note": (
                    "there is no 100% parse gate; parse failures are counted as "
                    "wrong answers and reported separately"
                ),
            },
            "no_label_leakage": True,
        },
        "forbidden_claims": [
            "full2638 is an untouched confirmation set",
            "a development gate pass is a publication or deployment claim",
            "an edge score may be compared to the frozen native-resolution dense score",
            "the 2026-10-06 native-resolution DEVELOPMENT_STOP is rewritten",
            "the edge576 v2 re-test retroactively changes the native-resolution v2 result",
            "the r1 preregistration was executed or is executable",
        ],
        "next_allowed_action": (
            "Adapt the three arm runners to consume the edge576 manifest with the "
            "pinned processor budgets (min_pixels=65536, max_pixels=589824) and "
            "per-sample image_grid_thw assertions, then run Dense@edge576, the "
            "corrected integer L2@B1/K96 control and the frozen existing "
            "conservative_l2_guarded_rerank_v2@B1/K96 candidate to the output "
            "paths named here; score with cv_bench_metric_v2 (reporting parse "
            "success/failure counts and rates) and audit before any external "
            "validation or hardware claim."
        ),
        "runner_adaptation_entry": {
            "dense": (
                "evaluation/run_hf_shallow_block_pruning.py with --retentions 1.0 "
                "(the frozen producer of the dense reference shallow_r1000.jsonl), "
                "re-pointed to the edge576 manifest and processor budgets"
            ),
            "corrected_l2_integer": str(CORRECTED_L2_RUNNER.relative_to(ROOT)),
            "v2_candidate": str(V2_RUNNER.relative_to(ROOT)),
            "required_edge576_inputs": {
                "manifest": str(EDGE_MANIFEST.relative_to(ROOT)),
                "min_pixels": 65536,
                "max_pixels": 589824,
                "assert_image_grid_thw_equals_manifest": True,
            },
            "status": "RUNNER_ADAPTATION_NOT_YET_IMPLEMENTED_NO_MODEL_RUN",
        },
    }
    return config


def render_markdown(config: Mapping[str, Any]) -> str:
    manifest = config["manifest_contract"]
    hist = manifest["g_histogram"]
    pins = config["pins"]
    lines = [
        "# edge576 CV-Bench 三臂未来预注册 r2（2026-10-07）",
        "",
        f"- 状态：`{config['status']}`（`executed=false`、`gpu_runs=0`、仅 CPU/只读、"
        "未加载模型权重）；第三臂状态 `NOT_EXECUTED`",
        f"- 取代已作废的 r1：`{pins['r1_prereg_config']['path']}` SHA "
        f"`{pins['r1_prereg_config']['sha256']}`（r1 `PREEXECUTION_SUPERSEDED`，"
        "从未执行且不得运行）",
        f"- edge576 manifest：`{pins['edge576_manifest']['path']}` SHA "
        f"`{pins['edge576_manifest']['sha256']}`",
        f"- 源冻结 full manifest：SHA `{pins['source_full_manifest']['sha256']}`",
        f"- 修正后审计器：`{pins['auditor']['path']}` SHA `{pins['auditor']['sha256']}`；"
        f"重新审计报告：`{pins['edge576_audit_r2_report']['path']}` SHA "
        f"`{pins['edge576_audit_r2_report']['sha256']}`",
        f"- 样本数：{manifest['sample_count']}（唯一 {manifest['unique_sample_id_count']}）；"
        f"全部 `G<=576`：{manifest['all_g_within_budget']}",
        f"- G：min/max = {manifest['g_min']}/{manifest['g_max']}；分桶 "
        f"<=96={hist['g_le_96']}、97-256={hist['g_97_256']}、"
        f"257-512={hist['g_257_512']}、513-576={hist['g_513_576']}",
        "",
        "## r1 三项缺陷与 r2 修正",
        "",
        "1. **A（第三臂未预注册）**：r1 第三臂 `mechanism_id=null`，不是已预注册候选。"
        "r2 将其明确冻结为**现有** `conservative_l2_guarded_rerank_v2`，"
        "`is_new_mechanism=false`，不发明新机制。",
        "2. **B（记录完整性口径错误）**：r1 写「2638 全部可解析」与诚实解析口径冲突"
        "（Dense 也会有不可解析输出）。r2 只要求「2638 个唯一完整输出」，"
        "`parsed_answer` 可 `null`，解析失败按错误计并单独报告成功/失败数/率，"
        "**不设 100% 解析门**。",
        "3. **C（审计器 `--model` 丢失）**：r1 审计器 CLI 接受 `--model` 却未传入 "
        "`audit()`，`verify_processor` 硬用 `DEFAULT_MODEL`。r2 已修复：`audit()` 显式"
        "接受 `model_path`，`main` 传入 `--model`，报告记录实际处理器路径。",
        "",
        "## 三臂共享合同（已锁定）",
        "",
        "- 同一 edge576 manifest 与同一 Prompt；BF16/SDPA；`cuda:0`；",
        "- `do_sample=false`（greedy）、`max_new_tokens=16`；",
        "- 处理参数 `min_pixels=65536`、`max_pixels=589824`、保持长宽比、"
        "`patch_size=16`、`merge_size=2`；",
        "- 每臂必须断言 processor 的 `image_grid_thw` 与 manifest 记录逐样本一致，"
        "否则 fail closed；禁止使用原生分辨率预算。",
        "",
        "## 三臂（与 r1 相同的输入与切点）",
        "",
        "1. `dense_edge576`（同输入质量分母）；",
        "2. `corrected_l2_integer_b1_k96`（B1 after vision block 0、K96、grid3、cap16、"
        "`G<=96` identity bypass）——冻结整数 L2 控制臂；",
        "3. `v2_edge576_b1_k96`（同切点、同 K）——冻结现有 "
        "`conservative_l2_guarded_rerank_v2`，输出路径 `"
        "artifacts/token_selector/2026-10-07-edge576-cvbench/v2_edge576_k96_full2638.jsonl`，"
        "`status=NOT_EXECUTED`、`output_sha256=null`。",
        "",
        "## 成功门（保留 r1 的有效门）",
        "",
        "- 主指标仅官方 `cv_bench_accuracy`（`((ADE20K+COCO)/2+Omni3D)/2`）；",
        "- 相对**同输入** Dense@edge576 保留率 `>= 0.90`；",
        "- 相对同切点/同 K 修正整数 L2，按 `task::source_dataset` 分层配对 bootstrap"
        "（5000 次、seed 20261007、95%）CI 下界 `>=0`；",
        "- 布局 K96/87 real+9 context（`G<=96` identity bypass）；",
        "- 记录完整性：2638 个唯一完整输出（解析可失败、可 `null`，失败计错并报告）；",
        "- 无标签/答案/task/source 泄漏。",
        "",
        "## 口径与边界",
        "",
        "- 这是**现有 v2 机制在新边缘运行域（G<=576）的发展性复测**，不是新机制，"
        "也不在 edge576 上调参；",
        "- 2026-10-06 原生分辨率 v2 `DEVELOPMENT_STOP` 不追溯改写，原生结果只读；",
        "- full2638 是反复观测的发展集，通过只允许一次新的未触碰外部基准确认；",
        "- 本预注册不代表任何精度/部署/加速/FPGA 时延主张，也未启动任何模型推理；",
        "- r1 从未执行、不得运行，仅作为未执行设计缺陷证据保留字节不变。",
        "",
        "## 下一步 runner 适配入口",
        "",
        f"- Dense 适配入口：`{config['runner_adaptation_entry']['dense']}`",
        f"- 修正整数 L2 入口：`{config['runner_adaptation_entry']['corrected_l2_integer']}`",
        f"- v2 候选入口：`{config['runner_adaptation_entry']['v2_candidate']}`",
        "- 必须消费 edge576 manifest，`min_pixels=65536`、`max_pixels=589824`，"
        "并逐样本断言 `image_grid_thw` 等于 manifest；未适配前不运行任何模型。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    args = parser.parse_args(argv)

    config = build_config()
    config["generated_at"] = datetime.now(timezone.utc).isoformat()
    config["generator"] = "scripts/preregister_edge576_cvbench_r2_20261007.py"
    config["generator_sha256"] = sha256_file(Path(__file__).resolve())

    # Self-check: every declared pin must re-verify on the spot.
    for name, entry in config["pins"].items():
        observed = sha256_file(ROOT / entry["path"])
        if observed != entry["sha256"]:
            raise PreregistrationError(
                f"pin {name} mismatch: {observed} != {entry['sha256']}"
            )

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "status": config["status"],
        "arm_status": config["arm_status"],
        "revision": config["revision"],
        "date": config["date"],
        "gpu_runs": 0,
        "executed": False,
        "pins_verified": True,
        "pin_count": len(config["pins"]),
        "pins": config["pins"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "supersedes": config["supersedes"],
        "manifest_contract": config["manifest_contract"],
        "shared_arm_contract": config["shared_arm_contract"],
        "arms": config["arms"],
        "frozen_candidate_mechanism": config["frozen_candidate_mechanism"],
        "success_gates": config["success_gates"],
        "runner_adaptation_entry": config["runner_adaptation_entry"],
        "next_allowed_action": config["next_allowed_action"],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(render_markdown(config), encoding="utf-8")
    print(
        "EDGE576_R2_PREREGISTERED "
        f"config={args.config.name} sha256={report['config_sha256']} "
        f"pins={report['pin_count']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreregistrationError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
