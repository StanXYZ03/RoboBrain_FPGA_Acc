#!/usr/bin/env python3
"""Generate the 2026-10-07 edge576 CV-Bench three-arm preregistration JSON.

CPU-only, read-only over every pinned input.  This script never runs a model:
it recomputes the SHA256 of the edge576 builder/auditor/tests, the built edge576
manifest and summary, the frozen full manifest, the model identity manifest, the
future-only metric module and the two frozen comparison-arm outputs, then writes
``configs/2026-10-07-edge576-cvbench-three-arm-development.preregistered.json``
plus a Chinese preregistration report.

The preregistration locks the *shared* contract of the three edge arms
(Dense@edge576, corrected integer L2@B1/K96, next candidate@B1/K96): identical
edge576 input, identical Prompt, greedy decoding and ``max_new_tokens=16``.
The next-candidate mechanism itself stays ``TO_BE_SELECTED`` under the
independent policy review; only its slot in the comparison is frozen.
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
    / "2026-10-07-edge576-cvbench-three-arm-development.preregistered.json"
)
DEFAULT_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg.md"
)

EDGE_MANIFEST = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.jsonl"
EDGE_SUMMARY = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.summary.json"
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
MODEL_IDENTITY = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
METRIC_MODULE = ROOT / "evaluation" / "cv_bench_metric_v2.py"
BUILDER = ROOT / "evaluation" / "build_edge576_cv_bench_manifest.py"
AUDITOR = ROOT / "evaluation" / "audit_edge576_cv_bench_manifest.py"
BUILDER_TEST = ROOT / "evaluation" / "tests" / "test_build_edge576_cv_bench_manifest.py"
AUDITOR_TEST = ROOT / "evaluation" / "tests" / "test_audit_edge576_cv_bench_manifest.py"
PREREG_TEST = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg.py"
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
    import json as _json  # local alias for clarity

    summary = _json.loads(EDGE_SUMMARY.read_text(encoding="utf-8"))
    config: dict[str, Any] = {
        "version": "2026-10-07-edge576-cvbench-three-arm-development-r1",
        "date": "2026-10-07",
        "status": "PREREGISTERED_NOT_EXECUTED",
        "executed": False,
        "future_only": True,
        "retroactive": False,
        "gpu_runs": 0,
        "purpose": (
            "Future-only edge deployment study: re-preprocess the full CV-Bench "
            "2638 development set under the G<=576 edge runtime domain and "
            "compare Dense@edge576, corrected-integer-L2@B1/K96 and the next "
            "pre-registered candidate on identical inputs."
        ),
        "measurement_class": "development",
        "development_set_statement": (
            "CV-Bench full2638 has been repeatedly observed and is a development "
            "set only; passing the gates permits one fresh untouched external "
            "benchmark confirmation, never a publication or deployment claim."
        ),
        "non_retroactive_statement": (
            "The 2026-10-06 native-resolution conservative_l2_guarded_rerank_v2 "
            "DEVELOPMENT_STOP is frozen and is NOT modified, re-scored or "
            "re-adjudicated here; native-resolution results stay read-only."
        ),
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
            "auditor": pin(AUDITOR, role="edge576_auditor"),
            "builder_test": pin(BUILDER_TEST, role="edge576_builder_test"),
            "auditor_test": pin(AUDITOR_TEST, role="edge576_auditor_test"),
            "prereg_test": pin(PREREG_TEST, role="edge576_prereg_test"),
            "cv_bench_metric_v2": pin(METRIC_MODULE, role="future_metric_v2"),
            "model_identity_manifest": pin(MODEL_IDENTITY, role="model_identity"),
            "corrected_l2_runner": pin(CORRECTED_L2_RUNNER, role="corrected_l2_runner"),
            "corrected_l2_integer_jsonl": pin(
                CORRECTED_L2_INTEGER_JSONL, role="frozen_corrected_l2_control"
            ),
            "dense_full_jsonl": pin(DENSE_FULL_JSONL, role="frozen_native_dense_reference"),
        },
        "model_identity": {
            "default_model_path": MODEL_PATH,
            "manifest_path": str(MODEL_IDENTITY.relative_to(ROOT)),
            "weights_loaded_in_this_round": False,
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
            "next_candidate_b1_k96": {
                "role": "pre-registered candidate (mechanism pending policy review)",
                "mechanism_id": None,
                "status": "TO_BE_SELECTED_UNDER_INDEPENDENT_POLICY_REVIEW_20261005-001",
                "cutpoint": "B1_after_vision_block_0",
                "selector_budget": {
                    "K": 96,
                    "region_grid": 3,
                    "local_candidate_cap": 16,
                    "identity_bypass_g_le": 96,
                },
                "output_path": (
                    "artifacts/token_selector/2026-10-07-edge576-cvbench/"
                    "next_candidate_k96_full2638.jsonl"
                ),
                "output_sha256": None,
                "allowed_mechanism_scope": (
                    "single-variable change relative to corrected integer L2 at the "
                    "same cutpoint and K; no G/source/task routing, no label/answer "
                    "leakage, no post-hoc threshold tuning on full2638"
                ),
                "edge576_processor_budget_required": True,
            },
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
            "quality_retention": {
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
            "honest_parse_gate": {
                "source_field": "parsed_answer_key only",
                "must_report": [
                    "parse_success_rate",
                    "parse_success_count",
                    "parse_failure_count",
                ],
                "aliasing_correct_key_forbidden": True,
            },
            "no_label_leakage": True,
            "record_completeness": "all 2638 unique samples present and parseable",
        },
        "forbidden_claims": [
            "full2638 is an untouched confirmation set",
            "a development gate pass is a publication or deployment claim",
            "an edge score may be compared to the frozen native-resolution dense score",
            "the 2026-10-06 native-resolution DEVELOPMENT_STOP is rewritten",
            "any G/source/task routing may be used as the candidate mechanism",
        ],
        "next_allowed_action": (
            "Adapt the three arm runners to consume the edge576 manifest with the "
            "pinned processor budgets and per-sample image_grid_thw assertions, then "
            "run Dense@edge576, corrected integer L2@B1/K96 and the frozen next "
            "candidate; score with cv_bench_metric_v2 and audit before any external "
            "validation or hardware claim."
        ),
    }
    return config


def render_markdown(config: Mapping[str, Any]) -> str:
    manifest = config["manifest_contract"]
    hist = manifest["g_histogram"]
    lines = [
        "# edge576 CV-Bench 三臂未来预注册（2026-10-07）",
        "",
        f"- 状态：`{config['status']}`（`executed=false`、`gpu_runs=0`、"
        "仅 CPU/只读、未加载模型权重）",
        f"- edge576 manifest：`{config['pins']['edge576_manifest']['path']}` "
        f"SHA `{config['pins']['edge576_manifest']['sha256']}`",
        f"- 源冻结 full manifest：SHA "
        f"`{config['pins']['source_full_manifest']['sha256']}`",
        f"- 样本数：{manifest['sample_count']}（唯一 {manifest['unique_sample_id_count']}）；"
        f"全部 `G<=576`：{manifest['all_g_within_budget']}",
        f"- G：min/max = {manifest['g_min']}/{manifest['g_max']}；分桶 "
        f"<=96={hist['g_le_96']}、97-256={hist['g_97_256']}、"
        f"257-512={hist['g_257_512']}、513-576={hist['g_513_576']}",
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
        "## 三臂",
        "",
        "1. `dense_edge576`（质量分母）；",
        "2. `corrected_l2_integer_b1_k96`（B1 after vision block 0、K96、grid3、cap16、"
        "`G<=96` identity bypass）——冻结整数 L2 控制臂；",
        "3. `next_candidate_b1_k96`（同切点、同 K、单变量；机制待独立策略审查后选定，"
        "`mechanism_id=null`）。",
        "",
        "## 成功门",
        "",
        "- 质量保留率 = `selector_cv_bench_accuracy / dense_edge576_cv_bench_accuracy` "
        "`>= 0.90`；",
        "- 相对同切点/同 K 修正整数 L2，按 `task::source_dataset` 分层配对 bootstrap"
        "（5000 次、seed 20261007、95%）CI 下界 `>=0`；",
        "- 主指标仅 `cv_bench_accuracy`，诚实解析门使用显式 `parsed_answer_key`；",
        "- 无标签泄漏；2638 全集完整且可解析。",
        "",
        "## 口径与边界",
        "",
        "- full2638 是反复观测的发展集，通过只允许一次新的未触碰外部基准确认；",
        "- 2026-10-06 原生分辨率 `DEVELOPMENT_STOP` 不追溯修改；",
        "- 原生高分辨率仅作压力测试，不作边缘主档位晋升门；",
        "- 本预注册不代表任何精度/部署/加速/FPGA 时延主张，也未启动任何模型推理。",
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
    config["generator"] = "scripts/preregister_edge576_cvbench_20261007.py"
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
        "date": config["date"],
        "gpu_runs": 0,
        "executed": False,
        "pins_verified": True,
        "pins": config["pins"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": sha256_file(args.config),
        "manifest_contract": config["manifest_contract"],
        "shared_arm_contract": config["shared_arm_contract"],
        "arms": config["arms"],
        "success_gates": config["success_gates"],
        "next_allowed_action": config["next_allowed_action"],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(render_markdown(config), encoding="utf-8")
    print(
        "EDGE576_PREREGISTERED "
        f"config={args.config.name} sha256={report['config_sha256']} "
        f"pins={len(config['pins'])}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreregistrationError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
