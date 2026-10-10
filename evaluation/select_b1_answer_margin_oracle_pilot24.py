#!/usr/bin/env python3
"""Freeze the 24-sample manifest for the B1 answer-margin oracle K96 pilot.

CPU-only and read-only over the manifest metadata.  It loads only the **local
processor/tokenizer** (``local_files_only=True``, no model weights, no CUDA, no
network) to obtain the authoritative processor grid and the real-tokenizer
label-branch contract.  It never reads dense/L2/correctness outputs.

Selection rule (pre-registered, deterministic)
---------------------------------------------
For each of exactly three fixed ``source_dataset`` groups
(``ADE20K Validation Set``, ``COCO 2017 Validation Set``, ``Omni3D_Hypersim``):

1. start from the frozen ``datasets/cv_bench_full2638.jsonl`` metadata;
2. exclude ``sample_id == "1"``, every id in
   ``datasets/cv_bench_screening100.jsonl`` and every id in the same-image
   counterfactual 100 manifest;
3. require the real-tokenizer label-branch contract
   (common prefix + exactly one distinct branch token + common suffix);
4. require ``full_llm_visual_tokens > 96`` (compression active at K=96);
5. sort the remaining candidates by ``sha256(sample_id)`` ascending (ties by
   ``sample_id``) and take the first eight.

Any source with fewer than eight eligible candidates fails closed.  The chosen
manifest and the pre-registered configuration are written together with every
input/script/manifest SHA256 pin.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.b1_answer_margin_oracle import (  # noqa: E402
    ARM_ORACLE,
    BUDGET,
    CUTPOINT,
    DESIGN,
    EXPECTED_CONTEXT,
    EXPECTED_OUTPUT,
    EXPECTED_REAL,
    GATE_ORACLE_MINUS_CONTROL_NET_MIN,
    GATE_ORACLE_PARSE_MIN,
    GATE_R_TO_W_MAX,
    GATE_W_TO_R_MIN,
    LABEL_AWARE_ORACLE,
    LOCAL_CANDIDATE_CAP,
    MAX_NEW_TOKENS,
    PILOT_PER_SOURCE,
    PILOT_SAMPLE_COUNT,
    PRUNE_AFTER_LAYER,
    REGION_GRID_FIXED,
    SALIENCY_DEFINITION,
    SOURCE_DATASETS,
    SOURCE_GROUP_OF,
    STATUS_PREREGISTERED_NOT_EXECUTED,
    OraclePilotError,
    file_sha256,
    stable_sha256_rank,
)

ROOT = Path(__file__).resolve().parents[1]

FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
SCREENING_MANIFEST = ROOT / "datasets" / "cv_bench_screening100.jsonl"
COUNTERFACTUAL_MANIFEST = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl"
)
DENSE_FULL_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
CONTROL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
OUTPUT_MANIFEST = (
    ROOT / "datasets" / "2026-10-05-b1-answer-margin-oracle-pilot24.jsonl"
)
OUTPUT_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-05-b1-answer-margin-oracle-pilot24.preregistered.json"
)
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_JSON = REPORT_DIR / "2026-10-05-b1-answer-margin-oracle-pilot-prereg.json"
REPORT_MD = REPORT_DIR / "2026-10-05-b1-answer-margin-oracle-pilot-prereg.md"
ORACLE_OUTPUT = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-05-b1-answer-margin-oracle-pilot24"
    / "oracle_answer_margin_b1_saliency_k96.jsonl"
)

SELECTION_SCRIPT = ROOT / "evaluation" / "select_b1_answer_margin_oracle_pilot24.py"
SHARED_HELPERS = ROOT / "evaluation" / "b1_answer_margin_oracle.py"
GPU_RUNNER = ROOT / "evaluation" / "run_b1_answer_margin_oracle_pilot.py"
SUMMARIZER = ROOT / "evaluation" / "summarize_b1_answer_margin_oracle_pilot.py"
SELECTION_TEST = (
    ROOT / "evaluation" / "tests" / "test_select_b1_answer_margin_oracle_pilot24.py"
)
RUNNER_TEST = (
    ROOT / "evaluation" / "tests" / "test_run_b1_answer_margin_oracle_pilot.py"
)
SUMMARIZER_TEST = (
    ROOT
    / "evaluation"
    / "tests"
    / "test_summarize_b1_answer_margin_oracle_pilot.py"
)

SELECTION_CRITERIA = (
    "frozen datasets/cv_bench_full2638.jsonl manifest metadata only",
    "real local tokenizer label branch contract (prefix + one distinct branch "
    "token + common suffix)",
    "full_llm_visual_tokens > 96 (compression active at K=96)",
    "stable sha256(sample_id) ascending, first 8 per source_dataset",
)


class SelectionError(RuntimeError):
    """Raised when the selection or a pin invariant is violated."""


# ---------------------------------------------------------------------------
# Pure, CPU-testable selection (no processor, no correctness)
# ---------------------------------------------------------------------------
def select_from_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    exclusions: Sequence[str] | set[str],
    per_source: int = PILOT_PER_SOURCE,
    source_datasets: Sequence[str] = SOURCE_DATASETS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select ``per_source`` candidates per source; never reads correctness.

    Each candidate must carry ``sample_id``, ``source_dataset``,
    ``full_llm_visual_tokens`` and ``label_branch_ok``.  Any extra field (for
    example a ``correct`` flag) is ignored: this function has no correctness
    input.
    """

    if isinstance(per_source, bool) or not isinstance(per_source, int) or per_source <= 0:
        raise OraclePilotError("per_source must be a positive integer")
    sources = tuple(str(source) for source in source_datasets)
    if len(set(sources)) != len(sources):
        raise OraclePilotError("source_datasets must be unique")
    excluded = {str(value) for value in exclusions}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for candidate in candidates:
        for field in ("sample_id", "source_dataset", "full_llm_visual_tokens", "label_branch_ok"):
            if field not in candidate:
                raise OraclePilotError(f"candidate is missing field {field!r}")
        sample_id = str(candidate["sample_id"])
        if sample_id in seen_ids:
            raise OraclePilotError(f"duplicate candidate sample_id {sample_id!r}")
        seen_ids.add(sample_id)
        source = str(candidate["source_dataset"])
        if source not in sources:
            continue
        if sample_id in excluded:
            continue
        try:
            full_groups = int(candidate["full_llm_visual_tokens"])
        except (TypeError, ValueError) as exc:
            raise OraclePilotError(
                f"candidate {sample_id!r} full_llm_visual_tokens is not an integer"
            ) from exc
        if full_groups <= BUDGET:
            continue
        if not bool(candidate["label_branch_ok"]):
            continue
        grouped[source].append(dict(candidate))

    selected: list[dict[str, Any]] = []
    per_source_stats: list[dict[str, Any]] = []
    for source in sources:
        pool = sorted(
            grouped.get(source, []),
            key=lambda row: (stable_sha256_rank(row["sample_id"]), str(row["sample_id"])),
        )
        if len(pool) < per_source:
            raise OraclePilotError(
                f"source_dataset {source!r} has only {len(pool)} eligible "
                f"candidates (< {per_source}); fail closed"
            )
        for rank, row in enumerate(pool[:per_source]):
            record = dict(row)
            record["source_dataset"] = source
            record["source_group"] = SOURCE_GROUP_OF.get(source, source)
            record["selection_rank"] = rank
            record["selection_sha256_rank"] = stable_sha256_rank(row["sample_id"])
            selected.append(record)
        per_source_stats.append(
            {
                "source_dataset": source,
                "source_group": SOURCE_GROUP_OF.get(source, source),
                "n_eligible": len(pool),
                "selected": per_source,
                "min_sha256_rank_sample_id": pool[0]["sample_id"] if pool else None,
                "max_selected_sha256_rank_sample_id": pool[per_source - 1]["sample_id"],
            }
        )

    if len(selected) != len(sources) * per_source:
        raise OraclePilotError("internal error: selected count does not match sources")
    metadata = {
        "criteria": list(SELECTION_CRITERIA),
        "per_source": per_source,
        "source_datasets": list(sources),
        "exclusion_count": len(excluded),
        "sort": "sha256(sample_id) ascending, tie-break sample_id",
        "per_source_stats": per_source_stats,
        "reads_correctness": False,
        "reads_dense_or_l2_outputs": False,
    }
    return selected, metadata


# ---------------------------------------------------------------------------
# Processor-backed measurement
# ---------------------------------------------------------------------------
def measure_candidate(
    processor: Any,
    sample: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Measure processor grid G and the real-tokenizer label branch contract."""

    from PIL import Image  # noqa: PLC0415

    from evaluation.run_b1_answer_margin_gpu_smoke import (  # noqa: PLC0415
        answer_index,
        build_option_labels,
        parse_label_branches,
        tokenize_labels,
    )
    from evaluation.run_hf_token_reduction import (  # noqa: PLC0415
        make_messages,
        processor_inputs,
        resolve_image,
    )

    sample_id = str(sample["sample_id"])
    source = str(sample["source_dataset"])
    choices = [str(choice) for choice in sample.get("choices", [])]
    labels = build_option_labels(len(choices))
    answer = str(sample.get("answer", ""))
    correct_index = answer_index(answer, labels)

    image_path = resolve_image(manifest_path, str(sample["image"]))
    messages = make_messages(image_path, str(sample["prompt"]))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    processed = processor_inputs(processor, messages, image)
    grid = [int(value) for value in processed.image_grid_thw[0].tolist()]
    full_patch_tokens = int(processed.image_grid_thw.prod().item())
    merge_unit = 4
    if full_patch_tokens % merge_unit:
        raise SelectionError(f"sample {sample_id!r} grid is not merger-aligned")
    full_groups = full_patch_tokens // merge_unit

    label_token_ids = tokenize_labels(processor.tokenizer, labels)
    branches = parse_label_branches(label_token_ids)
    return {
        "sample_id": sample_id,
        "source_dataset": source,
        "source_group": SOURCE_GROUP_OF.get(source, source),
        "task": str(sample.get("task")),
        "source": str(sample.get("source")),
        "full_llm_visual_tokens": int(full_groups),
        "image_grid_thw": grid,
        "labels": labels,
        "label_token_ids": label_token_ids,
        "answer": answer,
        "correct_index": int(correct_index),
        "common_prefix": list(branches.common_prefix),
        "common_suffix": list(branches.common_suffix),
        "branch_token_ids": list(branches.branch_token_ids),
        "label_branch_ok": True,
    }


def build_candidates(
    manifest_rows: Sequence[Mapping[str, Any]],
    exclusions: Sequence[str] | set[str],
    processor: Any,
    manifest_path: Path,
    *,
    per_source: int = PILOT_PER_SOURCE,
) -> list[dict[str, Any]]:
    """Measure candidates in stable SHA order until each source has 8 eligible."""

    excluded = {str(value) for value in exclusions}
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        source = str(row.get("source_dataset"))
        if source not in SOURCE_DATASETS:
            continue
        if str(row["sample_id"]) in excluded:
            continue
        by_source[source].append(row)
    for source in SOURCE_DATASETS:
        by_source[source].sort(
            key=lambda row: (stable_sha256_rank(row["sample_id"]), str(row["sample_id"]))
        )

    measured: list[dict[str, Any]] = []
    for source in SOURCE_DATASETS:
        found = 0
        for row in by_source.get(source, []):
            candidate = measure_candidate(processor, row, manifest_path)
            measured.append(candidate)
            if candidate["full_llm_visual_tokens"] > BUDGET:
                found += 1
                if found >= per_source:
                    break
        if found < per_source:
            raise OraclePilotError(
                f"source_dataset {source!r} exhausted after {found} eligible "
                f"candidates (< {per_source}); fail closed"
            )
        print(
            f"SELECT source={source!r} eligible={found} measured={len(measured)}",
            flush=True,
        )
    return measured


# ---------------------------------------------------------------------------
# Config / report assembly
# ---------------------------------------------------------------------------
def _pin(path: Path, root: Path = ROOT) -> dict[str, Any]:
    if not path.is_file():
        raise SelectionError(f"required pin target is missing: {path} (fail closed)")
    try:
        relative = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        relative = str(path)
    return {"path": relative, "sha256": file_sha256(path)}


def load_exclusions(root: Path = ROOT) -> dict[str, Any]:
    screening = {
        str(json.loads(line)["sample_id"])
        for line in SCREENING_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    counterfactual = {
        str(json.loads(line)["sample_id"])
        for line in COUNTERFACTUAL_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    manual = {"1"}
    exclusions = manual | screening | counterfactual
    return {
        "sample_id_1": sorted(manual),
        "screening100": sorted(screening),
        "counterfactual100": sorted(counterfactual),
        "all": sorted(exclusions),
        "count": len(exclusions),
    }


def build_config(
    *,
    selected_records: Sequence[Mapping[str, Any]],
    selection_metadata: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "design": DESIGN,
        "scope": (
            "Label-aware oracle upper-bound pilot for the B1 answer-margin "
            "saliency; CPU-only pre-registration, GPU not executed."
        ),
        "status": STATUS_PREREGISTERED_NOT_EXECUTED,
        "date": "2026-10-05",
        "ground_truth_answer_leakage": True,
        "label_aware_oracle": True,
        "deployable_selector": False,
        "is_paper_accuracy_claim": False,
        "is_selector_accuracy_evidence": False,
        "gpu_executed": False,
        "model_was_run": False,
        "training": False,
        "optimizer_constructed": False,
        "pilot_sample_count": PILOT_SAMPLE_COUNT,
        "arm": ARM_ORACLE,
        "selection": {
            "criteria": list(SELECTION_CRITERIA),
            "source_datasets": list(SOURCE_DATASETS),
            "per_source": PILOT_PER_SOURCE,
            "stable_rank": "sha256(str(sample_id)) ascending, tie-break sample_id",
            "exclusions": dict(exclusions),
            "reads_correctness": False,
            "reads_dense_or_l2_outputs": False,
            **{k: v for k, v in selection_metadata.items() if k != "exclusion_count"},
        },
        "manifest": {
            "path": str(OUTPUT_MANIFEST.relative_to(ROOT)),
            "sha256": manifest_sha256,
            "count": len(selected_records),
            "source_counts": {
                source: sum(
                    1 for row in selected_records if row["source_dataset"] == source
                )
                for source in SOURCE_DATASETS
            },
            "sample_ids": [str(row["sample_id"]) for row in selected_records],
        },
        "runtime_contract": {
            "model_path": "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B",
            "model_label": "RoboBrain2.5-4B-BF16",
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "device": "cuda:0",
            "decoding": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "prune_after_layer": PRUNE_AFTER_LAYER,
            "cutpoint": CUTPOINT,
            "budget": BUDGET,
            "region_grid": REGION_GRID_FIXED,
            "local_candidate_cap": LOCAL_CANDIDATE_CAP,
            "expected_output_group_count": EXPECTED_OUTPUT,
            "expected_real_group_count": EXPECTED_REAL,
            "expected_context_group_count": EXPECTED_CONTEXT,
            "saliency_definition": SALIENCY_DEFINITION,
            "score_injection": (
                "existing integer_scorer hook of "
                "forward_hardware_regular_b1_vision -> combined_scores of "
                "select_hardware_regular_groups (layout unchanged)"
            ),
            "model_parameters_frozen": True,
            "parameter_grads_must_all_be_none": True,
        },
        "inputs_sha256": {
            "full_manifest": _pin(FULL_MANIFEST),
            "screening100_manifest": _pin(SCREENING_MANIFEST),
            "counterfactual100_manifest": _pin(COUNTERFACTUAL_MANIFEST),
            "selection_script": _pin(SELECTION_SCRIPT),
            "shared_helpers": _pin(SHARED_HELPERS),
            "gpu_runner": _pin(GPU_RUNNER),
            "summarizer": _pin(SUMMARIZER),
            "selection_test": _pin(SELECTION_TEST),
            "runner_test": _pin(RUNNER_TEST),
            "summarizer_test": _pin(SUMMARIZER_TEST),
        },
        "summarizer_inputs": {
            "dense_full2638_jsonl": _pin(DENSE_FULL_JSONL),
            "control_corrected_l2_integer_jsonl": _pin(CONTROL_JSONL),
            "oracle_output_jsonl": {
                "path": str(ORACLE_OUTPUT.relative_to(ROOT)),
                "sha256": None,
                "pending_gpu": True,
            },
        },
        "output_paths": {"oracle": str(ORACLE_OUTPUT.relative_to(ROOT))},
        "pilot_gate": {
            "oracle_minus_control_net_min": GATE_ORACLE_MINUS_CONTROL_NET_MIN,
            "w_to_r_min": GATE_W_TO_R_MIN,
            "r_to_w_max": GATE_R_TO_W_MAX,
            "oracle_parse_min": GATE_ORACLE_PARSE_MIN,
            "oracle_parse_denominator": PILOT_SAMPLE_COUNT,
            "exact_k96_layout_all_required": True,
            "on_pass": "GO",
            "on_fail": "STOP",
            "uses_confidence_interval": False,
        },
        "required_statements": {
            "ground_truth_leakage": (
                "oracle 使用官方正确答案分叉 token 计算 saliency，属于 "
                "ground-truth answer leakage，只能作为可证伪上界。"
            ),
            "not_deployable": "该 oracle 不可部署；它不是 selector，不能上板、不能用于推理。",
            "not_paper_accuracy": (
                "24 样本只是泄漏 oracle 筛选，不是论文精度，不得使用 CI 或"
                "『精度提升已确认』措辞。"
            ),
            "gpu_not_executed": "本轮未运行 GPU、未加载模型权重、未训练。",
        },
        "forbidden_claims": [
            "deployable selector",
            "paper accuracy",
            "CV-Bench accuracy improvement confirmed",
            "FPGA speedup or latency",
            "using dense/L2 correctness to select samples",
        ],
    }


def build_prereg_report(
    config: Mapping[str, Any],
    selection_metadata: Mapping[str, Any],
    exclusions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "date": "2026-10-05",
        "design": DESIGN,
        "status": STATUS_PREREGISTERED_NOT_EXECUTED,
        "scope": config["scope"],
        "label_aware_oracle": True,
        "ground_truth_answer_leakage": True,
        "deployable_selector": False,
        "is_paper_accuracy_claim": False,
        "gpu_executed": False,
        "model_was_run": False,
        "gpu_runs": 0,
        "training": False,
        "pilot_sample_count": PILOT_SAMPLE_COUNT,
        "per_source": PILOT_PER_SOURCE,
        "source_datasets": list(SOURCE_DATASETS),
        "selection": dict(selection_metadata),
        "exclusions": dict(exclusions),
        "manifest": dict(config["manifest"]),
        "runtime_contract": dict(config["runtime_contract"]),
        "pilot_gate": dict(config["pilot_gate"]),
        "inputs_sha256": dict(config["inputs_sha256"]),
        "summarizer_inputs": dict(config["summarizer_inputs"]),
        "label_aware_oracle_note": LABEL_AWARE_ORACLE,
        "required_statements": dict(config["required_statements"]),
        "forbidden_claims": list(config["forbidden_claims"]),
        "next_command": (
            "python evaluation/run_b1_answer_margin_oracle_pilot.py "
            "--model /home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B "
            "--model-label RoboBrain2.5-4B-BF16 "
            f"--manifest {OUTPUT_MANIFEST.relative_to(ROOT)} "
            f"--config {OUTPUT_CONFIG.relative_to(ROOT)} "
            f"--output {ORACLE_OUTPUT.relative_to(ROOT)}"
        ),
    }


def render_prereg_markdown(report: Mapping[str, Any]) -> str:
    manifest = report["manifest"]
    gate = report["pilot_gate"]
    lines = [
        "# B1 answer-margin oracle K96 pilot 预注册（未执行 GPU）",
        "",
        f"- 日期：{report['date']}",
        f"- 状态：**{report['status']}**",
        f"- 设计：`{report['design']}`",
        f"- 范围：{report['scope']}",
        "",
        "## 1. 结论口径（必须保留）",
        "",
        "- **ground-truth answer leakage = true**：oracle 用官方正确答案分叉 token 计算",
        "  B1 `|grad×activation|` group saliency，只是可证伪的**上界**。",
        "- **不可部署**：这不是 selector，不能上板、不能用于真实推理。",
        "- **非论文精度**：24 样本只是泄漏 oracle 筛选；本轮 **未运行 GPU**、未加载模型权重、",
        "  未训练、未生成任何 oracle 输出，因此不得使用 CI 或『精度提升已确认』措辞。",
        "- `gpu_executed = false`、`model_was_run = false`、`gpu_runs = 0`。",
        "",
        "## 2. 固定 24 样本",
        "",
        f"- manifest：`{manifest['path']}`（SHA256 `{manifest['sha256']}`，"
        f"{manifest['count']} 条）。",
        "- 三个固定 `source_dataset` 各 8 条："
        + "、".join(f"`{s}`" for s in report["source_datasets"])
        + "。",
        f"- 样本 ID：`{manifest['sample_ids']}`。",
        f"- 各源数量：`{manifest['source_counts']}`。",
        "",
        "## 3. 选样规则（不读正确性）",
        "",
        "- 只用 `datasets/cv_bench_full2638.jsonl` 原始元数据、真实本地 tokenizer 标签分叉",
        "  合同、`full_llm_visual_tokens > 96`，并按 `sha256(sample_id)` 升序取前 8。",
        "- 排除：`sample_id=1`、`datasets/cv_bench_screening100.jsonl` 全部 ID、",
        "  同图反事实 100 manifest 全部 ID。",
        f"- 排除计数：`{report['exclusions']['count']}`。",
        "- **不读取 dense/L2 正确性**；选样函数没有任何 correctness 输入。",
        "",
        "## 4. 固定运行合同",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 模型 | `{report['runtime_contract']['model_path']}`，BF16/SDPA |",
        f"| cutpoint | `{report['runtime_contract']['cutpoint']}`（prune_after_layer="
        f"{report['runtime_contract']['prune_after_layer']}） |",
        f"| K / region_grid / local_cap | {report['runtime_contract']['budget']} / "
        f"{report['runtime_contract']['region_grid']} / "
        f"{report['runtime_contract']['local_candidate_cap']} |",
        f"| 解码 | greedy，max_new_tokens={report['runtime_contract']['max_new_tokens']} |",
        f"| 期望布局 | output={report['runtime_contract']['expected_output_group_count']}，"
        f"real={report['runtime_contract']['expected_real_group_count']}，"
        f"context={report['runtime_contract']['expected_context_group_count']} |",
        "",
        f"- saliency 定义：{report['runtime_contract']['saliency_definition']}",
        f"- score 注入：{report['runtime_contract']['score_injection']}",
        "",
        "## 5. 汇总结论门（仅 24 样本泄漏 oracle 筛选）",
        "",
        f"- oracle − control net ≥ {gate['oracle_minus_control_net_min']}；",
        f"- W→R ≥ {gate['w_to_r_min']}；R→W ≤ {gate['r_to_w_max']}；",
        f"- oracle parse ≥ {gate['oracle_parse_min']}/{gate['oracle_parse_denominator']}；",
        "- 24/24 精确 K96 布局合同通过；",
        f"- 满足 → `{gate['on_pass']}`，否则 `{gate['on_fail']}`；不使用置信区间。",
        "",
        "## 6. 输入与脚本 SHA256 pin",
        "",
    ]
    for name, spec in report["inputs_sha256"].items():
        lines.append(f"- `{name}`: `{spec['path']}` → `{spec['sha256']}`")
    lines += [
        "",
        "## 7. 汇总器输入",
        "",
        f"- dense：`{report['summarizer_inputs']['dense_full2638_jsonl']['path']}` "
        f"→ `{report['summarizer_inputs']['dense_full2638_jsonl']['sha256']}`",
        f"- control：`{report['summarizer_inputs']['control_corrected_l2_integer_jsonl']['path']}` "
        f"→ `{report['summarizer_inputs']['control_corrected_l2_integer_jsonl']['sha256']}`",
        f"- oracle（待 GPU 生成）：`{report['summarizer_inputs']['oracle_output_jsonl']['path']}`",
        "",
        "## 8. 下一条命令（尚未执行）",
        "",
        "```",
        report["next_command"],
        "```",
        "",
        "禁止据此报告声称：" + "、".join(report["forbidden_claims"]) + "。",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, default=FULL_MANIFEST)
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--output-config", type=Path, default=OUTPUT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.is_dir():
        raise SelectionError(
            f"--model must be a local directory (processor only): {args.model}"
        )

    from transformers import AutoProcessor  # noqa: PLC0415

    for path in (
        SELECTION_SCRIPT,
        SHARED_HELPERS,
        GPU_RUNNER,
        SUMMARIZER,
        SELECTION_TEST,
        RUNNER_TEST,
        SUMMARIZER_TEST,
        DENSE_FULL_JSONL,
        CONTROL_JSONL,
    ):
        if not path.is_file():
            raise SelectionError(
                f"required pin target is missing: {path} (fail closed)"
            )

    exclusions = load_exclusions()
    manifest_path: Path = args.full_manifest
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

    measured = build_candidates(
        manifest_rows, exclusions["all"], processor, manifest_path
    )
    selected, selection_metadata = select_from_candidates(
        measured, exclusions=exclusions["all"]
    )

    manifest_records: list[dict[str, Any]] = []
    measured_index = {str(row["sample_id"]): row for row in measured}
    raw_index = {str(row["sample_id"]): row for row in manifest_rows}
    for candidate in selected:
        sample_id = str(candidate["sample_id"])
        record = dict(raw_index[sample_id])
        measured_row = measured_index[sample_id]
        record["oracle_pilot"] = {
            "design": DESIGN,
            "source_group": measured_row["source_group"],
            "selection_rank": int(candidate["selection_rank"]),
            "selection_sha256_rank": stable_sha256_rank(sample_id),
            "full_llm_visual_tokens": int(measured_row["full_llm_visual_tokens"]),
            "image_grid_thw": list(measured_row["image_grid_thw"]),
            "labels": list(measured_row["labels"]),
            "label_token_ids": list(measured_row["label_token_ids"]),
            "common_prefix": list(measured_row["common_prefix"]),
            "common_suffix": list(measured_row["common_suffix"]),
            "branch_token_ids": list(measured_row["branch_token_ids"]),
            "correct_index": int(measured_row["correct_index"]),
            "answer": measured_row["answer"],
            "label_branch_contract_ok": True,
            "label_aware_oracle": True,
            "deployable_selector": False,
        }
        manifest_records.append(record)

    output_manifest = args.output_manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in manifest_records
        ),
        encoding="utf-8",
    )
    manifest_sha256 = file_sha256(output_manifest)

    config = build_config(
        selected_records=manifest_records,
        selection_metadata=selection_metadata,
        exclusions=exclusions,
        manifest_sha256=manifest_sha256,
    )
    output_config = args.output_config
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    config_sha256 = file_sha256(output_config)

    report = build_prereg_report(config, selection_metadata, exclusions)
    report["config"] = {
        "path": str(output_config.relative_to(ROOT)),
        "sha256": config_sha256,
    }
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    report_json = report_dir / REPORT_JSON.name
    report_md = report_dir / REPORT_MD.name
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_md.write_text(render_prereg_markdown(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "status": STATUS_PREREGISTERED_NOT_EXECUTED,
                "manifest": str(output_manifest.relative_to(ROOT)),
                "manifest_sha256": manifest_sha256,
                "config": str(output_config.relative_to(ROOT)),
                "config_sha256": config_sha256,
                "sample_ids": [str(row["sample_id"]) for row in manifest_records],
                "source_counts": config["manifest"]["source_counts"],
                "report_json": str(report_json.relative_to(ROOT)),
                "report_md": str(report_md.relative_to(ROOT)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
