#!/usr/bin/env python3
"""Freeze the 24-sample challenge/control manifest for the B1 oracle pilot24 v2.

CPU-only.  It reads the frozen dense and corrected-L2 outputs on purpose
(**outcome/label leakage**) to split every cell into two challenge and two
regression-control samples, and it loads only the **local processor/tokenizer**
(``local_files_only=True``, no model weights, no CUDA, no network) to obtain the
authoritative processor grid and the real-tokenizer label-branch contract.

Selection rule (pre-registered, deterministic)
---------------------------------------------
For each of exactly six fixed ``source x task`` cells:

1. start from the frozen ``datasets/cv_bench_full2638.jsonl`` metadata;
2. exclude ``sample_id == "1"``, every id in ``datasets/cv_bench_screening100.jsonl``,
   every id in the same-image counterfactual 100 manifest and every id in the
   existing random v1 pilot24 manifest;
3. keep only samples whose dense BF16 output is officially correct;
4. require the real-tokenizer label-branch contract and the processor-derived
   ``full_llm_visual_tokens > 96``;
5. split by corrected integer-L2 B1/K96 correctness into ``challenge`` (wrong)
   and ``control`` (correct), then take the first two by
   ``sha256(str(sample_id))`` then sample id in each stratum;
6. across the eight Omni3D picks require ``Omni3D_Hypersim`` / ``Omni3D_SUNRGBD`` /
   ``Omni3D_nuScenes`` each to contribute at least two samples.  If the naive
   first-two rule does not satisfy this, run the deterministic constrained
   selection that minimises the ordered SHA-rank tuple; fail closed if
   impossible.

This manifest is an outcome-enriched diagnostic.  It is **non-deployable,
non-representative and forbidden for official absolute accuracy claims**.
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

from evaluation.b1_answer_margin_oracle_challenge_control import (  # noqa: E402
    ARM_ORACLE,
    BUDGET,
    CELLS,
    CELL_BY_KEY,
    CHALLENGE,
    CONTROL,
    DESIGN,
    EXPECTED_CONTEXT,
    EXPECTED_OUTPUT,
    EXPECTED_REAL,
    GATE_CHALLENGE_REPAIR_MIN,
    GATE_CONTROL_REGRESSION_MAX,
    GATE_ORACLE_PARSE_MIN,
    LABEL_AWARE_ORACLE,
    LOCAL_CANDIDATE_CAP,
    MAX_NEW_TOKENS,
    OMNI3D_MIN_PER_SUBDATASET,
    OMNI3D_SOURCE,
    OMNI3D_SUBDATASETS,
    PER_CELL,
    PER_STRATUM,
    PILOT_SAMPLE_COUNT,
    PRUNE_AFTER_LAYER,
    REGION_GRID_FIXED,
    SALIENCY_DEFINITION,
    SELECTION_CRITERIA,
    STATUS_PREREGISTERED_NOT_EXECUTED,
    STRATA,
    OracleChallengeControlError,
    challenge_control_label,
    constrained_omni3d_selection,
    file_sha256,
    load_exclusions,
    omni3d_coverage,
    omni3d_coverage_ok,
    rank_key,
)
from evaluation.b1_answer_margin_oracle import (  # noqa: E402
    CUTPOINT,
    stable_sha256_rank,
)
from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice  # noqa: E402
from evaluation.hardware_regular_token_selector import MERGE_UNIT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

STEM = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24"

FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
SCREENING_MANIFEST = ROOT / "datasets" / "cv_bench_screening100.jsonl"
COUNTERFACTUAL_MANIFEST = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl"
)
RANDOM_V1_MANIFEST = (
    ROOT / "datasets" / "2026-10-05-b1-answer-margin-oracle-pilot24.jsonl"
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
OUTPUT_MANIFEST = ROOT / "datasets" / f"{STEM}.jsonl"
OUTPUT_CONFIG = ROOT / "configs" / f"{STEM}.preregistered.json"
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_STEM = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot-prereg"
REPORT_JSON = REPORT_DIR / f"{REPORT_STEM}.json"
REPORT_MD = REPORT_DIR / f"{REPORT_STEM}.md"
ORACLE_OUTPUT = (
    ROOT
    / "artifacts"
    / "token_selector"
    / STEM
    / "oracle_answer_margin_b1_saliency_k96.jsonl"
)

SELECTION_SCRIPT = (
    ROOT / "evaluation" / "select_b1_answer_margin_oracle_challenge_control_pilot24.py"
)
SHARED_HELPERS = ROOT / "evaluation" / "b1_answer_margin_oracle.py"
CHALLENGE_CONTROL_HELPERS = (
    ROOT / "evaluation" / "b1_answer_margin_oracle_challenge_control.py"
)
GPU_RUNNER = ROOT / "evaluation" / "run_b1_answer_margin_oracle_pilot.py"
SUMMARIZER = (
    ROOT / "evaluation" / "summarize_b1_answer_margin_oracle_challenge_control_pilot.py"
)
TESTS = (
    ROOT
    / "evaluation"
    / "tests"
    / "test_b1_answer_margin_oracle_challenge_control.py",
    ROOT
    / "evaluation"
    / "tests"
    / "test_select_b1_answer_margin_oracle_challenge_control_pilot24.py",
    ROOT
    / "evaluation"
    / "tests"
    / "test_run_b1_answer_margin_oracle_challenge_control_pilot.py",
    ROOT
    / "evaluation"
    / "tests"
    / "test_summarize_b1_answer_margin_oracle_challenge_control_pilot.py",
)


class SelectionError(RuntimeError):
    """Raised when the selection or a pin invariant is violated."""


# ---------------------------------------------------------------------------
# Pure, CPU-testable selection helpers
# ---------------------------------------------------------------------------
def official_correct(
    manifest_row: Mapping[str, Any], output_row: Mapping[str, Any]
) -> tuple[bool, str | None]:
    """Official parse + ground-truth comparison for one arm."""

    choices = [str(choice) for choice in manifest_row.get("choices", [])]
    truth = answer_letter(str(manifest_row["answer"]))
    parsed = parse_choice(str(output_row.get("output_text", "")), choices)
    return bool(parsed is not None and parsed == truth), parsed


def select_pilot_cells(
    pools: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, Any]]]],
) -> tuple[dict[tuple[str, str], dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Select two challenge and two control samples per cell (pure)."""

    selected: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for cell in CELLS:
        entry = pools.get(cell.key, {})
        selected[cell.key] = {}
        for stratum in STRATA:
            pool = sorted(
                (dict(candidate) for candidate in entry.get(stratum, [])),
                key=lambda candidate: rank_key(candidate["sample_id"]),
            )
            if len(pool) < PER_STRATUM:
                raise OracleChallengeControlError(
                    f"cell {cell.label()} stratum {stratum!r} has only "
                    f"{len(pool)} eligible candidates (< {PER_STRATUM}); fail closed"
                )
            selected[cell.key][stratum] = pool[:PER_STRATUM]

    omni_cells = [cell for cell in CELLS if cell.source == OMNI3D_SOURCE]
    naive = [
        candidate
        for cell in omni_cells
        for stratum in STRATA
        for candidate in selected[cell.key][stratum]
    ]
    if omni3d_coverage_ok(naive):
        omni_metadata: dict[str, Any] = {
            "strategy": "naive_first_two",
            "coverage": omni3d_coverage(naive),
            "min_per_subdataset": OMNI3D_MIN_PER_SUBDATASET,
        }
    else:
        buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for cell in omni_cells:
            for stratum in STRATA:
                buckets[(cell.task, stratum)] = [
                    dict(candidate)
                    for candidate in pools.get(cell.key, {}).get(stratum, [])
                ]
        chosen, omni_metadata = constrained_omni3d_selection(buckets)
        omni_metadata = dict(omni_metadata)
        omni_metadata["strategy"] = "constrained_lexicographic_rank_tuple"
        for cell in omni_cells:
            for stratum in STRATA:
                selected[cell.key][stratum] = [
                    dict(candidate) for candidate in chosen[(cell.task, stratum)]
                ]

    for cell in CELLS:
        if len(selected[cell.key][CHALLENGE]) != PER_STRATUM or len(
            selected[cell.key][CONTROL]
        ) != PER_STRATUM:
            raise OracleChallengeControlError(
                f"cell {cell.label()} does not hold {PER_STRATUM}+{PER_STRATUM} samples"
            )
    return selected, omni_metadata


# ---------------------------------------------------------------------------
# Processor-backed measurement
# ---------------------------------------------------------------------------
def measure_candidate(
    processor: Any,
    sample: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    """Measure the processor grid G and the real-tokenizer label branch."""

    from PIL import Image  # noqa: PLC0415

    from evaluation.run_b1_answer_margin_gpu_smoke import (  # noqa: PLC0415
        AnswerMarginSmokeError,
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
    source = str(sample["source"])
    task = str(sample["task"])
    if (source, task) not in CELL_BY_KEY:
        raise SelectionError(f"sample {sample_id!r} is outside the six fixed cells")
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
    if full_patch_tokens % MERGE_UNIT:
        raise SelectionError(f"sample {sample_id!r} grid is not merger-aligned")
    full_groups = full_patch_tokens // MERGE_UNIT

    label_token_ids = tokenize_labels(processor.tokenizer, labels)
    try:
        branches = parse_label_branches(label_token_ids)
    except AnswerMarginSmokeError:
        return {
            "sample_id": sample_id,
            "source": source,
            "task": task,
            "source_group": source,
            "source_dataset": str(sample.get("source_dataset")),
            "full_llm_visual_tokens": int(full_groups),
            "image_grid_thw": grid,
            "label_branch_ok": False,
        }
    return {
        "sample_id": sample_id,
        "source": source,
        "task": task,
        "source_group": source,
        "source_dataset": str(sample.get("source_dataset")),
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


def build_candidate_pools(
    manifest_rows: Sequence[Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    control_index: Mapping[str, Mapping[str, Any]],
    exclusions: Sequence[str] | set[str],
    processor: Any,
    manifest_path: Path,
) -> tuple[dict[tuple[str, str], dict[str, list[dict[str, Any]]]], dict[str, int]]:
    """Measure every dense-correct candidate in the six cells (label leakage)."""

    excluded = {str(value) for value in exclusions}
    pools: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {
        cell.key: {CHALLENGE: [], CONTROL: []} for cell in CELLS
    }
    stats = {
        "dense_correct_measured": 0,
        "label_branch_rejected": 0,
        "g_le_96_rejected": 0,
    }
    for row in manifest_rows:
        cell_key = (str(row.get("source")), str(row.get("task")))
        if cell_key not in CELL_BY_KEY:
            continue
        sample_id = str(row["sample_id"])
        if sample_id in excluded:
            continue
        if sample_id not in dense_index or sample_id not in control_index:
            raise SelectionError(
                f"sample {sample_id!r} is missing from the frozen dense/control outputs"
            )
        dense_correct, _ = official_correct(row, dense_index[sample_id])
        if not dense_correct:
            continue
        control_correct, _ = official_correct(row, control_index[sample_id])
        candidate = measure_candidate(processor, row, manifest_path)
        stats["dense_correct_measured"] += 1
        if not candidate["label_branch_ok"]:
            stats["label_branch_rejected"] += 1
            continue
        if int(candidate["full_llm_visual_tokens"]) <= BUDGET:
            stats["g_le_96_rejected"] += 1
            continue
        candidate["dense_correct"] = True
        candidate["control_correct"] = bool(control_correct)
        stratum = challenge_control_label(True, control_correct)
        if stratum is None:
            raise SelectionError("internal error: no stratum for a dense-correct sample")
        candidate["stratum"] = stratum
        candidate["cell"] = f"{cell_key[0]} x {cell_key[1]}"
        candidate["selection_sha256_rank"] = stable_sha256_rank(sample_id)
        pools[cell_key][stratum].append(candidate)
    return pools, stats


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


def build_config(
    *,
    selected_records: Sequence[Mapping[str, Any]],
    selection_metadata: Mapping[str, Any],
    exclusions: Mapping[str, Any],
    manifest_sha256: str,
    manifest_pin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "design": DESIGN,
        "scope": (
            "Outcome-enriched challenge/control diagnostic for the B1 answer-margin "
            "oracle; CPU-only pre-registration, GPU not executed."
        ),
        "status": STATUS_PREREGISTERED_NOT_EXECUTED,
        "date": "2026-10-05",
        "outcome_enriched_diagnostic": True,
        "outcome_label_leakage": True,
        "non_deployable": True,
        "non_representative": True,
        "forbidden_for_official_absolute_accuracy": True,
        "ground_truth_answer_leakage": True,
        "label_aware_oracle": True,
        "deployable_selector": False,
        "is_paper_accuracy_claim": False,
        "is_selector_accuracy_evidence": False,
        "gpu_executed": False,
        "model_was_run": False,
        "gpu_runs": 0,
        "training": False,
        "optimizer_constructed": False,
        "pilot_sample_count": PILOT_SAMPLE_COUNT,
        "arm": ARM_ORACLE,
        "cells": [
            {
                "cell": cell.label(),
                "source": cell.source,
                "task": cell.task,
                "samples": PER_CELL,
                "challenge": PER_STRATUM,
                "control": PER_STRATUM,
            }
            for cell in CELLS
        ],
        "selection": {
            "criteria": list(SELECTION_CRITERIA),
            "per_cell": PER_CELL,
            "per_stratum": PER_STRATUM,
            "stable_rank": "sha256(str(sample_id)) ascending, tie-break sample_id",
            "dense_reference": "evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl",
            "control_reference": "artifacts/token_selector/2026-10-04-b1-corrected-l2-prompt-ablation-full2638/corrected_l2_integer_k96_full2638.jsonl",
            "exclusions": dict(exclusions),
            "reads_correctness": True,
            "outcome_enriched_diagnostic": True,
            "outcome_label_leakage": True,
            "non_representative": True,
            **{
                key: value
                for key, value in selection_metadata.items()
                if key not in {"exclusion_count"}
            },
        },
        "manifest": {
            "path": str(OUTPUT_MANIFEST.relative_to(ROOT)),
            "sha256": manifest_sha256,
            "count": len(selected_records),
            "cell_counts": {
                cell.label(): sum(
                    1
                    for row in selected_records
                    if row["oracle_pilot"]["cell"] == cell.label()
                )
                for cell in CELLS
            },
            "outcome_counts": {
                stratum: sum(
                    1
                    for row in selected_records
                    if row["oracle_pilot"]["stratum"] == stratum
                )
                for stratum in STRATA
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
            "runner_reused": "evaluation/run_b1_answer_margin_oracle_pilot.py",
            "model_parameters_frozen": True,
            "parameter_grads_must_all_be_none": True,
        },
        "inputs_sha256": {
            "full_manifest": _pin(FULL_MANIFEST),
            "screening100_manifest": _pin(SCREENING_MANIFEST),
            "counterfactual100_manifest": _pin(COUNTERFACTUAL_MANIFEST),
            "random_v1_pilot24_manifest": _pin(RANDOM_V1_MANIFEST),
            "pilot_manifest": (
                dict(manifest_pin)
                if manifest_pin is not None
                else _pin(OUTPUT_MANIFEST)
            ),
            "shared_helpers": _pin(SHARED_HELPERS),
            "challenge_control_helpers": _pin(CHALLENGE_CONTROL_HELPERS),
            "selection_script": _pin(SELECTION_SCRIPT),
            "gpu_runner": _pin(GPU_RUNNER),
            "summarizer": _pin(SUMMARIZER),
            "helpers_test": _pin(TESTS[0]),
            "selection_test": _pin(TESTS[1]),
            "runner_test": _pin(TESTS[2]),
            "summarizer_test": _pin(TESTS[3]),
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
            "challenge_repair_min": GATE_CHALLENGE_REPAIR_MIN,
            "challenge_denominator": PER_CELL * 3,
            "control_regression_max": GATE_CONTROL_REGRESSION_MAX,
            "control_denominator": PER_CELL * 3,
            "oracle_parse_min": GATE_ORACLE_PARSE_MIN,
            "oracle_parse_denominator": PILOT_SAMPLE_COUNT,
            "exact_k96_layout_all_required": True,
            "dense_is_only_eligibility_reference": True,
            "on_pass": "GO",
            "on_fail": "STOP",
            "uses_confidence_interval": False,
        },
        "required_statements": {
            "outcome_label_leakage": (
                "manifest 用 dense/control 正确性分层，属于 outcome/label leakage，"
                "非代表性、不可部署，禁止用于官方绝对精度主张。"
            ),
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
            "official CV-Bench accuracy from 24 enriched samples",
            "CV-Bench accuracy improvement confirmed",
            "FPGA speedup or latency",
            "representative sampling",
            "using the 24 enriched samples for training or threshold tuning",
        ],
    }


def build_prereg_report(
    config: Mapping[str, Any],
    selection_metadata: Mapping[str, Any],
    exclusions: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "date": "2026-10-05",
        "design": DESIGN,
        "status": STATUS_PREREGISTERED_NOT_EXECUTED,
        "scope": config["scope"],
        "outcome_enriched_diagnostic": True,
        "outcome_label_leakage": True,
        "non_deployable": True,
        "non_representative": True,
        "forbidden_for_official_absolute_accuracy": True,
        "ground_truth_answer_leakage": True,
        "label_aware_oracle": True,
        "deployable_selector": False,
        "is_paper_accuracy_claim": False,
        "gpu_executed": False,
        "model_was_run": False,
        "gpu_runs": 0,
        "training": False,
        "pilot_sample_count": PILOT_SAMPLE_COUNT,
        "cells": list(config["cells"]),
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
        "# B1 answer-margin oracle challenge/control pilot24 v2 预注册（未执行 GPU）",
        "",
        f"- 日期：{report['date']}",
        f"- 状态：**{report['status']}**",
        f"- 设计：`{report['design']}`",
        f"- 范围：{report['scope']}",
        "",
        "## 1. 结论口径（必须保留）",
        "",
        "- **outcome/label leakage = true**：样本按 dense/control 正确性分层，是结果富集诊断，",
        "  **非代表性、不可部署**，禁止用于官方绝对精度主张。",
        "- **ground-truth answer leakage = true**：oracle 用官方正确答案分叉 token 计算",
        "  B1 `|grad×activation|` group saliency，只是可证伪上界。",
        "- 24 样本**不是论文精度**；本轮 **未运行 GPU**、未加载模型权重、未训练、",
        "  未生成任何 oracle 输出，因此不得使用 CI 或『精度提升已确认』措辞。",
        "- `gpu_executed = false`、`model_was_run = false`、`gpu_runs = 0`。",
        "",
        "## 2. 固定 24 样本（六单元各 4：2 challenge + 2 control）",
        "",
        f"- manifest：`{manifest['path']}`（SHA256 `{manifest['sha256']}`，"
        f"{manifest['count']} 条）。",
        f"- 样本 ID：`{manifest['sample_ids']}`。",
        f"- 各单元数量：`{manifest['cell_counts']}`。",
        f"- 结果分层数量：`{manifest['outcome_counts']}`。",
        "",
        "## 3. 选样规则（明确使用 dense/control 正确性 = outcome leakage）",
        "",
        "- 只用 `datasets/cv_bench_full2638.jsonl` 原始元数据 + 冻结 dense BF16 与",
        "  corrected integer-L2 B1/K96 正确性 + 真实本地 tokenizer 标签分叉合同 +",
        "  权威 processor `G>96`。",
        "- 每单元 challenge=dense 对且 corrected L2 错；control=dense 对且 corrected L2 对。",
        "- 按 `sha256(str(sample_id))` 升序取每层前 2；Omni3D 8 条要求 Hypersim/SUNRGBD/",
        "  nuScenes 各 ≥2，否则用最小化有序 SHA-rank 元组的确定性约束选样。",
        "- 排除：`sample_id=1`、screening100 全部 ID、同图反事实 100 全部 ID、既有随机 v1",
        f"  pilot24 全部 ID。排除计数：`{report['exclusions']['count']}`。",
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
        f"| runner | `{report['runtime_contract']['runner_reused']}`（复用，未改） |",
        "",
        f"- saliency 定义：{report['runtime_contract']['saliency_definition']}",
        f"- score 注入：{report['runtime_contract']['score_injection']}",
        "",
        "## 5. 汇总 GO/STOP 门（仅此富集诊断）",
        "",
        f"- challenge 修复 ≥ {gate['challenge_repair_min']}/{gate['challenge_denominator']}；",
        f"- control 回归 ≤ {gate['control_regression_max']}/{gate['control_denominator']}；",
        f"- oracle parse = {gate['oracle_parse_min']}/{gate['oracle_parse_denominator']}；",
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
        f"- dense（仅资格参照）：`"
        f"{report['summarizer_inputs']['dense_full2638_jsonl']['path']}` "
        f"→ `{report['summarizer_inputs']['dense_full2638_jsonl']['sha256']}`",
        f"- control：`"
        f"{report['summarizer_inputs']['control_corrected_l2_integer_jsonl']['path']}` "
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.model.is_dir():
        raise SelectionError(
            f"--model must be a local directory (processor only): {args.model}"
        )

    for path in (
        SELECTION_SCRIPT,
        SHARED_HELPERS,
        CHALLENGE_CONTROL_HELPERS,
        GPU_RUNNER,
        SUMMARIZER,
        *TESTS,
        DENSE_FULL_JSONL,
        CONTROL_JSONL,
    ):
        if not path.is_file():
            raise SelectionError(
                f"required pin target is missing: {path} (fail closed)"
            )

    from transformers import AutoProcessor  # noqa: PLC0415

    exclusions = load_exclusions(
        ROOT,
        screening_manifest=SCREENING_MANIFEST,
        counterfactual_manifest=COUNTERFACTUAL_MANIFEST,
        random_v1_manifest=RANDOM_V1_MANIFEST,
    )
    manifest_path: Path = args.full_manifest
    manifest_rows = _load_jsonl(manifest_path)
    dense_rows = _load_jsonl(DENSE_FULL_JSONL)
    control_rows = _load_jsonl(CONTROL_JSONL)
    dense_index = {str(row["sample_id"]): row for row in dense_rows}
    control_index = {str(row["sample_id"]): row for row in control_rows}
    if len(dense_index) != len(dense_rows) or len(control_index) != len(control_rows):
        raise SelectionError("dense or control JSONL contains duplicate sample_id")

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    pools, pool_stats = build_candidate_pools(
        manifest_rows,
        dense_index,
        control_index,
        exclusions["all"],
        processor,
        manifest_path,
    )
    selected, omni_metadata = select_pilot_cells(pools)

    raw_index = {str(row["sample_id"]): row for row in manifest_rows}
    measured_index: dict[str, dict[str, Any]] = {}
    for cell in CELLS:
        for stratum in STRATA:
            for candidate in selected[cell.key][stratum]:
                measured_index[str(candidate["sample_id"])] = candidate

    manifest_records: list[dict[str, Any]] = []
    for cell in CELLS:
        for stratum in STRATA:
            for rank, candidate in enumerate(selected[cell.key][stratum]):
                sample_id = str(candidate["sample_id"])
                record = dict(raw_index[sample_id])
                record["oracle_pilot"] = {
                    "design": DESIGN,
                    "cell": cell.label(),
                    "source_group": cell.source_group,
                    "stratum": stratum,
                    "selection_rank": int(rank),
                    "selection_sha256_rank": stable_sha256_rank(sample_id),
                    "dense_correct": True,
                    "control_correct": bool(candidate["control_correct"]),
                    "outcome_enriched_diagnostic": True,
                    "outcome_label_leakage": True,
                    "non_representative": True,
                    "deployable_selector": False,
                    "full_llm_visual_tokens": int(candidate["full_llm_visual_tokens"]),
                    "image_grid_thw": list(candidate["image_grid_thw"]),
                    "labels": list(candidate["labels"]),
                    "label_token_ids": list(candidate["label_token_ids"]),
                    "common_prefix": list(candidate["common_prefix"]),
                    "common_suffix": list(candidate["common_suffix"]),
                    "branch_token_ids": list(candidate["branch_token_ids"]),
                    "correct_index": int(candidate["correct_index"]),
                    "answer": candidate["answer"],
                    "label_branch_contract_ok": True,
                    "label_aware_oracle": True,
                }
                manifest_records.append(record)

    if len(manifest_records) != PILOT_SAMPLE_COUNT:
        raise SelectionError(
            f"selected {len(manifest_records)} samples != {PILOT_SAMPLE_COUNT}"
        )

    output_manifest = args.output_manifest
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in manifest_records
        ),
        encoding="utf-8",
    )
    manifest_sha256 = file_sha256(output_manifest)

    selection_metadata = {
        "criteria": list(SELECTION_CRITERIA),
        "per_cell": PER_CELL,
        "per_stratum": PER_STRATUM,
        "exclusion_count": exclusions["count"],
        "exclusions": dict(exclusions),
        "omni3d": omni_metadata,
        "pool_stats": pool_stats,
        "reads_correctness": True,
        "reads_dense_or_l2_outputs": True,
        "outcome_enriched_diagnostic": True,
        "outcome_label_leakage": True,
        "non_representative": True,
    }
    config = build_config(
        selected_records=manifest_records,
        selection_metadata={k: v for k, v in selection_metadata.items() if k != "exclusions"},
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
                "cell_counts": config["manifest"]["cell_counts"],
                "outcome_counts": config["manifest"]["outcome_counts"],
                "omni3d": omni_metadata,
                "pool_stats": pool_stats,
                "exclusions_count": exclusions["count"],
                "report_json": str(report_json.relative_to(ROOT)),
                "report_md": str(report_md.relative_to(ROOT)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
