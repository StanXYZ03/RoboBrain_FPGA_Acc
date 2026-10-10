#!/usr/bin/env python3
"""Attribute matched-budget L2 failures using target coverage and strata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_q8_expanded_diagnostic_report import target_coverage
from evaluation.metrics import load_jsonl, score_record


def first_strategy(path: Path, strategy: str) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        if str(record.get("strategy")) != strategy:
            continue
        sample_id = str(record["sample_id"])
        if sample_id not in selected or int(record.get("repeat", 0)) < int(
            selected[sample_id].get("repeat", 0)
        ):
            selected[sample_id] = record
    return selected


def unique(path: Path) -> dict[str, dict[str, Any]]:
    records = load_jsonl(path)
    output = {str(record["sample_id"]): record for record in records}
    if len(output) != len(records):
        raise ValueError(f"duplicate sample IDs in {path}")
    return output


def primary_score(sample: dict[str, Any], prediction: dict[str, Any]) -> float:
    scored = score_record(sample, prediction)
    return float(scored["iou"] if sample["task"] == "grounding" else scored["pck_0_1"])


def retention_bin(value: float) -> str:
    if value < 0.20:
        return "[0.00,0.20)"
    if value < 0.35:
        return "[0.20,0.35)"
    if value < 0.50:
        return "[0.35,0.50)"
    return "[0.50,1.00]"


def coverage_bin(value: float) -> str:
    if value == 0.0:
        return "0"
    if value < 0.5:
        return "(0,0.5)"
    if value < 1.0:
        return "[0.5,1)"
    return "1"


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "retention_mean": statistics.fmean(float(row["retention"]) for row in rows),
        "target_coverage_mean": statistics.fmean(
            float(row["target_coverage"]) for row in rows
        ),
        "target_center_retention_rate": statistics.fmean(
            bool(row["target_center_cell_retained"]) for row in rows
        ),
        "dense_primary_mean": statistics.fmean(
            float(row["dense_primary"]) for row in rows
        ),
        "roi_primary_mean": statistics.fmean(float(row["roi_primary"]) for row in rows),
        "l2_primary_mean": statistics.fmean(float(row["l2_primary"]) for row in rows),
        "l2_minus_dense_mean": statistics.fmean(
            float(row["l2_minus_dense"]) for row in rows
        ),
        "l2_minus_roi_mean": statistics.fmean(
            float(row["l2_minus_roi"]) for row in rows
        ),
    }


def grouped_summary(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for field in fields:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        output[field] = {key: summarize(values) for key, values in sorted(groups.items())}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--matched-l2", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = {str(row["sample_id"]): row for row in load_jsonl(args.manifest)}
    dense = first_strategy(args.native_results, "dense")
    roi = first_strategy(args.native_results, "roi_only")
    l2 = unique(args.matched_l2)
    for label, records in (("dense", dense), ("roi", roi), ("l2", l2)):
        if set(records) != set(samples):
            raise ValueError(f"{label} sample IDs do not match the manifest")

    rows: list[dict[str, Any]] = []
    selection_by_image: dict[int, set[str]] = defaultdict(set)
    for sample_id, sample in samples.items():
        l2_record = l2[sample_id]
        coverage = target_coverage(l2_record, sample["ground_truth"]["bbox_1000"])
        dense_primary = primary_score(sample, dense[sample_id])
        roi_primary = primary_score(sample, roi[sample_id])
        l2_primary = primary_score(sample, l2_record)
        retention = int(l2_record["llm_visual_tokens"]) / int(
            l2_record["baseline_llm_visual_tokens"]
        )
        row = {
            "sample_id": sample_id,
            "coco_image_id": int(sample["coco_image_id"]),
            "task": str(sample["task"]),
            "category": str(sample["category"]),
            "retention": retention,
            "retention_bin": retention_bin(retention),
            "target_area_fraction": float(sample["stratification"]["target_area_fraction"]),
            "target_area_bin": str(sample["stratification"]["target_area_bin"]),
            "edge_bin": str(sample["stratification"]["edge_bin"]),
            "overlap_bin": str(sample["stratification"]["overlap_bin"]),
            "scene_complexity_bin": str(
                sample["stratification"]["scene_complexity_bin"]
            ),
            **coverage,
            "coverage_bin": coverage_bin(float(coverage["target_coverage"])),
            "dense_primary": dense_primary,
            "roi_primary": roi_primary,
            "l2_primary": l2_primary,
            "roi_minus_dense": roi_primary - dense_primary,
            "l2_minus_dense": l2_primary - dense_primary,
            "l2_minus_roi": l2_primary - roi_primary,
        }
        rows.append(row)
        selection_by_image[int(sample["coco_image_id"])].add(
            str(l2_record["selected_group_indices_sha256"])
        )

    task_selection_mismatches = [
        image_id for image_id, hashes in selection_by_image.items() if len(hashes) != 1
    ]
    fields = (
        "retention_bin",
        "coverage_bin",
        "target_center_cell_retained",
        "target_area_bin",
        "edge_bin",
        "overlap_bin",
        "scene_complexity_bin",
    )
    by_task: dict[str, Any] = {}
    for task in ("grounding", "pointing"):
        task_rows = [row for row in rows if row["task"] == task]
        by_task[task] = {
            "overall": summarize(task_rows),
            "groups": grouped_summary(task_rows, fields),
            "correlation": {
                "retention_vs_l2_minus_dense": pearson(
                    [float(row["retention"]) for row in task_rows],
                    [float(row["l2_minus_dense"]) for row in task_rows],
                ),
                "target_coverage_vs_l2_minus_dense": pearson(
                    [float(row["target_coverage"]) for row in task_rows],
                    [float(row["l2_minus_dense"]) for row in task_rows],
                ),
            },
        }

    target_cells_total = sum(int(row["target_intersecting_cells"]) for row in rows)
    target_cells_kept = sum(int(row["target_retained_cells"]) for row in rows)
    report = {
        "status": "ROI_L2_MATCHED_FAILURE_ATTRIBUTION_COMPLETE",
        "evidence_level": "post-hoc descriptive attribution on revealed development data",
        "records": len(rows),
        "unique_images": len(selection_by_image),
        "same_l2_selection_for_paired_tasks": len(task_selection_mismatches) == 0,
        "task_selection_mismatch_image_ids": task_selection_mismatches,
        "target_coverage_audit": {
            "centers_retained": sum(bool(row["target_center_cell_retained"]) for row in rows),
            "samples": len(rows),
            "all_target_cells_retained": sum(bool(row["target_fully_retained"]) for row in rows),
            "target_cells_retained": target_cells_kept,
            "target_cells_total": target_cells_total,
            "target_cell_retention_rate": target_cells_kept / target_cells_total,
        },
        "by_task": by_task,
        "limitations": [
            "Subgroup and correlation results are post-hoc descriptive evidence, not confirmatory tests.",
            "Target-box cell coverage does not measure semantic context, object parts, or feature distortion.",
            "Association cannot prove that omitted target cells caused the output error."
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit = report["target_coverage_audit"]
    lines = [
        "# ROI/L2同预算失效归因",
        "",
        "状态：`ROI_L2_MATCHED_FAILURE_ATTRIBUTION_COMPLETE`。这是已揭示开发集上的事后描述分析，不是新假设检验。",
        "",
        "## 目标覆盖审计",
        "",
        f"L2保留目标中心网格：`{audit['centers_retained']}/{audit['samples']}`；完整保留全部目标相交网格：`{audit['all_target_cells_retained']}/{audit['samples']}`；目标相交网格总体保留率：`{audit['target_cell_retention_rate']:.3f}`。",
        f"同图Grounding/Pointing的L2选择集合一致：`{report['same_l2_selection_for_paired_tasks']}`。",
        "",
        "## 关键分组",
        "",
        "`Delta`为L2主指标减Dense主指标；分组只用于定位机制，不作显著性结论。",
        "",
        "| 任务 | 分组 | n | 保留率 | 目标覆盖 | L2-Dense Delta | L2-ROI Delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in ("grounding", "pointing"):
        for field in ("retention_bin", "target_center_cell_retained", "coverage_bin"):
            for value, item in by_task[task]["groups"][field].items():
                lines.append(
                    f"| {task} | {field}={value} | {item['n']} | "
                    f"{item['retention_mean']:.3f} | {item['target_coverage_mean']:.3f} | "
                    f"{item['l2_minus_dense_mean']:+.4f} | {item['l2_minus_roi_mean']:+.4f} |"
                )
    lines.extend([
        "",
        "## 相关性",
        "",
        "| 任务 | 保留率与L2-Dense | 目标覆盖与L2-Dense |",
        "|---|---:|---:|",
    ])
    for task in ("grounding", "pointing"):
        correlation = by_task[task]["correlation"]
        lines.append(
            f"| {task} | {correlation['retention_vs_l2_minus_dense']:.4f} | "
            f"{correlation['target_coverage_vs_l2_minus_dense']:.4f} |"
        )
    lines.extend([
        "",
        "## 解释边界",
        "",
        "如果保留目标中心或提高目标网格覆盖只能部分缩小损失，则失败不能简化为单个中心Token漏选；还可能涉及部件细节、连续邻域、全局参照和裁剪后重采样差异。当前分析只能形成下一动作空间的设计依据，不能证明因果。",
    ])
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "coverage": audit}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
