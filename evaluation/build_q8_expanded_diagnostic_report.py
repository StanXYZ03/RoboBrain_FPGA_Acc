#!/usr/bin/env python3
"""Build paired confidence intervals and failure attribution for expanded Q8 runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record


PRIMARY = {
    "pointing_pck_0_1": ("pointing", "pck_0_1", -0.05),
    "pointing_pck_0_2": ("pointing", "pck_0_2", -0.05),
    "grounding_iou_mean": ("grounding", "iou", -0.03),
    "grounding_acc_0_5": ("grounding", "acc_0_5", -0.03),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_bootstrap(differences: list[float], seed: int, replicates: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(differences)
    means = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(replicates)
    ]
    return {
        "n": n,
        "mean_delta": statistics.fmean(differences),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def area_bin(box: list[float]) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1]) / 1_000_000.0
    return "small" if area < 0.05 else "medium" if area < 0.20 else "large"


def position_bin(box: list[float]) -> str:
    x = (box[0] + box[2]) / 2000.0
    y = (box[1] + box[3]) / 2000.0
    return "border" if x < 0.25 or x > 0.75 or y < 0.25 or y > 0.75 else "center"


def target_coverage(mask: dict[str, Any], box: list[float]) -> dict[str, Any]:
    _, patch_h, patch_w = (int(value) for value in mask["image_grid_thw"])
    grid_h, grid_w = patch_h // 2, patch_w // 2
    selected = {int(value) for value in mask["selected_group_indices"]}
    x1, y1, x2, y2 = box
    intersecting = []
    for row in range(grid_h):
        cell_y1, cell_y2 = row * 1000.0 / grid_h, (row + 1) * 1000.0 / grid_h
        for col in range(grid_w):
            cell_x1, cell_x2 = col * 1000.0 / grid_w, (col + 1) * 1000.0 / grid_w
            if min(cell_x2, x2) > max(cell_x1, x1) and min(cell_y2, y2) > max(cell_y1, y1):
                intersecting.append(row * grid_w + col)
    center_col = min(grid_w - 1, int(((x1 + x2) / 2.0) * grid_w / 1000.0))
    center_row = min(grid_h - 1, int(((y1 + y2) / 2.0) * grid_h / 1000.0))
    retained = sum(index in selected for index in intersecting)
    return {
        "target_intersecting_cells": len(intersecting),
        "target_retained_cells": retained,
        "target_coverage": retained / len(intersecting),
        "target_fully_retained": retained == len(intersecting),
        "target_center_cell_retained": center_row * grid_w + center_col in selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--stock-raw", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--compact-mode", default="oracle0875")
    parser.add_argument("--label", default="Q8-B")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    records = load_jsonl(args.raw)
    if args.stock_raw is not None:
        records.extend(
            record for record in load_jsonl(args.stock_raw) if record["mode"] == "stock"
        )
    record_by_key = {(r["sample_id"], r["mode"]): r for r in records}
    masks = (
        {record["sample_id"]: record for record in load_jsonl(args.masks)}
        if args.masks is not None else {}
    )
    expected = {(sample["sample_id"], mode) for sample in samples for mode in ("stock", args.compact_mode)}
    missing = expected - set(record_by_key)
    if missing:
        raise SystemExit(f"incomplete paired run: missing {len(missing)} records")

    pairs = []
    differences: dict[str, list[float]] = defaultdict(list)
    diagnostic_ids: set[str] = set()
    for sample in samples:
        stock_record = record_by_key[(sample["sample_id"], "stock")]
        compact_record = record_by_key[(sample["sample_id"], args.compact_mode)]
        stock = score_record(sample, stock_record)
        compact = score_record(sample, compact_record)
        box = [float(value) for value in sample["ground_truth"]["bbox_1000"]]
        row: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "task": sample["task"],
            "category": sample["category"],
            "image": sample["image"],
            "bbox_area_bin": area_bin(box),
            "bbox_position_bin": position_bin(box),
            "full_visual_tokens": stock_record["full_visual_tokens"],
            "kept_visual_tokens": compact_record["kept_visual_tokens"],
            "stock_output": stock_record["output_text"],
            "compact_output": compact_record["output_text"],
            "output_equal": stock_record["output_text"] == compact_record["output_text"],
            "first_token_divergence": first_divergence(stock_record["token_ids"], compact_record["token_ids"]),
            "stock_parse": stock["parse_success"],
            "compact_parse": compact["parse_success"],
        }
        if masks:
            row.update(target_coverage(masks[sample["sample_id"]], box))
        if sample["task"] == "pointing":
            for key in ("pck_0_1", "pck_0_2"):
                delta = float(compact[key]) - float(stock[key])
                row[f"stock_{key}"] = stock[key]
                row[f"compact_{key}"] = compact[key]
                row[f"delta_{key}"] = delta
                differences[f"pointing_{key}"].append(delta)
            stock_nme = stock["point_nme_bbox_diag"]
            compact_nme = compact["point_nme_bbox_diag"]
            row["stock_error"] = stock_nme
            row["compact_error"] = compact_nme
            row["error_delta"] = (
                float(compact_nme) - float(stock_nme)
                if stock_nme is not None and compact_nme is not None else None
            )
            if any(row[f"delta_{key}"] < 0 for key in ("pck_0_1", "pck_0_2")) or (
                row["error_delta"] is not None and row["error_delta"] > 0.02
            ):
                diagnostic_ids.add(sample["sample_id"])
        else:
            for key, output_key in (("iou", "grounding_iou_mean"), ("acc_0_5", "grounding_acc_0_5")):
                delta = float(compact[key]) - float(stock[key])
                row[f"stock_{key}"] = stock[key]
                row[f"compact_{key}"] = compact[key]
                row[f"delta_{key}"] = delta
                differences[output_key].append(delta)
            row["stock_error"] = stock["center_error_1000"]
            row["compact_error"] = compact["center_error_1000"]
            row["error_delta"] = (
                float(compact["center_error_1000"]) - float(stock["center_error_1000"])
                if stock["center_error_1000"] is not None and compact["center_error_1000"] is not None else None
            )
            if row["delta_acc_0_5"] < 0 or row["delta_iou"] < -0.03:
                diagnostic_ids.add(sample["sample_id"])
        pairs.append(row)

    statistics_by_metric = {}
    for offset, (name, (_, _, margin)) in enumerate(PRIMARY.items()):
        result = paired_bootstrap(differences[name], args.seed + offset, args.bootstrap)
        row_key = {
            "pointing_pck_0_1": "pck_0_1",
            "pointing_pck_0_2": "pck_0_2",
            "grounding_iou_mean": "iou",
            "grounding_acc_0_5": "acc_0_5",
        }[name]
        task = PRIMARY[name][0]
        metric_rows = [row for row in pairs if row["task"] == task]
        result["stock_mean"] = statistics.fmean(float(row[f"stock_{row_key}"]) for row in metric_rows)
        result["compact_mean"] = statistics.fmean(float(row[f"compact_{row_key}"]) for row in metric_rows)
        result["loss_count"] = sum(float(row[f"delta_{row_key}"]) < 0 for row in metric_rows)
        result["gain_count"] = sum(float(row[f"delta_{row_key}"]) > 0 for row in metric_rows)
        result["noninferiority_margin"] = margin
        result["decision"] = (
            "PASS" if result["ci95_low"] >= margin
            else "DEGRADATION" if result["ci95_high"] < margin
            else "INCONCLUSIVE"
        )
        result["effect_vs_zero"] = (
            "DECREASE" if result["ci95_high"] < 0.0
            else "INCREASE" if result["ci95_low"] > 0.0
            else "UNCERTAIN"
        )
        statistics_by_metric[name] = result
    decisions = [item["decision"] for item in statistics_by_metric.values()]
    overall = "PASS" if all(x == "PASS" for x in decisions) else (
        "DEGRADATION" if any(x == "DEGRADATION" for x in decisions) else "INCONCLUSIVE"
    )

    grouped = {}
    for task in ("grounding", "pointing"):
        grouped[task] = {}
        task_rows = [row for row in pairs if row["task"] == task]
        group_fields = ["category", "bbox_area_bin", "bbox_position_bin", "full_visual_tokens"]
        if masks:
            group_fields.append("target_fully_retained")
        for field in group_fields:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in task_rows:
                groups[str(row[field])].append(row)
            grouped[task][field] = {
                name: {
                    "n": len(items),
                    "output_change_rate": statistics.fmean(not item["output_equal"] for item in items),
                    "mean_error_delta_present": statistics.fmean(
                        item["error_delta"] for item in items if item["error_delta"] is not None
                    ) if any(item["error_delta"] is not None for item in items) else None,
                }
                for name, items in sorted(groups.items())
            }

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in pairs for key in row})
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pairs)
    with args.diagnostic_manifest.open("w", encoding="utf-8") as stream:
        for sample in samples:
            if sample["sample_id"] in diagnostic_ids:
                stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    coverage_summary = None
    if masks:
        coverage_summary = {
            "target_centers_retained": sum(row["target_center_cell_retained"] for row in pairs),
            "target_centers_total": len(pairs),
            "target_intersecting_cells_retained": sum(row["target_retained_cells"] for row in pairs),
            "target_intersecting_cells_total": sum(row["target_intersecting_cells"] for row in pairs),
            "samples_with_full_target_coverage": sum(row["target_fully_retained"] for row in pairs),
            "samples_total": len(pairs),
        }
        coverage_summary["target_intersecting_cell_retention_rate"] = (
            coverage_summary["target_intersecting_cells_retained"]
            / coverage_summary["target_intersecting_cells_total"]
        )

    payload = {
        "format": "q8-expanded-paired-diagnostic-report-v1",
        "status": overall,
        "expanded_diagnostic_not_blind_validation": True,
        "oracle_uses_ground_truth_bbox": True,
        "manifest_sha256": sha256(args.manifest),
        "raw_sha256": sha256(args.raw),
        "stock_raw_sha256": sha256(args.stock_raw) if args.stock_raw is not None else None,
        "masks_sha256": sha256(args.masks) if args.masks is not None else None,
        "paired_records": len(pairs),
        "unique_images": len({sample["image"] for sample in samples}),
        "primary_metrics": statistics_by_metric,
        "output_change_rate": statistics.fmean(not row["output_equal"] for row in pairs),
        "diagnostic_subset_records": len(diagnostic_ids),
        "grouped_attribution_by_task": grouped,
        "target_coverage_audit": coverage_summary,
        "limitations": [
            "This 96-image set was revealed in prior L2/residual work and is not a new blind test.",
            "The geometric oracle reads ground-truth boxes and is not deployable.",
            "Association by category, area, position, or grid does not establish causality.",
            "Target coverage association cannot by itself distinguish semantic context loss from implementation effects.",
        ],
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {args.label} 87.5%扩大配对诊断",
        "",
        f"状态：{overall}。这是96图、192任务的扩大诊断，不是新盲测；Oracle读取真实框，不能部署。",
        "",
        "## 配对统计",
        "",
        "| 指标 | n | 完整 | 裁剪 | 裁剪-完整 | 95% Bootstrap区间 | 相对零变化 | 退化/改善样本 | 非劣界限 | 非劣判定 |",
        "|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for name, item in statistics_by_metric.items():
        lines.append(
            f"| {name} | {item['n']} | {item['stock_mean']:.4f} | {item['compact_mean']:.4f} | {item['mean_delta']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
            f"{item['effect_vs_zero']} | {item['loss_count']}/{item['gain_count']} | "
            f"{item['noninferiority_margin']:+.3f} | {item['decision']} |"
        )
    lines += [
        "",
        f"- 输出文本变化率：{payload['output_change_rate']:.3f}。",
        f"- 自动归因子集：{len(diagnostic_ids)}条任务记录。",
        "- " + (
            "Q8-A已在同一完整集运行；应结合阶段归因报告判断早期视觉裁剪与Token集合影响。"
            if args.label == "Q8-B" else
            "Q8-A完整集结果应与同一数据上的Q8-B直接配对比较；不能只根据单一路径归因。"
        ),
    ]
    if coverage_summary is not None:
        lines += [
            "",
            "## 目标区域覆盖审计",
            "",
            f"- 目标中心所在网格保留：{coverage_summary['target_centers_retained']}/{coverage_summary['target_centers_total']}。",
            f"- 与目标框相交网格保留：{coverage_summary['target_intersecting_cells_retained']}/{coverage_summary['target_intersecting_cells_total']}（{coverage_summary['target_intersecting_cell_retention_rate']:.3%}）。",
            f"- 目标相交网格全部保留的任务：{coverage_summary['samples_with_full_target_coverage']}/{coverage_summary['samples_total']}。",
            "- 因此若退化仍出现在目标完整覆盖样本中，不能解释为目标区域被直接删除；应继续检查上下文删除、裁剪切入点及量化交互。",
        ]
    lines += [
        "",
        "分组结果和逐样本输出保存在同名JSON/CSV中。类别、目标大小、位置和网格只用于发现关联，不能直接作为因果结论。",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "pairs": len(pairs), "diagnostic_subset": len(diagnostic_ids)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
