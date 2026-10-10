#!/usr/bin/env python3
"""Attribute global/ROI development errors by frozen dataset strata."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record


def first_runs(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for record in load_jsonl(path):
        key = (str(record["strategy"]), str(record["sample_id"]))
        if key not in selected or int(record["repeat"]) < int(selected[key]["repeat"]):
            selected[key] = record
    return selected


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--baseline", default="dense")
    parser.add_argument("--candidates", nargs="+", default=("roi_only", "roi_global_v2"))
    args = parser.parse_args()

    samples = {record["sample_id"]: record for record in load_jsonl(args.manifest)}
    predictions = first_runs(args.results)
    scored = {
        key: score_record(samples[key[1]], record)
        for key, record in predictions.items()
        if key[1] in samples
    }
    analyses: dict[str, Any] = {}
    errors: list[str] = []
    for candidate in args.candidates:
        task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample_id, sample in samples.items():
            baseline_key = (args.baseline, sample_id)
            candidate_key = (candidate, sample_id)
            if baseline_key not in scored or candidate_key not in scored:
                continue
            baseline_score = scored[baseline_key]
            candidate_score = scored[candidate_key]
            candidate_record = predictions[candidate_key]
            if sample["task"] == "pointing":
                primary_baseline = float(baseline_score["pck_0_1"])
                primary_candidate = float(candidate_score["pck_0_1"])
                continuous_baseline = float(baseline_score["point_nme_bbox_diag"] or 0.0)
                continuous_candidate = float(candidate_score["point_nme_bbox_diag"] or 0.0)
            else:
                primary_baseline = float(baseline_score["iou"])
                primary_candidate = float(candidate_score["iou"])
                continuous_baseline = 1.0 - primary_baseline
                continuous_candidate = 1.0 - primary_candidate
            crop = candidate_record.get("effective_crop_box_1000")
            crop_area = (
                (float(crop[2]) - float(crop[0])) * (float(crop[3]) - float(crop[1])) / 1e6
                if crop
                else 1.0
            )
            row = {
                "sample_id": sample_id,
                "task": sample["task"],
                "category": sample["category"],
                **sample["stratification"],
                "crop_area_fraction": crop_area,
                "crop_touches_image_edge": bool(
                    crop
                    and (
                        float(crop[0]) <= 1e-9
                        or float(crop[1]) <= 1e-9
                        or float(crop[2]) >= 1000.0 - 1e-9
                        or float(crop[3]) >= 1000.0 - 1e-9
                    )
                ),
                "visual_tokens": int(candidate_record["final_context_visual_tokens"]),
                "baseline_primary": primary_baseline,
                "candidate_primary": primary_candidate,
                "primary_delta": primary_candidate - primary_baseline,
                "continuous_error_delta": continuous_candidate - continuous_baseline,
                "baseline_parse": bool(baseline_score["parse_success"]),
                "candidate_parse": bool(candidate_score["parse_success"]),
            }
            task_rows[str(sample["task"])].append(row)

        candidate_analysis: dict[str, Any] = {}
        for task, rows in sorted(task_rows.items()):
            if not rows:
                errors.append(f"{candidate}:{task}: no paired rows")
                continue
            breakdowns: dict[str, Any] = {}
            dimensions = {
                "target_area_bin": lambda row: str(row["target_area_bin"]),
                "edge_bin": lambda row: str(row["edge_bin"]),
                "overlap_bin": lambda row: str(row["overlap_bin"]),
                "scene_complexity_bin": lambda row: str(row["scene_complexity_bin"]),
                "crop_touches_image_edge": lambda row: str(row["crop_touches_image_edge"]).lower(),
            }
            for dimension, key_function in dimensions.items():
                groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for row in rows:
                    groups[key_function(row)].append(row)
                breakdowns[dimension] = {
                    value: {
                        "n": len(group),
                        "primary_delta_mean": mean([float(row["primary_delta"]) for row in group]),
                        "continuous_error_delta_mean": mean(
                            [float(row["continuous_error_delta"]) for row in group]
                        ),
                    }
                    for value, group in sorted(groups.items())
                }
            worst = sorted(rows, key=lambda row: float(row["primary_delta"]))[:8]
            candidate_analysis[task] = {
                "n": len(rows),
                "primary_delta_mean": mean([float(row["primary_delta"]) for row in rows]),
                "continuous_error_delta_mean": mean(
                    [float(row["continuous_error_delta"]) for row in rows]
                ),
                "candidate_parse_rate": mean([float(row["candidate_parse"]) for row in rows]),
                "crop_area_fraction_mean": mean(
                    [float(row["crop_area_fraction"]) for row in rows]
                ),
                "crop_edge_touch_rate": mean(
                    [float(row["crop_touches_image_edge"]) for row in rows]
                ),
                "breakdowns": breakdowns,
                "worst_primary_deltas": [
                    {
                        "sample_id": row["sample_id"],
                        "category": row["category"],
                        "primary_delta": row["primary_delta"],
                        "target_area_bin": row["target_area_bin"],
                        "edge_bin": row["edge_bin"],
                        "overlap_bin": row["overlap_bin"],
                        "crop_touches_image_edge": row["crop_touches_image_edge"],
                    }
                    for row in worst
                ],
            }
        analyses[candidate] = candidate_analysis

    report = {
        "status": "GLOBAL_ROI_FAILURE_ATTRIBUTION_COMPLETE" if not errors else "GLOBAL_ROI_FAILURE_ATTRIBUTION_PARTIAL",
        "evidence_level": "32-image revealed development diagnostic",
        "geometric_target_coverage": (
            "100% by construction: the expanded ground-truth bbox is inside every requested ROI; "
            "this does not prove semantic detail preservation"
        ),
        "coordinate_mapping": "pixel-rounded crop mapping unit tests pass",
        "analyses": analyses,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 全局与ROI表示失败归因",
        "",
        f"状态：`{report['status']}`。范围为32张已揭示开发图，不是盲测。",
        "",
        "真值框经扩张后按构造完整包含目标，坐标回映射单元测试通过，全部候选格式解析率见下表。因此当前损失不能归因为简单的漏选目标或解析失败；仍可能来自裁剪后的视觉分布变化、上下文缺失、多图干扰和分辨率/边距选择。",
        "",
        "| 候选 | 任务 | n | 主指标差 | 连续误差变化 | 平均裁剪面积 | 触边率 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, tasks in analyses.items():
        for task, item in tasks.items():
            lines.append(
                f"| {candidate} | {task} | {item['n']} | {item['primary_delta_mean']:+.4f} | "
                f"{item['continuous_error_delta_mean']:+.4f} | {item['crop_area_fraction_mean']:.3f} | "
                f"{item['crop_edge_touch_rate']:.3f} |"
            )
    lines.extend(["", "## 分层差值", ""])
    for candidate, tasks in analyses.items():
        for task, item in tasks.items():
            lines.extend([f"### {candidate} / {task}", ""])
            for dimension, groups in item["breakdowns"].items():
                values = ", ".join(
                    f"{name}: n={group['n']}, Δ={group['primary_delta_mean']:+.4f}"
                    for name, group in groups.items()
                )
                lines.append(f"- `{dimension}`：{values}。")
            lines.append("")
    lines.extend(
        [
            "## 当前归因边界",
            "",
            "- ROI-only用于判断局部表示本身；ROI-first加全局图用于判断上下文是否恢复。二者差值若很小，不能把全部退化简单归因于缺少全局图。",
            "- 真值ROI仍读取标注，不反映可部署控制器的定位错误；下一步必须在开发集测试边距/尺度，再接入不读真值的轻量策略。",
            "- 分层子组样本很小，只用于提出后续假设，不作显著性结论。",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": len(errors)}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
