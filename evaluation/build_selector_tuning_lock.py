#!/usr/bin/env python3
"""Rank preregistered selector weights and freeze one configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate, load_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(summary: dict[str, Any]) -> float:
    point = summary["tasks"]["pointing"]
    ground = summary["tasks"]["grounding"]
    return (point["pck_0_1"] + point["pck_0_2"] + ground["iou_mean"] + ground["acc_0_5"]) / 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--expanded-results", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blind-validation-manifest", type=Path)
    args = parser.parse_args()

    specs = (
        (0.0, args.input_dir / "w0000_g3/residual_r0750.jsonl"),
        (0.25, args.expanded_results / "residual_r0750.jsonl"),
        (0.5, args.input_dir / "w0500_g3/residual_r0750.jsonl"),
        (0.75, args.input_dir / "w0750_g3/residual_r0750.jsonl"),
        (1.0, args.input_dir / "w1000_g3/residual_r0750.jsonl"),
    )
    candidates = []
    for weight, path in specs:
        records = load_jsonl(path)
        if len(records) != 192 or len({record["sample_id"] for record in records}) != 192:
            raise ValueError(f"{path}: expected 192 unique predictions")
        _, metrics = evaluate(args.manifest, path)
        candidates.append(
            {
                "energy_weight": weight,
                "path": str(path),
                "sha256": sha256(path),
                "objective": score(metrics),
                "metrics": metrics,
            }
        )
    candidates.sort(key=lambda item: (-item["objective"], item["energy_weight"]))
    selected = candidates[0]
    l2_path = args.expanded_results / "shallow_r0750.jsonl"
    _, l2_metrics = evaluate(args.manifest, l2_path)
    point = selected["metrics"]["tasks"]["pointing"]
    ground = selected["metrics"]["tasks"]["grounding"]
    l2_point = l2_metrics["tasks"]["pointing"]
    l2_ground = l2_metrics["tasks"]["grounding"]
    delta_l2 = {
        "pck_0_1": point["pck_0_1"] - l2_point["pck_0_1"],
        "pck_0_2": point["pck_0_2"] - l2_point["pck_0_2"],
        "iou_mean": ground["iou_mean"] - l2_ground["iou_mean"],
        "acc_0_5": ground["acc_0_5"] - l2_ground["acc_0_5"],
    }
    blind_validation = None
    if args.blind_validation_manifest:
        blind_records = load_jsonl(args.blind_validation_manifest)
        development_ids = {int(record["coco_image_id"]) for record in load_jsonl(args.manifest)}
        blind_ids = {int(record["coco_image_id"]) for record in blind_records}
        if len(blind_records) != 192 or len(blind_ids) != 96:
            raise ValueError("blind validation manifest must contain 96 images / 192 tasks")
        if development_ids & blind_ids:
            raise ValueError("blind validation overlaps the development manifest")
        blind_validation = {
            "path": str(args.blind_validation_manifest),
            "sha256": sha256(args.blind_validation_manifest),
            "images": len(blind_ids),
            "samples": len(blind_records),
            "development_overlap": 0,
        }
    lock = {
        "date": "2026-08-25",
        "status": (
            "FROZEN_WITH_BLIND_MANIFEST_BEFORE_INFERENCE"
            if blind_validation
            else "FROZEN_BEFORE_BLIND_VALIDATION"
        ),
        "development_manifest": str(args.manifest),
        "development_manifest_sha256": sha256(args.manifest),
        "selection_objective": "equal mean of PCK@0.1, PCK@0.2, Mean IoU, Acc@0.5",
        "preregistered_weights": [0.0, 0.25, 0.5, 0.75, 1.0],
        "fixed_parameters": {
            "retention": 0.75,
            "prune_after_layer": 5,
            "region_grid": 3,
            "selector": "residual_quota",
        },
        "selected_energy_weight": selected["energy_weight"],
        "selected_objective": selected["objective"],
        "selected_path": selected["path"],
        "selected_sha256": selected["sha256"],
        "delta_vs_l2_same_budget": delta_l2,
        "l2_reference": {"path": str(l2_path), "sha256": sha256(l2_path)},
        "ranking": candidates,
        "blind_validation_manifest": blind_validation,
        "blind_validation_results": None,
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 统一空间选择器参数冻结",
        "",
        "日期：2026-08-25  ",
        "状态：`FROZEN_WITH_BLIND_MANIFEST_BEFORE_INFERENCE`"
        if blind_validation
        else "状态：`FROZEN_BEFORE_BLIND_VALIDATION`",
        "",
        "固定开发目标：PCK@0.1、PCK@0.2、Mean IoU、Acc@0.5 等权平均；",
        "固定计算配置：75% Token、layer-5、3×3 区域配额。GPU 延迟不参与选择。",
        "",
        "| 能量权重 | 四指标均值 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        candidate_point = candidate["metrics"]["tasks"]["pointing"]
        candidate_ground = candidate["metrics"]["tasks"]["grounding"]
        lines.append(
            f"| {candidate['energy_weight']:.2f} | {candidate['objective']:.4f} | "
            f"{candidate_point['pck_0_1']:.4f} | {candidate_point['pck_0_2']:.4f} | "
            f"{candidate_ground['iou_mean']:.4f} | {candidate_ground['acc_0_5']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"冻结结果：能量权重 `{selected['energy_weight']:.2f}`，即纯四邻域特征残差。",
            "该配置在开发集相对同预算 L2 的四项变化为："
            f"PCK@0.1 {delta_l2['pck_0_1']:+.4f}、PCK@0.2 {delta_l2['pck_0_2']:+.4f}、"
            f"Mean IoU {delta_l2['iou_mean']:+.4f}、Acc@0.5 {delta_l2['acc_0_5']:+.4f}。",
            "",
            "从此文件生成后，不再根据盲验证结果更改该配置；失败则记录负结果并返回新一轮开发。",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected_energy_weight": selected["energy_weight"], "objective": selected["objective"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
