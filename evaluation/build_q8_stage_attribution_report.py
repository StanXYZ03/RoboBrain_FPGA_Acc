#!/usr/bin/env python3
"""Compare stock, Q8-A and Q8-B on a Q8-B failure-attribution subset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record
from evaluation.build_q8_expanded_diagnostic_report import paired_bootstrap


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--q8b-raw", type=Path, required=True)
    parser.add_argument("--q8a-raw", type=Path, required=True)
    parser.add_argument("--mode", default="oracle0875")
    parser.add_argument("--subset-selected", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    q8b_records = load_jsonl(args.q8b_raw)
    q8a_records = load_jsonl(args.q8a_raw)
    q8b = {(r["sample_id"], r["mode"]): r for r in q8b_records}
    q8a = {r["sample_id"]: r for r in q8a_records}
    rows: list[dict[str, Any]] = []
    classifications: Counter[str] = Counter()

    for sample in samples:
        sample_id = sample["sample_id"]
        stock_record = q8b[(sample_id, "stock")]
        q8b_record = q8b[(sample_id, args.mode)]
        q8a_record = q8a[sample_id]
        stock = score_record(sample, stock_record)
        score_a = score_record(sample, q8a_record)
        score_b = score_record(sample, q8b_record)
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "task": sample["task"],
            "category": sample["category"],
            "stock_output": stock_record["output_text"],
            "q8a_output": q8a_record["output_text"],
            "q8b_output": q8b_record["output_text"],
            "q8a_exact_stock": q8a_record["output_text"] == stock_record["output_text"],
            "q8b_exact_stock": q8b_record["output_text"] == stock_record["output_text"],
        }
        if sample["task"] == "pointing":
            stock_error = stock["point_nme_bbox_diag"]
            a_error = score_a["point_nme_bbox_diag"]
            b_error = score_b["point_nme_bbox_diag"]
            row.update(
                stock_error=stock_error, q8a_error=a_error, q8b_error=b_error,
                q8a_error_delta=None if a_error is None or stock_error is None else a_error - stock_error,
                q8b_error_delta=None if b_error is None or stock_error is None else b_error - stock_error,
                stock_primary=float(stock["pck_0_1"]),
                q8a_primary=float(score_a["pck_0_1"]),
                q8b_primary=float(score_b["pck_0_1"]),
                stock_secondary=float(stock["pck_0_2"]),
                q8a_secondary=float(score_a["pck_0_2"]),
                q8b_secondary=float(score_b["pck_0_2"]),
            )
            b_bad = float(score_b["pck_0_1"]) < float(stock["pck_0_1"]) or (
                row["q8b_error_delta"] is not None and row["q8b_error_delta"] > 0.02
            )
            a_recovers = float(score_a["pck_0_1"]) >= float(stock["pck_0_1"]) and (
                row["q8a_error_delta"] is None or row["q8a_error_delta"] <= 0.02
            )
        else:
            stock_error = stock["iou"]
            a_error = score_a["iou"]
            b_error = score_b["iou"]
            row.update(
                stock_error=stock_error, q8a_error=a_error, q8b_error=b_error,
                q8a_error_delta=a_error - stock_error,
                q8b_error_delta=b_error - stock_error,
                stock_primary=float(stock["acc_0_5"]),
                q8a_primary=float(score_a["acc_0_5"]),
                q8b_primary=float(score_b["acc_0_5"]),
                stock_secondary=float(stock["iou"]),
                q8a_secondary=float(score_a["iou"]),
                q8b_secondary=float(score_b["iou"]),
            )
            b_bad = float(score_b["acc_0_5"]) < float(stock["acc_0_5"]) or b_error - stock_error < -0.03
            a_recovers = float(score_a["acc_0_5"]) >= float(stock["acc_0_5"]) and a_error - stock_error >= -0.03
        classification = (
            "EARLY_PRUNING_SUPPORTED" if b_bad and a_recovers
            else "TOKEN_SET_OR_SHARED_COMPACTION_SUPPORTED" if b_bad
            else "NO_LONGER_PRIMARY_FAILURE"
        )
        row["classification"] = classification
        classifications[classification] += 1
        rows.append(row)

    task_summary = {}
    for task in ("grounding", "pointing"):
        items = [row for row in rows if row["task"] == task]
        task_summary[task] = {
            "n": len(items),
            "q8a_exact_stock_rate": statistics.fmean(row["q8a_exact_stock"] for row in items),
            "q8b_exact_stock_rate": statistics.fmean(row["q8b_exact_stock"] for row in items),
            "q8a_primary_mean": statistics.fmean(row["q8a_primary"] for row in items),
            "q8b_primary_mean": statistics.fmean(row["q8b_primary"] for row in items),
            "stock_primary_mean": statistics.fmean(row["stock_primary"] for row in items),
            "q8a_error_delta_mean_present": statistics.fmean(
                row["q8a_error_delta"] for row in items if row["q8a_error_delta"] is not None
            ),
            "q8b_error_delta_mean_present": statistics.fmean(
                row["q8b_error_delta"] for row in items if row["q8b_error_delta"] is not None
            ),
        }

    q8b_minus_q8a = {}
    comparison_metrics = {
        "pointing_pck_0_1": ("pointing", "primary"),
        "pointing_pck_0_2": ("pointing", "secondary"),
        "grounding_acc_0_5": ("grounding", "primary"),
        "grounding_iou_mean": ("grounding", "secondary"),
    }
    for offset, (name, (task, metric)) in enumerate(comparison_metrics.items()):
        differences = [
            row[f"q8b_{metric}"] - row[f"q8a_{metric}"]
            for row in rows if row["task"] == task
        ]
        item = paired_bootstrap(differences, 20261011 + offset, 10000)
        item["effect_vs_zero"] = (
            "Q8B_LOWER" if item["ci95_high"] < 0.0
            else "Q8B_HIGHER" if item["ci95_low"] > 0.0
            else "UNCERTAIN"
        )
        q8b_minus_q8a[name] = item

    fields = sorted({key for row in rows for key in row})
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "format": "q8-stage-attribution-v1",
        "status": (
            "ATTRIBUTION_COMPLETE_NOT_POPULATION_ESTIMATE"
            if args.subset_selected else
            "ATTRIBUTION_COMPLETE_EXPANDED_DIAGNOSTIC"
        ),
        "selection_bias_warning": (
            "Subset selected after observing Q8-B degradation; do not use subset means as full-set accuracy."
            if args.subset_selected else None
        ),
        "sample_count": len(rows),
        "classifications": dict(classifications),
        "task_summary": task_summary,
        "q8b_minus_q8a_paired": q8b_minus_q8a,
        "manifest_sha256": sha256(args.manifest),
        "q8b_raw_sha256": sha256(args.q8b_raw),
        "q8a_raw_sha256": sha256(args.q8a_raw),
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Q8 87.5%裁剪阶段归因",
        "",
        (
            "本报告分析根据Q8-B退化事后选出的样本，不能把子集均值解释为总体精度。"
            if args.subset_selected else
            "本报告分析完整96图、192任务扩大诊断集，不存在按本轮Q8-B结果筛选样本的问题；但数据曾用于旧L2研究，仍不是新盲测。"
        ) + "Q8-A与Q8-B使用同一保留集合；差别是Q8-A在Merger后压紧，Q8-B在视觉块1后压紧。",
        "",
        "## 分类结果",
        "",
        f"- 支持早期视觉裁剪为主要原因：{classifications['EARLY_PRUNING_SUPPORTED']}条。",
        f"- Q8-A也未恢复，支持Token集合或共享压紧路径为原因：{classifications['TOKEN_SET_OR_SHARED_COMPACTION_SUPPORTED']}条。",
        f"- 按复核规则不再构成主失败：{classifications['NO_LONGER_PRIMARY_FAILURE']}条。",
        "",
        "## 分任务摘要",
        "",
        "| 任务 | n | stock主指标 | Q8-A主指标 | Q8-B主指标 | Q8-A与stock完全同输出 | Q8-B与stock完全同输出 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task, item in task_summary.items():
        lines.append(
            f"| {task} | {item['n']} | {item['stock_primary_mean']:.4f} | "
            f"{item['q8a_primary_mean']:.4f} | {item['q8b_primary_mean']:.4f} | "
            f"{item['q8a_exact_stock_rate']:.4f} | {item['q8b_exact_stock_rate']:.4f} |"
        )
    lines += [
        "",
        "Pointing主指标为PCK@0.1，Grounding主指标为Acc@0.5。逐样本误差与分类保存在CSV；分类只提供机制证据，不替代在完整集上运行Q8-A的总体比较。",
        "",
        "## Q8-B相对Q8-A的完整集配对差异",
        "",
        "| 指标 | n | Q8-B减Q8-A | 95% Bootstrap区间 | 方向 |",
        "|---|---:|---:|---:|---|",
    ]
    for name, item in q8b_minus_q8a.items():
        lines.append(
            f"| {name} | {item['n']} | {item['mean_delta']:+.4f} | "
            f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | {item['effect_vs_zero']} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "classifications": dict(classifications)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
