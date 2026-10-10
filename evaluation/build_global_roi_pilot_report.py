#!/usr/bin/env python3
"""Summarize the revealed global-plus-ROI mechanics pilot."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = {item["sample_id"]: item for item in load_jsonl(args.manifest)}
    records = load_jsonl(args.results)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["strategy"], record["task"])].append(record)
    summary: dict[str, Any] = {
        "status": "REVEALED_ORACLE_MECHANICS_PILOT_NOT_VALIDATION",
        "result_records": len(records),
        "groups": {},
    }
    for (strategy, task), group in sorted(grouped.items()):
        by_sample: dict[str, dict[str, Any]] = {}
        for record in group:
            by_sample.setdefault(record["sample_id"], record)
        scored = [score_record(samples[sample_id], record) for sample_id, record in by_sample.items()]
        latencies = [float(record["application_end_to_end_ms"]) for record in group]
        tokens = [float(record["final_context_visual_tokens"]) for record in group]
        entry: dict[str, Any] = {
            "strategy": strategy,
            "task": task,
            "unique_samples": len(scored),
            "repeats_total": len(group),
            "parse_rate": statistics.fmean(bool(item["parse_success"]) for item in scored),
            "visual_tokens_mean": statistics.fmean(tokens),
            "application_latency_ms_median": statistics.median(latencies),
            "application_latency_ms_p95": p95(latencies),
        }
        if task == "pointing":
            entry.update(
                pck_0_1=statistics.fmean(bool(item["pck_0_1"]) for item in scored),
                pck_0_2=statistics.fmean(bool(item["pck_0_2"]) for item in scored),
                point_nme_bbox_diag_mean=statistics.fmean(
                    float(item["point_nme_bbox_diag"])
                    for item in scored
                    if item["point_nme_bbox_diag"] is not None
                ),
            )
        else:
            entry.update(
                iou_mean=statistics.fmean(float(item["iou"]) for item in scored),
                acc_0_5=statistics.fmean(bool(item["acc_0_5"]) for item in scored),
            )
        summary["groups"][f"{strategy}:{task}"] = entry

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 全局图加ROI BF16小样本机制实验",
        "",
        "状态：已揭示真值ROI机制实验，不是盲测，也不能用于精度或加速结论。目的仅是验证双图模型前向、MRoPE/DeepStack执行、坐标回映射和计时字段。",
        "",
        "| 策略 | 任务 | 图像数 | 平均视觉Token | 应用延迟p50/p95 ms | 质量 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for entry in summary["groups"].values():
        quality = (
            f"PCK@0.1={entry['pck_0_1']:.3f}, PCK@0.2={entry['pck_0_2']:.3f}"
            if entry["task"] == "pointing"
            else f"IoU={entry['iou_mean']:.3f}, Acc@0.5={entry['acc_0_5']:.3f}"
        )
        lines.append(
            f"| {entry['strategy']} | {entry['task']} | {entry['unique_samples']} | "
            f"{entry['visual_tokens_mean']:.1f} | "
            f"{entry['application_latency_ms_median']:.1f}/{entry['application_latency_ms_p95']:.1f} | "
            f"{quality} |"
        )
    lines.extend(
        [
            "",
            "注意：ROI策略读取真值框；小样本质量数值只用于发现接口错误。最终成本必须加入可部署Scout/策略的全部开销，并在全新隔离数据上评价。",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "records": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
