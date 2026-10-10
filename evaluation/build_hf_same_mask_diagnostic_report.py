#!/usr/bin/env python3
"""Compare BF16 stock and same-mask cut points on a revealed diagnostic set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_q8_expanded_diagnostic_report import paired_bootstrap
from evaluation.metrics import load_jsonl, score_record
from evaluation.run_hf_cutpoint_oracle import output_path


METRICS = {
    "pointing_pck_0_1": ("pointing", "pck_0_1"),
    "pointing_pck_0_2": ("pointing", "pck_0_2"),
    "grounding_iou_mean": ("grounding", "iou"),
    "grounding_acc_0_5": ("grounding", "acc_0_5"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--cut-points", nargs="+", default=("vision_block_1", "post_merger"))
    parser.add_argument("--baseline-cut-point", default="vision_block_1")
    parser.add_argument("--retention", type=float, default=0.875)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    baseline_path = output_path(args.results_dir, args.baseline_cut_point, 1.0)
    baseline_records = load_jsonl(baseline_path)
    baseline = {record["sample_id"]: record for record in baseline_records}
    if set(baseline) != set(sample_by_id):
        raise SystemExit("baseline records do not exactly match manifest")

    rows: dict[str, Any] = {}
    raw_hashes = {str(baseline_path): sha256(baseline_path)}
    for cut_offset, cut_name in enumerate(args.cut_points):
        compact_path = output_path(args.results_dir, cut_name, args.retention)
        compact_records = load_jsonl(compact_path)
        compact = {record["sample_id"]: record for record in compact_records}
        if set(compact) != set(sample_by_id):
            raise SystemExit(f"{cut_name} records do not exactly match manifest")
        if not all(record.get("expected_mask_match") is True for record in compact_records):
            raise SystemExit(f"{cut_name} lacks exact expected-mask evidence")
        raw_hashes[str(compact_path)] = sha256(compact_path)
        cut_metrics = {}
        for metric_offset, (name, (task, key)) in enumerate(METRICS.items()):
            differences = []
            stock_values = []
            compact_values = []
            for sample_id, sample in sample_by_id.items():
                if sample["task"] != task:
                    continue
                stock_score = score_record(sample, baseline[sample_id])
                compact_score = score_record(sample, compact[sample_id])
                stock_values.append(float(stock_score[key]))
                compact_values.append(float(compact_score[key]))
                differences.append(float(compact_score[key]) - float(stock_score[key]))
            result = paired_bootstrap(
                differences,
                args.seed + cut_offset * len(METRICS) + metric_offset,
                args.bootstrap,
            )
            result.update(
                stock_mean=statistics.fmean(stock_values),
                compact_mean=statistics.fmean(compact_values),
                effect_vs_zero=(
                    "DECREASE" if result["ci95_high"] < 0.0 else
                    "INCREASE" if result["ci95_low"] > 0.0 else
                    "UNCERTAIN"
                ),
            )
            cut_metrics[name] = result
        rows[cut_name] = cut_metrics

    payload = {
        "format": "hf-same-mask-expanded-diagnostic-v1",
        "status": "REVEALED_DIAGNOSTIC_NOT_BLIND_VALIDATION",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "retention": args.retention,
        "all_compact_masks_exactly_matched": True,
        "raw_sha256": raw_hashes,
        "cut_points": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# BF16同掩码扩大诊断",
        "",
        "状态：已揭示诊断，不是新盲测。所有裁剪记录均通过与Q8固定掩码的逐索引一致性检查。",
        "",
        "| 切入点 | 指标 | stock | 裁剪 | 差值 | 95%配对Bootstrap区间 | 相对零变化 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for cut_name, metrics in rows.items():
        for name, item in metrics.items():
            lines.append(
                f"| {cut_name} | {name} | {item['stock_mean']:.4f} | "
                f"{item['compact_mean']:.4f} | {item['mean_delta']:+.4f} | "
                f"[{item['ci95_low']:+.4f}, {item['ci95_high']:+.4f}] | "
                f"{item['effect_vs_zero']} |"
            )
    lines += [
        "",
        "解释规则：Merger后也退化支持Token集合/上下文缺失；仅视觉块1后退化支持早期视觉传播；BF16均不退化而Q8退化则应优先审查量化交互或Q8后端。",
    ]
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(args.json), "markdown": str(args.markdown)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
