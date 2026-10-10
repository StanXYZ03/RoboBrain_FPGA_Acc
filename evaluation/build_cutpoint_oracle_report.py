#!/usr/bin/env python3
"""Summarize the development-only multi-cut-point geometric-oracle scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.cutpoint_oracle import CUT_POINTS
from evaluation.metrics import evaluate, load_jsonl
from evaluation.run_hf_cutpoint_oracle import output_path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def med(records: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument(
        "--cut-points", nargs="+", choices=tuple(CUT_POINTS), default=tuple(CUT_POINTS)
    )
    parser.add_argument(
        "--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25)
    )
    args = parser.parse_args()

    rows = []
    missing = []
    output_texts: dict[tuple[str, float], dict[str, str]] = {}
    for cut_name in args.cut_points:
        for retention in args.retentions:
            path = output_path(args.results_dir, cut_name, retention)
            if not path.is_file():
                missing.append(str(path))
                continue
            records = load_jsonl(path)
            scored, metrics = evaluate(
                args.manifest, path, limit_per_task=args.limit_per_task
            )
            output_texts[(cut_name, retention)] = {
                str(record["sample_id"]): str(record["output_text"]) for record in records
            }
            point = metrics["tasks"]["pointing"]
            grounding = metrics["tasks"]["grounding"]
            combined_ratios = [
                (
                    float(record["encoder_macs_proxy"])
                    + float(record["llm_prefill_macs_proxy"])
                )
                /
                (
                    float(record["baseline_encoder_macs_proxy"])
                    + float(record["baseline_llm_prefill_macs_proxy"])
                )
                for record in records
            ]
            rows.append(
                {
                    "cut_name": cut_name,
                    "completed_vision_blocks": CUT_POINTS[cut_name].completed_vision_blocks,
                    "retention": retention,
                    "raw": str(path),
                    "raw_sha256": file_sha256(path),
                    "sample_count": len(scored),
                    "pck_0_1": point["pck_0_1"],
                    "pck_0_2": point["pck_0_2"],
                    "iou_mean": grounding["iou_mean"],
                    "acc_0_5": grounding["acc_0_5"],
                    "vision_seconds_median": med(records, "vision_encoder_seconds"),
                    "llm_prefill_seconds_median": med(records, "llm_prefill_seconds"),
                    "elapsed_seconds_median": med(records, "elapsed_seconds"),
                    "combined_mac_ratio_median": float(statistics.median(combined_ratios)),
                    "one_way_10gbe_ms_median": 1000.0
                    * float(
                        statistics.median(
                            record["communication"]["one_way_serialization_seconds"][
                                "10GbE"
                            ]
                            for record in records
                        )
                    ),
                    "round_trip_10gbe_ms_median": 1000.0
                    * float(
                        statistics.median(
                            record["communication"]["round_trip_serialization_seconds"][
                                "10GbE"
                            ]
                            for record in records
                        )
                    ),
                }
            )
    if missing:
        raise SystemExit("missing result files:\n" + "\n".join(missing))

    baseline = next(
        row
        for row in rows
        if row["cut_name"] == "patch_embedding" and row["retention"] == 1.0
    )
    full_maps = [output_texts[(cut_name, 1.0)] for cut_name in args.cut_points]
    full_equivalent = all(mapping == full_maps[0] for mapping in full_maps[1:])
    candidates = []
    for row in rows:
        if row["retention"] == 1.0:
            row["smoke_gate"] = "BASELINE"
            continue
        passes = (
            row["pck_0_1"] >= baseline["pck_0_1"] - 0.125
            and row["pck_0_2"] >= baseline["pck_0_2"] - 0.125
            and row["iou_mean"] >= baseline["iou_mean"] - 0.05
            and row["acc_0_5"] >= baseline["acc_0_5"] - 0.125
            and row["combined_mac_ratio_median"] < 0.90
            and row["one_way_10gbe_ms_median"] < 40.0
        )
        row["smoke_gate"] = "CANDIDATE" if passes else "FAIL"
        if passes:
            candidates.append(row)
    selected = (
        min(
            candidates,
            key=lambda row: (
                row["completed_vision_blocks"],
                row["combined_mac_ratio_median"],
                row["retention"],
            ),
        )
        if candidates
        else None
    )
    status = (
        "SMOKE_CANDIDATE_REQUIRES_SCALEUP"
        if selected is not None and full_equivalent
        else "STOP_EXTERNAL_SELECTOR_ROUTE"
    )
    payload = {
        "format": "cutpoint-geometric-oracle-summary-v1",
        "status": status,
        "development_only": True,
        "not_an_accuracy_claim": True,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "full_retention_outputs_equivalent_across_cutpoints": full_equivalent,
        "gate": {
            "pck_0_1_max_drop": 0.125,
            "pck_0_2_max_drop": 0.125,
            "iou_mean_max_drop": 0.05,
            "acc_0_5_max_drop": 0.125,
            "combined_mac_ratio_max_exclusive": 0.90,
            "one_way_10gbe_ms_max_exclusive": 40.0,
            "note": "With four samples per task, this is only a fail-fast smoke gate; a candidate must be rerun on the remaining development set and then frozen before a new untouched test set is created."
        },
        "selected_smoke_candidate": selected,
        "rows": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# RoboBrain2.5-4B BF16 多切入点几何Oracle门控",
        "",
        f"状态：`{status}`。这是BF16模型上每任务4条的新开发集烟雾实验，Oracle读取真实框，不能部署，不能作为最终Q8_0精度结论。",
        "",
        f"100%保留路径跨切入点输出一致：`{'PASS' if full_equivalent else 'FAIL'}`。",
        "",
        "| 切入点 | 视觉块数 | 保留率 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | 主MAC比例 | 10GbE单向/ms | GPU总时中位/s | 门控 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cut_name']} | {row['completed_vision_blocks']} | {row['retention']:.2f} | "
            f"{row['pck_0_1']:.3f} | {row['pck_0_2']:.3f} | {row['iou_mean']:.3f} | "
            f"{row['acc_0_5']:.3f} | {row['combined_mac_ratio_median']:.3f} | "
            f"{row['one_way_10gbe_ms_median']:.3f} | {row['elapsed_seconds_median']:.3f} | "
            f"{row['smoke_gate']} |"
        )
    lines.extend(["", "## 决策", ""])
    if selected is None:
        lines.append("没有切入点通过几何Oracle烟雾门，停止外置Selector硬件路线，回到驻留连续子图或单板稀疏计算面。")
    else:
        lines.append(
            f"BF16首个候选为`{selected['cut_name']}@{selected['retention']:.2f}`。先在最终目标Q8_0 GGUF执行路径复验；Q8未通过不得冻结描述符或RTL边界。"
        )
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": status, "selected": selected, "json": str(args.json), "markdown": str(args.markdown)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
