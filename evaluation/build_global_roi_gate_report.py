#!/usr/bin/env python3
"""Build paired development evidence for global/ROI representation strategies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record


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


def bootstrap_delta(values: list[float], seed: int, iterations: int) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires nonempty paired differences")
    rng = random.Random(seed)
    n = len(values)
    means = [
        statistics.fmean(values[rng.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    ]
    return {
        "n": n,
        "mean_delta": statistics.fmean(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "iterations": iterations,
    }


def p95(values: list[float]) -> float:
    return percentile(values, 0.95)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--baseline", default="dense")
    parser.add_argument(
        "--include-strategies",
        nargs="+",
        help="Only summarize these strategy labels from the input result files.",
    )
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--omit-latency-comparison",
        action="store_true",
        help="Hide latency when result files were collected under different GPU load.",
    )
    args = parser.parse_args()

    manifest_records = {record["sample_id"]: record for record in load_jsonl(args.manifest)}
    records = [record for path in args.results for record in load_jsonl(path)]
    if args.include_strategies:
        included = set(args.include_strategies)
        records = [record for record in records if str(record["strategy"]) in included]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["strategy"]), str(record["task"]))].append(record)
        by_key[(str(record["strategy"]), str(record["sample_id"]))].append(record)

    errors: list[str] = []
    outputs_deterministic = True
    first: dict[tuple[str, str], dict[str, Any]] = {}
    for key, runs in by_key.items():
        if len(runs) != args.expected_repeats:
            errors.append(f"{key}: expected {args.expected_repeats} repeats, got {len(runs)}")
        if len({str(run["output_text"]) for run in runs}) != 1:
            outputs_deterministic = False
            errors.append(f"{key}: output changed across deterministic repeats")
        first[key] = min(runs, key=lambda run: int(run["repeat"]))

    strategies = sorted({strategy for strategy, _ in grouped})
    tasks = sorted({task for _, task in grouped})
    if args.baseline not in strategies:
        errors.append(f"baseline strategy {args.baseline!r} is missing")

    selected_by_image: dict[int, dict[str, Any]] = {}
    for (_, sample_id), record in first.items():
        sample = manifest_records.get(sample_id)
        if sample is not None:
            selected_by_image[int(sample["coco_image_id"])] = sample
    sample_distribution = {
        key: dict(
            sorted(
                Counter(
                    str(sample["stratification"][key])
                    for sample in selected_by_image.values()
                ).items()
            )
        )
        for key in (
            "target_area_bin",
            "edge_bin",
            "overlap_bin",
            "scene_complexity_bin",
        )
    }

    metrics: dict[str, dict[str, Any]] = {}
    scored: dict[tuple[str, str], dict[str, float]] = {}
    for (strategy, task), runs in sorted(grouped.items()):
        unique_runs = [
            record for (one_strategy, _), record in first.items()
            if one_strategy == strategy and str(record["task"]) == task
        ]
        unique_runs.sort(key=lambda record: str(record["sample_id"]))
        for record in unique_runs:
            sample_id = str(record["sample_id"])
            if sample_id not in manifest_records:
                errors.append(f"prediction {sample_id} is absent from manifest")
                continue
            result = score_record(manifest_records[sample_id], record)
            if task == "pointing":
                scored[(strategy, sample_id)] = {
                    "primary": float(result["pck_0_1"]),
                    "secondary": float(result["pck_0_2"]),
                    "continuous": float(result["point_nme_bbox_diag"] or 0.0),
                    "parse": float(result["parse_success"]),
                }
            else:
                scored[(strategy, sample_id)] = {
                    "primary": float(result["iou"]),
                    "secondary": float(result["acc_0_5"]),
                    "continuous": float(result["iou"]),
                    "parse": float(result["parse_success"]),
                }

        task_scores = [scored[(strategy, str(record["sample_id"]))] for record in unique_runs]
        latencies = [float(record["application_end_to_end_ms"]) for record in runs]
        # Reconstruct the frozen budget definition so older raw pilots remain comparable.
        model_latencies = [
            float(record["roi_fetch_resize_ms"])
            + float(record["final_preprocess_ms"])
            + float(record["host_to_device_ms"])
            + float(record["generate_ms"])
            + float(record["postprocess_ms"])
            for record in runs
        ]
        vision_latencies = [float(record["final_vision_ms"]) for record in runs]
        prefill_latencies = [float(record["final_prefill_ms"]) for record in runs]
        decode_latencies = [float(record["final_decode_ms"]) for record in runs]
        preprocess_latencies = [
            float(record["image_read_decode_ms"])
            + float(record["roi_fetch_resize_ms"])
            + float(record["final_preprocess_ms"])
            for record in runs
        ]
        tokens = [float(record["final_context_visual_tokens"]) for record in unique_runs]
        metrics[f"{strategy}:{task}"] = {
            "strategy": strategy,
            "task": task,
            "unique_samples": len(unique_runs),
            "repeats": len(runs),
            "parse_rate": statistics.fmean(item["parse"] for item in task_scores),
            "primary_mean": statistics.fmean(item["primary"] for item in task_scores),
            "secondary_mean": statistics.fmean(item["secondary"] for item in task_scores),
            "continuous_mean": statistics.fmean(item["continuous"] for item in task_scores),
            "visual_tokens_mean": statistics.fmean(tokens),
            "application_latency_ms_p50": statistics.median(latencies),
            "application_latency_ms_p95": p95(latencies),
            "model_latency_ms_p50": statistics.median(model_latencies),
            "vision_ms_p50": statistics.median(vision_latencies),
            "prefill_ms_p50": statistics.median(prefill_latencies),
            "decode_ms_p50": statistics.median(decode_latencies),
            "host_preprocess_ms_p50": statistics.median(preprocess_latencies),
        }

    comparisons: dict[str, dict[str, Any]] = {}
    for strategy_index, strategy in enumerate(strategies):
        if strategy == args.baseline:
            continue
        comparisons[strategy] = {}
        for task_index, task in enumerate(tasks):
            candidate_ids = {
                sample_id for (one_strategy, sample_id), record in first.items()
                if one_strategy == strategy and str(record["task"]) == task
            }
            baseline_ids = {
                sample_id for (one_strategy, sample_id), record in first.items()
                if one_strategy == args.baseline and str(record["task"]) == task
            }
            if candidate_ids != baseline_ids:
                errors.append(f"{strategy}:{task}: sample IDs do not match baseline")
                continue
            sample_ids = sorted(candidate_ids)
            task_comparison: dict[str, Any] = {}
            for metric_index, metric in enumerate(("primary", "secondary", "parse")):
                differences = [
                    scored[(strategy, sample_id)][metric]
                    - scored[(args.baseline, sample_id)][metric]
                    for sample_id in sample_ids
                ]
                task_comparison[metric] = bootstrap_delta(
                    differences,
                    args.seed + 1000 * strategy_index + 100 * task_index + metric_index,
                    args.bootstrap,
                )
            candidate = metrics[f"{strategy}:{task}"]
            baseline = metrics[f"{args.baseline}:{task}"]
            task_comparison["visual_token_ratio"] = (
                candidate["visual_tokens_mean"] / baseline["visual_tokens_mean"]
            )
            task_comparison["application_p50_ratio"] = (
                None
                if args.omit_latency_comparison
                else candidate["application_latency_ms_p50"]
                / baseline["application_latency_ms_p50"]
            )
            comparisons[strategy][task] = task_comparison

    status = "DEVELOPMENT_REPRESENTATION_GATE_COMPLETE" if not errors else "DEVELOPMENT_REPRESENTATION_GATE_FAIL"
    report = {
        "status": status,
        "evidence_level": "development diagnostic with revealed ground-truth ROI; not deployable policy or blind validation",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "results": [str(path) for path in args.results],
        "results_sha256": {str(path): sha256(path) for path in args.results},
        "result_records": len(records),
        "selected_unique_images": len(selected_by_image),
        "selected_image_ids": sorted(selected_by_image),
        "sample_distribution": sample_distribution,
        "outputs_deterministic_across_repeats": outputs_deterministic,
        "latency_comparable": not args.omit_latency_comparison,
        "metrics": metrics,
        "paired_vs_dense": comparisons,
        "errors": errors,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 全局与ROI表示开发门控",
        "",
        f"状态：`{status}`。本报告使用已揭示真值ROI，只验证表示可行性，不代表可部署策略或盲测精度。",
        "",
        f"分层抽样图像数：`{len(selected_by_image)}`；分布：`{json.dumps(sample_distribution, ensure_ascii=False, sort_keys=True)}`。",
        "",
        "| 策略 | 任务 | n | 视觉Token | 应用p50/p95 ms | 主指标 | 次指标 | 格式率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in metrics.values():
        primary_name = "PCK@0.1" if item["task"] == "pointing" else "Mean IoU"
        secondary_name = "PCK@0.2" if item["task"] == "pointing" else "Acc@0.5"
        latency = (
            "N/A"
            if args.omit_latency_comparison
            else f"{item['application_latency_ms_p50']:.1f}/{item['application_latency_ms_p95']:.1f}"
        )
        lines.append(
            f"| {item['strategy']} | {item['task']} | {item['unique_samples']} | "
            f"{item['visual_tokens_mean']:.1f} | {latency} | {primary_name}={item['primary_mean']:.4f} | "
            f"{secondary_name}={item['secondary_mean']:.4f} | {item['parse_rate']:.4f} |"
        )
    if not args.omit_latency_comparison:
        lines.extend(
            [
                "",
                "## 阶段中位延迟",
                "",
                "| 策略 | 任务 | Host预处理 ms | Vision ms | Prefill ms | Decode ms |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in metrics.values():
            lines.append(
                f"| {item['strategy']} | {item['task']} | {item['host_preprocess_ms_p50']:.1f} | "
                f"{item['vision_ms_p50']:.1f} | {item['prefill_ms_p50']:.1f} | "
                f"{item['decode_ms_p50']:.1f} |"
            )
    else:
        lines.extend(
            [
                "",
                "延迟列已禁用：输入结果来自不同日期和不同GPU占用状态，只进行质量与Token配对。",
            ]
        )
    lines.extend(
        [
            "",
            "## 相对Dense的逐图配对差值",
            "",
            "质量项格式为 `均值差 [95% bootstrap CI]`；Token和延迟为候选/Dense比值。",
            "",
            "| 策略 | 任务 | 主指标差 | 次指标差 | 格式率差 | Token比 | p50延迟比 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy, task_values in comparisons.items():
        for task, item in task_values.items():
            def ci(metric: str) -> str:
                value = item[metric]
                return f"{value['mean_delta']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"

            latency_ratio = (
                "N/A"
                if item["application_p50_ratio"] is None
                else f"{item['application_p50_ratio']:.3f}"
            )
            lines.append(
                f"| {strategy} | {task} | {ci('primary')} | {ci('secondary')} | "
                f"{ci('parse')} | {item['visual_token_ratio']:.3f} | "
                f"{latency_ratio} |"
            )
    lines.extend(
        [
            "",
            f"三次确定性输出一致：`{str(outputs_deterministic).lower()}`。所有模型计时均位于独立warmup之后，GPU区间前后同步。",
            "",
            "证据边界：真值ROI读取标注，尚未计入可部署Scout/控制器的正确率和开销；开发结果只能用于选择后续候选，不能作为最终泛化结论。",
        ]
    )
    if errors:
        lines.extend(["", "## 错误", ""] + [f"- {error}" for error in errors])
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "records": len(records), "errors": len(errors)}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
