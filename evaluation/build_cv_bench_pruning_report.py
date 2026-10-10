#!/usr/bin/env python3
"""Build a paired CV-Bench report for dense and frozen L2 token pruning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl


def parse_choice(text: str, choices: list[str]) -> str | None:
    candidate = text.strip()
    if "</think>" in candidate:
        candidate = candidate.rsplit("</think>", 1)[-1].strip()
    for pattern in (
        r"(?:answer|option|choice)\s*(?:is|:|=)?\s*\(?([A-Z])\)?\b",
        r"\(([A-Z])\)",
    ):
        matches = re.findall(pattern, candidate, flags=re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if ord(letter) - ord("A") < len(choices):
                return letter
    exact = re.fullmatch(r"\s*\(?([A-Z])\)?[\s.!]*", candidate, flags=re.IGNORECASE)
    if exact:
        letter = exact.group(1).upper()
        if ord(letter) - ord("A") < len(choices):
            return letter
    normalized = re.sub(r"\W+", " ", candidate).strip().casefold()
    matches = [
        chr(ord("A") + index)
        for index, choice in enumerate(choices)
        if normalized == re.sub(r"\W+", " ", choice).strip().casefold()
    ]
    return matches[0] if len(matches) == 1 else None


def answer_letter(answer: str) -> str:
    match = re.search(r"\(([A-Z])\)", answer.upper())
    if not match:
        raise ValueError(f"unsupported CV-Bench answer: {answer!r}")
    return match.group(1)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": percentile(values, 0.95),
    }


def score(records: list[dict[str, Any]], correct_key: str) -> dict[str, Any]:
    source_accuracy = {}
    for source in ("ADE20K", "COCO", "Omni3D"):
        subset = [row for row in records if row["source"] == source]
        source_accuracy[source] = statistics.fmean(bool(row[correct_key]) for row in subset)
    accuracy_2d = (source_accuracy["ADE20K"] + source_accuracy["COCO"]) / 2
    accuracy_3d = source_accuracy["Omni3D"]
    tasks = sorted({str(row["task"]) for row in records})
    return {
        "sample_count": len(records),
        "parse_success_rate": statistics.fmean(
            row[correct_key.replace("_correct", "_answer")] is not None for row in records
        ),
        "accuracy": statistics.fmean(bool(row[correct_key]) for row in records),
        "source_accuracy": source_accuracy,
        "task_accuracy": {
            task: statistics.fmean(
                bool(row[correct_key]) for row in records if row["task"] == task
            )
            for task in tasks
        },
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "cv_bench_accuracy": (accuracy_2d + accuracy_3d) / 2,
    }


def paired_bootstrap_delta(
    records: list[dict[str, Any]], left: str, right: str, seed: int, repeats: int = 5000
) -> list[float]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[f"{row['task']}::{row['source_dataset']}"].append(row)
    rng = random.Random(seed)
    deltas = []
    for _ in range(repeats):
        sample = []
        for group in groups.values():
            sample.extend(rng.choice(group) for _ in range(len(group)))
        deltas.append(score(sample, left)["cv_bench_accuracy"] - score(sample, right)["cv_bench_accuracy"])
    deltas.sort()
    return [deltas[int(0.025 * (repeats - 1))], deltas[int(0.975 * (repeats - 1))]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--dense-jsonl", type=Path, required=True)
    parser.add_argument("--pruned-jsonl", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--retention", type=float, default=0.875)
    args = parser.parse_args()
    if not (0.0 < args.retention <= 1.0):
        parser.error("--retention must be in (0, 1]")
    candidate_label = f"l2_{round(args.retention * 1000):04d}"
    candidate_display = f"L2@{args.retention:.1%}"

    manifest = {row["sample_id"]: row for row in load_jsonl(args.manifest)}
    baseline_payload = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    baseline = {str(row["idx"]): row for row in baseline_payload["records"]}
    dense_rows = load_jsonl(args.dense_jsonl)
    pruned_rows = load_jsonl(args.pruned_jsonl)
    dense = {str(row["sample_id"]): row for row in dense_rows}
    pruned = {str(row["sample_id"]): row for row in pruned_rows}
    if len(dense_rows) != len(dense) or len(pruned_rows) != len(pruned):
        raise ValueError("duplicate result sample IDs")
    for label, records, retention in (("dense", dense_rows, 1.0), ("pruned", pruned_rows, args.retention)):
        for row in records:
            if (
                row["selector"] != "l2_anchor"
                or row["prune_after_layer"] != 5
                or row["anchor_grid"] != 3
                or abs(row["requested_retention"] - retention) > 1e-12
            ):
                raise ValueError(f"{label} result does not use the frozen strategy")
    expected = set(manifest)
    for label, records in (("baseline", baseline), ("dense", dense), ("pruned", pruned)):
        if set(records) != expected:
            raise ValueError(f"{label} sample IDs do not exactly match the frozen manifest")

    joined = []
    for sample_id in manifest:
        meta, native, full, cut = manifest[sample_id], baseline[sample_id], dense[sample_id], pruned[sample_id]
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        native_answer = native.get("parsed_answer") or parse_choice(native["output_text"], choices)
        dense_answer = parse_choice(full["output_text"], choices)
        pruned_answer = parse_choice(cut["output_text"], choices)
        joined.append(
            {
                "sample_id": sample_id,
                "task": meta["task"],
                "type": meta["type"],
                "source": meta["source"],
                "source_dataset": meta["source_dataset"],
                "ground_truth": truth,
                "native_answer": native_answer,
                "dense_answer": dense_answer,
                "pruned_answer": pruned_answer,
                "native_correct": native_answer == truth,
                "dense_correct": dense_answer == truth,
                "pruned_correct": pruned_answer == truth,
                "native_output": native["output_text"],
                "dense_output": full["output_text"],
                "pruned_output": cut["output_text"],
                "native_dense_text_equal": native["output_text"] == full["output_text"],
                "native_dense_answer_equal": native_answer == dense_answer,
                "dense_pruned_text_equal": full["output_text"] == cut["output_text"],
                "dense_pruned_answer_equal": dense_answer == pruned_answer,
                "full_visual_tokens": int(cut["baseline_llm_visual_tokens"]),
                "pruned_visual_tokens": int(cut["llm_visual_tokens"]),
                "actual_token_retention": float(cut["actual_token_retention"]),
                "dense_vision_seconds": float(full["vision_encoder_seconds"]),
                "pruned_vision_seconds": float(cut["vision_encoder_seconds"]),
                "dense_prefill_seconds": float(full["llm_prefill_seconds"]),
                "pruned_prefill_seconds": float(cut["llm_prefill_seconds"]),
                "dense_decode_seconds": float(full["decoder_seconds"]),
                "pruned_decode_seconds": float(cut["decoder_seconds"]),
                "dense_inference_seconds": float(full["elapsed_seconds"]),
                "pruned_inference_seconds": float(cut["elapsed_seconds"]),
                "dense_preprocess_seconds": float(full["preprocess_seconds"]),
                "pruned_preprocess_seconds": float(cut["preprocess_seconds"]),
                "dense_peak_cuda_bytes": int(full["peak_cuda_bytes"]),
                "pruned_peak_cuda_bytes": int(cut["peak_cuda_bytes"]),
                "vision_macs_ratio": float(cut["encoder_macs_proxy"]) / float(cut["baseline_encoder_macs_proxy"]),
                "prefill_macs_ratio": float(cut["llm_prefill_macs_proxy"]) / float(cut["baseline_llm_prefill_macs_proxy"]),
            }
        )

    scores = {
        "native_bf16": score(joined, "native_correct"),
        "custom_dense": score(joined, "dense_correct"),
        candidate_label: score(joined, "pruned_correct"),
    }
    dense_score = scores["custom_dense"]["cv_bench_accuracy"]
    pruned_score = scores[candidate_label]["cv_bench_accuracy"]
    quality = {
        "absolute_delta_vs_custom_dense": pruned_score - dense_score,
        "relative_score_retention_vs_custom_dense": pruned_score / dense_score,
        "paired_delta_ci95": paired_bootstrap_delta(
            joined, "pruned_correct", "dense_correct", args.seed + 17
        ),
        "native_dense_exact_text_rate": statistics.fmean(row["native_dense_text_equal"] for row in joined),
        "native_dense_answer_agreement": statistics.fmean(row["native_dense_answer_equal"] for row in joined),
        "dense_pruned_exact_text_rate": statistics.fmean(row["dense_pruned_text_equal"] for row in joined),
        "dense_pruned_answer_agreement": statistics.fmean(row["dense_pruned_answer_equal"] for row in joined),
        "dense_correct_to_pruned_incorrect": sum(row["dense_correct"] and not row["pruned_correct"] for row in joined),
        "dense_incorrect_to_pruned_correct": sum(not row["dense_correct"] and row["pruned_correct"] for row in joined),
    }
    timing = {}
    for prefix in ("dense", "pruned"):
        timing[prefix] = {
            name: stats([row[f"{prefix}_{name}_seconds"] * 1000 for row in joined])
            for name in ("vision", "prefill", "decode", "inference", "preprocess")
        }
    reductions = {
        "mean_token_retention": statistics.fmean(row["actual_token_retention"] for row in joined),
        "mean_token_reduction": 1 - statistics.fmean(row["actual_token_retention"] for row in joined),
        "mean_vision_macs_ratio": statistics.fmean(row["vision_macs_ratio"] for row in joined),
        "mean_prefill_macs_ratio": statistics.fmean(row["prefill_macs_ratio"] for row in joined),
        "inference_speedup": timing["dense"]["inference"]["mean"] / timing["pruned"]["inference"]["mean"],
        "vision_speedup": timing["dense"]["vision"]["mean"] / timing["pruned"]["vision"]["mean"],
        "prefill_speedup": timing["dense"]["prefill"]["mean"] / timing["pruned"]["prefill"]["mean"],
    }
    sample_count = len(joined)
    is_full = sample_count == 2638
    scope = (
        "CV-Bench complete 2638-sample benchmark"
        if is_full else f"fixed {sample_count}-sample stratified CV-Bench screening"
    )
    strategy = json.loads(args.lock_file.read_text(encoding="utf-8"))["strategy"]
    strategy["retention"] = args.retention
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "evaluation_scope": scope,
        "strategy": strategy,
        "scores": scores,
        "quality_comparison": quality,
        "timing_ms": timing,
        "reductions": reductions,
        "records": joined,
    }
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)

    lines = [
        "# CV-Bench 冻结 Token 裁剪配对评测",
        "",
        (
            "> 已评测 CV-Bench 全部 2638 条，按官方来源分组组合公式计分。"
            if is_full else f"> 这是固定分层 {sample_count} 样本开发筛选，不是 2638 样本完整官方评测。"
        ),
        "",
        "## 实验设置",
        "",
        f"- 策略：视觉第 5 层后，按 2×2 patch group 的 L2 范数排序，保留 {args.retention:.1%}，并保留 3×3 区域锚点。",
        "- 位置：保留原始 MRoPE 位置，不对剩余 Token 重新编号。",
        f"- 对照：原生 `model.generate()` BF16、相同自定义路径 Dense、相同路径 {candidate_display}。",
        "- 解码：greedy，`do_sample=False` 等价路径，最多 16 个新 Token。",
        "",
        "## 准确率",
        "",
        "| 配置 | CV-Bench组合分数 | 2D | 3D |",
        "|---|---:|---:|---:|",
    ]
    for label in ("native_bf16", "custom_dense", candidate_label):
        item = scores[label]
        lines.append(f"| {label} | {item['cv_bench_accuracy']:.4f} | {item['accuracy_2d']:.4f} | {item['accuracy_3d']:.4f} |")
    lines.extend([
        "",
        f"- {candidate_display} 相对自定义 Dense：绝对变化 {quality['absolute_delta_vs_custom_dense']:+.4f}，相对保留 {quality['relative_score_retention_vs_custom_dense']:.4f}。",
        f"- 配对分层 bootstrap 的分数差 95% CI：[{quality['paired_delta_ci95'][0]:+.4f}, {quality['paired_delta_ci95'][1]:+.4f}]。",
        f"- Dense 正确→裁剪错误：{quality['dense_correct_to_pruned_incorrect']}；Dense 错误→裁剪正确：{quality['dense_incorrect_to_pruned_correct']}。",
        f"- 原生 BF16 与自定义 Dense：答案一致率 {quality['native_dense_answer_agreement']:.4f}，逐字一致率 {quality['native_dense_exact_text_rate']:.4f}。",
        f"- 答案解析成功率：Dense {scores['custom_dense']['parse_success_rate']:.4f}，裁剪 {scores[candidate_label]['parse_success_rate']:.4f}。",
        "",
        f"| 任务 | Dense | {candidate_display} |",
        "|---|---:|---:|",
    ])
    for task in scores["custom_dense"]["task_accuracy"]:
        lines.append(
            f"| {task} | {scores['custom_dense']['task_accuracy'][task]:.4f} | {scores[candidate_label]['task_accuracy'][task]:.4f} |"
        )
    lines.extend([
        "",
        f"| 来源 | Dense | {candidate_display} |",
        "|---|---:|---:|",
    ])
    for source in scores["custom_dense"]["source_accuracy"]:
        lines.append(
            f"| {source} | {scores['custom_dense']['source_accuracy'][source]:.4f} | {scores[candidate_label]['source_accuracy'][source]:.4f} |"
        )
    lines.extend([
        "",
        "## 性能与计算量",
        "",
        f"- 实际视觉 Token 平均保留 {reductions['mean_token_retention']:.4f}，减少 {reductions['mean_token_reduction']:.2%}。",
        f"- Vision MAC 代理保留 {reductions['mean_vision_macs_ratio']:.4f}；LLM Prefill MAC 代理保留 {reductions['mean_prefill_macs_ratio']:.4f}。",
        f"- 自定义推理核心均值：Dense {timing['dense']['inference']['mean']:.3f} ms，裁剪 {timing['pruned']['inference']['mean']:.3f} ms，加速 {reductions['inference_speedup']:.3f}×。",
        f"- 自定义推理核心 p50/p95：Dense {timing['dense']['inference']['median']:.3f}/{timing['dense']['inference']['p95']:.3f} ms，裁剪 {timing['pruned']['inference']['median']:.3f}/{timing['pruned']['inference']['p95']:.3f} ms。",
        f"- Vision 均值：Dense {timing['dense']['vision']['mean']:.3f} ms，裁剪 {timing['pruned']['vision']['mean']:.3f} ms，加速 {reductions['vision_speedup']:.3f}×。",
        f"- Prefill 均值：Dense {timing['dense']['prefill']['mean']:.3f} ms，裁剪 {timing['pruned']['prefill']['mean']:.3f} ms，加速 {reductions['prefill_speedup']:.3f}×。",
        "",
        "## 结论边界",
        "",
        (
            "完整集已完成，但只覆盖 CV-Bench 的空间多选问答，不等价于全部具身任务或机器人闭环成功率。"
            if is_full else "该结果只用于判断冻结轻度裁剪策略在 CV-Bench 上是否值得保留，不能替代完整数据集评测。"
        ),
        "推理核心包括自定义视觉前向、多模态输入整理、Prefill、Decode及必要CPU调度，不含图片打开/解码和processor。阶段时间使用CUDA事件；MAC为主Transformer矩阵计算代理，不含patch embedding、merger、DeepStack投影与文本Token。",
        "每个样本每配置确定性推理一次、每配置预热一次；性能数字是本次顺序运行观测，尚未做交错多轮时延复测或GPU功耗测量。不能推导ZCU102端到端速度或能耗。",
        "",
    ])
    args.output_stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"scores": scores, "quality": quality, "reductions": reductions}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
