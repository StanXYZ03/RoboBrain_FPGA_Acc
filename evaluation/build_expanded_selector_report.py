#!/usr/bin/env python3
"""Build the expanded-set comparison of L2 and spatial-residual selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate, load_jsonl, score_record


PS_VISION_SECONDS = 625.425
PS_LLM_PREFILL_SECONDS = 1003.155
PS_FIXED_SECONDS = 65.498
PS_BASELINE_SECONDS = 1695.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(records: list[dict[str, Any]], value: str, baseline: str) -> float:
    return sum(float(record[value]) for record in records) / sum(
        float(record[baseline]) for record in records
    )


def median(records: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(record[key]) for record in records)


def aggregate(label: str, path: Path, manifest: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    if len(records) != 192 or len({record["sample_id"] for record in records}) != 192:
        raise ValueError(f"{path}: expected 192 unique predictions")
    _, metrics = evaluate(manifest, path)
    encoder = ratio(records, "encoder_macs_proxy", "baseline_encoder_macs_proxy")
    llm = ratio(records, "llm_prefill_macs_proxy", "baseline_llm_prefill_macs_proxy")
    projected = PS_VISION_SECONDS * encoder + PS_LLM_PREFILL_SECONDS * llm + PS_FIXED_SECONDS
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "prediction_count": len(records),
        "actual_token_retention": ratio(
            records, "encoder_tokens_after_prune", "baseline_encoder_tokens"
        ),
        "encoder_compute_retention": encoder,
        "llm_prefill_compute_retention": llm,
        "a6000_median_seconds": {
            key: median(records, key)
            for key in (
                "vision_encoder_seconds",
                "llm_prefill_seconds",
                "decoder_seconds",
                "elapsed_seconds",
            )
        },
        "zcu102_projection": {
            "seconds_per_image": projected,
            "speedup": PS_BASELINE_SECONDS / projected,
            "hz": 1.0 / projected,
        },
        "metrics": metrics,
    }


def metric_values(
    manifest: Path, prediction: Path
) -> dict[str, dict[str, list[float]]]:
    samples = load_jsonl(manifest)
    predictions = {record["sample_id"]: record for record in load_jsonl(prediction)}
    values = {
        "pointing": {"pck_0_1": [], "pck_0_2": []},
        "grounding": {"iou_mean": [], "acc_0_5": []},
    }
    for sample in samples:
        scored = score_record(sample, predictions[sample["sample_id"]])
        if sample["task"] == "pointing":
            values["pointing"]["pck_0_1"].append(float(scored["pck_0_1"]))
            values["pointing"]["pck_0_2"].append(float(scored["pck_0_2"]))
        else:
            values["grounding"]["iou_mean"].append(float(scored["iou"]))
            values["grounding"]["acc_0_5"].append(float(scored["acc_0_5"]))
    return values


def paired_bootstrap(
    left: list[float], right: list[float], *, seed: int, iterations: int
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired vectors must have the same nonzero length")
    deltas = [lvalue - rvalue for lvalue, rvalue in zip(left, right)]
    generator = random.Random(seed)
    samples = []
    for _ in range(iterations):
        samples.append(
            sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas)
        )
    samples.sort()
    return {
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": samples[int(0.025 * (iterations - 1))],
        "ci95_high": samples[int(0.975 * (iterations - 1))],
        "paired_samples": len(deltas),
        "iterations": iterations,
    }


def paired_comparison(
    manifest: Path,
    candidate: Path,
    reference: Path,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    candidate_values = metric_values(manifest, candidate)
    reference_values = metric_values(manifest, reference)
    result: dict[str, Any] = {}
    for task, metrics in candidate_values.items():
        result[task] = {}
        for offset, (metric, values) in enumerate(metrics.items()):
            result[task][metric] = paired_bootstrap(
                values,
                reference_values[task][metric],
                seed=seed + offset + (0 if task == "pointing" else 100),
                iterations=iterations,
            )
    return result


def result_row(result: dict[str, Any]) -> str:
    point = result["metrics"]["tasks"]["pointing"]
    ground = result["metrics"]["tasks"]["grounding"]
    return "| " + " | ".join(
        (
            result["label"],
            f"{result['actual_token_retention']:.3f}",
            f"{result['encoder_compute_retention']:.3f}",
            f"{result['llm_prefill_compute_retention']:.3f}",
            f"{point['pck_0_1']:.4f}",
            f"{point['pck_0_2']:.4f}",
            f"{ground['iou_mean']:.4f}",
            f"{ground['acc_0_5']:.4f}",
            f"{result['zcu102_projection']['seconds_per_image']:.1f}",
            f"{result['zcu102_projection']['speedup']:.2f}×",
        )
    ) + " |"


def ci_text(value: dict[str, float]) -> str:
    return f"{value['mean_delta']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    specs = (
        ("full@1.000", "shallow_r1000.jsonl"),
        ("L2+anchor@0.875", "shallow_r0875.jsonl"),
        ("residual+quota@0.875", "residual_r0875.jsonl"),
        ("L2+anchor@0.750", "shallow_r0750.jsonl"),
        ("residual+quota@0.750", "residual_r0750.jsonl"),
    )
    results = [aggregate(label, args.input_dir / filename, args.manifest) for label, filename in specs]
    comparisons = {
        "retention_0875_residual_minus_l2": paired_comparison(
            args.manifest,
            args.input_dir / "residual_r0875.jsonl",
            args.input_dir / "shallow_r0875.jsonl",
            seed=20260825,
            iterations=args.bootstrap_iterations,
        ),
        "retention_0750_residual_minus_l2": paired_comparison(
            args.manifest,
            args.input_dir / "residual_r0750.jsonl",
            args.input_dir / "shallow_r0750.jsonl",
            seed=20260826,
            iterations=args.bootstrap_iterations,
        ),
    }
    expanded = load_jsonl(args.manifest)
    smoke = load_jsonl(args.smoke_manifest)
    expanded_images = {int(record["coco_image_id"]) for record in expanded}
    smoke_images = {int(record["coco_image_id"]) for record in smoke}
    overlap = sorted(expanded_images & smoke_images)
    if overlap:
        raise ValueError(f"expanded set overlaps smoke set: {overlap[:3]}")

    comparison_0750 = comparisons["retention_0750_residual_minus_l2"]
    deltas_0750 = [
        comparison_0750["pointing"]["pck_0_1"]["mean_delta"],
        comparison_0750["pointing"]["pck_0_2"]["mean_delta"],
        comparison_0750["grounding"]["iou_mean"]["mean_delta"],
        comparison_0750["grounding"]["acc_0_5"]["mean_delta"],
    ]
    strict_dominance = all(delta >= 0 for delta in deltas_0750) and any(
        delta > 0 for delta in deltas_0750
    )
    resolved_improvement = any(
        metric["ci95_low"] > 0
        for task in comparison_0750.values()
        for metric in task.values()
    )
    gate = {
        "expanded_data_gate_pass": True,
        "coordinate_preservation_signal": statistics.fmean(deltas_0750) > 0,
        "strict_metric_dominance": strict_dominance,
        "bootstrap_resolved_improvement": resolved_improvement,
        "algorithm_research_continue": True,
        "pl_integration_pass": False,
        "reason": (
            "残差配额在 75% 预算的点估计提高 Pointing 与 Mean IoU，但 Acc@0.5 下降，"
            "且 96 样本的成对置信区间仍不足以确认稳定优势；继续算法消融，暂不固化硬件"
        ),
    }
    summary = {
        "date": "2026-08-25",
        "scope": "96 held-out COCO images; 96 Pointing + 96 Grounding; 960 inferences",
        "dataset": {
            "manifest": str(args.manifest),
            "sha256": sha256(args.manifest),
            "images": len(expanded_images),
            "samples": len(expanded),
            "smoke_image_overlap": len(overlap),
        },
        "method": {
            "name": "spatial residual + regional quota",
            "score": "0.75*four-neighbor feature residual + 0.25*feature RMS energy",
            "selection_unit": "Qwen3-VL 2x2 merger group",
            "hardware_mapping": "four-neighbor stencil/line buffer, region sums, bounded regional top-k",
            "task_specific_configuration": False,
        },
        "gate": gate,
        "results": results,
        "paired_bootstrap": comparisons,
        "projection_status": "first-order workload projection, not board measurement",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c875 = comparisons["retention_0875_residual_minus_l2"]
    c750 = comparisons["retention_0750_residual_minus_l2"]
    lines = [
        "# 扩大集：坐标敏感结构化选择器对照",
        "",
        "日期：2026-08-25  ",
        "状态：`96_IMAGES / 192_TASKS / 960_INFERENCES / ALGORITHM_CONTINUE / PL_HOLD`",
        "",
        "## 1. 数据与方法",
        "",
        "从 COCO val2017 重新抽取 96 张图片，每张构造同一目标的 Pointing 与 Grounding，",
        "显式排除原 smoke 集 32 张，图片重叠为 0。该集合扩大了统计样本，但仍是 COCO 派生",
        "空间定位集，不等同于论文级具身 affordance/trajectory benchmark。",
        "",
        "新选择器在视觉第 5 层按 Qwen3-VL 2×2 merger group 计算四邻域特征残差，",
        "与小权重 RMS 能量融合；每个 3×3 分区至少保留配额，剩余预算按区域分数质量分配。",
        "Pointing/Grounding 使用完全相同的评分、预算与硬件数据通路。",
        "",
        "## 2. 完整结果",
        "",
        "| 配置 | Token | Encoder MAC | LLM MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | PS 投影/s | 投影加速 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(result_row(result) for result in results)
    lines.extend(
        (
            "",
            "## 3. 相对同预算 L2 的成对差值",
            "",
            "格式为 `均值差 [95% paired bootstrap CI]`，每项任务 n=96，10,000 次重采样。",
            "",
            "| 预算 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |",
            "|---:|---:|---:|---:|---:|",
            "| 0.875 | "
            + " | ".join(
                (
                    ci_text(c875["pointing"]["pck_0_1"]),
                    ci_text(c875["pointing"]["pck_0_2"]),
                    ci_text(c875["grounding"]["iou_mean"]),
                    ci_text(c875["grounding"]["acc_0_5"]),
                )
            )
            + " |",
            "| 0.750 | "
            + " | ".join(
                (
                    ci_text(c750["pointing"]["pck_0_1"]),
                    ci_text(c750["pointing"]["pck_0_2"]),
                    ci_text(c750["grounding"]["iou_mean"]),
                    ci_text(c750["grounding"]["acc_0_5"]),
                )
            )
            + " |",
            "",
            "## 4. 门控结论",
            "",
            gate["reason"] + "。",
            "75% 预算下，残差配额相对 L2 恢复了 Pointing 指标并提高 Mean IoU，说明局部空间",
            "残差比单纯特征范数更贴近坐标任务；但 Acc@0.5 的反向变化证明当前固定融合权重",
            "仍未形成统一 Pareto 支配，也不能称为最终创新。",
            "",
            "下一步只做软件消融：在开发子集扫描残差/能量权重与区域配额，在隔离评测子集",
            "确认后再决定是否形成可编程 PL selector。当前不写 RTL/HLS、不生成 bitstream。",
            "",
            "## 5. 复现",
            "",
            f"- 数据清单：`{args.manifest}`",
            f"- 推理结果：`{args.input_dir}`",
            f"- 机器汇总：`{args.summary}`",
            "- 选择器：`evaluation/shallow_block_pruning.py`",
            "- 推理器：`evaluation/run_hf_shallow_block_pruning.py`",
            "- 汇总器：`evaluation/build_expanded_selector_report.py`",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
