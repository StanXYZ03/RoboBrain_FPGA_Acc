#!/usr/bin/env python3
"""Re-evaluate the frozen L2@87.5% token-pruning baseline under the 2026-09-16 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import evaluate, load_jsonl, score_record


EXPECTED_HASHES = {
    "manifest": "9ae7aaf1d14cb206b5b924020ed0b9ed015c2483a72062fe5f6df15344560cd9",
    "full": "c8d763f33a25014bdc329d9748d1e570c22b3f78ef082e6b09a5322349e8114d",
    "l2": "119e67e0d322e6a835b901ab60c02614fe2899536d08d83e323d9db63f9ee9e4",
}

STAGE_KEYS = (
    "preprocess_seconds",
    "vision_encoder_seconds",
    "llm_prefill_seconds",
    "decoder_seconds",
    "timed_stage_seconds",
    "elapsed_seconds",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of empty input")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def descriptive(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped["combined"].extend(records)
    for record in records:
        grouped[str(record["task"])].append(record)
    result: dict[str, Any] = {}
    for task, task_records in sorted(grouped.items()):
        result[task] = {}
        for key in STAGE_KEYS:
            values = [float(record[key]) for record in task_records if record.get(key) is not None]
            if values:
                result[task][key] = descriptive(values)
    return result


def compact_quality(summary: dict[str, Any]) -> dict[str, float]:
    pointing = summary["tasks"]["pointing"]
    grounding = summary["tasks"]["grounding"]
    return {
        "pointing_format_success": pointing["parse_rate"],
        "grounding_format_success": grounding["parse_rate"],
        "format_success_macro": statistics.fmean(
            (pointing["parse_rate"], grounding["parse_rate"])
        ),
        "pck_0_1": pointing["pck_0_1"],
        "pck_0_2": pointing["pck_0_2"],
        "point_nme_bbox_diag_mean": pointing["point_nme_bbox_diag_mean"],
        "iou_mean": grounding["iou_mean"],
        "acc_0_5": grounding["acc_0_5"],
    }


def safe_ratio(candidate: float, baseline: float) -> float:
    if baseline == 0.0:
        raise ZeroDivisionError("baseline metric is zero")
    return candidate / baseline


def quality_retention(full: dict[str, float], l2: dict[str, float]) -> dict[str, float]:
    return {
        "pck_0_1": safe_ratio(l2["pck_0_1"], full["pck_0_1"]),
        "pck_0_2": safe_ratio(l2["pck_0_2"], full["pck_0_2"]),
        "point_nme_bbox_diag_mean": safe_ratio(
            full["point_nme_bbox_diag_mean"], l2["point_nme_bbox_diag_mean"]
        ),
        "iou_mean": safe_ratio(l2["iou_mean"], full["iou_mean"]),
        "acc_0_5": safe_ratio(l2["acc_0_5"], full["acc_0_5"]),
        "format_success_macro": safe_ratio(
            l2["format_success_macro"], full["format_success_macro"]
        ),
    }


def score_by_image(
    manifest: Path, prediction: Path
) -> dict[int, dict[str, dict[str, float]]]:
    samples = load_jsonl(manifest)
    predictions = {row["sample_id"]: row for row in load_jsonl(prediction)}
    result: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)
    for sample in samples:
        scored = score_record(sample, predictions[sample["sample_id"]])
        image_id = int(sample["coco_image_id"])
        if sample["task"] == "pointing":
            result[image_id]["pointing"] = {
                "pck_0_1": float(scored["pck_0_1"]),
                "pck_0_2": float(scored["pck_0_2"]),
                "point_nme_bbox_diag_mean": float(scored["point_nme_bbox_diag"]),
            }
        elif sample["task"] == "grounding":
            result[image_id]["grounding"] = {
                "iou_mean": float(scored["iou"]),
                "acc_0_5": float(scored["acc_0_5"]),
            }
    incomplete = [image_id for image_id, tasks in result.items() if len(tasks) != 2]
    if incomplete:
        raise ValueError(f"images without both tasks: {incomplete[:3]}")
    return result


def image_bootstrap(
    manifest: Path,
    full_path: Path,
    l2_path: Path,
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    full = score_by_image(manifest, full_path)
    l2 = score_by_image(manifest, l2_path)
    image_ids = sorted(full)
    if image_ids != sorted(l2) or len(image_ids) != 96:
        raise ValueError("expected the same 96 images for full and L2")

    metrics = {
        "pck_0_1": ("pointing", "pck_0_1", False),
        "pck_0_2": ("pointing", "pck_0_2", False),
        "point_nme_bbox_diag_mean": ("pointing", "point_nme_bbox_diag_mean", True),
        "iou_mean": ("grounding", "iou_mean", False),
        "acc_0_5": ("grounding", "acc_0_5", False),
    }
    generator = random.Random(seed)
    bootstrap_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        selected = [image_ids[generator.randrange(len(image_ids))] for _ in image_ids]
        ratios: dict[str, float] = {}
        for metric, (task, key, lower_is_better) in metrics.items():
            full_mean = statistics.fmean(full[i][task][key] for i in selected)
            l2_mean = statistics.fmean(l2[i][task][key] for i in selected)
            ratio = safe_ratio(full_mean, l2_mean) if lower_is_better else safe_ratio(l2_mean, full_mean)
            ratios[metric] = ratio
            bootstrap_values[metric].append(ratio)
        bootstrap_values["coco_spatial_diagnostic_mean_retention"].append(
            statistics.fmean((ratios["pck_0_1"], ratios["iou_mean"]))
        )
        bootstrap_values["coco_spatial_diagnostic_all_metrics_retention"].append(
            statistics.fmean(ratios.values())
        )

    result: dict[str, Any] = {
        "unit": "image_id_with_pointing_and_grounding_resampled_together",
        "image_count": len(image_ids),
        "iterations": iterations,
        "seed": seed,
        "confidence_intervals": {},
    }
    for metric, values in bootstrap_values.items():
        result["confidence_intervals"][metric] = {
            "bootstrap_mean": statistics.fmean(values),
            "ci95_low": percentile(values, 0.025),
            "ci95_high": percentile(values, 0.975),
        }
    return result


def aggregate_compute(records: list[dict[str, Any]]) -> dict[str, float]:
    def ratio(value: str, baseline: str) -> float:
        return sum(float(row[value]) for row in records) / sum(float(row[baseline]) for row in records)

    return {
        "token_retention": ratio("encoder_tokens_after_prune", "baseline_encoder_tokens"),
        "encoder_compute_retention": ratio("encoder_macs_proxy", "baseline_encoder_macs_proxy"),
        "llm_prefill_compute_retention": ratio("llm_prefill_macs_proxy", "baseline_llm_prefill_macs_proxy"),
    }


def speedup_summary(full: dict[str, Any], l2: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in full:
        result[task] = {}
        for stage in full[task]:
            result[task][stage] = {
                statistic: safe_ratio(full[task][stage][statistic], l2[task][stage][statistic])
                for statistic in ("mean", "p50", "p95")
            }
    return result


def f4(value: float) -> str:
    return f"{value:.4f}"


def build_markdown(report: dict[str, Any]) -> str:
    full = report["quality"]["full"]
    l2 = report["quality"]["l2"]
    retention = report["quality"]["retention"]
    ci = report["paired_image_bootstrap"]["confidence_intervals"]
    compute = report["compute_proxy"]["l2"]
    speed = report["timing"]["speedup_full_over_l2"]
    assessment = report["assessment"]
    rows = []
    labels = (
        ("pck_0_1", "Pointing PCK@0.1"),
        ("pck_0_2", "Pointing PCK@0.2"),
        ("point_nme_bbox_diag_mean", "Pointing NME（越低越好）"),
        ("iou_mean", "Grounding Mean IoU"),
        ("acc_0_5", "Grounding Acc@0.5"),
        ("format_success_macro", "格式成功率"),
    )
    for key, label in labels:
        rows.append(f"| {label} | {f4(full[key])} | {f4(l2[key])} | {f4(retention[key])} |")

    ci_rows = []
    for key, label in (
        ("pck_0_1", "PCK@0.1保持率"),
        ("iou_mean", "Mean IoU保持率"),
        ("coco_spatial_diagnostic_mean_retention", "COCO两项诊断平均保持率"),
    ):
        value = ci[key]
        ci_rows.append(
            f"| {label} | {f4(value['bootstrap_mean'])} | "
            f"[{f4(value['ci95_low'])}, {f4(value['ci95_high'])}] |"
        )

    timing_rows = []
    for task, label in (("pointing", "Pointing"), ("grounding", "Grounding"), ("combined", "合并")):
        full_elapsed = report["timing"]["full"][task]["elapsed_seconds"]
        l2_elapsed = report["timing"]["l2"][task]["elapsed_seconds"]
        timing_rows.append(
            f"| {label} | {full_elapsed['mean'] * 1000:.2f} | {l2_elapsed['mean'] * 1000:.2f} | "
            f"{speed[task]['elapsed_seconds']['mean']:.3f}x | "
            f"{speed[task]['elapsed_seconds']['p50']:.3f}x |"
        )

    return "\n".join(
        [
            "# L2@87.5% Token裁剪按文献通行方式复核",
            "",
            "日期：2026-09-16  ",
            "对象：layer-5 L2选择 + 3x3区域锚定，保留87.5%视觉group  ",
            f"结论：`{assessment['decision']}`",
            "",
            "## 这次复核回答什么",
            "",
            "本报告不重新选参，也不把已揭盲的COCO数据冒充正式测试集。它只用冻结前生成的96张图、",
            "192项空间任务原始输出，检查该方案是否仍可作为可信基线。",
            "原始文件哈希全部与2026-08-26锁定记录一致。",
            "",
            "论文主质量表应按RoboBrain2.5原论文分别报告CV-Bench、CrossPoint、RoboSpatial、",
            "RefSpatial和EmbSpatial的官方分数及算术平均；通用VLM基准另表报告。",
            "本地尚无这些公开基准结果，因此本报告不是论文主质量评价。",
            "",
            "## 冻结空间任务结果",
            "",
            "保持率定义为裁剪模型相对Full BF16；NME越低越好，因此该项使用Full/L2。",
            "",
            "| 指标 | Full | L2@87.5% | 保持率 |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            f"PCK@0.1保持率与Mean IoU保持率的诊断性平均 = "
            f"**{retention['coco_spatial_diagnostic_mean_retention']:.4f}**。该值只用于概括本地诊断，",
            "不作为新的综合评分标准。",
            "",
            "## 配对不确定性",
            "",
            "Bootstrap以图像为重采样单位，使同一图像的Pointing和Grounding始终成对出现；固定种子，10000次。",
            "",
            "| 指标 | Bootstrap均值 | 95%区间 |",
            "|---|---:|---:|",
            *ci_rows,
            "",
            "两项诊断平均保持率的区间包含1.0，说明数据与“轻微下降、无变化或轻微提高”都相容；",
            "由于没有预先注册非劣界限，不能据此宣称统计非劣或性能改善。",
            "",
            "## 计算量与实测时间",
            "",
            f"- 视觉Token保持率：{compute['token_retention']:.4f}（减少{(1-compute['token_retention'])*100:.1f}%）",
            f"- 视觉编码MAC代理保持率：{compute['encoder_compute_retention']:.4f}（减少{(1-compute['encoder_compute_retention'])*100:.1f}%）",
            f"- LLM Prefill MAC代理保持率：{compute['llm_prefill_compute_retention']:.4f}（减少{(1-compute['llm_prefill_compute_retention'])*100:.1f}%）",
            "",
            "下面是原始记录中的`elapsed_seconds`，不是新跑的基准；每个任务96张图、每张一次。",
            "旧实现从视觉前向开始计时，Processor与主机到GPU传输发生在计时前，因此应称为推理核心时间，",
            "不能冒充新标准定义的模型端到端时间。",
            "",
            "| 任务 | Full均值(ms) | L2均值(ms) | 均值加速比 | p50加速比 |",
            "|---|---:|---:|---:|---:|",
            *timing_rows,
            "",
            "A6000推理核心实测只提高约0.7%，没有形成有实质意义的端到端加速证据；",
            "原因是只减少12.5%视觉Token，而且Decode基本不受益。",
            "当前没有这一路线的ZCU102完整模型板级延迟或能耗数据。",
            "",
            "## 按文献通行方式的证据状态",
            "",
            f"- 冻结数据完整性：{'通过' if assessment['data_integrity_pass'] else '不通过'}。",
            "- 本地COCO诊断：各项点估计未下降，但没有建立正式非劣性。",
            "- RoboBrain2.5五个官方空间基准：未评估。",
            "- 通用VLM鲁棒性基准：未评估。",
            "- 正式模型端到端性能：未评估；旧计时缺少H2D。",
            "- ZCU102完整模型延迟、功耗与能效：未评估。",
            "",
            "因此L2@87.5%仍是目前最可信、可实现、质量风险最低的Token裁剪基线；",
            "但它的收益不足以单独支撑论文主加速方案。下一次正式评价应直接接入官方基准，",
            "并把它作为安全基线与更高压缩率或其他异构优化组合比较。",
            "",
            "## 复现",
            "",
            "```bash",
            "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python evaluation/recheck_frozen_token_pruning.py",
            "```",
            "",
            f"机器可读结果：`{report['output_json']}`",
        ]
    ) + "\n"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "datasets/dac_spatial_l2res_blind.jsonl")
    parser.add_argument(
        "--full",
        type=Path,
        default=root / "evaluation/results/l2_residual_blind_validation/shallow_r1000.jsonl",
    )
    parser.add_argument(
        "--l2",
        type=Path,
        default=root / "evaluation/results/l2_residual_blind_validation/shallow_r0875.jsonl",
    )
    parser.add_argument("--protocol", type=Path, default=root / "configs/evaluation_protocol.json")
    parser.add_argument(
        "--output-json", type=Path, default=root / "reports/2026-09-16-l2-0875-new-standard-recheck.json"
    )
    parser.add_argument(
        "--output-md", type=Path, default=root / "reports/2026-09-16-l2-0875-new-standard-recheck.md"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    paths = {"manifest": args.manifest, "full": args.full, "l2": args.l2}
    hashes = {name: sha256(path) for name, path in paths.items()}
    integrity = {name: hashes[name] == EXPECTED_HASHES[name] for name in paths}
    if not all(integrity.values()):
        raise ValueError(f"frozen evidence hash mismatch: {integrity}")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    full_scored, full_summary = evaluate(args.manifest, args.full)
    l2_scored, l2_summary = evaluate(args.manifest, args.l2)
    if len(full_scored) != 192 or len(l2_scored) != 192:
        raise ValueError("expected 192 predictions for each configuration")

    full_quality = compact_quality(full_summary)
    l2_quality = compact_quality(l2_summary)
    retention = quality_retention(full_quality, l2_quality)
    retention["coco_spatial_diagnostic_mean_retention"] = statistics.fmean(
        (retention["pck_0_1"], retention["iou_mean"])
    )
    retention["coco_spatial_diagnostic_all_metrics_retention"] = statistics.fmean(
        retention[key]
        for key in ("pck_0_1", "pck_0_2", "point_nme_bbox_diag_mean", "iou_mean", "acc_0_5")
    )

    full_records = load_jsonl(args.full)
    l2_records = load_jsonl(args.l2)
    full_timing = timing_summary(full_records)
    l2_timing = timing_summary(l2_records)
    speedups = speedup_summary(full_timing, l2_timing)
    bootstrap = image_bootstrap(
        args.manifest,
        args.full,
        args.l2,
        seed=args.seed,
        iterations=args.bootstrap_iterations,
    )

    diagnostic_point_estimates_no_degradation = all(
        retention[key] >= 1.0
        for key in ("pck_0_1", "pck_0_2", "point_nme_bbox_diag_mean", "iou_mean", "acc_0_5")
    )
    assessment = {
        "data_integrity_pass": all(integrity.values()),
        "coco_diagnostic_point_estimates_no_degradation": diagnostic_point_estimates_no_degradation,
        "formal_noninferiority_established": False,
        "robobrain_official_five_benchmark_results_available": False,
        "general_vlm_robustness_results_available": False,
        "custom_weighted_quality_score_used": False,
        "formal_model_end_to_end_performance_available": False,
        "zcu102_on_board_performance_available": False,
        "zcu102_energy_available": False,
        "decision": "RETAIN_AS_FROZEN_SAFE_BASELINE / NOT_MAIN_ACCELERATION_SCHEME",
    }
    report: dict[str, Any] = {
        "date": "2026-09-16",
        "scheme": {
            "name": "layer-5 L2 + 3x3 regional anchor",
            "requested_retention": 0.875,
            "selection": "L2 top-k with one regional anchor per non-empty 3x3 cell",
            "parameter_tuning_in_this_recheck": False,
        },
        "protocol": {
            "path": str(args.protocol.relative_to(root)),
            "sha256": sha256(args.protocol),
            "version": protocol["version"],
        },
        "evidence_scope": {
            "dataset": "COCO 2017 validation-derived frozen spatial set",
            "distinct_images": 96,
            "tasks": ["pointing", "grounding"],
            "samples_per_configuration": 192,
            "role": "revealed frozen reproduction and COCO spatial diagnostic only",
            "not_valid_for": [
                "new hyperparameter selection",
                "RoboBrain2.5 official five-benchmark quality table",
                "general VLM robustness table",
                "a newly proposed universal quality score",
            ],
        },
        "integrity": {"actual_sha256": hashes, "expected_sha256": EXPECTED_HASHES, "matches": integrity},
        "quality": {"full": full_quality, "l2": l2_quality, "retention": retention},
        "paired_image_bootstrap": bootstrap,
        "compute_proxy": {
            "full": aggregate_compute(full_records),
            "l2": aggregate_compute(l2_records),
        },
        "timing": {
            "scope": (
                "archived A6000 single-run-per-image inference-core records; elapsed_seconds starts "
                "after processor and host-to-device transfer and therefore is not the formal model-E2E boundary"
            ),
            "full": full_timing,
            "l2": l2_timing,
            "speedup_full_over_l2": speedups,
        },
        "assessment": assessment,
        "limitations": [
            "The frozen COCO set is already revealed and cannot validate a new method.",
            "The official RoboBrain2.5 spatial suite and general VLM robustness benchmarks have not been run.",
            "Latency records have one measured run per distinct image, not the new protocol's repeated timing design.",
            "The archived elapsed_seconds excludes processor and host-to-device transfer, so it cannot close the formal model-E2E gate.",
            "No full-model ZCU102 latency or board-energy measurement exists for this pruning scheme.",
        ],
        "output_json": str(args.output_json.relative_to(root)),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(build_markdown(report), encoding="utf-8")
    print(json.dumps(assessment, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
