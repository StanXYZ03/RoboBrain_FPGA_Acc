#!/usr/bin/env python3
"""Build the revealed-development ROI/L2 per-sample matched-budget report."""

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

from evaluation.metrics import load_jsonl, score_record
from evaluation.token_reduction import qwen3_vl_compute_proxy


METRICS = {
    "grounding": ("iou", "acc_0_5"),
    "pointing": ("pck_0_1", "pck_0_2"),
}


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


def paired_bootstrap(
    deltas: list[float], *, seed: int, iterations: int
) -> dict[str, float | int]:
    if not deltas:
        raise ValueError("paired bootstrap requires nonempty differences")
    generator = random.Random(seed)
    n = len(deltas)
    means = [
        statistics.fmean(deltas[generator.randrange(n)] for _ in range(n))
        for _ in range(iterations)
    ]
    return {
        "n": n,
        "mean_delta": statistics.fmean(deltas),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
        "iterations": iterations,
    }


def deterministic_strategy_records(
    path: Path, strategy: str, expected_repeats: int
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(path):
        if str(record.get("strategy")) == strategy:
            grouped[str(record["sample_id"])].append(record)
    errors: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    for sample_id, records in grouped.items():
        if len(records) != expected_repeats:
            errors.append(
                f"{strategy}:{sample_id}: expected {expected_repeats} repeats, got {len(records)}"
            )
        if len({str(record["output_text"]) for record in records}) != 1:
            errors.append(f"{strategy}:{sample_id}: nondeterministic output")
        selected[sample_id] = min(records, key=lambda item: int(item.get("repeat", 0)))
    return selected, errors


def unique_records(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records = load_jsonl(path)
    selected: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in selected:
            errors.append(f"{path}:{sample_id}: duplicate prediction")
        selected[sample_id] = record
    return selected, errors


def score_methods(
    manifest: dict[str, dict[str, Any]],
    methods: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    scored: dict[tuple[str, str], dict[str, float]] = {}
    summaries: dict[str, dict[str, float]] = {}
    for method, records in methods.items():
        for sample_id, record in records.items():
            sample = manifest[sample_id]
            result = score_record(sample, record)
            if sample["task"] == "grounding":
                scored[(method, sample_id)] = {
                    "primary": float(result["iou"]),
                    "secondary": float(result["acc_0_5"]),
                    "parse": float(result["parse_success"]),
                }
            else:
                scored[(method, sample_id)] = {
                    "primary": float(result["pck_0_1"]),
                    "secondary": float(result["pck_0_2"]),
                    "parse": float(result["parse_success"]),
                }
        for task in ("grounding", "pointing"):
            ids = sorted(
                sample_id
                for sample_id, sample in manifest.items()
                if sample["task"] == task
            )
            values = [scored[(method, sample_id)] for sample_id in ids]
            summaries[f"{method}:{task}"] = {
                "n": len(values),
                "primary_mean": statistics.fmean(value["primary"] for value in values),
                "secondary_mean": statistics.fmean(value["secondary"] for value in values),
                "parse_rate": statistics.fmean(value["parse"] for value in values),
            }
    return summaries, scored


def method_tokens(
    method: str, records: dict[str, dict[str, Any]]
) -> tuple[int, int, float]:
    if method in {"native_dense", "roi384"}:
        kept = sum(int(record["final_context_visual_tokens"]) for record in records.values())
        if method == "native_dense":
            full = kept
        else:
            full = sum(
                qwen3_vl_compute_proxy(int(record["global_encoder_tokens"])).llm_visual_tokens
                if int(record.get("global_encoder_tokens", 0)) > 0
                else int(record["final_context_visual_tokens"])
                for record in records.values()
            )
            # ROI-only records do not carry their original dense grid. The
            # caller overwrites this denominator from native_dense below.
    else:
        kept = sum(int(record["llm_visual_tokens"]) for record in records.values())
        full = sum(int(record["baseline_llm_visual_tokens"]) for record in records.values())
    return kept, full, kept / full


def comparison(
    candidate: str,
    reference: str,
    manifest: dict[str, dict[str, Any]],
    scored: dict[tuple[str, str], dict[str, float]],
    *,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task_index, task in enumerate(("grounding", "pointing")):
        ids = sorted(
            sample_id for sample_id, sample in manifest.items() if sample["task"] == task
        )
        output[task] = {}
        for metric_index, metric in enumerate(("primary", "secondary", "parse")):
            deltas = [
                scored[(candidate, sample_id)][metric]
                - scored[(reference, sample_id)][metric]
                for sample_id in ids
            ]
            output[task][metric] = paired_bootstrap(
                deltas,
                seed=seed + 100 * task_index + metric_index,
                iterations=iterations,
            )
    return output


def ci(value: dict[str, Any]) -> str:
    return (
        f"{value['mean_delta']:+.4f} "
        f"[{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-results", type=Path, required=True)
    parser.add_argument("--custom-dense", type=Path, required=True)
    parser.add_argument("--matched-l2", type=Path, required=True)
    parser.add_argument("--matched-geometric", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--expected-native-repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--noninferiority-margin", type=float, default=-0.02)
    args = parser.parse_args()

    manifest_rows = load_jsonl(args.manifest)
    manifest = {str(row["sample_id"]): row for row in manifest_rows}
    native_dense, errors = deterministic_strategy_records(
        args.native_results, "dense", args.expected_native_repeats
    )
    roi384, more_errors = deterministic_strategy_records(
        args.native_results, "roi_only", args.expected_native_repeats
    )
    errors.extend(more_errors)
    custom_dense, more_errors = unique_records(args.custom_dense)
    errors.extend(more_errors)
    matched_l2, more_errors = unique_records(args.matched_l2)
    errors.extend(more_errors)
    methods = {
        "native_dense": native_dense,
        "custom_dense": custom_dense,
        "roi384": roi384,
        "l2_matched_roi384": matched_l2,
    }
    if args.matched_geometric is not None:
        matched_geometric, more_errors = unique_records(args.matched_geometric)
        errors.extend(more_errors)
        methods["geometric_matched_roi384"] = matched_geometric
    expected_ids = set(manifest)
    for method, records in methods.items():
        if set(records) != expected_ids:
            errors.append(
                f"{method}: sample-ID mismatch, missing={len(expected_ids - set(records))}, "
                f"extra={len(set(records) - expected_ids)}"
            )
    if errors:
        raise ValueError("; ".join(errors[:10]))

    summaries, scored = score_methods(manifest, methods)
    dense_token_total = sum(
        int(record["final_context_visual_tokens"]) for record in native_dense.values()
    )
    token_summary: dict[str, Any] = {}
    for method, records in methods.items():
        kept, full, ratio = method_tokens(method, records)
        if method == "roi384":
            full = dense_token_total
            ratio = kept / full
        token_summary[method] = {
            "kept_visual_tokens": kept,
            "dense_visual_tokens": full,
            "ratio_to_dense": ratio,
        }

    target_mismatches = []
    capped = []
    for sample_id, l2_record in matched_l2.items():
        roi_tokens = int(roi384[sample_id]["final_context_visual_tokens"])
        l2_tokens = int(l2_record["llm_visual_tokens"])
        dense_tokens = int(l2_record["baseline_llm_visual_tokens"])
        expected = min(roi_tokens, dense_tokens)
        if l2_tokens != expected:
            target_mismatches.append(sample_id)
        if roi_tokens > dense_tokens:
            capped.append(sample_id)

    exact_dense_output_matches = sum(
        str(custom_dense[sample_id]["output_text"])
        == str(native_dense[sample_id]["output_text"])
        for sample_id in sorted(expected_ids)
    )
    comparisons = {
        "custom_dense_minus_native_dense": comparison(
            "custom_dense", "native_dense", manifest, scored,
            seed=args.seed, iterations=args.bootstrap,
        ),
        "roi384_minus_native_dense": comparison(
            "roi384", "native_dense", manifest, scored,
            seed=args.seed + 1000, iterations=args.bootstrap,
        ),
        "l2_matched_minus_native_dense": comparison(
            "l2_matched_roi384", "native_dense", manifest, scored,
            seed=args.seed + 2000, iterations=args.bootstrap,
        ),
        "l2_matched_minus_roi384": comparison(
            "l2_matched_roi384", "roi384", manifest, scored,
            seed=args.seed + 3000, iterations=args.bootstrap,
        ),
    }
    if "geometric_matched_roi384" in methods:
        comparisons.update({
            "geometric_matched_minus_native_dense": comparison(
                "geometric_matched_roi384", "native_dense", manifest, scored,
                seed=args.seed + 4000, iterations=args.bootstrap,
            ),
            "geometric_matched_minus_roi384": comparison(
                "geometric_matched_roi384", "roi384", manifest, scored,
                seed=args.seed + 5000, iterations=args.bootstrap,
            ),
            "geometric_matched_minus_l2_matched": comparison(
                "geometric_matched_roi384", "l2_matched_roi384", manifest, scored,
                seed=args.seed + 6000, iterations=args.bootstrap,
            ),
        })
    l2_noninferior = {
        task: comparisons["l2_matched_minus_native_dense"][task]["primary"]["ci95_low"]
        >= args.noninferiority_margin
        for task in ("grounding", "pointing")
    }
    l2_encoder_ratio = sum(
        int(record["encoder_macs_proxy"]) for record in matched_l2.values()
    ) / sum(int(record["baseline_encoder_macs_proxy"]) for record in matched_l2.values())
    roi_encoder_ratio = sum(
        qwen3_vl_compute_proxy(int(record["encoder_tokens_total"])).encoder_macs
        for record in roi384.values()
    ) / sum(
        qwen3_vl_compute_proxy(int(record["encoder_tokens_total"])).encoder_macs
        for record in native_dense.values()
    )
    encoder_ratios = {
        "roi384": roi_encoder_ratio,
        "l2_matched_roi384": l2_encoder_ratio,
    }
    geometric_noninferior = None
    if "geometric_matched_roi384" in methods:
        matched_geometric = methods["geometric_matched_roi384"]
        encoder_ratios["geometric_matched_roi384"] = sum(
            int(record["encoder_macs_proxy"]) for record in matched_geometric.values()
        ) / sum(
            int(record["baseline_encoder_macs_proxy"])
            for record in matched_geometric.values()
        )
        geometric_noninferior = {
            task: comparisons["geometric_matched_minus_native_dense"][task]["primary"]["ci95_low"]
            >= args.noninferiority_margin
            for task in ("grounding", "pointing")
        }
    report = {
        "status": "ROI_L2_MATCHED_BUDGET192_COMPLETE",
        "evidence_level": "revealed 192-image development comparison; not blind validation",
        "files": {
            "manifest": str(args.manifest),
            "native_results": str(args.native_results),
            "custom_dense": str(args.custom_dense),
            "matched_l2": str(args.matched_l2),
            "matched_geometric": (
                str(args.matched_geometric) if args.matched_geometric is not None else None
            ),
        },
        "sha256": {
            "manifest": sha256(args.manifest),
            "native_results": sha256(args.native_results),
            "custom_dense": sha256(args.custom_dense),
            "matched_l2": sha256(args.matched_l2),
            "matched_geometric": (
                sha256(args.matched_geometric)
                if args.matched_geometric is not None
                else None
            ),
        },
        "records": len(manifest),
        "unique_images": len({int(row["coco_image_id"]) for row in manifest_rows}),
        "token_matching": {
            "mismatch_count": len(target_mismatches),
            "mismatch_sample_ids": target_mismatches,
            "capped_roi_targets_count": len(capped),
            "capped_roi_target_sample_ids": capped,
        },
        "custom_dense_native_output_audit": {
            "exact_matches": exact_dense_output_matches,
            "total": len(expected_ids),
            "rate": exact_dense_output_matches / len(expected_ids),
        },
        "token_summary": token_summary,
        "encoder_compute_proxy_ratio_to_dense": encoder_ratios,
        "metrics": summaries,
        "paired_bootstrap": comparisons,
        "gate": {
            "primary_noninferiority_margin": args.noninferiority_margin,
            "l2_primary_noninferior_by_ci": l2_noninferior,
            "joint_noninferiority_pass": all(l2_noninferior.values()),
            "geometric_primary_noninferior_by_ci": geometric_noninferior,
            "geometric_joint_noninferiority_pass": (
                all(geometric_noninferior.values())
                if geometric_noninferior is not None
                else None
            ),
            "test_split_accessed": False,
            "calibration_split_accessed": False,
        },
        "limitations": [
            "ROI-384 uses the revealed ground-truth box and is not deployable.",
            "L2 selection occurs after vision layer 5, so matched final tokens do not mean matched total vision compute.",
            "Latency from different collection runs is not compared.",
            "Development evidence cannot establish held-out generalization."
        ]
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# ROI与L2逐样本同预算开发对照",
        "",
        "状态：`ROI_L2_MATCHED_BUDGET192_COMPLETE`。本报告使用192张已揭示开发图像，校准集和测试集未访问。",
        "",
        "## 预算审计",
        "",
        f"L2与ROI-384逐样本Token不一致数：`{len(target_mismatches)}`；ROI目标超过Dense而封顶的任务数：`{len(capped)}`。",
        f"自定义Dense与原生Dense文本逐字一致：`{exact_dense_output_matches}/{len(expected_ids)}`。",
        "",
        "| 方法 | 最终视觉Token比 | Vision Encoder计算代理比 |",
        "|---|---:|---:|",
        f"| 原生Dense | {token_summary['native_dense']['ratio_to_dense']:.3f} | 1.000 |",
        f"| 真值ROI-384 | {token_summary['roi384']['ratio_to_dense']:.3f} | {roi_encoder_ratio:.3f} |",
        f"| L2逐样本匹配 | {token_summary['l2_matched_roi384']['ratio_to_dense']:.3f} | {l2_encoder_ratio:.3f} |",
    ]
    if "geometric_matched_roi384" in methods:
        lines.append(
            f"| 几何Oracle逐样本匹配 | "
            f"{token_summary['geometric_matched_roi384']['ratio_to_dense']:.3f} | "
            f"{encoder_ratios['geometric_matched_roi384']:.3f} |"
        )
    lines.extend([
        "",
        "## 真实标注质量",
        "",
        "| 方法 | 任务 | 主指标 | 次指标 | 格式率 |",
        "|---|---|---:|---:|---:|",
    ])
    names = {
        "native_dense": "原生Dense",
        "custom_dense": "自定义Dense",
        "roi384": "真值ROI-384",
        "l2_matched_roi384": "L2逐样本匹配",
        "geometric_matched_roi384": "几何Oracle逐样本匹配",
    }
    for method in names:
        if f"{method}:grounding" not in summaries:
            continue
        for task in ("grounding", "pointing"):
            item = summaries[f"{method}:{task}"]
            lines.append(
                f"| {names[method]} | {task} | {item['primary_mean']:.4f} | "
                f"{item['secondary_mean']:.4f} | {item['parse_rate']:.4f} |"
            )
    lines.extend([
        "",
        "## 成对差值",
        "",
        "差值均为左侧方法减右侧方法，格式为均值差 `[95% bootstrap CI]`。",
        "",
        "| 对照 | 任务 | 主指标差 | 次指标差 |",
        "|---|---|---:|---:|",
    ])
    comparison_names = {
        "custom_dense_minus_native_dense": "自定义Dense - 原生Dense",
        "roi384_minus_native_dense": "ROI-384 - 原生Dense",
        "l2_matched_minus_native_dense": "L2匹配 - 原生Dense",
        "l2_matched_minus_roi384": "L2匹配 - ROI-384",
        "geometric_matched_minus_native_dense": "几何Oracle匹配 - 原生Dense",
        "geometric_matched_minus_roi384": "几何Oracle匹配 - ROI-384",
        "geometric_matched_minus_l2_matched": "几何Oracle匹配 - L2匹配",
    }
    for key, label in comparison_names.items():
        if key not in comparisons:
            continue
        for task in ("grounding", "pointing"):
            item = comparisons[key][task]
            lines.append(
                f"| {label} | {task} | {ci(item['primary'])} | {ci(item['secondary'])} |"
            )
    lines.extend([
        "",
        "## 判定边界",
        "",
        f"预设主指标非劣界限为`{args.noninferiority_margin:+.2f}`；L2对Dense的Grounding/Pointing CI门控分别为"
        f"`{l2_noninferior['grounding']}`/`{l2_noninferior['pointing']}`，联合通过=`{all(l2_noninferior.values())}`。",
        "",
        "ROI读取真值框，只是表示诊断；L2在第5层后选择，前6个视觉块仍处理完整网格。因此本报告可以区分局部裁图与全局Token选择的质量差异，但不能把最终Token相同写成总计算相同，也不进行跨运行延迟比较。",
    ])
    if geometric_noninferior is not None:
        lines.insert(
            lines.index("" , lines.index("## 判定边界") + 2) + 1,
            f"几何Oracle对Dense的Grounding/Pointing CI门控分别为"
            f"`{geometric_noninferior['grounding']}`/`{geometric_noninferior['pointing']}`，"
            f"联合通过=`{all(geometric_noninferior.values())}`。",
        )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gate": report["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
