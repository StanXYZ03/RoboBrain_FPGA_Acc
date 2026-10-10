#!/usr/bin/env python3
"""Summarize the complete VSR zero-shot development budget curve."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.vsr_zeroshot_dev_contract import (
    ARMS, ARM_DENSE, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED, EVAL_MANIFEST,
    OUTPUT_DIR, SAMPLE_COUNT, canonical_record_sha256, load_eval_manifest,
    output_path, sha256_file, verify_preregistration,
)

ROOT = Path(__file__).resolve().parents[1]
INFEASIBLE_ARM = "v2_edge576_b1_k192"
MEASURED_ARMS = tuple(arm for arm in ARMS if arm != INFEASIBLE_ARM)
REPORT_JSON = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.json"
REPORT_CSV = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.csv"
REPORT_MD = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.md"


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty values")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def describe(values: Sequence[float]) -> dict[str, float]:
    data = [float(value) for value in values]
    return {
        "mean": statistics.fmean(data), "median": statistics.median(data),
        "min": min(data), "max": max(data),
        "std": statistics.stdev(data) if len(data) > 1 else 0.0,
        "p95": percentile(data, 0.95),
    }


def load_arm(arm: str, expected_ids: set[str]) -> list[dict[str, Any]]:
    path = output_path(arm)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row.get("sample_id")) for row in rows]
    if len(rows) != SAMPLE_COUNT or len(set(ids)) != SAMPLE_COUNT or set(ids) != expected_ids:
        raise ValueError(f"{arm}: incomplete or mismatched sample set")
    for row in rows:
        if row.get("arm") != arm or row.get("record_sha256") != canonical_record_sha256(row):
            raise ValueError(f"{arm}: record identity/hash mismatch")
    return rows


def paired_bootstrap(left: Mapping[str, bool], right: Mapping[str, bool], repetitions: int = BOOTSTRAP_REPETITIONS, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    ids = sorted(left)
    if set(ids) != set(right):
        raise ValueError("bootstrap sample sets differ")
    observed = statistics.fmean(float(left[sid]) - float(right[sid]) for sid in ids)
    rng = random.Random(seed)
    deltas = []
    for _ in range(repetitions):
        sampled = [ids[rng.randrange(len(ids))] for _ in range(len(ids))]
        deltas.append(statistics.fmean(float(left[sid]) - float(right[sid]) for sid in sampled))
    return {"observed_difference": observed, "ci_95_lower": percentile(deltas, 0.025), "ci_95_upper": percentile(deltas, 0.975), "seed": seed, "repetitions": repetitions}


def stage_values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float] | None:
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def arm_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relation: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        relation[str(row["relation"])].append(bool(row["correct"]))
    timing_fields = (
        "runner_wall_seconds", "model_inference_seconds", "elapsed_seconds",
        "vision_total_seconds", "vision_encoder_seconds", "llm_prefill_seconds",
        "decoder_seconds", "selector_cpu_roundtrip_seconds", "preprocess_seconds",
    )
    timing = {}
    for field in timing_fields:
        values = stage_values(rows, field)
        if values is not None:
            timing[field] = describe(values)
    return {
        "sample_count": len(rows),
        "correct_count": sum(bool(row["correct"]) for row in rows),
        "accuracy": statistics.fmean(bool(row["correct"]) for row in rows),
        "parse_success_count": sum(bool(row["parse_success"]) for row in rows),
        "parse_success_rate": statistics.fmean(bool(row["parse_success"]) for row in rows),
        "mean_full_visual_tokens": statistics.fmean(int(row["full_llm_visual_tokens"]) for row in rows),
        "mean_output_visual_tokens": statistics.fmean(int(row["output_group_count"]) for row in rows),
        "mean_visual_token_retention": statistics.fmean(int(row["output_group_count"]) / int(row["full_llm_visual_tokens"]) for row in rows),
        "identity_bypass_count": sum(bool(row["identity_bypass"]) for row in rows),
        "mean_output_tokens": statistics.fmean(int(row.get("output_tokens", 0)) for row in rows),
        "peak_cuda_bytes_max": max(int(row.get("peak_cuda_bytes", 0)) for row in rows),
        "accuracy_by_relation": {
            key: {"count": len(values), "accuracy": statistics.fmean(values)}
            for key, values in sorted(relation.items())
        },
        "timing_seconds": timing,
    }


def summarize(repetitions: int = BOOTSTRAP_REPETITIONS) -> dict[str, Any]:
    config = verify_preregistration()
    samples = load_eval_manifest(config["pins"]["eval_manifest"]["sha256"])
    expected_ids = {str(row["sample_id"]) for row in samples}
    by_arm = {arm: load_arm(arm, expected_ids) for arm in MEASURED_ARMS}
    metrics = {arm: arm_metrics(rows) for arm, rows in by_arm.items()}
    dense_accuracy = metrics[ARM_DENSE]["accuracy"]
    dense_wall = metrics[ARM_DENSE]["timing_seconds"]["runner_wall_seconds"]["mean"]
    dense_correct = {row["sample_id"]: bool(row["correct"]) for row in by_arm[ARM_DENSE]}
    comparisons = {}
    for arm in MEASURED_ARMS:
        if arm == ARM_DENSE:
            continue
        correct = {row["sample_id"]: bool(row["correct"]) for row in by_arm[arm]}
        wall = metrics[arm]["timing_seconds"]["runner_wall_seconds"]["mean"]
        comparisons[arm] = {
            "accuracy_retention_vs_dense": metrics[arm]["accuracy"] / dense_accuracy if dense_accuracy else None,
            "accuracy_delta_vs_dense": metrics[arm]["accuracy"] - dense_accuracy,
            "paired_bootstrap_accuracy_delta_vs_dense": paired_bootstrap(correct, dense_correct, repetitions),
            "observed_runner_wall_speedup_vs_dense": dense_wall / wall,
            "observed_runner_wall_delta_ms": (wall - dense_wall) * 1000.0,
            "development_viability_88pct_quality_and_positive_software_speedup": (
                dense_accuracy > 0 and metrics[arm]["accuracy"] / dense_accuracy >= 0.88 and dense_wall / wall > 1.0
            ),
        }
    l2_vs_v2 = {}
    for budget in (96, 144):
        l2 = f"corrected_l2_integer_b1_k{budget}"
        v2 = f"v2_edge576_b1_k{budget}"
        l2_correct = {row["sample_id"]: bool(row["correct"]) for row in by_arm[l2]}
        v2_correct = {row["sample_id"]: bool(row["correct"]) for row in by_arm[v2]}
        l2_vs_v2[str(budget)] = paired_bootstrap(v2_correct, l2_correct, repetitions)
    return {
        "status": "VSR_ZEROSHOT_DEV_BUDGET_CURVE_COMPLETE_WITH_V2_K192_INFEASIBLE",
        "scope": "development_only_not_external_confirmation",
        "official_primary_metric": "overall binary accuracy on all 340 zero-shot dev examples",
        "test_split_opened": False,
        "manifest_sha256": sha256_file(EVAL_MANIFEST),
        "arms": metrics, "comparisons_vs_dense": comparisons,
        "v2_minus_l2_same_budget": l2_vs_v2,
        "structurally_infeasible_arm": {
            "arm": INFEASIBLE_ARM,
            "scored_records": 0,
            "reason": "The frozen integer prompt scorer internally invokes a cap=16 selection audit. At K192 its 144-candidate pool cannot supply 183 required detail groups. The selection is not consumed downstream, but bypassing it would modify the frozen, HLS-audited scorer core.",
            "decision": "Do not modify the frozen core in this experiment; report K192 for L2 only and compare v2 vs L2 at K96/K144.",
        },
        "interpretation_rule": {
            "quality_reference": "ReCoVLM-referenced 88.9% relative task performance is contextual only; 0.88 is a permissive development screen, not an official VSR threshold.",
            "latency": "runner_wall_seconds is synchronized wall time from immediately before the frozen per-sample producer through decoded output; model load and warmup are excluded.",
            "no_board_claim": True,
        },
    }


def write_reports(report: Mapping[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arm", "accuracy", "relative_accuracy", "parse_rate", "mean_G", "mean_K", "token_retention", "mean_wall_ms", "p95_wall_ms", "speedup_vs_dense"))
        writer.writeheader()
        for arm in MEASURED_ARMS:
            metric = report["arms"][arm]
            comparison = report["comparisons_vs_dense"].get(arm, {})
            wall = metric["timing_seconds"]["runner_wall_seconds"]
            writer.writerow({"arm": arm, "accuracy": metric["accuracy"], "relative_accuracy": comparison.get("accuracy_retention_vs_dense", 1.0), "parse_rate": metric["parse_success_rate"], "mean_G": metric["mean_full_visual_tokens"], "mean_K": metric["mean_output_visual_tokens"], "token_retention": metric["mean_visual_token_retention"], "mean_wall_ms": wall["mean"] * 1000, "p95_wall_ms": wall["p95"] * 1000, "speedup_vs_dense": comparison.get("observed_runner_wall_speedup_vs_dense", 1.0)})
    lines = [
        "# VSR Zero-Shot Dev Token Selector 预算曲线", "",
        "本报告只用于方法开发，不是外部测试结论。官方 zero-shot test 未解析、未评分。主指标是 340 条 dev 的总体 True/False 准确率；关系类别只作诊断。", "",
        "| 方案 | 准确率 | 相对 Dense | 解析成功率 | 平均 G→K | Token 保留 | 完整时延 mean/p95 (ms) | 实测加速 |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in MEASURED_ARMS:
        metric = report["arms"][arm]
        comparison = report["comparisons_vs_dense"].get(arm, {})
        wall = metric["timing_seconds"]["runner_wall_seconds"]
        lines.append(f"| {arm} | {metric['accuracy']:.4f} | {comparison.get('accuracy_retention_vs_dense', 1.0):.4f} | {metric['parse_success_rate']:.4f} | {metric['mean_full_visual_tokens']:.1f}→{metric['mean_output_visual_tokens']:.1f} | {metric['mean_visual_token_retention']:.3f} | {wall['mean']*1000:.1f}/{wall['p95']*1000:.1f} | {comparison.get('observed_runner_wall_speedup_vs_dense', 1.0):.3f}× |")
    lines += ["", "## 同预算 v2 相对 L2", ""]
    for budget, item in report["v2_minus_l2_same_budget"].items():
        lines.append(f"- K{budget}: 准确率差 {item['observed_difference']:+.4f}，95% CI [{item['ci_95_lower']:+.4f}, {item['ci_95_upper']:+.4f}]。")
    lines += ["", "## 结构性不可执行项", "", "- v2 K192 在任何正式记录写入前 fail-closed：冻结整数 Prompt scorer 内部仍执行 cap=16 的审计选择，144 个候选不足以提供 183 个细节组。该中间选择虽不被后续推理消费，但绕过它会修改已完成 HLS 数值审计的核心，因此本轮不改核心、不生成 v2 K192 数据。", ""]
    lines += ["", "## 解释边界", "", "- 完整时延采用每条样本冻结生产器外层的 CUDA 同步墙钟时间，包含图片打开、处理、传输、视觉、Prefill、Decode 和输出解码；不含模型加载及每臂一次 warmup。", "- 当前 Selector 仍是 CPU 往返软件实现；其时延不能作为 ZCU102 上板实测，也不能据此声称板端加速。", "- 0.88 相对质量仅是宽松的发展阶段筛选线，不是 VSR 官方标准，也不替代一次性外部测试。", ""]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    args = parser.parse_args(argv)
    report = summarize(args.bootstrap_repetitions)
    write_reports(report)
    print(json.dumps({"status": report["status"], "json": str(REPORT_JSON), "csv": str(REPORT_CSV), "markdown": str(REPORT_MD)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
