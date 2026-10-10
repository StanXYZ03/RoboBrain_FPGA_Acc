#!/usr/bin/env python3
"""Aggregate token-reduction JSONL runs into an auditable Pareto report."""

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

from evaluation.metrics import evaluate, load_jsonl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(records: list[dict[str, Any]], key: str) -> float:
    return statistics.median(float(record[key]) for record in records if record.get(key) is not None)


def aggregate(path: Path, manifest: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    if len(records) != 64:
        raise ValueError(f"{path}: expected 64 records, observed {len(records)}")
    _, metrics = evaluate(manifest, path)
    first = records[0]
    encoder_tokens = sum(int(record["encoder_tokens"]) for record in records)
    baseline_tokens = sum(int(record["baseline_encoder_tokens"]) for record in records)
    encoder_macs = sum(int(record["encoder_macs_proxy"]) for record in records)
    baseline_encoder_macs = sum(int(record["baseline_encoder_macs_proxy"]) for record in records)
    llm_macs = sum(int(record["llm_prefill_macs_proxy"]) for record in records)
    baseline_llm_macs = sum(int(record["baseline_llm_prefill_macs_proxy"]) for record in records)
    return {
        "file": str(path),
        "sha256": sha256(path),
        "strategy": first["strategy"],
        "requested_retention": float(first["requested_retention"]),
        "sample_count": len(records),
        "actual_token_retention": encoder_tokens / baseline_tokens,
        "encoder_compute_retention": encoder_macs / baseline_encoder_macs,
        "llm_prefill_compute_retention": llm_macs / baseline_llm_macs,
        "vision_encoder_seconds_median": median(records, "vision_encoder_seconds"),
        "llm_prefill_seconds_median": median(records, "llm_prefill_seconds"),
        "decoder_seconds_median": median(records, "decoder_seconds"),
        "elapsed_seconds_median": median(records, "elapsed_seconds"),
        "peak_cuda_bytes_max": max(int(record["peak_cuda_bytes"]) for record in records),
        "metrics": metrics,
    }


def f(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def table_row(result: dict[str, Any], baseline: dict[str, Any]) -> str:
    point = result["metrics"]["tasks"]["pointing"]
    grounding = result["metrics"]["tasks"]["grounding"]
    vision_speedup = baseline["vision_encoder_seconds_median"] / result["vision_encoder_seconds_median"]
    prefill_speedup = baseline["llm_prefill_seconds_median"] / result["llm_prefill_seconds_median"]
    return "| " + " | ".join(
        (
            f(result["requested_retention"], 3),
            f(result["actual_token_retention"], 3),
            f(result["encoder_compute_retention"], 3),
            f(result["llm_prefill_compute_retention"], 3),
            f(vision_speedup, 3) + "×",
            f(prefill_speedup, 3) + "×",
            f(point["pck_0_1"]),
            f(point["pck_0_2"]),
            f(grounding["iou_mean"]),
            f(grounding["acc_0_5"]),
        )
    ) + " |"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results = [aggregate(path, args.manifest) for path in sorted(args.input_dir.glob("*.jsonl"))]
    if len(results) != 10:
        raise ValueError(f"expected 10 complete configurations, observed {len(results)}")
    results.sort(key=lambda result: (result["strategy"], -result["requested_retention"]))
    baseline = next(
        result
        for result in results
        if result["strategy"] == "uniform" and result["requested_retention"] == 1.0
    )
    base_point = baseline["metrics"]["tasks"]["pointing"]
    base_grounding = baseline["metrics"]["tasks"]["grounding"]
    for result in results:
        point = result["metrics"]["tasks"]["pointing"]
        grounding = result["metrics"]["tasks"]["grounding"]
        result["delta_vs_uniform_r1000"] = {
            "pck_0_1": point["pck_0_1"] - base_point["pck_0_1"],
            "pck_0_2": point["pck_0_2"] - base_point["pck_0_2"],
            "iou_mean": grounding["iou_mean"] - base_grounding["iou_mean"],
            "acc_0_5": grounding["acc_0_5"] - base_grounding["acc_0_5"],
        }
        # With n=32 per task, one binary outcome is 3.125 percentage points.
        # This smoke gate therefore cannot resolve the originally proposed 2
        # percentage-point tolerance. This provisional gate allows at most one
        # changed binary outcome and 0.02 mean-IoU loss.
        result["provisional_smoke_gate"] = bool(
            result["actual_token_retention"] <= 0.5
            and point["pck_0_1"] >= base_point["pck_0_1"] - 1 / 32
            and point["pck_0_2"] >= base_point["pck_0_2"] - 1 / 32
            and grounding["acc_0_5"] >= base_grounding["acc_0_5"] - 1 / 32
            and grounding["iou_mean"] >= base_grounding["iou_mean"] - 0.02
        )

    passing = [
        result
        for result in results
        if result["provisional_smoke_gate"]
    ]
    uniform_quarter = next(
        result
        for result in results
        if result["strategy"] == "uniform" and result["requested_retention"] == 0.25
    )
    oracle_quarter = next(
        result
        for result in results
        if result["strategy"] == "oracle_crop" and result["requested_retention"] == 0.25
    )
    uniform_quarter_point = uniform_quarter["metrics"]["tasks"]["pointing"]
    oracle_quarter_point = oracle_quarter["metrics"]["tasks"]["pointing"]
    uniform_quarter_grounding = uniform_quarter["metrics"]["tasks"]["grounding"]
    oracle_quarter_grounding = oracle_quarter["metrics"]["tasks"]["grounding"]
    matched_quarter_comparison = {
        "uniform_actual_token_retention": uniform_quarter["actual_token_retention"],
        "oracle_actual_token_retention": oracle_quarter["actual_token_retention"],
        "oracle_minus_uniform": {
            "pck_0_1": oracle_quarter_point["pck_0_1"] - uniform_quarter_point["pck_0_1"],
            "pck_0_2": oracle_quarter_point["pck_0_2"] - uniform_quarter_point["pck_0_2"],
            "iou_mean": oracle_quarter_grounding["iou_mean"] - uniform_quarter_grounding["iou_mean"],
            "acc_0_5": oracle_quarter_grounding["acc_0_5"] - uniform_quarter_grounding["acc_0_5"],
        },
    }
    summary = {
        "date": "2026-08-25",
        "scope": "32 COCO images; 32 Pointing + 32 Grounding; smoke/pipeline gate only",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "configuration_count": len(results),
        "prediction_count": sum(result["sample_count"] for result in results),
        "formal_two_percent_gate_resolvable": False,
        "provisional_gate_definition": "actual token retention <=0.5; <=1/32 loss on PCK/Acc; <=0.02 mean IoU loss",
        "provisional_passing_configs": [
            f"{result['strategy']}@{result['requested_retention']:.3f}" for result in passing
        ],
        "matched_quarter_budget_comparison": matched_quarter_comparison,
        "decision": "selector_research_go_but_accuracy_gate_not_passed",
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# RoboBrain2.5-4B BF16 视觉 Token 缩减可行性扫描",
        "",
        "日期：2026-08-25  ",
        "状态：`640_INFERENCES_COMPLETE / METRICS_COMPLETE / SMOKE_ONLY`",
        "",
        "## 1. 实验边界",
        "",
        "冻结集包含 32 张 COCO 图片，每张各有 Pointing 和 Grounding 查询，共 64 条。",
        "`uniform` 是真实降低输入网格的统一分辨率基线；`oracle_crop` 使用真实标注框",
        "选择目标区域并将预测坐标映射回原图，只表示空间选择方法的不可部署上界。",
        "两者都实际改变 Encoder 和 LLM 视觉 Token 数，不是保持计算量不变的 mask。",
        "",
        "MAC proxy 覆盖 Vision/LLM block 的主投影、FFN 和 attention，明确不含 patch",
        "embedding、merger、DeepStack projection、文本 token，因此只用于相对比较。",
        "",
        "## 2. Pareto 结果",
        "",
        "表中速度为分阶段 CUDA event 中位数相对 uniform@1.0；不是完整图片推理速度。",
        "",
    ]
    header = "| 请求保留率 | 实际 Token | Encoder MAC | LLM prefill MAC | Vision 加速 | Prefill 加速 | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |"
    divider = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    for strategy in ("uniform", "oracle_crop"):
        lines.extend((f"### {strategy}", "", header, divider))
        lines.extend(table_row(result, baseline) for result in results if result["strategy"] == strategy)
        lines.append("")

    matched_delta = matched_quarter_comparison["oracle_minus_uniform"]
    lines.extend(
        (
            "### 匹配的约 1/4 Token 预算",
            "",
            f"uniform 实际保留 {uniform_quarter['actual_token_retention']:.3f}，oracle 实际保留 "
            f"{oracle_quarter['actual_token_retention']:.3f}。在几乎相同预算下，oracle 相对 uniform：",
            "",
            f"- PCK@0.1：{matched_delta['pck_0_1']:+.4f}；PCK@0.2：{matched_delta['pck_0_2']:+.4f}。",
            f"- Mean IoU：{matched_delta['iou_mean']:+.4f}；Acc@0.5：{matched_delta['acc_0_5']:+.4f}。",
            "",
            "这不是可部署结果，但证明精度损失主要取决于保留哪些空间区域，而不只是",
            "Token 数；因此存在开发无真值、空间保持型选择器的研究价值。",
            "",
        )
    )

    lines.extend(
        (
            "## 3. 门控解释",
            "",
            "原定‘精度下降不超过 2 个百分点’无法由每任务 32 条的集合分辨：一个样本",
            "就是 3.125 个百分点。因此本报告只给烟雾门：实际 Token≤50%，PCK/Acc",
            "最多下降 1/32，Mean IoU 最多下降 0.02。它不能替代扩大后的置信区间。",
            "",
            "烟雾门通过配置：" + (", ".join(summary["provisional_passing_configs"]) or "无") + "。",
            "阶段决策：精度门未通过，不进入论文精度声明；但匹配预算的 oracle 优势",
            "足够明确，进入无真值结构化选择器研究。",
            "",
            "## 4. 结论使用限制",
            "",
            "- uniform 只回答分辨率基线，不是创新方法。",
            "- oracle_crop 使用真值框，不能用于部署或论文主结果。",
            "- 当前集没有 Affordance、复杂 referring expression 或 Trajectory 标注。",
            "- 下一步只有在 oracle 明显优于 uniform 时，才值得开发无真值的文本/视觉",
            "  重要性预测器，并把动态 Token 索引接入 PL。",
            "",
            "## 5. 复现",
            "",
            f"- 原始结果目录：`{args.input_dir}`",
            f"- 机器可读汇总：`{args.summary}`",
            "- 运行器：`evaluation/run_hf_token_reduction.py`",
            "- 纯函数与 MAC proxy：`evaluation/token_reduction.py`",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["provisional_passing_configs"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
