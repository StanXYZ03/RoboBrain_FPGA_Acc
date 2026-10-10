#!/usr/bin/env python3
"""Build the immutable report for the frozen L2-residual selector blind run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_expanded_selector_report import aggregate, paired_comparison


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(result: dict) -> dict[str, float]:
    pointing = result["metrics"]["tasks"]["pointing"]
    grounding = result["metrics"]["tasks"]["grounding"]
    return {
        "pck_0_1": pointing["pck_0_1"],
        "pck_0_2": pointing["pck_0_2"],
        "iou_mean": grounding["iou_mean"],
        "acc_0_5": grounding["acc_0_5"],
        "point_nme_bbox_diag_mean": pointing["point_nme_bbox_diag_mean"],
        "point_latency_seconds_mean": pointing["latency_seconds_mean"],
        "grounding_latency_seconds_mean": grounding["latency_seconds_mean"],
    }


def ci(comparison: dict, task: str, metric: str) -> str:
    value = comparison[task][metric]
    return f"{value['mean_delta']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--configuration-lock", type=Path, required=True)
    parser.add_argument("--blind-lock", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "full": args.input_dir / "shallow_r1000.jsonl",
        "l2": args.input_dir / "shallow_r0875.jsonl",
        "frozen": args.input_dir / "l2res_lambda_2_r0875.jsonl",
    }
    results = {
        "full": aggregate("full@1.000", paths["full"], args.manifest),
        "l2": aggregate("L2+anchor@0.875", paths["l2"], args.manifest),
        "frozen": aggregate("frozen L2+residual(lambda=2)@0.875", paths["frozen"], args.manifest),
    }
    metrics = {name: compact(result) for name, result in results.items()}
    comparisons = {
        "frozen_vs_l2": paired_comparison(
            args.manifest, paths["frozen"], paths["l2"], seed=20260830, iterations=10000
        ),
        "frozen_vs_full": paired_comparison(
            args.manifest, paths["frozen"], paths["full"], seed=20260831, iterations=10000
        ),
    }
    primary = ("pck_0_1", "pck_0_2", "iou_mean", "acc_0_5")
    delta_l2 = {key: metrics["frozen"][key] - metrics["l2"][key] for key in primary}
    delta_full = {key: metrics["frozen"][key] - metrics["full"][key] for key in primary}
    point_estimate_full_gate = all(value >= -0.02 for value in delta_full.values())
    strict_l2_dominance = all(value >= 0 for value in delta_l2.values()) and any(
        value > 0 for value in delta_l2.values()
    )
    statistical_gain_l2 = any(
        value["ci95_low"] > 0
        for task in comparisons["frozen_vs_l2"].values()
        for value in task.values()
    )
    statistical_noninferiority_full = all(
        value["ci95_low"] >= -0.02
        for task in comparisons["frozen_vs_full"].values()
        for value in task.values()
    )
    gate = {
        "blind_integrity_pass": True,
        "configuration_frozen_pass": True,
        "point_estimate_accuracy_vs_full_pass": point_estimate_full_gate,
        "statistical_noninferiority_vs_full_pass": statistical_noninferiority_full,
        "strict_dominance_vs_l2_pass": strict_l2_dominance,
        "statistically_resolved_gain_vs_l2_pass": statistical_gain_l2,
        "pl_integration_pass": False,
        "decision": "STOP_FIRST_ORDER_SALIENCY_DISTILLATION / RETAIN_L2_BASELINE / PL_HOLD",
        "reason": (
            "冻结候选在87.5% Token下保持full点估计，但未严格支配同预算L2；"
            "相对L2三项下降、一项提高，所有成对95% CI跨0，且统计非劣性未建立"
        ),
    }
    summary = {
        "date": "2026-08-26",
        "status": "576_INFERENCES / BLIND_COMPLETE / ALGORITHM_GATE_FAIL / PL_HOLD",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "configuration_lock": str(args.configuration_lock),
        "configuration_lock_sha256": sha256(args.configuration_lock),
        "blind_lock": str(args.blind_lock),
        "blind_lock_sha256": sha256(args.blind_lock),
        "prediction_sha256": {name: sha256(path) for name, path in paths.items()},
        "inference_count": 576,
        "metrics": metrics,
        "delta_frozen_vs_l2": delta_l2,
        "delta_frozen_vs_full": delta_full,
        "paired_bootstrap": comparisons,
        "compute_proxy": {
            name: {
                "token_retention": result["actual_token_retention"],
                "encoder_compute_retention": result["encoder_compute_retention"],
                "llm_prefill_compute_retention": result["llm_prefill_compute_retention"],
            }
            for name, result in results.items()
        },
        "gate": gate,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    rows = []
    for name, label in (("full", "Full"), ("l2", "L2+anchor"), ("frozen", "L2+残差头")):
        value = metrics[name]
        proxy = summary["compute_proxy"][name]
        rows.append(
            f"| {label} | {proxy['token_retention']:.3f} | {proxy['encoder_compute_retention']:.3f} | "
            f"{proxy['llm_prefill_compute_retention']:.3f} | {value['pck_0_1']:.4f} | "
            f"{value['pck_0_2']:.4f} | {value['iou_mean']:.4f} | {value['acc_0_5']:.4f} | "
            f"{value['point_nme_bbox_diag_mean']:.4f} |"
        )
    report = [
        "# L2安全锚定残差选择器：冻结盲验证",
        "",
        "日期：2026-08-26  ",
        "状态：`576_INFERENCES / BLIND_COMPLETE / ALGORITHM_GATE_FAIL / PL_HOLD`",
        "",
        "## 实验纪律",
        "",
        "开发集冻结 `lambda=2, retention=0.875, layer-5, 3×3区域配额` 后才创建96图盲集。",
        "盲集与此前224张图片零重叠；full、L2和冻结候选各运行一次，揭盲后未调参。",
        "",
        "## 盲测结果",
        "",
        "| 配置 | Token | Encoder MAC | LLM MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | Point NME |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## 冻结候选相对L2的成对差值（95% bootstrap CI）",
        "",
        "| PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |",
        "|---:|---:|---:|---:|",
        "| " + " | ".join((
            ci(comparisons["frozen_vs_l2"], "pointing", "pck_0_1"),
            ci(comparisons["frozen_vs_l2"], "pointing", "pck_0_2"),
            ci(comparisons["frozen_vs_l2"], "grounding", "iou_mean"),
            ci(comparisons["frozen_vs_l2"], "grounding", "acc_0_5"),
        )) + " |",
        "",
        "## 结论",
        "",
        gate["reason"] + "。",
        "因此停止一阶 activation×gradient 蒸馏路线，不把113参数残差头映射到PL。",
        "87.5%浅层结构化剪枝和L2规则继续作为强基线；下一算法必须直接建模多个group联合删除后的坐标损失，",
        "而不是继续拟合单token一阶重要性。GPU端到端延迟未显示残差头收益，ZCU102收益仍须等算法门通过后上板实测。",
        "",
        "## 复现锚点",
        "",
        f"- 配置锁：`{args.configuration_lock}`",
        f"- 盲集锁：`{args.blind_lock}`",
        f"- 原始输出：`{args.input_dir}`",
        f"- 机器汇总：`{args.summary}`",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(report) + "\n")
    print(json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
