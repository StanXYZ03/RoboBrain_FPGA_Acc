#!/usr/bin/env python3
"""Report the frozen selector on the untouched validation manifest."""

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


def compact_metrics(result: dict) -> dict[str, float]:
    point = result["metrics"]["tasks"]["pointing"]
    ground = result["metrics"]["tasks"]["grounding"]
    return {
        "pck_0_1": point["pck_0_1"],
        "pck_0_2": point["pck_0_2"],
        "iou_mean": ground["iou_mean"],
        "acc_0_5": ground["acc_0_5"],
    }


def row(result: dict) -> str:
    metrics = compact_metrics(result)
    return (
        f"| {result['label']} | {result['actual_token_retention']:.3f} | "
        f"{result['encoder_compute_retention']:.3f} | {result['llm_prefill_compute_retention']:.3f} | "
        f"{metrics['pck_0_1']:.4f} | {metrics['pck_0_2']:.4f} | "
        f"{metrics['iou_mean']:.4f} | {metrics['acc_0_5']:.4f} | "
        f"{result['zcu102_projection']['seconds_per_image']:.1f} | "
        f"{result['zcu102_projection']['speedup']:.2f}× |"
    )


def ci(comparison: dict, task: str, metric: str) -> str:
    value = comparison[task][metric]
    return f"{value['mean_delta']:+.4f} [{value['ci95_low']:+.4f}, {value['ci95_high']:+.4f}]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--tuning-lock", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    full_path = args.input_dir / "shallow_r1000.jsonl"
    l2_path = args.input_dir / "shallow_r0750.jsonl"
    frozen_path = args.input_dir / "residual_r0750.jsonl"
    results = [
        aggregate("full@1.000", full_path, args.manifest),
        aggregate("L2+anchor@0.750", l2_path, args.manifest),
        aggregate("frozen residual+quota@0.750", frozen_path, args.manifest),
    ]
    by_label = {result["label"]: result for result in results}
    frozen_metrics = compact_metrics(by_label["frozen residual+quota@0.750"])
    l2_metrics = compact_metrics(by_label["L2+anchor@0.750"])
    full_metrics = compact_metrics(by_label["full@1.000"])
    frozen_vs_l2 = paired_comparison(
        args.manifest, frozen_path, l2_path, seed=20260827, iterations=10000
    )
    frozen_vs_full = paired_comparison(
        args.manifest, frozen_path, full_path, seed=20260828, iterations=10000
    )
    delta_l2 = {key: frozen_metrics[key] - l2_metrics[key] for key in frozen_metrics}
    delta_full = {key: frozen_metrics[key] - full_metrics[key] for key in frozen_metrics}
    strict_dominance = all(value >= 0 for value in delta_l2.values()) and any(
        value > 0 for value in delta_l2.values()
    )
    gate = {
        "blind_manifest_integrity_pass": True,
        "frozen_configuration_audit_pass": True,
        "strict_dominance_vs_l2": strict_dominance,
        "statistically_resolved_gain_vs_l2": any(
            metric["ci95_low"] > 0
            for task in frozen_vs_l2.values()
            for metric in task.values()
        ),
        "accuracy_gate_vs_full": (
            delta_full["pck_0_1"] >= -0.02
            and delta_full["pck_0_2"] >= -0.02
            and delta_full["iou_mean"] >= -0.02
            and delta_full["acc_0_5"] >= -0.02
        ),
        "pl_integration_pass": False,
        "decision": "STOP_PURE_RESIDUAL_SELECTOR",
        "reason": (
            "开发集四指标同时改善未在盲集复现；相对 L2 仅 Mean IoU 点估计略升，"
            "PCK@0.1 与 Acc@0.5 各下降 1/96，所有成对 95% CI 均跨 0"
        ),
    }
    summary = {
        "date": "2026-08-25",
        "status": "BLIND_VALIDATION_COMPLETE / SELECTOR_GATE_FAIL / PL_HOLD",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "tuning_lock": str(args.tuning_lock),
        "tuning_lock_sha256": sha256(args.tuning_lock),
        "inference_count": 576,
        "frozen_configuration": {
            "retention": 0.75,
            "prune_after_layer": 5,
            "region_grid": 3,
            "energy_weight": 0.0,
        },
        "gate": gate,
        "delta_frozen_vs_l2": delta_l2,
        "delta_frozen_vs_full": delta_full,
        "paired_bootstrap_frozen_vs_l2": frozen_vs_l2,
        "paired_bootstrap_frozen_vs_full": frozen_vs_full,
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# 冻结空间选择器盲验证",
        "",
        "日期：2026-08-25  ",
        "状态：`576_INFERENCES / BLIND_VALIDATION_COMPLETE / GATE_FAIL / PL_HOLD`",
        "",
        "## 1. 实验纪律",
        "",
        "配置在开发集上按四指标等权均值冻结为：75% Token、layer-5、3×3 区域配额、",
        "纯四邻域特征残差。盲集 96 张图片在首次推理前写入锁文件，与 smoke/开发集均零重叠。",
        "盲验证只运行 full、同预算 L2 和冻结配置，未根据结果重新调参。",
        "",
        "## 2. 盲验证结果",
        "",
        "| 配置 | Token | Encoder MAC | LLM MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | PS 投影/s | 投影加速 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(row(result) for result in results)
    lines.extend(
        (
            "",
            "## 3. 冻结配置相对 L2 的成对差值",
            "",
            "| PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 |",
            "|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                (
                    ci(frozen_vs_l2, "pointing", "pck_0_1"),
                    ci(frozen_vs_l2, "pointing", "pck_0_2"),
                    ci(frozen_vs_l2, "grounding", "iou_mean"),
                    ci(frozen_vs_l2, "grounding", "acc_0_5"),
                )
            )
            + " |",
            "",
            "## 4. 门控结论",
            "",
            gate["reason"] + "。",
            "纯残差选择器未形成可泛化优势，停止该固定规则，不进入 HLS/RTL。",
            "保留其为负结果和基线；下一算法候选必须引入可训练或文本条件信息，",
            "同时维持规则块输出和原始 MRoPE/DeepStack 对齐，不能继续手调单一视觉分数。",
            "",
            "ZCU102 时间仍为 MAC 比例的一阶工作量投影，不是该选择器的板端实测。",
            "",
            "## 5. 复现",
            "",
            f"- 冻结锁：`{args.tuning_lock}`",
            f"- 盲集：`{args.manifest}`",
            f"- 原始结果：`{args.input_dir}`",
            f"- 机器汇总：`{args.summary}`",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
