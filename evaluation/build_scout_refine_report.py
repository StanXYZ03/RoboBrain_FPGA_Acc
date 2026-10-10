#!/usr/bin/env python3
"""Build the auditable no-GT scout-refine selector report."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def aggregate(label: str, path: Path, manifest: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    if len(records) != 64 or len({record["sample_id"] for record in records}) != 64:
        raise ValueError(f"{path}: expected 64 unique predictions")
    _, metrics = evaluate(manifest, path)

    def ratio(value: str, baseline: str) -> float:
        return sum(int(record[value]) for record in records) / sum(
            int(record[baseline]) for record in records
        )

    return {
        "label": label,
        "path": str(path),
        "sha256": sha256(path),
        "prediction_count": len(records),
        "actual_token_retention": ratio("encoder_tokens", "baseline_encoder_tokens"),
        "encoder_compute_retention": ratio("encoder_macs_proxy", "baseline_encoder_macs_proxy"),
        "llm_prefill_compute_retention": ratio(
            "llm_prefill_macs_proxy", "baseline_llm_prefill_macs_proxy"
        ),
        "selector_parse_rate": sum(bool(record.get("selector_parse_success", True)) for record in records)
        / len(records),
        "metrics": metrics,
    }


def write_hybrid(manifest: Path, tight: Path, wide: Path, output: Path) -> None:
    samples = load_jsonl(manifest)
    tight_by_id = {record["sample_id"]: record for record in load_jsonl(tight)}
    wide_by_id = {record["sample_id"]: record for record in load_jsonl(wide)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for sample in samples:
            # This branch uses only the prompt/task protocol, never the label:
            # Pointing uses the fixed square crop; Grounding uses the wider
            # expanded predicted box. PL would still receive the same crop
            # descriptor and compute workload.
            source = tight_by_id if sample["task"] == "pointing" else wide_by_id
            record = dict(source[sample["sample_id"]])
            record["profile"] = "task_protocol_hybrid"
            record["hybrid_source_profile"] = "tight" if sample["task"] == "pointing" else "wide"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def f(value: float) -> str:
    return f"{value:.4f}"


def row(result: dict[str, Any]) -> str:
    point = result["metrics"]["tasks"]["pointing"]
    grounding = result["metrics"]["tasks"]["grounding"]
    return "| " + " | ".join(
        (
            result["label"],
            f(result["actual_token_retention"]),
            f(result["encoder_compute_retention"]),
            f(result["llm_prefill_compute_retention"]),
            f(point["pck_0_1"]),
            f(point["pck_0_2"]),
            f(grounding["iou_mean"]),
            f(grounding["acc_0_5"]),
            f(point["latency_seconds_median"]),
            f(grounding["latency_seconds_median"]),
        )
    ) + " |"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hybrid-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    tight_high = args.input_dir / "scout_r0125_refine_r0250_tight.jsonl"
    wide_high = args.input_dir / "scout_r0125_refine_r0250_wide.jsonl"
    write_hybrid(args.manifest, tight_high, wide_high, args.hybrid_output)

    specs = (
        ("uniform@1.0", args.baseline_dir / "uniform_r1000.jsonl"),
        ("uniform@0.25", args.baseline_dir / "uniform_r0250.jsonl"),
        ("uniform@0.50", args.baseline_dir / "uniform_r0500.jsonl"),
        ("oracle@0.25", args.baseline_dir / "oracle_crop_r0250.jsonl"),
        ("scout-refine low/tight", args.input_dir / "scout_r0125_refine_r0125_tight.jsonl"),
        ("scout-refine low/wide", args.input_dir / "scout_r0125_refine_r0125_wide.jsonl"),
        ("scout-refine high/tight", tight_high),
        ("scout-refine high/wide", wide_high),
        ("scout-refine task hybrid", args.hybrid_output),
    )
    results = [aggregate(label, path, args.manifest) for label, path in specs]
    by_label = {result["label"]: result for result in results}
    full = by_label["uniform@1.0"]
    uniform_half = by_label["uniform@0.50"]
    hybrid = by_label["scout-refine task hybrid"]
    hp = hybrid["metrics"]["tasks"]["pointing"]
    hg = hybrid["metrics"]["tasks"]["grounding"]
    bp = full["metrics"]["tasks"]["pointing"]
    bg = full["metrics"]["tasks"]["grounding"]
    up = uniform_half["metrics"]["tasks"]["pointing"]
    ug = uniform_half["metrics"]["tasks"]["grounding"]
    board_fixed_one_pass = 0.204 + 63.732 + 1.562
    board_projected_seconds = (
        625.425 * hybrid["encoder_compute_retention"]
        + 1003.155 * hybrid["llm_prefill_compute_retention"]
        + 2 * board_fixed_one_pass
    )

    summary = {
        "date": "2026-08-25",
        "scope": "32 Pointing + 32 Grounding smoke set; no-ground-truth selection",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "inference_count": 256,
        "hybrid": hybrid,
        "hybrid_delta_vs_full": {
            "pck_0_1": hp["pck_0_1"] - bp["pck_0_1"],
            "pck_0_2": hp["pck_0_2"] - bp["pck_0_2"],
            "iou_mean": hg["iou_mean"] - bg["iou_mean"],
            "acc_0_5": hg["acc_0_5"] - bg["acc_0_5"],
        },
        "hybrid_delta_vs_uniform_r0500": {
            "token_retention": hybrid["actual_token_retention"]
            - uniform_half["actual_token_retention"],
            "pck_0_1": hp["pck_0_1"] - up["pck_0_1"],
            "pck_0_2": hp["pck_0_2"] - up["pck_0_2"],
            "iou_mean": hg["iou_mean"] - ug["iou_mean"],
            "acc_0_5": hg["acc_0_5"] - ug["acc_0_5"],
        },
        "gate": {
            "accuracy_gate_pass": False,
            "pl_integration_gate_pass": False,
            "reason": "hybrid is a useful Pareto point but loses grounding accuracy and doubles sequential passes",
        },
        "zcu102_first_order_projection": {
            "seconds_per_image": board_projected_seconds,
            "speedup_vs_1695_01_seconds": 1695.01 / board_projected_seconds,
            "hz": 1.0 / board_projected_seconds,
            "assumption": "visual stages scale with MAC proxy; fixed text/decode cost paid twice; selector overhead otherwise zero",
        },
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    delta = summary["hybrid_delta_vs_uniform_r0500"]
    lines = [
        "# 无真值 Scout–Refine 规则区域选择器实验",
        "",
        "日期：2026-08-25  ",
        "状态：`256_INFERENCES_COMPLETE / ACCURACY_GATE_FAIL / PL_GATE_FAIL`",
        "",
        "## 1. 方法",
        "",
        "第一遍用 12.5% 请求分辨率输出预测点/框；第二遍不读取标注，Pointing 在预测点周围",
        "取固定方形区域，Grounding 对预测框按固定比例扩展，再将第二遍坐标映射回原图。",
        "`task hybrid` 只依据公开任务协议选参数：Pointing=tight，Grounding=wide。",
        "总 Token、MAC proxy 和延迟均包含两遍，没有隐藏 scout 开销。",
        "",
        "## 2. 完整结果",
        "",
        "| 配置 | 实际 Token | Encoder MAC | LLM prefill MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | Point p50/s | Ground p50/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(row(result) for result in results)
    lines.extend(
        (
            "",
            "## 3. 门控结论",
            "",
            f"task hybrid 实际使用 {hybrid['actual_token_retention']:.1%} Token。相对 uniform@0.50，",
            f"Token 减少 {-delta['token_retention']:.1%}（绝对保留率），PCK@0.1 {delta['pck_0_1']:+.4f}，",
            f"但 Mean IoU {delta['iou_mean']:+.4f}、Acc@0.5 {delta['acc_0_5']:+.4f}。",
            "",
            "它形成一个可复现的低 Token Pareto 点，但没有满足完整输入的精度门，也没有统一超过",
            "matched uniform。两遍 A6000 中位延迟约为单遍的两倍；在 PS 上还会重复文本 prefill",
            "与 decode。因此该方案只保留为软件基线，不接入 PL。",
            "",
            f"按既有 ZCU102 PS 阶段时间一阶投影，并把文本 prefill/decode 固定成本支付两次，",
            f"task hybrid 约为 {board_projected_seconds:.1f} s/图、{1695.01 / board_projected_seconds:.2f}×、",
            f"{1.0 / board_projected_seconds:.6f} Hz；这只是工作量投影，不是板端实测。",
            "",
            "## 4. 负结果带来的设计约束",
            "",
            "- Pointing 与 Grounding 的最佳固定裁剪范围不同，固定统一参数不可行。",
            "- 极低分辨率 scout 的定位误差会把第二遍高分辨率计算集中到错误区域。",
            "- 裁剪重建局部坐标系，无法天然保持原图 MRoPE/DeepStack 的全局空间参考。",
            "- 下一候选应是单遍浅层特征选择：保留原 position ID，以规则 2×2/4×4 块压缩，",
            "  同时保留分区全局锚点；不能再依赖完整 scout decode。",
            "",
            "## 5. 复现",
            "",
            f"- 原始结果：`{args.input_dir}`",
            f"- 派生 hybrid：`{args.hybrid_output}`",
            f"- 机器汇总：`{args.summary}`",
            "- 运行器：`evaluation/run_hf_scout_refine.py`",
            "- 汇总器：`evaluation/build_scout_refine_report.py`",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
