#!/usr/bin/env python3
"""Audit and summarize the single-pass shallow structured pruning scan."""

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


RETENTIONS = (1.0, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125)
PS_VISION_SECONDS = 625.425
PS_LLM_PREFILL_SECONDS = 1003.155
PS_FIXED_SECONDS = 65.498
PS_MEASURED_BASELINE_SECONDS = 1695.01


def tag(retention: float) -> str:
    return f"{round(retention * 1000):04d}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median_present(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.median(values) if values else None


def ratio(records: list[dict[str, Any]], value: str, baseline: str) -> float:
    return sum(float(record[value]) for record in records) / sum(
        float(record[baseline]) for record in records
    )


def aggregate(retention: float, path: Path, manifest: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    unique_ids = {record["sample_id"] for record in records}
    if len(records) != 64 or len(unique_ids) != 64:
        raise ValueError(f"{path}: expected 64 records with 64 unique IDs")
    _, metrics = evaluate(manifest, path)
    encoder_compute = ratio(records, "encoder_macs_proxy", "baseline_encoder_macs_proxy")
    llm_compute = ratio(records, "llm_prefill_macs_proxy", "baseline_llm_prefill_macs_proxy")
    board_seconds = (
        PS_VISION_SECONDS * encoder_compute
        + PS_LLM_PREFILL_SECONDS * llm_compute
        + PS_FIXED_SECONDS
    )
    return {
        "requested_retention": retention,
        "path": str(path),
        "sha256": sha256(path),
        "prediction_count": len(records),
        "actual_token_retention": ratio(
            records, "encoder_tokens_after_prune", "baseline_encoder_tokens"
        ),
        "encoder_compute_retention": encoder_compute,
        "llm_prefill_compute_retention": llm_compute,
        "weighted_variable_compute_retention": (
            PS_VISION_SECONDS * encoder_compute + PS_LLM_PREFILL_SECONDS * llm_compute
        )
        / (PS_VISION_SECONDS + PS_LLM_PREFILL_SECONDS),
        "a6000_median_seconds": {
            key: median_present(records, key)
            for key in (
                "vision_prefix_block_seconds",
                "vision_pruned_block_seconds",
                "vision_other_seconds",
                "vision_encoder_seconds",
                "llm_prefill_seconds",
                "decoder_seconds",
                "timed_stage_seconds",
                "elapsed_seconds",
            )
        },
        "zcu102_first_order_projection": {
            "seconds_per_image": board_seconds,
            "speedup_vs_measured_baseline": PS_MEASURED_BASELINE_SECONDS / board_seconds,
            "hz": 1.0 / board_seconds,
        },
        "metrics": metrics,
    }


def accuracy_delta(result: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    point = result["metrics"]["tasks"]["pointing"]
    ground = result["metrics"]["tasks"]["grounding"]
    base_point = baseline["metrics"]["tasks"]["pointing"]
    base_ground = baseline["metrics"]["tasks"]["grounding"]
    return {
        "pck_0_1": point["pck_0_1"] - base_point["pck_0_1"],
        "pck_0_2": point["pck_0_2"] - base_point["pck_0_2"],
        "iou_mean": ground["iou_mean"] - base_ground["iou_mean"],
        "acc_0_5": ground["acc_0_5"] - base_ground["acc_0_5"],
    }


def exact_output_audit(shallow: Path, stock: Path) -> dict[str, Any]:
    shallow_records = load_jsonl(shallow)
    stock_records = load_jsonl(stock)
    shallow_by_id = {record["sample_id"]: record for record in shallow_records}
    stock_by_id = {record["sample_id"]: record for record in stock_records}
    if set(shallow_by_id) != set(stock_by_id):
        raise ValueError("100% shallow and stock baseline sample-ID sets differ")
    mismatches = [
        sample_id
        for sample_id in sorted(stock_by_id)
        if shallow_by_id[sample_id]["output_text"] != stock_by_id[sample_id]["output_text"]
    ]
    return {
        "stock_path": str(stock),
        "stock_sha256": sha256(stock),
        "compared_count": len(stock_by_id),
        "exact_match_count": len(stock_by_id) - len(mismatches),
        "mismatch_count": len(mismatches),
        "first_mismatch_id": mismatches[0] if mismatches else None,
        "pass": not mismatches,
    }


def metric_row(result: dict[str, Any], delta: dict[str, float]) -> str:
    point = result["metrics"]["tasks"]["pointing"]
    ground = result["metrics"]["tasks"]["grounding"]
    projection = result["zcu102_first_order_projection"]
    return "| " + " | ".join(
        (
            f"{result['requested_retention']:.3f}",
            f"{result['actual_token_retention']:.3f}",
            f"{result['encoder_compute_retention']:.3f}",
            f"{result['llm_prefill_compute_retention']:.3f}",
            f"{point['pck_0_1']:.4f} ({delta['pck_0_1']:+.4f})",
            f"{point['pck_0_2']:.4f} ({delta['pck_0_2']:+.4f})",
            f"{ground['iou_mean']:.4f} ({delta['iou_mean']:+.4f})",
            f"{ground['acc_0_5']:.4f} ({delta['acc_0_5']:+.4f})",
            f"{projection['seconds_per_image']:.1f}",
            f"{projection['speedup_vs_measured_baseline']:.2f}×",
        )
    ) + " |"


def timing_row(result: dict[str, Any], baseline_elapsed: float) -> str:
    times = result["a6000_median_seconds"]
    return "| " + " | ".join(
        (
            f"{result['requested_retention']:.3f}",
            f"{times['vision_prefix_block_seconds']:.4f}",
            f"{times['vision_pruned_block_seconds']:.4f}",
            f"{times['vision_encoder_seconds']:.4f}",
            f"{times['llm_prefill_seconds']:.4f}",
            f"{times['decoder_seconds']:.4f}",
            f"{times['elapsed_seconds']:.4f}",
            f"{baseline_elapsed / times['elapsed_seconds']:.2f}×",
        )
    ) + " |"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--stock-baseline", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    results = [
        aggregate(retention, args.input_dir / f"shallow_r{tag(retention)}.jsonl", args.manifest)
        for retention in RETENTIONS
    ]
    baseline = results[0]
    deltas = {
        tag(result["requested_retention"]): accuracy_delta(result, baseline)
        for result in results
    }
    output_audit = exact_output_audit(
        args.input_dir / "shallow_r1000.jsonl", args.stock_baseline
    )
    if not output_audit["pass"]:
        raise ValueError(f"100% implementation audit failed: {output_audit}")

    # A one-sample change is already 3.125 percentage points on each 32-sample task.
    # This is only a smoke-set gate, not a statistical equivalence claim.
    provisional_candidates = []
    for result in results[1:]:
        delta = deltas[tag(result["requested_retention"])]
        accuracy_pass = (
            delta["pck_0_1"] >= -0.03125
            and delta["pck_0_2"] >= -0.03125
            and delta["iou_mean"] >= -0.02
            and delta["acc_0_5"] >= -0.03125
        )
        compute_pass = result["weighted_variable_compute_retention"] <= 0.90
        result["gate"] = {
            "provisional_accuracy_pass": accuracy_pass,
            "meaningful_compute_pass": compute_pass,
            "provisional_candidate": accuracy_pass and compute_pass,
        }
        if accuracy_pass and compute_pass:
            provisional_candidates.append(result)
    selected = min(
        provisional_candidates,
        key=lambda item: item["zcu102_first_order_projection"]["seconds_per_image"],
        default=None,
    )
    gate_reason = (
        "存在临时软件候选，但每项任务 32 条不足以证明精度等价，且 L2+锚点选择器"
        "属于基线而非论文创新"
        if selected
        else "没有保留率同时满足临时精度门与计算量门"
    )
    gate = {
        "baseline_implementation_pass": True,
        "provisional_smoke_candidate_found": selected is not None,
        "selected_retention": selected["requested_retention"] if selected else None,
        "pl_rtl_integration_pass": False,
        "reason": gate_reason,
    }
    summary = {
        "date": "2026-08-25",
        "scope": "single-pass layer-5 structured 2x2-group pruning; 32 Pointing + 32 Grounding",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "inference_count": 64 * len(RETENTIONS),
        "output_equivalence_audit": output_audit,
        "gate_thresholds": {
            "pck_0_1_min_delta": -0.03125,
            "pck_0_2_min_delta": -0.03125,
            "iou_mean_min_delta": -0.02,
            "acc_0_5_min_delta": -0.03125,
            "weighted_variable_compute_retention_max": 0.90,
            "status": "provisional smoke-set thresholds, not statistical equivalence",
        },
        "gate": gate,
        "results": results,
        "accuracy_delta_vs_r1000": deltas,
        "projection_model": {
            "formula": "625.425*encoder_ratio + 1003.155*LLM_prefill_ratio + 65.498",
            "baseline_measured_seconds": PS_MEASURED_BASELINE_SECONDS,
            "status": "first-order workload projection, not a board measurement",
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    baseline_elapsed = baseline["a6000_median_seconds"]["elapsed_seconds"]
    lines = [
        "# 单遍浅层结构化视觉 Token 剪枝实验",
        "",
        "日期：2026-08-25  ",
        "状态：`448_INFERENCES_COMPLETE / 100%_OUTPUT_EQUIVALENCE_PASS / PL_RTL_HOLD`",
        "",
        "## 1. 实现与审计",
        "",
        "RoboBrain2.5-4B（原始 BF16）视觉编码器第 0–5 层处理完整 patch；在第 5 层后按",
        "2×2 merger group 的特征 L2 范数选块，并在 3×3 自适应分区内保留锚点；第 6–23 层",
        "和 LLM prefill 仅处理保留组。被保留 token 沿用原始视觉 RoPE 与 MRoPE position ID。",
        "选择器只读取运行时特征，不读取标注或答案，且不增加第二遍模型推理。",
        "",
        f"100% 保留路径与既有 stock 路径逐条比较：{output_audit['exact_match_count']}/"
        f"{output_audit['compared_count']} 输出完全一致，mismatch={output_audit['mismatch_count']}。",
        "",
        "## 2. 精度与计算量",
        "",
        "括号内为相对 100% 保留的绝对变化。每项任务仅 32 条，因此 0.03125 即 1 个样本。",
        "",
        "| 请求保留率 | 实际 Token | Encoder MAC | LLM prefill MAC | PCK@0.1 | PCK@0.2 | Mean IoU | Acc@0.5 | PS 投影/s | 投影加速 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        metric_row(result, deltas[tag(result["requested_retention"])]) for result in results
    )
    lines.extend(
        (
            "",
            "## 3. A6000 阶段中位时间",
            "",
            "该自定义逐 token greedy 路径用于功能与阶段计时，不代表 CUDA 生产吞吐；decode 占比大时，",
            "视觉 token 剪枝不会等比例转化为端到端 GPU 加速。",
            "",
            "| 保留率 | Vision 0–5/s | Vision 6–23/s | Vision 总计/s | LLM prefill/s | Decode/s | 端到端/s | 相对 100% |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    lines.extend(timing_row(result, baseline_elapsed) for result in results)
    lines.extend(
        (
            "",
            "## 4. 门控结论",
            "",
            f"临时 smoke gate 候选：{gate['selected_retention'] if selected else '无'}。",
            gate["reason"] + "。",
            "因此本轮不直接写 RTL/HLS 或烧录板卡；先把规则选择器作为可审计软件基线，",
            "下一阶段扩大评测并围绕选择分数、任务约束和可流式硬件映射形成真正的联合创新。",
            "",
            "ZCU102 数值按既有实测阶段 `625.425 s vision + 1003.155 s LLM prefill +",
            "65.498 s fixed` 乘计算量比例得到，只是工作量投影，不是剪枝版板端实测。",
            "",
            "## 5. 复现",
            "",
            f"- 原始结果：`{args.input_dir}`",
            f"- 机器汇总：`{args.summary}`",
            "- 运行器：`evaluation/run_hf_shallow_block_pruning.py`",
            "- 汇总器：`evaluation/build_shallow_block_report.py`",
        )
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
