#!/usr/bin/env python3
"""Decision analysis for the completed VSR zero-shot dev budget curve."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.project_current_selector_heterogeneous_latency import (
    load_accepted_cycle_model,
    load_transfer_model,
    project_sample,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "artifacts/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve"
SUMMARY = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve.json"
OUTPUT_JSON = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-outcome-analysis.json"
OUTPUT_MD = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-outcome-analysis.md"
HLS_REPORT = ROOT / "reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean_ms(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return 1000.0 * statistics.fmean(float(row[key]) for row in rows)


def project_k96(v2_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cycle_model = load_accepted_cycle_model()
    transfer = load_transfer_model()
    projected = []
    for row in v2_rows:
        sample = {
            "sample_id": row["sample_id"], "task": row["task"], "source": "VSR",
            "group_count": row["full_llm_visual_tokens"],
            "grid_h": int(row["image_grid_thw"][1]) // 2,
            "grid_w": int(row["image_grid_thw"][2]) // 2,
            "model_inference_seconds": row["model_inference_seconds"],
            "selector_software_seconds": row["selector_cpu_roundtrip_seconds"],
            "application_end_to_end_seconds": row["application_end_to_end_seconds"],
        }
        projected.append(project_sample(96, sample, cycle_model, transfer))

    def mean(key: str) -> float:
        return statistics.fmean(float(row[key]) for row in projected)

    return {
        "scope": "pure analytic projection; no Jetson, ZCU102, link, bitstream, or board measurement",
        "accepted_hls_budget_limit": 96,
        "sample_count": len(projected),
        "mean_group_count": statistics.fmean(int(row["group_count"]) for row in projected),
        "identity_bypass_count": sum(bool(row["identity_bypass"]) for row in projected),
        "mean_transfer_bytes": mean("transfer_bytes"),
        "mean_fpga_compute_ms": 1000.0 * mean("fpga_compute_seconds"),
        "mean_transfer_1gbe_110MBps_ms": 1000.0 * mean("transfer_1gbe_110MBps_seconds"),
        "mean_transfer_10gbe_assumed_1p1GBps_ms": 1000.0 * mean("transfer_10gbe_1p1GBps_assumed_seconds"),
        "projected_model_1gbe_ms": 1000.0 * mean("projected_model_e2e_1gbe_110MBps_seconds"),
        "projected_model_10gbe_ms": 1000.0 * mean("projected_model_e2e_10gbe_1p1GBps_assumed_seconds"),
        "projected_application_1gbe_ms": 1000.0 * mean("projected_application_e2e_1gbe_110MBps_seconds"),
        "projected_application_10gbe_ms": 1000.0 * mean("projected_application_e2e_10gbe_1p1GBps_assumed_seconds"),
        "assumptions": [
            "Wv remains resident on FPGA and is not transferred per image.",
            "Jetson-to-FPGA, FPGA compute, and FPGA-to-Jetson execute serially without overlap.",
            "1 GbE uses 110 MB/s effective payload; 10 GbE uses an unmeasured 1.1 GB/s assumption.",
            "Accepted compact HLS formula and 200 MHz target are synthesis-derived estimates, not board measurements.",
        ],
    }


def analyze() -> dict[str, Any]:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    dense = load_jsonl(RAW_DIR / "dense_edge576.jsonl")
    v2_k96 = load_jsonl(RAW_DIR / "v2_edge576_b1_k96.jsonl")
    l2_k96 = load_jsonl(RAW_DIR / "corrected_l2_integer_b1_k96.jsonl")
    v2_k144 = load_jsonl(RAW_DIR / "v2_edge576_b1_k144.jsonl")
    dense_correct = sum(bool(row["correct"]) for row in dense)
    k96_correct = sum(bool(row["correct"]) for row in v2_k96)
    required_correct = int(0.88 * dense_correct + 0.999999999)
    projection = project_k96(v2_k96)
    dense_model_ms = mean_ms(dense, "elapsed_seconds")
    projection["dense_model_inference_ms"] = dense_model_ms
    projection["speedup_vs_dense_model_1gbe"] = dense_model_ms / projection["projected_model_1gbe_ms"]
    projection["speedup_vs_dense_model_10gbe_assumed"] = dense_model_ms / projection["projected_model_10gbe_ms"]
    projection["zero_transfer_projected_model_ms"] = (
        mean_ms(v2_k96, "model_inference_seconds")
        - mean_ms(v2_k96, "selector_cpu_roundtrip_seconds")
        + projection["mean_fpga_compute_ms"]
    )
    projection["zero_transfer_speedup_vs_dense_model"] = dense_model_ms / projection["zero_transfer_projected_model_ms"]

    l2_by_id = {row["sample_id"]: row for row in l2_k96}
    transitions = {
        "l2_wrong_v2_correct": sum(not bool(l2_by_id[row["sample_id"]]["correct"]) and bool(row["correct"]) for row in v2_k96),
        "l2_correct_v2_wrong": sum(bool(l2_by_id[row["sample_id"]]["correct"]) and not bool(row["correct"]) for row in v2_k96),
    }
    return {
        "status": "CURRENT_K96_NEAR_QUALITY_GATE_BUT_NO_NET_1GBE_BENEFIT",
        "scope": "VSR zero-shot dev only; not external test confirmation",
        "quality": {
            "dense_correct": dense_correct,
            "dense_accuracy": summary["arms"]["dense_edge576"]["accuracy"],
            "v2_k96_correct": k96_correct,
            "v2_k96_accuracy": summary["arms"]["v2_edge576_b1_k96"]["accuracy"],
            "v2_k96_relative_to_dense": summary["comparisons_vs_dense"]["v2_edge576_b1_k96"]["accuracy_retention_vs_dense"],
            "permissive_development_screen": 0.88,
            "minimum_correct_for_screen": required_correct,
            "correct_example_deficit": required_correct - k96_correct,
            "v2_k96_minus_l2_k96": summary["v2_minus_l2_same_budget"]["96"],
            "v2_k96_l2_transitions": transitions,
            "v2_k144_relative_to_dense": summary["comparisons_vs_dense"]["v2_edge576_b1_k144"]["accuracy_retention_vs_dense"],
            "v2_k144_hls_compatible": False,
            "v2_k144_hls_incompatibility": "accepted compact HLS interface requires K<=96",
        },
        "latency": projection,
        "decision": {
            "current_k96_promote": False,
            "reason": "K96 misses the permissive quality screen by four correct examples and projected 1 GbE inference is slower than Dense.",
            "extend_k144_hardware_now": False,
            "reason_k144": "K144 passes the quality screen, but v2-L2 improvement is not statistically established and the accepted HLS core does not support K144.",
            "next_primary_work": "Improve relation-preserving, prompt-conditioned K96 selection on VSR zero-shot train while keeping the existing K96 hardware contract fixed.",
            "next_hardware_work": "Do not widen K yet; first make K96 quality pass. In parallel, treat inter-board transfer as a first-class architecture constraint and prefer PCIe/shared-memory or move the cut point to avoid a 1 GbE round trip.",
        },
        "inputs": {
            "summary": {"path": str(SUMMARY), "sha256": sha256_file(SUMMARY)},
            "dense": {"path": str(RAW_DIR / "dense_edge576.jsonl"), "sha256": sha256_file(RAW_DIR / "dense_edge576.jsonl")},
            "l2_k96": {"path": str(RAW_DIR / "corrected_l2_integer_b1_k96.jsonl"), "sha256": sha256_file(RAW_DIR / "corrected_l2_integer_b1_k96.jsonl")},
            "v2_k96": {"path": str(RAW_DIR / "v2_edge576_b1_k96.jsonl"), "sha256": sha256_file(RAW_DIR / "v2_edge576_b1_k96.jsonl")},
            "v2_k144": {"path": str(RAW_DIR / "v2_edge576_b1_k144.jsonl"), "sha256": sha256_file(RAW_DIR / "v2_edge576_b1_k144.jsonl")},
            "accepted_hls_report": {"path": str(HLS_REPORT), "sha256": sha256_file(HLS_REPORT)},
        },
    }


def render(report: Mapping[str, Any]) -> str:
    q = report["quality"]
    l = report["latency"]
    return "\n".join(
        [
            "# VSR Zero-Shot Dev 结果与下一步决策", "",
            "本报告只使用 zero-shot dev；test 未打开。FPGA 与链路数值均为分析投影，不是上板测量。", "",
            "## 结论", "",
            f"- Dense：{q['dense_accuracy']:.4f}（{q['dense_correct']}/340）。",
            f"- v2 K96：{q['v2_k96_accuracy']:.4f}，相对 Dense {q['v2_k96_relative_to_dense']:.4f}；达到 0.88 宽松开发线至少需 {q['minimum_correct_for_screen']} 条正确，当前差 {q['correct_example_deficit']} 条。",
            f"- v2 K96 相对 L2 K96：{q['v2_k96_minus_l2_k96']['observed_difference']:+.4f}，95% CI [{q['v2_k96_minus_l2_k96']['ci_95_lower']:+.4f}, {q['v2_k96_minus_l2_k96']['ci_95_upper']:+.4f}]，尚未证明稳定优于 L2。",
            f"- v2 K144 相对 Dense 为 {q['v2_k144_relative_to_dense']:.4f}，质量线通过，但当前已接受 HLS 接口只支持 K<=96。", "",
            "## 异构时延投影", "",
            f"- 本轮 Dense 模型推理均值：{l['dense_model_inference_ms']:.2f} ms。",
            f"- v2 K96 FPGA 计算估计：{l['mean_fpga_compute_ms']:.2f} ms；平均双向载荷 {l['mean_transfer_bytes']/1e6:.3f} MB。",
            f"- 1 GbE（110 MB/s）串行投影：{l['projected_model_1gbe_ms']:.2f} ms，相对 Dense {l['speedup_vs_dense_model_1gbe']:.3f}×。",
            f"- 假设 10 GbE（1.1 GB/s）串行投影：{l['projected_model_10gbe_ms']:.2f} ms，相对 Dense {l['speedup_vs_dense_model_10gbe_assumed']:.3f}×。",
            f"- 理想零传输投影：{l['zero_transfer_projected_model_ms']:.2f} ms，相对 Dense {l['zero_transfer_speedup_vs_dense_model']:.3f}×。", "",
            "投影公式为：实测 v2 模型时间 - 实测 CPU Selector 往返 + HLS 计算估计 + 双向传输。Wv 假定常驻 FPGA，阶段严格串行。10 GbE 未实测，所有 HLS 数值来自 csynth 周期公式。", "",
            "## 下一步", "",
            "1. 固定 K96、3x3 anchor、INT8/整数接口和输出布局，不扩大硬件预算。",
            "2. 使用 VSR zero-shot train 发展关系保持选择目标，重点修复仅差 4 条的质量缺口，并要求同预算相对 L2 的配对区间不再跨 0。",
            "3. 把通信作为架构约束：1 GbE 往返没有净收益，优先验证 PCIe/共享内存，或调整切点以避免把完整 block0 特征往返传输。",
            "4. 方法冻结前不打开 VSR zero-shot test；当前结果不能用于外部泛化或板端加速声明。", "",
        ]
    )


def main() -> int:
    report = analyze()
    OUTPUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "json": str(OUTPUT_JSON), "markdown": str(OUTPUT_MD)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
