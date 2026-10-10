#!/usr/bin/env python3
"""Summarize PE8/PE16 HLS reports and project the frozen board workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
BASELINE_ROOT = WORKSPACE / "fpga_accel" / "q8q8_output_burst_hls"
DEFAULT_BASELINE_XML = (
    BASELINE_ROOT
    / "build/q8q8_output_burst_prj/solution1/syn/report/csynth.xml"
)
DEFAULT_BASELINE_COMPUTE_XML = (
    BASELINE_ROOT
    / "build/q8q8_output_burst_prj/solution1/syn/report/"
    "p_anonymous_namespace_compute_cached_tile_csynth.xml"
)
DEFAULT_CANDIDATE_XML = (
    ROOT / "build/q8q8_pe16_prj/solution1/syn/report/csynth.xml"
)
DEFAULT_CANDIDATE_COMPUTE_XML = (
    ROOT
    / "build/q8q8_pe16_prj/solution1/syn/report/"
    "p_anonymous_namespace_compute_cached_tile_csynth.xml"
)
DEFAULT_CSIM_LOG = ROOT / "artifacts/vitis_hls_csynth.log"
DEFAULT_BOARD_LOG = (
    WORKSPACE / "reports/2026-08-25-zcu102-q8q8-v3-uart.log"
)

QK = 32
TM = 16
TN = 64
CLOCK_MHZ = 200
RESOURCE_KEYS = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
WORKLOADS = (
    ("language_prefill", 512, 512, 9728, 2560),
    ("vision_encoder", 4320, 512, 4096, 1024),
    ("decoder_gemv", 1, 1, 9728, 2560),
)


def required_text(node: ElementTree.Element, path: str) -> str:
    value = node.findtext(path)
    if value is None:
        raise ValueError(f"missing XML field: {path}")
    return value.strip()


def parse_hls_xml(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    resources = {
        key: int(required_text(root, f"./AreaEstimates/Resources/{key}"))
        for key in RESOURCE_KEYS
    }
    available = {
        key: int(required_text(root, f"./AreaEstimates/AvailableResources/{key}"))
        for key in RESOURCE_KEYS
    }
    utilization = {
        key: (resources[key] / available[key] if available[key] else None)
        for key in RESOURCE_KEYS
    }
    return {
        "path": str(path),
        "tool_version": required_text(root, "./ReportVersion/Version"),
        "part": required_text(root, "./UserAssignments/Part"),
        "target_clock_period_ns": float(
            required_text(root, "./UserAssignments/TargetClockPeriod")
        ),
        "estimated_clock_period_ns": float(
            required_text(
                root,
                "./PerformanceEstimates/SummaryOfTimingAnalysis/"
                "EstimatedClockPeriod",
            )
        ),
        "resources": resources,
        "available_resources": available,
        "utilization": utilization,
    }


def parse_compute_xml(path: Path) -> dict[str, Any]:
    root = ElementTree.parse(path).getroot()
    loops = root.findall("./PerformanceEstimates/SummaryOfLoopLatency/*")
    if len(loops) != 1:
        raise ValueError(f"expected one compute loop in {path}, found {len(loops)}")
    loop = loops[0]
    return {
        "path": str(path),
        "loop_name": loop.tag,
        "pipeline_ii": int(required_text(loop, "./PipelineII")),
        "pipeline_depth": int(required_text(loop, "./PipelineDepth")),
        "estimated_clock_period_ns": float(
            required_text(
                root,
                "./PerformanceEstimates/SummaryOfTimingAnalysis/"
                "EstimatedClockPeriod",
            )
        ),
    }


def component_model_us(
    m: int,
    n: int,
    k: int,
    pe_n: int,
    compute_ii: int = 1,
    compute_depth: int = 25,
) -> float:
    if TN % pe_n:
        raise ValueError(f"TN={TN} is not divisible by PE_N={pe_n}")
    m_tiles = math.ceil(m / TM)
    n_tiles = math.ceil(n / TN)
    k_blocks = k // QK
    weight_cycles = TN * k_blocks + 14
    activation_cycles = TM * k_blocks + 14
    compute_cycles = (
        TM * (TN // pe_n) * k_blocks * compute_ii + compute_depth
    )
    init_cycles = 1026
    store_cycles = 1034
    cycles = n_tiles * (
        weight_cycles
        + m_tiles
        * (
            activation_cycles
            + init_cycles
            + compute_cycles
            + store_cycles
        )
    )
    return cycles / CLOCK_MHZ


def workload_model_us(
    total_m: int,
    chunk_m: int,
    n: int,
    k: int,
    pe_n: int,
    compute_ii: int = 1,
    compute_depth: int = 25,
) -> float:
    total = 0.0
    for m0 in range(0, total_m, chunk_m):
        total += component_model_us(
            min(chunk_m, total_m - m0),
            n,
            k,
            pe_n,
            compute_ii,
            compute_depth,
        )
    return total


def parse_board_results(path: Path) -> dict[str, dict[str, float]]:
    pattern = re.compile(
        r"BENCH_RESULT case=(?P<name>\S+).*?"
        r"average_kernel_us=(?P<kernel>[0-9.]+) "
        r"component_model_us=(?P<model>[0-9.]+).*?"
        r"effective_gops=(?P<gops>[0-9.]+)"
    )
    results: dict[str, dict[str, float]] = {}
    for match in pattern.finditer(path.read_text(encoding="utf-8")):
        results.setdefault(
            match.group("name"),
            {
                "average_kernel_us": float(match.group("kernel")),
                "component_model_us": float(match.group("model")),
                "effective_gops": float(match.group("gops")),
            },
        )
    missing = sorted({row[0] for row in WORKLOADS} - results.keys())
    if missing:
        raise ValueError(f"missing board results in {path}: {missing}")
    return results


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_workload_rows(
    board_results: dict[str, dict[str, float]],
    candidate_ii: int,
    candidate_depth: int,
) -> list[dict[str, Any]]:
    rows = []
    for name, total_m, chunk_m, n, k in WORKLOADS:
        pe8_us = workload_model_us(total_m, chunk_m, n, k, 8)
        pe16_ideal_us = workload_model_us(total_m, chunk_m, n, k, 16)
        pe16_us = workload_model_us(
            total_m,
            chunk_m,
            n,
            k,
            16,
            candidate_ii,
            candidate_depth,
        )
        measured_us = board_results[name]["average_kernel_us"]
        fixed_overhead_us = measured_us - pe8_us
        projected_us = fixed_overhead_us + pe16_us
        rows.append(
            {
                "name": name,
                "shape": {
                    "total_m": total_m,
                    "chunk_m": chunk_m,
                    "n": n,
                    "k": k,
                },
                "pe8_schedule_model_us": pe8_us,
                "pe16_ideal_ii1_schedule_model_us": pe16_ideal_us,
                "pe16_schedule_model_us": pe16_us,
                "schedule_only_speedup": pe8_us / pe16_us,
                "v3_board_measured_us": measured_us,
                "v3_model_residual_us": fixed_overhead_us,
                "pe16_first_order_projected_us": projected_us,
                "first_order_speedup_vs_v3_board": measured_us / projected_us,
                "projection_status": (
                    "first-order projection; keeps the measured PE8/model "
                    "residual fixed; not a PE16 board measurement"
                ),
            }
        )
    return rows


def render_markdown(summary: dict[str, Any]) -> str:
    base = summary["baseline_pe8"]
    cand = summary["candidate_pe16"]
    gate = summary["gate"]
    lines = [
        "# ZCU102 Q8×Q8 PE_N=16 HLS 综合评估",
        "",
        f"生成时间：{summary['generated_at']}",
        f"状态：`{gate['decision']}`",
        "",
        "## 验证边界",
        "",
        "- PE8为已上板v3参考；PE16当前只有C仿真和HLS综合证据。",
        "- HLS资源和时钟为pre-place估算，不是Vivado routed结果。",
        "- PE16延迟是一阶投影，不是板端实测；完整模型端到端延迟未在本阶段测量。",
        "",
        "## C仿真与综合门控",
        "",
        f"- C testbench：`{'PASS' if gate['csim_pass'] else 'FAIL'}`，"
        f"{summary['csim']['passing_cases']}个有效用例通过。",
        f"- 主计算循环II：PE8={base['compute_loop']['pipeline_ii']}，"
        f"PE16={cand['compute_loop']['pipeline_ii']}。",
        f"- PE16估算时钟周期：{cand['estimated_clock_period_ns']:.3f} ns，"
        f"200 MHz门控={'通过' if gate['timing_pass'] else '失败'}。",
        "",
        "## HLS资源对比",
        "",
        "| 资源 | PE8 | PE16 | 变化 | PE16占ZCU102 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in RESOURCE_KEYS:
        base_value = base["resources"][key]
        cand_value = cand["resources"][key]
        utilization = cand["utilization"][key]
        utilization_text = (
            "N/A" if utilization is None else f"{100.0 * utilization:.2f}%"
        )
        lines.append(
            f"| {key} | {base_value} | {cand_value} | "
            f"{cand_value - base_value:+d} | {utilization_text} |"
        )
    lines += [
        "",
        "## 固定板端workload投影",
        "",
        "| workload | PE8排程模型/ms | PE16排程模型/ms | 排程加速 | "
        "v3板测/ms | PE16一阶投影/ms | 一阶加速 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["workloads"]:
        lines.append(
            f"| {row['name']} | "
            f"{row['pe8_schedule_model_us'] / 1000.0:.3f} | "
            f"{row['pe16_schedule_model_us'] / 1000.0:.3f} | "
            f"{row['schedule_only_speedup']:.3f}× | "
            f"{row['v3_board_measured_us'] / 1000.0:.3f} | "
            f"{row['pe16_first_order_projected_us'] / 1000.0:.3f} | "
            f"{row['first_order_speedup_vs_v3_board']:.3f}× |"
        )
    lines += [
        "",
        "## 决策",
        "",
        gate["explanation_zh"],
        "",
        "机器可读证据："
        f"`{summary['output_json']}`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-xml", type=Path, default=DEFAULT_BASELINE_XML)
    parser.add_argument(
        "--baseline-compute-xml",
        type=Path,
        default=DEFAULT_BASELINE_COMPUTE_XML,
    )
    parser.add_argument("--candidate-xml", type=Path, default=DEFAULT_CANDIDATE_XML)
    parser.add_argument(
        "--candidate-compute-xml",
        type=Path,
        default=DEFAULT_CANDIDATE_COMPUTE_XML,
    )
    parser.add_argument("--csim-log", type=Path, default=DEFAULT_CSIM_LOG)
    parser.add_argument("--board-log", type=Path, default=DEFAULT_BOARD_LOG)
    parser.add_argument(
        "--source-metadata-json",
        type=Path,
        help="Reuse frozen source hashes when re-rendering an earlier attempt.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/pe16_hls_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=WORKSPACE / "reports/2026-08-27-zcu102-q8q8-pe16-hls.md",
    )
    args = parser.parse_args()

    baseline = parse_hls_xml(args.baseline_xml)
    baseline["compute_loop"] = parse_compute_xml(args.baseline_compute_xml)
    candidate = parse_hls_xml(args.candidate_xml)
    candidate["compute_loop"] = parse_compute_xml(args.candidate_compute_xml)

    csim_text = args.csim_log.read_text(encoding="utf-8", errors="replace")
    passing_cases = len(re.findall(r"TB_CASE .* result=PASS", csim_text))
    failing_cases = len(re.findall(r"TB_CASE .* result=FAIL", csim_text))
    csim_pass = "TB_RESULT=PASS" in csim_text and failing_cases == 0
    timing_pass = candidate["estimated_clock_period_ns"] <= 5.0
    ii_pass = candidate["compute_loop"]["pipeline_ii"] == 1
    resource_pass = all(
        value is None or value <= 0.80
        for value in candidate["utilization"].values()
    )
    proceed = csim_pass and timing_pass and ii_pass and resource_pass
    if proceed:
        decision = "HLS_GATE_PASS / VIVADO_IMPLEMENTATION_NEXT"
        explanation_zh = (
            "PE16通过C数值、主循环II=1、200 MHz估算和80%资源上限门控。"
            "下一步可创建独立Vivado实现分支，必须以布局布线时序和系统资源决定"
            "是否上板；当前不能声称PE16已经实现或获得投影中的加速。"
        )
    else:
        decision = "HLS_GATE_FAIL / DATAFLOW_NEXT"
        explanation_zh = (
            "PE16未同时通过C数值、II=1、200 MHz估算和80%资源门控。"
            "停止直接扩PE，下一步转入双缓冲DATAFLOW，不创建板端实现。"
        )

    source_files = [
        ROOT / "include/q8q8_gemm.h",
        ROOT / "src/q8q8_gemm.cpp",
        ROOT / "tb/tb_q8q8_gemm.cpp",
        ROOT / "scripts/run_hls.tcl",
        ROOT / "scripts/summarize_hls.py",
    ]
    if args.source_metadata_json:
        frozen = json.loads(
            args.source_metadata_json.read_text(encoding="utf-8")
        )
        source_sha256 = frozen["inputs"]["source_sha256"]
    else:
        source_sha256 = {
            str(path.relative_to(WORKSPACE)): sha256(path)
            for path in source_files
        }
    output_json = args.output_json.resolve()
    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "PE16 HLS pre-place evaluation; no PE16 Vivado or board result",
        "baseline_pe8": baseline,
        "candidate_pe16": candidate,
        "resource_delta_pe16_minus_pe8": {
            key: candidate["resources"][key] - baseline["resources"][key]
            for key in RESOURCE_KEYS
        },
        "csim": {
            "log": str(args.csim_log),
            "passing_cases": passing_cases,
            "failing_cases": failing_cases,
            "tb_result_pass": csim_pass,
        },
        "workloads": make_workload_rows(
            parse_board_results(args.board_log),
            candidate["compute_loop"]["pipeline_ii"],
            candidate["compute_loop"]["pipeline_depth"],
        ),
        "gate": {
            "csim_pass": csim_pass,
            "timing_pass": timing_pass,
            "compute_ii_pass": ii_pass,
            "resource_under_80_percent_pass": resource_pass,
            "decision": decision,
            "explanation_zh": explanation_zh,
        },
        "inputs": {
            "board_log": str(args.board_log),
            "board_log_sha256": sha256(args.board_log),
            "source_sha256": source_sha256,
            "report_generator_sha256": sha256(
                ROOT / "scripts/summarize_hls.py"
            ),
        },
        "output_json": str(output_json),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary["gate"], ensure_ascii=False))
    print(f"JSON={output_json}")
    print(f"MARKDOWN={args.output_md.resolve()}")
    return 0 if proceed else 1


if __name__ == "__main__":
    raise SystemExit(main())
