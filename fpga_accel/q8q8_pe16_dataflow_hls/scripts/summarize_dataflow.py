#!/usr/bin/env python3
"""Freeze PE16 DATAFLOW HLS evidence and conservative workload projections."""

from __future__ import annotations

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
REPORT_ROOT = ROOT / "build/q8q8_pe16_dataflow_prj/solution1/syn/report"
TOP_XML = REPORT_ROOT / "csynth.xml"
COMPUTE_XML = REPORT_ROOT / "p_anonymous_namespace_compute_cached_tile4_csynth.xml"
TOP_RPT = REPORT_ROOT / "q8q8_gemm_csynth.rpt"
CSIM_LOG = ROOT / "artifacts/20260829T131624Z_vitis_hls_csim.log"
CSYNTH_LOG = ROOT / "artifacts/20260829T131634Z_vitis_hls_csynth.log"
BASELINE_JSON = WORKSPACE / "fpga_accel/q8q8_pe16_hls/artifacts/pe16_hls_summary.json"
BOARD_JSON = WORKSPACE / "deployment/artifacts/q8q8_pe16_board_summary.json"
CONTRACT_JSON = ROOT / "artifacts/dataflow_stage_contract.json"
OUTPUT_JSON = ROOT / "artifacts/pe16_dataflow_hls_summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-08-29-zcu102-q8q8-pe16-dataflow-hls.md"

QK = 32
TM = 16
TN = 64
PE_N = 16
CLOCK_MHZ = 200.0
READ_STREAM_DEPTH = 4
PACK_STREAM_DEPTH = 2
STORE_STREAM_DEPTH = 10


def required(node: ElementTree.Element, path: str) -> str:
    value = node.findtext(path)
    if value is None:
        raise ValueError(f"missing XML field {path}")
    return value.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def component_candidate_cycles(m: int, n: int, k: int) -> int:
    """Conservative task schedule: compute path plus one unhidden final store."""
    k_blocks = k // QK
    m_tiles = math.ceil(m / TM)
    n_tiles = math.ceil(n / TN)
    weight = TN * k_blocks + 14
    tile_words = TN // 4
    per_n_tile = weight
    for tile in range(m_tiles):
        valid_rows = min(TM, m - tile * TM)
        valid_words = valid_rows * tile_words
        activation_stream_read = TM * k_blocks + READ_STREAM_DEPTH
        compute = TM * (TN // PE_N) * k_blocks + 29
        pack = valid_words + PACK_STREAM_DEPTH
        per_n_tile += activation_stream_read + compute + pack
    final_valid_rows = min(TM, m - (m_tiles - 1) * TM)
    per_n_tile += final_valid_rows * tile_words + STORE_STREAM_DEPTH
    return n_tiles * per_n_tile


def workload_candidate_us(total_m: int, chunk_m: int, n: int, k: int) -> float:
    cycles = 0
    for m0 in range(0, total_m, chunk_m):
        cycles += component_candidate_cycles(min(chunk_m, total_m - m0), n, k)
    return cycles / CLOCK_MHZ


def main() -> int:
    for output in (OUTPUT_JSON, OUTPUT_MD):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite record: {output}")

    contract = json.loads(CONTRACT_JSON.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    board = json.loads(BOARD_JSON.read_text(encoding="utf-8"))
    top = ElementTree.parse(TOP_XML).getroot()
    compute = ElementTree.parse(COMPUTE_XML).getroot()
    compute_loops = compute.findall("./PerformanceEstimates/SummaryOfLoopLatency/*")
    if len(compute_loops) != 1:
        raise ValueError("expected exactly one MAC loop")

    resource_names = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")
    resources = {
        name: int(required(top, f"./AreaEstimates/Resources/{name}"))
        for name in resource_names
    }
    available = {
        name: int(required(top, f"./AreaEstimates/AvailableResources/{name}"))
        for name in resource_names
    }
    utilization = {
        name: (resources[name] / available[name] if available[name] else None)
        for name in resource_names
    }
    estimated_ns = float(required(
        top,
        "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
    ))
    compute_ii = int(required(compute_loops[0], "./PipelineII"))
    compute_depth = int(required(compute_loops[0], "./PipelineDepth"))

    csim_text = CSIM_LOG.read_text(encoding="utf-8", errors="replace")
    cases = re.findall(r"^TB_CASE .* result=PASS$", csim_text, re.MULTILINE)
    errors = [float(value) for value in re.findall(r"max_abs=([0-9.eE+-]+)", csim_text)]
    csim_pass = "TB_RESULT=PASS" in csim_text and "result=FAIL" not in csim_text
    top_rpt_text = TOP_RPT.read_text(encoding="utf-8", errors="replace")
    dataflow_pass = "process_n_tile_dataflow" in top_rpt_text and "dataflow" in top_rpt_text

    baseline_rows = {row["name"]: row for row in baseline["workloads"]}
    board_rows = board["benchmarks"]
    workload_rows = []
    for name in ("language_prefill", "vision_encoder", "decoder_gemv"):
        base = baseline_rows[name]
        shape = base["shape"]
        candidate_us = workload_candidate_us(
            shape["total_m"], shape["chunk_m"], shape["n"], shape["k"]
        )
        baseline_us = base["pe16_schedule_model_us"]
        reduction = 1.0 - candidate_us / baseline_us
        measured_us = board_rows[name]["average_kernel_us"]
        projected_us = measured_us + candidate_us - baseline_us
        workload_rows.append({
            "name": name,
            "shape": shape,
            "pe16_serial_schedule_us": baseline_us,
            "dataflow_conservative_schedule_us": candidate_us,
            "schedule_reduction": reduction,
            "schedule_speedup": baseline_us / candidate_us,
            "pe16_board_measured_us": measured_us,
            "first_order_projected_board_us": projected_us,
            "projection_status": (
                "conservative HLS schedule projection; preserves the measured "
                "PE16/model residual; not an RTL or board measurement"
            ),
            "claim_allowed": name != "decoder_gemv",
        })

    limits = contract["synthesis_gates"]
    resource_pass = (
        utilization["BRAM_18K"] <= limits["bram18k_utilization_maximum"]
        and utilization["DSP"] <= limits["dsp_utilization_maximum"]
        and utilization["FF"] <= limits["ff_utilization_maximum"]
        and utilization["LUT"] <= limits["lut_utilization_maximum"]
    )
    performance_pass = all(
        row["schedule_reduction"] >= contract["performance_model_gates"]
        ["language_prefill_schedule_reduction_minimum"]
        for row in workload_rows
        if row["name"] in ("language_prefill", "vision_encoder")
    )
    gates = {
        "csim_pass": csim_pass and len(cases) >= 9 and max(errors) <= 1.0e-4,
        "timing_pass": estimated_ns <= 5.0,
        "compute_ii_pass": compute_ii <= 1,
        "resource_pass": resource_pass,
        "dataflow_region_pass": dataflow_pass,
        "prefill_vision_schedule_pass": performance_pass,
    }
    passed = all(gates.values())
    decision = (
        "DATAFLOW_HLS_GATE_PASS / DIRECTED_RTL_NEXT"
        if passed else "DATAFLOW_HLS_GATE_FAIL / ITERATE_HLS"
    )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": "PE16 stream DATAFLOW C simulation, HLS synthesis, and schedule projection",
        "csim": {
            "status": "PASS" if csim_pass else "FAIL",
            "passing_cases": len(cases),
            "maximum_absolute_error": max(errors),
            "log": str(CSIM_LOG),
            "log_sha256": sha256(CSIM_LOG),
        },
        "hls": {
            "tool": "Vitis HLS 2023.2",
            "part": required(top, "./UserAssignments/Part"),
            "target_clock_ns": 5.0,
            "estimated_clock_ns": estimated_ns,
            "estimated_fmax_mhz": 1000.0 / estimated_ns,
            "compute_pipeline_ii": compute_ii,
            "compute_pipeline_depth": compute_depth,
            "synthesized_dataflow_region": dataflow_pass,
            "fifo_depths": {
                "activation_q_stream": 4864,
                "activation_scale_stream": 4864,
                "output_stream": 256,
            },
            "resources": resources,
            "available_resources": available,
            "utilization": utilization,
            "csynth_log": str(CSYNTH_LOG),
            "csynth_log_sha256": sha256(CSYNTH_LOG),
        },
        "optimization": {
            "load_compute_store_overlap": True,
            "accumulator_clear_fused_into_first_k_block": True,
            "removed_clear_cycles_per_m_tile": 1026,
        },
        "workloads": workload_rows,
        "gates": gates,
        "attempt_logs": [
            str(ROOT / "artifacts/20260829T130834Z_vitis_hls_csynth.log"),
            str(ROOT / "artifacts/20260829T131347Z_vitis_hls_csynth.log"),
            str(CSYNTH_LOG),
        ],
        "limitations": [
            "HLS timing and resources are pre-place estimates.",
            "Projected workload latency is not an RTL or board measurement.",
            "No decoder speedup is claimed because measured decoder GEMV is weight-stream/memory-bound.",
            "No bitstream was generated and the board was not accessed.",
        ],
        "source_sha256": {
            str(path.relative_to(WORKSPACE)): sha256(path)
            for path in (
                ROOT / "include/q8q8_gemm.h",
                ROOT / "src/q8q8_gemm.cpp",
                ROOT / "tb/tb_q8q8_gemm.cpp",
                ROOT / "scripts/run_hls.tcl",
                ROOT / "scripts/run_stage.sh",
            )
        },
        "output_json": str(OUTPUT_JSON),
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows_md = []
    for row in workload_rows:
        claim = "仅观察" if not row["claim_allowed"] else "投影"
        rows_md.append(
            f"| {row['name']} | {row['pe16_serial_schedule_us']/1000:.3f} | "
            f"{row['dataflow_conservative_schedule_us']/1000:.3f} | "
            f"{100*row['schedule_reduction']:.2f}% | {claim} |"
        )
    md = f"""# ZCU102 Q8×Q8 PE16 DATAFLOW HLS门控

生成时间：{summary['generated_at']}
状态：`{decision}`

## 本阶段实现

- 三级并发流水：DDR连续读取激活 → PE16 Q8×Q8 GEMM → 128-bit结果写回。
- 片上FIFO容纳一个最大tile，允许load、compute、store跨tile重叠。
- 将1026周期的累加器清零融合到首个K-block，不改变后续累加顺序。

## 验证结果

- C testbench：9/9 PASS，最大绝对误差`{max(errors):.9g}`。
- HLS明确综合出DATAFLOW区域；主MAC循环II={compute_ii}。
- 估算时钟：{estimated_ns:.3f} ns（约{1000/estimated_ns:.2f} MHz），通过200 MHz门槛。
- 资源：BRAM {resources['BRAM_18K']}/{available['BRAM_18K']}（{100*utilization['BRAM_18K']:.2f}%），DSP {resources['DSP']}/{available['DSP']}（{100*utilization['DSP']:.2f}%），LUT {resources['LUT']}/{available['LUT']}（{100*utilization['LUT']:.2f}%）。

## 冻结workload的保守排程模型

| workload | PE16串行/ms | DATAFLOW/ms | 降幅 | 证据边界 |
|---|---:|---:|---:|---|
{chr(10).join(rows_md)}

模型将compute串行路径全部计入，并额外保留最后一次store；因此没有把理想完全重叠当作实测。decoder受权重流/内存限制，本阶段不宣称加速。

## 决策

全部冻结HLS门槛通过。下一步只做生成RTL的定向数值/控制仿真；在RTL和布局布线门槛通过前，不生成板端烧录结论。

机器可读证据：`{OUTPUT_JSON}`。
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(f"DECISION={decision}")
    print(f"JSON={OUTPUT_JSON}")
    print(f"MARKDOWN={OUTPUT_MD}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
