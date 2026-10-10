#!/usr/bin/env python3
"""Freeze the six-stream Decoder GEMV HLS and RTL-cosim evidence."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
REPORT_ROOT = ROOT / "build/q8q8_gemv_stream6_prj/solution1/syn/report"
TOP_XML = REPORT_ROOT / "csynth.xml"
TOP_RPT = REPORT_ROOT / "csynth.rpt"
LOAD_XML = REPORT_ROOT / (
    "p_anonymous_namespace_load_weight_records_"
    "Pipeline_VITIS_LOOP_122_1_csynth.xml"
)
COMPUTE_XML = REPORT_ROOT / (
    "p_anonymous_namespace_compute_record_streams_"
    "Pipeline_VITIS_LOOP_165_1_csynth.xml"
)
ACTIVATION_XML = REPORT_ROOT / "p_anonymous_namespace_load_activation_csynth.xml"
PARTITION_JSON = (
    WORKSPACE
    / "fpga_accel/heterogeneous_partition/artifacts/partition_gate_20260901/summary.json"
)
OUTPUT_DIR = ROOT / "artifacts/stream6_hls_gate_20260901"
OUTPUT_JSON = OUTPUT_DIR / "summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-09-01-q8q8-gemv-stream6-hls.md"
RESOURCE_KEYS = ("BRAM_18K", "DSP", "FF", "LUT", "URAM")


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


def only_loop(path: Path) -> dict[str, int | str]:
    root = ElementTree.parse(path).getroot()
    loops = root.findall("./PerformanceEstimates/SummaryOfLoopLatency/*")
    if len(loops) != 1:
        raise ValueError(f"expected one loop in {path}, found {len(loops)}")
    loop = loops[0]
    return {
        "name": loop.tag,
        "pipeline_ii": int(required(loop, "./PipelineII")),
        "pipeline_depth": int(required(loop, "./PipelineDepth")),
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite frozen stream6 evidence")
    cosim_logs = sorted((ROOT / "artifacts").glob("*_vitis_hls_cosim.log"))
    if not cosim_logs:
        raise FileNotFoundError("no RTL co-simulation log found")
    cosim_log = cosim_logs[-1]
    cosim_text = cosim_log.read_text(encoding="utf-8", errors="replace")
    top_text = TOP_RPT.read_text(encoding="utf-8", errors="replace")
    top = ElementTree.parse(TOP_XML).getroot()

    resources = {
        key: int(required(top, f"./AreaEstimates/Resources/{key}"))
        for key in RESOURCE_KEYS
    }
    available = {
        key: int(required(top, f"./AreaEstimates/AvailableResources/{key}"))
        for key in RESOURCE_KEYS
    }
    utilization = {
        key: (resources[key] / available[key] if available[key] else None)
        for key in RESOURCE_KEYS
    }
    target_ns = float(required(top, "./UserAssignments/TargetClockPeriod"))
    estimated_ns = float(required(
        top,
        "./PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod",
    ))
    uncertainty_ns = 1.35
    load_loop = only_loop(LOAD_XML)
    compute_loop = only_loop(COMPUTE_XML)
    activation_root = ElementTree.parse(ACTIVATION_XML).getroot()
    activation_loops = activation_root.findall(
        "./PerformanceEstimates/SummaryOfLoopLatency/*"
    )
    if len(activation_loops) != 1:
        raise ValueError("expected one activation load loop")
    activation_ii = int(required(activation_loops[0], "./PipelineII"))

    weight_bursts = sorted(set(re.findall(
        r"\| m_axi_gmem_w([0-5]) \| read\s+\| variable \| 256",
        top_text,
    )))
    case_rows = re.findall(
        r"^GEMV6_CASE name=(\S+).*?max_abs=([0-9.eE+-]+).*?result=(PASS|FAIL)$",
        cosim_text,
        re.MULTILINE,
    )
    unique_cases = {}
    for name, error, status in case_rows:
        unique_cases[name] = {"maximum_absolute_error": float(error), "status": status}
    invalid_pass = (
        "GEMV6_CASE name=invalid a=-2 b=-3 c=-1 unchanged=1 result=PASS"
        in cosim_text
    )
    numeric_max = max(
        (row["maximum_absolute_error"] for row in unique_cases.values()),
        default=float("inf"),
    )
    rtl_transactions = [int(value) for value in re.findall(
        r"RTL Simulation : (\d+) / 9", cosim_text
    )]
    cosim_pass = (
        "C/RTL co-simulation finished: PASS" in cosim_text
        and len(unique_cases) == 6
        and all(row["status"] == "PASS" for row in unique_cases.values())
        and invalid_pass
        and numeric_max == 0.0
        and rtl_transactions
        and max(rtl_transactions) == 9
    )

    partition = json.loads(PARTITION_JSON.read_text(encoding="utf-8"))
    decoder = partition["decoder_weight_stream"]
    bandwidth_rows = {
        row["bandwidth"]: row for row in decoder["bandwidth_cases"]
    }
    line = bandwidth_rows["DDR4_2133_line_rate"]
    scenario85 = bandwidth_rows["DDR4_2133_85pct_scenario"]

    native_bytes_per_port_record = 8 * (32 + 2)
    transferred_bytes_per_port_record = 9 * 32
    clock_hz = 200_000_000
    compute_macs_s = 6 * 32 * (8 / 9) * clock_hz
    timing_issue = bool(re.search(
        r"\|\+ q8q8_gemv_stream6\s+\|\s*Timing\|", top_text
    ))
    resource_gate = (
        utilization["BRAM_18K"] <= 0.25
        and utilization["DSP"] <= 0.10
        and utilization["FF"] <= 0.10
        and utilization["LUT"] <= 0.15
    )
    structural_gate = (
        load_loop["pipeline_ii"] == 1
        and compute_loop["pipeline_ii"] == 1
        and activation_ii == 2
        and weight_bursts == list("012345")
    )
    decision = (
        "STREAM6_RTL_COSIM_PASS / OOC_TIMING_NEXT"
        if cosim_pass and resource_gate and structural_gate
        else "STREAM6_GATE_FAIL / ITERATE_HLS"
    )

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": (
            "independent six-stream Q8_0 batch-one Decoder GEMV; C simulation, "
            "HLS synthesis and Verilog C/RTL co-simulation only"
        ),
        "verification": {
            "c_rtl_cosim_pass": cosim_pass,
            "rtl_transactions_completed": max(rtl_transactions, default=0),
            "numeric_cases": unique_cases,
            "invalid_dimension_case_pass": invalid_pass,
            "maximum_absolute_error": numeric_max,
            "cosim_log": str(cosim_log.relative_to(WORKSPACE)),
            "cosim_log_sha256": sha256(cosim_log),
        },
        "hls": {
            "tool": required(top, "./ReportVersion/Version"),
            "part": required(top, "./UserAssignments/Part"),
            "target_clock_ns": target_ns,
            "estimated_clock_ns": estimated_ns,
            "estimated_fmax_mhz": 1000.0 / estimated_ns,
            "clock_uncertainty_ns": uncertainty_ns,
            "hls_uncertainty_timing_clean": not timing_issue,
            "note": (
                "estimated period is below 5 ns, but HLS reports a Timing issue "
                "against its uncertainty-adjusted budget; OOC/implementation is required"
            ),
            "activation_load_ii": activation_ii,
            "weight_load_loop": load_loop,
            "compute_loop": compute_loop,
            "inferred_weight_burst_ports": [f"gmem_w{x}" for x in weight_bursts],
            "resources": resources,
            "available_resources": available,
            "utilization": utilization,
            "csynth_xml": str(TOP_XML.relative_to(WORKSPACE)),
            "csynth_xml_sha256": sha256(TOP_XML),
        },
        "architecture": {
            "weight_ports": 6,
            "hls_weight_port_width_bits": 256,
            "outputs_per_group": 48,
            "record_words_per_port_per_k_block": 9,
            "native_q8_0_bytes_per_port_record": native_bytes_per_port_record,
            "transferred_bytes_per_port_record": transferred_bytes_per_port_record,
            "packing_efficiency": (
                native_bytes_per_port_record / transferred_bytes_per_port_record
            ),
            "traffic_overhead_over_native": (
                transferred_bytes_per_port_record / native_bytes_per_port_record - 1.0
            ),
            "requested_internal_weight_bandwidth_at_200mhz_bytes_s": (
                6 * 32 * clock_hz
            ),
            "integer_compute_capacity_gmac_s": compute_macs_s / 1e9,
            "integer_compute_capacity_gops_s_two_ops_per_mac": 2 * compute_macs_s / 1e9,
            "note": (
                "internal request/capacity figures are not DDR measurements; the six PS "
                "ports and one DDR controller will apply width conversion and backpressure"
            ),
        },
        "decoder_system_bounds_from_partition_gate": {
            "source": str(PARTITION_JSON.relative_to(WORKSPACE)),
            "source_sha256": sha256(PARTITION_JSON),
            "full_vocabulary_weight_bytes_per_token": decoder[
                "full_vocabulary_bytes_per_token"
            ],
            "ddr4_2133_line_rate_upper_bound_token_s": line[
                "full_vocabulary_upper_bound_token_per_second"
            ],
            "ddr4_2133_85pct_scenario_token_s": scenario85[
                "full_vocabulary_upper_bound_token_per_second"
            ],
            "five_token_s_required_traffic_reduction_at_85pct": scenario85[
                "targets"
            ]["5_token_per_second"]["required_reduction_from_full"],
        },
        "gates": {
            "c_rtl_cosim": cosim_pass,
            "six_weight_bursts_and_ii1": structural_gate,
            "standalone_hls_resource_budget": resource_gate,
            "gross_5ns_estimate": estimated_ns <= target_ns,
            "uncertainty_adjusted_hls_timing": not timing_issue,
        },
        "limitations": [
            "No exported-IP OOC synthesis, place-and-route, bitstream or board measurement yet.",
            "The 256-bit HLS ports do not prove six simultaneous 256-bit PS DDR beats.",
            "Token/s values are bandwidth-only upper bounds and exclude KV, activation, control and contention.",
            "This branch does not replace the previously board-validated PE16 prefill GEMM.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md = f"""# 六通道 Q8_0 Decoder GEMV：HLS 与 RTL 协同仿真记录

## 结论

判定：`{decision}`。

- C/RTL 协同仿真通过，9/9 个 RTL 事务完成；6 个数值用例和 1 个非法维度用例均通过，最大绝对误差为 `{numeric_max:g}`。
- 六个权重 AXI 端口均推断出连续 256-bit burst；权重读取循环 II=1，九相计算循环 II=1。
- HLS 估算周期 `{estimated_ns:.3f} ns`（约 `{1000.0 / estimated_ns:.2f} MHz`），低于 5 ns 名义目标；但在 1.35 ns 时钟不确定度下仍报告 Timing 风险，必须继续做导出 IP 后的 OOC/实现时序验证。
- 资源估算：BRAM18K {resources['BRAM_18K']}（{utilization['BRAM_18K']:.2%}）、DSP {resources['DSP']}（{utilization['DSP']:.2%}）、LUT {resources['LUT']}（{utilization['LUT']:.2%}）、FF {resources['FF']}（{utilization['FF']:.2%}）。

## 数据组织与吞吐含义

每个端口、每个 K-block 使用 9 个 256-bit 字：1 个尺度头和 8 个 Q8 权重块，对应 288 B 传输承载 272 B 原生 Q8_0 数据，打包效率 {native_bytes_per_port_record / transferred_bytes_per_port_record:.2%}。200 MHz 下，HLS 内部六端口请求上限为 38.4 GB/s，整数计算能力为 {compute_macs_s / 1e9:.2f} GMAC/s；这两个数都不是 ZCU102 DDR 实测值，接入 PS 端口后会受到位宽转换、仲裁和 DDR 总带宽限制。

按已冻结的模型权重流量，每生成 1 token 需读取约 {decoder['full_vocabulary_bytes_per_token'] / 1e9:.3f} GB 权重。DDR4-2133 线速只给出 {line['full_vocabulary_upper_bound_token_per_second']:.2f} token/s 的理论上界；按线速 85% 的规划情形为 {scenario85['full_vocabulary_upper_bound_token_per_second']:.2f} token/s。要达到 5 token/s，仍需减少约 {scenario85['targets']['5_token_per_second']['required_reduction_from_full']:.1%} 的每 token 权重流量。

## 证据

- 机器可读摘要：`{OUTPUT_JSON.relative_to(WORKSPACE)}`
- RTL 协同仿真日志：`{cosim_log.relative_to(WORKSPACE)}`
- HLS 综合报告：`{TOP_RPT.relative_to(WORKSPACE)}`
- 异构边界与 Decoder 带宽约束：`{PARTITION_JSON.relative_to(WORKSPACE)}`

## 尚未完成

尚未进行导出 IP 的 OOC 综合、全系统布线、真实 DDR 并发带宽测试和板端推理集成，因此不能据此声称板端 token/s 或端到端加速比。下一门槛是导出 IP，并在 200 MHz 目标下获得实现级时序和资源结果。
"""
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "summary": str(OUTPUT_JSON),
        "report": str(OUTPUT_MD),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
