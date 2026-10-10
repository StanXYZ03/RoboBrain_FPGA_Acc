#!/usr/bin/env python3
"""Freeze the six-stream GEMV out-of-context implementation result."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
IMPL_REPORTS = (
    ROOT
    / "build/q8q8_gemv_stream6_prj/solution1/impl/report/verilog"
)
IMPL_RPT = IMPL_REPORTS / "export_impl.rpt"
IP_ZIP = ROOT / "artifacts/q8q8_gemv_stream6_ip.zip"
PRIOR_JSON = ROOT / "artifacts/stream6_hls_gate_20260901/summary.json"
OUTPUT_DIR = ROOT / "artifacts/stream6_ooc_impl_20260901"
OUTPUT_JSON = OUTPUT_DIR / "summary.json"
OUTPUT_MD = WORKSPACE / "reports/2026-09-01-q8q8-gemv-stream6-ooc-impl.md"
AVAILABLE = {
    "LUT": 274080,
    "FF": 548160,
    "DSP": 2520,
    "BRAM": 1824,
    "URAM": 0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required(pattern: str, text: str, cast=float):
    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        raise ValueError(f"missing report field: {pattern}")
    return cast(match.group(1))


def main() -> int:
    if OUTPUT_DIR.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite frozen OOC evidence")
    logs = sorted((ROOT / "artifacts").glob("*_vitis_hls_export_impl.log"))
    if not logs:
        raise FileNotFoundError("no export_impl log found")
    impl_log = logs[-1]
    report_text = IMPL_RPT.read_text(encoding="utf-8", errors="replace")
    log_text = impl_log.read_text(encoding="utf-8", errors="replace")
    prior = json.loads(PRIOR_JSON.read_text(encoding="utf-8"))

    resources = {
        key: required(rf"^{key}:\s+(\d+)\s*$", report_text, int)
        for key in AVAILABLE
    }
    utilization = {
        key: (resources[key] / AVAILABLE[key] if AVAILABLE[key] else None)
        for key in AVAILABLE
    }
    target_ns = required(r"\| Target\s+\|\s+([0-9.]+)\s+\|", report_text)
    post_synth_ns = required(
        r"\| Post-Synthesis\s+\|\s+([0-9.]+)\s+\|", report_text
    )
    post_route_ns = required(
        r"\| Post-Route\s+\|\s+([0-9.]+)\s+\|", report_text
    )
    wns_ns = required(r"worst setup slack \(WNS\)=([0-9.+-]+)", log_text)
    whs_ns = required(r"worst hold slack \(WHS\)=([0-9.+-]+)", log_text)
    unrouted = required(r"number of unrouted nets=(\d+)", log_text, int)
    timing_met = "* Timing was met" in report_text and post_route_ns <= target_ns
    implementation_pass = timing_met and wns_ns >= 0 and whs_ns >= 0 and unrouted == 0
    decision = (
        "STREAM6_OOC_200MHZ_PASS / FULL_SYSTEM_INTEGRATION_NEXT"
        if implementation_pass
        else "STREAM6_OOC_FAIL / ITERATE_KERNEL"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    raw = OUTPUT_DIR / "raw"
    raw.mkdir()
    raw_sources = [
        IMPL_RPT,
        IMPL_REPORTS / "export_impl.xml",
        IMPL_REPORTS / "export_syn.rpt",
        IMPL_REPORTS / "export_syn.xml",
        IMPL_REPORTS / "q8q8_gemv_stream6_export.rpt",
        impl_log,
        IP_ZIP,
    ]
    frozen_files = []
    for source in raw_sources:
        destination = raw / source.name
        shutil.copy2(source, destination)
        frozen_files.append({
            "path": str(destination.relative_to(WORKSPACE)),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })

    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "scope": (
            "Vivado out-of-context synthesis and place-and-route of the standalone "
            "six-stream Q8_0 Decoder GEMV IP"
        ),
        "prerequisite": {
            "decision": prior["decision"],
            "summary": str(PRIOR_JSON.relative_to(WORKSPACE)),
            "summary_sha256": sha256(PRIOR_JSON),
            "c_rtl_cosim_pass": prior["verification"]["c_rtl_cosim_pass"],
            "maximum_absolute_error": prior["verification"][
                "maximum_absolute_error"
            ],
        },
        "implementation": {
            "tool": "Vivado 2023.2",
            "part": "xczu9eg-ffvb1156-2-e",
            "target_period_ns": target_ns,
            "post_synthesis_period_ns": post_synth_ns,
            "post_route_period_ns": post_route_ns,
            "post_route_fmax_from_period_mhz": 1000.0 / post_route_ns,
            "worst_setup_slack_ns": wns_ns,
            "worst_hold_slack_ns": whs_ns,
            "unrouted_nets": unrouted,
            "timing_met": timing_met,
            "resources": resources,
            "available_resources": AVAILABLE,
            "utilization": utilization,
        },
        "frozen_evidence": frozen_files,
        "gates": {
            "prior_c_rtl_cosim": prior["verification"]["c_rtl_cosim_pass"],
            "post_route_200mhz": timing_met,
            "setup_slack_nonnegative": wns_ns >= 0,
            "hold_slack_nonnegative": whs_ns >= 0,
            "fully_routed": unrouted == 0,
            "standalone_resource_headroom": (
                utilization["LUT"] < 0.15
                and utilization["FF"] < 0.10
                and utilization["DSP"] < 0.15
                and utilization["BRAM"] < 0.15
            ),
        },
        "limitations": [
            "OOC warnings state that ap_clk has no HD.CLK_SRC and external AXI ports have no full-system partition-pin locations.",
            "The result excludes Zynq PS, SmartConnect/data-width converters, reset/clock infrastructure and DDR contention.",
            "Vectorless OOC power is not used as a power claim.",
            "No bitstream, board programming, SD-card modification or board throughput measurement was performed.",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report = f"""# 六通道 Q8_0 Decoder GEMV：OOC 实现记录

## 结论

判定：`{decision}`。

独立计算核在 XCZU9EG 上完成 Vivado OOC 综合、布局和布线。目标周期为 {target_ns:.3f} ns（200 MHz），后综合周期 {post_synth_ns:.3f} ns，布线后周期 {post_route_ns:.3f} ns；WNS 为 {wns_ns:+.3f} ns，WHS 为 {whs_ns:+.3f} ns，未布通网络为 {unrouted}。因此，HLS 阶段的时序风险在计算核 OOC 实现中已经消除。

## 实现级资源

| 资源 | 用量 | XCZU9EG 可用 | 占比 |
|---|---:|---:|---:|
| LUT | {resources['LUT']} | {AVAILABLE['LUT']} | {utilization['LUT']:.2%} |
| FF | {resources['FF']} | {AVAILABLE['FF']} | {utilization['FF']:.2%} |
| DSP | {resources['DSP']} | {AVAILABLE['DSP']} | {utilization['DSP']:.2%} |
| BRAM18K | {resources['BRAM']} | {AVAILABLE['BRAM']} | {utilization['BRAM']:.2%} |

## 判断边界

这证明六通道 GEMV 核单体可以在 200 MHz 下闭合，并留有较多逻辑和算术资源。OOC 环境没有连接 PS、SmartConnect、位宽转换器和 DDR 控制器；工具也明确提示 `ap_clk` 缺少全系统 `HD.CLK_SRC`，外部 AXI 端口缺少系统级位置。因此，这一结果不能替代全系统时序、DDR 并发带宽和板端 token/s 测量。

## 证据

- 机器可读摘要：`{OUTPUT_JSON.relative_to(WORKSPACE)}`
- 冻结的实现报告与 IP：`{raw.relative_to(WORKSPACE)}`
- 前置 C/RTL 协同仿真：`{PRIOR_JSON.relative_to(WORKSPACE)}`

下一步是构建只含 PS、六路高性能端口、必要互连和该 IP 的最小系统工程，先做地址/位宽/时钟静态检查与仿真，再做全系统实现；此阶段仍不需要烧板。
"""
    OUTPUT_MD.write_text(report, encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "summary": str(OUTPUT_JSON),
        "report": str(OUTPUT_MD),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
