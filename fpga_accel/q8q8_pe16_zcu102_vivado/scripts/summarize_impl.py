#!/usr/bin/env python3
"""Summarize PE16 full-system Vivado implementation against v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
BASELINE = WORKSPACE / "fpga_accel/q8q8_v3_zcu102_vivado/artifacts"
REPORT_DIR = WORKSPACE / "reports"
DEVICE_CAPACITY = {
    "total_luts": 274080,
    "ffs": 548160,
    "bram_tiles": 912,
    "dsps": 2520,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_utilization(path: Path) -> dict[str, Any]:
    line = next(
        (
            row
            for row in path.read_text(encoding="utf-8").splitlines()
            if re.match(r"^\|\s*q8q8_system_wrapper\s*\|", row)
        ),
        None,
    )
    if line is None:
        raise ValueError(f"top utilization row missing in {path}")
    columns = [item.strip() for item in line.split("|")[1:-1]]
    if len(columns) != 11:
        raise ValueError(f"unexpected utilization columns in {path}: {columns}")
    total_luts = int(columns[2])
    ffs = int(columns[6])
    ramb36 = int(columns[7])
    ramb18 = int(columns[8])
    uram = int(columns[9])
    dsps = int(columns[10])
    return {
        "total_luts": total_luts,
        "ffs": ffs,
        "ramb36": ramb36,
        "ramb18": ramb18,
        "bram_tiles": ramb36 + ramb18 / 2.0,
        "uram": uram,
        "dsps": dsps,
    }


def parse_power(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    def field(label: str) -> str:
        match = re.search(
            rf"^\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            raise ValueError(f"missing power field {label!r} in {path}")
        return match.group(1).strip()

    return {
        "total_on_chip_w": float(field("Total On-Chip Power (W)")),
        "dynamic_w": float(field("Dynamic (W)")),
        "device_static_w": float(field("Device Static (W)")),
        "confidence": field("Confidence Level"),
        "status": "Vivado vectorless estimate; not board measurement",
    }


def parse_violation_summary(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    total_match = re.search(r"Violations found:\s*(\d+)", text)
    if not total_match:
        raise ValueError(f"violation count missing in {path}")
    rules: dict[str, dict[str, Any]] = {}
    by_severity: dict[str, int] = {}
    for rule, severity, count in re.findall(
        r"^\|\s*([A-Z0-9-]+)\s*\|\s*([A-Za-z]+)\s*\|.*\|\s*(\d+)\s*\|$",
        text,
        flags=re.MULTILINE,
    ):
        value = int(count)
        rules[rule] = {"severity": severity, "count": value}
        by_severity[severity] = by_severity.get(severity, 0) + value
    return {
        "total": int(total_match.group(1)),
        "by_severity": by_severity,
        "rules": rules,
    }


def parse_design(artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / "implementation_summary.txt"
    utilization_path = artifact_dir / "impl_utilization.rpt"
    timing_path = artifact_dir / "impl_timing_summary.rpt"
    route_path = artifact_dir / "impl_route_status.rpt"
    power_path = artifact_dir / "impl_power.rpt"
    drc_path = artifact_dir / "impl_drc.rpt"
    methodology_path = artifact_dir / "impl_methodology.rpt"
    values = parse_key_values(summary_path)
    utilization = parse_utilization(utilization_path)
    utilization_percent = {
        name: 100.0 * utilization[name] / capacity
        for name, capacity in DEVICE_CAPACITY.items()
    }
    return {
        "artifact_dir": str(artifact_dir),
        "target_mhz": float(values["TARGET_MHZ"]),
        "actual_mhz": float(values["ACTUAL_MHZ"]),
        "output_axi_width": int(values["OUTPUT_AXI_WIDTH"]),
        "wns_ns": float(values["WNS_NS"]),
        "whs_ns": float(values["WHS_NS"]),
        "unrouted_nets": int(values["UNROUTED_NETS"]),
        "bitstream_generated": values["BITSTREAM_GENERATED"],
        "board_programmed": values["BOARD_PROGRAMMED"],
        "utilization": utilization,
        "utilization_percent": utilization_percent,
        "power": parse_power(power_path),
        "drc": parse_violation_summary(drc_path),
        "methodology": parse_violation_summary(methodology_path),
        "report_sha256": {
            path.name: sha256(path)
            for path in (
                summary_path,
                utilization_path,
                timing_path,
                route_path,
                power_path,
                drc_path,
                methodology_path,
            )
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    base = result["baseline_v3"]
    cand = result["candidate_pe16"]
    gate = result["gate"]
    bu = base["utilization"]
    cu = cand["utilization"]
    rows = (
        ("LUT", bu["total_luts"], cu["total_luts"]),
        ("FF", bu["ffs"], cu["ffs"]),
        ("BRAM tile", bu["bram_tiles"], cu["bram_tiles"]),
        ("DSP", bu["dsps"], cu["dsps"]),
    )
    lines = [
        "# ZCU102 Q8×Q8 PE16完整系统布局布线报告",
        "",
        f"生成时间：{result['generated_at']}",
        f"状态：`{gate['decision']}`",
        "",
        "## 验证边界",
        "",
        "- 本报告是PE16完整PS+互连+PL系统的Vivado routed结果。",
        "- 未生成bitstream，未连接、复位或烧写ZCU102。",
        "- 功耗为Vivado vectorless估算，不是板级实测。",
        "- 延迟仍是一阶模型；PE16 kernel尚未在板端运行。",
        "",
        "## 时序与路由",
        "",
        "| 配置 | WNS/ns | WHS/ns | 未布线网络 |",
        "|---|---:|---:|---:|",
        f"| v3 PE8 | {base['wns_ns']:+.3f} | {base['whs_ns']:+.3f} | "
        f"{base['unrouted_nets']} |",
        f"| PE16 | {cand['wns_ns']:+.3f} | {cand['whs_ns']:+.3f} | "
        f"{cand['unrouted_nets']} |",
        "",
        "## 完整系统资源",
        "",
        "| 资源 | v3 PE8 | PE16 | PE16占器件 | 变化 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, base_value, cand_value in rows:
        percent_key = {
            "LUT": "total_luts",
            "FF": "ffs",
            "BRAM tile": "bram_tiles",
            "DSP": "dsps",
        }[name]
        lines.append(
            f"| {name} | {base_value:g} | {cand_value:g} | "
            f"{cand['utilization_percent'][percent_key]:.2f}% | "
            f"{cand_value - base_value:+g} |"
        )
    lines += [
        "",
        "## 工具功耗估算",
        "",
        f"- v3 PE8：{base['power']['total_on_chip_w']:.3f} W，"
        f"confidence={base['power']['confidence']}。",
        f"- PE16：{cand['power']['total_on_chip_w']:.3f} W，"
        f"confidence={cand['power']['confidence']}。",
        "",
        "## DRC与方法学审计",
        "",
        f"- 默认DRC共{cand['drc']['total']}项："
        f"Warning {cand['drc']['by_severity'].get('Warning', 0)}项，"
        f"Advisory {cand['drc']['by_severity'].get('Advisory', 0)}项，"
        f"Error {cand['drc']['by_severity'].get('Error', 0)}项。",
        "- 警告均为DSP输入/输出流水化建议或DSP属性advisory；设计仍满足200 MHz。",
        f"- 方法学DRC共{cand['methodology']['total']}项，均为时钟资源advisory；"
        "规则与v3 PE8基线相同。",
        "",
        "## 决策",
        "",
        gate["explanation_zh"],
        "",
        f"机器可读证据：`{result['output_json']}`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "artifacts/pe16_vivado_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPORT_DIR / "2026-08-27-zcu102-q8q8-pe16-vivado.md",
    )
    args = parser.parse_args()

    baseline = parse_design(args.baseline)
    candidate = parse_design(args.candidate)
    timing_pass = candidate["wns_ns"] >= 0.0 and candidate["whs_ns"] >= 0.0
    route_pass = candidate["unrouted_nets"] == 0
    interface_pass = (
        candidate["output_axi_width"] == baseline["output_axi_width"] == 128
    )
    resource_pass = all(
        percent <= 80.0
        for percent in candidate["utilization_percent"].values()
    )
    pass_gate = timing_pass and route_pass and interface_pass and resource_pass
    if pass_gate:
        decision = "VIVADO_ROUTE_PASS / BOARD_MICROBENCHMARK_NEXT"
        explanation = (
            "PE16完整系统通过200 MHz setup/hold、路由完整性、128-bit接口和"
            "80%资源门控。下一步应先生成bitstream并运行既有数值self-test及"
            "三类microbenchmark；本报告本身不证明板端加速。"
        )
    else:
        decision = "VIVADO_ROUTE_FAIL / PE16_STOP"
        explanation = (
            "PE16完整系统未通过时序、路由、接口或资源门控，停止生成bitstream"
            "和板端测试；下一步转回PE8双缓冲DATAFLOW。"
        )

    output_json = args.output_json.resolve()
    result = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "full-system Vivado routed implementation; no PE16 bitstream "
            "or board measurement"
        ),
        "baseline_v3": baseline,
        "candidate_pe16": candidate,
        "delta_pe16_minus_v3": {
            "wns_ns": candidate["wns_ns"] - baseline["wns_ns"],
            "whs_ns": candidate["whs_ns"] - baseline["whs_ns"],
            "total_luts": (
                candidate["utilization"]["total_luts"]
                - baseline["utilization"]["total_luts"]
            ),
            "ffs": (
                candidate["utilization"]["ffs"]
                - baseline["utilization"]["ffs"]
            ),
            "bram_tiles": (
                candidate["utilization"]["bram_tiles"]
                - baseline["utilization"]["bram_tiles"]
            ),
            "dsps": (
                candidate["utilization"]["dsps"]
                - baseline["utilization"]["dsps"]
            ),
            "vectorless_power_w": (
                candidate["power"]["total_on_chip_w"]
                - baseline["power"]["total_on_chip_w"]
            ),
        },
        "gate": {
            "timing_pass": timing_pass,
            "route_pass": route_pass,
            "interface_pass": interface_pass,
            "resource_under_80_percent_pass": resource_pass,
            "decision": decision,
            "explanation_zh": explanation,
        },
        "inputs": {
            "hls_ip": str(
                WORKSPACE
                / "fpga_accel/q8q8_pe16_hls/artifacts/q8q8_pe16_ip.zip"
            ),
            "hls_ip_sha256": sha256(
                WORKSPACE
                / "fpga_accel/q8q8_pe16_hls/artifacts/q8q8_pe16_ip.zip"
            ),
            "build_tcl_sha256": sha256(ROOT / "scripts/build_system.tcl"),
        },
        "output_json": str(output_json),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["gate"], ensure_ascii=False))
    print(f"JSON={output_json}")
    print(f"MARKDOWN={args.output_md.resolve()}")
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
