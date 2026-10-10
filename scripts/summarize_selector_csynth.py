#!/usr/bin/env python3
"""Build the small machine-readable summary for a controlled selector csynth run.

Only reads Vitis HLS ``csynth_design`` reports (no cosim, no IP export, no
bitstream).  Emits JSON with the keys required by the gate:
status/algorithm_equivalence/tests/csynth/latency/resources/timing/warnings/
files_changed/next_gate are assembled by the caller; this script fills the
``csynth``/``latency``/``resources``/``timing``/``warnings`` blocks from the
tool reports.

Usage:
    summarize_selector_csynth.py --report-dir DIR --console-log FILE \
        --df-before FILE --project-dir DIR --clock-ns 5.0 --out FILE
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _number(value: str | None):
    if value is None:
        return None
    value = value.strip()
    if value in {"undef", "?", ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_csynth_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    estimated = _number(_text(root, "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"))
    best = _number(_text(root, "PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency"))
    average = _number(_text(root, "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency"))
    worst = _number(_text(root, "PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency"))
    resources = {}
    available = {}
    area = root.find("AreaEstimates/Resources")
    if area is not None:
        for child in area:
            resources[child.tag] = int(child.text.strip())
    avail = root.find("AreaEstimates/AvailableResources")
    if avail is not None:
        for child in avail:
            available[child.tag] = int(child.text.strip())
    bundles: dict[str, list[str]] = {}
    ports = root.find("InterfaceSummary")
    if ports is not None:
        for port in ports.findall("RtlPorts"):
            obj = port.find("Object")
            protocol = port.find("IOProtocol")
            if obj is None or protocol is None:
                continue
            name = obj.text.strip() if obj.text else ""
            proto = protocol.text.strip() if protocol.text else ""
            if proto == "m_axi":
                bundles.setdefault(name, []).append(
                    (port.findtext("name") or "").strip()
                )
    assignments = root.find("UserAssignments")
    meta = {}
    if assignments is not None:
        for child in assignments:
            meta[child.tag] = (child.text or "").strip()
    return {
        "estimated_clock_ns": estimated,
        "top_latency_cycles": {
            "best": best,
            "average": average,
            "worst": worst,
        },
        "resources_used": resources,
        "resources_available": available,
        "m_axi_bundles": {name: len(ports) for name, ports in bundles.items()},
        "lifetime_meta": meta,
    }


def parse_top_report(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    timing = re.search(
        r"ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|", text
    )
    latency_summary = re.search(
        r"Latency \(cycles\) \|  Latency \(absolute\).*?\n.*?\n\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|\s*(\S+)\s*\|",
        text,
        re.DOTALL,
    )
    result = {
        "clock_target_ns": float(timing.group(1)) if timing else None,
        "clock_estimated_ns": float(timing.group(2)) if timing else None,
        "clock_uncertainty_ns": float(timing.group(3)) if timing else None,
    }
    if latency_summary:
        result["latency_min"] = _number(latency_summary.group(1))
        result["latency_max"] = _number(latency_summary.group(2))
        result["latency_min_abs_ns"] = _number(latency_summary.group(3))
        result["latency_max_abs_ns"] = _number(latency_summary.group(4))
    return result


def parse_loop_table(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    loops = []
    pattern = re.compile(r"^\s*\|\s*[o+]\s+(\S+)\s*\|(.*)\|\s*$")
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(1)
        if name.startswith("token_selector_top") or name == "token_selector_top":
            continue
        cols = [col.strip() for col in match.group(2).split("|")]
        if len(cols) < 12:
            continue
        loops.append(
            {
                "loop": name,
                "issue_type": cols[0],
                "slack": cols[1],
                "latency_cycles": cols[2],
                "latency_ns": cols[3],
                "iteration_latency": cols[4],
                "interval": cols[5],
                "trip_count": cols[6],
                "pipelined": cols[7],
            }
        )
    return loops


def parse_console(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    error_count = len(re.findall(r"^ERROR:", text, re.MULTILINE))
    warning_count = len(re.findall(r"^WARNING:", text, re.MULTILINE))
    ii_final = re.findall(r"Final II = (\d+)", text)
    loop_constraints = "All loop constraints were NOT satisfied" in text
    return {
        "error_count": error_count,
        "warning_count": warning_count,
        "ii_final_values": sorted({int(value) for value in ii_final}),
        "all_loop_constraints_met": not loop_constraints,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--df-before", type=Path, required=True)
    parser.add_argument("--project-dir", type=str, default="")
    parser.add_argument("--top", type=str, default="token_selector_top")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    xml = parse_csynth_xml(args.report_dir / "csynth.xml")
    top = parse_top_report(args.report_dir / f"{args.top}_csynth.rpt")
    loops = parse_loop_table(args.report_dir / "csynth.rpt")
    console = parse_console(args.console_log)

    target = top["clock_target_ns"] or args.clock_ns
    uncertainty = top["clock_uncertainty_ns"]
    effective = (
        target - uncertainty if uncertainty is not None and target is not None else None
    )
    estimated = xml["estimated_clock_ns"] or top["clock_estimated_ns"]
    slack = (
        round(effective - estimated, 4)
        if effective is not None and estimated is not None
        else None
    )

    used = xml["resources_used"]
    available = xml["resources_available"]
    util = {}
    for key, value in used.items():
        total = available.get(key, 0)
        util[key] = round(100.0 * value / total, 2) if total else 0.0

    latency = xml["top_latency_cycles"]
    finite = all(value is not None for value in latency.values())
    worst_cycles = latency["worst"]
    at_200mhz_ms = (
        round(worst_cycles / 200_000_000.0 * 1000.0, 4)
        if worst_cycles is not None
        else None
    )

    summary = {
        "tool": xml["lifetime_meta"].get("TopModelName"),
        "requested_top": args.top,
        "part": xml["lifetime_meta"].get("Part"),
        "flow_target": xml["lifetime_meta"].get("FlowTarget"),
        "project_dir": args.project_dir,
        "clock_target_ns": target,
        "clock_uncertainty_ns": uncertainty,
        "effective_budget_ns": round(effective, 4) if effective is not None else None,
        "estimated_clock_ns": estimated,
        "estimated_fmax_mhz": round(1000.0 / estimated, 2) if estimated else None,
        "timing_slack_ns": slack,
        "timing_met": bool(slack is not None and slack >= 0),
        "latency": latency,
        "latency_finite": finite,
        "worst_latency_ms_at_200mhz": at_200mhz_ms,
        "latency_under_33p3ms_at_200mhz": bool(
            at_200mhz_ms is not None and at_200mhz_ms < 33.3
        ),
        "resources_used": used,
        "resources_available": available,
        "resource_utilization_percent": util,
        "any_resource_over_50_percent": any(value >= 50.0 for value in util.values()),
        "m_axi_bundles": xml["m_axi_bundles"],
        "console": console,
        "loops": loops,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in (
        "estimated_clock_ns", "timing_slack_ns", "timing_met", "latency",
        "latency_finite", "worst_latency_ms_at_200mhz",
        "latency_under_33p3ms_at_200mhz", "resource_utilization_percent",
        "any_resource_over_50_percent", "m_axi_bundles")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
