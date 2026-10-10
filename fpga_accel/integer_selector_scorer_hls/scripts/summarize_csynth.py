#!/usr/bin/env python3
"""Summarise the controlled integer_selector_scorer csynth_design run.

Only reads Vitis HLS ``csynth_design`` reports (no cosim, no IP export, no
bitstream, no board).  Emits the small machine-readable ``summary.json`` used by
the gate: timing, resource percentages, the reported finite latency and a
**derived** per-group latency for G = 80/260/400 at 200 MHz.

Usage:
    summarize_csynth.py --report-dir DIR --console-log FILE --out FILE
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

TOP = "integer_selector_scorer"
SUBMODULE = "p_anonymous_namespace_is_scorer_core_false_s"
TARGET_FMAX_MHZ = 200.0


def _text(root: ET.Element, path: str) -> str | None:
    node = root.find(path)
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _number(value):
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
    resources = {}
    area = root.find("AreaEstimates/Resources")
    if area is not None:
        for child in area:
            resources[child.tag] = int(float(child.text))
    available = {}
    avail = root.find("AreaEstimates/AvailableResources")
    if avail is not None:
        for child in avail:
            available[child.tag] = int(float(child.text))
    meta = {}
    assignments = root.find("UserAssignments")
    if assignments is not None:
        for child in assignments:
            meta[child.tag] = (child.text or "").strip()
    return {
        "estimated_clock_ns": _number(
            _text(root, "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod")
        ),
        "worst_latency_cycles": _number(
            _text(root, "PerformanceEstimates/SummaryOfOverallLatency/Worst-caseLatency")
        ),
        "best_latency_cycles": _number(
            _text(root, "PerformanceEstimates/SummaryOfOverallLatency/Best-caseLatency")
        ),
        "average_latency_cycles": _number(
            _text(root, "PerformanceEstimates/SummaryOfOverallLatency/Average-caseLatency")
        ),
        "resources_used": resources,
        "resources_available": available,
        "lifetime_meta": meta,
    }


def parse_clock_table(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|", text)
    if not match:
        return {"clock_target_ns": None, "clock_estimated_ns": None, "clock_uncertainty_ns": None}
    return {
        "clock_target_ns": float(match.group(1)),
        "clock_estimated_ns": float(match.group(2)),
        "clock_uncertainty_ns": float(match.group(3)),
    }


def parse_pipeline_instances(path: Path) -> dict[str, float]:
    """Instance name -> max latency (cycles) from the submodule latency table.

    Only the performance-estimate latency table (which carries ``ms`` absolute
    columns) is parsed, so the utilization/register tables with the same
    instance names cannot overwrite the values.
    """
    result: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Pipeline_VITIS_LOOP" not in line or not line.strip().startswith("|"):
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 10 or not any(
            re.search(r"\b(ns|us|ms)\b", col) for col in cols
        ):
            continue
        instance = cols[1]
        name_match = re.search(r"(Pipeline_VITIS_LOOP_\d+_\d+)_", instance)
        if not name_match:
            continue
        value = _number(cols[4])
        if value is not None:
            result[name_match.group(1)] = value
    return result


def parse_loop_table(path: Path) -> list[dict]:
    loops: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "VITIS_LOOP" not in line or not line.strip().startswith("|"):
            continue
        cols = [col.strip() for col in line.split("|")]
        # | - VITIS_LOOP_x_y | min | max | iter latency | ii | target | trip | pipelined |
        if len(cols) < 9:
            continue
        name = cols[1].lstrip("- ").strip()
        if not name.startswith("VITIS_LOOP"):
            continue
        iteration = cols[4]
        iteration_values = [float(v) for v in re.findall(r"[\d.]+", iteration)]
        loops.append(
            {
                "loop": name,
                "latency_min_cycles": _number(cols[2]),
                "latency_max_cycles": _number(cols[3]),
                "iteration_latency": iteration,
                "iteration_latency_max": max(iteration_values) if iteration_values else None,
                "trip_count": cols[7],
                "pipelined": cols[8],
            }
        )
    return loops


def parse_console(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "error_count": len(re.findall(r"^ERROR:", text, re.MULTILINE)),
        "warning_count": len(re.findall(r"^WARNING:", text, re.MULTILINE)),
        "estimated_fmax_mhz": _number(
            (re.search(r"Estimated Fmax:\s*([\d.]+)\s*MHz", text) or [None, None])[1]
        ),
        "timing_warning_200_871": "HLS 200-871" in text,
        "critical_path_warning_200_1016": "HLS 200-1016" in text,
        "all_loop_constraints_met": "All loop constraints were satisfied" in text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clock-ns", type=float, default=5.0)
    args = parser.parse_args()

    report_dir = args.report_dir
    xml = parse_csynth_xml(report_dir / "csynth.xml")
    clock = parse_clock_table(report_dir / f"{TOP}_csynth.rpt")
    pipelines = parse_pipeline_instances(report_dir / f"{SUBMODULE}_csynth.rpt")
    loops = parse_loop_table(report_dir / f"{SUBMODULE}_csynth.rpt")
    console = parse_console(args.console_log)

    target = clock["clock_target_ns"] or args.clock_ns
    uncertainty = clock["clock_uncertainty_ns"]
    effective = target - uncertainty if uncertainty is not None else None
    estimated = clock["clock_estimated_ns"] or xml["estimated_clock_ns"]
    slack = round(effective - estimated, 4) if effective is not None else None

    used = xml["resources_used"]
    available = xml["resources_available"]
    utilization = {
        key: round(100.0 * value / available.get(key, 0), 2) if available.get(key) else 0.0
        for key, value in used.items()
    }

    # The Wv cache is the only one-time 8192-iteration pipeline, so it is the
    # largest fixed pipeline latency in the submodule.
    wv_load_cycles = max(pipelines.values()) if pipelines else None

    per_group = {}
    for loop in loops:
        cycle_cost = loop["iteration_latency_max"]
        if cycle_cost is not None:
            per_group[loop["loop"]] = cycle_cost
    # Identify the two group loops structurally (never by source line number):
    # pass 1 has the largest total latency, pass 2 the second largest.
    ordered = sorted(
        (loop for loop in loops if loop["latency_max_cycles"] is not None),
        key=lambda loop: loop["latency_max_cycles"],
        reverse=True,
    )
    pass1 = ordered[0]["iteration_latency_max"] if len(ordered) > 0 else None
    pass2 = ordered[1]["iteration_latency_max"] if len(ordered) > 1 else None
    per_group_total = (pass1 or 0.0) + (pass2 or 0.0)

    derived = {}
    if wv_load_cycles is not None and per_group_total > 0:
        for g in (80, 260, 400):
            cycles = wv_load_cycles + per_group_total * g
            derived[str(g)] = {
                "group_count": g,
                "cycles": cycles,
                "ms_at_200mhz": round(cycles / (TARGET_FMAX_MHZ * 1e6) * 1000.0, 4),
            }

    derived_g400_cycles = derived.get("400", {}).get("cycles")
    gate = {
        "zero_errors": console["error_count"] == 0,
        "timing_met": bool(slack is not None and slack >= 0),
        "all_resources_under_50_percent": all(v < 50.0 for v in utilization.values()),
        "derived_g400_finite": derived_g400_cycles is not None,
        "derived_g400_le_4m_cycles": bool(
            derived_g400_cycles is not None and derived_g400_cycles <= 4_000_000
        ),
        "derived_g400_le_20ms": bool(
            derived_g400_cycles is not None
            and derived_g400_cycles / (TARGET_FMAX_MHZ * 1e6) * 1000.0 <= 20.0
        ),
    }
    gate["pass"] = all(gate.values())

    summary = {
        "tool": xml["lifetime_meta"].get("TopModelName"),
        "top": TOP,
        "part": xml["lifetime_meta"].get("Part"),
        "flow_target": xml["lifetime_meta"].get("FlowTarget"),
        "clock_target_ns": target,
        "clock_uncertainty_ns": uncertainty,
        "effective_budget_ns": round(effective, 4) if effective is not None else None,
        "estimated_clock_ns": estimated,
        "estimated_fmax_mhz": round(1000.0 / estimated, 2) if estimated else None,
        "timing_slack_ns": slack,
        "reported_latency_cycles": {
            "best": xml["best_latency_cycles"],
            "average": xml["average_latency_cycles"],
            "worst": xml["worst_latency_cycles"],
        },
        "reported_latency_ms_at_200mhz": round(
            (xml["worst_latency_cycles"] or 0.0) / (TARGET_FMAX_MHZ * 1e6) * 1000.0, 4
        ),
        "wv_load_cycles": wv_load_cycles,
        "per_group_cycles": {"pass1": pass1, "pass2": pass2, "total": per_group_total},
        "derived_latency": derived,
        "resources_used": used,
        "resources_available": available,
        "resource_utilization_percent": utilization,
        "console": console,
        "pipelines": pipelines,
        "loops": loops,
        "gate": gate,
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    print(json.dumps({
        "estimated_clock_ns": summary["estimated_clock_ns"],
        "timing_slack_ns": slack,
        "wv_load_cycles": wv_load_cycles,
        "per_group_cycles": summary["per_group_cycles"],
        "derived_latency": derived,
        "resource_utilization_percent": utilization,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
