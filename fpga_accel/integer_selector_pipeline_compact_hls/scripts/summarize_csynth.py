#!/usr/bin/env python3
"""Summarise the controlled compact-pipeline ``csynth_design`` run.

Only reads Vitis HLS ``csynth_design`` reports (no cosim, no IP export, no
bitstream, no board).  Emits the small machine-readable ``summary.json`` used by
the gate:

* timing (target / uncertainty / estimated period / slack / Fmax),
* absolute and percentage ZCU102 resources,
* the reported best / average / worst latency,
* the scorer phase latencies (Wv load, pass 1, pass 2) as reported by the
  pipeline's own scorer submodule report,
* **formula-derived** per-configuration cycle counts for G = 80/260/400 and
  K = 48/64/96, including the compact envelope overhead (arena pre-zero +
  bounded metadata packing),
* the generated interface audit: exactly three ``m_axi`` bundles, exactly one
  AXI-Lite ``control`` bundle and no ``score_q`` port,
* HLS error / warning counts and the timing / II warnings.

The per-configuration cycle counts are an explicit **model**, not a tool
measurement (the tool reports latency only for compile-time trip-count bounds).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.hardware_regular_selector_fixedpoint import (  # noqa: E402
    build_fixedpoint_regions,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    fpga_transfer_layout,
)

TOP = "integer_selector_pipeline_compact"
SCORER_SUBMODULE = "p_anonymous_namespace_is_scorer_core_false_s"
TARGET_FMAX_MHZ = 200.0
CAP = 16
GROUP_ELEMENTS = 4096
GROUP_WORDS = 256

EXPECTED_M_AXI = {"input_arena", "wv_q", "output_arena"}
EXPECTED_AXI_LITE = {"control"}

C_ANCHOR = 5.5
C_SCAN = 4.0
C_POOL = 1.4
C_IDENTITY_PER_GROUP = 260.0
FIXED_BOOKKEEPING = 60000.0

CONFIG_GRIDS = {80: (8, 10), 260: (13, 20), 400: (20, 20)}
CONFIG_BUDGETS = (48, 64, 96)


def _text(root: ET.Element, path: str):
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
    match = re.search(
        r"ap_clk\s*\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|\s*([\d.]+)\s*ns\|", text
    )
    if not match:
        return {"clock_target_ns": None, "clock_estimated_ns": None, "clock_uncertainty_ns": None}
    return {
        "clock_target_ns": float(match.group(1)),
        "clock_estimated_ns": float(match.group(2)),
        "clock_uncertainty_ns": float(match.group(3)),
    }


def parse_interface(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    section = text.split("== Interface", 1)
    if len(section) < 2:
        return {"m_axi_bundles": [], "s_axi_bundles": [], "port_names": [], "score_ports": []}
    body = section[1].split("== Register", 1)[0]
    m_axi: set[str] = set()
    s_axi: set[str] = set()
    ports: set[str] = set()
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cols = [col.strip() for col in line.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        name, _direction, _bits, protocol, source, _ctype = cols[:6]
        if name in {"RTL Ports", ""} or name.startswith("-"):
            continue
        if protocol == "m_axi":
            m_axi.add(source)
            ports.add(name)
        elif protocol == "s_axi":
            s_axi.add(source)
            ports.add(name)
    score_ports = sorted(name for name in ports if "score" in name.lower())
    return {
        "m_axi_bundles": sorted(m_axi),
        "s_axi_bundles": sorted(s_axi),
        "axi_lite_bundle_count": len(s_axi),
        "port_names": sorted(ports),
        "score_ports": score_ports,
    }


def parse_pipeline_instances(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Pipeline_VITIS_LOOP" not in line or not line.strip().startswith("|"):
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 10 or not any(re.search(r"\b(ns|us|ms)\b", col) for col in cols):
            continue
        instance = cols[1]
        value = _number(cols[4])
        if value is not None and "ap_start_reg" not in instance:
            result[instance] = value
    return result


def parse_instance_table(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip().startswith("|grp_"):
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 6:
            continue
        if not any(re.search(r"\b(ns|us|ms)\b", col) for col in cols):
            continue
        instance = cols[1]
        value = _number(cols[4])
        if value is not None:
            result[instance] = int(value)
    return result


def parse_loop_table(path: Path) -> list[dict]:
    loops: list[dict] = []
    if not path.exists():
        return loops
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "VITIS_LOOP" not in line or not line.strip().startswith("|"):
            continue
        cols = [col.strip() for col in line.split("|")]
        if len(cols) < 9:
            continue
        name = cols[1].lstrip("-+ ").strip()
        if not name.startswith("VITIS_LOOP"):
            continue
        iteration = cols[4]
        values = [float(v) for v in re.findall(r"[\d.]+", iteration)]
        loops.append(
            {
                "loop": name,
                "latency_min_cycles": _number(cols[2]),
                "latency_max_cycles": _number(cols[3]),
                "iteration_latency": iteration,
                "iteration_latency_max": max(values) if values else None,
                "trip_count": cols[7],
                "pipelined": cols[8],
            }
        )
    return loops


def parse_console(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    warning_lines = re.findall(r"^WARNING:.*$", text, re.MULTILINE)
    return {
        "error_count": len(re.findall(r"^ERROR:", text, re.MULTILINE)),
        "warning_count": len(warning_lines),
        "estimated_fmax_mhz": _number(
            (re.search(r"Estimated Fmax:\s*([\d.]+)\s*MHz", text) or [None, None])[1]
        ),
        "timing_warning_200_871": "HLS 200-871" in text,
        "critical_path_warning_200_1016": "HLS 200-1016" in text,
        "loop_constraint_warning_200_790": "All loop constraints were NOT satisfied" in text,
        "ii_warning_200_885": len(re.findall(r"WARNING: \[HLS 200-885\]", text)),
        "ii_warning_200_878": len(re.findall(r"WARNING: \[HLS 200-878\]", text)),
        "all_loop_constraints_met": "All loop constraints were satisfied" in text,
    }


def _selector_cycles(group_count: int, budget: int, grid_h: int, grid_w: int) -> float:
    if group_count <= budget:
        return group_count * C_IDENTITY_PER_GROUP + FIXED_BOOKKEEPING
    regions = build_fixedpoint_regions(grid_h, grid_w)
    context = set(regions.context_regions)
    row_bounds = [(r * grid_h // 3, (r + 1) * grid_h // 3) for r in range(3)]
    col_bounds = [(c * grid_w // 3, (c + 1) * grid_w // 3) for c in range(3)]
    local = 0.0
    for rr in range(3):
        for cc in range(3):
            rid = rr * 3 + cc
            members = len(regions.members[rid])
            if members <= 0:
                continue
            band = (row_bounds[rr][1] - row_bounds[rr][0]) * (
                col_bounds[cc][1] - col_bounds[cc][0]
            )
            reserve = 1 if rid in context else 0
            allowed = max(0, members - 1 - reserve)
            local += min(CAP, allowed) * band
    pool_words = sum(
        max(0, len(regions.members[rid]) - 2) * GROUP_WORDS
        for rid in regions.context_regions
    )
    bookkeeping = FIXED_BOOKKEEPING + 2.0 * budget * 144.0 + 2.0 * budget * budget
    return (
        C_ANCHOR * group_count
        + C_SCAN * local
        + C_POOL * pool_words
        + bookkeeping
    )


def _compact_overhead_cycles(budget: int) -> float:
    """Arena pre-zero (one 128-bit word/cycle) + bounded metadata packing."""
    total_words = fpga_transfer_layout(budget).total_bytes // 16
    metadata_words = (2 * budget + 3 * budget + 15) // 16
    return float(total_words + metadata_words + 32)


def derive_latency(wv_load, pass1, pass2) -> dict:
    derived: dict[str, dict] = {}
    per_group = (pass1 or 0.0) + (pass2 or 0.0)
    for group_count in (80, 260, 400):
        grid_h, grid_w = CONFIG_GRIDS[group_count]
        for budget in CONFIG_BUDGETS:
            scorer = (wv_load or 0.0) + group_count * per_group
            selector = _selector_cycles(group_count, budget, grid_h, grid_w)
            overhead = _compact_overhead_cycles(budget)
            cycles = scorer + selector + overhead
            derived[f"G{group_count}_K{budget}"] = {
                "group_count": group_count,
                "budget": budget,
                "grid": [grid_h, grid_w],
                "scorer_cycles": scorer,
                "selector_cycles_estimate": selector,
                "compact_overhead_cycles_estimate": overhead,
                "total_cycles_estimate": cycles,
                "ms_at_200mhz": round(cycles / (TARGET_FMAX_MHZ * 1e6) * 1000.0, 4),
            }
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clock-ns", type=float, default=5.0)
    args = parser.parse_args()

    report_dir = args.report_dir
    xml = parse_csynth_xml(report_dir / "csynth.xml")
    top_rpt = report_dir / f"{TOP}_csynth.rpt"
    clock = parse_clock_table(top_rpt)
    interface = parse_interface(top_rpt)
    scorer_rpt = report_dir / f"{SCORER_SUBMODULE}_csynth.rpt"
    scorer_loops = parse_loop_table(scorer_rpt)
    scorer_pipelines = parse_pipeline_instances(scorer_rpt)
    if not scorer_pipelines:
        for candidate in sorted(report_dir.glob("*scorer_core*_csynth.rpt")):
            scorer_pipelines = parse_pipeline_instances(candidate)
            scorer_loops = parse_loop_table(candidate)
            if scorer_pipelines:
                break
    top_loops = parse_loop_table(top_rpt)
    top_pipelines = parse_instance_table(top_rpt)
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

    wv_load = max(scorer_pipelines.values()) if scorer_pipelines else None
    ordered = sorted(
        (loop for loop in scorer_loops if loop["iteration_latency_max"] is not None),
        key=lambda loop: loop["latency_max_cycles"] or 0.0,
        reverse=True,
    )
    pass1 = ordered[0]["iteration_latency_max"] if len(ordered) > 0 else None
    pass2 = ordered[1]["iteration_latency_max"] if len(ordered) > 1 else None
    per_group = (pass1 or 0.0) + (pass2 or 0.0)

    derived = derive_latency(wv_load, pass1, pass2)

    interface_ok = (
        set(interface["m_axi_bundles"]) == EXPECTED_M_AXI
        and set(interface["s_axi_bundles"]) == EXPECTED_AXI_LITE
        and interface["score_ports"] == []
    )

    gate = {
        "zero_errors": console["error_count"] == 0,
        "timing_met": bool(slack is not None and slack >= 0.0),
        "estimated_period_le_5ns": bool(estimated is not None and estimated <= 5.0),
        "all_resources_under_50_percent": all(v < 50.0 for v in utilization.values()),
        "exactly_three_m_axi_bundles": set(interface["m_axi_bundles"]) == EXPECTED_M_AXI,
        "exactly_one_axilite_bundle": set(interface["s_axi_bundles"]) == EXPECTED_AXI_LITE,
        "no_external_score_port": interface_ok,
        "derived_configs_finite": all(
            row["total_cycles_estimate"] is not None for row in derived.values()
        ),
        "derived_g400_k96_le_6m_cycles": bool(
            derived.get("G400_K96") is not None
            and derived["G400_K96"]["total_cycles_estimate"] <= 6_000_000
        ),
        "derived_g80_k48_le_2m_cycles": bool(
            derived.get("G80_K48") is not None
            and derived["G80_K48"]["total_cycles_estimate"] <= 2_000_000
        ),
        "derived_g260_k64_le_4m_cycles": bool(
            derived.get("G260_K64") is not None
            and derived["G260_K64"]["total_cycles_estimate"] <= 4_000_000
        ),
    }
    gate["csynth_pass"] = all(gate.values())

    summary = {
        "tool": xml["lifetime_meta"].get("TopModelName"),
        "flow_target": xml["lifetime_meta"].get("FlowTarget"),
        "top": TOP,
        "part": xml["lifetime_meta"].get("Part"),
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
        "scorer_phase": {
            "wv_load_cycles": wv_load,
            "pass1_cycles_per_group": pass1,
            "pass2_cycles_per_group": pass2,
            "per_group_cycles": per_group,
        },
        "derived_model": {
            "kind": "formula estimate, not a measurement",
            "constants": {
                "c_anchor": C_ANCHOR,
                "c_scan": C_SCAN,
                "c_pool": C_POOL,
                "c_identity_per_group": C_IDENTITY_PER_GROUP,
                "fixed_bookkeeping": FIXED_BOOKKEEPING,
                "local_candidate_cap": CAP,
                "scorer_per_group": per_group,
            },
            "configs": derived,
        },
        "interface": interface,
        "resources_used": used,
        "resources_available": available,
        "resource_utilization_percent": utilization,
        "console": console,
        "critical_path": {
            "warning_200_1016": console["critical_path_warning_200_1016"],
            "note": "no timing violation -> the tool reports no violating critical "
                    "path; the effective budget (target - uncertainty) is the "
                    "limiting constraint",
        },
        "scorer_loops": scorer_loops,
        "scorer_pipelines": scorer_pipelines,
        "selector_pipelines": top_pipelines,
        "selector_loops": top_loops,
        "gate": gate,
    }
    scorer_max = (wv_load or 0.0) + 4096 * per_group
    selector_max = _selector_cycles(4096, 96, 64, 64)
    summary["derived_model"]["validation_at_max_bounds"] = {
        "scorer_cycles": scorer_max,
        "selector_cycles_estimate": selector_max,
        "compact_overhead_cycles_estimate": _compact_overhead_cycles(96),
        "total_cycles_estimate": scorer_max + selector_max + _compact_overhead_cycles(96),
        "tool_reported_worst_cycles": xml["worst_latency_cycles"],
        "model_vs_tool_ratio": round(
            (scorer_max + selector_max + _compact_overhead_cycles(96))
            / (xml["worst_latency_cycles"] or 1.0),
            4,
        ),
        "note": "G=4096, K=96, local_cap=16, 64x64 grid (compile-time bounds)",
    }
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["gate"], indent=2))
    print(json.dumps({
        "estimated_clock_ns": summary["estimated_clock_ns"],
        "timing_slack_ns": slack,
        "scorer_phase": summary["scorer_phase"],
        "derived": {k: derived[k]["total_cycles_estimate"]
                    for k in ("G80_K48", "G260_K64", "G400_K96")},
        "resource_utilization_percent": utilization,
        "m_axi_bundles": interface["m_axi_bundles"],
        "s_axi_bundles": interface["s_axi_bundles"],
        "score_ports": interface["score_ports"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
