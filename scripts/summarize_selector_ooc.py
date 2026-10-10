#!/usr/bin/env python3
"""Collect and summarize the Selector v4 HLS-IP OOC implementation reports.

This is the read-only collector for the flow driven by
``scripts/export_selector_v4_ip_ooc.tcl`` (Vitis HLS ``csynth_design`` +
``export_design -rtl verilog -format ip_catalog -flow impl``, i.e. the Vivado
out-of-context ``synth_design -mode out_of_context`` + ``opt_design`` +
``place_design`` + ``route_design`` that HLS generates).

Only small text/JSON summaries are copied into the repository.  The Vivado
project, caches, checkpoints and the exported IP zip stay under ``/tmp``.

Usage:
    python3 scripts/summarize_selector_ooc.py \
        --project-dir /tmp/robobrain_token_selector_hls_export_v4 \
        --ip-zip /tmp/robobrain_selector_ip_ooc_v4/ip/token_selector_top_packed_ip.zip \
        --artifact-dir artifacts/token_selector/2026-09-28-selector-ooc-v4 \
        --hls-log /tmp/robobrain_selector_export_run_v4/console_export_v4.log \
        --df-before /tmp/robobrain_selector_export_run_v4/df_before.txt \
        --df-after /tmp/robobrain_selector_export_run_v4/df_after.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FLOAT = r"-?\d+\.\d+"
INT = r"\d+"


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_route_status(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def grab(label: str) -> Optional[int]:
        m = re.search(rf"#\s*of\s*{re.escape(label)}\s*[.\s]*:\s*(\d+)", text)
        return int(m.group(1)) if m else None

    out["logical_nets"] = grab("logical nets")
    out["routable_nets"] = grab("routable nets")
    out["fully_routed_nets"] = grab("fully routed nets")
    out["routing_errors"] = grab("nets with routing errors")
    if out["routable_nets"] is not None and out["fully_routed_nets"] is not None:
        out["unrouted_nets"] = out["routable_nets"] - out["fully_routed_nets"]
    else:
        out["unrouted_nets"] = None
    return out


def parse_timing_summary(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    idx = text.find("Design Timing Summary")
    if idx < 0:
        return out
    seg = text[idx: idx + 4000]
    row = re.compile(
        rf"^\s*({FLOAT})\s+({FLOAT})\s+({INT})\s+({INT})\s+"
        rf"({FLOAT})\s+({FLOAT})\s+({INT})\s+({INT})\s+"
        rf"({FLOAT})\s+({FLOAT})\s+({INT})\s+({INT})\s*$",
        re.M,
    )
    m = row.search(seg)
    if not m:
        return out
    keys = [
        "wns_ns", "tns_ns", "tns_failing_endpoints", "tns_total_endpoints",
        "whs_ns", "ths_ns", "ths_failing_endpoints", "ths_total_endpoints",
        "wpws_ns", "tpws_ns", "tpws_failing_endpoints", "tpws_total_endpoints",
    ]
    for k, v in zip(keys, m.groups()):
        out[k] = float(v) if "." in v else int(v)
    cidx = text.find("Clock Summary")
    if cidx >= 0:
        cseg = text[cidx: cidx + 1500]
        cm = re.search(
            rf"^\s*(\S+)\s+\{{[^}}]*\}}\s+({FLOAT})\s+({FLOAT})\s*$", cseg, re.M
        )
        if cm:
            out["clock_name"] = cm.group(1)
            out["clock_period_ns"] = float(cm.group(2))
            out["clock_frequency_mhz"] = float(cm.group(3))
    return out


def parse_utilization(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    labels = {
        "lut": r"CLB LUTs",
        "ff": r"CLB Registers",
        "bram_tile": r"Block RAM Tile",
        "ramb36": r"RAMB36/FIFO\*?",
        "ramb18": r"RAMB18",
        "dsp": r"DSPs",
    }
    for key, label in labels.items():
        m = re.search(
            rf"^\|\s*{label}\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            text,
            re.M,
        )
        if m:
            out[key] = {
                "used": float(m.group(1)) if "." in m.group(1) else int(m.group(1)),
                "fixed": int(m.group(2)),
                "prohibited": int(m.group(3)),
                "available": int(m.group(4)),
                "util_percent": float(m.group(5)),
            }
    # RAMB18-equivalent BRAM utilisation (csynth reports BRAM_18K / 1824).
    ramb36 = out.get("ramb36", {}).get("used")
    ramb18 = out.get("ramb18", {}).get("used")
    if ramb36 is not None and ramb18 is not None:
        out["bram18_equivalent"] = {
            "used": 2 * ramb36 + ramb18,
            "available": 1824,
            "util_percent": round(100.0 * (2 * ramb36 + ramb18) / 1824.0, 2),
        }
    return out


def parse_drc(text: str) -> Dict[str, Any]:
    idx = text.find("REPORT SUMMARY")
    seg = text[idx: idx + 8000] if idx >= 0 else text
    row = re.compile(
        r"^\|\s*([A-Za-z0-9\-_]+)\s*\|\s*(Error|Critical Warning|Warning|Advisory)\s*\|\s*(.*?)\s*\|\s*(\d+)\s*\|\s*$",
        re.M,
    )
    rules: List[Dict[str, Any]] = []
    by_severity: Dict[str, int] = {}
    for m in row.finditer(seg):
        rule, severity, desc, count = m.group(1), m.group(2), m.group(3), int(m.group(4))
        rules.append({"rule": rule, "severity": severity, "description": desc.strip(), "violations": count})
        by_severity[severity] = by_severity.get(severity, 0) + count
    out = {
        "by_severity_violations": by_severity,
        "error_violations": by_severity.get("Error", 0),
        "critical_warning_violations": by_severity.get("Critical Warning", 0),
        "warning_violations": by_severity.get("Warning", 0),
        "advisory_violations": by_severity.get("Advisory", 0),
        "rules": rules,
    }
    m = re.search(r"Violations found:\s*(\d+)", text)
    if m:
        out["total_violations"] = int(m.group(1))
    return out


def parse_critical_path(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    patterns = {
        "slack": r"Slack\s*\((MET|VIOLATED)\)\s*:\s*(" + FLOAT + r")ns",
        "source": r"^\s*Source:\s*(.+)$",
        "destination": r"^\s*Destination:\s*(.+)$",
        "path_group": r"^\s*Path Group:\s*(.+)$",
        "path_type": r"^\s*Path Type:\s*(.+)$",
        "requirement": r"^\s*Requirement:\s*(.+)$",
        "data_path_delay": r"^\s*Data Path Delay:\s*(.+)$",
        "logic_levels": r"^\s*Logic Levels:\s*(.+)$",
        "clock_uncertainty": r"^\s*Clock Uncertainty:\s*(.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.M)
        if m:
            out[key] = m.group(1).strip()
    if "slack" in out:
        m = re.search(r"Slack\s*\((MET|VIOLATED)\)\s*:\s*(" + FLOAT + r")ns", text)
        if m:
            out["slack_status"] = m.group(1)
            out["slack_ns"] = float(m.group(2))
    # first path block = up to the next "Max Delay Paths"/"Slack ("
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if "Slack (" in ln), None)
    if start is not None:
        block: List[str] = []
        for ln in lines[start:]:
            if block and re.match(r"^\s*Slack \(", ln):
                break
            block.append(ln)
            if len(block) >= 60:
                break
        out["first_path_block"] = "\n".join(block)
    return out


def parse_tool_version(text: str) -> Optional[str]:
    m = re.search(r"\|\s*Tool Version\s*:\s*(.+?)\s*\|", text)
    return m.group(1).strip() if m else None


def parse_design_name(text: str) -> Optional[str]:
    m = re.search(r"\|\s*Design\s*:\s*(.+?)\s*\|", text)
    return m.group(1).strip() if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-dir", default="/tmp/robobrain_token_selector_hls_export_v4")
    ap.add_argument("--ip-zip", default="/tmp/robobrain_selector_ip_ooc_v4/ip/token_selector_top_packed_ip.zip")
    ap.add_argument("--artifact-dir", default="artifacts/token_selector/2026-09-28-selector-ooc-v4")
    ap.add_argument("--hls-log", default="")
    ap.add_argument("--df-before", default="")
    ap.add_argument("--df-after", default="")
    ap.add_argument("--repo-root", default="")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parent.parent
    project_dir = Path(args.project_dir)
    art = Path(args.artifact_dir)
    if not art.is_absolute():
        art = repo_root / art
    art.mkdir(parents=True, exist_ok=True)

    impl_root = project_dir / "solution1" / "impl" / "verilog"
    runs_dir = impl_root / "project.runs" / "impl_1"
    hls_report_dir = impl_root / "report"

    summary: Dict[str, Any] = {
        "schema": "token-selector-ooc-summary-v4",
        "date": "2026-09-28",
        "flow": "Vitis HLS 2023.2 csynth_design + export_design -rtl verilog -format ip_catalog -flow impl (Vivado OOC synth+opt+place+route)",
        "synthesis_top": "token_selector_top_packed",
        "part": "xczu9eg-ffvb1156-2-e",
        "clock": {"name": "ap_clk", "period_ns": 5.0, "target_mhz": 200.0,
                  "hls_default_uncertainty_ns": 1.35,
                  "uncertainty_source": "Vitis HLS default create_clock uncertainty (27% of period), unchanged; exported OOC XDC keeps only create_clock -period 5.000"},
        "tmp_paths": {
            "hls_project": str(project_dir),
            "vivado_impl": str(impl_root),
            "ip_zip": args.ip_zip,
        },
        "reproduce": {
            "hls_tcl": "scripts/export_selector_v4_ip_ooc.tcl",
            "collector": "scripts/summarize_selector_ooc.py",
            "hls_command": ("export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH && "
                            "/home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f scripts/export_selector_v4_ip_ooc.tcl"),
        },
    }

    # ---- source hashes (frozen design) -------------------------------------
    source_hashes: Dict[str, str] = {}
    for rel in ["fpga_accel/token_selector_hls/src/token_selector_hls.cpp",
                "fpga_accel/token_selector_hls/include/token_selector_hls.h",
                "scripts/export_selector_v4_ip_ooc.tcl",
                "scripts/summarize_selector_ooc.py"]:
        p = repo_root / rel
        if p.exists():
            source_hashes[rel] = sha256_file(p)
    summary["source_sha256"] = source_hashes

    # ---- IP zip ------------------------------------------------------------
    ip_zip = Path(args.ip_zip)
    ip_info: Dict[str, Any] = {"path": str(ip_zip), "exists": ip_zip.exists()}
    if ip_zip.exists():
        ip_info["size_bytes"] = ip_zip.stat().st_size
        ip_info["sha256"] = sha256_file(ip_zip)
    summary["export_ip"] = ip_info

    # ---- locate reports ----------------------------------------------------
    def first(directory: Path, patterns: List[str]) -> Optional[Path]:
        if not directory.exists():
            return None
        for pat in patterns:
            hits = sorted(directory.glob(pat))
            if hits:
                return hits[0]
        return None

    route_src = first(runs_dir, ["*_route_status.rpt"])
    timing_sum_src = first(runs_dir, ["*_timing_summary_routed.rpt"])
    util_src = first(runs_dir, ["*_utilization_placed.rpt"])
    drc_src = first(runs_dir, ["*_drc_routed.rpt"])
    clock_src = first(runs_dir, ["*_clock_utilization_routed.rpt"])
    paths_src = first(hls_report_dir, ["*_timing_paths_routed.rpt"])
    hls_status_src = first(hls_report_dir, ["*_status_routed.rpt"])
    hls_timing_src = first(hls_report_dir, ["*_timing_routed.rpt"])
    hls_util_src = first(hls_report_dir, ["*_utilization_routed.rpt"])

    copied: Dict[str, str] = {}

    def copy_report(src: Optional[Path], dest_name: str) -> Optional[str]:
        if src is None or not src.exists():
            return None
        dest = art / dest_name
        shutil.copy2(src, dest)
        copied[dest_name] = sha256_file(dest)
        return str(dest)

    copy_report(route_src, "route_status_routed.rpt")
    copy_report(timing_sum_src, "timing_summary_routed.rpt")
    copy_report(util_src, "utilization_placed.rpt")
    copy_report(clock_src, "clock_utilization_routed.rpt")
    copy_report(paths_src, "critical_paths_routed.rpt")
    copy_report(hls_status_src, "route_status_hls_routed.rpt")
    copy_report(hls_timing_src, "timing_hls_routed.rpt")
    copy_report(hls_util_src, "utilization_hls_routed.rpt")

    # DRC full report is ~0.6 MiB; keep the summary section in the repo.
    if drc_src and drc_src.exists():
        text = read_text(drc_src)
        # The table of contents also contains "1. REPORT SUMMARY"; take the
        # last occurrence (the real section header) up to the details header.
        idx = text.rfind("1. REPORT SUMMARY")
        if idx >= 0:
            end = text.find("2. REPORT DETAILS", idx)
            trimmed = text[:end] if end > idx else text[: idx + 6000]
        else:
            trimmed = text[:20000]
        (art / "drc_routed_summary.rpt").write_text(trimmed)
        copied["drc_routed_summary.rpt"] = sha256_file(art / "drc_routed_summary.rpt")

    # ---- parse -------------------------------------------------------------
    route_text = read_text(route_src) if route_src else ""
    timing_text = read_text(timing_sum_src) if timing_sum_src else ""
    util_text = read_text(util_src) if util_src else ""
    drc_text = read_text(drc_src) if drc_src else ""
    paths_text = read_text(paths_src) if paths_src else ""

    summary["tool_version"] = parse_tool_version(timing_text or util_text or route_text)
    summary["design"] = parse_design_name(timing_text or util_text or route_text)

    route = parse_route_status(route_text)
    timing = parse_timing_summary(timing_text)
    util = parse_utilization(util_text)
    drc = parse_drc(drc_text)
    crit = parse_critical_path(paths_text)

    summary["ooc_route"] = route
    summary["timing"] = timing
    summary["resources"] = util
    summary["drc"] = drc

    # ---- hard gates --------------------------------------------------------
    def ok(cond: Optional[bool]) -> bool:
        return bool(cond)

    wns = timing.get("wns_ns")
    whs = timing.get("whs_ns")
    gates = {
        "wns_ge_0": ok(wns is not None and wns >= 0.0),
        "whs_ge_0": ok(whs is not None and whs >= 0.0),
        "unrouted_eq_0": ok(route.get("unrouted_nets") == 0 and route.get("routing_errors") == 0),
        "drc_error_eq_0": ok(drc.get("error_violations") == 0),
        "lut_lt_50": ok(util.get("lut", {}).get("util_percent", 100.0) < 50.0),
        "ff_lt_50": ok(util.get("ff", {}).get("util_percent", 100.0) < 50.0),
        "bram_lt_50": ok(util.get("bram_tile", {}).get("util_percent", 100.0) < 50.0
                         and util.get("bram18_equivalent", {}).get("util_percent", 100.0) < 50.0),
        "dsp_lt_50": ok(util.get("dsp", {}).get("util_percent", 100.0) < 50.0),
    }
    summary["gate_checks"] = gates
    summary["status"] = "PASS" if all(gates.values()) else "REDESIGN_REQUIRED"

    # ---- HLS export log excerpt (errors / warnings) ------------------------
    hls_log_info: Dict[str, Any] = {}
    if args.hls_log:
        logp = Path(args.hls_log)
        if logp.exists():
            ltext = read_text(logp)
            hls_log_info["path"] = str(logp)
            hls_log_info["size_bytes"] = logp.stat().st_size
            hls_log_info["sha256"] = sha256_file(logp)
            hls_log_info["error_lines"] = len(re.findall(r"^ERROR:", ltext, re.M))
            hls_log_info["hls_error_lines"] = len(re.findall(r"ERROR: \[HLS", ltext))
            m = re.search(r"Vitis HLS.*?v(\d{4}\.\d)", ltext)
            if m:
                hls_log_info["hls_version"] = "Vitis HLS " + m.group(1)
    summary["hls_log"] = hls_log_info

    # ---- critical path artifact -------------------------------------------
    crit_md = ["# Selector v4 OOC post-route critical path (Vivado report_timing)", ""]
    for k in ["slack_status", "slack_ns", "source", "destination", "path_group",
              "path_type", "requirement", "data_path_delay", "logic_levels",
              "clock_uncertainty"]:
        if k in crit:
            crit_md.append(f"- {k}: {crit[k]}")
    crit_md.append("")
    crit_md.append("```text")
    crit_md.append(crit.get("first_path_block", "(no timing path report found)"))
    crit_md.append("```")
    (art / "critical_path_routed.md").write_text("\n".join(crit_md) + "\n")
    copied["critical_path_routed.md"] = sha256_file(art / "critical_path_routed.md")
    summary["critical_path"] = {k: v for k, v in crit.items() if k != "first_path_block"}

    # ---- df snapshots ------------------------------------------------------
    df_info: Dict[str, Any] = {}
    for tag, src in [("before", args.df_before), ("after", args.df_after)]:
        if src and Path(src).exists():
            text = read_text(Path(src))
            dest = art / f"df_{tag}.txt"
            dest.write_text(text)
            copied[dest.name] = sha256_file(dest)
            df_info[tag] = text.strip().splitlines()
    if df_info:
        summary["df"] = df_info

    summary["artifacts"] = copied
    (art / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")

    # ---- machine-readable mirror for the harness --------------------------
    print(json.dumps({
        "status": summary["status"],
        "gate_checks": gates,
        "timing": timing,
        "ooc_route": route,
        "resources": {k: v for k, v in util.items()},
        "drc_error_violations": drc.get("error_violations"),
        "critical_path": summary["critical_path"],
        "export_ip": ip_info,
        "artifact_dir": str(art),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
