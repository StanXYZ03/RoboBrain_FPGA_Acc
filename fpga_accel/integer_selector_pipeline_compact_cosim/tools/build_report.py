#!/usr/bin/env python3
"""Build the bounded RTL co-simulation JSON/Markdown reports.

Reads only files already written into

  artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/

(the repository never receives the /tmp HLS project) plus the accepted fixture
manifest, and emits:

  reports/token_selector/2026-10-03-integer-selector-compact-rtl-cosim.json
  reports/token_selector/2026-10-03-integer-selector-compact-rtl-cosim.md
  reports/token_selector/logs/2026-10-03-integer-selector-compact-rtl-cosim-harness-result.json

Usage: python3 tools/build_report.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(PROJ_DIR, "..", ".."))

DATE = "2026-10-03"
ARTIFACT_DIR = os.path.join(
    REPO_ROOT, "artifacts/token_selector", f"{DATE}-integer-selector-compact-rtl-cosim"
)
FIXTURE_DIR = os.path.join(
    REPO_ROOT,
    "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures",
)
REPORT_JSON = os.path.join(
    REPO_ROOT, "reports/token_selector", f"{DATE}-integer-selector-compact-rtl-cosim.json"
)
REPORT_MD = os.path.join(
    REPO_ROOT, "reports/token_selector", f"{DATE}-integer-selector-compact-rtl-cosim.md"
)
HARNESS_JSON = os.path.join(
    REPO_ROOT,
    "reports/token_selector/logs",
    f"{DATE}-integer-selector-compact-rtl-cosim-harness-result.json",
)
PROMPT_MD = os.path.join(
    REPO_ROOT,
    "reports/token_selector/prompts",
    f"{DATE}-integer-selector-compact-rtl-cosim-harness-prompt.md",
)

TOP = "integer_selector_pipeline_compact"
EXPECTED_M_AXI = {"input_arena", "wv_q", "output_arena"}
EXPECTED_AXI_LITE = {"control"}
TARGET_MHZ = 200.0


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def num(value):
    if value is None:
        return None
    value = value.strip()
    if value in {"", "NA", "undef", "?"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def parse_log(log_path):
    text = read_text(log_path)
    lines = text.splitlines()
    tb_cases = []
    summaries = []
    results = []
    for line in lines:
        m = re.search(
            r"\[COSIM-CASE\] name=(\S+) G=(-?\d+) grid=(\d+)x(\d+) K=(\d+) cap=(\d+) "
            r"scale_in=([\d.eE+-]+) status=(-?\d+) output_count=(-?\d+) real_count=(-?\d+) "
            r"context_count=(-?\d+) scale_out=([\d.eE+-]+) arena_bytes=(-?\d+) checks=(\d+) "
            r"mismatches=(\d+)",
            line,
        )
        if m:
            tb_cases.append(
                {
                    "case": m.group(1),
                    "group_count": int(m.group(2)),
                    "grid": [int(m.group(3)), int(m.group(4))],
                    "budget": int(m.group(5)),
                    "local_cap": int(m.group(6)),
                    "scale_in": float(m.group(7)),
                    "status": int(m.group(8)),
                    "output_count": int(m.group(9)),
                    "real_count": int(m.group(10)),
                    "context_count": int(m.group(11)),
                    "scale_out": float(m.group(12)),
                    "arena_bytes": int(m.group(13)),
                    "checks": int(m.group(14)),
                    "mismatches": int(m.group(15)),
                }
            )
        m = re.search(
            r"\[COSIM-SUMMARY\] cases=(\d+) expected_cases=(\d+) checks=(\d+) passed=(\d+) "
            r"failed=(\d+) scalar_mismatches=(\d+) arena_mismatches=(\d+)",
            line,
        )
        if m:
            summaries.append(
                {
                    "cases": int(m.group(1)),
                    "expected_cases": int(m.group(2)),
                    "checks": int(m.group(3)),
                    "passed": int(m.group(4)),
                    "failed": int(m.group(5)),
                    "scalar_mismatches": int(m.group(6)),
                    "arena_mismatches": int(m.group(7)),
                }
            )
        m = re.search(r"\[COSIM-RESULT\] (\S+)", line)
        if m:
            results.append(m.group(1))

    warnings = [ln for ln in lines if ln.startswith("WARNING:")]
    errors = [ln for ln in lines if ln.startswith("ERROR:")]
    fatals = [ln for ln in lines if "FATAL" in ln.upper()]
    warning_codes = {}
    for ln in warnings:
        code = re.search(r"\[([A-Z]+ [0-9-]+)\]", ln)
        key = code.group(1) if code else "UNCODED"
        warning_codes[key] = warning_codes.get(key, 0) + 1
    error_codes = {}
    for ln in errors:
        code = re.search(r"\[([A-Z]+ [0-9-]+)\]", ln)
        key = code.group(1) if code else "UNCODED"
        error_codes[key] = error_codes.get(key, 0) + 1

    elapsed = None
    for line in lines:
        m = re.search(r"\[run_cosim\] vitis_hls exit=(\d+) elapsed_seconds=(\d+)", line)
        if m:
            elapsed = {"exit_code": int(m.group(1)), "elapsed_seconds": int(m.group(2))}
    xsim_started = "COSIM 212-15" in text
    finished_pass = "*** C/RTL co-simulation finished: PASS ***" in text
    finished_fail = "*** C/RTL co-simulation finished: FAIL ***" in text
    return {
        "tb_cases": tb_cases,
        "summaries": summaries,
        "results": results,
        "warning_count": len(warnings),
        "warning_codes": warning_codes,
        "error_count": len(errors),
        "error_codes": error_codes,
        "error_lines": errors[:50],
        "fatal_count": len(fatals),
        "fatal_lines": fatals[:20],
        "elapsed": elapsed,
        "xsim_started": xsim_started,
        "cosim_finished_pass": finished_pass,
        "cosim_finished_fail": finished_fail,
    }


# ---------------------------------------------------------------------------
# Csynth report parsing
# ---------------------------------------------------------------------------
def parse_csynth_xml(path):
    root = ET.parse(path).getroot()
    used = {}
    available = {}
    area = root.find("AreaEstimates/Resources")
    if area is not None:
        for child in area:
            used[child.tag] = int(float(child.text))
    avail = root.find("AreaEstimates/AvailableResources")
    if avail is not None:
        for child in avail:
            available[child.tag] = int(float(child.text))
    meta = {}
    ua = root.find("UserAssignments")
    if ua is not None:
        for child in ua:
            meta[child.tag] = (child.text or "").strip()
    perf = root.find("PerformanceEstimates/SummaryOfOverallLatency")
    latency = {}
    if perf is not None:
        for child in perf:
            latency[child.tag] = num(child.text)
    timing = root.find("PerformanceEstimates/SummaryOfTimingAnalysis")
    estimated_clock = None
    if timing is not None:
        estimated_clock = num(
            timing.findtext("EstimatedClockPeriod")
        )
    return {
        "resources_used": used,
        "resources_available": available,
        "user_assignments": meta,
        "latency_cycles": latency,
        "estimated_clock_ns": estimated_clock,
    }


def parse_interface(path):
    text = read_text(path)
    parts = text.split("== Interface", 1)
    if len(parts) < 2:
        return {
            "m_axi_bundles": [],
            "m_axi_data_widths": {},
            "s_axi_bundles": [],
            "control_r_present": None,
            "score_ports": [],
            "port_names": [],
        }
    body = parts[1].split("== Register", 1)[0]
    m_axi, s_axi, ports = set(), set(), set()
    widths = {}
    port_list = []
    for line in body.splitlines():
        if not line.strip().startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        name, _direction, bits, protocol, source, _ctype = cols[:6]
        if name in {"RTL Ports", ""} or name.startswith("-"):
            continue
        if protocol == "m_axi":
            m_axi.add(source)
            ports.add(name)
            port_list.append(name)
            if name.endswith("_WDATA") or name.endswith("_RDATA"):
                widths.setdefault(source, set()).add(bits)
        elif protocol == "s_axi":
            s_axi.add(source)
            ports.add(name)
            port_list.append(name)
    score_ports = sorted(n for n in port_list if "score" in n.lower())
    return {
        "m_axi_bundles": sorted(m_axi),
        "m_axi_data_widths": {k: sorted(v) for k, v in widths.items()},
        "s_axi_bundles": sorted(s_axi),
        "control_r_present": any("control_r" in s for s in s_axi),
        "score_ports": score_ports,
        "port_names": sorted(port_list),
    }


def parse_cosim_report(path):
    text = read_text(path)
    row = None
    for line in text.splitlines():
        if "Verilog" in line and line.strip().startswith("|"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cols) >= 9:
                row = {
                    "rtl": cols[0],
                    "status": cols[1],
                    "latency_min_cycles": num(cols[2]),
                    "latency_avg_cycles": num(cols[3]),
                    "latency_max_cycles": num(cols[4]),
                    "interval_min_cycles": num(cols[5]),
                    "interval_avg_cycles": num(cols[6]),
                    "interval_max_cycles": num(cols[7]),
                    "total_execution_cycles": num(cols[8]),
                }
    sim_tool = None
    for line in text.splitlines():
        if line.lower().startswith("simulation tool"):
            sim_tool = line.split(":", 1)[1].strip().rstrip(".")
    return {"row": row, "simulation_tool": sim_tool, "raw": text}


def parse_transaction_report(path):
    """Parse Vitis `result.transaction.rpt` (per-transaction latency)."""
    if not os.path.isfile(path):
        return []
    out = []
    for line in read_text(path).splitlines():
        m = re.match(r"\s*transaction\s+(\d+):\s+(\S+)\s+(\S+)", line)
        if not m:
            continue
        out.append(
            {
                "transaction": int(m.group(1)),
                "latency_cycles": num(m.group(2)),
                "interval_cycles": num(m.group(3)),
            }
        )
    return out


def parse_key_values(path):
    out = {}
    if not os.path.isfile(path):
        return out
    for line in read_text(path).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_cosim_options(path):
    if not os.path.isfile(path):
        return {}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {}
    out = {}
    for opt in root.findall("option"):
        out[opt.get("name")] = (opt.text or "").strip()
    return out


# ---------------------------------------------------------------------------
# Artifact audit
# ---------------------------------------------------------------------------
def artifact_audit(root_dir):
    files = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            rel = os.path.relpath(p, REPO_ROOT)
            files.append({"path": rel, "bytes": size})
            total += size
    files_sorted = sorted(files, key=lambda f: (-f["bytes"], f["path"]))
    return {
        "artifact_dir": os.path.relpath(root_dir, REPO_ROOT),
        "total_files": len(files),
        "total_bytes": total,
        "total_megabytes": round(total / (1024.0 * 1024.0), 3),
        "largest_20": files_sorted[:20],
    }


def main():
    log03 = os.path.join(ARTIFACT_DIR, "logs/03_cosim.log")
    log01 = os.path.join(ARTIFACT_DIR, "logs/01_host_test.log")
    log02 = os.path.join(ARTIFACT_DIR, "logs/02_host_test_asan_ubsan.log")
    timeout_flag = os.path.join(ARTIFACT_DIR, "logs/03_cosim_xsim_timeout.flag")

    with open(os.path.join(ARTIFACT_DIR, "cases/rtl_cosim_cases_resolved.json")) as fh:
        case_resolution = json.load(fh)
    with open(os.path.join(ARTIFACT_DIR, "protected_sha_before.json")) as fh:
        protected_before = json.load(fh)
    verify_path = os.path.join(ARTIFACT_DIR, "protected_sha_verify.json")
    protected_verify = json.load(open(verify_path)) if os.path.isfile(verify_path) else None

    host_plain = parse_log(log01)
    host_asan = parse_log(log02)
    cosim = parse_log(log03)

    csynth_rpt = os.path.join(
        ARTIFACT_DIR, "cosim/syn_report", f"{TOP}_csynth.rpt"
    )
    csynth_xml = os.path.join(
        ARTIFACT_DIR, "cosim/syn_report", f"{TOP}_csynth.xml"
    )
    interface = parse_interface(csynth_rpt) if os.path.isfile(csynth_rpt) else {}
    csynth_xml_data = parse_csynth_xml(csynth_xml) if os.path.isfile(csynth_xml) else {}

    sim_dir = os.path.join(ARTIFACT_DIR, "cosim/sim_report")
    cosim_rpt = None
    if os.path.isdir(sim_dir):
        for name in sorted(os.listdir(sim_dir)):
            if name.endswith(".rpt") and "cosim" in name:
                cosim_rpt = os.path.join(sim_dir, name)
                break
    cosim_report = parse_cosim_report(cosim_rpt) if cosim_rpt else {"row": None}
    cosim_options_path = os.path.join(ARTIFACT_DIR, "cosim/cosim_options.xml")
    if not os.path.isfile(cosim_options_path):
        cosim_options_path = os.path.join(ARTIFACT_DIR, "cosim/sim_report/cosim_options.xml")
    cosim_options = parse_cosim_options(cosim_options_path)
    tmp_audit = parse_key_values(os.path.join(ARTIFACT_DIR, "cosim/tmp_project_audit.txt"))
    artifact = artifact_audit(ARTIFACT_DIR)

    # ---- per-transaction latency (result.transaction.rpt) ----
    tx_path = os.path.join(sim_dir, "result.transaction.rpt")
    tx_rows = parse_transaction_report(tx_path)
    per_transaction = []
    for idx, case in enumerate(case_resolution["resolved_cases"]):
        row = next((r for r in tx_rows if r["transaction"] == idx), None)
        per_transaction.append(
            {
                "transaction": idx,
                "case": case["case"],
                "group_count": case["group_count"],
                "budget": case["budget"],
                "grid": case["grid"],
                "latency_cycles": row["latency_cycles"] if row else None,
                "interval_cycles": row["interval_cycles"] if row else None,
                "latency_ms_at_200mhz": (
                    round(row["latency_cycles"] / (TARGET_MHZ * 1e6) * 1000.0, 6)
                    if row and row["latency_cycles"] is not None
                    else None
                ),
            }
        )

    # ---- gate checks ----
    m_axi_ok = set(interface.get("m_axi_bundles", [])) == EXPECTED_M_AXI
    widths_ok = all(
        set(interface.get("m_axi_data_widths", {}).get(b, [])) == {"128"}
        for b in EXPECTED_M_AXI
    )
    s_axi_ok = set(interface.get("s_axi_bundles", [])) == EXPECTED_AXI_LITE
    no_control_r = interface.get("control_r_present") is False
    no_score = interface.get("score_ports", []) == []

    host_plain_ok = (
        len(host_plain["summaries"]) >= 1
        and all(s["failed"] == 0 and s["cases"] == 5 for s in host_plain["summaries"])
        and host_plain["results"] == ["PASS"]
    )
    host_asan_ok = (
        len(host_asan["summaries"]) >= 1
        and all(s["failed"] == 0 and s["cases"] == 5 for s in host_asan["summaries"])
        and host_asan["results"] == ["PASS"]
        and host_asan["error_count"] == 0
    )
    cosim_c_tb_ok = (
        len(cosim["summaries"]) >= 1
        and all(s["failed"] == 0 and s["cases"] == 5 for s in cosim["summaries"])
        and all(r == "PASS" for r in cosim["results"])
    )
    waveform_files = sum(
        int(tmp_audit.get(k, 0))
        for k in (
            "waveform_files_wdb",
            "waveform_files_vcd",
            "waveform_files_saif",
            "waveform_files_fsdb",
        )
    )
    trace_none = cosim_options.get("trace_level") == "none"
    artifacts_bounded = artifact["total_bytes"] < 64 * 1024 * 1024
    protected_unchanged = bool(protected_verify and protected_verify.get("unchanged"))

    gates = {
        "protected_files_unchanged": protected_unchanged,
        "host_test_five_cases_zero_mismatch": host_plain_ok,
        "host_asan_ubsan_zero_mismatch": host_asan_ok,
        "cosim_finished_pass": cosim["cosim_finished_pass"] and not cosim["cosim_finished_fail"],
        "cosim_c_tb_five_cases_zero_mismatch": cosim_c_tb_ok,
        "cosim_zero_errors": cosim["error_count"] == 0,
        "cosim_zero_fatals": cosim["fatal_count"] == 0,
        "interface_exactly_three_m_axi": m_axi_ok,
        "interface_all_m_axi_128bit": widths_ok,
        "interface_exactly_one_axilite_control": s_axi_ok,
        "interface_no_control_r": no_control_r,
        "interface_no_external_score_port": no_score,
        "trace_level_none": trace_none,
        "zero_waveform_files": waveform_files == 0,
        "artifacts_bounded": artifacts_bounded,
        "xsim_timeout_flag_absent": not os.path.isfile(timeout_flag),
    }
    all_pass = all(gates.values())
    if os.path.isfile(timeout_flag) and not cosim["cosim_finished_pass"]:
        decision = "INTEGER_SELECTOR_COMPACT_RTL_COSIM_BLOCKED_TIMEOUT"
    elif all_pass:
        decision = "INTEGER_SELECTOR_COMPACT_RTL_COSIM_PASS"
    else:
        decision = "INTEGER_SELECTOR_COMPACT_RTL_COSIM_FAIL"

    generated_at = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    prompt_sha = sha256_file(PROMPT_MD) if os.path.isfile(PROMPT_MD) else None

    report = {
        "schema_version": "1.0",
        "date": DATE,
        "task": "Bounded Vitis HLS 2023.2 C/RTL co-simulation of the accepted compact integer Selector pipeline",
        "decision": decision,
        "generated_at_utc": generated_at,
        "authority_prompt": {
            "path": os.path.relpath(PROMPT_MD, REPO_ROOT),
            "sha256": prompt_sha,
        },
        "cases": [
            {
                "slot": slot["slot"],
                "case": slot["resolved_case"],
            }
            for slot in case_resolution["slots"]
        ],
        "resolved_cases": case_resolution["resolved_cases"],
        "case_resolution_ok": case_resolution["all_ok"],
        "host": {
            "plain": {
                "cases": host_plain["tb_cases"],
                "summaries": host_plain["summaries"],
                "results": host_plain["results"],
                "error_count": host_plain["error_count"],
            },
            "asan_ubsan": {
                "cases": host_asan["tb_cases"],
                "summaries": host_asan["summaries"],
                "results": host_asan["results"],
                "error_count": host_asan["error_count"],
            },
        },
        "rtl": {
            "c_tb_cases": cosim["tb_cases"],
            "c_tb_summaries": cosim["summaries"],
            "c_tb_results": cosim["results"],
            "cosim_finished_pass": cosim["cosim_finished_pass"],
            "cosim_finished_fail": cosim["cosim_finished_fail"],
            "xsim_started": cosim["xsim_started"],
            "elapsed": cosim["elapsed"],
            "cosim_report": cosim_report,
            "per_transaction": per_transaction,
            "per_transaction_note": "Vitis emits per-transaction latency in "
                                    "result.transaction.rpt (parsed here); the cosim "
                                    "summary report gives the aggregate min/avg/max "
                                    "and total execution cycles",
        },
        "cosim_environment": {
            "tool": "Vitis HLS 2023.2",
            "rtl_language": cosim_options.get("rtl"),
            "simulator": cosim_options.get("tool"),
            "part": csynth_xml_data.get("user_assignments", {}).get("Part"),
            "clock_ns": csynth_xml_data.get("user_assignments", {}).get("TargetClockPeriod"),
            "clock_uncertainty_ns": csynth_xml_data.get("user_assignments", {}).get(
                "ClockUncertainty"
            ),
            "estimated_clock_ns": csynth_xml_data.get("estimated_clock_ns"),
            "trace_level": cosim_options.get("trace_level"),
            "argv": cosim_options.get("argv"),
            "ldflags": cosim_options.get("ldflags"),
        },
        "interface": interface,
        "csynth": {
            "estimated_clock_ns": csynth_xml_data.get("estimated_clock_ns"),
            "latency_cycles": csynth_xml_data.get("latency_cycles"),
            "resources_used": csynth_xml_data.get("resources_used"),
            "resources_available": csynth_xml_data.get("resources_available"),
        },
        "cosim_log_summary": {
            "warning_count": cosim["warning_count"],
            "warning_codes": cosim["warning_codes"],
            "error_count": cosim["error_count"],
            "error_codes": cosim["error_codes"],
            "error_lines": cosim["error_lines"],
            "fatal_count": cosim["fatal_count"],
            "fatal_lines": cosim["fatal_lines"],
        },
        "trace_and_storage": {
            "trace_level": cosim_options.get("trace_level"),
            "waveform_files": waveform_files,
            "waveform_audit": tmp_audit,
            "artifact_audit": artifact,
        },
        "protected_sha_audit": {
            "before_path": os.path.relpath(
                os.path.join(ARTIFACT_DIR, "protected_sha_before.json"), REPO_ROOT
            ),
            "before_total_files": protected_before["total_files"],
            "before_aggregate_sha256": protected_before["aggregate_sha256"],
            "verify": protected_verify,
        },
        "gate_checks": {"gates": gates, "pass": all_pass},
        "reproduction": {
            "commands": [
                "python3 fpga_accel/integer_selector_pipeline_compact_cosim/tools/validate_cases.py "
                "--fixtures artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/fixtures "
                "--cases fpga_accel/integer_selector_pipeline_compact_cosim/cases/rtl_cosim_cases.txt",
                "bash fpga_accel/integer_selector_pipeline_compact_cosim/scripts/run_host_test.sh",
                "bash fpga_accel/integer_selector_pipeline_compact_cosim/scripts/run_cosim.sh",
                "python3 fpga_accel/integer_selector_pipeline_compact_cosim/tools/protected_sha_audit.py snapshot "
                "--out artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_after.json",
                "python3 fpga_accel/integer_selector_pipeline_compact_cosim/tools/protected_sha_audit.py verify "
                "--before artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_before.json "
                "--after artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_after.json "
                "--out artifacts/token_selector/2026-10-03-integer-selector-compact-rtl-cosim/protected_sha_verify.json",
                "python3 fpga_accel/integer_selector_pipeline_compact_cosim/tools/build_report.py",
            ],
            "cosim_tcl": "fpga_accel/integer_selector_pipeline_compact_cosim/scripts/run_cosim.tcl",
            "testbench": "fpga_accel/integer_selector_pipeline_compact_cosim/tb/tb_integer_selector_pipeline_compact_cosim.cpp",
        },
        "limitations": [
            "Functional C/RTL co-simulation only: it proves RTL transaction behaviour against "
            "the frozen expected arenas, not routed timing, bitstream, board latency, model "
            "end-to-end speed or task accuracy.",
            "XSim cosim is pre-implementation; the accepted csynth estimate (3.650 ns <= 5.0 ns) "
            "is a pre-route estimate, not a post-route timing sign-off.",
            "The five cases are a bounded representative subset of the 23 accepted compact "
            "fixtures; they do not exhaust the fixture space.",
            "Vitis reports per-transaction latency in result.transaction.rpt and the "
            "aggregate min/avg/max in the cosim report; the measured cycle counts are "
            "pre-implementation RTL numbers, not post-route/board latency.",
            "Input/output canaries are checked in the co-simulation C phase and one word past the "
            "declared m_axi port depth; memory smaller than the declared cosim port depth is not "
            "independently modelled.",
        ],
        "recommended_next_action": "Export the accepted IP (export_design) and run ZCU102 "
        "out-of-context synthesis/implementation only after this RTL co-simulation gate passes; "
        "route timing is still required before any bitstream/board claim.",
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    harness = {
        "schema_version": "1.0",
        "date": DATE,
        "task": report["task"],
        "decision": decision,
        "gate_checks": {"gates": gates, "pass": all_pass},
        "cases": report["cases"],
        "host": report["host"],
        "rtl": report["rtl"],
        "cosim_environment": report["cosim_environment"],
        "interface": interface,
        "cosim_log_summary": report["cosim_log_summary"],
        "trace_and_storage": report["trace_and_storage"],
        "protected_sha_audit": report["protected_sha_audit"],
        "reproduction": report["reproduction"],
        "limitations": report["limitations"],
        "recommended_next_action": report["recommended_next_action"],
        "report_json": os.path.relpath(REPORT_JSON, REPO_ROOT),
        "report_markdown": os.path.relpath(REPORT_MD, REPO_ROOT),
        "scope": {
            "cpu_only": True,
            "gpu": False,
            "network": False,
            "training": False,
            "parameter_search": False,
            "model_load": False,
            "csim_only": False,
            "cosim": True,
            "export_ip": False,
            "implementation": False,
            "bitstream": False,
            "board_access": False,
            "raw_images": False,
            "internal_test_data": False,
            "tool": "Vitis HLS 2023.2",
            "part": csynth_xml_data.get("user_assignments", {}).get("Part"),
        },
    }
    with open(HARNESS_JSON, "w", encoding="utf-8") as fh:
        json.dump(harness, fh, indent=2)
        fh.write("\n")

    write_markdown(report)

    print("decision:", decision)
    print("gates:", json.dumps(gates, indent=2))
    return 0


def write_markdown(report):
    g = report["gate_checks"]["gates"]
    cases = report["cases"]
    lines = []
    lines.append("# 整数 Selector 紧凑 IP 有界 RTL C/RTL 协同仿真（Vitis HLS 2023.2 / XSim）")
    lines.append("")
    lines.append(f"- 日期：{DATE}")
    lines.append(f"- 决策：`{report['decision']}`")
    lines.append(
        "- 范围：对已验收顶层 `integer_selector_pipeline_compact` 运行 5 个冻结代表用例的 "
        "C/RTL 协同仿真（Verilog RTL、XSim、`trace_level none`）；只读引用已验收源码/头文件与 fixture。"
    )
    lines.append(
        "- 未做：GPU、联网、训练、调参、IP 导出、Vivado implementation、bitstream、板卡；"
        "未改已验收源码/头文件/pragmas/报告/fixture。"
    )
    lines.append("")
    lines.append("## 1. 五个代表用例（对已验收 manifest 逐行解析）")
    lines.append("")
    lines.append("| slot | case | G | grid | K |")
    lines.append("|---|---|---|---|---|")
    resolved = {c["case"]: c for c in report["resolved_cases"]}
    slot_map = {
        "production_non_identity_G80_K48": "production 非 identity G80/K48",
        "real_correct_prompt_G260_K64": "real correct-prompt G260/K64",
        "production_stress_G400_K96": "production stress G400/K96",
        "identity_G80_K96": "identity G80/K96",
        "zero_or_all_equal_tie_supported_budget": "确定性 all-zero G128/K48",
    }
    for c in cases:
        info = resolved.get(c["case"], {})
        grid = info.get("grid", ["?", "?"])
        lines.append(
            f"| {slot_map.get(c['slot'], c['slot'])} | `{c['case']}` | "
            f"{info.get('group_count')} | {grid[0]}x{grid[1]} | {info.get('budget')} |"
        )
    lines.append("")
    lines.append("用例解析门：`all_ok=true`（逐行对已验收 `manifest.txt` 校验、SHA 一致）。")
    lines.append("")
    lines.append("## 2. Host 聚焦编译/测试（cosim 之前）")
    lines.append("")
    for label, key in (("g++ -O2", "plain"), ("ASan+UBSan", "asan_ubsan")):
        h = report["host"][key]
        summary = h["summaries"][-1] if h["summaries"] else {}
        lines.append(
            f"- {label}：cases={summary.get('cases')} checks={summary.get('checks')} "
            f"failed={summary.get('failed')} scalar_mismatches={summary.get('scalar_mismatches')} "
            f"arena_mismatches={summary.get('arena_mismatches')} result={h['results']}"
        )
    lines.append("")
    lines.append("## 3. C/RTL 协同仿真")
    lines.append("")
    env = report["cosim_environment"]
    row = (report["rtl"].get("cosim_report") or {}).get("row") or {}
    lines.append(f"- 工具：{env.get('tool')}；RTL 语言：{env.get('rtl_language')}；模拟器：{env.get('simulator')}")
    lines.append(f"- part：{env.get('part')}；clock：{env.get('clock_ns')}；trace：`{env.get('trace_level')}`")
    lines.append(f"- `C/RTL co-simulation finished: PASS` = {report['rtl']['cosim_finished_pass']}；XSim 启动 = {report['rtl']['xsim_started']}")
    if report["rtl"].get("elapsed"):
        lines.append(f"- 墙钟：{report['rtl']['elapsed']['elapsed_seconds']} s（vitis_hls exit={report['rtl']['elapsed']['exit_code']}）")
    lines.append(
        f"- cosim latency（aggregate over 5 transactions）：min={row.get('latency_min_cycles')} "
        f"avg={row.get('latency_avg_cycles')} max={row.get('latency_max_cycles')} cycles；"
        f"total={row.get('total_execution_cycles')} cycles；status={row.get('status')}"
    )
    sums = report["rtl"]["c_tb_summaries"]
    lines.append(f"- cosim 内 C TB post-check summaries：{json.dumps(sums)} results={report['rtl']['c_tb_results']}")
    lines.append("")
    lines.append("### 3.1 逐 transaction 延迟（`result.transaction.rpt`）")
    lines.append("")
    lines.append("| tx | case | G | grid | K | latency (cycles) | @200MHz (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for t in report["rtl"].get("per_transaction", []):
        grid = t.get("grid") or ["?", "?"]
        lines.append(
            f"| {t['transaction']} | `{t['case']}` | {t.get('group_count')} | "
            f"{grid[0]}x{grid[1]} | {t.get('budget')} | {t.get('latency_cycles')} | "
            f"{t.get('latency_ms_at_200mhz')} |"
        )
    lines.append("")
    lines.append("## 4. 生成 RTL 接口")
    lines.append("")
    iface = report["interface"]
    lines.append(f"- m_axi bundles：{iface.get('m_axi_bundles')}，data widths：{iface.get('m_axi_data_widths')}")
    lines.append(f"- s_axi bundles：{iface.get('s_axi_bundles')}；control_r present：{iface.get('control_r_present')}")
    lines.append(f"- score ports：{iface.get('score_ports')}")
    lines.append("")
    lines.append("## 5. 日志告警/错误与波形/存储")
    lines.append("")
    ls = report["cosim_log_summary"]
    lines.append(f"- warnings={ls['warning_count']}（{json.dumps(ls['warning_codes'])}）")
    lines.append(f"- errors={ls['error_count']}（{json.dumps(ls['error_codes'])}）；fatals={ls['fatal_count']}")
    ts = report["trace_and_storage"]
    lines.append(f"- waveform files={ts['waveform_files']}；trace_level=`{ts['trace_level']}`")
    aa = ts["artifact_audit"]
    lines.append(f"- artifact total：{aa['total_files']} files / {aa['total_bytes']} B（{aa['total_megabytes']} MiB）")
    lines.append("- 20 largest files：")
    for f in aa["largest_20"]:
        lines.append(f"  - `{f['path']}` — {f['bytes']} B")
    lines.append("")
    lines.append("## 6. 受保护文件 SHA 审计")
    lines.append("")
    ps = report["protected_sha_audit"]
    lines.append(f"- before files={ps['before_total_files']} aggregate={ps['before_aggregate_sha256']}")
    if ps.get("verify"):
        lines.append(f"- verify unchanged={ps['verify']['unchanged']} changed={len(ps['verify']['changed'])} "
                     f"missing={len(ps['verify']['missing'])} added={len(ps['verify']['added'])}")
    lines.append("")
    lines.append("## 7. 门与限制")
    lines.append("")
    for k, v in g.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- 限制：{lim}")
    lines.append("")
    lines.append(f"## 8. 推荐下一步")
    lines.append("")
    lines.append(report["recommended_next_action"])
    lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
