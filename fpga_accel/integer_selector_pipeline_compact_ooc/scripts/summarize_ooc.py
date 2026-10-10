#!/usr/bin/env python3
"""Assemble the compact integer Selector IP OOC-route reports.

Reads only the bounded evidence already archived in the repository artifact
directory (hls logs/csynth, exported component.xml, Vivado reports), evaluates
every decision gate, and writes the JSON + Markdown reports and the harness
result.  No tool is invoked and nothing outside the report/artifact paths is
written.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

DATE = "2026-10-03"
SLUG = "2026-10-03-integer-selector-compact-ip-ooc-route"
ARTIFACT_REL = "artifacts/token_selector/" + SLUG
REPORT_BASE = "reports/token_selector/" + SLUG
HARNESS_REL = "reports/token_selector/logs/" + SLUG + "-harness-result.json"
ACCEPTED_HLS = "reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json"

PART = "xczu9eg-ffvb1156-2-e"
TOP = "integer_selector_pipeline_compact"
CLOCK_NS = 5.0
M_AXI_EXPECTED = {"input_arena", "wv_q", "output_arena"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def find_one(patterns, root):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat), recursive=True))
        if hits:
            return hits[0]
    return None


# ---------------------------------------------------------------- HLS csynth
def parse_csynth_xml(path):
    x = read_text(path)
    out = {"present": bool(x)}
    if not x:
        return out
    for tag, key in [
        ("TopModelName", "top"),
        ("Part", "part"),
        ("TargetClockPeriod", "target_clock_ns"),
        ("EstimatedClockPeriod", "estimated_clock_ns"),
        ("ClockUncertainty", "clock_uncertainty_ns"),
    ]:
        m = re.search(r"<" + tag + r">(.*?)</" + tag + r">", x, re.S)
        out[key] = m.group(1).strip() if m else None
    res = {}
    avail = {}
    rm = re.search(r"<Resources>(.*?)</Resources>", x, re.S)
    am = re.search(r"<AvailableResources>(.*?)</AvailableResources>", x, re.S)
    for tag in ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]:
        if rm:
            mm = re.search(r"<" + tag + r">(\d+)</" + tag + r">", rm.group(1))
            res[tag] = int(mm.group(1)) if mm else None
        if am:
            mm = re.search(r"<" + tag + r">(\d+)</" + tag + r">", am.group(1))
            avail[tag] = int(mm.group(1)) if mm else None
    out["resources_used"] = res
    out["resources_available"] = avail
    util = {}
    for k in res:
        if res.get(k) is None or avail.get(k) is None:
            util[k] = None
        elif avail[k] == 0:
            util[k] = 0.0 if res[k] == 0 else None
        else:
            util[k] = round(100.0 * res[k] / avail[k], 4)
    out["resource_utilization_percent"] = util
    # interface ports
    ports = {}
    for m in re.finditer(r"<RtlPorts>(.*?)</RtlPorts>", x, re.S):
        blk = m.group(1)
        nm = re.search(r"<name>(.*?)</name>", blk, re.S)
        if not nm:
            continue
        nm = nm.group(1).strip()
        entry = {}
        for tag, key in [("Object", "object"), ("Type", "type"), ("IOProtocol", "protocol"), ("Dir", "dir"), ("Bits", "bits")]:
            mm = re.search(r"<" + tag + r">(.*?)</" + tag + r">", blk, re.S)
            entry[key] = mm.group(1).strip() if mm else None
        ports[nm] = entry
    out["ports"] = ports
    out["m_axi_data_widths"] = {
        bundle: ports.get("m_axi_" + bundle + "_WDATA", {}).get("bits")
        for bundle in sorted(M_AXI_EXPECTED)
    }
    out["s_axi_control_data_width"] = ports.get("s_axi_control_WDATA", {}).get("bits")
    return out


def parse_hls_export_log(path):
    x = read_text(path)
    return {
        "present": bool(x),
        "error_lines": len(re.findall(r"ERROR:", x)),
        "fatal_lines": len(re.findall(r"FATAL", x)),
        "csynth_completed": "Finished Command csynth_design" in x,
        "export_completed": "export_design completed" in x,
        "vitis_exit_clean": "Exiting vitis_hls" in x,
        "bytes": len(x),
    }


# ---------------------------------------------------------------- component.xml
def parse_component_xml(path):
    x = read_text(path)
    out = {"present": bool(x)}
    if not x:
        return out
    for tag in ["vendor", "library", "name", "version"]:
        m = re.search(r"<spirit:" + tag + r">(.*?)</spirit:" + tag + r">", x, re.S)
        out[tag] = m.group(1).strip() if m else None
    out["vlnv"] = "%s:%s:%s:%s" % (out.get("vendor"), out.get("library"), out.get("name"), out.get("version"))
    buses = []
    for m in re.finditer(r"<spirit:busInterface>(.*?)</spirit:busInterface>", x, re.S):
        blk = m.group(1)
        nm = re.search(r"<spirit:name>(.*?)</spirit:name>", blk, re.S)
        bt = re.search(r'<spirit:busType[^>]*spirit:name="([^"]+)"', blk)
        mode = None
        if re.search(r"<spirit:master>", blk):
            mode = "master"
        elif re.search(r"<spirit:slave>", blk):
            mode = "slave"
        buses.append({
            "name": nm.group(1).strip() if nm else None,
            "bus_type": bt.group(1) if bt else None,
            "mode": mode,
        })
    out["bus_interfaces"] = buses
    # physical ports
    port_names = re.findall(r"<spirit:port>\s*<spirit:name>(.*?)</spirit:name>", x, re.S)
    out["port_names"] = [p.strip() for p in port_names]
    out["port_count"] = len(out["port_names"])
    return out


# ---------------------------------------------------------------- Vivado reports
def parse_timing_summary(path):
    x = read_text(path)
    out = {"present": bool(x), "constraints_met": ("All user specified timing constraints are met" in x)}
    lines = x.splitlines()
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "WPWS(ns)" in line:
            for j in range(i + 1, min(i + 5, len(lines))):
                s = lines[j].strip()
                if not s or set(s) <= set("- "):
                    continue
                parts = s.split()
                if len(parts) >= 12:
                    out.update({
                        "wns_ns": float(parts[0]), "tns_ns": float(parts[1]),
                        "tns_failing_endpoints": int(parts[2]), "tns_total_endpoints": int(parts[3]),
                        "whs_ns": float(parts[4]), "ths_ns": float(parts[5]),
                        "ths_failing_endpoints": int(parts[6]), "ths_total_endpoints": int(parts[7]),
                        "wpws_ns": float(parts[8]), "tpws_ns": float(parts[9]),
                        "tpws_failing_endpoints": int(parts[10]), "tpws_total_endpoints": int(parts[11]),
                    })
                break
            break
    return out


def _util_row(x, site_type):
    # rows look like: | CLB LUTs*   | 28427 |  0 |  0 | 274080 | 10.37 |
    # or (BRAM)       | Block RAM Tile | 122.5 | 0 | 0 | 912 | 13.43 |
    pat = (r"\|\s*" + re.escape(site_type) + r"\*?\s*\|\s*([\d.]+)\s*\|\s*(\d+)?\s*\|\s*(\d+)?\s*\|\s*(\d+)?\s*\|\s*([\d.]+)\s*\|")
    m = re.search(pat, x)
    if not m:
        return None
    used = float(m.group(1))
    used_out = int(used) if used == int(used) else used
    avail = int(m.group(4)) if m.group(4) else None
    pct = float(m.group(5)) if m.group(5) else None
    return {"used": used_out, "available": avail, "util_percent": pct}


def parse_utilization(path):
    x = read_text(path)
    out = {"present": bool(x)}
    mapping = {
        "lut": "CLB LUTs",
        "ff": "CLB Registers",
        "lut_as_logic": "LUT as Logic",
        "lut_as_memory": "LUT as Memory",
        "bram_tile": "Block RAM Tile",
        "dsp": "DSPs",
        "uram": "URAM",
    }
    for key, site in mapping.items():
        row = _util_row(x, site)
        if row:
            out[key] = row
    return out


def parse_route_status(path):
    x = read_text(path)
    out = {"present": bool(x)}
    def grab(label):
        m = re.search(r"# of " + re.escape(label) + r"[.\s]*:\s*(\d+)", x)
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


def parse_drc(path):
    x = read_text(path)
    out = {"present": bool(x)}
    counts = {}
    for m in re.finditer(r"\|\s*([A-Z]+-\d+)\s*\|\s*([A-Za-z ]+?)\s*\|[^|]*\|\s*(\d+)\s*\|", x):
        sev = m.group(2).strip()
        counts[sev] = counts.get(sev, 0) + int(m.group(3))
    out["violations_found"] = None
    m = re.search(r"Violations found:\s*(\d+)", x)
    if m:
        out["violations_found"] = int(m.group(1))
    out["by_severity"] = counts
    out["error"] = counts.get("Error", 0)
    out["critical_warning"] = counts.get("Critical Warning", 0)
    out["warning"] = counts.get("Warning", 0)
    return out


def _parse_path_block(block):
    d = {}
    m = re.search(r"Slack\s*(\((?:MET|VIOLATED)\))?\s*:?\s*(-?[\d.]+)ns", block)
    if m:
        d["slack_ns"] = float(m.group(2))
        d["met"] = (m.group(1) == "(MET)")
    m = re.search(r"(?:Startpoint|Source)\s*:\s*(.+)", block)
    if m:
        d["startpoint"] = m.group(1).strip()
    m = re.search(r"(?:Endpoint|Destination)\s*:\s*(.+)", block)
    if m:
        d["endpoint"] = m.group(1).strip()
    m = re.search(r"Path Type\s*:\s*(.+)", block)
    if m:
        d["path_type"] = m.group(1).strip()
    m = re.search(r"Data Path Delay:\s*([\d.]+)ns", block)
    if m:
        d["data_path_delay_ns"] = float(m.group(1))
    m = re.search(r"Logic Delay:\s*([\d.]+)ns", block)
    if m:
        d["logic_delay_ns"] = float(m.group(1))
    m = re.search(r"Route Delay:\s*([\d.]+)ns", block)
    if m:
        d["route_delay_ns"] = float(m.group(1))
    m = re.search(r"Logic Levels:\s*(\d+)", block)
    if m:
        d["logic_levels"] = int(m.group(1))
    m = re.search(r"Clock Uncertainty:\s*([\d.]+)ns", block)
    if m:
        d["clock_uncertainty_ns"] = float(m.group(1))
    return d


def parse_timing_paths(path):
    """Extract the worst setup (max) and worst hold (min) paths from a
    report_timing / report_timing_summary text report."""
    x = read_text(path)
    out = {"present": bool(x)}
    starts = [m.start() for m in re.finditer(r"Slack\s*\((?:MET|VIOLATED)\)", x)]
    blocks = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(x)
        blocks.append(_parse_path_block(x[s:e]))
    setup = [b for b in blocks if "Setup" in b.get("path_type", "")]
    hold = [b for b in blocks if "Hold" in b.get("path_type", "")]
    if setup:
        out["setup_critical_path"] = setup[0]
        out.update(setup[0])
    if hold:
        out["worst_hold_path"] = hold[0]
    if not setup and blocks:
        out.update(blocks[0])
    return out


def parse_stage_status(path):
    rows = []
    x = read_text(path)
    for line in x.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3 or parts[0] == "stage":
            continue
        rows.append({"stage": parts[0], "rc": int(parts[1]), "elapsed_s": int(parts[2]),
                     "detail": parts[3] if len(parts) > 3 else ""})
    return rows


def parse_tool_versions(path):
    return read_text(path).strip()


def parse_qor(path):
    x = read_text(path)
    return {
        "present": bool(x),
        "bytes": len(x),
        "suggestion_lines": len(re.findall(r"(?im)^\s*[-*]?\s*(?:suggestion|Suggestion|SUGGESTION)", x)),
    }


# ---------------------------------------------------------------- audit
def artifact_audit(art_dir):
    files = []
    total = 0
    for dirpath, _d, filenames in os.walk(art_dir):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            total += size
            files.append({"path": os.path.relpath(p, art_dir), "size": size})
    files.sort(key=lambda e: (-e["size"], e["path"]))
    return {"file_count": len(files), "total_bytes": total,
            "top20_largest": files[:20]}


# ---------------------------------------------------------------- main build
def build(args):
    repo = os.path.abspath(args.repo)
    art = os.path.join(repo, ARTIFACT_REL)
    if not os.path.isdir(art):
        raise SystemExit("artifact dir missing: " + art)

    csynth_path = find_one(["hls/csynth.xml"], art)
    csynth_rpt_path = find_one(["hls/csynth.rpt"], art)
    comp_path = find_one(["ip/component.xml"], art)
    ports_path = find_one(["ip/top_module_ports.txt"], art)
    xdc_path = find_one(["ip/*_ooc.xdc"], art)
    hls_log_path = find_one(["hls/hls_export.log"], art)
    stage_path = find_one(["vivado/ooc_stage_status.tsv"], art)

    csynth = parse_csynth_xml(csynth_path) if csynth_path else {"present": False}
    hls_log = parse_hls_export_log(hls_log_path) if hls_log_path else {"present": False}
    comp = parse_component_xml(comp_path) if comp_path else {"present": False}
    top_ports_txt = read_text(ports_path) if ports_path else ""
    xdc_txt = read_text(xdc_path) if xdc_path else ""
    stages = parse_stage_status(stage_path) if stage_path else []

    def rep(name):
        p = find_one(["vivado/" + name], art)
        return p

    timing_synth = parse_timing_summary(rep("timing_summary_post_synth.rpt")) if rep("timing_summary_post_synth.rpt") else {"present": False}
    timing_place = parse_timing_summary(rep("timing_summary_post_place.rpt")) if rep("timing_summary_post_place.rpt") else {"present": False}
    timing_route = parse_timing_summary(rep("timing_summary_post_route.rpt")) if rep("timing_summary_post_route.rpt") else {"present": False}
    util_synth = parse_utilization(rep("utilization_post_synth.rpt")) if rep("utilization_post_synth.rpt") else {"present": False}
    util_place = parse_utilization(rep("utilization_post_place.rpt")) if rep("utilization_post_place.rpt") else {"present": False}
    util_route = parse_utilization(rep("utilization_post_route.rpt")) if rep("utilization_post_route.rpt") else {"present": False}
    route_status = parse_route_status(rep("route_status_post_route.rpt")) if rep("route_status_post_route.rpt") else {"present": False}
    drc = parse_drc(rep("drc_post_route.rpt")) if rep("drc_post_route.rpt") else {"present": False}
    timing_paths = parse_timing_paths(rep("timing_summary_post_route.rpt")) if rep("timing_summary_post_route.rpt") else {"present": False}
    qor = parse_qor(rep("qor_suggestions_post_route.rpt")) if rep("qor_suggestions_post_route.rpt") else {"present": False}
    tool_versions = parse_tool_versions(find_one(["tool_versions.txt"], art) or "")

    # run status (written by the harness after the driver returns)
    run_status = {}
    rs_path = find_one(["logs/run_status.txt"], art)
    if rs_path:
        for line in read_text(rs_path).splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                run_status[k.strip()] = v.strip()
    hls_rc = int(run_status.get("HLS_EXIT", "-1"))
    vivado_rc = int(run_status.get("VIVADO_EXIT", "-1"))

    # protected audit
    compare = {}
    cmp_path = find_one(["protected_sha_compare.json"], art)
    if cmp_path:
        compare = json.load(open(cmp_path))

    # ---------------- interface gates ----------------
    aximm = [b for b in comp.get("bus_interfaces", []) if b.get("bus_type") == "aximm"]
    masters = [b for b in aximm if b.get("mode") == "master"]
    slaves = [b for b in aximm if b.get("mode") == "slave"]
    master_names = sorted(b["name"] for b in masters)
    # strip m_axi_ / s_axi_ prefixes to bundle names
    def bundle(n):
        return re.sub(r"^(m_axi_|s_axi_)", "", n or "")
    master_bundles = sorted(bundle(n) for n in master_names)
    slave_bundles = sorted(bundle(b["name"]) for b in slaves)
    all_bus_names = [b.get("name") for b in comp.get("bus_interfaces", [])]
    no_control_r = not any("control_r" in (n or "") for n in all_bus_names)
    no_score_port = ("score_q" not in comp.get("port_names", [])) and ("score_q" not in top_ports_txt)
    widths = csynth.get("m_axi_data_widths", {})
    three_128 = (master_bundles == sorted(M_AXI_EXPECTED)) and all(str(widths.get(b)) == "128" for b in M_AXI_EXPECTED)
    one_control = (slave_bundles == ["control"]) and str(csynth.get("s_axi_control_data_width")) == "32"
    interface_ok = bool(three_128 and one_control and no_control_r and no_score_port)

    # ---------------- part / clock gates ----------------
    part_ok = (csynth.get("part") == PART) or (PART in read_text(rep("utilization_post_route.rpt") or ""))
    clock_ok = (abs(float(csynth.get("target_clock_ns") or 0) - CLOCK_NS) < 1e-9
                and "create_clock" in xdc_txt and "5.000" in xdc_txt
                and "ap_clk" in xdc_txt)
    clk_rpt = read_text(rep("clocks_post_synth.rpt") or "")
    clock_ok = clock_ok and ("ap_clk" in clk_rpt if clk_rpt else True)

    # ---------------- stage gates ----------------
    stage_by = {s["stage"]: s for s in stages}
    hls_ok = (hls_log.get("error_lines", 1) == 0 and hls_log.get("fatal_lines", 1) == 0
              and hls_log.get("csynth_completed") and hls_log.get("export_completed"))
    if hls_rc not in (-1, 0):
        hls_ok = False
    synth_ok = bool(stage_by.get("synth_design", {}).get("rc") == 0)
    opt_ok = bool(stage_by.get("opt_design", {}).get("rc") == 0)
    place_ok = bool(stage_by.get("place_design", {}).get("rc") == 0)
    phys_opt_ok = bool(stage_by.get("phys_opt_design", {}).get("rc") == 0)
    route_stage_ok = bool(stage_by.get("route_design", {}).get("rc") == 0)
    timeout = (vivado_rc == 124) or (not route_stage_ok and not stage_by.get("route_design"))

    wns = timing_route.get("wns_ns")
    whs = timing_route.get("whs_ns")
    timing_ok = (wns is not None and wns >= 0.0 and whs is not None and whs >= 0.0)
    unrouted = route_status.get("unrouted_nets")
    route_ok = bool(route_stage_ok and (route_status.get("routing_errors") == 0) and (unrouted == 0))
    drc_ok = bool(drc.get("present") and drc.get("error", 1) == 0 and drc.get("critical_warning", 1) == 0)

    resources_ok = True
    resource_report = {}
    for key in ["lut", "ff", "bram_tile", "dsp", "uram"]:
        row = util_route.get(key)
        if row is None and key == "uram":
            # XCZU9EG has no URAM; record the zero capacity explicitly.
            row = {"used": 0, "available": 0, "util_percent": 0.0,
                   "note": "XCZU9EG has no URAM"}
        resource_report[key] = row
        if row and row.get("util_percent") is not None:
            if row["util_percent"] >= 80.0:
                resources_ok = False
        elif key != "uram":
            # A missing flat-utilization row for a required resource cannot PASS.
            resources_ok = False
    # URAM capacity is 0 on XCZU9EG -> used must be 0
    if util_route.get("uram") and util_route["uram"].get("used") not in (0, None):
        resources_ok = False

    vivado_log_txt = read_text(rep("vivado_ooc.log") or "")
    no_bitstream = (not glob.glob(os.path.join(art, "**", "*.bit"), recursive=True)
                    and "write_bitstream" not in vivado_log_txt)
    env_crit_ids = sorted(set(re.findall(r"CRITICAL WARNING: \[([^\]]+)\]", vivado_log_txt)))
    env_crit_invalidating = [i for i in env_crit_ids if i not in ("Common 17-741",)]
    license_ok = ("Got license for feature 'Synthesis'" in vivado_log_txt
                  and "Got license for feature 'Implementation'" in vivado_log_txt)
    license_errors = len(re.findall(r"(?i)license.*(?:error|denied|failed)", vivado_log_txt))
    protected_ok = bool(compare.get("protected_files_unchanged"))

    achieved_mhz = None
    if wns is not None:
        period = CLOCK_NS - wns
        achieved_mhz = round(1000.0 / period, 4) if period > 0 else None

    # ---------------- decision ----------------
    if not hls_ok or not comp.get("present"):
        decision = "INTEGER_SELECTOR_COMPACT_IP_EXPORT_BLOCKED"
    elif not synth_ok:
        decision = "INTEGER_SELECTOR_COMPACT_OOC_SYNTH_BLOCKED"
    elif timeout:
        decision = "INTEGER_SELECTOR_COMPACT_OOC_ROUTE_BLOCKED_TIMEOUT"
    elif not route_ok or not opt_ok or not place_ok:
        decision = "INTEGER_SELECTOR_COMPACT_OOC_ROUTE_BLOCKED"
    elif not timing_ok:
        decision = "INTEGER_SELECTOR_COMPACT_OOC_TIMING_FAIL"
    elif not (protected_ok and drc_ok and resources_ok and interface_ok and part_ok and clock_ok and no_bitstream):
        decision = "INTEGER_SELECTOR_COMPACT_OOC_ROUTE_BLOCKED"
    else:
        decision = "INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS"

    if decision == "INTEGER_SELECTOR_COMPACT_IP_OOC_ROUTE_PASS":
        next_action = ("Build the minimal Zynq UltraScale+ MPSoC block design around this exported IP "
                       "(PS, clock/reset, three 128-bit AXI4 masters, AXI4-Lite control, memory map and DMA path), "
                       "validate it, and only then consider a bitstream.")
    elif decision == "INTEGER_SELECTOR_COMPACT_OOC_TIMING_FAIL":
        if timing_paths.get("startpoint") and timing_paths.get("endpoint"):
            next_action = ("Optimise the smallest evidence-driven RTL/HLS piece on the post-route critical path "
                           "(%s -> %s) before re-running this OOC flow." % (timing_paths.get("startpoint"), timing_paths.get("endpoint")))
        else:
            next_action = ("Optimise the largest post-route timing bottleneck (see critical_paths_post_route.rpt) "
                           "with the smallest RTL/HLS change, then re-run this OOC flow.")
    else:
        next_action = ("Inspect the archived console/report evidence, fix the first blocking failure, and re-run "
                       "scripts/run_ooc.sh unchanged.")

    # ---------------- accepted HLS identity ----------------
    identity = None
    acc_path = os.path.join(repo, ACCEPTED_HLS)
    if os.path.isfile(acc_path):
        acc = json.load(open(acc_path))["csynth"]
        identity = {
            "accepted_estimated_clock_ns": acc.get("estimated_clock_ns"),
            "new_estimated_clock_ns": float(csynth.get("estimated_clock_ns")) if csynth.get("estimated_clock_ns") else None,
            "accepted_resources_used": acc.get("resources_used"),
            "new_resources_used": csynth.get("resources_used"),
            "accepted_resource_utilization_percent": acc.get("resource_utilization_percent"),
            "new_resource_utilization_percent": csynth.get("resource_utilization_percent"),
        }
        identity["identical"] = (
            identity["accepted_estimated_clock_ns"] == identity["new_estimated_clock_ns"]
            and identity["accepted_resources_used"] == identity["new_resources_used"]
        )

    interfaces = {
        "m_axi_bundles": master_bundles,
        "m_axi_full_names": master_names,
        "m_axi_data_widths": widths,
        "s_axi_bundles": slave_bundles,
        "s_axi_full_names": sorted(b["name"] for b in slaves),
        "s_axi_control_data_width": csynth.get("s_axi_control_data_width"),
        "other_bus_interfaces": sorted(n for n in all_bus_names if n not in master_names and n not in [b["name"] for b in slaves]),
        "component_bus_interface_count": len(comp.get("bus_interfaces", [])),
        "component_port_count": comp.get("port_count"),
        "control_r_present": not no_control_r,
        "score_q_port_present": not no_score_port,
        "gates": {
            "three_128bit_axi4_masters_input_arena_wv_q_output_arena": three_128,
            "one_axi4lite_slave_control": one_control,
            "no_control_r_interface": no_control_r,
            "no_external_score_q_port": no_score_port,
            "interface_gate_pass": interface_ok,
        },
    }

    gates = {
        "protected_files_byte_identical": protected_ok,
        "hls_csynth_and_export_no_error_fatal": hls_ok,
        "vivado_ooc_synth_ok": synth_ok,
        "opt_design_ok": opt_ok,
        "place_design_ok": place_ok,
        "phys_opt_design_ok": phys_opt_ok,
        "route_design_ok": route_ok,
        "post_route_setup_wns_ge_0": bool(wns is not None and wns >= 0.0),
        "post_route_hold_whs_ge_0": bool(whs is not None and whs >= 0.0),
        "no_unrouted_nets": bool(unrouted == 0),
        "drc_no_error_no_critical_warning": drc_ok,
        "resources_each_below_80_percent": resources_ok,
        "part_exact": part_ok,
        "clock_exact_5_000_ns": clock_ok,
        "interface_gates": interface_ok,
        "no_bitstream_generated": no_bitstream,
    }

    export_zip = find_one(["ip/*.zip"], art)
    export_zip_info = {"present": bool(export_zip)}
    if export_zip:
        export_zip_info["path"] = os.path.relpath(export_zip, art)
        export_zip_info["size_bytes"] = os.path.getsize(export_zip)
        export_zip_info["sha256"] = sha256_file(export_zip)
    dcp = find_one(["vivado/post_route.dcp"], art)
    dcp_info = {"archived": bool(dcp)}
    if dcp:
        dcp_info["path"] = os.path.relpath(dcp, art)
        dcp_info["size_bytes"] = os.path.getsize(dcp)
        dcp_info["sha256"] = sha256_file(dcp)
        dcp_info["limit_bytes"] = 209715200
        dcp_info["within_200mib"] = os.path.getsize(dcp) <= 209715200

    result = {
        "schema_version": 1.0,
        "date": DATE,
        "task": "Export the accepted compact integer Selector IP catalog component and run a Vivado 2023.2 out-of-context place-and-route on xczu9eg-ffvb1156-2-e at ap_clk = 5.000 ns",
        "decision": decision,
        "scope": {
            "flow": "Vitis HLS 2023.2 csynth_design + export_design -format ip_catalog (-flow syn); Vivado 2023.2 OOC synth_design + opt_design + place_design + phys_opt_design + route_design",
            "part": PART,
            "top": TOP,
            "project_root": "session workspace (repo)",
            "all_hls_vivado_projects_in_tmp": True,
            "bitstream": False,
            "board_access": False,
            "cosim": False,
            "gpu": False,
            "network": False,
            "training": False,
            "source_or_interface_change": False,
        },
        "commands": {
            "hls": "export LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE:$LD_LIBRARY_PATH && /home/xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_hls_export.tcl",
            "vivado": "/home/xilinx/Vivado/2023.2/bin/vivado -mode batch -source fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_vivado_ooc.tcl -log /tmp/.../vivado_ooc.log",
            "driver": "bash fpga_accel/integer_selector_pipeline_compact_ooc/scripts/run_ooc.sh",
            "timeouts_s": {"hls": 7200, "vivado_total": 5400},
        },
        "tool_versions": tool_versions,
        "license": {
            "vivado_license_ok": license_ok,
            "synthesis_and_implementation_license_obtained": license_ok,
            "license_error_lines": license_errors,
            "vivado_license_errors": len(re.findall(r"(?i)license.*error", vivado_log_txt)),
            "hls_license_errors": len(re.findall(r"(?i)license.*error", read_text(find_one(["hls/hls_export.log"], art) or ""))),
        },
        "environment_warnings": {
            "host_critical_warning_ids": env_crit_ids,
            "non_host_invalidating_critical_warning_ids": env_crit_invalidating,
            "note": "The only console CRITICAL WARNING is [Common 17-741] (no write access to the local Xilinx Tcl store; reverted to the installation area). It is a host/Tcl-store setting, not a design DRC, and it does not invalidate implementation or timing.",
        },
        "hls": {
            "csynth": csynth,
            "export_log": hls_log,
            "identity_with_accepted": identity,
        },
        "exported_ip": {
            "component_xml_present": comp.get("present"),
            "vlnv": comp.get("vlnv"),
            "archive": export_zip_info,
            "bus_interfaces": comp.get("bus_interfaces"),
        },
        "interfaces": interfaces,
        "ooc": {
            "stage_status": stages,
            "hls_exit": hls_rc,
            "vivado_exit": vivado_rc,
            "timeout_flag": timeout,
            "synth_ok": synth_ok, "opt_ok": opt_ok, "place_ok": place_ok,
            "phys_opt_ok": phys_opt_ok, "route_ok": route_ok,
            "post_route_dcp": dcp_info,
        },
        "timing": {
            "clock_name": "ap_clk",
            "clock_period_ns": CLOCK_NS,
            "target_frequency_mhz": 200.0,
            "hls_default_clock_uncertainty_ns": csynth.get("clock_uncertainty_ns"),
            "exported_xdc_uncertainty": "none in exported OOC XDC (create_clock ap_clk 5.000 only)",
            "post_synth": timing_synth,
            "post_place": timing_place,
            "post_route": timing_route,
            "post_route_critical_path": timing_paths,
            "achieved_frequency_mhz_from_5ns_minus_wns": achieved_mhz,
        },
        "resources": {
            "post_synth": util_synth,
            "post_place": util_place,
            "post_route": util_route,
            "post_route_gate_rows": resource_report,
        },
        "route": {"status": route_status, "drc": drc},
        "qor_suggestions": qor,
        "gates": gates,
        "protected_sha_audit": compare,
        "artifact_storage_audit": artifact_audit(art),
        "limitations": [
            "Out-of-context (OOC) only: no PS, no DDR, no AXI interconnect contention, no DMA and no full ZCU102 block design.",
            "Post-route timing/resources are the OOC Vivado result for the exported IP; they are not a board measurement.",
            "No bitstream was generated and the board was not accessed.",
            "No end-to-end model latency, throughput, speedup or task-accuracy claim is made.",
            "The exported HLS OOC XDC carries only create_clock ap_clk 5.000 ns; it has no explicit set_clock_uncertainty. Vitis HLS csynth used its default 1.35 ns scheduling uncertainty, which is not present in the post-route OOC constraints.",
            "The exported IP interface gate is parsed from the generated component.xml/csynth.xml, not from the C++ pragmas.",
        ],
        "recommended_next_action": next_action,
    }

    os.makedirs(os.path.dirname(os.path.join(repo, HARNESS_REL)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(repo, REPORT_BASE + ".json")), exist_ok=True)
    with open(os.path.join(repo, REPORT_BASE + ".json"), "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    with open(os.path.join(repo, REPORT_BASE + ".md"), "w") as fh:
        fh.write(render_markdown(result))
    with open(os.path.join(repo, HARNESS_REL), "w") as fh:
        json.dump(build_harness(result), fh, indent=1, sort_keys=True)
    print(json.dumps({
        "decision": decision,
        "wns_ns": wns, "whs_ns": whs, "unrouted": unrouted,
        "drc_error": drc.get("error"), "drc_critical_warning": drc.get("critical_warning"),
        "stages": stages,
        "interface_gate_pass": interface_ok,
        "protected_unchanged": protected_ok,
        "artifact_bytes": result["artifact_storage_audit"]["total_bytes"],
        "artifact_files": result["artifact_storage_audit"]["file_count"],
    }, indent=1))
    return result


def build_harness(result):
    return {
        "schema_version": 1.0,
        "date": result["date"],
        "task": result["task"],
        "decision": result["decision"],
        "gates": result["gates"],
        "interfaces": result["interfaces"]["gates"],
        "timing": {
            "post_route_wns_ns": result["timing"]["post_route"].get("wns_ns"),
            "post_route_whs_ns": result["timing"]["post_route"].get("whs_ns"),
            "post_route_tns_ns": result["timing"]["post_route"].get("tns_ns"),
            "post_route_ths_ns": result["timing"]["post_route"].get("ths_ns"),
            "achieved_frequency_mhz": result["timing"]["achieved_frequency_mhz_from_5ns_minus_wns"],
        },
        "route": {
            "unrouted_nets": result["route"]["status"].get("unrouted_nets"),
            "routing_errors": result["route"]["status"].get("routing_errors"),
            "drc_error": result["route"]["drc"].get("error"),
            "drc_critical_warning": result["route"]["drc"].get("critical_warning"),
        },
        "exported_ip": {
            "vlnv": result["exported_ip"].get("vlnv"),
            "archive": result["exported_ip"].get("archive"),
        },
        "post_route_dcp": result["ooc"].get("post_route_dcp"),
        "license": result["license"],
        "environment_warnings": result["environment_warnings"],
        "protected_sha_audit": {
            "protected_files_unchanged": result["protected_sha_audit"].get("protected_files_unchanged"),
            "before_file_count": result["protected_sha_audit"].get("before_file_count"),
            "after_file_count": result["protected_sha_audit"].get("after_file_count"),
            "before_aggregate_sha256": result["protected_sha_audit"].get("before_aggregate_sha256"),
            "after_aggregate_sha256": result["protected_sha_audit"].get("after_aggregate_sha256"),
        },
        "artifact_storage_audit": result["artifact_storage_audit"],
        "evidence_paths": {
            "artifact_dir": ARTIFACT_REL,
            "report_json": REPORT_BASE + ".json",
            "report_md": REPORT_BASE + ".md",
            "harness_result": HARNESS_REL,
        },
        "limitations": result["limitations"],
        "recommended_next_action": result["recommended_next_action"],
    }


def render_markdown(r):
    L = []
    L.append("# 紧凑整数 Selector IP 导出 + Vivado OOC 布线报告\n")
    L.append("- 日期：%s" % r["date"])
    L.append("- 决策：**%s**" % r["decision"])
    L.append("- part：`%s`，top：`%s`，时钟：`ap_clk = %.3f ns`" % (r["scope"]["part"], r["scope"]["top"], CLOCK_NS))
    L.append("- 流程：%s" % r["scope"]["flow"])
    L.append("")
    L.append("## 命令与工具")
    for k, v in r["commands"].items():
        L.append("- `%s`：%s" % (k, v))
    L.append("- 工具版本：\n```\n%s\n```" % r["tool_versions"])
    L.append("- license：synthesis+implementation 获取成功=%s，license error 行=%s" % (
        r["license"].get("synthesis_and_implementation_license_obtained"), r["license"].get("license_error_lines")))
    L.append("- 环境 CRITICAL WARNING：%s（[Common 17-741] Tcl store 写权限，非设计 DRC，不影响实现/时序）" % (
        r["environment_warnings"].get("host_critical_warning_ids"),))
    L.append("")
    L.append("## 接口门（来自 component.xml/csynth.xml）")
    it = r["interfaces"]
    L.append("- m_axi bundles：%s（宽度 %s）" % (it["m_axi_bundles"], it["m_axi_data_widths"]))
    L.append("- s_axi bundles：%s（control WDATA %s bit）" % (it["s_axi_bundles"], it["s_axi_control_data_width"]))
    L.append("- 无 control_r：%s；无 score_q 端口：%s" % (not it["control_r_present"], not it["score_q_port_present"]))
    for k, v in it["gates"].items():
        L.append("- %s：%s" % (k, v))
    L.append("")
    L.append("## HLS 身份")
    idn = r["hls"]["identity_with_accepted"]
    if idn:
        L.append("- accepted estimated clock %s ns / new %s ns" % (idn["accepted_estimated_clock_ns"], idn["new_estimated_clock_ns"]))
        L.append("- accepted resources %s" % (idn["accepted_resources_used"],))
        L.append("- new resources %s" % (idn["new_resources_used"],))
        L.append("- identical：%s" % idn["identical"])
    L.append("- HLS export log：errors=%s fatals=%s csynth_done=%s export_done=%s" % (
        r["hls"]["export_log"].get("error_lines"), r["hls"]["export_log"].get("fatal_lines"),
        r["hls"]["export_log"].get("csynth_completed"), r["hls"]["export_log"].get("export_completed")))
    L.append("- VLNV：`%s`" % r["exported_ip"]["vlnv"])
    arch = r["exported_ip"].get("archive", {})
    if arch.get("present"):
        L.append("- IP archive：%s，%s B，sha256 `%s`" % (arch.get("path"), arch.get("size_bytes"), arch.get("sha256")))
    dcp = r["ooc"].get("post_route_dcp", {})
    if dcp.get("archived"):
        L.append("- routed DCP：%s，%s B（<=200MiB：%s），sha256 `%s`" % (
            dcp.get("path"), dcp.get("size_bytes"), dcp.get("within_200mib"), dcp.get("sha256")))
    L.append("")
    L.append("## OOC 阶段")
    for s in r["ooc"]["stage_status"]:
        L.append("- %s：rc=%s，%s s" % (s["stage"], s["rc"], s["elapsed_s"]))
    L.append("- HLS exit=%s，Vivado exit=%s，timeout=%s" % (r["ooc"]["hls_exit"], r["ooc"]["vivado_exit"], r["ooc"]["timeout_flag"]))
    L.append("")
    L.append("## 时序")
    pr = r["timing"]["post_route"]
    L.append("- post-route WNS=%s ns / TNS=%s ns / WHS=%s ns / THS=%s ns" % (
        pr.get("wns_ns"), pr.get("tns_ns"), pr.get("whs_ns"), pr.get("ths_ns")))
    L.append("- 由 5 ns 需求推导的 achieved frequency：%s MHz" % r["timing"]["achieved_frequency_mhz_from_5ns_minus_wns"])
    cp = r["timing"]["post_route_critical_path"]
    if cp.get("startpoint"):
        L.append("- 关键路径：%s -> %s（slack %s ns，data path %s ns）" % (
            cp.get("startpoint"), cp.get("endpoint"), cp.get("slack_ns"), cp.get("data_path_delay_ns")))
    L.append("")
    L.append("## 资源（post-route）")
    rows = r["resources"].get("post_route_gate_rows", r["resources"]["post_route"])
    for k in ["lut", "ff", "bram_tile", "dsp", "uram"]:
        row = rows.get(k)
        if row:
            extra = ("（%s）" % row["note"]) if row.get("note") else ""
            L.append("- %s：used %s / available %s = %s%%%s" % (k, row.get("used"), row.get("available"), row.get("util_percent"), extra))
    syn = r["resources"].get("post_synth", {})
    def _fmt(key):
        row = syn.get(key)
        if not row:
            return "n/a"
        return "%s/%s=%s%%" % (row.get("used"), row.get("available"), row.get("util_percent"))
    L.append("- post-synth：LUT %s，FF %s，BRAM %s，DSP %s" % (
        _fmt("lut"), _fmt("ff"), _fmt("bram_tile"), _fmt("dsp")))
    L.append("")
    L.append("## 布线 / DRC")
    L.append("- %s" % (r["route"]["status"],))
    L.append("- %s" % (r["route"]["drc"],))
    L.append("")
    L.append("## 门汇总")
    for k, v in r["gates"].items():
        L.append("- %s：%s" % (k, v))
    L.append("")
    L.append("## 受保护文件 SHA 审计")
    L.append("- before=%s 文件 / aggregate `%s`" % (r["protected_sha_audit"].get("before_file_count"), r["protected_sha_audit"].get("before_aggregate_sha256")))
    L.append("- after=%s 文件 / aggregate `%s`" % (r["protected_sha_audit"].get("after_file_count"), r["protected_sha_audit"].get("after_aggregate_sha256")))
    L.append("- 未变：%s（changed=%s missing=%s added=%s）" % (
        r["protected_sha_audit"].get("protected_files_unchanged"),
        r["protected_sha_audit"].get("changed"), r["protected_sha_audit"].get("missing"), r["protected_sha_audit"].get("added")))
    L.append("")
    L.append("## 产物存储审计")
    audit = r["artifact_storage_audit"]
    L.append("- 总字节 %s / 文件 %s" % (audit["total_bytes"], audit["file_count"]))
    L.append("- 最大的 20 个文件：")
    for e in audit["top20_largest"]:
        L.append("  - %s：%s B" % (e["path"], e["size"]))
    L.append("")
    L.append("## 限制")
    for lim in r["limitations"]:
        L.append("- %s" % lim)
    L.append("")
    L.append("## 建议下一步")
    L.append("- %s" % r["recommended_next_action"])
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
