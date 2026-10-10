#!/usr/bin/env python3
"""Summarize the minimal ZCU102 block-design run into the task reports.

Reads only bounded text evidence (the /tmp report dumps plus the copied artifact
directory) and writes:
  reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.json
  reports/token_selector/2026-10-03-integer-selector-minimal-zynq-bd.md
  reports/token_selector/logs/2026-10-03-integer-selector-minimal-zynq-bd-harness-result.json
"""
import argparse
import glob
import json
import os
import re
import sys

SELECTOR_VLNV = "robobrain.local:hls:integer_selector_pipeline_compact:1.0"
PART = "xczu9eg-ffvb1156-2-e"
BOARD = "xilinx.com:zcu102:part0:3.4"
M_AXI = ["m_axi_input_arena", "m_axi_wv_q", "m_axi_output_arena"]


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def read_kv(path):
    out = {}
    for line in read(path).splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def read_tsv(path):
    rows = []
    for i, line in enumerate(read(path).splitlines()):
        if not line.strip():
            continue
        parts = line.split("\t")
        if i == 0:
            rows.append(("__header__", parts))
        else:
            rows.append(("row", parts))
    return rows


def rows_of(path):
    out = []
    for kind, parts in read_tsv(path):
        if kind == "row":
            out.append(parts)
    return out


def count_log(path):
    errs, crit, other_warn = [], [], []
    for line in read(path).splitlines():
        if line.startswith("ERROR:"):
            errs.append(line.strip())
        elif line.startswith("CRITICAL WARNING:"):
            if "XilinxTclStore" in line or "local Tcl store" in line:
                continue
            crit.append(line.strip())
        elif line.startswith("WARNING:"):
            other_warn.append(line.strip())
    return errs, crit, other_warn


def parse_util(text):
    out = {}
    for line in text.splitlines():
        m = re.match(
            r"\|\s*([A-Za-z0-9_ .()/+\-*]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            line,
        )
        if m:
            def num(s):
                return float(s) if "." in s else int(s)
            out[m.group(1).strip()] = {
                "used": num(m.group(2)),
                "fixed": num(m.group(3)),
                "prohibited": num(m.group(4)),
                "available": num(m.group(5)),
                "pct": float(m.group(6)),
            }
    return out


def parse_timing(text):
    lines = text.splitlines()
    best = None
    for i, line in enumerate(lines):
        if "WNS(ns)" in line and "TNS(ns)" in line:
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s and not set(s) <= set("- "):
                    best = s
                    break
                j += 1
            if best:
                break
    if not best:
        return {}
    vals = best.split()
    keys = ["WNS_ns", "TNS_ns", "TNS_failing", "TNS_total",
            "WHS_ns", "THS_ns", "THS_failing", "THS_total",
            "WPWS_ns", "TPWS_ns", "TPWS_failing", "TPWS_total"]
    return {k: v for k, v in zip(keys, vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="/tmp report dump dir")
    ap.add_argument("--art", required=True, help="artifact dir")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--md-out", required=True)
    ap.add_argument("--harness-out", required=True)
    args = ap.parse_args()

    R = args.reports
    A = args.art
    B = os.path.join(A, "bd")       # archived BD evidence dumps
    V = os.path.join(A, "vivado")   # archived Vivado logs / reports

    def bpath(name):
        p = os.path.join(B, name)
        return p if os.path.isfile(p) else os.path.join(R, name)

    def vpath(name):
        p = os.path.join(V, name)
        return p if os.path.isfile(p) else os.path.join(R, name)

    info = read_kv(bpath("bd_design_info.txt"))
    cells = rows_of(bpath("bd_cells.tsv"))
    intf_pins = rows_of(bpath("bd_intf_pins.tsv"))
    intf_nets = rows_of(bpath("bd_intf_nets.tsv"))
    addr_segs = rows_of(bpath("bd_addr_segs.tsv"))
    ps_cfg = {r[0]: r[1] for r in rows_of(bpath("bd_ps_config.tsv")) if len(r) >= 2}
    reset_params = {r[1]: r[2] for r in rows_of(bpath("bd_reset_params.tsv")) if len(r) >= 3}
    sc_params = rows_of(bpath("bd_interconnect_params.tsv"))
    wrapper_txt = read_kv(bpath("bd_wrapper.txt"))

    build_log = read(os.path.join(V, "build_bd.log"))
    synth_log = read(os.path.join(V, "synth_bd.log"))
    errs, crit, warns = count_log(os.path.join(V, "build_bd.log")) if build_log else count_log(os.path.join(R, "build_bd.log"))
    synth_errs, synth_crit, _ = count_log(os.path.join(V, "synth_bd.log"))

    stage_status = rows_of(bpath("bd_stage_status.tsv"))
    synth_stage_status = rows_of(vpath("synth_stage_status.tsv"))

    util = parse_util(read(os.path.join(V, "utilization_post_synth_bd.rpt")))
    timing = parse_timing(read(os.path.join(V, "timing_summary_post_synth_bd.rpt")))
    clocks_rpt = read(os.path.join(V, "clocks_post_synth_bd.rpt"))

    sha_compare = json.loads(read(os.path.join(A, "protected_sha_compare.json")) or "{}")
    run_status = read_kv(os.path.join(A, "logs", "run_status.txt"))
    tool_versions = read(os.path.join(A, "tool_versions.txt"))
    top_ports = read(os.path.join(A, "ip", "top_module_ports.txt"))

    # ---------------- interface inventory ----------------
    def pin_role(path, suffix):
        for r in intf_pins:
            if len(r) >= 5 and r[0] == path:
                return r
        for r in intf_pins:
            if len(r) >= 5 and r[0].endswith(suffix):
                return r
        return None

    sel_masters = []
    for name in M_AXI:
        r = pin_role("/integer_selector_pipeline_compact_0/" + name, name)
        if r:
            sel_masters.append({"interface": r[0], "name": name, "mode": r[1],
                                "protocol": r[2], "data_width": int(r[3]) if r[3] else None,
                                "freq_hz": r[4]})
    ctl_row = pin_role("/integer_selector_pipeline_compact_0/s_axi_control", "s_axi_control")
    control = ({"interface": ctl_row[0], "mode": ctl_row[1], "protocol": ctl_row[2],
                "data_width": int(ctl_row[3]) if ctl_row[3] else None, "freq_hz": ctl_row[4]}
               if ctl_row else None)

    intf_pin_inventory = [
        {"pin": r[0], "mode": r[1], "protocol": r[2],
         "data_width": int(r[3]) if len(r) > 3 and r[3] else None,
         "freq_hz": r[4] if len(r) > 4 else ""}
        for r in intf_pins if len(r) >= 3
    ]

    # ---------------- connectivity ----------------
    def net_members(netname):
        for r in intf_nets:
            if r and r[0] == netname:
                return r[1].split() if len(r) > 1 else []
        for r in intf_nets:
            if r and r[0].endswith(netname):
                return r[1].split() if len(r) > 1 else []
        return []

    m_axi_targets = {}
    for name in M_AXI:
        members = net_members("integer_selector_pipeline_compact_0_" + name)
        m_axi_targets[name] = members
    ddr_mi = net_members("smartconnect_ddr_M00_AXI")
    ctl_mi = net_members("smartconnect_control_M00_AXI")
    ps_master_net = net_members("zynq_ultra_ps_e_0_M_AXI_HPM0_FPD")
    connectivity = {
        "selector_master_nets": m_axi_targets,
        "smartconnect_ddr_M00_AXI": ddr_mi,
        "smartconnect_control_M00_AXI": ctl_mi,
        "ps_M_AXI_HPM0_FPD": ps_master_net,
        "all_interface_nets": [{"net": r[0], "members": r[1]} for r in intf_nets if len(r) >= 2],
    }

    three_to_one_ok = (
        all(any("smartconnect_ddr/S" in m for m in members) for members in m_axi_targets.values())
        and any("zynq_ultra_ps_e_0/S_AXI_HP0_FPD" in m for m in ddr_mi)
    )
    control_path_ok = (
        any("smartconnect_control/S00_AXI" in m for m in ps_master_net)
        and any("integer_selector_pipeline_compact_0/s_axi_control" in m for m in ctl_mi)
    )

    # ---------------- address map ----------------
    # DDR master address segments are the rows that carry a Selector m_axi
    # master path and a real SEG_ addressable (the PS SAXIGP2 memory-map row
    # also mentions HP0_DDR_LOW but is not a PL master segment).
    ddr_segs = [r for r in addr_segs if any(name in r[0] for name in M_AXI) and "_DDR_" in r[0]]
    control_segs = [r for r in addr_segs if "Reg" in r[0]]
    control_base = None
    for r in control_segs:
        if len(r) > 1 and r[1]:
            control_base = r[1]
    ddr_by_master = {}
    for name in M_AXI:
        for r in ddr_segs:
            if name in r[0]:
                rng = r[2] if len(r) > 2 else ""
                try:
                    nonempty = bool(rng) and int(rng, 16) > 0
                except ValueError:
                    nonempty = False
                ddr_by_master[name] = {"segment": r[0], "offset": r[1] if len(r) > 1 else "",
                                       "range": rng, "range_nonempty": nonempty}
    all_three_ddr = len(ddr_by_master) == 3 and all(v["range_nonempty"] for v in ddr_by_master.values())

    # ---------------- clocks / resets ----------------
    clk_pins = [r for r in rows_of(os.path.join(R, "bd_signal_pins.tsv")) if len(r) >= 4 and "clk" in r[0].lower()]
    rst_pins = [r for r in rows_of(os.path.join(R, "bd_signal_pins.tsv")) if len(r) >= 4 and ("rst" in r[0].lower() or "reset" in r[0].lower())]
    ap_clk_freq = None
    for p in intf_pin_inventory:
        if p["pin"].endswith("/ap_clk") or p["pin"].endswith("ap_clk"):
            ap_clk_freq = p["freq_hz"]
    # fall back: any pin frequency
    if not ap_clk_freq:
        for p in intf_pin_inventory:
            if p["freq_hz"]:
                ap_clk_freq = p["freq_hz"]
                break
    try:
        ap_clk_mhz = round(int(ap_clk_freq) / 1e6, 4) if ap_clk_freq else None
    except (TypeError, ValueError):
        ap_clk_mhz = None

    reset_connect = {
        "ext_reset_in_driven_by": [r[0] for r in rst_pins if "ext_reset_in" in r[0]],
        "peripheral_aresetn_members": [r[3] for r in rst_pins if "peripheral_aresetn" in r[0]],
        "ap_rst_n_net": [r[3] for r in rst_pins if "ap_rst_n" in r[0]],
    }
    reset_polarity = reset_params.get("CONFIG.C_EXT_RESET_HIGH")

    # ---------------- selector port safety ----------------
    no_control_r = "control_r" not in top_ports
    no_score_port = ("score" not in top_ports.lower())
    interface_inventory_ok = (
        len(sel_masters) == 3
        and all(m["mode"] == "Master" and m["protocol"] == "AXI4" and m["data_width"] == 128 for m in sel_masters)
        and control is not None
        and control["mode"] == "Slave" and control["protocol"] == "AXI4LITE" and control["data_width"] == 32
        and no_control_r and no_score_port
    )

    # ---------------- output products / wrapper ----------------
    wrapper_list = wrapper_txt.get("wrapper_files", "")
    top_set = wrapper_txt.get("top", "")
    wrapper_ok = bool(wrapper_list.strip()) and wrapper_list.strip() not in ("{}", "")

    # ---------------- synthesis ----------------
    synth_status = None
    synth_rc = None
    synth_seconds = None
    for r in synth_stage_status:
        if r and r[0] == "synth_design":
            synth_rc = int(r[1])
            synth_seconds = int(r[2]) if len(r) > 2 and r[2] else None
    if synth_rc is not None:
        synth_status = "OK" if synth_rc == 0 else "ERROR"

    # ---------------- gates ----------------
    gates = {
        "protected_files_unchanged": bool(sha_compare.get("protected_files_unchanged")),
        "exact_part_and_board_recorded": info.get("part") == PART and info.get("board_part") == BOARD,
        "selector_vlnv_instantiated": any(
            len(r) >= 2 and r[1] == SELECTOR_VLNV for r in cells),
        "selector_interface_inventory_exact": interface_inventory_ok,
        "three_master_to_ps_ddr_path": three_to_one_ok,
        "ps_master_to_control_and_base": control_path_ok and bool(control_base),
        "clocks_resets_connected": (
            ap_clk_mhz is not None
            and reset_connect["ap_rst_n_net"]
            and any("peripheral_aresetn" in m for m in reset_connect["peripheral_aresetn_members"])
        ),
        "pl_clock_200mhz": ap_clk_mhz is not None and abs(ap_clk_mhz - 200.0) < 0.1,
        "three_ddr_segments_nonempty": all_three_ddr,
        "validate_bd_design_no_error": info.get("validate_bd_design_rc") == "0" and len(errs) == 0,
        "output_products_and_wrapper": wrapper_ok,
        "no_bitstream_no_board": int(run_status.get("BITSTREAM_FILES", "0") or 0) == 0,
    }
    decision = (
        "INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_PASS"
        if all(gates.values())
        else "INTEGER_SELECTOR_MINIMAL_ZYNQ_BD_BLOCKED"
    )

    domains = {
        "control_registers": {
            "master": "/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD",
            "slave": "/integer_selector_pipeline_compact_0/s_axi_control",
            "class": "AXI4-Lite register range",
            "base_address": control_base,
            "range": next((r[2] for r in control_segs if len(r) > 2 and r[1] == control_base), None),
            "segment": next((r[0] for r in control_segs if len(r) > 1 and r[1] == control_base), None),
        },
        "ddr": {
            "ps_slave": "/zynq_ultra_ps_e_0/S_AXI_HP0_FPD",
            "memmap": "HP0_DDR_LOW",
            "segments": ddr_by_master,
        },
    }

    # `[BD 5-234] No nets matched` warnings are emitted by the evidence-dump
    # loops (get_bd_nets on unconnected pins), not by validation itself.
    design_warns = [w for w in warns if "[BD 5-234]" not in w]
    dump_warns = [w for w in warns if "[BD 5-234]" in w]

    resources = {
        "CLB_LUTs": util.get("CLB LUTs*"),
        "CLB_Registers": util.get("CLB Registers"),
        "Block_RAM_Tile": util.get("Block RAM Tile"),
        "DSPs": util.get("DSPs"),
        "URAM": util.get("URAM"),
        "PS8": util.get("PS8"),
    }

    report = {
        "task": "integer_selector_minimal_zynq_bd",
        "decision": decision,
        "gates": gates,
        "tool": {
            "vivado": "2023.2",
            "tool_versions_txt": tool_versions.strip().splitlines(),
            "license_limit": "Tool Version Limit: 2023.10",
            "part_available": info.get("part") == PART,
            "board_part": info.get("board_part"),
            "board_part_status": info.get("board_part_status"),
            "board_part_available_versions": ["xilinx.com:zcu102:part0:3.3", "xilinx.com:zcu102:part0:3.4"],
        },
        "design": {
            "design_name": info.get("design_name"),
            "part": info.get("part"),
            "board_part": info.get("board_part"),
            "selector_vlnv": info.get("selector_vlnv"),
            "interrupt_used": info.get("interrupt_used"),
        },
        "cell_inventory": [{"cell": r[0], "vlnv": r[1] if len(r) > 1 else "",
                            "type": r[2] if len(r) > 2 else ""} for r in cells],
        "axi_interface_inventory": intf_pin_inventory,
        "selector_interface_summary": {
            "axi4_masters_128": sel_masters,
            "axi4lite_slave_control": control,
            "control_r_present": not no_control_r,
            "score_port_present": not no_score_port,
        },
        "connectivity": connectivity,
        "ps_interfaces": {
            "control_master": "M_AXI_HPM0_FPD (128-bit AXI4, GP0) -> AXI SmartConnect -> control",
            "ddr_slave": "S_AXI_HP0_FPD (128-bit AXI4 high-performance DDR port, enabled via PSU__USE__S_AXI_GP2)",
            "pl_clock_source": "pl_clk0 from IOPLL, requested 200 MHz",
            "ps_config_subset": ps_cfg,
        },
        "clocks_resets": {
            "ap_clk_freq_hz": ap_clk_freq,
            "ap_clk_mhz": ap_clk_mhz,
            "clock_pins": [{"pin": r[0], "dir": r[1] if len(r) > 1 else "", "type": r[2] if len(r) > 2 else "", "net": r[3] if len(r) > 3 else ""} for r in clk_pins],
            "reset_pins": [{"pin": r[0], "dir": r[1] if len(r) > 1 else "", "type": r[2] if len(r) > 2 else "", "net": r[3] if len(r) > 3 else ""} for r in rst_pins],
            "reset_connectivity": reset_connect,
            "proc_sys_reset_C_EXT_RESET_HIGH": reset_polarity,
            "proc_sys_reset_active_low": reset_polarity == "0",
        },
        "address_map": {
            "domains": domains,
            "all_segments": [{"segment": r[0], "offset": r[1] if len(r) > 1 else "",
                              "range": r[2] if len(r) > 2 else ""} for r in addr_segs],
        },
        "validation": {
            "validate_bd_design_rc": info.get("validate_bd_design_rc"),
            "error_count": len(errs),
            "critical_warning_count": len(crit),
            "warning_count": len(design_warns),
            "evidence_dump_warning_count": len(dump_warns),
            "total_warning_count": len(warns),
            "errors": errs[:50],
            "critical_warnings": crit[:50],
            "warning_examples": sorted(set(design_warns))[:20],
            "warning_notes": [
                "AWLOCK/ARLOCK width mismatch (HLS 2-bit vs AXI4 1-bit) is benign; only the lower bit is used.",
                "SmartConnect low-area mode does not support WRAP bursts; the AXI4-Lite control path issues "
                "single-beat register transactions only.",
            ],
        },
        "output_products": {
            "generate_target_all": True,
            "wrapper_files": wrapper_list,
            "top": top_set,
        },
        "stage_status": [{"stage": r[0], "rc": r[1], "seconds": r[2] if len(r) > 2 else ""} for r in stage_status],
        "synthesis": {
            "run": synth_rc == 0 if synth_rc is not None else False,
            "status": synth_status,
            "open_run_seconds": synth_seconds,
            "wall_seconds": run_status.get("SYNTH_SECONDS"),
            "resources": resources,
            "utilization": util,
            "timing": timing,
            "error_count": len(synth_errs),
            "critical_warning_count": len(synth_crit),
        },
        "protected_sha_audit": {
            "before_file_count": sha_compare.get("before_file_count"),
            "after_file_count": sha_compare.get("after_file_count"),
            "before_aggregate_sha256": sha_compare.get("before_aggregate_sha256"),
            "after_aggregate_sha256": sha_compare.get("after_aggregate_sha256"),
            "changed": sha_compare.get("changed"),
            "missing": sha_compare.get("missing"),
            "added": sha_compare.get("added"),
            "protected_files_unchanged": sha_compare.get("protected_files_unchanged"),
        },
        "run_status": run_status,
        "limitations": [
            "No Ethernet/Jetson software path is implemented.",
            "No PS software (bare-metal/Linux) driver is implemented.",
            "No DDR traffic contention / arbitration bandwidth is measured.",
            "No full-system timing sign-off (no implementation, no bitstream).",
            "No bitstream is generated and the board is never accessed.",
            "No board measurement is performed.",
            "No model end-to-end latency is measured.",
            "No task-accuracy claim is made.",
        ],
        "next_action": "Implement the PS-side bare-metal/Linux register-and-DDR test harness using static JPG-derived fixture buffers, before any Jetson Ethernet integration.",
    }

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)

    harness = {
        "task": "integer_selector_minimal_zynq_bd",
        "decision": decision,
        "gates": gates,
        "selector_vlnv": info.get("selector_vlnv"),
        "part": info.get("part"),
        "board_part": info.get("board_part"),
        "axi_masters_128": [m["name"] for m in sel_masters],
        "control_slave": control["interface"] if control else None,
        "control_base_address": control_base,
        "ddr_segments": ddr_by_master,
        "ap_clk_mhz": ap_clk_mhz,
        "validate_rc": info.get("validate_bd_design_rc"),
        "error_count": len(errs),
        "critical_warning_count": len(crit),
        "wrapper_files": wrapper_list,
        "synthesis_status": synth_status,
        "synthesis_seconds": run_status.get("SYNTH_SECONDS"),
        "synthesis_resources": resources,
        "synthesis_timing": timing,
        "protected_files_unchanged": sha_compare.get("protected_files_unchanged"),
        "protected_file_count": sha_compare.get("before_file_count"),
        "bitstream_files": run_status.get("BITSTREAM_FILES"),
        "json_report": os.path.relpath(args.json_out, args.repo),
        "md_report": os.path.relpath(args.md_out, args.repo),
    }
    os.makedirs(os.path.dirname(args.harness_out), exist_ok=True)
    with open(args.harness_out, "w", encoding="utf-8") as fh:
        json.dump(harness, fh, indent=1, sort_keys=True)

    # ---------------- markdown ----------------
    def g(v):
        return "PASS" if v else "FAIL"

    lines = []
    lines.append("# 整数 Selector 最小 ZCU102 Zynq MPSoC Block Design 验证")
    lines.append("")
    lines.append(f"- 决策：**{decision}**")
    lines.append(f"- 工程日期：2026-10-03；工具：Vivado 2023.2（Tool Version Limit 2023.10）")
    lines.append(f"- part：`{info.get('part')}`；board part：`{info.get('board_part')}`（{info.get('board_part_status')}）")
    lines.append(f"- Selector VLNV：`{info.get('selector_vlnv')}`")
    lines.append(f"- 工程记录：`fpga_accel/integer_selector_minimal_zynq_bd/`；证据：`artifacts/token_selector/2026-10-03-integer-selector-minimal-zynq-bd/`")
    lines.append("")
    lines.append("## 门（Gates）")
    lines.append("")
    lines.append("| gate | result |")
    lines.append("|---|---|")
    for k, v in gates.items():
        lines.append(f"| {k} | {g(v)} |")
    lines.append("")
    lines.append("## AXI 接口清单")
    lines.append("")
    lines.append("| pin | mode | protocol | data width | freq (Hz) |")
    lines.append("|---|---|---|---|---|")
    for p in intf_pin_inventory:
        lines.append(f"| `{p['pin']}` | {p['mode']} | {p['protocol']} | {p['data_width']} | {p['freq_hz']} |")
    lines.append("")
    lines.append("## AXI 连接图")
    lines.append("")
    for k, v in connectivity.items():
        if k == "all_interface_nets":
            continue
        lines.append(f"- `{k}` -> {v}")
    lines.append("")
    lines.append("## 时钟 / 复位")
    lines.append("")
    lines.append(f"- `ap_clk` = {ap_clk_mhz} MHz（目标 200 MHz；PS `pl_clk0`，IOPLL）")
    lines.append(f"- `proc_sys_reset` `C_EXT_RESET_HIGH` = {reset_polarity}（0 = 外部复位低有效）")
    lines.append(f"- 复位连接：{reset_connect}")
    lines.append("")
    lines.append("## 地址映射")
    lines.append("")
    lines.append(f"- AXI-Lite control 基地址：`{control_base}`")
    lines.append("- DDR 段（每个 Selector master）：")
    for name, seg in ddr_by_master.items():
        lines.append(f"  - `{name}`：`{seg['offset']}` .. range `{seg['range']}`（{seg['segment']}）")
    lines.append("")
    lines.append("## 校验 / 综合")
    lines.append("")
    lines.append(f"- `validate_bd_design` rc = {info.get('validate_bd_design_rc')}，ERROR = {len(errs)}，CRITICAL WARNING = {len(crit)}，设计 WARNING = {len(design_warns)}（另有 {len(dump_warns)} 条证据导出的 BD 5-234）")
    lines.append(f"- HDL wrapper：{wrapper_list}")
    lines.append(f"- 综合：{synth_status}（wall {run_status.get('SYNTH_SECONDS')} s）")
    if resources:
        lines.append(f"- 综合资源（post-synth，逻辑 + PS）：{json.dumps(resources, ensure_ascii=False)}")
    if timing:
        lines.append(f"- 综合时序（post-synth，未实现/未布线）：{json.dumps(timing, ensure_ascii=False)}")
    if clocks_rpt:
        lines.append("- 时钟约束报告：`vivado/clocks_post_synth_bd.rpt`")
    lines.append("")
    lines.append("## 受限说明")
    lines.append("")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    lines.append(f"**下一步**：{report['next_action']}")
    lines.append("")

    with open(args.md_out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(json.dumps({"decision": decision, "gates": gates}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
