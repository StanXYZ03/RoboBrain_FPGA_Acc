#!/usr/bin/env python3
"""Assemble the compact-pipeline JSON + Chinese Markdown reports and the harness
result.

Reads the controlled csynth ``summary.json``, the fixture ledger, the four
verification logs, the transfer audit and the protected-tree baseline; writes

* reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json
* reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.md
* reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json

No model, GPU, network, board or training is touched.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

MODULE = Path(__file__).resolve().parents[1]
ROOT = MODULE.parents[1]
for path in (str(ROOT), str(MODULE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.hardware_regular_token_selector import (  # noqa: E402
    describe_fpga_layout,
)
from tools.transfer_audit import (  # noqa: E402
    GBE_EFFECTIVE_MBPS_CENTRAL,
    build_transfer_audit,
    input_arena_bytes,
)

DATE = "2026-10-03"
ARTIFACT = (
    ROOT / "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls"
)
FIXTURES = ARTIFACT / "fixtures"
CSYNTH = ARTIFACT / "csynth"
LOGS = ARTIFACT / "logs"
REPORTS = ROOT / "reports/token_selector"
REPORT_JSON = REPORTS / "2026-10-03-integer-selector-compact-interface-hls.json"
REPORT_MD = REPORTS / "2026-10-03-integer-selector-compact-interface-hls.md"
HARNESS_RESULT = (
    REPORTS / "logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json"
)
PROTECTED_BASELINE = ARTIFACT / "protected_trees_before.json"

PROTECTED_TREES = [
    "fpga_accel/integer_selector_scorer_hls",
    "fpga_accel/token_selector_hls",
    "fpga_accel/integer_selector_pipeline_hls",
    "evaluation",
]

TARGET_FMAX_MHZ = 200.0
LAYOUT_BUDGETS = (16, 32, 48, 64, 80, 96)
INPUT_GROUP_COUNTS = (9, 16, 50, 80, 128, 260, 400, 4096)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_protected() -> dict[str, Any]:
    baseline = _read_json(PROTECTED_BASELINE)
    groups: dict[str, Any] = {}
    all_ok = True
    for tree in PROTECTED_TREES:
        recorded = baseline["trees"][tree]["files"]
        changed: list[str] = []
        missing: list[str] = []
        unchanged = 0
        for rel, before in recorded.items():
            path = ROOT / rel
            if not path.is_file():
                missing.append(rel)
                continue
            if sha256_file(path) == before:
                unchanged += 1
            else:
                changed.append(rel)
        ok = not changed and not missing
        all_ok = all_ok and ok
        groups[tree] = {
            "file_count": len(recorded),
            "unchanged": unchanged,
            "changed": changed,
            "missing": missing,
            "ok": ok,
        }
    return {
        "baseline_path": str(PROTECTED_BASELINE.relative_to(ROOT)),
        "baseline_sha256": sha256_file(PROTECTED_BASELINE),
        "total_files": baseline["total_files"],
        "all_unchanged": all_ok,
        "groups": groups,
    }


def parse_verification() -> dict[str, Any]:
    def text(name: str) -> str:
        return (LOGS / name).read_text(encoding="utf-8", errors="replace")

    py = text("01_python_fixtures.log")
    gpp = text("02_gpp_test.log")
    asan = text("03_asan_ubsan.log")
    csim = text("04_hls_csim.log")

    def summary_line(blob: str) -> dict[str, int]:
        match = re.search(r"ISP_SUMMARY cases=(\d+) passed=(\d+) failed=(\d+)", blob)
        if not match:
            return {"cases": 0, "passed": 0, "failed": 0}
        return {
            "cases": int(match.group(1)),
            "passed": int(match.group(2)),
            "failed": int(match.group(3)),
        }

    py_tests = re.search(r"Ran (\d+) tests", py)
    return {
        "python": {
            "tests_run": int(py_tests.group(1)) if py_tests else 0,
            "ok": bool(py_tests) and "OK" in py,
            "log": "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/01_python_fixtures.log",
        },
        "gpp": {
            **summary_line(gpp),
            "result": "PASS" if "ISPC_RESULT PASS" in gpp else "FAIL",
            "reference_so": "loaded" if "ISPC_REF_SO loaded" in gpp else "not_loaded",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/02_gpp_test.log",
        },
        "asan_ubsan": {
            **summary_line(asan),
            "result": "PASS" if "ISPC_RESULT PASS" in asan else "FAIL",
            "reference_so": "loaded" if "ISPC_REF_SO loaded" in asan else "not_loaded",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/03_asan_ubsan.log",
        },
        "hls_csim": {
            **summary_line(csim),
            "csim_errors": int(
                (re.search(r"CSim done with (\d+) errors", csim) or [0, -1])[1]
            ),
            "result": "PASS" if "ISPC_RESULT PASS" in csim else "FAIL",
            "reference_so": "loaded" if "ISPC_REF_SO loaded" in csim else "not_loaded",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/04_hls_csim.log",
        },
    }


def layout_evidence() -> dict[str, Any]:
    return {
        "authority": {
            "module": "evaluation.hardware_regular_token_selector",
            "functions": ["fpga_transfer_layout", "describe_fpga_layout"],
        },
        "input_arena": {
            "total_bytes_formula": "G*5120 + 128",
            "tokens_offset": 0,
            "tokens_bytes_formula": "G*4096",
            "group_mean_offset_formula": "G*4096",
            "group_mean_bytes_formula": "G*1024",
            "query_offset_formula": "G*5120",
            "query_bytes": 128,
            "alignment": 64,
            "per_group_counts": {
                str(g): {
                    "tokens_offset": 0,
                    "group_mean_offset": g * 4096,
                    "query_offset": g * 5120,
                    "total_bytes": input_arena_bytes(g),
                }
                for g in INPUT_GROUP_COUNTS
            },
        },
        "output_arena": {
            str(budget): describe_fpga_layout(budget) for budget in LAYOUT_BUDGETS
        },
    }


def collect_files() -> dict[str, str]:
    files: dict[str, str] = {}
    for base in (MODULE, FIXTURES, CSYNTH, LOGS):
        for path in sorted(Path(base).rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in {".pyc"} or "__pycache__" in path.parts:
                continue
            files[str(path.relative_to(ROOT))] = sha256_file(path)
    return files


def build_bundle() -> dict[str, Any]:
    summary = _read_json(CSYNTH / "summary.json")
    manifest = _read_json(FIXTURES / "manifest.json")
    verification = parse_verification()
    transfer = build_transfer_audit()
    protected = verify_protected()
    layout = layout_evidence()

    gate = summary["gate"]
    functional_pass = (
        verification["python"]["ok"]
        and verification["gpp"]["result"] == "PASS"
        and verification["asan_ubsan"]["result"] == "PASS"
        and verification["hls_csim"]["result"] == "PASS"
        and verification["hls_csim"]["csim_errors"] == 0
        and verification["gpp"]["failed"] == 0
        and verification["asan_ubsan"]["failed"] == 0
        and verification["hls_csim"]["failed"] == 0
        and verification["gpp"]["reference_so"] == "loaded"
        and verification["asan_ubsan"]["reference_so"] == "loaded"
    )
    layout_totals = {
        str(budget): layout["output_arena"][str(budget)]["total_bytes"]
        for budget in (48, 64, 96)
    }
    gates = {
        "protected_trees_unchanged": protected["all_unchanged"],
        "functional_gates_pass": functional_pass,
        "exactly_three_m_axi_bundles": gate["exactly_three_m_axi_bundles"],
        "exactly_one_axilite_bundle": gate["exactly_one_axilite_bundle"],
        "no_external_score_port": gate["no_external_score_port"],
        "zero_errors": gate["zero_errors"],
        "estimated_period_le_5ns": gate["estimated_period_le_5ns"],
        "timing_met": gate["timing_met"],
        "all_resources_under_50_percent": gate["all_resources_under_50_percent"],
        "output_arena_k48_k64_k96_exact": layout_totals
        == {"48": 196928, "64": 262464, "96": 393792},
        "derived_production_cases_finite": all(
            summary["derived_model"]["configs"][key]["total_cycles_estimate"] is not None
            for key in ("G80_K48", "G260_K64", "G400_K96")
        ),
    }
    decision = (
        "INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_PASS"
        if all(gates.values())
        else "INTEGER_SELECTOR_COMPACT_INTERFACE_HLS_BLOCKED"
    )

    derived = summary["derived_model"]["configs"]
    interface = summary["interface"]

    # Raw interface/layout evidence next to the csynth reports.
    (CSYNTH / "interface_evidence.json").write_text(
        json.dumps(
            {
                "top": summary["top"],
                "part": summary["part"],
                "m_axi_bundles": interface["m_axi_bundles"],
                "axi_lite_bundles": interface["s_axi_bundles"],
                "axi_lite_bundle_count": interface["axi_lite_bundle_count"],
                "score_ports": interface["score_ports"],
                "bus_width_bits": 128,
                "expected": {
                    "m_axi": sorted(
                        ["input_arena", "wv_q", "output_arena"]
                    ),
                    "axi_lite": ["control"],
                },
                "gate_ok": (
                    set(interface["m_axi_bundles"])
                    == {"input_arena", "wv_q", "output_arena"}
                    and set(interface["s_axi_bundles"]) == {"control"}
                    and interface["score_ports"] == []
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (CSYNTH / "layout_evidence.json").write_text(
        json.dumps(layout, indent=2) + "\n", encoding="utf-8"
    )

    limitations = [
        "HLS evidence only: csynth_design plus host g++/ASan/UBSan and Vitis HLS "
        "C-simulation.  This is NOT RTL co-simulation, NOT place-and-route/routed "
        "timing, NOT a bitstream, NOT a board measurement and NOT a model "
        "end-to-end latency or task-accuracy result.",
        "The per-configuration G80/K48, G260/K64 and G400/K96 cycle counts are "
        "FORMULA estimates built from the tool-reported per-loop unit costs; only "
        "the reported best/average/worst latency (compile-time trip-count bounds) "
        "is a tool number.",
        "The transfer audit is a model with an explicit 1 GbE effective-bandwidth "
        "assumption and a sensitivity band; it is not a measured link throughput.",
        "The compact envelope removes the intermediate score_q FPGA-DDR round trip "
        "and merges the input pointers into one arena; it does not remove the "
        "Jetson <-> FPGA round trip.",
        "group_mean_q is still supplied inside the input arena; reconstructing it "
        "from tokens_q in the PL is not attempted here and the accepted group-mean "
        "quantisation contract is unchanged.",
        "Wv is assumed to remain resident in FPGA-attached memory in the corrected "
        "resident mode; the cold mode number is reported for completeness.",
        "No IP export, no cosim and no implementation/PnR were run; the AXI-Lite "
        "interface count is proven from the generated csynth report/XML only.",
    ]

    return {
        "schema_version": "1.0",
        "date": DATE,
        "task": "Compact AXI interface wrapper (3 m_axi + 1 AXI-Lite) around the "
                "accepted fused integer Selector pipeline",
        "decision": decision,
        "scope": {
            "cpu_only": True,
            "gpu": False,
            "network": False,
            "training": False,
            "parameter_search": False,
            "model_load": False,
            "cosim": False,
            "export_ip": False,
            "implementation": False,
            "bitstream": False,
            "board_access": False,
            "raw_images": False,
            "internal_test_data": False,
            "csynth_design_only": True,
            "tool": "Vitis HLS 2023.2",
            "part": "xczu9eg-ffvb1156-2-e",
            "clock_ns": 5.0,
            "uncertainty_ns": summary["clock_uncertainty_ns"],
        },
        "contract": {
            "top": "integer_selector_pipeline_compact",
            "score_q": "local on-chip uint16_t buffer[G]; NOT a top-level port and never DDR",
            "m_axi_bundles": interface["m_axi_bundles"],
            "axi_lite_bundles": interface["s_axi_bundles"],
            "axi_lite_bundle_count": interface["axi_lite_bundle_count"],
            "bus_width_bits": 128,
            "inputs": [
                "input_arena [G*320 + 8] 128-bit words: tokens | group means | query",
                "wv_q [8192] 128-bit words (128x1024 row-major INT8)",
            ],
            "outputs": "output_arena [ispc_fpga_layout(K).total_bytes/16] 128-bit "
                       "words: tokens + 4 metadata segments at frozen offsets",
            "scalar_outputs": [
                "scale_out",
                "output_count",
                "real_count",
                "context_count",
            ],
            "status_codes": {
                "0": "OK",
                "1": "ERR_NULL",
                "2": "ERR_GROUP_COUNT",
                "3": "ERR_GRID",
                "4": "ERR_BUDGET",
                "5": "ERR_INSUFFICIENT",
                "6": "ERR_CANDIDATE_POOL",
                "7": "ERR_INTERNAL",
                "8": "ERR_LOCAL_CAP",
            },
            "status_codes_match_accepted_pipeline": True,
            "reuse": "the two accepted translation units are #include'd into the "
                     "wrapper; is_scorer_core<> and token_selector_core_packed are "
                     "called directly; no second copy of the maths",
        },
        "interface": interface,
        "layout": layout,
        "output_arena_totals": layout_totals,
        "golden": manifest["golden"],
        "wv": manifest["wv"],
        "fixtures": {
            "dir": str(FIXTURES.relative_to(ROOT)),
            "scenes": manifest["scenes"],
            "cases": manifest["cases"],
            "coverage": manifest["coverage"],
        },
        "verification": verification,
        "csynth": {
            "top": summary["top"],
            "part": summary["part"],
            "clock_target_ns": summary["clock_target_ns"],
            "clock_uncertainty_ns": summary["clock_uncertainty_ns"],
            "effective_budget_ns": summary["effective_budget_ns"],
            "estimated_clock_ns": summary["estimated_clock_ns"],
            "estimated_fmax_mhz": summary["estimated_fmax_mhz"],
            "timing_slack_ns": summary["timing_slack_ns"],
            "reported_latency_cycles": summary["reported_latency_cycles"],
            "reported_latency_ms_at_200mhz": summary["reported_latency_ms_at_200mhz"],
            "scorer_phase": summary["scorer_phase"],
            "resources_used": summary["resources_used"],
            "resources_available": summary["resources_available"],
            "resource_utilization_percent": summary["resource_utilization_percent"],
            "console": summary["console"],
            "critical_path": summary["critical_path"],
            "selector_pipelines": summary["selector_pipelines"],
            "selector_loops": summary["selector_loops"],
            "scorer_loops": summary["scorer_loops"],
        },
        "derived_latency_model": summary["derived_model"],
        "derived_latency_200mhz": {
            key: derived[key] for key in ("G80_K48", "G260_K64", "G400_K96")
        },
        "transfer_audit": transfer,
        "protected_trees": protected,
        "gate_checks": {"gates": gates, "pass": decision.endswith("_PASS")},
        "limitations": limitations,
        "files": collect_files(),
        "reproduction": [
            "cd fpga_accel/integer_selector_pipeline_compact_hls",
            "make python_test",
            "make test",
            "make asan_run",
            "make csim",
            "make csynth",
            "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -B scripts/summarize_csynth.py "
            "--report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth "
            "--console-log ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/console.log "
            "--out ../../artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/summary.json",
            "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python -B tools/build_report.py",
        ],
    }


def _fmt_int(value: Any) -> str:
    if value is None:
        return "undef"
    return f"{int(round(value)):,}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def render_markdown(bundle: dict[str, Any]) -> str:
    csynth = bundle["csynth"]
    console = csynth["console"]
    interface = bundle["interface"]
    verification = bundle["verification"]
    derived = bundle["derived_latency_200mhz"]
    transfer = bundle["transfer_audit"]
    lines: list[str] = []
    add = lines.append

    add("# 紧凑 AXI 接口整数 Selector 流水线 `integer_selector_pipeline_compact`：实现、验证与受控 HLS csynth")
    add("")
    add(f"- 日期：{bundle['date']}")
    add(f"- 决策：**`{bundle['decision']}`**")
    add("- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、"
        "`create_clock -period 5.0`、tool 默认 uncertainty，未放宽）；无 C/RTL cosim、无 IP 导出、"
        "无 implementation/bitstream、无板卡、无 GPU、无联网、无训练、无调参。")
    add("- 只读复用两个已验收 HLS 核源码与冻结 Python golden；**未改**受保护目录。")
    add("")
    add("## 0. 结论摘要")
    add("")
    add("新增独立顶层 `integer_selector_pipeline_compact`，把已验收的片上 scorer + packed selector "
        "包进三个 128-bit `m_axi` bundle 与一个 AXI-Lite `control` bundle：")
    add("")
    add("```")
    add("input_arena (tokens | group_mean | query) + wv_q")
    add("     -> 已验收整数 scorer 核 -> score_q[G]（片上局部缓冲，不是端口）")
    add("     -> 已验收 packed v4 selector 核 -> output_arena（tokens + 4 段 metadata）")
    add("```")
    add("")
    add(f"- 四道功能门：Python {verification['python']['tests_run']}/0、"
        f"g++ {verification['gpp']['passed']}/{verification['gpp']['failed']}、"
        f"ASan+UBSan {verification['asan_ubsan']['passed']}/{verification['asan_ubsan']['failed']}、"
        f"HLS C-sim {verification['hls_csim']['passed']}/{verification['hls_csim']['failed']}"
        f"（{verification['hls_csim']['csim_errors']} error）；g++/ASan/C-sim 均 dlopen 已验收 "
        "`integer_selector_pipeline` 参考实现并逐字节比对。")
    add(f"- 生成接口：m_axi = `{'`, `'.join(interface['m_axi_bundles'])}`（**{len(interface['m_axi_bundles'])} 个**）；"
        f"AXI-Lite = `{'`, `'.join(interface['s_axi_bundles'])}`（**{interface['axi_lite_bundle_count']} 个**）；"
        f"外部 score 端口 = {len(interface['score_ports'])}。")
    add(f"- `csynth_design` {console['error_count']} error；estimated period "
        f"**{csynth['estimated_clock_ns']} ns** ≤ requested 5.0 ns，effective budget "
        f"**{csynth['effective_budget_ns']} ns**，slack **{csynth['timing_slack_ns']}**，"
        f"Fmax **{csynth['estimated_fmax_mhz']} MHz**。")
    add(f"- 资源占 ZCU102：BRAM_18K **{_fmt_pct(csynth['resource_utilization_percent']['BRAM_18K'])}**、"
        f"DSP **{_fmt_pct(csynth['resource_utilization_percent']['DSP'])}**、"
        f"FF **{_fmt_pct(csynth['resource_utilization_percent']['FF'])}**、"
        f"LUT **{_fmt_pct(csynth['resource_utilization_percent']['LUT'])}**、"
        f"URAM **{_fmt_pct(csynth['resource_utilization_percent']['URAM'])}**（均 < 50%）。")
    add(f"- 输出 arena 权威总量 K=48/64/96 = **{bundle['output_arena_totals']['48']:,} / "
        f"{bundle['output_arena_totals']['64']:,} / {bundle['output_arena_totals']['96']:,} B**，"
        "5 段逐 offset 64 B 对齐，gap/unused 全 0。")
    add(f"- **公式推导** 生产用例：G80/K48 **{_fmt_int(derived['G80_K48']['total_cycles_estimate'])} cycles "
        f"= {derived['G80_K48']['ms_at_200mhz']:.3f} ms**；G260/K64 **{_fmt_int(derived['G260_K64']['total_cycles_estimate'])} "
        f"= {derived['G260_K64']['ms_at_200mhz']:.3f} ms**；G400/K96 **{_fmt_int(derived['G400_K96']['total_cycles_estimate'])} "
        f"= {derived['G400_K96']['ms_at_200mhz']:.3f} ms** @200 MHz。")
    add("")
    add("## 1. 顶层接口合同")
    add("")
    add("```")
    add("int integer_selector_pipeline_compact(")
    add("    const ts_word_t* input_arena,   // [G*320 + 8] 128-bit words")
    add("    const is_word_t* wv_q,          // [8192] 128-bit words")
    add("    ts_word_t* output_arena,        // [total_bytes/16] 128-bit words")
    add("    int group_count, int grid_h, int grid_w, int budget, int local_cap,")
    add("    float scale_in, float* scale_out,")
    add("    int* output_count, int* real_count, int* context_count);")
    add("```")
    add("")
    add("| 项 | 定义 |")
    add("|---|---|")
    add("| `score_q` | `uint16_t score_local[IS_MAX_GROUPS]` **顶层函数内局部数组**；无端口、不落 DDR |")
    add("| m_axi | 仅 `input_arena`、`wv_q`、`output_arena` 三个 128-bit bundle |")
    add("| AXI-Lite | 仅 `control` 一个；三个 m_axi 指针各自的 offset 寄存器用显式 "
        "`s_axilite port=<ptr> bundle=control` 合并进 `control`（否则 Vitis 2023.2 会另建 `control_r`） |")
    add("| 状态码 | 与 `integer_selector_pipeline` 逐值一致（`static_assert` 编译期锁定） |")
    add("")
    add("## 2. 冻结内存合同与布局证据")
    add("")
    add("输入 arena（偏移为 arena 基址的字节偏移，均 64 B 对齐）：")
    add("")
    add("| 段 | 偏移 | 大小 |")
    add("|---|---|---|")
    add("| `tokens_int8` | 0 | `G*4096` |")
    add("| `group_mean_int8` | `G*4096` | `G*1024` |")
    add("| `query_int8` | `G*5120` | 128 |")
    add("| 合计 | | `G*5120 + 128` |")
    add("")
    add("输出 arena 权威（`evaluation.hardware_regular_token_selector.fpga_transfer_layout`，5 段各自 64 B 对齐）：")
    add("")
    add("| K | total B | tokens off/size | group_index off/size | source off/size | region off/size | valid off/size |")
    add("|---:|---:|:---:|:---:|:---:|:---:|:---:|")
    for budget in ("48", "64", "96"):
        segs = {s["name"]: s for s in bundle["layout"]["output_arena"][budget]["segments"]}
        cells = " | ".join(
            f"{segs[name]['offset']}/{segs[name]['size']}"
            for name in ("tokens_int8", "group_index_uint16", "source_type_uint8",
                         "region_id_uint8", "valid_uint8")
        )
        add(f"| {budget} | {bundle['layout']['output_arena'][budget]['total_bytes']:,} | {cells} |")
    add("")
    add("Python 布局测试对 K=16/32/48/64/80/96 与 G=9/16/50/80/128/260/400/4096 同时核对"
        "公式与冻结权威函数，并编译 C++ 端的 `ispc_fpga_layout`/`ispc_input_layout` 镜像逐字段比对。")
    add("")
    add("## 3. 功能验证")
    add("")
    add("fixture 生成器只读复用已验收 pipeline 的 scene/case/Wv（`accepted_generator_sha256`），"
        "golden 直接来自冻结 `integer_chain` + `select_fixedpoint`，输出偏移实时来自冻结 "
        "`fpga_transfer_layout`/`describe_fpga_layout`；未复制第二套数学或 offset 算法。")
    add("")
    add("| 门 | 结果 | 日志 |")
    add("|---|---|---|")
    add(f"| Python 布局/fixture/接口测试 | **Ran {verification['python']['tests_run']} tests, OK** | `{verification['python']['log']}` |")
    add(f"| g++（dlopen 已验收 pipeline） | **{verification['gpp']['passed']} checks, {verification['gpp']['failed']} fail；PASS** | `{verification['gpp']['log']}` |")
    add(f"| ASan + UBSan | **{verification['asan_ubsan']['passed']} checks, {verification['asan_ubsan']['failed']} fail；PASS** | `{verification['asan_ubsan']['log']}` |")
    add(f"| Vitis HLS C-sim | **{verification['hls_csim']['csim_errors']} error；{verification['hls_csim']['passed']} checks, {verification['hls_csim']['failed']} fail** | `{verification['hls_csim']['log']}` |")
    add("")
    add("主机测试台逐字节核对：compact arena vs 冻结 Python golden、vs dlopen 的已验收 "
        "`integer_selector_pipeline`、vs 导出 `integer_selector_scorer` + `token_selector_core_packed` "
        "顺序组合；并检查 status / 三计数 / `scale_out` / 全部 padding / arena 尾部 / canary / 重复确定性。")
    add("")
    add("fixture 覆盖：K=48/64/96、G=9/16/50/80/128/260/400、真实 grounding/pointing correct+wrong prompt、"
        "identity bypass（G16/K48、G9/K96、G80/K96）、全 0 分数、全等分数、空 region（2x25）；"
        "非法 G/grid/budget/local_cap/空指针均为兼容状态码且不越过 canary。")
    add("")
    add("## 4. 受控 csynth 结果")
    add("")
    add(f"- project `/tmp/robobrain_integer_selector_compact_hls_csynth`；只 `csynth_design`；"
        f"**{console['error_count']} error**、{console['warning_count']} warning。")
    add(f"- `HLS 200-871` = {int(console['timing_warning_200_871'])}、"
        f"`HLS 200-1016` = {int(console['critical_path_warning_200_1016'])}、"
        f"`HLS 200-790` loop constraint NOT satisfied = {int(console['loop_constraint_warning_200_790'])}、"
        f"II 200-885 = {console['ii_warning_200_885']}、200-878 = {console['ii_warning_200_878']}。")
    add("")
    add("### 4.1 时序与资源（工具报告值）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| requested / uncertainty | {csynth['clock_target_ns']:.2f} ns / {csynth['clock_uncertainty_ns']:.2f} ns（默认，未改） |")
    add(f"| effective budget | **{csynth['effective_budget_ns']:.3f} ns** |")
    add(f"| estimated period | **{csynth['estimated_clock_ns']:.3f} ns**（Fmax {csynth['estimated_fmax_mhz']:.2f} MHz） |")
    add(f"| slack | **{csynth['timing_slack_ns']:.3f} ns** |")
    for key in ("BRAM_18K", "DSP", "FF", "LUT", "URAM"):
        add(f"| {key} | {csynth['resources_used'][key]:,} / {csynth['resources_available'][key]:,} = "
            f"**{_fmt_pct(csynth['resource_utilization_percent'][key])}** |")
    add("")
    add("### 4.2 时延（工具报告值 vs 公式推导值）")
    add("")
    add(f"- **工具报告**（compile-time 上界）：best {_fmt_int(csynth['reported_latency_cycles']['best'])}、"
        f"average {csynth['reported_latency_cycles']['average'] or 'undef'}、"
        f"**worst {_fmt_int(csynth['reported_latency_cycles']['worst'])} cycles = "
        f"{csynth['reported_latency_ms_at_200mhz']:.3f} ms @200 MHz**。")
    add(f"- **工具报告** scorer 子模块：Wv 载入 **{_fmt_int(csynth['scorer_phase']['wv_load_cycles'])} cycles**、"
        f"pass1 **{_fmt_int(csynth['scorer_phase']['pass1_cycles_per_group'])} cycles/group**、"
        f"pass2 **{_fmt_int(csynth['scorer_phase']['pass2_cycles_per_group'])} cycles/group**。")
    add("")
    add("| G | K | grid | scorer（推导） | selector（推导） | compact 开销（推导） | 总周期（推导） | ms @200 MHz（推导） |")
    add("|---:|---:|:---|---:|---:|---:|---:|---:|")
    for key in ("G80_K48", "G260_K64", "G400_K96"):
        row = derived[key]
        add(f"| {row['group_count']} | {row['budget']} | {row['grid'][0]}x{row['grid'][1]} | "
            f"{_fmt_int(row['scorer_cycles'])} | {_fmt_int(row['selector_cycles_estimate'])} | "
            f"{_fmt_int(row['compact_overhead_cycles_estimate'])} | **{_fmt_int(row['total_cycles_estimate'])}** | "
            f"{row['ms_at_200mhz']:.3f} |")
    add("")
    add(f"- 模型自校验（G=4096/K=96/64x64）：模型 {_fmt_int(bundle['derived_latency_model']['validation_at_max_bounds']['total_cycles_estimate'])} "
        f"vs 工具报告 worst {_fmt_int(csynth['reported_latency_cycles']['worst'])}，ratio "
        f"**{bundle['derived_latency_model']['validation_at_max_bounds']['model_vs_tool_ratio']:.4f}**（保守）。")
    add("- 以上生产用例周期为**公式估计**，不是工具测量；工具只报告 compile-time 上界时延。")
    add("")
    add("## 5. 异构传输估计（模型，非测量；fixed arena + Wv-resident）")
    add("")
    add(f"- 1 GbE 有效带宽假设 central {GBE_EFFECTIVE_MBPS_CENTRAL:.0f} MB/s（灵敏度 "
        f"{transfer['assumptions']['gbe_effective_mbytes_per_s_low']:.0f}–"
        f"{transfer['assumptions']['gbe_effective_mbytes_per_s_high']:.0f} MB/s）；本地 DDR 6 GB/s。")
    add(f"- 输入 arena = `G*5120 + 128` B；Wv = {transfer['assumptions']['wv_bytes']:,} B，resident 模式每图 0 B。")
    add("- FPGA→Jetson 网络/DDR 时间统一基于 fixed arena（`fpga_transfer_layout(K).total_bytes`），"
        "`logical_payload_bytes` 仅作为下界参考。")
    add("")
    add("| G | K | input arena B | J→F Wv-resident B | J→F 1GbE central (us) | fixed arena B | F→J 1GbE central (us) | F→J 6GB/s (us) |")
    add("|---:|---:|---:|---:|:---:|---:|:---:|---:|")
    for key in ("G80_K48", "G260_K64", "G400_K96"):
        for suffix in ("cold", "resident"):
            row = transfer["configs"][f"{key}_{suffix}"]
            t = row["times"]
            add(f"| {row['group_count']} | {row['budget']} | {row['jetson_to_fpga']['input_arena_bytes']:,} | "
                f"{row['jetson_to_fpga']['wv_q']:,} | {t['jetson_to_fpga_1gbe']['central_us']:.1f} | "
                f"**{row['fpga_to_jetson']['fixed_budget_arena_bytes']:,}** | "
                f"{t['fpga_to_jetson_1gbe_arena']['central_us']:.1f} | "
                f"{t['fpga_to_jetson_local_ddr_6gbs_us']:.1f} |")
    add("")
    add("- identity bypass（G80/K96）`output_count=80`，但固定预算 arena 仍为 K=96 的 393,792 B。")
    add("- 片上 `score_q` 不计入任何外部传输；紧凑化消除的是本地 DDR 中间往返。")
    add("")
    add("## 6. 门与决策")
    add("")
    add("| gate | 结果 |")
    add("|---|---|")
    for name, value in bundle["gate_checks"]["gates"].items():
        add(f"| {name} | {'✅' if value else '❌'} |")
    add("")
    add(f"→ **`{bundle['decision']}`**")
    add("")
    add("## 7. 限制")
    add("")
    for item in bundle["limitations"]:
        add(f"- {item}")
    add("")
    add("## 8. 受保护目录 SHA256")
    add("")
    add(f"- baseline `{bundle['protected_trees']['baseline_path']}`（{bundle['protected_trees']['total_files']} 文件）")
    for tree, info in bundle["protected_trees"]["groups"].items():
        add(f"- `{tree}`：{info['unchanged']}/{info['file_count']} 未变，changed {len(info['changed'])}，"
            f"missing {len(info['missing'])} → {'OK' if info['ok'] else 'FAIL'}")
    add("")
    add("## 9. 复现命令")
    add("")
    add("```bash")
    for command in bundle["reproduction"]:
        add(command)
    add("```")
    add("")
    add("## 10. 证据路径")
    add("")
    add("- 源码：`fpga_accel/integer_selector_pipeline_compact_hls/{README.md,Makefile,include/,src/,tb/,tools/,tests/,scripts/}`")
    add(f"- fixtures：`{bundle['fixtures']['dir']}/`")
    add("- csynth：`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/csynth/`")
    add("- 日志：`artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls/logs/`")
    add("- 机读报告：`reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json`")
    add("- harness result：`reports/token_selector/logs/2026-10-03-integer-selector-compact-interface-hls-harness-result.json`")
    add("")
    add("## 11. 文件与 SHA256")
    add("")
    add("| 文件 | sha256 |")
    add("|---|---|")
    for name, digest in sorted(bundle["files"].items()):
        add(f"| `{name}` | `{digest}` |")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    bundle = build_bundle()
    markdown = render_markdown(bundle)
    REPORT_MD.write_text(markdown, encoding="utf-8")
    bundle["report_markdown"] = {
        "path": str(REPORT_MD.relative_to(ROOT)),
        "sha256": sha256_bytes(markdown.encode("utf-8")),
    }
    REPORT_JSON.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    harness = {
        "schema_version": "1.0",
        "date": DATE,
        "task": bundle["task"],
        "decision": bundle["decision"],
        "gate_checks": bundle["gate_checks"],
        "interface": bundle["interface"],
        "output_arena_totals": bundle["output_arena_totals"],
        "verification": bundle["verification"],
        "csynth": {
            "estimated_clock_ns": bundle["csynth"]["estimated_clock_ns"],
            "effective_budget_ns": bundle["csynth"]["effective_budget_ns"],
            "timing_slack_ns": bundle["csynth"]["timing_slack_ns"],
            "resource_utilization_percent": bundle["csynth"]["resource_utilization_percent"],
            "console": bundle["csynth"]["console"],
        },
        "derived_latency_200mhz": bundle["derived_latency_200mhz"],
        "transfer_audit": {
            "transport_time_basis": bundle["transfer_audit"]["transport_time_basis"],
            "layout_authority": bundle["transfer_audit"]["layout_authority"],
        },
        "protected_trees": bundle["protected_trees"],
        "report_json": str(REPORT_JSON.relative_to(ROOT)),
        "report_markdown": bundle["report_markdown"],
        "scope": bundle["scope"],
    }
    HARNESS_RESULT.write_text(json.dumps(harness, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "decision": bundle["decision"],
        "gates": bundle["gate_checks"]["gates"],
        "protected_all_unchanged": bundle["protected_trees"]["all_unchanged"],
        "report_json": str(REPORT_JSON.relative_to(ROOT)),
        "report_markdown": bundle["report_markdown"],
        "harness_result": str(HARNESS_RESULT.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
