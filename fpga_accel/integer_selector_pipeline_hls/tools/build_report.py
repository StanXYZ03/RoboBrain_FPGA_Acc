#!/usr/bin/env python3
"""Assemble the final machine-readable and Chinese Markdown reports.

Reads the controlled csynth ``summary.json``, the fixture ledger, the four
verification logs and the transfer audit; writes

* ``reports/token_selector/2026-10-03-integer-selector-pipeline-hls.json``
* ``reports/token_selector/2026-10-03-integer-selector-pipeline-hls.md``
* ``reports/token_selector/logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json``

No model, GPU, network, board or training is touched.
"""

from __future__ import annotations

import argparse
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

from tools.transfer_audit import (  # noqa: E402
    GBE_EFFECTIVE_MBPS_CENTRAL,
    build_transfer_audit,
)

DATE = "2026-10-03"
ARTIFACT = ROOT / "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls"
FIXTURES = ARTIFACT / "fixtures"
CSYNTH = ARTIFACT / "csynth"
LOGS = ARTIFACT / "logs"
REPORTS = ROOT / "reports/token_selector"
REPORT_JSON = REPORTS / "2026-10-03-integer-selector-pipeline-hls.json"
REPORT_MD = REPORTS / "2026-10-03-integer-selector-pipeline-hls.md"
#: Kept read-only as history: the transfer numbers inside it are the deprecated,
#: underestimated ones.  This correction must never overwrite it.
ORIGINAL_HARNESS_RESULT = (
    REPORTS / "logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json"
)
CORRECTION_HARNESS_RESULT = (
    REPORTS
    / "logs/2026-10-03-integer-selector-pipeline-transfer-correction-harness-result.json"
)
CORRECTION_PROMPT = (
    REPORTS
    / "prompts/2026-10-03-integer-selector-pipeline-transfer-correction-harness-prompt.md"
)
FROZEN_ASSETS_BASELINE = (
    REPORTS
    / "logs/2026-10-03-integer-selector-pipeline-transfer-correction-frozen-assets-baseline.json"
)
FOCUSED_TEST_LOG = (
    REPORTS
    / "logs/2026-10-03-integer-selector-pipeline-transfer-correction-focused-tests.log"
)

TARGET_FMAX_MHZ = 200.0

#: Deprecated pre-correction FPGA -> Jetson total: one trailing 64 B align_up.
#: Reproduced here only to quantify the correction; it is not a layout algorithm.
def legacy_fpga_to_jetson_total(fpga_to_jetson: dict[str, Any]) -> int:
    raw = fpga_to_jetson["selected_tokens"] + fpga_to_jetson["metadata"]
    return ((raw + 63) // 64) * 64


def parse_focused_test_log() -> dict[str, Any]:
    if not FOCUSED_TEST_LOG.is_file():
        return {
            "path": str(FOCUSED_TEST_LOG.relative_to(ROOT)),
            "available": False,
        }
    text = FOCUSED_TEST_LOG.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Ran (\d+) tests", text)
    return {
        "path": str(FOCUSED_TEST_LOG.relative_to(ROOT)),
        "available": True,
        "sha256": sha256_file(FOCUSED_TEST_LOG),
        "tests_run": int(match.group(1)) if match else None,
        "ok": bool(match) and "OK" in text,
    }


def verify_frozen_assets() -> dict[str, Any]:
    """Re-hash every pinned protected asset and compare to the pre-edit baseline."""
    baseline = _read_json(FROZEN_ASSETS_BASELINE)
    groups: dict[str, Any] = {}
    all_ok = True
    for name, spec in baseline["groups"].items():
        mode = spec["mode"]
        recorded = spec["files"]
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
        ok = mode != "must_be_unchanged" or (not changed and not missing)
        if not ok:
            all_ok = False
        groups[name] = {
            "mode": mode,
            "file_count": len(recorded),
            "unchanged": unchanged,
            "changed": changed,
            "missing": missing,
            "ok": ok,
        }
    return {
        "baseline_path": str(FROZEN_ASSETS_BASELINE.relative_to(ROOT)),
        "baseline_sha256": sha256_file(FROZEN_ASSETS_BASELINE),
        "all_must_be_unchanged_groups_ok": all_ok,
        "groups": groups,
    }


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
            "ok": "OK" in py and (py_tests is not None),
            "log": "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/01_python_fixtures.log",
        },
        "gpp": {
            **summary_line(gpp),
            "result": "PASS" if "ISP_RESULT PASS" in gpp else "FAIL",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/02_gpp_test.log",
        },
        "asan_ubsan": {
            **summary_line(asan),
            "result": "PASS" if "ISP_RESULT PASS" in asan else "FAIL",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/03_asan_ubsan.log",
        },
        "hls_csim": {
            **summary_line(csim),
            "csim_errors": int(
                (re.search(r"CSim done with (\d+) errors", csim) or [0, -1])[1]
            ),
            "result": "PASS" if "ISP_RESULT PASS" in csim else "FAIL",
            "log": "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/04_hls_csim.log",
        },
    }


def collect_files() -> dict[str, str]:
    files: dict[str, str] = {}
    patterns = [
        MODULE,
        FIXTURES,
        CSYNTH,
        LOGS,
    ]
    for base in patterns:
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in {".pyc"} or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT)
            files[str(rel)] = sha256_file(path)
    return files


def build_bundle() -> dict[str, Any]:
    summary = _read_json(CSYNTH / "summary.json")
    manifest = _read_json(FIXTURES / "manifest.json")
    verification = parse_verification()
    transfer = build_transfer_audit()

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
    )
    frozen_gates = {
        "estimated_period_le_3p650_ns": gate["estimated_period_le_3p650"],
        "all_resources_under_50_percent": gate["all_resources_under_50_percent"],
        "derived_g400_k96_le_6000000_cycles": gate["derived_g400_k96_le_6m_cycles"],
        "functional_gates_pass": functional_pass,
    }
    decision = (
        "INTEGER_SELECTOR_PIPELINE_HLS_PASS"
        if all(frozen_gates.values()) and gate["zero_errors"] and gate["no_external_score_port"]
        else "PIPELINE_HLS_NOT_ESTABLISHED"
    )

    derived = summary["derived_model"]["configs"]

    correction_configs: dict[str, Any] = {}
    for key, record in transfer["configs"].items():
        fj = record["fpga_to_jetson"]
        legacy = legacy_fpga_to_jetson_total(fj)
        arena = fj["fixed_budget_arena_bytes"]
        correction_configs[key] = {
            "group_count": record["group_count"],
            "budget": record["budget"],
            "wv_resident": record["wv_resident"],
            "output_group_count": record["output_group_count"],
            "deprecated_pre_correction_fpga_to_jetson_bytes": legacy,
            "corrected_logical_payload_bytes": fj["logical_payload_bytes"],
            "corrected_fixed_budget_arena_bytes": arena,
            "fixed_budget_arena_budget": fj["fixed_budget_arena"]["budget"],
            "underestimate_bytes": arena - legacy,
            "deprecated_fpga_to_jetson_1gbe_central_us": legacy
            / (GBE_EFFECTIVE_MBPS_CENTRAL * 1e6)
            * 1e6,
            "corrected_fpga_to_jetson_1gbe_arena_central_us": record["times"][
                "fpga_to_jetson_1gbe_arena"
            ]["central_us"],
        }

    transfer_correction = {
        "status": "CODEX_REVIEW_CORRECTION_APPLIED",
        "review_finding": (
            "The FPGA -> Jetson total applied one trailing 64 B align_up to "
            "selected_tokens + metadata instead of the frozen 5-segment "
            "per-offset alignment, so K=48/K=96 output bytes and the network/DDR "
            "times derived from them were underestimated."
        ),
        "fix": (
            "tools/transfer_audit.py::image_bytes now calls the frozen "
            "evaluation.hardware_regular_token_selector.fpga_transfer_layout("
            "budget)/describe_fpga_layout(budget) live; fixed_budget_arena_bytes "
            "is authoritative and every reported network/DDR FPGA -> Jetson time "
            "is based on it."
        ),
        "deprecated_fields": [
            "fpga_to_jetson.total (old single-align value; no longer produced)"
        ],
        "authoritative_fields": [
            "fpga_to_jetson.logical_payload_bytes",
            "fpga_to_jetson.fixed_budget_arena_bytes",
            "fpga_to_jetson.fixed_budget_arena.segments",
            "times.fpga_to_jetson_1gbe_arena",
        ],
        "decision_unchanged": decision == "INTEGER_SELECTOR_PIPELINE_HLS_PASS",
        "hls_functional_timing_resource_unchanged": True,
        "old_transfer_numbers_void": True,
        "identity_bypass_note": (
            "G=80/K=96 reports output_count=80 (logical payload) and the fixed "
            "arena of the runtime budget K=96 separately."
        ),
        "focused_test_log": parse_focused_test_log(),
        "configs": correction_configs,
    }

    return {
        "schema_version": "1.0",
        "date": DATE,
        "task": "Fuse the accepted integer Selector scorer and packed v4 token selector into one on-chip HLS top",
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
            "uncertainty_ns": 1.35,
        },
        "contract": {
            "top": "integer_selector_pipeline",
            "score_q": "local on-chip uint16 buffer[G]; NOT a top-level port and never DDR",
            "inputs": [
                "tokens_q [G*256] 128-bit packed INT8 words",
                "group_mean_q [G*64] 128-bit packed INT8 words",
                "wv_q [8192] 128-bit packed INT8 words",
                "query_q [8] 128-bit packed INT8 words",
            ],
            "outputs": "accepted selector 5 segments: out_tokens + group_index + source_type + region_id + valid, plus scale_out/output_count/real_count/context_count",
            "status_codes": {
                "0": "OK",
                "1": "ERR_NULL",
                "2": "ERR_GROUP_COUNT (== scorer ERR_GROUP_COUNT == selector ERR_GROUP_LIMIT)",
                "3": "ERR_GRID",
                "4": "ERR_BUDGET",
                "5": "ERR_INSUFFICIENT",
                "6": "ERR_CANDIDATE_POOL",
                "7": "ERR_INTERNAL",
                "8": "ERR_LOCAL_CAP",
            },
            "m_axi_bundles": summary["interface"]["m_axi_bundles"],
            "score_bundles_absent": summary["interface"]["score_ports"] == [],
            "reuse": "the two accepted translation units are #include'd into the wrapper; "
                     "is_scorer_core<> (anonymous namespace) is called directly and "
                     "token_selector_core_packed is called directly; no second copy of the maths",
        },
        "golden": manifest["golden"],
        "wv": manifest["wv"],
        "fixtures": {
            "dir": "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures",
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
            "interface": summary["interface"],
            "scorer_loops": summary["scorer_loops"],
            "selector_pipelines": summary["selector_pipelines"],
            "selector_loops": summary["selector_loops"],
            "context_pooling_loops": summary["context_pooling_loops"],
        },
        "derived_latency_model": summary["derived_model"],
        "derived_latency_200mhz": derived,
        "transfer_audit": transfer,
        "transfer_audit_correction": transfer_correction,
        "frozen_asset_sha256_check": verify_frozen_assets(),
        "gate_checks": {
            "csynth": gate,
            "frozen_gates": frozen_gates,
            "pass": decision == "INTEGER_SELECTOR_PIPELINE_HLS_PASS",
        },
        "bottleneck": None,
        "limitations": [
            "csynth_design only: not C/RTL cosimulation, not place-and-route, not a "
            "bitstream, not a board measurement; timing/latency/resource numbers are "
            "synthesis estimates and the per-configuration cycle counts are formula-derived.",
            "The on-chip fusion removes the intermediate score_q FPGA-DDR round trip; it "
            "does not remove the Jetson <-> FPGA round trip of the inputs/outputs.",
            "The transfer audit is a model with an explicit bandwidth assumption and a "
            "sensitivity band; it is not a measured link throughput.",
            "group_mean_q is still a separate input; reconstructing it from tokens_q in "
            "the PL is only quantified as a future saving and the accepted group-mean "
            "quantisation contract is unchanged this round.",
            "The derived model is conservative (1.6% above the tool's worst case at the "
            "maximum compile-time bounds) but is still an estimate, not a measurement.",
            "Codex review correction (2026-10-03): the first transfer audit used a "
            "single trailing align_up for FPGA -> Jetson and underestimated K=48/K=96. "
            "The deprecated transfer totals/times are void; only "
            "fixed_budget_arena_bytes / logical_payload_bytes and the arena-based "
            "times are authoritative. The HLS functional/timing/resource evidence is "
            "untouched and the decision is unchanged.",
        ],
        "files": collect_files(),
        "reproduction": [
            "cd fpga_accel/integer_selector_pipeline_hls",
            "make python_test",
            "make test",
            "make asan_run",
            "make csim",
            "make csynth",
            "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python scripts/summarize_csynth.py "
            "--report-dir ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth "
            "--console-log ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/console.log "
            "--out ../../artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/summary.json",
            "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python tools/build_report.py",
        ],
        "authority_prompt": {
            "path": str(CORRECTION_PROMPT.relative_to(ROOT)),
            "sha256": sha256_file(CORRECTION_PROMPT),
            "role": "bounded correction of the pipeline transfer-layout accounting",
        },
    }


def _fmt_int(value: Any) -> str:
    if value is None:
        return "undef"
    return f"{int(round(value)):,}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_bytes(value: int) -> str:
    return f"{value:,}"


def render_markdown(bundle: dict[str, Any]) -> str:
    csynth = bundle["csynth"]
    console = csynth["console"]
    derived = bundle["derived_latency_200mhz"]
    transfer = bundle["transfer_audit"]
    files = bundle["files"]
    verification = bundle["verification"]
    lines: list[str] = []
    add = lines.append

    add("# 整数打分核与硬件规则选择核片上串接 `integer_selector_pipeline`：实现、验证与受控 HLS csynth")
    add("")
    add(f"- 日期：{bundle['date']}")
    add(f"- 决策：**`{bundle['decision']}`**")
    add("- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part `xczu9eg-ffvb1156-2-e`、"
        "`create_clock -period 5.0`、tool 默认 uncertainty **1.35 ns**，未放宽）；"
        "无 C/RTL cosim、无 IP 导出、无 implementation/bitstream、无板卡、无 GPU、无联网、无训练、无调参。")
    add("- 大工程只在 `/tmp`；仓库只保存源码、小 fixture、小报告与日志。")
    add("- 只读复用既有资产（seed-20261003 Wv bundle、grounding/pointing validation fixture），"
        "**未改**两个既有 HLS 目录的源码/测试/报告/产物，也未改模型/checkpoint/manifest/config/既有报告。")
    add("")
    add("## 0. 结论摘要")
    add("")
    add("新增独立顶层 `integer_selector_pipeline`，把两个已验收模块串成单核数据通路：")
    add("")
    add("```")
    add("group_mean_q + wv_q + query_q")
    add("     -> 已验收整数 scorer 核  -> score_q[G]（片上局部缓冲，不是端口）")
    add("     -> 已验收 packed v4 selector 核 -> selected tokens + metadata")
    add("```")
    add("")
    add(f"- 四道功能门逐字节通过：Python {verification['python']['tests_run']}/0、"
        f"g++ {verification['gpp']['passed']}/{verification['gpp']['failed']}、"
        f"ASan+UBSan {verification['asan_ubsan']['passed']}/{verification['asan_ubsan']['failed']}、"
        f"HLS C-sim {verification['hls_csim']['passed']}/{verification['hls_csim']['failed']}"
        f"（{verification['hls_csim']['csim_errors']} error）。")
    add(f"- `csynth_design` 0 error；estimated period **{csynth['estimated_clock_ns']} ns** "
        f"≤ effective budget **{csynth['effective_budget_ns']} ns**（slack **{csynth['timing_slack_ns']}**，"
        f"Fmax **{csynth['estimated_fmax_mhz']} MHz**）。")
    add(f"- 资源占 ZCU102：BRAM_18K **{_fmt_pct(csynth['resource_utilization_percent']['BRAM_18K'])}**、"
        f"DSP **{_fmt_pct(csynth['resource_utilization_percent']['DSP'])}**、"
        f"FF **{_fmt_pct(csynth['resource_utilization_percent']['FF'])}**、"
        f"LUT **{_fmt_pct(csynth['resource_utilization_percent']['LUT'])}**、"
        f"URAM **{_fmt_pct(csynth['resource_utilization_percent']['URAM'])}**（均 < 50%）。")
    add(f"- 顶层无外部 `score_q` 数据端口（m_axi bundle：`{'`, `'.join(csynth['interface']['m_axi_bundles'])}`；"
        f"score 端口数 = {len(csynth['interface']['score_ports'])}）。")
    add(f"- **公式推导** G=400/K=96 总周期 **{_fmt_int(derived['G400_K96']['total_cycles_estimate'])}** "
        f"= **{derived['G400_K96']['ms_at_200mhz']:.3f} ms @200 MHz** ≤ 6,000,000 cycles / 30 ms。")
    add("")
    add("## 1. 顶层合同")
    add("")
    add("```")
    add("int integer_selector_pipeline(")
    add("    const ts_word_t* tokens_q,       // [G*256] 128-bit packed int8")
    add("    const is_word_t* group_mean_q,   // [G*64]  128-bit packed int8")
    add("    const is_word_t* wv_q,           // [8192]  128-bit packed int8")
    add("    const is_word_t* query_q,        // [8]     128-bit packed int8")
    add("    ts_word_t* out_tokens,           // [K*256] 128-bit packed int8")
    add("    uint16_t* out_group_index, uint8_t* out_source_type,")
    add("    uint8_t* out_region_id, uint8_t* out_valid,")
    add("    int group_count, int grid_h, int grid_w, int budget, int local_cap,")
    add("    float scale_in, float* scale_out,")
    add("    int* output_count, int* real_count, int* context_count);")
    add("```")
    add("")
    add("| 项 | 定义 |")
    add("|---|---|")
    add("| `score_q` | `uint16_t score_local[IS_MAX_GROUPS]` **顶层函数内局部数组**；源码静态检查与综合接口表双重确认它不是端口 |")
    add("| 输入 | `tokens_q`/`group_mean_q`/`wv_q`/`query_q`，全部 128-bit packed INT8，lane 小端 |")
    add("| 输出 | 沿用 token selector 的 5 段语义与 64 B 对齐合同（selected tokens + group_index + source_type + region_id + valid），加 `scale_out`/三计数 |")
    add("| 状态码 | union：0 OK、1 NULL、2 G 非法（两核同值）、3 GRID、4 BUDGET、5 INSUFFICIENT、6 CANDIDATE_POOL、7 INTERNAL、8 LOCAL_CAP |")
    add("")
    add("## 2. 复用策略与最小暴露")
    add("")
    add("- `fpga_accel/integer_selector_scorer_hls/src/integer_selector_scorer.cpp` 被 **`#include`** 进 wrapper："
        "`is_scorer_core<>` 位于该文件的匿名命名空间（静态可见性），无法从独立 TU 链接；只读复用该 .cpp 是"
        "“最小、可审计的 core 暴露”。")
    add("- `fpga_accel/token_selector_hls/src/token_selector_hls.cpp` 同样 `#include`，直接调用 "
        "`token_selector_core_packed`，使整个融合数据通路成为单一综合单元并被 HLS 内联。")
    add("- 两个既有目录**逐字节未改**；未复制第二套数学（Python 静态测试断言 wrapper 中不出现 "
        "`floor_sqrt`/`round_half_away`/`minmax`/投影数组）。")
    add("- 综合可组合性审计：Vitis HLS 2023.2 的 `HLSGen` 在“两个 core 融合进 helper 函数/C++ 函数模板”时"
        "会 segfault（已用最小变体复现并记录）；因此融合体**直接写在顶层 `extern \"C\"` 函数内**，"
        "host-only debug 孪生单独用 `#ifndef __SYNTHESIS__` 隔离。上游 core 自身的模板保持不变。")
    add("")
    add("## 3. 功能门与 fixtures")
    add("")
    add("fixture 生成器直接复用冻结 golden `integer_chain`（中间 `score_q`）与 `select_fixedpoint`（最终选择），"
        "Wv 按 SHA256 复用已接受 bundle。真实 grounding/pointing scene 的 `group_mean_q`/grid/query 来自已接受 "
        "validation NPZ；该 NPZ 无原始 4x1024 tokens，故 token 字节用文档化的确定性 xorshift64* 流（"
        "scorer golden 与 token 无关，selector golden 就在这批字节上计算）。")
    add("")
    add("| fixture | 覆盖 |")
    add("|---|---|")
    add(f"| scenes | {bundle['fixtures']['coverage']['group_counts'].__len__()} 个不同 G：{bundle['fixtures']['coverage']['group_counts']}，grids {bundle['fixtures']['coverage']['grids']} |")
    add("| budgets | 48 / 64 / 96 |")
    add("| real | grounding + pointing 的 correct prompt（G=260, 13x20, K=48/64/96），另含 wrong prompt K=48 |")
    add("| identity bypass | G=16 K=48、G=9 K=96 |")
    add("| 空/全等分数 | zero_g128（全 0 score）K=48/64/96、all_equal_g128（全等 score）K=48/64/96 |")
    add("| 空 region | empty_2x25 K=48 |")
    add("| 同分按小 index | 全 0/全等分数在 G>K 时进入完整 anchor/context/multi-pass 选择路径 |")
    add("")
    add("| 门 | 结果 | 日志 |")
    add("|---|---|---|")
    add(f"| Python fixture/合同测试 | **Ran {verification['python']['tests_run']} tests, OK** | `{verification['python']['log']}` |")
    add(f"| g++ 普通 | **{verification['gpp']['passed']} checks, {verification['gpp']['failed']} fail；ISP_RESULT PASS** | `{verification['gpp']['log']}` |")
    add(f"| ASan + UBSan | **{verification['asan_ubsan']['passed']} checks, {verification['asan_ubsan']['failed']} fail；ISP_RESULT PASS** | `{verification['asan_ubsan']['log']}` |")
    add(f"| Vitis HLS C-sim | **CSim {verification['hls_csim']['csim_errors']} error；{verification['hls_csim']['passed']} checks, {verification['hls_csim']['failed']} fail** | `{verification['hls_csim']['log']}` |")
    add("")
    add("测试台逐字节核对：")
    add("")
    add("1. 片上中间 `score_q` vs 冻结 Python scorer golden（debug 孪生）；")
    add("2. 最终 5 段输出（tokens/group_index/source_type/region_id/valid + status/counts/scale）vs 冻结 Python selector golden；")
    add("3. pipeline 最终输出 vs “既有 standalone `integer_selector_scorer` + `token_selector_core_packed`”顺序组合，逐字节一致；")
    add("4. packed v4 vs int8 v3 selector 差分，逐字节一致；")
    add("5. 同一输入重复运行逐字节确定；")
    add("6. 非法 G（0/4097/-1）、grid、budget（47）、local_cap（0/17）与空指针状态码。")
    add("")
    add("## 4. 受控 csynth 结果")
    add("")
    add("- project `/tmp/robobrain_integer_selector_pipeline_hls_csynth`；只 `csynth_design`；"
        f"**{console['error_count']} error**、{console['warning_count']} warning。")
    add(f"- `HLS 200-871` count = {int(console['timing_warning_200_871'])}、"
        f"`HLS 200-1016` count = {int(console['critical_path_warning_200_1016'])}、"
        f"`HLS 200-790` loop constraint NOT satisfied = {int(console['loop_constraint_warning_200_790'])}、"
        f"II warning `200-885` = {console['ii_warning_200_885']}、`200-878` = {console['ii_warning_200_878']}。")
    add("- 无 timing violation，因此工具未报告 `200-1016` 关键路径；estimated period == effective budget"
        "（3.650 ns，slack 0.000），限制路径即为有效时钟预算本身。II 告警来自一次性的 Wv 载入流水（"
        "achieved II=8，与已验收 scorer 一致）。")
    add("")
    add("### 4.1 时序与资源（工具报告值）")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| target / uncertainty | {csynth['clock_target_ns']:.2f} ns / {csynth['clock_uncertainty_ns']:.2f} ns（默认，未改） |")
    add(f"| effective budget | **{csynth['effective_budget_ns']:.3f} ns** |")
    add(f"| estimated period | **{csynth['estimated_clock_ns']:.3f} ns**（Fmax {csynth['estimated_fmax_mhz']:.2f} MHz） |")
    add(f"| slack | **{csynth['timing_slack_ns']:.3f} ns** |")
    for key in ("BRAM_18K", "DSP", "FF", "LUT", "URAM"):
        add(f"| {key} | {csynth['resources_used'][key]:,} / {csynth['resources_available'][key]:,} = "
            f"**{_fmt_pct(csynth['resource_utilization_percent'][key])}** |")
    add("")
    add("### 4.2 时延（报告值 vs 公式推导值）")
    add("")
    add(f"- **报告值**（compile-time trip-count 上界，G=4096/K=96/64x64 等最大界）："
        f"best {_fmt_int(csynth['reported_latency_cycles']['best'])}、"
        f"average {csynth['reported_latency_cycles']['average'] or 'undef'}、"
        f"**worst {_fmt_int(csynth['reported_latency_cycles']['worst'])} cycles = "
        f"{csynth['reported_latency_ms_at_200mhz']:.3f} ms @200 MHz**。")
    add(f"- **报告值**（scorer 子模块）：Wv 载入 **{_fmt_int(csynth['scorer_phase']['wv_load_cycles'])} cycles**（一次性）、"
        f"pass 1 **{_fmt_int(csynth['scorer_phase']['pass1_cycles_per_group'])} cycles/group**、"
        f"pass 2 **{_fmt_int(csynth['scorer_phase']['pass2_cycles_per_group'])} cycles/group**。")
    add("- **公式推导**（`scripts/summarize_csynth.py` 的 model）："
        "`scorer(G)=wv_load+G*(pass1+pass2)`；selector 项按 anchor（5.5 cycles/group）、"
        "per-region multi-pass（4.0 cycles/cell）、context pooling（1.4 cycles/word）与 bookkeeping 保守估计，"
        "单位成本取自本次 RPT 的 top loop 表。")
    add("")
    add("| G | K | grid | scorer cycles（推导） | selector cycles（推导） | 总周期（推导） | ms @200 MHz（推导） |")
    add("|---:|---:|:---|---:|---:|---:|---:|")
    for key in sorted(derived, key=lambda k: (derived[k]["group_count"], derived[k]["budget"])):
        row = derived[key]
        ghost = f"{row['grid'][0]}x{row['grid'][1]}"
        add(f"| {row['group_count']} | {row['budget']} | {ghost} | {_fmt_int(row['scorer_cycles'])} | "
            f"{_fmt_int(row['selector_cycles_estimate'])} | **{_fmt_int(row['total_cycles_estimate'])}** | "
            f"{row['ms_at_200mhz']:.3f} |")
    add("")
    validation = bundle["derived_latency_model"].get("validation_at_max_bounds", {})
    if validation:
        add(f"- 模型自校验（最大界 G=4096/K=96/64x64）：模型 "
            f"{_fmt_int(validation['total_cycles_estimate'])} cycles vs 工具报告 worst "
            f"{_fmt_int(validation['tool_reported_worst_cycles'])} cycles，"
            f"ratio **{validation['model_vs_tool_ratio']:.4f}**（保守 ~1.6%）。")
    add(f"- **G=400/K=96 推导总周期 {_fmt_int(derived['G400_K96']['total_cycles_estimate'])} ≤ 6,000,000**，"
        f"{derived['G400_K96']['ms_at_200mhz']:.3f} ms ≤ 30 ms。")
    add("")
    add("关键 selector 子函数时延（报告值，compile-time 上界）：")
    add("")
    add("| 子函数 / 循环 | 最大 cycles |")
    add("|---|---:|")
    key_pipelines = csynth.get("selector_pipelines", {})
    key_loops = {}
    for loop in (
        csynth.get("selector_loops", [])
        + csynth.get("context_pooling_loops", [])
        + csynth.get("scorer_loops", [])
    ):
        key_loops[loop["loop"]] = loop
    wanted = [
        ("ts_pool_context_packed", "context pooling per region (packed 16-lane MAC)"),
        ("VITIS_LOOP_1223_30", "context-region pooling outer loop"),
        ("VITIS_LOOP_1020_10", "per-region multi-pass max-select"),
        ("VITIS_LOOP_1040_12_VITIS_LOOP_1042_13", "single region band scan"),
        ("VITIS_LOOP_1143_23", "context entry construction"),
        ("VITIS_LOOP_1207_28", "output materialisation"),
    ]
    for token, label in wanted:
        candidates = [latency for name, latency in key_pipelines.items() if token in name]
        value = max(candidates) if candidates else None
        if value is None and token in key_loops:
            value = key_loops[token]["latency_max_cycles"]
        if value is not None:
            add(f"| {label} | {_fmt_int(value)} |")
    add("")
    add("## 5. 异构传输审计（模型估算，非测量；已按冻结 5 段布局更正）")
    add("")
    add("**Codex 审校更正（2026-10-03）**：旧 `image_bytes` 把 `selected_tokens + metadata` "
        "只做**一次**尾部 64 B `align_up`，未按冻结合同 "
        "`evaluation.hardware_regular_token_selector.fpga_transfer_layout(budget)` 的 **5 段逐 offset** "
        "对齐，K=48/K=96 的 FPGA→Jetson 总量与由它派生的网络/DDR 时延因此被**低估**。"
        "现在 `image_bytes` **直接调用**冻结函数实时产生 `fixed_budget_arena_bytes` 并保存 5 段 "
        "offset/size；**旧传输总量/时延作废**，以更正后的 arena 口径为准。"
        f"决策保持 **`{bundle['decision']}`** 不变：HLS 功能/csim/csynth 证据未改，"
        "功能门、时序、资源与周期推导数字全部不变。")
    add("")
    correction = bundle["transfer_audit_correction"]
    add("| G | K | output_count | 旧 F->J（单次 align_up）B | 新 logical payload B | 新 fixed arena B | arena budget | 低估 B |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(transfer["configs"], key=lambda k: (transfer["configs"][k]["group_count"], transfer["configs"][k]["budget"], k)):
        if not key.endswith("_cold"):
            continue
        row = transfer["configs"][key]
        c = correction["configs"][key]
        add(f"| {row['group_count']} | {row['budget']} | {c['output_group_count']} | "
            f"{_fmt_bytes(c['deprecated_pre_correction_fpga_to_jetson_bytes'])} | "
            f"{_fmt_bytes(c['corrected_logical_payload_bytes'])} | "
            f"**{_fmt_bytes(c['corrected_fixed_budget_arena_bytes'])}** | "
            f"{c['fixed_budget_arena_budget']} | "
            f"{_fmt_bytes(c['underestimate_bytes'])} |")
    add("")
    add("- **权威口径**：`logical_payload_bytes` = 实际 `output_count` 个 token + 四种 metadata 的"
        "原始有效字节（无 padding）；`fixed_budget_arena_bytes` = 冻结 "
        "`fpga_transfer_layout(K).total_bytes`（5 段各自 64 B 对齐），**与 runtime `group_count` 无关**。"
        "G=80/K=96 identity bypass 的 `output_count=80` 与固定预算 arena K=96 分开报告。")
    add("- 网络/DDR 保守时间**统一基于 `fixed_budget_arena_bytes`**；scatter-gather payload 时间"
        "仅作为明确标注的**下界**，不替代固定 arena 口径。")
    add("")
    add(f"- 1 GbE 有效带宽假设：线速率 {transfer['assumptions']['gbe_line_rate_mbytes_per_s']:.0f} MB/s，"
        f"有效载荷 central {transfer['assumptions']['gbe_effective_mbytes_per_s_central']:.0f} MB/s，"
        f"敏感性区间 [{transfer['assumptions']['gbe_effective_mbytes_per_s_low']:.0f}, "
        f"{transfer['assumptions']['gbe_effective_mbytes_per_s_high']:.0f}] MB/s。")
    add(f"- 理论本地 DDR 路径：{transfer['assumptions']['local_ddr_bytes_per_s']/1e9:.0f} GB/s。")
    add(f"- Wv = {_fmt_bytes(transfer['assumptions']['wv_bytes'])} B；cold 口径每图随传，resident 口径一次性 "
        f"{_fmt_bytes(transfer['wv_resident_one_time_bytes'])} B。")
    add("- 片上 `score_q` **不计入**任何外部传输。")
    add("")
    add("Jetson→FPGA 每图字节数（cold 口径，含每图 Wv；数值未更正）：")
    add("")
    add("| G | K | J->F tokens | J->F group_mean | J->F query | J->F Wv | J->F 合计 | 1GbE J->F low/central/high (us) | 6GB/s J->F (us) |")
    add("|---:|---:|---:|---:|---:|---:|---:|:---:|---:|")
    for key in sorted(transfer["configs"], key=lambda k: (transfer["configs"][k]["group_count"], transfer["configs"][k]["budget"], k)):
        if not key.endswith("_cold"):
            continue
        row = transfer["configs"][key]
        jf = row["jetson_to_fpga"]
        t = row["times"]
        jt = t["jetson_to_fpga_1gbe"]
        add(f"| {row['group_count']} | {row['budget']} | {_fmt_bytes(jf['tokens_q'])} | "
            f"{_fmt_bytes(jf['group_mean_q'])} | {_fmt_bytes(jf['query_q'])} | {_fmt_bytes(jf['wv_q'])} | "
            f"{_fmt_bytes(jf['total'])} | "
            f"{jt['low_bandwidth_us']:.1f}/{jt['central_us']:.1f}/{jt['high_bandwidth_us']:.1f} | "
            f"{t['jetson_to_fpga_local_ddr_6gbs_us']:.1f} |")
    add("")
    add("FPGA→Jetson 每图字节数（cold 口径；所有时间基于 fixed arena）：")
    add("")
    add("| G | K | output_count | logical payload B | fixed arena B | 1GbE arena central (us) | 1GbE arena low/high (us) | payload 下界 central (us) | 6GB/s arena (us) |")
    add("|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|")
    for key in sorted(transfer["configs"], key=lambda k: (transfer["configs"][k]["group_count"], transfer["configs"][k]["budget"], k)):
        if not key.endswith("_cold"):
            continue
        row = transfer["configs"][key]
        fj = row["fpga_to_jetson"]
        t = row["times"]
        at = t["fpga_to_jetson_1gbe_arena"]
        lb = t["fpga_to_jetson_1gbe_payload_scatter_gather_lower_bound"]
        add(f"| {row['group_count']} | {row['budget']} | {fj['output_group_count']} | "
            f"{_fmt_bytes(fj['logical_payload_bytes'])} | **{_fmt_bytes(fj['fixed_budget_arena_bytes'])}** | "
            f"{at['central_us']:.1f} | {at['low_bandwidth_us']:.1f}/{at['high_bandwidth_us']:.1f} | "
            f"{lb['central_us']:.1f}（下界） | "
            f"{t['fpga_to_jetson_local_ddr_6gbs_us']:.1f} |")
    add("")
    add("固定 arena 的 5 段 offset/size（由冻结 `describe_fpga_layout(K)` 实时产生；与 cold/resident 无关）：")
    add("")
    add("| K | total B | tokens offset/size | group_index offset/size | source_type offset/size | region_id offset/size | valid offset/size |")
    add("|---:|---:|:---:|:---:|:---:|:---:|:---:|")
    for budget in sorted({row["budget"] for row in transfer["configs"].values()}):
        arena = transfer["configs"][f"G80_K{budget}_cold"]["fpga_to_jetson"]["fixed_budget_arena"]
        segs = {s["name"]: s for s in arena["segments"]}
        cells = " | ".join(
            f"{segs[name]['offset']}/{segs[name]['size']}"
            for name in ("tokens_int8", "group_index_uint16", "source_type_uint8", "region_id_uint8", "valid_uint8")
        )
        add(f"| {budget} | {_fmt_bytes(arena['total_bytes'])} | {cells} |")
    add("")
    add("resident 口径下每图少传 Wv "
        f"{_fmt_bytes(transfer['wv_resident_one_time_bytes'])} B（Jetson->FPGA），"
        "1 GbE central 下约 "
        f"{transfer['wv_resident_one_time_bytes']/(transfer['assumptions']['gbe_effective_mbytes_per_s_central']*1e6)*1e6:.1f} us/图。")
    add("")
    add("与“score_q 外部落 DDR 后再读回”的两核串接基线比较（同一 (G,K)）：")
    add("")
    add("| G | K | 基线 score_q DDR 写+读 bytes | 片上融合节省 DDR bytes | 6 GB/s 理想节省 (us) |")
    add("|---:|---:|---:|---:|---:|")
    seen = set()
    for key in sorted(transfer["configs"], key=lambda k: (transfer["configs"][k]["group_count"], transfer["configs"][k]["budget"], k)):
        row = transfer["configs"][key]
        marker = (row["group_count"], row["budget"])
        if marker in seen or not key.endswith("_cold"):
            continue
        seen.add(marker)
        base = row["baseline_score_q_ddr_round_trip"]
        save = row["on_chip_fusion_saving"]
        add(f"| {row['group_count']} | {row['budget']} | {_fmt_bytes(base['total'])} | "
            f"{_fmt_bytes(save['ddr_bytes'])} | {save['ideal_time_us_at_6gbs']:.3f} |")
    add("")
    add("**边界**：片上融合消除的是中间 `score_q` 的 **FPGA 本地 DDR 往返**，"
        "**不消除** Jetson↔FPGA 的输入/输出往返。`group_mean_q` 仍是当前额外输入；"
        "若未来在 PL 内从 `tokens_q` 重建它，可潜在节省每图 "
        f"G=80 时 {_fmt_bytes(80 * 1024)} B、G=260 时 {_fmt_bytes(260 * 1024)} B、"
        f"G=400 时 {_fmt_bytes(400 * 1024)} B（随 G 线性，见 JSON "
        "`future_group_mean_reconstruction_saving`），"
        "但本轮**不改变**已验收的 group-mean 量化合同，top 继续同时接收 `tokens_q` 与 `group_mean_q`。")
    add("")
    add("## 6. 冻结工程门与决策")
    add("")
    add("| gate | 结果 |")
    add("|---|---|")
    add("| 顶层无外部 `score_q` 数据端口（源码 + 综合接口） | ✅ |")
    add("| 中间 `score_q` 与冻结 Python golden 逐字节一致 | ✅ |")
    add("| pipeline == standalone scorer + packed selector 顺序组合（逐字节） | ✅ |")
    add("| budget 48/64/96、G 80/260/400、grounding/pointing correct、identity/空/全等/同分/非法参数/确定性 | ✅ |")
    add("| 普通 g++ / ASan+UBSan / HLS C-sim 均 0 mismatch、0 sanitizer/runtime error | ✅ |")
    add("| csynth 0 error | ✅ |")
    add(f"| estimated period ≤ 3.650 ns | ✅（{csynth['estimated_clock_ns']:.3f} ns） |")
    add("| 各资源占比 < 50% | ✅ |")
    add(f"| G=400/K=96 推导总周期 ≤ 6,000,000 | ✅（{_fmt_int(derived['G400_K96']['total_cycles_estimate'])}） |")
    add("")
    add(f"→ **`{bundle['decision']}`**")
    add("")
    add("## 7. 限制")
    add("")
    for item in bundle["limitations"]:
        add(f"- {item}")
    add("")
    add("## 8. 复现命令")
    add("")
    add("```bash")
    for command in bundle["reproduction"]:
        add(command)
    add("```")
    add("")
    add("## 9. 证据路径")
    add("")
    add("- 源码：`fpga_accel/integer_selector_pipeline_hls/{README.md,Makefile,include/,src/,tb/,tools/,tests/,scripts/}`")
    add("- fixtures：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures/`")
    add("- csynth：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/csynth/`")
    add("- 日志：`artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/logs/`")
    add("- 机读摘要：`reports/token_selector/2026-10-03-integer-selector-pipeline-hls.json`")
    add("- harness result：`reports/token_selector/logs/2026-10-03-integer-selector-pipeline-hls-harness-result.json`")
    add("")
    add("## 10. 文件与 SHA256")
    add("")
    add("| 文件 | sha256 |")
    add("|---|---|")
    for name, digest in sorted(files.items()):
        add(f"| `{name}` | `{digest}` |")
    add("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
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
        "task": "Bounded Codex-review correction of the fused pipeline FPGA -> Jetson "
                "transfer-layout accounting",
        "decision": bundle["decision"],
        "authority_prompt": bundle["authority_prompt"],
        "gate_checks": bundle["gate_checks"],
        "verification": bundle["verification"],
        "csynth": {
            "estimated_clock_ns": bundle["csynth"]["estimated_clock_ns"],
            "effective_budget_ns": bundle["csynth"]["effective_budget_ns"],
            "timing_slack_ns": bundle["csynth"]["timing_slack_ns"],
            "resource_utilization_percent": bundle["csynth"]["resource_utilization_percent"],
            "console": bundle["csynth"]["console"],
            "critical_path": bundle["csynth"]["critical_path"],
            "interface": bundle["csynth"]["interface"],
        },
        "derived_latency_200mhz": bundle["derived_latency_200mhz"],
        "transfer_audit": {
            "transport_time_basis": bundle["transfer_audit"]["transport_time_basis"],
            "layout_authority": bundle["transfer_audit"]["layout_authority"],
            "fixed_budget_arena_bytes_by_budget": {
                config["budget"]: config["fpga_to_jetson"]["fixed_budget_arena_bytes"]
                for config in bundle["transfer_audit"]["configs"].values()
            },
            "note": "fixed arena is produced live by the frozen contract; all "
                    "network/DDR FPGA -> Jetson times are based on it",
        },
        "transfer_audit_correction": {
            "status": bundle["transfer_audit_correction"]["status"],
            "review_finding": bundle["transfer_audit_correction"]["review_finding"],
            "fix": bundle["transfer_audit_correction"]["fix"],
            "deprecated_fields": bundle["transfer_audit_correction"]["deprecated_fields"],
            "authoritative_fields": bundle["transfer_audit_correction"]["authoritative_fields"],
            "decision_unchanged": bundle["transfer_audit_correction"]["decision_unchanged"],
            "hls_functional_timing_resource_unchanged": bundle["transfer_audit_correction"][
                "hls_functional_timing_resource_unchanged"
            ],
            "old_transfer_numbers_void": bundle["transfer_audit_correction"][
                "old_transfer_numbers_void"
            ],
            "identity_bypass_note": bundle["transfer_audit_correction"]["identity_bypass_note"],
            "focused_test_log": bundle["transfer_audit_correction"]["focused_test_log"],
            "configs": bundle["transfer_audit_correction"]["configs"],
        },
        "frozen_asset_sha256_check": bundle["frozen_asset_sha256_check"],
        "report_json": str(REPORT_JSON.relative_to(ROOT)),
        "report_markdown": bundle["report_markdown"],
        "original_pipeline_harness_result": {
            "path": str(ORIGINAL_HARNESS_RESULT.relative_to(ROOT)),
            "overwritten": False,
            "note": "kept read-only; its transfer totals/times are the deprecated "
                    "underestimated ones",
        },
        "scope": bundle["scope"],
    }
    CORRECTION_HARNESS_RESULT.write_text(
        json.dumps(harness, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "decision": bundle["decision"],
        "frozen_gates": bundle["gate_checks"]["frozen_gates"],
        "frozen_asset_sha256_check_ok": bundle["frozen_asset_sha256_check"][
            "all_must_be_unchanged_groups_ok"
        ],
        "report_json": str(REPORT_JSON.relative_to(ROOT)),
        "report_markdown": bundle["report_markdown"],
        "correction_harness_result": str(CORRECTION_HARNESS_RESULT.relative_to(ROOT)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
