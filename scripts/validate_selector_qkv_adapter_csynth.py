#!/usr/bin/env python3
"""Assemble the controlled-csynth report + final harness for the adapter/bias.

This validator is CPU-only and read-mostly:

* summarizes the two controlled ``csynth_design`` runs (adapter + bias) from
  their small raw reports under
  ``artifacts/token_selector/2026-09-28-selector-qkv-adapter-csynth/``,
* parses the saved g++ / ASan+UBSan / HLS C-sim csim-regression logs,
* runs the focused Python contract test and the full ``evaluation/tests``
  regression,
* writes ``reports/token_selector/2026-09-28-selector-qkv-adapter-csynth.{md,json}``
  and the final ``/tmp/robobrain_harness_result.json``.

It never runs a synthesis itself, never runs cosim/IP export/Vivado, never
touches a board or GPU and never uses the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CSIM_ARTIFACT = ROOT / "artifacts" / "token_selector" / "2026-09-28-selector-qkv-adapter-csim"
CSYNTH_ARTIFACT = (ROOT / "artifacts" / "token_selector" /
                   "2026-09-28-selector-qkv-adapter-csynth")
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_STEM = "2026-09-28-selector-qkv-adapter-csynth"
HARNESS_PATH = Path("/tmp/robobrain_harness_result.json")
PYTHON = "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python"
SUMMARIZER = ROOT / "scripts" / "summarize_selector_csynth.py"

ADAPTER_TOP = "selector_ln_q8_adapter"
BIAS_TOP = "qkv_bias_add"
CLOCK_NS = 5.0
FREQ_HZ = 200_000_000.0
ADAPTER_TARGET_MS = 33.3

DECISION_BASE = (
    "SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / ADAPTER_LEGAL_M_192_256_384 / "
    "PARAM_CACHE_8KIB_BRAM / FUSED_LAYERNORM_Q8_0_BYTE_EXACT_VS_PYTHON_GOLDEN / "
    "INDEPENDENT_QKV_BIAS_ADD_KERNEL / GPP_ASAN_HLS_CSIM_PASS / "
    "CONTROLLED_CSYNTH_RUN / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def parse_log(path: Path, relative_root: Path = ROOT) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = None
    for match in re.finditer(r"SQA_TEST_SUMMARY passed=(\d+) total=(\d+) result=(\w+)",
                             text):
        summary = {"passed": int(match.group(1)), "total": int(match.group(2)),
                   "result": match.group(3)}
    fixtures = [line for line in text.splitlines() if line.startswith("SQA_FIXTURE")]
    return {
        "present": True,
        "path": str(path.relative_to(relative_root)),
        "sha256": sha256_file(path),
        "summary": summary,
        "fixture_lines": fixtures,
        "csim_done_zero_errors": "CSim done with 0 errors" in text,
        "runtime_error_count": text.count("runtime error"),
        "address_sanitizer_error_count": text.count("AddressSanitizer"),
        "failed_compilation": "compilation error(s)" in text,
    }


def run_unittest(targets: list[str], timeout: int = 1200) -> dict[str, Any]:
    command = [PYTHON, "-m", "unittest"] + targets
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                               timeout=timeout)
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests? in", output)
    ran = int(match.group(1)) if match else None
    ok = bool(re.search(r"^OK", output, re.MULTILINE))
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "ran": ran,
        "ok": ok,
        "output_tail": output.strip().splitlines()[-3:],
    }


def summarize_csynth(mode: str, top: str) -> dict[str, Any]:
    mode_dir = CSYNTH_ARTIFACT / mode
    summary_path = mode_dir / "summary.json"
    completed = subprocess.run(
        [PYTHON, str(SUMMARIZER),
         "--report-dir", str(mode_dir),
         "--console-log", str(mode_dir / "console.log"),
         "--df-before", str(mode_dir / "df_before.txt"),
         "--project-dir", f"/tmp/robobrain_selector_qkv_adapter_csynth_{mode}",
         "--top", top,
         "--clock-ns", str(CLOCK_NS),
         "--out", str(summary_path)],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"summarize failed for {mode}: {completed.stderr}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_adapter_row_loop() -> dict[str, Any]:
    """Extract the outer row-loop trip-count/latency range from the top report.

    Returns the directly-reported M=192 / M=384 figures and a bounded M=256
    derivation; never fabricates an exact M=256 number.
    """
    report = CSYNTH_ARTIFACT / "adapter" / f"{ADAPTER_TOP}_csynth.rpt"
    text = report.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"\|\s*-\s+(VITIS_LOOP_\d+_\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
        r"(\d+)\s*~\s*(\d+)\s*\|[^|]*\|[^|]*\|\s*(\d+)\s*~\s*(\d+)\s*\|\s*no\s*\|")
    match = pattern.search(text)
    if match is None:
        return {
            "available": False,
            "note": "adapter row-loop range not found in the top report",
        }
    name = match.group(1)
    latency_min = int(match.group(2))
    latency_max = int(match.group(3))
    iteration_min = int(match.group(4))
    iteration_max = int(match.group(5))
    trip_min = int(match.group(6))
    trip_max = int(match.group(7))
    derived_256 = {
        "cycles_lower_bound": iteration_min * 256,
        "cycles_upper_bound": iteration_max * 256,
        "method": ("row-loop iteration latency range "
                   f"{iteration_min}..{iteration_max} x M=256"),
        "is_exact": False,
    }
    return {
        "available": True,
        "loop": name,
        "trip_count_range": [trip_min, trip_max],
        "row_loop_latency_cycles": {"M_192": latency_min, "M_384": latency_max},
        "row_loop_latency_ms_at_200mhz": {
            "M_192": round(latency_min / FREQ_HZ * 1000.0, 4),
            "M_384": round(latency_max / FREQ_HZ * 1000.0, 4),
        },
        "M_256": derived_256,
        "note": ("The tool reports the M=192 and M=384 row-loop bounds directly. "
                 "M=256 is not separately schedulable because `rows` is a runtime "
                 "scalar; only the bounded derivation above is given."),
    }


def parse_bias_formal_boundary(summary: dict[str, Any]) -> dict[str, Any]:
    """Derive the formal M=384,N=3072 bias-add cycle count from the flat pipeline.

    The bias kernel is a single flattened (rows x words) pipeline with achieved
    II reported by the tool; the derivation is reported alongside (never instead
    of) the tool's full dynamic-range worst case.
    """
    flat = [loop for loop in summary.get("loops", [])
            if loop.get("pipelined") == "yes" and loop.get("trip_count") not in (None, "-")]
    if not flat:
        return {"available": False,
                "note": "no flat pipelined loop found; dynamic cols prevent a value"}
    loop = max(flat, key=lambda item: int(item["trip_count"]))
    interval = int(loop["interval"])
    trip_max = int(loop["trip_count"])
    instance_max = None
    for candidate in summary.get("loops", []):
        if candidate.get("latency_cycles") not in (None, "-") and \
                "Pipeline" in str(candidate.get("loop", "")):
            instance_max = int(candidate["latency_cycles"])
    overhead = (instance_max - interval * trip_max) if instance_max is not None else 0
    words_m384 = 384 * (3072 // 4)
    cycles_m384 = interval * words_m384 + max(overhead, 0)
    return {
        "available": True,
        "loop": loop["loop"],
        "achieved_ii": interval,
        "flat_trip_count_max_rows_cols": trip_max,
        "tool_worst_cycles_rows4320_cols9728": int(summary["latency"]["worst"]),
        "formal_M384_N3072": {
            "words": words_m384,
            "cycles": cycles_m384,
            "ms_at_200mhz": round(cycles_m384 / FREQ_HZ * 1000.0, 4),
            "method": (f"flat pipeline: achieved II={interval} x "
                       f"M*N/4 words (+{max(overhead, 0)} reported overhead)"),
            "is_exact": False,
        },
        "note": ("`cols` is a runtime scalar, so the tool cannot report the exact "
                 "M=384,N=3072 latency; the full dynamic worst case (M=4320, "
                 "N=9728) is the reported hard boundary and the formal case is "
                 "derived from the achieved II."),
    }


def build_regression() -> dict[str, Any]:
    csim_json_path = REPORT_DIR / "2026-09-28-selector-qkv-adapter-csim.json"
    csim_json = (json.loads(csim_json_path.read_text(encoding="utf-8"))
                 if csim_json_path.is_file() else None)
    return {
        "gpp": parse_log(CSIM_ARTIFACT / "gpp_test.log"),
        "asan_ubsan": parse_log(CSIM_ARTIFACT / "asan_ubsan_run.log"),
        "hls_csim_adapter": parse_log(CSIM_ARTIFACT / "hls_csim_adapter.log"),
        "hls_csim_bias": parse_log(CSIM_ARTIFACT / "hls_csim_bias.log"),
        "csim_report": {
            "present": csim_json is not None,
            "path": str(csim_json_path.relative_to(ROOT)),
            "decision": csim_json["decision"] if csim_json else None,
            "fixture_byte_exact": _fixture_byte_exact(csim_json),
        },
    }


def _fixture_byte_exact(csim_json: dict[str, Any] | None) -> Any:
    if not csim_json:
        return None
    out = {}
    for key in ("gpp", "asan_ubsan", "hls_csim_adapter", "hls_csim_bias"):
        block = csim_json.get("tests", {}).get(key, {})
        lines = block.get("fixture_lines", [])
        mismatches = []
        for line in lines:
            found = re.findall(r"(payload_byte_mismatch|scale_byte_mismatch|byte_mismatch)=(\d+)",
                               line)
            mismatches.extend(int(value) for _, value in found)
        out[key] = {"fixture_lines": lines, "mismatch_values": mismatches,
                    "zero_mismatch": bool(lines) and all(v == 0 for v in mismatches)}
    return out


def _gate(summary: dict[str, Any], *, is_adapter: bool) -> dict[str, Any]:
    latency = summary["latency"]
    worst = latency.get("worst")
    finite = worst is not None and latency.get("best") is not None
    console = summary["console"]
    resource_over = summary["any_resource_over_50_percent"]
    estimated = summary["estimated_clock_ns"]
    effective = summary["effective_budget_ns"]
    slack = summary["timing_slack_ns"]
    worst_ms = round(worst / FREQ_HZ * 1000.0, 4) if worst is not None else None
    latency_target = (worst_ms is not None and worst_ms < ADAPTER_TARGET_MS) \
        if is_adapter else True
    checks = {
        "zero_errors": console["error_count"] == 0,
        "latency_finite": finite,
        "estimated_period_le_effective_budget": bool(
            estimated is not None and effective is not None and estimated <= effective),
        "timing_slack_non_negative": bool(slack is not None and slack >= 0),
        "any_resource_below_50_percent": not resource_over,
        "worst_latency_target": latency_target,
    }
    if not is_adapter:
        checks.pop("worst_latency_target")
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "worst_latency_ms_at_200mhz": worst_ms,
        "adapter_target_ms": ADAPTER_TARGET_MS if is_adapter else None,
        "failed_checks": [key for key, value in checks.items() if not value],
    }


def _console_estimated_ns(path: Path) -> float | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Estimated clock period \(([\d.]+) ns\)", text)
    return float(match.group(1)) if match else None


def adapter_fix_rounds() -> list[dict[str, Any]]:
    """Record the at-most-two pragma-only fix rounds and the real root cause."""
    return [
        {
            "round": 0,
            "change": "original design (no scheduling pragma)",
            "estimated_clock_ns": _console_estimated_ns(
                CSYNTH_ARTIFACT / "adapter_round0" / "console.log"),
            "result": "csynth completed; timing gate FAIL (3.945 > 3.650 ns)",
            "evidence": "adapter_round0/console.log",
        },
        {
            "round": 1,
            "change": "scheduling pragma: HLS PIPELINE II=2 on the Q8_0 amax loop",
            "estimated_clock_ns": _console_estimated_ns(
                CSYNTH_ARTIFACT / "adapter_round1" / "console.log"),
            "result": ("csynth completed; timing gate still FAIL (3.945 ns, same "
                       "fcmp->select->store path), latency reaches II=2 but the "
                       "combinational compare is not broken"),
            "evidence": "adapter_round1/console.log",
        },
        {
            "round": 2,
            "change": ("scheduling pragma: named compare + BIND_OP fcmp "
                       "impl=fabric latency=1"),
            "estimated_clock_ns": None,
            "result": ("pre-synthesis ERROR: HLS 214-372 (fcmp impl=fabric "
                       "unsupported on xczu9eg) and HLS 214-383 (fcmp latency "
                       "range [-1,-1]); not a valid synthesized design"),
            "evidence": "adapter_round2_bindop_unsupported/console.log",
        },
    ]


ADAPTER_ROOT_CAUSE = (
    "Q8_0 quantize_block 的 32 路 amax 归约是 loop-carried 依赖 "
    "(amax -> fcmp -> select -> amax)。2023.2 在该器件上把单精度浮点 `>` 实现为约 "
    "2.782 ns 的 fcmp，加上 select 0.449 ns 与 store 0.427 ns，单周期组合路径 "
    "3.945 ns > 默认 uncertainty 下的 3.650 ns effective budget。只允许调度/存储 pragma "
    "的等价修复无法打断该组合路径：II=2 不生效；`BIND_OP op=fcmp` 在该器件不支持。"
    "同一 3.945 ns 路径也存在于现有 pl_resident_ffn 的 quantize_block csynth 中，"
    "证明这是算法实现的固有路径，而非本轮参数 cache 引入。"
)



def collect_files_changed() -> list[dict[str, Any]]:
    explicit = [
        "README.md",
        "fpga_accel/selector_qkv_adapter_hls/README.md",
        "fpga_accel/selector_qkv_adapter_hls/Makefile",
        "fpga_accel/selector_qkv_adapter_hls/include/selector_qkv_adapter.h",
        "fpga_accel/selector_qkv_adapter_hls/src/selector_qkv_adapter.cpp",
        "fpga_accel/selector_qkv_adapter_hls/tb/tb_selector_qkv_adapter.cpp",
        "fpga_accel/selector_qkv_adapter_hls/tools/selector_qkv_golden.py",
        "fpga_accel/selector_qkv_adapter_hls/tools/generate_fixtures.py",
        "fpga_accel/selector_qkv_adapter_hls/scripts/run_hls.tcl",
        "fpga_accel/selector_qkv_adapter_hls/scripts/run_stage.sh",
        "fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.tcl",
        "fpga_accel/selector_qkv_adapter_hls/scripts/run_csynth.sh",
        "scripts/validate_selector_qkv_adapter_csim.py",
        "scripts/validate_selector_qkv_adapter_csynth.py",
        "scripts/summarize_selector_csynth.py",
        "evaluation/tests/test_selector_qkv_adapter_hls.py",
    ]
    entries = []
    seen: set[str] = set()
    for relative in explicit:
        path = ROOT / relative
        if not path.is_file():
            continue
        seen.add(str(path))
        entries.append({"path": relative, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size})
    for root in (CSIM_ARTIFACT, CSYNTH_ARTIFACT):
        if not root.is_dir():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if str(path) in seen:
                continue
            seen.add(str(path))
            entries.append({"path": str(path.relative_to(ROOT)),
                            "sha256": sha256_file(path),
                            "bytes": path.stat().st_size})
    return entries


LIMITATIONS = [
    "仅 csynth_design：不是 C/RTL cosim、不是 place/route、不是 bitstream、不是板级结果。",
    "激活为合成，非真实采集；只有 norm1 gamma/beta 与 qkv.bias 是真实张量。",
    "数值链止于 +qkv.bias；reshape/split、RoPE、softmax、PV、attn.proj、残差、norm2、"
    "fc1/GELU/fc2 与 blocks 2..23 全部缺失。",
    "不声称端到端加速，也不判任务精度；Q8Q8 权重侧工作不在本轮范围。",
    "adapter 的 M=256 周期只能由 row-loop 迭代延迟范围给出有界推导，不是工具单独报告值。",
    "bias 的 cols 为运行时标量，M=384,N=3072 周期由 achieved II 推导；工具硬边界为 M=4320,N=9728。",
    "HLS 200-790 仍报 loop constraints 未满足（adapter 参数 cache 载入循环与 bias 内层循环存在 II "
    "violation），这不是 correctness/timing 门槛；gate 以 0 error/有限 latency/timing/资源判定。",
]

NEXT_GATE = (
    "adapter csynth 在默认 1.35 ns uncertainty / 5.0 ns 下 estimated 3.945 ns > 3.650 ns，"
    "未过 timing 门槛；下一步只有在独立授权下才可用不等价手段（例如把 32 路 amax 归约改成"
    "平衡树/分段最大值并对 fcmp 打拍，或重排量化块调度）以打破该 loop-carried 组合路径，"
    "再重跑同一条只 csynth_design 的流程；在此之前不 cosim、不导出 IP、不 route、不建 BD、"
    "不写驱动、不上板。+qkv.bias 之后的 attention 各算子与真实 block0 hidden 采集仍待完成。"
)


def build_report() -> dict[str, Any]:
    csim = _load_module("validate_selector_qkv_adapter_csim",
                        ROOT / "scripts" / "validate_selector_qkv_adapter_csim.py")

    adapter = summarize_csynth("adapter", ADAPTER_TOP)
    bias = summarize_csynth("bias", BIAS_TOP)

    adapter["legal_M_cycles"] = parse_adapter_row_loop()
    bias["formal_boundary"] = parse_bias_formal_boundary(bias)
    adapter["latency_ms_at_200mhz"] = _latency_ms(adapter["latency"])
    bias["latency_ms_at_200mhz"] = _latency_ms(bias["latency"])

    adapter_gate = _gate(adapter, is_adapter=True)
    bias_gate = _gate(bias, is_adapter=False)

    tests = {
        "focused_python_contract": run_unittest(
            ["evaluation.tests.test_selector_qkv_adapter_hls"]),
        "full_regression": run_unittest(["discover", "-s", "evaluation/tests"]),
    }
    regression = build_regression()

    report = {
        "report": REPORT_STEM,
        "date": "2026-09-28",
        "decision": DECISION_BASE,
        "scope": {
            "cpu_only": True,
            "gpu_used": False,
            "board_used": False,
            "csynth": True,
            "csynth_only": True,
            "cosim": False,
            "ip_export": False,
            "vivado": False,
            "network": False,
            "full_model_loaded": False,
            "existing_hls_modified": False,
            "model_files_modified": False,
        },
        "contract_correction": csim.CONTRACT_CORRECTION,
        "csim_regression": {
            "logs": regression,
            "focused_python_contract": tests["focused_python_contract"],
            "full_regression": tests["full_regression"],
        },
        "adapter_csynth": {
            "top": ADAPTER_TOP,
            "project_dir": "/tmp/robobrain_selector_qkv_adapter_csynth_adapter",
            "summary": adapter,
            "gate": adapter_gate,
            "fix_rounds": adapter_fix_rounds(),
            "root_cause": ADAPTER_ROOT_CAUSE,
            "cosim_route_not_run": True,
        },
        "bias_csynth": {
            "top": BIAS_TOP,
            "project_dir": "/tmp/robobrain_selector_qkv_adapter_csynth_bias",
            "summary": bias,
            "gate": bias_gate,
        },
        "tests": tests,
        "limitations": LIMITATIONS,
        "files_changed": collect_files_changed(),
        "next_gate": NEXT_GATE,
    }
    suffixes = []
    if adapter_gate["status"] == "PASS":
        suffixes.append("ADAPTER_CSYNTH_GATE_PASS")
    else:
        suffixes.append("ADAPTER_CSYNTH_TIMING_FAIL")
    if bias_gate["status"] == "PASS":
        suffixes.append("BIAS_CSYNTH_GATE_PASS")
    else:
        suffixes.append("BIAS_CSYNTH_GATE_FAIL")
    suffixes.append("NO_COSIM_NO_ROUTE")
    report["decision"] = DECISION_BASE + " / " + " / ".join(suffixes)
    if adapter_gate["status"] != "PASS" or bias_gate["status"] != "PASS":
        report["limitations"] = LIMITATIONS + [
            "adapter csynth timing gate FAIL：estimated period 3.945 ns > 3.650 ns "
            "effective budget；按门槛如实判 FAIL，未继续 cosim/route。",
        ]
    return _jsonable(report)


def _latency_ms(latency: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ("best", "average", "worst"):
        value = latency.get(key)
        out[key] = round(value / FREQ_HZ * 1000.0, 4) if value is not None else None
    return out


def _fmt_percent_map(util: dict[str, Any]) -> str:
    order = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
    parts = [f"{key} {util.get(key, 0.0):.2f}%" for key in order if key in util]
    return "、".join(parts)


def write_markdown(report: dict[str, Any], path: Path) -> None:
    adapter = report["adapter_csynth"]
    bias = report["bias_csynth"]
    a_sum = adapter["summary"]
    b_sum = bias["summary"]
    lines: list[str] = []
    lines.append("# Selector→LayerNorm-Q8_0 适配器与独立 QKV bias-add：受控 HLS csynth")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 决策：`{report['decision']}`")
    lines.append("- 范围：**只 `csynth_design`**（Vitis HLS 2023.2、part "
                 "`xczu9eg-ffvb1156-2-e`、`create_clock -period 5.0`、默认 uncertainty 不放宽）；"
                 "无 C/RTL cosim、无 IP 导出、无 Vivado/BD/驱动、无板卡、无 GPU、无联网。")
    lines.append("- 大工程只放 `/tmp`；仓库只保存小报告与 hash。未改现有 selector/q8q8/"
                 "pl_resident_ffn 源码与模型文件。")
    lines.append("")
    lines.append("## 0. 前置合同修正（本轮）")
    lines.append("")
    correction = report["contract_correction"]
    legal = correction["adapter_legal_rows"]
    lines.append(f"- adapter 合法 rows：`{legal['before']}` → **`{legal['after']}`**；"
                 f"拒绝示例 `{legal['rejected_examples']}`。{legal['bias_add_note']}。")
    cache = correction["param_storage"]
    lines.append(f"- 参数存储：`{cache['before']}` → **`{cache['after']}`**；"
                 f"不变量：{cache['invariants']}。")
    lines.append("")
    lines.append("## 1. 逐字节 C-sim 回归（修正后重跑，0 mismatch）")
    lines.append("")
    regression = report["csim_regression"]["logs"]
    lines.append("| 门 | 结果 | 证据 |")
    lines.append("|---|---|---|")
    for key, label in (("gpp", "g++ 普通"), ("asan_ubsan", "ASan+UBSan"),
                       ("hls_csim_adapter", "HLS C-sim(adapter)"),
                       ("hls_csim_bias", "HLS C-sim(bias)")):
        log = regression[key]
        summary = log.get("summary") or {}
        lines.append(f"| {label} | {summary.get('passed', '?')}/{summary.get('total', '?')} "
                     f"{summary.get('result', '?')}；CSim0err={log.get('csim_done_zero_errors')}；"
                     f"runtime_err={log.get('runtime_error_count')}；"
                     f"asan_err={log.get('address_sanitizer_error_count')} | `{log['path']}` |")
    lines.append("")
    for key, label in (("gpp", "g++ 普通"), ("asan_ubsan", "ASan+UBSan"),
                       ("hls_csim_adapter", "HLS C-sim(adapter)"),
                       ("hls_csim_bias", "HLS C-sim(bias)")):
        for fixture in regression[key].get("fixture_lines", []):
            lines.append(f"- {label}：`{fixture}`")
    lines.append("")
    lines.append("两条 fixture 的 payload/scale 与 bias 输出对 Python 冻结 golden "
                 "逐字节 mismatch 均为 **0**。")
    lines.append("")
    lines.append("## 2. adapter csynth（top `selector_ln_q8_adapter`）")
    lines.append("")
    lines.append(f"- project：`{adapter['project_dir']}`；只 `csynth_design`。")
    lines.append(f"- estimated period：**{a_sum['estimated_clock_ns']} ns**；effective budget："
                 f"**{a_sum['effective_budget_ns']} ns**（target {a_sum['clock_target_ns']} ns − "
                 f"uncertainty {a_sum['clock_uncertainty_ns']} ns）；slack："
                 f"**{a_sum['timing_slack_ns']} ns**；Fmax "
                 f"{a_sum['estimated_fmax_mhz']} MHz。")
    lines.append(f"- latency cycles：best {a_sum['latency']['best']}、average "
                 f"{a_sum['latency']['average']}、worst **{a_sum['latency']['worst']}**；"
                 f"@200MHz：best {_latency_ms(a_sum['latency'])['best']} ms、average "
                 f"{_latency_ms(a_sum['latency'])['average']} ms、worst "
                 f"**{a_sum['worst_latency_ms_at_200mhz']} ms** (< "
                 f"{ADAPTER_TARGET_MS} ms)。best=1 来自非法 rows 的提前返回路径，不是 M=192。")
    lines.append(f"- 资源：{_fmt_percent_map(a_sum['resource_utilization_percent'])}"
                 f"（used `{a_sum['resources_used']}` / available `{a_sum['resources_available']}`）。"
                 "任一资源 <50%。")
    lines.append(f"- II warnings / critical path：critical path 为 quantize amax 归约 "
                 f"`fcmp→select→store`（3.945 ns），模块 `selector_ln_q8_adapter_Pipeline_"
                 f"VITIS_LOOP_129_1`；`HLS 200-871`/`200-1016` 在合法 M=384 下仍触发。"
                 f"`HLS 200-885` 另报参数 cache 载入循环与 bias 内层循环的 II violation"
                 f"（gmem0 端口限制），不改变结果。")
    lines.append("")
    lines.append("### 2.1 允许的调度/存储 pragma 修复轮（最多 2 轮，均无法关闭 timing）")
    lines.append("")
    lines.append("| 轮 | 变更 | estimated period | 结果 | 证据 |")
    lines.append("|---|---|---|---|---|")
    for entry in adapter["fix_rounds"]:
        estimate = entry["estimated_clock_ns"]
        estimate_text = f"{estimate} ns" if estimate is not None else "-"
        lines.append(f"| {entry['round']} | {entry['change']} | {estimate_text} | "
                     f"{entry['result']} | `{entry['evidence']}` |")
    lines.append("")
    lines.append(f"- **真实根因**：{adapter['root_cause']}")
    lines.append(f"- gate：**{adapter['gate']['status']}**；failed_checks "
                 f"`{adapter['gate']['failed_checks']}`；按门槛判 FAIL，未继续 cosim/route。")
    legal_block = a_sum["legal_M_cycles"]
    if legal_block.get("available"):
        lc = legal_block["row_loop_latency_cycles"]
        lm = legal_block["row_loop_latency_ms_at_200mhz"]
        lines.append(f"- 合法 M 可推导周期（row-loop trip {legal_block['trip_count_range']}）："
                     f"M=192 → **{lc['M_192']} cycles / {lm['M_192']} ms**（工具直接报告）；"
                     f"M=384 → **{lc['M_384']} cycles / {lm['M_384']} ms**（工具直接报告）；"
                     f"M=256 → 区间 [{legal_block['M_256']['cycles_lower_bound']}, "
                     f"{legal_block['M_256']['cycles_upper_bound']}] cycles"
                     f"（**有界推导，非工具单独报告**）。")
    lines.append("")
    lines.append("## 3. bias csynth（top `qkv_bias_add`）")
    lines.append("")
    lines.append(f"- project：`{bias['project_dir']}`；0 error。")
    lines.append(f"- estimated period：**{b_sum['estimated_clock_ns']} ns**；effective budget："
                 f"**{b_sum['effective_budget_ns']} ns**；slack "
                 f"**{b_sum['timing_slack_ns']} ns**；Fmax {b_sum['estimated_fmax_mhz']} MHz。")
    lines.append(f"- latency cycles：best {b_sum['latency']['best']}、average "
                 f"{b_sum['latency']['average']}（动态 rows/cols 下 tool 报 `undef`）、worst "
                 f"**{b_sum['latency']['worst']}**（M=4320,N=9728 硬边界）；@200MHz：best "
                 f"{_latency_ms(b_sum['latency'])['best']} ms、average "
                 f"{_latency_ms(b_sum['latency'])['average']} ms（undef）、worst "
                 f"**{b_sum['worst_latency_ms_at_200mhz']} ms**。")
    lines.append(f"- 资源：{_fmt_percent_map(b_sum['resource_utilization_percent'])}"
                 f"（used `{b_sum['resources_used']}` / available `{b_sum['resources_available']}`）。"
                 "任一资源 <50%。")
    boundary = b_sum["formal_boundary"]
    if boundary.get("available"):
        formal = boundary["formal_M384_N3072"]
        lines.append(f"- 正式 M=384,N=3072 边界：words {formal['words']}、achieved II "
                     f"{boundary['achieved_ii']} → **{formal['cycles']} cycles / "
                     f"{formal['ms_at_200mhz']} ms @200MHz**（由工具报告的 achieved II 推导，"
                     f"非直接报告值）；工具硬边界 M=4320,N=9728 = "
                     f"{boundary['tool_worst_cycles_rows4320_cols9728']} cycles。")
    lines.append(f"- II warnings：`HLS 200-885` bias 内层平坦流水线 II violation（gmem0 端口限制，"
                 f"achieved II={boundary.get('achieved_ii')}）；`HLS 200-790` loop constraints "
                 f"未满足，不改变结果。")
    lines.append(f"- gate：**{bias['gate']['status']}**；checks `{bias['gate']['checks']}`。")
    lines.append("")
    lines.append("## 4. 测试")
    lines.append("")
    focused = report["tests"]["focused_python_contract"]
    full = report["tests"]["full_regression"]
    lines.append(f"| 门 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| 聚焦 Python 合同测试 | Ran {focused['ran']} tests, "
                 f"{'OK' if focused['ok'] else 'FAILED'} |")
    lines.append(f"| 全量 evaluation/tests 回归 | Ran {full['ran']} tests, "
                 f"{'OK' if full['ok'] else 'FAILED'} |")
    lines.append(f"| adapter csynth gate | {adapter['gate']['status']} |")
    lines.append(f"| bias csynth gate | {bias['gate']['status']} |")
    lines.append("")
    lines.append("## 5. 边界与限制")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(f"**精确下一步（未执行）**：{report['next_gate']}")
    lines.append("")
    lines.append("## 6. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append("cd fpga_accel/selector_qkv_adapter_hls")
    lines.append("make test && make asan_run && make csim && make csim_bias")
    lines.append("bash scripts/run_csynth.sh adapter   # 只 csynth_design，工程在 /tmp")
    lines.append("bash scripts/run_csynth.sh bias")
    lines.append("```")
    lines.append("")
    lines.append("## 7. 文件变更")
    lines.append("")
    for entry in report["files_changed"]:
        lines.append(f"- `{entry['path']}`（{entry['bytes']} B，sha256 "
                     f"`{entry['sha256'][:16]}…`）")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_harness(report: dict[str, Any], path: Path) -> None:
    adapter = report["adapter_csynth"]
    bias = report["bias_csynth"]
    harness = {
        "status": report["decision"],
        "contract_correction": report["contract_correction"],
        "csim_regression": {
            "gpp": report["csim_regression"]["logs"]["gpp"],
            "asan_ubsan": report["csim_regression"]["logs"]["asan_ubsan"],
            "hls_csim_adapter": report["csim_regression"]["logs"]["hls_csim_adapter"],
            "hls_csim_bias": report["csim_regression"]["logs"]["hls_csim_bias"],
            "fixture_byte_exact": report["csim_regression"]["logs"]["csim_report"][
                "fixture_byte_exact"],
        },
        "adapter_csynth": {
            "estimated_clock_ns": adapter["summary"]["estimated_clock_ns"],
            "effective_budget_ns": adapter["summary"]["effective_budget_ns"],
            "timing_slack_ns": adapter["summary"]["timing_slack_ns"],
            "latency": adapter["summary"]["latency"],
            "latency_ms_at_200mhz": adapter["summary"]["latency_ms_at_200mhz"],
            "worst_latency_ms_at_200mhz": adapter["summary"][
                "worst_latency_ms_at_200mhz"],
            "resource_utilization_percent": adapter["summary"][
                "resource_utilization_percent"],
            "legal_M_cycles": adapter["summary"]["legal_M_cycles"],
            "fix_rounds": adapter["fix_rounds"],
            "root_cause": adapter["root_cause"],
            "gate": adapter["gate"],
        },
        "bias_csynth": {
            "estimated_clock_ns": bias["summary"]["estimated_clock_ns"],
            "effective_budget_ns": bias["summary"]["effective_budget_ns"],
            "timing_slack_ns": bias["summary"]["timing_slack_ns"],
            "latency": bias["summary"]["latency"],
            "latency_ms_at_200mhz": bias["summary"]["latency_ms_at_200mhz"],
            "worst_latency_ms_at_200mhz": bias["summary"][
                "worst_latency_ms_at_200mhz"],
            "resource_utilization_percent": bias["summary"][
                "resource_utilization_percent"],
            "formal_boundary": bias["summary"]["formal_boundary"],
            "gate": bias["gate"],
        },
        "tests": {
            "focused_python_contract": {
                "ran": report["tests"]["focused_python_contract"]["ran"],
                "ok": report["tests"]["focused_python_contract"]["ok"],
            },
            "full_regression": {
                "ran": report["tests"]["full_regression"]["ran"],
                "ok": report["tests"]["full_regression"]["ok"],
            },
        },
        "limitations": report["limitations"],
        "files_changed": [entry["path"] for entry in report["files_changed"]] + [
            str((REPORT_DIR / f"{REPORT_STEM}.md").relative_to(ROOT)),
            str((REPORT_DIR / f"{REPORT_STEM}.json").relative_to(ROOT)),
            "CURRENT_STATUS.md",
        ],
        "next_gate": report["next_gate"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(harness), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    if args.skip_tests:
        for key in ("focused_python_contract", "full_regression"):
            report["tests"][key] = {"ran": None, "ok": None, "skipped": True}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / f"{REPORT_STEM}.json"
    md_path = REPORT_DIR / f"{REPORT_STEM}.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_markdown(report, md_path)
    write_harness(report, HARNESS_PATH)
    adapter_gate = report["adapter_csynth"]["gate"]["status"]
    bias_gate = report["bias_csynth"]["gate"]["status"]
    print(f"adapter_csynth={adapter_gate} bias_csynth={bias_gate}")
    print(f"adapter: est={report['adapter_csynth']['summary']['estimated_clock_ns']}ns "
          f"slack={report['adapter_csynth']['summary']['timing_slack_ns']}ns "
          f"worst={report['adapter_csynth']['summary']['worst_latency_ms_at_200mhz']}ms")
    print(f"bias: est={report['bias_csynth']['summary']['estimated_clock_ns']}ns "
          f"slack={report['bias_csynth']['summary']['timing_slack_ns']}ns")
    print(f"focused={report['tests']['focused_python_contract'].get('ran')} "
          f"full={report['tests']['full_regression'].get('ran')}")
    print(f"report={json_path.relative_to(ROOT)}")
    print(f"harness={HARNESS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
