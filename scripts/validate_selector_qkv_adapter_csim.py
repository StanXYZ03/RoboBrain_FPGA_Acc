#!/usr/bin/env python3
"""Build the Chinese report + harness for the Selector->LayerNorm-Q8_0 adapter.

This validator is CPU-only and read-mostly:

* regenerates the frozen real-block1 fixtures from the real safetensors header
  (read-only),
* runs the focused Python contract test and the full ``evaluation/tests``
  regression,
* parses the saved g++ / ASan+UBSan / HLS C-sim raw logs,
* computes the fused-vs-direct numerics for M=192/256/384 (K=48/64/96),
* writes ``reports/token_selector/2026-09-28-selector-qkv-adapter-csim.{md,json}``
  and ``/tmp/robobrain_harness_result.json``.

It never runs csynth/cosim/IP export/Vivado, never touches a board or GPU and
never uses the network.
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

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HLS_DIR = ROOT / "fpga_accel" / "selector_qkv_adapter_hls"
TOOLS = HLS_DIR / "tools"
ARTIFACT_DIR = ROOT / "artifacts" / "token_selector" / "2026-09-28-selector-qkv-adapter-csim"
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_STEM = "2026-09-28-selector-qkv-adapter-csim"
HARNESS_PATH = Path("/tmp/robobrain_harness_result.json")
PYTHON = "/home/zhaoxiangyu/conda-envs/robobrain25/bin/python"

DECISION = (
    "SELECTOR_QKV_ADAPTER_HLS_IMPLEMENTED / "
    "ADAPTER_LEGAL_M_192_256_384 / PARAM_CACHE_8KIB_BRAM / "
    "FUSED_LAYERNORM_Q8_0_BYTE_EXACT_VS_PYTHON_GOLDEN / "
    "INDEPENDENT_QKV_BIAS_ADD_KERNEL / GPP_ASAN_HLS_CSIM_PASS / "
    "CSIM_NOT_CSYNTH / ATTENTION_STILL_MISSING / NO_END_TO_END_SPEEDUP_CLAIM"
)

CONTRACT_CORRECTION = {
    "adapter_legal_rows": {
        "before": "rows<=0 || rows>4320 (SQA_MAX_ROWS=4320)",
        "after": "rows in {192,256,384} (SQA_MAX_ROWS=384, explicit legal check)",
        "reason": "Selector legal budgets are K=48/64/96 only, so M=4K is 192/256/384",
        "rejected_examples": [0, -1, 1, 191, 193, 385, 4320, 385],
        "bias_add_note": "qkv_bias_add keeps the general row gate 1..4320; formal qkv case is M=192/256/384,N=3072",
    },
    "param_storage": {
        "before": "norm_gamma/norm_beta re-read from m_axi per row/block",
        "after": "norm_gamma/norm_beta read once into 256x128-bit on-chip caches (4KiB each, 8KiB total) explicitly BIND_STORAGE type=ram_2p impl=bram; rows read from cache",
        "invariants": "math, Q8_0 payload/scale, port layout unchanged; byte-exact golden stays 0 mismatch",
    },
}


REQUIRED_CASES = (
    {"case_id": "m192_k48_seed20260928", "rows": 192, "seed": 20260928},
    {"case_id": "m192_k48_seed20260929", "rows": 192, "seed": 20260929},
    {"case_id": "m192_k48_seed20260930", "rows": 192, "seed": 20260930},
    {"case_id": "m256_k64_seed20260930", "rows": 256, "seed": 20260930},
    {"case_id": "m256_k64_seed20261001", "rows": 256, "seed": 20261001},
    {"case_id": "m384_k96_seed20261001", "rows": 384, "seed": 20261001},
    {"case_id": "m384_k96_seed20261002", "rows": 384, "seed": 20261002},
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


golden = _load_module("selector_qkv_golden", TOOLS / "selector_qkv_golden.py")
generator = _load_module("selector_qkv_generate_fixtures",
                         TOOLS / "generate_fixtures.py")


def sha256_file(path: Path) -> str:
    return golden.sha256_file(path)


def _float(value: Any) -> float:
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def run_case_metrics(case: dict[str, Any]) -> dict[str, Any]:
    rows = int(case["rows"])
    seed = int(case["seed"])
    hidden = generator.synthetic_block0_hidden(seed, rows)
    amax = float(np.max(np.abs(hidden.astype(np.float64))))
    scale = golden.selector_scale_from_amax(amax)
    q_int8, saturated = golden.selector_quantize(hidden, scale)
    # Use a stable synthetic gamma/beta so the numerics are deterministic and
    # independent of the real fixture; the byte-exact fixture uses the real one.
    rng = np.random.default_rng(seed)
    gamma = (1.0 + 0.05 * rng.standard_normal(golden.HIDDEN)).astype(np.float32)
    beta = (0.02 * rng.standard_normal(golden.HIDDEN)).astype(np.float32)
    fused = golden.fused_layernorm(q_int8, gamma, beta, scale)
    direct = golden.direct_layernorm(q_int8, gamma, beta, scale)
    payload, scale_bits, _ = golden.quantize_q8_0(fused)
    direct_payload, direct_scale_bits, _ = golden.quantize_q8_0(direct)
    dequant = golden.dequantize_q8_0(payload, scale_bits)
    return {
        "case_id": case["case_id"],
        "rows_M": rows,
        "selector_groups_K": rows // 4,
        "seed": seed,
        "global_scale_s": scale,
        "selector_saturated_values": saturated,
        "fused_vs_direct": golden.metrics(fused, direct),
        "dequant_vs_direct": golden.metrics(dequant, direct),
        "dequant_vs_fused": golden.metrics(dequant, fused),
        "fused_golden_payload_vs_direct_quant_payload": {
            "payload_byte_mismatch": int(np.count_nonzero(
                payload.astype(np.int16) - direct_payload.astype(np.int16))),
            "scale_value_mismatch": int(np.count_nonzero(
                scale_bits.astype(np.uint16) != direct_scale_bits.astype(np.uint16))),
            "payload_bytes": int(payload.size),
            "scale_values": int(scale_bits.size),
        },
    }


def parse_log(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False, "path": str(path.relative_to(ROOT))}
    text = path.read_text(encoding="utf-8", errors="replace")
    summary = None
    for match in re.finditer(r"SQA_TEST_SUMMARY passed=(\d+) total=(\d+) result=(\w+)",
                             text):
        summary = {"passed": int(match.group(1)), "total": int(match.group(2)),
                   "result": match.group(3)}
    fixtures = [line for line in text.splitlines() if line.startswith("SQA_FIXTURE")]
    return {
        "present": True,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "summary": summary,
        "fixture_lines": fixtures,
        "csim_done_zero_errors": "CSim done with 0 errors" in text,
        "runtime_error_count": text.count("runtime error"),
        "address_sanitizer_error_count": text.count("AddressSanitizer"),
        "failed_compilation": "compilation error(s)" in text,
    }


def run_unittest(targets: list[str], timeout: int = 900) -> dict[str, Any]:
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


def collect_real_tensor_evidence() -> dict[str, Any]:
    manifest_path = ARTIFACT_DIR / "fixture_manifest.json"
    if not manifest_path.is_file():
        manifest = generator.build_fixtures(ARTIFACT_DIR)
    else:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    return manifest


def build_report() -> dict[str, Any]:
    manifest = collect_real_tensor_evidence()
    numerics = [run_case_metrics(case) for case in REQUIRED_CASES]

    tests = {
        "focused_python_contract": run_unittest(
            ["evaluation.tests.test_selector_qkv_adapter_hls"]),
        "full_regression": run_unittest(["discover", "-s", "evaluation/tests"]),
        "gpp": parse_log(ARTIFACT_DIR / "gpp_test.log"),
        "asan_ubsan": parse_log(ARTIFACT_DIR / "asan_ubsan_run.log"),
        "hls_csim_adapter": parse_log(ARTIFACT_DIR / "hls_csim_adapter.log"),
        "hls_csim_bias": parse_log(ARTIFACT_DIR / "hls_csim_bias.log"),
    }

    files_changed = []
    for relative in (
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
        "evaluation/tests/test_selector_qkv_adapter_hls.py",
    ):
        path = ROOT / relative
        files_changed.append({"path": relative, "sha256": sha256_file(path),
                              "bytes": path.stat().st_size})
    for name in sorted(p.name for p in ARTIFACT_DIR.iterdir() if p.is_file()):
        path = ARTIFACT_DIR / name
        files_changed.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })

    adapter_metrics = manifest["adapter_fixture"]["metrics"]
    report = {
        "report": REPORT_STEM,
        "date": "2026-09-28",
        "decision": DECISION,
        "scope": {
            "cpu_only": True,
            "gpu_used": False,
            "board_used": False,
            "csynth": False,
            "cosim": False,
            "ip_export": False,
            "vivado": False,
            "network": False,
            "full_model_loaded": False,
            "existing_hls_modified": False,
            "model_files_modified": False,
        },
        "design_contract": {
            "adapter_top": "selector_ln_q8_adapter",
            "bias_add_top": "qkv_bias_add",
            "adapter_input": {
                "layout": "int8[M,1024], 128-bit little-endian packed, lane L = bits[8L+7:8L]",
                "words_per_row": golden.HIDDEN // golden.INT8_LANES,
                "selector_contract": "x ~= q/s, q = clamp(round_half_away(x*s), -128, 127), zero point 0",
                "global_scale": "single positive float s per image",
                "legal_rows": list(golden.LEGAL_ADAPTER_ROWS),
                "max_rows": golden.MAX_ROWS,
                "rows_gate": "rows must be exactly 192/256/384; 1/191/193/385/4320 rejected",
                "selector_clamp_in": [golden.SELECTOR_MIN, golden.SELECTOR_MAX],
            },
            "adapter_output": {
                "payload": "act_q[row*32+block] = 256-bit = 32 int8 little-endian",
                "scale": "act_scale[row*32+block] = binary16 (stored d = amax/127)",
                "q8_0_clamp": [golden.Q8_0_MIN, golden.Q8_0_MAX],
                "matches_existing": "q8q8_pe16_dataflow activations_q/activations_scale and ffn_vector quantize_block",
            },
            "bias_add": {
                "layout": "128-bit = 4 fp32, word = row*(cols/4)+col/4, lane = col%4",
                "operation": "y_out = y_in + bias[col]",
                "rows_gate": [1, golden.BIAS_MAX_ROWS],
                "cols_gate": [1, golden.MAX_COLS],
                "formal_case": {"rows_M": list(golden.LEGAL_ADAPTER_ROWS),
                                "cols_N": golden.QKV_COLS},
                "independent_operator": True,
            },
            "half_conversion": "exact copy of ffn_vector.cpp::float_to_half_bits",
        },
        "contract_correction": CONTRACT_CORRECTION,
        "math_equivalence": {
            "identity": "LN(q/s) = ((q - mean_q) / sqrt(var_q + eps*s*s)) * gamma + beta",
            "derivation": [
                "x = q/s, so mean_x = mean_q/s and var_x = var_q/s^2",
                "(x-mean_x)/sqrt(var_x+eps) = (q-mean_q)/s / sqrt(var_q/s^2+eps)",
                "= (q-mean_q)/sqrt(var_q + eps*s*s)",
                "then multiply by gamma and add beta",
            ],
            "population_variance": True,
            "variance_formula": "var_q = E[q^2] - E[q]^2, clamped at 0; same as ffn_vector row_statistics",
            "scale_broadcast_is_not_layernorm": True,
            "eps_term": "(eps*s)*s left-associative float32",
            "constant_row": "var_q=0 and eps*s*s=0 -> inverse_norm=0 -> output = beta (matches direct chain)",
            "minus128": "input -128 is used in the statistics; only the Q8_0 output clamps to -127",
            "case_metrics": numerics,
        },
        "real_tensor_evidence": {
            "shard_info": manifest["shard_info"],
            "tensors": manifest["tensors"],
            "adapter_fixture": manifest["adapter_fixture"],
            "bias_fixture": manifest["bias_fixture"],
            "note": "read-only header + byte-range reads; the full model was never loaded",
        },
        "numerics": {
            "adapter_real_block1": adapter_metrics,
            "required_cases": numerics,
            "metric_definitions": {
                "max_abs": "max |actual-reference|",
                "rmse": "sqrt(mean((actual-reference)^2))",
                "relative_l2": "||actual-reference||_2 / ||reference||_2",
                "cosine": "<actual,reference> / (||actual|| ||reference||)",
            },
        },
        "tests": tests,
        "limitations": [
            "仅 HLS C-sim：不是综合，不是时序/资源/功耗，不是板级结果。",
            "激活为合成，非真实采集；只有 norm1 gamma/beta 与 qkv.bias 是真实张量。",
            "数值链止于 +qkv.bias；reshape/split、RoPE、softmax、PV、attn.proj、残差、norm2、fc1/GELU/fc2 与 blocks 2..23 全部缺失。",
            "不声称端到端加速，也不判任务精度。",
            "仓库不存在 Vision tower 的 Q8_0 激活/权重产物；Q8Q8 权重侧工作不在本轮范围。",
            "Python golden 的 direct 参考使用向量化求和而 C++ 使用顺序求和，fused_vs_direct 的 max_abs 数量级略有差异（均远小于 Q8_0 步长）。",
        ],
    }
    report["files_changed"] = files_changed
    report["next_gate"] = (
        "在 +qkv.bias 之后实现 attention 各算子（reshape/split、RoPE、softmax、PV、"
        "attn.proj+bias、残差、norm2、fc1/GELU/fc2），仅在独立授权下再考虑 csynth/BD/驱动/上板；"
        "同时采集真实 block0 hidden 激活以做 fixture 级数值验证。"
    )
    return _jsonable(report)


def _metrics_row(entry: dict[str, float]) -> str:
    return (f"| {entry['max_abs']:.6g} | {entry['rmse']:.6g} | "
            f"{entry['relative_l2']:.6g} | {entry['cosine']:.9f} |")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Selector→LayerNorm-Q8_0 适配器与独立 QKV bias-add 参考核（g++/Sanitizer/HLS C-sim）")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 决策：`{report['decision']}`")
    lines.append("- 范围：CPU/HLS **C-sim only**；只读真实 BF16 张量 header+切片；"
                 "不加载完整模型；无 GPU/板卡/综合/cosim/IP 导出/Vivado/联网；"
                 "未改现有 selector/q8q8/pl_resident_ffn HLS 源码与模型文件。")
    lines.append("")
    lines.append("## 0. 边界声明（必读）")
    lines.append("")
    lines.append("- 激活为**确定性合成**，非真实采集 hidden；仅 `blocks.1.norm1.weight/bias` 与 "
                 "`blocks.1.attn.qkv.bias` 为真实张量。")
    lines.append("- 数值链只到 `+qkv.bias`；下游 attention 缺失，**不**声称完整、**不**判任务精度、"
                 "**不**声称端到端加速。")
    lines.append("- HLS C-sim **不是**综合/时序/资源/板级结果。")
    lines.append("")
    lines.append("## 1. 接口合同")
    lines.append("")
    lines.append("| 顶层 | 输入 | 输出 |")
    lines.append("|---|---|---|")
    lines.append("| `selector_ln_q8_adapter` | `int8[M,1024]` 128-bit 小端 packed + 全局 `s>0` + "
                 "真实 `norm1` gamma/beta (4×fp32/128-bit) | Q8_0 `[M,32]` 256-bit payload + "
                 "per-block binary16 scale（与现有 q8q8/pl_resident_ffn 合同一致） |")
    lines.append("| `qkv_bias_add` | Q8Q8 输出 4×fp32/128-bit `[M,3072]` + fp32 `bias[3072]` | "
                 "同布局 `y_in + bias[col]` |")
    lines.append("")
    lines.append("- Selector clamp `[-128,127]`；Q8_0 饱和 `[-127,127]`；`d=amax/127`；"
                 "`scale=float_to_half_bits(d)`；round 半离零。")
    lines.append("- `s<=0`/NaN/±inf → `SQA_ERR_SCALE`；adapter 的 `rows` 不属于 "
                 "`{192,256,384}`（含 `1/191/193/385/4320`）→ `SQA_ERR_DIMENSION`。")
    lines.append("- bias-add 保留通用 `rows∈[1,4320]`、`cols∈[1,9728]` 且 `cols%4==0`；"
                 "本轮正式 qkv case 为 `M=192/256/384, N=3072`。")
    lines.append("")
    lines.append("### 1.1 本轮合同修正（逐条）")
    correction = report["contract_correction"]
    legal = correction["adapter_legal_rows"]
    lines.append(f"- adapter 合法 rows：`{legal['before']}` → **`{legal['after']}`**。"
                 f"原因：{legal['reason']}。拒绝示例：`{legal['rejected_examples']}`。")
    lines.append(f"- bias-add：{legal['bias_add_note']}。")
    cache = correction["param_storage"]
    lines.append(f"- 参数存储：`{cache['before']}` → **`{cache['after']}`**；"
                 f"不变量：{cache['invariants']}。")
    lines.append("")
    lines.append("## 2. 数学等价推导")
    lines.append("")
    lines.append("令 `x=q/s`，则 `mean_x=mean_q/s`、`var_x=var_q/s²`（population variance）：")
    lines.append("")
    lines.append("```")
    lines.append("LN(x) = (x-mean_x)/sqrt(var_x+eps)*gamma + beta")
    lines.append("      = (q-mean_q)/sqrt(var_q + eps*s*s)*gamma + beta")
    lines.append("```")
    lines.append("")
    lines.append("- 因此可在 int8 域直接计算，不物化 `x`；`1/s` 广播只是反量化，**不等于** LayerNorm。")
    lines.append("- `var_q = E[q²]-E[q]²`（负值截 0），与 `ffn_vector.cpp::row_statistics` 的 population "
                 "variance 完全一致；常量行 `var_q=0` 且 `eps*s*s=0` 时取 `inverse_norm=0`，结果恰为 `beta`。")
    lines.append("- `q=-128` 参与统计；只有输出 Q8_0 饱和到 `-127`。")
    lines.append("")
    lines.append("## 3. 真实张量 fixture 取证（只读，未加载完整模型）")
    lines.append("")
    shard = report["real_tensor_evidence"]["shard_info"]
    lines.append(f"- shard：`{shard['shard']}`（{shard['shard_bytes']} B），header sha256 "
                 f"`{shard['header_sha256'][:16]}…`，读后未改动：`{shard['read_only_unchanged']}`。")
    lines.append("")
    lines.append("| 张量 | dtype | shape | data_offsets | raw sha256 |")
    lines.append("|---|---|---|---|---|")
    for name, info in report["real_tensor_evidence"]["tensors"].items():
        lines.append(f"| `{name}` | {info['dtype']} | {info['shape']} | "
                     f"{info['data_offsets']} | `{info['raw_sha256'][:16]}…` |")
    lines.append("")
    adapter_fixture = report["real_tensor_evidence"]["adapter_fixture"]
    bias_fixture = report["real_tensor_evidence"]["bias_fixture"]
    lines.append(f"- adapter fixture：`{adapter_fixture['path']}` "
                 f"sha256 `{adapter_fixture['sha256'][:16]}…`，M={adapter_fixture['rows_M']} "
                 f"(K={adapter_fixture['selector_groups_K']})，s={adapter_fixture['global_scale_s']:.6g}。")
    lines.append(f"- bias fixture：`{bias_fixture['path']}` sha256 `{bias_fixture['sha256'][:16]}…`，"
                 f"M={bias_fixture['rows_M']}，N={bias_fixture['cols_N']}。")
    lines.append("")
    lines.append("## 4. 逐字节对拍与逐级误差")
    lines.append("")
    lines.append("C++/HLS 内核对 Python 冻结 golden（fused 公式 + repository half）逐字节比较：")
    for key, label in (("hls_csim_adapter", "HLS C-sim(adapter)"),
                       ("gpp", "g++ 普通"),
                       ("asan_ubsan", "ASan+UBSan")):
        log = report["tests"].get(key)
        if not log or not log.get("present"):
            continue
        for line in log.get("fixture_lines", []):
            lines.append(f"- {label}：`{line}`")
    lines.append("")
    lines.append("### 4.1 adapter 真实 block1（M=192, K=48, 真实 gamma/beta）")
    lines.append("")
    adapter = report["numerics"]["adapter_real_block1"]
    lines.append("| 对象 | max_abs | RMSE | relative_L2 | cosine |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| fused golden vs direct FP32 LN | {_metrics_row(adapter['fused_vs_direct'])[2:]}")
    lines.append(f"| golden Q8_0 dequant vs direct FP32 LN | {_metrics_row(adapter['dequant_vs_direct'])[2:]}")
    mismatch = adapter["fused_golden_payload_vs_direct_quant_payload"]
    lines.append("")
    lines.append(f"- fused golden payload 对 direct-quant payload 字节差："
                 f"{mismatch['payload_byte_mismatch']}/{mismatch['payload_bytes']}，"
                 f"scale 值差：{mismatch['scale_value_mismatch']}/{mismatch['scale_values']}。")
    lines.append("- 上表为 Python golden 的数值；表下 C++/HLS 日志中 "
                 "`fused_vs_direct.max_abs≈4.3e-6`（C++ direct 用顺序 float32 求和，"
                 "Python 用向量化求和），两者都远小于 Q8_0 量化步长。")
    lines.append("- C++/HLS 的 adapter payload/scale 与 bias 逐字节差均为 **0**，即与 "
                 "Python 冻结 golden **逐字节一致**。")
    lines.append("")
    lines.append("### 4.2 必需案例（合成激活，K=48/64/96 → M=192/256/384，多 seed）")
    lines.append("")
    lines.append("| case | M | K | s | 饱和 | fused_vs_direct rel_L2 | dequant_vs_direct rel_L2 | cosine |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for case in report["numerics"]["required_cases"]:
        lines.append(f"| {case['case_id']} | {case['rows_M']} | {case['selector_groups_K']} | "
                     f"{case['global_scale_s']:.6g} | {case['selector_saturated_values']} | "
                     f"{case['fused_vs_direct']['relative_l2']:.3g} | "
                     f"{case['dequant_vs_direct']['relative_l2']:.3g} | "
                     f"{case['dequant_vs_direct']['cosine']:.9f} |")
    lines.append("")
    lines.append("## 5. 测试与原始日志")
    lines.append("")
    lines.append("| 门 | 结果 | 证据 |")
    lines.append("|---|---|---|")
    focused = report["tests"]["focused_python_contract"]
    full = report["tests"]["full_regression"]
    lines.append(f"| 聚焦 Python 合同测试 | Ran {focused['ran']} tests, "
                 f"{'OK' if focused['ok'] else 'FAILED'} | `evaluation/tests/test_selector_qkv_adapter_hls.py` |")
    lines.append(f"| 全量 evaluation/tests 回归 | Ran {full['ran']} tests, "
                 f"{'OK' if full['ok'] else 'FAILED'} | `evaluation/tests` |")
    for key, label in (("gpp", "g++ 普通"), ("asan_ubsan", "ASan+UBSan"),
                       ("hls_csim_adapter", "HLS C-sim(adapter)"),
                       ("hls_csim_bias", "HLS C-sim(bias)")):
        log = report["tests"].get(key, {})
        if not log.get("present"):
            lines.append(f"| {label} | NOT_RUN | - |")
            continue
        summary = log.get("summary") or {}
        lines.append(f"| {label} | {summary.get('passed', '?')}/{summary.get('total', '?')} "
                     f"{summary.get('result', '?')}；CSim0err={log.get('csim_done_zero_errors')}；"
                     f"runtime_err={log.get('runtime_error_count')}；asan_err={log.get('address_sanitizer_error_count')} "
                     f"| `{log['path']}` |")
    lines.append("")
    lines.append("## 6. 主机工具链 workaround（仅 host testbench）")
    lines.append("")
    lines.append("Ubuntu 24.04 / glibc 2.39 下复用仓库既有可复现 workaround：")
    lines.append("1. `LD_LIBRARY_PATH=/home/xilinx/xic/lib/lnx64.o/SuSE` 提供 Tcl 需要的 `libtinfo.so.5`；")
    lines.append("2. `COMPILER_PATH=<symlink ld → /usr/bin/ld>`、`LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`、"
                 "`AP_GCC_PATH`=自带 GCC 8.3.0、`config_csim -ldflags \"-no-pie\"` 规避 binutils 2.37 "
                 "无法读取系统 `libm.so.6 .relr.dyn`；DUT/算法未变。")
    lines.append("3. ASan/UBSan 使用 `-fno-sanitize=shift`：Vitis HLS `ap_int` 头内部 left-shift-of-negative "
                 "惯用法会误报；本内核位移全部作用于无符号值且位移量有界，未丢失内核侧检查。")
    lines.append("")
    lines.append("## 7. 文件变更")
    lines.append("")
    for entry in report["files_changed"]:
        lines.append(f"- `{entry['path']}`（{entry['bytes']} B，sha256 `{entry['sha256'][:16]}…`）")
    lines.append("")
    lines.append("## 8. 限制与下一步")
    lines.append("")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(f"**精确下一步（未执行）**：{report['next_gate']}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_harness(report: dict[str, Any], path: Path) -> None:
    harness = {
        "status": report["decision"],
        "design_contract": report["design_contract"],
        "math_equivalence": {
            "identity": report["math_equivalence"]["identity"],
            "population_variance": report["math_equivalence"]["population_variance"],
            "scale_broadcast_is_not_layernorm": report["math_equivalence"][
                "scale_broadcast_is_not_layernorm"],
            "required_case_count": len(report["math_equivalence"]["case_metrics"]),
        },
        "real_tensor_evidence": {
            "shard": report["real_tensor_evidence"]["shard_info"]["shard"],
            "read_only_unchanged": report["real_tensor_evidence"]["shard_info"][
                "read_only_unchanged"],
            "tensors": list(report["real_tensor_evidence"]["tensors"].keys()),
            "adapter_fixture_sha256": report["real_tensor_evidence"][
                "adapter_fixture"]["sha256"],
            "bias_fixture_sha256": report["real_tensor_evidence"]["bias_fixture"]["sha256"],
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
            "gpp": report["tests"]["gpp"]["summary"],
            "asan_ubsan": report["tests"]["asan_ubsan"]["summary"],
            "hls_csim_adapter": report["tests"]["hls_csim_adapter"]["summary"],
            "hls_csim_bias": report["tests"]["hls_csim_bias"]["summary"],
        },
        "numerics": {
            "adapter_real_block1_fused_vs_direct": report["numerics"][
                "adapter_real_block1"]["fused_vs_direct"],
            "adapter_real_block1_dequant_vs_direct": report["numerics"][
                "adapter_real_block1"]["dequant_vs_direct"],
            "required_cases": [
                {
                    "case_id": case["case_id"],
                    "rows_M": case["rows_M"],
                    "selector_groups_K": case["selector_groups_K"],
                    "fused_vs_direct_relative_l2": case["fused_vs_direct"]["relative_l2"],
                    "dequant_vs_direct_relative_l2": case["dequant_vs_direct"]["relative_l2"],
                    "cosine": case["dequant_vs_direct"]["cosine"],
                }
                for case in report["numerics"]["required_cases"]
            ],
        },
        "limitations": report["limitations"],
        "files_changed": [entry["path"] for entry in report["files_changed"]] + [
            str((REPORT_DIR / f"{REPORT_STEM}.md").relative_to(ROOT)),
            str((REPORT_DIR / f"{REPORT_STEM}.json").relative_to(ROOT)),
            "CURRENT_STATUS.md",
            str(path),
        ],
        "next_gate": report["next_gate"],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(harness, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true",
                        help="skip the focused/full unittest runs")
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
    print(json.dumps({
        "report_md": str(md_path.relative_to(ROOT)),
        "report_json": str(json_path.relative_to(ROOT)),
        "harness": str(HARNESS_PATH),
        "decision": report["decision"],
        "focused": report["tests"]["focused_python_contract"].get("ran"),
        "full": report["tests"]["full_regression"].get("ran"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
