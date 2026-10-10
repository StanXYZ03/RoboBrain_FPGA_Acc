#!/usr/bin/env python3
"""Freeze one PE16 DATAFLOW board run into append-only JSON and Chinese MD."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path


PE16_BASELINE_US = {
    "language_prefill": 242_808.0,
    "vision_encoder": 448_839.0,
    "decoder_gemv": 14_647.0,
}

STAGES = {
    "kernel_language_prefill_fixed512": ("language_prefill", 512, 512, 9728, 2560, None),
    "visual_prefill_fixed512": ("visual_prefill", 512, 1080, 9728, 2560, None),
    "visual_prefill_auto": ("visual_prefill", 976, 1080, 9728, 2560, None),
    "vision_encoder_fixed512": ("vision_encoder", 512, 4320, 4096, 1024, None),
    "vision_encoder_auto": ("vision_encoder", 3584, 4320, 4096, 1024, None),
    "decoder_gemv_fixed": ("decoder_gemv", 1, 1, 9728, 2560, None),
    "decoder_gemv_auto": ("decoder_gemv", 1, 1, 9728, 2560, None),
}
for _workload in ("visual_prefill", "vision_encoder", "decoder_gemv"):
    _chunk = {"visual_prefill": 976, "vision_encoder": 3584, "decoder_gemv": 1}[_workload]
    _shape = {
        "visual_prefill": (1080, 9728, 2560),
        "vision_encoder": (4320, 4096, 1024),
        "decoder_gemv": (1, 9728, 2560),
    }[_workload]
    for _profile in ("traffic", "latency"):
        STAGES[f"{_workload}_auto_apm_{_profile}"] = (
            _workload,
            _chunk,
            *_shape,
            _profile,
        )

EXPECTED_STAGE_ORDER = (
    "pre_selftest",
    "kernel_language_prefill_fixed512",
    "visual_prefill_fixed512",
    "visual_prefill_auto",
    "vision_encoder_fixed512",
    "vision_encoder_auto",
    "decoder_gemv_fixed",
    "decoder_gemv_auto",
    "visual_prefill_auto_apm_traffic",
    "visual_prefill_auto_apm_latency",
    "vision_encoder_auto_apm_traffic",
    "vision_encoder_auto_apm_latency",
    "decoder_gemv_auto_apm_traffic",
    "decoder_gemv_auto_apm_latency",
    "post_selftest",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fields(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))


def require(text: str, token: str) -> None:
    if token not in text:
        raise RuntimeError(f"missing gate: {token}")


def parse_linux_memory(uart: str) -> tuple[int, int]:
    matches = re.findall(r"Memory:\s+(\d+)K/(\d+)K available", uart)
    if len(matches) != 1:
        raise RuntimeError(f"expected one Linux memory line, found {len(matches)}")
    available_kib, total_kib = (int(value) for value in matches[0])
    # total_kib is fixed by the 16 GiB device tree.  available_kib changes with
    # the versioned initramfs size, so gate a conservative usable range rather
    # than one build-specific value.
    if total_kib != 16_776_192 or not (13_000_000 <= available_kib < total_kib):
        raise RuntimeError(
            f"Linux memory gate failed: available={available_kib} total={total_kib}"
        )
    return available_kib, total_kib


def protocol_window(uart: str) -> list[str]:
    lines = uart.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "DATAFLOW_MEASURE_BEGIN protocol=1 dry_run=0"]
    ends = [i for i, line in enumerate(lines) if line == "DATAFLOW_MEASURE_PASS protocol=1"]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise RuntimeError("expected one complete live DATAFLOW protocol window")
    return lines[starts[0] : ends[0] + 1]


def split_stages(lines: list[str]) -> dict[str, list[str]]:
    stages: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    for line in lines:
        if line.startswith("DATAFLOW_MEASURE_STAGE "):
            item = fields(line)
            label = item.get("label")
            state = item.get("state")
            if not label or state not in {"begin", "done"}:
                raise RuntimeError(f"bad stage marker: {line}")
            if state == "begin":
                if current is not None or label in stages:
                    raise RuntimeError(f"nested or duplicate stage: {label}")
                current = label
                stages[label] = []
                order.append(label)
            else:
                if current != label:
                    raise RuntimeError(f"stage completion mismatch: {line}")
                current = None
        elif current is not None:
            stages[current].append(line)
    if current is not None:
        raise RuntimeError(f"unterminated stage: {current}")
    if tuple(order) != EXPECTED_STAGE_ORDER:
        raise RuntimeError(f"stage order mismatch: {order}")
    return stages


def one_prefixed(lines: list[str], prefix: str) -> str:
    matches = [line for line in lines if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix!r}, found {len(matches)}")
    return matches[0]


def parse_benchmark_stage(label: str, lines: list[str]) -> dict[str, object]:
    case, expected_chunk, expected_m, expected_n, expected_k, profile = STAGES[label]
    policy = fields(one_prefixed(lines, "BENCH_CHUNK_POLICY "))
    result = fields(one_prefixed(lines, "BENCH_RESULT "))
    integer_expectations = {
        "m": expected_m,
        "n": expected_n,
        "k": expected_k,
        "chunk_m": expected_chunk,
        "repeats": 3,
    }
    if result.get("case") != case or result.get("result") != "PASS":
        raise RuntimeError(f"{label}: result identity/gate mismatch")
    for name, expected in integer_expectations.items():
        if int(result.get(name, "-1")) != expected:
            raise RuntimeError(f"{label}: {name} mismatch")
    if int(policy.get("chunk_m", "-1")) != expected_chunk:
        raise RuntimeError(f"{label}: chunk policy mismatch")
    expected_mode = "auto" if "_auto" in label else "manual"
    if policy.get("mode") != expected_mode:
        raise RuntimeError(f"{label}: expected policy mode {expected_mode}")

    chunks = math.ceil(expected_m / expected_chunk)
    compare = [line for line in lines if line.startswith("BENCH_STAGE ") and "name=compare state=done" in line]
    if len(compare) != chunks * 3:
        raise RuntimeError(f"{label}: compare coverage mismatch")
    if any(fields(line).get("max_abs") != "0" for line in compare):
        raise RuntimeError(f"{label}: nonzero numerical error")
    repeats = [fields(line) for line in lines if line.startswith("BENCH_REPEAT ")]
    if len(repeats) != 3 or any(int(item.get("chunks", "-1")) != chunks for item in repeats):
        raise RuntimeError(f"{label}: repeat/chunk coverage mismatch")

    parsed: dict[str, object] = {
        "case": case,
        "m": expected_m,
        "n": expected_n,
        "k": expected_k,
        "chunk_m": expected_chunk,
        "chunks": chunks,
        "repeats": 3,
        "average_kernel_us": float(result["average_kernel_us"]),
        "effective_gops": float(result["effective_gops"]),
        "average_prepare_us": float(result["average_prepare_us"]),
        "weight_prepare_us": int(result["weight_prepare_us"]),
        "modeled_ddr_read_bytes": float(result["modeled_ddr_read_bytes"]),
        "modeled_ddr_write_bytes": float(result["modeled_ddr_write_bytes"]),
        "max_abs_error": 0.0,
        "result": "PASS",
    }

    if profile is None:
        if any(line.startswith("APM_") for line in lines):
            raise RuntimeError(f"{label}: unexpected APM output")
        return parsed

    apm_result = fields(one_prefixed(lines, "APM_RESULT "))
    if apm_result.get("case") != case or apm_result.get("profile") != profile:
        raise RuntimeError(f"{label}: APM identity mismatch")
    samples = [fields(line) for line in lines if line.startswith("APM_SAMPLE ")]
    if len(samples) != chunks * 3 or any(item.get("overflow") != "0x00000000" for item in samples):
        raise RuntimeError(f"{label}: APM sample/overflow gate mismatch")
    totals: dict[int, int] = {}
    for line in lines:
        if not line.startswith("APM_TOTAL "):
            continue
        item = fields(line)
        if item.get("case") != case or item.get("profile") != profile:
            raise RuntimeError(f"{label}: foreign APM total")
        totals[int(item["counter"])] = int(item["value"])
    if set(totals) != set(range(10)):
        raise RuntimeError(f"{label}: APM counter coverage mismatch")
    parsed["apm"] = {
        "profile": profile,
        "global_cycles": int(apm_result["global_cycles"]),
        "derived_clock_mhz": float(apm_result["derived_clock_mhz"]),
        "totals": totals,
        "overflow": 0,
    }
    return parsed


def safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise RuntimeError(f"zero denominator: {label}")
    return numerator / denominator


def summarize(uart_path: Path, xsct_path: Path) -> dict[str, object]:
    uart = uart_path.read_text(errors="replace")
    xsct = xsct_path.read_text(errors="replace")
    for token in (
        "RAM_BOOT_DDR magic=0x53504434 ranks=2 total_bytes=17179869184",
        "RAM_BOOT_Q8Q8_PL_GATE=PASS",
        "RAM_BOOT_PS_PL_ISOLATION_REMOVAL=PASS",
        "RAM_BOOT_PS_PL_RESET_RELEASE=PASS",
        "RAM_BOOT_Q8Q8_AXI_GATE=PASS",
        "RAM_BOOT_POST_UBOOT_IMAGE_GATE=PASS",
        "RAM_BOOT_Q8Q8_JTAG_HANDOFF_PASS",
    ):
        require(xsct, token)
    available_memory_kib, total_memory_kib = parse_linux_memory(uart)
    for token in (
        "RAM_LINUX_UIO device=uio0 name=q8q8-gemm",
        "RAM_LINUX_PL_SELFTEST_PASS",
        "DATAFLOW_MEASURE_FINAL_CONTROL value=0x00000004",
    ):
        require(uart, token)
    if re.search(r"(?:BENCH|SELFTEST|RAM_LINUX_PL_SELFTEST|DATAFLOW_MEASURE)_FAIL", uart):
        raise RuntimeError("failure marker present in UART log")

    window = protocol_window(uart)
    stages = split_stages(window)
    for label, lines in stages.items():
        if lines.count("DATAFLOW_MEASURE_IDLE control=0x00000004") != 1:
            raise RuntimeError(f"{label}: per-stage ap_idle gate missing")
    for label in ("pre_selftest", "post_selftest"):
        if sum("SELFTEST_PASS m=2 n=4 k=32 outputs=8 max_abs=0" in line for line in stages[label]) != 1:
            raise RuntimeError(f"{label}: exact numerical self-test missing")

    parsed = {label: parse_benchmark_stage(label, stages[label]) for label in STAGES}
    pairs: dict[str, dict[str, float]] = {}
    for workload in ("visual_prefill", "vision_encoder", "decoder_gemv"):
        fixed = parsed[f"{workload}_{'fixed512' if workload != 'decoder_gemv' else 'fixed'}"]
        auto = parsed[f"{workload}_auto"]
        fixed_us = float(fixed["average_kernel_us"])
        auto_us = float(auto["average_kernel_us"])
        pairs[workload] = {
            "fixed_us": fixed_us,
            "auto_us": auto_us,
            "speedup": safe_ratio(fixed_us, auto_us, f"{workload} speedup"),
            "time_reduction_percent": (1.0 - safe_ratio(auto_us, fixed_us, workload)) * 100.0,
        }

    kernel_comparison: dict[str, dict[str, float]] = {}
    for workload, label in (
        ("language_prefill", "kernel_language_prefill_fixed512"),
        ("vision_encoder", "vision_encoder_fixed512"),
        ("decoder_gemv", "decoder_gemv_fixed"),
    ):
        candidate_us = float(parsed[label]["average_kernel_us"])
        kernel_comparison[workload] = {
            "pe16_baseline_us": PE16_BASELINE_US[workload],
            "dataflow_us": candidate_us,
            "speedup": safe_ratio(PE16_BASELINE_US[workload], candidate_us, workload),
        }

    apm_summary: dict[str, dict[str, object]] = {}
    for workload in ("visual_prefill", "vision_encoder", "decoder_gemv"):
        traffic = parsed[f"{workload}_auto_apm_traffic"]["apm"]
        latency = parsed[f"{workload}_auto_apm_latency"]["apm"]
        traffic_totals = {int(k): int(v) for k, v in traffic["totals"].items()}
        latency_totals = {int(k): int(v) for k, v in latency["totals"].items()}
        repeats = 3
        bytes_per_run = sum(traffic_totals[index] for index in (0, 2, 4, 6)) / repeats
        auto = parsed[f"{workload}_auto"]
        operations = 2.0 * int(auto["m"]) * int(auto["n"]) * int(auto["k"])
        kernel_us = float(auto["average_kernel_us"])
        # Both profiles aggregate over the same repeats and chunks. Therefore
        # average latency is total latency / total transactions; dividing the
        # transaction count by repeats here would incorrectly multiply by 3.
        apm_summary[workload] = {
            "bytes_per_run": bytes_per_run,
            "bandwidth_gbs": bytes_per_run / (kernel_us * 1000.0),
            "arithmetic_intensity_op_per_byte": safe_ratio(operations, bytes_per_run, workload),
            "traffic_totals": traffic_totals,
            "latency_totals": latency_totals,
            "latency_cycles": {
                "cci1_average_read": safe_ratio(latency_totals[0], traffic_totals[1], "cci1"),
                "cci1_max_read": float(latency_totals[2]),
                "cci2_average_read": safe_ratio(latency_totals[3], traffic_totals[3], "cci2"),
                "cci2_max_read": float(latency_totals[5]),
                "hp0_average_read": safe_ratio(latency_totals[6], traffic_totals[5], "hp0 read"),
                "hp0_average_write": safe_ratio(latency_totals[8], traffic_totals[7], "hp0 write"),
            },
        }

    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": "DATAFLOW_BOARD_PASS / NUMERIC_PASS / CHUNK_AB_PASS / APM_PASS",
        "board_state": {
            "programmed": True,
            "boot_mode": "JTAG volatile RAM Linux; no SD/QSPI write",
            "physical_ddr_bytes": 17179869184,
            "linux_available_memory_kib": available_memory_kib,
            "linux_total_memory_kib": total_memory_kib,
            "uio": "uio0:q8q8-gemm",
            "final_control": "0x00000004",
            "uart_shell": "alive",
        },
        "gates": {
            "jtag_boot": "PASS",
            "ddr_16gb_dual_rank": "PASS",
            "pl_configured": "PASS",
            "ps_pl_isolation_and_reset": "PASS",
            "axi_lite_idle": "PASS_AFTER_EACH_OF_15_STAGES",
            "initial_and_final_protocol_selftest": "PASS_MAX_ABS_0",
            "benchmark_numeric_samples": "PASS_MAX_ABS_0",
            "apm_overflow": 0,
            "task_fields": 0,
        },
        "benchmarks_by_stage": parsed,
        "kernel_comparison_vs_pe16": kernel_comparison,
        "chunk_policy_ab": pairs,
        "apm_auto_policy": apm_summary,
        "raw_evidence": {
            str(uart_path): {"bytes": uart_path.stat().st_size, "sha256": sha256(uart_path)},
            str(xsct_path): {"bytes": xsct_path.stat().st_size, "sha256": sha256(xsct_path)},
        },
        "limitations": [
            "These are accelerator microbenchmarks, not end-to-end RoboBrain inference times.",
            "Automatic chunk speedups isolate DMA job sizing and still reread weights on each kernel launch.",
            "APM latency uses total latency divided by total transactions across equal repeat counts.",
            "No board power or temperature result is claimed by this protocol.",
        ],
    }


def markdown(result: dict[str, object]) -> str:
    kernels = result["kernel_comparison_vs_pe16"]
    pairs = result["chunk_policy_ab"]
    apm = result["apm_auto_policy"]
    kernel_rows = []
    for name in ("language_prefill", "vision_encoder", "decoder_gemv"):
        item = kernels[name]
        kernel_rows.append(
            f"| {name} | {item['pe16_baseline_us'] / 1000:.3f} ms | "
            f"{item['dataflow_us'] / 1000:.3f} ms | {item['speedup']:.3f}× |"
        )
    pair_rows = []
    for name in ("visual_prefill", "vision_encoder", "decoder_gemv"):
        item = pairs[name]
        pair_rows.append(
            f"| {name} | {item['fixed_us'] / 1000:.3f} ms | "
            f"{item['auto_us'] / 1000:.3f} ms | {item['speedup']:.3f}× |"
        )
    apm_rows = []
    for name in ("visual_prefill", "vision_encoder", "decoder_gemv"):
        item = apm[name]
        latency = item["latency_cycles"]
        apm_rows.append(
            f"| {name} | {item['bytes_per_run'] / 1e6:.3f} MB | "
            f"{item['bandwidth_gbs']:.3f} GB/s | {item['arithmetic_intensity_op_per_byte']:.3f} | "
            f"{latency['cci1_average_read']:.2f} / {latency['cci2_average_read']:.2f} / "
            f"{latency['hp0_average_read']:.2f} |"
        )
    return f"""# ZCU102 Q8×Q8 PE16 DATAFLOW 板端验证

日期：2026-08-30  
状态：`{result['decision']}`

## DATAFLOW 与已冻结 PE16 基线

| 同形状工作负载 | PE16基线 | DATAFLOW | 加速比 |
|---|---:|---:|---:|
{chr(10).join(kernel_rows)}

## DMA分块A/B

| 完整工作负载 | 固定512 | 自动分块 | 加速比 |
|---|---:|---:|---:|
{chr(10).join(pair_rows)}

自动分块只改变单次调用的M行数；当前PL仍在每次调用重新读取权重，因此这里不把收益
解释为跨调用权重常驻。所有阶段前后均保持`ap_idle`，协议前后自检及全部数值抽检
`max_abs=0`。

## 自动分块APM

| 工作负载 | 每次流量 | 实效带宽 | OP/B | CCI1/CCI2/HP0平均读延迟(cycle) |
|---|---:|---:|---:|---:|
{chr(10).join(apm_rows)}

平均延迟按“累计延迟÷累计事务数”计算；两个profile均覆盖相同3次重复和相同chunk，
不能先把事务数除以3，否则结果会被错误放大3倍。

## 边界

- 结果仅是矩阵核microbenchmark，不是完整RoboBrain端到端推理。
- 未测板级功耗和温度，不给出能效结论。
- UART：`{next(iter(result['raw_evidence']))}`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--xsct", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.json_out, args.markdown_out):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite append-only result: {output}")
    result = summarize(args.uart, args.xsct)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.markdown_out.write_text(markdown(result))
    print(
        "DATAFLOW_BOARD_SUMMARY_PASS "
        + " ".join(
            f"{name}_chunk_speedup={result['chunk_policy_ab'][name]['speedup']:.6f}"
            for name in ("visual_prefill", "vision_encoder", "decoder_gemv")
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
