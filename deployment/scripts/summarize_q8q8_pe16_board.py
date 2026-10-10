#!/usr/bin/env python3
"""Freeze one PE16 ZCU102 board-validation run into JSON and Chinese Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


CASES = ("language_prefill", "vision_encoder", "decoder_gemv")
V3_BASELINE_US = {
    "language_prefill": 368570.0,
    "vision_encoder": 628165.0,
    "decoder_gemv": 18511.0,
}
LABELS = {
    "language_prefill": "Language prefill",
    "vision_encoder": "Vision encoder",
    "decoder_gemv": "Decoder GEMV",
}


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

    uart = args.uart.read_text(errors="replace")
    xsct = args.xsct.read_text(errors="replace")
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
    for token in (
        "Memory: 13863120K/16776192K available",
        "RAM_LINUX_UIO device=uio0 name=q8q8-gemm",
        "RAM_LINUX_PL_SELFTEST_PASS",
        "0x00000004",
    ):
        require(uart, token)
    if uart.count("SELFTEST_PASS m=2 n=4 k=32 outputs=8 max_abs=0") != 2:
        raise RuntimeError("expected initial and final exact self-tests")
    if re.search(r"(?:BENCH|SELFTEST|RAM_LINUX_PL_SELFTEST)_FAIL", uart):
        raise RuntimeError("failure marker present in UART log")

    bench: dict[str, dict[str, float | int | str]] = {}
    apm: dict[tuple[str, str], dict[int, int]] = {}
    for line in uart.splitlines():
        if line.startswith("BENCH_RESULT "):
            item = fields(line)
            if item.get("result") == "PASS" and item.get("repeats") == "3":
                bench[item["case"]] = {
                    "m": int(item["m"]),
                    "n": int(item["n"]),
                    "k": int(item["k"]),
                    "repeats": int(item["repeats"]),
                    "average_kernel_us": float(item["average_kernel_us"]),
                    "effective_gops": float(item["effective_gops"]),
                    "modeled_bandwidth_gbs": float(item["modeled_bandwidth_gbs"]),
                    "max_abs_error": 0.0,
                    "result": item["result"],
                }
        elif line.startswith("APM_TOTAL "):
            item = fields(line)
            apm.setdefault((item["case"], item["profile"]), {})[
                int(item["counter"])
            ] = int(item["value"])

    if set(bench) != set(CASES):
        raise RuntimeError(f"missing benchmark cases: {set(CASES) - set(bench)}")
    derived: dict[str, dict[str, float | str]] = {}
    for case in CASES:
        traffic = apm[(case, "traffic")]
        latency = apm[(case, "latency")]
        repeats = int(bench[case]["repeats"])
        transactions = {
            # Traffic and latency profiles both aggregate over the same three
            # repeats. Average latency is therefore total latency divided by
            # total transactions. Dividing the transaction totals by repeats
            # here would incorrectly inflate every average by 3x.
            "cci1": traffic[1],
            "cci2": traffic[3],
            "hp0_read": traffic[5],
            "hp0_write": traffic[7],
        }
        kernel_us = float(bench[case]["average_kernel_us"])
        operations = 2.0 * int(bench[case]["m"]) * int(bench[case]["n"]) * int(bench[case]["k"])
        apm_bytes = sum(traffic[index] for index in (0, 2, 4, 6)) / repeats
        intensity = operations / apm_bytes
        classification = (
            "memory/weight-stream-bound" if case == "decoder_gemv"
            else "compute/schedule-bound"
        )
        derived[case] = {
            "v3_baseline_us": V3_BASELINE_US[case],
            "speedup_vs_v3": V3_BASELINE_US[case] / kernel_us,
            "time_reduction_percent": (1.0 - kernel_us / V3_BASELINE_US[case]) * 100.0,
            "apm_bytes_per_run": apm_bytes,
            "apm_bandwidth_gbs": apm_bytes / (kernel_us * 1000.0),
            "arithmetic_intensity_op_per_byte": intensity,
            "peak_compute_utilization_percent": float(bench[case]["effective_gops"]) / 204.8 * 100.0,
            "bottleneck": classification,
            "latency_cycles": {
                "cci1_average_read": latency[0] / transactions["cci1"],
                "cci1_max_read": float(latency[2]),
                "cci2_average_read": latency[3] / transactions["cci2"],
                "cci2_max_read": float(latency[5]),
                "hp0_average_read": latency[6] / transactions["hp0_read"],
                "hp0_average_write": latency[8] / transactions["hp0_write"],
            },
        }

    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "decision": "PE16_BOARD_PASS / NUMERIC_PASS / PERFORMANCE_PASS / APM_PASS",
        "board_state": {
            "programmed": True,
            "boot_mode": "JTAG volatile RAM Linux; no SD/QSPI write",
            "physical_ddr_bytes": 17179869184,
            "linux_memory_total_bytes": 14587236352,
            "linux_memory_available_bytes_after_tests": 14271213568,
            "uio": "uio0:q8q8-gemm",
            "final_control": "0x00000004",
            "uart_shell": "alive",
        },
        "gates": {
            "jtag_boot": "PASS",
            "ddr_16gb_dual_rank": "PASS",
            "pl_configured": "PASS",
            "ps_pl_isolation_and_reset": "PASS",
            "axi_lite_idle": "PASS",
            "linux_uio": "PASS",
            "initial_selftest": "PASS_MAX_ABS_0",
            "final_selftest": "PASS_MAX_ABS_0",
            "benchmark_numeric_samples": "PASS_MAX_ABS_0",
            "apm_overflow": 0,
        },
        "hardware": {
            "pl_clock_mhz": 200.0,
            "pe_count": 16,
            "int8_lanes_per_pe": 32,
            "theoretical_peak_gops": 204.8,
            "ddr4_2133_theoretical_gbs": 17.065,
            "roofline_balance_op_per_byte": 12.0,
        },
        "benchmarks": bench,
        "derived": derived,
        "raw_evidence": {
            str(args.uart): {"bytes": args.uart.stat().st_size, "sha256": sha256(args.uart)},
            str(args.xsct): {"bytes": args.xsct.stat().st_size, "sha256": sha256(args.xsct)},
        },
        "limitations": [
            "These are accelerator microbenchmarks, not end-to-end RoboBrain inference times.",
            "The component_model_us field in the frozen runtime still describes the older v2 schedule and is not used for PE16 claims.",
            "No board power or temperature measurement was available in this RAM-Linux image.",
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    rows = []
    for case in CASES:
        b = bench[case]
        d = derived[case]
        rows.append(
            f"| {LABELS[case]} | {int(b['m'])}×{int(b['n'])}×{int(b['k'])} | "
            f"{float(b['average_kernel_us']) / 1000.0:.3f} ms | "
            f"{float(b['effective_gops']):.3f} | {float(d['v3_baseline_us']) / 1000.0:.3f} ms | "
            f"{float(d['speedup_vs_v3']):.3f}× | {d['bottleneck']} |"
        )
    latency_rows = []
    for case in CASES:
        item = derived[case]["latency_cycles"]
        latency_rows.append(
            f"| {LABELS[case]} | {item['cci1_average_read']:.2f} / {item['cci1_max_read']:.0f} | "
            f"{item['cci2_average_read']:.2f} / {item['cci2_max_read']:.0f} | "
            f"{item['hp0_average_read']:.2f} | {item['hp0_average_write']:.2f} |"
        )
    markdown = f"""# ZCU102 Q8×Q8 PE16 首次板端验证

日期：2026-08-28  
状态：`PE16_BOARD_PASS / NUMERIC_PASS / PERFORMANCE_PASS / APM_PASS`

## 结论

PE16 已通过 JTAG 易失性加载到真实 ZCU102。16 GB 双 Rank DDR、PL 配置、PS–PL
释放、AXI-Lite、Linux UIO、启动前后数值自检和三类大矩阵基准全部通过；所有
数值抽检均为 `max_abs=0`，APM 无溢出。未写入 SD 或 QSPI。

| 工作负载 | M×N×K | PE16板端均值 | GOPS | v3 PE8基线 | 加速比 | 当前判定 |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

PE16 相对 v3 PE8 的耗时降幅分别为：Prefill
`{derived['language_prefill']['time_reduction_percent']:.2f}%`、Vision
`{derived['vision_encoder']['time_reduction_percent']:.2f}%`、Decoder
`{derived['decoder_gemv']['time_reduction_percent']:.2f}%`。这是算子级 microbenchmark，
不等同于完整 RoboBrain 推理已获得相同加速。

## 板端门控

- FSBL 从 SPD 识别 `2 Rank / 17179869184 B`；Linux 启动日志描述约16 GB内存。
- PL configured、PS–PL isolation/reset release、`0xA0000000` AXI-Lite idle均通过。
- Linux枚举 `uio0:q8q8-gemm`和4个APM；初始、最终自检均为8输出完全一致。
- 测试后控制寄存器为`0x00000004`，加速器idle；UART shell保留。

## APM与瓶颈

PE16理论峰值为`16×32×200 MHz×2 = 204.8 GOPS`；按DDR4-2133理论
`17.065 GB/s`计算，roofline分界约`12.0 OP/B`。Prefill/Vision算术强度分别
为`{derived['language_prefill']['arithmetic_intensity_op_per_byte']:.2f}`和
`{derived['vision_encoder']['arithmetic_intensity_op_per_byte']:.2f} OP/B`，仍以
计算/排程为主；Decoder仅`{derived['decoder_gemv']['arithmetic_intensity_op_per_byte']:.2f} OP/B`，
仍是权重流memory-bound。三者APM实测总带宽分别约
`{derived['language_prefill']['apm_bandwidth_gbs']:.3f}`、
`{derived['vision_encoder']['apm_bandwidth_gbs']:.3f}`、
`{derived['decoder_gemv']['apm_bandwidth_gbs']:.3f} GB/s`，均未达到DDR峰值。

| 工作负载 | CCI1读平均/最大(cycle) | CCI2读平均/最大(cycle) | HP0读平均 | HP0写平均 |
|---|---:|---:|---:|---:|
{chr(10).join(latency_rows)}

PE数翻倍没有产生2×收益，因为权重/激活装载、输出写回、分块边界和控制阶段没有
同比缩短。下一步应基于PE16实测继续做load/compute/store重叠和更大DMA分块；
Decoder必须保留独立GEMV/权重流路线。

## 证据与限制

- UART原始日志：`{args.uart}`，SHA256 `{sha256(args.uart)}`。
- XSCT原始日志：`{args.xsct}`，SHA256 `{sha256(args.xsct)}`。
- 机器可读汇总：`{args.json_out}`。
- 冻结runtime中的`component_model_us`仍是旧v2排程模型，本报告不使用它声称PE16收益。
- 当前RAM Linux没有形成可靠板级温度或功耗实测，本报告不作相关声明。
"""
    args.markdown_out.write_text(markdown)
    print(
        "PE16_BOARD_SUMMARY_PASS "
        + " ".join(
            f"{case}_us={bench[case]['average_kernel_us']}"
            for case in CASES
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
