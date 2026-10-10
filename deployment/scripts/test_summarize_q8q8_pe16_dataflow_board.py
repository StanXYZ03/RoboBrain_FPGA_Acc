#!/usr/bin/env python3
"""Offline parser/formula tests for the DATAFLOW board summarizer."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "deployment/scripts/summarize_q8q8_pe16_dataflow_board.py"
SPEC = importlib.util.spec_from_file_location("dataflow_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def stage_lines(label: str) -> list[str]:
    lines = [f"DATAFLOW_MEASURE_STAGE label={label} state=begin"]
    if label in {"pre_selftest", "post_selftest"}:
        lines.append("SELFTEST_PASS m=2 n=4 k=32 outputs=8 max_abs=0")
    else:
        case, chunk, m, n, k, profile = MODULE.STAGES[label]
        mode = "auto" if "_auto" in label else "manual"
        lines.append(f"BENCH_CHUNK_POLICY mode={mode} chunk_m={chunk}")
        chunks = (m + chunk - 1) // chunk
        for repeat in range(1, 4):
            for part in range(1, chunks + 1):
                lines.append(
                    f"BENCH_STAGE case={case} repeat={repeat} chunk={part} "
                    "name=compare state=done samples=12 max_abs=0"
                )
                if profile is not None:
                    lines.append(
                        f"APM_SAMPLE case={case} repeat={repeat} chunk={part} "
                        f"profile={profile} global_cycles=100 overflow=0x00000000 "
                        "derived_clock_mhz=200.000000"
                    )
            lines.append(
                f"BENCH_REPEAT case={case} repeat={repeat} chunks={chunks} "
                "kernel_us=500 prepare_us=10"
            )
        average = 500.0 if "_auto" in label else 1000.0
        lines.append(
            f"BENCH_RESULT case={case} m={m} n={n} k={k} chunk_m={chunk} "
            f"repeats=3 average_kernel_us={average:.3f} effective_gops=100.0 "
            "average_prepare_us=10.0 weight_prepare_us=20 "
            "modeled_ddr_read_bytes=10000 modeled_ddr_write_bytes=2000 result=PASS"
        )
        if profile is not None:
            lines.append(
                f"APM_RESULT case={case} profile={profile} repeats=3 "
                "global_cycles=300 derived_clock_mhz=200.000000"
            )
            traffic = [3000, 300, 6000, 600, 9000, 900, 12000, 1200, 1500, 1800]
            latency = [3000, 0, 50, 12000, 0, 60, 27000, 0, 48000, 0]
            values = traffic if profile == "traffic" else latency
            for counter, value in enumerate(values):
                aggregation = "max" if profile == "latency" and counter in {2, 5} else "sum"
                lines.append(
                    f"APM_TOTAL case={case} profile={profile} counter={counter} "
                    f"slot=1 metric=1 name=synthetic value={value} aggregation={aggregation}"
                )
    lines.append("DATAFLOW_MEASURE_IDLE control=0x00000004")
    lines.append(f"DATAFLOW_MEASURE_STAGE label={label} state=done")
    return lines


def make_uart() -> str:
    lines = [
        "Memory: 13863056K/16776192K available",
        "RAM_LINUX_UIO device=uio0 name=q8q8-gemm",
        "RAM_LINUX_PL_SELFTEST_PASS",
        "DATAFLOW_MEASURE_BEGIN protocol=1 dry_run=0",
    ]
    for label in MODULE.EXPECTED_STAGE_ORDER:
        lines.extend(stage_lines(label))
    lines.extend(
        [
            "DATAFLOW_MEASURE_FINAL_CONTROL value=0x00000004",
            "DATAFLOW_MEASURE_PASS protocol=1",
        ]
    )
    return "\n".join(lines) + "\n"


def make_xsct() -> str:
    return "\n".join(
        [
            "RAM_BOOT_DDR magic=0x53504434 ranks=2 total_bytes=17179869184",
            "RAM_BOOT_Q8Q8_PL_GATE=PASS",
            "RAM_BOOT_PS_PL_ISOLATION_REMOVAL=PASS",
            "RAM_BOOT_PS_PL_RESET_RELEASE=PASS",
            "RAM_BOOT_Q8Q8_AXI_GATE=PASS",
            "RAM_BOOT_POST_UBOOT_IMAGE_GATE=PASS",
            "RAM_BOOT_Q8Q8_JTAG_HANDOFF_PASS",
        ]
    ) + "\n"


def expect_failure(uart: str, xsct: Path, temp: Path, label: str) -> None:
    path = temp / f"{label}.log"
    path.write_text(uart)
    try:
        MODULE.summarize(path, xsct)
    except RuntimeError:
        return
    raise AssertionError(f"negative gate did not fail: {label}")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        uart = temp / "uart.log"
        xsct = temp / "xsct.log"
        uart_text = make_uart()
        uart.write_text(uart_text)
        xsct.write_text(make_xsct())
        result = MODULE.summarize(uart, xsct)
        assert result["decision"].startswith("DATAFLOW_BOARD_PASS")
        assert result["chunk_policy_ab"]["visual_prefill"]["speedup"] == 2.0
        latency = result["apm_auto_policy"]["visual_prefill"]["latency_cycles"]
        assert latency["cci1_average_read"] == 10.0
        assert latency["cci2_average_read"] == 20.0
        assert latency["hp0_average_read"] == 30.0
        assert latency["hp0_average_write"] == 40.0
        assert "累计延迟÷累计事务数" in MODULE.markdown(result)

        expect_failure(
            uart_text.replace("DATAFLOW_MEASURE_IDLE control=0x00000004\n", "", 1),
            xsct,
            temp,
            "missing_idle",
        )
        expect_failure(
            uart_text.replace("max_abs=0", "max_abs=1", 1),
            xsct,
            temp,
            "numeric_error",
        )
        expect_failure(
            uart_text.replace("overflow=0x00000000", "overflow=0x00000001", 1),
            xsct,
            temp,
            "apm_overflow",
        )
        expect_failure(
            uart_text.replace("13863056K/16776192K", "13863056K/4194304K"),
            xsct,
            temp,
            "wrong_linux_memory_total",
        )
    print("DATAFLOW_BOARD_SUMMARIZER_TEST_PASS cases=5 latency_formula=total_over_total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
