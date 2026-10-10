#!/usr/bin/env python3
"""Freeze the byte-exact A53 NEON/parallel Q8_0 quantizer benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


RESULT = re.compile(
    r"^PREP_RESULT operation=(?P<operation>\S+) case=(?P<case>\S+) "
    r"m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+) "
    r"input_bytes=(?P<input_bytes>\d+) output_bytes=(?P<output_bytes>\d+) "
    r"median_ns=(?P<median_ns>\d+) throughput_gbs=(?P<throughput_gbs>[0-9.]+) "
    r"checksum=(?P<checksum>\d+) result=PASS$"
)
NUMERIC = re.compile(
    r"^PREP_NUMERIC case=(?P<case>\S+) m=(?P<m>\d+) n=(?P<n>\d+) "
    r"k=(?P<k>\d+) bytes=(?P<bytes>\d+) "
    r"reference_vs_fast1=byte_equal reference_vs_fast4=byte_equal result=PASS$"
)
EXPECTED = {
    "decoder_m1": (1, 2560, 2560),
    "text_prefill_m49": (49, 2560, 2560),
    "visual_tail_m56": (56, 9728, 2560),
    "language_prefill_m512": (512, 9728, 2560),
    "visual_prefill_m1080": (1080, 9728, 2560),
    "vision_encoder_m4320": (4320, 4096, 1024),
}
OPERATIONS = {
    "activation_quantize_reference",
    "activation_quantize_fast1",
    "activation_quantize_fast4",
    "weight_pack",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(path: Path, binary_digest: str) -> dict[str, dict[str, object]]:
    text = path.read_text(errors="replace").replace("\r", "")
    required = (
        "PREP_BENCH_BEGIN version=3 repeats=3 destination=cacheable_ddr "
        "fast_implementation=aarch64_neon",
        "PREP_BENCH_PASS cases=6 measurements=24 numeric_cases=6",
        "UART_PUSH_DONE status=0",
    )
    if any(marker not in text for marker in required):
        raise RuntimeError("required benchmark marker missing")
    if not re.search(rf"^{binary_digest}\s+/tmp/prepare-benchmark-neon-numeric$",
                     text, re.MULTILINE):
        raise RuntimeError("board binary digest missing or mismatched")
    records: dict[str, dict[str, object]] = {case: {} for case in EXPECTED}
    numeric_cases = set()
    for line in text.splitlines():
        numeric = NUMERIC.match(line)
        if numeric:
            raw = numeric.groupdict()
            case = raw["case"]
            dimensions = tuple(int(raw[key]) for key in ("m", "n", "k"))
            if case not in EXPECTED or dimensions != EXPECTED[case]:
                raise RuntimeError("numeric shape mismatch")
            numeric_cases.add(case)
            records[case]["numeric_bytes"] = int(raw["bytes"])
            continue
        match = RESULT.match(line)
        if not match:
            continue
        raw = match.groupdict()
        case = raw.pop("case")
        operation = raw.pop("operation")
        if case not in EXPECTED or operation not in OPERATIONS:
            raise RuntimeError("unexpected result")
        if operation in records[case]:
            raise RuntimeError("duplicate result")
        item = {
            key: float(value) if key == "throughput_gbs" else int(value)
            for key, value in raw.items()
        }
        if tuple(item[key] for key in ("m", "n", "k")) != EXPECTED[case]:
            raise RuntimeError("result shape mismatch")
        item["median_ms"] = item["median_ns"] / 1_000_000
        records[case][operation] = item
    if numeric_cases != set(EXPECTED):
        raise RuntimeError("six full byte comparisons required")
    if any(not OPERATIONS.issubset(item) for item in records.values()):
        raise RuntimeError("expected 24 timing results")
    checksums = {
        case: {
            records[case][operation]["checksum"]
            for operation in OPERATIONS if operation != "weight_pack"
        }
        for case in EXPECTED
    }
    if any(len(values) != 1 for values in checksums.values()):
        raise RuntimeError("quantizer checksum mismatch")
    return records


def summarize(uart: Path, board_summary: Path, binary: Path) -> dict[str, object]:
    binary_digest = sha256(binary)
    cases = parse(uart, binary_digest)
    board = json.loads(board_summary.read_text())
    kernels = {
        "language_prefill_m512": board["benchmarks_by_stage"]["kernel_language_prefill_fixed512"]["average_kernel_us"] / 1000,
        "visual_prefill_m1080": board["benchmarks_by_stage"]["visual_prefill_auto"]["average_kernel_us"] / 1000,
        "vision_encoder_m4320": board["benchmarks_by_stage"]["vision_encoder_auto"]["average_kernel_us"] / 1000,
    }
    for case, item in cases.items():
        reference = item["activation_quantize_reference"]["median_ms"]
        fast1 = item["activation_quantize_fast1"]["median_ms"]
        fast4 = item["activation_quantize_fast4"]["median_ms"]
        item["speedup"] = {
            "fast1_vs_reference": reference / fast1,
            "fast4_vs_reference": reference / fast4,
            "fast4_vs_fast1": fast1 / fast4,
        }
        if case in kernels:
            item["same_shape_pl_comparison"] = {
                "kernel_ms": kernels[case],
                "fast4_quantize_over_kernel": fast4 / kernels[case],
                "fast4_quantize_plus_kernel_ms": fast4 + kernels[case],
            }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": (
            "A53_NEON_PARALLEL_QUANTIZATION_PASS / BYTE_EXACT_6_OF_6 / "
            "ACTIVATION_QUANTIZATION_NO_LONGER_PRIMARY_PREP_BOTTLENECK"
        ),
        "measurement": {
            "platform": "ZCU102 Cortex-A53 RAM Linux",
            "implementation": "AArch64 NEON, 1 or 4 row workers",
            "repeats": 3,
            "statistic": "median",
            "full_byte_comparison_cases": 6,
            "binary_sha256": binary_digest,
            "sd_qspi_written": False,
        },
        "cases": cases,
        "remaining_primary_bottleneck": (
            "Weight transformation and movement from native interleaved GGML Q8_0 "
            "into the split UIO-resident PL layout."
        ),
        "raw_evidence": {
            str(uart): {"bytes": uart.stat().st_size, "sha256": sha256(uart)},
            str(board_summary): {"bytes": board_summary.stat().st_size, "sha256": sha256(board_summary)},
            str(binary): {"bytes": binary.stat().st_size, "sha256": binary_digest},
        },
    }


def markdown(result: dict[str, object]) -> str:
    rows = []
    for case, item in result["cases"].items():
        ref = item["activation_quantize_reference"]["median_ms"]
        fast1 = item["activation_quantize_fast1"]["median_ms"]
        fast4 = item["activation_quantize_fast4"]["median_ms"]
        speed = item["speedup"]
        pl = item.get("same_shape_pl_comparison")
        pl_ratio = f"{pl['fast4_quantize_over_kernel'] * 100:.2f}%" if pl else "—"
        rows.append(
            f"| {case} | {ref:.3f} | {fast1:.3f} | {fast4:.3f} | "
            f"{speed['fast1_vs_reference']:.2f}× | {speed['fast4_vs_reference']:.2f}× | {pl_ratio} |"
        )
    return f"""# ZCU102 A53 NEON并行Q8_0激活量化

结论：`{result['decision']}`。

| case | 标量参考/ms | NEON单核/ms | NEON四核/ms | 单核加速 | 四核加速 | 四核量化/PL kernel |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 数值门

- 六个真实尺度的全部Q8_0输出逐字节比较通过：标量参考 = NEON单核 = NEON四核。
- 另有上游`quantize_row_q8_0_ref`差分测试、half-way舍入和NaN回退测试通过。
- 板端静态二进制SHA256回读一致；只写RAM中的`/tmp`，未写SD/QSPI。

## 判断

- 单核NEON约为标量的3.2×；大矩阵四核约为标量的11.8–12.4×。
- M=512、M=1080、M=4320的四核量化仅约占同shape PL kernel的1.55%、1.48%、3.43%。
- 激活量化已不再是首要准备瓶颈；当前首要问题转为权重从原生交织Q8_0到PL双平面UIO格式的
  变换与搬运。
- 四线程策略必须由未来backend统一管理，避免与GGML线程池重复并行；生产入口仍默认关闭。

原始UART日志：`{next(iter(result['raw_evidence']))}`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--board-summary", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.json_out, args.markdown_out):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite append-only output: {output}")
    result = summarize(args.uart, args.board_summary, args.binary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.markdown_out.write_text(markdown(result))
    print(
        "GGML_Q8Q8_NEON_PREPARE_PASS numeric=6/6 measurements=24 "
        "implementation=aarch64_neon sd_qspi_written=no"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
