#!/usr/bin/env python3
"""Freeze the A53 UIO-to-DDR output readback benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


RESULT = re.compile(
    r"^OUTPUT_PATH_RESULT case=(?P<case>\S+) m=(?P<m>\d+) n=(?P<n>\d+) "
    r"bytes=(?P<bytes>\d+) reference_ns=(?P<reference_ns>\d+) "
    r"fast1_ns=(?P<fast1_ns>\d+) fast4_ns=(?P<fast4_ns>\d+) "
    r"fast1_speedup=(?P<fast1_speedup>[0-9.]+) "
    r"fast4_speedup=(?P<fast4_speedup>[0-9.]+) byte_equal=yes result=PASS$"
)
EXPECTED = {
    "language_m512": (512, 9728),
    "visual_chunk_m976": (976, 9728),
    "vision_chunk_m3584": (3584, 4096),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(uart: Path, board_summary: Path, binary: Path) -> dict[str, object]:
    digest = sha256(binary)
    text = uart.read_text(errors="replace").replace("\r", "")
    for marker in (
        "OUTPUT_PATH_BEGIN version=1 repeats=3 source=uio_noncached pl_control_access=no sd_qspi_access=no",
        "OUTPUT_PATH_PASS cases=3 byte_equal=yes pl_control_access=no sd_qspi_written=no",
        "UART_PUSH_DONE status=0",
    ):
        if marker not in text:
            raise RuntimeError(f"missing marker: {marker}")
    if not re.search(rf"^{digest}\s+/tmp/output-path-benchmark$", text,
                     re.MULTILINE):
        raise RuntimeError("binary digest mismatch")
    cases = {}
    for line in text.splitlines():
        match = RESULT.match(line)
        if not match:
            continue
        raw = match.groupdict()
        case = raw.pop("case")
        if case not in EXPECTED or case in cases:
            raise RuntimeError("unexpected or duplicate case")
        item = {
            key: float(value) if "speedup" in key else int(value)
            for key, value in raw.items()
        }
        if (item["m"], item["n"]) != EXPECTED[case]:
            raise RuntimeError("shape mismatch")
        for key in ("reference_ns", "fast1_ns", "fast4_ns"):
            item[key.replace("_ns", "_ms")] = item[key] / 1_000_000
        item["fast4_effective_gbs"] = item["bytes"] / item["fast4_ns"]
        cases[case] = item
    if set(cases) != set(EXPECTED):
        raise RuntimeError("expected three output cases")
    board = json.loads(board_summary.read_text())
    comparisons = {
        "language_m512": board["benchmarks_by_stage"]["kernel_language_prefill_fixed512"]["average_kernel_us"] / 1000,
        "visual_chunk_m976": board["benchmarks_by_stage"]["visual_prefill_auto"]["average_kernel_us"] / 1000,
        "vision_chunk_m3584": board["benchmarks_by_stage"]["vision_encoder_auto"]["average_kernel_us"] / 1000,
    }
    for case, kernel_ms in comparisons.items():
        cases[case]["comparison"] = {
            "workload_kernel_ms": kernel_ms,
            "single_chunk_fast4_readback_over_workload_kernel":
                cases[case]["fast4_ms"] / kernel_ms,
        }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": (
            "FAST_UIO_READBACK_PASS / BYTE_EXACT_3_OF_3 / "
            "PER_OP_PS_ROUNDTRIP_REMAINS_MAJOR / PL_RESIDENT_BLOCK_PIPELINE_REQUIRED"
        ),
        "measurement": {
            "platform": "ZCU102 Cortex-A53 RAM Linux",
            "source": "UIO non-cached reserved DDR",
            "repeats": 3,
            "statistic": "median",
            "pl_control_access": False,
            "sd_qspi_written": False,
            "binary_sha256": digest,
        },
        "cases": cases,
        "architecture_action": (
            "Keep intermediate tensors in PL-accessible DDR and fuse/requantize "
            "adjacent operators; do not return every MUL_MAT output to PS."
        ),
        "limitations": [
            "Visual and vision records time the largest auto chunk, not the sum of all chunks.",
            "No PL kernel was started; only UIO data memory was read.",
        ],
        "raw_evidence": {
            str(uart): {"bytes": uart.stat().st_size, "sha256": sha256(uart)},
            str(board_summary): {"bytes": board_summary.stat().st_size, "sha256": sha256(board_summary)},
            str(binary): {"bytes": binary.stat().st_size, "sha256": digest},
        },
    }


def markdown(result: dict[str, object]) -> str:
    rows = []
    for case, item in result["cases"].items():
        rows.append(
            f"| {case} | {item['bytes'] / 1e6:.2f} | {item['reference_ms']:.3f} | "
            f"{item['fast1_ms']:.3f} | {item['fast4_ms']:.3f} | "
            f"{item['fast4_speedup']:.2f}× | {item['fast4_effective_gbs']:.3f} | "
            f"{item['comparison']['single_chunk_fast4_readback_over_workload_kernel'] * 100:.2f}% |"
        )
    return f"""# ZCU102 UIO输出回读实测

结论：`{result['decision']}`。

| case | 输出/MB | 逐字节参考/ms | NEON单核/ms | NEON四核/ms | 四核加速 | 四核GB/s | 单chunk回读/workload kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

三种输出均完成全量逐字节一致性验证。基准只读取UIO数据区，没有访问PL控制寄存器、启动kernel
或写SD/QSPI。

## 设计结论

- NEON四核消除了旧逐字节实现的灾难性开销，但非缓存UIO回读仍只有约0.39–0.45 GB/s。
- M=512完整输出回读仍相当于同shape kernel的28.05%；视觉/Encoder表中只计最大chunk，
  完整workload还有尾chunk，不能把该比例误当完整回读总量。
- 若每个MUL_MAT都回到PS，输出回读会成为端到端主要开销。因此完整Block不能只是“逐算子RPC”。
- 下一架构门：中间张量驻留PL可访问DDR，融合RMSNorm/残差/SiLU或在PL端直接requantize，
  只在Block边界或确有CPU消费者时回读。

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
    print("GGML_Q8Q8_OUTPUT_PATH_PASS shapes=3 byte_equal=yes pl_control_access=no sd_qspi_written=no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
