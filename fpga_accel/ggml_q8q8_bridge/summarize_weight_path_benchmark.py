#!/usr/bin/env python3
"""Freeze the A53 cacheable-DDR/UIO Q8_0 weight staging benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path


RESULT = re.compile(
    r"^WEIGHT_PATH_RESULT destination=(?P<destination>\S+) "
    r"case=(?P<case>\S+) n=(?P<n>\d+) k=(?P<k>\d+) bytes=(?P<bytes>\d+) "
    r"reference_ns=(?P<reference_ns>\d+) fast1_ns=(?P<fast1_ns>\d+) "
    r"fast4_ns=(?P<fast4_ns>\d+) fast1_speedup=(?P<fast1_speedup>[0-9.]+) "
    r"fast4_speedup=(?P<fast4_speedup>[0-9.]+) byte_equal=yes result=PASS$"
)
EXPECTED = {
    "llm_square_2560": (2560, 2560),
    "llm_ffn_9728": (9728, 2560),
    "vision_fc_4096": (4096, 1024),
}
DESTINATIONS = {"cacheable_ddr", "uio_noncached"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse(path: Path, binary_digest: str) -> dict[str, dict[str, object]]:
    text = path.read_text(errors="replace").replace("\r", "")
    for marker in (
        "WEIGHT_PATH_BEGIN version=1 repeats=3 destinations=cacheable_ddr,uio_noncached pl_control_access=no sd_qspi_access=no",
        "WEIGHT_PATH_PASS cases=3 destinations=2 byte_equal=yes pl_control_access=no sd_qspi_written=no",
        "UART_PUSH_DONE status=0",
    ):
        if marker not in text:
            raise RuntimeError(f"required marker missing: {marker}")
    if not re.search(rf"^{binary_digest}\s+/tmp/weight-path-benchmark$",
                     text, re.MULTILINE):
        raise RuntimeError("board binary digest missing or mismatched")
    records: dict[str, dict[str, object]] = {case: {} for case in EXPECTED}
    for line in text.splitlines():
        match = RESULT.match(line)
        if not match:
            continue
        raw = match.groupdict()
        case = raw.pop("case")
        destination = raw.pop("destination")
        if case not in EXPECTED or destination not in DESTINATIONS:
            raise RuntimeError("unexpected result")
        item = {
            key: float(value) if "speedup" in key else int(value)
            for key, value in raw.items()
        }
        if (item["n"], item["k"]) != EXPECTED[case]:
            raise RuntimeError("shape mismatch")
        if destination in records[case]:
            raise RuntimeError("duplicate result")
        for key in ("reference_ns", "fast1_ns", "fast4_ns"):
            item[key.replace("_ns", "_ms")] = item[key] / 1_000_000
        records[case][destination] = item
    if any(set(item) != DESTINATIONS for item in records.values()):
        raise RuntimeError("expected 3 shapes x 2 destinations")
    return records


def summarize(uart: Path, board_summary: Path, neon_summary: Path,
              binary: Path) -> dict[str, object]:
    digest = sha256(binary)
    cases = parse(uart, digest)
    board = json.loads(board_summary.read_text())
    neon = json.loads(neon_summary.read_text())
    stages = board["benchmarks_by_stage"]
    combined = {}
    mapping = {
        "language_prefill_m512": (
            "llm_ffn_9728", "language_prefill_m512",
            stages["kernel_language_prefill_fixed512"]["average_kernel_us"] / 1000,
        ),
        "visual_prefill_m1080": (
            "llm_ffn_9728", "visual_prefill_m1080",
            stages["visual_prefill_auto"]["average_kernel_us"] / 1000,
        ),
        "vision_encoder_m4320": (
            "vision_fc_4096", "vision_encoder_m4320",
            stages["vision_encoder_auto"]["average_kernel_us"] / 1000,
        ),
    }
    for name, (weight_case, activation_case, kernel_ms) in mapping.items():
        weight_ms = cases[weight_case]["uio_noncached"]["fast4_ms"]
        activation_ms = neon["cases"][activation_case]["activation_quantize_fast4"]["median_ms"]
        combined[name] = {
            "weight_stage_fast4_ms": weight_ms,
            "activation_quantize_fast4_ms": activation_ms,
            "kernel_ms": kernel_ms,
            "measured_prepare_components_ms": weight_ms + activation_ms,
            "measured_prepare_components_over_kernel":
                (weight_ms + activation_ms) / kernel_ms,
            "measured_components_plus_kernel_ms": weight_ms + activation_ms + kernel_ms,
        }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": (
            "FAST_UIO_WEIGHT_STAGING_PASS / BYTE_EXACT_3_OF_3 / "
            "LARGE_M_STAGING_FEASIBLE / NATIVE_Q8_PL_FRONTEND_DEFERRED"
        ),
        "measurement": {
            "platform": "ZCU102 Cortex-A53 RAM Linux",
            "destinations": ["cacheable DDR", "UIO non-cached reserved DDR"],
            "repeats": 3,
            "statistic": "median",
            "pl_control_access": False,
            "sd_qspi_written": False,
            "binary_sha256": digest,
        },
        "cases": cases,
        "combined_measured_components": combined,
        "limitations": [
            "Combined totals omit activation Q8_0 split/copy, output copy, backend scheduling, and GGML graph overhead.",
            "The benchmark mapped only UIO data memory and did not access accelerator control registers.",
            "M=1 decoder remains CPU-routed even though staging itself is faster.",
        ],
        "raw_evidence": {
            str(uart): {"bytes": uart.stat().st_size, "sha256": sha256(uart)},
            str(board_summary): {"bytes": board_summary.stat().st_size, "sha256": sha256(board_summary)},
            str(neon_summary): {"bytes": neon_summary.stat().st_size, "sha256": sha256(neon_summary)},
            str(binary): {"bytes": binary.stat().st_size, "sha256": digest},
        },
    }


def markdown(result: dict[str, object]) -> str:
    rows = []
    for case, destinations in result["cases"].items():
        cache = destinations["cacheable_ddr"]
        uio = destinations["uio_noncached"]
        rows.append(
            f"| {case} | {cache['bytes'] / 1e6:.2f} | {cache['reference_ms']:.3f} | "
            f"{cache['fast4_ms']:.3f} | {uio['reference_ms']:.3f} | "
            f"{uio['fast4_ms']:.3f} | {uio['fast4_speedup']:.2f}× |"
        )
    combined_rows = []
    for case, item in result["combined_measured_components"].items():
        combined_rows.append(
            f"| {case} | {item['weight_stage_fast4_ms']:.3f} | "
            f"{item['activation_quantize_fast4_ms']:.3f} | {item['kernel_ms']:.3f} | "
            f"{item['measured_prepare_components_over_kernel'] * 100:.2f}% |"
        )
    return f"""# ZCU102 Q8_0权重数据路径实测

结论：`{result['decision']}`。

| 权重shape | 原生Q8_0/MB | cache参考/ms | cache四核/ms | UIO参考/ms | UIO四核/ms | UIO加速 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

三种shape在cacheable DDR和真实UIO非缓存区均完成全量逐字节对照。基准只映射UIO数据区，
未访问PL控制寄存器，未启动计算核，未写SD/QSPI。

## 与已测kernel的组合口径

| workload | 权重stage/ms | 激活NEON四核/ms | PL kernel/ms | 已测准备组件/kernel |
|---|---:|---:|---:|---:|
{chr(10).join(combined_rows)}

- 对大M算子，快速权重stage与激活量化合计约为当前kernel的4.30–8.44%，可先用于完整block接入。
- 原生交织Q8_0 PL前端仍可能进一步降低带宽和PS占用，但不再是首个Transformer Block的前置门。
- 下一步必须把相同快速拆分用于每个activation chunk；当前runtime内部仍调用旧串行拆分。
- 上表尚未包含activation拆分/拷贝、输出拷贝、backend调度和GGML图开销，不能当端到端延迟。
- Decoder继续CPU回退，避免为M=1支付无收益的PL kernel与stage开销。

原始UART日志：`{next(iter(result['raw_evidence']))}`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uart", type=Path, required=True)
    parser.add_argument("--board-summary", type=Path, required=True)
    parser.add_argument("--neon-summary", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.json_out, args.markdown_out):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite append-only output: {output}")
    result = summarize(args.uart, args.board_summary, args.neon_summary,
                       args.binary)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.markdown_out.write_text(markdown(result))
    print(
        "GGML_Q8Q8_WEIGHT_PATH_PASS shapes=3 destinations=2 byte_equal=yes "
        "pl_control_access=no sd_qspi_written=no"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
