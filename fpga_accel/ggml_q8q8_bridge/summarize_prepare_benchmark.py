#!/usr/bin/env python3
"""Freeze and summarize the A53 GGML-to-PL preparation benchmark."""

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

EXPECTED = {
    "decoder_m1": (1, 2560, 2560),
    "text_prefill_m49": (49, 2560, 2560),
    "visual_tail_m56": (56, 9728, 2560),
    "language_prefill_m512": (512, 9728, 2560),
    "visual_prefill_m1080": (1080, 9728, 2560),
    "vision_encoder_m4320": (4320, 4096, 1024),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_uart(path: Path, binary_digest: str) -> dict[str, dict[str, dict[str, object]]]:
    text = path.read_text(errors="replace").replace("\r", "")
    if "PREP_BENCH_PASS cases=6 measurements=12" not in text:
        raise RuntimeError("benchmark pass marker missing")
    if "UART_PUSH_PASS" in text:
        raise RuntimeError("host-only marker unexpectedly present in raw UART log")
    if not re.search(rf"^{binary_digest}\s+/tmp/prepare-benchmark$", text, re.MULTILINE):
        raise RuntimeError("board-side binary digest missing or mismatched")
    records: dict[str, dict[str, dict[str, object]]] = {}
    for line in text.splitlines():
        match = RESULT.match(line)
        if not match:
            continue
        raw = match.groupdict()
        case = raw.pop("case")
        operation = raw.pop("operation")
        if case not in EXPECTED or operation not in {
            "activation_quantize", "weight_pack"
        }:
            raise RuntimeError(f"unexpected result: {case}/{operation}")
        item: dict[str, object] = {}
        for key, value in raw.items():
            item[key] = float(value) if key == "throughput_gbs" else int(value)
        dimensions = (item["m"], item["n"], item["k"])
        if dimensions != EXPECTED[case]:
            raise RuntimeError(f"shape mismatch for {case}: {dimensions}")
        if operation in records.setdefault(case, {}):
            raise RuntimeError(f"duplicate result: {case}/{operation}")
        item["median_ms"] = item["median_ns"] / 1_000_000
        records[case][operation] = item
    if len(records) != len(EXPECTED) or any(len(item) != 2 for item in records.values()):
        raise RuntimeError("expected exactly 6 cases and 12 results")
    return records


def summarize(uart: Path, board_summary: Path, binary: Path) -> dict[str, object]:
    binary_digest = sha256(binary)
    records = parse_uart(uart, binary_digest)
    board = json.loads(board_summary.read_text())
    stages = board["benchmarks_by_stage"]
    comparisons = {
        "language_prefill_m512": stages["kernel_language_prefill_fixed512"]["average_kernel_us"],
        "visual_prefill_m1080": stages["visual_prefill_auto"]["average_kernel_us"],
        "vision_encoder_m4320": stages["vision_encoder_auto"]["average_kernel_us"],
    }
    for case, kernel_us in comparisons.items():
        quantize_ms = records[case]["activation_quantize"]["median_ms"]
        kernel_ms = kernel_us / 1000.0
        records[case]["same_shape_pl_comparison"] = {
            "kernel_ms": kernel_ms,
            "quantize_over_kernel": quantize_ms / kernel_ms,
            "cached_weight_quantize_plus_kernel_ms": quantize_ms + kernel_ms,
            "uncached_weight_prepare_plus_quantize_plus_kernel_ms": (
                records[case]["weight_pack"]["median_ms"] + quantize_ms + kernel_ms
            ),
        }
    old_weight = {
        "language_prefill_m512": stages["kernel_language_prefill_fixed512"]["weight_prepare_us"],
        "visual_prefill_m1080": stages["visual_prefill_auto"]["weight_prepare_us"],
        "vision_encoder_m4320": stages["vision_encoder_auto"]["weight_prepare_us"],
    }
    for case, uio_us in old_weight.items():
        cacheable_ms = records[case]["weight_pack"]["median_ms"]
        records[case]["uio_weight_prepare_comparison"] = {
            "existing_uio_weight_prepare_ms": uio_us / 1000.0,
            "cacheable_ddr_weight_pack_ms": cacheable_ms,
            "uio_over_cacheable_ratio": (uio_us / 1000.0) / cacheable_ms,
        }
    return {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds"),
        "decision": (
            "PREPARE_COST_MEASURED / WEIGHT_REPACK_PER_CALL_REJECTED / "
            "ACTIVATION_QUANTIZATION_OPTIMIZATION_REQUIRED / DECODER_CPU_FALLBACK"
        ),
        "measurement": {
            "platform": "ZCU102 Cortex-A53 RAM Linux",
            "repeats": 3,
            "statistic": "median",
            "destination": "cacheable DDR malloc buffers",
            "board_binary_sha256": binary_digest,
            "sd_qspi_written": False,
        },
        "cases": records,
        "existing_uio_prepare_scope": (
            "The prior board benchmark includes deterministic activation generation and "
            "packing into the UIO reserved arena; it is not interchangeable with the "
            "new isolated cacheable-DDR quantizer measurement."
        ),
        "architecture_actions": [
            "Do not dispatch M=1 decoder GEMV to the current PL kernel.",
            "Never split and repack an unchanged weight tensor on every invocation.",
            "Add a bounded persistent/prepacked weight policy or make the PL stream native interleaved GGML Q8_0.",
            "Optimize FP32-to-Q8_0 activation quantization with AArch64 NEON and row parallelism before end-to-end claims.",
            "Keep the production offload entry disabled by default until numerical and timing gates pass.",
        ],
        "raw_evidence": {
            str(uart): {"bytes": uart.stat().st_size, "sha256": sha256(uart)},
            str(board_summary): {"bytes": board_summary.stat().st_size, "sha256": sha256(board_summary)},
            str(binary): {"bytes": binary.stat().st_size, "sha256": binary_digest},
        },
    }


def markdown(result: dict[str, object]) -> str:
    records = result["cases"]
    rows = []
    for case, item in records.items():
        quant = item["activation_quantize"]
        weight = item["weight_pack"]
        comparison = item.get("same_shape_pl_comparison")
        kernel = f"{comparison['kernel_ms']:.3f}" if comparison else "—"
        ratio = f"{comparison['quantize_over_kernel'] * 100:.1f}%" if comparison else "—"
        rows.append(
            f"| {case} | {quant['m']} | {quant['n']} | {quant['k']} | "
            f"{quant['median_ms']:.3f} | {weight['median_ms']:.3f} | {kernel} | {ratio} |"
        )
    return f"""# ZCU102 GGML→PL准备开销实测

结论：`{result['decision']}`。

| case | M | N | K | 激活FP32→Q8_0 | 权重拆分（cacheable DDR） | 同shape PL kernel | 量化/kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

单位均为ms，三次取中位数。同shape对照只列入M/N/K完全一致的三项。板端程序经UART写入
`/tmp`并执行，SHA256回读一致；未写SD或QSPI。

## 关键判断

- `M=512`与`M=1080`的激活量化约占当前PL kernel的20.6%；`M=4320` Encoder约占48.3%。
- `N=9728,K=2560`权重在普通可缓存DDR中拆分仍需约147–149 ms；既有UIO实测约354 ms。
  不允许生产路径在每次算子调用时重复这一步。
- 需要比较两条路线：受控的预打包权重驻留/复用，或PL直接消费原生34-byte GGML Q8_0块。
- 当前decoder GEMV板端没有kernel收益，保持CPU回退，避免权重准备反向拖慢。
- 下一门是NEON/多线程激活量化与权重数据路径原型；未完成前不宣称端到端加速。

## 口径边界

本次新测量的目标是普通cacheable DDR；既有`average_prepare_us`还包含确定性激活生成和向UIO
保留区打包，二者不能直接相减或当成相同测量。全部结果仍是算子桥接开销，不是完整模型延迟。

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
        "GGML_Q8Q8_PREPARE_BOARD_PASS cases=6 measurements=12 "
        "weight_repack_per_call=REJECT decoder_route=CPU sd_qspi_written=no"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
