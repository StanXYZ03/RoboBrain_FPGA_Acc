#!/usr/bin/env python3
"""Audit existing GGML profile records against the Q8xQ8 bridge contract."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path


LINE = re.compile(
    r"^GGML_OP_PROFILE graph=(?P<graph>\d+) node=(?P<node>\d+) "
    r"op=(?P<op>\S+) name=(?P<name>.*?) out_type=(?P<out_type>\S+) "
    r"out_ne=(?P<out_ne>\S+) src0=(?P<src0>.*?) src0_type=(?P<src0_type>\S+) "
    r"src0_ne=(?P<src0_ne>\S+) src1=(?P<src1>.*?) src1_type=(?P<src1_type>\S+) "
    r"src1_ne=(?P<src1_ne>\S+) src2=.*? src2_type=\S+ src2_ne=\S+ "
    r"elapsed_us=(?P<elapsed_us>\d+) flops=(?P<flops>\d+) bytes=(?P<bytes>\d+)$"
)

ARENA_BYTES = 64 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shape(value: str) -> tuple[int, int, int, int]:
    parts = tuple(int(part) for part in value.split("x"))
    if len(parts) != 4:
        raise ValueError(f"invalid shape: {value}")
    return parts


def align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def minimum_arena_bytes(m: int, n: int, k: int) -> int:
    # The runtime stores split INT8 payload and FP16 scales, plus F32 output.
    return sum(
        align(value)
        for value in (
            n * k,
            n * (k // 32) * 2,
            m * k,
            m * (k // 32) * 2,
            m * n * 4,
        )
    )


def classify(item: dict[str, str]) -> tuple[str, tuple[int, int, int] | None, int | None]:
    if item["op"] != "MUL_MAT":
        return "not_mul_mat", None, None
    if item["src0_type"] != "q8_0":
        return "weight_type_not_q8_0", None, None
    if item["src1_type"] != "f32":
        return "activation_type_not_f32", None, None
    if item["out_type"] != "f32":
        return "output_type_not_f32", None, None
    if any("(view)" in item[key] or "(permuted)" in item[key]
           for key in ("name", "src0", "src1")):
        return "view_or_permutation", None, None
    w = shape(item["src0_ne"])
    x = shape(item["src1_ne"])
    y = shape(item["out_ne"])
    if w[2:] != (1, 1) or x[2:] != (1, 1) or y[2:] != (1, 1):
        return "batch_or_broadcast", None, None
    k, n = w[0], w[1]
    m = x[1]
    if x[0] != k or y[0] != n or y[1] != m or min(m, n, k) <= 0:
        return "shape_mismatch", None, None
    if n > 9728 or k > 9728 or n % 4 or k % 32:
        return "hardware_limit", (m, n, k), None
    arena = minimum_arena_bytes(1, n, k)
    if arena > ARENA_BYTES:
        return "arena_capacity", (m, n, k), arena
    return "eligible", (m, n, k), arena


def read_records(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    records = []
    with opener(path, "rt", errors="replace") as stream:
        for line in stream:
            match = LINE.match(line.rstrip("\n\r"))
            if match:
                records.append(match.groupdict())
    return records


def analyze(log_path: Path, profile_path: Path) -> dict[str, object]:
    profile = json.loads(profile_path.read_text())
    graph_stages = {str(key): value["stage"] for key, value in profile["graphs"].items()}
    records = read_records(log_path)
    if not records:
        raise RuntimeError("no GGML_OP_PROFILE records parsed")
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    shapes: dict[tuple[str, str, int, int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    parsed_mul_mat = 0
    for item in records:
        if item["op"] != "MUL_MAT":
            continue
        parsed_mul_mat += 1
        stage = graph_stages.get(item["graph"], "unknown")
        reason, dimensions, arena = classify(item)
        elapsed = int(item["elapsed_us"])
        flops = int(item["flops"])
        bucket = totals[stage]
        bucket["calls"] += 1
        bucket["elapsed_us"] += elapsed
        bucket["flops"] += flops
        bucket[f"reason_{reason}_calls"] += 1
        bucket[f"reason_{reason}_elapsed_us"] += elapsed
        bucket[f"reason_{reason}_flops"] += flops
        if dimensions is not None:
            m, n, k = dimensions
            entry = shapes[(stage, reason, m, n, k)]
            entry["calls"] += 1
            entry["elapsed_us"] += elapsed
            entry["flops"] += flops
            if arena is not None:
                entry["minimum_arena_bytes"] = arena
    stages = {}
    for stage, bucket in sorted(totals.items()):
        calls = int(bucket["calls"])
        elapsed = int(bucket["elapsed_us"])
        flops = int(bucket["flops"])
        eligible_calls = int(bucket.get("reason_eligible_calls", 0))
        eligible_elapsed = int(bucket.get("reason_eligible_elapsed_us", 0))
        eligible_flops = int(bucket.get("reason_eligible_flops", 0))
        stage_total_us = float(profile["stage_time_ms"].get(stage, 0.0)) * 1000.0
        reasons = {}
        for key, value in bucket.items():
            match = re.fullmatch(r"reason_(.+)_calls", key)
            if match:
                reason = match.group(1)
                reasons[reason] = {
                    "calls": int(value),
                    "elapsed_us": int(bucket[f"reason_{reason}_elapsed_us"]),
                    "flops": int(bucket[f"reason_{reason}_flops"]),
                }
        noneligible_stage_us = max(stage_total_us - eligible_elapsed, 0.0)
        stages[stage] = {
            "mul_mat_calls": calls,
            "mul_mat_elapsed_us": elapsed,
            "mul_mat_flops": flops,
            "eligible_calls": eligible_calls,
            "eligible_elapsed_us": eligible_elapsed,
            "eligible_flops": eligible_flops,
            "eligible_call_percent": 100.0 * eligible_calls / calls,
            "eligible_mul_mat_time_percent": 100.0 * eligible_elapsed / elapsed,
            "eligible_mul_mat_flop_percent": 100.0 * eligible_flops / flops,
            "eligible_stage_time_percent": 100.0 * eligible_elapsed / stage_total_us
            if stage_total_us else 0.0,
            "zero_time_amdahl_upper_speedup": stage_total_us / noneligible_stage_us
            if noneligible_stage_us else math.inf,
            "reasons": reasons,
        }
    shape_rows = [
        {
            "stage": stage,
            "reason": reason,
            "m": m,
            "n": n,
            "k": k,
            "calls": int(values["calls"]),
            "elapsed_us": int(values["elapsed_us"]),
            "flops": int(values["flops"]),
            **({"minimum_arena_bytes": int(values["minimum_arena_bytes"])}
               if "minimum_arena_bytes" in values else {}),
        }
        for (stage, reason, m, n, k), values in shapes.items()
    ]
    shape_rows.sort(key=lambda row: (-row["elapsed_us"], row["stage"], row["m"]))
    return {
        "schema_version": 1,
        "decision": "PROFILE_COVERAGE_AUDIT_PASS / NO_RUNTIME_DISPATCH_CLAIM",
        "contract": {
            "weights": "q8_0 contiguous 2D",
            "activations": "f32 contiguous 2D dynamically quantized to q8_0",
            "output": "f32 contiguous 2D",
            "limits": {"max_n": 9728, "max_k": 9728, "n_multiple": 4,
                       "k_multiple": 32, "arena_bytes": ARENA_BYTES},
        },
        "parsed_records": len(records),
        "parsed_mul_mat_records": parsed_mul_mat,
        "stages": stages,
        "shapes": shape_rows,
        "source": {
            "profile_log": str(log_path),
            "profile_log_sha256": sha256(log_path),
            "profile_json": str(profile_path),
            "profile_json_sha256": sha256(profile_path),
        },
        "limitations": [
            "Coverage is reconstructed from an existing instrumented profile, not a live PL dispatch.",
            "Profile names are used to conservatively reject explicit views/permutations.",
            "A zero-time Amdahl value is an upper bound, not a speedup prediction.",
            "Contiguous strides must still be checked by the production tensor adapter at runtime.",
        ],
    }


def markdown(result: dict[str, object]) -> str:
    rows = []
    for stage, item in result["stages"].items():
        rows.append(
            f"| {stage} | {item['eligible_calls']}/{item['mul_mat_calls']} | "
            f"{item['eligible_mul_mat_time_percent']:.2f}% | "
            f"{item['eligible_stage_time_percent']:.2f}% | "
            f"{item['zero_time_amdahl_upper_speedup']:.3f}× |"
        )
    top_shapes = []
    for item in result["shapes"][:15]:
        top_shapes.append(
            f"| {item['stage']} | {item['reason']} | {item['m']} | {item['n']} | "
            f"{item['k']} | {item['calls']} | {item['elapsed_us']/1e6:.3f} s |"
        )
    return f"""# 现有GGML日志的Q8×Q8 PL桥接覆盖率审计

状态：`{result['decision']}`

## 按阶段覆盖

| 阶段 | 可命中/矩阵乘调用 | 可命中MUL_MAT耗时 | 可命中阶段耗时 | 零耗时Amdahl上界 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

“可命中”要求Q8_0权重、FP32激活/输出、二维非view、shape满足PL上限且至少一行能装入
64 MiB保留区。零耗时Amdahl只回答即使可命中部分瞬间完成，整个阶段最多能加速多少，
不是实际PL速度预测。

## 耗时最高的shape/拒绝原因

| 阶段 | 分类 | M | N | K | 调用 | CPU profile耗时 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(top_shapes)}

## 边界

- 数据来自既有插桩日志，未重新运行模型，也未读取任何盲测精度数据。
- 本报告不证明生产后端已经调用PL；真实运行仍必须逐张量检查stride和view。
- 原始日志：`{result['source']['profile_log']}`。
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.json_out, args.markdown_out):
        if output.exists():
            raise RuntimeError(f"refusing to overwrite append-only output: {output}")
    result = analyze(args.log, args.profile)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    args.markdown_out.write_text(markdown(result))
    print(
        "GGML_Q8Q8_PROFILE_COVERAGE_PASS "
        + " ".join(
            f"{stage}={item['eligible_calls']}/{item['mul_mat_calls']}"
            for stage, item in result["stages"].items()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
