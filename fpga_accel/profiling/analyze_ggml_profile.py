#!/usr/bin/env python3
"""Aggregate opt-in GGML CPU node timing logs into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


STAGE_RE = re.compile(r"inference stage = ([a-z_]+)")
BEGIN_RE = re.compile(r"GGML_GRAPH_PROFILE_BEGIN graph=(\d+) nodes=(\d+) threads=(\d+)")
END_RE = re.compile(r"GGML_GRAPH_PROFILE_END graph=(\d+) status=(-?\d+)")
OP_RE = re.compile(
    r"GGML_OP_PROFILE graph=(?P<graph>\d+) node=(?P<node>\d+) "
    r"op=(?P<op>\S+) name=(?P<name>.*?) "
    r"out_type=(?P<out_type>\S+) out_ne=(?P<out_ne>\S+) "
    r"src0=(?P<src0>.*?) src0_type=(?P<src0_type>\S+) src0_ne=(?P<src0_ne>\S+) "
    r"src1=(?P<src1>.*?) src1_type=(?P<src1_type>\S+) src1_ne=(?P<src1_ne>\S+) "
    r"(?:src2=(?P<src2>.*?) src2_type=(?P<src2_type>\S+) src2_ne=(?P<src2_ne>\S+) )?"
    r"elapsed_us=(?P<elapsed_us>\d+) flops=(?P<flops>\d+) bytes=(?P<bytes>\d+)"
)


def parse_log(path: Path) -> tuple[list[dict], dict[int, dict]]:
    current_stage = "startup"
    graph_stage: dict[int, str] = {}
    graphs: dict[int, dict] = {}
    records: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            stage = STAGE_RE.search(line)
            if stage:
                current_stage = stage.group(1)
            begin = BEGIN_RE.search(line)
            if begin:
                graph_id = int(begin.group(1))
                graph_stage[graph_id] = current_stage
                graphs[graph_id] = {
                    "stage": current_stage,
                    "declared_nodes": int(begin.group(2)),
                    "threads": int(begin.group(3)),
                    "complete": False,
                }
                continue
            end = END_RE.search(line)
            if end:
                graph_id = int(end.group(1))
                graphs.setdefault(graph_id, {"stage": graph_stage.get(graph_id, current_stage)})
                graphs[graph_id]["complete"] = True
                graphs[graph_id]["status"] = int(end.group(2))
                continue
            match = OP_RE.search(line)
            if not match:
                continue
            record = match.groupdict(default="-")
            for key in ("graph", "node", "elapsed_us", "flops", "bytes"):
                record[key] = int(record[key])
            record["stage"] = graph_stage.get(record["graph"], current_stage)
            record["line"] = line_no
            records.append(record)
    return records, graphs


def shape(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split("x"))
    except ValueError:
        return ()


def effective_flops(item: dict) -> int:
    if item["flops"]:
        return item["flops"]
    if item["op"] == "FLASH_ATTN_EXT":
        query = shape(item["src0_ne"])
        key = shape(item["src1_ne"])
        if len(query) == 4 and len(key) == 4:
            head_dim, query_tokens, query_heads, batch = query
            key_tokens = key[1]
            # Q*K^T and P*V each require 2*D*Nq*Nk per query head.
            return 4 * head_dim * query_tokens * key_tokens * query_heads * batch
    return 0


def role(item: dict) -> str:
    op = item["op"]
    source = item["src0"]
    if op == "FLASH_ATTN_EXT":
        return "attention_core"
    if op == "MUL_MAT":
        if "ffn_" in source:
            return "ffn_linear"
        if "attn_" in source:
            return "attention_projection"
        if "deepstack" in source:
            return "vision_deepstack"
        if source.startswith("mm."):
            return "multimodal_projector"
        if source == "output.weight":
            return "lm_head"
        return "matmul_other"
    if op in {"GELU", "SWIGLU"}:
        return "activation"
    if op in {"NORM", "RMS_NORM"}:
        return "normalization"
    if op in {"ADD", "MUL", "DIV"}:
        return "elementwise"
    if op in {"CPY", "CONT", "CONCAT", "SET_ROWS", "GET_ROWS", "PAD"}:
        return "data_movement"
    return op.lower()


def aggregate(records: list[dict], peak_gflops: float | None, bandwidth_gbps: float | None) -> dict:
    buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"calls": 0, "elapsed_us": 0, "flops": 0, "bytes": 0}
    )
    role_buckets: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"calls": 0, "elapsed_us": 0, "flops": 0, "bytes": 0}
    )
    kernel_buckets: dict[tuple[str, str, str, str, str, str], dict] = defaultdict(
        lambda: {"calls": 0, "elapsed_us": 0, "flops": 0, "bytes": 0}
    )
    stage_totals: dict[str, int] = defaultdict(int)
    for item in records:
        flops = effective_flops(item)
        item_role = role(item)
        key = (item["stage"], item["op"])
        role_key = (item["stage"], item_role)
        kernel_key = (
            item["stage"], item_role, item["op"], item["src0_type"],
            item["src0_ne"], item["out_ne"],
        )
        for bucket in (buckets[key], role_buckets[role_key], kernel_buckets[kernel_key]):
            bucket["calls"] += 1
            bucket["elapsed_us"] += item["elapsed_us"]
            bucket["flops"] += flops
            bucket["bytes"] += item["bytes"]
        stage_totals[item["stage"]] += item["elapsed_us"]

    def finish(stage: str, bucket: dict) -> dict:
        elapsed_s = bucket["elapsed_us"] / 1e6
        row = {"stage": stage, **bucket}
        row["time_ms"] = bucket["elapsed_us"] / 1000
        row["stage_share_pct"] = 100 * bucket["elapsed_us"] / max(stage_totals[stage], 1)
        row["gflops"] = bucket["flops"] / elapsed_s / 1e9 if elapsed_s else None
        row["bandwidth_gbps"] = bucket["bytes"] / elapsed_s / 1e9 if elapsed_s else None
        row["arithmetic_intensity"] = bucket["flops"] / bucket["bytes"] if bucket["bytes"] else None
        if peak_gflops and bandwidth_gbps and row["arithmetic_intensity"] is not None:
            ridge = peak_gflops / bandwidth_gbps
            row["roofline_bound"] = "memory" if row["arithmetic_intensity"] < ridge else "compute"
        else:
            row["roofline_bound"] = "unclassified"
        return row

    rows = []
    for (stage, op), bucket in buckets.items():
        row = {"op": op, **finish(stage, bucket)}
        rows.append(row)
    rows.sort(key=lambda row: row["elapsed_us"], reverse=True)

    role_rows = []
    for (stage, item_role), bucket in role_buckets.items():
        role_rows.append({"role": item_role, **finish(stage, bucket)})
    role_rows.sort(key=lambda row: row["elapsed_us"], reverse=True)

    kernel_rows = []
    for (stage, item_role, op, source_type, source_shape, output_shape), bucket in kernel_buckets.items():
        kernel_rows.append({
            "role": item_role, "op": op, "source_type": source_type,
            "source_shape": source_shape, "output_shape": output_shape,
            **finish(stage, bucket),
        })
    kernel_rows.sort(key=lambda row: row["elapsed_us"], reverse=True)
    return {
        "method": {
            "timing": "barrier-synchronized CPU wall time per GGML graph node",
            "flops": "2*K per output element for matmul; QK and PV terms for FLASH_ATTN_EXT; softmax is excluded",
            "bytes": "ideal lower-bound tensor traffic; cache reuse and write allocation are not modeled",
            "warning": "profiling changes scheduling and logging overhead; use shares and operator ranking, not profiled end-to-end latency",
        },
        "configuration": {"peak_gflops": peak_gflops, "measured_bandwidth_gbps": bandwidth_gbps},
        "stage_time_ms": {stage: value / 1000 for stage, value in sorted(stage_totals.items())},
        "roles": role_rows,
        "kernels": kernel_rows,
        "operators": rows,
    }


def render_markdown(summary: dict, graph_count: int, record_count: int) -> str:
    lines = [
        "# ZCU102 GGML operator profile",
        "",
        f"Parsed {record_count} operator samples from {graph_count} graphs.",
        "",
        "> The instrumented run inserts synchronization and log output. Operator shares and rankings are valid for bottleneck discovery; use an uninstrumented run for end-to-end latency.",
        "",
        "## Measured operator time by stage",
        "",
        "| Stage | Functional block | Calls | Time (ms) | Stage share | GFLOP/s | AI (FLOP/B) | Roofline |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["roles"]:
        gflops = "-" if row["gflops"] is None else f'{row["gflops"]:.2f}'
        intensity = "-" if row["arithmetic_intensity"] is None else f'{row["arithmetic_intensity"]:.3f}'
        lines.append(
            f'| {row["stage"]} | {row["role"]} | {row["calls"]} | {row["time_ms"]:.3f} | '
            f'{row["stage_share_pct"]:.1f}% | {gflops} | {intensity} | {row["roofline_bound"]} |'
        )
    lines.extend([
        "",
        "## GGML operators",
        "",
        "| Stage | Operator | Calls | Time (ms) | Stage share | Matmul GFLOP/s | Ideal traffic (GB/s) | AI (FLOP/B) | Roofline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in summary["operators"]:
        gflops = "-" if row["gflops"] is None else f'{row["gflops"]:.2f}'
        bandwidth = "-" if row["bandwidth_gbps"] is None else f'{row["bandwidth_gbps"]:.2f}'
        intensity = "-" if row["arithmetic_intensity"] is None else f'{row["arithmetic_intensity"]:.3f}'
        lines.append(
            f'| {row["stage"]} | {row["op"]} | {row["calls"]} | {row["time_ms"]:.3f} | '
            f'{row["stage_share_pct"]:.1f}% | {gflops} | {bandwidth} | {intensity} | {row["roofline_bound"]} |'
        )
    lines.extend(["", "## Interpretation constraints", ""])
    for key, value in summary["method"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--peak-gflops", type=float)
    parser.add_argument("--bandwidth-gbps", type=float)
    args = parser.parse_args()
    records, graphs = parse_log(args.log)
    if not records:
        parser.error("no GGML_OP_PROFILE records found")
    summary = aggregate(records, args.peak_gflops, args.bandwidth_gbps)
    summary["source_log"] = str(args.log)
    summary["graphs"] = graphs
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary, len(graphs), len(records)), encoding="utf-8")
    print(f"PROFILE_ANALYSIS_PASS records={len(records)} graphs={len(graphs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
