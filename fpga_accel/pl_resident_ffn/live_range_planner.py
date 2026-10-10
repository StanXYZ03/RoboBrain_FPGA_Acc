#!/usr/bin/env python3
"""Live-range-aware 64 MiB arena planner for PL-resident FFN subgraphs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


ALIGNMENT = 64
Q8_NUMERATOR = 34
Q8_DENOMINATOR = 32


@dataclass(frozen=True)
class Buffer:
    name: str
    bytes: int
    birth: int
    death: int


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def q8_bytes(rows: int, width: int) -> int:
    if width % 32:
        raise ValueError("Q8_0 width must be divisible by 32")
    return rows * width * Q8_NUMERATOR // Q8_DENOMINATOR


def weight_bytes(n: int, k: int) -> int:
    return q8_bytes(n, k)


def buffers(kind: str, m: int) -> list[Buffer]:
    if m <= 0:
        raise ValueError("M must be positive")
    if kind == "llm_swiglu":
        return [
            Buffer("residual_f32", m * 2560 * 4, 0, 8),
            Buffer("weight_stage", weight_bytes(9728, 2560), 1, 7),
            Buffer("input_q8", q8_bytes(m, 2560), 1, 3),
            Buffer("gate_f32", m * 9728 * 4, 2, 5),
            Buffer("up_f32", m * 9728 * 4, 3, 5),
            Buffer("ffn_q8", q8_bytes(m, 9728), 5, 6),
            Buffer("down_f32", m * 2560 * 4, 6, 8),
        ]
    if kind == "vision_gelu":
        return [
            Buffer("residual_f32", m * 1024 * 4, 0, 7),
            Buffer("weight_stage", weight_bytes(4096, 1024), 1, 6),
            Buffer("input_q8", q8_bytes(m, 1024), 1, 3),
            Buffer("fc1_f32", m * 4096 * 4, 2, 4),
            Buffer("fc1_q8", q8_bytes(m, 4096), 4, 5),
            Buffer("fc2_f32", m * 1024 * 4, 5, 7),
        ]
    raise ValueError(f"unknown subgraph: {kind}")


def lifetimes_overlap(left: Buffer, right: Buffer) -> bool:
    return left.birth < right.death and right.birth < left.death


def allocate(kind: str, m: int) -> dict[str, object]:
    items = buffers(kind, m)
    placements: dict[str, tuple[int, int]] = {}
    for item in sorted(items, key=lambda value: (value.birth, -value.bytes, value.name)):
        offset = 0
        while True:
            offset = align(offset)
            conflict_end = None
            for other in items:
                if other.name not in placements or not lifetimes_overlap(item, other):
                    continue
                other_offset, other_size = placements[other.name]
                if offset < other_offset + other_size and other_offset < offset + align(item.bytes):
                    conflict_end = max(conflict_end or 0, other_offset + other_size)
            if conflict_end is None:
                placements[item.name] = (offset, align(item.bytes))
                break
            offset = conflict_end
    peak = max(offset + size for offset, size in placements.values())
    timeline = []
    last_stage = max(item.death for item in items)
    for stage in range(last_stage):
        live = [item for item in items if item.birth <= stage < item.death]
        timeline.append({
            "stage": stage,
            "live": [item.name for item in live],
            "live_bytes_unaliased": sum(align(item.bytes) for item in live),
        })
    return {
        "subgraph": kind,
        "m": m,
        "arena_bytes": peak,
        "buffers": {
            item.name: {
                "bytes": item.bytes,
                "aligned_bytes": placements[item.name][1],
                "offset": placements[item.name][0],
                "birth": item.birth,
                "death": item.death,
            }
            for item in items
        },
        "timeline": timeline,
    }


def validate(plan: dict[str, object], capacity: int) -> None:
    if plan["arena_bytes"] > capacity:
        raise ValueError("arena capacity exceeded")
    entries = plan["buffers"]
    names = list(entries)
    for index, name in enumerate(names):
        item = entries[name]
        if item["offset"] % ALIGNMENT or item["offset"] + item["aligned_bytes"] > capacity:
            raise ValueError("invalid aligned placement")
        for other_name in names[index + 1:]:
            other = entries[other_name]
            lifetime = item["birth"] < other["death"] and other["birth"] < item["death"]
            memory = (item["offset"] < other["offset"] + other["aligned_bytes"] and
                      other["offset"] < item["offset"] + item["aligned_bytes"])
            if lifetime and memory:
                raise ValueError(f"overlap: {name}/{other_name}")


def maximum_chunk(kind: str, total_m: int, capacity: int) -> dict[str, object]:
    low, high, best = 1, total_m, None
    while low <= high:
        middle = (low + high) // 2
        candidate = allocate(kind, middle)
        if candidate["arena_bytes"] <= capacity:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError("capacity cannot fit one row")
    chunk = best["m"]
    if chunk >= 16 and chunk < total_m:
        chunk -= chunk % 16
        best = allocate(kind, chunk)
    validate(best, capacity)
    best["capacity_bytes"] = capacity
    best["total_m"] = total_m
    best["chunk_count"] = (total_m + chunk - 1) // chunk
    best["utilization"] = best["arena_bytes"] / capacity
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subgraph", choices=("llm_swiglu", "vision_gelu"), required=True)
    parser.add_argument("--total-m", type=int, required=True)
    parser.add_argument("--capacity", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    plan = maximum_chunk(args.subgraph, args.total_m, args.capacity)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
