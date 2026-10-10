#!/usr/bin/env python3
"""Compare repeated teacher saliency records by value and selection rank."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_norm = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_norm = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_norm * right_norm)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    for rank, index in enumerate(order):
        result[index] = float(rank)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--retention", type=float, default=0.75)
    args = parser.parse_args()
    left = json.loads(args.left.read_text(encoding="utf-8").splitlines()[0])
    right = json.loads(args.right.read_text(encoding="utf-8").splitlines()[0])
    a = [float(value) for value in left["group_saliency"]]
    b = [float(value) for value in right["group_saliency"]]
    keep = math.ceil(len(a) * args.retention)
    top_a = set(sorted(range(len(a)), key=lambda index: (-a[index], index))[:keep])
    top_b = set(sorted(range(len(b)), key=lambda index: (-b[index], index))[:keep])
    result = {
        "sample_id_match": left["sample_id"] == right["sample_id"],
        "teacher_loss_absolute_difference": abs(left["teacher_loss"] - right["teacher_loss"]),
        "pearson": correlation(a, b),
        "spearman": correlation(ranks(a), ranks(b)),
        "topk": keep,
        "topk_overlap": len(top_a & top_b) / keep,
        "max_relative_mean_scaled_difference": max(abs(x - y) for x, y in zip(a, b))
        / (sum(a) / len(a)),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
