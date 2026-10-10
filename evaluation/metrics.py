#!/usr/bin/env python3
"""Parse structured spatial outputs and compute labeled Pointing/Grounding metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


NUMBER = r"[-+]?\d+(?:\.\d+)?"
POINT_PATTERN = re.compile(rf"[\(\[]\s*({NUMBER})\s*,\s*({NUMBER})\s*[\)\]]")
BOX_PATTERN = re.compile(
    rf"[\[\(]\s*({NUMBER})\s*,\s*({NUMBER})\s*,\s*"
    rf"({NUMBER})\s*,\s*({NUMBER})\s*[\]\)]"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return records


def parse_point(text: str) -> tuple[float, float] | None:
    match = POINT_PATTERN.search(text)
    if match is None:
        return None
    point = float(match.group(1)), float(match.group(2))
    return point if all(0.0 <= value <= 1000.0 for value in point) else None


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    match = BOX_PATTERN.search(text)
    if match is None:
        return None
    box = tuple(float(match.group(index)) for index in range(1, 5))
    if not all(0.0 <= value <= 1000.0 for value in box):
        return None
    x1, y1, x2, y2 = box
    return box if x2 > x1 and y2 > y1 else None


def box_iou(
    predicted: Iterable[float], ground_truth: Iterable[float]
) -> float:
    px1, py1, px2, py2 = predicted
    gx1, gy1, gx2, gy2 = ground_truth
    intersection_width = max(0.0, min(px2, gx2) - max(px1, gx1))
    intersection_height = max(0.0, min(py2, gy2) - max(py1, gy1))
    intersection = intersection_width * intersection_height
    predicted_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    ground_truth_area = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = predicted_area + ground_truth_area - intersection
    return intersection / union if union > 0 else 0.0


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(ax - bx, ay - by)


def score_record(sample: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    task = sample["task"]
    text = str(prediction.get("output_text", ""))
    result: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": task,
        "output_text": text,
        "elapsed_seconds": prediction.get("elapsed_seconds"),
    }
    if task == "pointing":
        predicted = parse_point(text)
        truth = tuple(float(value) for value in sample["ground_truth"]["point_1000"])
        result["parse_success"] = predicted is not None
        if predicted is None:
            result.update(
                point_error_1000=None,
                point_nme_image_diag=None,
                point_nme_bbox_diag=None,
                pck_0_1=False,
                pck_0_2=False,
            )
            return result
        bbox = tuple(float(value) for value in sample["ground_truth"]["bbox_1000"])
        bbox_diagonal = math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1])
        error = euclidean(predicted, truth)
        normalized_bbox_error = error / bbox_diagonal if bbox_diagonal > 0 else math.inf
        result.update(
            predicted=list(predicted),
            point_error_1000=error,
            point_nme_image_diag=error / math.hypot(1000.0, 1000.0),
            point_nme_bbox_diag=normalized_bbox_error,
            pck_0_1=normalized_bbox_error <= 0.1,
            pck_0_2=normalized_bbox_error <= 0.2,
        )
        return result
    if task == "grounding":
        predicted = parse_box(text)
        truth = tuple(float(value) for value in sample["ground_truth"]["bbox_1000"])
        result["parse_success"] = predicted is not None
        if predicted is None:
            result.update(iou=0.0, acc_0_5=False, center_error_1000=None)
            return result
        iou = box_iou(predicted, truth)
        predicted_center = ((predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2)
        truth_center = ((truth[0] + truth[2]) / 2, (truth[1] + truth[3]) / 2)
        result.update(
            predicted=list(predicted),
            iou=iou,
            acc_0_5=iou >= 0.5,
            center_error_1000=euclidean(predicted_center, truth_center),
        )
        return result
    raise ValueError(f"unsupported task: {task}")


def mean_present(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return statistics.fmean(values) if values else None


def latency_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    values = sorted(
        float(record["elapsed_seconds"])
        for record in records
        if record.get("elapsed_seconds") is not None
    )
    if not values:
        return {}
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "latency_seconds_mean": statistics.fmean(values),
        "latency_seconds_median": statistics.median(values),
        "latency_seconds_p95": values[p95_index],
    }


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in scored:
        grouped[record["task"]].append(record)
    summary: dict[str, Any] = {"sample_count": len(scored), "tasks": {}}
    for task, records in sorted(grouped.items()):
        common = {
            "sample_count": len(records),
            "parse_rate": statistics.fmean(bool(record["parse_success"]) for record in records),
        }
        common.update(latency_summary(records))
        if task == "pointing":
            common.update(
                point_error_1000_mean=mean_present(records, "point_error_1000"),
                point_nme_image_diag_mean=mean_present(records, "point_nme_image_diag"),
                point_nme_bbox_diag_mean=mean_present(records, "point_nme_bbox_diag"),
                pck_0_1=statistics.fmean(bool(record["pck_0_1"]) for record in records),
                pck_0_2=statistics.fmean(bool(record["pck_0_2"]) for record in records),
            )
        elif task == "grounding":
            common.update(
                iou_mean=statistics.fmean(float(record["iou"]) for record in records),
                acc_0_5=statistics.fmean(bool(record["acc_0_5"]) for record in records),
                center_error_1000_mean=mean_present(records, "center_error_1000"),
            )
        summary["tasks"][task] = common
    return summary


def evaluate(
    manifest_path: Path,
    prediction_path: Path,
    *,
    limit_per_task: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = load_jsonl(manifest_path)
    if limit_per_task is not None:
        selected = []
        counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            task = str(sample["task"])
            if counts[task] < limit_per_task:
                selected.append(sample)
                counts[task] += 1
        samples = selected
    predictions = load_jsonl(prediction_path)
    by_id = {record["sample_id"]: record for record in predictions}
    duplicate_count = len(predictions) - len(by_id)
    if duplicate_count:
        raise ValueError(f"prediction file contains {duplicate_count} duplicate sample IDs")
    missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in by_id]
    if missing:
        raise ValueError(f"missing {len(missing)} predictions; first missing ID: {missing[0]}")
    scored = [score_record(sample, by_id[sample["sample_id"]]) for sample in samples]
    return scored, summarize(scored)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--scored", type=Path)
    parser.add_argument("--limit-per-task", type=int)
    args = parser.parse_args()
    scored, summary = evaluate(
        args.manifest,
        args.predictions,
        limit_per_task=args.limit_per_task,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.scored:
        args.scored.parent.mkdir(parents=True, exist_ok=True)
        with args.scored.open("w", encoding="utf-8") as stream:
            for record in scored:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
