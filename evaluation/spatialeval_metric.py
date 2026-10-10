#!/usr/bin/env python3
"""Draft, deterministic answer-only metric for SpatialEval adapter testing.

This is deliberately not the official repository's model-specific heuristic
parser. The final external-confirmation parser must be frozen only after the
real schema and official evaluator have been audited.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Mapping, Sequence


TASK_PREFIXES = ("spatialmap", "mazenav", "spatialgrid", "spatialreal")


class SpatialEvalMetricError(ValueError):
    pass


def task_from_sample_id(sample_id: str) -> str:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise SpatialEvalMetricError("sample_id must be a non-empty string")
    compact = re.sub(r"[^a-z]", "", sample_id.lower())
    for task in TASK_PREFIXES:
        if compact.startswith(task):
            return task
    raise SpatialEvalMetricError(f"unknown SpatialEval task prefix in {sample_id!r}")


def normalize_short_answer(value: str) -> str | None:
    """Normalize a short answer without consulting the expected answer."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\n" in text or len(text) > 96:
        return None
    text = re.sub(r"^(?:answer|final answer)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`*_ ")
    text = re.sub(r"[.!?]+$", "", text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    option = re.fullmatch(r"[\[(]?([a-d])[\])]?(?:[.:])?", text)
    if option:
        return option.group(1)
    if not text or len(text.split()) > 12:
        return None
    return text


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(records, (list, tuple)) or not records:
        raise SpatialEvalMetricError("records must be a non-empty list/tuple")
    if expected_count is not None and len(records) != expected_count:
        raise SpatialEvalMetricError(
            f"record count {len(records)} != expected_count {expected_count}"
        )
    seen: set[str] = set()
    grouped: dict[str, list[bool]] = defaultdict(list)
    parse_failures = 0
    for index, record in enumerate(records):
        for key in ("sample_id", "output_text", "expected_answer"):
            if key not in record:
                raise SpatialEvalMetricError(f"record {index} missing {key!r}")
        sample_id = record["sample_id"]
        if sample_id in seen:
            raise SpatialEvalMetricError(f"duplicate sample_id {sample_id!r}")
        seen.add(sample_id)
        task = task_from_sample_id(sample_id)
        parsed = normalize_short_answer(record["output_text"])
        expected = normalize_short_answer(record["expected_answer"])
        if expected is None:
            raise SpatialEvalMetricError(
                f"record {index} expected_answer is outside the draft short-answer contract"
            )
        if parsed is None:
            parse_failures += 1
        grouped[task].append(parsed is not None and parsed == expected)
    per_task = {
        task: {"count": len(values), "accuracy": sum(values) / len(values)}
        for task, values in sorted(grouped.items())
    }
    correct = sum(sum(values) for values in grouped.values())
    return {
        "metric": "draft_answer_only_exact_match_not_official_parser",
        "count": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "parse_success_count": len(records) - parse_failures,
        "parse_failure_count": parse_failures,
        "parse_success_rate": (len(records) - parse_failures) / len(records),
        "per_task": per_task,
    }
