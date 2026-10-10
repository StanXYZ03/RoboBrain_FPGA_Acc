#!/usr/bin/env python3
"""Lifecycle-safe CV-Bench metric and evaluation-protocol v2 helpers.

This module is the future-only scoring contract that replaces the frozen
``evaluation.build_cv_bench_pruning_report.score`` primary metric *without*
touching it.  The frozen function is preserved byte-for-byte for
reproducibility; this module re-implements its official CV-Bench weighting and
fixes the two audit findings:

1. ``parse_success_rate`` must be derived from an *explicit*
   ``parsed_answer_key``.  The frozen ``score`` infers the key via
   ``correct_key.replace("_correct", "_answer")``; when ``correct_key ==
   "correct"`` that derivation aliases the boolean correctness field and the
   parse gate silently reads 1.0 while honest parse failures are dense.
2. Inputs (record count, required keys, types, sources and tasks) are validated
   and the computation fails closed instead of raising an opaque
   ``StatisticsError`` or returning a dishonest number.

It also carries the explicit lifecycle states and the lifecycle-safe probe-path
guard used by ``configs/evaluation_protocol_v2.json``.

CPU-only, stdlib-only.  ``parse_choice`` / ``answer_letter`` are imported from
the frozen module so parsing behaviour is reused, never re-invented.
"""

from __future__ import annotations

import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.build_cv_bench_pruning_report import (  # noqa: F401  (re-exported)
    answer_letter,
    parse_choice,
)

__all__ = [
    "CV_BENCH_SOURCES",
    "MetricV2Error",
    "LIFECYCLE_STATES",
    "TERMINAL_LIFECYCLE_STATES",
    "validate_lifecycle_state",
    "validate_metric_inputs",
    "compute_cv_bench_metric",
    "parse_gate",
    "is_temporary_path",
    "require_lifecycle_safe_probe_path",
    "answer_letter",
    "parse_choice",
]

#: Official CV-Bench source groups used by the combined score.
CV_BENCH_SOURCES: tuple[str, ...] = ("ADE20K", "COCO", "Omni3D")

#: Explicit future-only lifecycle states (see configs/evaluation_protocol_v2.json).
LIFECYCLE_STATES: tuple[str, ...] = (
    "PREREGISTERED_NOT_EXECUTED",
    "EXECUTED_UNAUDITED",
    "AUDITED_STOP",
    "AUDITED_GO_EXTERNAL_VALIDATION",
)

#: States from which no automatic continuation is allowed by this protocol.
TERMINAL_LIFECYCLE_STATES: tuple[str, ...] = (
    "AUDITED_STOP",
    "AUDITED_GO_EXTERNAL_VALIDATION",
)


class MetricV2Error(ValueError):
    """Raised when the v2 metric contract is violated (fail closed)."""


def _fail(message: str) -> None:
    raise MetricV2Error(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def validate_lifecycle_state(state: str) -> str:
    """Return ``state`` if it is a declared lifecycle state, else fail closed."""

    _require(
        isinstance(state, str) and state in LIFECYCLE_STATES,
        f"unknown lifecycle state {state!r}; expected one of {LIFECYCLE_STATES}",
    )
    return state


def validate_metric_inputs(
    records: Sequence[Mapping[str, Any]],
    *,
    correct_key: str,
    parsed_answer_key: str,
    expected_count: int | None = None,
    sources: Sequence[str] = CV_BENCH_SOURCES,
    id_key: str = "sample_id",
) -> None:
    """Validate record count, required keys and types; raise ``MetricV2Error``.

    ``correct_key`` and ``parsed_answer_key`` are both mandatory and must be
    distinct: the parsed-answer key is never inferred from the correctness key.
    """

    _require(
        correct_key is not None and parsed_answer_key is not None,
        "both correct_key and parsed_answer_key are required",
    )
    _require(
        isinstance(correct_key, str) and bool(correct_key),
        f"correct_key must be a non-empty str, got {correct_key!r}",
    )
    _require(
        isinstance(parsed_answer_key, str) and bool(parsed_answer_key),
        f"parsed_answer_key must be a non-empty str, got {parsed_answer_key!r}",
    )
    _require(
        correct_key != parsed_answer_key,
        "correct_key and parsed_answer_key must be distinct fields; the "
        "parsed-answer key is never silently inferred from the correctness key",
    )
    _require(
        isinstance(records, (list, tuple)),
        f"records must be a list/tuple, got {type(records).__name__}",
    )
    _require(len(records) > 0, "records must not be empty")
    if expected_count is not None:
        _require(
            isinstance(expected_count, int) and not isinstance(expected_count, bool),
            f"expected_count must be an int, got {expected_count!r}",
        )
        _require(
            len(records) == expected_count,
            f"record count {len(records)} != expected_count {expected_count}",
        )

    source_set = tuple(sources)
    _require(len(source_set) > 0, "sources must not be empty")
    missing_sources = [source for source in CV_BENCH_SOURCES if source not in source_set]
    _require(
        not missing_sources,
        f"sources must include the official CV-Bench sources {missing_sources}",
    )
    seen_ids: set[Any] = set()
    for index, record in enumerate(records):
        _require(
            isinstance(record, Mapping),
            f"record {index} is not a mapping: {type(record).__name__}",
        )
        _require(correct_key in record, f"record {index} missing {correct_key!r}")
        _require(
            parsed_answer_key in record,
            f"record {index} missing {parsed_answer_key!r}",
        )
        correct = record[correct_key]
        _require(
            isinstance(correct, bool),
            f"record {index} {correct_key!r} must be bool, got {type(correct).__name__}",
        )
        parsed = record[parsed_answer_key]
        _require(
            parsed is None or isinstance(parsed, str),
            f"record {index} {parsed_answer_key!r} must be str or None, "
            f"got {type(parsed).__name__}",
        )
        _require("source" in record, f"record {index} missing 'source'")
        _require(
            record["source"] in source_set,
            f"record {index} source {record['source']!r} not in {source_set}",
        )
        _require("task" in record, f"record {index} missing 'task'")
        _require(
            record["task"] is not None and str(record["task"]) != "",
            f"record {index} has an empty 'task'",
        )
        if id_key in record:
            sample_id = record[id_key]
            _require(
                sample_id not in seen_ids,
                f"record {index} duplicates {id_key}={sample_id!r}",
            )
            seen_ids.add(sample_id)

    for source in source_set:
        _require(
            any(record["source"] == source for record in records),
            f"no records for required source {source!r}",
        )


def compute_cv_bench_metric(
    records: Sequence[Mapping[str, Any]],
    *,
    correct_key: str,
    parsed_answer_key: str,
    expected_count: int | None = None,
    sources: Sequence[str] = CV_BENCH_SOURCES,
) -> dict[str, Any]:
    """Official CV-Bench combined score plus honest parse statistics.

    The primary score is identical in weighting to the frozen
    ``score(...)["cv_bench_accuracy"]``::

        ((accuracy_ADE20K + accuracy_COCO) / 2 + accuracy_Omni3D) / 2

    ``parse_success_rate`` is computed from ``parsed_answer_key`` only and is
    therefore honest even when the correctness field is a boolean.
    """

    validate_metric_inputs(
        records,
        correct_key=correct_key,
        parsed_answer_key=parsed_answer_key,
        expected_count=expected_count,
        sources=sources,
    )

    source_accuracy: dict[str, float] = {}
    for source in sources:
        subset = [record for record in records if record["source"] == source]
        source_accuracy[source] = statistics.fmean(
            bool(record[correct_key]) for record in subset
        )
    accuracy_2d = (source_accuracy["ADE20K"] + source_accuracy["COCO"]) / 2
    accuracy_3d = source_accuracy["Omni3D"]
    tasks = sorted({str(record["task"]) for record in records})

    parsed = [record for record in records if record[parsed_answer_key] is not None]
    parse_success_count = len(parsed)
    parse_failure_count = len(records) - parse_success_count

    return {
        "sample_count": len(records),
        "correct_key": correct_key,
        "parsed_answer_key": parsed_answer_key,
        "parse_success_rate": parse_success_count / len(records),
        "parse_success_count": parse_success_count,
        "parse_failure_count": parse_failure_count,
        "accuracy": statistics.fmean(bool(record[correct_key]) for record in records),
        "source_accuracy": source_accuracy,
        "task_accuracy": {
            task: statistics.fmean(
                bool(record[correct_key])
                for record in records
                if str(record["task"]) == task
            )
            for task in tasks
        },
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "cv_bench_accuracy": (accuracy_2d + accuracy_3d) / 2,
    }


def parse_gate(
    metric: Mapping[str, Any], *, require_all_parsed: bool = False
) -> dict[str, Any]:
    """Turn an honest metric result into an explicit parse gate.

    ``passed`` is true only when the parse statistics came from a distinct
    explicit parsed-answer field and (optionally) every record parsed.
    """

    _require(
        "parsed_answer_key" in metric and "correct_key" in metric,
        "metric result must carry correct_key/parsed_answer_key provenance",
    )
    _require(
        metric["parsed_answer_key"] != metric["correct_key"],
        "metric result reused the correctness field as the parsed-answer field",
    )
    failures = int(metric["parse_failure_count"])
    passed = failures == 0 if require_all_parsed else True
    return {
        "field": metric["parsed_answer_key"],
        "correctness_field": metric["correct_key"],
        "distinct": True,
        "parse_success_count": int(metric["parse_success_count"]),
        "parse_failure_count": failures,
        "parse_success_rate": float(metric["parse_success_rate"]),
        "require_all_parsed": bool(require_all_parsed),
        "passed": bool(passed),
    }


def is_temporary_path(path: str | Path, *, temp_root: str | Path | None = None) -> bool:
    """True when ``path`` resolves inside the temporary root."""

    probe = Path(path).resolve()
    root = (
        Path(tempfile.gettempdir()).resolve()
        if temp_root is None
        else Path(temp_root).resolve()
    )
    return probe == root or root in probe.parents


def require_lifecycle_safe_probe_path(
    probe_path: str | Path,
    *,
    default_output_paths: Sequence[str | Path],
    temp_root: str | Path | None = None,
) -> Path:
    """Validate a missing-output probe path and return its resolved form.

    Lifecycle-safe rule: a test that asserts a future output is *missing* must
    probe a temporary or explicit scratch path, never the real default output.
    Probing the real default would make pre-execution tests state-stale (they
    pass only while the arm is absent and break after the arm exists) and could
    observe or clobber a frozen artifact.
    """

    _require(
        isinstance(default_output_paths, (list, tuple)) and default_output_paths,
        "default_output_paths must be a non-empty list/tuple",
    )
    probe = Path(probe_path).resolve()
    _require(
        is_temporary_path(probe, temp_root=temp_root),
        f"missing-output probe {str(probe)!r} must use a temporary/explicit "
        "scratch path, never a repository output path",
    )
    for default in default_output_paths:
        resolved = Path(default).resolve()
        if probe == resolved:
            _fail(
                f"missing-output probe {str(probe)!r} equals the real default "
                "output path and would be state-stale; use a temporary path"
            )
        _require(
            resolved not in probe.parents,
            f"missing-output probe {str(probe)!r} is nested under the real "
            f"default output {str(resolved)!r}; use a temporary path",
        )
    return probe
