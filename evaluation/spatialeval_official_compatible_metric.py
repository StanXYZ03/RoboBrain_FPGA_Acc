#!/usr/bin/env python3
"""SpatialEval VQA metric pinned to the official three-task parser semantics.

The upstream evaluator (commit ``d82ba382``) defines *only* per-task
``Acc = correct / line_count`` and has no valid ``spatialreal`` dispatch and no
cross-task total-score formula.  This module therefore ports only the three
tasks that the official code can actually score -- ``spatialmap``,
``mazenav`` and ``spatialgrid`` -- by loading the untouched upstream
``evaluation.py`` through :mod:`importlib` and calling its answer parsers with
the frozen ``model_name='robobrain25'``.

Primary per-record rule (byte-for-byte the upstream comparison)::

    int(str(oracle_answer).lower() in str(parsed_answer).lower())

A second, stricter diagnostic is reported alongside it::

    int(str(oracle_answer).strip().lower() == str(parsed_answer).strip().lower())

The reported ``micro`` and equal-task ``macro`` numbers are **project-defined
aggregations**, *not* an official SpatialEval total score.  The metric name
states that explicitly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_EVALUATOR = (
    ROOT
    / "artifacts"
    / "provenance"
    / "spatialeval-official-d82ba382"
    / "evaluation.py"
)
OFFICIAL_COMMIT = "d82ba382805265f205693cb83a30c4866ea6c35b"
MODEL_NAME = "robobrain25"

TASKS = ("spatialmap", "mazenav", "spatialgrid")
QUESTION_IDS = (0, 1, 2)
TASK_PARSERS = {
    "spatialmap": "extract_answer_from_text_spatialmap",
    "mazenav": "extract_answer_from_text_mazenav",
    "spatialgrid": "extract_answer_from_text_spatialgrid",
}

EXPECTED_PRIMARY_COUNT = 4500
EXPECTED_PER_TASK_COUNT = 1500
EXPECTED_PER_TASK_QID_COUNT = 500

METRIC_NAME = (
    "spatialeval_official_parser_compatible_three_task_substring_accuracy"
    "__macro_micro_are_project_aggregation_not_official_total_score"
)


class SpatialEvalOfficialMetricError(ValueError):
    """Raised when inputs violate the frozen external-evaluation contract."""


def _ensure_module_available(name: str) -> bool:
    """Make ``name`` importable, returning True iff a placeholder was installed.

    The pinned upstream ``evaluation.py`` imports :mod:`pandas` at module scope
    but only uses it inside its CSV ``main()``.  When the scientific stack is
    absent we inject an inert placeholder so the *unmodified* parser functions
    can still be loaded; any attempted use of the missing dependency raises.
    """

    if name in sys.modules:
        return False
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return False
    stub = types.ModuleType(name)

    def _unavailable(*_args: Any, **_kwargs: Any) -> Any:
        raise ModuleNotFoundError(
            f"{name!r} is not installed; the official SpatialEval evaluator only "
            f"needs it for its CSV main(), not for the answer parsers"
        )

    stub.__getattr__ = _unavailable
    sys.modules[name] = stub
    return True


def load_official_evaluator(path: Path = OFFICIAL_EVALUATOR) -> ModuleType:
    """Import the frozen upstream evaluator from *path* via importlib."""

    path = Path(path)
    if not path.is_file():
        raise SpatialEvalOfficialMetricError(f"official evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "spatialeval_official_d82ba382_evaluation", path
    )
    if spec is None or spec.loader is None:
        raise SpatialEvalOfficialMetricError(
            f"cannot create import spec for official evaluator: {path}"
        )
    module = importlib.util.module_from_spec(spec)
    installed_stub = _ensure_module_available("pandas")
    try:
        spec.loader.exec_module(module)
    finally:
        if installed_stub:
            sys.modules.pop("pandas", None)
    missing = [name for name in TASK_PARSERS.values() if not callable(getattr(module, name, None))]
    if missing:
        raise SpatialEvalOfficialMetricError(
            f"official evaluator {path} is missing answer parsers: {missing}"
        )
    return module


def _record_value(record: Mapping[str, Any], primary: str, alias: str) -> Any:
    if primary in record:
        return record[primary]
    if alias in record:
        return record[alias]
    raise SpatialEvalOfficialMetricError(
        f"record is missing both {primary!r} and {alias!r}"
    )


def resolve_identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    """Return ``(sample_id, task, question_id)`` using the official split rule."""

    sample_id = str(_record_value(record, "sample_id", "id"))
    parts = sample_id.split(".")
    if len(parts) < 2:
        raise SpatialEvalOfficialMetricError(f"invalid SpatialEval id: {sample_id!r}")
    task = parts[0]
    if task not in TASKS:
        raise SpatialEvalOfficialMetricError(
            f"unsupported primary SpatialEval task {task!r} in {sample_id!r}"
        )
    try:
        question_id = int(parts[-1])
    except ValueError as exc:
        raise SpatialEvalOfficialMetricError(
            f"invalid question id in {sample_id!r}"
        ) from exc
    if question_id not in QUESTION_IDS:
        raise SpatialEvalOfficialMetricError(
            f"unsupported question id {question_id} in {sample_id!r}"
        )
    return sample_id, task, question_id


def parse_with_official_evaluator(
    task: str,
    question_id: int,
    output_text: str,
    *,
    official_module: ModuleType,
) -> Any:
    """Call the matching upstream parser with the frozen model name."""

    try:
        parser_name = TASK_PARSERS[task]
    except KeyError as exc:  # pragma: no cover - guarded by resolve_identity
        raise SpatialEvalOfficialMetricError(
            f"no official parser registered for task {task!r}"
        ) from exc
    parser = getattr(official_module, parser_name)
    return parser(output_text, question_id, MODEL_NAME)


def _strict_normalize(value: Any) -> str:
    return str(value).strip().lower()


def _evaluate_with_identity(
    sample_id: str,
    task: str,
    question_id: int,
    record: Mapping[str, Any],
    official_module: ModuleType,
) -> dict[str, Any]:
    output_text = _record_value(record, "output_text", "answer")
    if not isinstance(output_text, str):
        raise SpatialEvalOfficialMetricError(
            f"output text for {sample_id!r} must be a string"
        )
    if "oracle_answer" not in record:
        raise SpatialEvalOfficialMetricError(
            f"record {sample_id!r} is missing 'oracle_answer'"
        )
    oracle_answer = record["oracle_answer"]

    parse_error: str | None = None
    try:
        parsed_answer = parse_with_official_evaluator(
            task, question_id, output_text, official_module=official_module
        )
    except Exception as exc:  # parser failure == wrong answer for this benchmark
        parsed_answer = None
        parse_error = f"{type(exc).__name__}: {exc}"

    parse_success = parsed_answer is not None
    if parse_success:
        # Exact reproduction of the official upstream comparison.
        official_compatible_correct = int(
            str(oracle_answer).lower() in str(parsed_answer).lower()
        )
        strict_equal_correct = int(
            _strict_normalize(oracle_answer) == _strict_normalize(parsed_answer)
        )
    else:
        official_compatible_correct = 0
        strict_equal_correct = 0

    return {
        "sample_id": sample_id,
        "task": task,
        "question_id": question_id,
        "parsed_answer": parsed_answer,
        "parse_success": parse_success,
        "parse_error": parse_error,
        "official_compatible_correct": official_compatible_correct,
        "strict_equal_correct": strict_equal_correct,
    }


def evaluate_record(
    record: Mapping[str, Any], *, official_module: ModuleType | None = None
) -> dict[str, Any]:
    module = official_module or load_official_evaluator()
    sample_id, task, question_id = resolve_identity(record)
    return _evaluate_with_identity(
        sample_id, task, question_id, record, module
    )


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    primary_correct = sum(int(row["official_compatible_correct"]) for row in rows)
    strict_correct = sum(int(row["strict_equal_correct"]) for row in rows)
    parse_success = sum(bool(row["parse_success"]) for row in rows)
    return {
        "count": count,
        "official_compatible_correct": primary_correct,
        "official_compatible_accuracy": primary_correct / count if count else None,
        "strict_equal_correct": strict_correct,
        "strict_equal_accuracy": strict_correct / count if count else None,
        "parse_success_count": parse_success,
        "parse_failure_count": count - parse_success,
        "parse_success_rate": parse_success / count if count else None,
    }


def score_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int | None = EXPECTED_PRIMARY_COUNT,
    official_module: ModuleType | None = None,
    require_all_tasks: bool = True,
    include_record_rows: bool = True,
) -> dict[str, Any]:
    """Score records under the frozen three-task official-parser contract."""

    if not isinstance(records, (list, tuple)) or not records:
        raise SpatialEvalOfficialMetricError("records must be a non-empty list/tuple")
    if expected_count is not None and len(records) != expected_count:
        raise SpatialEvalOfficialMetricError(
            f"record count {len(records)} != frozen expected count {expected_count}"
        )
    module = official_module or load_official_evaluator()

    seen: set[str] = set()
    evaluated: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sample_id, task, question_id = resolve_identity(record)
        if sample_id in seen:
            raise SpatialEvalOfficialMetricError(f"duplicate sample id {sample_id!r}")
        seen.add(sample_id)
        row = _evaluate_with_identity(sample_id, task, question_id, record, module)
        evaluated.append(row)
        by_task[task].append(row)
        by_task_qid[f"{task}::{question_id}"].append(row)

    # The frozen 4500 distribution lock is authoritative and checked first.
    if expected_count == EXPECTED_PRIMARY_COUNT:
        counts = {task: len(by_task.get(task, ())) for task in TASKS}
        if any(value != EXPECTED_PER_TASK_COUNT for value in counts.values()):
            raise SpatialEvalOfficialMetricError(
                f"frozen task counts must be {EXPECTED_PER_TASK_COUNT} each, got {counts}"
            )
        qid_counts = {
            f"{task}::{qid}": len(by_task_qid.get(f"{task}::{qid}", ()))
            for task in TASKS
            for qid in QUESTION_IDS
        }
        if any(value != EXPECTED_PER_TASK_QID_COUNT for value in qid_counts.values()):
            raise SpatialEvalOfficialMetricError(
                "frozen task::qid counts must be "
                f"{EXPECTED_PER_TASK_QID_COUNT} each, got {qid_counts}"
            )

    missing_tasks = [task for task in TASKS if task not in by_task]
    if require_all_tasks and missing_tasks:
        raise SpatialEvalOfficialMetricError(
            f"three-task macro requires all tasks, missing {missing_tasks}"
        )

    per_task = {task: _aggregate(by_task[task]) for task in sorted(by_task)}
    per_task_qid = {
        key: _aggregate(by_task_qid[key]) for key in sorted(by_task_qid)
    }
    macro_tasks = [task for task in TASKS if task in per_task]
    macro_official = [
        float(per_task[task]["official_compatible_accuracy"]) for task in macro_tasks
    ]
    macro_strict = [
        float(per_task[task]["strict_equal_accuracy"]) for task in macro_tasks
    ]
    macro_project_summary = {
        "weighting": "equal_task",
        "task_count": len(macro_tasks),
        "tasks": macro_tasks,
        "official_compatible_accuracy": sum(macro_official) / len(macro_official),
        "strict_equal_accuracy": sum(macro_strict) / len(macro_strict),
    }

    result: dict[str, Any] = {
        "metric": METRIC_NAME,
        "official_commit": OFFICIAL_COMMIT,
        "model_name_for_official_parser": MODEL_NAME,
        "official_total_score_exists": False,
        "metric_name_marks_aggregation": (
            "the macro/micro fields are project aggregation, not an official "
            "SpatialEval total score"
        ),
        "project_aggregation_notice": (
            "micro and equal-task macro are project-defined summaries, not "
            "official SpatialEval total scores"
        ),
        "expected_count": expected_count,
        "micro_project_summary": _aggregate(evaluated),
        "macro_project_summary": macro_project_summary,
        "equal_task_macro_project_summary": macro_project_summary,
        "per_task": per_task,
        "per_task_question_id": per_task_qid,
    }
    if include_record_rows:
        result["evaluated_records"] = evaluated
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SpatialEvalOfficialMetricError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(record, dict):
                raise SpatialEvalOfficialMetricError(
                    f"line {line_number} of {path} is not a JSON object"
                )
            records.append(record)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="output JSON")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_PRIMARY_COUNT,
        help="locked record count (default 4500 = 3 tasks x 1500)",
    )
    parser.add_argument(
        "--no-count-lock",
        action="store_true",
        help="do not lock the record count (partial/debug runs)",
    )
    parser.add_argument(
        "--allow-partial-tasks",
        action="store_true",
        help="do not require all three tasks for the macro",
    )
    parser.add_argument(
        "--no-record-rows",
        action="store_true",
        help="omit the per-record diagnostic rows from the JSON output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    expected_count = None if args.no_count_lock else args.expected_count
    result = score_records(
        load_jsonl(args.input),
        expected_count=expected_count,
        require_all_tasks=not args.allow_partial_tasks,
        include_record_rows=not args.no_record_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
