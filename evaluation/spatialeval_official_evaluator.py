#!/usr/bin/env python3
"""Official SpatialEval VQA evaluator adapter (CPU-only, no weights, no CUDA).

This module loads the pinned upstream ``evaluation.py`` (SpatialEval commit
``d82ba382805265f205693cb83a30c4866ea6c35b``) with :mod:`importlib` and calls
the *original* ``extract_answer_from_text_spatialmap`` /
``extract_answer_from_text_mazenav`` / ``extract_answer_from_text_spatialgrid``
parsers with ``model_name="robobrain25"`` fixed.  No parser is copied or
rewritten here.

Main (official-compatible) comparison
-------------------------------------
The upstream line is reproduced verbatim for a successfully parsed answer::

    int(str(oracle_answer).lower() in str(parsed_answer).lower())

That is a **substring inclusion** test (oracle is the needle, the parsed answer
is the haystack), not equality.  Two upstream behaviours are deliberately made
fail-closed here instead of being reproduced, because the project forbids
silently counting them as anything but wrong:

* a parser that raises an exception counts **wrong** (upstream skips the row via
  ``except ValueError: continue`` and leaves it out of the numerator while still
  dividing by a line count that was incremented before the ``try``);
* a parser that returns ``None`` counts **wrong** (upstream would stringify it to
  ``"none"`` via ``str(None).lower()``, which can spuriously contain short
  oracles such as ``"no"``/``"one"``).

Aggregation naming
------------------
The upstream artifact defines only a per-task ``Acc = correct / line_count`` and
contains **no cross-task total**.  ``project_task_macro`` (mean of the three
per-task accuracies) and ``project_micro`` (pooled correct / pooled count) are
therefore *project-side* summaries and are explicitly labelled as such.  They
must never be reported as an official SpatialEval score.

All functions here are pure CPU; importing this module never loads a model
weight or touches CUDA.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Pinned official provenance
# ---------------------------------------------------------------------------
OFFICIAL_DIR = ROOT / "artifacts" / "provenance" / "spatialeval-official-d82ba382"
OFFICIAL_EVALUATION_PY = OFFICIAL_DIR / "evaluation.py"
OFFICIAL_EVALUATION_SHA256 = (
    "88d2a1b35abe02d9c517a513e8b1a5f442b61837b6cd742b1ec1fa23c6ccd952"
)
OFFICIAL_INFERENCE_VLM_PY = OFFICIAL_DIR / "inference_vlm.py"
OFFICIAL_INFERENCE_VLM_SHA256 = (
    "0a69b4cd14707001c6975ab9ae9113a90df2ce2cd73119bceabe99cfddb39337"
)
OFFICIAL_INFERENCE_CONFIGS_PY = OFFICIAL_DIR / "inference_configs.py"
OFFICIAL_INFERENCE_CONFIGS_SHA256 = (
    "79d9c59a8a45be78e18206bde8f6fc13c35827a208ffee99155b29efbbe44972"
)
OFFICIAL_COMMIT = "d82ba382805265f205693cb83a30c4866ea6c35b"

#: Fixed model name handed to the official per-model heuristic branches.
MODEL_NAME = "robobrain25"

#: Tasks for which the upstream evaluator actually defines a handler.
HANDLER_TASKS = ("spatialmap", "mazenav", "spatialgrid")
#: Task present in the dataset but *not* handled by the upstream evaluator.
NON_HANDLER_TASKS = ("spatialreal",)
TASKS_PER_ARM = 3
SAMPLES_PER_TASK = 1500
EXPECTED_SAMPLES = TASKS_PER_ARM * SAMPLES_PER_TASK  # 4500

#: The three parser names as they exist in the upstream artifact.
PARSER_NAMES = {
    "spatialmap": "extract_answer_from_text_spatialmap",
    "mazenav": "extract_answer_from_text_mazenav",
    "spatialgrid": "extract_answer_from_text_spatialgrid",
}

NORMALIZATION_DESCRIPTION = (
    "strip -> lower -> collapse whitespace -> strip trailing .!,?:; "
    "(project diagnostic only; not the official comparison)"
)


class SpatialEvalOfficialEvaluatorError(RuntimeError):
    """Raised when the official evaluator contract is violated (fail closed)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pandas_is_importable() -> bool:
    """Return whether the real pandas package can be imported."""

    try:  # pragma: no cover - depends on the environment
        import pandas  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def load_official_evaluation_module(
    path: Path = OFFICIAL_EVALUATION_PY,
    *,
    expected_sha256: str | None = OFFICIAL_EVALUATION_SHA256,
    module_name: str = "spatialeval_official_d82ba382_evaluation",
) -> types.ModuleType:
    """Load the pinned upstream ``evaluation.py`` and return the module.

    The file SHA256 is verified before execution, so a drifted official artifact
    can never be silently used.  The module is *not* registered in
    ``sys.modules`` and ``main`` is never invoked.
    """

    path = Path(path)
    if not path.is_file():
        raise SpatialEvalOfficialEvaluatorError(
            f"missing official evaluation.py {path} (fail closed)"
        )
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise SpatialEvalOfficialEvaluatorError(
                f"official evaluation.py SHA256 {actual} != pinned "
                f"{expected_sha256} (fail closed)"
            )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SpatialEvalOfficialEvaluatorError(
            f"cannot build an import spec for {path} (fail closed)"
        )
    module = importlib.util.module_from_spec(spec)
    stubbed = not _pandas_is_importable()
    saved = sys.modules.get("pandas")
    try:
        if stubbed:
            # The upstream artifact imports pandas only for the CSV writer in
            # ``main``; the parsers we call never touch it.
            sys.modules["pandas"] = types.ModuleType("pandas")
        spec.loader.exec_module(module)
    finally:
        if stubbed:
            if saved is None:
                sys.modules.pop("pandas", None)
            else:  # pragma: no cover - pandas present after all
                sys.modules["pandas"] = saved
    for parser_name in PARSER_NAMES.values():
        if not callable(getattr(module, parser_name, None)):
            raise SpatialEvalOfficialEvaluatorError(
                f"official module has no callable {parser_name!r} (fail closed)"
            )
    return module


_DEFAULT_MODULE: types.ModuleType | None = None


def official_parsers(module: types.ModuleType | None = None) -> dict[str, Callable[..., Any]]:
    """Return the three original parser callables, keyed by task."""

    global _DEFAULT_MODULE
    if module is None:
        if _DEFAULT_MODULE is None:
            _DEFAULT_MODULE = load_official_evaluation_module()
        module = _DEFAULT_MODULE
    parsers: dict[str, Callable[..., Any]] = {}
    for task, name in PARSER_NAMES.items():
        parser = getattr(module, name, None)
        if not callable(parser):
            raise SpatialEvalOfficialEvaluatorError(
                f"official module has no callable {name!r} (fail closed)"
            )
        parsers[task] = parser
    return parsers


# ---------------------------------------------------------------------------
# Sample-id structure
# ---------------------------------------------------------------------------
def task_question_id(sample_id: str) -> tuple[str, int]:
    """Split ``<task>.vqa.<int>[|sa_<int>].<question_id>`` like upstream does."""

    if not isinstance(sample_id, str) or not sample_id:
        raise SpatialEvalOfficialEvaluatorError(
            "sample_id must be a non-empty string (fail closed)"
        )
    parts = sample_id.split(".")
    if len(parts) < 2:
        raise SpatialEvalOfficialEvaluatorError(
            f"sample_id {sample_id!r} has no '<task>...<question_id>' structure "
            "(fail closed)"
        )
    task = parts[0]
    try:
        question_id = int(parts[-1])
    except ValueError as exc:
        raise SpatialEvalOfficialEvaluatorError(
            f"sample_id {sample_id!r} trailing question_id is not an int "
            "(fail closed)"
        ) from exc
    return task, question_id


def normalize_strict_answer(value: Any) -> str | None:
    """Project-side normalization used only by the strict-equality diagnostic."""

    if value is None:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\.,!?:;]+$", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _parse_one(
    task: str,
    output_text: Any,
    question_id: int,
    parsers: Mapping[str, Callable[..., Any]],
    *,
    model_name: str,
) -> tuple[Any, bool]:
    """Return ``(parsed_answer, raised)`` using the original upstream parser."""

    if not isinstance(output_text, str):
        output_text = "" if output_text is None else str(output_text)
    try:
        parsed = parsers[task](output_text, question_id, model_name)
    except Exception:
        return None, True
    return parsed, False


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    oracle_by_id: Mapping[str, Any],
    *,
    parsers: Mapping[str, Callable[..., Any]] | None = None,
    model_name: str = MODEL_NAME,
    expected_sample_ids: Sequence[str] | None = None,
    allow_excluded_tasks: bool = False,
    include_per_sample: bool = False,
) -> dict[str, Any]:
    """Score one arm with the official-compatible rule and the strict diagnostic.

    ``predictions`` is a sequence of mappings carrying at least ``sample_id`` and
    ``output_text``.  ``oracle_by_id`` maps ``sample_id`` to the frozen
    ``oracle_answer`` (taken from the pinned reference manifest, never from the
    prediction record).  Both the main official-compatible score and the
    project-side strict-equality diagnostic are returned.
    """

    if parsers is None:
        parsers = official_parsers()
    if not isinstance(predictions, Sequence) or isinstance(predictions, (str, bytes)):
        raise SpatialEvalOfficialEvaluatorError(
            "predictions must be a non-empty sequence (fail closed)"
        )
    expected_set: set[str] | None = None
    if expected_sample_ids is not None:
        expected = [str(value) for value in expected_sample_ids]
        if len(expected) != len(set(expected)):
            raise SpatialEvalOfficialEvaluatorError(
                "expected_sample_ids contains duplicates (fail closed)"
            )
        expected_set = set(expected)
        if len(predictions) != len(expected):
            raise SpatialEvalOfficialEvaluatorError(
                f"prediction count {len(predictions)} != expected {len(expected)} "
                "(fail closed)"
            )

    per_task: dict[str, dict[str, Any]] = {}
    per_task_qid: dict[str, dict[str, Any]] = {}
    strict_per_task: dict[str, dict[str, int]] = {}
    per_sample: list[dict[str, Any]] = []
    seen: set[str] = set()
    official_correct = 0
    strict_correct = 0
    parse_failures = 0
    total = 0

    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, Mapping):
            raise SpatialEvalOfficialEvaluatorError(
                f"prediction {index} is not a mapping (fail closed)"
            )
        for key in ("sample_id", "output_text"):
            if key not in prediction:
                raise SpatialEvalOfficialEvaluatorError(
                    f"prediction {index} is missing {key!r} (fail closed)"
                )
        sample_id = str(prediction["sample_id"])
        if sample_id in seen:
            raise SpatialEvalOfficialEvaluatorError(
                f"duplicate prediction sample_id {sample_id!r} (fail closed)"
            )
        seen.add(sample_id)
        task, question_id = task_question_id(sample_id)
        if task in NON_HANDLER_TASKS and not allow_excluded_tasks:
            raise SpatialEvalOfficialEvaluatorError(
                f"sample {sample_id!r} belongs to task {task!r} which the official "
                "evaluator has no handler for (fail closed)"
            )
        if task not in parsers:
            raise SpatialEvalOfficialEvaluatorError(
                f"no official parser for task {task!r} (fail closed)"
            )
        if sample_id not in oracle_by_id:
            raise SpatialEvalOfficialEvaluatorError(
                f"no frozen oracle_answer for sample {sample_id!r} (fail closed)"
            )
        oracle = oracle_by_id[sample_id]
        parsed, raised = _parse_one(
            task, prediction["output_text"], question_id, parsers, model_name=model_name
        )
        if parsed is None:
            official = 0
            parse_failures += 1
        else:
            official = int(
                str(oracle).lower() in str(parsed).lower()
            )
        normalized_oracle = normalize_strict_answer(oracle)
        normalized_parsed = normalize_strict_answer(parsed)
        strict = int(
            normalized_oracle is not None
            and normalized_parsed is not None
            and normalized_oracle == normalized_parsed
        )
        official_correct += official
        strict_correct += strict
        total += 1
        task_bucket = per_task.setdefault(
            task, {"count": 0, "correct": 0, "parse_failure_count": 0}
        )
        task_bucket["count"] += 1
        task_bucket["correct"] += official
        task_bucket["parse_failure_count"] += 1 if parsed is None else 0
        strict_bucket = strict_per_task.setdefault(task, {"count": 0, "correct": 0})
        strict_bucket["count"] += 1
        strict_bucket["correct"] += strict
        qid_bucket = per_task_qid.setdefault(
            f"{task}::{question_id}",
            {"task": task, "question_id": question_id, "count": 0, "correct": 0,
             "parse_failure_count": 0},
        )
        qid_bucket["count"] += 1
        qid_bucket["correct"] += official
        qid_bucket["parse_failure_count"] += 1 if parsed is None else 0
        if include_per_sample:
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "question_id": question_id,
                    "official_correct": int(official),
                    "strict_correct": int(strict),
                    "parse_failure": parsed is None,
                }
            )

    if total == 0:
        raise SpatialEvalOfficialEvaluatorError(
            "predictions is empty (fail closed)"
        )
    if expected_set is not None and seen != expected_set:
        missing = sorted(expected_set - seen)
        extra = sorted(seen - expected_set)
        raise SpatialEvalOfficialEvaluatorError(
            f"prediction sample set differs from expected: missing={missing[:5]} "
            f"extra={extra[:5]} (fail closed)"
        )

    for bucket in per_task.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"]
    for bucket in per_task_qid.values():
        bucket["accuracy"] = bucket["correct"] / bucket["count"]
    for task, bucket in strict_per_task.items():
        bucket["accuracy"] = bucket["correct"] / bucket["count"]

    present_tasks = [task for task in HANDLER_TASKS if task in per_task]
    if len(present_tasks) != len(per_task):
        unexpected = sorted(set(per_task) - set(HANDLER_TASKS))
        raise SpatialEvalOfficialEvaluatorError(
            f"predictions contain non-handler tasks {unexpected} (fail closed)"
        )
    macro = (
        sum(per_task[task]["accuracy"] for task in present_tasks) / len(present_tasks)
        if present_tasks
        else 0.0
    )
    strict_macro = (
        sum(strict_per_task[task]["accuracy"] for task in present_tasks)
        / len(present_tasks)
        if present_tasks
        else 0.0
    )
    return {
        "metric": "spatialeval_official_compatible_string_inclusion",
        "model_name": model_name,
        "official_semantics": {
            "comparison": "int(str(oracle_answer).lower() in str(parsed_answer).lower())",
            "none_or_exception": "counted_wrong",
            "official_cross_task_total_exists": False,
            "official_source": "artifacts/provenance/spatialeval-official-d82ba382/evaluation.py",
            "official_commit": OFFICIAL_COMMIT,
            "upstream_note": (
                "upstream skips ValueError rows and stringifies None to 'none'; "
                "this adapter is fail-closed and counts both as wrong"
            ),
        },
        "task_set": list(HANDLER_TASKS),
        "non_handler_tasks": list(NON_HANDLER_TASKS),
        "count": total,
        "correct": official_correct,
        "accuracy": official_correct / total,
        "parse_failure_count": parse_failures,
        "parse_success_count": total - parse_failures,
        "parse_success_rate": (total - parse_failures) / total,
        "per_task": {
            task: {
                "count": bucket["count"],
                "correct": bucket["correct"],
                "accuracy": bucket["accuracy"],
                "parse_failure_count": bucket["parse_failure_count"],
            }
            for task, bucket in sorted(per_task.items())
        },
        "per_task_qid": {
            key: {
                "task": bucket["task"],
                "question_id": bucket["question_id"],
                "count": bucket["count"],
                "correct": bucket["correct"],
                "accuracy": bucket["accuracy"],
                "parse_failure_count": bucket["parse_failure_count"],
            }
            for key, bucket in sorted(per_task_qid.items())
        },
        "project_task_macro": {
            "accuracy": macro,
            "tasks": present_tasks,
            "semantics": "project_task_macro_mean_of_handler_tasks_not_official_total",
        },
        "project_micro": {
            "accuracy": official_correct / total,
            "count": total,
            "correct": official_correct,
            "semantics": "project_pooled_micro_not_official_total",
        },
        "strict_equality_diagnostic": {
            "normalization": NORMALIZATION_DESCRIPTION,
            "is_official": False,
            "correct": strict_correct,
            "accuracy": strict_correct / total,
            "per_task": {
                task: {
                    "count": bucket["count"],
                    "correct": bucket["correct"],
                    "accuracy": bucket["accuracy"],
                }
                for task, bucket in sorted(strict_per_task.items())
            },
            "project_task_macro": {
                "accuracy": strict_macro,
                "tasks": present_tasks,
                "semantics": "project_task_macro_mean_of_handler_tasks_not_official_total",
            },
            "project_micro": {
                "accuracy": strict_correct / total,
                "count": total,
                "correct": strict_correct,
                "semantics": "project_pooled_micro_not_official_total",
            },
        },
        "per_sample": per_sample if include_per_sample else None,
    }
