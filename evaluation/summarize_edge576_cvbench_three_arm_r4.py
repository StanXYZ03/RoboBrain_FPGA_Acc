#!/usr/bin/env python3
"""CPU-only, additive, fail-closed summariser for the 2026-10-07 edge576 r4 run.

This module reads the three completed full2638 raw arm JSONLs produced for the
edge576 development run and emits a quality / latency / token summary.  It is
strictly additive: it imports the frozen helpers and constants, and it never
writes to, repairs or normalises any raw arm JSONL, the r4 preregistration
config, the model, ``CURRENT_STATUS.md`` or any existing report.

Contract enforced fail-closed
-----------------------------
* each arm is exactly ``2638`` records with unique ``sample_id`` and all three
  arms share exactly the same ``sample_id`` set (and the pinned edge576
  manifest set);
* each arm's declared file SHA256 equals the frozen pin;
* every record carries the arm / mechanism / grid / G / K contract, a valid
  ``record_sha256``, an *explicit* ``parsed_answer`` (``str`` or ``None``) and a
  real ``bool`` ``correct``;
* the official score is produced **only** by
  :func:`evaluation.cv_bench_metric_v2.compute_cv_bench_metric` with the
  explicit ``correct_key='correct'`` / ``parsed_answer_key='parsed_answer'`` and
  ``expected_count=2638``; the parsed-answer key is never inferred;
* paired deltas reuse the exact resampling / percentile semantics of the frozen
  :func:`evaluation.build_cv_bench_pruning_report.paired_bootstrap_delta`
  (``task::source_dataset`` strata, fixed seed, fixed repeats).

No threshold is invented here.  The only pass/fail thresholds reported come
verbatim from the frozen r4 preregistration config and are labelled as such.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import cv_bench_metric_v2
from evaluation.build_cv_bench_pruning_report import stats as _frozen_stats
from evaluation.edge576_cvbench_runner_contract import (
    ARM_DENSE,
    ARM_EXTRA_RECORD_FIELDS,
    ARM_L2,
    ARM_MECHANISM_IDS,
    ARM_OUTPUT_PATHS,
    ARM_V2,
    ARMS,
    ATTENTION_IMPLEMENTATION,
    DECODING,
    DO_SAMPLE,
    DTYPE,
    EDGE576_MANIFEST,
    EDGE576_MANIFEST_SHA256,
    EDGE576_SAMPLE_COUNT,
    K_BUDGET,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    REQUIRED_EDGE_RECORD_FIELDS,
    record_sha256 as contract_record_sha256,
)
from evaluation.metrics import load_jsonl

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Frozen inputs and outputs
# ---------------------------------------------------------------------------
#: SHA256 pins supplied with the completed run (the same three files that the
#: r4 config declares as arms).
INPUT_SHA256: dict[str, str] = {
    ARM_DENSE: "707868c61e683cefc24962bdf3c2b508bd2286d355d4427200da78533190e784",
    ARM_L2: "e11dba93b324f5bcce18b84434b2cda2d022a09e66f2cee2db895dcc4bf88a8c",
    ARM_V2: "8b2a28854ec8741b53be61bfd461d1e4170be2f3d9bef863ec05222cc2f792f8",
}

R4_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json"
)

REPORTS_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-08-edge576-cvbench-three-arm-r4-summary.json"
)
REPORTS_CSV = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-08-edge576-cvbench-three-arm-r4-summary.csv"
)
REPORTS_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-08-edge576-cvbench-three-arm-r4-summary.md"
)
ARTIFACTS_JSON = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-08-edge576-cvbench-three-arm-r4-summary.json"
)

SEED = 20261007
REPEATS = 5000
CONFIDENCE = 0.95
EXPECTED_COUNT = EDGE576_SAMPLE_COUNT

#: Short keys used inside the paired joined table.
ARM_SHORT = {ARM_DENSE: "dense", ARM_L2: "l2", ARM_V2: "v2"}
RESULT_KEY = {"dense": "dense_correct", "l2": "l2_correct", "v2": "v2_correct"}

#: Metric identity, recorded verbatim so the report is self-describing.
METRIC_MODULE = "evaluation/cv_bench_metric_v2.py"
METRIC_WEIGHTING = "((accuracy_ADE20K + accuracy_COCO) / 2 + accuracy_Omni3D) / 2"
CORRECT_KEY = "correct"
PARSED_ANSWER_KEY = "parsed_answer"

#: Frozen percentile CI indices used by the frozen paired bootstrap.
CI_LOW_INDEX_FACTOR = 0.025
CI_HIGH_INDEX_FACTOR = 0.975

#: Honest latency field mapping.  A ``None`` value means the arm's producer did
#: not record that quantity and the summary must report ``null`` rather than
#: substitute a different stage.  Candidate tuples are tried left-to-right per
#: record.
LATENCY_MAPPING: dict[str, dict[str, tuple[str, ...] | None]] = {
    "application_end_to_end_seconds": {
        ARM_DENSE: None,
        ARM_L2: ("application_end_to_end_seconds",),
        ARM_V2: ("application_end_to_end_seconds",),
    },
    "model_inference_seconds": {
        # The frozen dense producer measures inference from just before the
        # vision call through decode as ``elapsed_seconds``; the L2/v2 producers
        # record the equivalent window as ``model_inference_seconds``.
        ARM_DENSE: ("elapsed_seconds",),
        ARM_L2: ("model_inference_seconds",),
        ARM_V2: ("model_inference_seconds",),
    },
    "vision_seconds": {
        ARM_DENSE: ("vision_encoder_seconds",),
        ARM_L2: ("vision_total_seconds", "vision_encoder_seconds"),
        ARM_V2: ("vision_total_seconds", "vision_encoder_seconds"),
    },
    "selector_cpu_roundtrip_seconds": {
        # Dense has no selector, so there is no honest field: explicit null.
        ARM_DENSE: None,
        ARM_L2: ("selector_cpu_roundtrip_seconds",),
        ARM_V2: ("selector_cpu_roundtrip_seconds",),
    },
    "llm_prefill_seconds": {
        ARM_DENSE: ("llm_prefill_seconds",),
        ARM_L2: ("llm_prefill_seconds",),
        ARM_V2: ("llm_prefill_seconds",),
    },
    "decode_seconds": {
        ARM_DENSE: ("decoder_seconds",),
        ARM_L2: ("decoder_seconds",),
        ARM_V2: ("decoder_seconds",),
    },
}

#: Quantities that must never be presented as the primary metric.
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

DELTA_PAIRS = (
    ("l2_minus_dense", "l2", "dense"),
    ("v2_minus_dense", "v2", "dense"),
    ("v2_minus_l2", "v2", "l2"),
)


class SummaryError(RuntimeError):
    """Raised when the fail-closed summary contract is violated."""


def _fail(message: str) -> None:
    raise SummaryError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Loading and per-record validation
# ---------------------------------------------------------------------------
def load_arm_jsonl(
    path: Path,
    arm: str,
    *,
    expected_sha256: str,
    expected_count: int = EXPECTED_COUNT,
) -> tuple[list[dict[str, Any]], str]:
    """Load one arm after verifying its SHA256, count and id uniqueness."""

    _require(arm in ARMS, f"unknown arm {arm!r}")
    path = Path(path)
    _require(path.is_file(), f"missing required arm file {path} (fail closed)")
    actual_sha = sha256_file(path)
    _require(
        actual_sha == expected_sha256,
        f"{arm} SHA256 {actual_sha} != pinned {expected_sha256} (fail closed)",
    )
    records = load_jsonl(path)
    _require(
        len(records) == expected_count,
        f"{arm} record count {len(records)} != expected {expected_count} "
        "(fail closed)",
    )
    ids = [str(record.get("sample_id")) for record in records]
    _require(
        len(set(ids)) == len(ids),
        f"{arm} has duplicate sample_id values (fail closed)",
    )
    return records, actual_sha


def _require_bool(record: Mapping[str, Any], field: str, label: str) -> bool:
    value = record[field]
    _require(
        isinstance(value, bool),
        f"{label} {field} must be a real bool, got {type(value).__name__} "
        "(fail closed)",
    )
    return value


def _require_number(
    record: Mapping[str, Any], field: str, label: str
) -> float:
    value = record[field]
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} {field} must be numeric, got {type(value).__name__} "
        "(fail closed)",
    )
    return float(value)


def validate_arm_record(
    record: Mapping[str, Any], arm: str, *, label: str = "record"
) -> dict[str, Any]:
    """Fail-closed validation of a single arm record; returns its contract bits.

    Checks exactly the fields the task requires: arm / mechanism / grid / G / K,
    explicit ``parsed_answer``, real ``bool correct`` and the recomputed
    ``record_sha256``.
    """

    _require(arm in ARMS, f"unknown arm {arm!r}")
    _require(isinstance(record, Mapping), f"{label} is not a mapping")

    required = list(REQUIRED_EDGE_RECORD_FIELDS) + list(ARM_EXTRA_RECORD_FIELDS[arm])
    missing = [field for field in required if field not in record]
    _require(
        not missing,
        f"{label} is missing required fields {missing} (fail closed)",
    )

    _require(
        str(record.get("arm")) == arm,
        f"{label} arm={record.get('arm')!r} != {arm!r} (fail closed)",
    )
    expected_mechanism = ARM_MECHANISM_IDS[arm]
    _require(
        str(record.get("mechanism_id")) == expected_mechanism,
        f"{label} mechanism_id={record.get('mechanism_id')!r} != "
        f"{expected_mechanism!r} (fail closed)",
    )
    _require(
        record.get("manifest_sha256") == EDGE576_MANIFEST_SHA256,
        f"{label} manifest_sha256 differs from the pinned edge576 manifest "
        "(fail closed)",
    )

    grid = [int(value) for value in record["image_grid_thw"]]
    _require(
        len(grid) == 3 and grid[0] == 1 and grid[1] > 0 and grid[2] > 0,
        f"{label} image_grid_thw={grid!r} is not a positive single-image THW "
        "(fail closed)",
    )
    full_llm = int(record["full_llm_visual_tokens"])
    expected_g = (grid[0] * grid[1] * grid[2]) // 4
    _require(
        full_llm == expected_g,
        f"{label} G={full_llm} != grid-derived G={expected_g} (fail closed)",
    )

    output_group_count = int(record["output_group_count"])
    selected = [int(value) for value in record["selected_group_indices"]]
    _require(
        len(selected) == output_group_count,
        f"{label} has {len(selected)} selected groups != output_group_count "
        f"{output_group_count} (fail closed)",
    )
    _require(
        not selected
        or all(later > earlier for earlier, later in zip(selected, selected[1:])),
        f"{label} selected_group_indices are not strictly ascending (fail closed)",
    )

    if arm == ARM_DENSE:
        _require(
            output_group_count == full_llm,
            f"{label} dense output_group_count {output_group_count} != G "
            f"{full_llm} (fail closed)",
        )
    else:
        _require(
            int(record["requested_budget"]) == K_BUDGET,
            f"{label} requested_budget={record.get('requested_budget')!r} != "
            f"K={K_BUDGET} (fail closed)",
        )
        if _require_bool(record, "identity_bypass", label):
            _require(
                full_llm <= K_BUDGET,
                f"{label} identity_bypass with G={full_llm} > K={K_BUDGET} "
                "(fail closed)",
            )
            _require(
                output_group_count == full_llm,
                f"{label} identity bypass must emit G={full_llm}, got "
                f"{output_group_count} (fail closed)",
            )
        else:
            _require(
                output_group_count == K_BUDGET,
                f"{label} compressed sample must emit K={K_BUDGET}, got "
                f"{output_group_count} (fail closed)",
            )

    parsed = record["parsed_answer"]
    _require(
        parsed is None or isinstance(parsed, str),
        f"{label} parsed_answer must be str or None, got "
        f"{type(parsed).__name__} (fail closed)",
    )
    correct = _require_bool(record, "correct", label)

    for field, expected in (
        ("min_pixels", MIN_PIXELS),
        ("max_pixels", MAX_PIXELS),
        ("max_new_tokens", MAX_NEW_TOKENS),
        ("do_sample", DO_SAMPLE),
        ("decoding", DECODING),
        ("dtype", DTYPE),
        ("attention_implementation", ATTENTION_IMPLEMENTATION),
    ):
        _require(
            record.get(field) == expected,
            f"{label} {field}={record.get(field)!r} != pinned {expected!r} "
            "(fail closed)",
        )

    recomputed = contract_record_sha256(record)
    _require(
        str(record.get("record_sha256")) == recomputed,
        f"{label} record_sha256 mismatch (stale or corrupted record, fail "
        "closed)",
    )

    return {
        "sample_id": str(record["sample_id"]),
        "G": full_llm,
        "K": K_BUDGET if arm != ARM_DENSE else None,
        "output_group_count": output_group_count,
        "identity_bypass": bool(record["identity_bypass"]),
        "correct": bool(correct),
        "parsed": parsed,
        "grid": grid,
    }


def validate_arm_records(
    records: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    """Validate every record in one arm and return aggregate contract counts."""

    identity_bypass_count = 0
    compressed_count = 0
    parsed_count = 0
    grid_ok = 0
    hash_ok = 0
    for index, record in enumerate(records):
        info = validate_arm_record(record, arm, label=f"{arm} record {index}")
        grid_ok += 1
        hash_ok += 1
        if info["parsed"] is not None:
            parsed_count += 1
        if arm != ARM_DENSE:
            if info["identity_bypass"]:
                identity_bypass_count += 1
            else:
                compressed_count += 1
    _require(
        grid_ok == len(records) and hash_ok == len(records),
        f"{arm} grid/hash validation did not cover every record (fail closed)",
    )
    return {
        "record_count": len(records),
        "unique_sample_id_count": len({str(r["sample_id"]) for r in records}),
        "grid_ok_count": grid_ok,
        "record_sha256_ok_count": hash_ok,
        "identity_bypass_count": (
            identity_bypass_count if arm != ARM_DENSE else None
        ),
        "compressed_count": compressed_count if arm != ARM_DENSE else None,
        "parsed_answer_present_count": parsed_count,
        "parsed_answer_null_count": len(records) - parsed_count,
    }


def load_edge576_manifest_ids(
    path: Path = EDGE576_MANIFEST,
    *,
    expected_sha256: str = EDGE576_MANIFEST_SHA256,
    expected_count: int = EXPECTED_COUNT,
) -> dict[str, dict[str, Any]]:
    """Load the pinned edge576 manifest as ``sample_id -> contract fields``."""

    path = Path(path)
    _require(path.is_file(), f"missing edge576 manifest {path} (fail closed)")
    actual_sha = sha256_file(path)
    _require(
        actual_sha == expected_sha256,
        f"edge576 manifest SHA256 {actual_sha} != pinned {expected_sha256} "
        "(fail closed)",
    )
    rows = load_jsonl(path)
    _require(
        len(rows) == expected_count,
        f"edge576 manifest count {len(rows)} != {expected_count} (fail closed)",
    )
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        _require(
            sample_id not in by_id,
            f"duplicate edge576 manifest sample_id {sample_id!r} (fail closed)",
        )
        by_id[sample_id] = row
    return by_id


def validate_manifest_alignment(
    records: Sequence[Mapping[str, Any]],
    arm: str,
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Check every arm record against the pinned edge576 manifest."""

    arm_ids = {str(record["sample_id"]) for record in records}
    manifest_ids = set(manifest_by_id)
    _require(
        arm_ids == manifest_ids,
        f"{arm} sample_id set does not exactly match the edge576 manifest "
        "(fail closed)",
    )
    _require(
        len(arm_ids) == len(records),
        f"{arm} has duplicate sample_id values (fail closed)",
    )
    for record in records:
        sample_id = str(record["sample_id"])
        meta = manifest_by_id[sample_id]
        for field in ("task", "source", "source_dataset", "type"):
            if field not in meta:
                continue
            _require(
                str(record.get(field)) == str(meta.get(field)),
                f"{arm} sample {sample_id!r} {field}={record.get(field)!r} != "
                f"manifest {meta.get(field)!r} (fail closed)",
            )
        _require(
            [int(v) for v in record["image_grid_thw"]]
            == [int(v) for v in meta["image_grid_thw"]],
            f"{arm} sample {sample_id!r} image_grid_thw differs from the "
            "edge576 manifest (fail closed)",
        )
        _require(
            int(record["full_llm_visual_tokens"])
            == int(meta["full_llm_visual_tokens"]),
            f"{arm} sample {sample_id!r} G differs from the edge576 manifest "
            "(fail closed)",
        )
    return {"sample_id_set_matches_manifest": True, "aligned_record_count": len(records)}


# ---------------------------------------------------------------------------
# Official metric and paired bootstrap
# ---------------------------------------------------------------------------
def arm_official_metric(
    records: Sequence[Mapping[str, Any]], *, expected_count: int = EXPECTED_COUNT
) -> dict[str, Any]:
    """Call the frozen v2 metric with the explicit r4 keys and expected count."""

    metric = cv_bench_metric_v2.compute_cv_bench_metric(
        records,
        correct_key=CORRECT_KEY,
        parsed_answer_key=PARSED_ANSWER_KEY,
        expected_count=expected_count,
    )
    return {
        "official_cv_bench_accuracy": metric["cv_bench_accuracy"],
        "metric_module": METRIC_MODULE,
        "metric_weighting": METRIC_WEIGHTING,
        "correct_key": CORRECT_KEY,
        "parsed_answer_key": PARSED_ANSWER_KEY,
        "expected_count": expected_count,
        "unweighted_sample_accuracy": metric["accuracy"],
        "accuracy_2d": metric["accuracy_2d"],
        "accuracy_3d": metric["accuracy_3d"],
        "source_accuracy": dict(metric["source_accuracy"]),
        "task_accuracy": dict(metric["task_accuracy"]),
        "parse_success_count": metric["parse_success_count"],
        "parse_failure_count": metric["parse_failure_count"],
        "parse_success_rate": metric["parse_success_rate"],
        "parse_failure_rate": 1.0 - metric["parse_success_rate"],
    }


def _weighted_cv_bench_accuracy(
    records: Sequence[Mapping[str, Any]], correct_key: str
) -> float:
    """Frozen official weighting over an arbitrary record subset.

    Identical arithmetic to the frozen ``score(...)["cv_bench_accuracy"]`` and to
    ``compute_cv_bench_metric``; re-derived here only so the paired bootstrap can
    score 5000 resamples without re-running full input validation each time.
    """

    source_accuracy: dict[str, float] = {}
    for source in cv_bench_metric_v2.CV_BENCH_SOURCES:
        subset = [record for record in records if record["source"] == source]
        _require(
            bool(subset),
            f"bootstrap resample is missing source {source!r} (fail closed)",
        )
        source_accuracy[source] = statistics.fmean(
            bool(record[correct_key]) for record in subset
        )
    accuracy_2d = (source_accuracy["ADE20K"] + source_accuracy["COCO"]) / 2
    return (accuracy_2d + source_accuracy["Omni3D"]) / 2


def paired_bootstrap_delta_v2(
    joined: Sequence[Mapping[str, Any]],
    left_key: str,
    right_key: str,
    *,
    seed: int = SEED,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """Percentile paired bootstrap delta, frozen semantics.

    Same strata (``task::source_dataset``), same ``random.Random(seed)`` draw
    order (``rng.choice`` once per in-stratum record per repeat), same
    ``cv_bench_accuracy`` weighting and the same sorted percentile indices as
    :func:`evaluation.build_cv_bench_pruning_report.paired_bootstrap_delta`.
    """

    _require(repeats > 1, "repeats must be > 1 (fail closed)")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in joined:
        groups[f"{row['task']}::{row['source_dataset']}"].append(row)
    _require(bool(groups), "no bootstrap strata (fail closed)")
    group_list = list(groups.values())

    rng = random.Random(seed)
    deltas: list[float] = []
    sources = cv_bench_metric_v2.CV_BENCH_SOURCES
    for _ in range(repeats):
        left_sums: dict[str, float] = defaultdict(float)
        right_sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for group in group_list:
            for _ in range(len(group)):
                record = rng.choice(group)
                source = record["source"]
                left_sums[source] += bool(record[left_key])
                right_sums[source] += bool(record[right_key])
                counts[source] += 1

        def _acc(sums: Mapping[str, float]) -> float:
            accuracy: dict[str, float] = {}
            for source in sources:
                _require(
                    counts.get(source, 0) > 0,
                    f"bootstrap resample is missing source {source!r} "
                    "(fail closed)",
                )
                accuracy[source] = sums[source] / counts[source]
            return ((accuracy["ADE20K"] + accuracy["COCO"]) / 2 + accuracy[
                "Omni3D"
            ]) / 2

        deltas.append(_acc(left_sums) - _acc(right_sums))

    deltas.sort()
    low = deltas[int(CI_LOW_INDEX_FACTOR * (repeats - 1))]
    high = deltas[int(CI_HIGH_INDEX_FACTOR * (repeats - 1))]
    point = _weighted_cv_bench_accuracy(joined, left_key) - (
        _weighted_cv_bench_accuracy(joined, right_key)
    )
    return {
        "left_key": left_key,
        "right_key": right_key,
        "point_estimate": point,
        "ci95": [low, high],
        "ci95_low": low,
        "ci95_high": high,
        "seed": seed,
        "repeats": repeats,
        "confidence": CONFIDENCE,
        "strata": "task::source_dataset",
        "strata_count": len(group_list),
    }


def build_joined(
    arms_records: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Join the three arms on ``sample_id`` after checking identical id sets."""

    id_sets = {arm: {str(r["sample_id"]) for r in arms_records[arm]} for arm in ARMS}
    reference = id_sets[ARM_DENSE]
    for arm in ARMS:
        _require(
            id_sets[arm] == reference,
            f"{arm} sample_id set differs from dense (fail closed)",
        )
    _require(
        all(len(arms_records[arm]) == len(reference) for arm in ARMS),
        "an arm has duplicate sample_id values (fail closed)",
    )

    by_arm = {
        arm: {str(r["sample_id"]): r for r in arms_records[arm]} for arm in ARMS
    }
    joined: list[dict[str, Any]] = []
    for sample_id in sorted(reference):
        meta = by_arm[ARM_DENSE][sample_id]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "task": meta["task"],
            "source": meta["source"],
            "source_dataset": meta["source_dataset"],
        }
        for arm in ARMS:
            record = by_arm[arm][sample_id]
            for field in ("task", "source", "source_dataset"):
                _require(
                    str(record[field]) == str(meta[field]),
                    f"sample {sample_id!r} {field} differs between arms "
                    "(fail closed)",
                )
            row[RESULT_KEY[ARM_SHORT[arm]]] = bool(record["correct"])
        joined.append(row)
    return joined


# ---------------------------------------------------------------------------
# Latency / token summaries
# ---------------------------------------------------------------------------
def latency_stats(values: Sequence[float]) -> dict[str, float] | None:
    """Frozen mean/median/p95/min/max/std plus an honest present count."""

    if not values:
        return None
    numeric = [float(value) for value in values]
    base = _frozen_stats(numeric)
    return {
        "count": len(numeric),
        "mean": base["mean"],
        "median": base["median"],
        "p95": base["p95"],
        "min": base["min"],
        "max": base["max"],
        "std": base["std"],
    }


def _latency_values(
    records: Sequence[Mapping[str, Any]], candidates: Sequence[str]
) -> list[float]:
    values: list[float] = []
    for record in records:
        for field in candidates:
            value = record.get(field)
            if value is None:
                continue
            _require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"latency field {field} must be numeric (fail closed)",
            )
            values.append(float(value))
            break
    return values


def latency_summary(
    records: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    """Per-arm latency stats with an explicit null for unrecorded fields."""

    summary: dict[str, Any] = {}
    mapping: dict[str, Any] = {}
    for field, per_arm in LATENCY_MAPPING.items():
        candidates = per_arm.get(arm)
        mapping[field] = list(candidates) if candidates else None
        values = _latency_values(records, candidates) if candidates else []
        summary[field] = latency_stats(values)
    return {"stats": summary, "field_mapping": mapping}


def token_summary(
    records: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    """G / K / retention / G<=96 identity census for one arm."""

    g_values = [int(record["full_llm_visual_tokens"]) for record in records]
    k_values = []
    for record in records:
        if arm != ARM_DENSE:
            k_values.append(int(record["requested_budget"]))
    retention = [
        int(record["output_group_count"]) / int(record["full_llm_visual_tokens"])
        for record in records
    ]
    histogram: dict[str, int] = defaultdict(int)
    for g in g_values:
        histogram[str(g)] += 1
    return {
        "G_stats": latency_stats(g_values),
        "K_stats": latency_stats(k_values) if k_values else None,
        "K_value": K_BUDGET if arm != ARM_DENSE else None,
        "mean_llm_visual_token_retention": statistics.fmean(retention),
        "min_llm_visual_token_retention": min(retention),
        "max_llm_visual_token_retention": max(retention),
        "g_le_96_identity_count": sum(1 for g in g_values if g <= 96),
        "g_gt_96_count": sum(1 for g in g_values if g > 96),
        "identity_bypass_flag_count": sum(
            1 for record in records if bool(record.get("identity_bypass"))
        ),
        "G_histogram": dict(sorted(histogram.items(), key=lambda kv: int(kv[0]))),
    }


# ---------------------------------------------------------------------------
# Pre-registered gates (read verbatim from the frozen r4 config)
# ---------------------------------------------------------------------------
def _parse_lower_bound(expr: Any) -> float:
    text = str(expr).strip()
    _require(
        text.startswith(">="),
        f"unsupported gate bound expression {expr!r} (fail closed)",
    )
    return float(text[2:])


def compute_gates(
    scores: Mapping[str, Mapping[str, Any]],
    deltas: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate only the gates frozen in the r4 config; invent none."""

    success = config["success_gates"]
    quality = success["quality_retention_vs_same_input_dense_edge576"]
    quality_threshold = float(quality["threshold"])
    quality_comparison = str(quality["comparison"])
    _require(
        quality_comparison == ">=",
        f"unsupported quality comparison {quality_comparison!r} (fail closed)",
    )
    dense_score = scores[ARM_DENSE]["official_cv_bench_accuracy"]
    v2_score = scores[ARM_V2]["official_cv_bench_accuracy"]
    retention = v2_score / dense_score

    relative = success["relative_to_corrected_integer_l2"]
    bootstrap = relative["paired_bootstrap"]
    lower_bound = _parse_lower_bound(bootstrap["lower_bound"])
    point_bound = _parse_lower_bound(relative["point_estimate"])
    v2_minus_l2 = deltas["v2_minus_l2"]

    return {
        "source_config": str(R4_CONFIG.relative_to(ROOT)),
        "quality_retention_vs_same_input_dense_edge576": {
            "arm": ARM_V2,
            "value": retention,
            "threshold": quality_threshold,
            "comparison": ">=",
            "passed": retention >= quality_threshold,
        },
        "relative_to_corrected_integer_l2": {
            "point_estimate": v2_minus_l2["point_estimate"],
            "point_estimate_bound": point_bound,
            "point_estimate_passed": v2_minus_l2["point_estimate"] >= point_bound,
            "ci95": list(v2_minus_l2["ci95"]),
            "lower_bound": lower_bound,
            "ci_lower_passed": v2_minus_l2["ci95_low"] >= lower_bound,
        },
        "official_cv_bench_accuracy_required": bool(
            success["official_cv_bench_accuracy"]["required"]
        ),
        "record_completeness_expected_count": int(
            success["record_completeness"]["sample_count"]
        ),
        "honest_parse_requires_100_percent": bool(
            success["honest_parse_reporting"]["requires_100_percent_parse"]
        ),
    }


# ---------------------------------------------------------------------------
# Top-level summary
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_edge576_three_arm(
    *,
    dense_path: Path = ARM_OUTPUT_PATHS[ARM_DENSE],
    l2_path: Path = ARM_OUTPUT_PATHS[ARM_L2],
    v2_path: Path = ARM_OUTPUT_PATHS[ARM_V2],
    manifest_path: Path = EDGE576_MANIFEST,
    config_path: Path = R4_CONFIG,
    seed: int = SEED,
    repeats: int = REPEATS,
) -> dict[str, Any]:
    """Run the full fail-closed summarisation and return the report payload."""

    config_path = Path(config_path)
    _require(
        config_path.is_file(),
        f"missing r4 config {config_path} (fail closed)",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frozen_seed = int(
        config["success_gates"]["relative_to_corrected_integer_l2"][
            "paired_bootstrap"
        ]["seed"]
    )
    frozen_repeats = int(
        config["success_gates"]["relative_to_corrected_integer_l2"][
            "paired_bootstrap"
        ]["repeats"]
    )
    _require(
        frozen_seed == seed and frozen_repeats == repeats,
        f"r4 config bootstrap seed/repeats {frozen_seed}/{frozen_repeats} != "
        f"requested {seed}/{repeats} (fail closed)",
    )

    manifest_by_id = load_edge576_manifest_ids(manifest_path)
    paths = {ARM_DENSE: Path(dense_path), ARM_L2: Path(l2_path), ARM_V2: Path(v2_path)}

    arms_records: dict[str, list[dict[str, Any]]] = {}
    sha_actual: dict[str, str] = {}
    arm_validation: dict[str, Any] = {}
    for arm in ARMS:
        records, actual_sha = load_arm_jsonl(
            paths[arm], arm, expected_sha256=INPUT_SHA256[arm]
        )
        arms_records[arm] = records
        sha_actual[arm] = actual_sha
        arm_validation[arm] = validate_arm_records(records, arm)
        arm_validation[arm].update(
            validate_manifest_alignment(records, arm, manifest_by_id)
        )

    id_sets = {
        arm: {str(record["sample_id"]) for record in arms_records[arm]}
        for arm in ARMS
    }
    _require(
        id_sets[ARM_DENSE] == id_sets[ARM_L2] == id_sets[ARM_V2],
        "the three arms do not share exactly the same sample_id set (fail closed)",
    )
    _require(
        all(len(s) == EXPECTED_COUNT for s in id_sets.values()),
        f"not every arm has {EXPECTED_COUNT} unique sample_ids (fail closed)",
    )

    scores = {
        arm: arm_official_metric(arms_records[arm], expected_count=EXPECTED_COUNT)
        for arm in ARMS
    }

    joined = build_joined(arms_records)
    deltas = {
        name: paired_bootstrap_delta_v2(
            joined,
            RESULT_KEY[left],
            RESULT_KEY[right],
            seed=seed,
            repeats=repeats,
        )
        for name, left, right in DELTA_PAIRS
    }
    relative = {
        "quality_retention_vs_same_input_dense_edge576": {
            arm: scores[arm]["official_cv_bench_accuracy"]
            / scores[ARM_DENSE]["official_cv_bench_accuracy"]
            for arm in ARMS
        },
        "deltas": deltas,
    }

    latency = {arm: latency_summary(arms_records[arm], arm) for arm in ARMS}
    tokens = {arm: token_summary(arms_records[arm], arm) for arm in ARMS}
    gates = compute_gates(scores, deltas, config)

    return {
        "report": "edge576_cvbench_three_arm_r4_summary",
        "schema_version": "1.0",
        "generated_at": _now(),
        "scope": {
            "sample_count": EXPECTED_COUNT,
            "arms": {
                arm: {
                    "role": ARM_MECHANISM_IDS[arm],
                    "output_path": str(paths[arm].relative_to(ROOT)),
                    "sha256": INPUT_SHA256[arm],
                }
                for arm in ARMS
            },
            "edge576_manifest": {
                "path": str(Path(manifest_path).relative_to(ROOT)),
                "sha256": EDGE576_MANIFEST_SHA256,
            },
            "r4_config": {
                "path": str(config_path.relative_to(ROOT)),
                "sha256": sha256_file(config_path),
            },
            "metric": {
                "module": METRIC_MODULE,
                "name": "cv_bench_accuracy",
                "weighting": METRIC_WEIGHTING,
                "correct_key": CORRECT_KEY,
                "parsed_answer_key": PARSED_ANSWER_KEY,
                "expected_count": EXPECTED_COUNT,
            },
            "paired_bootstrap": {
                "strata": "task::source_dataset",
                "seed": seed,
                "repeats": repeats,
                "confidence": CONFIDENCE,
            },
        },
        "validation": {
            "each_arm_count": {
                arm: arm_validation[arm]["record_count"] for arm in ARMS
            },
            "unique_sample_id_counts": {
                arm: arm_validation[arm]["unique_sample_id_count"] for arm in ARMS
            },
            "same_sample_id_set": True,
            "sample_id_set_matches_manifest": all(
                arm_validation[arm]["sample_id_set_matches_manifest"]
                for arm in ARMS
            ),
            "record_sha256_all_match": {
                arm: arm_validation[arm]["record_sha256_ok_count"] for arm in ARMS
            },
            "sha256_pins_match": True,
            "sha256_actual": sha_actual,
            "per_arm": arm_validation,
        },
        "scores": scores,
        "relative": relative,
        "tokens": tokens,
        "latency": {
            arm: latency[arm]["stats"] for arm in ARMS
        },
        "latency_field_mapping": {
            arm: latency[arm]["field_mapping"] for arm in ARMS
        },
        "pre_registered_gates": gates,
        "limitations": {
            "software_selector_latency_is_not_fpga_measurement": True,
            "selector_cpu_roundtrip_is_not_acceleration_claim": True,
            "full2638_is_development_not_external_validation": True,
            "no_thresholds_invented": True,
            "cpu_only_read_only_summary": True,
            "parse_failures_counted_as_wrong": True,
            "notes": [
                "The selector CPU roundtrip is a software Python timer; it is "
                "not an FPGA/board measurement and not an acceleration claim.",
                "The full2638 edge576 result is a development-set result, not "
                "an untouched external validation.",
                "Dense records no application end-to-end timestamp; that cell "
                "is reported as null rather than substituted.",
                "All pass/fail thresholds above are read from the frozen r4 "
                "preregistration config; none is invented here.",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _json_dumps(payload) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scores = summary["scores"]
    for arm in ARMS:
        score = scores[arm]
        for metric in (
            "official_cv_bench_accuracy",
            "unweighted_sample_accuracy",
            "accuracy_2d",
            "accuracy_3d",
            "parse_success_count",
            "parse_failure_count",
            "parse_success_rate",
            "parse_failure_rate",
        ):
            rows.append(
                {
                    "category": "score",
                    "arm": arm,
                    "metric": metric,
                    "value": score[metric],
                    "ci_low": "",
                    "ci_high": "",
                    "note": "",
                }
            )
        for source, value in score["source_accuracy"].items():
            rows.append(
                {
                    "category": "score",
                    "arm": arm,
                    "metric": f"source_accuracy::{source}",
                    "value": value,
                    "ci_low": "",
                    "ci_high": "",
                    "note": "",
                }
            )
        for task, value in score["task_accuracy"].items():
            rows.append(
                {
                    "category": "score",
                    "arm": arm,
                    "metric": f"task_accuracy::{task}",
                    "value": value,
                    "ci_low": "",
                    "ci_high": "",
                    "note": "",
                }
            )

    for name, delta in summary["relative"]["deltas"].items():
        rows.append(
            {
                "category": "delta",
                "arm": name,
                "metric": "point_estimate",
                "value": delta["point_estimate"],
                "ci_low": delta["ci95_low"],
                "ci_high": delta["ci95_high"],
                "note": f"strata={delta['strata']};seed={delta['seed']};"
                f"repeats={delta['repeats']}",
            }
        )
    for arm, value in summary["relative"][
        "quality_retention_vs_same_input_dense_edge576"
    ].items():
        rows.append(
            {
                "category": "retention",
                "arm": arm,
                "metric": "official_cv_bench_accuracy_vs_dense",
                "value": value,
                "ci_low": "",
                "ci_high": "",
                "note": "",
            }
        )

    for arm in ARMS:
        token = summary["tokens"][arm]
        for stat in ("mean", "median", "p95", "min", "max", "std"):
            rows.append(
                {
                    "category": "token",
                    "arm": arm,
                    "metric": f"G_{stat}",
                    "value": token["G_stats"][stat],
                    "ci_low": "",
                    "ci_high": "",
                    "note": f"G_count={token['G_stats']['count']}",
                }
            )
        if token["K_stats"] is not None:
            rows.append(
                {
                    "category": "token",
                    "arm": arm,
                    "metric": "K",
                    "value": token["K_stats"]["mean"],
                    "ci_low": "",
                    "ci_high": "",
                    "note": f"K_count={token['K_stats']['count']}",
                }
            )
        else:
            rows.append(
                {
                    "category": "token",
                    "arm": arm,
                    "metric": "K",
                    "value": None,
                    "ci_low": "",
                    "ci_high": "",
                    "note": "dense arm has no selector budget",
                }
            )
        for metric in (
            "mean_llm_visual_token_retention",
            "min_llm_visual_token_retention",
            "max_llm_visual_token_retention",
            "g_le_96_identity_count",
            "g_gt_96_count",
            "identity_bypass_flag_count",
        ):
            rows.append(
                {
                    "category": "token",
                    "arm": arm,
                    "metric": metric,
                    "value": token[metric],
                    "ci_low": "",
                    "ci_high": "",
                    "note": "",
                }
            )

    for arm in ARMS:
        for field, stats in summary["latency"][arm].items():
            if stats is None:
                rows.append(
                    {
                        "category": "latency",
                        "arm": arm,
                        "metric": f"{field}::mean",
                        "value": None,
                        "ci_low": "",
                        "ci_high": "",
                        "note": "field not recorded by this arm; explicit null",
                    }
                )
                continue
            for stat in ("mean", "median", "p95", "min", "max", "std"):
                rows.append(
                    {
                        "category": "latency",
                        "arm": arm,
                        "metric": f"{field}::{stat}",
                        "value": stats[stat],
                        "ci_low": "",
                        "ci_high": "",
                        "note": f"count={stats['count']}",
                    }
                )

    for gate_name, gate in summary["pre_registered_gates"].items():
        if not isinstance(gate, Mapping):
            continue
        value = gate.get("value")
        if value is None:
            value = gate.get("point_estimate")
        rows.append(
            {
                "category": "pre_registered_gate",
                "arm": gate.get("arm", ARM_V2),
                "metric": gate_name,
                "value": value,
                "ci_low": (gate.get("ci95") or ["", ""])[0],
                "ci_high": (gate.get("ci95") or ["", ""])[1],
                "note": json.dumps(
                    {
                        k: v
                        for k, v in gate.items()
                        if k in ("threshold", "passed", "lower_bound", "ci_lower_passed")
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def write_csv(path: Path, summary: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _csv_rows(summary)
    fieldnames = ["category", "arm", "metric", "value", "ci_low", "ci_high", "note"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return sha256_file(path)


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _latency_table(summary: Mapping[str, Any]) -> list[str]:
    fields = list(LATENCY_MAPPING)
    lines = [
        "| 阶段 | 字段映射 Dense | Dense mean / p95 | L2 mean / p95 | v2 mean / p95 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for field in fields:
        dense_map = summary["latency_field_mapping"][ARM_DENSE][field]
        dense_stats = summary["latency"][ARM_DENSE][field]
        l2_stats = summary["latency"][ARM_L2][field]
        v2_stats = summary["latency"][ARM_V2][field]
        lines.append(
            "| {field} | {mapping} | {d} | {l} | {v} |".format(
                field=field,
                mapping=", ".join(dense_map) if dense_map else "null",
                d=(
                    "null"
                    if dense_stats is None
                    else f"{_fmt(dense_stats['mean'])} / {_fmt(dense_stats['p95'])}"
                ),
                l=(
                    "null"
                    if l2_stats is None
                    else f"{_fmt(l2_stats['mean'])} / {_fmt(l2_stats['p95'])}"
                ),
                v=(
                    "null"
                    if v2_stats is None
                    else f"{_fmt(v2_stats['mean'])} / {_fmt(v2_stats['p95'])}"
                ),
            )
        )
    return lines


def write_md(path: Path, summary: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = summary["scores"]
    deltas = summary["relative"]["deltas"]
    gates = summary["pre_registered_gates"]

    lines: list[str] = []
    lines.append("# edge576 CV-Bench 三臂 r4 汇总（CPU-only 只读）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：`{summary['generated_at']}`")
    lines.append(
        f"- 样本数：每臂 `{EXPECTED_COUNT}`（唯一 `sample_id`，三臂同集合，且与"
        "固定 edge576 manifest 一致）"
    )
    lines.append(
        "- 主指标：官方 `cv_bench_accuracy = "
        "((ADE20K+COCO)/2 + Omni3D)/2`，由 "
        "`evaluation/cv_bench_metric_v2.compute_cv_bench_metric"
        "(correct_key='correct', parsed_answer_key='parsed_answer', "
        "expected_count=2638)` 现场计算。"
    )
    lines.append(
        f"- 配对 bootstrap：`task::source_dataset` 分层、seed `{SEED}`、"
        f"`{REPEATS}` 次、95% 百分位区间；实现语义与冻结 "
        "`evaluation.build_cv_bench_pruning_report.paired_bootstrap_delta` "
        "一致（聚焦测试逐位对照）。"
    )
    lines.append("")
    lines.append("## 口径与限制（必须先读）")
    lines.append("")
    lines.append(
        "- **软件 Selector 时间不是 FPGA 实测**：`selector_cpu_roundtrip_seconds` "
        "是 Python 软件桥计时，不代表 FPGA/上板时延，也不是加速主张。"
    )
    lines.append(
        "- **全量结果不是外部验证**：full2638 是被反复观测的发展集，本报告不是"
        "未触碰确认集，也不构成论文级精度主张。"
    )
    lines.append(
        "- **诚实解析**：`parsed_answer` 显式字段为 `str` 或 `null`；解析失败计为"
        "错误答案并单独报告失败数/率，不设 100% 解析门。"
    )
    lines.append(
        "- **未发明阈值**：本报告仅对预注册 r4 config 中冻结的门做参考判定，"
        "阈值原文读取，未自行新增或放宽任何阈值。"
    )
    lines.append(
        "- 本汇总 CPU-only、只读；未修改三份原始 JSONL、r4 config、模型或 "
        "`CURRENT_STATUS.md`。"
    )
    lines.append("")

    lines.append("## 1. 输入与 SHA256 校验")
    lines.append("")
    lines.append("| 臂 | mechanism_id | 原始 JSONL | SHA256 |")
    lines.append("| --- | --- | --- | --- |")
    for arm in ARMS:
        lines.append(
            "| {arm} | {mech} | `{path}` | `{sha}` |".format(
                arm=arm,
                mech=ARM_MECHANISM_IDS[arm],
                path=summary["scope"]["arms"][arm]["output_path"],
                sha=summary["scope"]["arms"][arm]["sha256"],
            )
        )
    lines.append("")
    lines.append(
        f"edge576 manifest `{summary['scope']['edge576_manifest']['path']}` "
        f"SHA256 `{summary['scope']['edge576_manifest']['sha256']}`；"
        f"r4 config `{summary['scope']['r4_config']['path']}` "
        f"SHA256 `{summary['scope']['r4_config']['sha256']}`。"
    )
    lines.append("")
    lines.append("## 2. 官方质量分（官方 cv_bench_accuracy 及来源/任务分解）")
    lines.append("")
    lines.append(
        "| 臂 | 官方 cv_bench_accuracy | 相对同输入 Dense | 未加权样本精度 | 2D | 3D |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        score = scores[arm]
        retention = summary["relative"][
            "quality_retention_vs_same_input_dense_edge576"
        ][arm]
        lines.append(
            f"| {arm} | {_fmt(score['official_cv_bench_accuracy'])} | "
            f"{_fmt(retention)} | {_fmt(score['unweighted_sample_accuracy'])} | "
            f"{_fmt(score['accuracy_2d'])} | {_fmt(score['accuracy_3d'])} |"
        )
    lines.append("")
    lines.append("| 臂 | ADE20K | COCO | Omni3D | Count | Depth | Distance | Relation |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        score = scores[arm]
        src = score["source_accuracy"]
        task = score["task_accuracy"]
        lines.append(
            f"| {arm} | {_fmt(src['ADE20K'])} | {_fmt(src['COCO'])} | "
            f"{_fmt(src['Omni3D'])} | {_fmt(task.get('Count'))} | "
            f"{_fmt(task.get('Depth'))} | {_fmt(task.get('Distance'))} | "
            f"{_fmt(task.get('Relation'))} |"
        )
    lines.append("")
    lines.append("| 臂 | 解析成功 | 解析失败 | 解析成功率 | 解析失败率 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for arm in ARMS:
        score = scores[arm]
        lines.append(
            f"| {arm} | {score['parse_success_count']} | "
            f"{score['parse_failure_count']} | "
            f"{_fmt(score['parse_success_rate'])} | "
            f"{_fmt(score['parse_failure_rate'])} |"
        )
    lines.append("")

    lines.append("## 3. 配对差值（官方 cv_bench_accuracy）")
    lines.append("")
    lines.append("| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 | 分层 |")
    lines.append("| --- | --- | --- | --- | --- |")
    label_map = {
        "l2_minus_dense": "L2 − Dense",
        "v2_minus_dense": "v2 − Dense",
        "v2_minus_l2": "v2 − L2",
    }
    for name, delta in deltas.items():
        lines.append(
            f"| {label_map[name]} | {_fmt(delta['point_estimate'])} | "
            f"{_fmt(delta['ci95_low'])} | {_fmt(delta['ci95_high'])} | "
            f"{delta['strata']} |"
        )
    lines.append("")

    lines.append("## 4. 预注册门参考判定（阈值来自 r4 config，未发明）")
    lines.append("")
    q = gates["quality_retention_vs_same_input_dense_edge576"]
    r = gates["relative_to_corrected_integer_l2"]
    lines.append(
        f"- v2 官方分 / Dense 官方分 = `{_fmt(q['value'])}`，预注册阈值 "
        f"`{q['comparison']} {q['threshold']}` → "
        f"**{'PASS' if q['passed'] else 'FAIL'}**。"
    )
    lines.append(
        f"- v2 − L2 点估计 = `{_fmt(r['point_estimate'])}`（阈值 "
        f"`>= {r['point_estimate_bound']}`）→ "
        f"**{'PASS' if r['point_estimate_passed'] else 'FAIL'}**；95% CI "
        f"`[{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}]`，下界阈值 "
        f"`>= {r['lower_bound']}` → "
        f"**{'PASS' if r['ci_lower_passed'] else 'FAIL'}**。"
    )
    lines.append("")

    lines.append("## 5. 时延（秒；null = 该臂未记录该字段）")
    lines.append("")
    lines.extend(_latency_table(summary))
    lines.append("")
    lines.append(
        "- Dense 未记录 `application_end_to_end_seconds`，故 Dense 的"
        "应用端到端为 `null`；Dense 的 `model_inference_seconds` 诚实映射其真实"
        "字段 `elapsed_seconds`（vision 调用到 decode），"
        "`vision_seconds` 映射 `vision_encoder_seconds`。"
    )
    lines.append(
        "- L2/v2 的应用端到端、模型推理、vision、selector CPU roundtrip 均来自各"
        "自真实字段；`vision_seconds` 取 `vision_total_seconds`。"
    )
    lines.append("")

    lines.append("## 6. Token G/K、平均保留率、G<=96 identity 数")
    lines.append("")
    lines.append(
        "| 臂 | G mean / median / p95 / min / max | K | 平均 LLM 视觉 token 保留率 | "
        "G<=96 数 | G>96 数 | identity_bypass 标志数 |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        token = summary["tokens"][arm]
        g = token["G_stats"]
        k = "null" if token["K_stats"] is None else str(token["K_value"])
        lines.append(
            f"| {arm} | {_fmt(g['mean'])} / {_fmt(g['median'])} / "
            f"{_fmt(g['p95'])} / {g['min']} / {g['max']} | {k} | "
            f"{_fmt(token['mean_llm_visual_token_retention'])} | "
            f"{token['g_le_96_identity_count']} | {token['g_gt_96_count']} | "
            f"{token['identity_bypass_flag_count']} |"
        )
    lines.append("")
    lines.append(
        "注：Dense 不裁剪（每样本 `output_group_count == G`，保留率 1.0），其 "
        "`identity_bypass` 字段恒为 true，属 producer 语义，不代表压缩臂的 "
        "G<=96 绕过；压缩臂的 G<=96 identity 数以上表为准。"
    )
    lines.append("")

    lines.append("## 7. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python3 evaluation/summarize_edge576_cvbench_three_arm_r4.py"
    )
    lines.append("```")
    lines.append("")

    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Read-back verification
# ---------------------------------------------------------------------------
def verify_artifacts_json(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read the written artifacts JSON and confirm the load-bearing values."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    checks: list[tuple[str, bool]] = []
    checks.append(("report_name", payload.get("report") == summary["report"]))
    checks.append(
        ("sample_count", payload["scope"]["sample_count"] == EXPECTED_COUNT)
    )
    for arm in ARMS:
        checks.append(
            (
                f"{arm}_score",
                abs(
                    payload["scores"][arm]["official_cv_bench_accuracy"]
                    - summary["scores"][arm]["official_cv_bench_accuracy"]
                )
                < 1e-15,
            )
        )
        checks.append(
            (
                f"{arm}_sha",
                payload["scope"]["arms"][arm]["sha256"] == INPUT_SHA256[arm],
            )
        )
    for name in deltas_names():
        checks.append(
            (
                f"delta_{name}",
                payload["relative"]["deltas"][name]["ci95"]
                == summary["relative"]["deltas"][name]["ci95"],
            )
        )
    failed = [name for name, ok in checks if not ok]
    _require(
        not failed,
        f"artifacts JSON read-back failed checks {failed} (fail closed)",
    )
    return {"path": str(Path(path)), "checks_passed": len(checks), "failed": []}


def deltas_names() -> tuple[str, ...]:
    return tuple(name for name, _, _ in DELTA_PAIRS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only additive summariser for the 2026-10-07 edge576 "
            "three-arm r4 full2638 run."
        )
    )
    parser.add_argument("--dense-jsonl", type=Path, default=ARM_OUTPUT_PATHS[ARM_DENSE])
    parser.add_argument("--l2-jsonl", type=Path, default=ARM_OUTPUT_PATHS[ARM_L2])
    parser.add_argument("--v2-jsonl", type=Path, default=ARM_OUTPUT_PATHS[ARM_V2])
    parser.add_argument("--manifest", type=Path, default=EDGE576_MANIFEST)
    parser.add_argument("--config", type=Path, default=R4_CONFIG)
    parser.add_argument("--reports-json", type=Path, default=REPORTS_JSON)
    parser.add_argument("--reports-csv", type=Path, default=REPORTS_CSV)
    parser.add_argument("--reports-md", type=Path, default=REPORTS_MD)
    parser.add_argument("--artifacts-json", type=Path, default=ARTIFACTS_JSON)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_edge576_three_arm(
        dense_path=args.dense_jsonl,
        l2_path=args.l2_jsonl,
        v2_path=args.v2_jsonl,
        manifest_path=args.manifest,
        config_path=args.config,
        seed=args.seed,
        repeats=args.repeats,
    )

    artifacts_sha = write_json(args.artifacts_json, summary)
    # The reports JSON is the same machine-readable payload plus output paths.
    reports_payload = {
        "report": "edge576_cvbench_three_arm_r4_summary_report",
        "generated_at": summary["generated_at"],
        "artifacts_json": {
            "path": str(Path(args.artifacts_json).relative_to(ROOT)),
            "sha256": artifacts_sha,
        },
        "summary": summary,
    }
    reports_sha = write_json(args.reports_json, reports_payload)
    csv_sha = write_csv(args.reports_csv, summary)
    md_sha = write_md(args.reports_md, summary)

    verification = verify_artifacts_json(args.artifacts_json, summary)

    print(_json_dumps({
        "artifacts_json": {"path": str(args.artifacts_json), "sha256": artifacts_sha},
        "reports_json": {"path": str(args.reports_json), "sha256": reports_sha},
        "reports_csv": {"path": str(args.reports_csv), "sha256": csv_sha},
        "reports_md": {"path": str(args.reports_md), "sha256": md_sha},
        "scores": {
            arm: summary["scores"][arm]["official_cv_bench_accuracy"]
            for arm in ARMS
        },
        "deltas": {
            name: {
                "point_estimate": summary["relative"]["deltas"][name]["point_estimate"],
                "ci95": summary["relative"]["deltas"][name]["ci95"],
            }
            for name in deltas_names()
        },
        "latency_application": {
            arm: summary["latency"][arm]["application_end_to_end_seconds"]
            for arm in ARMS
        },
        "read_back": verification,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
