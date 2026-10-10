#!/usr/bin/env python3
"""Shared contracts for the B1 answer-margin Oracle challenge/control pilot24 v2.

This is the *outcome-enriched diagnostic* companion to the random v1 pilot
(``evaluation/b1_answer_margin_oracle.py``).  It keeps the exact same
label-aware oracle mechanics (re-exported below) but changes only the **sample
construction**:

* six fixed ``source x task`` cells, four samples each (24 total):
  ADE20K x Count, ADE20K x Relation, COCO x Count, COCO x Relation,
  Omni3D x Depth, Omni3D x Distance;
* inside every cell exactly two **challenge** samples (dense BF16 correct AND
  corrected integer-L2 B1/K96 wrong) and two **regression-control** samples
  (dense BF16 correct AND corrected integer-L2 B1/K96 correct);
* deterministic ranking by ``sha256(str(sample_id))`` then sample id;
* across the eight Omni3D picks, ``Omni3D_Hypersim`` / ``Omni3D_SUNRGBD`` /
  ``Omni3D_nuScenes`` must each contribute at least two samples.

Scientific status (must never be softened)
------------------------------------------
* Selection uses the frozen dense and corrected-L2 outputs on purpose.  This is
  **outcome/label leakage**: the manifest is enriched on the very outcome it
  later diagnoses.  It is non-deployable, non-representative and forbidden for
  official absolute accuracy claims.
* The oracle still uses the official ground-truth answer branch token.  That is
  ground-truth answer leakage and can only ever be a falsifiable upper bound.
* 24 samples are a screening decision, never official CV-Bench accuracy and
  never an FPGA / latency / acceleration claim.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.b1_answer_margin_oracle import (
    ALLOWED_ARMS,
    ARM_ORACLE,
    BUDGET,
    CUTPOINT,
    EXPECTED_CONTEXT,
    EXPECTED_OUTPUT,
    EXPECTED_REAL,
    LABEL_AWARE_ORACLE,
    LOCAL_CANDIDATE_CAP,
    MAX_NEW_TOKENS,
    PRUNE_AFTER_LAYER,
    REGION_GRID_FIXED,
    SALIENCY_DEFINITION,
    OraclePilotError,
    array_sha256,
    assert_layout_contract,
    build_oracle_scorer,
    file_sha256,
    group_saliency_from_grad_activation,
    inject_oracle_selection,
    layout_contract,
    record_sha256,
    retained_mass,
    saliency_summary,
    selection_jaccard,
    selection_overlap,
    stable_sha256_rank,
)

__all__ = [
    "ALLOWED_ARMS",
    "ARM_ORACLE",
    "BUDGET",
    "CELLS",
    "CELL_BY_KEY",
    "CHALLENGE",
    "CONTROL",
    "CUTPOINT",
    "DESIGN",
    "EXPECTED_CONTEXT",
    "EXPECTED_OUTPUT",
    "EXPECTED_REAL",
    "GATE_CHALLENGE_REPAIR_MIN",
    "GATE_CONTROL_REGRESSION_MAX",
    "GATE_ORACLE_PARSE_MIN",
    "LABEL_AWARE_ORACLE",
    "LOCAL_CANDIDATE_CAP",
    "MAX_NEW_TOKENS",
    "OMNI3D_MIN_PER_SUBDATASET",
    "OMNI3D_SUBDATASETS",
    "PER_CELL",
    "PER_STRATUM",
    "PILOT_SAMPLE_COUNT",
    "PRUNE_AFTER_LAYER",
    "REGION_GRID_FIXED",
    "SALIENCY_DEFINITION",
    "SELECTION_CRITERIA",
    "STATUS_GO",
    "STATUS_PREREGISTERED_NOT_EXECUTED",
    "STATUS_STOP",
    "STRATA",
    "Cell",
    "OracleChallengeControlError",
    "array_sha256",
    "assert_layout_contract",
    "build_oracle_scorer",
    "cell_counts",
    "challenge_control_label",
    "constrained_omni3d_selection",
    "file_sha256",
    "group_saliency_from_grad_activation",
    "inject_oracle_selection",
    "layout_contract",
    "load_exclusions",
    "omni3d_coverage",
    "omni3d_coverage_ok",
    "rank_key",
    "record_sha256",
    "retained_mass",
    "saliency_summary",
    "selection_jaccard",
    "selection_overlap",
    "stable_sha256_rank",
    "subdataset_of",
]

# ---------------------------------------------------------------------------
# Frozen pilot identity
# ---------------------------------------------------------------------------
DESIGN = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24"

PILOT_SAMPLE_COUNT = 24
PER_CELL = 4
PER_STRATUM = 2

CHALLENGE = "challenge"
CONTROL = "control"
STRATA = (CHALLENGE, CONTROL)

OMNI3D_SOURCE = "Omni3D"
OMNI3D_SUBDATASETS = (
    "Omni3D_Hypersim",
    "Omni3D_SUNRGBD",
    "Omni3D_nuScenes",
)
OMNI3D_MIN_PER_SUBDATASET = 2

#: Pre-registered decision gate for this outcome-enriched diagnostic.
GATE_CHALLENGE_REPAIR_MIN = 8  # of the 12 challenge samples
GATE_CONTROL_REGRESSION_MAX = 1  # of the 12 regression-control samples
GATE_ORACLE_PARSE_MIN = 24  # of 24

STATUS_PREREGISTERED_NOT_EXECUTED = "PREREGISTERED_NOT_EXECUTED"
STATUS_GO = "PILOT_GO"
STATUS_STOP = "PILOT_STOP"


class OracleChallengeControlError(ValueError):
    """Raised when a pure v2 helper violates its contract."""


@dataclass(frozen=True)
class Cell:
    """One fixed ``source x task`` cell."""

    source: str
    task: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.source, self.task)

    @property
    def source_group(self) -> str:
        return self.source

    def label(self) -> str:
        return f"{self.source} x {self.task}"


#: The six fixed cells, in the canonical manifest order.
CELLS: tuple[Cell, ...] = (
    Cell("ADE20K", "Count"),
    Cell("ADE20K", "Relation"),
    Cell("COCO", "Count"),
    Cell("COCO", "Relation"),
    Cell(OMNI3D_SOURCE, "Depth"),
    Cell(OMNI3D_SOURCE, "Distance"),
)
CELL_BY_KEY: dict[tuple[str, str], Cell] = {cell.key: cell for cell in CELLS}

SELECTION_CRITERIA = (
    "frozen datasets/cv_bench_full2638.jsonl manifest metadata only",
    "dense BF16 correct AND corrected integer-L2 B1/K96 outcome from the frozen "
    "JSONLs (deliberate outcome/label leakage for diagnostic sampling)",
    "real local tokenizer label branch contract (prefix + one distinct branch "
    "token + common suffix)",
    "processor-derived full_llm_visual_tokens > 96 (compression active at K=96)",
    "rank by sha256(str(sample_id)) then sample_id; first two per outcome "
    "stratum per cell (constrained lexicographic rank tuple for Omni3D "
    "subdataset coverage)",
)


# ---------------------------------------------------------------------------
# Ranking / grouping helpers
# ---------------------------------------------------------------------------
def rank_key(sample_id: Any) -> tuple[str, str]:
    """Stable deterministic order: ``sha256(str(sample_id))`` then id."""

    return (stable_sha256_rank(sample_id), str(sample_id))


def subdataset_of(candidate: Mapping[str, Any]) -> str:
    """Return the Omni3D sub-dataset of a candidate (``source_dataset``)."""

    for field in ("subdataset", "source_dataset"):
        if field in candidate:
            return str(candidate[field])
    raise OracleChallengeControlError(
        "candidate has neither 'subdataset' nor 'source_dataset'"
    )


def challenge_control_label(
    dense_correct: bool, control_correct: bool
) -> str | None:
    """Return the eligible stratum, or ``None`` when the sample is ineligible.

    Both strata require the dense BF16 arm to be correct; the stratum is then
    decided by whether the corrected integer-L2 B1/K96 arm is wrong
    (``challenge``) or correct (``control``).
    """

    if not dense_correct:
        return None
    return CONTROL if control_correct else CHALLENGE


def cell_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        cell = str(record.get("cell") or f"{record['source']} x {record['task']}")
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def omni3d_coverage(
    records: Sequence[Mapping[str, Any]],
    subdatasets: Sequence[str] = OMNI3D_SUBDATASETS,
) -> dict[str, int]:
    counts = {name: 0 for name in subdatasets}
    for record in records:
        name = subdataset_of(record)
        if name in counts:
            counts[name] += 1
    return counts


def omni3d_coverage_ok(
    records: Sequence[Mapping[str, Any]],
    *,
    min_per_subdataset: int = OMNI3D_MIN_PER_SUBDATASET,
    subdatasets: Sequence[str] = OMNI3D_SUBDATASETS,
) -> bool:
    counts = omni3d_coverage(records, subdatasets)
    return all(counts[name] >= min_per_subdataset for name in subdatasets)


# ---------------------------------------------------------------------------
# Constrained Omni3D sub-dataset coverage
# ---------------------------------------------------------------------------
def _feasible(
    remaining_caps: Sequence[int],
    available: Sequence[Sequence[int]],
    shortfall: Sequence[int],
) -> bool:
    """Exact feasibility of finishing with the required sub-dataset coverage."""

    sub_count = len(shortfall)
    if sum(remaining_caps) < sum(shortfall):
        return False

    options: list[list[tuple[int, ...]]] = []
    for bucket_index, cap in enumerate(remaining_caps):
        bucket_options: list[tuple[int, ...]] = []
        bucket_available = available[bucket_index]

        def _rec(sub: int, remaining: int, prefix: tuple[int, ...]) -> None:
            if sub == sub_count - 1:
                if remaining <= bucket_available[sub]:
                    bucket_options.append(prefix + (remaining,))
                return
            upper = min(remaining, bucket_available[sub])
            for value in range(0, upper + 1):
                _rec(sub + 1, remaining - value, prefix + (value,))

        _rec(0, int(cap), ())
        if not bucket_options:
            return False
        options.append(bucket_options)

    reachable: set[tuple[int, ...]] = {(0,) * sub_count}
    for bucket_options in options:
        new: set[tuple[int, ...]] = set()
        for state in reachable:
            for option in bucket_options:
                new.add(
                    tuple(
                        min(shortfall[d], state[d] + option[d])
                        for d in range(sub_count)
                    )
                )
        reachable = new
        if not reachable:
            return False
    return any(
        all(state[d] >= shortfall[d] for d in range(sub_count))
        for state in reachable
    )


def constrained_omni3d_selection(
    buckets: Mapping[Any, Sequence[Mapping[str, Any]]],
    *,
    subdatasets: Sequence[str] = OMNI3D_SUBDATASETS,
    per_bucket: int = PER_STRATUM,
    min_per_subdataset: int = OMNI3D_MIN_PER_SUBDATASET,
    key: Callable[[Mapping[str, Any]], tuple[str, str]] = None,  # type: ignore[assignment]
) -> tuple[dict[Any, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministically cover all Omni3D sub-datasets with minimal ranks.

    Every bucket (``task::stratum``) contributes exactly ``per_bucket`` picks.
    Among all selections that meet the per-sub-dataset minimum, this returns the
    one whose **globally ordered ``(sha256 rank, sample_id)`` tuple** is
    lexicographically smallest.  It is implemented as a greedy over the global
    rank order that includes a candidate iff an exact feasibility oracle still
    permits a valid completion from the not-yet-considered candidates.  The
    result is order-independent and reproducible.

    Raises :class:`OracleChallengeControlError` when no feasible selection
    exists (fail closed).
    """

    if key is None:
        key = lambda candidate: rank_key(candidate["sample_id"])  # noqa: E731
    bucket_keys = list(buckets)
    if not bucket_keys:
        raise OracleChallengeControlError("at least one bucket is required")
    if isinstance(per_bucket, bool) or not isinstance(per_bucket, int) or per_bucket <= 0:
        raise OracleChallengeControlError("per_bucket must be a positive integer")
    sub_index = {name: index for index, name in enumerate(subdatasets)}
    sub_count = len(subdatasets)

    flat: list[tuple[str, str, int, int, dict[str, Any]]] = []
    for bucket_index, bucket_key in enumerate(bucket_keys):
        for candidate in buckets[bucket_key]:
            sample_id = str(candidate["sample_id"])
            subdataset = subdataset_of(candidate)
            if subdataset not in sub_index:
                raise OracleChallengeControlError(
                    f"candidate {sample_id!r} has unknown subdataset {subdataset!r}"
                )
            rank, tie = key(candidate)
            flat.append(
                (rank, tie, bucket_index, sub_index[subdataset], dict(candidate))
            )
    flat.sort(key=lambda item: (item[0], item[1]))

    total = len(flat)
    suffix: list[tuple[tuple[int, ...], ...]] = [None] * (total + 1)  # type: ignore[list-item]
    suffix[total] = tuple((0,) * sub_count for _ in bucket_keys)
    for index in range(total - 1, -1, -1):
        previous = [list(row) for row in suffix[index + 1]]
        previous[flat[index][2]][flat[index][3]] += 1
        suffix[index] = tuple(tuple(row) for row in previous)

    chosen: dict[Any, list[dict[str, Any]]] = {bucket_key: [] for bucket_key in bucket_keys}
    coverage = [0] * sub_count
    for index, (_rank, _tie, bucket_index, sub_index_value, candidate) in enumerate(flat):
        bucket_key = bucket_keys[bucket_index]
        if len(chosen[bucket_key]) >= per_bucket:
            continue
        caps = [per_bucket - len(chosen[item]) for item in bucket_keys]
        caps[bucket_index] -= 1
        tentative = list(coverage)
        tentative[sub_index_value] += 1
        shortfall = [
            max(0, min_per_subdataset - tentative[d]) for d in range(sub_count)
        ]
        if _feasible(caps, suffix[index + 1], shortfall):
            chosen[bucket_key].append(candidate)
            coverage = tentative

    if any(len(chosen[bucket_key]) != per_bucket for bucket_key in bucket_keys):
        raise OracleChallengeControlError(
            "no feasible Omni3D selection covers the sub-dataset minimum; fail closed"
        )
    metadata = {
        "algorithm": "greedy_lexicographic_min_sorted_sha256_rank_tuple",
        "bucket_order": [str(item) for item in bucket_keys],
        "per_bucket": per_bucket,
        "subdatasets": list(subdatasets),
        "min_per_subdataset": min_per_subdataset,
        "coverage": {
            name: coverage[index] for index, name in enumerate(subdatasets)
        },
        "selected_sample_ids": [
            str(candidate["sample_id"])
            for bucket_key in bucket_keys
            for candidate in chosen[bucket_key]
        ],
    }
    return chosen, metadata


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------
def load_exclusions(
    root: Path,
    *,
    full_manifest: Path | None = None,
    screening_manifest: Path | None = None,
    counterfactual_manifest: Path | None = None,
    random_v1_manifest: Path | None = None,
) -> dict[str, Any]:
    """Load the frozen exclusion union (sample 1 + screening100 + cf100 + v1)."""

    root = Path(root)
    paths = {
        "screening100": screening_manifest
        or root / "datasets" / "cv_bench_screening100.jsonl",
        "counterfactual100": counterfactual_manifest
        or root
        / "datasets"
        / "2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl",
        "random_v1_pilot24": random_v1_manifest
        or root / "datasets" / "2026-10-05-b1-answer-margin-oracle-pilot24.jsonl",
    }
    sets: dict[str, set[str]] = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.is_file():
            raise OracleChallengeControlError(
                f"exclusion manifest {label!r} is missing: {path} (fail closed)"
            )
        sets[label] = {
            str(json.loads(line)["sample_id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    manual = {"1"}
    union = set(manual)
    for values in sets.values():
        union |= values
    return {
        "sample_id_1": sorted(manual),
        "screening100": sorted(sets["screening100"]),
        "counterfactual100": sorted(sets["counterfactual100"]),
        "random_v1_pilot24": sorted(sets["random_v1_pilot24"]),
        "all": sorted(union),
        "count": len(union),
        "counts": {label: len(values) for label, values in sets.items()},
    }
