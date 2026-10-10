#!/usr/bin/env python3
"""Shared pure contracts for the B1 answer-margin oracle K96 pilot.

This module is the *label-aware oracle upper bound* contract shared by the
CPU-only selection script, the (not executed) GPU runner and the fail-closed
summariser.  It is deliberately small and dependency-light: ``torch`` is
imported lazily inside the single GPU-only helper, so the selection script and
the summariser can import it on a CPU-only host.

Scientific status (must never be softened)
------------------------------------------
* The oracle score is derived from the **official ground-truth answer branch
  token** for each sample.  This is deliberate ground-truth answer leakage: it
  can only ever be a *falsifiable upper bound*, never a deployable selector.
* The pilot uses 24 samples.  It is a leakage-oracle screening, **not** a paper
  accuracy result, **not** a CV-Bench accuracy claim and **not** an FPGA /
  latency / acceleration claim.
* The hardware-rule layout produced by injecting the oracle score through the
  existing ``select_hardware_regular_groups`` contract must be identical to the
  frozen corrected integer L2 layout (``G>96`` -> ``output=96``, ``real=87``,
  ``context=9``, same 3x3 anchors / context pooling / original positions).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.hardware_regular_b1_corrected_l2 import array_sha256
from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    DEFAULT_LOCAL_CANDIDATE_CAP,
    FEATURE_DIM,
    MAX_CONTEXT_SLOTS,
    MERGE_UNIT,
    REGION_GRID,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    HardwareSelection,
    SelectorContractError,
    select_hardware_regular_groups,
)

__all__ = [
    "ARM_ORACLE",
    "ALLOWED_ARMS",
    "BUDGET",
    "CUTPOINT",
    "DESIGN",
    "EXPECTED_CONTEXT",
    "EXPECTED_OUTPUT",
    "EXPECTED_REAL",
    "GATE_ORACLE_PARSE_MIN",
    "GATE_ORACLE_MINUS_CONTROL_NET_MIN",
    "GATE_R_TO_W_MAX",
    "GATE_W_TO_R_MIN",
    "LABEL_AWARE_ORACLE",
    "LOCAL_CANDIDATE_CAP",
    "MAX_NEW_TOKENS",
    "PILOT_PER_SOURCE",
    "PILOT_SAMPLE_COUNT",
    "PRUNE_AFTER_LAYER",
    "REGION_GRID",
    "REGION_GRID_FIXED",
    "SALIENCY_DEFINITION",
    "SOURCE_DATASETS",
    "SOURCE_GROUP_OF",
    "STATUS_GO",
    "STATUS_PREREGISTERED_NOT_EXECUTED",
    "STATUS_STOP",
    "OraclePilotError",
    "array_sha256",
    "assert_layout_contract",
    "build_oracle_scorer",
    "file_sha256",
    "group_saliency_from_grad_activation",
    "inject_oracle_selection",
    "layout_contract",
    "record_sha256",
    "retained_mass",
    "saliency_summary",
    "selection_jaccard",
    "selection_overlap",
    "stable_sha256_rank",
]

# ---------------------------------------------------------------------------
# Frozen pilot identity
# ---------------------------------------------------------------------------
DESIGN = "2026-10-05-b1-answer-margin-oracle-pilot24"
ARM_ORACLE = "oracle_answer_margin_b1_saliency"
ALLOWED_ARMS = (ARM_ORACLE,)

BUDGET = 96
REGION_GRID_FIXED = REGION_GRID
LOCAL_CANDIDATE_CAP = DEFAULT_LOCAL_CANDIDATE_CAP
PRUNE_AFTER_LAYER = 0
CUTPOINT = "B1_after_vision_block_0"
MAX_NEW_TOKENS = 16

EXPECTED_OUTPUT = BUDGET
EXPECTED_REAL = 87
EXPECTED_CONTEXT = MAX_CONTEXT_SLOTS  # 9 reserved context slots for G > K

PILOT_SAMPLE_COUNT = 24
PILOT_PER_SOURCE = 8

#: The three CV-Bench ``source_dataset`` groups fixed at pre-registration.
#: They cover the official ``source`` triple ``ADE20K`` / ``COCO`` / ``Omni3D``
#: required by :func:`evaluation.build_cv_bench_pruning_report.score`.
SOURCE_DATASETS = (
    "ADE20K Validation Set",
    "COCO 2017 Validation Set",
    "Omni3D_Hypersim",
)
SOURCE_GROUP_OF = {
    "ADE20K Validation Set": "ADE20K",
    "COCO 2017 Validation Set": "COCO",
    "Omni3D_Hypersim": "Omni3D",
}

SALIENCY_DEFINITION = (
    "per merged group g: sum over its 4 patches j and feature dim d of "
    "abs(grad[g*4+j, d] * activation[g*4+j, d]), from the full (uncut) B1 "
    "visual sequence and the official correct-answer branch token"
)
LABEL_AWARE_ORACLE = (
    "label-aware oracle upper bound; uses the official ground-truth answer "
    "branch token to compute the saliency score. Ground-truth answer leakage. "
    "Not deployable. Not paper accuracy."
)

STATUS_PREREGISTERED_NOT_EXECUTED = "PRE-REGISTERED_NOT_EXECUTED"
STATUS_GO = "PILOT_GO"
STATUS_STOP = "PILOT_STOP"

#: Pre-registered pilot GO gate (24-sample leakage-oracle screening only).
GATE_ORACLE_MINUS_CONTROL_NET_MIN = 3
GATE_W_TO_R_MIN = 4
GATE_R_TO_W_MAX = 1
GATE_ORACLE_PARSE_MIN = 23


class OraclePilotError(ValueError):
    """Raised when a pure oracle-pilot helper violates its contract."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def file_sha256(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself."""

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def stable_sha256_rank(sample_id: Any) -> str:
    """Stable selection rank: ``sha256`` of the manifest ``sample_id`` string."""

    return hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Oracle saliency
# ---------------------------------------------------------------------------
def group_saliency_from_grad_activation(
    b1_hidden: Any,
    b1_grad: Any,
    merge_unit: int = MERGE_UNIT,
) -> list[float]:
    """Return the per-group ``|grad * activation|`` oracle saliency.

    ``b1_hidden`` / ``b1_grad`` are ``[G * merge_unit, D]`` tensors from the
    full (retention ``1.0``, ``prune_after_layer=0``) B1 visual sequence.  The
    returned list has exactly ``G`` finite, non-negative entries.
    """

    import torch  # noqa: PLC0415 - lazy so CPU-only importers stay light

    if b1_hidden is None or b1_grad is None:
        raise OraclePilotError("b1_hidden and b1_grad must both be present")
    if not torch.is_tensor(b1_hidden) or not torch.is_tensor(b1_grad):
        raise OraclePilotError("b1_hidden and b1_grad must be torch tensors")
    if b1_hidden.ndim != 2 or b1_grad.ndim != 2:
        raise OraclePilotError("b1_hidden and b1_grad must be [P, D]")
    if tuple(b1_hidden.shape) != tuple(b1_grad.shape):
        raise OraclePilotError(
            f"b1_hidden {tuple(b1_hidden.shape)} and b1_grad "
            f"{tuple(b1_grad.shape)} shapes differ"
        )
    if isinstance(merge_unit, bool) or not isinstance(merge_unit, int) or merge_unit <= 0:
        raise OraclePilotError("merge_unit must be a positive integer")
    patch_tokens = int(b1_hidden.shape[0])
    if patch_tokens == 0 or patch_tokens % merge_unit != 0:
        raise OraclePilotError(
            f"patch tokens {patch_tokens} must be a positive multiple of "
            f"merge_unit {merge_unit}"
        )
    group_count = patch_tokens // merge_unit
    activation = b1_hidden.detach().to(torch.float32)
    gradient = b1_grad.to(torch.float32)
    product = (gradient * activation).reshape(group_count, merge_unit, -1)
    per_group = product.abs().sum(dim=(1, 2))
    if not bool(torch.isfinite(per_group).all().item()):
        raise OraclePilotError("oracle saliency contains non-finite values")
    if bool((per_group < 0).any().item()):
        raise OraclePilotError("oracle saliency must be non-negative")
    return [float(value) for value in per_group.detach().cpu().tolist()]


def saliency_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise OraclePilotError("saliency vector must not be empty")
    if not np.isfinite(array).all():
        raise OraclePilotError("saliency vector must be finite")
    if np.any(array < 0):
        raise OraclePilotError("saliency vector must be non-negative")
    return {
        "group_count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "sum": float(array.sum()),
        "all_finite": bool(np.isfinite(array).all()),
        "all_nonzero": bool((array > 0).all()),
    }


# ---------------------------------------------------------------------------
# Oracle score injection (existing hardware-regular contract)
# ---------------------------------------------------------------------------
def build_oracle_scorer(saliency: Sequence[float]) -> Any:
    """Build the ``integer_scorer`` callback that injects oracle saliency.

    The callback reuses the *existing* ``integer_scorer`` hook of
    :func:`evaluation.hardware_regular_b1_integration.forward_hardware_regular_b1_vision`:
    it returns the oracle saliency as ``combined_scores`` so the frozen
    :func:`select_hardware_regular_groups` context/anchor contract runs
    unchanged.  No integer maths is performed and no score is regenerated.
    """

    array = np.asarray(saliency, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise OraclePilotError("oracle saliency must not be empty")
    if not np.isfinite(array).all():
        raise OraclePilotError("oracle saliency must be finite")
    if np.any(array < 0):
        raise OraclePilotError("oracle saliency must be non-negative")
    frozen = array.copy()

    def scorer(
        hidden_np: Any,
        grid_h: Any,
        grid_w: Any,
        budget: Any,
        text_embeddings: Any,
    ) -> dict[str, Any]:
        hidden_array = np.asarray(hidden_np)
        if hidden_array.ndim != 3 or int(hidden_array.shape[0]) != frozen.size:
            raise OraclePilotError(
                f"oracle saliency has {frozen.size} groups but hidden has "
                f"shape {hidden_array.shape}"
            )
        return {
            "combined_scores": frozen.copy(),
            "score_source": "oracle_b1_grad_x_activation",
            "is_label_aware_oracle": True,
            "is_deployable_selector": False,
        }

    return scorer


def inject_oracle_selection(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    saliency: Sequence[float],
    *,
    budget: int = BUDGET,
    region_grid: int = REGION_GRID_FIXED,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
) -> HardwareSelection:
    """Inject oracle saliency as the score into the frozen selection contract."""

    array = np.asarray(saliency, dtype=np.float64).reshape(-1)
    return select_hardware_regular_groups(
        hidden,
        grid_h,
        grid_w,
        budget=budget,
        combined_scores=array,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        boundary=BOUNDARY_MAIN,
    )


# ---------------------------------------------------------------------------
# Layout contract (identical to the frozen corrected integer L2 layout)
# ---------------------------------------------------------------------------
def layout_contract(
    selection: HardwareSelection,
    *,
    full_groups: int,
    budget: int = BUDGET,
) -> dict[str, Any]:
    """Return the per-field checks of the frozen hardware-rule layout."""

    if not isinstance(selection, HardwareSelection):
        raise OraclePilotError("selection must be a HardwareSelection")
    complete = int(selection.bypass)
    groups = np.asarray(selection.group_indices, dtype=np.int64).reshape(-1)
    patch = np.asarray(selection.patch_indices, dtype=np.int64)
    source = np.asarray(selection.source_type, dtype=np.int64).reshape(-1)
    region = np.asarray(selection.region_id, dtype=np.int64).reshape(-1)
    valid = np.asarray(selection.valid, dtype=bool).reshape(-1)
    output_count = int(groups.size)
    real_count = int((source == SOURCE_REAL).sum())
    context_count = int((source == SOURCE_CONTEXT).sum())

    ascending = bool(
        output_count <= 1 or np.all(np.diff(groups) > 0)
    )
    in_range = bool(
        output_count == 0
        or (int(groups.min()) >= 0 and int(groups.max()) < int(full_groups))
    )
    expected_patch = (
        groups[:, None] * MERGE_UNIT + np.arange(MERGE_UNIT, dtype=np.int64)[None, :]
        if output_count
        else np.zeros((0, MERGE_UNIT), dtype=np.int64)
    )
    patch_ok = bool(patch.shape == expected_patch.shape) and bool(
        np.array_equal(patch, expected_patch)
    )

    checks: list[dict[str, Any]] = [
        {
            "name": "grid_matches_full_groups",
            "passed": int(selection.grid_h) * int(selection.grid_w) == int(full_groups),
            "detail": f"grid {selection.grid_h}x{selection.grid_w} vs G={full_groups}",
        },
        {
            "name": "output_count_is_expected",
            "passed": output_count
            == (int(full_groups) if complete else int(budget)),
            "detail": f"output={output_count} budget={budget} G={full_groups}",
        },
        {
            "name": "source_type_counts_are_expected",
            "passed": (
                (real_count == int(full_groups) and context_count == 0)
                if complete
                else (
                    real_count == EXPECTED_REAL
                    and context_count == EXPECTED_CONTEXT
                )
            ),
            "detail": (
                f"real={real_count} context={context_count} "
                f"(expected {EXPECTED_REAL}/{EXPECTED_CONTEXT} when G>K, "
                f"{full_groups}/0 when G<=K)"
            ),
        },
        {
            "name": "bypass_matches_G_le_K",
            "passed": bool(selection.bypass) == (int(full_groups) <= int(budget)),
            "detail": f"bypass={bool(selection.bypass)} G={full_groups} K={budget}",
        },
        {
            "name": "group_indices_strictly_ascending_and_in_range",
            "passed": ascending and in_range,
            "detail": f"{output_count} groups, ascending={ascending}, in_range={in_range}",
        },
        {
            "name": "patch_indices_are_original_positions",
            "passed": patch_ok,
            "detail": "patch rows equal group*4 + [0,1,2,3] (original positions)",
        },
        {
            "name": "all_valid",
            "passed": bool(valid.size == output_count and valid.all()),
            "detail": f"{int(valid.sum())}/{output_count} valid",
        },
        {
            "name": "source_type_values_known",
            "passed": bool(source.size == output_count and np.isin(source, (SOURCE_REAL, SOURCE_CONTEXT)).all()),
            "detail": f"source_type values {sorted(set(int(v) for v in source.tolist()))}",
        },
        {
            "name": "region_ids_in_range",
            "passed": bool(
                region.size == output_count
                and region.size > 0
                and int(region.min()) >= 0
                and int(region.max()) < MAX_CONTEXT_SLOTS
            ),
            "detail": f"region_id range {int(region.min()) if region.size else 'n/a'}..{int(region.max()) if region.size else 'n/a'}",
        },
    ]
    passed = all(bool(item["passed"]) for item in checks)
    return {
        "passed": passed,
        "full_groups": int(full_groups),
        "budget": int(budget),
        "output_group_count": output_count,
        "real_group_count": real_count,
        "context_group_count": context_count,
        "identity_bypass": bool(selection.bypass),
        "checks": checks,
    }


def assert_layout_contract(
    selection: HardwareSelection,
    *,
    full_groups: int,
    budget: int = BUDGET,
) -> dict[str, Any]:
    contract = layout_contract(selection, full_groups=full_groups, budget=budget)
    if not contract["passed"]:
        failures = [
            item["name"] for item in contract["checks"] if not item["passed"]
        ]
        raise OraclePilotError(
            f"oracle selection violates the frozen hardware layout contract: {failures}"
        )
    return contract


# ---------------------------------------------------------------------------
# Diagnostic-only selection comparisons
# ---------------------------------------------------------------------------
def retained_mass(
    saliency: Sequence[float], selected_group_indices: Sequence[int]
) -> float:
    """Diagnostic-only fraction of total saliency mass kept by the selection."""

    array = np.asarray(saliency, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise OraclePilotError("saliency vector must not be empty")
    total = float(array.sum())
    if total <= 0.0:
        return 0.0
    indices = np.asarray(list(selected_group_indices), dtype=np.int64).reshape(-1)
    if indices.size == 0:
        return 0.0
    if int(indices.min()) < 0 or int(indices.max()) >= array.size:
        raise OraclePilotError("selected group index outside the saliency vector")
    return float(array[indices].sum() / total)


def selection_overlap(
    left: Sequence[int], right: Sequence[int]
) -> dict[str, Any]:
    """Diagnostic-only overlap of two selected-group index sets."""

    left_set = {int(value) for value in left}
    right_set = {int(value) for value in right}
    union = left_set | right_set
    intersection = left_set & right_set
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": (
            float(len(intersection) / len(union)) if union else 0.0
        ),
        "exact_match": left_set == right_set,
    }


def selection_jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    return float(selection_overlap(left, right)["jaccard"])
