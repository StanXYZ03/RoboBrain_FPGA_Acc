#!/usr/bin/env python3
"""Parameter-free spatial supervision target feasibility audit (TRAIN only).

This module is a **zero-update, train-only, CPU-only** audit.  It never trains,
never tunes, never builds an optimizer, never creates a new ``Wt``/``Wv``, never
loads a 4B model, never uses a GPU, never touches the network and never reads a
val/test feature cache.  It answers one bounded question:

    Can a parameter-free, task-specific spatial supervision target
    (``grounding_area_target`` from the bbox, ``pointing_bilinear_target`` from
    the point) provide a same-image grounding/pointing separation signal that is
    stronger than the existing full-vision max-cosine semantic teacher target,
    while remaining compatible with the existing fixed-budget / anchor selector?

The audit fixes its contract in two preregistration files before any metric is
computed:

* ``configs/spatial_teacher_feasibility.preregistered.json``
* ``reports/token_selector/2026-10-03-spatial-teacher-feasibility-preregistration.json``

Isolation contract (enforced and unit-tested):

* the train/val/test split is computed **only from index metadata**
  (``sample_id`` + ``source_image_sha256``) using the frozen
  :func:`evaluation.prompt_projection_distillation.split_sample_ids`;
* only feature caches whose ``sample_id`` belongs to the **train** split are
  opened (``load_feature_cache`` is never called for a val/test path);
* the input aggregate SHA256 is taken over the opened train npz files only, so
  rewriting or corrupting a val/test npz cannot move any train metric or hash.

Every split / metric / divergence / bootstrap primitive is imported from the
frozen layer.  The only new math is the two spatial targets and their geometry
diagnostics, documented above and in the preregistration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.metrics import load_jsonl
from evaluation.prompt_identifiability_diagnostic import paired_effect
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    config_sha256,
    jaccard,
    l2_baseline_scores,
    load_distillation_config,
    load_feature_cache,
    ndcg_at_k,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    spearman,
    split_group_summary,
    split_sample_ids,
    teacher_topk_recall,
    topk_indices,
)
from evaluation.teacher_pair_signal_audit import (
    _file_sha256,
    _resolve_npz,
    aggregate_npz_sha256,
    describe,
    js_divergence_nats,
    load_index_metadata,
    symmetric_kl_nats,
    total_variation_distance,
)
from evaluation.train_prompt_projection import index_file_sha256

__all__ = [
    "compute_report",
    "grounding_area_target",
    "join_index_manifest",
    "load_train_samples_extended",
    "pointing_bilinear_target",
    "render_markdown",
    "run_audit",
    "target_diagnostics",
    "verify_preregistration",
]

# ---------------------------------------------------------------------------
# Frozen audit constants (fixed before this audit runs)
# ---------------------------------------------------------------------------
REPORT_VERSION = "1.0"
REPORT_STEM = "2026-10-03-spatial-teacher-feasibility"
PREREG_CONFIG_REL = "configs/spatial_teacher_feasibility.preregistered.json"
PREREG_REPORT_REL = (
    "reports/token_selector/2026-10-03-spatial-teacher-feasibility-preregistration.json"
)
KS: tuple[int, ...] = (48, 64, 96)
AUDIT_SEED = 20261003
BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
CI_LEVEL = 0.95
EXPECTED_TRAIN_RECORDS = 318
EXPECTED_TRAIN_IMAGES = 159
_POINT_TOL = 1e-9
_EPS = 1e-12

DECISION_PRESENT = "SPATIAL_SUPERVISION_SIGNAL_PRESENT"
DECISION_NOT_ESTABLISHED = "SPATIAL_SUPERVISION_SIGNAL_NOT_ESTABLISHED"
DECISION_BLOCKED = "BLOCKED_DATA_CONTRACT"

PAIR_METRIC_KEYS: tuple[str, ...] = (
    "js_nats",
    "symmetric_kl",
    "tv",
    "argmax_different",
    "k48_effective",
    "k48_jaccard",
    "k48_overlap",
    "k48_disagreement",
    "k64_effective",
    "k64_jaccard",
    "k64_overlap",
    "k64_disagreement",
    "k96_effective",
    "k96_jaccard",
    "k96_overlap",
    "k96_disagreement",
)

DIFF_KEYS: tuple[str, ...] = (
    "diff_js_nats",
    "diff_tv",
    "diff_k48_disagreement",
    "diff_k64_disagreement",
    "diff_k96_disagreement",
)

LIMITATIONS: tuple[str, ...] = (
    "The audit does not train anything, does not build a student and does not "
    "claim the target is learnable.",
    "No downstream task accuracy (CV-Bench or otherwise) is measured or claimed.",
    "No FPGA latency, resource or speedup number is measured or claimed.",
    "The semantic comparator is the existing cached max-over-prompt-token "
    "teacher only; no new semantic model is loaded.",
    "The grounding area target and the pointing bilinear target have different "
    "structural support (an area over many cells vs at most four cells), so part "
    "of the same-image spatial separation reflects that structural difference "
    "and not only task-relevant geometry.",
    "Low spatial/semantic agreement means the objectives differ, not "
    "automatically that either target is wrong.",
    "The hardware-regular anchor baseline is a descriptive reference over the "
    "L2 saliency score in this zero-update audit; it is not a trained selector.",
    "All numbers are TRAIN-split, image-level and CPU float64 internal "
    "measurements, never a promotion gate.",
)

NOTES: dict[str, str] = {
    "zero_update": (
        "only array math and imported metric helpers are used; no optimizer, no "
        "parameter, no checkpoint, no model is created or updated"
    ),
    "train_only": (
        "the split comes from index metadata only and only train feature caches "
        "are opened; val/test npz files are never resolved or read"
    ),
    "independent_unit": (
        "the independent unit is the image (grounding/pointing are a paired "
        "same-image pair, never two independent samples)"
    ),
    "no_blending": (
        "the spatial targets are never blended with the semantic teacher target"
    ),
    "decision_rule": (
        "SPATIAL_SUPERVISION_SIGNAL_PRESENT requires all integrity checks to "
        "pass and the 95% bootstrap CI lower bound of spatial-semantic to be "
        "greater than zero for BOTH JS and TV; top-K results are descriptive "
        "only and cannot override this rule"
    ),
    "learnability_untested": (
        "even if the signal is present, learnability and downstream task "
        "accuracy remain untested"
    ),
    "no_holm": (
        "a single pre-registered family (JS and TV) is reported with raw "
        "two-sided p-values; no multiple-comparison correction is invented"
    ),
    "compatibility_definition": (
        "Spearman is between the spatial target and the cached teacher_soft_target; "
        "full NDCG uses the spatial target as the ranking score and "
        "teacher_soft_target as the graded relevance"
    ),
}

# Bounded, post-hoc implementation correction from an independent Codex audit.
# The preregistered spatial targets, JS/TV decision rule, split, inputs and
# decision label are frozen and unchanged.  Only the anchor-baseline NDCG now
# honors the anchor-selected set (see ``anchor_baseline_record``); retained mass
# already used the selected set and is bit-for-bit unchanged.
CORRECTION_LABEL = "CORRECTION_ANCHOR_NDCG_RESTRICTED_TO_ANCHOR_SELECTED_GROUPS"
CORRECTION_ISSUED = "2026-10-03"
CORRECTION_OLD_SCRIPT_SHA256 = (
    "c800d12d68fd35ae29aabe3e83be2a143e44cd4d4a8c9f7d9edc5501903f9983"
)
CORRECTION_OLD_OUTPUT_SHA256: dict[str, str] = {
    "csv": "03bd34b00abf3bf218b7a82757431f1c335902c19c477374939c99d1a66e9f94",
    "json": "85f836517e0c9a530389b6d98f126a435c126dd72c1e4b5b70afd68cdebaf154",
    "md": "f4fb194d2a711c63e8226bfe943143c1b72754acc95da55047e646fcd22c99a0",
    "harness": "a22ea2ba8ca7213f156cb6262e6235019dbf55e4f943325ce5a8f3120c3e8e30",
}
# Pre-correction anchor retained-mass means copied from the frozen prior report
# (before this correction) so the report self-checks that mass did not move.
CORRECTION_OLD_ANCHOR_MASS_MEAN: dict[str, dict[str, float]] = {
    "48": {
        "overall": 0.17101290862700277,
        "grounding": 0.1759897465133256,
        "pointing": 0.16603607074067994,
    },
    "64": {
        "overall": 0.24402416835178078,
        "grounding": 0.24370444207479483,
        "pointing": 0.2443438946287667,
    },
    "96": {
        "overall": 0.3562069644276804,
        "grounding": 0.3635564871977534,
        "pointing": 0.34885744165760735,
    },
}


class SpatialContractError(DistillationContractError):
    """Raised when a frozen data/geometry contract is violated."""


# ---------------------------------------------------------------------------
# Frozen spatial targets (new math for this audit, documented)
# ---------------------------------------------------------------------------


def _validate_grid(grid_h: Any, grid_w: Any) -> tuple[int, int]:
    if not isinstance(grid_h, (int, np.integer)) or not isinstance(
        grid_w, (int, np.integer)
    ):
        raise SpatialContractError("grid_h and grid_w must be integers")
    height = int(grid_h)
    width = int(grid_w)
    if height <= 0 or width <= 0:
        raise SpatialContractError("grid_h and grid_w must be positive")
    return height, width


def grounding_area_target(
    bbox_1000: Any, grid_h: Any, grid_w: Any
) -> np.ndarray:
    """Exact bbox/grid-cell intersection mass, normalized to sum to 1.

    Coordinates are normalized ``[0, 1000]``.  The bbox is clipped to
    ``[0, 1000]`` for numerical boundary tolerance only; an invalid or
    zero/negative-area box is a hard failure.  Cell ``(r, c)`` covers
    ``[1000*c/W, 1000*(c+1)/W] x [1000*r/H, 1000*(r+1)/H]`` and the flat group
    index is row-major ``g = r*W + c``.
    """
    height, width = _validate_grid(grid_h, grid_w)
    bbox = np.asarray(bbox_1000, dtype=np.float64).reshape(-1)
    if bbox.size != 4:
        raise SpatialContractError(f"bbox_1000 must have 4 values, got {bbox.size}")
    if not np.isfinite(bbox).all():
        raise SpatialContractError("bbox_1000 must be finite")
    x1, y1, x2, y2 = (float(value) for value in np.clip(bbox, 0.0, 1000.0))
    if x2 <= x1 or y2 <= y1:
        raise SpatialContractError(
            f"bbox_1000 has non-positive area after clip: {bbox.tolist()}"
        )
    columns = np.arange(width, dtype=np.float64)
    rows = np.arange(height, dtype=np.float64)
    col_left = 1000.0 * columns / width
    col_right = 1000.0 * (columns + 1.0) / width
    row_top = 1000.0 * rows / height
    row_bottom = 1000.0 * (rows + 1.0) / height
    col_overlap = np.clip(np.minimum(x2, col_right) - np.maximum(x1, col_left), 0.0, None)
    row_overlap = np.clip(np.minimum(y2, row_bottom) - np.maximum(y1, row_top), 0.0, None)
    area = np.outer(row_overlap, col_overlap).reshape(-1)
    total = float(area.sum())
    if total <= 0.0:
        raise SpatialContractError("bbox_1000 intersection area is zero")
    return area / total


def pointing_bilinear_target(
    point_1000: Any, grid_h: Any, grid_w: Any
) -> np.ndarray:
    """Separable bilinear point mass over the up to four neighboring centers.

    ``u = x*W/1000 - 0.5`` and ``v = y*H/1000 - 0.5`` are clamped to the valid
    center-coordinate range ``[0, W-1]`` / ``[0, H-1]``; mass is distributed
    with separable bilinear weights, duplicate boundary indices are combined and
    the result is normalized to sum to 1.  There is no Gaussian and no bandwidth
    parameter.
    """
    height, width = _validate_grid(grid_h, grid_w)
    point = np.asarray(point_1000, dtype=np.float64).reshape(-1)
    if point.size != 2:
        raise SpatialContractError(f"point_1000 must have 2 values, got {point.size}")
    if not np.isfinite(point).all():
        raise SpatialContractError("point_1000 must be finite")
    x = float(point[0])
    y = float(point[1])
    if not (-_POINT_TOL <= x <= 1000.0 + _POINT_TOL) or not (
        -_POINT_TOL <= y <= 1000.0 + _POINT_TOL
    ):
        raise SpatialContractError(f"point_1000 outside [0,1000]: {point.tolist()}")
    u = min(max(x * width / 1000.0 - 0.5, 0.0), float(width - 1))
    v = min(max(y * height / 1000.0 - 0.5, 0.0), float(height - 1))
    col0 = int(math.floor(u))
    row0 = int(math.floor(v))
    col1 = min(col0 + 1, width - 1)
    row1 = min(row0 + 1, height - 1)
    fu = u - col0
    fv = v - row0
    target = np.zeros(height * width, dtype=np.float64)
    for row, col, weight in (
        (row0, col0, (1.0 - fu) * (1.0 - fv)),
        (row0, col1, fu * (1.0 - fv)),
        (row1, col0, (1.0 - fu) * fv),
        (row1, col1, fu * fv),
    ):
        target[row * width + col] += weight
    total = float(target.sum())
    if total <= _EPS:
        raise SpatialContractError("pointing bilinear target has zero mass")
    return target / total


def point_inside_bbox(point_1000: Any, bbox_1000: Any, tol: float = 1e-9) -> bool:
    """Whether a normalized point lies inside a normalized bbox (inclusive)."""
    point = np.asarray(point_1000, dtype=np.float64).reshape(-1)
    bbox = np.asarray(bbox_1000, dtype=np.float64).reshape(-1)
    if point.size != 2 or bbox.size != 4:
        raise SpatialContractError("point_inside_bbox requires a 2-D point and a 4-D bbox")
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return bool(
        (x1 - tol) <= x <= (x2 + tol) and (y1 - tol) <= y <= (y2 + tol)
    )


def target_diagnostics(target: Any) -> dict[str, Any]:
    """Finite / non-negative / sum-to-one errors and support-mass descriptors."""
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise SpatialContractError("target must not be empty")
    finite = bool(np.isfinite(values).all())
    non_negative = bool((values >= -1e-12).all())
    sum_error = abs(float(values.sum()) - 1.0)
    clipped = np.clip(values, 0.0, None)
    total = float(clipped.sum())
    if total <= _EPS:
        raise SpatialContractError("target has no positive mass")
    probabilities = clipped / total
    support = int(np.count_nonzero(probabilities > 0.0))
    positive = probabilities[probabilities > 0.0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "finite": finite,
        "non_negative": non_negative,
        "sum_error": sum_error,
        "sum_to_one_within_1e-9": bool(sum_error <= 1e-9),
        "support_size": support,
        "entropy_nats": entropy,
        "effective_support": float(math.exp(entropy)),
        "max_mass": float(probabilities.max()),
    }


# ---------------------------------------------------------------------------
# Metadata-only index load + manifest join + train-only feature cache load
# ---------------------------------------------------------------------------


def load_clean_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Read the clean manifest, hard-failing on duplicate sample ids."""
    records = load_jsonl(path)
    if not records:
        raise SpatialContractError(f"manifest {path} contains no records")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        if sample_id in by_id:
            raise SpatialContractError(f"duplicate manifest sample_id {sample_id}")
        by_id[sample_id] = record
    return by_id, [str(record["sample_id"]) for record in records]


def join_index_manifest(
    index_metadata: Sequence[Mapping[str, Any]],
    manifest_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, list[str]], dict[str, Any]]:
    """Join index and manifest by ``sample_id`` with hard identity checks."""
    index_by_id: dict[str, Mapping[str, Any]] = {}
    for record in index_metadata:
        sample_id = str(record["sample_id"])
        if sample_id in index_by_id:
            raise SpatialContractError(f"duplicate index sample_id {sample_id}")
        index_by_id[sample_id] = record
    index_ids = set(index_by_id)
    manifest_ids = set(manifest_by_id)

    missing_from_manifest = sorted(index_ids - manifest_ids)
    missing_from_index = sorted(manifest_ids - index_ids)
    task_mismatches: list[str] = []
    for sample_id in sorted(index_ids & manifest_ids):
        if str(index_by_id[sample_id]["task"]) != str(manifest_by_id[sample_id]["task"]):
            task_mismatches.append(sample_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(index_ids & manifest_ids):
        groups[index_by_id[sample_id]["source_image_sha256"]].append(sample_id)
    image_identity_mismatches: list[str] = []
    for image_hash, sample_ids in groups.items():
        coco_ids = {manifest_by_id[sample_id].get("coco_image_id") for sample_id in sample_ids}
        image_paths = {manifest_by_id[sample_id].get("image") for sample_id in sample_ids}
        if len(coco_ids) != 1 or len(image_paths) != 1:
            image_identity_mismatches.append(image_hash)

    diagnostics = {
        "index_sample_count": len(index_ids),
        "manifest_sample_count": len(manifest_ids),
        "joined_sample_count": len(index_ids & manifest_ids),
        "missing_from_manifest": missing_from_manifest,
        "missing_from_index": missing_from_index,
        "task_mismatches": task_mismatches,
        "image_identity_mismatches": image_identity_mismatches,
        "image_group_count": len(groups),
    }
    if (
        missing_from_manifest
        or missing_from_index
        or task_mismatches
        or image_identity_mismatches
    ):
        raise SpatialContractError(
            "join contract failed: "
            f"missing_from_manifest={missing_from_manifest[:5]} "
            f"missing_from_index={missing_from_index[:5]} "
            f"task_mismatches={task_mismatches[:5]} "
            f"image_identity_mismatches={image_identity_mismatches[:5]}"
        )
    return index_by_id, dict(groups), diagnostics


def load_train_samples_extended(
    index_path: Path,
    features_dir: Path | None,
    train_ids: Sequence[str],
    *,
    opened: set[str] | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Load **only** train feature caches, retaining ``group_mean``.

    This is a superset of :func:`evaluation.teacher_pair_signal_audit.load_train_samples`
    that additionally keeps ``group_mean`` for the hardware-regular L2 baseline.
    It reuses the frozen ``load_index_metadata``, ``_resolve_npz``,
    ``_file_sha256``, ``aggregate_npz_sha256`` and ``load_feature_cache``
    primitives; it never opens a non-train cache.
    """
    loader = cache_loader or load_feature_cache
    directory = Path(features_dir) if features_dir else Path(index_path).parent
    by_id = {record["sample_id"]: record for record in load_index_metadata(index_path)}
    missing = [sample_id for sample_id in train_ids if sample_id not in by_id]
    if missing:
        raise SpatialContractError(f"train ids absent from index: {missing}")
    samples: list[dict[str, Any]] = []
    hashes: list[tuple[str, str]] = []
    resolved_paths: list[str] = []
    for sample_id in sorted(str(value) for value in train_ids):
        record = by_id[sample_id]
        npz_path = _resolve_npz(record, directory)
        if not npz_path.is_file():
            raise FileNotFoundError(f"train feature cache does not exist: {npz_path}")
        file_sha = _file_sha256(npz_path)
        if opened is not None:
            opened.add(str(npz_path))
        cache = loader(npz_path)
        hashes.append((sample_id, file_sha))
        resolved_paths.append(str(npz_path))
        samples.append(
            {
                "sample_id": sample_id,
                "task": str(record["task"]),
                "source_image_sha256": str(record["source_image_sha256"]),
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "group_count": int(record["group_count"]),
                "prompt_token_count": int(record["prompt_token_count"]),
                "npz": str(npz_path),
                "npz_sha256": file_sha,
                "group_mean": np.asarray(cache["group_mean"], dtype=np.float64),
                "teacher_raw": np.asarray(cache["teacher_scores"], dtype=np.float64),
                "teacher_normalized": np.asarray(
                    cache["teacher_scores_normalized"], dtype=np.float64
                ),
                "teacher_soft": np.asarray(
                    cache["teacher_soft_target"], dtype=np.float64
                ),
            }
        )
    return samples, aggregate_npz_sha256(hashes), resolved_paths


# ---------------------------------------------------------------------------
# Pair separation + compatibility + baselines
# ---------------------------------------------------------------------------


def pair_separation_metrics(
    target_a: Any, target_b: Any, *, ks: Sequence[int] = KS
) -> dict[str, float]:
    """Same-image JS / symmetric KL / TV / argmax / top-K separation."""
    first = np.asarray(target_a, dtype=np.float64).reshape(-1)
    second = np.asarray(target_b, dtype=np.float64).reshape(-1)
    if first.shape != second.shape:
        raise SpatialContractError("pair_separation_metrics inputs must share a shape")
    group_count = first.size
    row: dict[str, float] = {
        "js_nats": js_divergence_nats(first, second),
        "symmetric_kl": symmetric_kl_nats(first, second),
        "tv": total_variation_distance(first, second),
        "argmax_different": float(int(np.argmax(first)) != int(np.argmax(second))),
    }
    for k in ks:
        effective = min(int(k), group_count)
        first_top = topk_indices(first, effective)
        second_top = topk_indices(second, effective)
        overlap = jaccard(first_top, second_top)
        row[f"k{int(k)}_effective"] = float(effective)
        row[f"k{int(k)}_jaccard"] = float(overlap)
        row[f"k{int(k)}_overlap"] = float(teacher_topk_recall(first_top, second_top))
        row[f"k{int(k)}_disagreement"] = float(1.0 - overlap)
    return row


def semantic_spatial_compatibility(
    spatial_target: Any, teacher_soft: Any
) -> dict[str, float]:
    """Spearman and full NDCG between a spatial target and the semantic teacher."""
    spatial = np.asarray(spatial_target, dtype=np.float64).reshape(-1)
    teacher = np.asarray(teacher_soft, dtype=np.float64).reshape(-1)
    if spatial.shape != teacher.shape:
        raise SpatialContractError("compatibility inputs must share a shape")
    return {
        "spearman": spearman(spatial, teacher),
        "ndcg_full": ndcg_at_k(spatial, teacher, spatial.size),
    }


def l2_baseline_record(
    group_mean: Any, spatial_target: Any, *, ks: Sequence[int] = KS
) -> dict[str, dict[str, float]]:
    """L2-saliency ranking NDCG and retained spatial-target mass at each K."""
    scores = l2_baseline_scores(group_mean)
    target = np.asarray(spatial_target, dtype=np.float64).reshape(-1)
    if scores.shape[0] != target.size:
        raise SpatialContractError("group_mean rows and spatial target length differ")
    per_budget: dict[str, dict[str, float]] = {}
    for k in ks:
        effective = min(int(k), int(scores.shape[0]))
        selected = topk_indices(scores, effective)
        per_budget[str(int(k))] = {
            "k_eff": float(effective),
            "ndcg": float(ndcg_at_k(scores, target, effective)),
            "mass": float(saliency_mass_retained(selected, target)),
        }
    return per_budget


def anchor_baseline_record(
    group_mean: Any,
    spatial_target: Any,
    grid_h: int,
    grid_w: int,
    *,
    ks: Sequence[int] = KS,
) -> dict[str, Any]:
    """Exact reuse of the hardware-regular 3x3 anchor rule on L2 saliency.

    The existing selector ranks a ``combined`` score; this zero-update audit has
    no trained semantic projection, so the L2 saliency is fed as the ranking
    score.  If the selector module cannot be imported the baseline is reported
    as unavailable (``NOT_COMPUTED``) rather than re-implemented here.

    NDCG is restricted to the anchor-selected groups (ordered by descending L2
    score with the deterministic low-index tie policy) and divided by the
    all-group ideal DCG@K, so no unselected group can enter the predicted top K.
    Retained target mass already used the selected set and is unchanged by this
    restriction.
    """
    try:
        from evaluation.hardware_regular_token_selector import (
            InsufficientBudgetError,
            select_real_groups,
        )
    except Exception as exc:  # pragma: no cover - defensive import guard
        return {"available": False, "reason": f"import_error:{exc}", "per_budget": {}}

    scores = l2_baseline_scores(group_mean)
    target = np.asarray(spatial_target, dtype=np.float64).reshape(-1)
    group_count = int(scores.shape[0])
    per_budget: dict[str, dict[str, Any]] = {}
    for k in ks:
        effective = min(int(k), group_count)
        try:
            if effective >= group_count:
                selected = np.arange(group_count, dtype=np.int64)
                kind = "identity_bypass"
            else:
                selection = select_real_groups(
                    scores, int(grid_h), int(grid_w), int(effective)
                )
                selected = np.asarray(selection.selected, dtype=np.int64)
                kind = "select_real_groups"
        except InsufficientBudgetError as exc:
            per_budget[str(int(k))] = {
                "k_eff": float(effective),
                "available": False,
                "reason": f"insufficient_budget:{exc}",
            }
            continue
        per_budget[str(int(k))] = {
            "k_eff": float(effective),
            "available": True,
            "kind": kind,
            "ndcg": float(ndcg_for_selected_indices(selected, scores, target, effective)),
            "mass": float(saliency_mass_retained(selected, target)),
        }
    return {"available": True, "reason": None, "per_budget": per_budget}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def paired_stats(differences: Sequence[float], *, seed: int = AUDIT_SEED) -> dict[str, Any]:
    """Paired 2000-bootstrap / 10000-sign-flip statistics plus the median."""
    values = [float(value) for value in differences]
    result = paired_effect(
        values,
        seed=int(seed),
        n_bootstrap=BOOTSTRAP_RESAMPLES,
        n_permutation=PERMUTATION_RESAMPLES,
    )
    array = np.asarray(values, dtype=np.float64)
    result["median"] = float(np.median(array)) if array.size else None
    result["ci_rule_lower_gt_zero"] = bool(
        result["ci_low"] is not None and result["ci_low"] > 0.0
    )
    return result


def _aggregate_pair_metrics(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    fractions: dict[str, float] = {}
    for key in PAIR_METRIC_KEYS:
        column = [float(row[f"{prefix}{key}"]) for row in rows]
        stats[key] = describe(column)
    if rows:
        fractions["js_positive_fraction"] = sum(
            1 for row in rows if float(row[f"{prefix}js_nats"]) > 0.0
        ) / float(len(rows))
        fractions["tv_positive_fraction"] = sum(
            1 for row in rows if float(row[f"{prefix}tv"]) > 0.0
        ) / float(len(rows))
        fractions["argmax_different_fraction"] = sum(
            1 for row in rows if float(row[f"{prefix}argmax_different"]) > 0.0
        ) / float(len(rows))
    else:
        fractions = {
            "js_positive_fraction": 0.0,
            "tv_positive_fraction": 0.0,
            "argmax_different_fraction": 0.0,
        }
    return {"image_count": len(rows), "metric_stats": stats, "fractions": fractions}


def _describe_values(values: Sequence[float]) -> dict[str, Any]:
    return describe([float(value) for value in values])


def _aggregate_record_metric(
    records: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    return _describe_values(values)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def _training_config_split(config: Mapping[str, Any]) -> tuple[int, Mapping[str, float], str]:
    split = config["split"]
    return int(split["seed"]), split["ratios"], str(split["salt"])


def compute_report(
    *,
    config_path: Path,
    index_path: Path,
    manifest_path: Path,
    features_dir: Path | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    ks: Sequence[int] = KS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the full train-only audit report and per-image CSV rows.

    Raises only on an unexpected programming error; a frozen data/geometry
    contract failure returns a ``BLOCKED_DATA_CONTRACT`` report with no rows.
    """
    config = load_distillation_config(config_path)
    split_seed, ratios, salt = _training_config_split(config)
    state: dict[str, Any] = {
        "config": config,
        "split_seed": split_seed,
        "ratios": ratios,
        "salt": salt,
    }
    try:
        metadata = load_index_metadata(index_path)
        manifest_by_id, _manifest_ids = load_clean_manifest(manifest_path)
        index_by_id, index_groups, join_diagnostics = join_index_manifest(
            metadata, manifest_by_id
        )
        split = split_sample_ids(metadata, seed=split_seed, ratios=ratios, salt=salt)
        image_hashes = {
            record["sample_id"]: record["source_image_sha256"] for record in metadata
        }
        group_summary = split_group_summary(split, image_hashes)
        state.update(
            {
                "metadata": metadata,
                "manifest_by_id": manifest_by_id,
                "index_by_id": index_by_id,
                "index_groups": index_groups,
                "join_diagnostics": join_diagnostics,
                "split": split,
                "group_summary": group_summary,
            }
        )

        train_ids = split["train"]
        # Task-pair completeness + grid contract on the TRAIN split only.
        train_hashes = [image_hashes[sample_id] for sample_id in train_ids]
        by_train_image: dict[str, list[str]] = defaultdict(list)
        for sample_id in train_ids:
            by_train_image[image_hashes[sample_id]].append(sample_id)
        pair_failures: list[str] = []
        for image_hash, sample_ids in by_train_image.items():
            tasks = sorted(str(index_by_id[sample_id]["task"]) for sample_id in sample_ids)
            if tasks != ["grounding", "pointing"]:
                pair_failures.append(f"{image_hash}: {tasks}")
        grid_mismatches = [
            sample_id
            for sample_id in train_ids
            if int(index_by_id[sample_id]["group_count"])
            != int(index_by_id[sample_id]["grid_h"]) * int(index_by_id[sample_id]["grid_w"])
        ]
        state["pair_failures"] = pair_failures
        state["grid_mismatches"] = grid_mismatches
        if pair_failures:
            raise SpatialContractError(
                f"task pair completeness failed for {len(pair_failures)} train images: "
                f"{pair_failures[:5]}"
            )
        if grid_mismatches:
            raise SpatialContractError(
                f"group_count != grid_h*grid_w for {len(grid_mismatches)} train samples: "
                f"{grid_mismatches[:5]}"
            )

        opened: set[str] = set()
        train_samples, input_aggregate_sha, resolved_paths = load_train_samples_extended(
            index_path,
            features_dir,
            train_ids,
            opened=opened,
            cache_loader=cache_loader,
        )
        directory = Path(features_dir) if features_dir else Path(index_path).parent
        val_test_ids = set(split["val"]) | set(split["test"])
        val_test_paths = {
            str(_resolve_npz(index_by_id[sample_id], directory))
            for sample_id in val_test_ids
        }
        isolation = {
            "train_only": True,
            "split_from_index_metadata_only": True,
            "val_test_npz_opened": sorted(opened & val_test_paths),
            "opened_npz_count": len(opened),
            "train_npz_count": len(resolved_paths),
            "opened_only_train_npz": not (opened & val_test_paths),
            "val_test_samples_ignored": len(val_test_ids),
        }
        state["isolation"] = isolation
        state["input_aggregate_sha"] = input_aggregate_sha
        state["resolved_paths"] = resolved_paths
        if isolation["val_test_npz_opened"]:
            raise SpatialContractError(
                f"isolation violation: val/test caches opened: "
                f"{isolation['val_test_npz_opened']}"
            )

        # Build per-sample spatial targets and geometry diagnostics.
        by_image: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for sample in train_samples:
            manifest_record = manifest_by_id[sample["sample_id"]]
            ground_truth = manifest_record["ground_truth"]
            grid_h = int(sample["grid_h"])
            grid_w = int(sample["grid_w"])
            if sample["task"] == "grounding":
                target = grounding_area_target(ground_truth["bbox_1000"], grid_h, grid_w)
            elif sample["task"] == "pointing":
                target = pointing_bilinear_target(
                    ground_truth["point_1000"], grid_h, grid_w
                )
            else:  # pragma: no cover - guarded by the join contract
                raise SpatialContractError(f"unknown task {sample['task']!r}")
            diagnostics = target_diagnostics(target)
            spatial_target_size = int(target.size)
            if spatial_target_size != int(sample["group_count"]):
                raise SpatialContractError(
                    f"{sample['sample_id']}: target size != group_count"
                )
            if np.asarray(sample["teacher_soft"]).size != spatial_target_size:
                raise SpatialContractError(
                    f"{sample['sample_id']}: teacher target length != group_count"
                )
            bbox = np.asarray(ground_truth["bbox_1000"], dtype=np.float64)
            point = np.asarray(ground_truth["point_1000"], dtype=np.float64)
            sample["spatial_target"] = target
            sample["spatial_diagnostics"] = diagnostics
            sample["point_inside_bbox"] = point_inside_bbox(point, bbox)
            by_image[sample["source_image_sha256"]][sample["task"]] = sample
        state["by_image"] = by_image

        rows: list[dict[str, Any]] = []
        record_level: list[dict[str, Any]] = []
        integrity_target_errors: list[str] = []
        for image_hash in sorted(by_image):
            tasks = by_image[image_hash]
            grounding = tasks["grounding"]
            pointing = tasks["pointing"]
            group_count = int(grounding["group_count"])
            grid_h = int(grounding["grid_h"])
            grid_w = int(grounding["grid_w"])
            if group_count != int(pointing["group_count"]):
                raise SpatialContractError(
                    f"{image_hash}: grounding/pointing group_count differ"
                )

            semantic = pair_separation_metrics(
                grounding["teacher_soft"], pointing["teacher_soft"], ks=ks
            )
            spatial = pair_separation_metrics(
                grounding["spatial_target"], pointing["spatial_target"], ks=ks
            )
            for task_name, sample in (("grounding", grounding), ("pointing", pointing)):
                diagnostics = sample["spatial_diagnostics"]
                if not (
                    diagnostics["finite"]
                    and diagnostics["non_negative"]
                    and diagnostics["sum_to_one_within_1e-9"]
                ):
                    integrity_target_errors.append(sample["sample_id"])
                compatibility = semantic_spatial_compatibility(
                    sample["spatial_target"], sample["teacher_soft"]
                )
                l2_per_budget = l2_baseline_record(
                    sample["group_mean"], sample["spatial_target"], ks=ks
                )
                anchor = anchor_baseline_record(
                    sample["group_mean"],
                    sample["spatial_target"],
                    grid_h,
                    grid_w,
                    ks=ks,
                )
                record_level.append(
                    {
                        "sample_id": sample["sample_id"],
                        "task": task_name,
                        "group_count": group_count,
                        "compat_spearman": compatibility["spearman"],
                        "compat_ndcg_full": compatibility["ndcg_full"],
                        "l2": l2_per_budget,
                        "anchor": anchor,
                        "point_inside_bbox": bool(sample["point_inside_bbox"]),
                    }
                )

            row: dict[str, Any] = {
                "image_sha256": image_hash,
                "grounding_sample_id": grounding["sample_id"],
                "pointing_sample_id": pointing["sample_id"],
                "group_count": group_count,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "point_inside_bbox": 0.5
                * (
                    (1.0 if grounding["point_inside_bbox"] else 0.0)
                    + (1.0 if pointing["point_inside_bbox"] else 0.0)
                ),
                "diff_js_nats": float(spatial["js_nats"] - semantic["js_nats"]),
                "diff_tv": float(spatial["tv"] - semantic["tv"]),
            }
            for key in PAIR_METRIC_KEYS:
                row[f"semantic_{key}"] = float(semantic[key])
                row[f"spatial_{key}"] = float(spatial[key])
            for k in ks:
                row[f"diff_k{int(k)}_disagreement"] = float(
                    spatial[f"k{int(k)}_disagreement"]
                    - semantic[f"k{int(k)}_disagreement"]
                )
            for task_name, sample in (("grounding", grounding), ("pointing", pointing)):
                diagnostics = sample["spatial_diagnostics"]
                row[f"spatial_{task_name}_support_size"] = float(diagnostics["support_size"])
                row[f"spatial_{task_name}_entropy_nats"] = float(diagnostics["entropy_nats"])
                row[f"spatial_{task_name}_effective_support"] = float(
                    diagnostics["effective_support"]
                )
                row[f"spatial_{task_name}_max_mass"] = float(diagnostics["max_mass"])
                row[f"spatial_{task_name}_sum_error"] = float(diagnostics["sum_error"])
            rows.append(row)
        state["integrity_target_errors"] = integrity_target_errors
    except DistillationContractError as exc:
        return _blocked_report(state, str(exc)), []

    # --- Aggregation (no contract exceptions expected below here) ------------
    semantic_signal = _aggregate_pair_metrics(rows, "semantic_")
    spatial_signal = _aggregate_pair_metrics(rows, "spatial_")
    paired_differences = {
        key: paired_stats([float(row[key]) for row in rows]) for key in DIFF_KEYS
    }

    compatibility = {
        "overall": {
            "spearman": _describe_values([record["compat_spearman"] for record in record_level]),
            "ndcg_full": _describe_values([record["compat_ndcg_full"] for record in record_level]),
        },
        "grounding": {
            "spearman": _describe_values(
                [record["compat_spearman"] for record in record_level if record["task"] == "grounding"]
            ),
            "ndcg_full": _describe_values(
                [record["compat_ndcg_full"] for record in record_level if record["task"] == "grounding"]
            ),
        },
        "pointing": {
            "spearman": _describe_values(
                [record["compat_spearman"] for record in record_level if record["task"] == "pointing"]
            ),
            "ndcg_full": _describe_values(
                [record["compat_ndcg_full"] for record in record_level if record["task"] == "pointing"]
            ),
        },
    }

    l2_baseline = _aggregate_baseline(record_level, "l2", ks)
    anchor_baseline = _aggregate_baseline(record_level, "anchor", ks)
    correction = _anchor_correction_note(l2_baseline, anchor_baseline, ks)

    integrity = {
        "join_diagnostics": state["join_diagnostics"],
        "task_pair_completeness": {
            "train_images": len(rows),
            "pair_failures": list(state.get("pair_failures", [])),
        },
        "grid_contract": {
            "group_count_equals_grid_h_times_grid_w": True,
            "mismatches": list(state.get("grid_mismatches", [])),
            "grid_shape_distribution": _grid_shape_distribution(rows),
        },
        "geometry": {
            "bbox_valid_count": len(record_level),
            "bbox_invalid_count": 0,
            "point_valid_count": len(record_level),
            "point_invalid_count": 0,
            "point_inside_bbox_fraction": (
                float(
                    np.mean(
                        [1.0 if record["point_inside_bbox"] else 0.0 for record in record_level]
                    )
                )
                if record_level
                else None
            ),
            "point_inside_bbox_count": int(
                sum(1 for record in record_level if record["point_inside_bbox"])
            ),
        },
        "target_contracts": {
            "finite": True,
            "non_negative": True,
            "sum_to_one_within_1e-9": True,
            "violations": list(state.get("integrity_target_errors", [])),
            "max_abs_sum_error": max(
                [
                    abs(float(row[f"spatial_{task}_sum_error"]))
                    for row in rows
                    for task in ("grounding", "pointing")
                ]
                or [0.0]
            ),
        },
        "expected_train_records": EXPECTED_TRAIN_RECORDS,
        "expected_train_images": EXPECTED_TRAIN_IMAGES,
        "train_record_count": len(record_level),
        "train_image_count": len(rows),
        "matches_expected": (
            len(record_level) == EXPECTED_TRAIN_RECORDS
            and len(rows) == EXPECTED_TRAIN_IMAGES
        ),
    }
    integrity_ok = bool(
        integrity["train_record_count"] > 0
        and integrity["train_image_count"] > 0
        and not integrity["task_pair_completeness"]["pair_failures"]
        and not integrity["grid_contract"]["mismatches"]
        and not integrity["target_contracts"]["violations"]
        and integrity["geometry"]["bbox_invalid_count"] == 0
        and integrity["geometry"]["point_invalid_count"] == 0
        and state["isolation"]["opened_only_train_npz"]
        and state["isolation"]["val_test_npz_opened"] == []
    )

    js_stats = paired_differences["diff_js_nats"]
    tv_stats = paired_differences["diff_tv"]
    if not integrity_ok:
        decision_label = DECISION_BLOCKED
    elif js_stats["ci_rule_lower_gt_zero"] and tv_stats["ci_rule_lower_gt_zero"]:
        decision_label = DECISION_PRESENT
    else:
        decision_label = DECISION_NOT_ESTABLISHED

    provenance = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": AUDIT_SEED,
        "split_seed": int(split_seed),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "permutation_resamples": PERMUTATION_RESAMPLES,
        "ci_level": CI_LEVEL,
        "config_sha256": config_sha256(config_path),
        "index_sha256": index_file_sha256(index_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_npz_aggregate_sha256": state["input_aggregate_sha"],
        "input_npz_scope": "train split only; aggregate over sorted (sample_id, file sha256)",
        "input_npz_count": len(state["resolved_paths"]),
        "gpu_used": False,
        "model_loaded": False,
        "network_used": False,
    }
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "report_version": REPORT_VERSION,
        "report_stem": REPORT_STEM,
        "status": "COMPUTED",
        "decision_label": decision_label,
        "scope": (
            "parameter-free spatial supervision target feasibility audit; "
            "TRAIN split only; zero-update, CPU-only float64; no model load, no "
            "GPU, no network, no val/test npz, no CV-Bench"
        ),
        "inputs": {
            "config": str(config_path),
            "index": str(index_path),
            "manifest": str(manifest_path),
            "features_dir": str(features_dir) if features_dir else str(Path(index_path).parent),
        },
        "split_counts": {name: len(value) for name, value in state["split"].items()},
        "split_group_summary": state["group_summary"],
        "isolation": state["isolation"],
        "integrity": integrity,
        "geometry": _geometry_summary(rows),
        "semantic_pair_signal": semantic_signal,
        "spatial_pair_signal": spatial_signal,
        "paired_differences": paired_differences,
        "semantic_spatial_compatibility": compatibility,
        "l2_baseline": l2_baseline,
        "anchor_baseline": anchor_baseline,
        "correction": correction,
        "provenance": provenance,
        "ks": [int(k) for k in ks],
        "js_base": "natural_log",
        "notes": dict(NOTES),
        "limitations": list(LIMITATIONS),
    }
    return report, rows


def _grid_shape_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = defaultdict(int)
    for row in rows:
        distribution[f"{int(row['grid_h'])}x{int(row['grid_w'])}"] += 1
    return dict(sorted(distribution.items()))


def _geometry_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"image_count": 0}
    result: dict[str, Any] = {"image_count": len(rows)}
    for task in ("grounding", "pointing"):
        result[task] = {
            key: describe([float(row[f"spatial_{task}_{key}"]) for row in rows])
            for key in (
                "support_size",
                "entropy_nats",
                "effective_support",
                "max_mass",
                "sum_error",
            )
        }
    return result


def _baseline_metric_means(
    baseline: Mapping[str, Any], metric: str, ks: Sequence[int]
) -> dict[str, dict[str, float | None]]:
    return {
        str(int(k)): {
            scope: baseline["per_budget"][str(int(k))][scope][metric]["mean"]
            for scope in ("overall", "grounding", "pointing")
        }
        for k in ks
    }


def _anchor_correction_note(
    l2_baseline: Mapping[str, Any],
    anchor_baseline: Mapping[str, Any],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Self-checking correction record comparing old and restored anchor NDCG."""
    corrected_mass = _baseline_metric_means(anchor_baseline, "mass", ks)
    max_mass_diff = 0.0
    for k in ks:
        key = str(int(k))
        for scope in ("overall", "grounding", "pointing"):
            old = CORRECTION_OLD_ANCHOR_MASS_MEAN[key][scope]
            new = corrected_mass[key][scope]
            if old is None or new is None:
                max_mass_diff = float("inf")
            else:
                max_mass_diff = max(max_mass_diff, abs(float(old) - float(new)))
    return {
        "label": CORRECTION_LABEL,
        "issued": CORRECTION_ISSUED,
        "defect": (
            "anchor_baseline_record computed NDCG as "
            "ndcg_at_k(scores, target, effective), which ignores the "
            "anchor-selected set and therefore duplicated the pure-L2 NDCG even "
            "when anchor retained mass differed."
        ),
        "old_definition": "ndcg_at_k(all L2 scores, spatial_target, effective)",
        "corrected_definition": (
            "ndcg_for_selected_indices(anchor_selected, L2 scores, "
            "spatial_target, effective): DCG over only the selected groups "
            "ordered by descending L2 score with deterministic low-index "
            "tie-break, divided by the all-group ideal DCG@K"
        ),
        "old_script_sha256": CORRECTION_OLD_SCRIPT_SHA256,
        "old_output_sha256": dict(CORRECTION_OLD_OUTPUT_SHA256),
        "anchor_ndcg_old": _baseline_metric_means(l2_baseline, "ndcg", ks),
        "anchor_ndcg_corrected": _baseline_metric_means(
            anchor_baseline, "ndcg", ks
        ),
        "anchor_mass_unchanged": bool(max_mass_diff <= 1e-12),
        "anchor_mass_max_abs_diff": float(max_mass_diff),
        "primary_metrics_unchanged": True,
        "note": (
            "The pre-correction anchor NDCG equals the pure-L2 NDCG because the "
            "ranking ignored the anchor-selected set.  Retained target mass "
            "already used the selected set and is unchanged.  The preregistered "
            "spatial targets, JS/TV decision rule, data split, inputs and "
            "decision label were not modified."
        ),
    }


def _aggregate_baseline(
    record_level: Sequence[Mapping[str, Any]], name: str, ks: Sequence[int]
) -> dict[str, Any]:
    if name == "anchor":
        available = bool(record_level) and all(
            bool(record.get("anchor", {}).get("available")) for record in record_level
        )
        if not available:
            reasons = sorted(
                {
                    str(record.get("anchor", {}).get("reason"))
                    for record in record_level
                    if not bool(record.get("anchor", {}).get("available"))
                }
            )
            return {"available": False, "status": "NOT_COMPUTED", "reasons": reasons}
    else:
        available = True
    per_budget: dict[str, Any] = {}
    kind_counts: dict[str, int] = defaultdict(int)
    for k in ks:
        key = str(int(k))
        budget: dict[str, Any] = {}
        for scope in ("overall", "grounding", "pointing"):
            entries = [
                (
                    record["anchor"]["per_budget"].get(key, {})
                    if name == "anchor"
                    else record["l2"].get(key, {})
                )
                for record in record_level
                if scope == "overall" or record["task"] == scope
            ]
            usable = [
                entry
                for entry in entries
                if entry.get("available", True) and entry.get("ndcg") is not None
            ]
            budget[scope] = {
                "count": len(usable),
                "k_eff": (
                    float(np.mean([entry["k_eff"] for entry in entries]))
                    if entries
                    else None
                ),
                "ndcg": _describe_values([entry["ndcg"] for entry in usable]),
                "mass": _describe_values([entry["mass"] for entry in usable]),
            }
        if name == "anchor":
            for entry in (
                record["anchor"]["per_budget"].get(key, {}) for record in record_level
            ):
                if entry.get("available"):
                    kind_counts[str(entry.get("kind"))] += 1
        per_budget[key] = budget
    result: dict[str, Any] = {
        "available": True,
        "per_budget": per_budget,
        "record_count": len(record_level),
    }
    if name == "anchor":
        result["selection_kind_counts"] = dict(kind_counts)
    return result


def _blocked_report(state: Mapping[str, Any], violation: str) -> dict[str, Any]:
    split = state.get("split")
    group_summary = state.get("group_summary")
    isolation = state.get("isolation")
    join_diagnostics = state.get("join_diagnostics")
    return {
        "schema_version": "1.0",
        "report_version": REPORT_VERSION,
        "report_stem": REPORT_STEM,
        "status": DECISION_BLOCKED,
        "decision_label": DECISION_BLOCKED,
        "scope": (
            "parameter-free spatial supervision target feasibility audit; "
            "TRAIN split only; zero-update, CPU-only float64"
        ),
        "inputs": {},
        "split_counts": (
            {name: len(value) for name, value in split.items()} if split else {}
        ),
        "split_group_summary": group_summary or {},
        "isolation": isolation or {},
        "integrity": {
            "join_diagnostics": join_diagnostics or {},
            "task_pair_completeness": {
                "pair_failures": list(state.get("pair_failures", []))
            },
            "grid_contract": {"mismatches": list(state.get("grid_mismatches", []))},
            "contract_violations": [violation],
        },
        "provenance": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "seed": AUDIT_SEED,
            "gpu_used": False,
            "model_loaded": False,
            "network_used": False,
        },
        "ks": [int(k) for k in KS],
        "notes": dict(NOTES),
        "limitations": list(LIMITATIONS),
        "contract_violations": [violation],
    }


# ---------------------------------------------------------------------------
# Preregistration verification
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_preregistration(
    prereg_config_path: Path,
    prereg_report_path: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Check the pinned input hashes and the preregistration pair linkage."""
    base = Path(root) if root is not None else _project_root()
    config = json.loads(Path(prereg_config_path).read_text(encoding="utf-8"))
    report = json.loads(Path(prereg_report_path).read_text(encoding="utf-8"))
    pinned_config = config["inputs"]
    pinned_report = report["frozen_inputs"]
    results: dict[str, Any] = {}
    names = sorted(set(pinned_config) | set(pinned_report))
    for name in names:
        entry = pinned_config.get(name, pinned_report.get(name))
        relative = str(entry["path"])
        resolved = Path(relative)
        if not resolved.is_absolute():
            resolved = base / relative
        pinned = str(entry["sha256"])
        actual = _sha256_file(resolved)
        report_pin = pinned_report.get(name, {}).get("sha256")
        results[name] = {
            "path": relative,
            "pinned_config_sha256": pinned,
            "pinned_report_sha256": report_pin,
            "actual_sha256": actual,
            "match_config": actual == pinned,
            "match_report": actual == report_pin,
        }
    config_sha = _sha256_file(prereg_config_path)
    report_config_pin = str(report["config_preregistration"]["sha256"])
    return {
        "preregistration_config_path": str(prereg_config_path),
        "preregistration_report_path": str(prereg_report_path),
        "preregistration_config_sha256": config_sha,
        "preregistration_report_sha256": _sha256_file(prereg_report_path),
        "config_prereg_pin_match": config_sha == report_config_pin,
        "pinned_inputs": results,
        "all_match": bool(
            all(entry["match_config"] and entry["match_report"] for entry in results.values())
            and config_sha == report_config_pin
        ),
    }


def run_audit(
    *,
    config_path: Path,
    index_path: Path,
    manifest_path: Path,
    features_dir: Path | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    ks: Sequence[int] = KS,
    prereg_config_path: Path | None = None,
    prereg_report_path: Path | None = None,
    verify_preregistration_hashes: bool = True,
    require_prereg_match: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the audit, optionally verifying the preregistration before/after."""
    before: dict[str, Any] | None = None
    if verify_preregistration_hashes:
        if prereg_config_path is None or prereg_report_path is None:
            raise SpatialContractError(
                "preregistration paths are required when verification is enabled"
            )
        before = verify_preregistration(prereg_config_path, prereg_report_path)
        if require_prereg_match and not before["all_match"]:
            raise SpatialContractError(
                "preregistration hash mismatch before the run: "
                + json.dumps(
                    {
                        name: entry
                        for name, entry in before["pinned_inputs"].items()
                        if not (entry["match_config"] and entry["match_report"])
                    },
                    sort_keys=True,
                )
            )
    report, rows = compute_report(
        config_path=config_path,
        index_path=index_path,
        manifest_path=manifest_path,
        features_dir=features_dir,
        cache_loader=cache_loader,
        ks=ks,
    )
    if verify_preregistration_hashes and before is not None:
        after = verify_preregistration(prereg_config_path, prereg_report_path)
        unchanged = {
            name: before["pinned_inputs"][name]["actual_sha256"]
            == after["pinned_inputs"][name]["actual_sha256"]
            for name in before["pinned_inputs"]
        }
        report["preregistration"] = {
            "preregistration_config_path": before["preregistration_config_path"],
            "preregistration_config_sha256": before["preregistration_config_sha256"],
            "preregistration_report_path": before["preregistration_report_path"],
            "preregistration_report_sha256": before["preregistration_report_sha256"],
            "config_prereg_pin_match": before["config_prereg_pin_match"],
            "pinned_inputs": after["pinned_inputs"],
            "all_pinned_match_after": after["all_match"],
            "byte_for_byte_unchanged": bool(all(unchanged.values())),
            "unchanged_by_input": unchanged,
        }
    return report, rows


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
CSV_FIELDS: tuple[str, ...] = (
    "image_sha256",
    "grounding_sample_id",
    "pointing_sample_id",
    "group_count",
    "grid_h",
    "grid_w",
    "point_inside_bbox",
    "semantic_js_nats",
    "semantic_symmetric_kl",
    "semantic_tv",
    "semantic_argmax_different",
    "semantic_k48_effective",
    "semantic_k48_jaccard",
    "semantic_k48_overlap",
    "semantic_k48_disagreement",
    "semantic_k64_jaccard",
    "semantic_k64_overlap",
    "semantic_k64_disagreement",
    "semantic_k96_jaccard",
    "semantic_k96_overlap",
    "semantic_k96_disagreement",
    "spatial_js_nats",
    "spatial_symmetric_kl",
    "spatial_tv",
    "spatial_argmax_different",
    "spatial_k48_effective",
    "spatial_k48_jaccard",
    "spatial_k48_overlap",
    "spatial_k48_disagreement",
    "spatial_k64_jaccard",
    "spatial_k64_overlap",
    "spatial_k64_disagreement",
    "spatial_k96_jaccard",
    "spatial_k96_overlap",
    "spatial_k96_disagreement",
    "diff_js_nats",
    "diff_tv",
    "diff_k48_disagreement",
    "diff_k64_disagreement",
    "diff_k96_disagreement",
    "spatial_grounding_support_size",
    "spatial_grounding_entropy_nats",
    "spatial_grounding_effective_support",
    "spatial_grounding_max_mass",
    "spatial_grounding_sum_error",
    "spatial_pointing_support_size",
    "spatial_pointing_entropy_nats",
    "spatial_pointing_effective_support",
    "spatial_pointing_max_mass",
    "spatial_pointing_sum_error",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv as _csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = _csv.DictWriter(
            stream, fieldnames=list(CSV_FIELDS), extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _stat_line(name: str, stats: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {_fmt(stats.get('count'))} | {_fmt(stats.get('mean'))} | "
        f"{_fmt(stats.get('median'))} | {_fmt(stats.get('std'))} | "
        f"{_fmt(stats.get('p5'))} | {_fmt(stats.get('p95'))} | {_fmt(stats.get('min'))} | "
        f"{_fmt(stats.get('max'))} |"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    if report.get("decision_label") == DECISION_BLOCKED:
        lines = [
            "# 空间监督目标可行性审计（TRAIN-only，零更新）",
            "",
            f"- 决策：`{DECISION_BLOCKED}`",
            f"- 合同违规：{report.get('contract_violations')}",
            "",
            "本报告不训练、不调参、不加载模型、不使用 GPU、不读取 val/test NPZ。",
        ]
        return "\n".join(lines) + "\n"

    lines = [
        "# 空间监督目标可行性审计（TRAIN-only，零更新）",
        "",
        f"- 决策：`{report['decision_label']}`",
        f"- 训练记录/图像：{report['integrity']['train_record_count']} / "
        f"{report['integrity']['train_image_count']}",
        f"- 隔离：`val_test_npz_opened={report['isolation']['val_test_npz_opened']}`，"
        f"opened_npz_count={report['isolation']['opened_npz_count']}",
        f"- 预注册：`{report.get('preregistration', {}).get('preregistration_config_sha256')}`",
        "",
        "## 目标定义（冻结，不调参，不与语义 teacher 混合）",
        "",
        "- `grounding_area_target`：bbox 与网格单元精确交集面积，按总面积归一化。",
        "- `pointing_bilinear_target`：point→连续中心坐标→四邻可分离双线性权重，"
        "合并重复边界索引并归一化；无高斯、无带宽参数。",
        "",
        "## 同图任务分离（image 为独立单位）",
        "",
        "### 语义 teacher（缓存 teacher_soft_target）",
        "",
        "| 指标 | count | mean | median | std | p5 | p95 | min | max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in ("js_nats", "symmetric_kl", "tv"):
        lines.append(_stat_line(key, report["semantic_pair_signal"]["metric_stats"][key]))
    lines += [
        "",
        f"- argmax 不同比例："
        f"{_fmt(report['semantic_pair_signal']['fractions']['argmax_different_fraction'])}",
        "",
        "### 空间 teacher（grounding 面积 vs pointing 双线性）",
        "",
        "| 指标 | count | mean | median | std | p5 | p95 | min | max |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in ("js_nats", "symmetric_kl", "tv"):
        lines.append(_stat_line(key, report["spatial_pair_signal"]["metric_stats"][key]))
    lines += [
        "",
        f"- argmax 不同比例："
        f"{_fmt(report['spatial_pair_signal']['fractions']['argmax_different_fraction'])}",
        "",
        "## 配对差 spatial − semantic（bootstrap 2000 / sign-flip 10000，seed 20261003）",
        "",
        "| 指标 | n | mean | median | std | CI95 low | CI95 high | p(双侧) | Cohen dz |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in DIFF_KEYS:
        stats = report["paired_differences"][key]
        lines.append(
            f"| {key} | {_fmt(stats.get('n'))} | {_fmt(stats.get('mean'))} | "
            f"{_fmt(stats.get('median'))} | {_fmt(stats.get('std'))} | "
            f"{_fmt(stats.get('ci_low'))} | {_fmt(stats.get('ci_high'))} | "
            f"{_fmt(stats.get('p_value_two_sided'))} | {_fmt(stats.get('cohens_dz'))} |"
        )
    lines += [
        "",
        "无多重候选搜索，报告原始 p 值，不自创 Holm 校正。",
        "",
        "## 语义/空间兼容性（Spearman 与 full NDCG，relevance=teacher_soft_target）",
        "",
        "| 范围 | Spearman mean | NDCG mean |",
        "| --- | --- | --- |",
    ]
    for scope in ("overall", "grounding", "pointing"):
        compatibility = report["semantic_spatial_compatibility"][scope]
        lines.append(
            f"| {scope} | {_fmt(compatibility['spearman']['mean'])} | "
            f"{_fmt(compatibility['ndcg_full']['mean'])} |"
        )
    lines += [
        "",
        "低一致性只说明目标不同，并不自动说明某一方错误。",
        "",
        "## 硬件规整基线（L2 / 3x3 anchor，K=48/64/96）",
        "",
        "| 基线 | K | 范围 | NDCG mean | 保留目标质量 mean | count |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, label in (("l2_baseline", "L2"), ("anchor_baseline", "anchor")):
        baseline = report.get(name, {})
        if not baseline.get("available"):
            lines.append(f"| {label} | - | - | NOT_COMPUTED | NOT_COMPUTED | 0 |")
            continue
        for k in (48, 64, 96):
            for scope in ("overall", "grounding", "pointing"):
                budget = baseline["per_budget"][str(k)][scope]
                lines.append(
                    f"| {label} | {k} | {scope} | {_fmt(budget['ndcg']['mean'])} | "
                    f"{_fmt(budget['mass']['mean'])} | {_fmt(budget['count'])} |"
                )
    correction = report.get("correction")
    if correction:
        lines += [
            "",
            "anchor 行 NDCG 为**仅对 anchor 选定组按 L2 降序排序**的受限 NDCG"
            "（低索引打破平局，分母为全体组 ideal DCG@K）；与其保留目标质量使用同一选定集合。",
            "",
            "## 更正记录：anchor NDCG 选择集缺陷（旧值 vs 校正值）",
            "",
            f"- 标签：`{correction['label']}`（{correction['issued']}）",
            f"- 缺陷：{correction['defect']}",
            f"- 旧定义：`{correction['old_definition']}`",
            f"- 校正定义：`{correction['corrected_definition']}`",
            f"- 保留目标质量不变：`{correction['anchor_mass_unchanged']}`"
            f"（最大绝对差 {_fmt(correction['anchor_mass_max_abs_diff'])}）",
            f"- 主指标/JS/TV 决策不变：`{correction['primary_metrics_unchanged']}`",
            "",
            "| K | 范围 | anchor NDCG 旧 | anchor NDCG 校正 |",
            "| --- | --- | --- | --- |",
        ]
        for k in (48, 64, 96):
            for scope in ("overall", "grounding", "pointing"):
                old = correction["anchor_ndcg_old"][str(k)][scope]
                new = correction["anchor_ndcg_corrected"][str(k)][scope]
                lines.append(f"| {k} | {scope} | {_fmt(old)} | {_fmt(new)} |")
        lines += ["", f"- {correction['note']}"]
    lines += [
        "",
        "## 限制",
        "",
    ]
    lines += [f"- {item}" for item in report["limitations"]]
    lines += [
        "",
        f"> 决策规则：仅当所有完整性检查通过且 `spatial-semantic` 的 95% bootstrap "
        f"CI 下界对 JS 与 TV 同时 > 0 时，才判 `{DECISION_PRESENT}`；"
        "top-K 仅为描述性，不能覆盖该规则。",
        "> 即使 signal present，可学习性与下游任务精度仍未验证；本报告不声称精度提升或 FPGA 加速。",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "prompt_projection_distillation.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=root
        / "artifacts"
        / "token_selector"
        / "prompt_projection_features"
        / "index.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "datasets" / "prompt_projection_clean_inductive_v1.jsonl",
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=root / "reports" / "token_selector")
    parser.add_argument(
        "--prereg-config", type=Path, default=root / PREREG_CONFIG_REL
    )
    parser.add_argument(
        "--prereg-report", type=Path, default=root / PREREG_REPORT_REL
    )
    parser.add_argument("--no-prereg-verify", action="store_true")
    parser.add_argument(
        "--harness-result",
        type=Path,
        default=Path("/tmp/robobrain_harness_result.json"),
    )
    parser.add_argument(
        "--harness-durable",
        type=Path,
        default=root
        / "reports"
        / "token_selector"
        / "logs"
        / f"{REPORT_STEM}-harness-result.json",
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()
    report, rows = run_audit(
        config_path=args.config,
        index_path=args.index,
        manifest_path=args.manifest,
        features_dir=args.features_dir,
        prereg_config_path=args.prereg_config,
        prereg_report_path=args.prereg_report,
        verify_preregistration_hashes=not args.no_prereg_verify,
    )
    runtime_seconds = time.perf_counter() - started
    peak_rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    report["runtime_seconds"] = runtime_seconds
    report["peak_rss_mib"] = peak_rss_mib

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{REPORT_STEM}.csv"
    json_path = out_dir / f"{REPORT_STEM}.json"
    md_path = out_dir / f"{REPORT_STEM}.md"
    write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")

    output_hashes = {
        "csv": _sha256_file(csv_path),
        "json": _sha256_file(json_path),
        "md": _sha256_file(md_path),
    }
    harness = {
        "status": report["status"],
        "decision_label": report["decision_label"],
        "report": REPORT_STEM,
        "scope": report["scope"],
        "environment": {
            "python": report["provenance"]["python"],
            "numpy": report["provenance"]["numpy"],
            "platform": report["provenance"]["platform"],
        },
        "inputs": {
            "config_sha256": report["provenance"]["config_sha256"],
            "index_sha256": report["provenance"]["index_sha256"],
            "manifest_sha256": report["provenance"]["manifest_sha256"],
            "script_sha256": report["provenance"]["script_sha256"],
            "input_npz_aggregate_sha256": report["provenance"][
                "input_npz_aggregate_sha256"
            ],
        },
        "preregistration": report.get("preregistration", {}),
        "correction": report.get("correction", {}),
        "output_sha256": output_hashes,
        "artifacts": {
            "script": "evaluation/spatial_teacher_feasibility_audit.py",
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "focused_tests": "evaluation/tests/test_spatial_teacher_feasibility_audit.py",
        },
        "isolation": report.get("isolation", {}),
        "key_metrics": {
            "train_records": report.get("integrity", {}).get("train_record_count"),
            "train_images": report.get("integrity", {}).get("train_image_count"),
            "semantic_pair_signal": {
                key: report.get("semantic_pair_signal", {})
                .get("metric_stats", {})
                .get(key, {})
                .get("mean")
                for key in ("js_nats", "symmetric_kl", "tv")
            },
            "spatial_pair_signal": {
                key: report.get("spatial_pair_signal", {})
                .get("metric_stats", {})
                .get(key, {})
                .get("mean")
                for key in ("js_nats", "symmetric_kl", "tv")
            },
            "paired_differences": report.get("paired_differences", {}),
        },
        "runtime_seconds": runtime_seconds,
        "peak_rss_mib": peak_rss_mib,
        "no_gpu_model_network": {
            "gpu_used": False,
            "model_loaded": False,
            "network_used": False,
        },
        "limitations": report.get("limitations", []),
        "next_decision": (
            "Codex/PI decision only: if decision_label == "
            "SPATIAL_SUPERVISION_SIGNAL_PRESENT, decide whether to design a "
            "trainable scorer and a promotion-gated experiment; if "
            "NOT_ESTABLISHED, no spatial supervision target is authorized by this "
            "audit; learnability and task accuracy remain untested either way."
        ),
    }
    args.harness_result.parent.mkdir(parents=True, exist_ok=True)
    args.harness_result.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.harness_durable.parent.mkdir(parents=True, exist_ok=True)
    args.harness_durable.write_text(
        json.dumps(harness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"SPATIAL_TEACHER_FEASIBILITY status={report['status']} "
        f"decision={report['decision_label']} images={report.get('integrity', {}).get('train_image_count')}"
    )
    print(f"isolation opened_only_train_npz={report.get('isolation', {}).get('opened_only_train_npz')}")
    if report.get("status") == "COMPUTED":
        diffs = report["paired_differences"]
        for key in DIFF_KEYS:
            stats = diffs[key]
            print(
                f"{key}: mean={stats['mean']:.10f} ci_low={stats['ci_low']:.10f} "
                f"ci_high={stats['ci_high']:.10f} p={stats['p_value_two_sided']:.6g} "
                f"dz={stats['cohens_dz']:.6f}"
            )
    print(f"runtime_seconds={runtime_seconds:.3f} peak_rss_mib={peak_rss_mib:.1f}")
    for name, digest in output_hashes.items():
        print(f"output_{name}_sha256={digest}")
    print(f"harness_result={args.harness_result}")
    print(f"harness_durable={args.harness_durable}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
