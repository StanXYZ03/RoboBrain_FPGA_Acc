"""CPU-only, hardware-regular visual-token selection and FPGA layout contract.

This module is intentionally free of CUDA, model weights and any training
loop.  It operates on plain NumPy arrays and implements the frozen contract for
the main insertion boundary ``B1`` (after vision block 0) with ``B0`` exposed as
a same-interface ablation label only.

Contract summary
----------------
* Minimum selection unit is one complete 2x2 merge group, so the hidden input
  has shape ``[G, 4, 1024]`` (``G`` merged groups, 4 patch positions, 1024
  features).
* Prompt input is ``text_embeddings[T, 2560]``.  Only the *interfaces* of the
  trainable low-rank projections ``Wt: 2560 -> 128`` and ``Wv: 1024 -> 128``
  are defined; weights must be supplied explicitly and are never trained or
  randomly generated here.
* Scores are returned separately as semantic cosine, group L2 saliency and a
  min-max normalised weighted combination with an explicit ``lambda_semantic``.
* Output is an exact merged-group budget ``K`` (multiple of 16, the production
  budgets are 48/64/96).  ``G <= K`` is an element-wise identity bypass.
* A fixed 3x3 spatial partition keeps at least one real anchor per non-empty
  region; the remaining real detail groups come from a *two-level* Top-K
  (bounded per-region local candidates, then a global pick over that bounded
  candidate set).  No global full sort over all ``G`` groups is performed.
* Context slots are dynamic and at most 9.  A context group is emitted only for
  a *context-eligible* region, i.e. a region that has at least two input groups
  and still owns at least one unselected group after compression.  Every
  eligible region is forced to keep at least one unselected group.
* Each context group pools **all** unselected groups of its region with
  non-negative normalised weights into a ``[4, 1024]`` synthetic group.  Its
  representative ``original_group_index`` is the unselected group closest to the
  weighted centroid (stable tie-break: lower index), so no context ever
  degenerates to a zero/``valid=False`` padding token.
* The real budget is dynamic: ``real = K - C`` where ``C`` is the number of
  context-eligible regions, and every non-empty region still keeps at least one
  real anchor.
* All real and context representative ``original_group_index`` values are
  globally unique, so hardware can tell real from context purely through the
  ``source_type`` metadata.  Output entries are ordered by representative index
  ascending.  Equal scores break ties by lower index.

The FPGA transfer layout is only *described* (offsets/sizes), never written.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MERGE_UNIT = 4
FEATURE_DIM = 1024
PROMPT_DIM = 2560
SCORE_DIM = 128
REGION_GRID = 3
# Maximum number of context groups (one per 3x3 region).  The actual number is
# dynamic and only counts context-eligible regions, so it is ``<=`` this bound.
MAX_CONTEXT_SLOTS = REGION_GRID * REGION_GRID
# Retained as a deprecated alias for the historical fixed-slot constant.
CONTEXT_SLOTS = MAX_CONTEXT_SLOTS
REQUIRED_BUDGET_ALIGNMENT = 16
SUPPORTED_BUDGETS = (48, 64, 96)
BOUNDARY_MAIN = "B1"
BOUNDARY_ABLATION = "B0"
ALLOWED_BOUNDARIES = (BOUNDARY_MAIN, BOUNDARY_ABLATION)
SOURCE_REAL = 0
SOURCE_CONTEXT = 1
FPGA_LAYOUT_ALIGNMENT = 64
DEFAULT_LOCAL_CANDIDATE_CAP = 16
DEFAULT_LAMBDA_SEMANTIC = 0.5
_TIE_EPS = 1e-12
_CONFIG_FILENAME = "hardware_regular_token_selector.json"


class SelectorContractError(ValueError):
    """Base class for all contract violations raised by this module."""


class InsufficientBudgetError(SelectorContractError):
    """The requested budget cannot be met under the fixed region rules."""


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _as_float_array(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise SelectorContractError(
            f"{name} must have ndim={ndim}, got ndim={array.ndim}"
        )
    if array.size == 0:
        raise SelectorContractError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise SelectorContractError(f"{name} must contain only finite values")
    return array


def _as_float_matrix(
    name: str, value: Any, shape: Sequence[int | None]
) -> np.ndarray:
    array = _as_float_array(name, value, 2)
    if len(shape) != 2:
        raise SelectorContractError("shape specification must have two entries")
    for axis, expected in enumerate(shape):
        if expected is not None and array.shape[axis] != expected:
            raise SelectorContractError(
                f"{name} must have shape axis {axis} == {expected}, "
                f"got {array.shape[axis]}"
            )
    if array.shape[0] < 1:
        raise SelectorContractError(f"{name} must have at least one row")
    return array


def validate_hidden(hidden: Any) -> np.ndarray:
    """Validate ``[G, 4, 1024]`` merged-group hidden states."""
    array = _as_float_array("hidden", hidden, 3)
    if array.shape[1] != MERGE_UNIT:
        raise SelectorContractError(
            f"hidden patch axis must be {MERGE_UNIT}, got {array.shape[1]}"
        )
    if array.shape[2] != FEATURE_DIM:
        raise SelectorContractError(
            f"hidden feature axis must be {FEATURE_DIM}, got {array.shape[2]}"
        )
    return array


def validate_budget(budget: Any) -> int:
    """Validate the merged-group budget ``K`` (positive multiple of 16)."""
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise SelectorContractError("budget must be an integer")
    value = int(budget)
    if value <= 0:
        raise SelectorContractError("budget must be positive")
    if value % REQUIRED_BUDGET_ALIGNMENT != 0:
        raise SelectorContractError(
            f"budget must be a multiple of {REQUIRED_BUDGET_ALIGNMENT}, got {value}"
        )
    return value


def validate_boundary(boundary: Any) -> str:
    if boundary not in ALLOWED_BOUNDARIES:
        raise SelectorContractError(
            f"boundary must be one of {ALLOWED_BOUNDARIES}, got {boundary!r}"
        )
    return str(boundary)


# ---------------------------------------------------------------------------
# Trainable low-rank projection interface (never trained here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LowRankProjection:
    """Interface for ``Wt: 2560 -> 128`` and ``Wv: 1024 -> 128``.

    Both matrices are stored as ``[SCORE_DIM, source_dim]`` so that
    ``features @ W.T`` performs the projection.  The module never initialises
    these weights: callers must provide explicit values.
    """

    Wt: np.ndarray
    Wv: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "Wt", _as_float_matrix("Wt", self.Wt, (SCORE_DIM, PROMPT_DIM))
        )
        object.__setattr__(
            self, "Wv", _as_float_matrix("Wv", self.Wv, (SCORE_DIM, FEATURE_DIM))
        )


# ---------------------------------------------------------------------------
# Score interfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroupScores:
    """Reported scores.

    ``combined`` is always present because the selection rule ranks by it.
    ``semantic`` and ``l2`` are optional: the ``l2`` score mode and other
    callers that supply precomputed scores do not have a prompt projection.
    """

    semantic: np.ndarray | None
    l2: np.ndarray | None
    combined: np.ndarray
    lambda_semantic: float | None


def group_representation(hidden: Any) -> np.ndarray:
    """Return the mean of the 4 patch positions, shape ``[G, 1024]``."""
    return validate_hidden(hidden).mean(axis=1)


def group_l2_saliency(hidden: Any) -> np.ndarray:
    """L2 norm of the per-group mean representation, shape ``[G]``."""
    representation = group_representation(hidden)
    return np.linalg.norm(representation, axis=1)


def semantic_cosine_scores(
    hidden: Any, text_embeddings: Any, projection: LowRankProjection
) -> np.ndarray:
    """Maximum cosine similarity between each visual group and prompt tokens."""
    if not isinstance(projection, LowRankProjection):
        raise SelectorContractError("projection must be a LowRankProjection")
    representation = group_representation(hidden)
    text = _as_float_matrix("text_embeddings", text_embeddings, (None, PROMPT_DIM))
    visual_proj = representation @ projection.Wv.T
    text_proj = text @ projection.Wt.T
    visual_norm = np.linalg.norm(visual_proj, axis=1)
    text_norm = np.linalg.norm(text_proj, axis=1)
    if np.any(visual_norm <= _TIE_EPS) or np.any(text_norm <= _TIE_EPS):
        raise SelectorContractError(
            "semantic cosine is undefined for zero-norm projected features"
        )
    visual_unit = visual_proj / visual_norm[:, None]
    text_unit = text_proj / text_norm[:, None]
    return (visual_unit @ text_unit.T).max(axis=1)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    high = float(values.max())
    if high - low <= _TIE_EPS:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def minmax_normalize(values: Any) -> np.ndarray:
    """Public 1-D min-max normalisation used by the ``combined`` score term."""
    return _minmax_normalize(_as_float_array("values", values, 1))



def combine_scores(
    semantic: Any, l2: Any, lambda_semantic: float
) -> np.ndarray:
    """Min-max normalise both terms and combine with an explicit weight."""
    semantic_array = _as_float_array("semantic", semantic, 1)
    l2_array = _as_float_array("l2", l2, 1)
    if semantic_array.shape != l2_array.shape:
        raise SelectorContractError("semantic and l2 must share the same shape")
    weight = float(lambda_semantic)
    if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
        raise SelectorContractError("lambda_semantic must be in [0, 1]")
    combined = weight * _minmax_normalize(semantic_array) + (
        1.0 - weight
    ) * _minmax_normalize(l2_array)
    return combined


def score_groups(
    hidden: Any,
    text_embeddings: Any,
    projection: LowRankProjection,
    lambda_semantic: float,
) -> GroupScores:
    """Return semantic, L2 and combined scores in one call."""
    hidden_array = validate_hidden(hidden)
    semantic = semantic_cosine_scores(hidden_array, text_embeddings, projection)
    l2 = group_l2_saliency(hidden_array)
    combined = combine_scores(semantic, l2, lambda_semantic)
    return GroupScores(
        semantic=semantic,
        l2=l2,
        combined=combined,
        lambda_semantic=float(lambda_semantic),
    )


def scores_from_precomputed(
    semantic: Any | None = None,
    l2: Any | None = None,
    combined: Any | None = None,
    lambda_semantic: float | None = None,
) -> GroupScores:
    """Assemble ``GroupScores`` from caller-supplied arrays.

    This never generates or trains a projection.  ``combined`` is mandatory or
    derived from explicit ``semantic`` + ``l2`` + ``lambda_semantic``.  The
    ``l2`` score mode only needs ``l2`` + ``combined``.
    """
    if semantic is None and l2 is None and combined is None:
        raise SelectorContractError(
            "at least one of semantic, l2 or combined must be provided"
        )
    semantic_array = (
        _as_float_array("semantic", semantic, 1) if semantic is not None else None
    )
    l2_array = _as_float_array("l2", l2, 1) if l2 is not None else None
    combined_array = (
        _as_float_array("combined", combined, 1) if combined is not None else None
    )
    reference = (
        combined_array
        if combined_array is not None
        else l2_array
        if l2_array is not None
        else semantic_array
    )
    assert reference is not None  # guarded by the emptiness check above
    size = int(reference.size)
    for name, array in (
        ("semantic", semantic_array),
        ("l2", l2_array),
        ("combined", combined_array),
    ):
        if array is not None and int(array.size) != size:
            raise SelectorContractError(
                f"{name} has {array.size} entries but the score vector has {size}"
            )
    if combined_array is None:
        if semantic_array is None or l2_array is None:
            raise SelectorContractError(
                "combined can only be derived when semantic and l2 are both given"
            )
        if lambda_semantic is None:
            raise SelectorContractError(
                "lambda_semantic is required to derive combined from semantic and l2"
            )
        combined_array = combine_scores(semantic_array, l2_array, lambda_semantic)
    weight: float | None = None
    if lambda_semantic is not None:
        weight = float(lambda_semantic)
        if not math.isfinite(weight) or not (0.0 <= weight <= 1.0):
            raise SelectorContractError("lambda_semantic must be in [0, 1]")
    return GroupScores(
        semantic=semantic_array,
        l2=l2_array,
        combined=combined_array,
        lambda_semantic=weight,
    )



# ---------------------------------------------------------------------------
# Fixed 3x3 spatial regions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialRegion:
    region_id: int
    grid_h: int
    grid_w: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    members: tuple[int, ...]
    center_row: float
    center_col: float


def build_regions(
    grid_h: int, grid_w: int, region_grid: int = REGION_GRID
) -> tuple[SpatialRegion, ...]:
    """Partition the merged grid into ``region_grid x region_grid`` regions."""
    if not isinstance(grid_h, (int, np.integer)) or not isinstance(
        grid_w, (int, np.integer)
    ):
        raise SelectorContractError("grid_h and grid_w must be integers")
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h <= 0 or grid_w <= 0:
        raise SelectorContractError("grid_h and grid_w must be positive")
    if not isinstance(region_grid, (int, np.integer)) or int(region_grid) <= 0:
        raise SelectorContractError("region_grid must be a positive integer")
    region_grid = int(region_grid)
    regions: list[SpatialRegion] = []
    for region_row in range(region_grid):
        row_start = region_row * grid_h // region_grid
        row_end = (region_row + 1) * grid_h // region_grid
        for region_col in range(region_grid):
            col_start = region_col * grid_w // region_grid
            col_end = (region_col + 1) * grid_w // region_grid
            members = tuple(
                row * grid_w + col
                for row in range(row_start, row_end)
                for col in range(col_start, col_end)
            )
            center_row = (
                (row_start + row_end - 1) / 2.0 if row_end > row_start else float("nan")
            )
            center_col = (
                (col_start + col_end - 1) / 2.0 if col_end > col_start else float("nan")
            )
            regions.append(
                SpatialRegion(
                    region_id=region_row * region_grid + region_col,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    row_start=row_start,
                    row_end=row_end,
                    col_start=col_start,
                    col_end=col_end,
                    members=members,
                    center_row=center_row,
                    center_col=center_col,
                )
            )
    return tuple(regions)


def region_representative(region: SpatialRegion) -> int:
    """Nearest original group to the region centre; ties pick the lower index."""
    if not region.members:
        return -1

    def key(group: int) -> tuple[float, int]:
        row = group // region.grid_w
        col = group % region.grid_w
        distance = (row - region.center_row) ** 2 + (col - region.center_col) ** 2
        return (distance, group)

    return int(min(region.members, key=key))


def region_context_representative(
    region: SpatialRegion, remaining: Sequence[int], weights: Any
) -> int:
    """Unselected group closest to the weighted centroid of unselected groups.

    ``weights`` are the non-negative normalised pooling weights aligned with
    ``remaining``.  Empty ``remaining`` returns ``-1``; ties resolve to the
    lower ``original_group_index`` so the choice is fully deterministic.
    """
    indices = [int(group) for group in remaining]
    if not indices:
        return -1
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (len(indices),):
        raise SelectorContractError(
            "context representative weights must align with the remaining groups"
        )
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
        raise SelectorContractError(
            "context representative weights must be finite and non-negative"
        )
    total = float(weight_array.sum())
    if total <= _TIE_EPS:
        weight_array = np.full((len(indices),), 1.0 / len(indices), dtype=np.float64)
    else:
        weight_array = weight_array / total
    rows = np.array([group // region.grid_w for group in indices], dtype=np.float64)
    cols = np.array([group % region.grid_w for group in indices], dtype=np.float64)
    centroid_row = float((weight_array * rows).sum())
    centroid_col = float((weight_array * cols).sum())

    def key(group: int) -> tuple[float, int]:
        row = group // region.grid_w
        col = group % region.grid_w
        distance = (row - centroid_row) ** 2 + (col - centroid_col) ** 2
        return (distance, group)

    return int(min(indices, key=key))


def region_id_by_group(
    grid_h: int, grid_w: int, region_grid: int = REGION_GRID
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for region in build_regions(grid_h, grid_w, region_grid):
        for group in region.members:
            mapping[group] = region.region_id
    return mapping


# ---------------------------------------------------------------------------
# Two-level Top-K real-group selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealGroupSelection:
    anchors: tuple[int, ...]
    details: tuple[int, ...]
    selected: tuple[int, ...]
    effective_candidate_cap: int
    region_ids: tuple[int, ...]


def select_real_groups(
    combined: Any,
    grid_h: int,
    grid_w: int,
    real_budget: int,
    *,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = DEFAULT_LOCAL_CANDIDATE_CAP,
    context_region_ids: Iterable[int] = (),
) -> RealGroupSelection:
    """Select real groups with anchors plus a bounded two-level Top-K.

    The first level takes at most ``effective_candidate_cap`` local candidates
    per region (anchors excluded).  The second level ranks only that bounded
    candidate pool, never all ``G`` groups.  Ties always resolve to the lower
    ``original_group_index``.

    ``context_region_ids`` names the regions that must keep at least one
    unselected group for context pooling; for those regions the local candidate
    pool is reduced by one so the reservation is enforced structurally.
    """
    scores = _as_float_array("combined", combined, 1)
    regions = build_regions(grid_h, grid_w, region_grid)
    if grid_h * grid_w != scores.size:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {scores.size} scored groups"
        )
    if real_budget <= 0:
        raise InsufficientBudgetError("real group budget must be positive")
    if real_budget > scores.size:
        raise InsufficientBudgetError(
            f"real group budget {real_budget} exceeds available groups {scores.size}"
        )
    if not isinstance(local_candidate_cap, (int, np.integer)) or int(
        local_candidate_cap
    ) <= 0:
        raise SelectorContractError("local_candidate_cap must be a positive integer")

    non_empty = [region for region in regions if region.members]
    if real_budget < len(non_empty):
        raise InsufficientBudgetError(
            f"real budget {real_budget} cannot cover one anchor in each of "
            f"{len(non_empty)} non-empty regions"
        )

    reserved_ids = {int(region_id) for region_id in context_region_ids}
    region_by_id = {region.region_id: region for region in non_empty}
    unknown = sorted(reserved_ids - set(region_by_id))
    if unknown:
        raise SelectorContractError(
            f"context_region_ids {unknown} are not non-empty regions"
        )
    for region_id in sorted(reserved_ids):
        if len(region_by_id[region_id].members) < 2:
            raise SelectorContractError(
                f"region {region_id} cannot reserve an unselected group with "
                f"fewer than 2 members"
            )

    anchor_by_region = {
        region.region_id: int(
            max(region.members, key=lambda group: (scores[group], -group))
        )
        for region in non_empty
    }
    anchors = tuple(anchor_by_region[region.region_id] for region in non_empty)
    anchor_set = set(anchors)

    effective_cap = int(local_candidate_cap)
    candidates: set[int] = set()
    for region in non_empty:
        anchor = anchor_by_region[region.region_id]
        ranked = sorted(
            (group for group in region.members if group != anchor),
            key=lambda group: (-scores[group], group),
        )
        reserve = 1 if region.region_id in reserved_ids else 0
        allowed = max(0, len(region.members) - 1 - reserve)
        take = min(len(ranked), effective_cap, allowed)
        candidates.update(int(group) for group in ranked[:take])

    pool = sorted(candidates - anchor_set, key=lambda group: (-scores[group], group))
    details_needed = real_budget - len(anchors)
    if len(pool) < details_needed:
        raise InsufficientBudgetError(
            f"bounded local candidate pool has {len(pool)} groups but "
            f"{details_needed} detail groups are required; increase "
            f"local_candidate_cap"
        )
    details = tuple(int(group) for group in pool[:details_needed])
    selected = tuple(sorted(anchors + details))
    mapping = region_id_by_group(grid_h, grid_w, region_grid)
    return RealGroupSelection(
        anchors=anchors,
        details=details,
        selected=selected,
        effective_candidate_cap=effective_cap,
        region_ids=tuple(mapping[group] for group in selected),
    )


# ---------------------------------------------------------------------------
# Context pooling
# ---------------------------------------------------------------------------


def context_pooling_weights(combined: Any, remaining: Sequence[int]) -> np.ndarray:
    """Non-negative normalised pooling weights for the remaining groups."""
    values = np.asarray(combined, dtype=np.float64)
    indices = [int(group) for group in remaining]
    if not indices:
        return np.zeros((0,), dtype=np.float64)
    raw = np.maximum(values[indices], 0.0)
    total = float(raw.sum())
    if total <= _TIE_EPS:
        return np.full((len(indices),), 1.0 / len(indices), dtype=np.float64)
    return raw / total


def pool_region_context(
    hidden: Any, remaining: Sequence[int], combined: Any
) -> tuple[np.ndarray, bool]:
    """Pool unselected groups into a ``[4, 1024]`` synthetic context group.

    Returns ``(tokens, valid)``.  With no remaining group the result is an
    all-zero context and ``valid=False`` (never NaN).
    """
    hidden_array = validate_hidden(hidden)
    indices = [int(group) for group in remaining]
    if not indices:
        return np.zeros((MERGE_UNIT, FEATURE_DIM), dtype=np.float64), False
    for group in indices:
        if group < 0 or group >= hidden_array.shape[0]:
            raise SelectorContractError(
                f"context group index {group} is outside [0, {hidden_array.shape[0]})"
            )
    weights = context_pooling_weights(combined, indices)
    tokens = np.zeros((MERGE_UNIT, FEATURE_DIM), dtype=np.float64)
    for weight, group in zip(weights, indices):
        tokens += weight * hidden_array[group]
    return tokens, True


# ---------------------------------------------------------------------------
# Main selection entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareSelection:
    boundary: str
    budget: int
    bypass: bool
    groups: np.ndarray
    group_indices: np.ndarray
    patch_indices: np.ndarray
    source_type: np.ndarray
    region_id: np.ndarray
    valid: np.ndarray
    scores: GroupScores | None
    real_group_count: int
    context_group_count: int
    grid_h: int
    grid_w: int
    anchors: tuple[int, ...]
    details: tuple[int, ...]
    lambda_semantic: float | None

    @property
    def output_group_count(self) -> int:
        return int(self.group_indices.shape[0])


def _identity_bypass(hidden: np.ndarray, grid_h: int, grid_w: int, budget: int, boundary: str) -> HardwareSelection:
    group_count = hidden.shape[0]
    group_indices = np.arange(group_count, dtype=np.int64)
    patch_indices = (
        group_indices[:, None] * MERGE_UNIT + np.arange(MERGE_UNIT, dtype=np.int64)
    )
    region_ids = region_id_by_group(grid_h, grid_w)
    return HardwareSelection(
        boundary=boundary,
        budget=budget,
        bypass=True,
        groups=hidden.copy(),
        group_indices=group_indices,
        patch_indices=patch_indices,
        source_type=np.full((group_count,), SOURCE_REAL, dtype=np.uint8),
        region_id=np.array(
            [region_ids[int(group)] for group in group_indices], dtype=np.int16
        ),
        valid=np.ones((group_count,), dtype=bool),
        scores=None,
        real_group_count=group_count,
        context_group_count=0,
        grid_h=grid_h,
        grid_w=grid_w,
        anchors=(),
        details=(),
        lambda_semantic=None,
    )


def select_hardware_regular_groups(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    text_embeddings: Any | None = None,
    projection: LowRankProjection | None = None,
    lambda_semantic: float | None = None,
    semantic_scores: Any | None = None,
    l2_scores: Any | None = None,
    combined_scores: Any | None = None,
    region_grid: int = REGION_GRID,
    max_context_slots: int = MAX_CONTEXT_SLOTS,
    local_candidate_cap: int = DEFAULT_LOCAL_CANDIDATE_CAP,
    boundary: str = BOUNDARY_MAIN,
) -> HardwareSelection:
    """Select exactly ``budget`` output groups (or identity-bypass ``G <= K``).

    The compression path accepts either precomputed ``semantic_scores`` /
    ``l2_scores`` / ``combined_scores`` (used as-is, never regenerated) or the
    existing explicit ``text_embeddings`` + ``LowRankProjection`` prompt path.
    The identity bypass needs neither.  The context count ``C`` is dynamic and
    ``<= max_context_slots``: only regions with at least two input groups that
    still own an unselected group after compression contribute one context
    group each.  The real count is therefore ``K - C``.
    """
    hidden_array = validate_hidden(hidden)
    budget = validate_budget(budget)
    boundary = validate_boundary(boundary)
    if not isinstance(max_context_slots, (int, np.integer)) or int(
        max_context_slots
    ) <= 0:
        raise SelectorContractError(
            "max_context_slots must be a positive integer"
        )
    max_context_slots = int(max_context_slots)

    group_count = hidden_array.shape[0]
    if grid_h * grid_w != group_count:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} merged groups"
        )

    if group_count <= budget:
        return _identity_bypass(hidden_array, grid_h, grid_w, budget, boundary)

    explicit_scores = any(
        value is not None
        for value in (semantic_scores, l2_scores, combined_scores)
    )
    if explicit_scores:
        scores = scores_from_precomputed(
            semantic_scores, l2_scores, combined_scores, lambda_semantic
        )
        if int(scores.combined.size) != group_count:
            raise SelectorContractError(
                f"precomputed combined scores have {scores.combined.size} entries "
                f"but there are {group_count} merged groups"
            )
    else:
        if text_embeddings is None:
            raise SelectorContractError(
                "text_embeddings are required when compression is active"
            )
        if not isinstance(projection, LowRankProjection):
            raise SelectorContractError(
                "projection must be a LowRankProjection when compression is active"
            )
        if lambda_semantic is None:
            raise SelectorContractError(
                "lambda_semantic must be configured explicitly when compression is active"
            )
        scores = score_groups(
            hidden_array, text_embeddings, projection, lambda_semantic
        )

    regions = build_regions(grid_h, grid_w, region_grid)
    non_empty = [region for region in regions if region.members]
    context_regions = [
        region for region in non_empty if len(region.members) >= 2
    ]
    context_count = len(context_regions)
    if context_count > max_context_slots:
        raise SelectorContractError(
            f"{context_count} context-eligible regions exceed max_context_slots "
            f"{max_context_slots}"
        )
    if budget - context_count < len(non_empty):
        raise InsufficientBudgetError(
            f"budget {budget} leaves {budget - context_count} real groups but "
            f"{len(non_empty)} non-empty regions each need an anchor"
        )

    real_budget = budget - context_count
    reserved_region_ids = [region.region_id for region in context_regions]

    real = select_real_groups(
        scores.combined,
        grid_h,
        grid_w,
        real_budget,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        context_region_ids=reserved_region_ids,
    )
    selected_set = set(real.selected)
    region_lookup = region_id_by_group(grid_h, grid_w, region_grid)

    entries: list[tuple[int, int, int, bool, np.ndarray, np.ndarray]] = []
    for group in real.selected:
        patches = group * MERGE_UNIT + np.arange(MERGE_UNIT, dtype=np.int64)
        entries.append(
            (
                int(group),
                SOURCE_REAL,
                int(region_lookup[group]),
                True,
                hidden_array[group].copy(),
                patches,
            )
        )

    for region in context_regions:
        remaining = [group for group in region.members if group not in selected_set]
        if not remaining:
            raise SelectorContractError(
                f"context region {region.region_id} has no unselected group; the "
                f"reservation invariant was violated"
            )
        weights = context_pooling_weights(scores.combined, remaining)
        context_tokens, valid = pool_region_context(
            hidden_array, remaining, scores.combined
        )
        if not valid:
            raise SelectorContractError(
                f"context region {region.region_id} produced an invalid context"
            )
        representative = region_context_representative(region, remaining, weights)
        representative_patches = representative * MERGE_UNIT + np.arange(
            MERGE_UNIT, dtype=np.int64
        )
        entries.append(
            (
                int(representative),
                SOURCE_CONTEXT,
                int(region.region_id),
                True,
                context_tokens,
                representative_patches,
            )
        )

    # Representative original_group_index values are globally unique, so the
    # ascending order alone is deterministic; source_type/region_id are kept as
    # a defensive tie-break for metadata.
    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))

    output_count = len(entries)
    if output_count != budget:
        raise SelectorContractError(
            f"internal error: emitted {output_count} groups for budget {budget}"
        )
    groups = np.stack([entry[4] for entry in entries], axis=0)
    group_indices = np.array([entry[0] for entry in entries], dtype=np.int64)
    patch_indices = np.stack([entry[5] for entry in entries], axis=0)
    source_type = np.array([entry[1] for entry in entries], dtype=np.uint8)
    region_id = np.array([entry[2] for entry in entries], dtype=np.int16)
    valid = np.array([entry[3] for entry in entries], dtype=bool)

    return HardwareSelection(
        boundary=boundary,
        budget=budget,
        bypass=False,
        groups=groups,
        group_indices=group_indices,
        patch_indices=patch_indices,
        source_type=source_type,
        region_id=region_id,
        valid=valid,
        scores=scores,
        real_group_count=int((source_type == SOURCE_REAL).sum()),
        context_group_count=int((source_type == SOURCE_CONTEXT).sum()),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        anchors=real.anchors,
        details=real.details,
        lambda_semantic=scores.lambda_semantic,
    )


# ---------------------------------------------------------------------------
# FPGA transfer layout description (never writes binary data)
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int = FPGA_LAYOUT_ALIGNMENT) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise SelectorContractError("align_up requires a non-negative integer")
    if not isinstance(alignment, (int, np.integer)) or int(alignment) <= 0:
        raise SelectorContractError("alignment must be a positive integer")
    value = int(value)
    alignment = int(alignment)
    return ((value + alignment - 1) // alignment) * alignment


def int8_token_bytes(budget: int) -> int:
    return validate_budget(budget) * MERGE_UNIT * FEATURE_DIM


def uint16_group_index_bytes(budget: int) -> int:
    return validate_budget(budget) * 2


def uint8_source_bytes(budget: int) -> int:
    return validate_budget(budget)


def uint8_region_bytes(budget: int) -> int:
    return validate_budget(budget)


def uint8_valid_bytes(budget: int) -> int:
    return validate_budget(budget)


_FPGA_SEGMENTS: tuple[tuple[str, str, int, Any], ...] = (
    ("tokens_int8", "int8", 1, int8_token_bytes),
    ("group_index_uint16", "uint16", 2, uint16_group_index_bytes),
    ("source_type_uint8", "uint8", 1, uint8_source_bytes),
    ("region_id_uint8", "uint8", 1, uint8_region_bytes),
    ("valid_uint8", "uint8", 1, uint8_valid_bytes),
)


@dataclass(frozen=True)
class FpgaLayoutSegment:
    name: str
    dtype: str
    element_bytes: int
    element_count: int
    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


@dataclass(frozen=True)
class FpgaTransferLayout:
    budget: int
    alignment: int
    segments: tuple[FpgaLayoutSegment, ...]
    total_bytes: int

    def offsets(self) -> dict[str, int]:
        return {segment.name: segment.offset for segment in self.segments}

    def sizes(self) -> dict[str, int]:
        return {segment.name: segment.size for segment in self.segments}


def fpga_transfer_layout(budget: int) -> FpgaTransferLayout:
    """Describe the 64-byte aligned FPGA transfer layout for one sample."""
    budget = validate_budget(budget)
    segments: list[FpgaLayoutSegment] = []
    offset = 0
    for name, dtype, element_bytes, size_fn in _FPGA_SEGMENTS:
        offset = align_up(offset, FPGA_LAYOUT_ALIGNMENT)
        size = int(size_fn(budget))
        if size % element_bytes != 0:
            raise SelectorContractError(
                f"segment {name} size {size} is not a multiple of {element_bytes}"
            )
        segments.append(
            FpgaLayoutSegment(
                name=name,
                dtype=dtype,
                element_bytes=element_bytes,
                element_count=size // element_bytes,
                offset=offset,
                size=size,
            )
        )
        offset = offset + size
    return FpgaTransferLayout(
        budget=budget,
        alignment=FPGA_LAYOUT_ALIGNMENT,
        segments=tuple(segments),
        total_bytes=align_up(offset, FPGA_LAYOUT_ALIGNMENT),
    )


def fpga_segment_offsets(budget: int) -> dict[str, int]:
    return fpga_transfer_layout(budget).offsets()


def fpga_segment_sizes(budget: int) -> dict[str, int]:
    return fpga_transfer_layout(budget).sizes()


def describe_fpga_layout(budget: int) -> dict[str, Any]:
    layout = fpga_transfer_layout(budget)
    return {
        "budget": layout.budget,
        "alignment": layout.alignment,
        "total_bytes": layout.total_bytes,
        "segments": [
            {
                "name": segment.name,
                "dtype": segment.dtype,
                "element_bytes": segment.element_bytes,
                "element_count": segment.element_count,
                "offset": segment.offset,
                "size": segment.size,
            }
            for segment in layout.segments
        ],
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / _CONFIG_FILENAME


def load_selector_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the frozen selector config."""
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = (
        "boundary",
        "budgets",
        "alignment",
        "context_slots_policy",
        "max_context_slots",
        "region_grid",
        "feature_dim",
        "prompt_dim",
        "score_dim",
        "merge_unit",
        "lambda_semantic",
        "local_candidate_cap",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise SelectorContractError(f"config is missing keys: {missing}")
    validate_boundary(config["boundary"])
    if int(config["alignment"]) != REQUIRED_BUDGET_ALIGNMENT:
        raise SelectorContractError("config alignment must be 16")
    if config["context_slots_policy"] != "dynamic_up_to_max":
        raise SelectorContractError(
            "config context_slots_policy must be 'dynamic_up_to_max'"
        )
    if int(config["max_context_slots"]) != MAX_CONTEXT_SLOTS:
        raise SelectorContractError(
            f"config max_context_slots must be {MAX_CONTEXT_SLOTS}"
        )
    if int(config["region_grid"]) != REGION_GRID:
        raise SelectorContractError("config region_grid must be 3")
    if int(config["feature_dim"]) != FEATURE_DIM:
        raise SelectorContractError("config feature_dim must be 1024")
    if int(config["prompt_dim"]) != PROMPT_DIM:
        raise SelectorContractError("config prompt_dim must be 2560")
    if int(config["score_dim"]) != SCORE_DIM:
        raise SelectorContractError("config score_dim must be 128")
    if int(config["merge_unit"]) != MERGE_UNIT:
        raise SelectorContractError("config merge_unit must be 4")
    for budget in config["budgets"]:
        validate_budget(budget)
    return config


__all__ = [
    "ALLOWED_BOUNDARIES",
    "BOUNDARY_ABLATION",
    "BOUNDARY_MAIN",
    "CONTEXT_SLOTS",
    "DEFAULT_LAMBDA_SEMANTIC",
    "DEFAULT_LOCAL_CANDIDATE_CAP",
    "FEATURE_DIM",
    "FPGA_LAYOUT_ALIGNMENT",
    "FpgaLayoutSegment",
    "FpgaTransferLayout",
    "GroupScores",
    "HardwareSelection",
    "InsufficientBudgetError",
    "LowRankProjection",
    "MAX_CONTEXT_SLOTS",
    "MERGE_UNIT",
    "PROMPT_DIM",
    "REGION_GRID",
    "REQUIRED_BUDGET_ALIGNMENT",
    "RealGroupSelection",
    "SCORE_DIM",
    "SOURCE_CONTEXT",
    "SOURCE_REAL",
    "SUPPORTED_BUDGETS",
    "SelectorContractError",
    "SpatialRegion",
    "align_up",
    "build_regions",
    "combine_scores",
    "context_pooling_weights",
    "default_config_path",
    "describe_fpga_layout",
    "fpga_segment_offsets",
    "fpga_segment_sizes",
    "fpga_transfer_layout",
    "group_l2_saliency",
    "group_representation",
    "int8_token_bytes",
    "load_selector_config",
    "minmax_normalize",
    "pool_region_context",
    "region_id_by_group",
    "region_context_representative",
    "region_representative",
    "score_groups",
    "scores_from_precomputed",
    "select_hardware_regular_groups",
    "select_real_groups",
    "semantic_cosine_scores",
    "uint16_group_index_bytes",
    "uint8_region_bytes",
    "uint8_source_bytes",
    "uint8_valid_bytes",
    "validate_boundary",
    "validate_budget",
    "validate_hidden",
]
