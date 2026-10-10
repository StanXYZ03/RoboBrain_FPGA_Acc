"""Bit-exact fixed-point golden for the hardware-regular token selector.

This module is the integer counterpart of ``hardware_regular_token_selector``.
It performs **selection only** (never scoring): the caller supplies an already
quantised score vector and the quantised merged-group tokens.

Frozen interface
----------------
* ``tokens_q``      : ``int8[G, 4, 1024]`` 2x2 merge groups, fully quantised.
* ``score_q``       : ``uint16`` Q0.15 combined score in ``[0, 32767]``.
* ``scale``         : one global symmetric scale shared by the whole image.
  It is **pass-through / register semantics** only and never participates in the
  selection or in the context pooling.
* ``1 <= grid_h, grid_w <= MAX_GRID_DIM`` (128) and ``grid_h * grid_w == G``
  with ``G <= 4096``.  A single dimension above ``MAX_GRID_DIM`` yields
  ``ERR_GRID``.  This bound is what makes the ``int64`` worst case of the
  weighted-centroid cross-multiplication provably overflow-free; see
  ``artifacts/token_selector/2026-09-27-selector-overflow-proof.json``.
* ``budget`` (``K``) is a positive multiple of 16 with ``K <= MAX_BUDGET = 96``;
  the production budgets are ``48/64/96``.  ``G <= K`` is an element-wise
  identity bypass.  The upper bound makes every HLS output/selected/entry loop
  statically sized, and the per-region candidate pool is at most
  ``NUM_REGIONS * MAX_LOCAL_CANDIDATE_CAP = 144``.

Selection maths (identical ordering to the float CPU selector)
--------------------------------------------------------------
1. 3x3 spatial regions.  Each non-empty region keeps exactly one real anchor:
   highest ``score_q``, ties to the lower ``original_group_index``.
2. Every region with ``>= 2`` input groups reserves at least one unselected
   group and therefore produces exactly one context group; ``C`` is dynamic.
3. ``real = K - C``.  Details come from a two-level Top-K: at most
   ``local_cap`` bounded local candidates per region (anchors excluded, one
   slot removed for context-eligible regions), then the highest scoring
   candidates over that bounded global pool.
4. Output representatives are globally unique and emitted by ascending
   representative index.

Integer arithmetic
------------------
* Context pooling uses all unselected groups of the region with the
  non-negative raw ``score_q`` weights; if every weight is zero the weights
  become uniform.  Accumulation is ``int64`` and the single division uses
  round-half-away-from-zero.
* The context representative is the unselected group closest to the weighted
  centroid of the unselected groups, compared with integer cross-multiplication
  of the squared distance; ties pick the lower index.
* Every intermediate is provably within signed 64-bit range for the frozen
  ``MAX_GRID_DIM = 128`` bound: the analytic worst case is
  ``distance <= 9.13e17 < 2**63-1`` and ``|accumulator| <= 2.06e9``.  See
  ``artifacts/token_selector/2026-09-27-selector-overflow-proof.json``; the
  Python golden uses arbitrary precision but never diverges from the C++ kernel.

Only new files are introduced by this contract; the existing float selector is
not modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

MERGE_UNIT = 4
FEATURE_DIM = 1024
GROUP_ELEMENTS = MERGE_UNIT * FEATURE_DIM
REGION_GRID = 3
NUM_REGIONS = REGION_GRID * REGION_GRID
LOCAL_CANDIDATE_CAP = 16
# Frozen compile-time cap for the per-region bounded insertion (candidate pool
# is at most NUM_REGIONS * MAX_LOCAL_CANDIDATE_CAP = 9 * 16 = 144).
MAX_LOCAL_CANDIDATE_CAP = 16
SUPPORTED_BUDGETS = (48, 64, 96)
REQUIRED_BUDGET_ALIGNMENT = 16
# Frozen compile-time budget bound: every output/selected/entry loop is sized
# by this value.  Runtime budgets stay positive multiples of 16 and <= 96 so the
# boundary fixtures (16/32/48/64/80/96) are preserved; production is 48/64/96.
MAX_BUDGET = 96
SCORE_Q_MAX = 32767
SCORE_Q_SCALE = 1 << 15  # Q0.15 denominator
MAX_GROUPS = 4096
MAX_GROUP_INDEX = 65535  # uint16 representative index guard
MAX_GRID_DIM = 128  # proven int64-safe per-dimension grid bound (see overflow proof)
FPGA_LAYOUT_ALIGNMENT = 64

SOURCE_REAL = 0
SOURCE_CONTEXT = 1

STATUS_OK = 0
STATUS_ERR_NULL = 1
STATUS_ERR_GROUP_LIMIT = 2
STATUS_ERR_GRID = 3
STATUS_ERR_BUDGET = 4
STATUS_ERR_INSUFFICIENT = 5
STATUS_ERR_CANDIDATE_POOL = 6
STATUS_ERR_INTERNAL = 7
STATUS_ERR_LOCAL_CAP = 8

STATUS_NAMES = {
    STATUS_OK: "OK",
    STATUS_ERR_NULL: "ERR_NULL",
    STATUS_ERR_GROUP_LIMIT: "ERR_GROUP_LIMIT",
    STATUS_ERR_GRID: "ERR_GRID",
    STATUS_ERR_BUDGET: "ERR_BUDGET",
    STATUS_ERR_INSUFFICIENT: "ERR_INSUFFICIENT",
    STATUS_ERR_CANDIDATE_POOL: "ERR_CANDIDATE_POOL",
    STATUS_ERR_INTERNAL: "ERR_INTERNAL",
    STATUS_ERR_LOCAL_CAP: "ERR_LOCAL_CAP",
}


class FixedpointContractError(ValueError):
    """Malformed *programmer* input (array shape/dtype), never a data status."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_tokens_q(tokens_q: Any) -> np.ndarray:
    """Return a contiguous ``int8[G, 4, 1024]`` array with ``G >= 1``."""
    array = np.asarray(tokens_q)
    if array.dtype != np.int8:
        try:
            array = array.astype(np.int8)
        except (TypeError, ValueError) as exc:  # pragma: no cover - numpy guard
            raise FixedpointContractError("tokens_q must be int8") from exc
    if array.ndim != 3 or array.shape[1] != MERGE_UNIT or array.shape[2] != FEATURE_DIM:
        raise FixedpointContractError(
            f"tokens_q must have shape [G, {MERGE_UNIT}, {FEATURE_DIM}], "
            f"got {array.shape}"
        )
    if array.shape[0] < 1:
        raise FixedpointContractError("tokens_q must contain at least one group")
    return np.ascontiguousarray(array, dtype=np.int8)


def validate_score_q(score_q: Any, expected_groups: int | None = None) -> np.ndarray:
    """Return a contiguous ``uint16[G]`` Q0.15 score vector in ``[0, 32767]``."""
    array = np.asarray(score_q)
    if array.dtype != np.uint16:
        array = array.astype(np.uint16)
    if array.ndim != 1:
        raise FixedpointContractError("score_q must be one-dimensional")
    if expected_groups is not None and array.shape[0] != expected_groups:
        raise FixedpointContractError(
            f"score_q has {array.shape[0]} entries but there are "
            f"{expected_groups} groups"
        )
    if array.size and int(array.max()) > SCORE_Q_MAX:
        raise FixedpointContractError(
            f"score_q values must be within [0, {SCORE_Q_MAX}]"
        )
    return np.ascontiguousarray(array, dtype=np.uint16)


def group_limit_ok(group_count: int) -> bool:
    """``True`` iff ``1 <= G <= 65535`` (the uint16 representative guard)."""
    return 1 <= int(group_count) <= MAX_GROUP_INDEX


def grid_dims_ok(grid_h: Any, grid_w: Any) -> bool:
    """``True`` iff both grid dimensions are within ``[1, MAX_GRID_DIM]``.

    Only the per-dimension bound is checked here; the ``grid_h * grid_w == G``
    and ``G <= MAX_GROUPS`` consistency checks stay in :func:`select_fixedpoint`
    so that the rejection status matches the C++ kernel.
    """
    if isinstance(grid_h, bool) or isinstance(grid_w, bool):
        return False
    if not isinstance(grid_h, (int, np.integer)) or not isinstance(
        grid_w, (int, np.integer)
    ):
        return False
    return 1 <= int(grid_h) <= MAX_GRID_DIM and 1 <= int(grid_w) <= MAX_GRID_DIM


def budget_status(budget: Any) -> int:
    """Validate the budget, mirroring the frozen ``TS_MAX_BUDGET`` contract.

    A budget is accepted iff it is a positive multiple of
    ``REQUIRED_BUDGET_ALIGNMENT`` (16) **and** ``<= MAX_BUDGET`` (96).  The
    upper bound is the compile-time bound used by the HLS kernel, so every
    output/selected/entry loop is statically sized.
    """
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        return STATUS_ERR_BUDGET
    value = int(budget)
    if value <= 0 or value % REQUIRED_BUDGET_ALIGNMENT != 0 or value > MAX_BUDGET:
        return STATUS_ERR_BUDGET
    return STATUS_OK


# ---------------------------------------------------------------------------
# 3x3 spatial regions
# ---------------------------------------------------------------------------


def _region_bounds(size: int) -> list[tuple[int, int]]:
    return [(r * size // REGION_GRID, (r + 1) * size // REGION_GRID) for r in range(REGION_GRID)]


@dataclass(frozen=True)
class FixedpointRegions:
    grid_h: int
    grid_w: int
    members: tuple[tuple[int, ...], ...]
    region_of_group: np.ndarray

    @property
    def non_empty(self) -> tuple[int, ...]:
        return tuple(r for r in range(NUM_REGIONS) if self.members[r])

    @property
    def context_regions(self) -> tuple[int, ...]:
        return tuple(r for r in self.non_empty if len(self.members[r]) >= 2)


def build_fixedpoint_regions(grid_h: int, grid_w: int) -> FixedpointRegions:
    """Partition the merged grid into the fixed 3x3 region layout."""
    if not isinstance(grid_h, (int, np.integer)) or not isinstance(
        grid_w, (int, np.integer)
    ):
        raise FixedpointContractError("grid_h and grid_w must be integers")
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h <= 0 or grid_w <= 0:
        raise FixedpointContractError("grid_h and grid_w must be positive")
    row_bounds = _region_bounds(grid_h)
    col_bounds = _region_bounds(grid_w)
    members: list[tuple[int, ...]] = []
    region_of = np.full((grid_h * grid_w,), -1, dtype=np.int32)
    for region_row in range(REGION_GRID):
        row_start, row_end = row_bounds[region_row]
        for region_col in range(REGION_GRID):
            col_start, col_end = col_bounds[region_col]
            region_id = region_row * REGION_GRID + region_col
            group_list: list[int] = []
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    group = row * grid_w + col
                    group_list.append(group)
                    region_of[group] = region_id
            members.append(tuple(group_list))
    return FixedpointRegions(
        grid_h=grid_h,
        grid_w=grid_w,
        members=tuple(members),
        region_of_group=region_of,
    )


# ---------------------------------------------------------------------------
# Integer context pooling helpers
# ---------------------------------------------------------------------------


def round_half_away_from_zero(numerator: int, denominator: int) -> int:
    """Exact integer division rounding half away from zero (``denominator > 0``)."""
    den = int(denominator)
    if den <= 0:
        raise FixedpointContractError("denominator must be positive")
    num = int(numerator)
    sign = -1 if num < 0 else 1
    magnitude = -num if num < 0 else num
    quotient = (2 * magnitude + den) // (2 * den)
    return sign * quotient


def context_pooling_weights_q(
    score_q: np.ndarray, remaining: Sequence[int]
) -> tuple[list[int], int]:
    """Return ``(weights, total)`` for the remaining unselected groups.

    Weights are the raw non-negative ``score_q`` values; when their total is
    zero the weights fall back to uniform (all ones).
    """
    indices = [int(group) for group in remaining]
    if not indices:
        return [], 0
    weights = [int(score_q[group]) for group in indices]
    total = int(sum(weights))
    if total == 0:
        weights = [1] * len(indices)
        total = len(indices)
    return weights, total


def pool_context_tokens_q(
    tokens_q: np.ndarray,
    score_q: np.ndarray,
    remaining: Sequence[int],
) -> np.ndarray:
    """Pool the remaining groups into an ``int8[4, 1024]`` context group."""
    indices = [int(group) for group in remaining]
    if not indices:
        raise FixedpointContractError("context pooling requires remaining groups")
    weights, total = context_pooling_weights_q(score_q, indices)
    accumulator = np.zeros((MERGE_UNIT, FEATURE_DIM), dtype=np.int64)
    for weight, group in zip(weights, indices):
        accumulator += weight * tokens_q[group].astype(np.int64)
    sign = np.where(accumulator < 0, -1, 1)
    quotient = (2 * np.abs(accumulator) + total) // (2 * total)
    pooled = sign * quotient
    return pooled.astype(np.int8)


def context_representative_q(
    score_q: np.ndarray,
    remaining: Sequence[int],
    grid_w: int,
) -> int:
    """Unselected group nearest the weighted centroid, ties to lower index."""
    indices = [int(group) for group in remaining]
    if not indices:
        return -1
    weights, total = context_pooling_weights_q(score_q, indices)
    row_sum = 0
    col_sum = 0
    rows: list[int] = []
    cols: list[int] = []
    for weight, group in zip(weights, indices):
        row = group // grid_w
        col = group % grid_w
        rows.append(row)
        cols.append(col)
        row_sum += weight * row
        col_sum += weight * col
    best_key: tuple[int, int] | None = None
    best_group = -1
    for position, group in enumerate(indices):
        delta_row = rows[position] * total - row_sum
        delta_col = cols[position] * total - col_sum
        distance = delta_row * delta_row + delta_col * delta_col
        key = (distance, group)
        if best_key is None or key < best_key:
            best_key = key
            best_group = group
    return int(best_group)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedpointSelection:
    status: int
    output_count: int
    real_count: int
    context_count: int
    tokens_out: np.ndarray
    group_index: np.ndarray
    source_type: np.ndarray
    region_id: np.ndarray
    valid: np.ndarray
    scale_out: float
    grid_h: int
    grid_w: int
    budget: int
    anchors: tuple[int, ...] = ()
    details: tuple[int, ...] = ()
    context_region_ids: tuple[int, ...] = ()
    status_name: str = field(default="OK")

    @property
    def output_group_count(self) -> int:
        return int(self.output_count)


def _error_selection(
    status: int, scale: float, grid_h: int, grid_w: int, budget: int
) -> FixedpointSelection:
    return FixedpointSelection(
        status=status,
        output_count=0,
        real_count=0,
        context_count=0,
        tokens_out=np.zeros((0, MERGE_UNIT, FEATURE_DIM), dtype=np.int8),
        group_index=np.zeros((0,), dtype=np.uint16),
        source_type=np.zeros((0,), dtype=np.uint8),
        region_id=np.zeros((0,), dtype=np.uint8),
        valid=np.zeros((0,), dtype=np.uint8),
        scale_out=float(scale),
        grid_h=int(grid_h) if isinstance(grid_h, (int, np.integer)) else 0,
        grid_w=int(grid_w) if isinstance(grid_w, (int, np.integer)) else 0,
        budget=int(budget) if isinstance(budget, (int, np.integer)) else 0,
        status_name=STATUS_NAMES.get(status, "UNKNOWN"),
    )


def select_fixedpoint(
    tokens_q: Any,
    score_q: Any,
    grid_h: int,
    grid_w: int,
    budget: int,
    *,
    local_cap: int = LOCAL_CANDIDATE_CAP,
    scale: float = 1.0,
) -> FixedpointSelection:
    """Bit-exact fixed-point selection.

    Array *shape/dtype* problems raise ``FixedpointContractError`` because they
    are programmer errors.  Contract/data problems (group limit, grid, budget,
    feasibility) are reported through ``FixedpointSelection.status`` so that the
    C++ kernel and the Python golden can be cross-checked on rejection paths.
    """
    tokens = validate_tokens_q(tokens_q)
    group_count = int(tokens.shape[0])
    scores = validate_score_q(score_q, group_count)

    if group_count > MAX_GROUP_INDEX or group_count > MAX_GROUPS:
        return _error_selection(STATUS_ERR_GROUP_LIMIT, scale, grid_h, grid_w, budget)

    if (
        not isinstance(grid_h, (int, np.integer))
        or not isinstance(grid_w, (int, np.integer))
        or int(grid_h) <= 0
        or int(grid_w) <= 0
        or int(grid_h) > MAX_GRID_DIM
        or int(grid_w) > MAX_GRID_DIM
        or int(grid_h) * int(grid_w) != group_count
    ):
        return _error_selection(STATUS_ERR_GRID, scale, grid_h, grid_w, budget)

    if budget_status(budget) != STATUS_OK:
        return _error_selection(STATUS_ERR_BUDGET, scale, grid_h, grid_w, budget)
    budget = int(budget)

    if (
        not isinstance(local_cap, (int, np.integer))
        or int(local_cap) <= 0
        or int(local_cap) > MAX_LOCAL_CANDIDATE_CAP
    ):
        return _error_selection(STATUS_ERR_LOCAL_CAP, scale, grid_h, grid_w, budget)
    local_cap = int(local_cap)
    grid_h = int(grid_h)
    grid_w = int(grid_w)

    regions = build_fixedpoint_regions(grid_h, grid_w)

    if group_count <= budget:
        return _identity_selection(tokens, regions, budget, scale)

    non_empty = regions.non_empty
    context_regions = regions.context_regions
    context_count = len(context_regions)
    if budget - context_count < len(non_empty):
        return _error_selection(
            STATUS_ERR_INSUFFICIENT, scale, grid_h, grid_w, budget
        )

    real_budget = budget - context_count
    anchor_by_region: dict[int, int] = {}
    for region_id in non_empty:
        region_members = regions.members[region_id]
        best_group = region_members[0]
        best_key = (-int(scores[best_group]), best_group)
        for group in region_members[1:]:
            key = (-int(scores[group]), group)
            if key < best_key:
                best_key = key
                best_group = group
        anchor_by_region[region_id] = int(best_group)
    anchors = tuple(anchor_by_region[region_id] for region_id in non_empty)
    anchor_set = set(anchors)

    reserved_ids = set(context_regions)
    candidates: set[int] = set()
    for region_id in non_empty:
        anchor = anchor_by_region[region_id]
        region_members = regions.members[region_id]
        ranked = sorted(
            (group for group in region_members if group != anchor),
            key=lambda group: (-int(scores[group]), group),
        )
        reserve = 1 if region_id in reserved_ids else 0
        allowed = max(0, len(region_members) - 1 - reserve)
        take = min(len(ranked), local_cap, allowed)
        candidates.update(int(group) for group in ranked[:take])

    pool = sorted(candidates - anchor_set, key=lambda group: (-int(scores[group]), group))
    details_needed = real_budget - len(anchors)
    if len(pool) < details_needed:
        return _error_selection(
            STATUS_ERR_CANDIDATE_POOL, scale, grid_h, grid_w, budget
        )
    details = tuple(int(group) for group in pool[:details_needed])
    selected = tuple(sorted(anchors + details))
    selected_set = set(selected)

    entries: list[tuple[int, int, int, np.ndarray]] = []
    for group in selected:
        entries.append(
            (
                int(group),
                SOURCE_REAL,
                int(regions.region_of_group[group]),
                np.ascontiguousarray(tokens[group].copy()),
            )
        )
    for region_id in context_regions:
        remaining = [
            group for group in regions.members[region_id] if group not in selected_set
        ]
        if not remaining:
            return _error_selection(
                STATUS_ERR_INTERNAL, scale, grid_h, grid_w, budget
            )
        context_tokens = pool_context_tokens_q(tokens, scores, remaining)
        representative = context_representative_q(scores, remaining, grid_w)
        entries.append(
            (
                int(representative),
                SOURCE_CONTEXT,
                int(region_id),
                np.ascontiguousarray(context_tokens),
            )
        )

    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    output_count = len(entries)
    if output_count != budget:
        return _error_selection(STATUS_ERR_INTERNAL, scale, grid_h, grid_w, budget)

    tokens_out = np.stack([entry[3] for entry in entries], axis=0).astype(np.int8)
    group_index = np.array([entry[0] for entry in entries], dtype=np.uint16)
    source_type = np.array([entry[1] for entry in entries], dtype=np.uint8)
    region_id_out = np.array([entry[2] for entry in entries], dtype=np.uint8)
    valid = np.ones((output_count,), dtype=np.uint8)
    return FixedpointSelection(
        status=STATUS_OK,
        output_count=output_count,
        real_count=int((source_type == SOURCE_REAL).sum()),
        context_count=int((source_type == SOURCE_CONTEXT).sum()),
        tokens_out=tokens_out,
        group_index=group_index,
        source_type=source_type,
        region_id=region_id_out,
        valid=valid,
        scale_out=float(scale),
        grid_h=grid_h,
        grid_w=grid_w,
        budget=budget,
        anchors=anchors,
        details=details,
        context_region_ids=context_regions,
        status_name="OK",
    )


def _identity_selection(
    tokens: np.ndarray,
    regions: FixedpointRegions,
    budget: int,
    scale: float,
) -> FixedpointSelection:
    group_count = int(tokens.shape[0])
    group_index = np.arange(group_count, dtype=np.uint16)
    return FixedpointSelection(
        status=STATUS_OK,
        output_count=group_count,
        real_count=group_count,
        context_count=0,
        tokens_out=np.ascontiguousarray(tokens.copy()),
        group_index=group_index,
        source_type=np.zeros((group_count,), dtype=np.uint8),
        region_id=np.asarray(regions.region_of_group, dtype=np.uint8),
        valid=np.ones((group_count,), dtype=np.uint8),
        scale_out=float(scale),
        grid_h=regions.grid_h,
        grid_w=regions.grid_w,
        budget=int(budget),
        anchors=(),
        details=(),
        context_region_ids=(),
        status_name="OK",
    )


# ---------------------------------------------------------------------------
# FPGA transfer layout (description only; never writes binary data)
# ---------------------------------------------------------------------------


def align_up(value: int, alignment: int = FPGA_LAYOUT_ALIGNMENT) -> int:
    if not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise FixedpointContractError("align_up requires a non-negative integer")
    if not isinstance(alignment, (int, np.integer)) or int(alignment) <= 0:
        raise FixedpointContractError("alignment must be a positive integer")
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _require_budget(budget: Any) -> int:
    if budget_status(budget) != STATUS_OK:
        raise FixedpointContractError(
            f"budget must be a positive multiple of {REQUIRED_BUDGET_ALIGNMENT} "
            f"and <= {MAX_BUDGET}"
        )
    return int(budget)


def int8_token_bytes(budget: int) -> int:
    return _require_budget(budget) * GROUP_ELEMENTS


def uint16_group_index_bytes(budget: int) -> int:
    return _require_budget(budget) * 2


def uint8_source_bytes(budget: int) -> int:
    return _require_budget(budget)


def uint8_region_bytes(budget: int) -> int:
    return _require_budget(budget)


def uint8_valid_bytes(budget: int) -> int:
    return _require_budget(budget)


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
    """Describe the 64-byte aligned 5-segment transfer layout for ``budget``."""
    budget = _require_budget(budget)
    segments: list[FpgaLayoutSegment] = []
    offset = 0
    for name, dtype, element_bytes, size_fn in _FPGA_SEGMENTS:
        offset = align_up(offset)
        size = int(size_fn(budget))
        if size % element_bytes != 0:
            raise FixedpointContractError(
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
        total_bytes=align_up(offset),
    )


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


def spec_payload() -> dict[str, Any]:
    """Machine-readable description of this fixed-point contract."""
    return {
        "schema_version": "1.0",
        "date": "2026-09-27",
        "scope": "bit-exact fixed-point hardware-regular token selector (selection only)",
        "module": "evaluation/hardware_regular_selector_fixedpoint.py",
        "constants": {
            "merge_unit": MERGE_UNIT,
            "feature_dim": FEATURE_DIM,
            "group_elements": GROUP_ELEMENTS,
            "region_grid": REGION_GRID,
            "num_regions": NUM_REGIONS,
            "local_candidate_cap": LOCAL_CANDIDATE_CAP,
            "max_local_candidate_cap": MAX_LOCAL_CANDIDATE_CAP,
            "supported_budgets": list(SUPPORTED_BUDGETS),
            "budget_alignment": REQUIRED_BUDGET_ALIGNMENT,
            "max_budget": MAX_BUDGET,
            "score_q_max": SCORE_Q_MAX,
            "score_q_scale": SCORE_Q_SCALE,
            "max_groups": MAX_GROUPS,
            "max_group_index": MAX_GROUP_INDEX,
            "max_grid_dim": MAX_GRID_DIM,
            "fpga_layout_alignment": FPGA_LAYOUT_ALIGNMENT,
        },
        "input_contract": {
            "tokens_q": "int8[G, 4, 1024]",
            "score_q": "uint16 Q0.15 combined score in [0, 32767]",
            "scale": "global symmetric scale, pass-through/register semantics only",
            "grid": "1 <= grid_h, grid_w <= 128, grid_h * grid_w == G, G <= 4096; single dim > 128 -> ERR_GRID",
            "budget": "positive multiple of 16; production 48/64/96; G <= K identity",
            "local_cap": LOCAL_CANDIDATE_CAP,
        },
        "integer_rules": {
            "pooling_weights": "raw non-negative score_q; uniform all-ones when the sum is zero",
            "pooling_accumulator": "int64",
            "pooling_rounding": "round-half-away-from-zero",
            "representative": "integer cross-multiplication of squared distance to the weighted centroid; ties lower index",
        },
        "status_codes": STATUS_NAMES,
        "output_segments": [
            {"name": name, "dtype": dtype}
            for name, dtype, _element_bytes, _fn in _FPGA_SEGMENTS
        ],
        "fpga_layout": {
            "alignment": FPGA_LAYOUT_ALIGNMENT,
            "layouts": [describe_fpga_layout(budget) for budget in SUPPORTED_BUDGETS],
        },
    }


def write_spec(path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(spec_payload(), handle, indent=2, sort_keys=False)
        handle.write("\n")


__all__ = [
    "FEATURE_DIM",
    "FPGA_LAYOUT_ALIGNMENT",
    "FixedpointContractError",
    "FixedpointRegions",
    "FixedpointSelection",
    "FpgaLayoutSegment",
    "FpgaTransferLayout",
    "GROUP_ELEMENTS",
    "LOCAL_CANDIDATE_CAP",
    "MAX_BUDGET",
    "MAX_GRID_DIM",
    "MAX_GROUPS",
    "MAX_GROUP_INDEX",
    "MAX_LOCAL_CANDIDATE_CAP",
    "MERGE_UNIT",
    "NUM_REGIONS",
    "REGION_GRID",
    "REQUIRED_BUDGET_ALIGNMENT",
    "SCORE_Q_MAX",
    "SCORE_Q_SCALE",
    "SOURCE_CONTEXT",
    "SOURCE_REAL",
    "STATUS_ERR_BUDGET",
    "STATUS_ERR_CANDIDATE_POOL",
    "STATUS_ERR_GRID",
    "STATUS_ERR_GROUP_LIMIT",
    "STATUS_ERR_INSUFFICIENT",
    "STATUS_ERR_INTERNAL",
    "STATUS_ERR_LOCAL_CAP",
    "STATUS_ERR_NULL",
    "STATUS_NAMES",
    "STATUS_OK",
    "SUPPORTED_BUDGETS",
    "align_up",
    "budget_status",
    "build_fixedpoint_regions",
    "context_pooling_weights_q",
    "context_representative_q",
    "describe_fpga_layout",
    "fpga_transfer_layout",
    "group_limit_ok",
    "grid_dims_ok",
    "int8_token_bytes",
    "pool_context_tokens_q",
    "round_half_away_from_zero",
    "select_fixedpoint",
    "spec_payload",
    "uint16_group_index_bytes",
    "uint8_region_bytes",
    "uint8_source_bytes",
    "uint8_valid_bytes",
    "validate_score_q",
    "validate_tokens_q",
    "write_spec",
]
