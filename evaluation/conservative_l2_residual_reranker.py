#!/usr/bin/env python3
"""Conservative corrected-L2 backbone + bounded integer task-residual reranker.

Mechanism
---------
``conservative_l2_residual_rerank_v1`` is a single additive selector family that
starts from the frozen corrected integer patch-L2 K96 real selection and applies
at most ``S = 16`` globally highest, strictly positive, non-conflicting
same-region one-for-one swaps driven only by a *signed integer task residual*:

    residual_q15[g] = semantic_norm_q15[g] - corrected_l2_norm_q15[g]

where ``semantic_norm_q15`` is the **already accepted** integer semantic chain
(seed-20261003 ``Wt`` / ``Wv_q`` Q0.15) and ``corrected_l2_norm_q15`` is the
**corrected integer patch-wise L2** Q0.15 vector.  The mechanism uses no labels,
answer tokens, gradients, floating ``lambda``, learned new weights or
task/source IDs.

Conservatism contract
---------------------
1. The baseline real selection is exactly the frozen
   :func:`evaluation.hardware_regular_token_selector.select_hardware_regular_groups`
   result on the corrected integer-L2 ``combined`` score, including one L2 anchor
   per non-empty 3x3 region and the existing dynamic context reservation.
2. Every L2 anchor is frozen and never replaced.
3. Every operation is a same-region one-for-one swap, so each region keeps its
   number of selected real groups and the reserved unselected context group can
   never be consumed.
4. Per region, unselected candidates are enumerated up to the existing
   ``local_candidate_cap = 16``; swaps are ranked by gain with deterministic
   lower-original-group-index tie breaking.
5. At most ``S = 16`` swaps are applied globally.  If fewer strictly positive
   legal swaps exist, fewer are applied.  ``K`` stays exactly 96 because every
   operation is one-for-one.
6. The 87 real + 9 context layout is rebuilt with the frozen layout primitives.
   The context pooling weights remain corrected integer-L2 based, so prompt
   information can only affect the bounded residual swaps.

The module is CPU-only (no CUDA, no model, no training) and only reuses accepted
quantisation / selection / pooling helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.hardware_regular_b1_corrected_l2 import (
    SCORE_Q_MAX,
    SCORE_Q_SCALE,
    array_sha256,
)
from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    MERGE_UNIT,
    REGION_GRID,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    GroupScores,
    HardwareSelection,
    build_regions,
    context_pooling_weights,
    pool_region_context,
    region_context_representative,
    region_id_by_group,
    select_hardware_regular_groups,
    validate_budget,
    validate_hidden,
)

#: Frozen mechanism identifier (also the raw-record ``arm`` value).
MECHANISM = "conservative_l2_residual_rerank_v1"
MECHANISM_FAMILY = "conservative_l2_residual_rerank"

#: The single fixed hardware work quantum.  This is one independent fixed
#: integer, never inferred from the frozen 24-sample answer-margin pilot and
#: deliberately unrelated to ``local_candidate_cap``.
MAX_SWAPS = 16
#: Existing per-region local candidate bound, reused verbatim.
LOCAL_CANDIDATE_CAP = 16
#: Existing B1 production budget.
BUDGET = 96
#: Exact signed residual definition.
RESIDUAL_FORMULA = "semantic_norm_q15 - corrected_l2_norm_q15"
#: Context pooling stays on this accepted term.
CONTEXT_POOLING_BASIS = "corrected_integer_l2_norm_q15"


class ResidualRerankError(ValueError):
    """A malformed input or a violated conservative-rerank contract."""


# ---------------------------------------------------------------------------
# Residual score
# ---------------------------------------------------------------------------
def residual_score_q15(semantic_norm_q15: Any, l2_norm_q15: Any) -> np.ndarray:
    """Signed integer task residual ``semantic_q15 - corrected_l2_q15``.

    Both inputs must be 1-D integer vectors in ``[0, 32767]`` of equal length.
    The result is a signed ``int64`` vector in ``[-32767, 32767]``.
    """

    semantic = np.asarray(semantic_norm_q15)
    l2 = np.asarray(l2_norm_q15)
    if semantic.ndim != 1 or l2.ndim != 1:
        raise ResidualRerankError(
            f"semantic_norm_q15 and l2_norm_q15 must be 1-D, got "
            f"{semantic.ndim}-D and {l2.ndim}-D"
        )
    if semantic.shape != l2.shape:
        raise ResidualRerankError(
            f"semantic_norm_q15 {semantic.shape} and l2_norm_q15 {l2.shape} "
            f"must share a shape"
        )
    if semantic.size == 0:
        raise ResidualRerankError("residual score needs a non-empty vector")
    if not np.issubdtype(semantic.dtype, np.integer) or not np.issubdtype(
        l2.dtype, np.integer
    ):
        raise ResidualRerankError("semantic_norm_q15 and l2_norm_q15 must be integer")
    for name, vector in (("semantic_norm_q15", semantic), ("l2_norm_q15", l2)):
        if int(vector.min()) < 0 or int(vector.max()) > SCORE_Q_MAX:
            raise ResidualRerankError(
                f"{name} must lie in [0, {SCORE_Q_MAX}], got "
                f"[{int(vector.min())}, {int(vector.max())}]"
            )
    return (semantic.astype(np.int64) - l2.astype(np.int64)).astype(np.int64)


def l2_combined_float(l2_norm_q15: Any) -> np.ndarray:
    """Dequantise the corrected integer-L2 Q0.15 vector for the frozen selector."""

    values = np.asarray(l2_norm_q15, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ResidualRerankError("l2_norm_q15 must be non-empty")
    return values.astype(np.float64) / float(SCORE_Q_SCALE)


# ---------------------------------------------------------------------------
# Swap plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResidualSwap:
    """One same-region one-for-one real-group swap."""

    region_id: int
    removed_group: int
    added_group: int
    residual_removed_q15: int
    residual_added_q15: int
    gain_q15: int

    def as_dict(self) -> dict[str, int]:
        return {
            "region_id": int(self.region_id),
            "removed_group": int(self.removed_group),
            "added_group": int(self.added_group),
            "residual_removed_q15": int(self.residual_removed_q15),
            "residual_added_q15": int(self.residual_added_q15),
            "gain_q15": int(self.gain_q15),
        }


@dataclass(frozen=True)
class ResidualRerankPlan:
    """The frozen baseline selection plus the applied residual swaps."""

    mechanism: str
    baseline_anchors: tuple[int, ...]
    baseline_details: tuple[int, ...]
    baseline_selected_real: tuple[int, ...]
    reranked_selected_real: tuple[int, ...]
    swaps: tuple[ResidualSwap, ...]
    region_ids: tuple[int, ...]
    candidate_cap: int
    max_swaps: int
    region_grid: int
    grid_h: int
    grid_w: int
    residual_q15: np.ndarray
    per_region_selected_counts_before: tuple[int, ...]
    per_region_selected_counts_after: tuple[int, ...]

    @property
    def swap_count(self) -> int:
        return len(self.swaps)

    def as_audit(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "residual_formula": RESIDUAL_FORMULA,
            "max_swaps": int(self.max_swaps),
            "local_candidate_cap": int(self.candidate_cap),
            "region_grid": int(self.region_grid),
            "swap_count": int(self.swap_count),
            "baseline_anchors": [int(value) for value in self.baseline_anchors],
            "baseline_details": [int(value) for value in self.baseline_details],
            "baseline_selected_real": [
                int(value) for value in self.baseline_selected_real
            ],
            "reranked_selected_real": [
                int(value) for value in self.reranked_selected_real
            ],
            "removed_group_indices": [
                int(swap.removed_group) for swap in self.swaps
            ],
            "added_group_indices": [int(swap.added_group) for swap in self.swaps],
            "swap_region_ids": [int(swap.region_id) for swap in self.swaps],
            "swap_residual_removed_q15": [
                int(swap.residual_removed_q15) for swap in self.swaps
            ],
            "swap_residual_added_q15": [
                int(swap.residual_added_q15) for swap in self.swaps
            ],
            "swap_gain_q15": [int(swap.gain_q15) for swap in self.swaps],
            "per_region_selected_counts_before": [
                int(value) for value in self.per_region_selected_counts_before
            ],
            "per_region_selected_counts_after": [
                int(value) for value in self.per_region_selected_counts_after
            ],
            "residual_q15": self.residual_q15,
        }


def _region_members(
    grid_h: int, grid_w: int, region_grid: int
) -> list[tuple[int, tuple[int, ...]]]:
    return [
        (region.region_id, region.members)
        for region in build_regions(grid_h, grid_w, region_grid)
        if region.members
    ]


def _candidate_ranking(
    unselected: Sequence[int], residual_q15: np.ndarray, candidate_cap: int
) -> list[int]:
    """Unselected candidates ranked by residual (desc), ties by lower index."""

    ranked = sorted(
        (int(group) for group in unselected),
        key=lambda group: (-int(residual_q15[group]), int(group)),
    )
    return ranked[: int(candidate_cap)]


def plan_residual_swaps(
    baseline: HardwareSelection,
    residual_q15: Any,
    *,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
) -> ResidualRerankPlan:
    """Plan the bounded conservative residual swaps over a baseline selection.

    The baseline is the frozen corrected integer-L2 K96 selection.  Anchors are
    frozen, candidates are bounded per region by ``local_candidate_cap`` and only
    strictly positive gains are eligible.  The globally highest positive,
    non-conflicting swaps are taken up to ``max_swaps``.
    """

    if baseline.bypass:
        raise ResidualRerankError(
            "the residual reranker is only defined for the compressed path; the "
            "G<=K identity bypass must be handled by the caller"
        )
    if not isinstance(local_candidate_cap, (int, np.integer)) or int(
        local_candidate_cap
    ) <= 0:
        raise ResidualRerankError("local_candidate_cap must be a positive integer")
    if not isinstance(max_swaps, (int, np.integer)) or int(max_swaps) <= 0:
        raise ResidualRerankError("max_swaps must be a positive integer")

    residual = np.asarray(residual_q15)
    if residual.ndim != 1:
        raise ResidualRerankError("residual_q15 must be 1-D")
    grid_h = int(baseline.grid_h)
    grid_w = int(baseline.grid_w)
    group_count = grid_h * grid_w
    if residual.size != group_count:
        raise ResidualRerankError(
            f"residual_q15 has {residual.size} entries but the selection grid is "
            f"{grid_h}x{grid_w}"
        )
    if not np.issubdtype(residual.dtype, np.integer):
        raise ResidualRerankError("residual_q15 must be an integer vector")

    anchors = tuple(sorted(int(value) for value in baseline.anchors))
    details = tuple(sorted(int(value) for value in baseline.details))
    baseline_selected_real = tuple(sorted(set(anchors) | set(details)))
    anchor_set = set(anchors)
    sorted_selected = set(baseline_selected_real)

    regions = _region_members(grid_h, grid_w, region_grid)
    selected_by_region: dict[int, list[int]] = {}
    unselected_by_region: dict[int, list[int]] = {}
    for region_id, members in regions:
        member_set = set(int(group) for group in members)
        selected_by_region[region_id] = sorted(member_set & sorted_selected)
        unselected_by_region[region_id] = sorted(member_set - sorted_selected)

    candidate_rankings: dict[int, list[int]] = {
        region_id: _candidate_ranking(
            unselected_by_region[region_id], np.asarray(residual), int(local_candidate_cap)
        )
        for region_id, _members in regions
    }

    eligible: list[tuple[int, int, int, int]] = []
    for region_id, _members in regions:
        selected_non_anchor = [
            group
            for group in selected_by_region[region_id]
            if group not in anchor_set
        ]
        for removed in selected_non_anchor:
            removed_residual = int(residual[removed])
            for added in candidate_rankings[region_id]:
                gain = int(residual[added]) - removed_residual
                if gain > 0:
                    eligible.append((gain, int(removed), int(added), int(region_id)))
    # Deterministic: highest gain, then lower original group index, then region.
    eligible.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    used_removed: set[int] = set()
    used_added: set[int] = set()
    swaps: list[ResidualSwap] = []
    for gain, removed, added, region_id in eligible:
        if removed in used_removed or added in used_added:
            continue
        used_removed.add(removed)
        used_added.add(added)
        swaps.append(
            ResidualSwap(
                region_id=region_id,
                removed_group=removed,
                added_group=added,
                residual_removed_q15=int(residual[removed]),
                residual_added_q15=int(residual[added]),
                gain_q15=int(gain),
            )
        )
        if len(swaps) >= int(max_swaps):
            break

    reranked_selected_real = tuple(
        sorted((set(baseline_selected_real) - used_removed) | used_added)
    )
    counts_before = tuple(
        len(selected_by_region[region_id]) for region_id, _members in regions
    )
    reranked_set = set(reranked_selected_real)
    counts_after = tuple(
        len(set(int(group) for group in members) & reranked_set)
        for _region_id, members in regions
    )
    return ResidualRerankPlan(
        mechanism=MECHANISM,
        baseline_anchors=anchors,
        baseline_details=details,
        baseline_selected_real=baseline_selected_real,
        reranked_selected_real=reranked_selected_real,
        swaps=tuple(swaps),
        region_ids=tuple(region_id for region_id, _members in regions),
        candidate_cap=int(local_candidate_cap),
        max_swaps=int(max_swaps),
        region_grid=int(region_grid),
        grid_h=grid_h,
        grid_w=grid_w,
        residual_q15=np.asarray(residual, dtype=np.int64),
        per_region_selected_counts_before=counts_before,
        per_region_selected_counts_after=counts_after,
    )


# ---------------------------------------------------------------------------
# Rebuild with the frozen layout contract
# ---------------------------------------------------------------------------
def rebuild_hardware_selection(
    hidden: Any,
    baseline: HardwareSelection,
    selected_real: Sequence[int],
    l2_combined: Any,
    *,
    region_grid: int = REGION_GRID,
) -> HardwareSelection:
    """Rebuild the 87 real + 9 context entries from an explicit real set.

    The entry construction, context reservation, pooling and ordering reuse the
    frozen :mod:`evaluation.hardware_regular_token_selector` primitives.  Context
    pooling weights come from ``l2_combined`` (the corrected integer-L2 vector),
    never from the residual.
    """

    hidden_array = validate_hidden(hidden)
    l2 = np.asarray(l2_combined, dtype=np.float64).reshape(-1)
    grid_h = int(baseline.grid_h)
    grid_w = int(baseline.grid_w)
    group_count = grid_h * grid_w
    if hidden_array.shape[0] != group_count:
        raise ResidualRerankError(
            f"hidden has {hidden_array.shape[0]} groups but the grid is "
            f"{grid_h}x{grid_w}"
        )
    if l2.size != group_count:
        raise ResidualRerankError(
            f"l2_combined has {l2.size} entries but the grid is {grid_h}x{grid_w}"
        )
    budget = int(baseline.budget)
    validate_budget(budget)

    selected = tuple(sorted(int(group) for group in selected_real))
    if len(set(selected)) != len(selected):
        raise ResidualRerankError("selected_real contains duplicate groups")
    if any(group < 0 or group >= group_count for group in selected):
        raise ResidualRerankError("selected_real contains an out-of-range group")
    anchor_set = set(int(value) for value in baseline.anchors)
    if not anchor_set.issubset(set(selected)):
        raise ResidualRerankError("a frozen L2 anchor is missing from selected_real")

    regions = build_regions(grid_h, grid_w, region_grid)
    non_empty = [region for region in regions if region.members]
    context_regions = [region for region in non_empty if len(region.members) >= 2]
    context_count = len(context_regions)
    real_budget = budget - context_count
    if len(selected) != real_budget:
        raise ResidualRerankError(
            f"selected_real has {len(selected)} groups but the layout requires "
            f"{real_budget} real groups (K={budget}, C={context_count})"
        )

    selected_set = set(selected)
    for region in context_regions:
        remaining_probe = [
            group for group in region.members if int(group) not in selected_set
        ]
        if not remaining_probe:
            raise ResidualRerankError(
                f"context region {region.region_id} would lose its reserved "
                f"unselected group"
            )

    region_lookup = region_id_by_group(grid_h, grid_w, region_grid)
    entries: list[tuple[int, int, int, bool, np.ndarray, np.ndarray]] = []
    for group in selected:
        patches = int(group) * MERGE_UNIT + np.arange(MERGE_UNIT, dtype=np.int64)
        entries.append(
            (
                int(group),
                SOURCE_REAL,
                int(region_lookup[int(group)]),
                True,
                hidden_array[int(group)].copy(),
                patches,
            )
        )
    for region in context_regions:
        remaining = [
            int(group) for group in region.members if int(group) not in selected_set
        ]
        weights = context_pooling_weights(l2, remaining)
        tokens, valid = pool_region_context(hidden_array, remaining, l2)
        if not valid:
            raise ResidualRerankError(
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
                tokens,
                representative_patches,
            )
        )
    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    if len(entries) != budget:
        raise ResidualRerankError(
            f"rebuilt layout emitted {len(entries)} groups for K={budget}"
        )
    group_indices = np.array([entry[0] for entry in entries], dtype=np.int64)
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise ResidualRerankError("rebuilt group_indices must be strictly ascending")
    groups = np.stack([entry[4] for entry in entries], axis=0)
    patch_indices = np.stack([entry[5] for entry in entries], axis=0)
    source_type = np.array([entry[1] for entry in entries], dtype=np.uint8)
    region_id = np.array([entry[2] for entry in entries], dtype=np.int16)
    valid = np.array([entry[3] for entry in entries], dtype=bool)
    scores = GroupScores(
        semantic=None, l2=l2, combined=l2, lambda_semantic=None
    )
    return HardwareSelection(
        boundary=baseline.boundary,
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
        grid_h=grid_h,
        grid_w=grid_w,
        anchors=baseline.anchors,
        details=tuple(
            int(group) for group in selected if int(group) not in anchor_set
        ),
        lambda_semantic=None,
    )


# ---------------------------------------------------------------------------
# Full selection entry point
# ---------------------------------------------------------------------------
def select_conservative_residual_rerank(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    semantic_norm_q15: Any,
    l2_norm_q15: Any,
    budget: int = BUDGET,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
) -> dict[str, Any]:
    """Baseline corrected integer-L2 K96 selection + bounded residual swaps.

    Returns a dict with ``selection`` (a :class:`HardwareSelection`), the
    baseline/reranked plan audit and the residual vectors.  ``G <= K`` returns the
    frozen identity bypass with ``swap_count == 0``.
    """

    hidden_array = validate_hidden(hidden)
    budget = validate_budget(budget)
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h * grid_w != hidden_array.shape[0]:
        raise ResidualRerankError(
            f"grid {grid_h}x{grid_w} does not match {hidden_array.shape[0]} groups"
        )
    residual = residual_score_q15(semantic_norm_q15, l2_norm_q15)
    if residual.size != hidden_array.shape[0]:
        raise ResidualRerankError(
            f"residual has {residual.size} entries but there are "
            f"{hidden_array.shape[0]} groups"
        )
    l2 = np.asarray(l2_norm_q15, dtype=np.int64).reshape(-1)
    combined = l2_combined_float(l2)

    baseline = select_hardware_regular_groups(
        hidden_array,
        grid_h,
        grid_w,
        budget=budget,
        combined_scores=combined,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
    )
    if baseline.bypass:
        return {
            "mechanism": MECHANISM,
            "selection": baseline,
            "bypass": True,
            "swap_count": 0,
            "baseline_plan": None,
            "semantic_norm_q15": np.asarray(semantic_norm_q15, dtype=np.int64),
            "l2_norm_q15": l2,
            "residual_q15": residual,
            "l2_combined": combined,
            "real_group_count": int(baseline.real_group_count),
            "context_group_count": int(baseline.context_group_count),
            "output_group_count": int(baseline.output_group_count),
            "context_pooling_basis": CONTEXT_POOLING_BASIS,
        }

    plan = plan_residual_swaps(
        baseline,
        residual,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        max_swaps=max_swaps,
    )
    _assert_plan_invariants(baseline, plan)
    selection = rebuild_hardware_selection(
        hidden_array,
        baseline,
        plan.reranked_selected_real,
        combined,
        region_grid=region_grid,
    )
    _assert_selection_invariants(baseline, selection, budget)

    audit = plan.as_audit()
    return {
        "mechanism": MECHANISM,
        "selection": selection,
        "bypass": False,
        "swap_count": int(plan.swap_count),
        "baseline_plan": audit,
        "semantic_norm_q15": np.asarray(semantic_norm_q15, dtype=np.int64),
        "l2_norm_q15": l2,
        "residual_q15": residual,
        "l2_combined": combined,
        "real_group_count": int(selection.real_group_count),
        "context_group_count": int(selection.context_group_count),
        "output_group_count": int(selection.output_group_count),
        "context_pooling_basis": CONTEXT_POOLING_BASIS,
    }


def _assert_plan_invariants(
    baseline: HardwareSelection, plan: ResidualRerankPlan
) -> None:
    anchor_set = set(int(value) for value in baseline.anchors)
    if not anchor_set.issubset(set(plan.reranked_selected_real)):
        raise ResidualRerankError("plan dropped a frozen anchor (fail closed)")
    if len(plan.reranked_selected_real) != len(plan.baseline_selected_real):
        raise ResidualRerankError(
            "plan changed the number of selected real groups (fail closed)"
        )
    if plan.swap_count > int(plan.max_swaps):
        raise ResidualRerankError("plan exceeded max_swaps (fail closed)")
    if plan.per_region_selected_counts_before != plan.per_region_selected_counts_after:
        raise ResidualRerankError(
            "plan changed a region's selected count (fail closed)"
        )
    removed = [swap.removed_group for swap in plan.swaps]
    added = [swap.added_group for swap in plan.swaps]
    if len(set(removed)) != len(removed) or len(set(added)) != len(added):
        raise ResidualRerankError("plan reuses a group across swaps (fail closed)")
    if set(removed) & set(added):
        raise ResidualRerankError("plan both removes and adds a group (fail closed)")
    for swap in plan.swaps:
        if swap.gain_q15 <= 0:
            raise ResidualRerankError("plan contains a non-positive swap (fail closed)")
        if swap.gain_q15 != swap.residual_added_q15 - swap.residual_removed_q15:
            raise ResidualRerankError("plan gain is inconsistent (fail closed)")


def _assert_selection_invariants(
    baseline: HardwareSelection, selection: HardwareSelection, budget: int
) -> None:
    if selection.bypass:
        raise ResidualRerankError("reranked selection must not be the identity bypass")
    if selection.output_group_count != budget:
        raise ResidualRerankError(
            f"reranked selection emitted {selection.output_group_count} groups for "
            f"K={budget} (fail closed)"
        )
    anchor_set = set(int(value) for value in baseline.anchors)
    if not anchor_set.issubset(set(int(value) for value in selection.group_indices)):
        raise ResidualRerankError("reranked selection dropped an anchor (fail closed)")
    real = int(selection.real_group_count)
    context = int(selection.context_group_count)
    if real + context != budget:
        raise ResidualRerankError("reranked real+context does not sum to K (fail closed)")
    if not np.all(selection.valid):
        raise ResidualRerankError("reranked selection has an invalid group (fail closed)")


# ---------------------------------------------------------------------------
# B1 ``integer_scorer``-shaped callback
# ---------------------------------------------------------------------------
class ConservativeResidualScorer:
    """B1 scorer callback computing the residual rerank payload for one image.

    Holds the frozen ``Wt`` / ``Wv_q`` bundle and calls the **unmodified**
    corrected-L2 arm-3 chain to obtain the accepted integer semantic Q0.15 and
    the corrected integer patch-L2 Q0.15, then applies the bounded residual
    rerank.  The returned payload contains ``selection`` (consumed by the new B1
    integration), ``combined_scores`` (the corrected integer-L2 vector, for
    record fidelity) and the full audit arrays.
    """

    def __init__(
        self,
        Wt: Any,
        wv_q: Any,
        *,
        budgets: Sequence[int] = (BUDGET,),
        region_grid: int = REGION_GRID,
        local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
        max_swaps: int = MAX_SWAPS,
    ) -> None:
        if Wt is None or wv_q is None:
            raise ResidualRerankError(
                "the residual reranker requires the frozen Wt and Wv_q bundle"
            )
        self.Wt = np.asarray(Wt)
        self.wv_q = np.asarray(wv_q, dtype=np.int64)
        self.budgets = tuple(int(value) for value in budgets)
        self.region_grid = int(region_grid)
        self.local_candidate_cap = int(local_candidate_cap)
        self.max_swaps = int(max_swaps)
        self.last_payload: dict[str, Any] | None = None

    def __call__(
        self,
        hidden_groups: Any,
        merged_h: int,
        merged_w: int,
        budget: int,
        text_embeddings: Any,
    ) -> dict[str, Any]:
        # Imported lazily so the module stays free of any heavy dependency.
        from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: PLC0415
            prompt_plus_corrected_l2_integer,
        )

        payload = prompt_plus_corrected_l2_integer(
            hidden_groups,
            text_embeddings,
            self.Wt,
            self.wv_q,
            merged_h,
            merged_w,
            budget=int(budget),
        )
        result = select_conservative_residual_rerank(
            hidden_groups,
            int(merged_h),
            int(merged_w),
            semantic_norm_q15=payload["semantic_norm_q15"],
            l2_norm_q15=payload["l2_norm_q15"],
            budget=int(budget),
            region_grid=self.region_grid,
            local_candidate_cap=self.local_candidate_cap,
            max_swaps=self.max_swaps,
        )
        result["integer_chain_payload"] = payload
        result["combined_scores"] = result["l2_combined"]
        result["budget"] = int(budget)
        result["mechanism"] = MECHANISM
        self.last_payload = result
        return result


def build_residual_scorer(
    Wt: Any,
    wv_q: Any,
    *,
    budgets: Sequence[int] = (BUDGET,),
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
) -> ConservativeResidualScorer:
    return ConservativeResidualScorer(
        Wt,
        wv_q,
        budgets=budgets,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        max_swaps=max_swaps,
    )


def audit_hashes(result: Mapping[str, Any]) -> dict[str, Any]:
    """Portable SHA256 digests of every residual-rerank audit array."""

    hashes: dict[str, Any] = {
        "semantic_norm_q15_sha256": array_sha256(
            np.asarray(result["semantic_norm_q15"], dtype=np.int64)
        ),
        "l2_norm_q15_sha256": array_sha256(
            np.asarray(result["l2_norm_q15"], dtype=np.int64)
        ),
        "residual_q15_sha256": array_sha256(
            np.asarray(result["residual_q15"], dtype=np.int64)
        ),
        "l2_combined_sha256": array_sha256(
            np.asarray(result["l2_combined"], dtype=np.float64)
        ),
        "reranked_selected_real_sha256": array_sha256(
            np.asarray(
                result["baseline_plan"]["reranked_selected_real"]
                if not result["bypass"]
                else [],
                dtype=np.int64,
            )
        ),
    }
    return hashes


__all__ = [
    "BUDGET",
    "CONTEXT_POOLING_BASIS",
    "ConservativeResidualScorer",
    "LOCAL_CANDIDATE_CAP",
    "MAX_SWAPS",
    "MECHANISM",
    "MECHANISM_FAMILY",
    "REGION_GRID",
    "RESIDUAL_FORMULA",
    "ResidualRerankError",
    "ResidualRerankPlan",
    "ResidualSwap",
    "audit_hashes",
    "build_residual_scorer",
    "l2_combined_float",
    "plan_residual_swaps",
    "rebuild_hardware_selection",
    "residual_score_q15",
    "select_conservative_residual_rerank",
]
