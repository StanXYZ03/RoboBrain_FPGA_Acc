#!/usr/bin/env python3
"""Conservative corrected-L2 backbone + L2-guarded integer reranker (v2).

This is the **additive, sign-corrected** successor of
``conservative_l2_residual_rerank_v1``.  A pre-execution review proved that the
preregistered v1 ranking quantity

    (semantic_q15[j] - l2_q15[j]) - (semantic_q15[i] - l2_q15[i])

has the *wrong sign* for an L2-protecting mechanism: it rewards an L2 loss.
For ``semantic_i = semantic_j`` and a severely smaller candidate L2 the v1 gain
is strongly positive, so v1 would accept a candidate with no semantic gain and a
large corrected-L2 regression.  v2 stops that defect before any GPU run; the v1
code, config and reports are left byte-identical.

Mechanism
---------
``conservative_l2_guarded_rerank_v2`` starts from the frozen corrected integer
patch-L2 K96 real selection and applies at most ``S = 16`` globally highest,
strictly positive, non-conflicting same-region one-for-one swaps driven by a
widened integer utility:

    utility_q16[g] = semantic_q15[g] + l2_q15[g]        (0..65534, int64)

For selected non-anchor ``i`` and same-region unselected candidate ``j``:

    semantic_gain = semantic_q15[j] - semantic_q15[i]
    l2_loss       = l2_q15[i]       - l2_q15[j]
    legal_gain    = semantic_gain - l2_loss
                  = utility_q16[j] - utility_q16[i]

Only ``legal_gain > 0`` is eligible.  A candidate is admitted when it improves
the accepted integer semantic score by *more* than the corrected integer-L2 mass
it gives up, so the corrected-L2 anchors and pooling are protected by
construction.

Contract
--------
1. The baseline real selection is exactly the frozen corrected integer-L2 K96
   path (one L2 anchor per non-empty 3x3 region + the existing dynamic context
   reservation).
2. Every L2 anchor is frozen and never replaced.
3. Every operation is a same-region one-for-one swap, so each region keeps its
   number of selected real groups and the reserved unselected context group can
   never be consumed.
4. Per region, unselected candidates are ranked by ``utility_q16`` descending
   (lower original group index first) and capped by the existing
   ``local_candidate_cap = 16``.
5. Globally at most ``S = 16`` swaps, ranked by ``legal_gain`` descending then
   ``(added_group_id, removed_group_id, region_id)`` ascending, are applied
   nonconflictingly.  ``K`` stays exactly 96.
6. The 87 real + 9 context layout is rebuilt with the frozen layout primitives.
   Context pooling weights remain corrected integer-L2 based, so prompt
   information can only affect the bounded legal swaps.

No labels, answer tokens, gradients, floating lambda, learned new weights or
task/source IDs are used.  The module is CPU-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.conservative_l2_residual_reranker import (
    l2_combined_float,
    rebuild_hardware_selection,
)
from evaluation.hardware_regular_b1_corrected_l2 import (
    SCORE_Q_MAX,
    SCORE_Q_SCALE,
    array_sha256,
)
from evaluation.hardware_regular_token_selector import (
    MERGE_UNIT,
    REGION_GRID,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    HardwareSelection,
    build_regions,
    region_id_by_group,
    select_hardware_regular_groups,
    validate_budget,
    validate_hidden,
)

#: Frozen mechanism identifier (also the raw-record ``arm`` value).
MECHANISM = "conservative_l2_guarded_rerank_v2"
MECHANISM_FAMILY = "conservative_l2_guarded_rerank"

#: Widened integer utility bound: ``semantic_q15 + l2_q15 <= 2 * 32767``.
UTILITY_MAX = 2 * SCORE_Q_MAX  # 65534

#: One fixed hardware work quantum, unrelated to ``local_candidate_cap``.
MAX_SWAPS = 16
#: Existing per-region local candidate bound, reused verbatim.
LOCAL_CANDIDATE_CAP = 16
#: Existing B1 production budget.
BUDGET = 96

#: Exact integer definitions recorded in every audit.
UTILITY_FORMULA = "semantic_q15 + corrected_l2_q15"
LEGAL_GAIN_FORMULA = "semantic_gain - l2_loss = utility_added - utility_removed"
#: Context pooling stays on the accepted corrected integer-L2 term.
CONTEXT_POOLING_BASIS = "corrected_integer_l2_norm_q15"
#: Name of the v2 candidate ordering key.
CANDIDATE_ORDER = "utility_q16_desc_then_lower_original_group_index"


class GuardedRerankError(ValueError):
    """A malformed input or a violated L2-guarded rerank contract."""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def utility_q16(semantic_norm_q15: Any, l2_norm_q15: Any) -> np.ndarray:
    """Widened integer utility ``semantic_q15 + corrected_l2_q15``.

    Both inputs must be 1-D integer vectors in ``[0, 32767]`` of equal length.
    The result is an ``int64`` vector in ``[0, 65534]``.
    """

    semantic = np.asarray(semantic_norm_q15)
    l2 = np.asarray(l2_norm_q15)
    if semantic.ndim != 1 or l2.ndim != 1:
        raise GuardedRerankError(
            f"semantic_norm_q15 and l2_norm_q15 must be 1-D, got "
            f"{semantic.ndim}-D and {l2.ndim}-D"
        )
    if semantic.shape != l2.shape:
        raise GuardedRerankError(
            f"semantic_norm_q15 {semantic.shape} and l2_norm_q15 {l2.shape} "
            f"must share a shape"
        )
    if semantic.size == 0:
        raise GuardedRerankError("utility score needs a non-empty vector")
    if not np.issubdtype(semantic.dtype, np.integer) or not np.issubdtype(
        l2.dtype, np.integer
    ):
        raise GuardedRerankError(
            "semantic_norm_q15 and l2_norm_q15 must be integer"
        )
    for name, vector in (("semantic_norm_q15", semantic), ("l2_norm_q15", l2)):
        if int(vector.min()) < 0 or int(vector.max()) > SCORE_Q_MAX:
            raise GuardedRerankError(
                f"{name} must lie in [0, {SCORE_Q_MAX}], got "
                f"[{int(vector.min())}, {int(vector.max())}]"
            )
    widened = semantic.astype(np.int64) + l2.astype(np.int64)
    if int(widened.max()) > UTILITY_MAX:
        raise GuardedRerankError(
            f"utility_q16 exceeds the widened bound {UTILITY_MAX}"
        )
    return widened


# ---------------------------------------------------------------------------
# Swap plan
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardedSwap:
    """One same-region one-for-one real-group swap with the full L2 audit."""

    region_id: int
    removed_group: int
    added_group: int
    semantic_removed_q15: int
    semantic_added_q15: int
    l2_removed_q15: int
    l2_added_q15: int
    utility_removed_q16: int
    utility_added_q16: int
    semantic_gain_q15: int
    l2_loss_q15: int
    legal_gain_q16: int

    def as_dict(self) -> dict[str, int]:
        return {
            "region_id": int(self.region_id),
            "removed_group": int(self.removed_group),
            "added_group": int(self.added_group),
            "semantic_removed_q15": int(self.semantic_removed_q15),
            "semantic_added_q15": int(self.semantic_added_q15),
            "l2_removed_q15": int(self.l2_removed_q15),
            "l2_added_q15": int(self.l2_added_q15),
            "utility_removed_q16": int(self.utility_removed_q16),
            "utility_added_q16": int(self.utility_added_q16),
            "semantic_gain_q15": int(self.semantic_gain_q15),
            "l2_loss_q15": int(self.l2_loss_q15),
            "legal_gain_q16": int(self.legal_gain_q16),
        }


@dataclass(frozen=True)
class GuardedRerankPlan:
    """The frozen corrected-L2 baseline plus the applied L2-guarded swaps."""

    mechanism: str
    baseline_anchors: tuple[int, ...]
    baseline_details: tuple[int, ...]
    baseline_selected_real: tuple[int, ...]
    reranked_selected_real: tuple[int, ...]
    swaps: tuple[GuardedSwap, ...]
    region_ids: tuple[int, ...]
    candidate_cap: int
    max_swaps: int
    region_grid: int
    grid_h: int
    grid_w: int
    utility_q16: np.ndarray
    per_region_selected_counts_before: tuple[int, ...]
    per_region_selected_counts_after: tuple[int, ...]

    @property
    def swap_count(self) -> int:
        return len(self.swaps)

    def as_audit(self) -> dict[str, Any]:
        return {
            "mechanism": self.mechanism,
            "utility_formula": UTILITY_FORMULA,
            "legal_gain_formula": LEGAL_GAIN_FORMULA,
            "candidate_order": CANDIDATE_ORDER,
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
            "swap_semantic_gain_q15": [
                int(swap.semantic_gain_q15) for swap in self.swaps
            ],
            "swap_l2_loss_q15": [int(swap.l2_loss_q15) for swap in self.swaps],
            "swap_legal_gain_q16": [
                int(swap.legal_gain_q16) for swap in self.swaps
            ],
            "swap_utility_removed_q16": [
                int(swap.utility_removed_q16) for swap in self.swaps
            ],
            "swap_utility_added_q16": [
                int(swap.utility_added_q16) for swap in self.swaps
            ],
            "per_region_selected_counts_before": [
                int(value) for value in self.per_region_selected_counts_before
            ],
            "per_region_selected_counts_after": [
                int(value) for value in self.per_region_selected_counts_after
            ],
            "utility_q16": self.utility_q16,
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
    unselected: Sequence[int], utility: np.ndarray, candidate_cap: int
) -> list[int]:
    """Unselected candidates ranked by utility (desc), ties by lower index."""

    ranked = sorted(
        (int(group) for group in unselected),
        key=lambda group: (-int(utility[group]), int(group)),
    )
    return ranked[: int(candidate_cap)]


def _swap_from_pair(
    region_id: int,
    removed: int,
    added: int,
    semantic: np.ndarray,
    l2: np.ndarray,
    utility: np.ndarray,
) -> GuardedSwap:
    semantic_gain = int(semantic[added]) - int(semantic[removed])
    l2_loss = int(l2[removed]) - int(l2[added])
    legal_gain = int(utility[added]) - int(utility[removed])
    if legal_gain != semantic_gain - l2_loss:
        raise GuardedRerankError(
            "internal error: legal_gain != semantic_gain - l2_loss (fail closed)"
        )
    return GuardedSwap(
        region_id=int(region_id),
        removed_group=int(removed),
        added_group=int(added),
        semantic_removed_q15=int(semantic[removed]),
        semantic_added_q15=int(semantic[added]),
        l2_removed_q15=int(l2[removed]),
        l2_added_q15=int(l2[added]),
        utility_removed_q16=int(utility[removed]),
        utility_added_q16=int(utility[added]),
        semantic_gain_q15=int(semantic_gain),
        l2_loss_q15=int(l2_loss),
        legal_gain_q16=int(legal_gain),
    )


def plan_guarded_swaps(
    baseline: HardwareSelection,
    semantic_norm_q15: Any,
    l2_norm_q15: Any,
    *,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
) -> GuardedRerankPlan:
    """Plan the bounded L2-guarded swaps over a corrected integer-L2 baseline."""

    if baseline.bypass:
        raise GuardedRerankError(
            "the guarded reranker is only defined for the compressed path; the "
            "G<=K identity bypass must be handled by the caller"
        )
    if not isinstance(local_candidate_cap, (int, np.integer)) or int(
        local_candidate_cap
    ) <= 0:
        raise GuardedRerankError("local_candidate_cap must be a positive integer")
    if not isinstance(max_swaps, (int, np.integer)) or int(max_swaps) <= 0:
        raise GuardedRerankError("max_swaps must be a positive integer")

    semantic = np.asarray(semantic_norm_q15)
    l2 = np.asarray(l2_norm_q15)
    grid_h = int(baseline.grid_h)
    grid_w = int(baseline.grid_w)
    group_count = grid_h * grid_w
    utility = utility_q16(semantic, l2)
    if utility.size != group_count:
        raise GuardedRerankError(
            f"utility_q16 has {utility.size} entries but the selection grid is "
            f"{grid_h}x{grid_w}"
        )

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
            unselected_by_region[region_id], utility, int(local_candidate_cap)
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
            for added in candidate_rankings[region_id]:
                legal_gain = int(utility[added]) - int(utility[removed])
                if legal_gain > 0:
                    eligible.append(
                        (legal_gain, int(added), int(removed), int(region_id))
                    )
    # Deterministic: highest legal gain, then added id, then removed id, region.
    eligible.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

    used_removed: set[int] = set()
    used_added: set[int] = set()
    swaps: list[GuardedSwap] = []
    for legal_gain, added, removed, region_id in eligible:
        if removed in used_removed or added in used_added:
            continue
        used_removed.add(removed)
        used_added.add(added)
        swap = _swap_from_pair(region_id, removed, added, semantic, l2, utility)
        if swap.legal_gain_q16 != legal_gain:
            raise GuardedRerankError(
                "internal error: planned legal gain is inconsistent (fail closed)"
            )
        swaps.append(swap)
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
    return GuardedRerankPlan(
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
        utility_q16=np.asarray(utility, dtype=np.int64),
        per_region_selected_counts_before=counts_before,
        per_region_selected_counts_after=counts_after,
    )


# ---------------------------------------------------------------------------
# Full selection entry point
# ---------------------------------------------------------------------------
def select_guarded_rerank(
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
    """Corrected integer-L2 K96 baseline + bounded L2-guarded swaps.

    Returns a dict with ``selection`` (a :class:`HardwareSelection`), the
    baseline/reranked audit and the utility vector.  ``G <= K`` returns the
    frozen identity bypass with ``swap_count == 0``.
    """

    hidden_array = validate_hidden(hidden)
    budget = validate_budget(budget)
    grid_h = int(grid_h)
    grid_w = int(grid_w)
    if grid_h * grid_w != hidden_array.shape[0]:
        raise GuardedRerankError(
            f"grid {grid_h}x{grid_w} does not match {hidden_array.shape[0]} groups"
        )
    semantic = np.asarray(semantic_norm_q15, dtype=np.int64).reshape(-1)
    l2 = np.asarray(l2_norm_q15, dtype=np.int64).reshape(-1)
    utility = utility_q16(semantic, l2)
    if utility.size != hidden_array.shape[0]:
        raise GuardedRerankError(
            f"utility has {utility.size} entries but there are "
            f"{hidden_array.shape[0]} groups"
        )
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
            "semantic_norm_q15": semantic,
            "l2_norm_q15": l2,
            "utility_q16": utility,
            "l2_combined": combined,
            "real_group_count": int(baseline.real_group_count),
            "context_group_count": int(baseline.context_group_count),
            "output_group_count": int(baseline.output_group_count),
            "context_pooling_basis": CONTEXT_POOLING_BASIS,
        }

    plan = plan_guarded_swaps(
        baseline,
        semantic,
        l2,
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
        "semantic_norm_q15": semantic,
        "l2_norm_q15": l2,
        "utility_q16": utility,
        "l2_combined": combined,
        "real_group_count": int(selection.real_group_count),
        "context_group_count": int(selection.context_group_count),
        "output_group_count": int(selection.output_group_count),
        "context_pooling_basis": CONTEXT_POOLING_BASIS,
    }


def _assert_plan_invariants(
    baseline: HardwareSelection, plan: GuardedRerankPlan
) -> None:
    anchor_set = set(int(value) for value in baseline.anchors)
    if not anchor_set.issubset(set(plan.reranked_selected_real)):
        raise GuardedRerankError("plan dropped a frozen anchor (fail closed)")
    if len(plan.reranked_selected_real) != len(plan.baseline_selected_real):
        raise GuardedRerankError(
            "plan changed the number of selected real groups (fail closed)"
        )
    if plan.swap_count > int(plan.max_swaps):
        raise GuardedRerankError("plan exceeded max_swaps (fail closed)")
    if plan.per_region_selected_counts_before != plan.per_region_selected_counts_after:
        raise GuardedRerankError(
            "plan changed a region's selected count (fail closed)"
        )
    removed = [swap.removed_group for swap in plan.swaps]
    added = [swap.added_group for swap in plan.swaps]
    if len(set(removed)) != len(removed) or len(set(added)) != len(added):
        raise GuardedRerankError("plan reuses a group across swaps (fail closed)")
    if set(removed) & set(added):
        raise GuardedRerankError("plan both removes and adds a group (fail closed)")
    for swap in plan.swaps:
        if swap.legal_gain_q16 <= 0:
            raise GuardedRerankError(
                "plan contains a non-positive legal gain (fail closed)"
            )
        if swap.legal_gain_q16 != (
            swap.utility_added_q16 - swap.utility_removed_q16
        ):
            raise GuardedRerankError(
                "plan legal_gain != utility_added - utility_removed (fail closed)"
            )
        if swap.semantic_gain_q15 != (
            swap.semantic_added_q15 - swap.semantic_removed_q15
        ):
            raise GuardedRerankError(
                "plan semantic_gain is inconsistent (fail closed)"
            )
        if swap.l2_loss_q15 != (swap.l2_removed_q15 - swap.l2_added_q15):
            raise GuardedRerankError("plan l2_loss is inconsistent (fail closed)")
        if swap.legal_gain_q16 != swap.semantic_gain_q15 - swap.l2_loss_q15:
            raise GuardedRerankError(
                "plan legal_gain != semantic_gain - l2_loss (fail closed)"
            )


def _assert_selection_invariants(
    baseline: HardwareSelection, selection: HardwareSelection, budget: int
) -> None:
    if selection.bypass:
        raise GuardedRerankError(
            "reranked selection must not be the identity bypass"
        )
    if selection.output_group_count != budget:
        raise GuardedRerankError(
            f"reranked selection emitted {selection.output_group_count} groups for "
            f"K={budget} (fail closed)"
        )
    anchor_set = set(int(value) for value in baseline.anchors)
    if not anchor_set.issubset(set(int(value) for value in selection.group_indices)):
        raise GuardedRerankError("reranked selection dropped an anchor (fail closed)")
    real = int(selection.real_group_count)
    context = int(selection.context_group_count)
    if real + context != budget:
        raise GuardedRerankError(
            "reranked real+context does not sum to K (fail closed)"
        )
    if not np.all(selection.valid):
        raise GuardedRerankError(
            "reranked selection has an invalid group (fail closed)"
        )


# ---------------------------------------------------------------------------
# B1 ``integer_scorer``-shaped callback
# ---------------------------------------------------------------------------
class GuardedRerankScorer:
    """B1 scorer callback computing the L2-guarded rerank payload for one image.

    Holds the frozen ``Wt`` / ``Wv_q`` bundle and calls the **unmodified**
    corrected-L2 arm-3 chain to obtain the accepted integer semantic Q0.15 and
    the corrected integer patch-L2 Q0.15, then applies the bounded L2-guarded
    rerank.  The returned payload contains ``selection`` (consumed by the v2 B1
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
            raise GuardedRerankError(
                "the guarded reranker requires the frozen Wt and Wv_q bundle"
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
        result = select_guarded_rerank(
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


def build_guarded_scorer(
    Wt: Any,
    wv_q: Any,
    *,
    budgets: Sequence[int] = (BUDGET,),
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
) -> GuardedRerankScorer:
    return GuardedRerankScorer(
        Wt,
        wv_q,
        budgets=budgets,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        max_swaps=max_swaps,
    )


def audit_hashes(result: Mapping[str, Any]) -> dict[str, Any]:
    """Portable SHA256 digests of every L2-guarded rerank audit array."""

    plan = result.get("baseline_plan")
    return {
        "semantic_norm_q15_sha256": array_sha256(
            np.asarray(result["semantic_norm_q15"], dtype=np.int64)
        ),
        "l2_norm_q15_sha256": array_sha256(
            np.asarray(result["l2_norm_q15"], dtype=np.int64)
        ),
        "utility_q16_sha256": array_sha256(
            np.asarray(result["utility_q16"], dtype=np.int64)
        ),
        "l2_combined_sha256": array_sha256(
            np.asarray(result["l2_combined"], dtype=np.float64)
        ),
        "baseline_selected_real_sha256": array_sha256(
            np.asarray(
                plan["baseline_selected_real"] if plan else [], dtype=np.int64
            )
        ),
        "reranked_selected_real_sha256": array_sha256(
            np.asarray(plan["reranked_selected_real"] if plan else [], dtype=np.int64)
        ),
    }


__all__ = [
    "BUDGET",
    "CANDIDATE_ORDER",
    "CONTEXT_POOLING_BASIS",
    "GuardedRerankError",
    "GuardedRerankPlan",
    "GuardedRerankScorer",
    "GuardedSwap",
    "LEGAL_GAIN_FORMULA",
    "LOCAL_CANDIDATE_CAP",
    "MAX_SWAPS",
    "MECHANISM",
    "MECHANISM_FAMILY",
    "REGION_GRID",
    "SOURCE_CONTEXT",
    "SOURCE_REAL",
    "UTILITY_FORMULA",
    "UTILITY_MAX",
    "audit_hashes",
    "build_guarded_scorer",
    "plan_guarded_swaps",
    "select_guarded_rerank",
    "utility_q16",
]
