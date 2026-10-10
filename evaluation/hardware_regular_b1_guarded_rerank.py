#!/usr/bin/env python3
"""Minimal B1 integration for the L2-guarded integer reranker (v2, additive).

The frozen :func:`evaluation.hardware_regular_b1_integration.forward_hardware_regular_b1_vision`
feeds a single ``combined_scores`` vector to the frozen selector, so it cannot
express a prebuilt :class:`~evaluation.hardware_regular_token_selector.HardwareSelection`.
The already tested **v1 additive** integration
:func:`evaluation.hardware_regular_b1_residual_rerank.forward_conservative_residual_rerank_b1_vision`
*does* accept a scorer callback that returns a prebuilt ``selection``; it is
therefore reused here verbatim.  The only v2-specific fix is the ``G <= K``
identity-bypass payload label, because the reused v1 helper stamps its own
mechanism name on that branch only.

Nothing in the v1 source is modified, and the compressed path is produced
entirely by the v2 :class:`~evaluation.conservative_l2_guarded_reranker.GuardedRerankScorer`.

No model weights, no CUDA at import, no training here.
"""

from __future__ import annotations

from typing import Any, Callable

from evaluation.conservative_l2_guarded_reranker import (
    BUDGET,
    CONTEXT_POOLING_BASIS,
    LOCAL_CANDIDATE_CAP,
    MAX_SWAPS,
    MECHANISM,
    UTILITY_FORMULA,
)
from evaluation.hardware_regular_b1_integration import (
    HardwareB1VisionOutput,
)
from evaluation.hardware_regular_b1_residual_rerank import (
    forward_conservative_residual_rerank_b1_vision as _v1_forward,
)
from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    REGION_GRID,
    HardwareSelection,
)


def _guarded_bypass_payload(selection: HardwareSelection) -> dict[str, Any]:
    """v2-labelled identity-bypass payload (``G <= K``, zero swaps)."""

    return {
        "mechanism": MECHANISM,
        "selection": selection,
        "bypass": True,
        "swap_count": 0,
        "baseline_plan": None,
        "semantic_norm_q15": None,
        "l2_norm_q15": None,
        "utility_q16": None,
        "l2_combined": None,
        "real_group_count": int(selection.real_group_count),
        "context_group_count": int(selection.context_group_count),
        "output_group_count": int(selection.output_group_count),
        "context_pooling_basis": CONTEXT_POOLING_BASIS,
        "utility_formula": UTILITY_FORMULA,
    }


def forward_guarded_rerank_b1_vision(
    visual: Any,
    pixel_values: Any,
    grid_thw: Any,
    *,
    text_embeddings: Any,
    guarded_scorer: Callable[..., dict[str, Any]],
    budget: int = BUDGET,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
    boundary: str = BOUNDARY_MAIN,
    synchronize: Callable[[], None] | None = None,
) -> HardwareB1VisionOutput:
    """Run B1 with the v2 L2-guarded reranker selection step.

    Reuses the tested v1 additive forward (which itself mirrors the frozen B1
    preamble / compaction / DeepStack / merger tail) and only relabels the
    identity-bypass payload for v2.  The compressed path returns the scorer's own
    v2 payload untouched.
    """

    output = _v1_forward(
        visual,
        pixel_values,
        grid_thw,
        text_embeddings=text_embeddings,
        residual_scorer=guarded_scorer,
        budget=budget,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        max_swaps=max_swaps,
        boundary=boundary,
        synchronize=synchronize,
    )
    payload = output.integer_payload
    if payload is not None and bool(payload.get("bypass")):
        output.integer_payload = _guarded_bypass_payload(output.selection)
    return output


__all__ = [
    "forward_guarded_rerank_b1_vision",
]
