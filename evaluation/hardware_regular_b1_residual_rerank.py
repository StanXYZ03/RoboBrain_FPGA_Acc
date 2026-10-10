#!/usr/bin/env python3
"""B1 integration of the conservative L2-residual reranker (new, additive).

The frozen :func:`evaluation.hardware_regular_b1_integration.forward_hardware_regular_b1_vision`
cannot express this mechanism: its single ``integer_scorer`` hook feeds one
``combined_scores`` vector that the frozen selector uses **both** for selection
ranking and for context pooling weights, while the conservative residual
reranker must keep selection ranking (residual) separate from context pooling
(corrected integer-L2).  This module therefore mirrors the frozen forward in a
new file (frozen code untouched) and swaps only the selection step.

Everything else is shared with the frozen path: the same vision preamble, the
same block-0 cut (``B1``), the same ``compact_hidden`` construction, the same
DeepStack/merger tail and the same ``HardwareB1VisionOutput`` dataclass.  With a
zero-swap plan the returned group indices / groups / source types / region ids
are element-wise identical to the frozen integration (checked by the focused
tests).

No model weights, no CUDA at import, no training here.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.conservative_l2_residual_reranker import (
    BUDGET,
    LOCAL_CANDIDATE_CAP,
    MAX_SWAPS,
    MECHANISM,
    ConservativeResidualScorer,
)
from evaluation.hardware_regular_b1_integration import (
    SCORE_MODE_INTEGER,
    HardwareB1VisionOutput,
)
from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    REGION_GRID,
    HardwareSelection,
    SelectorContractError,
    select_hardware_regular_groups,
)
from evaluation.shallow_block_pruning import PrunedVisionOutput


def _clock(synchronize: Callable[[], None] | None) -> float:
    if synchronize is not None:
        synchronize()
    return time.perf_counter()


def _bypass_payload(selection: HardwareSelection) -> dict[str, Any]:
    return {
        "mechanism": MECHANISM,
        "selection": selection,
        "bypass": True,
        "swap_count": 0,
        "baseline_plan": None,
        "semantic_norm_q15": None,
        "l2_norm_q15": None,
        "residual_q15": None,
        "l2_combined": None,
        "real_group_count": int(selection.real_group_count),
        "context_group_count": int(selection.context_group_count),
        "output_group_count": int(selection.output_group_count),
    }


def forward_conservative_residual_rerank_b1_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    text_embeddings: Any,
    residual_scorer: ConservativeResidualScorer | Callable[..., dict[str, Any]],
    budget: int = BUDGET,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    max_swaps: int = MAX_SWAPS,
    boundary: str = BOUNDARY_MAIN,
    synchronize: Callable[[], None] | None = None,
) -> HardwareB1VisionOutput:
    """Run B1 with the conservative residual reranker selection step."""

    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("B1 integration currently supports one still image")
    if len(visual.blocks) < 1:
        raise ValueError("the vision model must contain at least one block")
    if residual_scorer is None:
        raise SelectorContractError(
            "the residual reranker requires an explicit residual_scorer callback"
        )
    if text_embeddings is None:
        raise SelectorContractError(
            "the residual reranker requires the prompt text embeddings"
        )

    preamble_started = _clock(synchronize)
    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype))
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    seq_len = int(hidden_states.shape[0])
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    full_cos, full_sin = emb.cos(), emb.sin()

    merge_size = int(visual.spatial_merge_size)
    merge_unit = merge_size * merge_size
    full_groups = seq_len // merge_unit
    merged_h = int(grid_thw[0, 1].item()) // merge_size
    merged_w = int(grid_thw[0, 2].item()) // merge_size
    if full_groups != merged_h * merged_w:
        raise ValueError("vision token order is not merge-grid aligned")

    cu_seqlens = F.pad(
        torch.tensor([seq_len], dtype=torch.int32, device=hidden_states.device),
        (1, 0),
        value=0,
    )
    position_embeddings = (full_cos, full_sin)
    block_input_shapes: list[int] = [int(hidden_states.shape[0])]
    vision_preamble_seconds = _clock(synchronize) - preamble_started

    block0_started = _clock(synchronize)
    hidden_states = visual.blocks[0](
        hidden_states,
        cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings,
    )
    block0_seconds = _clock(synchronize) - block0_started
    block0_hidden = hidden_states

    selector_started = _clock(synchronize)
    hidden_groups = hidden_states.reshape(full_groups, merge_unit, -1)
    hidden_np = hidden_groups.detach().to(torch.float64).cpu().numpy()
    budget_int = int(budget)
    compression_active = full_groups > budget_int
    if compression_active:
        payload = residual_scorer(
            hidden_np, merged_h, merged_w, budget_int, text_embeddings
        )
        if not isinstance(payload, dict) or "selection" not in payload:
            raise SelectorContractError(
                "residual_scorer must return a payload containing 'selection'"
            )
        selection = payload["selection"]
        if not isinstance(selection, HardwareSelection):
            raise SelectorContractError(
                "residual_scorer 'selection' must be a HardwareSelection"
            )
        if selection.bypass:
            raise SelectorContractError(
                "residual reranker returned the identity bypass for a compressed "
                "sample (fail closed)"
            )
        if int(selection.budget) != budget_int:
            raise SelectorContractError(
                f"residual reranker selection budget {selection.budget} != "
                f"{budget_int} (fail closed)"
            )
    else:
        selection = select_hardware_regular_groups(
            hidden_np,
            merged_h,
            merged_w,
            budget=budget_int,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
            boundary=boundary,
        )
        payload = _bypass_payload(selection)

    output_count = selection.output_group_count
    if output_count != budget_int and not selection.bypass:
        raise SelectorContractError(
            f"selector emitted {output_count} groups for budget {budget_int}"
        )
    group_indices = np.asarray(selection.group_indices, dtype=np.int64)
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise SelectorContractError(
            "hardware selector group_indices must be strictly ascending and unique"
        )
    if not np.all(selection.valid):
        raise SelectorContractError("hardware selector produced an invalid group")

    compact_hidden = torch.as_tensor(
        selection.groups, dtype=hidden_states.dtype, device=hidden_states.device
    ).reshape(output_count * merge_unit, -1)
    patch_flat = torch.as_tensor(
        selection.patch_indices.reshape(-1),
        dtype=torch.long,
        device=hidden_states.device,
    )
    compact_cos = full_cos.index_select(0, patch_flat)
    compact_sin = full_sin.index_select(0, patch_flat)
    compact_cu_seqlens = torch.tensor(
        [0, output_count * merge_unit],
        dtype=torch.int32,
        device=hidden_states.device,
    )
    selector_cpu_seconds = _clock(synchronize) - selector_started

    remaining_started = _clock(synchronize)
    hidden_states = compact_hidden
    deepstack_indexes = [int(value) for value in visual.deepstack_visual_indexes]
    deepstack_features: list[torch.Tensor] = []
    for layer_num in range(1, len(visual.blocks)):
        block_input_shapes.append(int(hidden_states.shape[0]))
        hidden_states = visual.blocks[layer_num](
            hidden_states,
            cu_seqlens=compact_cu_seqlens,
            position_embeddings=(compact_cos, compact_sin),
        )
        if layer_num in deepstack_indexes:
            merger_index = deepstack_indexes.index(layer_num)
            deepstack_features.append(
                visual.deepstack_merger_list[merger_index](hidden_states)
            )
    image_embeds = visual.merger(hidden_states)
    if any(
        feature.shape[0] != image_embeds.shape[0] for feature in deepstack_features
    ):
        raise RuntimeError(
            "DeepStack features are not aligned with the compacted image tokens"
        )
    remaining_vision_seconds = _clock(synchronize) - remaining_started

    scores = selection.scores
    group_l2 = None
    if scores is not None and scores.l2 is not None:
        group_l2 = torch.as_tensor(scores.l2, dtype=torch.float32)
    group_scores = (
        torch.as_tensor(scores.combined, dtype=torch.float32)
        if scores is not None
        else None
    )
    pruned = PrunedVisionOutput(
        image_embeds=image_embeds,
        deepstack_embeds=deepstack_features,
        selected_group_indices=torch.as_tensor(
            group_indices, dtype=torch.long, device=hidden_states.device
        ),
        selected_patch_indices=torch.as_tensor(
            selection.patch_indices.reshape(-1),
            dtype=torch.long,
            device=hidden_states.device,
        ),
        group_scores=group_scores,
        full_patch_tokens=seq_len,
        full_llm_visual_tokens=full_groups,
        group_l2=group_l2,
    )
    return HardwareB1VisionOutput(
        pruned=pruned,
        selection=selection,
        full_patch_tokens=seq_len,
        full_llm_visual_tokens=full_groups,
        budget=budget_int,
        merged_h=merged_h,
        merged_w=merged_w,
        score_mode=SCORE_MODE_INTEGER,
        block_input_shapes=block_input_shapes,
        vision_preamble_seconds=vision_preamble_seconds,
        block0_seconds=block0_seconds,
        selector_cpu_seconds=selector_cpu_seconds,
        remaining_vision_seconds=remaining_vision_seconds,
        compact_hidden=compact_hidden,
        compact_cos=compact_cos,
        compact_sin=compact_sin,
        compact_cu_seqlens=compact_cu_seqlens,
        block0_hidden=block0_hidden,
        full_cos=full_cos,
        full_sin=full_sin,
        integer_payload=payload,
    )


__all__ = [
    "forward_conservative_residual_rerank_b1_vision",
]
