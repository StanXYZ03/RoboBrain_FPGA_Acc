"""Qwen3-VL B1 (after vision block 0) integration of the hardware-regular selector.

This module reuses the ``shallow_block_pruning`` skeleton but never edits it.
It runs vision block 0 on the *full* patch sequence ``P``, calls the CPU
``hardware_regular_token_selector`` on the ``[G, 4, 1024]`` merged groups, and
then continues blocks 1..N-1 / DeepStack / merger on the compacted
``K * 4`` patch sequence.

The one semantic difference against a pure real-group gather is the synthetic
context group: the compacted hidden rows always come from
``selection.groups.reshape(K * 4, 1024)`` (pooled values), while ``cos``/``sin``
are gathered with ``selection.patch_indices`` (representative positions), the
``cu_seqlens`` is rebuilt, and the placeholder / M-RoPE handling is delegated to
the existing ``prepare_pruned_multimodal_inputs`` so the full ``input_ids`` are
used before the unique ascending ``group_indices`` slice is applied.

No model weights, no CUDA requirement at import time, no training here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    DEFAULT_LOCAL_CANDIDATE_CAP,
    REGION_GRID,
    HardwareSelection,
    LowRankProjection,
    SelectorContractError,
    group_l2_saliency,
    minmax_normalize,
    select_hardware_regular_groups,
)
from evaluation.shallow_block_pruning import (
    PrunedMultimodalInputs,
    PrunedVisionOutput,
    prepare_pruned_multimodal_inputs,
)

SCORE_MODE_L2 = "l2"
SCORE_MODE_PROMPT = "prompt"
#: 2026-10-03 integer scorer + frozen hardware selection.  Requires an explicit
#: ``integer_scorer`` callback; it never regenerates scores from a float
#: projection.
SCORE_MODE_INTEGER = "integer"
ALLOWED_SCORE_MODES = (SCORE_MODE_L2, SCORE_MODE_PROMPT, SCORE_MODE_INTEGER)


@dataclass
class HardwareB1VisionOutput:
    """Vision-side result of the B1 hardware-regular path."""

    pruned: PrunedVisionOutput
    selection: HardwareSelection
    full_patch_tokens: int
    full_llm_visual_tokens: int
    budget: int
    merged_h: int
    merged_w: int
    score_mode: str
    block_input_shapes: list[int]
    vision_preamble_seconds: float
    block0_seconds: float
    selector_cpu_seconds: float
    remaining_vision_seconds: float
    compact_hidden: torch.Tensor
    compact_cos: torch.Tensor
    compact_sin: torch.Tensor
    compact_cu_seqlens: torch.Tensor
    block0_hidden: torch.Tensor
    full_cos: torch.Tensor
    full_sin: torch.Tensor
    integer_payload: dict[str, Any] | None = None

    @property
    def image_embeds(self) -> torch.Tensor:
        return self.pruned.image_embeds

    @property
    def deepstack_embeds(self) -> list[torch.Tensor]:
        return self.pruned.deepstack_embeds

    @property
    def group_indices(self) -> np.ndarray:
        return self.selection.group_indices

    @property
    def patch_indices(self) -> np.ndarray:
        return self.selection.patch_indices

    @property
    def source_type(self) -> np.ndarray:
        return self.selection.source_type

    @property
    def region_id(self) -> np.ndarray:
        return self.selection.region_id

    @property
    def valid(self) -> np.ndarray:
        return self.selection.valid

    @property
    def context_group_count(self) -> int:
        return self.selection.context_group_count

    @property
    def real_group_count(self) -> int:
        return self.selection.real_group_count

    @property
    def vision_total_seconds(self) -> float:
        """Wall-clock vision time: preamble + block0 + selector + remaining.

        The CPU selector round trip is included here as an explicit overhead
        term; it is never GPU compute and must not be reported as one.
        """
        return (
            self.vision_preamble_seconds
            + self.block0_seconds
            + self.selector_cpu_seconds
            + self.remaining_vision_seconds
        )

    @property
    def gpu_vision_seconds(self) -> float:
        """GPU-side vision time only: excludes the CPU selector round trip."""
        return (
            self.vision_preamble_seconds
            + self.block0_seconds
            + self.remaining_vision_seconds
        )

    def as_pruned_vision_output(self) -> PrunedVisionOutput:
        return self.pruned


def _clock(synchronize: Callable[[], None] | None) -> float:
    if synchronize is not None:
        synchronize()
    return time.perf_counter()


def _group_logical_scores(
    hidden_groups: np.ndarray,
    *,
    score_mode: str,
    text_embeddings: Any | None,
    projection: Any | None,
    lambda_semantic: float | None,
) -> dict[str, Any]:
    """Return selector keyword arguments for the requested score mode."""
    if score_mode == SCORE_MODE_L2:
        # The L2 mode derives ``combined`` from the B1 group-mean L2 without any
        # prompt or projection, and never randomly generates weights.
        l2 = group_l2_saliency(hidden_groups)
        return {"l2_scores": l2, "combined_scores": minmax_normalize(l2)}
    if score_mode == SCORE_MODE_PROMPT:
        if projection is None:
            raise SelectorContractError(
                "prompt score mode requires an explicit Wt/Wv projection; "
                "none was provided"
            )
        if not isinstance(projection, LowRankProjection):
            raise SelectorContractError(
                "prompt score mode requires a LowRankProjection instance"
            )
        if text_embeddings is None:
            raise SelectorContractError(
                "prompt score mode requires text_embeddings"
            )
        if lambda_semantic is None:
            raise SelectorContractError(
                "prompt score mode requires an explicit lambda_semantic"
            )
        return {
            "text_embeddings": text_embeddings,
            "projection": projection,
            "lambda_semantic": lambda_semantic,
        }
    raise SelectorContractError(
        f"score_mode must be one of {ALLOWED_SCORE_MODES}, got {score_mode!r}"
    )


def forward_hardware_regular_b1_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    budget: int,
    score_mode: str = SCORE_MODE_L2,
    text_embeddings: Any | None = None,
    projection: LowRankProjection | None = None,
    lambda_semantic: float | None = None,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = DEFAULT_LOCAL_CANDIDATE_CAP,
    boundary: str = BOUNDARY_MAIN,
    synchronize: Callable[[], None] | None = None,
    integer_scorer: Callable[..., dict[str, Any]] | None = None,
) -> HardwareB1VisionOutput:
    """Run B1: block 0 on full ``P``, selector compaction, blocks 1..N-1."""
    if score_mode not in ALLOWED_SCORE_MODES:
        raise SelectorContractError(
            f"score_mode must be one of {ALLOWED_SCORE_MODES}, got {score_mode!r}"
        )
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("B1 integration currently supports one still image")
    if len(visual.blocks) < 1:
        raise ValueError("the vision model must contain at least one block")

    # --- Vision preamble (patch embed / learned pos / rotary / cu_seqlens) --
    # Bracket every GPU timing segment with the caller supplied ``synchronize``
    # so the measured seconds exclude queued work from the previous segment.
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

    # --- B1 selection bridge (CPU numpy round trip) -----------------------
    selector_started = _clock(synchronize)
    hidden_groups = hidden_states.reshape(full_groups, merge_unit, -1)
    hidden_np = hidden_groups.detach().to(torch.float64).cpu().numpy()
    budget_int = int(budget)
    compression_active = full_groups > budget_int
    integer_payload: dict[str, Any] | None = None
    if compression_active:
        if score_mode == SCORE_MODE_INTEGER:
            if integer_scorer is None:
                raise SelectorContractError(
                    "integer score mode requires an integer_scorer callback"
                )
            integer_payload = integer_scorer(
                hidden_np, merged_h, merged_w, budget_int, text_embeddings
            )
            score_kwargs = {"combined_scores": integer_payload["combined_scores"]}
        else:
            score_kwargs = _group_logical_scores(
                hidden_np,
                score_mode=score_mode,
                text_embeddings=(
                    np.asarray(text_embeddings)
                    if text_embeddings is not None
                    else None
                ),
                projection=projection,
                lambda_semantic=lambda_semantic,
            )
        selection = select_hardware_regular_groups(
            hidden_np,
            merged_h,
            merged_w,
            budget=budget_int,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
            boundary=boundary,
            **score_kwargs,
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

    # --- Remaining vision blocks / DeepStack / merger ---------------------
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
        score_mode=score_mode,
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
        integer_payload=integer_payload,
    )


def prepare_b1_multimodal_inputs(
    model: torch.nn.Module,
    batch: Any,
    result: HardwareB1VisionOutput,
) -> PrunedMultimodalInputs:
    """Reuse the full ``input_ids`` then keep the unique ascending B1 groups."""
    group_indices = result.group_indices
    expected = int(result.pruned.image_embeds.shape[0])
    if group_indices.size != expected:
        raise ValueError(
            f"B1 produced {group_indices.size} groups but {expected} image embeds"
        )
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise ValueError("B1 group_indices must be strictly ascending and unique")
    if int(result.valid.sum()) != expected:
        raise ValueError("B1 valid mask does not match the image token count")
    return prepare_pruned_multimodal_inputs(model, batch, result.pruned)


__all__ = [
    "ALLOWED_SCORE_MODES",
    "HardwareB1VisionOutput",
    "SCORE_MODE_INTEGER",
    "SCORE_MODE_L2",
    "SCORE_MODE_PROMPT",
    "forward_hardware_regular_b1_vision",
    "prepare_b1_multimodal_inputs",
]
