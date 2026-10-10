"""Qwen3-VL layer-5 (after vision block 5) integration of the hardware selector.

This is a *separate* cut-point integration from
:mod:`evaluation.hardware_regular_b1_integration`; the frozen B1 module is never
imported for its control flow and is never edited.

Contract (all fixed by the registered screening100 experiment):

* one still image only;
* the cut point is fixed at ``prune_after_layer=5`` (the full-sequence vision
  prefix is blocks ``0..5`` inclusive); any other cut point is rejected;
* ``score_mode="l2"`` only.  The scoring reuses the frozen
  :func:`evaluation.hardware_regular_token_selector.group_l2_saliency` and
  :func:`evaluation.hardware_regular_token_selector.minmax_normalize` over the
  ``[G, 4, 1024]`` merged groups and the frozen
  :func:`evaluation.hardware_regular_token_selector.select_hardware_regular_groups`
  selection rule, keeping ``region_grid=3`` and ``local_candidate_cap=16``.

The **B1-trained ``Wv`` projection is never used here**: no prompt projection is
accepted, and passing ``projection`` / ``text_embeddings`` raises
:class:`SelectorContractError`.  The only caller-supplied score interface is the
L2 branch.

Pipeline: run vision blocks ``0..5`` on the *full* patch sequence ``P``, select
``K`` output groups with the frozen hardware selector, build the compact hidden
from ``selection.groups`` (pooled values) with ``cos``/``sin`` gathered by
``selection.patch_indices``, then:

* call the DeepStack ``deepstack_merger`` whose index equals block 5 on the
  **compact** hidden (never on the full hidden);
* run blocks ``6..len(blocks)-1`` on the compact sequence, invoking the
  remaining DeepStack mergers as their layer index is reached;
* call the final ``merger`` and assert every DeepStack feature is aligned with
  ``image_embeds``.

No model weights, no CUDA requirement at import time, no training here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.hardware_regular_token_selector import (
    BOUNDARY_MAIN,
    DEFAULT_LOCAL_CANDIDATE_CAP,
    REGION_GRID,
    HardwareSelection,
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

#: The only supported cut point: after the sixth vision block (index 5).
PRUNE_AFTER_LAYER = 5
#: The only supported score mode for this cut point.
SCORE_MODE_L2 = "l2"
ALLOWED_SCORE_MODES = (SCORE_MODE_L2,)
#: Fixed screening100 budget / grid / local cap.
BUDGET = 96
#: ``REGION_GRID`` is re-exported from the frozen token selector (value 3).
LOCAL_CANDIDATE_CAP = DEFAULT_LOCAL_CANDIDATE_CAP
BOUNDARY = BOUNDARY_MAIN


@dataclass
class HardwareLayer5VisionOutput:
    """Vision-side result of the layer-5 hardware-regular L2 path."""

    pruned: PrunedVisionOutput
    selection: HardwareSelection
    full_patch_tokens: int
    full_llm_visual_tokens: int
    budget: int
    merged_h: int
    merged_w: int
    score_mode: str
    prune_after_layer: int
    block_input_shapes: list[int]
    vision_preamble_seconds: float
    cut_blocks_seconds: float
    selector_cpu_seconds: float
    remaining_vision_seconds: float
    compact_hidden: torch.Tensor
    compact_cos: torch.Tensor
    compact_sin: torch.Tensor
    compact_cu_seqlens: torch.Tensor
    cut_hidden: torch.Tensor
    full_cos: torch.Tensor
    full_sin: torch.Tensor
    deepstack_cut_merger_index: int | None = None
    deepstack_pre_cut_count: int = 0
    deepstack_merger_indexes: tuple[int, ...] = field(default_factory=tuple)

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
    def block0_seconds(self) -> float:
        """B1-runner field-name alias for the pre-selector blocks.

        In this layer-5 path it is the *full* prefix blocks ``0..5``; the alias
        exists only so the runner record keeps the frozen B1 field name.
        """
        return self.cut_blocks_seconds

    @property
    def vision_total_seconds(self) -> float:
        """Wall-clock vision time: preamble + cut blocks + selector + remaining.

        The CPU selector round trip is included here as an explicit overhead
        term; it is never GPU compute and must not be reported as one.
        """
        return (
            self.vision_preamble_seconds
            + self.cut_blocks_seconds
            + self.selector_cpu_seconds
            + self.remaining_vision_seconds
        )

    @property
    def gpu_vision_seconds(self) -> float:
        """GPU-side vision time only: excludes the CPU selector round trip."""
        return (
            self.vision_preamble_seconds
            + self.cut_blocks_seconds
            + self.remaining_vision_seconds
        )

    def as_pruned_vision_output(self) -> PrunedVisionOutput:
        return self.pruned


def _clock(synchronize: Callable[[], None] | None) -> float:
    if synchronize is not None:
        synchronize()
    return time.perf_counter()


def _validate_cutpoint(prune_after_layer: Any) -> int:
    if isinstance(prune_after_layer, bool) or not isinstance(
        prune_after_layer, (int, np.integer)
    ):
        raise SelectorContractError(
            f"prune_after_layer must be the integer {PRUNE_AFTER_LAYER}, "
            f"got {prune_after_layer!r}"
        )
    value = int(prune_after_layer)
    if value != PRUNE_AFTER_LAYER:
        raise SelectorContractError(
            "the layer-5 integration only supports "
            f"prune_after_layer={PRUNE_AFTER_LAYER}, got {value}"
        )
    return value


def forward_hardware_regular_layer5_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    budget: int = BUDGET,
    score_mode: str = SCORE_MODE_L2,
    region_grid: int = REGION_GRID,
    local_candidate_cap: int = LOCAL_CANDIDATE_CAP,
    boundary: str = BOUNDARY,
    prune_after_layer: int = PRUNE_AFTER_LAYER,
    synchronize: Callable[[], None] | None = None,
    projection: Any | None = None,
    text_embeddings: Any | None = None,
) -> HardwareLayer5VisionOutput:
    """Run layer-5: blocks 0..5 on full ``P``, selector compaction, blocks 6..N-1.

    ``projection`` / ``text_embeddings`` are accepted only so the B1-trained
    ``Wv`` path can be explicitly rejected: the layer-5 experiment is L2-only and
    must never reuse the B1-trained ``Wv``.
    """
    if score_mode not in ALLOWED_SCORE_MODES:
        raise SelectorContractError(
            f"score_mode must be one of {ALLOWED_SCORE_MODES}, got {score_mode!r}"
        )
    if projection is not None or text_embeddings is not None:
        raise SelectorContractError(
            "layer-5 integration is L2-only and must not use the B1-trained "
            "Wv projection or text embeddings"
        )
    prune_after_layer = _validate_cutpoint(prune_after_layer)
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("layer-5 integration currently supports one still image")
    if len(visual.blocks) <= prune_after_layer:
        raise ValueError(
            "the vision model must contain more than "
            f"{prune_after_layer} blocks for a layer-5 cut"
        )

    # --- Vision preamble (patch embed / learned pos / rotary / cu_seqlens) --
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

    full_cu_seqlens = F.pad(
        torch.tensor([seq_len], dtype=torch.int32, device=hidden_states.device),
        (1, 0),
        value=0,
    )
    position_embeddings = (full_cos, full_sin)
    block_input_shapes: list[int] = []
    vision_preamble_seconds = _clock(synchronize) - preamble_started

    # --- Full-sequence prefix: blocks 0..prune_after_layer (inclusive) -----
    deepstack_indexes = [int(value) for value in visual.deepstack_visual_indexes]
    if prune_after_layer not in deepstack_indexes:
        raise SelectorContractError(
            f"the layer-5 cut requires DeepStack index {prune_after_layer} to be "
            f"present in deepstack_visual_indexes={deepstack_indexes}"
        )
    deepstack_cut_merger_index = deepstack_indexes.index(prune_after_layer)
    # DeepStack mergers for layers strictly before the cut are captured on the
    # full sequence and compacted by group index below.
    deepstack_pre_cut: list[tuple[int, torch.Tensor]] = []

    cut_started = _clock(synchronize)
    for layer_num in range(0, prune_after_layer + 1):
        block_input_shapes.append(int(hidden_states.shape[0]))
        hidden_states = visual.blocks[layer_num](
            hidden_states,
            cu_seqlens=full_cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_num in deepstack_indexes and layer_num < prune_after_layer:
            merger_index = deepstack_indexes.index(layer_num)
            deepstack_pre_cut.append(
                (
                    merger_index,
                    visual.deepstack_merger_list[merger_index](hidden_states),
                )
            )
    cut_hidden = hidden_states
    cut_blocks_seconds = _clock(synchronize) - cut_started

    # --- Hardware-regular L2 selection bridge (CPU numpy round trip) ------
    selector_started = _clock(synchronize)
    hidden_groups = hidden_states.reshape(full_groups, merge_unit, -1)
    hidden_np = hidden_groups.detach().to(torch.float64).cpu().numpy()
    budget_int = int(budget)
    l2_scores = group_l2_saliency(hidden_np)
    selection = select_hardware_regular_groups(
        hidden_np,
        merged_h,
        merged_w,
        budget=budget_int,
        region_grid=region_grid,
        local_candidate_cap=local_candidate_cap,
        boundary=boundary,
        l2_scores=l2_scores,
        combined_scores=minmax_normalize(l2_scores),
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
    group_indices_t = torch.as_tensor(
        group_indices, dtype=torch.long, device=hidden_states.device
    )
    deepstack_features: list[torch.Tensor] = []
    # Pre-cut DeepStack features (if any) are compacted by selected group index
    # so they stay aligned with the compact image tokens.
    for _merger_index, feature in sorted(deepstack_pre_cut):
        deepstack_features.append(feature.index_select(0, group_indices_t))
    # The DeepStack merger at the cut point is called on the *compact* hidden.
    deepstack_features.append(
        visual.deepstack_merger_list[deepstack_cut_merger_index](compact_hidden)
    )
    for layer_num in range(prune_after_layer + 1, len(visual.blocks)):
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
    return HardwareLayer5VisionOutput(
        pruned=pruned,
        selection=selection,
        full_patch_tokens=seq_len,
        full_llm_visual_tokens=full_groups,
        budget=budget_int,
        merged_h=merged_h,
        merged_w=merged_w,
        score_mode=score_mode,
        prune_after_layer=prune_after_layer,
        block_input_shapes=block_input_shapes,
        vision_preamble_seconds=vision_preamble_seconds,
        cut_blocks_seconds=cut_blocks_seconds,
        selector_cpu_seconds=selector_cpu_seconds,
        remaining_vision_seconds=remaining_vision_seconds,
        compact_hidden=compact_hidden,
        compact_cos=compact_cos,
        compact_sin=compact_sin,
        compact_cu_seqlens=compact_cu_seqlens,
        cut_hidden=cut_hidden,
        full_cos=full_cos,
        full_sin=full_sin,
        deepstack_cut_merger_index=deepstack_cut_merger_index,
        deepstack_pre_cut_count=len(deepstack_pre_cut),
        deepstack_merger_indexes=tuple(deepstack_indexes),
    )


def prepare_layer5_multimodal_inputs(
    model: torch.nn.Module,
    batch: Any,
    result: HardwareLayer5VisionOutput,
) -> PrunedMultimodalInputs:
    """Reuse the full ``input_ids`` then keep the unique ascending groups."""
    group_indices = result.group_indices
    expected = int(result.pruned.image_embeds.shape[0])
    if group_indices.size != expected:
        raise ValueError(
            f"layer-5 produced {group_indices.size} groups but {expected} image embeds"
        )
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise ValueError("layer-5 group_indices must be strictly ascending and unique")
    if int(result.valid.sum()) != expected:
        raise ValueError("layer-5 valid mask does not match the image token count")
    return prepare_pruned_multimodal_inputs(model, batch, result.pruned)


__all__ = [
    "ALLOWED_SCORE_MODES",
    "BUDGET",
    "HardwareLayer5VisionOutput",
    "LOCAL_CANDIDATE_CAP",
    "PRUNE_AFTER_LAYER",
    "REGION_GRID",
    "SCORE_MODE_L2",
    "forward_hardware_regular_layer5_vision",
    "prepare_layer5_multimodal_inputs",
]
