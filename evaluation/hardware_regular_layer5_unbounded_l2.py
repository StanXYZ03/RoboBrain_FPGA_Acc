"""Layer-5 unbounded real-only L2 + 3x3 anchor integration (diagnostic arm B).

This is a *separate*, independent layer-5 cut-point integration.  The frozen
layer-5, context-ablation and B1 modules are never edited and their control flow
is never imported.

Selection contract (arm B)
--------------------------
* Every output slot is a **real** original 2x2 merge group.  There is **no**
  context pooling, **no** context slot reservation and **no**
  ``local_candidate_cap``.
* The frozen 3x3 spatial partition still contributes one anchor per non-empty
  region.  The anchor is the region member with the highest L2-derived combined
  score (ties resolve to the lower original group index), exactly the frozen
  anchor rule.
* The remaining ``K - anchors`` slots are the **global** L2 ranking
  (``minmax_normalize(group_l2_saliency)``, descending, ties to the lower index)
  over every group that is not an anchor.  No bounded per-region candidate pool
  is used.
* ``G <= K`` is an element-wise identity bypass (all real, no compression).
* Selection metadata is emitted with ``source_type`` all ``SOURCE_REAL`` and
  ``context_group_count == 0``; ``groups`` therefore equals the raw selected
  groups.

Cut-point contract
------------------
Blocks ``0..5`` run on the full patch sequence, the compact hidden is built from
the selected real groups, the DeepStack merger whose index equals block 5 is
called on the **compact** hidden, and blocks ``6..23`` then run on the compact
sequence.

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
    MERGE_UNIT,
    REGION_GRID,
    SOURCE_REAL,
    HardwareSelection,
    InsufficientBudgetError,
    SelectorContractError,
    build_regions,
    group_l2_saliency,
    minmax_normalize,
    region_id_by_group,
    scores_from_precomputed,
    validate_boundary,
    validate_budget,
    validate_hidden,
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
#: Fixed screening100 budget / grid.
BUDGET = 96
#: This arm never applies a local candidate cap; the value is pinned to ``None``.
LOCAL_CANDIDATE_CAP = None
BOUNDARY = BOUNDARY_MAIN
#: Frozen label recorded in every record to identify the selection rule.
SELECTION_RULE = "unbounded_real_only_global_l2_3x3_anchor"
#: Independent arm-side label.
ARM_LABEL = "layer5_unbounded_real_only_block5"


@dataclass
class HardwareLayer5UnboundedL2Output:
    """Vision-side result of the layer-5 unbounded real-only L2 arm."""

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
    selection_rule: str = SELECTION_RULE
    uses_context_pooling: bool = False
    uses_context_slot_reservation: bool = False
    uses_local_candidate_cap: bool = False
    local_candidate_cap: None = None
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
    def identity_bypass(self) -> bool:
        return bool(self.selection.bypass)

    @property
    def block0_seconds(self) -> float:
        """B1-runner field-name alias for the pre-selector blocks."""
        return self.cut_blocks_seconds

    @property
    def vision_total_seconds(self) -> float:
        return (
            self.vision_preamble_seconds
            + self.cut_blocks_seconds
            + self.selector_cpu_seconds
            + self.remaining_vision_seconds
        )

    @property
    def gpu_vision_seconds(self) -> float:
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
            "the layer-5 unbounded L2 arm only supports "
            f"prune_after_layer={PRUNE_AFTER_LAYER}, got {value}"
        )
    return value


def _identity_bypass_real_only(
    hidden: np.ndarray, grid_h: int, grid_w: int, budget: int, boundary: str
) -> HardwareSelection:
    """All-real identity bypass for ``G <= K`` (frozen ``_identity_bypass`` shape)."""

    group_count = hidden.shape[0]
    group_indices = np.arange(group_count, dtype=np.int64)
    patch_indices = (
        group_indices[:, None] * MERGE_UNIT
        + np.arange(MERGE_UNIT, dtype=np.int64)
    )
    region_lookup = region_id_by_group(grid_h, grid_w)
    return HardwareSelection(
        boundary=boundary,
        budget=budget,
        bypass=True,
        groups=hidden.copy(),
        group_indices=group_indices,
        patch_indices=patch_indices,
        source_type=np.full((group_count,), SOURCE_REAL, dtype=np.uint8),
        region_id=np.array(
            [region_lookup[int(group)] for group in group_indices], dtype=np.int16
        ),
        valid=np.ones((group_count,), dtype=bool),
        scores=None,
        real_group_count=group_count,
        context_group_count=0,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        anchors=(),
        details=(),
        lambda_semantic=None,
    )


def select_unbounded_real_only_groups(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY_MAIN,
) -> HardwareSelection:
    """Select exactly ``budget`` real groups: anchors + global L2 ranking.

    No context is pooled, no context region is reserved and no local candidate
    cap is applied.  The two ranking rules are:

    1. one anchor per non-empty 3x3 region (highest combined score, lower index
       tie-break);
    2. the remaining ``K - anchors`` slots come from the global combined-score
       ranking over all non-anchor groups (ties resolve to the lower index).

    ``G <= budget`` returns the frozen-shaped all-real identity bypass.
    """

    hidden_array = validate_hidden(hidden)
    budget = validate_budget(budget)
    boundary = validate_boundary(boundary)
    group_count = hidden_array.shape[0]
    if grid_h * grid_w != group_count:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} merged groups"
        )
    if group_count <= budget:
        return _identity_bypass_real_only(
            hidden_array, grid_h, grid_w, budget, boundary
        )

    l2_scores = group_l2_saliency(hidden_array)
    combined = minmax_normalize(l2_scores)
    scores = scores_from_precomputed(l2=l2_scores, combined=combined)

    regions = build_regions(grid_h, grid_w, region_grid)
    non_empty = [region for region in regions if region.members]
    if budget < len(non_empty):
        raise InsufficientBudgetError(
            f"budget {budget} cannot cover one anchor in each of "
            f"{len(non_empty)} non-empty regions"
        )

    anchor_by_region = {
        region.region_id: int(
            max(region.members, key=lambda group: (combined[group], -group))
        )
        for region in non_empty
    }
    anchors = tuple(anchor_by_region[region.region_id] for region in non_empty)
    anchor_set = set(anchors)

    details_needed = budget - len(anchors)
    global_ranked = sorted(
        (group for group in range(group_count) if group not in anchor_set),
        key=lambda group: (-combined[group], group),
    )
    if len(global_ranked) < details_needed:
        raise InsufficientBudgetError(
            f"global ranking has {len(global_ranked)} non-anchor groups but "
            f"{details_needed} detail groups are required"
        )
    details = tuple(int(group) for group in global_ranked[:details_needed])
    selected = tuple(sorted(anchors + details))
    if len(selected) != budget:
        raise SelectorContractError(
            f"internal error: selected {len(selected)} real groups for budget {budget}"
        )

    region_lookup = region_id_by_group(grid_h, grid_w, region_grid)
    groups = np.stack([hidden_array[group].copy() for group in selected], axis=0)
    group_indices = np.array(selected, dtype=np.int64)
    patch_indices = (
        group_indices[:, None] * MERGE_UNIT
        + np.arange(MERGE_UNIT, dtype=np.int64)
    )
    source_type = np.full((budget,), SOURCE_REAL, dtype=np.uint8)
    region_id = np.array(
        [region_lookup[group] for group in selected], dtype=np.int16
    )
    valid = np.ones((budget,), dtype=bool)

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
        real_group_count=budget,
        context_group_count=0,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        anchors=anchors,
        details=details,
        lambda_semantic=None,
    )


def verify_real_only_selection(selection: HardwareSelection) -> bool:
    """Prove arm B kept the all-real / no-context / exact-K contract.

    Raises :class:`SelectorContractError` on any violation and returns ``True``
    otherwise.
    """

    if not isinstance(selection, HardwareSelection):
        raise SelectorContractError("selection must be a HardwareSelection")
    if int(selection.context_group_count) != 0:
        raise SelectorContractError(
            "unbounded real-only arm must not emit any context group, got "
            f"{selection.context_group_count}"
        )
    if np.any(selection.source_type != SOURCE_REAL):
        raise SelectorContractError(
            "unbounded real-only arm must emit only SOURCE_REAL slots"
        )
    if selection.bypass:
        if int(selection.output_group_count) != int(selection.real_group_count):
            raise SelectorContractError(
                "identity bypass must be all real and emit exactly G groups"
            )
    else:
        if int(selection.output_group_count) != int(selection.budget):
            raise SelectorContractError(
                f"non-bypass selection must emit exactly K={selection.budget} "
                f"groups, got {selection.output_group_count}"
            )
        if int(selection.real_group_count) != int(selection.budget):
            raise SelectorContractError(
                "non-bypass real group count must equal the requested budget"
            )
    group_indices = np.asarray(selection.group_indices, dtype=np.int64)
    if group_indices.size != int(selection.output_group_count):
        raise SelectorContractError(
            "group_indices length does not match the output group count"
        )
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise SelectorContractError(
            "unbounded real-only group_indices must be strictly ascending and unique"
        )
    if not np.all(selection.valid):
        raise SelectorContractError(
            "unbounded real-only selection produced an invalid group"
        )
    return True


def forward_hardware_regular_layer5_unbounded_l2_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    budget: int = BUDGET,
    score_mode: str = SCORE_MODE_L2,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY,
    prune_after_layer: int = PRUNE_AFTER_LAYER,
    synchronize: Callable[[], None] | None = None,
    projection: Any | None = None,
    text_embeddings: Any | None = None,
) -> HardwareLayer5UnboundedL2Output:
    """Run the layer-5 unbounded real-only L2 arm for one still image.

    Blocks ``0..5`` see the full patch sequence; the unbounded real-only selector
    is invoked once; the compact hidden is the raw selected real groups; blocks
    ``6..N-1`` and the DeepStack merger at index 5 then run on the compact
    sequence.
    """

    if score_mode not in ALLOWED_SCORE_MODES:
        raise SelectorContractError(
            f"score_mode must be one of {ALLOWED_SCORE_MODES}, got {score_mode!r}"
        )
    if projection is not None or text_embeddings is not None:
        raise SelectorContractError(
            "the layer-5 unbounded L2 arm is L2-only and must not use the "
            "B1-trained Wv projection or text embeddings"
        )
    prune_after_layer = _validate_cutpoint(prune_after_layer)
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError(
            "the layer-5 unbounded L2 arm currently supports one still image"
        )
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

    # --- Unbounded real-only L2 selection bridge (CPU numpy round trip) ----
    selector_started = _clock(synchronize)
    hidden_groups = hidden_states.reshape(full_groups, merge_unit, -1)
    hidden_np = hidden_groups.detach().to(torch.float64).cpu().numpy()
    budget_int = int(budget)
    selection = select_unbounded_real_only_groups(
        hidden_np,
        merged_h,
        merged_w,
        budget=budget_int,
        region_grid=region_grid,
        boundary=boundary,
    )
    verify_real_only_selection(selection)
    output_count = selection.output_group_count
    group_indices = np.asarray(selection.group_indices, dtype=np.int64)

    # All-real compact hidden: every output row is its raw 2x2 group.
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
    return HardwareLayer5UnboundedL2Output(
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


def prepare_layer5_unbounded_multimodal_inputs(
    model: torch.nn.Module,
    batch: Any,
    result: HardwareLayer5UnboundedL2Output,
) -> PrunedMultimodalInputs:
    """Reuse the full ``input_ids`` then keep the unique ascending groups."""

    group_indices = result.group_indices
    expected = int(result.pruned.image_embeds.shape[0])
    if group_indices.size != expected:
        raise ValueError(
            f"layer-5 unbounded arm produced {group_indices.size} groups but "
            f"{expected} image embeds"
        )
    if group_indices.size > 1 and not np.all(np.diff(group_indices) > 0):
        raise ValueError(
            "layer-5 unbounded group_indices must be strictly ascending and unique"
        )
    if int(result.valid.sum()) != expected:
        raise ValueError("layer-5 valid mask does not match the image token count")
    return prepare_pruned_multimodal_inputs(model, batch, result.pruned)


__all__ = [
    "ALLOWED_SCORE_MODES",
    "ARM_LABEL",
    "BUDGET",
    "BOUNDARY",
    "HardwareLayer5UnboundedL2Output",
    "LOCAL_CANDIDATE_CAP",
    "PRUNE_AFTER_LAYER",
    "REGION_GRID",
    "SCORE_MODE_L2",
    "SELECTION_RULE",
    "forward_hardware_regular_layer5_unbounded_l2_vision",
    "prepare_layer5_unbounded_multimodal_inputs",
    "select_unbounded_real_only_groups",
    "verify_real_only_selection",
]
