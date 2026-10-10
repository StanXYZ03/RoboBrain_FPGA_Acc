"""Layer-5 forced-index / control diagnostic ablation (CPU, real-only).

This is a *separate*, self-contained layer-5 cut-point integration.  The frozen
layer-5, context-ablation, B1 and unbounded-L2 modules are never edited and
their control flow is never imported.

Design contract
---------------
* One **unified** forward runs the whole layer-5 cut point.  ``control`` and
  ``forced`` share *all* of it; the only variable between the two modes is the
  resolved ``selected_group_indices`` (and the metadata derived from it).
* Blocks ``0..5`` run on the full patch sequence, every output slot is a
  complete real 2x2 merge group, the DeepStack merger whose index equals block 5
  is called on the **compact** hidden, blocks ``6..N-1`` run on the compact
  sequence and the final merger closes the vision tower.
* ``control``: given ``K_C`` (any positive integer, not restricted to a multiple
  of 16), keep one anchor per non-empty 3x3 region (highest combined L2 score,
  lower index tie-break) plus the global L2 ranking ``(-score, index)`` over the
  remaining non-anchor groups, until exactly ``K_C`` groups are kept.
* ``forced``: keep exactly the historical indices.  The list must be
  one-dimensional, integer, of length ``K_C``, unique and inside
  ``[0, G)``; it is deterministically sorted ascending.
* All output slots are real; there is no context pooling, no context slot
  reservation and no ``local_candidate_cap``.  ``G <= K_C`` is an identity
  bypass.
* This is a diagnostic arm.  ``hardware_alignment_relaxed_for_diagnostic`` is
  ``True``, ``causal_diagnostic_only`` is ``True`` and
  ``fpga_deployable_candidate`` is ``False``: the relaxed ``K_C`` alignment is
  explicitly *not* an FPGA-deployable configuration.

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
    validate_hidden,
)
from evaluation.shallow_block_pruning import PrunedVisionOutput

#: The only supported cut point: after the sixth vision block (index 5).
PRUNE_AFTER_LAYER = 5
#: Default control budget (any positive integer is accepted at runtime).
BUDGET = 96
BOUNDARY = BOUNDARY_MAIN
#: Unified-forward modes.  The only variable between them is the resolved
#: ``selected_group_indices``.
MODE_CONTROL = "control"
MODE_FORCED = "forced"
ALLOWED_MODES = (MODE_CONTROL, MODE_FORCED)
#: Frozen labels recorded in every output to identify the selection rule.
SELECTION_RULE_CONTROL = "control_global_l2_plus_3x3_anchor"
SELECTION_RULE_FORCED = "forced_exact_historical_group_indices"
ARM_LABEL = "layer5_forced_index_ablation_block5"
#: Diagnostic metadata flags (see the module docstring).
HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC = True
CAUSAL_DIAGNOSTIC_ONLY = True
FPGA_DEPLOYABLE_CANDIDATE = False


@dataclass
class HardwareLayer5ForcedIndexAblationOutput:
    """Vision-side result of the layer-5 forced-index / control ablation."""

    pruned: PrunedVisionOutput
    selection: HardwareSelection
    mode: str
    full_patch_tokens: int
    full_llm_visual_tokens: int
    budget: int
    merged_h: int
    merged_w: int
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
    hardware_alignment_relaxed_for_diagnostic: bool = (
        HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC
    )
    causal_diagnostic_only: bool = CAUSAL_DIAGNOSTIC_ONLY
    fpga_deployable_candidate: bool = FPGA_DEPLOYABLE_CANDIDATE
    uses_context_pooling: bool = False
    uses_context_slot_reservation: bool = False
    uses_local_candidate_cap: bool = False
    local_candidate_cap: None = None
    deepstack_cut_merger_index: int | None = None
    deepstack_pre_cut_count: int = 0
    deepstack_merger_indexes: tuple[int, ...] = field(default_factory=tuple)

    @property
    def selection_rule(self) -> str:
        if self.mode == MODE_FORCED:
            return SELECTION_RULE_FORCED
        return SELECTION_RULE_CONTROL

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
            "the layer-5 forced-index ablation only supports "
            f"prune_after_layer={PRUNE_AFTER_LAYER}, got {value}"
        )
    return value


def validate_diagnostic_budget(budget: Any) -> int:
    """Validate ``K_C``: any positive integer (alignment deliberately relaxed)."""
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)):
        raise SelectorContractError("budget must be an integer")
    value = int(budget)
    if value <= 0:
        raise SelectorContractError("budget must be positive")
    return value


def _make_selection(
    hidden_array: np.ndarray,
    grid_h: int,
    grid_w: int,
    selected: Any,
    *,
    budget: int,
    bypass: bool,
    boundary: str,
    region_grid: int,
    scores: Any | None,
    anchors: tuple[int, ...] = (),
    details: tuple[int, ...] = (),
) -> HardwareSelection:
    """Build the all-real, no-context ``HardwareSelection`` (shared builder)."""
    selected_array = np.asarray(selected, dtype=np.int64).reshape(-1)
    if selected_array.size:
        groups = np.stack(
            [hidden_array[int(group)].copy() for group in selected_array], axis=0
        )
    else:  # pragma: no cover - guarded by the selectors below
        groups = np.zeros(
            (0, MERGE_UNIT, hidden_array.shape[-1]), dtype=hidden_array.dtype
        )
    patch_indices = (
        selected_array[:, None] * MERGE_UNIT
        + np.arange(MERGE_UNIT, dtype=np.int64)
    )
    region_lookup = region_id_by_group(int(grid_h), int(grid_w), region_grid)
    region_id = np.array(
        [region_lookup[int(group)] for group in selected_array], dtype=np.int16
    )
    return HardwareSelection(
        boundary=boundary,
        budget=int(budget),
        bypass=bool(bypass),
        groups=groups,
        group_indices=selected_array,
        patch_indices=patch_indices,
        source_type=np.full((selected_array.size,), SOURCE_REAL, dtype=np.uint8),
        region_id=region_id,
        valid=np.ones((selected_array.size,), dtype=bool),
        scores=scores,
        real_group_count=int(selected_array.size),
        context_group_count=0,
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        anchors=tuple(anchors),
        details=tuple(details),
        lambda_semantic=None,
    )


def _identity_bypass_real_only(
    hidden_array: np.ndarray,
    grid_h: int,
    grid_w: int,
    budget: int,
    boundary: str,
    region_grid: int,
) -> HardwareSelection:
    group_count = hidden_array.shape[0]
    return _make_selection(
        hidden_array,
        grid_h,
        grid_w,
        np.arange(group_count, dtype=np.int64),
        budget=budget,
        bypass=True,
        boundary=boundary,
        region_grid=region_grid,
        scores=None,
    )


def select_control_groups(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY_MAIN,
) -> HardwareSelection:
    """Control rule: global L2 ranking + one anchor per non-empty 3x3 region.

    ``G <= budget`` returns the frozen all-real identity bypass so the two modes
    share the same edge case.
    """
    hidden_array = validate_hidden(hidden)
    budget = validate_diagnostic_budget(budget)
    boundary = validate_boundary(boundary)
    group_count = hidden_array.shape[0]
    if int(grid_h) * int(grid_w) != group_count:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} merged groups"
        )
    if group_count <= budget:
        return _identity_bypass_real_only(
            hidden_array, grid_h, grid_w, budget, boundary, region_grid
        )

    l2_scores = group_l2_saliency(hidden_array)
    combined = minmax_normalize(l2_scores)
    scores = scores_from_precomputed(l2=l2_scores, combined=combined)

    regions = [region for region in build_regions(grid_h, grid_w, region_grid) if region.members]
    if budget < len(regions):
        raise InsufficientBudgetError(
            f"budget {budget} cannot cover one anchor in each of "
            f"{len(regions)} non-empty regions"
        )
    anchors = tuple(
        int(max(region.members, key=lambda group: (combined[group], -group)))
        for region in regions
    )
    anchor_set = set(anchors)
    details_needed = budget - len(anchors)
    ranked = sorted(
        (group for group in range(group_count) if group not in anchor_set),
        key=lambda group: (-combined[group], group),
    )
    if len(ranked) < details_needed:
        raise InsufficientBudgetError(
            f"global ranking has {len(ranked)} non-anchor groups but "
            f"{details_needed} detail groups are required"
        )
    details = tuple(int(group) for group in ranked[:details_needed])
    selected = np.array(sorted(anchors + details), dtype=np.int64)
    if selected.size != budget:
        raise SelectorContractError(
            f"internal error: selected {selected.size} real groups for budget {budget}"
        )
    return _make_selection(
        hidden_array,
        grid_h,
        grid_w,
        selected,
        budget=budget,
        bypass=False,
        boundary=boundary,
        region_grid=region_grid,
        scores=scores,
        anchors=anchors,
        details=details,
    )


def normalize_forced_group_indices(
    forced_group_indices: Any, *, budget: int, group_count: int
) -> np.ndarray:
    """Validate and deterministically sort the forced historical indices.

    The array must be one-dimensional with an integer dtype, exactly ``budget``
    entries long, unique and inside ``[0, group_count)``.  The returned array is
    ascending.
    """
    budget = validate_diagnostic_budget(budget)
    array = np.asarray(forced_group_indices)
    if array.ndim != 1:
        raise SelectorContractError(
            f"forced_group_indices must be one-dimensional, got ndim={array.ndim}"
        )
    if array.dtype == bool or not np.issubdtype(array.dtype, np.integer):
        raise SelectorContractError(
            f"forced_group_indices must have an integer dtype, got {array.dtype}"
        )
    if int(array.size) != budget:
        raise SelectorContractError(
            f"forced_group_indices must have length K={budget}, got {array.size}"
        )
    values = array.astype(np.int64)
    if np.unique(values).size != values.size:
        raise SelectorContractError("forced_group_indices must be unique")
    if int(values.min()) < 0 or int(values.max()) >= int(group_count):
        raise SelectorContractError(
            f"forced_group_indices must lie in [0, {group_count}), got "
            f"min={int(values.min())} max={int(values.max())}"
        )
    return np.sort(values)


def select_forced_groups(
    hidden: Any,
    grid_h: int,
    grid_w: int,
    *,
    budget: int,
    forced_group_indices: Any,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY_MAIN,
) -> HardwareSelection:
    """Forced rule: keep exactly the historical indices (sorted ascending)."""
    hidden_array = validate_hidden(hidden)
    budget = validate_diagnostic_budget(budget)
    boundary = validate_boundary(boundary)
    group_count = hidden_array.shape[0]
    if int(grid_h) * int(grid_w) != group_count:
        raise SelectorContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} merged groups"
        )
    if group_count <= budget:
        return _identity_bypass_real_only(
            hidden_array, grid_h, grid_w, budget, boundary, region_grid
        )
    selected = normalize_forced_group_indices(
        forced_group_indices, budget=budget, group_count=group_count
    )
    # Scores are computed uniformly so that a forced selection with the same
    # index set as a control selection produces a bit-identical output.
    l2_scores = group_l2_saliency(hidden_array)
    combined = minmax_normalize(l2_scores)
    scores = scores_from_precomputed(l2=l2_scores, combined=combined)
    return _make_selection(
        hidden_array,
        grid_h,
        grid_w,
        selected,
        budget=budget,
        bypass=False,
        boundary=boundary,
        region_grid=region_grid,
        scores=scores,
    )


def verify_diagnostic_selection(selection: HardwareSelection) -> bool:
    """Prove the all-real / no-context / exact-``K_C`` diagnostic contract."""
    if not isinstance(selection, HardwareSelection):
        raise SelectorContractError("selection must be a HardwareSelection")
    if int(selection.context_group_count) != 0:
        raise SelectorContractError(
            "the forced-index ablation must not emit any context group, got "
            f"{selection.context_group_count}"
        )
    if np.any(selection.source_type != SOURCE_REAL):
        raise SelectorContractError(
            "the forced-index ablation must emit only SOURCE_REAL slots"
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
            "group_indices must be strictly ascending and unique"
        )
    if not np.all(selection.valid):
        raise SelectorContractError("selection produced an invalid group")
    return True


def _unified_forward(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    resolve_selection: Callable[[np.ndarray, int, int], HardwareSelection],
    budget: int,
    mode: str,
    prune_after_layer: int = PRUNE_AFTER_LAYER,
    synchronize: Callable[[], None] | None = None,
) -> HardwareLayer5ForcedIndexAblationOutput:
    """The one forward shared by control and forced modes.

    ``resolve_selection(hidden_groups_np, merged_h, merged_w)`` is the *only*
    mode-dependent step; it returns the all-real ``HardwareSelection`` whose
    ``selected_group_indices`` is the sole experimental variable.
    """
    prune_after_layer = _validate_cutpoint(prune_after_layer)
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError(
            "the layer-5 forced-index ablation currently supports one still image"
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

    # --- Selection bridge (CPU numpy round trip); the only variable ---------
    selector_started = _clock(synchronize)
    hidden_groups = hidden_states.reshape(full_groups, merge_unit, -1)
    hidden_np = hidden_groups.detach().to(torch.float64).cpu().numpy()
    selection = resolve_selection(hidden_np, merged_h, merged_w)
    verify_diagnostic_selection(selection)
    output_count = selection.output_group_count
    group_indices = np.asarray(selection.group_indices, dtype=np.int64)

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
    return HardwareLayer5ForcedIndexAblationOutput(
        pruned=pruned,
        selection=selection,
        mode=mode,
        full_patch_tokens=seq_len,
        full_llm_visual_tokens=full_groups,
        budget=int(budget),
        merged_h=merged_h,
        merged_w=merged_w,
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


def forward_hardware_regular_layer5_forced_index_ablation_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    mode: str = MODE_CONTROL,
    budget: int = BUDGET,
    forced_group_indices: Any | None = None,
    region_grid: int = REGION_GRID,
    boundary: str = BOUNDARY,
    prune_after_layer: int = PRUNE_AFTER_LAYER,
    synchronize: Callable[[], None] | None = None,
) -> HardwareLayer5ForcedIndexAblationOutput:
    """Run the unified layer-5 diagnostic forward for one still image.

    ``mode="control"`` selects ``budget`` groups with the control rule;
    ``mode="forced"`` keeps exactly ``forced_group_indices``.  Both modes run the
    same code path after the resolved indices.
    """
    budget = validate_diagnostic_budget(budget)
    boundary = validate_boundary(boundary)
    if mode == MODE_CONTROL:
        if forced_group_indices is not None:
            raise SelectorContractError(
                "control mode must not receive forced_group_indices"
            )

        def resolve_selection(
            hidden_np: np.ndarray, grid_h: int, grid_w: int
        ) -> HardwareSelection:
            return select_control_groups(
                hidden_np,
                grid_h,
                grid_w,
                budget=budget,
                region_grid=region_grid,
                boundary=boundary,
            )

    elif mode == MODE_FORCED:
        if forced_group_indices is None:
            raise SelectorContractError(
                "forced mode requires forced_group_indices"
            )

        def resolve_selection(
            hidden_np: np.ndarray, grid_h: int, grid_w: int
        ) -> HardwareSelection:
            return select_forced_groups(
                hidden_np,
                grid_h,
                grid_w,
                budget=budget,
                forced_group_indices=forced_group_indices,
                region_grid=region_grid,
                boundary=boundary,
            )

    else:
        raise SelectorContractError(
            f"mode must be one of {ALLOWED_MODES}, got {mode!r}"
        )

    return _unified_forward(
        visual,
        pixel_values,
        grid_thw,
        resolve_selection=resolve_selection,
        budget=budget,
        mode=mode,
        prune_after_layer=prune_after_layer,
        synchronize=synchronize,
    )


__all__ = [
    "ALLOWED_MODES",
    "ARM_LABEL",
    "BUDGET",
    "BOUNDARY",
    "CAUSAL_DIAGNOSTIC_ONLY",
    "FPGA_DEPLOYABLE_CANDIDATE",
    "HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC",
    "HardwareLayer5ForcedIndexAblationOutput",
    "MODE_CONTROL",
    "MODE_FORCED",
    "PRUNE_AFTER_LAYER",
    "REGION_GRID",
    "SELECTION_RULE_CONTROL",
    "SELECTION_RULE_FORCED",
    "forward_hardware_regular_layer5_forced_index_ablation_vision",
    "normalize_forced_group_indices",
    "select_control_groups",
    "select_forced_groups",
    "validate_diagnostic_budget",
    "verify_diagnostic_selection",
]
