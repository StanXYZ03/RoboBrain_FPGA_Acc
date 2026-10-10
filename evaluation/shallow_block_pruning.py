"""Single-pass shallow structured visual-token pruning for Qwen3-VL.

This module intentionally lives outside Transformers.  It reuses the loaded
model's original layers and weights, prunes only complete 2x2 patch groups,
and indexes the original learned/rotary positions instead of renumbering the
surviving tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from evaluation.joint_deletion import (
    candidate_interaction_features,
    candidate_set_utilities,
    region_preserving_swap_candidates,
)
from evaluation.learned_selector import normalize_per_image, scalar_selector_features


@dataclass
class PrunedVisionOutput:
    image_embeds: torch.Tensor
    deepstack_embeds: list[torch.Tensor]
    selected_group_indices: torch.Tensor
    selected_patch_indices: torch.Tensor
    group_scores: torch.Tensor
    full_patch_tokens: int
    full_llm_visual_tokens: int
    prune_hidden_states: torch.Tensor | None = None
    group_residual: torch.Tensor | None = None
    group_energy: torch.Tensor | None = None
    group_l2: torch.Tensor | None = None
    joint_candidate_id: str | None = None
    joint_candidate_utility: float | None = None


@dataclass
class PrunedMultimodalInputs:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    visual_pos_masks: torch.Tensor
    deepstack_visual_embeds: list[torch.Tensor]
    original_sequence_length: int


def validate_fixed_group_indices(
    indices: torch.Tensor,
    group_count: int,
    keep_count: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Validate and canonicalize an externally supplied fixed-token mask."""
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise ValueError("fixed group indices must be a one-dimensional tensor")
    if indices.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("fixed group indices must use an integer dtype")
    if group_count <= 0 or not (1 <= keep_count <= group_count):
        raise ValueError("invalid fixed-index group budget")
    if indices.numel() != keep_count:
        raise ValueError(
            f"fixed group indices contain {indices.numel()} entries; expected {keep_count}"
        )
    canonical = torch.sort(indices.to(device=device, dtype=torch.long)).values
    if canonical.numel() > 1 and bool((canonical[1:] == canonical[:-1]).any()):
        raise ValueError("fixed group indices must be unique")
    if int(canonical[0].item()) < 0 or int(canonical[-1].item()) >= group_count:
        raise ValueError("fixed group index is outside the merged visual grid")
    return canonical


def structured_group_indices(
    scores: torch.Tensor,
    grid_h: int,
    grid_w: int,
    retention: float,
    *,
    anchor_grid: int = 3,
) -> torch.Tensor:
    """Select high-L2 2x2 groups plus one anchor from each coarse region."""
    if scores.ndim != 1 or scores.numel() != grid_h * grid_w:
        raise ValueError("scores must match the merged 2D grid")
    if not (0.0 < retention <= 1.0):
        raise ValueError("retention must be in (0, 1]")
    if anchor_grid <= 0:
        raise ValueError("anchor_grid must be positive")
    group_count = scores.numel()
    keep_count = max(1, min(group_count, math.ceil(group_count * retention)))
    if keep_count == group_count:
        return torch.arange(group_count, device=scores.device, dtype=torch.long)

    anchor_rows = min(anchor_grid, grid_h, max(1, int(math.sqrt(keep_count))))
    anchor_cols = min(anchor_grid, grid_w, max(1, keep_count // anchor_rows))
    while anchor_rows * anchor_cols > keep_count:
        if anchor_cols >= anchor_rows and anchor_cols > 1:
            anchor_cols -= 1
        elif anchor_rows > 1:
            anchor_rows -= 1
        else:
            break

    anchors: list[int] = []
    score_grid = scores.view(grid_h, grid_w)
    for region_r in range(anchor_rows):
        r0 = region_r * grid_h // anchor_rows
        r1 = (region_r + 1) * grid_h // anchor_rows
        for region_c in range(anchor_cols):
            c0 = region_c * grid_w // anchor_cols
            c1 = (region_c + 1) * grid_w // anchor_cols
            region = score_grid[r0:r1, c0:c1].reshape(-1)
            local = int(torch.argmax(region).item())
            width = c1 - c0
            anchors.append((r0 + local // width) * grid_w + c0 + local % width)

    selected = set(anchors)
    # Stable sorting makes ties deterministic and favors the lower spatial ID.
    ranking = torch.argsort(scores, descending=True, stable=True).tolist()
    for index in ranking:
        selected.add(int(index))
        if len(selected) == keep_count:
            break
    return torch.tensor(sorted(selected), device=scores.device, dtype=torch.long)


def geometric_oracle_group_scores(
    grid_h: int,
    grid_w: int,
    bbox_1000: tuple[float, float, float, float] | list[float],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Score merged visual groups using the labeled target box.

    This is deliberately an unattainable oracle used only as a feasibility
    bound.  Cells intersecting the target are ranked first by overlap; the
    remaining cells are ranked by distance to the box so that extra budget
    preserves nearby context.  No model activation is consumed.
    """
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("oracle grid dimensions must be positive")
    if len(bbox_1000) != 4:
        raise ValueError("oracle bbox must contain four coordinates")
    x1, y1, x2, y2 = (float(value) for value in bbox_1000)
    if not (0.0 <= x1 < x2 <= 1000.0 and 0.0 <= y1 < y2 <= 1000.0):
        raise ValueError("oracle bbox must be a non-empty box in normalized 0-1000 space")

    rows = torch.arange(grid_h, dtype=torch.float32, device=device)
    cols = torch.arange(grid_w, dtype=torch.float32, device=device)
    cell_x1 = cols[None, :] * (1000.0 / grid_w)
    cell_x2 = (cols[None, :] + 1.0) * (1000.0 / grid_w)
    cell_y1 = rows[:, None] * (1000.0 / grid_h)
    cell_y2 = (rows[:, None] + 1.0) * (1000.0 / grid_h)
    overlap_w = (torch.minimum(cell_x2, torch.tensor(x2, device=device)) -
                 torch.maximum(cell_x1, torch.tensor(x1, device=device))).clamp_min(0.0)
    overlap_h = (torch.minimum(cell_y2, torch.tensor(y2, device=device)) -
                 torch.maximum(cell_y1, torch.tensor(y1, device=device))).clamp_min(0.0)
    overlap_fraction = overlap_w * overlap_h / ((1000.0 / grid_w) * (1000.0 / grid_h))

    center_x = (cell_x1 + cell_x2) * 0.5
    center_y = (cell_y1 + cell_y2) * 0.5
    dx = torch.maximum(torch.maximum(torch.tensor(x1, device=device) - center_x,
                                     center_x - torch.tensor(x2, device=device)),
                       torch.zeros((), device=device))
    dy = torch.maximum(torch.maximum(torch.tensor(y1, device=device) - center_y,
                                     center_y - torch.tensor(y2, device=device)),
                       torch.zeros((), device=device))
    diagonal = math.hypot(1000.0 / grid_w, 1000.0 / grid_h)
    proximity = torch.exp(-torch.sqrt(dx.square() + dy.square()) / max(diagonal, 1e-6))
    # The +2 term guarantees that every intersecting cell precedes every
    # non-intersecting cell; proximity only orders the remaining context.
    return (proximity + 2.0 * (overlap_fraction > 0).float() + overlap_fraction).reshape(-1)


def geometric_oracle_group_indices(
    grid_h: int,
    grid_w: int,
    retention: float,
    bbox_1000: tuple[float, float, float, float] | list[float],
    *,
    anchor_grid: int = 3,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-aware oracle groups plus their auditable score vector.

    All target-intersecting cells are retained first when the budget permits.
    Remaining capacity is filled with coarse spatial anchors and then nearby
    context.  This avoids allowing the coverage rule to evict the known target.
    """
    scores = geometric_oracle_group_scores(grid_h, grid_w, bbox_1000, device=device)
    keep_count = max(1, min(scores.numel(), math.ceil(scores.numel() * retention)))
    if keep_count == scores.numel():
        return torch.arange(scores.numel(), device=scores.device), scores

    target_ranking = torch.argsort(scores, descending=True, stable=True)
    target_ranking = target_ranking[scores.index_select(0, target_ranking) >= 2.0]
    selected_set = set(int(index) for index in target_ranking[:keep_count].tolist())

    if len(selected_set) < keep_count:
        anchors = structured_group_indices(
            scores, grid_h, grid_w, min(1.0, keep_count / scores.numel()),
            anchor_grid=anchor_grid,
        )
        for index in anchors.tolist():
            selected_set.add(int(index))
            if len(selected_set) == keep_count:
                break
    if len(selected_set) < keep_count:
        for index in torch.argsort(scores, descending=True, stable=True).tolist():
            selected_set.add(int(index))
            if len(selected_set) == keep_count:
                break
    selected = torch.tensor(sorted(selected_set), device=scores.device, dtype=torch.long)
    return selected, scores


def spatial_residual_energy_features(
    hidden_states: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_unit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw local residual and activation energy for each merger group.

    The stencil only uses the four neighboring merger groups, so it can be
    implemented with line buffers and reductions rather than global attention.
    """
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be [patch_tokens, hidden_size]")
    if hidden_states.shape[0] != grid_h * grid_w * merge_unit:
        raise ValueError("hidden_states do not match the merged grid")
    features = hidden_states.view(grid_h, grid_w, merge_unit, -1).float().mean(dim=2)
    neighbor_sum = torch.zeros_like(features)
    neighbor_count = torch.zeros((grid_h, grid_w, 1), device=features.device)
    neighbor_sum[1:] += features[:-1]
    neighbor_count[1:] += 1
    neighbor_sum[:-1] += features[1:]
    neighbor_count[:-1] += 1
    neighbor_sum[:, 1:] += features[:, :-1]
    neighbor_count[:, 1:] += 1
    neighbor_sum[:, :-1] += features[:, 1:]
    neighbor_count[:, :-1] += 1
    neighbor_mean = neighbor_sum / neighbor_count.clamp_min(1)
    residual = (features - neighbor_mean).square().mean(dim=-1).sqrt()
    energy = features.square().mean(dim=-1).sqrt()
    return residual.reshape(-1), energy.reshape(-1)


def spatial_residual_group_scores(
    hidden_states: torch.Tensor,
    grid_h: int,
    grid_w: int,
    merge_unit: int,
    *,
    energy_weight: float = 0.25,
) -> torch.Tensor:
    """Score spatial novelty with a local stencil plus a small energy term."""
    if not (0.0 <= energy_weight <= 1.0):
        raise ValueError("energy_weight must be in [0, 1]")
    residual, energy = spatial_residual_energy_features(
        hidden_states, grid_h, grid_w, merge_unit
    )
    eps = torch.finfo(residual.dtype).eps
    residual = residual / residual.mean().clamp_min(eps)
    energy = energy / energy.mean().clamp_min(eps)
    return (1.0 - energy_weight) * residual + energy_weight * energy


def regional_quota_indices(
    scores: torch.Tensor,
    grid_h: int,
    grid_w: int,
    retention: float,
    *,
    region_grid: int = 3,
) -> torch.Tensor:
    """Allocate a minimum spatial quota, then distribute budget by score mass."""
    if scores.ndim != 1 or scores.numel() != grid_h * grid_w:
        raise ValueError("scores must match the merged 2D grid")
    if not (0.0 < retention <= 1.0):
        raise ValueError("retention must be in (0, 1]")
    if region_grid <= 0:
        raise ValueError("region_grid must be positive")
    keep_count = max(1, min(scores.numel(), math.ceil(scores.numel() * retention)))
    if keep_count == scores.numel():
        return torch.arange(scores.numel(), device=scores.device, dtype=torch.long)

    region_rows = min(region_grid, grid_h, max(1, int(math.sqrt(keep_count))))
    region_cols = min(region_grid, grid_w, max(1, keep_count // region_rows))
    while region_rows * region_cols > keep_count:
        if region_cols >= region_rows and region_cols > 1:
            region_cols -= 1
        elif region_rows > 1:
            region_rows -= 1

    regions: list[torch.Tensor] = []
    masses: list[float] = []
    capacities: list[int] = []
    score_grid = scores.view(grid_h, grid_w)
    score_floor = float(scores.min().item())
    for region_r in range(region_rows):
        r0 = region_r * grid_h // region_rows
        r1 = (region_r + 1) * grid_h // region_rows
        for region_c in range(region_cols):
            c0 = region_c * grid_w // region_cols
            c1 = (region_c + 1) * grid_w // region_cols
            rows = torch.arange(r0, r1, device=scores.device)[:, None]
            cols = torch.arange(c0, c1, device=scores.device)[None, :]
            indices = (rows * grid_w + cols).reshape(-1)
            regions.append(indices)
            region_scores = score_grid[r0:r1, c0:c1]
            masses.append(float((region_scores - score_floor).sum().item()) + 1e-12)
            capacities.append(indices.numel())

    quotas = [1] * len(regions)
    for _ in range(keep_count - len(regions)):
        candidates = [
            (masses[index] / (quotas[index] + 1), -index, index)
            for index in range(len(regions))
            if quotas[index] < capacities[index]
        ]
        if not candidates:
            raise RuntimeError("regional quota allocation exhausted capacity")
        quotas[max(candidates)[2]] += 1

    selected: list[int] = []
    for indices, quota in zip(regions, quotas):
        local_scores = scores.index_select(0, indices)
        ranking = torch.argsort(local_scores, descending=True, stable=True)[:quota]
        selected.extend(int(index) for index in indices.index_select(0, ranking).tolist())
    return torch.tensor(sorted(selected), device=scores.device, dtype=torch.long)


def text_relevance_group_scores(
    projected_groups: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Return maximum cosine relevance of every visual group to prompt tokens."""
    if projected_groups.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("projected_groups and text_embeddings must be matrices")
    if projected_groups.shape[1] != text_embeddings.shape[1]:
        raise ValueError("visual and text features must share the LLM hidden dimension")
    if text_embeddings.shape[0] == 0:
        raise ValueError("at least one text embedding is required")
    visual = F.normalize(projected_groups.float(), dim=-1)
    text = F.normalize(text_embeddings.float(), dim=-1)
    return (visual @ text.transpose(0, 1)).amax(dim=1)


def text_residual_group_scores(
    projected_groups: torch.Tensor,
    text_embeddings: torch.Tensor,
    residual_scores: torch.Tensor,
    *,
    text_weight: float,
) -> torch.Tensor:
    """Fuse prompt relevance with local residual using the existing DeepStack space."""
    if residual_scores.shape != (projected_groups.shape[0],):
        raise ValueError("residual_scores must have one value per visual group")
    if not (0.0 <= text_weight <= 1.0):
        raise ValueError("text_weight must be in [0, 1]")
    relevance = text_relevance_group_scores(projected_groups, text_embeddings)
    relevance = relevance - relevance.min()
    relevance = relevance / relevance.mean().clamp_min(torch.finfo(relevance.dtype).eps)
    return (1.0 - text_weight) * residual_scores.float() + text_weight * relevance


def forward_shallow_pruned_vision(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    retention: float,
    prune_after_layer: int = 5,
    anchor_grid: int = 3,
    selector: str = "l2_anchor",
    residual_energy_weight: float = 0.25,
    text_embeddings: torch.Tensor | None = None,
    text_weight: float = 0.5,
    capture_prune_gradient: bool = False,
    capture_selector_features: bool = False,
    selector_head: torch.nn.Module | None = None,
    selector_feature_mask: tuple[int, ...] = (0, 1, 2, 3, 4),
    fixed_group_indices: torch.Tensor | None = None,
    joint_candidate_seed: int | None = None,
    joint_swap_counts: tuple[int, ...] = (1, 2, 4),
    joint_candidates_per_swap: int = 2,
    candidate_task_value: float | None = None,
    oracle_bbox_1000: tuple[float, float, float, float] | list[float] | None = None,
) -> PrunedVisionOutput:
    """Run the original vision model, pruning complete groups at one cut point.

    ``prune_after_layer=-1`` means immediately after patch embedding.  Layer
    indices are zero-based; the last legal index is ``len(blocks)-1``.
    """
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("prototype currently supports one still image per request")
    if not (-1 <= prune_after_layer < len(visual.blocks)):
        raise ValueError("prune_after_layer must be -1 or index a vision layer")
    if prune_after_layer == -1 and capture_prune_gradient:
        raise ValueError("gradient capture after patch embedding is not supported")

    hidden_states = visual.patch_embed(pixel_values.type(visual.dtype))
    pos_embeds = visual.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds
    rotary_pos_emb = visual.rot_pos_emb(grid_thw)
    seq_len = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    cu_seqlens = F.pad(
        torch.tensor([seq_len], dtype=torch.int32, device=hidden_states.device),
        (1, 0),
        value=0,
    )

    merge_size = int(visual.spatial_merge_size)
    merge_unit = merge_size * merge_size
    full_groups = seq_len // merge_unit
    merged_h = int(grid_thw[0, 1].item()) // merge_size
    merged_w = int(grid_thw[0, 2].item()) // merge_size
    if full_groups != merged_h * merged_w:
        raise ValueError("vision token order is not merge-grid aligned")

    selected_groups: torch.Tensor | None = None
    selected_patches: torch.Tensor | None = None
    group_scores: torch.Tensor | None = None
    prune_hidden_states: torch.Tensor | None = None
    group_residual: torch.Tensor | None = None
    group_energy: torch.Tensor | None = None
    group_l2: torch.Tensor | None = None
    joint_candidate_id: str | None = None
    joint_candidate_utility: float | None = None
    deepstack_features: list[torch.Tensor] = []

    # Patch-embedding cut.  It is intentionally limited to masks that do not
    # require a later feature, which keeps the semantics explicit.
    if prune_after_layer == -1:
        keep_count = max(1, min(full_groups, math.ceil(full_groups * retention)))
        if selector == "fixed_indices":
            if fixed_group_indices is None:
                raise ValueError("fixed_indices selector requires fixed group indices")
            selected_groups = validate_fixed_group_indices(
                fixed_group_indices, full_groups, keep_count, device=hidden_states.device
            )
            group_scores = hidden_states.view(full_groups, merge_unit, -1).float().norm(
                dim=-1
            ).mean(dim=1)
        elif selector == "geometric_oracle":
            if oracle_bbox_1000 is None:
                raise ValueError("geometric_oracle requires oracle_bbox_1000")
            selected_groups, group_scores = geometric_oracle_group_indices(
                merged_h,
                merged_w,
                retention,
                oracle_bbox_1000,
                anchor_grid=anchor_grid,
                device=hidden_states.device,
            )
        else:
            raise ValueError(
                "patch-embedding cut supports only fixed_indices or geometric_oracle"
            )
        offsets = torch.arange(merge_unit, device=hidden_states.device)
        selected_patches = (
            selected_groups[:, None] * merge_unit + offsets[None, :]
        ).reshape(-1)
        hidden_states = hidden_states.index_select(0, selected_patches)
        position_embeddings = (
            position_embeddings[0].index_select(0, selected_patches),
            position_embeddings[1].index_select(0, selected_patches),
        )
        cu_seqlens = torch.tensor(
            [0, selected_patches.numel()], dtype=torch.int32, device=hidden_states.device
        )

    for layer_num, block in enumerate(visual.blocks):
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_num == prune_after_layer and capture_prune_gradient:
            hidden_states = hidden_states.detach().requires_grad_(True)
            hidden_states.retain_grad()
            prune_hidden_states = hidden_states
        feature: torch.Tensor | None = None
        if layer_num in visual.deepstack_visual_indexes:
            merger_index = visual.deepstack_visual_indexes.index(layer_num)
            feature = visual.deepstack_merger_list[merger_index](hidden_states)

        if layer_num == prune_after_layer:
            if capture_selector_features:
                group_residual, group_energy = spatial_residual_energy_features(
                    hidden_states, merged_h, merged_w, merge_unit
                )
                group_l2 = hidden_states.view(full_groups, merge_unit, -1).float().norm(
                    dim=-1
                ).mean(dim=1)
            if selector == "fixed_indices":
                if fixed_group_indices is None:
                    raise ValueError("fixed_indices selector requires fixed group indices")
                keep_count = max(
                    1, min(full_groups, math.ceil(full_groups * retention))
                )
                selected_groups = validate_fixed_group_indices(
                    fixed_group_indices,
                    full_groups,
                    keep_count,
                    device=hidden_states.device,
                )
                group_scores = hidden_states.view(
                    full_groups, merge_unit, -1
                ).float().norm(dim=-1).mean(dim=1)
            elif selector == "l2_anchor":
                group_scores = hidden_states.view(full_groups, merge_unit, -1).float().norm(dim=-1).mean(dim=1)
                selected_groups = structured_group_indices(
                    group_scores,
                    merged_h,
                    merged_w,
                    retention,
                    anchor_grid=anchor_grid,
                )
            elif selector == "geometric_oracle":
                if oracle_bbox_1000 is None:
                    raise ValueError("geometric_oracle requires oracle_bbox_1000")
                selected_groups, group_scores = geometric_oracle_group_indices(
                    merged_h,
                    merged_w,
                    retention,
                    oracle_bbox_1000,
                    anchor_grid=anchor_grid,
                    device=hidden_states.device,
                )
            elif selector == "residual_quota":
                group_scores = spatial_residual_group_scores(
                    hidden_states,
                    merged_h,
                    merged_w,
                    merge_unit,
                    energy_weight=residual_energy_weight,
                )
                selected_groups = regional_quota_indices(
                    group_scores,
                    merged_h,
                    merged_w,
                    retention,
                    region_grid=anchor_grid,
                )
            elif selector == "text_residual_quota":
                if text_embeddings is None or feature is None:
                    raise ValueError("text_residual_quota requires text embeddings and DeepStack features")
                residual = spatial_residual_group_scores(
                    hidden_states,
                    merged_h,
                    merged_w,
                    merge_unit,
                    energy_weight=0.0,
                )
                group_scores = text_residual_group_scores(
                    feature,
                    text_embeddings,
                    residual,
                    text_weight=text_weight,
                )
                selected_groups = regional_quota_indices(
                    group_scores,
                    merged_h,
                    merged_w,
                    retention,
                    region_grid=anchor_grid,
                )
            elif selector == "learned_quota":
                if selector_head is None or feature is None:
                    raise ValueError("learned_quota requires a selector head and DeepStack features")
                if group_residual is None or group_energy is None:
                    group_residual, group_energy = spatial_residual_energy_features(
                        hidden_states, merged_h, merged_w, merge_unit
                    )
                if 2 in selector_feature_mask:
                    if text_embeddings is None:
                        raise ValueError("learned selector text feature requires text embeddings")
                    relevance = text_relevance_group_scores(feature, text_embeddings)
                else:
                    relevance = torch.zeros_like(group_residual)
                selector_features = scalar_selector_features(
                    group_residual,
                    group_energy,
                    relevance,
                    merged_h,
                    merged_w,
                )
                disabled = sorted(set(range(5)) - set(selector_feature_mask))
                if disabled:
                    selector_features[:, disabled] = 0
                group_scores = selector_head(selector_features).float()
                selected_groups = regional_quota_indices(
                    group_scores,
                    merged_h,
                    merged_w,
                    retention,
                    region_grid=anchor_grid,
                )
            elif selector == "l2_residual_quota":
                if selector_head is None or feature is None:
                    raise ValueError(
                        "l2_residual_quota requires a residual selector head and DeepStack features"
                    )
                if group_residual is None or group_energy is None:
                    group_residual, group_energy = spatial_residual_energy_features(
                        hidden_states, merged_h, merged_w, merge_unit
                    )
                if group_l2 is None:
                    group_l2 = hidden_states.view(full_groups, merge_unit, -1).float().norm(
                        dim=-1
                    ).mean(dim=1)
                if 2 in selector_feature_mask:
                    if text_embeddings is None:
                        raise ValueError("L2 residual text feature requires text embeddings")
                    relevance = text_relevance_group_scores(feature, text_embeddings)
                else:
                    relevance = torch.zeros_like(group_residual)
                selector_features = scalar_selector_features(
                    group_residual,
                    group_energy,
                    relevance,
                    merged_h,
                    merged_w,
                )
                normalized_l2 = normalize_per_image(group_l2[:, None]).squeeze(1)
                group_scores = selector_head(selector_features, normalized_l2).float()
                selected_groups = regional_quota_indices(
                    group_scores,
                    merged_h,
                    merged_w,
                    retention,
                    region_grid=anchor_grid,
                )
            elif selector == "joint_set_quota":
                if selector_head is None or feature is None:
                    raise ValueError(
                        "joint_set_quota requires a residual selector head and "
                        "DeepStack features"
                    )
                if joint_candidate_seed is None:
                    raise ValueError("joint_set_quota requires a candidate seed")
                if group_residual is None or group_energy is None:
                    group_residual, group_energy = spatial_residual_energy_features(
                        hidden_states, merged_h, merged_w, merge_unit
                    )
                if group_l2 is None:
                    group_l2 = hidden_states.view(
                        full_groups, merge_unit, -1
                    ).float().norm(dim=-1).mean(dim=1)
                if 2 in selector_feature_mask:
                    if text_embeddings is None:
                        raise ValueError(
                            "joint-deletion selector text feature requires text embeddings"
                        )
                    relevance = text_relevance_group_scores(feature, text_embeddings)
                else:
                    relevance = torch.zeros_like(group_residual)
                selector_features = scalar_selector_features(
                    group_residual,
                    group_energy,
                    relevance,
                    merged_h,
                    merged_w,
                )
                disabled = sorted(set(range(5)) - set(selector_feature_mask))
                if disabled:
                    selector_features[:, disabled] = 0
                normalized_l2 = normalize_per_image(group_l2[:, None]).squeeze(1)
                group_scores = selector_head(
                    selector_features, normalized_l2
                ).float()
                baseline_groups = structured_group_indices(
                    group_l2,
                    merged_h,
                    merged_w,
                    retention,
                    anchor_grid=anchor_grid,
                )
                candidates = region_preserving_swap_candidates(
                    baseline_groups,
                    merged_h,
                    merged_w,
                    swap_counts=joint_swap_counts,
                    candidates_per_swap=joint_candidates_per_swap,
                    region_grid=anchor_grid,
                    seed=joint_candidate_seed,
                )
                candidate_indices = [
                    candidate.selected_group_indices for candidate in candidates
                ]
                utilities = candidate_set_utilities(
                    group_scores, candidate_indices
                )
                chosen_index = int(torch.argmax(utilities).item())
                chosen = candidates[chosen_index]
                selected_groups = torch.tensor(
                    chosen.selected_group_indices,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                joint_candidate_id = chosen.candidate_id
                joint_candidate_utility = float(utilities[chosen_index].item())
            elif selector == "candidate_interaction_quota":
                if selector_head is None or feature is None:
                    raise ValueError(
                        "candidate_interaction_quota requires a candidate head "
                        "and DeepStack features"
                    )
                if joint_candidate_seed is None:
                    raise ValueError(
                        "candidate_interaction_quota requires a candidate seed"
                    )
                if candidate_task_value is None:
                    raise ValueError(
                        "candidate_interaction_quota requires a task value"
                    )
                if group_residual is None or group_energy is None:
                    group_residual, group_energy = spatial_residual_energy_features(
                        hidden_states, merged_h, merged_w, merge_unit
                    )
                if group_l2 is None:
                    group_l2 = hidden_states.view(
                        full_groups, merge_unit, -1
                    ).float().norm(dim=-1).mean(dim=1)
                if 2 in selector_feature_mask:
                    if text_embeddings is None:
                        raise ValueError(
                            "candidate interaction text feature requires text embeddings"
                        )
                    relevance = text_relevance_group_scores(feature, text_embeddings)
                else:
                    relevance = torch.zeros_like(group_residual)
                selector_features = scalar_selector_features(
                    group_residual,
                    group_energy,
                    relevance,
                    merged_h,
                    merged_w,
                )
                disabled = sorted(set(range(5)) - set(selector_feature_mask))
                if disabled:
                    selector_features[:, disabled] = 0
                normalized_l2 = normalize_per_image(group_l2[:, None]).squeeze(1)
                baseline_groups = structured_group_indices(
                    group_l2,
                    merged_h,
                    merged_w,
                    retention,
                    anchor_grid=anchor_grid,
                )
                candidates = region_preserving_swap_candidates(
                    baseline_groups,
                    merged_h,
                    merged_w,
                    swap_counts=joint_swap_counts,
                    candidates_per_swap=joint_candidates_per_swap,
                    region_grid=anchor_grid,
                    seed=joint_candidate_seed,
                )
                candidate_indices = [
                    candidate.selected_group_indices for candidate in candidates
                ]
                descriptors = candidate_interaction_features(
                    selector_features,
                    normalized_l2,
                    candidate_indices[0],
                    candidate_indices,
                    task_value=candidate_task_value,
                )
                utilities = selector_head(descriptors).float()
                utilities = utilities - utilities[0]
                chosen_index = int(torch.argmax(utilities).item())
                chosen = candidates[chosen_index]
                selected_groups = torch.tensor(
                    chosen.selected_group_indices,
                    dtype=torch.long,
                    device=hidden_states.device,
                )
                group_scores = group_l2
                joint_candidate_id = chosen.candidate_id
                joint_candidate_utility = float(utilities[chosen_index].item())
            else:
                raise ValueError(f"unsupported selector: {selector}")
            offsets = torch.arange(merge_unit, device=hidden_states.device)
            selected_patches = (selected_groups[:, None] * merge_unit + offsets[None, :]).reshape(-1)

            # A late cut must compact DeepStack tensors captured by earlier
            # layers as well as the current/future feature tensors.
            deepstack_features = [
                previous.index_select(0, selected_groups)
                for previous in deepstack_features
            ]

        if feature is not None:
            if layer_num == prune_after_layer:
                feature = feature.index_select(0, selected_groups)
            deepstack_features.append(feature)

        if layer_num == prune_after_layer:
            hidden_states = hidden_states.index_select(0, selected_patches)
            position_embeddings = (
                position_embeddings[0].index_select(0, selected_patches),
                position_embeddings[1].index_select(0, selected_patches),
            )
            cu_seqlens = torch.tensor(
                [0, selected_patches.numel()], dtype=torch.int32, device=hidden_states.device
            )

    if selected_groups is None or selected_patches is None or group_scores is None:
        raise RuntimeError("pruning point was not reached")
    image_embeds = visual.merger(hidden_states)
    if any(feature.shape[0] != image_embeds.shape[0] for feature in deepstack_features):
        raise RuntimeError("DeepStack features are not aligned with selected image tokens")
    return PrunedVisionOutput(
        image_embeds=image_embeds,
        deepstack_embeds=deepstack_features,
        selected_group_indices=selected_groups,
        selected_patch_indices=selected_patches,
        group_scores=group_scores,
        full_patch_tokens=seq_len,
        full_llm_visual_tokens=full_groups,
        prune_hidden_states=prune_hidden_states,
        group_residual=group_residual,
        group_energy=group_energy,
        group_l2=group_l2,
        joint_candidate_id=joint_candidate_id,
        joint_candidate_utility=joint_candidate_utility,
    )


def prepare_pruned_multimodal_inputs(
    model: torch.nn.Module,
    batch: Any,
    vision: PrunedVisionOutput,
) -> PrunedMultimodalInputs:
    """Remove matching placeholders while retaining their original MRoPE IDs."""
    input_ids = batch.input_ids
    attention_mask = batch.attention_mask
    if input_ids.shape[0] != 1:
        raise ValueError("prototype currently supports batch size one")
    image_token_id = int(model.config.image_token_id)
    image_positions = torch.nonzero(input_ids[0] == image_token_id, as_tuple=False).flatten()
    if image_positions.numel() != vision.full_llm_visual_tokens:
        raise ValueError("processor image placeholders do not match the full visual grid")

    full_position_ids, _ = model.model.get_rope_index(
        input_ids,
        batch.image_grid_thw,
        None,
        attention_mask=attention_mask,
    )
    keep_mask = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
    keep_mask[image_positions] = False
    keep_mask[image_positions.index_select(0, vision.selected_group_indices)] = True
    pruned_ids = input_ids[:, keep_mask]
    pruned_attention = attention_mask[:, keep_mask]
    pruned_positions = full_position_ids[:, :, keep_mask]
    inputs_embeds = model.get_input_embeddings()(pruned_ids)
    visual_mask = pruned_ids == image_token_id
    if int(visual_mask.sum().item()) != vision.image_embeds.shape[0]:
        raise RuntimeError("pruned placeholders do not match selected image embeddings")
    inputs_embeds[visual_mask] = vision.image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    deepstack = [feature.to(inputs_embeds.device, inputs_embeds.dtype) for feature in vision.deepstack_embeds]
    return PrunedMultimodalInputs(
        input_ids=pruned_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=pruned_attention,
        position_ids=pruned_positions,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=deepstack,
        original_sequence_length=input_ids.shape[1],
    )
