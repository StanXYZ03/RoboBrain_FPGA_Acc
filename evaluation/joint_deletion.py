"""Joint-deletion candidates and set-level coordinate-risk supervision."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from evaluation.learned_selector import region_ids


@dataclass(frozen=True)
class JointDeletionCandidate:
    candidate_id: str
    swap_count: int
    selected_group_indices: tuple[int, ...]


def stable_candidate_seed(base_seed: int, image_key: str) -> int:
    """Use one candidate set for every task attached to the same image."""
    digest = hashlib.sha256(f"{base_seed}:{image_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def require_development_manifest(path: str) -> None:
    """Reject previously revealed validation or blind manifests."""
    lowered = path.lower()
    forbidden = tuple(tag for tag in ("blind", "validation") if tag in lowered)
    if forbidden:
        raise ValueError(
            "joint-deletion development forbids revealed blind/validation manifests"
        )


def _canonical_indices(indices: Iterable[int], group_count: int) -> tuple[int, ...]:
    values = tuple(sorted(int(index) for index in indices))
    if not values:
        raise ValueError("a candidate must retain at least one visual group")
    if len(values) != len(set(values)):
        raise ValueError("candidate group indices must be unique")
    if values[0] < 0 or values[-1] >= group_count:
        raise ValueError("candidate group index is outside the merged grid")
    return values


def candidate_set_utilities(
    group_logits: torch.Tensor,
    candidate_indices: Sequence[Sequence[int]],
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Score retained sets by their mean group logit.

    Every candidate in the joint-deletion protocol has the same budget, so
    mean and sum induce the same deployed choice. The scale only controls the
    margin seen by a ranking loss; using the retained-set size reproduces a sum
    while preserving a grid-independent default.
    """
    if group_logits.ndim != 1:
        raise ValueError("group logits must be one-dimensional")
    if not candidate_indices:
        raise ValueError("at least one candidate is required")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("candidate utility scale must be finite and positive")
    utilities = []
    expected_budget: int | None = None
    for indices in candidate_indices:
        canonical = _canonical_indices(indices, group_logits.numel())
        if expected_budget is None:
            expected_budget = len(canonical)
        elif len(canonical) != expected_budget:
            raise ValueError("all joint-deletion candidates must share one budget")
        index_tensor = torch.tensor(canonical, device=group_logits.device)
        utilities.append(group_logits.index_select(0, index_tensor).mean() * scale)
    return torch.stack(utilities)


def candidate_interaction_features(
    group_features: torch.Tensor,
    normalized_l2: torch.Tensor,
    baseline_indices: Sequence[int],
    candidate_indices: Sequence[Sequence[int]],
    *,
    task_value: float,
) -> torch.Tensor:
    """Encode added/removed group interactions for a fixed candidate set."""
    if group_features.ndim != 2 or group_features.shape[1] != 5:
        raise ValueError("group features must have shape [groups, 5]")
    if normalized_l2.shape != (group_features.shape[0],):
        raise ValueError("normalized L2 must have one value per group")
    if not bool(
        torch.isfinite(group_features).all()
        and torch.isfinite(normalized_l2).all()
    ):
        raise ValueError("candidate interaction inputs must be finite")
    if not math.isfinite(task_value):
        raise ValueError("task value must be finite")

    group_count = group_features.shape[0]
    baseline = _canonical_indices(baseline_indices, group_count)
    baseline_set = set(baseline)
    augmented = torch.cat(
        (group_features.float(), normalized_l2.float().unsqueeze(1)), dim=1
    )
    global_mean = augmented.mean(dim=0)
    global_std = augmented.std(dim=0, unbiased=False)
    swaps = []
    canonical_candidates = []
    for indices in candidate_indices:
        candidate = _canonical_indices(indices, group_count)
        if len(candidate) != len(baseline):
            raise ValueError("all interaction candidates must match baseline budget")
        candidate_set = set(candidate)
        removed = tuple(sorted(baseline_set - candidate_set))
        added = tuple(sorted(candidate_set - baseline_set))
        if len(removed) != len(added):
            raise ValueError("candidate swap must add and remove equal group counts")
        swaps.append((removed, added))
        canonical_candidates.append(candidate)
    if canonical_candidates[0] != baseline:
        raise ValueError("the first interaction candidate must be the baseline")
    max_swap = max(1, max(len(removed) for removed, _ in swaps))

    descriptors = []
    zeros = torch.zeros(6, dtype=augmented.dtype, device=augmented.device)
    for removed, added in swaps:
        if added:
            added_index = torch.tensor(added, device=augmented.device)
            removed_index = torch.tensor(removed, device=augmented.device)
            added_mean = augmented.index_select(0, added_index).mean(dim=0)
            removed_mean = augmented.index_select(0, removed_index).mean(dim=0)
        else:
            added_mean = zeros
            removed_mean = zeros
        delta = added_mean - removed_mean
        descriptor = torch.cat(
            (
                delta,
                delta.abs(),
                added_mean * removed_mean,
                global_mean,
                global_std,
                torch.tensor(
                    [len(added) / max_swap, task_value],
                    dtype=augmented.dtype,
                    device=augmented.device,
                ),
            )
        )
        descriptors.append(descriptor)
    return torch.stack(descriptors)


def coarse_region_histogram(
    indices: Sequence[int],
    grid_h: int,
    grid_w: int,
    *,
    region_grid: int = 3,
) -> tuple[int, ...]:
    ids = region_ids(grid_h, grid_w, region_grid, torch.device("cpu"))
    region_count = int(ids.max().item()) + 1
    counts = [0] * region_count
    for index in _canonical_indices(indices, grid_h * grid_w):
        counts[int(ids[index].item())] += 1
    return tuple(counts)


def region_preserving_swap_candidates(
    baseline_indices: Sequence[int] | torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    swap_counts: Sequence[int] = (1, 2, 4),
    candidates_per_swap: int = 2,
    region_grid: int = 3,
    seed: int = 20260826,
) -> list[JointDeletionCandidate]:
    """Create same-budget candidates by swapping groups only within each region."""
    if candidates_per_swap <= 0:
        raise ValueError("candidates_per_swap must be positive")
    if not swap_counts or any(count <= 0 for count in swap_counts):
        raise ValueError("swap counts must contain positive integers")
    group_count = grid_h * grid_w
    if group_count <= 0:
        raise ValueError("grid dimensions must be positive")
    if isinstance(baseline_indices, torch.Tensor):
        baseline_indices = baseline_indices.detach().cpu().tolist()
    baseline = _canonical_indices(baseline_indices, group_count)
    if len(baseline) == group_count:
        raise ValueError("joint deletion requires at least one omitted group")

    ids = region_ids(grid_h, grid_w, region_grid, torch.device("cpu")).tolist()
    selected = set(baseline)
    selected_by_region: dict[int, list[int]] = {}
    omitted_by_region: dict[int, list[int]] = {}
    for index, region in enumerate(ids):
        target = selected_by_region if index in selected else omitted_by_region
        target.setdefault(int(region), []).append(index)

    rng = random.Random(seed)
    candidates = [JointDeletionCandidate("baseline", 0, baseline)]
    seen = {baseline}
    for swap_count in swap_counts:
        created = 0
        attempts = 0
        while created < candidates_per_swap and attempts < 1000:
            attempts += 1
            removable = {region: values.copy() for region, values in selected_by_region.items()}
            addable = {region: values.copy() for region, values in omitted_by_region.items()}
            removed: list[int] = []
            added: list[int] = []
            for _ in range(swap_count):
                eligible = sorted(
                    region
                    for region in removable
                    if removable[region] and addable.get(region)
                )
                if not eligible:
                    break
                region = rng.choice(eligible)
                old = rng.choice(removable[region])
                new = rng.choice(addable[region])
                removable[region].remove(old)
                addable[region].remove(new)
                removed.append(old)
                added.append(new)
            if len(removed) != swap_count:
                break
            proposal = _canonical_indices((selected - set(removed)) | set(added), group_count)
            if proposal in seen:
                continue
            if coarse_region_histogram(
                proposal, grid_h, grid_w, region_grid=region_grid
            ) != coarse_region_histogram(
                baseline, grid_h, grid_w, region_grid=region_grid
            ):
                raise RuntimeError("region-preserving swap changed a regional quota")
            seen.add(proposal)
            candidates.append(
                JointDeletionCandidate(
                    candidate_id=f"swap_{swap_count:03d}_{created:02d}",
                    swap_count=swap_count,
                    selected_group_indices=proposal,
                )
            )
            created += 1
        if created != candidates_per_swap:
            raise ValueError(
                f"could not create {candidates_per_swap} unique candidates with "
                f"{swap_count} region-preserving swaps"
            )
    return candidates


def pairwise_set_ranking_loss(
    group_logits: torch.Tensor,
    candidate_indices: Sequence[Sequence[int]],
    candidate_coordinate_nll: torch.Tensor,
    *,
    tie_epsilon: float = 1e-6,
    utility_scale: float = 1.0,
) -> torch.Tensor:
    """Rank whole retained sets by measured coordinate NLL.

    Candidate utility is the mean selected-group logit. Lower measured NLL must
    receive higher utility. Supervision therefore reflects joint deletion
    outcomes rather than independent per-group saliency labels.
    """
    if group_logits.ndim != 1:
        raise ValueError("group logits must be one-dimensional")
    if candidate_coordinate_nll.shape != (len(candidate_indices),):
        raise ValueError("one coordinate NLL is required per candidate")
    if len(candidate_indices) < 2:
        raise ValueError("at least two candidates are required")
    if tie_epsilon < 0:
        raise ValueError("tie_epsilon must be nonnegative")
    if not bool(torch.isfinite(candidate_coordinate_nll).all()):
        raise ValueError("candidate coordinate NLL values must be finite")

    utilities_tensor = candidate_set_utilities(
        group_logits, candidate_indices, scale=utility_scale
    )

    losses = []
    for left in range(len(candidate_indices)):
        for right in range(left + 1, len(candidate_indices)):
            delta = float(
                candidate_coordinate_nll[right] - candidate_coordinate_nll[left]
            )
            if abs(delta) <= tie_epsilon:
                continue
            preferred_margin = (
                utilities_tensor[left] - utilities_tensor[right]
                if delta > 0
                else utilities_tensor[right] - utilities_tensor[left]
            )
            losses.append(F.softplus(-preferred_margin))
    if not losses:
        raise ValueError("all candidate coordinate NLL values are tied")
    return torch.stack(losses).mean()
