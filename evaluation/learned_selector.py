"""Tiny coordinate-aware selector head and its offline distillation loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class CoordinateSelectorHead(torch.nn.Module):
    """A 113-parameter MLP over five hardware-computable group features."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != 5:
            raise ValueError("selector features must end in five channels")
        return self.layers(features).squeeze(-1)


class L2ResidualSelectorHead(torch.nn.Module):
    """Learn only a correction to a normalized L2 safety baseline."""

    def __init__(self, hidden: int = 16, *, zero_initialize: bool = True) -> None:
        super().__init__()
        self.correction = CoordinateSelectorHead(hidden)
        if zero_initialize:
            final = self.correction.layers[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, normalized_l2: torch.Tensor) -> torch.Tensor:
        if normalized_l2.shape != features.shape[:-1]:
            raise ValueError("normalized L2 must have one value per selector feature row")
        return normalized_l2.float() + self.correction(features)


class CandidateInteractionHead(torch.nn.Module):
    """A tiny MLP that scores a fixed set of safe deletion candidates."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden: int = 16,
        *,
        zero_initialize: bool = True,
    ) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        if zero_initialize:
            final = self.layers[-1]
            torch.nn.init.zeros_(final.weight)
            torch.nn.init.zeros_(final.bias)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        if descriptors.shape[-1] != self.layers[0].in_features:
            raise ValueError("candidate descriptors have the wrong width")
        return self.layers(descriptors).squeeze(-1)


def normalize_per_image(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=0, keepdim=True)
    return centered / centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)


def scalar_selector_features(
    residual: torch.Tensor,
    energy: torch.Tensor,
    text_relevance: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Build [residual, energy, text relevance, x, y] for each merger group."""
    group_count = grid_h * grid_w
    if any(values.shape != (group_count,) for values in (residual, energy, text_relevance)):
        raise ValueError("each scalar feature must match the merged grid")
    device = residual.device
    y = (torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5) / grid_h
    x = (torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5) / grid_w
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    content = normalize_per_image(
        torch.stack((residual.float(), energy.float(), text_relevance.float()), dim=-1)
    )
    return torch.cat((content, xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=-1)


def region_ids(grid_h: int, grid_w: int, region_grid: int, device: torch.device) -> torch.Tensor:
    rows = torch.arange(grid_h, device=device) * min(region_grid, grid_h) // grid_h
    cols = torch.arange(grid_w, device=device) * min(region_grid, grid_w) // grid_w
    return (rows[:, None] * min(region_grid, grid_w) + cols[None, :]).reshape(-1)


def selector_distillation_loss(
    logits: torch.Tensor,
    teacher_saliency: torch.Tensor,
    grid_h: int,
    grid_w: int,
    *,
    region_grid: int = 3,
    temperature: float = 1.0,
    region_weight: float = 0.25,
) -> torch.Tensor:
    """Match teacher saliency globally and at coarse spatial-region level."""
    if logits.shape != teacher_saliency.shape or logits.shape != (grid_h * grid_w,):
        raise ValueError("logits and teacher saliency must match the merged grid")
    if temperature <= 0 or not (0.0 <= region_weight <= 1.0):
        raise ValueError("invalid loss hyperparameters")
    saliency = teacher_saliency.float()
    if not bool(torch.isfinite(saliency).all()) or bool((saliency < 0).any()):
        raise ValueError("teacher saliency must be finite and nonnegative")
    saliency_mass = saliency.sum()
    if float(saliency_mass) <= 0:
        raise ValueError("teacher saliency must have positive mass")
    # Activation-times-gradient values are around 1e-5.  Treating them as
    # logits would collapse the teacher to uniform.  Normalize as probability
    # mass first; temperature then applies a scale-invariant power transform.
    teacher_probability = saliency / saliency_mass
    teacher = F.softmax(
        teacher_probability.clamp_min(torch.finfo(torch.float32).tiny).log() / temperature,
        dim=0,
    )
    log_predicted = F.log_softmax(logits.float() / temperature, dim=0)
    global_loss = F.kl_div(log_predicted, teacher, reduction="sum") * temperature**2

    ids = region_ids(grid_h, grid_w, region_grid, logits.device)
    region_count = int(ids.max().item()) + 1
    predicted = log_predicted.exp()
    teacher_regions = torch.zeros(region_count, device=logits.device).scatter_add_(0, ids, teacher)
    predicted_regions = torch.zeros(region_count, device=logits.device).scatter_add_(
        0, ids, predicted
    )
    region_loss = F.mse_loss(predicted_regions, teacher_regions, reduction="sum")
    return (1.0 - region_weight) * global_loss + region_weight * region_loss


def head_cost(hidden: int = 16) -> dict[str, int]:
    return {
        "parameters": 5 * hidden + hidden + hidden + 1,
        "macs_per_group": 5 * hidden + hidden,
    }


def candidate_head_cost(input_dim: int = 32, hidden: int = 16) -> dict[str, int]:
    return {
        "parameters": input_dim * hidden + hidden + hidden + 1,
        "macs_per_candidate": input_dim * hidden + hidden,
    }
