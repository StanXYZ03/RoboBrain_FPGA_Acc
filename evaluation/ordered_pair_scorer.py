#!/usr/bin/env python3
"""Ordered three-slot, bounded-partner scorer for hardware-regular token selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PROMPT_DIM = 2560
FEATURE_DIM = 1024
SCORE_DIM = 128
ROLE_COUNT = 3
GEOMETRY_DIM = 6


def grid_positions(grid_h: int, grid_w: int, *, device: Any = None) -> torch.Tensor:
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("grid dimensions must be positive")
    rows = torch.arange(grid_h, dtype=torch.float32, device=device)
    cols = torch.arange(grid_w, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack(((xx + 0.5) / grid_w, (yy + 0.5) / grid_h), dim=-1).reshape(-1, 2)


def pair_geometry(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = source - target
    return torch.cat((delta, delta.abs(), delta.square()), dim=-1)


class OrderedPairScorer(torch.nn.Module):
    """Three ordered prompt slots and at most ``2 * partner_k`` partners/group."""

    def __init__(self, *, seed: int, init_std: float = 0.02, max_prompt_tokens: int = 16, partner_k: int = 4):
        super().__init__()
        if max_prompt_tokens <= 0 or partner_k <= 0:
            raise ValueError("max_prompt_tokens and partner_k must be positive")
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.partner_k = int(partner_k)
        generator = torch.Generator(device="cpu"); generator.manual_seed(int(seed))

        def random_parameter(*shape: int) -> torch.nn.Parameter:
            return torch.nn.Parameter(torch.randn(*shape, generator=generator) * float(init_std))

        self.Wt = random_parameter(SCORE_DIM, PROMPT_DIM)
        self.Wv = random_parameter(SCORE_DIM, FEATURE_DIM)
        self.position = random_parameter(self.max_prompt_tokens, SCORE_DIM)
        self.A_prev = random_parameter(SCORE_DIM, SCORE_DIM)
        self.A_self = random_parameter(SCORE_DIM, SCORE_DIM)
        self.A_next = random_parameter(SCORE_DIM, SCORE_DIM)
        self.conv_bias = torch.nn.Parameter(torch.zeros(SCORE_DIM))
        self.role_vectors = random_parameter(ROLE_COUNT, SCORE_DIM)
        self.geometry_weights = random_parameter(GEOMETRY_DIM, SCORE_DIM)
        self.geometry_bias = torch.nn.Parameter(torch.zeros(GEOMETRY_DIM))

    def encode_prompt(self, prompt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prompt.ndim != 2 or prompt.shape[1] != PROMPT_DIM:
            raise ValueError(f"prompt must have shape [T,{PROMPT_DIM}]")
        token_count = int(prompt.shape[0])
        if token_count < 1 or token_count > self.max_prompt_tokens:
            raise ValueError(f"prompt token count {token_count} outside [1,{self.max_prompt_tokens}]")
        projected = F.linear(prompt, self.Wt) + self.position[:token_count]
        zero = torch.zeros_like(projected[:1])
        previous = torch.cat((zero, projected[:-1]), dim=0)
        following = torch.cat((projected[1:], zero), dim=0)
        hidden = F.relu(
            F.linear(previous, self.A_prev)
            + F.linear(projected, self.A_self)
            + F.linear(following, self.A_next)
            + self.conv_bias
        )
        attention = torch.softmax(hidden @ self.role_vectors.T, dim=0)
        queries = F.normalize(attention.T @ hidden, dim=-1)
        geometry = torch.clamp(F.linear(queries[2], self.geometry_weights, self.geometry_bias), -1.0, 1.0)
        return queries, geometry

    def project_visual(self, group_mean: torch.Tensor) -> torch.Tensor:
        if group_mean.ndim != 2 or group_mean.shape[1] != FEATURE_DIM:
            raise ValueError(f"group_mean must have shape [G,{FEATURE_DIM}]")
        return torch.clamp(F.linear(group_mean, self.Wv), -1.0, 1.0)

    def score_projected(
        self,
        visual: torch.Tensor,
        positions: torch.Tensor,
        queries: torch.Tensor,
        geometry_weights: torch.Tensor,
    ) -> torch.Tensor:
        group_count = int(visual.shape[0])
        if visual.shape != (group_count, SCORE_DIM) or positions.shape != (group_count, 2):
            raise ValueError("visual/position shape mismatch")
        if queries.shape != (ROLE_COUNT, SCORE_DIM) or geometry_weights.shape != (GEOMETRY_DIM,):
            raise ValueError("query/geometry shape mismatch")
        if group_count == 1:
            return visual @ (queries[0] + queries[1])

        subject = visual @ queries[0]
        obj = visual @ queries[1]
        relation = visual @ queries[2]
        count = min(self.partner_k, group_count)
        top_subject = torch.topk(subject, count, largest=True, sorted=True).indices
        top_object = torch.topk(obj, count, largest=True, sorted=True).indices
        all_indices = torch.arange(group_count, device=visual.device)

        # Every row is candidate subject i; columns are bounded object partners j.
        object_positions = positions[top_object]
        geometry_so = pair_geometry(positions[:, None, :], object_positions[None, :, :]) @ geometry_weights
        score_as_subject = (
            subject[:, None] + obj[top_object][None, :]
            + relation[:, None] - relation[top_object][None, :] + geometry_so
        )
        score_as_subject = score_as_subject.masked_fill(all_indices[:, None] == top_object[None, :], -torch.inf)

        # Every row is candidate object i; columns are bounded subject partners j.
        subject_positions = positions[top_subject]
        geometry_os = pair_geometry(subject_positions[None, :, :], positions[:, None, :]) @ geometry_weights
        score_as_object = (
            subject[top_subject][None, :] + obj[:, None]
            + relation[top_subject][None, :] - relation[:, None] + geometry_os
        )
        score_as_object = score_as_object.masked_fill(all_indices[:, None] == top_subject[None, :], -torch.inf)
        return torch.maximum(score_as_subject.max(dim=1).values, score_as_object.max(dim=1).values)

    def forward(self, group_mean: torch.Tensor, prompt: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        queries, geometry = self.encode_prompt(prompt)
        visual = self.project_visual(group_mean)
        return self.score_projected(visual, positions, queries, geometry)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def export_numpy(self) -> dict[str, np.ndarray]:
        return {name: value.detach().cpu().numpy().astype(np.float32) for name, value in self.state_dict().items()}
