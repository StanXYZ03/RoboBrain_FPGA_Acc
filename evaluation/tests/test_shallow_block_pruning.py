from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import torch

from evaluation.shallow_block_pruning import (
    geometric_oracle_group_indices,
    geometric_oracle_group_scores,
    regional_quota_indices,
    spatial_residual_energy_features,
    spatial_residual_group_scores,
    structured_group_indices,
    text_relevance_group_scores,
    text_residual_group_scores,
)
from evaluation.run_hf_shallow_block_pruning import matched_visual_token_targets


class ShallowBlockPruningTest(unittest.TestCase):
    def test_matched_token_targets_require_deterministic_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.jsonl"
            rows = [
                {
                    "sample_id": "a",
                    "strategy": "roi_only",
                    "final_context_visual_tokens": 60,
                    "repeat": repeat,
                }
                for repeat in range(3)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            self.assertEqual(
                matched_visual_token_targets(
                    path, "roi_only", expected_repeats=3
                ),
                {"a": 60},
            )

    def test_region_anchors_cover_a_three_by_three_partition(self) -> None:
        scores = torch.arange(36, dtype=torch.float32)
        selected = structured_group_indices(scores, 6, 6, 0.25, anchor_grid=3)
        self.assertEqual(selected.tolist(), [7, 9, 11, 19, 21, 23, 31, 33, 35])

    def test_selection_is_sorted_deterministic_and_exact_size(self) -> None:
        scores = torch.ones(40)
        first = structured_group_indices(scores, 5, 8, 0.3, anchor_grid=3)
        second = structured_group_indices(scores, 5, 8, 0.3, anchor_grid=3)
        self.assertEqual(first.numel(), 12)
        self.assertTrue(bool(torch.all(first[1:] > first[:-1])))
        self.assertTrue(torch.equal(first, second))

    def test_full_retention_is_identity(self) -> None:
        selected = structured_group_indices(torch.randn(15), 3, 5, 1.0)
        self.assertEqual(selected.tolist(), list(range(15)))

    def test_spatial_residual_scores_are_finite_and_local(self) -> None:
        hidden = torch.ones(4 * 4 * 4, 8)
        hidden[(2 * 4 + 2) * 4 : (2 * 4 + 3) * 4] = 4
        scores = spatial_residual_group_scores(hidden, 4, 4, 4)
        self.assertEqual(scores.shape, (16,))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertGreater(float(scores[10]), float(scores[0]))

    def test_raw_residual_energy_match_fused_extremes(self) -> None:
        hidden = torch.arange(32, dtype=torch.float32).view(8, 4)
        residual, energy = spatial_residual_energy_features(hidden, 2, 2, 2)
        residual_score = spatial_residual_group_scores(hidden, 2, 2, 2, energy_weight=0.0)
        energy_score = spatial_residual_group_scores(hidden, 2, 2, 2, energy_weight=1.0)
        self.assertTrue(torch.allclose(residual_score, residual / residual.mean()))
        self.assertTrue(torch.allclose(energy_score, energy / energy.mean()))

    def test_regional_quota_is_exact_and_covers_regions(self) -> None:
        scores = torch.arange(36, dtype=torch.float32)
        selected = regional_quota_indices(scores, 6, 6, 0.25, region_grid=3)
        self.assertEqual(selected.numel(), 9)
        self.assertEqual(selected.tolist(), [7, 9, 11, 19, 21, 23, 31, 33, 35])

    def test_regional_quota_full_retention_is_identity(self) -> None:
        selected = regional_quota_indices(torch.randn(15), 3, 5, 1.0)
        self.assertEqual(selected.tolist(), list(range(15)))

    def test_text_relevance_favors_matching_projected_group(self) -> None:
        projected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        text = torch.tensor([[0.0, 2.0]])
        residual = torch.ones(3)
        scores = text_residual_group_scores(
            projected,
            text,
            residual,
            text_weight=1.0,
        )
        self.assertEqual(int(torch.argmax(scores).item()), 1)

    def test_text_relevance_exposes_raw_cosine_maximum(self) -> None:
        projected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        text = torch.tensor([[2.0, 0.0]])
        expected = torch.tensor([1.0, 0.0, -1.0])
        self.assertTrue(torch.allclose(text_relevance_group_scores(projected, text), expected))

    def test_geometric_oracle_prioritizes_every_target_cell(self) -> None:
        scores = geometric_oracle_group_scores(4, 4, [250.0, 250.0, 750.0, 750.0])
        inside = {5, 6, 9, 10}
        outside = set(range(16)) - inside
        self.assertGreater(
            min(float(scores[index]) for index in inside),
            max(float(scores[index]) for index in outside),
        )

    def test_geometric_oracle_is_deterministic_exact_and_spatially_anchored(self) -> None:
        first, scores = geometric_oracle_group_indices(
            6, 6, 0.25, [420.0, 420.0, 580.0, 580.0], anchor_grid=3
        )
        second, _ = geometric_oracle_group_indices(
            6, 6, 0.25, [420.0, 420.0, 580.0, 580.0], anchor_grid=3
        )
        self.assertEqual(first.numel(), 9)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(scores.shape, (36,))
        self.assertIn(14, first.tolist())
        self.assertIn(21, first.tolist())

    def test_all_cutpoint_classes_keep_deepstack_aligned(self) -> None:
        class IdentityBlock(torch.nn.Module):
            def forward(self, hidden, **_kwargs):
                return hidden + 0.01

        class GroupMerger(torch.nn.Module):
            def forward(self, hidden):
                return hidden.view(-1, 4, hidden.shape[-1]).mean(dim=1)

        class FakeVisual(torch.nn.Module):
            dtype = torch.float32
            spatial_merge_size = 2
            deepstack_visual_indexes = [1, 3, 5]

            def __init__(self) -> None:
                super().__init__()
                self.patch_embed = torch.nn.Identity()
                self.blocks = torch.nn.ModuleList(IdentityBlock() for _ in range(6))
                self.deepstack_merger_list = torch.nn.ModuleList(
                    GroupMerger() for _ in range(3)
                )
                self.merger = GroupMerger()

            def fast_pos_embed_interpolate(self, grid):
                return torch.zeros(16, 8)

            def rot_pos_emb(self, grid):
                return torch.zeros(16, 4)

        from evaluation.shallow_block_pruning import forward_shallow_pruned_vision

        visual = FakeVisual()
        pixels = torch.arange(16 * 8, dtype=torch.float32).view(16, 8)
        grid = torch.tensor([[1, 4, 4]])
        for cut in (-1, 0, 2, 5):
            result = forward_shallow_pruned_vision(
                visual,
                pixels,
                grid,
                retention=0.5,
                prune_after_layer=cut,
                selector="geometric_oracle",
                oracle_bbox_1000=[400.0, 400.0, 600.0, 600.0],
            )
            self.assertEqual(result.image_embeds.shape, (2, 8))
            self.assertEqual(result.selected_patch_indices.numel(), 8)
            self.assertEqual(len(result.deepstack_embeds), 3)
            self.assertTrue(all(feature.shape == (2, 8) for feature in result.deepstack_embeds))


if __name__ == "__main__":
    unittest.main()
