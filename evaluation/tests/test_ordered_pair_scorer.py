from __future__ import annotations

import unittest

import numpy as np
import torch

from evaluation.ordered_pair_scorer import OrderedPairScorer, grid_positions, pair_geometry


class OrderedPairScorerTest(unittest.TestCase):
    def test_shapes_finite_parameter_count_and_gradients(self):
        model = OrderedPairScorer(seed=7)
        groups = torch.randn(12, 1024)
        prompt = torch.randn(9, 2560)
        score = model(groups, prompt, grid_positions(3, 4))
        self.assertEqual(tuple(score.shape), (12,))
        self.assertTrue(torch.isfinite(score).all())
        self.assertEqual(model.parameter_count(), 511238)
        score.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_word_order_changes_scores(self):
        torch.manual_seed(3)
        model = OrderedPairScorer(seed=11)
        groups = torch.randn(8, 1024)
        prompt = torch.randn(7, 2560)
        positions = grid_positions(2, 4)
        original = model(groups, prompt, positions)
        reversed_score = model(groups, prompt.flip(0), positions)
        self.assertGreater(float(torch.max(torch.abs(original - reversed_score))), 1e-6)

    def test_grid_positions_and_pair_geometry(self):
        positions = grid_positions(2, 2)
        np.testing.assert_allclose(positions.numpy(), [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
        geometry = pair_geometry(positions[0], positions[3])
        np.testing.assert_allclose(geometry.numpy(), [-0.5, -0.5, 0.5, 0.5, 0.25, 0.25])

    def test_rejects_caption_overflow(self):
        model = OrderedPairScorer(seed=1, max_prompt_tokens=16)
        with self.assertRaisesRegex(ValueError, "outside"):
            model.encode_prompt(torch.zeros(17, 2560))


if __name__ == "__main__":
    unittest.main()
