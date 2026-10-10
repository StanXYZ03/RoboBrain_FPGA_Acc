from __future__ import annotations

import unittest

import torch

from evaluation.learned_selector import (
    CandidateInteractionHead,
    CoordinateSelectorHead,
    L2ResidualSelectorHead,
    candidate_head_cost,
    head_cost,
    scalar_selector_features,
    selector_distillation_loss,
)
from evaluation.joint_deletion import (
    candidate_interaction_features,
    candidate_set_utilities,
    coarse_region_histogram,
    pairwise_set_ranking_loss,
    region_preserving_swap_candidates,
    require_development_manifest,
    stable_candidate_seed,
)
from evaluation.shallow_block_pruning import validate_fixed_group_indices


class LearnedSelectorTest(unittest.TestCase):
    def test_features_and_head_shape(self) -> None:
        values = torch.arange(6, dtype=torch.float32)
        features = scalar_selector_features(values, values + 1, values.flip(0), 2, 3)
        self.assertEqual(features.shape, (6, 5))
        logits = CoordinateSelectorHead()(features)
        self.assertEqual(logits.shape, (6,))
        self.assertEqual(head_cost()["parameters"], 113)
        self.assertEqual(sum(parameter.numel() for parameter in CoordinateSelectorHead().parameters()), 113)

    def test_aligned_teacher_has_lower_loss_than_reversed(self) -> None:
        teacher = torch.tensor([0.1, 0.2, 0.4, 1.0, 2.0, 3.0])
        aligned = selector_distillation_loss(teacher.log(), teacher, 2, 3)
        reversed_loss = selector_distillation_loss(teacher.flip(0).log(), teacher, 2, 3)
        self.assertLess(float(aligned), float(reversed_loss))

    def test_teacher_loss_is_saliency_scale_invariant(self) -> None:
        logits = torch.randn(6)
        teacher = torch.tensor([1e-7, 2e-7, 4e-7, 1e-6, 2e-6, 3e-6])
        first = selector_distillation_loss(logits, teacher, 2, 3)
        second = selector_distillation_loss(logits, teacher * 1e6, 2, 3)
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_loss_backpropagates(self) -> None:
        logits = torch.randn(9, requires_grad=True)
        loss = selector_distillation_loss(logits, torch.rand(9), 3, 3)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_l2_residual_head_starts_as_exact_l2(self) -> None:
        features = torch.randn(7, 5)
        l2 = torch.randn(7)
        head = L2ResidualSelectorHead()
        self.assertTrue(torch.equal(head(features, l2), l2))

    def test_fixed_group_indices_are_canonical_and_budget_checked(self) -> None:
        canonical = validate_fixed_group_indices(
            torch.tensor([4, 1, 3]), 6, 3, device=torch.device("cpu")
        )
        self.assertEqual(canonical.tolist(), [1, 3, 4])
        with self.assertRaises(ValueError):
            validate_fixed_group_indices(
                torch.tensor([1, 1, 3]), 6, 3, device=torch.device("cpu")
            )
        with self.assertRaises(ValueError):
            validate_fixed_group_indices(
                torch.tensor([1.0, 2.0, 3.0]), 6, 3, device=torch.device("cpu")
            )
        with self.assertRaises(ValueError):
            validate_fixed_group_indices(
                torch.tensor([1, 2]), 6, 3, device=torch.device("cpu")
            )
        with self.assertRaises(ValueError):
            validate_fixed_group_indices(
                torch.tensor([1, 2, 6]), 6, 3, device=torch.device("cpu")
            )

    def test_joint_candidates_preserve_budget_regions_and_are_deterministic(self) -> None:
        omitted = {1, 5, 9, 13, 17, 21}
        baseline = tuple(index for index in range(24) if index not in omitted)
        candidates = region_preserving_swap_candidates(
            baseline,
            4,
            6,
            swap_counts=(1, 2, 4),
            candidates_per_swap=2,
            seed=7,
        )
        repeated = region_preserving_swap_candidates(
            baseline,
            4,
            6,
            swap_counts=(1, 2, 4),
            candidates_per_swap=2,
            seed=7,
        )
        self.assertEqual(candidates, repeated)
        self.assertEqual(len(candidates), 7)
        self.assertEqual(
            len({candidate.selected_group_indices for candidate in candidates}), 7
        )
        baseline_histogram = coarse_region_histogram(baseline, 4, 6)
        for candidate in candidates:
            self.assertEqual(len(candidate.selected_group_indices), len(baseline))
            self.assertEqual(
                coarse_region_histogram(candidate.selected_group_indices, 4, 6),
                baseline_histogram,
            )
            difference = set(candidate.selected_group_indices) ^ set(baseline)
            self.assertEqual(len(difference), 2 * candidate.swap_count)

    def test_joint_set_ranking_loss_uses_coordinate_nll_order(self) -> None:
        candidates = ((0, 1), (2, 3))
        coordinate_nll = torch.tensor([0.1, 1.0])
        aligned_logits = torch.tensor([4.0, 3.0, 2.0, 1.0], requires_grad=True)
        reversed_logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
        aligned = pairwise_set_ranking_loss(
            aligned_logits, candidates, coordinate_nll
        )
        reversed_loss = pairwise_set_ranking_loss(
            reversed_logits, candidates, coordinate_nll
        )
        self.assertLess(float(aligned.detach()), float(reversed_loss))
        aligned.backward()
        self.assertTrue(bool(torch.isfinite(aligned_logits.grad).all()))

    def test_candidate_utilities_require_one_shared_budget(self) -> None:
        logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
        utilities = candidate_set_utilities(
            logits, ((0, 1), (2, 3)), scale=2.0
        )
        self.assertEqual(utilities.tolist(), [7.0, 3.0])
        with self.assertRaises(ValueError):
            candidate_set_utilities(logits, ((0, 1), (1, 2, 3)))

    def test_candidate_interaction_descriptor_and_head(self) -> None:
        features = torch.arange(30, dtype=torch.float32).view(6, 5)
        l2 = torch.linspace(-1.0, 1.0, 6)
        candidates = ((0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 4, 5))
        descriptors = candidate_interaction_features(
            features,
            l2,
            candidates[0],
            candidates,
            task_value=1.0,
        )
        self.assertEqual(descriptors.shape, (3, 32))
        head = CandidateInteractionHead()
        self.assertEqual(head(descriptors).tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(candidate_head_cost()["parameters"], 545)
        self.assertEqual(
            sum(parameter.numel() for parameter in head.parameters()), 545
        )

    def test_joint_deletion_rejects_revealed_manifests(self) -> None:
        require_development_manifest("datasets/development.jsonl")
        with self.assertRaises(ValueError):
            require_development_manifest("datasets/selector_validation.jsonl")
        with self.assertRaises(ValueError):
            require_development_manifest("datasets/new_blind.jsonl")

    def test_candidate_seed_depends_on_image_not_task(self) -> None:
        first = stable_candidate_seed(7, "images/example.jpg")
        second = stable_candidate_seed(7, "images/example.jpg")
        other = stable_candidate_seed(7, "images/other.jpg")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)




if __name__ == "__main__":
    unittest.main()
