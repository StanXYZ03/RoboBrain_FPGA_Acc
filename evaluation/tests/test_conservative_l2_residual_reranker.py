#!/usr/bin/env python3
"""CPU-only focused tests for the conservative L2-residual reranker mechanism.

The tests pin the frozen contract: baseline corrected integer-L2 selection,
frozen anchors, same-region one-for-one swaps, per-region count preservation,
strictly positive gains, bounded per-region candidates, the S=16 work quantum,
the exact K96 / 87 real + 9 context layout, corrected integer-L2 context
pooling and fail-closed validation.  No CUDA, model or network is touched.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import conservative_l2_residual_reranker as rr  # noqa: E402
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    SOURCE_CONTEXT,
    build_regions,
    context_pooling_weights,
    pool_region_context,
    region_id_by_group,
    select_hardware_regular_groups,
)

FULL_GRID = (21, 32)  # G = 672, 3x3 regions all non-empty with >= 2 members
SMALL_GRID = (15, 15)  # G = 225 > 96, still 9 context regions


def _synthetic_hidden(grid_h: int, grid_w: int, seed: int = 20261005) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = grid_h * grid_w
    scales = rng.normal(size=(groups, 1, 1))
    return (rng.normal(size=(groups, 4, 1024)) * scales).astype(np.float64)


def _baseline(hidden: np.ndarray, grid_h: int, grid_w: int):
    rng = np.random.default_rng(7)
    combined = rng.random(hidden.shape[0])
    return select_hardware_regular_groups(
        hidden,
        grid_h,
        grid_w,
        budget=96,
        combined_scores=combined,
        local_candidate_cap=rr.LOCAL_CANDIDATE_CAP,
    )


class ResidualScoreTest(unittest.TestCase):
    def test_signed_difference(self) -> None:
        semantic = np.array([10, 0, 32767], dtype=np.int64)
        l2 = np.array([0, 5, 32767], dtype=np.int64)
        residual = rr.residual_score_q15(semantic, l2)
        self.assertEqual(residual.tolist(), [10, -5, 0])
        self.assertTrue(np.issubdtype(residual.dtype, np.integer))

    def test_rejects_out_of_range_and_shape(self) -> None:
        with self.assertRaises(rr.ResidualRerankError):
            rr.residual_score_q15(np.array([1, 2]), np.array([1]))
        with self.assertRaises(rr.ResidualRerankError):
            rr.residual_score_q15(np.array([-1, 2]), np.array([0, 0]))
        with self.assertRaises(rr.ResidualRerankError):
            rr.residual_score_q15(np.array([1, 40000]), np.array([0, 0]))
        with self.assertRaises(rr.ResidualRerankError):
            rr.residual_score_q15(np.array([1.0, 2.0]), np.array([0, 0]))

    def test_l2_combined_scale(self) -> None:
        combined = rr.l2_combined_float(np.array([0, 32767], dtype=np.int64))
        self.assertAlmostEqual(combined[0], 0.0)
        self.assertAlmostEqual(combined[1], 32767 / 32768.0)


class PlanContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid_h, self.grid_w = SMALL_GRID
        self.hidden = _synthetic_hidden(self.grid_h, self.grid_w)
        self.baseline = _baseline(self.hidden, self.grid_h, self.grid_w)
        rng = np.random.default_rng(11)
        self.residual = rng.integers(-30000, 30000, size=self.hidden.shape[0])

    def test_anchors_frozen_and_region_counts_preserved(self) -> None:
        plan = rr.plan_residual_swaps(self.baseline, self.residual)
        anchors = set(int(v) for v in self.baseline.anchors)
        self.assertTrue(anchors.issubset(set(plan.reranked_selected_real)))
        self.assertEqual(
            len(plan.reranked_selected_real), len(plan.baseline_selected_real)
        )
        self.assertEqual(
            plan.per_region_selected_counts_before,
            plan.per_region_selected_counts_after,
        )

    def test_positive_gains_bounded_and_same_region(self) -> None:
        plan = rr.plan_residual_swaps(self.baseline, self.residual)
        self.assertLessEqual(plan.swap_count, rr.MAX_SWAPS)
        region_lookup = region_id_by_group(self.grid_h, self.grid_w)
        for swap in plan.swaps:
            self.assertGreater(swap.gain_q15, 0)
            self.assertEqual(
                swap.gain_q15,
                swap.residual_added_q15 - swap.residual_removed_q15,
            )
            self.assertEqual(
                region_lookup[swap.removed_group], region_lookup[swap.added_group]
            )

    def test_candidate_enumeration_bounded_by_cap(self) -> None:
        plan = rr.plan_residual_swaps(self.baseline, self.residual)
        baseline_real = set(plan.baseline_selected_real)
        region_lookup = region_id_by_group(self.grid_h, self.grid_w)
        unselected_by_region: dict[int, list[int]] = {}
        for region in build_regions(self.grid_h, self.grid_w):
            for group in region.members:
                if int(group) not in baseline_real:
                    unselected_by_region.setdefault(region.region_id, []).append(
                        int(group)
                    )
        for swap in plan.swaps:
            region_id = region_lookup[swap.added_group]
            ranked = sorted(
                unselected_by_region[region_id],
                key=lambda group: (-int(self.residual[group]), group),
            )[: rr.LOCAL_CANDIDATE_CAP]
            self.assertIn(swap.added_group, ranked)

    def test_deterministic_tie_break_by_lower_index(self) -> None:
        baseline_real = set(int(v) for v in self.baseline.anchors) | set(
            int(v) for v in self.baseline.details
        )
        anchors = set(int(v) for v in self.baseline.anchors)
        chosen_region = None
        selected_non_anchor: list[int] = []
        unselected: list[int] = []
        for region in build_regions(self.grid_h, self.grid_w):
            sel = sorted(
                g for g in region.members if g in baseline_real and g not in anchors
            )
            uns = sorted(g for g in region.members if g not in baseline_real)
            if len(sel) >= 2 and len(uns) >= 2:
                chosen_region = region
                selected_non_anchor = sel[:2]
                unselected = uns[:2]
                break
        self.assertIsNotNone(chosen_region)
        s0, s1 = selected_non_anchor
        c0, c1 = unselected
        # All selected groups share residual 100 and every unselected group
        # outside the chosen region sits at -30000, so every eligible swap has
        # the identical gain 100 and only the chosen region can swap.
        residual = np.full(self.hidden.shape[0], -30000, dtype=np.int64)
        for group in baseline_real:
            residual[group] = 100
        for group in chosen_region.members:
            if int(group) not in baseline_real:
                residual[int(group)] = 200
        all_selected = sorted(
            int(g)
            for g in chosen_region.members
            if int(g) in baseline_real and int(g) not in anchors
        )
        all_unselected = sorted(
            int(g) for g in chosen_region.members if int(g) not in baseline_real
        )
        expected_count = min(rr.MAX_SWAPS, len(all_selected), len(all_unselected))
        plan = rr.plan_residual_swaps(self.baseline, residual)
        plan_again = rr.plan_residual_swaps(self.baseline, residual)
        self.assertEqual(plan.swap_count, expected_count)
        self.assertEqual(
            {(swap.removed_group, swap.added_group) for swap in plan.swaps},
            {(all_selected[i], all_unselected[i]) for i in range(expected_count)},
        )
        self.assertEqual(plan.swaps, plan_again.swaps)
        # The first accepted swap is the lowest-index removed + added pair.
        self.assertEqual(plan.swaps[0].removed_group, s0)
        self.assertEqual(plan.swaps[0].added_group, c0)

    def test_rejects_bypass_baseline(self) -> None:
        small_hidden = _synthetic_hidden(8, 8)
        bypass = select_hardware_regular_groups(
            small_hidden, 8, 8, budget=96, combined_scores=np.zeros(64)
        )
        self.assertTrue(bypass.bypass)
        with self.assertRaises(rr.ResidualRerankError):
            rr.plan_residual_swaps(bypass, np.zeros(64, dtype=np.int64))


class SelectContractTest(unittest.TestCase):
    def test_zero_residual_matches_frozen_baseline_exactly(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=3)
        from evaluation.hardware_regular_b1_corrected_l2 import (
            corrected_patch_l2_integer,
        )

        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        baseline = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=96,
            combined_scores=l2.astype(np.float64) / 32768.0,
            local_candidate_cap=rr.LOCAL_CANDIDATE_CAP,
        )
        result = rr.select_conservative_residual_rerank(
            hidden, grid_h, grid_w, semantic_norm_q15=l2, l2_norm_q15=l2
        )
        self.assertEqual(result["swap_count"], 0)
        selection = result["selection"]
        self.assertTrue(np.array_equal(selection.group_indices, baseline.group_indices))
        self.assertTrue(np.allclose(selection.groups, baseline.groups))
        self.assertTrue(np.array_equal(selection.source_type, baseline.source_type))
        self.assertTrue(np.array_equal(selection.region_id, baseline.region_id))
        self.assertTrue(np.array_equal(selection.patch_indices, baseline.patch_indices))

    def test_exact_k96_layout(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=5)
        rng = np.random.default_rng(5)
        semantic = rng.integers(0, 32768, size=hidden.shape[0])
        from evaluation.hardware_regular_b1_corrected_l2 import (
            corrected_patch_l2_integer,
        )

        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        result = rr.select_conservative_residual_rerank(
            hidden, grid_h, grid_w, semantic_norm_q15=semantic, l2_norm_q15=l2
        )
        selection = result["selection"]
        self.assertFalse(selection.bypass)
        self.assertEqual(selection.output_group_count, 96)
        self.assertEqual(selection.real_group_count, 87)
        self.assertEqual(selection.context_group_count, 9)
        self.assertLessEqual(result["swap_count"], 16)
        self.assertTrue(np.all(np.diff(selection.group_indices) > 0))

    def test_context_pooling_uses_corrected_l2(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=9)
        rng = np.random.default_rng(9)
        semantic = rng.integers(0, 32768, size=hidden.shape[0])
        from evaluation.hardware_regular_b1_corrected_l2 import (
            corrected_patch_l2_integer,
        )

        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        result = rr.select_conservative_residual_rerank(
            hidden, grid_h, grid_w, semantic_norm_q15=semantic, l2_norm_q15=l2
        )
        selection = result["selection"]
        l2_combined = l2.astype(np.float64) / 32768.0
        selected_real = set(
            int(selection.group_indices[i])
            for i in range(selection.output_group_count)
            if int(selection.source_type[i]) != SOURCE_CONTEXT
        )
        regions = {
            region.region_id: region
            for region in build_regions(grid_h, grid_w)
        }
        context_seen = 0
        for position in range(selection.output_group_count):
            if int(selection.source_type[position]) != SOURCE_CONTEXT:
                continue
            context_seen += 1
            region_id = int(selection.region_id[position])
            region = regions[region_id]
            remaining = [
                int(g) for g in region.members if int(g) not in selected_real
            ]
            expected_tokens, valid = pool_region_context(
                hidden, remaining, l2_combined
            )
            self.assertTrue(valid)
            self.assertTrue(
                np.allclose(selection.groups[position], expected_tokens, atol=0.0)
            )
            weights = context_pooling_weights(l2_combined, remaining)
            self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(context_seen, 9)

    def test_identity_bypass(self) -> None:
        hidden = _synthetic_hidden(8, 8)
        result = rr.select_conservative_residual_rerank(
            hidden,
            8,
            8,
            semantic_norm_q15=np.zeros(64, dtype=np.int64),
            l2_norm_q15=np.zeros(64, dtype=np.int64),
        )
        self.assertTrue(result["bypass"])
        self.assertEqual(result["swap_count"], 0)
        self.assertEqual(result["selection"].output_group_count, 64)
        self.assertEqual(
            result["selection"].group_indices.tolist(), list(range(64))
        )

    def test_fail_closed_on_bad_inputs(self) -> None:
        hidden = _synthetic_hidden(*FULL_GRID)
        with self.assertRaises(rr.ResidualRerankError):
            rr.select_conservative_residual_rerank(
                hidden,
                *FULL_GRID,
                semantic_norm_q15=np.zeros(5, dtype=np.int64),
                l2_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
            )
        with self.assertRaises(rr.ResidualRerankError):
            rr.select_conservative_residual_rerank(
                hidden,
                7,
                7,
                semantic_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
                l2_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
            )


if __name__ == "__main__":
    unittest.main()
