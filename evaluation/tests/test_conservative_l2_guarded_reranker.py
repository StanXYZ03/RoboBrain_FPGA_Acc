#!/usr/bin/env python3
"""CPU-only focused tests for the L2-guarded integer reranker mechanism (v2).

The tests pin the corrected sign contract:
``legal_gain = semantic_gain - l2_loss = utility_added - utility_removed > 0``,
the frozen corrected integer-L2 baseline, frozen anchors, same-region
one-for-one swaps, per-region count preservation, ``local_candidate_cap`` and
``S=16`` bounds, the exact K96 / 87 real + 9 context layout, corrected
integer-L2 context pooling, deterministic tie breaking, zero-swap equivalence
and the frozen semantic / corrected-L2 provenance hashes.  No CUDA, model or
network is touched.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    conservative_l2_guarded_reranker as gr,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    corrected_patch_l2_float,
    corrected_patch_l2_integer,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    SOURCE_CONTEXT,
    build_regions,
    region_id_by_group,
    select_hardware_regular_groups,
)

FULL_GRID = (21, 32)  # G = 672, 3x3 regions all non-empty with >= 2 members
SMALL_GRID = (15, 15)  # G = 225 > 96, still 9 context regions

#: Frozen provenance pins (semantic Q0.15 chain + corrected patch-L2 chain +
#: deployment checkpoints).  These are re-verified in ``FrozenProvenanceTest``.
FROZEN_PROVENANCE = {
    "evaluation/selector_int8_numerical_audit.py": (
        "4329e11616d2790218670b6b2b7252cec847c75a56778fcba871044428ea05dc"
    ),
    "evaluation/selector_integer_scorer_audit.py": (
        "3fcf91d28f670e7406a41028b02fcaaa4f7694abf8ba76210494cd05502b16a9"
    ),
    "evaluation/hardware_regular_b1_corrected_l2.py": (
        "6c9bb24164da12c3dc5295b2e4a5d8dee1f4f46d05e421bb1788185e4fc3b585"
    ),
    "evaluation/current_selector_integer.py": (
        "34728308a0c5c59beab4615516d91fea5d79e3b75f34c8c119dff256217b8ed7"
    ),
    "artifacts/token_selector/spatial_prompt_contrastive_trainval/"
    "B_SPATIAL_KL_CONTRASTIVE/20261003/best.npz": (
        "1f1d379e3229f310d11fee4ff620480e13fea9005abefc762c66f0d331ca72ff"
    ),
    "artifacts/token_selector/selector_wv_scale_regularization/20261003/"
    "Wv_int8_rowmajor.bin": (
        "126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b"
    ),
    "artifacts/token_selector/selector_wv_scale_regularization/20261003/"
    "Wv_per_tensor_scale_f32le.bin": (
        "93585a3bd2d584a612b40d827f8d15f43c87acdc764d7e217d99df473428b13f"
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synthetic_hidden(grid_h: int, grid_w: int, seed: int = 20261006) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = grid_h * grid_w
    scales = rng.normal(size=(groups, 1, 1))
    return (rng.normal(size=(groups, 4, 1024)) * scales).astype(np.float64)


def _synthetic_baseline(
    grid_h: int,
    grid_w: int,
    anchors: tuple[int, ...],
    details: tuple[int, ...],
) -> "object":
    """Minimal HardwareSelection used only by ``plan_guarded_swaps``."""

    from evaluation.hardware_regular_token_selector import HardwareSelection

    selected = tuple(sorted(set(anchors) | set(details)))
    region_lookup = region_id_by_group(grid_h, grid_w)
    group_indices = np.asarray(selected, dtype=np.int64)
    return HardwareSelection(
        boundary="B1",
        budget=96,
        bypass=False,
        groups=np.zeros((len(selected), 4, 1024), dtype=np.float64),
        group_indices=group_indices,
        patch_indices=np.zeros((len(selected), 4), dtype=np.int64),
        source_type=np.zeros((len(selected),), dtype=np.uint8),
        region_id=np.asarray(
            [region_lookup[int(g)] for g in selected], dtype=np.int16
        ),
        valid=np.ones((len(selected),), dtype=bool),
        scores=None,
        real_group_count=len(selected),
        context_group_count=0,
        grid_h=grid_h,
        grid_w=grid_w,
        anchors=tuple(sorted(anchors)),
        details=tuple(sorted(details)),
        lambda_semantic=None,
    )


class UtilityTest(unittest.TestCase):
    def test_widened_sum_and_range(self) -> None:
        semantic = np.array([0, 10000, 32767], dtype=np.int64)
        l2 = np.array([0, 1000, 32767], dtype=np.int64)
        utility = gr.utility_q16(semantic, l2)
        self.assertEqual(utility.tolist(), [0, 11000, 65534])
        self.assertEqual(int(utility.max()), gr.UTILITY_MAX)
        self.assertTrue(np.issubdtype(utility.dtype, np.integer))

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(gr.GuardedRerankError):
            gr.utility_q16(np.array([1, 2]), np.array([1]))
        with self.assertRaises(gr.GuardedRerankError):
            gr.utility_q16(np.array([-1, 2]), np.array([0, 0]))
        with self.assertRaises(gr.GuardedRerankError):
            gr.utility_q16(np.array([1, 40000]), np.array([0, 0]))
        with self.assertRaises(gr.GuardedRerankError):
            gr.utility_q16(np.array([1.0, 2.0]), np.array([0, 0]))


class CounterexampleTest(unittest.TestCase):
    """Invariant 2: the explicit v1 counterexample must yield no swap."""

    def test_semantic_flat_l2_regression_is_never_accepted(self) -> None:
        grid_h, grid_w = SMALL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=101)
        group_count = grid_h * grid_w
        l2 = np.full(group_count, 20000, dtype=np.int64)
        l2[0] = 32767  # region-0 anchor
        l2[1] = 30000  # selected non-anchor i
        l2[2] = 1000  # same-region unselected candidate j
        semantic = np.full(group_count, 10000, dtype=np.int64)

        result = gr.select_guarded_rerank(
            hidden,
            grid_h,
            grid_w,
            semantic_norm_q15=semantic,
            l2_norm_q15=l2,
        )
        self.assertEqual(result["swap_count"], 0)
        selection = result["selection"]
        selected_real = set(int(v) for v in selection.group_indices)
        self.assertIn(1, selected_real)
        self.assertNotIn(2, selected_real)
        region_lookup = region_id_by_group(grid_h, grid_w)
        self.assertEqual(region_lookup[1], region_lookup[2])

        semantic_i = semantic_j = 10000
        l2_i, l2_j = 30000, 1000
        semantic_gain = semantic_j - semantic_i
        l2_loss = l2_i - l2_j
        legal_gain = semantic_gain - l2_loss
        self.assertEqual(semantic_gain, 0)
        self.assertEqual(l2_loss, 29000)
        self.assertEqual(legal_gain, -29000)
        self.assertLess(legal_gain, 0)
        # The defective v1 ranking quantity would have been strongly positive.
        legacy_v1_gain = (semantic_j - l2_j) - (semantic_i - l2_i)
        self.assertEqual(legacy_v1_gain, 29000)
        self.assertGreater(legacy_v1_gain, 0)

    def test_legal_gain_helper_is_sign_correct(self) -> None:
        semantic = np.array([10000, 10000], dtype=np.int64)
        l2 = np.array([30000, 1000], dtype=np.int64)
        utility = gr.utility_q16(semantic, l2)
        # removed index 0, added index 1
        legal = int(utility[1]) - int(utility[0])
        self.assertEqual(legal, -29000)


class PlanContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid_h, self.grid_w = SMALL_GRID
        self.hidden = _synthetic_hidden(self.grid_h, self.grid_w, seed=7)
        rng = np.random.default_rng(7)
        self.l2 = corrected_patch_l2_integer(self.hidden)["l2_norm_q15"]
        self.semantic = rng.integers(0, 32768, size=self.hidden.shape[0])
        baseline = select_hardware_regular_groups(
            self.hidden,
            self.grid_h,
            self.grid_w,
            budget=96,
            combined_scores=self.l2.astype(np.float64) / 32768.0,
            local_candidate_cap=gr.LOCAL_CANDIDATE_CAP,
        )
        self.plan = gr.plan_guarded_swaps(
            baseline, self.semantic, self.l2
        )

    def test_anchors_and_region_counts_preserved(self) -> None:
        anchors = set(int(v) for v in self.plan.baseline_anchors)
        self.assertTrue(anchors.issubset(set(self.plan.reranked_selected_real)))
        self.assertEqual(
            len(self.plan.reranked_selected_real),
            len(self.plan.baseline_selected_real),
        )
        self.assertEqual(
            self.plan.per_region_selected_counts_before,
            self.plan.per_region_selected_counts_after,
        )

    def test_every_swap_satisfies_widened_identity(self) -> None:
        region_lookup = region_id_by_group(self.grid_h, self.grid_w)
        utility = gr.utility_q16(self.semantic, self.l2)
        self.assertLessEqual(self.plan.swap_count, gr.MAX_SWAPS)
        for swap in self.plan.swaps:
            self.assertGreater(swap.legal_gain_q16, 0)
            self.assertEqual(
                swap.legal_gain_q16,
                swap.semantic_gain_q15 - swap.l2_loss_q15,
            )
            self.assertEqual(
                swap.legal_gain_q16,
                swap.utility_added_q16 - swap.utility_removed_q16,
            )
            self.assertEqual(
                swap.semantic_gain_q15,
                swap.semantic_added_q15 - swap.semantic_removed_q15,
            )
            self.assertEqual(
                swap.l2_loss_q15, swap.l2_removed_q15 - swap.l2_added_q15
            )
            self.assertEqual(swap.utility_added_q16, int(utility[swap.added_group]))
            self.assertEqual(
                swap.utility_removed_q16, int(utility[swap.removed_group])
            )
            self.assertEqual(
                region_lookup[swap.removed_group], region_lookup[swap.added_group]
            )

    def test_candidate_enumeration_bounded_and_ranked_by_utility(self) -> None:
        utility = gr.utility_q16(self.semantic, self.l2)
        baseline_real = set(self.plan.baseline_selected_real)
        region_lookup = region_id_by_group(self.grid_h, self.grid_w)
        unselected_by_region: dict[int, list[int]] = {}
        for region in build_regions(self.grid_h, self.grid_w):
            for group in region.members:
                if int(group) not in baseline_real:
                    unselected_by_region.setdefault(region.region_id, []).append(
                        int(group)
                    )
        for swap in self.plan.swaps:
            region_id = region_lookup[swap.added_group]
            ranked = sorted(
                unselected_by_region[region_id],
                key=lambda group: (-int(utility[group]), group),
            )[: gr.LOCAL_CANDIDATE_CAP]
            self.assertIn(swap.added_group, ranked)

    def test_global_sort_order(self) -> None:
        order = [
            (
                -swap.legal_gain_q16,
                swap.added_group,
                swap.removed_group,
                swap.region_id,
            )
            for swap in self.plan.swaps
        ]
        self.assertEqual(order, sorted(order))

    def test_rejects_bypass_baseline(self) -> None:
        small_hidden = _synthetic_hidden(8, 8)
        bypass = select_hardware_regular_groups(
            small_hidden, 8, 8, budget=96, combined_scores=np.zeros(64)
        )
        self.assertTrue(bypass.bypass)
        with self.assertRaises(gr.GuardedRerankError):
            gr.plan_guarded_swaps(
                bypass,
                np.zeros(64, dtype=np.int64),
                np.zeros(64, dtype=np.int64),
            )


class FlatSemanticTest(unittest.TestCase):
    """Invariant 1: flat semantic Q0.15 yields exactly zero swaps."""

    def test_flat_semantic_zero_swaps(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=13)
        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        for value in (0, 1, 12345, 32767):
            with self.subTest(value=value):
                semantic = np.full(hidden.shape[0], value, dtype=np.int64)
                result = gr.select_guarded_rerank(
                    hidden,
                    grid_h,
                    grid_w,
                    semantic_norm_q15=semantic,
                    l2_norm_q15=l2,
                )
                self.assertEqual(result["swap_count"], 0)
                self.assertEqual(result["baseline_plan"]["swap_count"], 0)


class ZeroSwapEquivalenceTest(unittest.TestCase):
    """Invariant 5: zero-swap v2 == corrected integer-L2 selection/layout."""

    def _compare_to_baseline(self, semantic: np.ndarray, seed: int) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=seed)
        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        baseline = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=96,
            combined_scores=l2.astype(np.float64) / 32768.0,
            local_candidate_cap=gr.LOCAL_CANDIDATE_CAP,
        )
        result = gr.select_guarded_rerank(
            hidden,
            grid_h,
            grid_w,
            semantic_norm_q15=semantic(l2),
            l2_norm_q15=l2,
        )
        self.assertEqual(result["swap_count"], 0)
        selection = result["selection"]
        self.assertTrue(
            np.array_equal(selection.group_indices, baseline.group_indices)
        )
        self.assertTrue(np.array_equal(selection.groups, baseline.groups))
        self.assertTrue(
            np.array_equal(selection.source_type, baseline.source_type)
        )
        self.assertTrue(np.array_equal(selection.region_id, baseline.region_id))
        self.assertTrue(
            np.array_equal(selection.patch_indices, baseline.patch_indices)
        )
        self.assertTrue(np.array_equal(selection.valid, baseline.valid))

    def test_flat_semantic_matches_baseline(self) -> None:
        self._compare_to_baseline(
            lambda l2: np.full(l2.shape, 1000, dtype=np.int64), seed=3
        )

    def test_utility_equal_to_l2_matches_baseline(self) -> None:
        # semantic == l2 makes utility strictly monotone in l2, so no swap.
        self._compare_to_baseline(lambda l2: l2.copy(), seed=5)


class LayoutInvariantTest(unittest.TestCase):
    """Invariant 4: anchors, counts, K96, 87+9, nonempty pools, L2 pooling."""

    def test_exact_layout_and_pooling(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=17)
        rng = np.random.default_rng(17)
        semantic = rng.integers(0, 32768, size=hidden.shape[0])
        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        result = gr.select_guarded_rerank(
            hidden,
            grid_h,
            grid_w,
            semantic_norm_q15=semantic,
            l2_norm_q15=l2,
        )
        selection = result["selection"]
        self.assertFalse(selection.bypass)
        self.assertEqual(selection.output_group_count, 96)
        self.assertEqual(selection.real_group_count, 87)
        self.assertEqual(selection.context_group_count, 9)
        self.assertLessEqual(result["swap_count"], 16)
        self.assertTrue(np.all(np.diff(selection.group_indices) > 0))
        anchors = set(int(v) for v in selection.anchors)
        self.assertTrue(anchors.issubset(set(int(v) for v in selection.group_indices)))

        l2_combined = l2.astype(np.float64) / 32768.0
        self.assertTrue(np.array_equal(selection.scores.l2, l2_combined))
        self.assertTrue(np.array_equal(selection.scores.combined, l2_combined))

        selected_real = set(
            int(selection.group_indices[i])
            for i in range(selection.output_group_count)
            if int(selection.source_type[i]) != SOURCE_CONTEXT
        )
        regions = {region.region_id: region for region in build_regions(grid_h, grid_w)}
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
            self.assertTrue(remaining)
            from evaluation.hardware_regular_token_selector import (
                context_pooling_weights,
                pool_region_context,
            )

            weights = context_pooling_weights(l2_combined, remaining)
            self.assertAlmostEqual(float(weights.sum()), 1.0)
            expected_tokens, valid = pool_region_context(
                hidden, remaining, l2_combined
            )
            self.assertTrue(valid)
            self.assertTrue(
                np.allclose(selection.groups[position], expected_tokens, atol=0.0)
            )
        self.assertEqual(context_seen, 9)

    def test_identity_bypass(self) -> None:
        hidden = _synthetic_hidden(8, 8)
        result = gr.select_guarded_rerank(
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
        with self.assertRaises(gr.GuardedRerankError):
            gr.select_guarded_rerank(
                hidden,
                *FULL_GRID,
                semantic_norm_q15=np.zeros(5, dtype=np.int64),
                l2_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
            )
        with self.assertRaises(gr.GuardedRerankError):
            gr.select_guarded_rerank(
                hidden,
                7,
                7,
                semantic_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
                l2_norm_q15=np.zeros(hidden.shape[0], dtype=np.int64),
            )


class TieBreakTest(unittest.TestCase):
    """Invariant 6: equal legal gains order by (added, removed, region)."""

    def test_equal_gain_order_is_added_then_removed_then_region(self) -> None:
        grid_h = grid_w = 6
        anchors = (0, 2, 4, 12, 14, 16, 24, 26, 28)
        details = (1, 3)
        baseline = _synthetic_baseline(grid_h, grid_w, anchors, details)
        semantic = np.zeros(grid_h * grid_w, dtype=np.int64)
        l2 = np.zeros(grid_h * grid_w, dtype=np.int64)
        l2[1] = 1000  # removed in region 0
        l2[6] = 1100  # added in region 0, gain 100
        l2[3] = 2000  # removed in region 1
        l2[8] = 2100  # added in region 1, gain 100

        plan = gr.plan_guarded_swaps(baseline, semantic, l2)
        plan_again = gr.plan_guarded_swaps(baseline, semantic, l2)
        self.assertEqual(plan.swaps, plan_again.swaps)
        self.assertEqual(plan.swap_count, 2)
        self.assertEqual(
            [(swap.added_group, swap.removed_group, swap.region_id) for swap in plan.swaps],
            [(6, 1, 0), (8, 3, 1)],
        )
        for swap in plan.swaps:
            self.assertEqual(swap.legal_gain_q16, 100)

    def test_candidate_tie_breaks_on_lower_index(self) -> None:
        grid_h = grid_w = 6
        anchors = (0, 2, 4, 12, 14, 16, 24, 26, 28)
        details = (1,)
        baseline = _synthetic_baseline(grid_h, grid_w, anchors, details)
        semantic = np.zeros(grid_h * grid_w, dtype=np.int64)
        l2 = np.zeros(grid_h * grid_w, dtype=np.int64)
        l2[1] = 1000  # removed in region 0
        l2[6] = 1000  # candidate A, gain 0 (not legal)
        l2[7] = 1000  # candidate B, gain 0 (not legal)
        plan = gr.plan_guarded_swaps(baseline, semantic, l2)
        self.assertEqual(plan.swap_count, 0)
        ranked = gr._candidate_ranking([7, 6], gr.utility_q16(semantic, l2), 16)
        self.assertEqual(ranked, [6, 7])


class FrozenProvenanceTest(unittest.TestCase):
    """Invariant 7: verify the accepted semantic and corrected-L2 provenance.

    The semantic Q0.15 chain and the corrected patch-L2 Q0.15 chain are pinned
    by physical SHA256; the corrected chain is explicitly the integer
    ``global-INT8 + floor_sqrt`` chain, never the older mean-of-patch float
    helper.
    """

    def test_source_and_checkpoint_hashes(self) -> None:
        for relative, expected in FROZEN_PROVENANCE.items():
            with self.subTest(relative=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file(), f"missing frozen input {relative}")
                self.assertEqual(_file_sha256(path), expected)

    def test_corrected_l2_is_integer_chain_not_float_helper(self) -> None:
        grid_h, grid_w = FULL_GRID
        hidden = _synthetic_hidden(grid_h, grid_w, seed=23)
        integer = corrected_patch_l2_integer(hidden)
        float_helper = corrected_patch_l2_float(hidden)
        self.assertIn("INT8", integer["formula"])
        self.assertIn("float", float_helper["formula"])
        # The integer chain derives from globally quantised raw patches plus an
        # exact INT64 sum of squares / integer floor sqrt; it is a distinct
        # quantity from the mean-of-patch float helper.
        self.assertEqual(integer["patch_q"].dtype, np.int64)
        self.assertLessEqual(int(np.abs(integer["patch_q"]).max()), 127)
        self.assertEqual(integer["patch_squared_sum"].dtype, np.int64)
        self.assertEqual(integer["l2_norm_q15"].dtype, np.int64)
        self.assertGreaterEqual(int(integer["l2_norm_q15"].min()), 0)
        self.assertLessEqual(int(integer["l2_norm_q15"].max()), 32767)
        self.assertFalse(
            np.array_equal(
                integer["l2_norm_q15"].astype(np.float64),
                float_helper["combined"] * 32768.0,
            )
        )

    def test_rounding_and_saturation_helpers_are_accepted(self) -> None:
        from evaluation.selector_integer_scorer_audit import (
            round_half_away_div,
        )
        from evaluation.selector_int8_numerical_audit import (
            quantize_activation,
        )

        self.assertEqual(round_half_away_div(1, 2), 1)
        self.assertEqual(round_half_away_div(3, 2), 2)
        self.assertEqual(round_half_away_div(-1, 2), -1)
        quantized, scale = quantize_activation(
            np.array([[-1e9, 0.0, 1e9]], dtype=np.float64)
        )
        self.assertTrue(np.isfinite(scale))
        self.assertEqual(int(quantized.min()), -128)
        self.assertEqual(int(quantized.max()), 127)


if __name__ == "__main__":
    unittest.main()
