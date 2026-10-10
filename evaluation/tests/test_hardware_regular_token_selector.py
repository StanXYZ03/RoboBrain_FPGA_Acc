from __future__ import annotations

import json
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.hardware_regular_token_selector import (
    ALLOWED_BOUNDARIES,
    BOUNDARY_ABLATION,
    BOUNDARY_MAIN,
    FEATURE_DIM,
    FPGA_LAYOUT_ALIGNMENT,
    MAX_CONTEXT_SLOTS,
    MERGE_UNIT,
    PROMPT_DIM,
    REGION_GRID,
    REQUIRED_BUDGET_ALIGNMENT,
    SCORE_DIM,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    InsufficientBudgetError,
    LowRankProjection,
    SelectorContractError,
    align_up,
    build_regions,
    combine_scores,
    context_pooling_weights,
    default_config_path,
    describe_fpga_layout,
    fpga_segment_offsets,
    fpga_segment_sizes,
    fpga_transfer_layout,
    group_l2_saliency,
    int8_token_bytes,
    load_selector_config,
    minmax_normalize,
    pool_region_context,
    region_context_representative,
    region_id_by_group,
    score_groups,
    scores_from_precomputed,
    select_hardware_regular_groups,
    select_real_groups,
    semantic_cosine_scores,
    uint16_group_index_bytes,
    uint8_region_bytes,
    uint8_source_bytes,
    uint8_valid_bytes,
    validate_budget,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _hidden(grid_h: int, grid_w: int) -> np.ndarray:
    """Deterministic non-degenerate [G, 4, 1024] fixture (no model weights)."""
    group = np.arange(grid_h * grid_w)[:, None, None]
    patch = np.arange(MERGE_UNIT)[None, :, None]
    feature = np.arange(FEATURE_DIM)[None, None, :]
    return 1.5 + np.sin(0.7 * group + 1.3 * patch + 0.11 * feature)


def _projection() -> LowRankProjection:
    """Explicit deterministic low-rank matrices; never randomly generated."""
    row = np.arange(SCORE_DIM)[:, None]
    text_col = np.arange(PROMPT_DIM)[None, :]
    feature_col = np.arange(FEATURE_DIM)[None, :]
    Wt = ((row * 13 + text_col * 7) % 11 - 5) / 5.0
    Wv = ((row * 5 + feature_col * 3) % 7 - 3) / 3.0
    return LowRankProjection(Wt=Wt, Wv=Wv)


def _text() -> np.ndarray:
    row = np.arange(5)[:, None]
    col = np.arange(PROMPT_DIM)[None, :]
    return 2.0 + np.cos(0.3 * row + 0.05 * col)


def _select(
    grid_h: int,
    grid_w: int,
    budget: int,
    **kwargs,
):
    return select_hardware_regular_groups(
        _hidden(grid_h, grid_w),
        grid_h,
        grid_w,
        budget=budget,
        text_embeddings=_text(),
        projection=_projection(),
        lambda_semantic=0.5,
        **kwargs,
    )


def _context_eligible_regions(grid_h: int, grid_w: int):
    return [region for region in build_regions(grid_h, grid_w) if len(region.members) >= 2]


class HardwareRegularTokenSelectorTest(unittest.TestCase):
    # -- budget / structure -------------------------------------------------
    def test_strict_budget_dynamic_context_and_unique_representatives(self) -> None:
        cases = [
            ((9, 9), 48),
            ((9, 9), 64),
            ((9, 9), 32),
            ((11, 11), 96),
            ((10, 12), 64),
            ((7, 8), 48),
        ]
        for (grid_h, grid_w), budget in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                self.assertGreater(group_count, budget)
                result = _select(grid_h, grid_w, budget)
                eligible = _context_eligible_regions(grid_h, grid_w)
                expected_context = len(eligible)
                self.assertLessEqual(expected_context, MAX_CONTEXT_SLOTS)

                self.assertFalse(result.bypass)
                self.assertEqual(result.output_group_count, budget)
                self.assertEqual(budget % REQUIRED_BUDGET_ALIGNMENT, 0)
                self.assertEqual(
                    result.real_group_count, budget - expected_context
                )
                self.assertEqual(result.context_group_count, expected_context)
                self.assertEqual(
                    result.real_group_count + result.context_group_count, budget
                )

                real_mask = result.source_type == SOURCE_REAL
                real_ids = result.group_indices[real_mask].tolist()
                self.assertEqual(len(real_ids), len(set(real_ids)))
                self.assertTrue(all(0 <= group < group_count for group in real_ids))

                lookup = region_id_by_group(grid_h, grid_w)
                per_region = Counter(lookup[group] for group in real_ids)
                non_empty = 0
                for region in build_regions(grid_h, grid_w):
                    if region.members:
                        non_empty += 1
                        self.assertGreaterEqual(
                            per_region[region.region_id],
                            1,
                            f"region {region.region_id} lost its anchor",
                        )
                self.assertEqual(len(result.anchors), non_empty)
                self.assertEqual(len(set(result.anchors)), len(result.anchors))
                self.assertEqual(
                    len(set(result.anchors) | set(result.details)),
                    budget - expected_context,
                )
                self.assertEqual(
                    sorted(result.anchors + result.details), sorted(real_ids)
                )

                # Global uniqueness across real and context representatives.
                representative_ids = result.group_indices.tolist()
                self.assertEqual(
                    len(set(representative_ids)), len(representative_ids)
                )
                # No invalid padding: every entry is a genuine group.
                self.assertTrue(result.valid.all())
                self.assertNotIn(-1, representative_ids)

    def test_real_groups_keep_four_contiguous_patches_and_values(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        result = _select(grid_h, grid_w, budget)
        expected_offsets = np.arange(MERGE_UNIT, dtype=np.int64)
        real_mask = result.source_type == SOURCE_REAL
        for index in np.flatnonzero(real_mask):
            group = int(result.group_indices[index])
            self.assertEqual(
                result.patch_indices[index].tolist(),
                (group * MERGE_UNIT + expected_offsets).tolist(),
            )
            np.testing.assert_array_equal(result.groups[index], hidden[group])

    def test_identity_bypass_is_elementwise_unchanged(self) -> None:
        for grid_h, grid_w, budget in [(4, 4, 48), (6, 6, 48), (8, 8, 64)]:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                hidden = _hidden(grid_h, grid_w)
                result = select_hardware_regular_groups(
                    hidden, grid_h, grid_w, budget=budget
                )
                self.assertTrue(result.bypass)
                np.testing.assert_array_equal(result.groups, hidden)
                self.assertEqual(result.output_group_count, hidden.shape[0])
                self.assertEqual(
                    result.group_indices.tolist(),
                    list(range(hidden.shape[0])),
                )
                self.assertEqual(result.context_group_count, 0)
                self.assertTrue((result.source_type == SOURCE_REAL).all())
                self.assertTrue(result.valid.all())
                self.assertIsNone(result.scores)

    def test_context_representative_is_unselected_weighted_centroid_group(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        result = _select(grid_h, grid_w, budget)
        offsets = np.arange(MERGE_UNIT, dtype=np.int64)
        real_ids = set(
            result.group_indices[result.source_type == SOURCE_REAL].tolist()
        )
        for region in _context_eligible_regions(grid_h, grid_w):
            locations = np.flatnonzero(
                (result.source_type == SOURCE_CONTEXT)
                & (result.region_id == region.region_id)
            )
            self.assertEqual(locations.size, 1)
            index = int(locations[0])
            remaining = [group for group in region.members if group not in real_ids]
            self.assertTrue(remaining, "context region lost its unselected group")
            weights = context_pooling_weights(result.scores.combined, remaining)
            expected = region_context_representative(region, remaining, weights)
            self.assertEqual(int(result.group_indices[index]), expected)
            # The representative must itself be an unselected group of the region.
            self.assertIn(expected, remaining)
            self.assertNotIn(expected, real_ids)
            self.assertEqual(
                result.patch_indices[index].tolist(),
                (expected * MERGE_UNIT + offsets).tolist(),
            )
            self.assertEqual(
                int(result.region_id[index]), region.region_id
            )

    # -- context pooling ----------------------------------------------------
    def test_context_pooling_numeric_and_empty_region(self) -> None:
        hidden = np.zeros((3, MERGE_UNIT, FEATURE_DIM))
        hidden[0] = 1.0
        hidden[1] = 5.0
        hidden[2] = 3.0
        combined = np.array([1.0, -1.0, 3.0])

        tokens, valid = pool_region_context(hidden, [0, 2], combined)
        self.assertTrue(valid)
        expected = 0.25 * hidden[0] + 0.75 * hidden[2]
        np.testing.assert_allclose(tokens, expected)

        weights = context_pooling_weights(combined, [0, 2])
        np.testing.assert_allclose(weights, [0.25, 0.75])
        self.assertGreaterEqual(weights.min(), 0.0)
        self.assertAlmostEqual(float(weights.sum()), 1.0)

        uniform = context_pooling_weights(np.array([-1.0, -2.0, -3.0]), [0, 1, 2])
        np.testing.assert_allclose(uniform, [1.0 / 3.0] * 3)

        empty_tokens, empty_valid = pool_region_context(hidden, [], combined)
        self.assertFalse(empty_valid)
        np.testing.assert_array_equal(empty_tokens, np.zeros((MERGE_UNIT, FEATURE_DIM)))
        self.assertTrue(np.isfinite(empty_tokens).all())

    def test_end_to_end_context_matches_remaining_group_pooling(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        result = _select(grid_h, grid_w, budget)
        real_ids = set(
            result.group_indices[result.source_type == SOURCE_REAL].tolist()
        )
        for region in _context_eligible_regions(grid_h, grid_w):
            locations = np.flatnonzero(
                (result.source_type == SOURCE_CONTEXT)
                & (result.region_id == region.region_id)
            )
            self.assertEqual(locations.size, 1)
            index = int(locations[0])
            remaining = [group for group in region.members if group not in real_ids]
            expected, expected_valid = pool_region_context(
                hidden, remaining, result.scores.combined
            )
            self.assertTrue(expected_valid)
            self.assertTrue(bool(result.valid[index]))
            np.testing.assert_allclose(result.groups[index], expected)
        self.assertTrue(np.isfinite(result.groups).all())

    def test_forbidden_zero_padding_for_empty_and_single_group_regions(self) -> None:
        # 5x4 leaves two regions with exactly one member and no empty region;
        # those regions must keep their single real anchor and emit no context.
        grid_h, grid_w, budget = 5, 4, 16
        result = _select(grid_h, grid_w, budget)
        self.assertEqual(result.output_group_count, budget)
        self.assertTrue(result.valid.all())
        self.assertNotIn(-1, result.group_indices.tolist())
        self.assertTrue((result.patch_indices >= 0).all())
        self.assertFalse(np.all(result.groups == 0.0, axis=(1, 2)).any())

        small_regions = [
            region for region in build_regions(grid_h, grid_w) if len(region.members) < 2
        ]
        self.assertTrue(small_regions)
        for region in small_regions:
            context_hits = np.flatnonzero(
                (result.source_type == SOURCE_CONTEXT)
                & (result.region_id == region.region_id)
            )
            self.assertEqual(context_hits.size, 0)
        eligible = _context_eligible_regions(grid_h, grid_w)
        self.assertEqual(result.context_group_count, len(eligible))

        # 2x9 additionally contains three empty regions that must be skipped.
        empty_grid_h, empty_grid_w = 2, 9
        empty_result = _select(empty_grid_h, empty_grid_w, 16)
        self.assertEqual(empty_result.output_group_count, 16)
        self.assertTrue(empty_result.valid.all())
        for region in build_regions(empty_grid_h, empty_grid_w):
            if not region.members:
                hits = np.flatnonzero(
                    (empty_result.source_type == SOURCE_CONTEXT)
                    & (empty_result.region_id == region.region_id)
                )
                self.assertEqual(hits.size, 0)
        self.assertEqual(
            empty_result.context_group_count,
            len(_context_eligible_regions(empty_grid_h, empty_grid_w)),
        )

    def test_context_reservation_keeps_one_unselected_group_per_region(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        result = _select(grid_h, grid_w, budget)
        real_ids = set(
            result.group_indices[result.source_type == SOURCE_REAL].tolist()
        )
        for region in _context_eligible_regions(grid_h, grid_w):
            remaining = [group for group in region.members if group not in real_ids]
            self.assertGreaterEqual(
                len(remaining),
                1,
                f"region {region.region_id} has no unselected group for its context",
            )

        # Direct check on the reservation-aware two-level Top-K.
        combined = np.linspace(1.0, 0.0, 6 * 6, dtype=np.float64)
        selection = select_real_groups(
            combined, 6, 6, 20, context_region_ids=[0, 4, 8]
        )
        selected = set(selection.selected)
        for region_id in (0, 4, 8):
            region = build_regions(6, 6)[region_id]
            remaining = [group for group in region.members if group not in selected]
            self.assertTrue(remaining)

    # -- determinism --------------------------------------------------------
    def test_output_indices_are_stable_ascending_and_tie_deterministic(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        arguments = dict(
            budget=budget,
            text_embeddings=_text(),
            projection=_projection(),
            lambda_semantic=0.5,
        )
        first = select_hardware_regular_groups(hidden, grid_h, grid_w, **arguments)
        second = select_hardware_regular_groups(hidden, grid_h, grid_w, **arguments)
        self.assertTrue(
            all(
                int(first.group_indices[index]) <= int(first.group_indices[index + 1])
                for index in range(first.output_group_count - 1)
            )
        )
        np.testing.assert_array_equal(first.group_indices, second.group_indices)
        np.testing.assert_array_equal(first.groups, second.groups)
        np.testing.assert_array_equal(first.patch_indices, second.patch_indices)
        np.testing.assert_array_equal(first.valid, second.valid)
        self.assertEqual(first.anchors, second.anchors)
        self.assertEqual(first.details, second.details)

    def test_equal_scores_break_ties_by_lower_original_index(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = np.ones((grid_h * grid_w, MERGE_UNIT, FEATURE_DIM))
        projection = LowRankProjection(
            Wt=np.ones((SCORE_DIM, PROMPT_DIM)),
            Wv=np.ones((SCORE_DIM, FEATURE_DIM)),
        )
        text = np.ones((3, PROMPT_DIM))
        result = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            text_embeddings=text,
            projection=projection,
            lambda_semantic=0.5,
        )
        np.testing.assert_allclose(result.scores.semantic, 1.0)
        for region in build_regions(grid_h, grid_w):
            expected_anchor = min(region.members)
            self.assertIn(expected_anchor, result.anchors)
        repeat = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            text_embeddings=text,
            projection=projection,
            lambda_semantic=0.5,
        )
        self.assertEqual(result.anchors, repeat.anchors)
        self.assertEqual(result.details, repeat.details)
        self.assertEqual(
            result.group_indices.tolist(), repeat.group_indices.tolist()
        )

    def test_two_level_topk_uses_only_bounded_local_candidates(self) -> None:
        grid_h, grid_w = 6, 6
        combined = np.full(grid_h * grid_w, 0.001, dtype=np.float64)
        # Region 0 members are groups 0, 1, 6, 7.
        combined[0] = 5.0
        combined[1] = 100.0
        combined[6] = 99.0
        combined[7] = 1.0
        selection = select_real_groups(
            combined,
            grid_h,
            grid_w,
            10,
            region_grid=REGION_GRID,
            local_candidate_cap=2,
        )
        self.assertIn(1, selection.selected)
        self.assertIn(6, selection.selected)
        self.assertNotIn(0, selection.selected)
        self.assertEqual(selection.effective_candidate_cap, 2)

    # -- scores -------------------------------------------------------------
    def test_score_interfaces_are_separate_and_combined_is_normalized(self) -> None:
        grid_h, grid_w = 9, 9
        hidden = _hidden(grid_h, grid_w)
        projection = _projection()
        text = _text()
        scores = score_groups(hidden, text, projection, lambda_semantic=0.25)
        self.assertEqual(scores.semantic.shape, (grid_h * grid_w,))
        self.assertEqual(scores.l2.shape, (grid_h * grid_w,))
        self.assertEqual(scores.combined.shape, (grid_h * grid_w,))
        self.assertLessEqual(float(scores.semantic.max()), 1.0 + 1e-9)
        self.assertGreaterEqual(float(scores.semantic.min()), -1.0 - 1e-9)
        self.assertTrue((scores.l2 >= 0.0).all())
        self.assertGreaterEqual(float(scores.combined.min()), -1e-12)
        self.assertLessEqual(float(scores.combined.max()), 1.0 + 1e-12)
        self.assertAlmostEqual(scores.lambda_semantic, 0.25)

        l2_only = combine_scores(scores.semantic, scores.l2, 0.0)
        semantic_only = combine_scores(scores.semantic, scores.l2, 1.0)
        expected_l2 = (scores.l2 - scores.l2.min()) / (
            scores.l2.max() - scores.l2.min()
        )
        np.testing.assert_allclose(l2_only, expected_l2, atol=1e-12)
        expected_semantic = (scores.semantic - scores.semantic.min()) / (
            scores.semantic.max() - scores.semantic.min()
        )
        np.testing.assert_allclose(semantic_only, expected_semantic, atol=1e-12)
        # Inputs are reported untouched; the combination is a separate array.
        self.assertFalse(np.shares_memory(scores.combined, scores.semantic))
        self.assertFalse(np.shares_memory(scores.combined, scores.l2))
        self.assertAlmostEqual(
            float(semantic_cosine_scores(hidden, text, projection)[0]),
            float(scores.semantic[0]),
        )
        np.testing.assert_allclose(group_l2_saliency(hidden), scores.l2)

    # -- FPGA layout --------------------------------------------------------
    def test_fpga_layout_offsets_are_aligned_and_total_is_correct(self) -> None:
        for budget in (48, 64, 96):
            with self.subTest(budget=budget):
                layout = fpga_transfer_layout(budget)
                self.assertEqual(layout.alignment, FPGA_LAYOUT_ALIGNMENT)
                for segment in layout.segments:
                    self.assertEqual(segment.offset % FPGA_LAYOUT_ALIGNMENT, 0)
                    self.assertEqual(
                        segment.size, segment.element_bytes * segment.element_count
                    )
                self.assertEqual(
                    layout.total_bytes,
                    align_up(layout.segments[-1].end, FPGA_LAYOUT_ALIGNMENT),
                )
                self.assertEqual(layout.total_bytes % FPGA_LAYOUT_ALIGNMENT, 0)
                self.assertGreaterEqual(
                    layout.total_bytes, sum(seg.size for seg in layout.segments)
                )
                self.assertEqual(
                    layout.offsets()["tokens_int8"], 0
                )
                self.assertEqual(layout.sizes()["tokens_int8"], int8_token_bytes(budget))
                self.assertEqual(
                    layout.sizes()["group_index_uint16"],
                    uint16_group_index_bytes(budget),
                )
                self.assertEqual(
                    layout.sizes()["source_type_uint8"], uint8_source_bytes(budget)
                )
                self.assertEqual(
                    layout.sizes()["region_id_uint8"], uint8_region_bytes(budget)
                )
                self.assertEqual(
                    layout.sizes()["valid_uint8"], uint8_valid_bytes(budget)
                )
                self.assertEqual(fpga_segment_offsets(budget), layout.offsets())
                self.assertEqual(fpga_segment_sizes(budget), layout.sizes())
                description = describe_fpga_layout(budget)
                self.assertEqual(description["budget"], budget)
                self.assertEqual(description["alignment"], FPGA_LAYOUT_ALIGNMENT)
                self.assertEqual(description["total_bytes"], layout.total_bytes)
                self.assertEqual(len(description["segments"]), len(layout.segments))
        self.assertEqual(align_up(0), 0)
        self.assertEqual(align_up(1), 64)
        self.assertEqual(align_up(64), 64)
        self.assertEqual(align_up(65), 128)

    # -- errors -------------------------------------------------------------
    def test_invalid_dimensions_and_budget_raise(self) -> None:
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                np.zeros((16, FEATURE_DIM)), 4, 4, budget=48
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                np.zeros((16, MERGE_UNIT - 1, FEATURE_DIM)), 4, 4, budget=48
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                np.zeros((16, MERGE_UNIT, FEATURE_DIM - 1)), 4, 4, budget=48
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                np.zeros((16, MERGE_UNIT, FEATURE_DIM)), 5, 5, budget=48
            )
        for bad_budget in (0, -16, 50, 17):
            with self.subTest(budget=bad_budget):
                with self.assertRaises(SelectorContractError):
                    validate_budget(bad_budget)
        with self.assertRaises(SelectorContractError):
            validate_budget("48")
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                np.zeros((16, MERGE_UNIT, FEATURE_DIM)), 4, 4, budget=50
            )
        with self.assertRaises(SelectorContractError):
            LowRankProjection(
                Wt=np.zeros((SCORE_DIM - 1, PROMPT_DIM)),
                Wv=np.zeros((SCORE_DIM, FEATURE_DIM)),
            )
        with self.assertRaises(SelectorContractError):
            combine_scores(np.zeros(4), np.zeros(4), 1.5)
        with self.assertRaises(SelectorContractError):
            _select(9, 9, 48, max_context_slots=0)

    def test_compression_requires_prompt_interface_and_boundary(self) -> None:
        hidden = _hidden(9, 9)
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(hidden, 9, 9, budget=48)
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                hidden,
                9,
                9,
                budget=48,
                text_embeddings=_text(),
                projection=_projection(),
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                hidden,
                9,
                9,
                budget=48,
                text_embeddings=np.zeros((4, PROMPT_DIM - 1)),
                projection=_projection(),
                lambda_semantic=0.5,
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                hidden,
                9,
                9,
                budget=48,
                text_embeddings=_text(),
                projection=_projection(),
                lambda_semantic=0.5,
                boundary="B2",
            )
        result = select_hardware_regular_groups(
            hidden,
            9,
            9,
            budget=48,
            text_embeddings=_text(),
            projection=_projection(),
            lambda_semantic=0.5,
            boundary=BOUNDARY_ABLATION,
        )
        self.assertEqual(result.boundary, BOUNDARY_ABLATION)
        self.assertIn(BOUNDARY_ABLATION, ALLOWED_BOUNDARIES)
        self.assertEqual(BOUNDARY_MAIN, "B1")

    def test_insufficient_budget_is_reported_explicitly(self) -> None:
        # 81 groups -> 9 context-eligible regions, but K=16 leaves only 7 real
        # groups, fewer than the 9 non-empty anchors.
        with self.assertRaises(InsufficientBudgetError):
            _select(9, 9, 16)
        # A deliberately tiny local candidate cap cannot fill the detail quota.
        with self.assertRaises(InsufficientBudgetError):
            _select(7, 7, 48, local_candidate_cap=1)

    # -- explicit precomputed scores ---------------------------------------
    def test_l2_only_explicit_scores_need_no_projection(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        l2 = group_l2_saliency(hidden)
        combined = minmax_normalize(l2)
        result = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            l2_scores=l2,
            combined_scores=combined,
        )
        self.assertFalse(result.bypass)
        self.assertEqual(result.output_group_count, budget)
        self.assertIsNone(result.scores.semantic)
        np.testing.assert_allclose(result.scores.l2, l2)
        np.testing.assert_allclose(result.scores.combined, combined)
        self.assertIsNone(result.lambda_semantic)
        # Equivalent to the prompt path evaluated purely on the L2 term.
        reference = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            text_embeddings=_text(),
            projection=_projection(),
            lambda_semantic=0.0,
        )
        np.testing.assert_array_equal(result.group_indices, reference.group_indices)
        np.testing.assert_allclose(
            result.scores.combined, reference.scores.combined
        )

    def test_explicit_scores_match_prompt_path_selection(self) -> None:
        grid_h, grid_w, budget = 9, 9, 64
        hidden = _hidden(grid_h, grid_w)
        scores = score_groups(hidden, _text(), _projection(), lambda_semantic=0.5)
        explicit = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            semantic_scores=scores.semantic,
            l2_scores=scores.l2,
            combined_scores=scores.combined,
            lambda_semantic=0.5,
        )
        reference = select_hardware_regular_groups(
            hidden,
            grid_h,
            grid_w,
            budget=budget,
            text_embeddings=_text(),
            projection=_projection(),
            lambda_semantic=0.5,
        )
        np.testing.assert_array_equal(explicit.group_indices, reference.group_indices)
        np.testing.assert_allclose(explicit.groups, reference.groups)
        self.assertAlmostEqual(explicit.lambda_semantic, 0.5)

    def test_scores_from_precomputed_derives_or_rejects(self) -> None:
        grid_h, grid_w = 9, 9
        hidden = _hidden(grid_h, grid_w)
        scores = score_groups(hidden, _text(), _projection(), lambda_semantic=0.25)
        derived = scores_from_precomputed(scores.semantic, scores.l2, None, 0.25)
        np.testing.assert_allclose(derived.combined, scores.combined)
        with self.assertRaises(SelectorContractError):
            scores_from_precomputed(scores.semantic, None, None, 0.25)
        with self.assertRaises(SelectorContractError):
            scores_from_precomputed(scores.semantic, scores.l2, None, None)
        with self.assertRaises(SelectorContractError):
            scores_from_precomputed()

    def test_explicit_score_size_mismatch_raises(self) -> None:
        hidden = _hidden(9, 9)
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                hidden, 9, 9, budget=64, combined_scores=np.zeros(80)
            )
        with self.assertRaises(SelectorContractError):
            select_hardware_regular_groups(
                hidden,
                9,
                9,
                budget=64,
                l2_scores=np.zeros(80),
                combined_scores=np.zeros(81),
            )
        with self.assertRaises(SelectorContractError):
            scores_from_precomputed(
                semantic=np.zeros(80), l2=np.zeros(81), combined=None,
                lambda_semantic=0.5,
            )

    def test_minmax_normalize_matches_combined_l2_term(self) -> None:
        values = np.array([1.0, 5.0, 3.0, -2.0])
        np.testing.assert_allclose(
            minmax_normalize(values), combine_scores(values, values, 0.0)
        )
        np.testing.assert_allclose(
            minmax_normalize(values), combine_scores(values, values, 1.0)
        )

    # -- config and environment independence --------------------------------
    def test_config_matches_module_constants(self) -> None:
        config = load_selector_config()
        self.assertEqual(config["boundary"], BOUNDARY_MAIN)
        self.assertEqual(sorted(config["budgets"]), [48, 64, 96])
        self.assertEqual(int(config["alignment"]), REQUIRED_BUDGET_ALIGNMENT)
        self.assertEqual(config["context_slots_policy"], "dynamic_up_to_max")
        self.assertEqual(int(config["max_context_slots"]), MAX_CONTEXT_SLOTS)
        self.assertEqual(int(config["max_context_slots"]), 9)
        self.assertEqual(int(config["region_grid"]), REGION_GRID)
        self.assertEqual(int(config["feature_dim"]), FEATURE_DIM)
        self.assertEqual(int(config["prompt_dim"]), PROMPT_DIM)
        self.assertEqual(int(config["score_dim"]), SCORE_DIM)
        self.assertEqual(int(config["merge_unit"]), MERGE_UNIT)
        self.assertIn("lambda_semantic", config)
        self.assertNotIn("context_slots", config)
        self.assertEqual(default_config_path().name, "hardware_regular_token_selector.json")
        self.assertTrue(default_config_path().exists())
        with open(default_config_path(), "r", encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["training"]["enabled"], False)

    def test_module_does_not_import_torch_or_cuda(self) -> None:
        script = (
            "import sys;"
            "import evaluation.hardware_regular_token_selector as m;"
            "assert 'torch' not in sys.modules, 'torch was imported';"
            "assert getattr(m, 'torch', None) is None;"
            "print(m.MERGE_UNIT, m.FEATURE_DIM)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), f"{MERGE_UNIT} {FEATURE_DIM}")


if __name__ == "__main__":
    unittest.main()
