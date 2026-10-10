"""CPU fake-visual tests for the layer-5 forced-index / control ablation.

These tests never load real weights and never require CUDA.  They exercise the
single unified forward of the new module, the control rule, the forced exact
historical indices, the relaxed ``K_C=25`` alignment, the identity bypass and
the DeepStack cut point.  No existing module is imported for control flow; the
reference fake vision tower (``hidden=1024``, ``merge_unit=4``,
``out_hidden=2560``, DeepStack at blocks 5/11/17, 24 blocks) is reproduced here.
"""

from __future__ import annotations

import math
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.hardware_regular_layer5_forced_index_ablation import (
    ALLOWED_MODES,
    BUDGET,
    CAUSAL_DIAGNOSTIC_ONLY,
    FPGA_DEPLOYABLE_CANDIDATE,
    HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC,
    MODE_CONTROL,
    MODE_FORCED,
    PRUNE_AFTER_LAYER,
    REGION_GRID,
    SELECTION_RULE_CONTROL,
    SELECTION_RULE_FORCED,
    forward_hardware_regular_layer5_forced_index_ablation_vision,
    normalize_forced_group_indices,
    select_control_groups,
    select_forced_groups,
    validate_diagnostic_budget,
    verify_diagnostic_selection,
)
from evaluation.hardware_regular_token_selector import (
    FEATURE_DIM,
    SOURCE_CONTEXT,
    SOURCE_REAL,
    SelectorContractError,
    build_regions,
    group_l2_saliency,
    minmax_normalize,
)

OUT_HIDDEN = 2560
DEPTH = 24
DEEPSTACK_INDEXES = [5, 11, 17]
#: grid=12 -> G=36 merged groups; K=25 is deliberately not a multiple of 16.
COMPRESS_GRID = 12
COMPRESS_BUDGET = 25
COMPRESS_GROUPS = COMPRESS_GRID * COMPRESS_GRID // 4


class _FakeBlock(nn.Module):
    """Position-aware block: output depends on the gathered cos/sin."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden, cu_seqlens=None, position_embeddings=None):
        output = hidden * self.scale
        if position_embeddings is None:
            return output
        cos, sin = position_embeddings
        rotary = torch.cat([cos, sin], dim=-1)
        if rotary.shape[-1] < self.hidden:
            rotary = F.pad(rotary, (0, self.hidden - rotary.shape[-1]))
        else:
            rotary = rotary[..., : self.hidden]
        return output + rotary


class _GroupMerger(nn.Module):
    """[N, 1024] -> view(-1, 4096) -> [N, 2560] with deterministic weights."""

    def __init__(self, hidden: int, out_hidden: int, salt: int = 0) -> None:
        super().__init__()
        fan_in = hidden * 4
        self.proj = nn.Linear(fan_in, out_hidden, bias=False)
        weight = (
            torch.arange(out_hidden * fan_in, dtype=torch.float32).reshape(
                out_hidden, fan_in
            )
            + salt
        ) % 17 - 8.0
        with torch.no_grad():
            self.proj.weight.copy_(weight / fan_in)

    def forward(self, hidden):
        return self.proj(hidden.reshape(-1, hidden.shape[-1] * 4))


class _FakeVisual(nn.Module):
    dtype = torch.float32
    spatial_merge_size = 2
    deepstack_visual_indexes = list(DEEPSTACK_INDEXES)

    def __init__(self, hidden: int = FEATURE_DIM, depth: int = DEPTH) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.blocks = nn.ModuleList(_FakeBlock(hidden) for _ in range(depth))
        self.deepstack_merger_list = nn.ModuleList(
            _GroupMerger(hidden, OUT_HIDDEN, salt=index)
            for index in range(len(DEEPSTACK_INDEXES))
        )
        self.merger = _GroupMerger(hidden, OUT_HIDDEN, salt=99)
        self.hidden = hidden

    def fast_pos_embed_interpolate(self, grid):
        count = int(grid.prod().item())
        return torch.zeros(count, self.hidden)

    def rot_pos_emb(self, grid):
        count = int(grid.prod().item())
        base = torch.arange(count, dtype=torch.float32).unsqueeze(1)
        return torch.cat([base * 0.01, base * 0.03], dim=1)


class _RecordingMerger(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, hidden):
        self.input_shapes.append(tuple(hidden.shape))
        return self.inner(hidden)


def _pixels(grid: int, hidden: int = FEATURE_DIM) -> torch.Tensor:
    count = grid * grid
    base = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    feature = torch.arange(hidden, dtype=torch.float32).unsqueeze(0)
    return torch.sin(0.01 * base + 0.001 * feature)


def _grid(grid: int) -> torch.Tensor:
    return torch.tensor([[1, grid, grid]])


def _forward(visual, grid, budget, **kwargs):
    return forward_hardware_regular_layer5_forced_index_ablation_vision(
        visual, _pixels(grid), _grid(grid), budget=budget, **kwargs
    )


def _combined_scores(result) -> np.ndarray:
    raw = (
        result.cut_hidden.detach()
        .to(torch.float64)
        .reshape(result.full_llm_visual_tokens, 4, FEATURE_DIM)
        .numpy()
    )
    return minmax_normalize(group_l2_saliency(raw))


class ControlRuleTest(unittest.TestCase):
    def test_constants_and_diagnostic_metadata(self) -> None:
        self.assertEqual(PRUNE_AFTER_LAYER, 5)
        self.assertEqual(REGION_GRID, 3)
        self.assertEqual(ALLOWED_MODES, ("control", "forced"))
        self.assertTrue(HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC)
        self.assertTrue(CAUSAL_DIAGNOSTIC_ONLY)
        self.assertFalse(FPGA_DEPLOYABLE_CANDIDATE)
        result = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        self.assertIs(result.hardware_alignment_relaxed_for_diagnostic, True)
        self.assertIs(result.causal_diagnostic_only, True)
        self.assertIs(result.fpga_deployable_candidate, False)
        self.assertEqual(result.selection_rule, SELECTION_RULE_CONTROL)

    def test_budget_25_is_accepted_and_emits_exactly_k_all_real(self) -> None:
        self.assertEqual(validate_diagnostic_budget(COMPRESS_BUDGET), 25)
        result = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        self.assertEqual(result.mode, MODE_CONTROL)
        self.assertFalse(result.selection.bypass)
        self.assertEqual(result.real_group_count, 25)
        self.assertEqual(result.context_group_count, 0)
        self.assertEqual(int(result.pruned.image_embeds.shape[0]), 25)
        self.assertEqual(result.group_indices.size, 25)
        self.assertTrue(np.all(result.source_type == SOURCE_REAL))
        self.assertFalse(np.any(result.source_type == SOURCE_CONTEXT))
        verify_diagnostic_selection(result.selection)

    def test_control_rule_is_global_l2_plus_one_anchor_per_region(self) -> None:
        result = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        combined = _combined_scores(result)
        regions = [
            region for region in build_regions(6, 6, 3) if region.members
        ]
        anchors = {
            max(region.members, key=lambda group: (combined[group], -group))
            for region in regions
        }
        self.assertEqual(set(result.selection.anchors), anchors)
        non_anchor = [group for group in range(COMPRESS_GROUPS) if group not in anchors]
        expected_details = sorted(
            non_anchor, key=lambda group: (-combined[group], group)
        )[: COMPRESS_BUDGET - len(anchors)]
        self.assertEqual(set(result.selection.details), set(expected_details))
        self.assertEqual(
            list(result.group_indices), sorted(anchors | set(expected_details))
        )
        self.assertTrue(bool(np.all(np.diff(result.group_indices) > 0)))


class ForcedIndexTest(unittest.TestCase):
    def test_forced_keeps_exactly_the_sorted_historical_indices(self) -> None:
        forced = [(7 * index) % COMPRESS_GROUPS for index in range(COMPRESS_BUDGET)]
        expected = np.sort(np.asarray(forced, dtype=np.int64))
        result = _forward(
            _FakeVisual(),
            COMPRESS_GRID,
            COMPRESS_BUDGET,
            mode=MODE_FORCED,
            forced_group_indices=forced,
        )
        self.assertEqual(result.mode, MODE_FORCED)
        self.assertEqual(result.selection_rule, SELECTION_RULE_FORCED)
        np.testing.assert_array_equal(result.group_indices, expected)
        self.assertEqual(result.real_group_count, COMPRESS_BUDGET)
        self.assertEqual(result.context_group_count, 0)
        self.assertTrue(np.all(result.source_type == SOURCE_REAL))
        # Forced output must not silently fall back to the control rule.
        control = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        self.assertFalse(
            np.array_equal(result.group_indices, control.group_indices)
        )

    def test_forced_shuffled_input_is_deterministically_sorted(self) -> None:
        rng = np.random.default_rng(0)
        shuffled = rng.permutation(COMPRESS_GROUPS)[:COMPRESS_BUDGET]
        self.assertEqual(np.unique(shuffled).size, COMPRESS_BUDGET)
        result = _forward(
            _FakeVisual(),
            COMPRESS_GRID,
            COMPRESS_BUDGET,
            mode=MODE_FORCED,
            forced_group_indices=shuffled,
        )
        np.testing.assert_array_equal(
            result.group_indices, np.sort(shuffled.astype(np.int64))
        )
        self.assertTrue(bool(np.all(np.diff(result.group_indices) > 0)))

    def test_same_indices_give_bit_identical_control_and_forced_outputs(self) -> None:
        control = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        forced = _forward(
            _FakeVisual(),
            COMPRESS_GRID,
            COMPRESS_BUDGET,
            mode=MODE_FORCED,
            forced_group_indices=control.group_indices.copy(),
        )
        for name in (
            "group_indices",
            "patch_indices",
            "source_type",
            "region_id",
            "valid",
        ):
            np.testing.assert_array_equal(
                getattr(control, name), getattr(forced, name)
            )
        self.assertEqual(control.context_group_count, forced.context_group_count)
        self.assertEqual(control.real_group_count, forced.real_group_count)
        self.assertEqual(control.identity_bypass, forced.identity_bypass)
        self.assertTrue(torch.equal(control.compact_hidden, forced.compact_hidden))
        self.assertEqual(control.block_input_shapes, forced.block_input_shapes)
        self.assertTrue(
            torch.equal(control.pruned.image_embeds, forced.pruned.image_embeds)
        )
        self.assertTrue(torch.equal(control.pruned.group_scores, forced.pruned.group_scores))
        self.assertTrue(torch.equal(control.pruned.group_l2, forced.pruned.group_l2))
        self.assertEqual(
            len(control.pruned.deepstack_embeds), len(forced.pruned.deepstack_embeds)
        )
        for left, right in zip(
            control.pruned.deepstack_embeds, forced.pruned.deepstack_embeds
        ):
            self.assertTrue(torch.equal(left, right))

    def test_illegal_forced_indices_are_rejected(self) -> None:
        hidden = np.zeros((COMPRESS_GROUPS, 4, FEATURE_DIM), dtype=np.float64)
        duplicates = np.arange(COMPRESS_BUDGET, dtype=np.int64)
        duplicates[1] = 0
        negative = np.arange(COMPRESS_BUDGET, dtype=np.int64)
        negative[0] = -1
        out_of_range = np.arange(COMPRESS_BUDGET, dtype=np.int64)
        out_of_range[-1] = COMPRESS_GROUPS
        cases = {
            "2d": np.zeros((5, 5), dtype=np.int64),
            "3d": np.zeros((5, 5, 1), dtype=np.int64),
            "scalar": 5,
            "float": np.arange(COMPRESS_BUDGET, dtype=np.float64),
            "bool": np.ones(COMPRESS_BUDGET, dtype=bool),
            "wrong_length": np.arange(COMPRESS_BUDGET - 1, dtype=np.int64),
            "duplicates": duplicates,
            "negative": negative,
            "out_of_range": out_of_range,
            "empty": np.zeros((0,), dtype=np.int64),
        }
        for name, bad in cases.items():
            with self.subTest(case=name):
                with self.assertRaises(SelectorContractError):
                    select_forced_groups(
                        hidden,
                        6,
                        6,
                        budget=COMPRESS_BUDGET,
                        forced_group_indices=bad,
                    )

    def test_normalize_returns_sorted_int64(self) -> None:
        source = [5, 1, 3]
        result = normalize_forced_group_indices(
            source, budget=3, group_count=10
        )
        self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(result, np.array([1, 3, 5]))

    def test_forced_mode_requires_indices_and_control_rejects_them(self) -> None:
        with self.assertRaises(SelectorContractError):
            _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET, mode=MODE_FORCED)
        with self.assertRaises(SelectorContractError):
            _forward(
                _FakeVisual(),
                COMPRESS_GRID,
                COMPRESS_BUDGET,
                mode=MODE_CONTROL,
                forced_group_indices=np.arange(COMPRESS_BUDGET),
            )

    def test_illegal_mode_is_rejected(self) -> None:
        for bad in ("bogus", "", None, 5):
            with self.subTest(mode=bad):
                with self.assertRaises(SelectorContractError):
                    _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET, mode=bad)


class BudgetAndIdentityTest(unittest.TestCase):
    def test_illegal_budget_is_rejected(self) -> None:
        for bad in (0, -1, 2.5, "25", True, None):
            with self.subTest(budget=bad):
                with self.assertRaises(SelectorContractError):
                    validate_diagnostic_budget(bad)

    def test_identity_bypass_when_group_count_not_above_k(self) -> None:
        grid, budget = 8, COMPRESS_BUDGET  # G = 16 <= 25 -> identity bypass
        result = _forward(_FakeVisual(), grid, budget)
        self.assertTrue(result.identity_bypass)
        self.assertEqual(result.real_group_count, 16)
        self.assertEqual(result.context_group_count, 0)
        np.testing.assert_array_equal(result.group_indices, np.arange(16))
        np.testing.assert_array_equal(
            result.source_type, np.full((16,), SOURCE_REAL, dtype=np.uint8)
        )
        self.assertTrue(torch.equal(result.compact_hidden, result.cut_hidden))
        self.assertEqual(result.prune_after_layer, 5)
        verify_diagnostic_selection(result.selection)

    def test_forced_identity_bypass_when_group_count_not_above_k(self) -> None:
        result = _forward(
            _FakeVisual(),
            8,
            COMPRESS_BUDGET,
            mode=MODE_FORCED,
            forced_group_indices=np.arange(COMPRESS_BUDGET),
        )
        self.assertTrue(result.identity_bypass)
        np.testing.assert_array_equal(result.group_indices, np.arange(16))
        self.assertEqual(result.real_group_count, 16)


class CutPointTest(unittest.TestCase):
    def test_deepstack_index5_is_called_on_the_compact_hidden(self) -> None:
        visual = _FakeVisual()
        cut_merger_index = visual.deepstack_visual_indexes.index(5)
        recorder = _RecordingMerger(visual.deepstack_merger_list[cut_merger_index])
        visual.deepstack_merger_list[cut_merger_index] = recorder
        result = _forward(visual, COMPRESS_GRID, COMPRESS_BUDGET)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(output_count, COMPRESS_BUDGET)
        self.assertEqual(
            recorder.input_shapes[0], (output_count * 4, FEATURE_DIM)
        )
        self.assertNotEqual(
            recorder.input_shapes[0][0], COMPRESS_GRID * COMPRESS_GRID
        )
        expected = visual.deepstack_merger_list[cut_merger_index].inner(
            result.compact_hidden
        )
        self.assertTrue(
            torch.equal(result.pruned.deepstack_embeds[cut_merger_index], expected)
        )

    def test_blocks0_to5_full_then_compact_sequence(self) -> None:
        result = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        full = COMPRESS_GRID * COMPRESS_GRID
        compact = COMPRESS_BUDGET * 4
        self.assertEqual(result.block_input_shapes[:6], [full] * 6)
        self.assertEqual(result.block_input_shapes[6:], [compact] * (DEPTH - 6))
        self.assertEqual(result.deepstack_cut_merger_index, 0)
        self.assertEqual(result.deepstack_pre_cut_count, 0)
        self.assertEqual(result.deepstack_merger_indexes, tuple(DEEPSTACK_INDEXES))
        for feature in result.pruned.deepstack_embeds:
            self.assertEqual(feature.shape, (COMPRESS_BUDGET, OUT_HIDDEN))

    def test_pre_cut_deepstack_is_compacted_by_group_index(self) -> None:
        class _EarlyDeepstackVisual(_FakeVisual):
            deepstack_visual_indexes = [2, 5, 11]

        result = _forward(_EarlyDeepstackVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        self.assertEqual(result.deepstack_pre_cut_count, 1)
        self.assertEqual(result.deepstack_cut_merger_index, 1)
        self.assertEqual(len(result.pruned.deepstack_embeds), 3)
        for feature in result.pruned.deepstack_embeds:
            self.assertEqual(feature.shape[0], COMPRESS_BUDGET)

    def test_illegal_cutpoint_is_rejected(self) -> None:
        for cut in (0, 4, 6, 23, -1, 5.0, "5", None, True):
            with self.subTest(cut=cut):
                with self.assertRaises(SelectorContractError):
                    _forward(
                        _FakeVisual(),
                        COMPRESS_GRID,
                        COMPRESS_BUDGET,
                        prune_after_layer=cut,
                    )

    def test_model_without_deepstack_at_cut_is_rejected(self) -> None:
        class _NoCutDeepstack(_FakeVisual):
            deepstack_visual_indexes = [11, 17]

        with self.assertRaises(SelectorContractError):
            _forward(_NoCutDeepstack(), COMPRESS_GRID, COMPRESS_BUDGET)

    def test_multi_image_is_rejected(self) -> None:
        grid_thw = torch.tensor([[2, 12, 12]])
        with self.assertRaises(ValueError):
            forward_hardware_regular_layer5_forced_index_ablation_vision(
                _FakeVisual(),
                _pixels(COMPRESS_GRID),
                grid_thw,
                budget=COMPRESS_BUDGET,
            )


class OutputSurfaceTest(unittest.TestCase):
    REQUIRED_FIELDS = (
        "image_embeds",
        "deepstack_embeds",
        "group_indices",
        "patch_indices",
        "source_type",
        "region_id",
        "valid",
        "context_group_count",
        "real_group_count",
        "identity_bypass",
        "vision_total_seconds",
        "gpu_vision_seconds",
    )

    def test_required_output_surface_is_present(self) -> None:
        result = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET)
        for name in self.REQUIRED_FIELDS:
            with self.subTest(field=name):
                self.assertTrue(hasattr(result, name))

    def test_timing_contract(self) -> None:
        sync_calls: list[float] = []

        def synchronize() -> None:
            sync_calls.append(time.perf_counter())

        result = _forward(
            _FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET, synchronize=synchronize
        )
        for value in (
            result.vision_preamble_seconds,
            result.cut_blocks_seconds,
            result.selector_cpu_seconds,
            result.remaining_vision_seconds,
        ):
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(
            result.vision_total_seconds,
            result.vision_preamble_seconds
            + result.cut_blocks_seconds
            + result.selector_cpu_seconds
            + result.remaining_vision_seconds,
        )
        self.assertEqual(
            result.gpu_vision_seconds,
            result.vision_preamble_seconds
            + result.cut_blocks_seconds
            + result.remaining_vision_seconds,
        )
        self.assertGreaterEqual(len(sync_calls), 8)
        self.assertEqual(sync_calls, sorted(sync_calls))

    def test_selector_helpers_reject_context_selection(self) -> None:
        selection = _forward(_FakeVisual(), COMPRESS_GRID, COMPRESS_BUDGET).selection
        broken = SimpleNamespace(
            budget=selection.budget,
            bypass=selection.bypass,
            real_group_count=selection.real_group_count,
            context_group_count=1,
            grid_h=selection.grid_h,
            grid_w=selection.grid_w,
            anchors=selection.anchors,
            details=selection.details,
            group_indices=selection.group_indices,
            source_type=selection.source_type,
            region_id=selection.region_id,
            valid=selection.valid,
        )
        with self.assertRaises(SelectorContractError):
            verify_diagnostic_selection(broken)

    def test_control_selector_direct_call_matches_forward(self) -> None:
        hidden = np.zeros((COMPRESS_GROUPS, 4, FEATURE_DIM), dtype=np.float64)
        selection = select_control_groups(hidden, 6, 6, budget=COMPRESS_BUDGET)
        verify_diagnostic_selection(selection)
        self.assertEqual(selection.output_group_count, COMPRESS_BUDGET)
        self.assertEqual(selection.region_id.shape, (COMPRESS_BUDGET,))


if __name__ == "__main__":
    unittest.main()
