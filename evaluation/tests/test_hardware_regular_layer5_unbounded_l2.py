"""CPU fake-visual/fake-model tests for the layer-5 unbounded real-only L2 arm.

These tests never load real weights and never require CUDA.  They exercise the
independent unbounded integration against the frozen pooled layer-5 integration
on the exact same fake vision tower (``hidden=1024``, ``merge_unit=4``,
``out_hidden=2560``, DeepStack at blocks 5/11/17, 24 vision blocks).
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

from evaluation.hardware_regular_layer5_unbounded_l2 import (
    ARM_LABEL,
    BUDGET,
    LOCAL_CANDIDATE_CAP,
    PRUNE_AFTER_LAYER,
    REGION_GRID,
    SCORE_MODE_L2,
    SELECTION_RULE,
    forward_hardware_regular_layer5_unbounded_l2_vision,
    prepare_layer5_unbounded_multimodal_inputs,
    select_unbounded_real_only_groups,
    verify_real_only_selection,
)
from evaluation.hardware_regular_layer5_integration import (
    forward_hardware_regular_layer5_vision,
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
from evaluation.shallow_block_pruning import forward_shallow_pruned_vision

OUT_HIDDEN = 2560
IMAGE_TOKEN_ID = 60
DEPTH = 24
DEEPSTACK_INDEXES = [5, 11, 17]


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


class _FakeTextModel(nn.Module):
    def __init__(self, image_token_id: int) -> None:
        super().__init__()
        self.image_token_id = image_token_id
        self.last_rope_input_length: int | None = None

    def get_rope_index(
        self,
        input_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        length = int(input_ids.shape[1])
        self.last_rope_input_length = length
        positions = (
            torch.arange(length, dtype=torch.long)
            .view(1, 1, length)
            .expand(3, 1, length)
            .clone()
        )
        return positions, torch.zeros(input_ids.shape[0], 1, dtype=torch.long)


class _FakeModel(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = OUT_HIDDEN) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = _FakeTextModel(IMAGE_TOKEN_ID)
        self.emb = nn.Embedding(vocab, hidden)
        weight = (
            torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden)
            % 23
            - 11.0
        ) / hidden
        with torch.no_grad():
            self.emb.weight.copy_(weight)

    def get_input_embeddings(self):
        return self.emb


def _pixels(grid: int, hidden: int = FEATURE_DIM) -> torch.Tensor:
    count = grid * grid
    base = torch.arange(count, dtype=torch.float32).unsqueeze(1)
    feature = torch.arange(hidden, dtype=torch.float32).unsqueeze(0)
    return torch.sin(0.01 * base + 0.001 * feature)


def _grid(grid: int) -> torch.Tensor:
    return torch.tensor([[1, grid, grid]])


def _batch(grid: int) -> SimpleNamespace:
    groups = grid * grid // 4
    prefix = torch.tensor([1, 2, 3], dtype=torch.long)
    image = torch.full((groups,), IMAGE_TOKEN_ID, dtype=torch.long)
    suffix = torch.tensor([4, 5], dtype=torch.long)
    input_ids = torch.cat([prefix, image, suffix]).unsqueeze(0)
    return SimpleNamespace(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        image_grid_thw=_grid(grid),
    )


class _RecordingMerger(nn.Module):
    """Wrap a merger and record the last input shape."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, hidden):
        self.input_shapes.append(tuple(hidden.shape))
        return self.inner(hidden)


def _forward(visual, grid, budget, **kwargs):
    return forward_hardware_regular_layer5_unbounded_l2_vision(
        visual, _pixels(grid), _grid(grid), budget=budget, **kwargs
    )


def _forward_frozen(visual, grid, budget):
    return forward_hardware_regular_layer5_vision(
        visual, _pixels(grid), _grid(grid), budget=budget
    )


def _combined_scores(result) -> np.ndarray:
    raw = (
        result.cut_hidden.detach()
        .to(torch.float64)
        .reshape(result.full_llm_visual_tokens, 4, FEATURE_DIM)
        .numpy()
    )
    return minmax_normalize(group_l2_saliency(raw))


class UnboundedIntegrationTest(unittest.TestCase):
    def test_constants_are_the_fixed_contract(self) -> None:
        self.assertEqual(PRUNE_AFTER_LAYER, 5)
        self.assertEqual(SCORE_MODE_L2, "l2")
        self.assertEqual(BUDGET, 96)
        self.assertEqual(REGION_GRID, 3)
        self.assertIsNone(LOCAL_CANDIDATE_CAP)
        self.assertEqual(SELECTION_RULE, "unbounded_real_only_global_l2_3x3_anchor")
        self.assertEqual(ARM_LABEL, "layer5_unbounded_real_only_block5")

    def test_identity_bypass_matches_dense_elementwise(self) -> None:
        grid, budget = 8, 48  # G = 16 <= 48 -> identity bypass
        visual = _FakeVisual()
        result = _forward(visual, grid, budget)
        self.assertTrue(result.selection.bypass)
        self.assertEqual(result.prune_after_layer, 5)
        self.assertEqual(result.context_group_count, 0)
        self.assertEqual(result.real_group_count, 16)
        self.assertEqual(result.identity_bypass, True)
        np.testing.assert_array_equal(
            result.source_type, np.zeros((16,), dtype=np.uint8)
        )
        np.testing.assert_allclose(
            result.compact_hidden.detach().numpy(),
            result.cut_hidden.detach().numpy(),
        )
        dense = forward_shallow_pruned_vision(
            visual,
            _pixels(grid),
            _grid(grid),
            retention=1.0,
            prune_after_layer=5,
            selector="l2_anchor",
        )
        self.assertTrue(torch.equal(result.pruned.image_embeds, dense.image_embeds))

    def test_compression_emits_exactly_k_all_real(self) -> None:
        grid, budget = 18, 48  # G = 81 > 48
        result = _forward(_FakeVisual(), grid, budget)
        self.assertFalse(result.selection.bypass)
        self.assertEqual(result.selection.output_group_count, budget)
        self.assertEqual(result.real_group_count, budget)
        self.assertEqual(result.context_group_count, 0)
        self.assertTrue(np.all(result.source_type == SOURCE_REAL))
        self.assertFalse(np.any(result.source_type == SOURCE_CONTEXT))
        verify_real_only_selection(result.selection)

    def test_selection_is_global_l2_ranking_plus_region_anchors(self) -> None:
        grid, budget = 18, 48
        result = _forward(_FakeVisual(), grid, budget)
        combined = _combined_scores(result)
        regions = [region for region in build_regions(9, 9, 3) if region.members]
        anchors = {
            max(region.members, key=lambda group: (combined[group], -group))
            for region in regions
        }
        self.assertEqual(set(result.selection.anchors), anchors)
        non_anchor = [group for group in range(81) if group not in anchors]
        expected_details = sorted(
            non_anchor, key=lambda group: (-combined[group], group)
        )[: budget - len(anchors)]
        self.assertEqual(set(result.selection.details), set(expected_details))
        self.assertEqual(
            list(result.group_indices), sorted(anchors | set(expected_details))
        )

    def test_anchors_match_the_frozen_pooled_anchors(self) -> None:
        grid, budget = 18, 48
        unbounded = _forward(_FakeVisual(), grid, budget)
        frozen = _forward_frozen(_FakeVisual(), grid, budget)
        self.assertEqual(
            tuple(unbounded.selection.anchors), tuple(frozen.selection.anchors)
        )
        # The changed constraint must actually change the selected index set.
        self.assertNotEqual(
            set(unbounded.group_indices.tolist()),
            set(frozen.selection.group_indices.tolist()),
        )

    def test_compact_hidden_rows_are_the_raw_selected_groups(self) -> None:
        grid, budget = 18, 48
        result = _forward(_FakeVisual(), grid, budget)
        raw_groups = result.cut_hidden.reshape(
            result.full_llm_visual_tokens, 4, FEATURE_DIM
        )
        for slot in range(result.group_indices.size):
            start = slot * 4
            group = int(result.group_indices[slot])
            self.assertTrue(
                torch.equal(
                    result.compact_hidden[start : start + 4], raw_groups[group]
                )
            )

    def test_compact_cos_sin_use_representative_patch_positions(self) -> None:
        grid, budget = 18, 48
        result = _forward(_FakeVisual(), grid, budget)
        patch_flat = torch.as_tensor(
            result.patch_indices.reshape(-1), dtype=torch.long
        )
        self.assertTrue(
            torch.equal(
                result.compact_cos, result.full_cos.index_select(0, patch_flat)
            )
        )
        self.assertTrue(
            torch.equal(
                result.compact_sin, result.full_sin.index_select(0, patch_flat)
            )
        )
        for index in range(result.group_indices.size):
            group = int(result.group_indices[index])
            self.assertEqual(
                result.patch_indices[index].tolist(),
                [group * 4, group * 4 + 1, group * 4 + 2, group * 4 + 3],
            )

    def test_deepstack_index5_is_called_on_the_compact_hidden(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        cut_merger_index = visual.deepstack_visual_indexes.index(5)
        recorder = _RecordingMerger(visual.deepstack_merger_list[cut_merger_index])
        visual.deepstack_merger_list[cut_merger_index] = recorder
        result = _forward(visual, grid, budget)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(
            recorder.input_shapes[0], (output_count * 4, FEATURE_DIM)
        )
        self.assertNotEqual(recorder.input_shapes[0][0], grid * grid)
        expected = visual.deepstack_merger_list[cut_merger_index].inner(
            result.compact_hidden
        )
        self.assertTrue(
            torch.equal(result.pruned.deepstack_embeds[cut_merger_index], expected)
        )

    def test_deepstack_and_merger_shapes_and_alignment(self) -> None:
        grid, budget = 18, 48
        result = _forward(_FakeVisual(), grid, budget)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(output_count, budget)
        self.assertEqual(result.pruned.image_embeds.shape, (output_count, OUT_HIDDEN))
        self.assertEqual(len(result.pruned.deepstack_embeds), len(DEEPSTACK_INDEXES))
        for feature in result.pruned.deepstack_embeds:
            self.assertEqual(feature.shape, (output_count, OUT_HIDDEN))
        self.assertEqual(result.deepstack_cut_merger_index, 0)
        self.assertEqual(result.deepstack_pre_cut_count, 0)
        self.assertEqual(result.deepstack_merger_indexes, tuple(DEEPSTACK_INDEXES))

    def test_pre_cut_deepstack_is_compacted_by_group_index(self) -> None:
        class _EarlyDeepstackVisual(_FakeVisual):
            deepstack_visual_indexes = [2, 5, 11]

        grid, budget = 18, 48
        result = _forward(_EarlyDeepstackVisual(), grid, budget)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(result.deepstack_pre_cut_count, 1)
        self.assertEqual(result.deepstack_cut_merger_index, 1)
        self.assertEqual(len(result.pruned.deepstack_embeds), 3)
        for feature in result.pruned.deepstack_embeds:
            self.assertEqual(feature.shape[0], output_count)

    def test_vision_timing_contract_and_synchronize_brackets(self) -> None:
        grid, budget = 18, 48
        sync_calls: list[float] = []

        def synchronize() -> None:
            sync_calls.append(time.perf_counter())

        result = _forward(_FakeVisual(), grid, budget, synchronize=synchronize)
        preamble = result.vision_preamble_seconds
        cut = result.cut_blocks_seconds
        selector_cpu = result.selector_cpu_seconds
        remaining = result.remaining_vision_seconds
        for value in (preamble, cut, selector_cpu, remaining):
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(
            result.vision_total_seconds, preamble + cut + selector_cpu + remaining
        )
        self.assertEqual(result.gpu_vision_seconds, preamble + cut + remaining)
        self.assertEqual(result.block0_seconds, cut)
        self.assertGreaterEqual(len(sync_calls), 8)
        self.assertEqual(sync_calls, sorted(sync_calls))

    def test_group_indices_unique_and_strictly_ascending(self) -> None:
        for grid, budget in ((18, 48), (18, 64), (8, 48), (12, 48)):
            with self.subTest(grid=grid, budget=budget):
                result = _forward(_FakeVisual(), grid, budget)
                indices = result.group_indices
                self.assertEqual(len(set(indices.tolist())), indices.size)
                if indices.size > 1:
                    self.assertTrue(bool(np.all(np.diff(indices) > 0)))
                self.assertEqual(result.source_type.shape, indices.shape)

    def test_no_local_candidate_cap_parameter(self) -> None:
        import inspect

        signature = inspect.signature(
            forward_hardware_regular_layer5_unbounded_l2_vision
        )
        self.assertNotIn("local_candidate_cap", signature.parameters)

    def test_prepare_keeps_k_placeholders_and_preserves_text_order(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        model = _FakeModel()
        batch = _batch(grid)
        result = _forward(visual, grid, budget)
        prepared = prepare_layer5_unbounded_multimodal_inputs(model, batch, result)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(int(prepared.visual_pos_masks.sum().item()), output_count)
        self.assertEqual(
            int((prepared.input_ids[0] == IMAGE_TOKEN_ID).sum().item()),
            output_count,
        )
        original_text = batch.input_ids[0][batch.input_ids[0] != IMAGE_TOKEN_ID]
        pruned_text = prepared.input_ids[0][prepared.input_ids[0] != IMAGE_TOKEN_ID]
        self.assertTrue(torch.equal(original_text, pruned_text))


class UnboundedSelectorUnitTest(unittest.TestCase):
    def test_matching_anchor_tie_break_uses_lower_index(self) -> None:
        hidden = np.zeros((64, 4, FEATURE_DIM), dtype=np.float64)
        selection = select_unbounded_real_only_groups(hidden, 8, 8, budget=48)
        verify_real_only_selection(selection)
        # Ties resolve to the lowest member index in each region.
        for region, anchor in zip(
            [r for r in build_regions(8, 8, 3) if r.members], selection.anchors
        ):
            self.assertEqual(anchor, min(region.members))

    def test_grid_mismatch_is_rejected(self) -> None:
        hidden = np.zeros((64, 4, FEATURE_DIM), dtype=np.float64)
        with self.assertRaises(SelectorContractError):
            select_unbounded_real_only_groups(hidden, 8, 9, budget=48)

    def test_verify_real_only_rejects_a_context_selection(self) -> None:
        selection = _forward(_FakeVisual(), 18, 48).selection
        broken = SimpleNamespace(
            **{
                name: getattr(selection, name)
                for name in (
                    "budget",
                    "bypass",
                    "real_group_count",
                    "grid_h",
                    "grid_w",
                    "anchors",
                    "details",
                )
            },
            context_group_count=1,
            group_indices=selection.group_indices,
            source_type=selection.source_type,
            region_id=selection.region_id,
            valid=selection.valid,
        )
        with self.assertRaises(SelectorContractError):
            verify_real_only_selection(broken)

    def test_verify_real_only_rejects_a_bad_k(self) -> None:
        selection = _forward(_FakeVisual(), 18, 48).selection
        broken = SimpleNamespace(
            **{
                name: getattr(selection, name)
                for name in (
                    "budget",
                    "bypass",
                    "real_group_count",
                    "context_group_count",
                    "grid_h",
                    "grid_w",
                    "anchors",
                    "details",
                    "group_indices",
                    "source_type",
                    "region_id",
                    "valid",
                )
            },
        )
        broken.budget = 64
        with self.assertRaises(SelectorContractError):
            verify_real_only_selection(broken)


class IllegalUnboundedContractTest(unittest.TestCase):
    def test_illegal_cutpoint_is_rejected(self) -> None:
        for cut in (0, 1, 4, 6, 23, -1):
            with self.subTest(cut=cut):
                with self.assertRaises(SelectorContractError):
                    _forward(_FakeVisual(), 18, 48, prune_after_layer=cut)

    def test_non_integer_cutpoint_is_rejected(self) -> None:
        for cut in (5.0, "5", None, True):
            with self.subTest(cut=cut):
                with self.assertRaises(SelectorContractError):
                    _forward(_FakeVisual(), 18, 48, prune_after_layer=cut)

    def test_illegal_score_mode_is_rejected(self) -> None:
        for mode in ("prompt", "integer", "float", "", None):
            with self.subTest(mode=mode):
                with self.assertRaises(SelectorContractError):
                    _forward(_FakeVisual(), 18, 48, score_mode=mode)

    def test_b1_trained_wv_projection_is_rejected(self) -> None:
        with self.assertRaises(SelectorContractError):
            _forward(_FakeVisual(), 18, 48, projection=object())
        with self.assertRaises(SelectorContractError):
            _forward(_FakeVisual(), 18, 48, text_embeddings=np.zeros((3, 2560)))

    def test_multi_image_is_rejected(self) -> None:
        grid_thw = torch.tensor([[2, 16, 16]])
        with self.assertRaises(ValueError):
            forward_hardware_regular_layer5_unbounded_l2_vision(
                _FakeVisual(),
                _pixels(16),
                grid_thw,
                budget=48,
            )

    def test_too_shallow_model_is_rejected(self) -> None:
        shallow = _FakeVisual(depth=5)
        with self.assertRaises(ValueError):
            _forward(shallow, 18, 48)

    def test_model_without_deepstack_at_cut_is_rejected(self) -> None:
        class _NoCutDeepstack(_FakeVisual):
            deepstack_visual_indexes = [11, 17]

        with self.assertRaises(SelectorContractError):
            _forward(_NoCutDeepstack(), 18, 48)


if __name__ == "__main__":
    unittest.main()
