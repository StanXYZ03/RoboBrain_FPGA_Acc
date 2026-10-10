"""CPU fake-visual/fake-model tests for the Qwen3-VL B1 hardware selector.

These tests never load real weights and never require CUDA.  They exercise the
integration through lightweight modules whose shapes match the Qwen3-VL vision
tower (``hidden=1024``, ``merge_unit=4``, ``out_hidden=2560``, DeepStack at
blocks 5/11/17).
"""

from __future__ import annotations

import math
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluation.hardware_regular_b1_integration import (
    SCORE_MODE_L2,
    SCORE_MODE_PROMPT,
    forward_hardware_regular_b1_vision,
    prepare_b1_multimodal_inputs,
)
from evaluation.hardware_regular_token_selector import (
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    LowRankProjection,
    SelectorContractError,
)
from evaluation.shallow_block_pruning import forward_shallow_pruned_vision

OUT_HIDDEN = 2560
IMAGE_TOKEN_ID = 60


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

    def __init__(self, hidden: int, out_hidden: int) -> None:
        super().__init__()
        fan_in = hidden * 4
        self.proj = nn.Linear(fan_in, out_hidden, bias=False)
        weight = (
            torch.arange(out_hidden * fan_in, dtype=torch.float32).reshape(
                out_hidden, fan_in
            )
            % 17
            - 8.0
        ) / fan_in
        with torch.no_grad():
            self.proj.weight.copy_(weight)

    def forward(self, hidden):
        return self.proj(hidden.reshape(-1, hidden.shape[-1] * 4))


class _FakeVisual(nn.Module):
    dtype = torch.float32
    spatial_merge_size = 2
    deepstack_visual_indexes = [5, 11, 17]

    def __init__(self, hidden: int = FEATURE_DIM, depth: int = 18) -> None:
        super().__init__()
        self.patch_embed = nn.Identity()
        self.blocks = nn.ModuleList(_FakeBlock(hidden) for _ in range(depth))
        self.deepstack_merger_list = nn.ModuleList(
            _GroupMerger(hidden, OUT_HIDDEN) for _ in range(3)
        )
        self.merger = _GroupMerger(hidden, OUT_HIDDEN)
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


def _text_embeddings(tokens: int) -> np.ndarray:
    row = np.arange(tokens)[:, None]
    col = np.arange(PROMPT_DIM)[None, :]
    return 1.0 + np.cos(0.2 * row + 0.01 * col)


def _projection() -> LowRankProjection:
    row = np.arange(SCORE_DIM)[:, None]
    text_col = np.arange(PROMPT_DIM)[None, :]
    feature_col = np.arange(FEATURE_DIM)[None, :]
    Wt = ((row * 13 + text_col * 7) % 11 - 5) / 5.0
    Wv = ((row * 5 + feature_col * 3) % 7 - 3) / 3.0
    return LowRankProjection(Wt=Wt, Wv=Wv)


def _batch(grid: int) -> SimpleNamespace:
    groups = grid * grid // 4
    vocab = 64
    prefix = torch.tensor([1, 2, 3], dtype=torch.long)
    image = torch.full((groups,), IMAGE_TOKEN_ID, dtype=torch.long)
    suffix = torch.tensor([4, 5], dtype=torch.long)
    input_ids = torch.cat([prefix, image, suffix]).unsqueeze(0)
    return SimpleNamespace(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        image_grid_thw=_grid(grid),
    )


class HardwareRegularB1IntegrationTest(unittest.TestCase):
    def test_identity_bypass_matches_dense_elementwise(self) -> None:
        grid, budget = 8, 48  # G = 16 <= 48 -> identity bypass
        visual = _FakeVisual()
        pixels = _pixels(grid)
        result = forward_hardware_regular_b1_vision(
            visual, pixels, _grid(grid), budget=budget
        )
        self.assertTrue(result.selection.bypass)
        group_count = grid * grid // 4
        self.assertEqual(result.group_indices.tolist(), list(range(group_count)))
        self.assertEqual(
            result.patch_indices.reshape(-1).tolist(),
            list(range(group_count * 4)),
        )
        self.assertEqual(result.context_group_count, 0)
        self.assertTrue(result.valid.all())
        np.testing.assert_allclose(
            result.compact_hidden.detach().numpy(),
            result.block0_hidden.detach().numpy(),
        )

        dense = forward_shallow_pruned_vision(
            visual,
            pixels,
            _grid(grid),
            retention=1.0,
            prune_after_layer=0,
            selector="l2_anchor",
        )
        self.assertTrue(
            torch.equal(result.pruned.image_embeds, dense.image_embeds)
        )
        self.assertEqual(
            len(result.pruned.deepstack_embeds), len(dense.deepstack_embeds)
        )
        for produced, reference in zip(
            result.pruned.deepstack_embeds, dense.deepstack_embeds
        ):
            self.assertTrue(torch.equal(produced, reference))

    def test_b1_only_compresses_after_block0(self) -> None:
        grid, budget = 18, 48  # G = 81 > 48 -> compression after block 0
        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        patch_tokens = grid * grid
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(output_count, budget)
        self.assertEqual(result.block_input_shapes[0], patch_tokens)
        self.assertTrue(
            all(shape == output_count * 4 for shape in result.block_input_shapes[1:])
        )
        self.assertEqual(len(result.block_input_shapes), len(visual.blocks))
        self.assertEqual(result.compact_hidden.shape[0], output_count * 4)

    def test_context_rows_are_pooled_values_not_raw_representatives(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        self.assertGreater(result.context_group_count, 0)
        raw_groups = result.block0_hidden.reshape(
            result.full_llm_visual_tokens, 4, FEATURE_DIM
        )
        pool_matches = 0
        pool_differs_from_raw = 0
        for index in range(result.pruned.image_embeds.shape[0]):
            chunk = result.compact_hidden[index * 4 : (index + 1) * 4]
            pooled = torch.as_tensor(
                result.selection.groups[index], dtype=chunk.dtype
            )
            np.testing.assert_allclose(
                chunk.detach().numpy(), pooled.numpy(), rtol=1e-5, atol=1e-5
            )
            pool_matches += 1
            if int(result.source_type[index]) == 1:
                representative = int(result.group_indices[index])
                raw = raw_groups[representative]
                if not torch.allclose(chunk, raw, rtol=1e-5, atol=1e-5):
                    pool_differs_from_raw += 1
        self.assertEqual(pool_matches, result.pruned.image_embeds.shape[0])
        self.assertGreater(pool_differs_from_raw, 0)
        # Real rows are still the original block-0 activation values.
        for index in np.flatnonzero(result.source_type == 0):
            group = int(result.group_indices[index])
            np.testing.assert_allclose(
                result.compact_hidden[index * 4 : (index + 1) * 4].detach().numpy(),
                raw_groups[group].detach().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    def test_compact_cos_sin_use_representative_patch_positions(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
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
        # Representative positions are the 4 contiguous patches of each output
        # group and are globally unique across the K groups.
        for index in range(result.group_indices.size):
            group = int(result.group_indices[index])
            self.assertEqual(
                result.patch_indices[index].tolist(),
                [group * 4, group * 4 + 1, group * 4 + 2, group * 4 + 3],
            )
        self.assertEqual(
            len(set(patch_flat.tolist())), result.group_indices.size * 4
        )

    def test_cu_seqlens_rebuilt_for_compact_sequence(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(
            result.compact_cu_seqlens.tolist(), [0, output_count * 4]
        )
        self.assertEqual(result.compact_cu_seqlens.dtype, torch.int32)

    def test_vision_timing_contract_and_synchronize_brackets(self) -> None:
        grid, budget = 18, 48
        sync_calls: list[float] = []

        def synchronize() -> None:
            sync_calls.append(time.perf_counter())

        result = forward_hardware_regular_b1_vision(
            _FakeVisual(),
            _pixels(grid),
            _grid(grid),
            budget=budget,
            synchronize=synchronize,
        )
        preamble = result.vision_preamble_seconds
        block0 = result.block0_seconds
        selector_cpu = result.selector_cpu_seconds
        remaining = result.remaining_vision_seconds
        for value in (preamble, block0, selector_cpu, remaining):
            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(value, 0.0)
        # The vision total is exactly the four reported sub-segments.
        self.assertEqual(
            result.vision_total_seconds,
            preamble + block0 + selector_cpu + remaining,
        )
        self.assertEqual(
            result.vision_total_seconds,
            result.vision_preamble_seconds
            + result.block0_seconds
            + result.selector_cpu_seconds
            + result.remaining_vision_seconds,
        )
        # GPU vision removes the CPU selector round trip.
        self.assertEqual(
            result.gpu_vision_seconds,
            preamble + block0 + remaining,
        )
        self.assertEqual(
            result.vision_total_seconds - result.gpu_vision_seconds,
            selector_cpu,
        )
        # Preamble, block 0, CPU selector and remaining are each bracketed by
        # the caller-supplied synchronize (start + end per segment).
        self.assertGreaterEqual(len(sync_calls), 8)
        self.assertEqual(sync_calls, sorted(sync_calls))

    def test_deepstack_and_merger_shapes(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(result.pruned.image_embeds.shape, (output_count, OUT_HIDDEN))
        self.assertEqual(len(result.pruned.deepstack_embeds), 3)
        for feature in result.pruned.deepstack_embeds:
            self.assertEqual(feature.shape, (output_count, OUT_HIDDEN))
        self.assertEqual(len(result.block_input_shapes), len(visual.blocks))

    def test_group_indices_unique_and_strictly_ascending(self) -> None:
        for grid, budget in ((18, 48), (18, 64), (8, 48), (12, 48)):
            with self.subTest(grid=grid, budget=budget):
                result = forward_hardware_regular_b1_vision(
                    _FakeVisual(), _pixels(grid), _grid(grid), budget=budget
                )
                indices = result.group_indices
                self.assertEqual(len(set(indices.tolist())), indices.size)
                if indices.size > 1:
                    self.assertTrue(bool(np.all(np.diff(indices) > 0)))
                self.assertEqual(result.source_type.shape, indices.shape)
                self.assertEqual(result.region_id.shape, indices.shape)
                self.assertEqual(result.valid.shape, indices.shape)

    def test_prepare_keeps_k_placeholders_and_preserves_text_order(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        model = _FakeModel()
        batch = _batch(grid)
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        prepared = prepare_b1_multimodal_inputs(model, batch, result)
        output_count = int(result.pruned.image_embeds.shape[0])
        self.assertEqual(int(prepared.visual_pos_masks.sum().item()), output_count)
        self.assertEqual(
            int((prepared.input_ids[0] == IMAGE_TOKEN_ID).sum().item()),
            output_count,
        )
        self.assertEqual(
            prepared.inputs_embeds.shape[1], prepared.input_ids.shape[1]
        )
        # The original text token subsequence is unchanged and in order.
        original_text = batch.input_ids[0][batch.input_ids[0] != IMAGE_TOKEN_ID]
        pruned_text = prepared.input_ids[0][
            prepared.input_ids[0] != IMAGE_TOKEN_ID
        ]
        self.assertTrue(torch.equal(original_text, pruned_text))
        # Exactly G - K image placeholders were removed.
        full_groups = grid * grid // 4
        self.assertEqual(
            int(prepared.original_sequence_length - prepared.input_ids.shape[1]),
            full_groups - output_count,
        )

    def test_mrope_is_computed_on_full_ids_then_sliced(self) -> None:
        grid, budget = 18, 48
        visual = _FakeVisual()
        model = _FakeModel()
        batch = _batch(grid)
        result = forward_hardware_regular_b1_vision(
            visual, _pixels(grid), _grid(grid), budget=budget
        )
        prepared = prepare_b1_multimodal_inputs(model, batch, result)
        # The rope index was computed on the full, unpruned input_ids.
        self.assertEqual(model.model.last_rope_input_length, batch.input_ids.shape[1])
        full_positions, _ = model.model.get_rope_index(
            batch.input_ids, batch.image_grid_thw
        )
        image_positions = torch.nonzero(
            batch.input_ids[0] == IMAGE_TOKEN_ID, as_tuple=False
        ).flatten()
        keep_mask = torch.ones(batch.input_ids.shape[1], dtype=torch.bool)
        keep_mask[image_positions] = False
        keep_mask[
            image_positions.index_select(
                0, torch.as_tensor(result.group_indices, dtype=torch.long)
            )
        ] = True
        expected = full_positions[:, :, keep_mask]
        self.assertTrue(torch.equal(prepared.position_ids, expected))
        # Text positions are untouched by the slice.
        text_columns = torch.nonzero(
            batch.input_ids[0] != IMAGE_TOKEN_ID, as_tuple=False
        ).flatten()
        expected_text_positions = full_positions[:, :, text_columns]
        produced_text_positions = prepared.position_ids[
            :, :, prepared.input_ids[0] != IMAGE_TOKEN_ID
        ]
        self.assertTrue(
            torch.equal(expected_text_positions, produced_text_positions)
        )

    def test_l2_mode_does_not_require_projection(self) -> None:
        grid, budget = 18, 48
        result = forward_hardware_regular_b1_vision(
            _FakeVisual(),
            _pixels(grid),
            _grid(grid),
            budget=budget,
            score_mode=SCORE_MODE_L2,
            projection=None,
            text_embeddings=None,
        )
        self.assertIsNone(result.selection.scores.semantic)
        self.assertIsNotNone(result.selection.scores.l2)
        self.assertIsNotNone(result.selection.scores.combined)

    def test_prompt_mode_requires_explicit_projection(self) -> None:
        grid, budget = 18, 48
        with self.assertRaises(SelectorContractError):
            forward_hardware_regular_b1_vision(
                _FakeVisual(),
                _pixels(grid),
                _grid(grid),
                budget=budget,
                score_mode=SCORE_MODE_PROMPT,
                projection=None,
                text_embeddings=_text_embeddings(5),
                lambda_semantic=0.5,
            )
        with self.assertRaises(SelectorContractError):
            forward_hardware_regular_b1_vision(
                _FakeVisual(),
                _pixels(grid),
                _grid(grid),
                budget=budget,
                score_mode=SCORE_MODE_PROMPT,
                projection=_projection(),
                text_embeddings=None,
                lambda_semantic=0.5,
            )

    def test_prompt_mode_runs_with_explicit_projection(self) -> None:
        grid, budget = 18, 48
        result = forward_hardware_regular_b1_vision(
            _FakeVisual(),
            _pixels(grid),
            _grid(grid),
            budget=budget,
            score_mode=SCORE_MODE_PROMPT,
            projection=_projection(),
            text_embeddings=_text_embeddings(5),
            lambda_semantic=0.5,
        )
        scores = result.selection.scores
        self.assertIsNotNone(scores.semantic)
        self.assertIsNotNone(scores.l2)
        self.assertIsNotNone(scores.combined)
        self.assertEqual(scores.combined.shape, (grid * grid // 4,))

    def test_identity_bypass_ignores_missing_projection(self) -> None:
        # G <= K must not require any score interface, even in prompt mode.
        grid, budget = 4, 48
        result = forward_hardware_regular_b1_vision(
            _FakeVisual(),
            _pixels(grid),
            _grid(grid),
            budget=budget,
            score_mode=SCORE_MODE_PROMPT,
            projection=None,
            text_embeddings=None,
            lambda_semantic=None,
        )
        self.assertTrue(result.selection.bypass)
        self.assertEqual(result.group_indices.size, grid * grid // 4)


class ProjectionLoadingTest(unittest.TestCase):
    """The prompt-mode runner must load Wt/Wv explicitly, never generate them."""

    def _write(self, directory: str, **overrides) -> Path:
        projection = _projection()
        payload = {
            "Wt": projection.Wt,
            "Wv": projection.Wv,
        }
        payload.update(overrides)
        path = Path(directory) / "projection.npz"
        np.savez(path, **payload)
        return path

    def test_valid_projection_loads_and_reports_hash(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import file_sha256, load_projection

        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory)
            loaded, digest = load_projection(path)
            self.assertEqual(digest, file_sha256(path))
            np.testing.assert_allclose(loaded.Wt, _projection().Wt)
            np.testing.assert_allclose(loaded.Wv, _projection().Wv)
            loaded2, _ = load_projection(path, expected_sha256=digest)
            np.testing.assert_allclose(loaded2.Wt, _projection().Wt)

    def test_missing_file_hash_mismatch_and_bad_shape_raise(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import load_projection

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.npz"
            with self.assertRaises(FileNotFoundError):
                load_projection(missing)

            path = self._write(
                directory, Wt=np.zeros((SCORE_DIM - 1, PROMPT_DIM))
            )
            with self.assertRaises(SelectorContractError):
                load_projection(path)

            path = self._write(directory)
            with self.assertRaises(ValueError):
                load_projection(path, expected_sha256="0" * 64)

    def test_missing_keys_raise(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import load_projection

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "only_wt.npz"
            np.savez(path, Wt=_projection().Wt)
            with self.assertRaises(ValueError):
                load_projection(path)

    def test_canonical_projection_sha256_stable_and_tamper_sensitive(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import (
            canonical_projection_sha256,
        )

        projection = _projection()
        digest = canonical_projection_sha256(projection.Wt, projection.Wv)
        self.assertEqual(
            digest,
            canonical_projection_sha256(projection.Wt, projection.Wv),
        )
        # Fixed dtype/shape/bytes: copies, a different source dtype and a
        # Fortran-ordered layout all hash to the same digest.
        self.assertEqual(
            digest,
            canonical_projection_sha256(
                np.array(projection.Wt, dtype=np.float64, order="F", copy=True),
                np.ascontiguousarray(projection.Wv, dtype=np.float32),
            ),
        )
        # Tampering one array element changes the digest.
        tampered = projection.Wt.copy()
        tampered[0, 0] += 1.0
        self.assertNotEqual(
            digest, canonical_projection_sha256(tampered, projection.Wv)
        )
        self.assertNotEqual(
            digest, canonical_projection_sha256(projection.Wt, tampered[:, :])
        )
        # A shape change also changes the digest.
        self.assertNotEqual(
            digest,
            canonical_projection_sha256(
                projection.Wt[:, : PROMPT_DIM - 1], projection.Wv
            ),
        )

    def test_weights_sha256_key_is_checked_against_canonical_content_hash(
        self,
    ) -> None:
        from evaluation.run_hf_hardware_regular_b1 import (
            canonical_projection_sha256,
            load_projection,
        )

        projection = _projection()
        good = canonical_projection_sha256(projection.Wt, projection.Wv)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, weights_sha256=np.array(good))
            loaded, _ = load_projection(path)
            np.testing.assert_allclose(loaded.Wt, projection.Wt)
            np.testing.assert_allclose(loaded.Wv, projection.Wv)

            path = self._write(directory, weights_sha256=np.array("0" * 64))
            with self.assertRaises(ValueError):
                load_projection(path)

            # Tampered arrays no longer match the stored content hash.
            tampered = projection.Wt.copy()
            tampered[0, 0] += 1.0
            path = self._write(
                directory,
                Wt=tampered,
                Wv=projection.Wv,
                weights_sha256=np.array(good),
            )
            with self.assertRaises(ValueError):
                load_projection(path)

            # A legacy self-file ``sha256`` key can never equal the file hash;
            # it is ignored instead of being compared.
            path = self._write(directory, sha256=np.array("deadbeef"))
            loaded, _ = load_projection(path)
            np.testing.assert_allclose(loaded.Wv, projection.Wv)


if __name__ == "__main__":
    unittest.main()
