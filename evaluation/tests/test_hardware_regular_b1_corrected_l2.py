#!/usr/bin/env python3
"""CPU-only tests for the corrected patch-wise L2 / Prompt scorer chain.

The tests pin the corrected chain against the accepted quantisation / rounding
/ min-max helpers and against the defective ``norm(mean)`` baseline.  No CUDA,
no model and no network are touched.  The cached representative activations are
read (read-only) from the frozen fixture.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import hardware_regular_b1_corrected_l2 as cl  # noqa: E402
from evaluation import selector_integer_scorer_audit as audit  # noqa: E402
from evaluation.hardware_regular_b1_integration import (  # noqa: E402
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    group_l2_saliency,
    minmax_normalize,
    select_hardware_regular_groups,
)

CACHED_ACTIVATIONS = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation"
    / "representative_activations.npz"
)


def _synthetic_hidden(group_count: int, seed: int = 20261004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = rng.normal(size=(group_count, 1, 1))
    patches = rng.normal(size=(group_count, 4, 1024)) * scales
    return patches.astype(np.float64)


def _select(combined: np.ndarray, grid_h: int, grid_w: int, budget: int = 96):
    selection = select_hardware_regular_groups(
        np.zeros((grid_h * grid_w, 4, 1024)),
        grid_h,
        grid_w,
        budget=budget,
        combined_scores=combined,
    )
    return tuple(int(value) for value in selection.group_indices)


def _jaccard(left, right) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / len(left_set | right_set)


class CancellationFormulaTest(unittest.TestCase):
    def test_cancellation_distinguishes_mean_norm_from_norm_mean(self) -> None:
        hidden = np.zeros((4, 4, 1024), dtype=np.float64)
        vector = np.zeros(1024, dtype=np.float64)
        vector[0] = 3.0
        # Group 0 cancels: patches [v, -v, v, -v] -> norm(mean) == 0 but
        # mean(norm) == 3.
        hidden[0, 0] = vector
        hidden[0, 1] = -vector
        hidden[0, 2] = vector
        hidden[0, 3] = -vector
        # Group 1 does not cancel: patches [v, v, v, v].
        hidden[1] = vector
        hidden[2] = 0.5
        hidden[3] = 0.25

        corrected = cl.corrected_patch_l2_float(hidden)
        defective = group_l2_saliency(hidden)

        self.assertAlmostEqual(float(corrected["group_scores"][0]), 3.0)
        self.assertAlmostEqual(float(defective[0]), 0.0)
        self.assertGreater(float(corrected["group_scores"][0]), 0.0)
        self.assertAlmostEqual(float(corrected["group_scores"][1]), 3.0)
        # The corrected vector is not the defective one in general.
        self.assertFalse(np.allclose(corrected["group_scores"], defective))
        # The corrected group 0 score is the mean of the four patch norms.
        self.assertAlmostEqual(
            float(corrected["patch_norms"][0].mean()),
            float(corrected["group_scores"][0]),
        )

    def test_float_combined_uses_accepted_minmax(self) -> None:
        hidden = _synthetic_hidden(20)
        payload = cl.corrected_patch_l2_float(hidden)
        expected = minmax_normalize(payload["group_scores"])
        self.assertTrue(np.allclose(payload["combined"], expected))


class IntegerPrimitiveTest(unittest.TestCase):
    def test_floor_sqrt_is_exact(self) -> None:
        for value in range(0, 200):
            root = audit.floor_sqrt_int64(value)
            self.assertLessEqual(root * root, value)
            self.assertGreater((root + 1) * (root + 1), value)
        self.assertEqual(audit.floor_sqrt_int64(10**18), 10**9)

    def test_minmax_q15_bounds_and_all_equal(self) -> None:
        self.assertTrue(
            np.array_equal(
                audit.minmax_normalize_q15(np.array([7, 7, 7])),
                np.zeros(3, dtype=np.int64),
            )
        )
        mapped = audit.minmax_normalize_q15(np.array([0, 5, 10], dtype=np.int64))
        self.assertEqual(int(mapped[0]), 0)
        self.assertEqual(int(mapped[-1]), cl.SCORE_Q_MAX)
        self.assertTrue(int(mapped[1]) > 0)

    def test_combine_q15_rounds_half_away_and_clamps(self) -> None:
        self.assertEqual(int(cl.combine_q15([1], [1])[0]), 1)
        self.assertEqual(int(cl.combine_q15([1], [2])[0]), 2)
        self.assertEqual(int(cl.combine_q15([2], [1])[0]), 2)
        self.assertEqual(int(cl.combine_q15([0], [1])[0]), 1)
        self.assertEqual(int(cl.combine_q15([0], [0])[0]), 0)
        self.assertEqual(int(cl.combine_q15([32767], [32767])[0]), 32767)
        with self.assertRaises(cl.CorrectedL2Error):
            cl.combine_q15([32768], [0])
        with self.assertRaises(cl.CorrectedL2Error):
            cl.combine_q15([0, 1], [0])

    def test_combine_q15_matches_accepted_integer_chain(self) -> None:
        rng = np.random.default_rng(7)
        for grid_h, grid_w in ((8, 8), (20, 13)):
            group_count = grid_h * grid_w
            mean_q = rng.integers(-128, 128, size=(group_count, 1024)).astype(
                np.int64
            )
            query_q = rng.integers(-128, 128, size=128).astype(np.int64)
            wv_q = rng.integers(-127, 128, size=(128, 1024)).astype(np.int64)
            chain = audit.integer_chain(
                mean_q, query_q, wv_q, grid_h, grid_w, budgets=[96]
            )
            mine = cl.combine_q15(
                chain["semantic_norm_q15"], chain["l2_norm_q15"]
            )
            self.assertTrue(np.array_equal(mine, chain["score_q"]))

    def test_integer_bounds_saturation_and_determinism(self) -> None:
        hidden = np.zeros((40, 4, 1024), dtype=np.float64)
        hidden[0] = 1e6
        hidden[1] = -1e6
        hidden[2:] = _synthetic_hidden(38, seed=11)
        first = cl.corrected_patch_l2_integer(hidden)
        second = cl.corrected_patch_l2_integer(hidden)
        self.assertTrue(
            np.array_equal(first["l2_norm_q15"], second["l2_norm_q15"])
        )
        bounds = first["bounds"]
        self.assertLessEqual(
            bounds["patch_squared_sum_max"],
            bounds["patch_squared_sum_analytic_bound"],
        )
        self.assertLessEqual(bounds["patch_norm_max"], bounds["patch_norm_analytic_bound"])
        self.assertLessEqual(
            bounds["group_norm_sum_max"], bounds["group_norm_sum_analytic_bound"]
        )
        self.assertLessEqual(bounds["l2_norm_q15_max"], cl.SCORE_Q_MAX)
        self.assertGreaterEqual(int(first["l2_norm_q15"].min()), 0)

    def test_all_equal_scores_map_to_zero(self) -> None:
        hidden = np.ones((20, 4, 1024), dtype=np.float64)
        payload = cl.corrected_patch_l2_integer(hidden)
        self.assertTrue(np.all(payload["l2_norm_q15"] == 0))
        self.assertTrue(np.all(payload["combined"] == 0.0))

    def test_deterministic_ties_pick_lower_index(self) -> None:
        hidden = np.zeros((100, 4, 1024), dtype=np.float64)
        hidden[:, :, 0] = 1.0
        # Every group has the same score, so selection must be deterministic.
        payload = cl.corrected_patch_l2_integer(hidden)
        self.assertEqual(len(set(payload["combined"].tolist())), 1)
        selected_first = _select(payload["combined"], 10, 10)
        selected_second = _select(payload["combined"], 10, 10)
        self.assertEqual(selected_first, selected_second)
        self.assertEqual(len(selected_first), 96)


class CorrectedSelectionTest(unittest.TestCase):
    def test_integer_vs_float_selection_on_synthetic(self) -> None:
        hidden = _synthetic_hidden(400)
        float_payload = cl.corrected_patch_l2_float(hidden)
        integer_payload = cl.corrected_patch_l2_integer(hidden)
        float_selected = _select(float_payload["combined"], 20, 20)
        integer_selected = _select(integer_payload["combined"], 20, 20)
        self.assertEqual(len(float_selected), 96)
        self.assertEqual(len(integer_selected), 96)
        self.assertGreaterEqual(_jaccard(float_selected, integer_selected), 0.9)
        # Determinism under repetition.
        self.assertEqual(
            integer_selected,
            _select(
                cl.corrected_patch_l2_integer(hidden)["combined"], 20, 20
            ),
        )

    def test_integer_vs_float_selection_on_cached_representative(self) -> None:
        self.assertTrue(CACHED_ACTIVATIONS.is_file())
        with np.load(CACHED_ACTIVATIONS, allow_pickle=False) as archive:
            patches = archive["representative_patches"].astype(np.float64)
            grid_h = int(archive["grid_h"])
            grid_w = int(archive["grid_w"])
        self.assertEqual(patches.shape, (grid_h * grid_w, 4, 1024))
        float_payload = cl.corrected_patch_l2_float(patches)
        integer_payload = cl.corrected_patch_l2_integer(patches)
        float_selected = _select(float_payload["combined"], grid_h, grid_w)
        integer_selected = _select(integer_payload["combined"], grid_h, grid_w)
        self.assertEqual(len(float_selected), 96)
        self.assertEqual(len(integer_selected), 96)
        self.assertGreaterEqual(_jaccard(float_selected, integer_selected), 0.9)
        # The corrected integer L2 genuinely differs from the defective one.
        defective = minmax_normalize(group_l2_saliency(patches))
        self.assertFalse(np.allclose(integer_payload["combined"], defective))


class PromptArmTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20261004)
        self.hidden = rng.normal(size=(40, 4, 1024)).astype(np.float64)
        self.Wt = rng.normal(size=(128, 2560)).astype(np.float64)
        self.wv_q = rng.integers(-127, 128, size=(128, 1024)).astype(np.int64)
        self.text = rng.normal(size=(12, 2560)).astype(np.float64)

    def test_prompt_arm_preserves_accepted_semantic_chain(self) -> None:
        payload = cl.prompt_plus_corrected_l2_integer(
            self.hidden, self.text, self.Wt, self.wv_q, 8, 5, budget=96
        )
        from evaluation import selector_int8_numerical_audit as base

        group_mean = self.hidden.mean(axis=1)
        mean_q, _ = base.quantize_activation(group_mean)
        query = base.primary_quantized_query(self.text, self.Wt)
        chain = audit.integer_chain(mean_q, query["query_q"], self.wv_q, 8, 5, budgets=[96])
        self.assertTrue(
            np.array_equal(payload["semantic_norm_q15"], chain["semantic_norm_q15"])
        )
        # The score is the accepted combine of the accepted semantic with the
        # corrected patch L2 (the old group-mean L2 is discarded).
        self.assertTrue(
            np.array_equal(
                payload["score_q"],
                cl.combine_q15(chain["semantic_norm_q15"], payload["l2_norm_q15"]),
            )
        )
        self.assertFalse(
            np.array_equal(payload["l2_norm_q15"], chain["l2_norm_q15"])
        )
        self.assertTrue(
            np.array_equal(
                payload["combined"],
                payload["score_q"].astype(np.float64) / cl.SCORE_Q_SCALE,
            )
        )

    def test_prompt_arm_requires_checkpoint(self) -> None:
        with self.assertRaises(cl.CorrectedL2Error):
            cl.prompt_plus_corrected_l2_integer(
                self.hidden, None, self.Wt, self.wv_q, 8, 5, budget=96
            )
        with self.assertRaises(cl.CorrectedL2Error):
            cl.prompt_plus_corrected_l2_integer(
                self.hidden, self.text, None, self.wv_q, 8, 5, budget=96
            )

    def test_scorer_factory_enforces_arm_inputs(self) -> None:
        with self.assertRaises(cl.CorrectedL2Error):
            cl.build_b1_scorer("not_an_arm")
        with self.assertRaises(cl.CorrectedL2Error):
            cl.build_b1_scorer(cl.ARM_FLOAT, Wt=self.Wt)
        with self.assertRaises(cl.CorrectedL2Error):
            cl.build_b1_scorer(cl.ARM_PROMPT_INTEGER)
        scorer = cl.build_b1_scorer(
            cl.ARM_PROMPT_INTEGER, Wt=self.Wt, wv_q=self.wv_q
        )
        payload = scorer(self.hidden, 8, 5, 96, self.text)
        self.assertEqual(payload["arm"], cl.ARM_PROMPT_INTEGER)
        self.assertEqual(payload["combined_scores"].shape, (40,))

    def test_two_stateless_callbacks_ignore_prompt(self) -> None:
        for arm, callback in (
            (cl.ARM_FLOAT, cl.score_corrected_l2_float),
            (cl.ARM_INTEGER, cl.score_corrected_l2_integer),
        ):
            with self.subTest(arm=arm):
                payload = callback(self.hidden, 8, 5, 96, None)
                self.assertEqual(payload["arm"], arm)
                self.assertEqual(payload["combined_scores"].shape, (40,))
                self.assertAlmostEqual(
                    float(payload["combined_scores"].max()), 1.0, delta=1e-3
                )


class HashTest(unittest.TestCase):
    def test_array_sha256_is_dtype_and_shape_explicit(self) -> None:
        ints = np.asarray([1, 2, 3], dtype=np.int64)
        floats = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
        self.assertNotEqual(cl.array_sha256(ints), cl.array_sha256(floats))
        self.assertEqual(cl.array_sha256(ints), cl.array_sha256(ints.copy()))


class FrozenB1PlugTest(unittest.TestCase):
    """Prove the callbacks plug into the unmodified frozen B1 forward."""

    GRID = 20  # 400 patches -> 100 merged groups > K=96
    BUDGET = 96

    def _run(self, scorer, text_embeddings=None):
        from evaluation.tests.test_hardware_regular_b1_integration import (  # noqa: PLC0415
            _FakeVisual,
            _grid,
            _pixels,
        )

        visual = _FakeVisual()
        result = forward_hardware_regular_b1_vision(
            visual,
            _pixels(self.GRID),
            _grid(self.GRID),
            budget=self.BUDGET,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=text_embeddings,
            integer_scorer=scorer,
        )
        return result

    def _hidden(self, result) -> np.ndarray:
        return (
            result.block0_hidden.detach()
            .to(torch.float64)
            .cpu()
            .numpy()
            .reshape(self.GRID * self.GRID // 4, 4, 1024)
        )

    def test_float_and_integer_callbacks_drive_selection(self) -> None:
        for arm, expected_fn in (
            (cl.ARM_FLOAT, cl.corrected_patch_l2_float),
            (cl.ARM_INTEGER, cl.corrected_patch_l2_integer),
        ):
            with self.subTest(arm=arm):
                scorer = cl.build_b1_scorer(arm)
                result = self._run(scorer)
                self.assertEqual(result.selection.output_group_count, self.BUDGET)
                self.assertEqual(result.integer_payload["arm"], arm)
                expected = expected_fn(self._hidden(result))["combined"]
                np.testing.assert_allclose(
                    result.integer_payload["combined_scores"], expected
                )

    def test_prompt_callback_drives_selection(self) -> None:
        from evaluation.tests.test_hardware_regular_b1_integration import (  # noqa: PLC0415
            _text_embeddings,
        )

        rng = np.random.default_rng(20261004)
        Wt = rng.normal(size=(128, 2560)).astype(np.float64)
        wv_q = rng.integers(-127, 128, size=(128, 1024)).astype(np.int64)
        text = _text_embeddings(4).astype(np.float64)
        scorer = cl.build_b1_scorer(cl.ARM_PROMPT_INTEGER, Wt=Wt, wv_q=wv_q)
        result = self._run(scorer, text_embeddings=text)
        self.assertEqual(result.selection.output_group_count, self.BUDGET)
        merged = self.GRID // 2  # spatial_merge_size == 2
        expected = cl.prompt_plus_corrected_l2_integer(
            self._hidden(result), text, Wt, wv_q, merged, merged, budget=self.BUDGET
        )["combined"]
        np.testing.assert_allclose(
            result.integer_payload["combined_scores"], expected
        )


if __name__ == "__main__":
    unittest.main()
