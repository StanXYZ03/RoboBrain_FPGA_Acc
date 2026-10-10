#!/usr/bin/env python3
"""CPU fake-visual tests for the residual-rerank B1 integration.

No real weights and no CUDA.  The tests pin the frozen integration equivalence
(zero swaps reproduces the corrected integer-L2 path element-wise), the exact
K96 / 87 real + 9 context layout, the identity bypass and the fail-closed scorer
contract.
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
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import test_hardware_regular_b1_integration as frozen_tests  # noqa: E402

from evaluation.conservative_l2_residual_reranker import (  # noqa: E402
    build_residual_scorer,
    select_conservative_residual_rerank,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ARM_INTEGER,
    build_b1_scorer,
    corrected_patch_l2_integer,
)
from evaluation.hardware_regular_b1_integration import (  # noqa: E402
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
)
from evaluation.hardware_regular_b1_residual_rerank import (  # noqa: E402
    forward_conservative_residual_rerank_b1_vision,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    SelectorContractError,
)


def _zero_swap_scorer(hidden_groups, merged_h, merged_w, budget, text_embeddings):
    del text_embeddings
    l2 = corrected_patch_l2_integer(hidden_groups)["l2_norm_q15"]
    return select_conservative_residual_rerank(
        hidden_groups,
        merged_h,
        merged_w,
        semantic_norm_q15=l2,
        l2_norm_q15=l2,
        budget=budget,
    )


def _fake_projection_bundle(seed: int = 13):
    rng = np.random.default_rng(seed)
    Wt = rng.normal(size=(SCORE_DIM, PROMPT_DIM))
    wv_q = rng.integers(-127, 128, size=(SCORE_DIM, FEATURE_DIM)).astype(np.int64)
    return Wt, wv_q


class ResidualRerankIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = 44  # G = 484 > 96
        self.visual = frozen_tests._FakeVisual()
        self.pixels = frozen_tests._pixels(self.grid)
        self.grid_tensor = frozen_tests._grid(self.grid)
        self.text_embeddings = frozen_tests._text_embeddings(5)

    def test_zero_swap_matches_frozen_corrected_l2(self) -> None:
        baseline = forward_hardware_regular_b1_vision(
            self.visual,
            self.pixels,
            self.grid_tensor,
            budget=96,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=None,
            integer_scorer=build_b1_scorer(ARM_INTEGER, budgets=(96,)),
        )
        new = forward_conservative_residual_rerank_b1_vision(
            self.visual,
            self.pixels,
            self.grid_tensor,
            text_embeddings=self.text_embeddings,
            residual_scorer=_zero_swap_scorer,
            budget=96,
        )
        self.assertEqual(new.integer_payload["swap_count"], 0)
        self.assertEqual(new.image_embeds.shape[0], 96)
        self.assertEqual(new.real_group_count, 87)
        self.assertEqual(new.context_group_count, 9)
        self.assertTrue(np.array_equal(baseline.group_indices, new.group_indices))
        self.assertTrue(np.allclose(baseline.selection.groups, new.selection.groups))
        self.assertTrue(np.array_equal(baseline.source_type, new.source_type))
        self.assertTrue(np.array_equal(baseline.region_id, new.region_id))
        self.assertTrue(np.array_equal(baseline.patch_indices, new.patch_indices))

    def test_rerank_layout_and_audit(self) -> None:
        Wt, wv_q = _fake_projection_bundle()
        scorer = build_residual_scorer(Wt, wv_q, budgets=(96,))
        new = forward_conservative_residual_rerank_b1_vision(
            self.visual,
            self.pixels,
            self.grid_tensor,
            text_embeddings=self.text_embeddings,
            residual_scorer=scorer,
            budget=96,
        )
        payload = new.integer_payload
        self.assertEqual(new.image_embeds.shape[0], 96)
        self.assertEqual(new.real_group_count, 87)
        self.assertEqual(new.context_group_count, 9)
        self.assertTrue(np.all(np.diff(new.group_indices) > 0))
        self.assertTrue(np.all(new.valid))
        self.assertLessEqual(payload["swap_count"], 16)
        plan = payload["baseline_plan"]
        self.assertEqual(plan["mechanism"], "conservative_l2_residual_rerank_v1")
        self.assertEqual(plan["residual_formula"], "semantic_norm_q15 - corrected_l2_norm_q15")
        self.assertEqual(len(plan["removed_group_indices"]), payload["swap_count"])
        self.assertEqual(len(plan["added_group_indices"]), payload["swap_count"])
        self.assertTrue(
            set(plan["baseline_anchors"]).issubset(set(plan["reranked_selected_real"]))
        )

    def test_identity_bypass(self) -> None:
        Wt, wv_q = _fake_projection_bundle()
        scorer = build_residual_scorer(Wt, wv_q, budgets=(96,))
        grid = 8  # G = 16 <= 96
        new = forward_conservative_residual_rerank_b1_vision(
            self.visual,
            frozen_tests._pixels(grid),
            frozen_tests._grid(grid),
            text_embeddings=self.text_embeddings,
            residual_scorer=scorer,
            budget=96,
        )
        self.assertTrue(new.selection.bypass)
        self.assertEqual(new.integer_payload["swap_count"], 0)
        self.assertEqual(new.image_embeds.shape[0], 16)

    def test_fail_closed_on_bad_scorer(self) -> None:
        def bad_scorer(hidden_groups, merged_h, merged_w, budget, text_embeddings):
            del hidden_groups, merged_h, merged_w, budget, text_embeddings
            return {}

        with self.assertRaises(SelectorContractError):
            forward_conservative_residual_rerank_b1_vision(
                self.visual,
                self.pixels,
                self.grid_tensor,
                text_embeddings=self.text_embeddings,
                residual_scorer=bad_scorer,
                budget=96,
            )

    def test_requires_text_embeddings(self) -> None:
        Wt, wv_q = _fake_projection_bundle()
        scorer = build_residual_scorer(Wt, wv_q, budgets=(96,))
        with self.assertRaises(SelectorContractError):
            forward_conservative_residual_rerank_b1_vision(
                self.visual,
                self.pixels,
                self.grid_tensor,
                text_embeddings=None,
                residual_scorer=scorer,
                budget=96,
            )


if __name__ == "__main__":
    unittest.main()
