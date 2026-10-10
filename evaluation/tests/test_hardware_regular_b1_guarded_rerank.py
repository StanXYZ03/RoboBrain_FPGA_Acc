#!/usr/bin/env python3
"""CPU fake-visual tests for the v2 L2-guarded B1 integration.

No real weights and no CUDA.  The tests pin the frozen integration equivalence
(zero swaps reproduces the corrected integer-L2 path element-wise), the exact
K96 / 87 real + 9 context layout, the v2 identity-bypass label and the
fail-closed scorer contract.
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

from evaluation.conservative_l2_guarded_reranker import (  # noqa: E402
    BUDGET,
    MECHANISM,
    build_guarded_scorer,
    select_guarded_rerank,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ARM_INTEGER,
    build_b1_scorer,
    corrected_patch_l2_integer,
)
from evaluation.hardware_regular_b1_guarded_rerank import (  # noqa: E402
    forward_guarded_rerank_b1_vision,
)
from evaluation.hardware_regular_b1_integration import (  # noqa: E402
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
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
    return select_guarded_rerank(
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


class GuardedRerankIntegrationTest(unittest.TestCase):
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
        new = forward_guarded_rerank_b1_vision(
            self.visual,
            self.pixels,
            self.grid_tensor,
            text_embeddings=self.text_embeddings,
            guarded_scorer=_zero_swap_scorer,
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
        scorer = build_guarded_scorer(Wt, wv_q, budgets=(96,))
        new = forward_guarded_rerank_b1_vision(
            self.visual,
            self.pixels,
            self.grid_tensor,
            text_embeddings=self.text_embeddings,
            guarded_scorer=scorer,
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
        self.assertEqual(plan["mechanism"], MECHANISM)
        self.assertEqual(plan["utility_formula"], "semantic_q15 + corrected_l2_q15")
        self.assertEqual(
            plan["legal_gain_formula"],
            "semantic_gain - l2_loss = utility_added - utility_removed",
        )
        self.assertEqual(len(plan["removed_group_indices"]), payload["swap_count"])
        self.assertEqual(len(plan["added_group_indices"]), payload["swap_count"])
        for index, legal in enumerate(plan["swap_legal_gain_q16"]):
            self.assertGreater(int(legal), 0)
            self.assertEqual(
                int(legal),
                int(plan["swap_semantic_gain_q15"][index])
                - int(plan["swap_l2_loss_q15"][index]),
            )
        self.assertTrue(
            set(plan["baseline_anchors"]).issubset(set(plan["reranked_selected_real"]))
        )

    def test_identity_bypass_is_labelled_v2(self) -> None:
        Wt, wv_q = _fake_projection_bundle()
        scorer = build_guarded_scorer(Wt, wv_q, budgets=(96,))
        grid = 8  # G = 16 <= 96
        new = forward_guarded_rerank_b1_vision(
            self.visual,
            frozen_tests._pixels(grid),
            frozen_tests._grid(grid),
            text_embeddings=self.text_embeddings,
            guarded_scorer=scorer,
            budget=96,
        )
        self.assertTrue(new.selection.bypass)
        payload = new.integer_payload
        self.assertEqual(payload["mechanism"], MECHANISM)
        self.assertEqual(payload["swap_count"], 0)
        self.assertEqual(
            payload["context_pooling_basis"], "corrected_integer_l2_norm_q15"
        )
        self.assertEqual(new.image_embeds.shape[0], 16)

    def test_fail_closed_on_bad_scorer(self) -> None:
        def bad_scorer(hidden_groups, merged_h, merged_w, budget, text_embeddings):
            del hidden_groups, merged_h, merged_w, budget, text_embeddings
            return {}

        with self.assertRaises(SelectorContractError):
            forward_guarded_rerank_b1_vision(
                self.visual,
                self.pixels,
                self.grid_tensor,
                text_embeddings=self.text_embeddings,
                guarded_scorer=bad_scorer,
                budget=96,
            )

    def test_requires_text_embeddings(self) -> None:
        Wt, wv_q = _fake_projection_bundle()
        scorer = build_guarded_scorer(Wt, wv_q, budgets=(96,))
        with self.assertRaises(SelectorContractError):
            forward_guarded_rerank_b1_vision(
                self.visual,
                self.pixels,
                self.grid_tensor,
                text_embeddings=None,
                guarded_scorer=scorer,
                budget=96,
            )

    def test_budget_default_is_frozen(self) -> None:
        self.assertEqual(BUDGET, 96)


if __name__ == "__main__":
    unittest.main()
