from __future__ import annotations

import unittest
from unittest import mock

from evaluation import summarize_layer5_adaptive_budget_screening100 as summary
from evaluation.build_cv_bench_pruning_report import score as official_score
from evaluation.layer5_adaptive_budget import compute_adaptive_budget


def manifest_row(sample_id: str, source: str, answer: str) -> dict:
    return {
        "sample_id": sample_id,
        "task": "Count",
        "type": "3D" if source == "Omni3D" else "2D",
        "source": source,
        "source_dataset": source + " test",
        "choices": ["zero", "one"],
        "answer": answer,
    }


def common_arm(sample_id: str, groups: int, output: str) -> dict:
    return {
        "sample_id": sample_id,
        "output_text": output,
        "score_mode": "l2",
        "prune_after_layer": 5,
        "selection_rule": summary.SELECTION_RULE,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "local_candidate_cap": None,
        "source_type": [0],
        "context_group_count": 0,
        "image_grid_thw": [1, 20, groups // 5 if groups % 5 == 0 else groups],
        "full_patch_tokens": groups * 4,
        "full_llm_visual_tokens": groups,
    }


def fixed_row(sample_id: str, groups: int, output: str) -> dict:
    row = common_arm(sample_id, groups, output)
    count = min(groups, 96)
    row.update({
        "requested_budget": 96,
        "identity_bypass": groups <= 96,
        "output_group_count": count,
        "real_group_count": count,
    })
    return row


def adaptive_row(sample_id: str, groups: int, output: str) -> dict:
    row = common_arm(sample_id, groups, output)
    budget = compute_adaptive_budget(groups)
    row.update(budget.as_record())
    row.update({
        "budget_mode": summary.BUDGET_MODE,
        "output_group_count": budget.effective_budget,
        "real_group_count": budget.effective_budget,
    })
    return row


class AdaptiveSummaryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = [
            manifest_row("a", "ADE20K", "(A)"),
            manifest_row("b", "COCO", "(A)"),
            manifest_row("c", "Omni3D", "(A)"),
        ]
        self.dense = [
            {"sample_id": row["sample_id"], "output_text": "(A) zero"}
            for row in self.manifest
        ]
        groups = [64, 385, 1400]
        self.fixed = [
            fixed_row("a", groups[0], "(B) one"),
            fixed_row("b", groups[1], "(A) zero"),
            fixed_row("c", groups[2], "(A) zero"),
        ]
        self.adaptive = [
            adaptive_row(row["sample_id"], group, "(A) zero")
            for row, group in zip(self.manifest, groups)
        ]

    def analyze(self):
        with mock.patch.object(summary, "BOOTSTRAP_REPEATS", 20):
            return summary.analyze(
                screening_rows=self.manifest,
                dense_full_rows=self.dense,
                fixed_k96_rows=self.fixed,
                adaptive_rows=self.adaptive,
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_uses_official_cv_bench_scorer(self) -> None:
        with mock.patch.object(summary, "score", wraps=official_score) as scorer:
            report = self.analyze()
        self.assertGreater(scorer.call_count, 0)
        self.assertEqual(
            report["metrics"]["adaptive"]["primary_metric"],
            'score(...)["cv_bench_accuracy"]',
        )
        self.assertAlmostEqual(
            report["metrics"]["adaptive"]["cv_bench_accuracy"], 1.0
        )
        self.assertAlmostEqual(
            report["metrics"]["fixed_k96"]["cv_bench_accuracy"], 0.75
        )
        self.assertFalse(
            report["project_interpretation"]["is_official_cv_bench_metric"]
        )

    def test_contract_rejects_tampered_budget(self) -> None:
        self.adaptive[1]["effective_budget"] = 96
        with self.assertRaises(summary.InputContractError):
            self.analyze()

    def test_identity_and_adaptive_larger_than_k96_are_valid(self) -> None:
        report = self.analyze()
        self.assertEqual(
            report["validation"]["adaptive_identity_or_exact_budget"], True
        )
        self.assertEqual(
            report["adaptive_budget_overview"]["identity_bypass_count"], 1
        )
        self.assertEqual(
            report["adaptive_records_exceeding_fixed_k96"], ["b", "c"]
        )


if __name__ == "__main__":
    unittest.main()
