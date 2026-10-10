#!/usr/bin/env python3
"""CPU-only tests for the current-Selector CV-Bench failure diagnostic.

Read-only: these tests open the frozen JSONL/report files but never rewrite them
and never touch the formal quality report or ``configs/evaluation_protocol.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import analyze_current_selector_failure as diag  # noqa: E402

# Frozen expected values (previous failure: the unweighted sample mean was
# mis-labelled as the CV-Bench primary).  Primary = score(...)["cv_bench_accuracy"].
EXPECTED_CV_BENCH_ACCURACY = {
    "K48": 0.6534285248201898,
    "K64": 0.7033306104226154,
    "K96": 0.7648694965313552,
}
EXPECTED_UNWEIGHTED_SAMPLE_ACCURACY = {
    "K48": 0.6493555724033359,
    "K64": 0.6990144048521607,
    "K96": 0.7615617892342684,
}
L2_LAYER5_CV_BENCH_ACCURACY = 0.7954859905017024
K96_MINUS_L2 = -0.03061649397034721


def _row(sample_id: str, group_count: int, task: str, source: str, correct: bool, text: str = "(A)") -> dict:
    return {
        "sample_id": sample_id,
        "full_llm_visual_tokens": group_count,
        "task": task,
        "source": source,
        "output_text": text,
        "official_correct": correct,
        "official_parsed": "A" if correct else None,
    }


class GBucketTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(diag.g_bucket(0), "G<=96")
        self.assertEqual(diag.g_bucket(96), "G<=96")
        self.assertEqual(diag.g_bucket(97), "97-256")
        self.assertEqual(diag.g_bucket(256), "97-256")
        self.assertEqual(diag.g_bucket(257), "257-768")
        self.assertEqual(diag.g_bucket(768), "257-768")
        self.assertEqual(diag.g_bucket(769), ">768")
        self.assertEqual(diag.g_bucket(40000), ">768")


class BucketAccuracyTest(unittest.TestCase):
    def test_accuracy_and_parse_rate(self) -> None:
        rows = [
            _row("0", 64, "Count", "ADE20K", True),
            _row("1", 64, "Count", "ADE20K", False),
            _row("2", 64, "Count", "ADE20K", False, text="unparsed"),
        ]
        rows[1]["official_parsed"] = "B"  # parsed but wrong
        stats = diag.bucket_accuracy(rows)
        self.assertEqual(stats["sample_count"], 3)
        self.assertAlmostEqual(stats["unweighted_sample_accuracy"], 1 / 3)
        self.assertAlmostEqual(stats["parse_success_rate"], 2 / 3)
        self.assertEqual(stats["metric"], diag.SECONDARY_METRIC_LABEL)

    def test_empty_bucket(self) -> None:
        stats = diag.bucket_accuracy([])
        self.assertEqual(stats["sample_count"], 0)
        self.assertIsNone(stats["unweighted_sample_accuracy"])


class IdentityVerificationTest(unittest.TestCase):
    def _fixture(self):
        manifest = {
            "0": {"choices": ["a", "b"], "answer": "(A)"},
            "1": {"choices": ["a", "b"], "answer": "(B)"},
        }
        rows = [
            {
                "sample_id": "0",
                "full_llm_visual_tokens": 64,
                "identity_bypass": True,
                "output_group_count": 64,
                "actual_token_retention": 1.0,
                "selected_group_indices": list(range(64)),
                "output_text": "(A)",
                "parsed_answer": "A",
            },
            {
                "sample_id": "1",
                "full_llm_visual_tokens": 128,
                "identity_bypass": False,
                "output_group_count": 64,
                "actual_token_retention": 0.5,
                "selected_group_indices": list(range(64)),
                "output_text": "(B)",
                "parsed_answer": "B",
            },
        ]
        dense = {"0": "(A)", "1": "(B)"}
        return rows, dense, manifest

    def test_identity_matches_dense(self) -> None:
        rows, dense, manifest = self._fixture()
        check = diag.verify_identity_bypass("K64", 64, rows, dense, manifest)
        self.assertTrue(check["identity_set_equals_g_le_k"])
        self.assertEqual(check["identity_bypass_count"], 1)
        self.assertEqual(check["dense_text_match_count"], 1)
        self.assertEqual(check["dense_answer_match_count"], 1)
        self.assertEqual(check["identity_full_sequence_match_count"], 1)
        self.assertTrue(check["all_identity_outputs_match_dense"])

    def test_dense_mismatch_detected(self) -> None:
        rows, dense, manifest = self._fixture()
        dense["0"] = "(B)"
        check = diag.verify_identity_bypass("K64", 64, rows, dense, manifest)
        self.assertFalse(check["all_identity_outputs_match_dense"])
        self.assertEqual(len(check["mismatches"]), 1)

    def test_spurious_identity_detected(self) -> None:
        rows, dense, manifest = self._fixture()
        rows[1]["identity_bypass"] = True
        check = diag.verify_identity_bypass("K64", 64, rows, dense, manifest)
        self.assertFalse(check["identity_set_equals_g_le_k"])
        self.assertEqual(check["spurious_identity_for_g_gt_k"], ["1"])


class RealArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = diag.build_report()
        cls.report_json = json.dumps(cls.report)

    def test_formal_scores_reproduced(self) -> None:
        self.assertTrue(self.report["formal_scores_reproduced"])
        for label, candidate in self.report["candidates"].items():
            self.assertEqual(candidate["sample_count"], 2638, label)
            self.assertTrue(candidate["formal_report_matches_recompute"], label)

    def test_primary_metric_is_cv_bench_accuracy(self) -> None:
        """Primary must be score(...)["cv_bench_accuracy"], never the sample mean."""
        self.assertEqual(self.report["primary_metric"], diag.PRIMARY_METRIC)
        self.assertIn("cv_bench_accuracy", diag.PRIMARY_METRIC)
        for label, candidate in self.report["candidates"].items():
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    candidate["cv_bench_accuracy"],
                    EXPECTED_CV_BENCH_ACCURACY[label],
                    places=12,
                )
                self.assertAlmostEqual(
                    candidate["unweighted_sample_accuracy"],
                    EXPECTED_UNWEIGHTED_SAMPLE_ACCURACY[label],
                    places=12,
                )
                # The two metrics genuinely differ, so mixing them is detectable.
                self.assertGreater(
                    abs(
                        candidate["cv_bench_accuracy"]
                        - candidate["unweighted_sample_accuracy"]
                    ),
                    1e-6,
                )

    def test_formal_report_match_uses_cv_bench_accuracy(self) -> None:
        """The formal match must compare candidate.cv_bench_accuracy, not accuracy."""
        for label, candidate in self.report["candidates"].items():
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    candidate["formal_report_cv_bench_accuracy"],
                    EXPECTED_CV_BENCH_ACCURACY[label],
                    places=12,
                )
                self.assertAlmostEqual(
                    candidate["cv_bench_accuracy"],
                    candidate["formal_report_cv_bench_accuracy"],
                    places=12,
                )
                # The old (wrong) comparison against candidate["accuracy"] would fail.
                self.assertGreater(
                    abs(
                        candidate["cv_bench_accuracy"]
                        - candidate["formal_report_unweighted_sample_accuracy"]
                    ),
                    1e-6,
                )

    def test_identity_bypass_counts_and_consistency(self) -> None:
        self.assertTrue(self.report["all_identity_checks_pass"])
        expected = {"K48": 0, "K64": 61, "K96": 143}
        for label, count in expected.items():
            check = self.report["identity_bypass_verification"][label]
            self.assertEqual(check["identity_bypass_count"], count, label)
            self.assertEqual(check["g_le_k_count"], count, label)
            self.assertEqual(check["dense_text_match_count"], count, label)
            self.assertTrue(check["all_identity_outputs_match_dense"], label)

    def test_k96_g_buckets(self) -> None:
        buckets = self.report["candidates"]["K96"]["g_buckets"]
        self.assertEqual(buckets["G<=96"]["sample_count"], 143)
        self.assertEqual(buckets["97-256"]["sample_count"], 781)
        self.assertEqual(buckets["257-768"]["sample_count"], 1144)
        self.assertEqual(buckets[">768"]["sample_count"], 570)
        self.assertEqual(sum(b["sample_count"] for b in buckets.values()), 2638)
        self.assertAlmostEqual(
            buckets["G<=96"]["unweighted_sample_accuracy"], 0.7272727272727273
        )

    def test_g_buckets_are_secondary_unweighted_only(self) -> None:
        """G-bucket means must be labelled secondary; never the CV-Bench primary."""
        for label, candidate in self.report["candidates"].items():
            for bucket_label, stat in candidate["g_buckets"].items():
                with self.subTest(label=label, bucket=bucket_label):
                    self.assertEqual(stat["metric"], diag.SECONDARY_METRIC_LABEL)
                    self.assertIn("unweighted", stat["metric"])
                    self.assertIn("secondary", stat["metric"])
                    self.assertIn("unweighted_sample_accuracy", stat)
                    # No key may present a bucket mean as the primary metric.
                    self.assertNotIn("accuracy", stat)
                    self.assertNotIn("cv_bench_accuracy", stat)

    def test_no_duplicate_absolute_delta_vs_dense_key(self) -> None:
        for label, candidate in self.report["candidates"].items():
            with self.subTest(label=label):
                self.assertNotIn("absolute_delta_vs_dense", candidate)

    def test_l2_cut_point_caveat(self) -> None:
        cmp = self.report["l2_layer5_comparison"]
        self.assertTrue(cmp["cut_point_differs"])
        self.assertTrue(cmp["not_attributable_to_ranker"])
        self.assertEqual(cmp["l2_layer5_prune_after_layer"], 5)
        self.assertIn("not", cmp["attribution_caveat"].lower())
        self.assertAlmostEqual(
            cmp["l2_layer5_cv_bench_accuracy"],
            L2_LAYER5_CV_BENCH_ACCURACY,
            places=12,
        )

    def test_l2_comparison_uses_current_cv_bench_accuracy(self) -> None:
        cmp = self.report["l2_layer5_comparison"]
        self.assertEqual(cmp["comparison_metric"], diag.PRIMARY_METRIC)
        for label, item in cmp["candidates"].items():
            with self.subTest(label=label):
                self.assertAlmostEqual(
                    item["current_selector_cv_bench_accuracy"],
                    EXPECTED_CV_BENCH_ACCURACY[label],
                    places=12,
                )
        k96 = cmp["candidates"]["K96"]
        self.assertAlmostEqual(k96["current_minus_l2_layer5"], K96_MINUS_L2, places=12)
        # The forbidden unweighted-based delta would be -0.033924..., not -0.030616...
        unweighted_delta = (
            EXPECTED_UNWEIGHTED_SAMPLE_ACCURACY["K96"] - L2_LAYER5_CV_BENCH_ACCURACY
        )
        self.assertGreater(
            abs(k96["current_minus_l2_layer5"] - unweighted_delta), 1e-6
        )

    def test_inputs_hashed(self) -> None:
        inputs = self.report["inputs_sha256"]
        self.assertEqual(
            set(inputs),
            {
                "formal_quality_report",
                "manifest",
                "dense_jsonl",
                "l2_layer5_report",
                "official_scorer_module",
                "candidate_K48_jsonl",
                "candidate_K64_jsonl",
                "candidate_K96_jsonl",
            },
        )
        for name, item in inputs.items():
            self.assertEqual(len(item["sha256"]), 64, name)
            int(item["sha256"], 16)

    def test_forbidden_primary_metrics_present(self) -> None:
        self.assertEqual(
            tuple(self.report["forbidden_primary_metrics"]),
            diag.FORBIDDEN_PRIMARY_METRICS,
        )

    def test_markdown_has_caveat_and_tables(self) -> None:
        md = diag.render_markdown(self.report)
        self.assertIn("归因限制", md)
        self.assertIn("K96", md)
        self.assertIn("G<=96", md)
        self.assertIn("identity bypass", md)

    def test_markdown_shows_both_metrics_and_explains_difference(self) -> None:
        md = diag.render_markdown(self.report)
        self.assertIn("CV-Bench 主分", md)
        self.assertIn("unweighted_sample_accuracy", md)
        self.assertIn("两种口径的区别", md)
        self.assertIn("不得写作 CV-Bench 主分", md)
        self.assertIn("-0.0306", md)
        # The G-bucket section must warn against promoting it to the primary.
        self.assertIn("secondary unweighted sample accuracy diagnostic", md)

    def test_sha256_matches_hashlib(self) -> None:
        digest = hashlib.sha256(diag.FORMAL_JSON.read_bytes()).hexdigest()
        self.assertEqual(
            self.report["inputs_sha256"]["formal_quality_report"]["sha256"], digest
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
