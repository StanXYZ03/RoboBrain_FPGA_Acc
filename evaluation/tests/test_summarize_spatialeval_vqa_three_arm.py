from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import summarize_spatialeval_vqa_three_arm as summarizer
from evaluation.spatialeval_vqa_contract import ARM_DENSE, ARM_L2, ARM_V2, ARMS

CORRECT_OUTPUT = {
    "spatialmap": "A. Northeast",
    "mazenav": "There are 3 right turns",
    "spatialgrid": "There are 3 blocks",
}
WRONG_OUTPUT = {
    "spatialmap": "A. Southwest",
    "mazenav": "There are 7 right turns",
    "spatialgrid": "There are 7 blocks",
}
ORACLE = {"spatialmap": "Northeast", "mazenav": "3", "spatialgrid": "3"}


def build_case(per_task: int = 10):
    """Return (predictions_by_arm, oracle_by_id, sample_ids) with small counts.

    Deterministic correctness patterns: dense 100%, corrected-L2 80%, v2 100%,
    so the default case passes both pre-registered gates.
    """

    sample_ids = [
        f"{task}.vqa.{index}.0"
        for task in ("spatialmap", "mazenav", "spatialgrid")
        for index in range(per_task)
    ]
    oracle = {
        sample_id: ORACLE[sample_id.split(".")[0]] for sample_id in sample_ids
    }

    def arm(hits):
        predictions = []
        for task in ("spatialmap", "mazenav", "spatialgrid"):
            for index in range(per_task):
                sample_id = f"{task}.vqa.{index}.0"
                correct = (index % 10) < hits
                output = CORRECT_OUTPUT if correct else WRONG_OUTPUT
                predictions.append(
                    {"sample_id": sample_id, "output_text": output[task]}
                )
        return predictions

    predictions_by_arm = {
        ARM_DENSE: arm(10),
        ARM_L2: arm(8),
        ARM_V2: arm(10),
    }
    return predictions_by_arm, oracle, sample_ids


class SummarizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions, self.oracle, self.sample_ids = build_case()
        self.count = len(self.sample_ids)

    def summarize(self, predictions=None, **kwargs):
        return summarizer.summarize_arms(
            predictions or self.predictions,
            self.oracle,
            expected_sample_ids=self.sample_ids,
            reference_manifest_sha256="ref",
            repetitions=200,
            expected_count=self.count,
            **kwargs,
        )

    def test_identical_sets_required(self) -> None:
        broken = {arm: list(values) for arm, values in self.predictions.items()}
        broken[ARM_V2] = broken[ARM_V2][:-1]
        with self.assertRaises(summarizer.SpatialEvalSummaryError):
            self.summarize(broken)

    def test_missing_arm_required(self) -> None:
        broken = {arm: values for arm, values in self.predictions.items() if arm != ARM_V2}
        with self.assertRaises(summarizer.SpatialEvalSummaryError):
            self.summarize(broken)

    def test_report_structure_and_gate_fields(self) -> None:
        report = self.summarize()
        self.assertEqual(report["count_per_arm"], self.count)
        self.assertEqual(set(report["arms"]), set(ARMS))
        self.assertIn("project_task_macro", report["arms"][ARM_V2])
        self.assertIn(
            "strict_equality_diagnostic", report["arms"][ARM_V2]
        )
        gate = report["project_gate"]
        self.assertEqual(gate["v2_dense_task_macro"]["minimum"], 0.90)
        self.assertEqual(
            gate["v2_minus_l2_task_macro_bootstrap"]["seed"], summarizer.BOOTSTRAP_SEED
        )
        self.assertEqual(
            gate["v2_minus_l2_task_macro_bootstrap"]["stratification"],
            "task::question_id",
        )
        self.assertTrue(report["declarations"]["parse_failure_counts_wrong"])
        self.assertTrue(report["declarations"]["macro_micro_are_project_aggregations"])
        self.assertFalse(report["declarations"]["official_cross_task_total_exists"])

    def test_ratio_gate_fails_when_v2_too_weak(self) -> None:
        weak = {arm: list(values) for arm, values in self.predictions.items()}
        weak[ARM_V2] = [
            {**item, "output_text": WRONG_OUTPUT[item["sample_id"].split(".")[0]]}
            for item in weak[ARM_V2]
        ]
        report = self.summarize(weak)
        self.assertFalse(report["project_gate"]["v2_dense_task_macro"]["pass"])
        self.assertEqual(report["status"], "SPATIALEVAL_EXTERNAL_GATES_FAIL")

    def test_bootstrap_is_deterministic_for_seed(self) -> None:
        first = self.summarize(seed=20261009)
        second = self.summarize(seed=20261009)
        third = self.summarize(seed=1)
        self.assertEqual(
            first["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci"],
            second["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci"],
        )
        self.assertEqual(
            first["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci_lower"],
            second["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci_lower"],
        )
        # Different seed should still produce a valid interval.
        self.assertLessEqual(
            third["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci_lower"],
            third["project_gate"]["v2_minus_l2_task_macro_bootstrap"]["ci_upper"],
        )

    def test_parse_failure_counts_wrong(self) -> None:
        broken = {arm: list(values) for arm, values in self.predictions.items()}
        broken[ARM_L2] = [
            {**item, "output_text": ""} for item in broken[ARM_L2]
        ]
        report = self.summarize(broken)
        self.assertEqual(report["arms"][ARM_L2]["correct"], 0)
        self.assertEqual(
            report["arms"][ARM_L2]["parse_failure_count"], self.count
        )


class LoadPredictionsTests(unittest.TestCase):
    def test_wrong_count_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            path.write_text(
                json.dumps(
                    {"sample_id": "spatialmap.vqa.1.0", "output_text": "x", "arm": ARM_DENSE}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(summarizer.SpatialEvalSummaryError):
                summarizer.load_arm_predictions(path, arm=ARM_DENSE, expected_count=2)

    def test_wrong_arm_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            path.write_text(
                json.dumps(
                    {"sample_id": "spatialmap.vqa.1.0", "output_text": "x", "arm": ARM_L2}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(summarizer.SpatialEvalSummaryError):
                summarizer.load_arm_predictions(path, arm=ARM_DENSE, expected_count=1)


class BootstrapStatisticTests(unittest.TestCase):
    def per_sample(self, treated_hits: int, control_hits: int, total: int = 30):
        treated = []
        control = []
        for index in range(total):
            task = ("spatialmap", "mazenav", "spatialgrid")[index % 3]
            sample_id = f"{task}.vqa.{index}.0"
            treated.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "question_id": 0,
                    "official_correct": int(index < treated_hits),
                }
            )
            control.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "question_id": 0,
                    "official_correct": int(index < control_hits),
                }
            )
        return {ARM_V2: treated, ARM_L2: control}

    def test_observed_macro_difference(self) -> None:
        result = summarizer.stratified_paired_bootstrap_macro(
            self.per_sample(20, 10),
            treated_arm=ARM_V2,
            control_arm=ARM_L2,
            repetitions=50,
        )
        self.assertGreater(result["observed"], 0.0)
        self.assertLessEqual(result["ci_lower"], result["ci_upper"])
        self.assertEqual(result["stratification"], "task::question_id")


if __name__ == "__main__":
    unittest.main()
