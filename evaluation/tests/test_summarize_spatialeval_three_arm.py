from __future__ import annotations

import types
import unittest

from evaluation import summarize_spatialeval_three_arm as summary
from evaluation import spatialeval_external_contract as contract


class FakeOfficial(types.SimpleNamespace):
    @staticmethod
    def extract_answer_from_text_spatialmap(text, qid, model):
        return text

    @staticmethod
    def extract_answer_from_text_mazenav(text, qid, model):
        return text

    @staticmethod
    def extract_answer_from_text_spatialgrid(text, qid, model):
        return text


def records(arm: str, outputs: list[str]) -> list[dict]:
    tasks = ("spatialmap", "mazenav", "spatialgrid")
    rows = []
    for index, (task, output) in enumerate(zip(tasks, outputs)):
        rows.append(
            {
                "sample_id": f"{task}.vqa.{index}.0",
                "output_text": output,
                "oracle_answer": "yes",
                "arm": arm,
            }
        )
    return rows


class SummarizeSpatialEvalTests(unittest.TestCase):
    def test_gate_pass_and_task_reporting(self) -> None:
        by_arm = {
            contract.ARM_DENSE: records(contract.ARM_DENSE, ["yes", "yes", "yes"]),
            contract.ARM_L2: records(contract.ARM_L2, ["yes", "no", "yes"]),
            contract.ARM_V2: records(contract.ARM_V2, ["yes", "yes", "yes"]),
        }
        result = summary.summarize_records(
            by_arm,
            expected_count=None,
            official_module=FakeOfficial(),
            bootstrap_repetitions=100,
        )
        self.assertTrue(result["gates"]["all_pass"])
        self.assertEqual(
            result["comparison"]["v2_dense_equal_task_macro_retention"], 1.0
        )
        self.assertIsNone(result["gates"]["per_task_gates"])
        self.assertFalse(result["official_total_score_exists"])

    def test_mismatched_sample_sets_fail(self) -> None:
        by_arm = {
            contract.ARM_DENSE: records(contract.ARM_DENSE, ["yes"] * 3),
            contract.ARM_L2: records(contract.ARM_L2, ["yes"] * 3),
            contract.ARM_V2: records(contract.ARM_V2, ["yes"] * 3)[:-1],
        }
        with self.assertRaises(contract.SpatialEvalExternalContractError):
            summary.summarize_records(
                by_arm,
                expected_count=None,
                official_module=FakeOfficial(),
                bootstrap_repetitions=10,
            )

    def test_bootstrap_is_deterministic(self) -> None:
        v2 = {
            "spatialmap.vqa.0.0": 1,
            "mazenav.vqa.0.0": 0,
            "spatialgrid.vqa.0.0": 1,
        }
        l2 = {key: 0 for key in v2}
        first = summary.stratified_paired_bootstrap(v2, l2, seed=7, repetitions=20)
        second = summary.stratified_paired_bootstrap(v2, l2, seed=7, repetitions=20)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
