from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import summarize_vsr_zeroshot_dev_budget_curve as summary
from evaluation import vsr_zeroshot_dev_contract as contract
from evaluation import run_vsr_zeroshot_dev_budget_curve as runner


class VSRContractTests(unittest.TestCase):
    def test_parse_true_false_is_strict(self) -> None:
        self.assertIs(contract.parse_true_false("True."), True)
        self.assertIs(contract.parse_true_false(" false \n"), False)
        self.assertIsNone(contract.parse_true_false("The answer is true"))

    def test_arm_budget_mapping(self) -> None:
        self.assertEqual(contract.arm_kind_budget("dense_edge576"), ("dense", None))
        self.assertEqual(contract.arm_kind_budget("v2_edge576_b1_k144"), ("v2", 144))
        with self.assertRaises(contract.VSRContractError):
            contract.arm_kind_budget("v2_edge576_b1_k100")

    def test_budget_specific_candidate_cap(self) -> None:
        self.assertEqual(contract.local_candidate_cap(96), 16)
        self.assertEqual(contract.local_candidate_cap(144), 16)
        self.assertEqual(contract.local_candidate_cap(192), 32)

    def test_selector_sanitization_removes_annotations(self) -> None:
        sample = {key: "x" for key in contract.SELECTOR_KEYS}
        sample.update({"official_label": 1, "relation": "on", "caption": "x"})
        sanitized = contract.sanitize_for_selector(sample)
        self.assertNotIn("official_label", sanitized)
        self.assertNotIn("relation", sanitized)

    def test_normalize_budget_and_hash(self) -> None:
        sample = {
            "sample_id": "s", "image_sha256": "a" * 64, "relation": "on",
            "official_label": 1, "prompt": "p",
        }
        frozen = {
            "output_text": "True", "image_grid_thw": [1, 20, 20],
            "selected_group_indices": list(range(96)), "identity_bypass": False,
            "output_tokens": 1,
        }
        row = runner.normalize("v2_edge576_b1_k96", sample, frozen, 0.1, "b" * 64, {})
        self.assertTrue(row["correct"])
        self.assertEqual(row["full_llm_visual_tokens"], 100)
        self.assertEqual(row["record_sha256"], contract.canonical_record_sha256(row))

    def test_resume_rejects_tampering(self) -> None:
        sample = {"sample_id": "s"}
        row = {"sample_id": "s", "arm": contract.ARM_DENSE, "manifest_sha256": "m", "requested_budget": None}
        row["record_sha256"] = contract.canonical_record_sha256(row)
        row["arm"] = "tampered"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(contract.VSRContractError):
                contract.validate_resume(path, [sample], contract.ARM_DENSE, "m")

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        left = {"a": True, "b": False, "c": True}
        right = {"a": False, "b": False, "c": True}
        first = summary.paired_bootstrap(left, right, repetitions=100, seed=7)
        second = summary.paired_bootstrap(left, right, repetitions=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["observed_difference"], 1 / 3)

    def test_describe(self) -> None:
        values = summary.describe([1.0, 2.0, 3.0])
        self.assertEqual(values["mean"], 2.0)
        self.assertEqual(values["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
