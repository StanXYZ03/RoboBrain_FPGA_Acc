from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.extract_vsr_b1_margin_teacher import (
    MANIFEST_SHA256,
    VSRTeacherError,
    canonical_record_sha256,
    teacher_targets,
    validate_label_contract,
    validate_resume,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": {"False": [4049], "True": [2514]}[text]}

    def decode(self, ids):
        return {4049: "False", 2514: "True"}[ids[0]]


class VSRTeacherPureTest(unittest.TestCase):
    def test_true_false_single_token_contract(self):
        result = validate_label_contract(FakeTokenizer())
        self.assertEqual(result["branch_token_ids"], [4049, 2514])
        self.assertTrue(result["single_distinct_token_per_label"])

    def test_teacher_targets_are_normalized(self):
        result = teacher_targets([0.0, 1.0, 3.0])
        np.testing.assert_allclose(result["normalized"], [0.0, 1 / 3, 1.0])
        self.assertAlmostEqual(float(result["soft"].sum()), 1.0, places=6)

    def test_bad_teacher_target_fails(self):
        with self.assertRaises(VSRTeacherError):
            teacher_targets([1.0, float("nan")])

    def test_resume_validates_record_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "a.npz"
            cache.write_bytes(b"cache")
            import hashlib
            record = {
                "sample_id": "a", "npz_name": "a.npz",
                "npz_file_sha256": hashlib.sha256(b"cache").hexdigest(),
                "manifest_sha256": MANIFEST_SHA256, "config_sha256": "c" * 64,
            }
            record["record_sha256"] = canonical_record_sha256(record)
            index = root / "index.jsonl"
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(validate_resume(index, root, {"a"}, "c" * 64), {"a"})
            record["npz_file_sha256"] = "0" * 64
            record["record_sha256"] = canonical_record_sha256(record)
            index.write_text(json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(VSRTeacherError):
                validate_resume(index, root, {"a"}, "c" * 64)


if __name__ == "__main__":
    unittest.main()
