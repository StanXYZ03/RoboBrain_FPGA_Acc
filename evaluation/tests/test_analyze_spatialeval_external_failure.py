from __future__ import annotations

import unittest

from evaluation.analyze_spatialeval_external_failure import _mean, _paired_contingency


class AnalyzeSpatialEvalExternalFailureTest(unittest.TestCase):
    def test_mean_requires_numeric_values(self) -> None:
        self.assertEqual(_mean([{"x": 1}, {"x": 3.0}], "x"), 2.0)
        with self.assertRaisesRegex(ValueError, "no numeric values"):
            _mean([{"x": None}], "x")

    def test_paired_contingency_counts_each_transition(self) -> None:
        left = {
            "a": {"task": "spatialmap", "official_compatible_correct": 1},
            "b": {"task": "spatialmap", "official_compatible_correct": 1},
            "c": {"task": "spatialmap", "official_compatible_correct": 0},
            "d": {"task": "spatialmap", "official_compatible_correct": 0},
        }
        right = {
            "a": {"task": "spatialmap", "official_compatible_correct": 1},
            "b": {"task": "spatialmap", "official_compatible_correct": 0},
            "c": {"task": "spatialmap", "official_compatible_correct": 1},
            "d": {"task": "spatialmap", "official_compatible_correct": 0},
        }
        expected = {
            "left_only_correct": 1,
            "right_only_correct": 1,
            "both_correct": 1,
            "both_wrong": 1,
        }
        result = _paired_contingency(left, right)
        self.assertEqual(result["spatialmap"], expected)
        self.assertEqual(result["all"], expected)

    def test_paired_contingency_rejects_mismatched_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "different sample sets"):
            _paired_contingency(
                {"a": {"task": "spatialmap", "official_compatible_correct": 1}},
                {"b": {"task": "spatialmap", "official_compatible_correct": 1}},
            )


if __name__ == "__main__":
    unittest.main()
