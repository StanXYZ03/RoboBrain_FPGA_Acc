from __future__ import annotations

import unittest

from evaluation.build_visual_refinement_splits import (
    assign_splits,
    intersection_fraction,
    proportional_sample,
)


class VisualRefinementSplitBuilderTest(unittest.TestCase):
    def test_intersection_fraction(self) -> None:
        self.assertAlmostEqual(intersection_fraction([0, 0, 10, 10], [5, 0, 10, 10]), 0.5)
        self.assertEqual(intersection_fraction([0, 0, 10, 10], [20, 20, 2, 2]), 0.0)

    def test_sampling_and_split_sizes_are_deterministic(self) -> None:
        candidates = []
        for index in range(20):
            candidates.append(
                {
                    "id": index,
                    "area_bin": "small" if index < 10 else "large",
                    "edge_bin": "edge" if index % 2 else "interior",
                    "overlap_bin": "clear",
                    "complexity_bin": "sparse",
                }
            )
        selected_a = proportional_sample(candidates, 10, 7)
        selected_b = proportional_sample(candidates, 10, 7)
        self.assertEqual([item["id"] for item in selected_a], [item["id"] for item in selected_b])
        splits = assign_splits(selected_a, {"development": 3, "calibration": 2, "test": 5}, 7)
        self.assertEqual({name: len(items) for name, items in splits.items()}, {"development": 3, "calibration": 2, "test": 5})


if __name__ == "__main__":
    unittest.main()
