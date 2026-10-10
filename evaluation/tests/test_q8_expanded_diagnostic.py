import unittest

from evaluation.build_q8_expanded_diagnostic_report import target_coverage


class TargetCoverageTest(unittest.TestCase):
    def test_reports_partial_target_coverage_and_retained_center(self):
        mask = {
            "image_grid_thw": [1, 4, 4],
            "selected_group_indices": [0, 3],
        }
        result = target_coverage(mask, [0.0, 0.0, 1000.0, 1000.0])
        self.assertEqual(result["target_intersecting_cells"], 4)
        self.assertEqual(result["target_retained_cells"], 2)
        self.assertEqual(result["target_coverage"], 0.5)
        self.assertFalse(result["target_fully_retained"])
        self.assertTrue(result["target_center_cell_retained"])

    def test_boundary_touch_does_not_count_as_intersection(self):
        mask = {
            "image_grid_thw": [1, 4, 4],
            "selected_group_indices": [0],
        }
        result = target_coverage(mask, [0.0, 0.0, 500.0, 500.0])
        self.assertEqual(result["target_intersecting_cells"], 1)
        self.assertEqual(result["target_retained_cells"], 1)
        self.assertTrue(result["target_fully_retained"])


if __name__ == "__main__":
    unittest.main()
