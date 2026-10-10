import unittest

import numpy as np

from evaluation.evaluate_task_conditioned_roi_proposals import (
    box_metrics,
    heatmap_center_1000,
    qwen_visual_tokens,
    ridge_apply,
    ridge_fit,
    shifted_box,
    smooth_heatmap,
)


class TaskConditionedRoiProposalTest(unittest.TestCase):
    def test_shifted_box_preserves_extent_at_edges(self):
        self.assertEqual(shifted_box(50, 950, 600, 400), [0.0, 600.0, 600.0, 1000.0])

    def test_smoothing_preserves_shape_and_constant(self):
        values = np.full((3, 5), 7.0)
        smoothed = smooth_heatmap(values)
        self.assertEqual(smoothed.shape, values.shape)
        self.assertTrue(np.allclose(smoothed, values))

    def test_heatmap_center_uses_cell_coordinates(self):
        values = np.zeros((3, 4))
        values[1, 2] = 10.0
        x, y = heatmap_center_1000(values)
        self.assertAlmostEqual(x, 625.0)
        self.assertAlmostEqual(y, 500.0)

    def test_box_metrics_reports_center_coverage_and_area(self):
        metrics = box_metrics([0, 0, 500, 500], [250, 250, 750, 750])
        self.assertTrue(metrics["target_center_recalled"])
        self.assertAlmostEqual(metrics["target_box_coverage"], 0.25)
        self.assertAlmostEqual(metrics["proposal_area_fraction"], 0.25)

    def test_ridge_recovers_simple_linear_relation(self):
        x = np.arange(10, dtype=np.float64)[:, None]
        y = 2.0 * x[:, 0] + 3.0
        model = ridge_fit(x, y, alpha=1e-9)
        prediction = ridge_apply(model, np.asarray([[4.5]]))[0]
        self.assertAlmostEqual(prediction, 12.0, places=6)

    def test_qwen_tokens_are_merge_aligned(self):
        encoder, llm = qwen_visual_tokens(256, 192, min_pixels=1024)
        self.assertEqual(encoder, 192)
        self.assertEqual(llm, 48)


if __name__ == "__main__":
    unittest.main()
