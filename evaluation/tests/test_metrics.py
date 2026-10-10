from __future__ import annotations

import unittest

from evaluation.metrics import box_iou, parse_box, parse_point, score_record, summarize


class SpatialMetricsTest(unittest.TestCase):
    def test_parse_structured_outputs(self) -> None:
        self.assertEqual(parse_point("[(123, 456)]"), (123.0, 456.0))
        self.assertEqual(parse_box("box=[10, 20, 110, 220]"), (10.0, 20.0, 110.0, 220.0))
        self.assertIsNone(parse_point("[(1001, 5)]"))
        self.assertIsNone(parse_box("[10, 20, 5, 30]"))

    def test_iou(self) -> None:
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)

    def test_point_score_uses_bbox_diagonal(self) -> None:
        sample = {
            "sample_id": "p0",
            "task": "pointing",
            "ground_truth": {"point_1000": [100, 100], "bbox_1000": [0, 0, 200, 200]},
        }
        score = score_record(sample, {"output_text": "[(110, 110)]"})
        self.assertTrue(score["parse_success"])
        self.assertTrue(score["pck_0_1"])

    def test_parse_failure_counts_as_metric_failure(self) -> None:
        point = score_record(
            {
                "sample_id": "p0",
                "task": "pointing",
                "ground_truth": {"point_1000": [100, 100], "bbox_1000": [0, 0, 200, 200]},
            },
            {"output_text": "unknown"},
        )
        grounding = score_record(
            {
                "sample_id": "g0",
                "task": "grounding",
                "ground_truth": {"bbox_1000": [0, 0, 200, 200]},
            },
            {"output_text": "unknown"},
        )
        summary = summarize([point, grounding])
        self.assertEqual(summary["tasks"]["pointing"]["pck_0_1"], 0.0)
        self.assertEqual(summary["tasks"]["grounding"]["iou_mean"], 0.0)

    def test_latency_is_propagated_and_summarized(self) -> None:
        sample = {
            "sample_id": "p0",
            "task": "pointing",
            "ground_truth": {"point_1000": [100, 100], "bbox_1000": [0, 0, 200, 200]},
        }
        records = [
            score_record(sample | {"sample_id": f"p{index}"}, {"output_text": "[100, 100]", "elapsed_seconds": value})
            for index, value in enumerate((1.0, 2.0, 8.0))
        ]
        latency = summarize(records)["tasks"]["pointing"]
        self.assertEqual(latency["latency_seconds_median"], 2.0)
        self.assertEqual(latency["latency_seconds_p95"], 8.0)


if __name__ == "__main__":
    unittest.main()
