from __future__ import annotations

import unittest
from pathlib import Path

from PIL import Image

from evaluation.run_hf_global_roi import choose_samples, prepare_strategy


class GlobalRoiRunnerTest(unittest.TestCase):
    def test_sampling_is_paired_deterministic_and_not_prefix_based(self) -> None:
        records = []
        for image_id in range(12):
            area = "small" if image_id < 6 else "large"
            edge = "edge" if image_id % 2 else "interior"
            for task in ("grounding", "pointing"):
                records.append(
                    {
                        "coco_image_id": image_id,
                        "sample_id": f"{image_id}-{task}",
                        "task": task,
                        "stratification": {
                            "target_area_bin": area,
                            "edge_bin": edge,
                            "overlap_bin": "clear",
                            "scene_complexity_bin": "sparse",
                        },
                    }
                )
        first = choose_samples(records, 6, 123)
        second = choose_samples(records, 6, 123)
        self.assertEqual(
            [record["sample_id"] for record in first],
            [record["sample_id"] for record in second],
        )
        selected_ids = {record["coco_image_id"] for record in first}
        self.assertEqual(len(selected_ids), 6)
        self.assertNotEqual(selected_ids, set(range(6)))
        self.assertEqual(sum(record["task"] == "grounding" for record in first), 6)
        self.assertEqual(sum(record["task"] == "pointing" for record in first), 6)

    def test_strategy_preparation_does_not_do_unrelated_image_work(self) -> None:
        sample = {
            "prompt": "Locate the cup.",
            "ground_truth": {"bbox_1000": [300, 300, 600, 600]},
        }
        image = Image.new("RGB", (640, 480))
        common = {
            "sample": sample,
            "image_path": Path("unused.jpg"),
            "original": image,
            "global_long_edge": 256,
            "roi_long_edge": 384,
            "oracle_margin": 0.5,
            "roi_upscale": True,
        }
        dense = prepare_strategy(strategy="dense", **common)
        global_only = prepare_strategy(strategy="global_only", **common)
        roi_only = prepare_strategy(strategy="roi_only", **common)
        self.assertIsNone(dense["global_image"])
        self.assertIsNone(dense["roi"])
        self.assertIsNotNone(global_only["global_image"])
        self.assertIsNone(global_only["roi"])
        self.assertIsNone(roi_only["global_image"])
        self.assertIsNotNone(roi_only["roi"])
        self.assertEqual(len(roi_only["images"]), 1)


if __name__ == "__main__":
    unittest.main()
