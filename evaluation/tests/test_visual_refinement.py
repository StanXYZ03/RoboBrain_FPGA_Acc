from __future__ import annotations

import unittest

from PIL import Image

from evaluation.visual_refinement import (
    crop_image_exact_1000,
    grid_token_counts,
    make_global_roi_messages,
    make_labeled_global_roi_messages,
    make_roi_only_messages,
    resize_long_edge,
)


class VisualRefinementTest(unittest.TestCase):
    def test_resize_does_not_upscale_by_default(self) -> None:
        image = Image.new("RGB", (100, 50))
        self.assertEqual(resize_long_edge(image, 200).size, (100, 50))
        self.assertEqual(resize_long_edge(image, 50).size, (50, 25))

    def test_resize_can_magnify_roi_for_encoder(self) -> None:
        image = Image.new("RGB", (100, 50))
        self.assertEqual(resize_long_edge(image, 200, allow_upscale=True).size, (200, 100))

    def test_crop_reports_effective_pixel_mapping(self) -> None:
        image = Image.new("RGB", (101, 203))
        crop, effective, pixels = crop_image_exact_1000(image, (100, 200, 600, 800))
        self.assertEqual(crop.size, (51, 122))
        self.assertEqual(pixels, (10, 40, 61, 162))
        self.assertAlmostEqual(effective[0], 1000 * 10 / 101)
        self.assertAlmostEqual(effective[3], 1000 * 162 / 203)

    def test_grid_tokens_match_merged_contract(self) -> None:
        counts = grid_token_counts([[1, 12, 18], [1, 14, 20]], merge_size=2)
        self.assertEqual(counts["encoder_tokens_total"], 496)
        self.assertEqual(counts["llm_visual_tokens_total"], 124)
        self.assertEqual(counts["llm_visual_tokens_per_image"], [54, 70])

    def test_prompt_has_two_image_items_and_roi_coordinate_contract(self) -> None:
        messages = make_global_roi_messages("Locate the cup.")
        image_items = [item for item in messages[0]["content"] if item["type"] == "image"]
        text = "".join(
            item["text"] for item in messages[0]["content"] if item["type"] == "text"
        )
        self.assertEqual(len(image_items), 2)
        self.assertIn("relative to Image 2", text)

    def test_roi_only_prompt_has_one_image(self) -> None:
        messages = make_roi_only_messages("Locate the cup.")
        image_items = [item for item in messages[0]["content"] if item["type"] == "image"]
        self.assertEqual(len(image_items), 1)

    def test_labeled_prompt_tracks_roi_order(self) -> None:
        messages = make_labeled_global_roi_messages("Locate the cup.", roi_first=True)
        image_items = [item for item in messages[0]["content"] if item["type"] == "image"]
        text = "".join(
            item["text"] for item in messages[0]["content"] if item["type"] == "text"
        )
        self.assertEqual(len(image_items), 2)
        self.assertIn("Picture 1 is an enlarged crop", text)
        self.assertIn("relative to Picture 1", text)


if __name__ == "__main__":
    unittest.main()
