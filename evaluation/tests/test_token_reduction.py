from __future__ import annotations

import unittest

from PIL import Image

from evaluation.token_reduction import (
    centered_square_1000,
    crop_image_1000,
    expanded_bbox_1000,
    qwen3_vl_compute_proxy,
    qwen3_vl_shallow_pruned_compute_proxy,
    remap_output_text,
    retention_label,
    scout_crop_box_1000,
)


class TokenReductionTest(unittest.TestCase):
    def test_crop_and_coordinate_remap(self) -> None:
        image = Image.new("RGB", (100, 200))
        crop_box = (100, 200, 600, 800)
        self.assertEqual(crop_image_1000(image, crop_box).size, (50, 120))
        self.assertEqual(
            remap_output_text("pointing", "[(500, 500)]", crop_box),
            "[(350.000000, 500.000000)]",
        )
        self.assertEqual(
            remap_output_text("grounding", "[0, 0, 1000, 1000]", crop_box),
            "[100.000000, 200.000000, 600.000000, 800.000000]",
        )

    def test_expansion_is_clamped(self) -> None:
        self.assertEqual(expanded_bbox_1000((0, 100, 200, 900), 0.5), (0.0, 0.0, 300.0, 1000.0))

    def test_compute_proxy_is_monotonic(self) -> None:
        full = qwen3_vl_compute_proxy(1024)
        half = qwen3_vl_compute_proxy(512)
        self.assertEqual(full.llm_visual_tokens, 256)
        self.assertEqual(half.llm_visual_tokens, 128)
        self.assertLess(half.encoder_macs, full.encoder_macs)
        self.assertLess(half.llm_prefill_macs, full.llm_prefill_macs)

    def test_shallow_pruned_proxy_accounts_for_full_prefix(self) -> None:
        full = qwen3_vl_compute_proxy(1024)
        same = qwen3_vl_shallow_pruned_compute_proxy(1024, 256)
        quarter = qwen3_vl_shallow_pruned_compute_proxy(1024, 64)
        uniform_quarter = qwen3_vl_compute_proxy(256)
        self.assertEqual(same, full)
        self.assertGreater(quarter.encoder_macs, uniform_quarter.encoder_macs)
        self.assertLess(quarter.encoder_macs, full.encoder_macs)
        self.assertEqual(quarter.llm_prefill_macs, uniform_quarter.llm_prefill_macs)

    def test_patch_embedding_cut_has_no_full_vision_blocks(self) -> None:
        patch_cut = qwen3_vl_shallow_pruned_compute_proxy(
            1024, 64, prune_after_layer=-1
        )
        uniform_quarter = qwen3_vl_compute_proxy(256)
        self.assertEqual(patch_cut.encoder_macs, uniform_quarter.encoder_macs)
        self.assertEqual(patch_cut.llm_prefill_macs, uniform_quarter.llm_prefill_macs)

    def test_centered_square_preserves_size_at_edges(self) -> None:
        self.assertEqual(centered_square_1000((950, 50), 250), (500.0, 0.0, 1000.0, 500.0))

    def test_scout_crop_uses_prediction_and_handles_parse_failure(self) -> None:
        self.assertEqual(
            scout_crop_box_1000(
                "pointing", "[(500, 600)]", grounding_margin=0.5, point_half_extent=250
            ),
            (250.0, 350.0, 750.0, 850.0),
        )
        self.assertEqual(
            scout_crop_box_1000(
                "grounding", "[200, 300, 400, 500]", grounding_margin=0.5, point_half_extent=250
            ),
            (100.0, 200.0, 500.0, 600.0),
        )
        self.assertIsNone(
            scout_crop_box_1000(
                "grounding", "not a box", grounding_margin=0.5, point_half_extent=250
            )
        )

    def test_retention_label(self) -> None:
        self.assertEqual(retention_label(1.0), "r1000")
        self.assertEqual(retention_label(0.125), "r0125")


if __name__ == "__main__":
    unittest.main()
