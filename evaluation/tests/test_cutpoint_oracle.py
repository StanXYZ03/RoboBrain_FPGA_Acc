from __future__ import annotations

import unittest

from evaluation.cutpoint_oracle import cut_point, transfer_accounting


class CutPointOracleTest(unittest.TestCase):
    def test_cut_point_indices_are_unambiguous(self) -> None:
        self.assertEqual(cut_point("patch_embedding").prune_after_layer, -1)
        self.assertEqual(cut_point("vision_block_3").completed_vision_blocks, 3)
        self.assertEqual(cut_point("deepstack_layer_5").prune_after_layer, 5)
        self.assertEqual(cut_point("post_merger").completed_vision_blocks, 24)

    def test_round_trip_is_larger_than_one_way(self) -> None:
        result = transfer_accounting(
            cut_name="vision_block_1", full_patch_tokens=1024, kept_groups=128
        )
        self.assertGreater(result["round_trip_bytes"], result["one_way_bytes"])
        self.assertEqual(result["compact_cut_tensor_bytes"], 128 * 4 * 1024 * 2)

    def test_post_merger_uses_llm_hidden_width(self) -> None:
        result = transfer_accounting(
            cut_name="post_merger", full_patch_tokens=1024, kept_groups=64
        )
        self.assertEqual(result["compact_cut_tensor_bytes"], 64 * 2560 * 2)
        self.assertLess(
            result["one_way_serialization_seconds"]["10GbE"],
            result["round_trip_serialization_seconds"]["10GbE"],
        )


if __name__ == "__main__":
    unittest.main()

