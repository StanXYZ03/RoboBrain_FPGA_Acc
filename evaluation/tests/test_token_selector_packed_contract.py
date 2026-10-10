"""CPU-only tests for the 128-bit packed token-word contract.

The frozen int8 golden stays authoritative; these tests prove that the packed
word view (little-endian 16-lane words, 256 words per 4096 B group) is a
byte-exact re-typing of the same contiguous token stream and that word-based
context pooling equals the int8 golden.
"""

from __future__ import annotations

import unittest

import numpy as np

from evaluation import hardware_regular_selector_fixedpoint as fp
from evaluation import token_selector_packed_contract as pk


def _tokens(group_count: int) -> np.ndarray:
    group = np.arange(group_count)[:, None, None]
    patch = np.arange(fp.MERGE_UNIT)[None, :, None]
    feature = np.arange(fp.FEATURE_DIM)[None, None, :]
    return (((group * 7 + patch * 3 + feature * 13) % 256) - 128).astype(np.int8)


def _scores(group_count: int, mode: str = "rand", seed: int = 12345) -> np.ndarray:
    if mode == "zero":
        return np.zeros((group_count,), dtype=np.uint16)
    if mode == "same":
        return np.full((group_count,), 12345, dtype=np.uint16)
    base = (np.arange(group_count).astype(np.uint64) * 1103515245 + seed) % 32768
    return base.astype(np.uint16)


def _remaining_by_region(selection: fp.FixedpointSelection, regions):
    real = {
        int(index)
        for index, source in zip(selection.group_index, selection.source_type)
        if int(source) == fp.SOURCE_REAL
    }
    return {
        rid: [group for group in regions.members[rid] if group not in real]
        for rid in regions.non_empty
    }


class PackedTokenContractTest(unittest.TestCase):
    def test_pack_unpack_roundtrip_arbitrary_byte_offsets(self) -> None:
        rng = np.random.default_rng(20260927)
        values = rng.integers(-128, 128, size=67, dtype=np.int64).astype(np.int8)
        for start in range(pk.PACKED_LANES):
            words = pk.pack_int8_to_words(
                values, start_byte=start, total_bytes=start + int(values.size)
            )
            self.assertEqual(words.shape[1], pk.PACKED_LANES)
            back = pk.unpack_words_to_int8(
                words, start_byte=start, byte_count=int(values.size)
            )
            np.testing.assert_array_equal(back, values)

    def test_pack_prefix_and_tail_are_zero(self) -> None:
        values = np.arange(1, 33, dtype=np.int64).astype(np.int8)
        words = pk.pack_int8_to_words(values, start_byte=5, total_bytes=64)
        flat = words.reshape(-1)
        np.testing.assert_array_equal(flat[:5], np.zeros(5, dtype=np.int8))
        np.testing.assert_array_equal(flat[5:37], values)
        np.testing.assert_array_equal(flat[37:], np.zeros(64 - 37, dtype=np.int8))

    def test_lane_order_is_little_endian(self) -> None:
        values = np.arange(0, 48, dtype=np.int64).astype(np.int8) - 24
        words = pk.pack_int8_to_words(values)
        # Row-major flattening of the (n_words, 16) array is the byte stream.
        np.testing.assert_array_equal(words.reshape(-1)[: values.size], values)
        self.assertEqual(int(words[0, 0]), int(values[0]))
        self.assertEqual(int(words[0, 15]), int(values[15]))
        self.assertEqual(int(words[1, 0]), int(values[16]))

    def test_byte_to_word_lane_mapping(self) -> None:
        self.assertEqual(pk.byte_to_word_lane(0), (0, 0))
        self.assertEqual(pk.byte_to_word_lane(15), (0, 15))
        self.assertEqual(pk.byte_to_word_lane(16), (1, 0))
        self.assertEqual(pk.byte_to_word_lane(4096), (256, 0))
        self.assertEqual(pk.byte_to_word_lane(4096 + 4095), (511, 15))

    def test_group_word_address_matches_byte_address(self) -> None:
        for group in (0, 1, 7, 4095):
            for word in (0, 1, 255):
                self.assertEqual(
                    pk.group_word_address(group, word),
                    (group * fp.GROUP_ELEMENTS + word * pk.WORD_BYTES) // pk.WORD_BYTES,
                )
        self.assertEqual(pk.GROUP_WORDS, fp.GROUP_ELEMENTS // pk.PACKED_LANES)
        self.assertEqual(pk.GROUP_WORDS, 256)

    def test_token_segment_words_and_bytes_unchanged(self) -> None:
        for budget in (16, 32, 48, 64, 80, 96):
            summary = pk.transfer_layout_word_summary(budget)
            self.assertEqual(summary["token_bytes"], budget * fp.GROUP_ELEMENTS)
            self.assertEqual(summary["token_words"], budget * pk.GROUP_WORDS)
            self.assertEqual(summary["token_words"] * pk.WORD_BYTES, summary["token_bytes"])
            self.assertEqual(summary["token_bytes"] % 64, 0)
            self.assertEqual(summary["token_offset"] % 64, 0)
            # Same total byte count as the frozen 5-segment layout.
            self.assertEqual(
                summary["total_bytes"], fp.fpga_transfer_layout(budget).total_bytes
            )

    def test_packed_view_is_contiguous_retype(self) -> None:
        group_count = 37
        tokens = _tokens(group_count)
        view = pk.packed_tokens_view(tokens)
        self.assertEqual(view.shape, (group_count, pk.GROUP_WORDS, pk.PACKED_LANES))
        self.assertTrue(np.array_equal(view.reshape(group_count, -1), tokens.reshape(group_count, -1)))
        # Re-pack per group and compare to the view.
        for group in (0, 5, group_count - 1):
            packed = pk.pack_int8_to_words(tokens[group])
            np.testing.assert_array_equal(packed, view[group])

    def test_packed_pooling_matches_golden(self) -> None:
        for grid_h, grid_w, budget, mode in (
            (9, 9, 48, "rand"),
            (10, 12, 64, "rand"),
            (9, 9, 48, "zero"),
            (5, 4, 16, "same"),
            (2, 25, 48, "rand"),
        ):
            group_count = grid_h * grid_w
            tokens = _tokens(group_count)
            scores = _scores(group_count, mode)
            selection = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
            self.assertEqual(selection.status, fp.STATUS_OK)
            regions = fp.build_fixedpoint_regions(grid_h, grid_w)
            remaining = _remaining_by_region(selection, regions)
            for rid, members in remaining.items():
                if not members:
                    continue
                golden = fp.pool_context_tokens_q(tokens, scores, members)
                packed = pk.pool_context_tokens_q_packed(tokens, scores, members)
                np.testing.assert_array_equal(packed, golden)

    def test_packed_pooling_all_zero_weights_uniform(self) -> None:
        group_count = 81
        tokens = _tokens(group_count)
        scores = np.zeros(group_count, dtype=np.uint16)
        members = list(range(0, 81, 3))
        golden = fp.pool_context_tokens_q(tokens, scores, members)
        packed = pk.pool_context_tokens_q_packed(tokens, scores, members)
        np.testing.assert_array_equal(packed, golden)


if __name__ == "__main__":
    unittest.main()
