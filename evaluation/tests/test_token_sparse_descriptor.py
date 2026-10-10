from __future__ import annotations

import unittest

from evaluation.token_sparse_descriptor import (
    HAS_POSITIONS,
    SORTED_UNIQUE,
    decode,
    encode,
)


class TokenSparseDescriptorPythonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = [
            {"original_index": 1, "position": [0, 0, 1, 0]},
            {"original_index": 4, "position": [0, 1, 1, 0]},
            {"original_index": 7, "position": [0, 2, 1, 0]},
        ]
        self.data = encode(8, self.entries, HAS_POSITIONS | SORTED_UNIQUE)

    def test_round_trip(self) -> None:
        self.assertEqual(decode(self.data)["entries"], self.entries)

    def test_payload_damage_is_rejected(self) -> None:
        damaged = bytearray(self.data)
        damaged[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "payload CRC"):
            decode(bytes(damaged))

    def test_duplicate_index_is_rejected(self) -> None:
        entries = [dict(entry) for entry in self.entries]
        entries[1] = {"original_index": 1, "position": [0, 1, 1, 0]}
        with self.assertRaises(ValueError):
            encode(8, entries, HAS_POSITIONS | SORTED_UNIQUE)


if __name__ == "__main__":
    unittest.main()
