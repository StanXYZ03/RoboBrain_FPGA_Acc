import math
import struct
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_capture import HEADER, MAGIC, parse_capture


class CaptureParserTest(unittest.TestCase):
    def make_capture(self, directory: Path, *, magic=MAGIC, payload=(1.0, -2.0, 3.5, 0.25)) -> Path:
        path = directory / "capture.bin"
        header = HEADER.pack(
            magic, 1, 0, 3, 30, 0, 2, 2, 1, 1,
            len(payload) * 4, b"ffn_inp-0\0",
        )
        path.write_bytes(header + struct.pack(f"<{len(payload)}f", *payload))
        return path

    def test_valid_capture(self):
        with tempfile.TemporaryDirectory() as raw:
            result = parse_capture(self.make_capture(Path(raw)))
            self.assertEqual(result["tensor_name"], "ffn_inp-0")
            self.assertEqual(tuple(result["ne"]), (2, 2, 1, 1))
            self.assertTrue(result["all_finite"])
            self.assertEqual(result["minimum"], -2.0)
            self.assertEqual(result["maximum"], 3.5)

    def test_wrong_magic_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaisesRegex(ValueError, "identity"):
                parse_capture(self.make_capture(Path(raw), magic=b"BADMAGIC"))

    def test_nonfinite_reported(self):
        with tempfile.TemporaryDirectory() as raw:
            result = parse_capture(self.make_capture(Path(raw), payload=(1.0, math.inf, 0.0, 2.0)))
            self.assertFalse(result["all_finite"])

    def test_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            path = self.make_capture(Path(raw))
            with path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "file size"):
                parse_capture(path)


if __name__ == "__main__":
    unittest.main()
