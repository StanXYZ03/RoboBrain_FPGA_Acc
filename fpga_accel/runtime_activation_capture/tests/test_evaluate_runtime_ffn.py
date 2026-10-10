from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parse_capture import HEADER, MAGIC, VERSION

try:
    import numpy as np
    from evaluate_runtime_ffn import load_rows
except ModuleNotFoundError:
    np = None
    load_rows = None


@unittest.skipIf(np is None, "NumPy is optional for parser-only test environments")
class RuntimeRowsTest(unittest.TestCase):
    def write_capture(self, path: Path, name: bytes = b"ffn_inp-0") -> np.ndarray:
        values = np.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype="<f4")
        header = HEADER.pack(
            MAGIC, VERSION, 0, 1, 30, 0, 3, 2, 1, 1,
            values.nbytes, name.ljust(64, b"\0"),
        )
        path.write_bytes(header + values.tobytes())
        return values

    def test_loads_ggml_ne0_major_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            expected = self.write_capture(path)
            metadata, actual = load_rows(path)
            self.assertEqual(metadata["ne"], (3, 2, 1, 1))
            np.testing.assert_array_equal(actual, expected)

    def test_rejects_wrong_node(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            self.write_capture(path, b"ffn_norm-0")
            with self.assertRaisesRegex(ValueError, "expected pre-norm"):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
