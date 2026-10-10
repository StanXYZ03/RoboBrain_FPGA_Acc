from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from artifact_server import digest, parse_range  # noqa: E402


class ArtifactServerTests(unittest.TestCase):
    def test_digest_and_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "llama-server"
            path.write_bytes(b"0123456789")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest(path), expected)
            self.assertEqual(parse_range(None, 10), (200, 0, 9))
            self.assertEqual(parse_range("bytes=3-6", 10), (206, 3, 6))
            self.assertEqual(parse_range("bytes=7-", 10), (206, 7, 9))
            with self.assertRaises(ValueError):
                parse_range("bytes=10-", 10)
            with self.assertRaises(ValueError):
                parse_range("bytes=4-3", 10)


if __name__ == "__main__":
    unittest.main()
