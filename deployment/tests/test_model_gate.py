from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from gguf_metadata import GGUFError, read_metadata  # noqa: E402
from model_gate import validate_file  # noqa: E402


def string(value: str) -> bytes:
    raw = value.encode()
    return struct.pack("<Q", len(raw)) + raw


def kv(key: str, value_type: int, value: bytes) -> bytes:
    return string(key) + struct.pack("<I", value_type) + value


def tiny_gguf(path: Path, architecture: str = "qwen3vl") -> None:
    entries = [
        kv("general.architecture", 8, string(architecture)),
        kv("general.name", 8, string("RoboBrain2.5-4B test")),
        kv("general.file_type", 4, struct.pack("<I", 7)),
        kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 2) + string("a") + string("b")),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(entries)) + b"".join(entries))


class GGUFMetadataTests(unittest.TestCase):
    def test_reads_scalars_and_skips_token_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            tiny_gguf(path)
            metadata = read_metadata(path)
            self.assertEqual(metadata["general.architecture"], "qwen3vl")
            self.assertEqual(metadata["general.file_type"], 7)
            self.assertNotIn("tokenizer.ggml.tokens", metadata)

    def test_rejects_non_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.gguf"
            path.write_bytes(b"nope")
            with self.assertRaises(GGUFError):
                read_metadata(path)

    def test_policy_rejects_wrong_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.gguf"
            tiny_gguf(path, "wrong")
            spec = {
                "role": "model", "name": path.name,
                "min_bytes": 1, "max_bytes": 10000,
                "architecture": "qwen3vl", "file_type": 7,
            }
            with self.assertRaises(RuntimeError):
                validate_file(path, spec)


if __name__ == "__main__":
    unittest.main()
