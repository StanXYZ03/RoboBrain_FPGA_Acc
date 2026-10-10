#!/usr/bin/env python3
"""CPU-only focused tests for scripts/build_model_identity_manifest.py.

Covers determinism, streaming hash, index-shard coverage, missing-shard
rejection and the real generated manifest's internal consistency. All fixture
paths are temporary; no existing artifact or report is rewritten.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_model_identity_manifest as builder  # noqa: E402

REAL_MANIFEST = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
HEX = set("0123456789abcdef")


def _write_fixture(root: Path) -> None:
    (root / "model-00001-of-00002.safetensors").write_bytes(b"A" * 4096)
    (root / "model-00002-of-00002.safetensors").write_bytes(b"B" * 8192)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12288},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("{{ x }}", encoding="utf-8")


class FixtureManifestTest(unittest.TestCase):
    def test_deterministic_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(root)
            first = builder.build_model_identity_manifest(root)
            second = builder.build_model_identity_manifest(root)
            self.assertEqual(first, second)
            self.assertEqual(
                builder.dumps_manifest(first), builder.dumps_manifest(second)
            )
            rels = [entry["relative_path"] for entry in first["files"]]
            self.assertEqual(rels, sorted(rels))
            self.assertEqual(len(rels), len(set(rels)))

    def test_index_shard_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(root)
            manifest = builder.build_model_identity_manifest(root)
            self.assertEqual(
                manifest["weight_shards"],
                [
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            )
            self.assertEqual(manifest["index_files"], ["model.safetensors.index.json"])
            self.assertEqual(
                manifest["referenced_shards"]["model.safetensors.index.json"],
                [
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
            )
            roles = {entry["relative_path"]: entry["role"] for entry in manifest["files"]}
            self.assertEqual(roles["model.safetensors.index.json"], "index")
            self.assertEqual(roles["model-00001-of-00002.safetensors"], "weight_shard")
            self.assertEqual(roles["chat_template.jinja"], "chat_template")

    def test_index_reference_in_subdirectory_rejected_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model-00001-of-00002.safetensors").write_bytes(b"x")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "shards/model-00001.safetensors"}}),
                encoding="utf-8",
            )
            with self.assertRaises(builder.ModelIdentityError):
                builder.build_model_identity_manifest(root)

    def test_missing_referenced_shard_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(root)
            (root / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaises(builder.ModelIdentityError):
                builder.build_model_identity_manifest(root)

    def test_no_shards_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(builder.ModelIdentityError):
                builder.build_model_identity_manifest(root)

    def test_missing_model_dir_rejected(self) -> None:
        with self.assertRaises(builder.ModelIdentityError):
            builder.build_model_identity_manifest("/nonexistent/model/dir")

    def test_invalid_index_json_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "model-00001.safetensors").write_bytes(b"x")
            (root / "model.safetensors.index.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(builder.ModelIdentityError):
                builder.build_model_identity_manifest(root)

    def test_sha256_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(root)
            manifest = builder.build_model_identity_manifest(root)
            by_rel = {entry["relative_path"]: entry for entry in manifest["files"]}
            payload = (root / "model-00001-of-00002.safetensors").read_bytes()
            self.assertEqual(
                by_rel["model-00001-of-00002.safetensors"]["sha256"],
                hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(
                by_rel["model-00001-of-00002.safetensors"]["bytes"], len(payload)
            )

    def test_check_cli_detects_match_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fixture(root)
            out = root / "identity.json"
            self.assertEqual(
                builder.main(["--model", str(root), "--output", str(out)]), 0
            )
            self.assertEqual(
                builder.main(
                    ["--model", str(root), "--output", str(out), "--check"]
                ),
                0,
            )
            out.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(builder.ModelIdentityError):
                builder.main(["--model", str(root), "--output", str(out), "--check"])


class RealManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))

    def test_internal_consistency(self) -> None:
        manifest = self.manifest
        rels = [entry["relative_path"] for entry in manifest["files"]]
        self.assertEqual(rels, sorted(rels))
        self.assertEqual(len(rels), len(set(rels)))
        self.assertEqual(manifest["file_count"], len(manifest["files"]))
        self.assertEqual(
            manifest["total_bytes"], sum(entry["bytes"] for entry in manifest["files"])
        )
        for entry in manifest["files"]:
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue(set(entry["sha256"]) <= HEX)
            self.assertGreater(entry["bytes"], 0)

    def test_weight_shard_recorded(self) -> None:
        manifest = self.manifest
        self.assertIn("model.safetensors", manifest["weight_shards"])
        shard = next(
            entry
            for entry in manifest["files"]
            if entry["relative_path"] == "model.safetensors"
        )
        self.assertEqual(shard["role"], "weight_shard")
        self.assertEqual(
            shard["sha256"],
            "e2f6e3c372f4ef76afb5e9e28ec56679944fcc3b4f2ce2ebda8b21264e96eb2f",
        )

    def test_identity_files_recorded(self) -> None:
        rels = {entry["relative_path"] for entry in self.manifest["files"]}
        for required in (
            "config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.json",
        ):
            self.assertIn(required, rels)


if __name__ == "__main__":
    unittest.main()
