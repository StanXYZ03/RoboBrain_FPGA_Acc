from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from evaluation import spatialeval_external_contract as contract


def image_bytes() -> bytes:
    stream = BytesIO()
    Image.new("RGB", (9, 7), color="red").save(stream, format="PNG")
    return stream.getvalue()


class SpatialEvalExternalContractTests(unittest.TestCase):
    def test_selector_projection_removes_all_oracles(self) -> None:
        sample = {
            "sample_id": "spatialmap.vqa.0.0",
            "task": "spatialmap",
            "image": "/tmp/a.png",
            "prompt": "question",
            "max_new_tokens": 64,
            "oracle_answer": "northwest",
            "oracle_option": "A",
            "oracle_full_answer": "A. northwest",
            "answer": "northwest",
        }
        projected = contract.sanitize_sample_for_selector(sample)
        self.assertEqual(
            set(projected),
            {"sample_id", "task", "image", "prompt", "max_new_tokens"},
        )
        self.assertTrue(set(projected).isdisjoint(contract.FORBIDDEN_SELECTOR_KEYS))

    def test_reference_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                contract.SpatialEvalExternalContractError, "SHA256"
            ):
                contract.load_reference_manifest(
                    path, expected_sha256="0" * 64, expected_total=1
                )

    def test_output_path_is_pinned(self) -> None:
        with self.assertRaises(contract.SpatialEvalExternalContractError):
            contract.ensure_output_path(contract.ARM_DENSE, Path("/tmp/wrong"))

    def test_materialization_detects_image_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            raw = image_bytes()
            row = {
                "id": "spatialmap.vqa.0.0",
                "text": "question",
                "image": {"bytes": raw, "path": None},
                "oracle_answer": "northwest",
                "oracle_option": "A",
                "oracle_full_answer": "A. northwest",
            }
            pq.write_table(pa.Table.from_pylist([row]), parquet)
            shard_sha = contract.sha256_file(parquet)
            reference = {
                "sample_id": row["id"],
                "source_parquet_path": str(parquet),
                "source_parquet_sha256": shard_sha,
                "source_parquet_row_index": 0,
                "image_sha256": "0" * 64,
                "width": 9,
                "height": 7,
                "text": "question",
                "oracle_answer": "northwest",
                "oracle_option": "A",
                "oracle_full_answer": "A. northwest",
            }
            shard = {
                "path": parquet,
                "sha256": shard_sha,
                "bytes": parquet.stat().st_size,
                "rows": 1,
            }
            with self.assertRaisesRegex(
                contract.SpatialEvalExternalContractError, "image SHA256"
            ):
                contract.materialize_primary_samples(
                    [reference],
                    image_cache=root / "images",
                    shards=[shard],
                    verify_shards=True,
                )

    def test_existing_cache_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "image.png"
            path.write_bytes(b"wrong")
            with self.assertRaises(contract.SpatialEvalExternalContractError):
                contract._atomic_write_new_or_equal(path, b"right")

    def test_constants_are_frozen(self) -> None:
        self.assertEqual(contract.PRIMARY_SAMPLE_COUNT, 4500)
        self.assertEqual(contract.MAX_NEW_TOKENS, 64)
        self.assertEqual((contract.MIN_PIXELS, contract.MAX_PIXELS), (65536, 589824))
        self.assertEqual(
            (contract.K_BUDGET, contract.REGION_GRID, contract.LOCAL_CANDIDATE_CAP),
            (96, 3, 16),
        )

    def test_preregistered_pin_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pinned = root / "input.txt"
            pinned.write_text("actual", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "status": "PREREGISTERED_NOT_EXECUTED",
                        "pins": {
                            "input": {
                                "path": str(pinned),
                                "sha256": hashlib.sha256(b"different").hexdigest(),
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                contract.SpatialEvalExternalContractError, "SHA256"
            ):
                contract.verify_preregistered_pins(config)


if __name__ == "__main__":
    unittest.main()
