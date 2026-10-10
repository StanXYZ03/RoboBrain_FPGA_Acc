from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from evaluation import build_spatialeval_vqa_manifest as builder
from evaluation import spatialeval_metric as metric


def png_bytes(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (9, 7), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def row(sample_id: str = "spatialmap-1", image: bytes | None = None) -> dict:
    return {
        "id": sample_id,
        "text": "Which direction?",
        "image": {"bytes": image if image is not None else png_bytes(), "path": None},
        "oracle_answer": "north",
        "oracle_option": "A",
        "oracle_full_answer": "A. north",
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ManifestTests(unittest.TestCase):
    def write_parquet(self, path: Path, rows: list[dict]) -> None:
        pq.write_table(pa.Table.from_pylist(rows), path)

    def test_builds_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row(), row("mazenav-2", png_bytes("blue"))])
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            images = root / "images"
            result = builder.build_manifest(
                parquet_path=parquet,
                output_path=manifest,
                summary_path=summary,
                image_dir=images,
            )
            self.assertEqual(result["row_count"], 2)
            self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(len(list(images.iterdir())), 2)
            self.assertTrue(summary.is_file())

    def test_duplicate_id_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row(), row()])
            paths = (root / "manifest", root / "summary", root / "images")
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=paths[0],
                    summary_path=paths[1],
                    image_dir=paths[2],
                )
            self.assertFalse(any(path.exists() for path in paths))

    def test_missing_field_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            bad = row()
            del bad["oracle_option"]
            pq.write_table(pa.Table.from_pylist([bad]), parquet)
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=root / "manifest",
                    summary_path=root / "summary",
                    image_dir=root / "images",
                )

    def test_bad_image_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row(image=b"not-an-image")])
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=root / "manifest",
                    summary_path=root / "summary",
                    image_dir=root / "images",
                )

    def test_existing_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row()])
            manifest = root / "manifest"
            manifest.write_text("keep", encoding="utf-8")
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=manifest,
                    summary_path=root / "summary",
                    image_dir=root / "images",
                )
            self.assertEqual(manifest.read_text(encoding="utf-8"), "keep")

    def test_nested_destinations_fail_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row()])
            output = root / "out" / "manifest.jsonl"
            image_dir = output / "images"
            summary = root / "summary.json"
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=output,
                    summary_path=summary,
                    image_dir=image_dir,
                )
            self.assertFalse(output.exists())
            self.assertFalse(image_dir.exists())
            self.assertFalse(summary.exists())

    def test_same_destination_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row()])
            shared = root / "manifest.jsonl"
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_path=parquet,
                    output_path=shared,
                    summary_path=shared,
                    image_dir=root / "images",
                )
            self.assertFalse(shared.exists())

    def test_success_leaves_no_staging_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row()])
            builder.build_manifest(
                parquet_path=parquet,
                output_path=root / "manifest.jsonl",
                summary_path=root / "summary.json",
                image_dir=root / "images",
            )
            leftovers = [path.name for path in root.iterdir() if path.name.startswith(".spatialeval")]
            self.assertEqual(leftovers, [])

    # --- multi-shard streaming / reference-mode contract -------------------

    def test_single_path_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "test.parquet"
            self.write_parquet(parquet, [row()])
            result = builder.build_manifest(
                parquet_paths=parquet,
                output_path=root / "manifest.jsonl",
                summary_path=root / "summary.json",
                image_dir=root / "images",
            )
            self.assertEqual(result["row_count"], 1)
            self.assertEqual(
                [shard["path"] for shard in result["source_parquets"]], [str(parquet)]
            )

    def test_two_shards_preserve_order_and_row_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "test-00000-of-00002.parquet"
            second = root / "test-00001-of-00002.parquet"
            self.write_parquet(first, [row(), row("mazenav-2", png_bytes("blue"))])
            self.write_parquet(second, [row("spatialgrid-3", png_bytes("green"))])
            manifest = root / "manifest.jsonl"
            result = builder.build_manifest(
                parquet_paths=[first, second],
                output_path=manifest,
                summary_path=root / "summary.json",
                image_dir=root / "images",
                storage_mode="extract",
                batch_size=1,
            )
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [record["sample_id"] for record in records],
                ["spatialmap-1", "mazenav-2", "spatialgrid-3"],
            )
            self.assertEqual(result["row_count"], 3)
            self.assertEqual(result["storage_mode"], "extract")
            self.assertEqual(
                [shard["row_count"] for shard in result["source_parquets"]], [2, 1]
            )
            self.assertEqual(
                [shard["path"] for shard in result["source_parquets"]],
                [str(first), str(second)],
            )
            self.assertEqual(len(list((root / "images").iterdir())), 3)
            self.assertTrue(
                all(Path(record["image_path"]).parent == root / "images" for record in records)
            )

    def test_cross_shard_duplicate_id_fails_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            second = root / "b.parquet"
            self.write_parquet(first, [row("dup-1")])
            self.write_parquet(second, [row("dup-1", png_bytes("blue"))])
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            images = root / "images"
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[first, second],
                    output_path=manifest,
                    summary_path=summary,
                    image_dir=images,
                )
            self.assertFalse(manifest.exists())
            self.assertFalse(summary.exists())
            self.assertFalse(images.exists())

    def test_missing_shard_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            self.write_parquet(first, [row()])
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[first, root / "missing.parquet"],
                    output_path=manifest,
                    summary_path=summary,
                    storage_mode="parquet_reference",
                )
            self.assertFalse(manifest.exists())
            self.assertFalse(summary.exists())

    def test_reference_mode_does_not_create_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            second = root / "b.parquet"
            self.write_parquet(first, [row(), row("mazenav-2", png_bytes("blue"))])
            self.write_parquet(second, [row("spatialgrid-3", png_bytes("green"))])
            manifest = root / "manifest.jsonl"
            images = root / "images"
            result = builder.build_manifest(
                parquet_paths=[first, second],
                output_path=manifest,
                summary_path=root / "summary.json",
                image_dir=images,  # reference mode must ignore and not create this
                storage_mode="parquet_reference",
                batch_size=1,
            )
            records = [
                json.loads(line)
                for line in manifest.read_text(encoding="utf-8").splitlines()
            ]
            self.assertFalse(images.exists())
            self.assertNotIn("image_dir", result)
            self.assertEqual(result["storage_mode"], "parquet_reference")
            expected = {
                "spatialmap-1": (0, first),
                "mazenav-2": (1, first),
                "spatialgrid-3": (0, second),
            }
            for record in records:
                self.assertNotIn("image_path", record)
                self.assertEqual(record["image_storage"], "parquet_embedded_or_local")
                expected_index, shard_path = expected[record["sample_id"]]
                self.assertEqual(record["source_parquet_path"], str(shard_path))
                self.assertEqual(record["source_parquet_row_index"], expected_index)
                self.assertEqual(record["source_parquet_sha256"], file_sha256(shard_path))
                self.assertGreater(record["width"], 0)
                self.assertGreater(record["height"], 0)
                self.assertEqual(len(record["image_sha256"]), 64)

    def test_reference_mode_still_validates_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "a.parquet"
            self.write_parquet(parquet, [row(image=b"not-an-image")])
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[parquet],
                    output_path=manifest,
                    summary_path=summary,
                    storage_mode="parquet_reference",
                )
            self.assertFalse(manifest.exists())
            self.assertFalse(summary.exists())

    def test_reference_mode_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "a.parquet"
            self.write_parquet(parquet, [row()])
            manifest = root / "manifest.jsonl"
            manifest.write_text("keep", encoding="utf-8")
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[parquet],
                    output_path=manifest,
                    summary_path=root / "summary.json",
                    storage_mode="parquet_reference",
                )
            self.assertEqual(manifest.read_text(encoding="utf-8"), "keep")
            self.assertFalse((root / "summary.json").exists())

    def test_summary_records_per_shard_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            second = root / "b.parquet"
            self.write_parquet(first, [row(), row("mazenav-2", png_bytes("blue"))])
            self.write_parquet(second, [row("spatialgrid-3")])
            summary = root / "summary.json"
            result = builder.build_manifest(
                parquet_paths=[first, second],
                output_path=root / "manifest.jsonl",
                summary_path=summary,
                storage_mode="parquet_reference",
                batch_size=1,
            )
            on_disk = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["storage_mode"], "parquet_reference")
            self.assertEqual(on_disk["row_count"], 3)
            self.assertEqual(on_disk["unique_id_count"], 3)
            self.assertEqual(on_disk, result)
            shards = on_disk["source_parquets"]
            self.assertEqual(len(shards), 2)
            for shard, path, expected_rows in ((shards[0], first, 2), (shards[1], second, 1)):
                self.assertEqual(
                    sorted(shard), ["bytes", "path", "row_count", "sha256"]
                )
                self.assertEqual(shard["path"], str(path))
                self.assertEqual(shard["sha256"], file_sha256(path))
                self.assertEqual(shard["bytes"], path.stat().st_size)
                self.assertEqual(shard["row_count"], expected_rows)

    def test_extract_requires_image_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "a.parquet"
            self.write_parquet(parquet, [row()])
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[parquet],
                    output_path=root / "manifest.jsonl",
                    summary_path=root / "summary.json",
                    storage_mode="extract",
                )

    def test_unknown_storage_mode_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parquet = root / "a.parquet"
            self.write_parquet(parquet, [row()])
            with self.assertRaises(builder.SpatialEvalManifestError):
                builder.build_manifest(
                    parquet_paths=[parquet],
                    output_path=root / "manifest.jsonl",
                    summary_path=root / "summary.json",
                    storage_mode="copy",
                )

    def test_streaming_path_never_calls_read_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            second = root / "b.parquet"
            self.write_parquet(first, [row(), row("mazenav-2", png_bytes("blue"))])
            self.write_parquet(second, [row("spatialgrid-3")])
            with mock.patch.object(
                pq,
                "read_table",
                side_effect=AssertionError("build_manifest must stream via iter_batches"),
            ):
                result = builder.build_manifest(
                    parquet_paths=[first, second],
                    output_path=root / "manifest.jsonl",
                    summary_path=root / "summary.json",
                    storage_mode="parquet_reference",
                    batch_size=2,
                )
            self.assertEqual(result["row_count"], 3)

    def test_builder_source_has_no_read_table(self) -> None:
        source = Path(builder.__file__).read_text(encoding="utf-8")
        self.assertNotIn("read_table", source)

    def test_cli_accepts_repeated_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.parquet"
            second = root / "b.parquet"
            self.write_parquet(first, [row()])
            self.write_parquet(second, [row("mazenav-2", png_bytes("blue"))])
            manifest = root / "manifest.jsonl"
            summary = root / "summary.json"
            code = builder.main(
                [
                    "--parquet",
                    str(first),
                    "--parquet",
                    str(second),
                    "--output",
                    str(manifest),
                    "--summary",
                    str(summary),
                    "--storage-mode",
                    "parquet_reference",
                ]
            )
            self.assertEqual(code, 0)
            self.assertEqual(len(manifest.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                json.loads(summary.read_text(encoding="utf-8"))["row_count"], 2
            )


class MetricTests(unittest.TestCase):
    def test_exact_match_and_grouping(self) -> None:
        result = metric.score_records(
            [
                {"sample_id": "spatialmap-1", "output_text": "Answer: A.", "expected_answer": "a"},
                {"sample_id": "mazenav-2", "output_text": "No", "expected_answer": "yes"},
            ]
        )
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(set(result["per_task"]), {"spatialmap", "mazenav"})

    def test_parse_failure_counts_wrong(self) -> None:
        result = metric.score_records(
            [{"sample_id": "spatialreal-1", "output_text": "line one\nline two", "expected_answer": "a"}]
        )
        self.assertEqual(result["accuracy"], 0.0)
        self.assertEqual(result["parse_failure_count"], 1)

    def test_duplicate_and_unknown_task_fail(self) -> None:
        with self.assertRaises(metric.SpatialEvalMetricError):
            metric.score_records(
                [
                    {"sample_id": "spatialgrid-1", "output_text": "a", "expected_answer": "a"},
                    {"sample_id": "spatialgrid-1", "output_text": "a", "expected_answer": "a"},
                ]
            )
        with self.assertRaises(metric.SpatialEvalMetricError):
            metric.score_records(
                [{"sample_id": "other-1", "output_text": "a", "expected_answer": "a"}]
            )


if __name__ == "__main__":
    unittest.main()
