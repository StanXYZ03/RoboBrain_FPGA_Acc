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

from evaluation import spatialeval_vqa_contract as contract


def png_bytes(color: str = "red", size: tuple[int, int] = (9, 7)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color=color).save(stream, format="PNG")
    return stream.getvalue()


def parquet_row(sample_id: str, image: bytes) -> dict:
    return {
        "id": sample_id,
        "text": f"Question for {sample_id}? Available options:\nA. north\nB. south",
        "image": {"bytes": image, "path": None},
        "oracle_answer": "north",
        "oracle_option": "A",
        "oracle_full_answer": "A. north",
    }


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ContractFixture(unittest.TestCase):
    """Two shards, six handler samples, two shared images (dedup)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.image_dir = self.root / "images"
        self.image_a = png_bytes("red", (9, 7))
        self.image_b = png_bytes("blue", (11, 13))
        self.image_c = png_bytes("green", (5, 5))
        shard0 = self.root / "test-00000-of-00002.parquet"
        shard1 = self.root / "test-00001-of-00002.parquet"
        # spatialmap x2 (shared image A), spatialgrid x1 (image B)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    parquet_row("spatialmap.vqa.2000.0", self.image_a),
                    parquet_row("spatialmap.vqa.2001.0", self.image_a),
                    parquet_row("spatialgrid.vqa.3000.1", self.image_b),
                ]
            ),
            shard0,
        )
        # mazenav x3 (image A, C, B)
        pq.write_table(
            pa.Table.from_pylist(
                [
                    parquet_row("mazenav.vqa.1000.2", self.image_a),
                    parquet_row("mazenav.vqa.1001.0", self.image_c),
                    parquet_row("mazenav.vqa.1002.1", self.image_b),
                ]
            ),
            shard1,
        )
        self.parquet_pins = (
            {
                "path": str(shard0),
                "sha256": file_sha(shard0),
                "bytes": shard0.stat().st_size,
                "row_count": 3,
            },
            {
                "path": str(shard1),
                "sha256": file_sha(shard1),
                "bytes": shard1.stat().st_size,
                "row_count": 3,
            },
        )
        self.task_counts = {"spatialmap": 2, "mazenav": 3, "spatialgrid": 1}
        self.selected = self._build_selected()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _reference(self, sample_id: str, image: bytes, parquet_path: Path, row_index: int) -> dict:
        with Image.open(BytesIO(image)) as opened:
            width, height = opened.size
        return {
            "sample_id": sample_id,
            "text": f"Question for {sample_id}? Available options:\nA. north\nB. south",
            "oracle_answer": "north",
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "width": width,
            "height": height,
            "source_parquet_path": str(parquet_path),
            "source_parquet_sha256": file_sha(parquet_path),
            "source_parquet_row_index": row_index,
        }

    def _build_selected(self) -> list[dict]:
        shard0 = Path(self.parquet_pins[0]["path"])
        shard1 = Path(self.parquet_pins[1]["path"])
        return [
            self._reference("spatialmap.vqa.2000.0", self.image_a, shard0, 0),
            self._reference("spatialmap.vqa.2001.0", self.image_a, shard0, 1),
            self._reference("spatialgrid.vqa.3000.1", self.image_b, shard0, 2),
            self._reference("mazenav.vqa.1000.2", self.image_a, shard1, 0),
            self._reference("mazenav.vqa.1001.0", self.image_c, shard1, 1),
            self._reference("mazenav.vqa.1002.1", self.image_b, shard1, 2),
        ]

    def write_run_manifest(self) -> Path:
        contract.materialize_images(
            self.selected, parquet_pins=self.parquet_pins, image_dir=self.image_dir
        )
        output = self.root / "run.jsonl"
        contract.write_run_manifest(
            self.selected,
            output_path=output,
            image_dir=self.image_dir,
            expected_count=6,
            task_sample_counts=self.task_counts,
        )
        return output


class MaterializationTests(ContractFixture):
    def test_materialization_deduplicates_and_verifies(self) -> None:
        summary = contract.materialize_images(
            self.selected, parquet_pins=self.parquet_pins, image_dir=self.image_dir
        )
        self.assertEqual(summary["selected_rows"], 6)
        self.assertEqual(summary["unique_images"], 3)
        files = sorted(path.name for path in self.image_dir.iterdir())
        self.assertEqual(len(files), 3)
        for name in files:
            self.assertTrue(name.endswith(".png"))
            self.assertEqual(
                hashlib.sha256((self.image_dir / name).read_bytes()).hexdigest(),
                name[:-4],
            )

    def test_parquet_sha_drift_fails_closed(self) -> None:
        broken = (
            {**self.parquet_pins[0], "sha256": "0" * 64},
            self.parquet_pins[1],
        )
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "SHA256"):
            contract.materialize_images(
                self.selected, parquet_pins=broken, image_dir=self.image_dir
            )

    def test_parquet_byte_drift_fails_closed(self) -> None:
        broken = (
            {**self.parquet_pins[0], "bytes": 1},
            self.parquet_pins[1],
        )
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "bytes"):
            contract.materialize_images(
                self.selected, parquet_pins=broken, image_dir=self.image_dir
            )

    def test_image_sha_drift_fails_closed(self) -> None:
        tampered = [dict(record) for record in self.selected]
        tampered[0]["image_sha256"] = "1" * 64
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "image SHA256"):
            contract.materialize_images(
                tampered, parquet_pins=self.parquet_pins, image_dir=self.image_dir
            )

    def test_image_size_drift_fails_closed(self) -> None:
        tampered = [dict(record) for record in self.selected]
        tampered[0]["width"] = 999
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "image size"):
            contract.materialize_images(
                tampered, parquet_pins=self.parquet_pins, image_dir=self.image_dir
            )

    def test_unpinned_parquet_fails_closed(self) -> None:
        tampered = [dict(record) for record in self.selected]
        tampered[0]["source_parquet_path"] = "/tmp/not-pinned.parquet"
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "un-pinned"):
            contract.materialize_images(
                tampered, parquet_pins=self.parquet_pins, image_dir=self.image_dir
            )


class RunManifestTests(ContractFixture):
    def test_run_manifest_is_label_free_and_uses_absolute_paths(self) -> None:
        output = self.write_run_manifest()
        records = contract.load_run_manifest(
            output, expected_count=6, task_sample_counts=self.task_counts
        )
        self.assertEqual(len(records), 6)
        for record in records:
            self.assertTrue(Path(record["image"]).is_absolute())
            self.assertEqual(
                record["prompt"], record["text"] + contract.W_REASON_SUFFIX
            )
            self.assertEqual(record["max_new_tokens"], contract.MAX_NEW_TOKENS)
            self.assertTrue(
                set(record).isdisjoint(contract.FORBIDDEN_SELECTOR_KEYS)
            )
            self.assertEqual(record["record_sha256"], contract.record_sha256(record))

    def test_run_manifest_refuses_existing_output(self) -> None:
        output = self.write_run_manifest()
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.write_run_manifest(
                self.selected,
                output_path=output,
                image_dir=self.image_dir,
                expected_count=6,
                task_sample_counts=self.task_counts,
            )

    def test_run_manifest_wrong_task_counts_fail(self) -> None:
        contract.materialize_images(
            self.selected, parquet_pins=self.parquet_pins, image_dir=self.image_dir
        )
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.write_run_manifest(
                self.selected,
                output_path=self.root / "run.jsonl",
                image_dir=self.image_dir,
                expected_count=6,
                task_sample_counts={"spatialmap": 3, "mazenav": 2, "spatialgrid": 1},
            )

    def test_prompt_without_w_reason_fails(self) -> None:
        output = self.write_run_manifest()
        lines = output.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["prompt"] = record["text"]
        record["record_sha256"] = contract.record_sha256(record)
        lines[0] = json.dumps(record, ensure_ascii=False)
        bad = self.root / "bad.jsonl"
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "w_reason"):
            contract.load_run_manifest(
                bad, expected_count=6, task_sample_counts=self.task_counts
            )

    def test_label_field_in_run_manifest_fails(self) -> None:
        output = self.write_run_manifest()
        lines = output.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["oracle_answer"] = "north"
        record["record_sha256"] = contract.record_sha256(record)
        lines[0] = json.dumps(record, ensure_ascii=False)
        bad = self.root / "labeled.jsonl"
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "labels"):
            contract.load_run_manifest(
                bad, expected_count=6, task_sample_counts=self.task_counts
            )

    def test_relative_image_path_fails(self) -> None:
        output = self.write_run_manifest()
        lines = output.read_text(encoding="utf-8").splitlines()
        record = json.loads(lines[0])
        record["image"] = "relative.png"
        record["record_sha256"] = contract.record_sha256(record)
        lines[0] = json.dumps(record, ensure_ascii=False)
        bad = self.root / "relative.jsonl"
        bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "absolute"):
            contract.load_run_manifest(
                bad, expected_count=6, task_sample_counts=self.task_counts
            )

    def test_tampered_image_bytes_fail(self) -> None:
        output = self.write_run_manifest()
        records = [
            json.loads(line)
            for line in output.read_text(encoding="utf-8").splitlines()
        ]
        victim = Path(records[0]["image"])
        victim.write_bytes(png_bytes("yellow", (9, 7)))
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "SHA256"):
            contract.load_run_manifest(
                output, expected_count=6, task_sample_counts=self.task_counts
            )


class SelectionTests(unittest.TestCase):
    def test_reference_manifest_requires_fields_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.jsonl"
            path.write_text(
                json.dumps({"sample_id": "spatialmap.vqa.1.0"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(contract.SpatialEvalContractError):
                contract.load_reference_manifest(path, expected_sha256=None)

    def test_reference_manifest_sha_drift_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(contract.SpatialEvalContractError, "SHA256"):
                contract.load_reference_manifest(path, expected_sha256="0" * 64)

    def test_select_handler_counts_fail_closed(self) -> None:
        records = [
            {"sample_id": "spatialmap.vqa.1.0"},
            {"sample_id": "mazenav.vqa.1.0"},
            {"sample_id": "spatialreal.vqa.1.0"},
        ]
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.select_handler_records(
                records,
                task_sample_counts={"spatialmap": 1, "mazenav": 1, "spatialgrid": 1},
                expected_count=3,
            )


class GridAndSanitizerTests(unittest.TestCase):
    def test_grid_range(self) -> None:
        sample = {"sample_id": "spatialmap.vqa.1.0"}
        self.assertEqual(contract.assert_spatialeval_grid(sample, [1, 8, 8]), 16)
        self.assertEqual(contract.assert_spatialeval_grid(sample, [1, 48, 48]), 576)
        for bad in (None, [2, 8, 8], [1, 1, 1], [1, 49, 48], [1, 8]):
            with self.assertRaises(contract.SpatialEvalContractError):
                contract.assert_spatialeval_grid(sample, bad)

    def test_sanitizer_drops_nothing_and_is_label_free(self) -> None:
        sample = {
            "sample_id": "spatialmap.vqa.1.0",
            "task": "spatialmap",
            "question_id": 0,
            "image": "/tmp/a.png",
            "prompt": "q",
            "max_new_tokens": 64,
        }
        self.assertEqual(
            set(contract.sanitize_sample_for_selector(sample)),
            set(contract.SELECTOR_SAMPLE_KEYS),
        )

    def test_sanitizer_rejects_labels(self) -> None:
        sample = {
            "sample_id": "spatialmap.vqa.1.0",
            "task": "spatialmap",
            "image": "/tmp/a.png",
            "prompt": "q",
            "max_new_tokens": 64,
            "oracle_answer": "north",
        }
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "labels"):
            contract.sanitize_sample_for_selector(sample)


class OutputPathTests(unittest.TestCase):
    def test_output_path_is_pinned(self) -> None:
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.ensure_output_path(contract.ARM_DENSE, Path("/tmp/wrong"))
        self.assertEqual(
            contract.ensure_output_path(contract.ARM_DENSE, None),
            contract.ARM_OUTPUT_PATHS[contract.ARM_DENSE],
        )
        self.assertEqual(
            contract.ensure_output_path(
                contract.ARM_DENSE, contract.ARM_OUTPUT_PATHS[contract.ARM_DENSE]
            ),
            contract.ARM_OUTPUT_PATHS[contract.ARM_DENSE],
        )


class ConfigPreregistrationTests(unittest.TestCase):
    def test_status_disclosure_and_naming(self) -> None:
        config = contract.load_config()
        self.assertEqual(config["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertFalse(config["executed"])
        self.assertEqual(config["gpu_runs"], 0)
        disclosure = config["data_disclosure"]
        self.assertTrue(disclosure["labels_observed"])
        self.assertTrue(disclosure["not_label_blind"])
        self.assertTrue(disclosure["not_used_for_selector_tuning"])
        self.assertFalse(config["dataset"]["spatialreal135"]["in_main_gate"])
        self.assertFalse(
            config["official_evaluator"]["official_cross_task_total_exists"]
        )
        self.assertFalse(
            config["project_aggregation_naming"]["is_official_total"]
        )
        self.assertEqual(config["success_gates"]["bootstrap"]["seed"], 20261009)
        self.assertEqual(config["success_gates"]["bootstrap"]["repetitions"], 5000)
        self.assertEqual(
            config["success_gates"][
                "v2_relative_to_same_input_dense_task_macro_minimum"
            ],
            0.9,
        )
        self.assertFalse(config["dataset"]["materialization"]["run_manifest_contains_labels"])

    def test_parquet_pins_present(self) -> None:
        config = contract.load_config()
        for name in ("parquet_shard_0", "parquet_shard_1"):
            entry = config["pins"][name]
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertGreater(entry["bytes"], 0)
            self.assertGreater(entry["rows"], 0)
        self.assertEqual(
            config["dataset"]["handler_sample_count"], contract.EXPECTED_SAMPLE_COUNT
        )

    def test_all_pins_match_workspace(self) -> None:
        result = contract.verify_dependency_pins()
        self.assertTrue(result["all_match"])
        self.assertEqual(result["verified_count"], result["requested_count"])

    def test_wrong_config_version_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"version": "wrong"}), encoding="utf-8")
            with self.assertRaises(contract.SpatialEvalContractError):
                contract.load_config(path)


if __name__ == "__main__":
    unittest.main()
