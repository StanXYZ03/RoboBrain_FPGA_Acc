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

from evaluation import run_spatialeval_vqa_three_arm as runner
from evaluation import spatialeval_vqa_contract as contract


def png_bytes(color: str = "red", size: tuple[int, int] = (9, 7)) -> bytes:
    stream = BytesIO()
    Image.new("RGB", size, color=color).save(stream, format="PNG")
    return stream.getvalue()


def build_case(root: Path):
    """Materialise two samples and a run manifest inside ``root``."""
    image_dir = root / "images"
    image = png_bytes("red", (9, 7))
    shard = root / "shard.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "id": "spatialmap.vqa.1.0",
                    "text": "q1",
                    "image": {"bytes": image, "path": None},
                    "oracle_answer": "north",
                    "oracle_option": "A",
                    "oracle_full_answer": "A. north",
                },
                {
                    "id": "mazenav.vqa.1.0",
                    "text": "q2",
                    "image": {"bytes": image, "path": None},
                    "oracle_answer": "north",
                    "oracle_option": "A",
                    "oracle_full_answer": "A. north",
                },
            ]
        ),
        shard,
    )
    sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    with Image.open(BytesIO(image)) as opened:
        width, height = opened.size
    image_sha = hashlib.sha256(image).hexdigest()
    selected = [
        {
            "sample_id": sample_id,
            "text": text,
            "oracle_answer": "north",
            "image_sha256": image_sha,
            "width": width,
            "height": height,
            "source_parquet_path": str(shard),
            "source_parquet_sha256": sha,
            "source_parquet_row_index": index,
        }
        for index, (sample_id, text) in enumerate(
            [("spatialmap.vqa.1.0", "q1"), ("mazenav.vqa.1.0", "q2")]
        )
    ]
    pins = (
        {
            "path": str(shard),
            "sha256": sha,
            "bytes": shard.stat().st_size,
            "row_count": 2,
        },
    )
    contract.materialize_images(selected, parquet_pins=pins, image_dir=image_dir)
    manifest = root / "run.jsonl"
    contract.write_run_manifest(
        selected,
        output_path=manifest,
        image_dir=image_dir,
        expected_count=2,
        task_sample_counts={"spatialmap": 1, "mazenav": 1, "spatialgrid": 0},
    )
    samples = contract.load_run_manifest(
        manifest,
        expected_count=2,
        task_sample_counts={"spatialmap": 1, "mazenav": 1, "spatialgrid": 0},
    )
    return samples, manifest, contract.sha256_file(manifest)


def frozen_record(grid=(1, 4, 4), groups=(0, 1, 2, 3), output="A. north", with_ids=False):
    record = {
        "output_text": output,
        "selected_group_indices": list(groups),
        "image_grid_thw": list(grid),
        "output_tokens": 5,
    }
    if with_ids:
        record["output_token_ids"] = [1, 2, 3, 4, 5]
    return record


class NormalizeRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.samples, self.manifest, self.sha = build_case(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dense_record_without_token_ids(self) -> None:
        record = runner.normalize_record(
            contract.ARM_DENSE,
            self.samples[0],
            frozen_record(),
            manifest_sha256=self.sha,
            run_metadata={},
            run_mode=contract.RUN_MODE,
        )
        self.assertFalse(record["output_token_ids_available"])
        self.assertNotIn("output_token_ids", record)
        self.assertEqual(record["output_token_count"], 5)
        self.assertEqual(record["full_llm_visual_tokens"], 4)
        self.assertTrue(record["identity_bypass"])
        self.assertEqual(record["record_sha256"], contract.record_sha256(record))
        self.assertFalse(record["uses_labels"])

    def test_l2_record_with_token_ids(self) -> None:
        record = runner.normalize_record(
            contract.ARM_L2,
            self.samples[0],
            frozen_record(with_ids=True),
            manifest_sha256=self.sha,
            run_metadata={},
            run_mode=contract.RUN_MODE,
        )
        self.assertTrue(record["output_token_ids_available"])
        self.assertEqual(record["output_token_count"], 5)

    def test_missing_output_text_fails(self) -> None:
        with self.assertRaises(contract.SpatialEvalContractError):
            runner.normalize_record(
                contract.ARM_DENSE,
                self.samples[0],
                {"selected_group_indices": [0], "image_grid_thw": [1, 4, 4]},
                manifest_sha256=self.sha,
                run_metadata={},
                run_mode=contract.RUN_MODE,
            )

    def test_grid_over_budget_fails(self) -> None:
        with self.assertRaises(contract.SpatialEvalContractError):
            runner.normalize_record(
                contract.ARM_DENSE,
                self.samples[0],
                frozen_record(grid=(1, 49, 48), groups=range(588)),
                manifest_sha256=self.sha,
                run_metadata={},
                run_mode=contract.RUN_MODE,
            )

    def test_token_count_disagreement_fails(self) -> None:
        bad = frozen_record(with_ids=True)
        bad["output_tokens"] = 7
        with self.assertRaises(contract.SpatialEvalContractError):
            runner.normalize_record(
                contract.ARM_L2,
                self.samples[0],
                bad,
                manifest_sha256=self.sha,
                run_metadata={},
                run_mode=contract.RUN_MODE,
            )


class ResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.samples, self.manifest, self.sha = build_case(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def write_record(self, arm=contract.ARM_DENSE, sample=None, output=None):
        sample = sample or self.samples[0]
        output = output or (self.root / "out.jsonl")
        record = runner.normalize_record(
            arm,
            sample,
            frozen_record(),
            manifest_sha256=self.sha,
            run_metadata={},
            run_mode=contract.RUN_MODE,
        )
        output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return record, output

    def test_valid_record_roundtrip(self) -> None:
        record, output = self.write_record()
        done = contract.validate_resume(
            output, self.samples, arm=contract.ARM_DENSE, manifest_sha256=self.sha
        )
        self.assertEqual(done, {record["sample_id"]})

    def test_record_sha_tamper_fails(self) -> None:
        record, output = self.write_record()
        record["output_text"] = "tampered"
        output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "record_sha256"):
            contract.validate_resume(
                output, self.samples, arm=contract.ARM_DENSE, manifest_sha256=self.sha
            )

    def test_wrong_arm_fails(self) -> None:
        _, output = self.write_record()
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.validate_resume(
                output, self.samples, arm=contract.ARM_L2, manifest_sha256=self.sha
            )

    def test_manifest_sha_mismatch_fails(self) -> None:
        _, output = self.write_record()
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.validate_resume(
                output,
                self.samples,
                arm=contract.ARM_DENSE,
                manifest_sha256="0" * 64,
            )

    def test_missing_field_fails(self) -> None:
        record, output = self.write_record()
        del record["image_grid_thw"]
        output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(contract.SpatialEvalContractError, "missing fields"):
            contract.validate_resume(
                output, self.samples, arm=contract.ARM_DENSE, manifest_sha256=self.sha
            )

    def test_unknown_sample_fails(self) -> None:
        record, output = self.write_record()
        record["sample_id"] = "spatialmap.vqa.999.0"
        record["record_sha256"] = contract.record_sha256(record)
        output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.validate_resume(
                output, self.samples, arm=contract.ARM_DENSE, manifest_sha256=self.sha
            )

    def test_dense_group_count_must_equal_g(self) -> None:
        record, output = self.write_record()
        record["output_group_count"] = 3
        record["selected_group_indices"] = [0, 1, 2]
        record["record_sha256"] = contract.record_sha256(record)
        output.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaises(contract.SpatialEvalContractError):
            contract.validate_resume(
                output, self.samples, arm=contract.ARM_DENSE, manifest_sha256=self.sha
            )


class RunArmOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.samples, self.manifest, self.sha = build_case(self.root)
        self.output = self.root / "out.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_complete_resume_skips_model_and_cuda(self) -> None:
        calls = {"model": 0, "cuda": 0}
        result = runner.run_arm(
            arm=contract.ARM_DENSE,
            samples=self.samples,
            manifest_path=self.manifest,
            output=self.output,
            model_path=None,
            model_label="x",
            device="cuda:0",
            validation_fn=lambda *a, **k: {"pending_count": 0, "existing_done_count": 2},
            require_cuda_fn=lambda device: calls.__setitem__("cuda", calls["cuda"] + 1),
            load_model_fn=lambda *a, **k: calls.__setitem__("model", calls["model"] + 1),
        )
        self.assertEqual(result["status"], "SPATIALEVAL_RUNNER_ARM_ALREADY_COMPLETE")
        self.assertEqual(calls, {"model": 0, "cuda": 0})
        self.assertEqual(result["gpu_runs"], 0)

    def test_cuda_required_before_model_identity(self) -> None:
        called = {"identity": False}

        def explode(device):
            raise contract.SpatialEvalContractError("no CUDA")

        with self.assertRaises(contract.SpatialEvalContractError):
            runner.run_arm(
                arm=contract.ARM_DENSE,
                samples=self.samples,
                manifest_path=self.manifest,
                output=self.output,
                model_path=Path("/model"),
                model_label="x",
                device="cuda:0",
                validation_fn=lambda *a, **k: {"pending_count": 2, "existing_done_count": 0},
                require_cuda_fn=explode,
                model_identity_fn=lambda *a, **k: called.__setitem__("identity", True),
                load_model_fn=lambda *a, **k: self.fail("model must not load"),
            )
        self.assertFalse(called["identity"])

    def test_full_path_calls_one_warmup_and_executes(self) -> None:
        order: list[str] = []
        state = {"scorer": None, "checkpoint_identity": {}}

        def warmup(arm, model, processor, sample, manifest, **kwargs):
            order.append("warmup")
            self.assertEqual(kwargs.get("manifest_sha256"), self.sha)
            return {"count": 1}

        def execute(**kwargs):
            order.append("execute")
            self.assertEqual(kwargs["arm"], contract.ARM_DENSE)
            return {"arm": contract.ARM_DENSE, "written": 2}

        with mock.patch.object(
            runner, "build_run_metadata", return_value={"stub": True}
        ):
            result = runner.run_arm(
                arm=contract.ARM_DENSE,
                samples=self.samples,
                manifest_path=self.manifest,
                output=self.output,
                model_path=Path("/model"),
                model_label="x",
                device="cuda:0",
                validation_fn=lambda *a, **k: {"pending_count": 2, "existing_done_count": 0},
                require_cuda_fn=lambda device: order.append("cuda"),
                model_identity_fn=lambda *a, **k: {"ok": True},
                build_state_fn=lambda arm: order.append("state") or state,
                load_model_fn=lambda *a, **k: order.append("model") or object(),
                load_processor_fn=lambda *a, **k: order.append("processor") or object(),
                warmup_fn=warmup,
                execute_fn=execute,
            )
        self.assertEqual(result["status"], "SPATIALEVAL_RUNNER_ARM_COMPLETE")
        self.assertEqual(result["warmup_count"], 1)
        self.assertEqual(result["gpu_runs"], 1)
        self.assertEqual(order, ["cuda", "state", "model", "processor", "warmup", "execute"])


class LockTests(unittest.TestCase):
    def test_frozen_flags_reject_overrides(self) -> None:
        parser = runner.build_parser()
        for flag in (
            ["--budget", "48"],
            ["--region-grid", "2"],
            ["--local-candidate-cap", "8"],
            ["--max-swaps", "4"],
            ["--min-pixels", "1"],
            ["--max-pixels", "2"],
        ):
            args = parser.parse_args(["--arm", contract.ARM_DENSE] + flag)
            with self.assertRaises(SystemExit):
                runner._enforce_locks(parser, args)

    def test_defaults_pass_locks(self) -> None:
        parser = runner.build_parser()
        args = parser.parse_args(["--arm", contract.ARM_DENSE])
        runner._enforce_locks(parser, args)


if __name__ == "__main__":
    unittest.main()
