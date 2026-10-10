#!/usr/bin/env python3
"""CPU-only focused tests for evaluation/audit_edge576_cv_bench_manifest.py.

Builds tiny synthetic edge576 fixtures in temporary directories (with real small
PNG images) and exercises every contract: record hashing, image hashing and
dimensions, the G formula, the budget, frozen-label identity, processing pins
and summary recomputation.  No CUDA, no model weights, no network.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import audit_edge576_cv_bench_manifest as auditor  # noqa: E402
from evaluation import build_edge576_cv_bench_manifest as builder  # noqa: E402


def _write_image(path: Path, size: tuple[int, int]) -> str:
    from PIL import Image

    Image.new("RGB", size, color=(10, 20, 30)).save(path)
    return auditor.sha256_file(path)


def _sample(sample_id: str, image_path: Path) -> dict:
    return {
        "sample_id": sample_id,
        "task": "Count",
        "type": "2D",
        "source": "ADE20K",
        "source_dataset": "ADE20K Validation Set",
        "image": str(image_path),
        "prompt": "How many? Select.\n(A) 1\n(B) 2",
        "choices": ["1", "2"],
        "answer": "(A)",
        "max_new_tokens": 16,
    }


class Fixture:
    def __init__(self, tmp: Path, count: int = 3) -> None:
        self.tmp = tmp
        self.samples = []
        self.records = []
        self.images: list[Path] = []
        for index in range(count):
            image_path = tmp / f"{index}.png"
            size = (64 + 16 * index, 64 + 16 * index)
            digest = _write_image(image_path, size)
            self.images.append(image_path)
            sample = _sample(str(index), image_path)
            self.samples.append(sample)
            grid = [1, 16 + 2 * index, 16 + 2 * index]
            record = builder.build_record(
                sample,
                image_grid_thw=grid,
                image_sha256=digest,
                original_width=size[0],
                original_height=size[1],
                model_path="/model",
                processor_class="Qwen3VLProcessor",
                image_processor_class="Qwen2VLImageProcessorFast",
                patch_size=16,
                merge_size=2,
                min_pixels=65536,
                max_pixels=589824,
                budget_g=576,
            )
            self.records.append(record)
        self.source_manifest = tmp / "full.jsonl"
        self.source_manifest.write_text(
            "\n".join(builder.canonical_json(s) for s in self.samples) + "\n",
            encoding="utf-8",
        )
        self.manifest = tmp / "edge.jsonl"
        self.manifest.write_text(
            "\n".join(builder.canonical_json(r) for r in self.records) + "\n",
            encoding="utf-8",
        )
        self.summary_path = tmp / "summary.json"
        self.summary = builder.build_summary(
            self.records,
            source_manifest=self.source_manifest,
            manifest_path=self.manifest,
            manifest_bytes=self.manifest.stat().st_size,
            manifest_sha256=builder.sha256_file(self.manifest),
            model_path="/model",
            model_identity_manifest=None,
            patch_size=16,
            merge_size=2,
            min_pixels=65536,
            max_pixels=589824,
            budget_g=576,
        )
        self._write_summary()

    def _write_summary(self) -> None:
        self.summary_path.write_text(
            json.dumps(self.summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def rewrite_records(self) -> None:
        self.manifest.write_text(
            "\n".join(builder.canonical_json(r) for r in self.records) + "\n",
            encoding="utf-8",
        )

    def run_audit(self):
        return auditor.audit(
            manifest=self.manifest,
            summary_path=self.summary_path,
            source_manifest=self.source_manifest,
            expected_count=len(self.records),
            budget_g=576,
            check_images=True,
        )


class AuditPassTest(unittest.TestCase):
    def test_complete_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            report = fixture.run_audit()
        self.assertEqual(report["status"], "EDGE576_MANIFEST_AUDIT_PASS")
        self.assertEqual(report["records_checked"], 3)
        self.assertEqual(report["image_hashes_checked"], 3)
        self.assertEqual(report["original_dims_checked"], 3)
        contracts = dict(report["contracts"])
        self.assertFalse(contracts.pop("processor_grid_reverified"))
        self.assertTrue(all(contracts.values()))

    def test_recompute_summary_matches_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            recomputed = auditor.recompute_summary(fixture.records)
        self.assertEqual(recomputed["sample_count"], 3)
        self.assertEqual(recomputed["g_histogram"]["g_le_96"], 2)
        self.assertEqual(recomputed["g_histogram"]["g_97_256"], 1)
        self.assertEqual(recomputed["source_counts"], {"ADE20K": 3})


class AuditFailClosedTest(unittest.TestCase):
    def _fixture(self) -> Fixture:
        tmp = tempfile.mkdtemp()
        return Fixture(Path(tmp))

    def test_record_hash_tamper(self) -> None:
        fixture = self._fixture()
        fixture.records[0]["record_sha256"] = "0" * 64
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_image_hash_tamper(self) -> None:
        fixture = self._fixture()
        fixture.records[0]["image_sha256"] = "0" * 64
        fixture.records[0]["record_sha256"] = builder.record_sha256(fixture.records[0])
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_group_formula_tamper(self) -> None:
        fixture = self._fixture()
        fixture.records[0]["full_llm_visual_tokens"] = 128
        fixture.records[0]["record_sha256"] = builder.record_sha256(fixture.records[0])
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_over_budget_rejected(self) -> None:
        fixture = self._fixture()
        record = fixture.records[0]
        record["image_grid_thw"] = [1, 64, 64]  # G=1024 > 576
        record["full_patch_tokens"] = 4096
        record["full_llm_visual_tokens"] = 1024
        record["record_sha256"] = builder.record_sha256(record)
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_label_drift_rejected(self) -> None:
        fixture = self._fixture()
        fixture.records[1]["answer"] = "(B)"
        fixture.records[1]["record_sha256"] = builder.record_sha256(fixture.records[1])
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_original_dimension_drift_rejected(self) -> None:
        fixture = self._fixture()
        fixture.records[0]["original_width"] = 999
        fixture.records[0]["record_sha256"] = builder.record_sha256(fixture.records[0])
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()

    def test_record_count_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            with self.assertRaises(auditor.Edge576AuditError):
                auditor.audit(
                    manifest=fixture.manifest,
                    summary_path=fixture.summary_path,
                    source_manifest=fixture.source_manifest,
                    expected_count=99,
                    budget_g=576,
                )

    def test_summary_sha_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            fixture.summary["manifest"]["sha256"] = "f" * 64
            fixture._write_summary()
            with self.assertRaises(auditor.Edge576AuditError):
                fixture.run_audit()

    def test_summary_statistics_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            fixture.summary["g_histogram"]["g_le_96"] = 999
            fixture._write_summary()
            with self.assertRaises(auditor.Edge576AuditError):
                fixture.run_audit()

    def test_processing_parameter_drift_rejected(self) -> None:
        fixture = self._fixture()
        fixture.records[0]["processing"]["max_pixels"] = 1
        fixture.records[0]["record_sha256"] = builder.record_sha256(fixture.records[0])
        fixture.rewrite_records()
        with self.assertRaises(auditor.Edge576AuditError):
            fixture.run_audit()


class PercentileTest(unittest.TestCase):
    def test_single_value(self) -> None:
        self.assertEqual(auditor._percentile([42], 0.9), 42.0)

    def test_interpolation(self) -> None:
        self.assertAlmostEqual(auditor._percentile([0, 10], 0.5), 5.0)


class ModelPathThreadingTest(unittest.TestCase):
    """Regression tests for the "CLI accepts --model but drops it" defect.

    Before the fix, ``main`` parsed ``--model`` but never passed it to
    ``audit``, and ``audit`` always opened ``DEFAULT_MODEL`` when
    ``--verify-processor`` was set.  These tests fail closed on the old
    behaviour and pass on the corrected explicit ``model_path`` threading.
    """

    def test_report_records_explicit_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            report = auditor.audit(
                manifest=fixture.manifest,
                summary_path=fixture.summary_path,
                source_manifest=fixture.source_manifest,
                expected_count=len(fixture.records),
                budget_g=576,
                check_images=True,
                model_path=Path("/custom/processor/model"),
            )
        self.assertEqual(report["processor_model_path"], "/custom/processor/model")
        self.assertFalse(report["processor_grid_verification_requested"])

    def test_verify_processor_uses_explicit_model_path(self) -> None:
        captured: dict = {}
        fake_processor = object()

        def fake_loader(model_path: Path):
            captured["model_path"] = model_path
            return fake_processor

        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            with mock.patch.object(
                auditor, "load_local_processor", side_effect=fake_loader
            ), mock.patch.object(
                auditor, "verify_processor_grids", return_value=len(fixture.records)
            ) as verify:
                report = auditor.audit(
                    manifest=fixture.manifest,
                    summary_path=fixture.summary_path,
                    source_manifest=fixture.source_manifest,
                    expected_count=len(fixture.records),
                    budget_g=576,
                    check_images=True,
                    verify_processor=True,
                    model_path=Path("/custom/processor/model"),
                )
        self.assertEqual(captured["model_path"], Path("/custom/processor/model"))
        self.assertIs(verify.call_args.args[0], fake_processor)
        self.assertEqual(report["processor_model_path"], "/custom/processor/model")
        self.assertTrue(report["contracts"]["processor_grid_reverified"])
        self.assertEqual(report["processor_grids_reverified"], len(fixture.records))

    def test_provided_processor_skips_loader(self) -> None:
        fake_processor = object()
        with tempfile.TemporaryDirectory() as tmp:
            fixture = Fixture(Path(tmp))
            with mock.patch.object(
                auditor,
                "load_local_processor",
                side_effect=AssertionError("provided processor must be used"),
            ), mock.patch.object(
                auditor, "verify_processor_grids", return_value=len(fixture.records)
            ):
                report = auditor.audit(
                    manifest=fixture.manifest,
                    summary_path=fixture.summary_path,
                    source_manifest=fixture.source_manifest,
                    expected_count=len(fixture.records),
                    budget_g=576,
                    check_images=True,
                    verify_processor=True,
                    processor=fake_processor,
                    model_path=Path("/custom/processor/model"),
                )
        self.assertEqual(report["processor_model_path"], "/custom/processor/model")
        self.assertEqual(report["processor_grids_reverified"], len(fixture.records))

    def test_main_forwards_model_flag_to_audit(self) -> None:
        captured: dict = {}
        canned = {
            "status": "EDGE576_MANIFEST_AUDIT_PASS",
            "records_checked": 2638,
            "image_hashes_checked": 2638,
            "processor_grids_reverified": 2638,
            "processor_model_path": "/custom/processor/model",
            "manifest": {"sha256": "0" * 64},
            "contracts": {"processor_grid_reverified": True},
        }

        def fake_audit(**kwargs):
            captured.update(kwargs)
            return canned

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "audit.json"
            with mock.patch.object(
                auditor, "audit", side_effect=fake_audit
            ), contextlib.redirect_stdout(io.StringIO()):
                exit_code = auditor.main(
                    [
                        "--model",
                        "/custom/processor/model",
                        "--report",
                        str(report_path),
                        "--verify-processor",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(report_path.is_file())
        self.assertEqual(captured["model_path"], Path("/custom/processor/model"))
        self.assertTrue(captured["verify_processor"])


if __name__ == "__main__":
    unittest.main()
