#!/usr/bin/env python3
"""CPU-only focused tests for evaluation/build_edge576_cv_bench_manifest.py.

Covers the deterministic pure helpers (grid/G formula, canonical record hashing,
fail-closed budget enforcement, frozen-label identity, summary statistics) and
the Chinese report renderer.  No CUDA, no model weights, no network and no
processor load: every fixture is a temporary path.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import build_edge576_cv_bench_manifest as builder  # noqa: E402


def _sample(sample_id: str, *, image: str = "/tmp/does-not-matter.image") -> dict:
    return {
        "sample_id": sample_id,
        "task": "Count",
        "type": "2D",
        "source": "ADE20K",
        "source_dataset": "ADE20K Validation Set",
        "image": image,
        "prompt": "How many? Select.\n(A) 1\n(B) 2",
        "choices": ["1", "2"],
        "answer": "(A)",
        "max_new_tokens": 16,
    }


def _record(sample_id: str, grid: list[int], *, image: str) -> dict:
    return builder.build_record(
        _sample(sample_id, image=image),
        image_grid_thw=grid,
        image_sha256="deadbeef",
        original_width=64,
        original_height=64,
        model_path="/model",
        processor_class="Qwen3VLProcessor",
        image_processor_class="Qwen2VLImageProcessorFast",
        patch_size=16,
        merge_size=2,
        min_pixels=65536,
        max_pixels=589824,
        budget_g=576,
    )


class GroupCountTest(unittest.TestCase):
    def test_product_and_merge_unit(self) -> None:
        patch, g = builder.compute_group_count([1, 48, 48], merge_size=2)
        self.assertEqual(patch, 2304)
        self.assertEqual(g, 576)

    def test_small_image(self) -> None:
        patch, g = builder.compute_group_count([1, 16, 16], merge_size=2)
        self.assertEqual((patch, g), (256, 64))

    def test_boolean_is_rejected(self) -> None:
        with self.assertRaises(builder.Edge576ManifestError):
            builder.compute_group_count([1, True, 16])

    def test_wrong_length_rejected(self) -> None:
        with self.assertRaises(builder.Edge576ManifestError):
            builder.compute_group_count([1, 16])

    def test_misalignment_fails_closed(self) -> None:
        with self.assertRaises(builder.Edge576ManifestError):
            builder.compute_group_count([1, 3, 3], merge_size=2)

    def test_non_positive_rejected(self) -> None:
        with self.assertRaises(builder.Edge576ManifestError):
            builder.compute_group_count([1, 0, 16])


class RecordHashTest(unittest.TestCase):
    def test_hash_is_deterministic_and_excludes_itself(self) -> None:
        record = _record("0", [1, 16, 16], image="/tmp/x.image")
        first = record["record_sha256"]
        mutated = dict(record)
        mutated["record_sha256"] = "0" * 64
        self.assertEqual(builder.record_sha256(mutated), first)

    def test_hash_changes_with_content(self) -> None:
        a = _record("0", [1, 16, 16], image="/tmp/x.image")
        b = _record("0", [1, 32, 32], image="/tmp/x.image")
        self.assertNotEqual(a["record_sha256"], b["record_sha256"])

    def test_build_record_contract(self) -> None:
        record = _record("7", [1, 32, 32], image="/tmp/x.image")
        self.assertEqual(record["sample_id"], "7")
        self.assertEqual(record["full_patch_tokens"], 1024)
        self.assertEqual(record["full_llm_visual_tokens"], 256)
        self.assertTrue(record["processing"]["edge576"])
        self.assertEqual(record["processing"]["merge_unit"], 4)
        self.assertEqual(record["processing"]["resize_factor"], 32)
        self.assertEqual(record["processing"]["max_pixels"], 589824)
        self.assertEqual(
            record["record_sha256"], builder.record_sha256(record)
        )


class BudgetTest(unittest.TestCase):
    def test_accepts_g_equal_to_budget(self) -> None:
        builder.assert_g_within_budget([{"sample_id": "0", "full_llm_visual_tokens": 576}], budget_g=576)

    def test_rejects_g_over_budget(self) -> None:
        with self.assertRaises(builder.Edge576ManifestError):
            builder.assert_g_within_budget(
                [{"sample_id": "9", "full_llm_visual_tokens": 577}], budget_g=576
            )


class LabelIdentityTest(unittest.TestCase):
    def test_matching_labels_pass(self) -> None:
        full = [_sample("0"), _sample("1")]
        edge = [_record("0", [1, 16, 16], image="/tmp/a"), _record("1", [1, 16, 16], image="/tmp/b")]
        builder.validate_label_identity(edge, full)

    def test_label_drift_fails_closed(self) -> None:
        full = [_sample("0")]
        edge = [_record("0", [1, 16, 16], image="/tmp/a")]
        edge[0]["answer"] = "(B)"
        edge[0]["record_sha256"] = builder.record_sha256(edge[0])
        with self.assertRaises(builder.Edge576ManifestError):
            builder.validate_label_identity(edge, full)

    def test_id_set_drift_fails_closed(self) -> None:
        full = [_sample("0"), _sample("1")]
        edge = [_record("0", [1, 16, 16], image="/tmp/a")]
        with self.assertRaises(builder.Edge576ManifestError):
            builder.validate_label_identity(edge, full)


class SummaryTest(unittest.TestCase):
    def test_summary_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_manifest = tmp_path / "full.jsonl"
            source_manifest.write_text(
                "\n".join(builder.canonical_json(_sample(str(i))) for i in range(3)) + "\n",
                encoding="utf-8",
            )
            manifest = tmp_path / "edge.jsonl"
            records = [
                _record("0", [1, 16, 16], image="/tmp/a"),  # G=64
                _record("1", [1, 32, 32], image="/tmp/b"),  # G=256
                _record("2", [1, 48, 48], image="/tmp/c"),  # G=576
            ]
            manifest.write_text(
                "\n".join(builder.canonical_json(r) for r in records) + "\n",
                encoding="utf-8",
            )
            summary = builder.build_summary(
                records,
                source_manifest=source_manifest,
                manifest_path=manifest,
                manifest_bytes=manifest.stat().st_size,
                manifest_sha256=builder.sha256_file(manifest),
                model_path="/model",
                model_identity_manifest=None,
                patch_size=16,
                merge_size=2,
                min_pixels=65536,
                max_pixels=589824,
                budget_g=576,
            )
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["g_min"], 64)
        self.assertEqual(summary["g_max"], 576)
        self.assertEqual(summary["g_histogram"]["g_le_96"], 1)
        self.assertEqual(summary["g_histogram"]["g_97_256"], 1)
        self.assertEqual(summary["g_histogram"]["g_513_576"], 1)
        self.assertEqual(summary["identity_bypass_count_g_le_96"], 1)
        self.assertEqual(summary["compressed_count_g_gt_96"], 2)
        self.assertTrue(summary["all_g_within_budget"])
        self.assertEqual(summary["builder"], builder.BUILDER_NAME)

    def test_summary_rejects_over_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_manifest = tmp_path / "full.jsonl"
            source_manifest.write_text(builder.canonical_json(_sample("0")) + "\n", encoding="utf-8")
            manifest = tmp_path / "edge.jsonl"
            manifest.write_text("placeholder\n", encoding="utf-8")
            over = _record("0", [1, 48, 48], image="/tmp/a")
            over["full_llm_visual_tokens"] = 577
            with self.assertRaises(builder.Edge576ManifestError):
                builder.build_summary(
                    [over],
                    source_manifest=source_manifest,
                    manifest_path=manifest,
                    manifest_bytes=1,
                    manifest_sha256="x",
                    model_path="/model",
                    model_identity_manifest=None,
                    patch_size=16,
                    merge_size=2,
                    min_pixels=65536,
                    max_pixels=589824,
                    budget_g=576,
                )


class ReportTest(unittest.TestCase):
    def test_report_contains_key_facts(self) -> None:
        summary = {
            "sample_count": 2638,
            "unique_sample_id_count": 2638,
            "source_manifest": {"path": "full.jsonl", "sha256": "abc"},
            "manifest": {"path": "edge.jsonl", "sha256": "def", "bytes": 1},
            "budget_g": 576,
            "processing_parameters": {
                "min_pixels": 65536,
                "max_pixels": 589824,
                "patch_size": 16,
                "merge_size": 2,
                "g_definition": builder.G_DEFINITION,
            },
            "g_min": 64,
            "g_median": 300,
            "g_mean": 300.0,
            "g_p90": 500,
            "g_p95": 550,
            "g_max": 576,
            "g_count_at_budget": 10,
            "identity_bypass_count_g_le_96": 143,
            "compressed_count_g_gt_96": 2495,
            "g_histogram": {
                "g_le_96": 143,
                "g_97_256": 800,
                "g_257_512": 1000,
                "g_513_576": 695,
            },
            "source_counts": {"ADE20K": 633},
            "task_counts": {"Count": 788},
        }
        markdown = builder.render_report_markdown(summary)
        self.assertIn("edge576", markdown)
        self.assertIn("589824", markdown)
        self.assertIn("2638", markdown)
        self.assertIn(builder.G_DEFINITION, markdown)


if __name__ == "__main__":
    unittest.main()
