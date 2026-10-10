#!/usr/bin/env python3
"""Focused tests for evaluation/summarize_current_selector_latency.py.

The tests use small synthetic JSONL files to exercise the completeness contract and
the statistics / speedup formulas. The production entry point still hard-validates
300 records = 100 samples x 3 repeats via the module defaults.
"""

from __future__ import annotations

import copy
import json
import statistics
import tempfile
import unittest
from pathlib import Path

from evaluation import summarize_current_selector_latency as s


def make_records(n_samples: int = 3, repeats: tuple[int, ...] = (0, 1, 2)) -> list[dict]:
    records = []
    for index in range(n_samples):
        for repeat in repeats:
            record = {
                "sample_id": str(index + 1),
                "repeat": repeat,
                "mode": "latency",
                "warmup": False,
                "run": {"warmup_per_sample": 1},
                "selector_path": "software",
                "selector_software_is_acceleration_claim": False,
            }
            for field in s.LATENCY_FIELDS:
                record[field] = 1.0 + index + 0.1 * repeat
            record["actual_token_retention"] = 1.0
            record["output_group_count"] = 48 + index
            record["full_llm_visual_tokens"] = 100 + index
            record["output_tokens"] = 5
            record["peak_cuda_bytes"] = 1000 + index
            records.append(record)
    return records


def make_quality(path: Path) -> Path:
    dense = {
        "sample_count": 2638,
        "parse_success_rate": 0.9848,
        "accuracy": 0.8681,
        "cv_bench_accuracy": 0.8727,
    }
    candidates = []
    for index, label in enumerate(s.SPEEDUP_TIERS):
        candidate = {
            "sample_count": 2638,
            "parse_success_rate": 0.98,
            "accuracy": 0.60 + 0.05 * index,
            "cv_bench_accuracy": 0.65 + 0.05 * index,
        }
        candidates.append(
            {
                "label": label,
                "candidate": candidate,
                "dense_bf16_reference": dense,
                "native_bf16_reference": dense,
                "absolute_delta_vs_dense": candidate["cv_bench_accuracy"] - dense["cv_bench_accuracy"],
                "relative_retention_vs_dense": candidate["cv_bench_accuracy"] / dense["cv_bench_accuracy"],
                "paired_bootstrap_ci95_vs_dense": [-0.2, -0.1],
                "paired_bootstrap_ci95_vs_native": [-0.21, -0.11],
                "bootstrap_seed": 20261004,
                "bootstrap_repeats": 5000,
                "candidate_jsonl": f"/tmp/{label}.jsonl",
            }
        )
    data = {
        "status": "COMPLETE",
        "formal_sample_count": 2638,
        "manifest": "/tmp/manifest.jsonl",
        "manifest_sha256": "deadbeef",
        "generated_at": "2026-10-03T17:44:25+00:00",
        "forbidden_primary_metrics": ["pointing", "grounding", "ndcg", "retained_mass"],
        "candidates": candidates,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class ContractConstantsTest(unittest.TestCase):
    def test_frozen_defaults(self):
        self.assertEqual(s.EXPECTED_RECORDS, 300)
        self.assertEqual(s.EXPECTED_SAMPLES, 100)
        self.assertEqual(s.EXPECTED_REPEATS, (0, 1, 2))
        self.assertEqual(s.REQUIRED_MODE, "latency")
        self.assertEqual(s.REQUIRED_WARMUP_PER_SAMPLE, 1)

    def test_required_latency_fields(self):
        self.assertEqual(
            s.LATENCY_FIELDS,
            [
                "model_inference_seconds",
                "application_end_to_end_seconds",
                "vision_total_seconds",
                "gpu_vision_seconds",
                "selector_software_seconds",
                "llm_prepare_seconds",
                "llm_prefill_seconds",
                "decoder_seconds",
                "processor_seconds",
                "host_to_device_seconds",
                "prompt_text_embedding_seconds",
                "postprocess_seconds",
            ],
        )

    def test_cross_run_reference_constant(self):
        self.assertEqual(s.OLD_INTERLEAVED_DENSE_CORE_MEAN_MS, 225.16090276495865)

    def test_forbidden_primary_metrics(self):
        self.assertEqual(s.FORBIDDEN_PRIMARY_METRICS, ["pointing", "grounding", "ndcg"])


class ValidationTest(unittest.TestCase):
    def test_accepts_well_formed_records(self):
        info = s.validate_records(make_records(), "identity", expected_records=9, expected_samples=3)
        self.assertTrue(info["passed"])
        self.assertEqual(info["records"], 9)
        self.assertEqual(info["unique_samples"], 3)
        self.assertEqual(info["duplicate_pairs"], 0)

    def test_defaults_reject_small_input(self):
        with self.assertRaises(ValueError):
            s.validate_records(make_records(), "identity")

    def test_rejects_wrong_record_count(self):
        with self.assertRaisesRegex(ValueError, "expected exactly 300 records"):
            s.validate_records(make_records(), "identity")

    def test_rejects_wrong_unique_sample_count(self):
        with self.assertRaisesRegex(ValueError, "unique sample_id"):
            s.validate_records(make_records(), "identity", expected_records=9, expected_samples=4)

    def test_rejects_missing_repeat(self):
        records = [record for record in make_records() if not (record["sample_id"] == "1" and record["repeat"] == 2)]
        with self.assertRaisesRegex(ValueError, "repeat set"):
            s.validate_records(records, "identity", expected_records=8, expected_samples=3)

    def test_rejects_duplicate_pair(self):
        records = make_records()
        records.append(copy.deepcopy(records[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            s.validate_records(records, "identity", expected_records=10, expected_samples=3)

    def test_rejects_warmup_true(self):
        records = make_records()
        records[0]["warmup"] = True
        with self.assertRaisesRegex(ValueError, "warmup"):
            s.validate_records(records, "identity", expected_records=9, expected_samples=3)

    def test_rejects_bad_warmup_per_sample(self):
        records = make_records()
        records[0]["run"]["warmup_per_sample"] = 0
        with self.assertRaisesRegex(ValueError, "warmup_per_sample"):
            s.validate_records(records, "identity", expected_records=9, expected_samples=3)

    def test_rejects_bad_mode(self):
        records = make_records()
        records[0]["mode"] = "quality"
        with self.assertRaisesRegex(ValueError, "mode"):
            s.validate_records(records, "identity", expected_records=9, expected_samples=3)

    def test_rejects_non_numeric_field(self):
        records = make_records()
        records[0]["output_tokens"] = "5"
        with self.assertRaisesRegex(ValueError, "output_tokens"):
            s.validate_records(records, "identity", expected_records=9, expected_samples=3)


class StatisticsTest(unittest.TestCase):
    def test_percentile_nearest_rank(self):
        self.assertEqual(s.percentile(list(range(1, 11)), 0.95), 10)
        self.assertEqual(s.percentile(list(range(1, 21)), 0.95), 19)
        self.assertEqual(s.percentile([5.0], 0.95), 5.0)

    def test_describe_reference_values(self):
        stats = s.describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["std"], (1.25) ** 0.5)
        self.assertEqual(stats["p95"], 4.0)

    def test_describe_std_is_population_ddof0(self):
        values = [1.0, 2.0, 3.0]
        stats = s.describe(values)
        self.assertAlmostEqual(stats["std"], statistics.pstdev(values))
        self.assertNotAlmostEqual(stats["std"], statistics.stdev(values))

    def test_per_sample_median_reduction(self):
        records = make_records(n_samples=1)
        for record in records:
            record["model_inference_seconds"] = {0: 1.0, 1: 3.0, 2: 100.0}[record["repeat"]]
        medians = s.per_sample_median_records(records)
        self.assertEqual(len(medians), 1)
        self.assertEqual(medians[0]["sample_id"], "1")
        self.assertEqual(medians[0]["model_inference_seconds"], 3.0)

    def test_summarize_tier_scopes(self):
        summary = s.summarize_tier(make_records(n_samples=2))
        self.assertEqual(summary["raw"]["n"], 6)
        self.assertEqual(summary["per_sample_median"]["n"], 2)
        self.assertEqual(summary["raw"]["derived_mean"]["output_tokens"], 5.0)
        # Per sample i the three latency values are 1+i, 1.1+i, 1.2+i so the median is 1.1+i.
        self.assertAlmostEqual(
            summary["per_sample_median"]["fields"]["model_inference_seconds"]["mean"],
            1.1 + 0.5,
        )
        self.assertEqual(summary["raw"]["peak_cuda_bytes_max"], 1001.0)
        self.assertEqual(summary["per_sample_median"]["peak_cuda_bytes_max"], 1001.0)


class SpeedupTest(unittest.TestCase):
    @staticmethod
    def _tier(model_mean: float, application_mean: float) -> dict:
        return {
            "raw": {"fields": {
                "model_inference_seconds": {"mean": model_mean},
                "application_end_to_end_seconds": {"mean": application_mean},
            }},
            "per_sample_median": {"fields": {
                "model_inference_seconds": {"mean": model_mean},
                "application_end_to_end_seconds": {"mean": application_mean},
            }},
        }

    def test_speedup_is_identity_over_candidate(self):
        tiers = {
            "identity": self._tier(10.0, 20.0),
            "K48": self._tier(5.0, 8.0),
            "K64": self._tier(4.0, 10.0),
            "K96": self._tier(2.0, 5.0),
        }
        speedups = s.speedups_vs_identity(tiers)
        self.assertEqual(speedups["K48"]["raw"]["model_inference_seconds"]["speedup"], 2.0)
        self.assertEqual(speedups["K48"]["raw"]["application_end_to_end_seconds"]["speedup"], 2.5)
        self.assertEqual(speedups["K96"]["per_sample_median"]["model_inference_seconds"]["speedup"], 5.0)
        self.assertEqual(speedups["K64"]["per_sample_median"]["application_end_to_end_seconds"]["speedup"], 2.0)


class ReportIntegrationTest(unittest.TestCase):
    def _build(self, root: Path) -> dict:
        input_paths = {}
        for label in s.TIER_ORDER:
            path = root / f"{label}.jsonl"
            path.write_text("\n".join(json.dumps(record) for record in make_records()) + "\n", encoding="utf-8")
            input_paths[label] = path
        quality_path = make_quality(root / "quality.json")
        return s.build_report(
            input_paths=input_paths,
            quality_path=quality_path,
            expected_records=9,
            expected_samples=3,
        )

    def test_build_report_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self._build(root)
            self.assertTrue(report["validation"]["passed"])
            self.assertEqual(report["forbidden_primary_metrics"], ["pointing", "grounding", "ndcg"])
            self.assertFalse(report["fpga_measured"])
            self.assertEqual(report["selector_implementation"], "software")
            self.assertFalse(report["cross_run_reference"]["usable_as_same_run_speedup"])
            self.assertFalse(report["cross_run_reference"]["same_run"])
            self.assertEqual(
                report["cross_run_reference"]["old_interleaved_native_dense_core_mean_ms"],
                225.16090276495865,
            )
            self.assertEqual(report["quality"]["tiers"]["identity"]["relative_retention_vs_bf16"], 1.0)
            self.assertEqual(report["latency"]["identity"]["raw"]["n"], 9)
            self.assertEqual(report["latency"]["identity"]["per_sample_median"]["n"], 3)

            output_json = root / "out.json"
            output_md = root / "out.md"
            s.write_report(report, output_json, output_md)
            self.assertTrue(output_json.exists() and output_md.exists())
            written = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(written["speedups_vs_identity"]["K48"]["raw"]["model_inference_seconds"]["speedup"], 1.0)
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("软件实现", markdown)
            self.assertIn("不是 FPGA 实测", markdown)
            self.assertIn("225.160903", markdown)


if __name__ == "__main__":
    unittest.main()
