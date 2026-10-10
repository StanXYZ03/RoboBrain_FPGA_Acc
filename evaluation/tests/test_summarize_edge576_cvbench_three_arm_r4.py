#!/usr/bin/env python3
"""CPU-only focused tests for the edge576 three-arm r4 summariser.

These tests never load a model, never touch CUDA and never modify any raw arm
JSONL, the r4 config, the model or ``CURRENT_STATUS.md``.  They exercise the
fail-closed validation logic with synthetic records, compare the re-implemented
paired bootstrap against the frozen function bit-for-bit on a small sample, and
(only when the real full2638 inputs exist) run the real summary once and compare
it against the frozen bootstrap implementation.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import summarize_edge576_cvbench_three_arm_r4 as summ  # noqa: E402
from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    paired_bootstrap_delta as frozen_paired_bootstrap_delta,
)
from evaluation.build_cv_bench_pruning_report import score as frozen_score  # noqa: E402
from evaluation.build_cv_bench_pruning_report import stats as frozen_stats  # noqa: E402
from evaluation.edge576_cvbench_runner_contract import (  # noqa: E402
    ARM_DENSE,
    ARM_L2,
    ARM_MECHANISM_IDS,
    ARM_V2,
    ARMS,
    ATTENTION_IMPLEMENTATION,
    DECODING,
    DO_SAMPLE,
    DTYPE,
    EDGE576_MANIFEST,
    EDGE576_MANIFEST_SHA256,
    MAX_NEW_TOKENS,
    MAX_PIXELS,
    MIN_PIXELS,
    record_sha256 as contract_record_sha256,
)

REAL_INPUTS = {
    ARM_DENSE: summ.ARM_OUTPUT_PATHS[ARM_DENSE],
    ARM_L2: summ.ARM_OUTPUT_PATHS[ARM_L2],
    ARM_V2: summ.ARM_OUTPUT_PATHS[ARM_V2],
}
REAL_INPUTS_AVAILABLE = (
    all(path.is_file() for path in REAL_INPUTS.values())
    and EDGE576_MANIFEST.is_file()
    and summ.R4_CONFIG.is_file()
)


def _record(
    arm: str,
    sample_id: str,
    *,
    grid: tuple[int, int, int] = (1, 16, 16),
    correct: bool = True,
    parsed: str | None = "A",
    source: str = "ADE20K",
    task: str = "Count",
    identity_bypass: bool | None = None,
) -> dict:
    g = (grid[0] * grid[1] * grid[2]) // 4
    if arm == ARM_DENSE:
        output_group_count = g
        identity = True
    else:
        identity = g <= 96 if identity_bypass is None else identity_bypass
        output_group_count = g if identity else 96
    record = {
        "sample_id": sample_id,
        "arm": arm,
        "mechanism_id": ARM_MECHANISM_IDS[arm],
        "manifest_sha256": EDGE576_MANIFEST_SHA256,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "image_grid_thw": [int(v) for v in grid],
        "full_llm_visual_tokens": g,
        "output_text": "(A)",
        "output_token_count": 1,
        "output_token_ids_available": False,
        "parsed_answer": parsed,
        "correct": correct,
        "source": source,
        "task": task,
        "source_dataset": f"{source} Validation Set",
        "type": "2D" if source != "Omni3D" else "3D",
        "prompt_sha256": "0" * 64,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "decoding": DECODING,
        "dtype": DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "output_group_count": output_group_count,
        "identity_bypass": identity,
        "run_mode": "full2638",
    }
    if arm == ARM_DENSE:
        record["actual_token_retention"] = output_group_count / g
        record["llm_visual_tokens"] = output_group_count
    else:
        record["requested_budget"] = 96
        record["region_grid"] = 3
        record["local_candidate_cap"] = 16
    if arm == ARM_V2:
        record.update(
            max_swaps=16,
            swap_count=0,
            utility_formula="utility",
            legal_gain_formula="legal_gain",
            uses_labels=False,
            uses_answer_tokens=False,
            uses_gradients=False,
        )
    record["selected_group_indices"] = list(range(output_group_count))
    record["record_sha256"] = contract_record_sha256(record)
    return record


def _joined_records(seed: int = 7, count: int = 120) -> list[dict]:
    rng = random.Random(seed)
    sources = ["ADE20K", "COCO", "Omni3D"]
    tasks = ["Count", "Relation", "Depth", "Distance"]
    datasets = ["A", "B"]
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_id": str(index),
                "task": tasks[index % len(tasks)],
                "source": sources[index % len(sources)],
                "source_dataset": datasets[index % len(datasets)],
                "a_correct": bool(rng.getrandbits(1)),
                "a_answer": "A" if rng.getrandbits(1) else None,
                "b_correct": bool(rng.getrandbits(1)),
                "b_answer": "B" if rng.getrandbits(1) else None,
            }
        )
    return rows


class PairedBootstrapTests(unittest.TestCase):
    def test_matches_frozen_on_small_sample(self) -> None:
        rows = _joined_records(seed=11, count=150)
        mine = summ.paired_bootstrap_delta_v2(
            rows, "a_correct", "b_correct", seed=123, repeats=500
        )
        frozen_ci = frozen_paired_bootstrap_delta(
            [dict(row) for row in rows], "a_correct", "b_correct", 123, 500
        )
        self.assertEqual(mine["ci95"], frozen_ci)
        point = (
            frozen_score(rows, "a_correct")["cv_bench_accuracy"]
            - frozen_score(rows, "b_correct")["cv_bench_accuracy"]
        )
        self.assertAlmostEqual(mine["point_estimate"], point, places=15)

    def test_weighted_accuracy_matches_frozen_score(self) -> None:
        rows = _joined_records(seed=3, count=90)
        mine = summ._weighted_cv_bench_accuracy(rows, "a_correct")
        self.assertAlmostEqual(
            mine, frozen_score(rows, "a_correct")["cv_bench_accuracy"], places=15
        )

    def test_rejects_too_few_repeats(self) -> None:
        with self.assertRaises(summ.SummaryError):
            summ.paired_bootstrap_delta_v2(
                _joined_records(count=20), "a_correct", "b_correct", repeats=1
            )


class MetricTests(unittest.TestCase):
    def test_official_metric_expected_value_and_parse_rate(self) -> None:
        records = []
        plan = [
            ("ADE20K", True, "A"),
            ("ADE20K", False, None),
            ("COCO", True, "B"),
            ("COCO", True, "C"),
            ("Omni3D", True, "A"),
            ("Omni3D", False, None),
        ]
        for index, (source, correct, parsed) in enumerate(plan):
            record = _record(
                ARM_DENSE, str(index), source=source, correct=correct, parsed=parsed
            )
            records.append(record)
        metric = summ.arm_official_metric(records, expected_count=6)
        self.assertAlmostEqual(
            metric["source_accuracy"]["ADE20K"], 0.5, places=15
        )
        self.assertAlmostEqual(metric["source_accuracy"]["COCO"], 1.0, places=15)
        self.assertAlmostEqual(
            metric["source_accuracy"]["Omni3D"], 0.5, places=15
        )
        expected = (((0.5 + 1.0) / 2) + 0.5) / 2
        self.assertAlmostEqual(
            metric["official_cv_bench_accuracy"], expected, places=15
        )
        self.assertEqual(metric["parse_success_count"], 4)
        self.assertEqual(metric["parse_failure_count"], 2)
        self.assertAlmostEqual(metric["parse_failure_rate"], 2 / 6, places=15)
        self.assertEqual(metric["correct_key"], "correct")
        self.assertEqual(metric["parsed_answer_key"], "parsed_answer")


class LatencyTests(unittest.TestCase):
    def test_latency_stats_match_frozen_stats(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mine = summ.latency_stats(values)
        theirs = frozen_stats(values)
        for key in ("mean", "median", "p95", "min", "max", "std"):
            self.assertAlmostEqual(mine[key], theirs[key], places=15)
        self.assertEqual(mine["count"], len(values))
        self.assertIsNone(summ.latency_stats([]))

    def test_dense_mapping_is_honest_and_selector_is_null(self) -> None:
        records = []
        for index in range(3):
            record = _record(ARM_DENSE, str(index))
            record["elapsed_seconds"] = 1.0 + index
            record["vision_encoder_seconds"] = 0.1 + index
            record["llm_prefill_seconds"] = 0.2 + index
            record["decoder_seconds"] = 0.3 + index
            records.append(record)
        summary = summ.latency_summary(records, ARM_DENSE)
        stats = summary["stats"]
        self.assertIsNone(stats["application_end_to_end_seconds"])
        self.assertAlmostEqual(stats["model_inference_seconds"]["mean"], 2.0)
        self.assertAlmostEqual(stats["vision_seconds"]["mean"], 1.1)
        self.assertIsNone(stats["selector_cpu_roundtrip_seconds"])
        self.assertEqual(
            summary["field_mapping"]["model_inference_seconds"],
            ["elapsed_seconds"],
        )

    def test_missing_field_is_explicit_null(self) -> None:
        records = [_record(ARM_L2, "0")]
        summary = summ.latency_summary(records, ARM_L2)
        self.assertIsNone(summary["stats"]["application_end_to_end_seconds"])
        self.assertIsNone(summary["stats"]["selector_cpu_roundtrip_seconds"])


class TokenTests(unittest.TestCase):
    def test_token_census_and_retention(self) -> None:
        records = [
            _record(ARM_L2, "0", grid=(1, 8, 8)),  # G=16 identity
            _record(ARM_L2, "1", grid=(1, 32, 32)),  # G=256 compressed to K=96
            _record(ARM_L2, "2", grid=(1, 16, 16)),  # G=64 identity
        ]
        token = summ.token_summary(records, ARM_L2)
        self.assertEqual(token["K_value"], 96)
        self.assertEqual(token["g_le_96_identity_count"], 2)
        self.assertEqual(token["g_gt_96_count"], 1)
        self.assertEqual(token["identity_bypass_flag_count"], 2)
        retentions = [1.0, 96 / 256, 1.0]
        self.assertAlmostEqual(
            token["mean_llm_visual_token_retention"],
            statistics.fmean(retentions),
            places=15,
        )
        self.assertEqual(token["G_stats"]["min"], 16)
        self.assertEqual(token["G_stats"]["max"], 256)

    def test_dense_has_no_k_budget(self) -> None:
        records = [_record(ARM_DENSE, "0")]
        token = summ.token_summary(records, ARM_DENSE)
        self.assertIsNone(token["K_stats"])
        self.assertIsNone(token["K_value"])


class ValidationTests(unittest.TestCase):
    def test_valid_records_pass(self) -> None:
        for arm in ARMS:
            record = _record(arm, "0")
            info = summ.validate_arm_record(record, arm)
            self.assertEqual(info["sample_id"], "0")
            self.assertTrue(info["correct"])
            self.assertIsInstance(info["parsed"], str)

    def test_wrong_arm_fails(self) -> None:
        record = _record(ARM_L2, "0")
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_V2)

    def test_wrong_mechanism_fails(self) -> None:
        record = _record(ARM_V2, "0")
        record["mechanism_id"] = "dense_edge576"
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_V2)

    def test_non_bool_correct_fails(self) -> None:
        record = _record(ARM_L2, "0")
        record["correct"] = 1
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_bad_parsed_answer_type_fails(self) -> None:
        record = _record(ARM_L2, "0")
        record["parsed_answer"] = 5
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_missing_parsed_answer_field_fails(self) -> None:
        record = _record(ARM_L2, "0")
        del record["parsed_answer"]
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_stale_record_sha256_fails(self) -> None:
        record = _record(ARM_L2, "0")
        record["record_sha256"] = "0" * 64
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_wrong_grid_fails(self) -> None:
        record = _record(ARM_L2, "0")
        record["image_grid_thw"] = [1, 8, 16]
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_compressed_sample_must_emit_k(self) -> None:
        record = _record(
            ARM_L2, "0", grid=(1, 32, 32), identity_bypass=False
        )
        record["output_group_count"] = 64
        record["selected_group_indices"] = list(range(64))
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_L2)

    def test_dense_wrong_group_count_fails(self) -> None:
        record = _record(ARM_DENSE, "0")
        record["output_group_count"] = 1
        record["selected_group_indices"] = [0]
        record["record_sha256"] = contract_record_sha256(record)
        with self.assertRaises(summ.SummaryError):
            summ.validate_arm_record(record, ARM_DENSE)

    def test_build_joined_detects_id_mismatch(self) -> None:
        records = {
            ARM_DENSE: [_record(ARM_DENSE, "0")],
            ARM_L2: [_record(ARM_L2, "1")],
            ARM_V2: [_record(ARM_V2, "0")],
        }
        with self.assertRaises(summ.SummaryError):
            summ.build_joined(records)


class LoadTests(unittest.TestCase):
    def test_load_arm_jsonl_verifies_sha_and_count(self) -> None:
        record = _record(ARM_DENSE, "0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            sha = summ.sha256_file(path)
            records, actual = summ.load_arm_jsonl(
                path, ARM_DENSE, expected_sha256=sha, expected_count=1
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(actual, sha)
            with self.assertRaises(summ.SummaryError):
                summ.load_arm_jsonl(
                    path, ARM_DENSE, expected_sha256="0" * 64, expected_count=1
                )
            with self.assertRaises(summ.SummaryError):
                summ.load_arm_jsonl(
                    path, ARM_DENSE, expected_sha256=sha, expected_count=2
                )

    def test_load_arm_jsonl_rejects_duplicate_ids(self) -> None:
        record = _record(ARM_DENSE, "0")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arm.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            sha = summ.sha256_file(path)
            with self.assertRaises(summ.SummaryError):
                summ.load_arm_jsonl(
                    path, ARM_DENSE, expected_sha256=sha, expected_count=2
                )


class WriterTests(unittest.TestCase):
    def _minimal_summary(self) -> dict:
        scores = {
            arm: {
                "official_cv_bench_accuracy": 0.5,
                "unweighted_sample_accuracy": 0.5,
                "accuracy_2d": 0.5,
                "accuracy_3d": 0.5,
                "source_accuracy": {"ADE20K": 0.5, "COCO": 0.5, "Omni3D": 0.5},
                "task_accuracy": {
                    "Count": 0.5,
                    "Depth": 0.5,
                    "Distance": 0.5,
                    "Relation": 0.5,
                },
                "parse_success_count": 2600,
                "parse_failure_count": 38,
                "parse_success_rate": 2600 / 2638,
                "parse_failure_rate": 38 / 2638,
            }
            for arm in ARMS
        }
        deltas = {
            name: {
                "point_estimate": 0.01,
                "ci95": [0.001, 0.02],
                "ci95_low": 0.001,
                "ci95_high": 0.02,
                "strata": "task::source_dataset",
                "seed": summ.SEED,
                "repeats": summ.REPEATS,
            }
            for name in summ.deltas_names()
        }
        token = {
            "G_stats": {
                "count": 2638,
                "mean": 300.0,
                "median": 300.0,
                "p95": 576.0,
                "min": 64.0,
                "max": 576.0,
                "std": 100.0,
            },
            "K_stats": None,
            "K_value": None,
            "mean_llm_visual_token_retention": 1.0,
            "min_llm_visual_token_retention": 1.0,
            "max_llm_visual_token_retention": 1.0,
            "g_le_96_identity_count": 143,
            "g_gt_96_count": 2495,
            "identity_bypass_flag_count": 2638,
            "G_histogram": {"64": 1},
        }
        return {
            "report": "edge576_cvbench_three_arm_r4_summary",
            "generated_at": "2026-10-08T00:00:00+00:00",
            "scope": {
                "sample_count": 2638,
                "arms": {
                    arm: {
                        "sha256": summ.INPUT_SHA256[arm],
                        "output_path": str(
                            summ.ARM_OUTPUT_PATHS[arm].relative_to(ROOT)
                        ),
                    }
                    for arm in ARMS
                },
                "edge576_manifest": {"path": "m", "sha256": "x"},
                "r4_config": {"path": "c", "sha256": "y"},
                "metric": {},
                "paired_bootstrap": {},
            },
            "scores": scores,
            "relative": {
                "quality_retention_vs_same_input_dense_edge576": {
                    arm: 1.0 for arm in ARMS
                },
                "deltas": deltas,
            },
            "tokens": {arm: dict(token) for arm in ARMS},
            "latency": {
                arm: {field: None for field in summ.LATENCY_MAPPING}
                for arm in ARMS
            },
            "latency_field_mapping": {
                arm: {field: None for field in summ.LATENCY_MAPPING}
                for arm in ARMS
            },
            "pre_registered_gates": {
                "quality_retention_vs_same_input_dense_edge576": {
                    "arm": ARM_V2,
                    "value": 1.0,
                    "threshold": 0.9,
                    "comparison": ">=",
                    "passed": True,
                },
                "relative_to_corrected_integer_l2": {
                    "point_estimate": 0.01,
                    "point_estimate_bound": 0.0,
                    "point_estimate_passed": True,
                    "ci95": [0.001, 0.02],
                    "lower_bound": 0.0,
                    "ci_lower_passed": True,
                },
            },
            "limitations": {"notes": ["software selector is not FPGA"]},
        }

    def test_writers_round_trip_and_verify(self) -> None:
        summary = self._minimal_summary()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            artifacts = tmp_path / "artifacts.json"
            reports = tmp_path / "reports.json"
            csv_path = tmp_path / "summary.csv"
            md_path = tmp_path / "summary.md"
            summ.write_json(artifacts, summary)
            verification = summ.verify_artifacts_json(artifacts, summary)
            self.assertEqual(verification["failed"], [])
            summ.write_json(reports, {"summary": summary})
            summ.write_csv(csv_path, summary)
            summ.write_md(md_path, summary)
            self.assertGreater(csv_path.stat().st_size, 0)
            text = md_path.read_text(encoding="utf-8")
            self.assertIn("FPGA", text)
            self.assertIn("不是外部验证", text)

    def test_verify_artifacts_detects_tamper(self) -> None:
        summary = self._minimal_summary()
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts.json"
            summ.write_json(artifacts, summary)
            payload = json.loads(artifacts.read_text(encoding="utf-8"))
            payload["scores"][ARM_DENSE]["official_cv_bench_accuracy"] = 0.0
            artifacts.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(summ.SummaryError):
                summ.verify_artifacts_json(artifacts, summary)


@unittest.skipUnless(REAL_INPUTS_AVAILABLE, "real full2638 inputs not present")
class RealInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = summ.summarize_edge576_three_arm(
            seed=summ.SEED, repeats=summ.REPEATS
        )
        cls.joined = summ.build_joined(
            {
                arm: summ.load_jsonl(REAL_INPUTS[arm])
                for arm in ARMS
            }
        )

    def test_pins_and_counts(self) -> None:
        validation = self.summary["validation"]
        self.assertEqual(
            validation["each_arm_count"],
            {arm: 2638 for arm in ARMS},
        )
        self.assertTrue(validation["same_sample_id_set"])
        self.assertTrue(validation["sha256_pins_match"])
        for arm in ARMS:
            self.assertEqual(validation["sha256_actual"][arm], summ.INPUT_SHA256[arm])
            self.assertEqual(validation["record_sha256_all_match"][arm], 2638)

    def test_known_official_scores(self) -> None:
        expected = {
            ARM_DENSE: 0.8739581554855612,
            ARM_L2: 0.79378695554051,
            ARM_V2: 0.8078459445801811,
        }
        for arm in ARMS:
            self.assertAlmostEqual(
                self.summary["scores"][arm]["official_cv_bench_accuracy"],
                expected[arm],
                places=15,
            )

    def test_known_parse_failures(self) -> None:
        expected = {ARM_DENSE: 36, ARM_L2: 42, ARM_V2: 41}
        for arm in ARMS:
            self.assertEqual(
                self.summary["scores"][arm]["parse_failure_count"], expected[arm]
            )

    def test_bootstrap_matches_frozen_on_real_joined_table(self) -> None:
        aliased = []
        for row in self.joined:
            copy = dict(row)
            copy["dense_answer"] = "A"
            copy["l2_answer"] = "A"
            copy["v2_answer"] = "A"
            aliased.append(copy)
        frozen = frozen_paired_bootstrap_delta(
            aliased, "v2_correct", "l2_correct", summ.SEED, summ.REPEATS
        )
        self.assertEqual(
            self.summary["relative"]["deltas"]["v2_minus_l2"]["ci95"], frozen
        )


if __name__ == "__main__":
    unittest.main()
