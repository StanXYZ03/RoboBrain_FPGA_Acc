from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

from evaluation import spatialeval_official_compatible_metric as metric


def record(sample_id: str, output: str, oracle: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "output_text": output,
        "oracle_answer": oracle,
    }


class FakeOfficial(types.SimpleNamespace):
    """Stand-in that records exactly which official parser was invoked."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, int, str]] = []

    def _parse(self, task: str, text: str, qid: int, model: str):
        self.calls.append((task, text, qid, model))
        if text == "NONE":
            return None
        if text == "ERROR":
            raise RuntimeError("parser failed")
        return text

    def extract_answer_from_text_spatialmap(self, text, qid, model):
        return self._parse("spatialmap", text, qid, model)

    def extract_answer_from_text_mazenav(self, text, qid, model):
        return self._parse("mazenav", text, qid, model)

    def extract_answer_from_text_spatialgrid(self, text, qid, model):
        return self._parse("spatialgrid", text, qid, model)


def full_4500(official_task_qid=0) -> list[dict[str, str]]:
    rows = []
    for task in metric.TASKS:
        for index in range(metric.EXPECTED_PER_TASK_COUNT):
            qid = official_task_qid or index % len(metric.QUESTION_IDS)
            rows.append(record(f"{task}.vqa.{index}.{qid}", "x", "x"))
    return rows


class OfficialCompatibleMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.official = FakeOfficial()

    # --- official parser port ------------------------------------------------

    def test_official_module_is_importlib_loaded_from_provenance(self) -> None:
        module = metric.load_official_evaluator()
        self.assertEqual(
            Path(module.__file__).resolve(), metric.OFFICIAL_EVALUATOR.resolve()
        )
        for parser_name in metric.TASK_PARSERS.values():
            self.assertTrue(callable(getattr(module, parser_name, None)))

    def test_real_official_parsers_are_invoked_end_to_end(self) -> None:
        module = metric.load_official_evaluator()
        cases = [
            ("spatialmap.vqa.7.0", "The correct answer is A. Northeast.", "Northeast"),
            ("spatialgrid.vqa.7.1", "I can see a rabbit there.", "rabbit"),
            ("mazenav.vqa.7.2", "The answer is yes.", "Yes"),
        ]
        for sample_id, output, oracle in cases:
            with self.subTest(sample_id=sample_id):
                row = metric.evaluate_record(
                    record(sample_id, output, oracle), official_module=module
                )
                self.assertTrue(row["parse_success"])
                self.assertEqual(row["official_compatible_correct"], 1)

    def test_dispatches_to_official_parser_with_fixed_model_name(self) -> None:
        result = metric.evaluate_record(
            record("spatialmap.vqa.0.1", "Northwest", "northwest"),
            official_module=self.official,
        )
        self.assertTrue(result["official_compatible_correct"])
        self.assertEqual(
            self.official.calls,
            [("spatialmap", "Northwest", 1, "robobrain25")],
        )

    def test_each_task_routes_to_its_own_official_parser(self) -> None:
        for task in metric.TASKS:
            self.official.calls.clear()
            metric.evaluate_record(
                record(f"{task}.vqa.5.2", "x", "x"), official_module=self.official
            )
            self.assertEqual(self.official.calls[0][0], task)

    def test_id_and_answer_aliases_are_accepted(self) -> None:
        result = metric.evaluate_record(
            {"id": "spatialmap.vqa.0.0", "answer": "northwest", "oracle_answer": "northwest"},
            official_module=self.official,
        )
        self.assertEqual(result["sample_id"], "spatialmap.vqa.0.0")
        self.assertEqual(result["official_compatible_correct"], 1)

    # --- primary substring vs strict diagnostic -------------------------------

    def test_substring_primary_and_strict_diagnostic_differ(self) -> None:
        result = metric.evaluate_record(
            record("spatialgrid.vqa.0.2", "rabbit near center", "rabbit"),
            official_module=self.official,
        )
        self.assertEqual(result["official_compatible_correct"], 1)
        self.assertEqual(result["strict_equal_correct"], 0)

    def test_numeric_substring_inflation_matches_official_rule(self) -> None:
        # Official rule is oracle-as-needle: "0" is contained in "10".
        result = metric.evaluate_record(
            record("spatialgrid.vqa.0.0", "10", "0"),
            official_module=self.official,
        )
        self.assertEqual(result["official_compatible_correct"], 1)
        self.assertEqual(result["strict_equal_correct"], 0)

    def test_strict_diagnostic_trims_and_lowercases(self) -> None:
        result = metric.evaluate_record(
            record("spatialmap.vqa.2.0", "Rabbit", " rabbit "),
            official_module=self.official,
        )
        # Whitespace makes the primary substring test fail...
        self.assertEqual(result["official_compatible_correct"], 0)
        # ...while the trim+lower equality diagnostic passes.
        self.assertEqual(result["strict_equal_correct"], 1)

    # --- failure handling -----------------------------------------------------

    def test_none_and_parser_exception_are_wrong(self) -> None:
        none_result = metric.evaluate_record(
            record("mazenav.vqa.0.2", "NONE", "no"),
            official_module=self.official,
        )
        error_result = metric.evaluate_record(
            record("mazenav.vqa.1.2", "ERROR", "no"),
            official_module=self.official,
        )
        self.assertFalse(none_result["parse_success"])
        self.assertFalse(error_result["parse_success"])
        self.assertEqual(none_result["official_compatible_correct"], 0)
        self.assertEqual(error_result["strict_equal_correct"], 0)
        self.assertIn("RuntimeError", error_result["parse_error"])

    def test_parse_failures_are_counted_in_aggregates(self) -> None:
        rows = [
            record("spatialmap.vqa.0.0", "northwest", "northwest"),
            record("mazenav.vqa.0.2", "NONE", "no"),
            record("spatialgrid.vqa.0.1", "rabbit", "rabbit"),
        ]
        result = metric.score_records(
            rows, expected_count=None, official_module=self.official
        )
        micro = result["micro_project_summary"]
        self.assertEqual(micro["parse_failure_count"], 1)
        self.assertEqual(micro["parse_success_count"], 2)
        self.assertAlmostEqual(micro["parse_success_rate"], 2 / 3)

    # --- contract rejection ---------------------------------------------------

    def test_unknown_task_is_rejected(self) -> None:
        for sample_id in ("spatialreal.vqa.sa_0.0", "spatialmapx.vqa.0.0"):
            with self.subTest(sample_id=sample_id):
                with self.assertRaises(metric.SpatialEvalOfficialMetricError):
                    metric.evaluate_record(
                        record(sample_id, "left", "left"),
                        official_module=self.official,
                    )

    def test_missing_fields_are_rejected(self) -> None:
        with self.assertRaises(metric.SpatialEvalOfficialMetricError):
            metric.evaluate_record(
                {"output_text": "x", "oracle_answer": "x"},
                official_module=self.official,
            )
        with self.assertRaises(metric.SpatialEvalOfficialMetricError):
            metric.evaluate_record(
                {"sample_id": "spatialmap.vqa.0.0", "output_text": "x"},
                official_module=self.official,
            )

    def test_duplicate_id_is_rejected(self) -> None:
        rows = [
            record("spatialmap.vqa.0.0", "northwest", "northwest"),
            record("spatialmap.vqa.0.0", "northwest", "northwest"),
        ]
        with self.assertRaisesRegex(
            metric.SpatialEvalOfficialMetricError, "duplicate"
        ):
            metric.score_records(
                rows, expected_count=None, official_module=self.official
            )

    def test_count_lock_rejects_non_4500(self) -> None:
        with self.assertRaisesRegex(metric.SpatialEvalOfficialMetricError, "count"):
            metric.score_records(
                [record("spatialmap.vqa.0.0", "x", "x")],
                expected_count=metric.EXPECTED_PRIMARY_COUNT,
                official_module=self.official,
            )

    def test_count_lock_rejects_wrong_task_or_qid_distribution(self) -> None:
        all_one_task = [
            record(f"spatialmap.vqa.{i}.{i % 3}", "x", "x") for i in range(4500)
        ]
        with self.assertRaisesRegex(metric.SpatialEvalOfficialMetricError, "1500"):
            metric.score_records(
                all_one_task,
                expected_count=metric.EXPECTED_PRIMARY_COUNT,
                official_module=self.official,
            )

        skewed_qids = []
        for task in metric.TASKS:
            for i in range(metric.EXPECTED_PER_TASK_COUNT):
                qid = 0 if i < 1000 else 1
                skewed_qids.append(record(f"{task}.vqa.{i}.{qid}", "x", "x"))
        with self.assertRaisesRegex(metric.SpatialEvalOfficialMetricError, "500"):
            metric.score_records(
                skewed_qids,
                expected_count=metric.EXPECTED_PRIMARY_COUNT,
                official_module=self.official,
            )

    def test_4500_count_lock_accepts_frozen_distribution(self) -> None:
        result = metric.score_records(
            full_4500(), official_module=self.official
        )
        self.assertEqual(result["micro_project_summary"]["count"], 4500)
        self.assertEqual(
            result["micro_project_summary"]["official_compatible_accuracy"], 1.0
        )
        self.assertEqual(result["macro_project_summary"]["task_count"], 3)
        for task in metric.TASKS:
            self.assertEqual(result["per_task"][task]["count"], 1500)
            for qid in metric.QUESTION_IDS:
                key = f"{task}::{qid}"
                self.assertEqual(result["per_task_question_id"][key]["count"], 500)

    # --- aggregation ----------------------------------------------------------

    def test_task_and_qid_aggregation_and_macro_micro(self) -> None:
        rows = [
            record("spatialmap.vqa.0.0", "northwest", "northwest"),
            record("spatialmap.vqa.1.1", "wrong", "target"),
            record("mazenav.vqa.0.2", "Yes", "yes"),
            record("spatialgrid.vqa.0.1", "rabbit extra", "rabbit"),
        ]
        result = metric.score_records(
            rows, expected_count=None, official_module=self.official
        )
        self.assertEqual(result["micro_project_summary"]["count"], 4)
        self.assertEqual(
            result["micro_project_summary"]["official_compatible_accuracy"], 0.75
        )
        self.assertEqual(
            result["micro_project_summary"]["strict_equal_accuracy"], 0.5
        )
        self.assertEqual(result["per_task"]["spatialmap"]["count"], 2)
        self.assertEqual(result["per_task"]["spatialmap"]["official_compatible_correct"], 1)
        self.assertEqual(result["per_task"]["mazenav"]["count"], 1)
        self.assertEqual(result["per_task"]["spatialgrid"]["count"], 1)
        self.assertEqual(result["per_task_question_id"]["mazenav::2"]["count"], 1)
        self.assertEqual(result["per_task_question_id"]["spatialgrid::1"]["count"], 1)

        # Equal-task macro ignores the uneven task sizes.
        expected_macro = (
            result["per_task"]["spatialmap"]["official_compatible_accuracy"]
            + result["per_task"]["mazenav"]["official_compatible_accuracy"]
            + result["per_task"]["spatialgrid"]["official_compatible_accuracy"]
        ) / 3
        self.assertAlmostEqual(
            result["macro_project_summary"]["official_compatible_accuracy"],
            expected_macro,
        )
        self.assertEqual(result["macro_project_summary"]["weighting"], "equal_task")

    def test_macro_requires_all_three_tasks(self) -> None:
        rows = [record("spatialmap.vqa.0.0", "northwest", "northwest")]
        with self.assertRaises(metric.SpatialEvalOfficialMetricError):
            metric.score_records(
                rows, expected_count=None, official_module=self.official
            )
        # The partial escape hatch keeps the per-task view usable.
        result = metric.score_records(
            rows,
            expected_count=None,
            require_all_tasks=False,
            official_module=self.official,
        )
        self.assertEqual(result["macro_project_summary"]["task_count"], 1)

    def test_metric_name_marks_project_aggregation_not_official_total(self) -> None:
        result = metric.score_records(
            full_4500(), include_record_rows=False, official_module=self.official
        )
        self.assertIn("project_aggregation", result["metric"])
        self.assertIn("not_official_total", result["metric"])
        self.assertFalse(result["official_total_score_exists"])
        self.assertIn("not official", result["project_aggregation_notice"])
        self.assertNotIn("evaluated_records", result)

    # --- CLI ------------------------------------------------------------------

    def test_cli_reads_jsonl_and_writes_json(self) -> None:
        rows = [
            record("spatialmap.vqa.0.0", "northwest", "northwest"),
            record("spatialmap.vqa.1.1", "wrong", "target"),
            record("mazenav.vqa.0.2", "Yes", "yes"),
            record("spatialgrid.vqa.0.1", "rabbit extra", "rabbit"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "predictions.jsonl"
            output_path = Path(tmp) / "metric.json"
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            exit_code = metric.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--no-count-lock",
                    "--no-record-rows",
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["micro_project_summary"]["count"], 4)
        self.assertEqual(
            payload["micro_project_summary"]["official_compatible_accuracy"], 0.75
        )
        self.assertEqual(payload["model_name_for_official_parser"], "robobrain25")


if __name__ == "__main__":
    unittest.main()
