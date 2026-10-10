from __future__ import annotations

import unittest
from pathlib import Path

from evaluation import spatialeval_official_evaluator as official


def prediction(sample_id: str, output_text):
    return {"sample_id": sample_id, "output_text": output_text}


class OfficialModuleTests(unittest.TestCase):
    def test_official_module_loads_and_sha_matches(self) -> None:
        module = official.load_official_evaluation_module()
        self.assertEqual(
            official.sha256_file(official.OFFICIAL_EVALUATION_PY),
            official.OFFICIAL_EVALUATION_SHA256,
        )
        self.assertTrue(hasattr(module, "extract_answer_from_text_spatialmap"))

    def test_sha_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            official.SpatialEvalOfficialEvaluatorError, "SHA256"
        ):
            official.load_official_evaluation_module(expected_sha256="0" * 64)

    def test_parsers_are_the_official_functions(self) -> None:
        parsers = official.official_parsers()
        self.assertEqual(
            set(parsers), {"spatialmap", "mazenav", "spatialgrid"}
        )
        for parser in parsers.values():
            self.assertTrue(
                str(parser.__code__.co_filename).endswith(
                    "spatialeval-official-d82ba382/evaluation.py"
                )
            )

    def test_missing_module_fails_closed(self) -> None:
        with self.assertRaises(official.SpatialEvalOfficialEvaluatorError):
            official.load_official_evaluation_module(
                Path("/tmp/does-not-exist-official.py")
            )


class OfficialComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parsers = official.official_parsers()

    def test_substring_inclusion_is_not_equality(self) -> None:
        # official: "north" in "northeast" -> correct; strict diagnostic: wrong.
        result = official.score_predictions(
            [prediction("spatialmap.vqa.1.0", "A. Northeast")],
            {"spatialmap.vqa.1.0": "north"},
            parsers=self.parsers,
        )
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["strict_equality_diagnostic"]["correct"], 0)

    def test_none_parse_counts_wrong_even_when_oracle_is_substring_of_none(self) -> None:
        # mazenav qid2 with no yes/no marker returns None.  Upstream would turn
        # None into "none", where "no" is a substring; this adapter counts it
        # wrong.
        result = official.score_predictions(
            [prediction("mazenav.vqa.1.2", "Maybe something else entirely.")],
            {"mazenav.vqa.1.2": "no"},
            parsers=self.parsers,
        )
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["parse_failure_count"], 1)

    def test_exception_counts_wrong(self) -> None:
        def boom(*args, **kwargs):
            raise RuntimeError("parser exploded")

        result = official.score_predictions(
            [prediction("spatialmap.vqa.1.0", "anything")],
            {"spatialmap.vqa.1.0": "north"},
            parsers={"spatialmap": boom, "mazenav": boom, "spatialgrid": boom},
        )
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["parse_failure_count"], 1)

    def test_parsed_integer_is_stringified(self) -> None:
        result = official.score_predictions(
            [prediction("spatialgrid.vqa.1.0", "There are 3 blocks.")],
            {"spatialgrid.vqa.1.0": "3"},
            parsers=self.parsers,
        )
        self.assertEqual(result["correct"], 1)

    def test_aggregations_and_naming(self) -> None:
        predictions = [
            prediction("spatialmap.vqa.1.0", "A. Northeast"),
            prediction("spatialmap.vqa.2.0", "unparseable"),
            prediction("mazenav.vqa.1.0", "There are 2 right turns"),
            prediction("spatialgrid.vqa.1.1", "cat"),
        ]
        oracle = {
            "spatialmap.vqa.1.0": "Northeast",
            "spatialmap.vqa.2.0": "North",
            "mazenav.vqa.1.0": "2",
            "spatialgrid.vqa.1.1": "cat",
        }
        result = official.score_predictions(predictions, oracle, parsers=self.parsers)
        self.assertEqual(result["count"], 4)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["project_micro"]["accuracy"], 0.75)
        self.assertEqual(result["project_task_macro"]["tasks"], ["spatialmap", "mazenav", "spatialgrid"])
        # macro = mean(0.5, 1.0, 1.0) = 0.8333...
        self.assertAlmostEqual(result["project_task_macro"]["accuracy"], 5 / 6)
        self.assertEqual(
            result["official_semantics"]["official_cross_task_total_exists"], False
        )
        self.assertIn("not_official_total", result["project_micro"]["semantics"])
        self.assertIn("spatialmap::0", result["per_task_qid"])

    def test_spatialreal_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            official.SpatialEvalOfficialEvaluatorError, "no handler"
        ):
            official.score_predictions(
                [prediction("spatialreal.vqa.1.0", "A. Northeast")],
                {"spatialreal.vqa.1.0": "Northeast"},
                parsers=self.parsers,
            )

    def test_expected_sample_set_must_match(self) -> None:
        with self.assertRaises(official.SpatialEvalOfficialEvaluatorError):
            official.score_predictions(
                [prediction("spatialmap.vqa.1.0", "A. Northeast")],
                {"spatialmap.vqa.1.0": "Northeast"},
                parsers=self.parsers,
                expected_sample_ids=["spatialmap.vqa.9.0"],
            )

    def test_duplicate_prediction_fails(self) -> None:
        with self.assertRaises(official.SpatialEvalOfficialEvaluatorError):
            official.score_predictions(
                [
                    prediction("spatialmap.vqa.1.0", "A. Northeast"),
                    prediction("spatialmap.vqa.1.0", "A. Northeast"),
                ],
                {"spatialmap.vqa.1.0": "Northeast"},
                parsers=self.parsers,
            )

    def test_missing_oracle_fails(self) -> None:
        with self.assertRaises(official.SpatialEvalOfficialEvaluatorError):
            official.score_predictions(
                [prediction("spatialmap.vqa.1.0", "A. Northeast")],
                {},
                parsers=self.parsers,
            )

    def test_task_question_id_parsing(self) -> None:
        self.assertEqual(
            official.task_question_id("spatialmap.vqa.|sa_12.2"), ("spatialmap", 2)
        )
        with self.assertRaises(official.SpatialEvalOfficialEvaluatorError):
            official.task_question_id("no-dots-here")


if __name__ == "__main__":
    unittest.main()
