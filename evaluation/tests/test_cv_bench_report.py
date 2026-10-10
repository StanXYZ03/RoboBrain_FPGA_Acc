import unittest

from evaluation.build_cv_bench_pruning_report import (
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
)


class CVBenchReportTest(unittest.TestCase):
    def test_parser_accepts_choice_without_treating_article_as_answer(self):
        choices = ["cup", "bottle"]
        self.assertEqual(parse_choice("(B) bottle", choices), "B")
        self.assertEqual(parse_choice("Answer: A", choices), "A")
        self.assertEqual(parse_choice("cup", choices), "A")
        self.assertIsNone(parse_choice("a cup is on the table", choices))
        self.assertIsNone(parse_choice("(C)", choices))

    def test_ground_truth_is_not_taken_from_model_output(self):
        self.assertEqual(answer_letter("(B) 3"), "B")
        with self.assertRaises(ValueError):
            answer_letter("3")

    def test_official_source_formula_is_not_raw_accuracy(self):
        rows = [
            {"source": "ADE20K", "source_dataset": "ade", "task": "Count", "dense_correct": True, "dense_answer": "A"},
            {"source": "COCO", "source_dataset": "coco", "task": "Count", "dense_correct": False, "dense_answer": None},
            {"source": "Omni3D", "source_dataset": "omni", "task": "Depth", "dense_correct": True, "dense_answer": "A"},
        ]
        result = score(rows, "dense_correct")
        self.assertAlmostEqual(result["cv_bench_accuracy"], 0.75)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["parse_success_rate"], 2 / 3)

    def test_paired_bootstrap_identity_has_zero_delta(self):
        rows = []
        for source in ("ADE20K", "COCO", "Omni3D"):
            rows.append({
                "source": source, "source_dataset": source, "task": "Count",
                "dense_correct": True, "dense_answer": "A",
                "pruned_correct": True, "pruned_answer": "A",
            })
        self.assertEqual(
            paired_bootstrap_delta(rows, "pruned_correct", "dense_correct", 42, repeats=100),
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
