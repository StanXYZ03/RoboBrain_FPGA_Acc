import unittest

from evaluation.run_cv_bench_refinement import detail_messages, rotated_strategies, validate_resume
from evaluation.build_cv_refinement_report import index_unique, summarize_latency


class RefinementTests(unittest.TestCase):
    def test_detail_keeps_original_question(self):
        prompt = "How many cups? (A) 1 (B) 2"
        content = detail_messages(prompt, [0, 100, 600, 700])[0]["content"]
        self.assertEqual(sum(row["type"] == "image" for row in content), 2)
        self.assertTrue(content[-1]["text"].endswith(prompt))
        self.assertIn("not another scene", content[-1]["text"])

    def test_rotation(self):
        values = ["dense", "global256", "cosine_global_roi"]
        self.assertEqual([rotated_strategies(values, 0, r)[0] for r in range(3)], values)

    def test_resume_rejects_duplicates_and_wrong_signature(self):
        row = {"sample_id": "0", "strategy": "dense", "repeat": 0, "run": {"v": 1}}
        expected = {("0", "dense", 0)}
        self.assertEqual(validate_resume([row], {"v": 1}, expected), expected)
        with self.assertRaises(ValueError):
            validate_resume([row, row], {"v": 1}, expected)
        with self.assertRaises(ValueError):
            validate_resume([row], {"v": 2}, expected)

    def test_accuracy_requires_exact_ids(self):
        row = {"sample_id": "0"}
        self.assertEqual(index_unique([row], {"0"}), {"0": row})
        with self.assertRaises(ValueError):
            index_unique([row, row], {"0"})
        with self.assertRaises(ValueError):
            index_unique([row], {"1"})

    def test_latency_checks_identity_and_repeat_determinism(self):
        rows = [{"sample_id": "0", "strategy": method, "repeat": repeat,
                 "output_text": "A", "effective_crop_1000": None,
                 "model_end_to_end_ms": 10.0, "output_tokens": 1,
                 "peak_cuda_mib": 100.0}
                for method in ("dense", "global256", "cosine_global_roi")
                for repeat in range(3)]
        summary, identities = summarize_latency(rows, {"0"}, {"0": {"output_text": "A"}})
        self.assertEqual(identities, 3)
        self.assertEqual(summary["global256"]["model_mean_speedup_vs_dense"], 1.0)
        with self.assertRaises(ValueError):
            summarize_latency(rows, {"0"}, {"0": {"output_text": "B"}})
        rows[-1]["output_text"] = "B"
        with self.assertRaises(ValueError):
            summarize_latency(rows, {"0"}, {"0": {"output_text": "A"}})
