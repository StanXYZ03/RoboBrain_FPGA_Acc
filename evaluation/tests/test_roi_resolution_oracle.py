import unittest

from evaluation.analyze_roi_resolution_oracle import analyze


def prediction(sample_id, tokens, text):
    return {
        "sample_id": sample_id,
        "final_context_visual_tokens": tokens,
        "output_text": text,
    }


class RoiResolutionOracleTest(unittest.TestCase):
    def test_active_escalation_charges_both_encodes(self):
        samples = [
            {
                "sample_id": "g1",
                "task": "grounding",
                "ground_truth": {"bbox_1000": [0, 0, 100, 100]},
                "stratification": {"target_area_bin": "small"},
            },
            {
                "sample_id": "g2",
                "task": "grounding",
                "ground_truth": {"bbox_1000": [0, 0, 100, 100]},
                "stratification": {"target_area_bin": "small"},
            },
        ]
        baseline = {
            "g1": prediction("g1", 100, "[0, 0, 100, 100]"),
            "g2": prediction("g2", 100, "[0, 0, 100, 100]"),
        }
        low = {
            "g1": prediction("g1", 20, "[0, 0, 50, 50]"),
            "g2": prediction("g2", 20, "[0, 0, 100, 100]"),
        }
        high = {
            "g1": prediction("g1", 40, "[0, 0, 100, 100]"),
            "g2": prediction("g2", 40, "[0, 0, 50, 50]"),
        }

        analyses, rows = analyze(samples, baseline, low, high)
        grounding = analyses["grounding"]
        self.assertEqual(grounding["action_complementarity"]["high_better"], 1)
        self.assertEqual(grounding["action_complementarity"]["high_worse"], 1)
        one_shot = grounding["policies"]["revealed_one_shot_oracle"]
        active = grounding["policies"]["revealed_active_escalation_oracle"]
        self.assertEqual(one_shot["mean_final_context_visual_tokens"], 30)
        self.assertEqual(one_shot["mean_cumulative_processed_visual_tokens"], 30)
        self.assertEqual(active["mean_final_context_visual_tokens"], 30)
        self.assertEqual(active["mean_cumulative_processed_visual_tokens"], 40)
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
