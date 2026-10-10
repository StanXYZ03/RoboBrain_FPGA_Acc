from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.train_vsr_caption_relation_student import apply_caption_spans


class CaptionRelationStudentTest(unittest.TestCase):
    def test_apply_caption_spans_slices_only_caption_tokens(self):
        samples = [{
            "sample_id": "sample-1",
            "prompt_embeddings": np.arange(30, dtype=np.float32).reshape(5, 6),
        }]
        payload = {
            "spans": {
                "sample-1": {"start": 1, "end_exclusive": 4, "full_token_count": 5},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spans.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            stats = apply_caption_spans(samples, path)

        np.testing.assert_array_equal(
            samples[0]["prompt_embeddings"],
            np.arange(30, dtype=np.float32).reshape(5, 6)[1:4],
        )
        self.assertEqual(stats, {"min": 3, "max": 3, "mean": 3.0})

    def test_apply_caption_spans_rejects_prompt_length_mismatch(self):
        samples = [{
            "sample_id": "sample-1",
            "prompt_embeddings": np.zeros((5, 6), dtype=np.float32),
        }]
        payload = {
            "spans": {
                "sample-1": {"start": 1, "end_exclusive": 4, "full_token_count": 6},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spans.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prompt token count mismatch"):
                apply_caption_spans(samples, path)


if __name__ == "__main__":
    unittest.main()
