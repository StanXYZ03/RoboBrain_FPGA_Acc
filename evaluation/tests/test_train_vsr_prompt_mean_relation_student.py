from __future__ import annotations

import unittest
import numpy as np
import torch

from evaluation.train_vsr_prompt_mean_relation_student import donor_map, prompt_mean_scores_numpy, prompt_mean_scores_torch


class RelationStudentTest(unittest.TestCase):
    def test_numpy_and_torch_scorers_match(self):
        rng = np.random.default_rng(7)
        group = rng.normal(size=(5, 4)).astype(np.float32)
        prompt = rng.normal(size=(3, 6)).astype(np.float32)
        wt = rng.normal(size=(2, 6)).astype(np.float32)
        wv = rng.normal(size=(2, 4)).astype(np.float32)
        expected = prompt_mean_scores_numpy(group, prompt, wt, wv)
        actual = prompt_mean_scores_torch(torch.tensor(group), torch.tensor(prompt), torch.tensor(wt), torch.tensor(wv)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_donor_map_never_uses_same_image(self):
        rows = [{"sample_id": "a", "source_image_sha256": "1"}, {"sample_id": "b", "source_image_sha256": "1"}, {"sample_id": "c", "source_image_sha256": "2"}]
        donors = donor_map(rows); by_id = {x["sample_id"]: x for x in rows}
        for row in rows:
            self.assertNotEqual(row["source_image_sha256"], by_id[donors[row["sample_id"]]]["source_image_sha256"])


if __name__ == "__main__": unittest.main()
