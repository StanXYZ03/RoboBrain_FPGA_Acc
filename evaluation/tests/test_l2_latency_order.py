import unittest

from evaluation.run_l2_interleaved_latency import completed_keys, configuration_order


class LatencyOrderTest(unittest.TestCase):
    def test_rotation_balances_first_configuration(self):
        orders = [configuration_order(5, sample, 0) for sample in range(5)]
        self.assertEqual(sorted(order[0] for order in orders), list(range(5)))
        for order in orders:
            self.assertEqual(sorted(order), list(range(5)))

    def test_resume_rejects_duplicates_and_wrong_model(self):
        meta = {"manifest_sha256": "a", "model_path": "b", "selector_source_sha256": "c", "repeats": 3, "retentions": [1, 0.5]}
        record = {"sample_id": "1", "repeat_index": 0, "requested_retention": 0.5, "run": meta}
        self.assertEqual(completed_keys([record], meta), {("1", 0, 0.5)})
        with self.assertRaises(ValueError):
            completed_keys([record, record], meta)
        with self.assertRaises(ValueError):
            completed_keys([record], {**meta, "model_path": "different"})


if __name__ == "__main__":
    unittest.main()
