import unittest

from evaluation.build_l2_retention_curve import paired_accuracy_ci


class CurveTest(unittest.TestCase):
    def test_identity_ci_is_zero(self):
        rows = [{"source": source, "source_dataset": source, "task": "Count", "pred_correct": True, "dense_correct": True} for source in ("ADE20K", "COCO", "Omni3D")]
        self.assertEqual(paired_accuracy_ci(rows), [0.0, 0.0])

    def test_constant_ade_loss_has_official_quarter_weight(self):
        rows = [{"source": source, "source_dataset": source, "task": "Count", "pred_correct": source != "ADE20K", "dense_correct": True} for source in ("ADE20K", "COCO", "Omni3D")]
        self.assertEqual(paired_accuracy_ci(rows), [-0.25, -0.25])


if __name__ == "__main__":
    unittest.main()
