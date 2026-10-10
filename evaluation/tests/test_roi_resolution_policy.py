import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.evaluate_roi_resolution_policy import (
    balanced_accuracy,
    cheap_roi_features,
    grouped_folds,
    ridge_predict,
)


class RoiResolutionPolicyTest(unittest.TestCase):
    def test_grouped_folds_are_deterministic_and_bounded(self):
        first = grouped_folds([3, 1, 2, 1, 3], folds=2, seed=7)
        second = grouped_folds([1, 2, 3], folds=2, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(set(first), {1, 2, 3})
        self.assertTrue(set(first.values()) <= {0, 1})

    def test_ridge_and_balanced_accuracy(self):
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.asarray([-2.0, -1.0, 1.0, 2.0])
        predicted = ridge_predict(x, y, x, alpha=0.1)
        self.assertEqual(balanced_accuracy(y, predicted), 1.0)

    def test_roi_features_are_finite_and_include_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), color=(10, 20, 30)).save(path)
            features, names = cheap_roi_features(path, (10, 20, 70, 60))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(features), len(names))
        self.assertAlmostEqual(features[names.index("crop_area_fraction")], 0.3)


if __name__ == "__main__":
    unittest.main()
