import unittest

import torch

from evaluation.extract_roi_vision_prefix_features import fixed_projection


class RoiVisionPrefixFeaturesTest(unittest.TestCase):
    def test_projection_is_deterministic_and_scaled(self):
        first = fixed_projection(8, 3, 17)
        second = fixed_projection(8, 3, 17)
        other = fixed_projection(8, 3, 18)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(torch.allclose(first.abs(), torch.full_like(first, 1 / (8 ** 0.5))))


if __name__ == "__main__":
    unittest.main()
