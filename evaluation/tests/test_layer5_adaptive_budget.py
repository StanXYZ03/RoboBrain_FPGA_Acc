from __future__ import annotations

import unittest

from evaluation.layer5_adaptive_budget import (
    BUDGET_FORMULA,
    compute_adaptive_budget,
    verify_adaptive_budget_fields,
)


class AdaptiveBudgetTest(unittest.TestCase):
    def test_frozen_boundary_cases(self) -> None:
        cases = {
            64: (96, 64, True),
            96: (96, 96, True),
            97: (96, 96, False),
            384: (96, 96, False),
            385: (97, 112, False),
            1400: (350, 352, False),
            3588: (897, 912, False),
        }
        for groups, expected in cases.items():
            with self.subTest(groups=groups):
                got = compute_adaptive_budget(groups)
                self.assertEqual(
                    (got.raw_budget, got.effective_budget, got.identity_bypass),
                    expected,
                )
                self.assertAlmostEqual(got.retention, got.effective_budget / groups)
                self.assertEqual(got.budget_formula, BUDGET_FORMULA)

    def test_invalid_group_count_fails_closed(self) -> None:
        for value in (0, -1, True, 1.5, "96"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    compute_adaptive_budget(value)  # type: ignore[arg-type]

    def test_record_round_trip_and_tamper_rejection(self) -> None:
        record = compute_adaptive_budget(385).as_record()
        self.assertEqual(verify_adaptive_budget_fields(record).effective_budget, 112)
        for field, bad in (
            ("raw_budget", 96),
            ("effective_budget", 96),
            ("requested_budget", 96),
            ("adaptive_retention", 0.25),
            ("identity_bypass", True),
        ):
            with self.subTest(field=field):
                tampered = dict(record)
                tampered[field] = bad
                with self.assertRaises(ValueError):
                    verify_adaptive_budget_fields(tampered)


if __name__ == "__main__":
    unittest.main()
