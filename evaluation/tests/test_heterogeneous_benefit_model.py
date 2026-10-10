from __future__ import annotations

import unittest

from evaluation.heterogeneous_benefit_model import estimate, serialization_ms


class HeterogeneousBenefitModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.full = {
            "vision_prefix_ms": 2.0,
            "vision_pruned_ms": 8.0,
            "vision_other_ms": 1.0,
            "llm_prefill_ms": 9.0,
        }
        self.compact = {
            "vision_prefix_ms": 2.0,
            "vision_pruned_ms": 4.0,
            "vision_other_ms": 1.0,
            "llm_prefill_ms": 6.0,
        }
        self.link = {
            "gbps": 10.0,
            "efficiency": 0.8,
            "fixed_one_way_ms": 0.0,
            "fixed_round_trip_ms": 0.0,
        }
        self.power = {
            "host_compute": 30.0,
            "accelerator_compute": 5.0,
            "link_active_total": 3.0,
        }

    def test_serialization_uses_decimal_link_rate(self) -> None:
        self.assertAlmostEqual(serialization_ms(1_000_000, 10.0, 0.8), 1.0)

    def test_round_trip_is_software_plus_link_lower_bound(self) -> None:
        row = estimate(
            full=self.full,
            compact=self.compact,
            one_way_bytes=1_000_000,
            round_trip_bytes=2_000_000,
            link=self.link,
            prefix_speedup=4.0,
            power=self.power,
        )
        self.assertAlmostEqual(row["software_compact_ms"], 13.0)
        self.assertAlmostEqual(row["round_trip_ms"], 15.0)
        self.assertGreater(row["round_trip_delta_vs_software_ms"], 0)
        self.assertAlmostEqual(row["break_even_prefix_speedup_vs_software"], 2.0)
        self.assertAlmostEqual(row["required_link_gbps_vs_software"], 20.0 / 3.0)

    def test_faster_prefix_cannot_make_resident_slower(self) -> None:
        slow = estimate(
            full=self.full, compact=self.compact, one_way_bytes=0,
            round_trip_bytes=0, link=self.link, prefix_speedup=1.0, power=self.power,
        )
        fast = estimate(
            full=self.full, compact=self.compact, one_way_bytes=0,
            round_trip_bytes=0, link=self.link, prefix_speedup=4.0, power=self.power,
        )
        self.assertLess(fast["resident_ms"], slow["resident_ms"])


if __name__ == "__main__":
    unittest.main()
