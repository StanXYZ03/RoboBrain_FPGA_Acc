import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_partition import analyze, boundary_elements, packed_bytes


class PartitionAnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((ROOT / "model_contract.json").read_text(encoding="utf-8"))

    def test_kv_elements_exact(self):
        boundaries = boundary_elements(self.contract)
        self.assertEqual(boundaries["prefill_to_decoder_kv"]["elements"], 84197376)

    def test_q8_packed_bytes_include_scales(self):
        self.assertEqual(packed_bytes(32, 34, 32), 34)
        self.assertEqual(packed_bytes(33, 34, 32), 36)

    def test_1gbe_kv_rejected(self):
        result = analyze(self.contract)
        self.assertEqual(
            result["decision"]["ordinary_1gbe_prefill_to_decoder_kv"],
            "REJECT_FOR_PER_FRESH_FRAME",
        )
        rows = [
            row for row in result["boundaries"]
            if row["boundary"] == "prefill_to_decoder_kv" and row["format"] == "BF16_or_FP16"
        ]
        link = next(x for x in rows[0]["links"] if x["link"] == "1GbE_80pct_scenario")
        self.assertGreater(link["transfer_lower_bound_ms"], 1600.0)
        self.assertFalse(link["within_20pct_period_budget"]["5Hz"])

    def test_decoder_peak_is_below_five_token_s(self):
        result = analyze(self.contract)
        cases = {
            x["bandwidth"]: x
            for x in result["decoder_weight_stream"]["bandwidth_cases"]
        }
        self.assertLess(
            cases["DDR4_2133_line_rate"]["full_vocabulary_upper_bound_token_per_second"],
            5.0,
        )
        reduction = cases["DDR4_2133_85pct_scenario"]["targets"]["5_token_per_second"][
            "required_reduction_from_full"
        ]
        self.assertGreater(reduction, 0.30)


if __name__ == "__main__":
    unittest.main()
