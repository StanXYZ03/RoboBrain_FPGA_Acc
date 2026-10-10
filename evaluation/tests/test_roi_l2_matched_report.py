from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.build_roi_l2_matched_report import (
    deterministic_strategy_records,
    paired_bootstrap,
)


class RoiL2MatchedReportTest(unittest.TestCase):
    def test_deterministic_strategy_records_selects_one_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            rows = [
                {
                    "sample_id": "sample",
                    "strategy": "roi_only",
                    "repeat": repeat,
                    "output_text": "[1, 2, 3, 4]",
                }
                for repeat in (2, 0, 1)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            records, errors = deterministic_strategy_records(path, "roi_only", 3)
            self.assertEqual(errors, [])
            self.assertEqual(records["sample"]["repeat"], 0)

    def test_paired_bootstrap_keeps_constant_delta(self) -> None:
        result = paired_bootstrap([0.25] * 8, seed=7, iterations=100)
        self.assertEqual(result["mean_delta"], 0.25)
        self.assertEqual(result["ci95_low"], 0.25)
        self.assertEqual(result["ci95_high"], 0.25)


if __name__ == "__main__":
    unittest.main()
