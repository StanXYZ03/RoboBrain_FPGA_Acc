from __future__ import annotations

import unittest

from evaluation.run_hf_hardware_regular_layer5_adaptive_budget import (
    ADAPTIVE_ARM_LABEL,
    BUDGET_MODE,
    SCORE_MODE,
    build_parser,
    budget_for,
)


class AdaptiveRunnerContractTest(unittest.TestCase):
    def test_budget_uses_frozen_formula(self) -> None:
        self.assertEqual(budget_for(385).effective_budget, 112)
        self.assertEqual(budget_for(1400).effective_budget, 352)
        self.assertEqual(SCORE_MODE, "l2")
        self.assertEqual(BUDGET_MODE, "adaptive_layer5_unbounded_real_only")
        self.assertIn("adaptive_budget", ADAPTIVE_ARM_LABEL)

    def test_cli_supports_limit_and_has_no_budget_override(self) -> None:
        parser = build_parser()
        options = {action.dest for action in parser._actions}
        self.assertIn("limit", options)
        self.assertNotIn("budget", options)
        args = parser.parse_args([
            "--model", "/tmp/model",
            "--model-label", "test",
            "--manifest", "/tmp/manifest.jsonl",
            "--output", "/tmp/output.jsonl",
            "--limit", "4",
        ])
        self.assertEqual(args.limit, 4)
        self.assertEqual(args.score_mode, "l2")
        self.assertEqual(args.prune_after_layer, 5)


if __name__ == "__main__":
    unittest.main()
