#!/usr/bin/env python3
"""CPU-only focused tests for the 2026-10-07 edge576 three-arm preregistration.

Verifies the preregistration status/lifecycle, that every declared SHA256 pin
re-verifies on disk, the frozen edge576 manifest/summary linkage, the shared
decode/processor contract and the success gates.  No CUDA, no model weights, no
network, no missing-output probe of a real default output path.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development.preregistered.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_status_is_pre_registered_and_not_executed(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertFalse(self.config["executed"])
        self.assertEqual(self.config["gpu_runs"], 0)
        self.assertTrue(self.config["future_only"])
        self.assertFalse(self.config["retroactive"])

    def test_every_pin_reverifies(self) -> None:
        pins = self.config["pins"]
        self.assertGreaterEqual(len(pins), 10)
        for name, entry in pins.items():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), f"pin {name} file missing: {path}")
            self.assertEqual(
                _sha256(path), entry["sha256"], f"pin {name} SHA256 drifted"
            )

    def test_manifest_and_summary_linkage(self) -> None:
        manifest = self.config["pins"]["edge576_manifest"]
        summary_path = ROOT / self.config["pins"]["edge576_summary"]["path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["manifest"]["sha256"], manifest["sha256"])
        self.assertTrue(summary["all_g_within_budget"])
        self.assertEqual(summary["sample_count"], 2638)
        self.assertEqual(summary["unique_sample_id_count"], 2638)
        self.assertLessEqual(summary["g_max"], 576)

    def test_three_arms_share_the_locked_input(self) -> None:
        arms = self.config["arms"]
        self.assertEqual(
            set(arms),
            {"dense_edge576", "corrected_l2_integer_b1_k96", "next_candidate_b1_k96"},
        )
        shared = self.config["shared_arm_contract"]
        self.assertEqual(
            shared["manifest"], self.config["pins"]["edge576_manifest"]["path"]
        )
        self.assertEqual(shared["manifest_sha256"], self.config["pins"]["edge576_manifest"]["sha256"])
        self.assertFalse(shared["do_sample"])
        self.assertEqual(shared["max_new_tokens"], 16)
        self.assertEqual(shared["processor"]["max_pixels"], 589824)
        self.assertEqual(shared["processor"]["min_pixels"], 65536)

    def test_next_candidate_is_unselected(self) -> None:
        candidate = self.config["arms"]["next_candidate_b1_k96"]
        self.assertIsNone(candidate["mechanism_id"])
        self.assertIn("TO_BE_SELECTED", candidate["status"])
        self.assertEqual(candidate["output_sha256"], None)

    def test_success_gates(self) -> None:
        gates = self.config["success_gates"]
        self.assertEqual(gates["quality_retention"]["threshold"], 0.90)
        self.assertEqual(gates["quality_retention"]["comparison"], ">=")
        bootstrap = gates["relative_to_corrected_integer_l2"]["paired_bootstrap"]
        self.assertEqual(bootstrap["strata"], "task::source_dataset")
        self.assertEqual(bootstrap["lower_bound"], ">=0")
        self.assertGreaterEqual(bootstrap["repeats"], 5000)
        self.assertTrue(gates["no_label_leakage"])
        self.assertTrue(gates["honest_parse_gate"]["aliasing_correct_key_forbidden"])

    def test_primary_metric_is_official_cv_bench_accuracy(self) -> None:
        metric = self.config["primary_metric"]
        self.assertEqual(metric["name"], "cv_bench_accuracy")
        self.assertEqual(
            metric["module_sha256"],
            self.config["pins"]["cv_bench_metric_v2"]["sha256"],
        )
        self.assertNotEqual(metric["correct_key"], metric["parsed_answer_key"])
        self.assertTrue(metric["parse_failures_count_as_wrong"])

    def test_development_and_stress_statements(self) -> None:
        self.assertEqual(self.config["measurement_class"], "development")
        self.assertIn("development set only", self.config["development_set_statement"])
        self.assertIn(
            "DEVELOPMENT_STOP", self.config["non_retroactive_statement"]
        )
        self.assertIn(
            "stress test",
            self.config["edge_runtime_domain"]["high_resolution_policy"],
        )


if __name__ == "__main__":
    unittest.main()
