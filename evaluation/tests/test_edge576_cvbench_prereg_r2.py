#!/usr/bin/env python3
"""CPU-only focused tests for the 2026-10-07 edge576 three-arm r2 preregistration.

Verifies the r2 lifecycle, that every declared SHA256 pin re-verifies on disk,
that the third arm is the existing frozen ``conservative_l2_guarded_rerank_v2``
mechanism (not a null slot and not a new mechanism), that the record-completeness
gate no longer demands 100% parsing, that the corrected auditor's ``--model``
path defect is fixed in the pinned re-audit report, and that the superseded r1
bytes are preserved.  No CUDA, no model weights, no network.
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
    / "2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json"
)
REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r2.json"
)
AUDIT_R2_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cv-bench-manifest-audit-r2.json"
)

# Frozen r1 evidence that must never change.
R1_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development.preregistered.json"
)
R1_REPORT_JSON = ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg.json"
R1_GENERATOR = ROOT / "scripts" / "preregister_edge576_cvbench_20261007.py"
R1_CONFIG_SHA = "8958cbd6f61589217b73e983e0f57257f0cf9bc951c5adbd8506df5d6156b3f3"
R1_REPORT_JSON_SHA = "c728c3125ffa5708abef0aa6d6fd050611ce8f864cd7bbbda3d85ad13b419afa"
R1_GENERATOR_SHA = "c01776756d37d45307fae9745c610d44ad2179d638d694a0f86372d343d06c57"
FROZEN_R1_AUDITOR_SHA = "45dcd04976e5c5c8acaf232989f219b1817b5a13fb87bfbe27dd4cbfe81ea683"

EXPECTED_MODEL_PATH = "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B"
V2_MECHANISM_ID = "conservative_l2_guarded_rerank_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class R2PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))

    def test_status_is_pre_registered_and_not_executed(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertEqual(self.config["arm_status"], "NOT_EXECUTED")
        self.assertFalse(self.config["executed"])
        self.assertEqual(self.config["gpu_runs"], 0)
        self.assertTrue(self.config["future_only"])
        self.assertFalse(self.config["retroactive"])

    def test_every_pin_reverifies(self) -> None:
        pins = self.config["pins"]
        self.assertGreaterEqual(len(pins), 25)
        for name, entry in pins.items():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), f"pin {name} file missing: {path}")
            self.assertEqual(
                _sha256(path), entry["sha256"], f"pin {name} SHA256 drifted"
            )

    def test_report_config_sha_matches(self) -> None:
        self.assertEqual(self.report["config_sha256"], _sha256(CONFIG))
        self.assertEqual(self.report["config"], str(CONFIG.relative_to(ROOT)))
        self.assertTrue(self.report["pins_verified"])
        self.assertEqual(self.report["pin_count"], len(self.config["pins"]))

    def test_generator_sha_is_recorded(self) -> None:
        generator = ROOT / "scripts" / "preregister_edge576_cvbench_r2_20261007.py"
        self.assertTrue(generator.is_file())
        self.assertEqual(self.config["generator_sha256"], _sha256(generator))
        self.assertEqual(
            self.config["pins"]["prereg_generator_r2"]["sha256"],
            self.config["generator_sha256"],
        )

    def test_third_arm_is_the_existing_v2_mechanism(self) -> None:
        arm = self.config["arms"]["v2_edge576_b1_k96"]
        self.assertEqual(arm["mechanism_id"], V2_MECHANISM_ID)
        self.assertIsNotNone(arm["mechanism_id"])
        self.assertFalse(arm["is_new_mechanism"])
        self.assertEqual(arm["status"], "NOT_EXECUTED")
        self.assertIsNone(arm["output_sha256"])
        self.assertIn("v2_edge576_k96", arm["output_path"])
        self.assertEqual(arm["cutpoint"], "B1_after_vision_block_0")
        self.assertEqual(arm["runner"], "evaluation/run_hf_conservative_l2_guarded_rerank.py")
        self.assertTrue(arm["reuses_native_resolution_v2_mechanism"])
        self.assertTrue(arm["native_resolution_result_not_rewritten"])
        self.assertEqual(arm["selector_budget"]["K"], 96)
        self.assertEqual(arm["selector_budget"]["max_swaps_S"], 16)

    def test_three_arms_share_the_locked_input(self) -> None:
        arms = self.config["arms"]
        self.assertEqual(
            set(arms),
            {"dense_edge576", "corrected_l2_integer_b1_k96", "v2_edge576_b1_k96"},
        )
        shared = self.config["shared_arm_contract"]
        self.assertEqual(
            shared["manifest"], self.config["pins"]["edge576_manifest"]["path"]
        )
        self.assertEqual(
            shared["manifest_sha256"], self.config["pins"]["edge576_manifest"]["sha256"]
        )
        self.assertFalse(shared["do_sample"])
        self.assertEqual(shared["max_new_tokens"], 16)
        self.assertEqual(shared["processor"]["max_pixels"], 589824)
        self.assertEqual(shared["processor"]["min_pixels"], 65536)

    def test_record_completeness_does_not_demand_full_parse(self) -> None:
        gate = self.config["success_gates"]["record_completeness"]
        self.assertTrue(gate["all_2638_unique_complete_outputs"])
        self.assertEqual(gate["sample_count"], 2638)
        self.assertEqual(gate["unique_sample_id_count"], 2638)
        self.assertTrue(gate["parsed_answer_may_be_null"])
        self.assertTrue(gate["parse_failures_count_as_wrong"])
        self.assertFalse(gate["requires_100_percent_parse"])
        for field in (
            "parse_success_count",
            "parse_failure_count",
            "parse_success_rate",
            "parse_failure_rate",
        ):
            self.assertIn(field, gate["must_report"])
        # No residual "all ... parseable" wording from r1.
        self.assertNotIn("parseable", json.dumps(gate).lower())

    def test_honest_parse_reporting_has_no_100pct_gate(self) -> None:
        gate = self.config["success_gates"]["honest_parse_reporting"]
        self.assertEqual(gate["source_field"], "parsed_answer_key only")
        self.assertTrue(gate["parsed_answer_may_be_null"])
        self.assertTrue(gate["parse_failures_count_as_wrong"])
        self.assertFalse(gate["requires_100_percent_parse"])
        self.assertTrue(gate["aliasing_correct_key_forbidden"])

    def test_success_gates(self) -> None:
        gates = self.config["success_gates"]
        retention = gates["quality_retention_vs_same_input_dense_edge576"]
        self.assertEqual(retention["threshold"], 0.90)
        self.assertEqual(retention["comparison"], ">=")
        bootstrap = gates["relative_to_corrected_integer_l2"]["paired_bootstrap"]
        self.assertEqual(bootstrap["strata"], "task::source_dataset")
        self.assertEqual(bootstrap["lower_bound"], ">=0")
        self.assertGreaterEqual(bootstrap["repeats"], 5000)
        self.assertEqual(bootstrap["seed"], 20261007)
        layout = gates["layout_contract"]
        self.assertEqual(layout["K"], 96)
        self.assertEqual(layout["real_slots"], 87)
        self.assertEqual(layout["context_slots"], 9)
        self.assertTrue(gates["no_label_leakage"])
        self.assertTrue(gates["official_cv_bench_accuracy"]["required"])

    def test_primary_metric_is_official_cv_bench_accuracy(self) -> None:
        metric = self.config["primary_metric"]
        self.assertEqual(metric["name"], "cv_bench_accuracy")
        self.assertEqual(
            metric["module_sha256"],
            self.config["pins"]["cv_bench_metric_v2"]["sha256"],
        )
        self.assertNotEqual(metric["correct_key"], metric["parsed_answer_key"])
        self.assertTrue(metric["parse_failures_count_as_wrong"])

    def test_frozen_v2_mechanism_is_not_retuned(self) -> None:
        mech = self.config["frozen_candidate_mechanism"]
        self.assertEqual(mech["mechanism_id"], V2_MECHANISM_ID)
        self.assertEqual(mech["utility_formula"], "semantic_q15 + corrected_l2_q15")
        self.assertTrue(mech["anchors_frozen"])
        self.assertEqual(mech["K_remains_exactly_96"], True)
        self.assertEqual(mech["real_group_count"], 87)
        self.assertEqual(mech["context_group_count"], 9)
        self.assertTrue(mech["no_labels_or_answer_tokens_or_gradients"])

    def test_development_retest_and_non_retroactive_statements(self) -> None:
        self.assertEqual(self.config["measurement_class"], "development")
        self.assertIn("EXISTING", self.config["development_retest_statement"])
        self.assertIn("No new mechanism", self.config["development_retest_statement"])
        self.assertIn("DEVELOPMENT_STOP", self.config["non_retroactive_statement"])
        self.assertIn(
            "stress test",
            self.config["edge_runtime_domain"]["high_resolution_policy"],
        )

    def test_r1_is_superseded_and_never_executed(self) -> None:
        supersedes = self.config["supersedes"]
        self.assertEqual(supersedes["status"], "PREEXECUTION_SUPERSEDED")
        self.assertTrue(supersedes["never_executed"])
        self.assertTrue(supersedes["must_not_be_run"])
        self.assertEqual(len(supersedes["defects"]), 3)
        # The pinned r1 bytes really are the frozen r1 bytes.
        self.assertEqual(supersedes["sha256"], R1_CONFIG_SHA)
        self.assertEqual(_sha256(R1_CONFIG), R1_CONFIG_SHA)

    def test_frozen_r1_bytes_unchanged(self) -> None:
        self.assertEqual(_sha256(R1_CONFIG), R1_CONFIG_SHA)
        self.assertEqual(_sha256(R1_REPORT_JSON), R1_REPORT_JSON_SHA)
        self.assertEqual(_sha256(R1_GENERATOR), R1_GENERATOR_SHA)
        # r1 pinned the pre-fix auditor; the corrected auditor differs.
        self.assertEqual(
            self.config["pins"]["auditor"]["sha256"] != FROZEN_R1_AUDITOR_SHA, True
        )

    def test_pinned_reaudit_report_records_actual_model_path(self) -> None:
        audit = json.loads(AUDIT_R2_JSON.read_text(encoding="utf-8"))
        self.assertEqual(audit["status"], "EDGE576_MANIFEST_AUDIT_PASS")
        self.assertEqual(audit["processor_model_path"], EXPECTED_MODEL_PATH)
        self.assertEqual(
            self.config["model_identity"]["processor_audit_model_path"],
            EXPECTED_MODEL_PATH,
        )
        self.assertTrue(audit["processor_grid_verification_requested"])
        self.assertEqual(audit["processor_grids_reverified"], 2638)
        self.assertEqual(
            self.config["pins"]["edge576_audit_r2_report"]["sha256"],
            _sha256(AUDIT_R2_JSON),
        )

    def test_v2_chain_pins_are_present(self) -> None:
        pins = self.config["pins"]
        for key in (
            "v2_mechanism_core",
            "v2_integration",
            "v2_runner",
            "v2_summarizer",
            "v2_test_mechanism",
            "v2_test_integration",
            "v2_test_runner",
            "v2_test_summarizer",
            "v2_projection_npz",
            "v2_wv_int8",
            "v2_wv_per_tensor_scale",
            "integer_token_selector_core",
            "integer_corrected_l2_core",
            "integer_b1_integration_core",
            "integer_current_selector_core",
            "v1_layout_rebuild_core",
        ):
            self.assertIn(key, pins, f"missing v2 chain pin {key}")

    def test_runner_adaptation_entry_points_to_existing_runner(self) -> None:
        entry = self.config["runner_adaptation_entry"]
        self.assertEqual(
            entry["v2_candidate"],
            "evaluation/run_hf_conservative_l2_guarded_rerank.py",
        )
        self.assertEqual(entry["required_edge576_inputs"]["min_pixels"], 65536)
        self.assertEqual(entry["required_edge576_inputs"]["max_pixels"], 589824)
        self.assertTrue(
            entry["required_edge576_inputs"]["assert_image_grid_thw_equals_manifest"]
        )
        self.assertIn("NO_MODEL_RUN", entry["status"])


if __name__ == "__main__":
    unittest.main()
