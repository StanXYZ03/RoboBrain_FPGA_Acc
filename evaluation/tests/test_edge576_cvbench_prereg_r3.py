#!/usr/bin/env python3
"""CPU-only focused tests for the 2026-10-07 edge576 three-arm r3 preregistration.

Verifies the r3 lifecycle, that every declared SHA256 pin (the 40 r2 frozen
dependencies plus the new runner-adapter / smoke-execution layer) re-verifies on
disk, that r3 freezes the r2 arms / metric / gates byte-for-byte unchanged, that
the supersede reason is *only* that r2 did not pin the new adapter and smoke
layer, that the new single-sample smoke contract and the execution order are
pinned, and that the r2 bytes and the r2 supersede evidence are preserved.

No CUDA, no model weights, no network.
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

R3_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r3.preregistered.json"
)
R3_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r3.json"
)
R3_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r3.md"
)
R3_GENERATOR = ROOT / "scripts" / "preregister_edge576_cvbench_r3_20261007.py"
R3_TEST = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r3.py"

R2_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r2.preregistered.json"
)
R2_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r2.json"
)
R2_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r2.md"
)
R2_GENERATOR = ROOT / "scripts" / "preregister_edge576_cvbench_r2_20261007.py"
R2_TEST = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r2.py"

R2_SUPERSEDED_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r2-preexecution-superseded.json"
)
R2_SUPERSEDED_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r2-preexecution-superseded.md"
)

SMOKE_ENTRY = ROOT / "evaluation" / "run_edge576_cvbench_three_arm_smoke.py"
SMOKE_TEST = (
    ROOT / "evaluation" / "tests" / "test_run_edge576_cvbench_three_arm_smoke.py"
)
RUNNER_CONTRACT = ROOT / "evaluation" / "edge576_cvbench_runner_contract.py"
RUNNER = ROOT / "evaluation" / "run_edge576_cvbench_three_arm.py"
RUNNER_TEST = ROOT / "evaluation" / "tests" / "test_run_edge576_cvbench_three_arm.py"
ADAPTATION_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-three-arm-runner-adaptation.json"
)
ADAPTATION_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-three-arm-runner-adaptation.md"
)

# Frozen r2 evidence that must never change.
R2_CONFIG_SHA = "8f265557bfb86998135b6985023e136d7f397eb8e06eb64551a97308cc958d09"
R2_REPORT_JSON_SHA = "ab2d22659850ea9757563dc84522ef5b50502cb28a788bf9a9e8bb285661d95b"
R2_REPORT_MD_SHA = "a2254d205e80240d9cfe6f1b8a11126f452edbe98de441398091e95a3b5e2b27"
R2_GENERATOR_SHA = "43ecab62deeb0248b7e4fb698cdd6d37aeeb85dd75b507be2dcf2dcbd6ea8430"
R2_TEST_SHA = "94608ed8a7dcca3efb97f38f1d09dacdb97e7ef0743371d0cf08566a38838b42"

EXPECTED_R2_DEPENDENCY_PIN_COUNT = 40
EXPECTED_R3_PIN_COUNT = 54
V2_MECHANISM_ID = "conservative_l2_guarded_rerank_v2"
SMOKE_OUTPUT_DIR = "artifacts/token_selector/2026-10-07-edge576-smoke"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class R3PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(R3_CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(R3_REPORT_JSON.read_text(encoding="utf-8"))
        cls.r2 = json.loads(R2_CONFIG.read_text(encoding="utf-8"))

    def test_status_is_pre_registered_and_not_executed(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertEqual(self.config["arm_status"], "NOT_EXECUTED")
        self.assertFalse(self.config["executed"])
        self.assertEqual(self.config["gpu_runs"], 0)
        self.assertTrue(self.config["future_only"])
        self.assertFalse(self.config["retroactive"])
        self.assertEqual(self.config["revision"], "r3")

    def test_every_pin_reverifies(self) -> None:
        pins = self.config["pins"]
        self.assertEqual(len(pins), EXPECTED_R3_PIN_COUNT)
        for name, entry in pins.items():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), f"pin {name} file missing: {path}")
            self.assertEqual(
                _sha256(path), entry["sha256"], f"pin {name} SHA256 drifted"
            )

    def test_report_config_sha_matches_and_pins_verified(self) -> None:
        self.assertEqual(self.report["config_sha256"], _sha256(R3_CONFIG))
        self.assertEqual(self.report["config"], str(R3_CONFIG.relative_to(ROOT)))
        self.assertTrue(self.report["pins_verified"])
        self.assertEqual(self.report["pin_count"], len(self.config["pins"]))
        self.assertEqual(self.report["pin_count"], EXPECTED_R3_PIN_COUNT)

    def test_generator_sha_is_recorded(self) -> None:
        self.assertTrue(R3_GENERATOR.is_file())
        self.assertEqual(self.config["generator_sha256"], _sha256(R3_GENERATOR))
        self.assertEqual(
            self.config["pins"]["prereg_generator_r3"]["sha256"],
            self.config["generator_sha256"],
        )

    def test_r3_pins_the_r2_supersede_evidence(self) -> None:
        pins = self.config["pins"]
        self.assertEqual(
            pins["r2_preexecution_superseded_json"]["sha256"],
            _sha256(R2_SUPERSEDED_JSON),
        )
        self.assertEqual(
            pins["r2_preexecution_superseded_md"]["sha256"],
            _sha256(R2_SUPERSEDED_MD),
        )
        evidence = json.loads(R2_SUPERSEDED_JSON.read_text(encoding="utf-8"))
        self.assertEqual(evidence["status"], "PREEXECUTION_SUPERSEDED")
        self.assertTrue(evidence["never_executed"])
        self.assertTrue(evidence["must_not_be_run"])
        self.assertTrue(evidence["bytes_preserved"])
        self.assertEqual(evidence["gpu_runs"], 0)

    def test_supersede_reason_is_only_the_missing_pin_layer(self) -> None:
        supersedes = self.config["supersedes"]
        self.assertEqual(supersedes["status"], "PREEXECUTION_SUPERSEDED")
        self.assertTrue(supersedes["never_executed"])
        self.assertTrue(supersedes["must_not_be_run"])
        reason = supersedes["reason"].lower()
        self.assertIn("adapter", reason)
        self.assertIn("smoke", reason)
        self.assertIn("pin", reason)
        self.assertIn("never executed", reason)
        # There is exactly one reason object, and no separate defect list.
        self.assertNotIn("defects", supersedes)
        self.assertEqual(
            supersedes["sha256"], R2_CONFIG_SHA
        )

    def test_r3_freezes_r2_arms_metric_and_gates_unchanged(self) -> None:
        for field in (
            "edge_runtime_domain",
            "shared_arm_contract",
            "arms",
            "frozen_candidate_mechanism",
            "primary_metric",
            "success_gates",
            "manifest_contract",
        ):
            self.assertEqual(
                self.config[field],
                self.r2[field],
                f"r3 changed the frozen r2 field {field!r}",
            )

    def test_three_arms_and_paths_unchanged(self) -> None:
        arms = self.config["arms"]
        self.assertEqual(
            set(arms),
            {"dense_edge576", "corrected_l2_integer_b1_k96", "v2_edge576_b1_k96"},
        )
        for arm, entry in arms.items():
            self.assertEqual(entry["status"], "NOT_EXECUTED")
            self.assertIsNone(entry["output_sha256"])
            self.assertEqual(
                entry["output_path"], self.r2["arms"][arm]["output_path"]
            )
        self.assertEqual(arms["v2_edge576_b1_k96"]["mechanism_id"], V2_MECHANISM_ID)
        self.assertFalse(arms["v2_edge576_b1_k96"]["is_new_mechanism"])

    def test_no_100_percent_parse_gate(self) -> None:
        gate = self.config["success_gates"]["record_completeness"]
        self.assertFalse(gate["requires_100_percent_parse"])
        self.assertTrue(gate["parsed_answer_may_be_null"])
        self.assertTrue(gate["parse_failures_count_as_wrong"])
        self.assertNotIn("parseable", json.dumps(gate).lower())

    def test_execution_order(self) -> None:
        order = self.config["execution_order"]
        ids = [step["id"] for step in order]
        self.assertEqual(
            ids,
            [
                "model_identity",
                "three_arm_single_sample_smoke",
                "dense_edge576_full2638",
                "corrected_l2_integer_b1_k96_full2638",
                "v2_edge576_b1_k96_full2638",
                "official_scoring_and_paired_bootstrap",
            ],
        )
        smoke_step = order[1]
        self.assertTrue(smoke_step["must_all_pass"])
        self.assertTrue(smoke_step["blocks_full2638_until_pass"])
        for step in order:
            self.assertFalse(step.get("executed", False))
        self.assertEqual(
            [step["step"] for step in order], [1, 2, 3, 4, 5, 6]
        )

    def test_smoke_contract_is_pinned(self) -> None:
        contract = self.config["smoke_contract"]
        self.assertEqual(
            contract["entry"],
            "evaluation/run_edge576_cvbench_three_arm_smoke.py",
        )
        self.assertEqual(contract["output_dir"], SMOKE_OUTPUT_DIR)
        self.assertEqual(contract["output_pattern"], "<arm>_<sample_id>.json")
        self.assertTrue(contract["one_sample_per_invocation"])
        self.assertTrue(contract["must_not_touch_full2638_outputs"])
        self.assertTrue(contract["warmup_not_counted_in_record"])
        self.assertTrue(contract["actual_grid_assertion"])
        self.assertFalse(contract["selection_algorithm_copied"])
        self.assertEqual(contract["real_warmup"]["max_new_tokens"], 1)
        self.assertTrue(contract["real_warmup"]["record_discarded"])
        self.assertEqual(contract["smoke_sample_id"], "1838")
        self.assertEqual(contract["smoke_sample_g"], 576)
        self.assertEqual(len(contract["commands"]), 3)
        for command in contract["commands"]:
            self.assertIn("run_edge576_cvbench_three_arm_smoke.py", command)
            self.assertIn("--sample-id 1838", command)
            self.assertIn(SMOKE_OUTPUT_DIR, command)

    def test_new_adapter_and_smoke_pins_are_present(self) -> None:
        pins = self.config["pins"]
        expected = {
            "edge576_runner_contract": RUNNER_CONTRACT,
            "edge576_runner": RUNNER,
            "edge576_runner_test": RUNNER_TEST,
            "runner_adaptation_report_json": ADAPTATION_JSON,
            "runner_adaptation_report_md": ADAPTATION_MD,
            "smoke_entry": SMOKE_ENTRY,
            "smoke_test": SMOKE_TEST,
            "prereg_generator_r3": R3_GENERATOR,
            "prereg_test_r3": R3_TEST,
            "r2_prereg_config": R2_CONFIG,
            "r2_prereg_report_json": R2_REPORT_JSON,
            "r2_prereg_report_md": R2_REPORT_MD,
            "r2_preexecution_superseded_json": R2_SUPERSEDED_JSON,
            "r2_preexecution_superseded_md": R2_SUPERSEDED_MD,
        }
        for name, path in expected.items():
            self.assertIn(name, pins, f"missing pin {name}")
            self.assertEqual(pins[name]["path"], str(path.relative_to(ROOT)))
            self.assertEqual(pins[name]["sha256"], _sha256(path))

    def test_r2_dependency_pins_are_inherited_verbatim(self) -> None:
        r2_pins = self.r2["pins"]
        self.assertEqual(len(r2_pins), EXPECTED_R2_DEPENDENCY_PIN_COUNT)
        r3_pins = self.config["pins"]
        for name, entry in r2_pins.items():
            self.assertIn(name, r3_pins, f"r3 dropped the r2 pin {name}")
            self.assertEqual(r3_pins[name]["path"], entry["path"])
            self.assertEqual(r3_pins[name]["sha256"], entry["sha256"])
            self.assertEqual(_sha256(ROOT / entry["path"]), entry["sha256"])

    def test_model_identity_is_pinned(self) -> None:
        pins = self.config["pins"]
        self.assertIn("model_identity_manifest", pins)
        self.assertEqual(
            pins["model_identity_manifest"]["sha256"],
            "bbfcda188605f54a9c7fea83e7fc2a46eca8ac635c086d23861b294c9950f8e3",
        )
        model = self.config["model_identity"]
        self.assertFalse(model["weights_loaded_in_this_round"])
        self.assertEqual(
            model["default_model_path"],
            "/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B",
        )

    def test_r2_bytes_are_preserved(self) -> None:
        self.assertEqual(_sha256(R2_CONFIG), R2_CONFIG_SHA)
        self.assertEqual(_sha256(R2_REPORT_JSON), R2_REPORT_JSON_SHA)
        self.assertEqual(_sha256(R2_REPORT_MD), R2_REPORT_MD_SHA)
        self.assertEqual(_sha256(R2_GENERATOR), R2_GENERATOR_SHA)
        self.assertEqual(_sha256(R2_TEST), R2_TEST_SHA)

    def test_r3_report_status_and_scope(self) -> None:
        self.assertEqual(self.report["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertEqual(self.report["revision"], "r3")
        self.assertFalse(self.report["executed"])
        self.assertEqual(self.report["gpu_runs"], 0)
        self.assertEqual(self.report["supersedes"]["sha256"], R2_CONFIG_SHA)
        self.assertIn("execution_order", self.report)
        self.assertIn("smoke_contract", self.report)
        self.assertTrue(R3_REPORT_MD.is_file())
        self.assertGreater(R3_REPORT_MD.stat().st_size, 0)

    def test_report_json_sha_is_pinned_by_config(self) -> None:
        # The r3 report is not self-referential, but the config must pin the
        # superseded r2 report and its own generator/test.
        pins = self.config["pins"]
        self.assertEqual(pins["r2_prereg_report_json"]["sha256"], R2_REPORT_JSON_SHA)
        self.assertEqual(pins["r2_prereg_report_md"]["sha256"], R2_REPORT_MD_SHA)

    def test_runner_adaptation_entry_status(self) -> None:
        entry = self.config["runner_adaptation_entry"]
        self.assertEqual(entry["smoke_entry"], str(SMOKE_ENTRY.relative_to(ROOT)))
        self.assertIn("IMPLEMENTED", entry["status"])
        self.assertFalse(entry["selection_algorithm_copied"])

    def test_shared_arm_contract_matches_r2(self) -> None:
        shared = self.config["shared_arm_contract"]
        self.assertEqual(shared["processor"]["min_pixels"], 65536)
        self.assertEqual(shared["processor"]["max_pixels"], 589824)
        self.assertFalse(shared["do_sample"])
        self.assertEqual(shared["max_new_tokens"], 16)
        self.assertEqual(shared, self.r2["shared_arm_contract"])

    def test_forbidden_claims_include_r2_not_executed(self) -> None:
        claims = " ".join(self.config["forbidden_claims"]).lower()
        self.assertIn("r2", claims)
        self.assertIn("never executed", claims)


if __name__ == "__main__":
    unittest.main()
