#!/usr/bin/env python3
"""CPU-only focused tests for the 2026-10-07 edge576 three-arm r4 preregistration.

Verifies the r4 lifecycle, that every declared SHA256 pin (the r3 frozen
dependencies plus the refreshed execution layer and the new r3 failure/supersede
evidence) re-verifies on disk, that r4 freezes the r3 arms / metric / gates
byte-for-byte unchanged, that the supersede reason is *only* the dense
``output_token_ids`` contract failure, that the new token contract (optional
ids, mandatory real ``output_token_count``) is pinned and enforced, and that the
r3 bytes and evidence are preserved.

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

from evaluation import edge576_cvbench_runner_contract as contract  # noqa: E402

R4_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-07-edge576-cvbench-three-arm-development-r4.preregistered.json"
)
R4_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r4.json"
)
R4_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cvbench-prereg-r4.md"
)
R4_GENERATOR = ROOT / "scripts" / "preregister_edge576_cvbench_r4_20261007.py"
R4_TEST = ROOT / "evaluation" / "tests" / "test_edge576_cvbench_prereg_r4.py"

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

R3_FAILURE_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.json"
)
R3_FAILURE_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-first-gpu-attempt-failure.md"
)
R3_SUPERSEDED_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-superseded.json"
)
R3_SUPERSEDED_MD = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-07-edge576-cvbench-r3-superseded.md"
)

RUNNER_CONTRACT = ROOT / "evaluation" / "edge576_cvbench_runner_contract.py"
SMOKE_ENTRY = ROOT / "evaluation" / "run_edge576_cvbench_three_arm_smoke.py"

# Frozen r3 evidence that must never change.
R3_CONFIG_SHA = "e7ff56c98a39f764deb6cd6b099211f5da45025ce6958b44e8f72c641ae83eed"
R3_REPORT_JSON_SHA = "42a4aabddd8895a2b502befff2052056ad757af80608a106f8f5ad76e3158c49"
R3_REPORT_MD_SHA = "ce3561e2635185375275e1cea22e1e6bd97ee233f5710ca206632f2bc79d9051"
R3_GENERATOR_SHA = "03ed9f42862f90b664e7a9f3fdc8732690831669de48e00e0c2ace26d38f3eaa"
R3_TEST_SHA = "632f988a0f07f86ab697823e97d97f01e5791491ebb335541b8364440321713e"

# r3 execution-layer bytes that the token-contract fix intentionally replaces.
R3_DRIFTED_PINS = {
    "edge576_runner_contract": (
        "fdea7df65556ef47c297a8082d9aa1238fcb645bdb1e5910d02f5453107590db"
    ),
    "edge576_runner": (
        "b1fb2ca1bc67af4355bfe6f0ce52725adf788f3339e77f5c6ea4188788e3eb85"
    ),
    "edge576_runner_test": (
        "6d4f513568faccd005ded47e6b963d89b783194b84411b5c80d25fabf700fa55"
    ),
    "smoke_entry": (
        "3c956d7ea9a64a52dc6affbec0c85ae032b44ce58197c8b166d89480dc2ec255"
    ),
    "smoke_test": (
        "458a45101d8ca2a861b4a4d7f507e650402bee00e164e5991ea0ca1c508d1068"
    ),
}

EXPECTED_R3_PIN_COUNT = 54
EXPECTED_R4_PIN_COUNT = 63
V2_MECHANISM_ID = "conservative_l2_guarded_rerank_v2"
SMOKE_OUTPUT_DIR = "artifacts/token_selector/2026-10-07-edge576-smoke"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class R4PreregistrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(R4_CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(R4_REPORT_JSON.read_text(encoding="utf-8"))
        cls.r3 = json.loads(R3_CONFIG.read_text(encoding="utf-8"))

    def test_status_is_pre_registered_and_not_executed(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertEqual(self.config["arm_status"], "NOT_EXECUTED")
        self.assertFalse(self.config["executed"])
        self.assertEqual(self.config["gpu_runs"], 0)
        self.assertTrue(self.config["future_only"])
        self.assertFalse(self.config["retroactive"])
        self.assertEqual(self.config["revision"], "r4")

    def test_every_pin_reverifies(self) -> None:
        pins = self.config["pins"]
        self.assertEqual(len(pins), EXPECTED_R4_PIN_COUNT)
        for name, entry in pins.items():
            path = ROOT / entry["path"]
            self.assertTrue(path.is_file(), f"pin {name} file missing: {path}")
            self.assertEqual(
                _sha256(path), entry["sha256"], f"pin {name} SHA256 drifted"
            )

    def test_report_config_sha_matches_and_pins_verified(self) -> None:
        self.assertEqual(self.report["config_sha256"], _sha256(R4_CONFIG))
        self.assertEqual(self.report["config"], str(R4_CONFIG.relative_to(ROOT)))
        self.assertTrue(self.report["pins_verified"])
        self.assertEqual(self.report["pin_count"], len(self.config["pins"]))
        self.assertEqual(self.report["pin_count"], EXPECTED_R4_PIN_COUNT)

    def test_generator_sha_is_recorded(self) -> None:
        self.assertTrue(R4_GENERATOR.is_file())
        self.assertEqual(self.config["generator_sha256"], _sha256(R4_GENERATOR))
        self.assertEqual(
            self.config["pins"]["prereg_generator_r4"]["sha256"],
            self.config["generator_sha256"],
        )

    def test_supersede_reason_is_only_the_dense_token_contract(self) -> None:
        supersedes = self.config["supersedes"]
        self.assertEqual(
            supersedes["status"], "SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE"
        )
        self.assertTrue(supersedes["executed"])
        self.assertEqual(supersedes["gpu_runs"], 1)
        self.assertEqual(supersedes["smoke_files_written"], 0)
        self.assertEqual(supersedes["records_written"], 0)
        self.assertFalse(supersedes["full2638_outputs_touched"])
        self.assertFalse(supersedes["never_executed"])
        self.assertTrue(supersedes["must_not_be_run"])
        self.assertIn("output_token_ids", supersedes["failure_error"])
        reason = supersedes["reason"].lower()
        self.assertIn("output_token_ids", reason)
        self.assertIn("dense", reason)
        self.assertIn("no record", reason)
        self.assertNotIn("defects", supersedes)
        self.assertEqual(supersedes["sha256"], R3_CONFIG_SHA)

    def test_r4_freezes_r3_arms_metric_and_gates_unchanged(self) -> None:
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
                self.r3[field],
                f"r4 changed the frozen r3 field {field!r}",
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
                entry["output_path"], self.r3["arms"][arm]["output_path"]
            )
        self.assertEqual(arms["v2_edge576_b1_k96"]["mechanism_id"], V2_MECHANISM_ID)
        self.assertFalse(arms["v2_edge576_b1_k96"]["is_new_mechanism"])

    def test_no_100_percent_parse_gate(self) -> None:
        gate = self.config["success_gates"]["record_completeness"]
        self.assertFalse(gate["requires_100_percent_parse"])
        self.assertTrue(gate["parsed_answer_may_be_null"])
        self.assertTrue(gate["parse_failures_count_as_wrong"])
        self.assertNotIn("parseable", json.dumps(gate).lower())

    def test_record_contract_pins_optional_token_ids(self) -> None:
        record = self.config["record_contract"]
        self.assertTrue(record["token_ids_never_fabricated"])
        self.assertTrue(record["token_field_consistency_fail_closed"])
        self.assertFalse(record["dense_producer_emits_output_token_ids"])
        self.assertTrue(record["l2_producer_emits_output_token_ids"])
        self.assertTrue(record["v2_producer_emits_output_token_ids"])
        self.assertFalse(record["metrics_require_token_ids"])
        self.assertIn("output_token_ids", record["optional_fields"])
        self.assertNotIn("output_token_ids", record["core_required_fields"])
        self.assertIn("output_token_count", record["required_audit_fields"])
        self.assertIn("output_token_ids_available", record["required_audit_fields"])

    def test_runtime_contract_matches_pinned_record_contract(self) -> None:
        self.assertIn(
            contract.OUTPUT_TOKEN_COUNT_FIELD,
            contract.REQUIRED_EDGE_RECORD_FIELDS,
        )
        self.assertNotIn(
            contract.OUTPUT_TOKEN_IDS_FIELD,
            contract.REQUIRED_EDGE_RECORD_FIELDS,
        )
        ids, count = contract.resolve_output_token_fields({"output_tokens": 4})
        self.assertIsNone(ids)
        self.assertEqual(count, 4)
        ids, count = contract.resolve_output_token_fields(
            {"output_token_ids": [1, 2, 3]}
        )
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(count, 3)
        with self.assertRaises(contract.Edge576ContractError):
            contract.resolve_output_token_fields({})
        with self.assertRaises(contract.Edge576ContractError):
            contract.resolve_output_token_fields(
                {"output_token_ids": [1, 2, 3], "output_tokens": 4}
            )

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
        self.assertEqual([step["step"] for step in order], [1, 2, 3, 4, 5, 6])

    def test_smoke_contract_is_pinned(self) -> None:
        smoke = self.config["smoke_contract"]
        self.assertEqual(
            smoke["entry"],
            "evaluation/run_edge576_cvbench_three_arm_smoke.py",
        )
        self.assertEqual(smoke["output_dir"], SMOKE_OUTPUT_DIR)
        self.assertEqual(smoke["output_pattern"], "<arm>_<sample_id>.json")
        self.assertTrue(smoke["one_sample_per_invocation"])
        self.assertTrue(smoke["must_not_touch_full2638_outputs"])
        self.assertTrue(smoke["warmup_not_counted_in_record"])
        self.assertTrue(smoke["actual_grid_assertion"])
        self.assertFalse(smoke["selection_algorithm_copied"])
        self.assertTrue(smoke["output_token_ids_optional"])
        self.assertTrue(smoke["output_token_count_required"])
        self.assertTrue(smoke["dense_arm_smoke_expects_missing_token_ids"])
        self.assertEqual(smoke["real_warmup"]["max_new_tokens"], 1)
        self.assertTrue(smoke["real_warmup"]["record_discarded"])
        self.assertEqual(smoke["smoke_sample_id"], "1838")
        self.assertEqual(smoke["smoke_sample_g"], 576)
        self.assertEqual(len(smoke["commands"]), 3)
        for command in smoke["commands"]:
            self.assertIn("run_edge576_cvbench_three_arm_smoke.py", command)
            self.assertIn("--sample-id 1838", command)
            self.assertIn(SMOKE_OUTPUT_DIR, command)

    def test_refreshed_execution_layer_pins_are_present(self) -> None:
        pins = self.config["pins"]
        self.assertEqual(
            pins["smoke_entry"]["path"],
            str(SMOKE_ENTRY.relative_to(ROOT)),
        )
        self.assertEqual(
            pins["edge576_runner_contract"]["path"],
            str(RUNNER_CONTRACT.relative_to(ROOT)),
        )
        for name, old_sha in R3_DRIFTED_PINS.items():
            self.assertIn(name, pins)
            self.assertNotEqual(
                pins[name]["sha256"], old_sha, f"{name} was not refreshed"
            )
            self.assertEqual(pins[name]["sha256"], _sha256(ROOT / pins[name]["path"]))

    def test_r3_nonexecution_pins_are_inherited_verbatim(self) -> None:
        r3_pins = self.r3["pins"]
        self.assertEqual(len(r3_pins), EXPECTED_R3_PIN_COUNT)
        r4_pins = self.config["pins"]
        for name, entry in r3_pins.items():
            self.assertIn(name, r4_pins, f"r4 dropped the r3 pin {name}")
            self.assertEqual(r4_pins[name]["path"], entry["path"])
            if name in R3_DRIFTED_PINS:
                continue
            self.assertEqual(
                r4_pins[name]["sha256"],
                entry["sha256"],
                f"r4 changed the non-execution pin {name}",
            )
            self.assertEqual(_sha256(ROOT / entry["path"]), entry["sha256"])

    def test_r3_failure_and_supersede_evidence_is_pinned(self) -> None:
        pins = self.config["pins"]
        for name, path in (
            ("r3_first_gpu_attempt_failure_json", R3_FAILURE_JSON),
            ("r3_first_gpu_attempt_failure_md", R3_FAILURE_MD),
            ("r3_superseded_json", R3_SUPERSEDED_JSON),
            ("r3_superseded_md", R3_SUPERSEDED_MD),
            ("r3_prereg_config", R3_CONFIG),
            ("r3_prereg_report_json", R3_REPORT_JSON),
            ("r3_prereg_report_md", R3_REPORT_MD),
            ("prereg_test_r4", R4_TEST),
        ):
            self.assertIn(name, pins, f"missing pin {name}")
            self.assertEqual(pins[name]["sha256"], _sha256(path))
        failure = json.loads(R3_FAILURE_JSON.read_text(encoding="utf-8"))
        self.assertEqual(failure["status"], "FIRST_GPU_SMOKE_ATTEMPT_FAIL_CLOSED")
        self.assertEqual(failure["attempt"]["gpu_runs"], 1)
        self.assertEqual(failure["attempt"]["records_written"], 0)
        self.assertFalse(failure["attempt"]["full2638_outputs_touched"])
        self.assertIn("output_token_ids", failure["attempt"]["error_message"])
        self.assertFalse(
            failure["root_cause"]["dense_producer_emits_output_token_ids"]
        )
        self.assertTrue(failure["root_cause"]["metrics_need_token_ids"] is False)
        superseded = json.loads(R3_SUPERSEDED_JSON.read_text(encoding="utf-8"))
        self.assertEqual(
            superseded["status"], "SUPERSEDED_AFTER_FIRST_GPU_SMOKE_FAILURE"
        )
        self.assertTrue(superseded["executed"])
        self.assertTrue(superseded["must_not_be_run"])
        self.assertTrue(superseded["bytes_preserved"])
        self.assertFalse(superseded["full2638_outputs_touched"])

    def test_r3_bytes_are_preserved(self) -> None:
        self.assertEqual(_sha256(R3_CONFIG), R3_CONFIG_SHA)
        self.assertEqual(_sha256(R3_REPORT_JSON), R3_REPORT_JSON_SHA)
        self.assertEqual(_sha256(R3_REPORT_MD), R3_REPORT_MD_SHA)
        self.assertEqual(_sha256(R3_GENERATOR), R3_GENERATOR_SHA)
        self.assertEqual(_sha256(R3_TEST), R3_TEST_SHA)

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

    def test_r4_report_status_and_scope(self) -> None:
        self.assertEqual(self.report["status"], "PREREGISTERED_NOT_EXECUTED")
        self.assertEqual(self.report["revision"], "r4")
        self.assertFalse(self.report["executed"])
        self.assertEqual(self.report["gpu_runs"], 0)
        self.assertEqual(self.report["supersedes"]["sha256"], R3_CONFIG_SHA)
        self.assertIn("execution_order", self.report)
        self.assertIn("record_contract", self.report)
        self.assertIn("smoke_contract", self.report)
        self.assertTrue(R4_REPORT_MD.is_file())
        self.assertGreater(R4_REPORT_MD.stat().st_size, 0)

    def test_runner_adaptation_entry_status(self) -> None:
        entry = self.config["runner_adaptation_entry"]
        self.assertEqual(entry["smoke_entry"], str(SMOKE_ENTRY.relative_to(ROOT)))
        self.assertIn("FIXED", entry["status"])
        self.assertFalse(entry["selection_algorithm_copied"])

    def test_shared_arm_contract_matches_r3(self) -> None:
        shared = self.config["shared_arm_contract"]
        self.assertEqual(shared["processor"]["min_pixels"], 65536)
        self.assertEqual(shared["processor"]["max_pixels"], 589824)
        self.assertFalse(shared["do_sample"])
        self.assertEqual(shared["max_new_tokens"], 16)
        self.assertEqual(shared, self.r3["shared_arm_contract"])

    def test_forbidden_claims_include_r3_failure(self) -> None:
        claims = " ".join(self.config["forbidden_claims"]).lower()
        self.assertIn("r3", claims)
        self.assertIn("fabricated", claims)


if __name__ == "__main__":
    unittest.main()
