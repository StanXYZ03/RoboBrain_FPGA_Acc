"""CPU-only tests for the reused oracle runner under the v2 challenge/control contract.

The GPU runner itself is reused unchanged
(``evaluation/run_b1_answer_margin_oracle_pilot.py``).  These tests exercise the
fail-closed resume validation, the exact K96 layout contract and the config-pin
verification on the v2 semantics.  No model weights, no CUDA.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.b1_answer_margin_oracle import (
    ARM_ORACLE,
    BUDGET,
    array_sha256,
    inject_oracle_selection,
    layout_contract,
    record_sha256,
)
from evaluation.run_b1_answer_margin_oracle_pilot import (
    OracleRunnerError,
    REQUIRED_RECORD_FIELDS,
    validate_resume,
    verify_config_pins,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
V2_CONFIG = (
    WORKSPACE_ROOT
    / "configs"
    / "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24.preregistered.json"
)


def valid_record(sample_id: str = "0", *, g: int = 256, **overrides) -> dict:
    values = np.linspace(0.1, 1.0, g, dtype=np.float64)
    record = {
        "sample_id": sample_id,
        "arm": ARM_ORACLE,
        "manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "requested_budget": BUDGET,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "output_text": "(A)",
        "output_token_ids": [1, 2],
        "full_patch_tokens": g * 4,
        "image_grid_thw": [1, 16, g * 4 // 16],
        "full_llm_visual_tokens": g,
        "output_group_count": BUDGET,
        "context_group_count": 9,
        "real_group_count": 87,
        "identity_bypass": False,
        "selected_group_indices": list(range(BUDGET)),
        "label_tokenization": {"contract_ok": True},
        "margin": {"value": 1.0},
        "saliency": {"definition": "x"},
        "saliency_sha256": array_sha256(values),
        "saliency_group_values": [float(value) for value in values],
        "timings": {"gpu_boundaries_synchronized": True},
        "peak_cuda_bytes": 1,
        "model_parameter_grads_all_none": True,
        "all_model_parameters_requires_grad_false": True,
        "layout": {
            "passed": True,
            "output_group_count": BUDGET,
            "real_group_count": 87,
            "context_group_count": 9,
            "identity_bypass": False,
        },
    }
    record.update(overrides)
    record["record_sha256"] = record_sha256(record)
    return record


class LayoutContractTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20261005)
        self.hidden = rng.random((256, 4, 1024))

    def test_g_gt_96_emits_exact_k96_87_9_layout(self) -> None:
        selection = inject_oracle_selection(
            self.hidden, 16, 16, np.linspace(0.1, 1.0, 256)
        )
        contract = layout_contract(selection, full_groups=256)
        self.assertTrue(contract["passed"])
        self.assertEqual(contract["output_group_count"], 96)
        self.assertEqual(contract["real_group_count"], 87)
        self.assertEqual(contract["context_group_count"], 9)


class ResumeContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [{"sample_id": "0"}, {"sample_id": "1"}]
        self.kwargs = dict(
            arm=ARM_ORACLE,
            manifest_sha256="a" * 64,
            config_sha256="b" * 64,
            budget=BUDGET,
            region_grid=3,
            local_candidate_cap=16,
        )

    def _write(self, tmp: str, records) -> Path:
        path = Path(tmp) / "out.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        return path

    def test_missing_and_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                validate_resume(Path(tmp) / "none.jsonl", self.samples, **self.kwargs),
                set(),
            )

    def test_valid_record_returns_done_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0")])
            self.assertEqual(validate_resume(path, self.samples, **self.kwargs), {"0"})

    def test_partial_record_fails_closed(self) -> None:
        record = valid_record("0")
        for field in REQUIRED_RECORD_FIELDS:
            if field == "record_sha256":
                continue
            broken = {key: value for key, value in record.items() if key != field}
            broken["record_sha256"] = record_sha256(broken)
            with tempfile.TemporaryDirectory() as tmp:
                path = self._write(tmp, [broken])
                with self.assertRaises(OracleRunnerError):
                    validate_resume(path, self.samples, **self.kwargs)

    def test_duplicate_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0"), valid_record("0")])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_mixed_arm_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", arm="other")])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_wrong_layout_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", context_group_count=8)])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_identity_bypass_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", identity_bypass=True)])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_g_le_96_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", full_llm_visual_tokens=96)])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_tampered_record_hash_fails_closed(self) -> None:
        record = valid_record("0")
        record["output_text"] = "(B)"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [record])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)


class ConfigPinTest(unittest.TestCase):
    def test_matching_pin_passes_and_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "artifact.txt"
            target.write_text("payload", encoding="utf-8")
            digest = hashlib.sha256(b"payload").hexdigest()
            good = {
                "inputs_sha256": {
                    "artifact": {"path": "artifact.txt", "sha256": digest}
                }
            }
            self.assertTrue(verify_config_pins(good, root=root)["pins_verified"])
            bad = {
                "inputs_sha256": {
                    "artifact": {"path": "artifact.txt", "sha256": "0" * 64}
                }
            }
            with self.assertRaises(OracleRunnerError):
                verify_config_pins(bad, root=root)

    @unittest.skipUnless(V2_CONFIG.is_file(), "v2 config not generated yet")
    def test_generated_v2_config_pins_verify(self) -> None:
        config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
        result = verify_config_pins(config, root=WORKSPACE_ROOT)
        self.assertTrue(result["pins_verified"])
        self.assertEqual(config["arm"], ARM_ORACLE)
        self.assertTrue(config["pilot_gate"]["dense_is_only_eligibility_reference"])
        self.assertFalse(config["pilot_gate"]["uses_confidence_interval"])


if __name__ == "__main__":
    unittest.main()
