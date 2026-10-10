"""CPU-only tests for the B1 answer-margin oracle GPU runner contract.

These tests never load model weights and never touch CUDA.  They cover the
oracle score injection into the frozen hardware layout (fake tensors produce
exactly the K96 / 87 real / 9 context layout), the saliency helper, the scorer
hook, record hashing, config-pin verification and the fail-closed resume
contract (partial / duplicate / mixed-arm / wrong-hash / wrong-layout records).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluation.b1_answer_margin_oracle import (
    ARM_ORACLE,
    BUDGET,
    OraclePilotError,
    array_sha256,
    build_oracle_scorer,
    group_saliency_from_grad_activation,
    inject_oracle_selection,
    layout_contract,
    record_sha256,
    retained_mass,
    selection_overlap,
)
from evaluation.run_b1_answer_margin_oracle_pilot import (
    OracleRunnerError,
    REQUIRED_RECORD_FIELDS,
    validate_resume,
    verify_config_pins,
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
    }
    record.update(overrides)
    record["record_sha256"] = record_sha256(record)
    return record


class SaliencyHelperTest(unittest.TestCase):
    def test_grad_activation_saliency_matches_manual_sum(self) -> None:
        hidden = torch.arange(12 * 5, dtype=torch.float32).reshape(12, 5) / 10.0
        hidden.requires_grad_(True)
        grad = torch.ones_like(hidden)
        saliency = group_saliency_from_grad_activation(hidden, grad, merge_unit=4)
        self.assertEqual(len(saliency), 3)
        manual = (hidden.detach() * grad).reshape(3, 4, 5).abs().sum(dim=(1, 2))
        self.assertTrue(
            np.allclose(saliency, manual.tolist(), atol=1e-6)
        )

    def test_shape_mismatch_fails_closed(self) -> None:
        with self.assertRaises(OraclePilotError):
            group_saliency_from_grad_activation(
                torch.zeros(8, 3), torch.zeros(4, 3), merge_unit=4
            )

    def test_non_finite_grad_fails_closed(self) -> None:
        hidden = torch.zeros(8, 3)
        grad = torch.full((8, 3), float("nan"))
        with self.assertRaises(OraclePilotError):
            group_saliency_from_grad_activation(hidden, grad, merge_unit=4)


class OracleScorerTest(unittest.TestCase):
    def test_scorer_returns_the_injected_combined_scores(self) -> None:
        saliency = np.linspace(1.0, 2.0, 256)
        scorer = build_oracle_scorer(saliency)
        payload = scorer(np.zeros((256, 4, 1024)), 16, 16, 96, None)
        self.assertTrue(np.array_equal(payload["combined_scores"], saliency))
        self.assertTrue(payload["is_label_aware_oracle"])
        self.assertFalse(payload["is_deployable_selector"])

    def test_scorer_size_mismatch_fails_closed(self) -> None:
        scorer = build_oracle_scorer(np.ones(256))
        with self.assertRaises(OraclePilotError):
            scorer(np.zeros((128, 4, 1024)), 8, 16, 96, None)

    def test_scorer_rejects_non_finite(self) -> None:
        with self.assertRaises(OraclePilotError):
            build_oracle_scorer([1.0, float("inf")])


class OraclLayoutInjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(20261005)
        self.hidden = rng.random((256, 4, 1024))

    def test_g_gt_96_emits_exact_k96_87_9_layout(self) -> None:
        selection = inject_oracle_selection(
            self.hidden, 16, 16, np.linspace(0.1, 1.0, 256)
        )
        contract = layout_contract(selection, full_groups=256)
        self.assertTrue(contract["passed"])
        self.assertFalse(selection.bypass)
        self.assertEqual(contract["output_group_count"], 96)
        self.assertEqual(contract["real_group_count"], 87)
        self.assertEqual(contract["context_group_count"], 9)
        self.assertEqual(
            [int(v) for v in selection.patch_indices[0]], [int(selection.group_indices[0]) * 4 + j for j in range(4)]
        )

    def test_g_le_96_is_identity_bypass(self) -> None:
        selection = inject_oracle_selection(
            self.hidden[:64], 8, 8, np.linspace(0.1, 1.0, 64)
        )
        contract = layout_contract(selection, full_groups=64)
        self.assertTrue(contract["passed"])
        self.assertTrue(selection.bypass)
        self.assertEqual(contract["output_group_count"], 64)
        self.assertEqual(contract["context_group_count"], 0)

    def test_mismatched_full_groups_fails_layout(self) -> None:
        selection = inject_oracle_selection(
            self.hidden, 16, 16, np.linspace(0.1, 1.0, 256)
        )
        contract = layout_contract(selection, full_groups=999)
        self.assertFalse(contract["passed"])

    def test_diagnostics(self) -> None:
        self.assertAlmostEqual(retained_mass([1.0, 1.0, 1.0, 1.0], [0, 2]), 0.5)
        overlap = selection_overlap([0, 1, 2], [1, 2, 3])
        self.assertEqual(overlap["intersection"], 2)
        self.assertAlmostEqual(overlap["jaccard"], 0.5)


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
            self.assertEqual(validate_resume(Path(tmp) / "none.jsonl", self.samples, **self.kwargs), set())
            empty = Path(tmp) / "empty.jsonl"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(validate_resume(empty, self.samples, **self.kwargs), set())

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

    def test_wrong_manifest_sha_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", manifest_sha256="c" * 64)])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_tampered_record_hash_fails_closed(self) -> None:
        record = valid_record("0")
        record["output_text"] = "(B)"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [record])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_wrong_layout_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("0", real_group_count=88)])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_bad_saliency_hash_fails_closed(self) -> None:
        record = valid_record("0")
        record["saliency_group_values"] = [0.0] * 256
        record["record_sha256"] = record_sha256(record)
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [record])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_stale_sample_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [valid_record("9")])
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)

    def test_corrupt_json_line_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(OracleRunnerError):
                validate_resume(path, self.samples, **self.kwargs)


class ConfigPinTest(unittest.TestCase):
    def test_matching_pin_passes_and_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "artifact.txt"
            target.write_text("payload", encoding="utf-8")
            digest = hashlib.sha256(b"payload").hexdigest()
            good = {"inputs_sha256": {"artifact": {"path": "artifact.txt", "sha256": digest}}}
            result = verify_config_pins(good, root=root)
            self.assertTrue(result["pins_verified"])
            bad = {"inputs_sha256": {"artifact": {"path": "artifact.txt", "sha256": "0" * 64}}}
            with self.assertRaises(OracleRunnerError):
                verify_config_pins(bad, root=root)

    def test_missing_pinned_file_fails_closed(self) -> None:
        config = {
            "inputs_sha256": {
                "artifact": {"path": "missing.txt", "sha256": "0" * 64}
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OracleRunnerError):
                verify_config_pins(config, root=Path(tmp))


if __name__ == "__main__":
    unittest.main()
