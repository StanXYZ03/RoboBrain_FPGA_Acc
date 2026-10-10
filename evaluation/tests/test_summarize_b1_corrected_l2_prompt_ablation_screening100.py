#!/usr/bin/env python3
"""CPU-only tests for the B1 corrected-L2 / Prompt-ablation summariser.

Synthetic inputs are written to a temporary directory; the real frozen artifacts
and configs are never modified.  The tests pin the preregistered decision rule,
the fail-closed input contract and the no-write-on-failure behaviour.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    summarize_b1_corrected_l2_prompt_ablation_screening100 as summ,
)
from evaluation.hardware_regular_b1_corrected_l2 import array_sha256  # noqa: E402

SOURCES = (
    ("s0", "ADE20K", "ADE20K Validation Set", "Count", "(A)", 3),
    ("s1", "COCO", "COCO", "Depth", "(B)", 3),
    ("s2", "Omni3D", "Omni3D", "Distance", "(C)", 3),
)


def _manifest_rows() -> list[dict]:
    return [
        {
            "sample_id": sid,
            "task": task,
            "type": "2D" if source in {"ADE20K", "COCO"} else "3D",
            "source": source,
            "source_dataset": source_dataset,
            "choices": ["alpha", "beta", "gamma"],
            "answer": answer,
            "image": "x.jpg",
            "prompt": "p",
            "max_new_tokens": 16,
        }
        for sid, source, source_dataset, task, answer, _ in SOURCES
    ]


def _arm_rows(arm: str, outputs: dict[str, str]) -> list[dict]:
    """New-arm records with K=96 (or identity bypass) and valid index hashes."""
    rows = []
    for sample_id in outputs:
        rows.append(
            {
                "sample_id": sample_id,
                "arm": arm,
                "output_text": outputs[sample_id],
                "requested_budget": 96,
                "region_grid": 3,
                "local_candidate_cap": 16,
                "full_patch_tokens": 4800,
                "image_grid_thw": [1, 60, 80],
                "full_llm_visual_tokens": 1200,
                "output_group_count": 96,
                "identity_bypass": False,
                "selected_group_indices": list(range(96)),
                "selected_group_indices_sha256": array_sha256(
                    np.asarray(list(range(96)), dtype=np.int64)
                ),
            }
        )
    return rows


def _ref_rows(outputs: dict[str, str]) -> list[dict]:
    return [
        {"sample_id": sample_id, "output_text": outputs[sample_id]}
        for sample_id in outputs
    ]


def _outputs() -> dict[str, dict[str, str]]:
    return {
        "arm_float": {"s0": "(A)", "s1": "(B)", "s2": "(A)"},
        "arm_integer": {"s0": "(A)", "s1": "(A)", "s2": "(C)"},
        "arm_prompt": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
        "dense": {"s0": "(A)", "s1": "(B)", "s2": "(C)"},
        "old_defective_l2": {"s0": "(A)", "s1": "(A)", "s2": "(C)"},
        "old_current_integer": {"s0": "(B)", "s1": "(B)", "s2": "(C)"},
    }


class DecisionRuleTest(unittest.TestCase):
    def test_quantization_failure_branch(self) -> None:
        decision = summ.decide([-0.03, 0.05], [0.10, 0.20])
        self.assertEqual(decision["outcome"], summ.OUTCOME_QUANTIZATION_FAIL)
        self.assertFalse(
            decision["integer_minus_float"]["quantization_gate_passed"]
        )

    def test_both_pass_branch(self) -> None:
        decision = summ.decide([-0.01, 0.05], [0.03, 0.10])
        self.assertEqual(decision["outcome"], summ.OUTCOME_PASS)
        self.assertTrue(decision["integer_minus_float"]["quantization_gate_passed"])
        self.assertTrue(decision["prompt_minus_integer_l2"]["prompt_gain_confirmed"])

    def test_quantization_pass_prompt_not_confirmed(self) -> None:
        decision = summ.decide([-0.01, 0.05], [-0.02, 0.04])
        self.assertEqual(
            decision["outcome"], summ.OUTCOME_PROMPT_NOT_CONFIRMED
        )

    def test_boundaries_are_inclusive_exclusive(self) -> None:
        on_floor = summ.decide([-0.02, 0.05], [0.0, 0.04])
        self.assertTrue(
            on_floor["integer_minus_float"]["quantization_gate_passed"]
        )
        self.assertEqual(on_floor["outcome"], summ.OUTCOME_PROMPT_NOT_CONFIRMED)
        just_below = summ.decide([-0.020001, 0.05], [0.5, 0.6])
        self.assertEqual(just_below["outcome"], summ.OUTCOME_QUANTIZATION_FAIL)
        strict_prompt = summ.decide([-0.01, 0.05], [0.0, 0.04])
        self.assertFalse(
            strict_prompt["prompt_minus_integer_l2"]["prompt_gain_confirmed"]
        )

    def test_only_primary_metric_is_used(self) -> None:
        decision = summ.decide([0.0, 0.1], [0.0, 0.1])
        self.assertTrue(decision["uses_only_primary_metric"])
        self.assertTrue(decision["descriptive_references_cannot_override"])
        self.assertEqual(decision["metric"], summ.PRIMARY_METRIC)


class ArmValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest_rows()
        self.outputs = _outputs()
        self.screening_index = summ.index_unique(self.manifest, "screening")

    def _indices(self):
        rows = {
            "arm_float": _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"]),
            "arm_integer": _arm_rows(
                summ.ARM_INTEGER, self.outputs["arm_integer"]
            ),
            "arm_prompt": _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"]),
        }
        return {
            "screening_index": self.screening_index,
            "full_index": summ.index_unique(self.manifest, "full"),
            "dense_index": summ.index_unique(
                _ref_rows(self.outputs["dense"]), "dense"
            ),
            "old_defective_index": summ.index_unique(
                _ref_rows(self.outputs["old_defective_l2"]), "old_defective"
            ),
            "old_current_index": summ.index_unique(
                _ref_rows(self.outputs["old_current_integer"]), "old_current"
            ),
            "arm_indices": {
                summ.ARM_FLOAT: summ.index_unique(
                    rows["arm_float"], "arm_float"
                ),
                summ.ARM_INTEGER: summ.index_unique(
                    rows["arm_integer"], "arm_integer"
                ),
                summ.ARM_PROMPT: summ.index_unique(rows["arm_prompt"], "arm_prompt"),
            },
        }

    def test_happy_path(self) -> None:
        result = summ.validate_arms(
            **self._indices(),
            expected_screening_count=3,
            expected_full_count=3,
        )
        self.assertTrue(result["sample_ids_identical"])
        self.assertTrue(result["input_grid_consistent_across_arms"])

    def test_id_set_mismatch_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, {"s0": "(A)", "s1": "(B)"})
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_wrong_arm_field_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_INTEGER, self.outputs["arm_integer"])
        for row in rows:
            row["arm"] = summ.ARM_FLOAT
        indices["arm_indices"][summ.ARM_INTEGER] = summ.index_unique(
            rows, "arm_integer"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_wrong_budget_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["requested_budget"] = 64
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_grid_mismatch_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"])
        rows[0]["image_grid_thw"] = [1, 40, 80]
        rows[0]["full_patch_tokens"] = 3200
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_bad_index_hash_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["selected_group_indices_sha256"] = "0" * 64
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_non_bypass_wrong_k_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["output_group_count"] = 95
        rows[0]["selected_group_indices"] = list(range(95))
        rows[0]["selected_group_indices_sha256"] = array_sha256(
            np.asarray(list(range(95)), dtype=np.int64)
        )
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices, expected_screening_count=3, expected_full_count=3
            )

    def test_checkpoint_hashes_cross_check(self) -> None:
        hashes = {
            "checkpoint_sha256": "c" * 64,
            "wv_sha256": "d" * 64,
            "wv_scale_sha256": "e" * 64,
        }
        indices = self._indices()
        rows = _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"])
        for row in rows:
            row.update(hashes)
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        summ.validate_arms(
            **indices,
            expected_screening_count=3,
            expected_full_count=3,
            checkpoint_hashes=hashes,
        )

        rows[0]["wv_sha256"] = "f" * 64
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **indices,
                expected_screening_count=3,
                expected_full_count=3,
                checkpoint_hashes=hashes,
            )

        clean = self._indices()
        float_rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        float_rows[0]["checkpoint_sha256"] = "c" * 64
        clean["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            float_rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                **clean,
                expected_screening_count=3,
                expected_full_count=3,
                checkpoint_hashes=hashes,
            )


class BuildAndWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.outputs = _outputs()
        self.manifest = _manifest_rows()
        self.paths = self._write_inputs()
        self._saved_repeats = summ.BOOTSTRAP_REPEATS
        summ.BOOTSTRAP_REPEATS = 32

    def tearDown(self) -> None:
        summ.BOOTSTRAP_REPEATS = self._saved_repeats
        self._tmp.cleanup()

    def _write(self, name: str, rows: list[dict]) -> Path:
        path = self.tmp / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def _write_inputs(self) -> summ.InputPaths:
        config = self.tmp / "config.json"
        config.write_text("{}", encoding="utf-8")
        return summ.InputPaths(
            preregistered=config,
            screening_manifest=self._write("screening.jsonl", self.manifest),
            full_manifest=self._write("full.jsonl", self.manifest),
            dense_full_jsonl=self._write(
                "dense.jsonl", _ref_rows(self.outputs["dense"])
            ),
            old_defective_l2_screening_jsonl=self._write(
                "old_l2.jsonl", _ref_rows(self.outputs["old_defective_l2"])
            ),
            old_current_integer_full_jsonl=self._write(
                "old_int.jsonl", _ref_rows(self.outputs["old_current_integer"])
            ),
            arm_float_jsonl=self._write(
                "arm_float.jsonl", _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
            ),
            arm_integer_jsonl=self._write(
                "arm_integer.jsonl",
                _arm_rows(summ.ARM_INTEGER, self.outputs["arm_integer"]),
            ),
            arm_prompt_jsonl=self._write(
                "arm_prompt.jsonl",
                _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"]),
            ),
        )

    def _run(self, paths=None, **kwargs):
        return summ.run(
            paths or self.paths,
            output_json=self.tmp / "report.json",
            output_csv=self.tmp / "report.csv",
            output_md=self.tmp / "report.md",
            verify_pins=False,
            verify_outputs=False,
            expected_screening_count=3,
            expected_full_count=3,
            **kwargs,
        )

    def test_analyze_produces_six_arms_and_primary_decisions(self) -> None:
        code, report = self._run()
        self.assertEqual(code, 0)
        self.assertIsNotNone(report)
        for arm in (*summ.ARM_ORDER, *summ.DESCRIPTIVE_ORDER):
            self.assertIn(arm, report["metrics"])
        self.assertIn("integer_minus_float", report["comparisons"])
        self.assertIn("prompt_minus_integer_l2", report["comparisons"])
        self.assertIn(
            report["decision"]["outcome"],
            {
                summ.OUTCOME_PASS,
                summ.OUTCOME_PROMPT_NOT_CONFIRMED,
                summ.OUTCOME_QUANTIZATION_FAIL,
            },
        )
        self.assertEqual(report["primary_metric"]["identifier"], summ.PRIMARY_METRIC)

    def test_success_writes_json_csv_and_markdown(self) -> None:
        code, _ = self._run()
        self.assertEqual(code, 0)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertTrue((self.tmp / suffix).is_file(), suffix)
        rows = (self.tmp / "report.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 4)  # header + 3 samples
        markdown = (self.tmp / "report.md").read_text(encoding="utf-8")
        self.assertIn("corrected-L2", markdown)
        self.assertIn("量化门", markdown)
        self.assertIn("Prompt 增益", markdown)
        for arm in summ.ARM_ORDER:
            self.assertIn(arm, markdown)

    def test_no_write_on_failure(self) -> None:
        broken = summ.InputPaths(**{
            **self.paths.__dict__,
            "arm_prompt_jsonl": self.tmp / "missing_prompt.jsonl",
        })
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertFalse((self.tmp / suffix).is_file(), suffix)

    def test_output_path_mismatch_fails_closed(self) -> None:
        config = {"output_paths": {arm: "wrong.jsonl" for arm in summ.ARM_ORDER}}
        with self.assertRaises(summ.InputContractError):
            summ.verify_output_paths(config, self.paths)


class ConfigPinTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config_path = summ.PREREGISTERED_CONFIG
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _paths(self) -> summ.InputPaths:
        # Real logical inputs; pending arm outputs point at existing temp files.
        arm = self.tmp / "arm.jsonl"
        arm.write_text("{}\n", encoding="utf-8")
        return summ.InputPaths(
            arm_float_jsonl=arm,
            arm_integer_jsonl=arm,
            arm_prompt_jsonl=arm,
        )

    def test_real_config_pins_verify(self) -> None:
        result = summ.verify_config_pins(
            self.config, self._paths(), root=ROOT
        )
        self.assertTrue(result["pins_verified"])
        self.assertTrue(result["checks"])

    def test_real_config_output_paths_match_defaults(self) -> None:
        result = summ.verify_output_paths(self.config, summ.DEFAULT_PATHS)
        self.assertTrue(result["output_paths_verified"])

    def test_tampered_config_pin_fails_closed(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["inputs_sha256"]["evaluation/metrics.py"]["sha256"] = "0" * 64
        with self.assertRaises(summ.InputContractError):
            summ.verify_config_pins(config, self._paths(), root=ROOT)

    def test_missing_pending_arm_output_fails_closed(self) -> None:
        paths = summ.InputPaths(
            arm_float_jsonl=self.tmp / "absent.jsonl",
            arm_integer_jsonl=self.tmp / "absent.jsonl",
            arm_prompt_jsonl=self.tmp / "absent.jsonl",
        )
        with self.assertRaises(summ.InputContractError):
            summ.verify_config_pins(self.config, paths, root=ROOT)


if __name__ == "__main__":
    unittest.main()
