#!/usr/bin/env python3
"""CPU-only focused tests for the full2638 corrected-L2 / Prompt summariser.

Synthetic inputs live in a temporary directory and small count overrides
(``expected_full_count=3``) are used so the real frozen 2638-sample artifacts
and configs are never touched.  The tests pin the three decision branches, the
relative-vs-absolute metric separation, the fail-closed full2638 input contract
(count / id / arm / record hash / manifest hash / checkpoint / grid / K) and the
no-write-on-failure behaviour.
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
    summarize_b1_corrected_l2_prompt_ablation_full2638 as summ,
)
from evaluation import (  # noqa: E402
    summarize_b1_corrected_l2_prompt_ablation_screening100 as screen,
)
from evaluation.hardware_regular_b1_corrected_l2 import array_sha256  # noqa: E402

MANIFEST_SHA = "a" * 64
CHECKPOINT_HASHES = {
    "checkpoint_sha256": "c" * 64,
    "wv_sha256": "d" * 64,
    "wv_scale_sha256": "e" * 64,
}
SOURCES = (
    ("s0", "ADE20K", "ADE20K Validation Set", "Count", "(A)"),
    ("s1", "COCO", "COCO", "Depth", "(B)"),
    ("s2", "Omni3D", "Omni3D", "Distance", "(C)"),
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
        for sid, source, source_dataset, task, answer in SOURCES
    ]


def _arm_rows(
    arm: str,
    outputs: dict[str, str],
    *,
    manifest_sha: str = MANIFEST_SHA,
    bypass: bool = False,
) -> list[dict]:
    """Arm records with a valid K=96 (or valid identity bypass) contract."""

    rows = []
    for sample_id in outputs:
        if bypass:
            full_llm, full_patch, grid, count = 8, 32, [1, 4, 8], 8
        else:
            full_llm, full_patch, grid, count = 1200, 4800, [1, 60, 80], 96
        row = {
            "sample_id": sample_id,
            "arm": arm,
            "task": dict((s[0], s[3]) for s in SOURCES)[sample_id],
            "output_text": outputs[sample_id],
            "manifest_sha256": manifest_sha,
            "requested_budget": 96,
            "region_grid": 3,
            "local_candidate_cap": 16,
            "full_patch_tokens": full_patch,
            "image_grid_thw": grid,
            "full_llm_visual_tokens": full_llm,
            "output_group_count": count,
            "identity_bypass": bypass,
            "selected_group_indices": list(range(count)),
            "selected_group_indices_sha256": array_sha256(
                np.asarray(list(range(count)), dtype=np.int64)
            ),
            "checkpoint_sha256": None,
            "wv_sha256": None,
            "wv_scale_sha256": None,
        }
        row["record_sha256"] = summ.record_sha256(row)
        rows.append(row)
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
        self.assertEqual(decision["outcome"], summ.OUTCOME_PROMPT_NOT_CONFIRMED)

    def test_boundaries_are_inclusive_exclusive(self) -> None:
        on_floor = summ.decide([-0.02, 0.05], [0.0, 0.04])
        self.assertTrue(on_floor["integer_minus_float"]["quantization_gate_passed"])
        self.assertEqual(on_floor["outcome"], summ.OUTCOME_PROMPT_NOT_CONFIRMED)
        just_below = summ.decide([-0.020001, 0.05], [0.5, 0.6])
        self.assertEqual(just_below["outcome"], summ.OUTCOME_QUANTIZATION_FAIL)
        strict_prompt = summ.decide([-0.01, 0.05], [0.0, 0.04])
        self.assertFalse(
            strict_prompt["prompt_minus_integer_l2"]["prompt_gain_confirmed"]
        )

    def test_only_primary_metric_and_full2638_outcomes(self) -> None:
        decision = summ.decide([0.0, 0.1], [0.0, 0.1])
        self.assertTrue(decision["uses_only_primary_metric"])
        self.assertTrue(decision["descriptive_references_cannot_override"])
        self.assertTrue(decision["formal_full2638_conclusion"])
        self.assertEqual(decision["metric"], summ.PRIMARY_METRIC)
        self.assertEqual(decision["ordered_rules"], list(summ.DECISION_ORDER))
        self.assertEqual(
            summ.OUTCOME_QUANTIZATION_FAIL, screen.OUTCOME_QUANTIZATION_FAIL
        )
        self.assertNotIn(summ.OUTCOME_PASS, screen.DECISION_ORDER)
        self.assertNotIn(summ.OUTCOME_PROMPT_NOT_CONFIRMED, screen.DECISION_ORDER)

    def test_record_sha256_matches_runner_algorithm(self) -> None:
        # Independent re-implementation of the frozen runner digest.
        import hashlib

        record = {"b": 1, "a": [1, 2, 3], "nested": {"x": None}}
        payload = {k: v for k, v in record.items() if k != "record_sha256"}
        blob = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        expected = hashlib.sha256(blob).hexdigest()
        self.assertEqual(summ.record_sha256(record), expected)


class ArmValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = _manifest_rows()
        self.outputs = _outputs()
        self.manifest_index = summ.index_unique(self.manifest, "full")

    def _indices(self, **overrides):
        rows = {
            "arm_float": _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"]),
            "arm_integer": _arm_rows(
                summ.ARM_INTEGER, self.outputs["arm_integer"]
            ),
            "arm_prompt": _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"]),
        }
        indices = {
            "manifest_index": self.manifest_index,
            "dense_index": summ.index_unique(
                _ref_rows(self.outputs["dense"]), "dense"
            ),
            "old_current_index": summ.index_unique(
                _ref_rows(self.outputs["old_current_integer"]), "old_current"
            ),
            "arm_indices": {
                summ.ARM_FLOAT: summ.index_unique(rows["arm_float"], "arm_float"),
                summ.ARM_INTEGER: summ.index_unique(
                    rows["arm_integer"], "arm_integer"
                ),
                summ.ARM_PROMPT: summ.index_unique(
                    rows["arm_prompt"], "arm_prompt"
                ),
            },
            "expected_full_count": 3,
            "manifest_sha256": MANIFEST_SHA,
        }
        indices.update(overrides)
        return indices

    def test_happy_path(self) -> None:
        result = summ.validate(**self._indices())
        self.assertTrue(result["exactly_2638_unique_ids_per_arm"])
        self.assertTrue(result["record_hashes_verified"])
        self.assertTrue(result["input_grid_consistent_across_arms"])
        self.assertTrue(result["all_non_bypass_arms_emit_K_96"])

    def test_manifest_missing_source_fails_closed(self) -> None:
        rows = [dict(row) for row in self.manifest]
        rows[0].pop("source")
        indices = self._indices()
        indices["manifest_index"] = summ.index_unique(rows, "full")
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_wrong_expected_full_count_fails_closed(self) -> None:
        indices = self._indices(expected_full_count=2638)
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_arm_missing_id_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, {"s0": "(A)", "s1": "(B)"})
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_descriptive_missing_id_fails_closed(self) -> None:
        indices = self._indices()
        indices["dense_index"] = summ.index_unique(
            _ref_rows({"s0": "(A)", "s1": "(B)"}), "dense"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_duplicate_sample_id_fails_closed(self) -> None:
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows.append(dict(rows[0]))
        with self.assertRaises(summ.InputContractError):
            summ.index_unique(rows, "arm_float")

    def test_wrong_arm_field_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_INTEGER, self.outputs["arm_integer"])
        for row in rows:
            row["arm"] = summ.ARM_FLOAT
            row["record_sha256"] = summ.record_sha256(row)
        indices["arm_indices"][summ.ARM_INTEGER] = summ.index_unique(
            rows, "arm_integer"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_wrong_budget_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["requested_budget"] = 64
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_manifest_sha_mismatch_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["manifest_sha256"] = "b" * 64
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_missing_manifest_sha_fails_closed(self) -> None:
        indices = self._indices(manifest_sha256=None)
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_record_sha_tamper_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["output_text"] = "(C)"
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_task_mismatch_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["task"] = "NotTheManifestTask"
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_source_mismatch_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["source"] = "WrongSource"
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_bad_index_hash_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["selected_group_indices_sha256"] = "0" * 64
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_non_ascending_indices_fail_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["selected_group_indices"] = [1, 0] + list(range(2, 96))
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_non_bypass_wrong_k_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["output_group_count"] = 95
        rows[0]["selected_group_indices"] = list(range(95))
        rows[0]["selected_group_indices_sha256"] = array_sha256(
            np.asarray(list(range(95)), dtype=np.int64)
        )
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_invalid_identity_bypass_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"], bypass=True)
        rows[0]["output_group_count"] = 7
        rows[0]["selected_group_indices"] = list(range(7))
        rows[0]["selected_group_indices_sha256"] = array_sha256(
            np.asarray(list(range(7)), dtype=np.int64)
        )
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_valid_identity_bypass_passes(self) -> None:
        indices = self._indices()
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"], bypass=True),
            "arm_float",
        )
        indices["arm_indices"][summ.ARM_INTEGER] = summ.index_unique(
            _arm_rows(summ.ARM_INTEGER, self.outputs["arm_integer"], bypass=True),
            "arm_integer",
        )
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"], bypass=True),
            "arm_prompt",
        )
        result = summ.validate(**indices)
        self.assertTrue(result["identity_bypass_matches_G"])

    def test_grid_invariant_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["image_grid_thw"] = [1, 60, 79]
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_grid_mismatch_across_arms_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"])
        rows[0]["image_grid_thw"] = [1, 40, 80]
        rows[0]["full_patch_tokens"] = 3200
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices)

    def test_checkpoint_hashes_cross_check(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_PROMPT, self.outputs["arm_prompt"])
        for row in rows:
            row.update(CHECKPOINT_HASHES)
            row["record_sha256"] = summ.record_sha256(row)
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        summ.validate(**indices, checkpoint_hashes=CHECKPOINT_HASHES)

        rows[0]["wv_sha256"] = "f" * 64
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_PROMPT] = summ.index_unique(
            rows, "arm_prompt"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices, checkpoint_hashes=CHECKPOINT_HASHES)

    def test_no_prompt_arm_carrying_checkpoint_fails_closed(self) -> None:
        indices = self._indices()
        rows = _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"])
        rows[0]["checkpoint_sha256"] = "c" * 64
        rows[0]["record_sha256"] = summ.record_sha256(rows[0])
        indices["arm_indices"][summ.ARM_FLOAT] = summ.index_unique(
            rows, "arm_float"
        )
        with self.assertRaises(summ.InputContractError):
            summ.validate(**indices, checkpoint_hashes=CHECKPOINT_HASHES)


class RelativeVsAbsoluteMetricTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.outputs = _outputs()
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
            full_manifest=self._write("full.jsonl", _manifest_rows()),
            dense_full_jsonl=self._write(
                "dense.jsonl", _ref_rows(self.outputs["dense"])
            ),
            old_current_integer_full_jsonl=self._write(
                "old_int.jsonl", _ref_rows(self.outputs["old_current_integer"])
            ),
            arm_float_jsonl=self._write(
                "arm_float.jsonl",
                _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"]),
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
            expected_full_count=3,
            manifest_sha256=MANIFEST_SHA,
            **kwargs,
        )

    def test_decision_uses_only_paired_delta_ci(self) -> None:
        # Absolute arm accuracies are not decision inputs at all.
        first = summ.decide([-0.005, 0.10], [0.02, 0.20])
        second = summ.decide([-0.005, 0.90], [0.02, 0.99])
        self.assertEqual(first["outcome"], second["outcome"])
        self.assertEqual(first["outcome"], summ.OUTCOME_PASS)

    def test_official_primary_separates_from_unweighted(self) -> None:
        metric = summ.arm_metrics(
            [
                {
                    "task": "Count",
                    "type": "2D",
                    "source": "ADE20K",
                    "source_dataset": "ADE20K Validation Set",
                    "left_correct": True,
                    "left_answer": "(A)",
                },
                {
                    "task": "Depth",
                    "type": "2D",
                    "source": "COCO",
                    "source_dataset": "COCO",
                    "left_correct": False,
                    "left_answer": "(B)",
                },
                {
                    "task": "Distance",
                    "type": "3D",
                    "source": "Omni3D",
                    "source_dataset": "Omni3D",
                    "left_correct": True,
                    "left_answer": "(C)",
                },
            ],
            "left",
            "left",
        )
        self.assertEqual(metric["primary_metric"], summ.PRIMARY_METRIC)
        # combined = ((1.0 + 0.0)/2 + 1.0)/2 = 0.75; unweighted = 2/3.
        self.assertAlmostEqual(metric["cv_bench_accuracy"], 0.75)
        self.assertAlmostEqual(metric["unweighted_sample_accuracy"], 2 / 3)
        self.assertNotEqual(
            metric["cv_bench_accuracy"], metric["unweighted_sample_accuracy"]
        )

    def test_paired_comparison_reports_primary_delta(self) -> None:
        from evaluation.build_cv_bench_pruning_report import score

        code, report = self._run()
        self.assertEqual(code, 0)
        joined = report["joined"]
        rows = screen._pair_score_rows(joined, summ.ARM_INTEGER, summ.ARM_FLOAT)
        expected = (
            score(rows, "left_correct")["cv_bench_accuracy"]
            - score(rows, "right_correct")["cv_bench_accuracy"]
        )
        comparison = report["comparisons"]["integer_minus_float"]
        self.assertEqual(comparison["metric"], summ.PRIMARY_METRIC)
        self.assertAlmostEqual(comparison["delta_point_estimate"], expected)

    def test_output_path_normalises_relative_and_absolute(self) -> None:
        # Absolute and relative declarations both resolve to the same paths.
        good = {
            "output_paths": {
                summ.ARM_FLOAT: str(self.paths.arm_float_jsonl.resolve()),
                summ.ARM_INTEGER: self.paths.arm_integer_jsonl.name,
                summ.ARM_PROMPT: str(self.paths.arm_prompt_jsonl),
            }
        }
        result = summ.verify_output_paths(good, self.paths, root=self.tmp)
        self.assertTrue(result["output_paths_verified"])

        good["output_paths"][summ.ARM_INTEGER] = "wrong.jsonl"
        with self.assertRaises(summ.InputContractError):
            summ.verify_output_paths(good, self.paths, root=self.tmp)


class BuildAndWriteTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.outputs = _outputs()
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
            full_manifest=self._write("full.jsonl", _manifest_rows()),
            dense_full_jsonl=self._write(
                "dense.jsonl", _ref_rows(self.outputs["dense"])
            ),
            old_current_integer_full_jsonl=self._write(
                "old_int.jsonl", _ref_rows(self.outputs["old_current_integer"])
            ),
            arm_float_jsonl=self._write(
                "arm_float.jsonl",
                _arm_rows(summ.ARM_FLOAT, self.outputs["arm_float"]),
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
            expected_full_count=3,
            manifest_sha256=MANIFEST_SHA,
            **kwargs,
        )

    def test_analyze_produces_five_arms_and_formal_decision(self) -> None:
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
        self.assertFalse(report["diagnostic_only"])
        self.assertTrue(report["formal_full2638_conclusion"])
        self.assertEqual(report["primary_metric"]["identifier"], summ.PRIMARY_METRIC)
        self.assertNotIn("old_defective_l2", report["metrics"])

    def test_success_writes_json_csv_and_markdown(self) -> None:
        code, _ = self._run()
        self.assertEqual(code, 0)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertTrue((self.tmp / suffix).is_file(), suffix)
        rows = (self.tmp / "report.csv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 4)  # header + 3 samples
        header = rows[0]
        self.assertNotIn("old_defective_l2", header)
        markdown = (self.tmp / "report.md").read_text(encoding="utf-8")
        self.assertIn("full2638", markdown)
        self.assertIn("量化门", markdown)
        self.assertIn("Prompt 增益", markdown)
        for arm in summ.ARM_ORDER:
            self.assertIn(arm, markdown)

    def test_no_write_on_missing_input(self) -> None:
        broken = summ.InputPaths(
            **{
                **self.paths.__dict__,
                "arm_prompt_jsonl": self.tmp / "missing_prompt.jsonl",
            }
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertFalse((self.tmp / suffix).is_file(), suffix)

    def test_no_write_on_contract_failure(self) -> None:
        bad_rows = _arm_rows(summ.ARM_INTEGER, self.outputs["arm_integer"])
        bad_rows[0]["output_group_count"] = 95
        broken = summ.InputPaths(
            **{
                **self.paths.__dict__,
                "arm_integer_jsonl": self._write("bad_integer.jsonl", bad_rows),
            }
        )
        code, report = self._run(paths=broken)
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        for suffix in ("report.json", "report.csv", "report.md"):
            self.assertFalse((self.tmp / suffix).is_file(), suffix)


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
        result = summ.verify_config_pins(self.config, self._paths(), root=ROOT)
        self.assertTrue(result["pins_verified"])
        self.assertTrue(result["checks"])

    def test_real_config_output_paths_match_defaults(self) -> None:
        result = summ.verify_output_paths(self.config, summ.DEFAULT_PATHS)
        self.assertTrue(result["output_paths_verified"])

    def test_tampered_config_pin_fails_closed(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["inputs_sha256"]["metrics"]["sha256"] = "0" * 64
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

    def test_declared_outcomes_match_decision_order(self) -> None:
        outcomes = [
            entry["outcome"]
            for entry in self.config["decision_rule"]["ordered_outcomes"]
        ]
        self.assertEqual(outcomes, list(summ.DECISION_ORDER))
        self.assertEqual(
            set(outcomes),
            {
                summ.OUTCOME_PASS,
                summ.OUTCOME_PROMPT_NOT_CONFIRMED,
                summ.OUTCOME_QUANTIZATION_FAIL,
            },
        )


if __name__ == "__main__":
    unittest.main()
