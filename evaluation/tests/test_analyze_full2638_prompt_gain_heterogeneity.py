#!/usr/bin/env python3
"""Focused tests for the full2638 Prompt-gain heterogeneity diagnostic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import analyze_full2638_prompt_gain_heterogeneity as analysis

MANIFEST_SHA = "test-manifest-sha"
CHECKPOINT_PINS = {
    "checkpoint_sha256": "c" * 64,
    "wv_sha256": "w" * 64,
    "wv_scale_sha256": "s" * 64,
}

#: (sample_id, task, source, source_dataset, answer, G, integer_text,
#:  prompt_text, integer_indices, prompt_indices)
SPECS = (
    ("1", "Count", "ADE20K", "ADE20K Validation Set", "(A)", 64, "(A)", "(A)", "bypass", "bypass"),
    ("2", "Count", "COCO", "COCO 2017 Validation Set", "(A)", 200, "(A)", "(B)", "range", "range"),
    ("3", "Depth", "Omni3D", "Omni3D_Hypersim", "(A)", 300, "(A)", "(A)", "range", "shift1"),
    ("4", "Depth", "Omni3D", "Omni3D_SUNRGBD", "(A)", 150, "(B)", "(B)", "range", "tail"),
    ("5", "Distance", "Omni3D", "Omni3D_nuScenes", "(B)", 1200, "(A)", "(B)", "range", "shift10"),
    ("6", "Relation", "COCO", "COCO 2017 Validation Set", "(A)", 250, "(B)", "(A)", "range", "range"),
)

NON_BYPASS_INDICES = {
    "range": list(range(96)),
    "shift1": list(range(1, 97)),
    "tail": list(range(95)) + [200],
    "shift10": list(range(10, 106)),
}


def _indices(kind: str, g: int) -> list[int]:
    if kind == "bypass":
        return list(range(g))
    return list(NON_BYPASS_INDICES[kind])


def _manifest_row(spec: tuple) -> dict:
    sample_id, task, source, source_dataset, answer, _g, _i, _p, _ii, _pi = spec
    return {
        "sample_id": sample_id,
        "task": task,
        "type": "3D" if source == "Omni3D" else "2D",
        "source": source,
        "source_dataset": source_dataset,
        "choices": ["zero", "one"],
        "answer": answer,
        "max_new_tokens": 16,
    }


def _arm_row(spec: tuple, output_text: str, kind: str, arm: str) -> dict:
    sample_id, task, _source, _source_dataset, _answer, g, _i, _p, _ii, _pi = spec
    selected = _indices(kind, g)
    bypass = g <= 96
    full_patch = g * analysis.MERGE_UNIT
    row = {
        "sample_id": sample_id,
        "task": task,
        "arm": arm,
        "score_mode": "integer",
        "integration_score_mode": "integer",
        "prune_after_layer": 0,
        "cutpoint": "B1_after_vision_block_0",
        "output_text": output_text,
        "raw_output_text": output_text,
        "requested_budget": 96,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "full_patch_tokens": full_patch,
        "image_grid_thw": [1, analysis.MERGE_UNIT, g],
        "full_llm_visual_tokens": g,
        "output_group_count": len(selected),
        "real_group_count": g if bypass else 87,
        "context_group_count": 0 if bypass else 9,
        "identity_bypass": bypass,
        "selected_group_indices": selected,
        "selected_group_indices_sha256": analysis.array_sha256(
            np.asarray(selected, dtype=np.int64)
        ),
        "manifest_sha256": MANIFEST_SHA,
        "dtype": "bfloat16",
        "decoding": "greedy_argmax",
        "attention_implementation": "sdpa",
        "do_sample": False,
        "max_new_tokens": 16,
        "checkpoint_sha256": None,
        "wv_sha256": None,
        "wv_scale_sha256": None,
    }
    if arm == analysis.ARM_PROMPT:
        row.update(CHECKPOINT_PINS)
    row["record_sha256"] = analysis.record_sha256(row)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Fixture:
    def __init__(self, tmp: str) -> None:
        self.root = Path(tmp)
        self.manifest_rows = [_manifest_row(spec) for spec in SPECS]
        self.arm_rows = {
            analysis.ARM_INTEGER: [
                _arm_row(spec, spec[6], spec[8], analysis.ARM_INTEGER)
                for spec in SPECS
            ],
            analysis.ARM_PROMPT: [
                _arm_row(spec, spec[7], spec[9], analysis.ARM_PROMPT)
                for spec in SPECS
            ],
        }
        self.manifest_path = self.root / "manifest.jsonl"
        self.integer_path = self.root / "integer.jsonl"
        self.prompt_path = self.root / "prompt.jsonl"
        self.report_path = self.root / "formal_report.json"
        _write_jsonl(self.manifest_path, self.manifest_rows)
        _write_jsonl(self.integer_path, self.arm_rows[analysis.ARM_INTEGER])
        _write_jsonl(self.prompt_path, self.arm_rows[analysis.ARM_PROMPT])
        self.manifest_index = {
            row["sample_id"]: row for row in self.manifest_rows
        }
        self.arm_indices = {
            arm: {row["sample_id"]: row for row in rows}
            for arm, rows in self.arm_rows.items()
        }
        self.report = self._build_report()
        self.write_report(self.report)

    # -- report ---------------------------------------------------------
    def _build_report(self) -> dict:
        joined = analysis.build_joined(
            manifest_index=self.manifest_index,
            arm_indices=self.arm_indices,
            formal_joined_index=None,
        )
        delta = analysis.official_accuracy(
            joined, analysis.ARM_PROMPT
        ) - analysis.official_accuracy(joined, analysis.ARM_INTEGER)
        ci = analysis.paired_bootstrap_delta(
            analysis._pair_score_rows(joined),
            "left_correct",
            "right_correct",
            analysis.BOOTSTRAP_SEED,
            repeats=analysis.BOOTSTRAP_REPEATS,
        )
        frozen_joined = [
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "source": row["source"],
                "source_dataset": row["source_dataset"],
                "corrected_l2_integer_correct": row[
                    f"{analysis.ARM_INTEGER}_correct"
                ],
                "corrected_l2_integer_answer": row[
                    f"{analysis.ARM_INTEGER}_answer"
                ],
                "prompt_plus_corrected_l2_integer_correct": row[
                    f"{analysis.ARM_PROMPT}_correct"
                ],
                "prompt_plus_corrected_l2_integer_answer": row[
                    f"{analysis.ARM_PROMPT}_answer"
                ],
                "dense_correct": True,
                "dense_answer": "A",
            }
            for row in joined
        ]
        return {
            "decision": {"outcome": analysis.FORMAL_OUTCOME},
            "fixed_configuration": {
                "manifest": str(self.manifest_path),
                "sample_count": len(SPECS),
                "budget_K": 96,
                "region_grid": 3,
                "local_candidate_cap": 16,
                "integration_score_mode": "integer",
                "bootstrap_seed": analysis.BOOTSTRAP_SEED,
                "bootstrap_repeats": analysis.BOOTSTRAP_REPEATS,
            },
            "comparisons": {
                "prompt_minus_integer_l2": {
                    "metric": analysis.PRIMARY_METRIC,
                    "delta_point_estimate": delta,
                    "delta_ci95": [float(ci[0]), float(ci[1])],
                }
            },
            "inputs_sha256": {
                "full_manifest": {"sha256": MANIFEST_SHA},
                "corrected_l2_integer_jsonl": {
                    "sha256": analysis.sha256_file(self.integer_path)
                },
                "prompt_plus_corrected_l2_integer_jsonl": {
                    "sha256": analysis.sha256_file(self.prompt_path)
                },
            },
            "config_pin_verification": {
                "checks": [
                    {
                        "name": "checkpoint_projection_npz",
                        "expected_sha256": CHECKPOINT_PINS["checkpoint_sha256"],
                    },
                    {
                        "name": "checkpoint_wv_int8",
                        "expected_sha256": CHECKPOINT_PINS["wv_sha256"],
                    },
                    {
                        "name": "checkpoint_wv_per_tensor_scale",
                        "expected_sha256": CHECKPOINT_PINS["wv_scale_sha256"],
                    },
                ]
            },
            "metrics": {"dense": {"cv_bench_accuracy": 0.75}},
            "joined": frozen_joined,
        }

    def write_report(self, report: dict) -> None:
        self.report = report
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def refresh_arm_shas(self) -> None:
        report = dict(self.report)
        inputs = dict(report["inputs_sha256"])
        inputs["corrected_l2_integer_jsonl"] = {
            "sha256": analysis.sha256_file(self.integer_path)
        }
        inputs["prompt_plus_corrected_l2_integer_jsonl"] = {
            "sha256": analysis.sha256_file(self.prompt_path)
        }
        report["inputs_sha256"] = inputs
        self.write_report(report)

    def paths(self) -> analysis.InputPaths:
        return analysis.InputPaths(
            formal_report=self.report_path,
            full_manifest=self.manifest_path,
            arm_integer_jsonl=self.integer_path,
            arm_prompt_jsonl=self.prompt_path,
        )

    def analyze(self) -> dict:
        return analysis.analyze(
            self.paths(),
            expected_pins=None,
            expected_sample_count=len(SPECS),
            manifest_sha256=MANIFEST_SHA,
        )


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)


class TestEndToEnd(BaseTest):
    def test_overall_transitions(self) -> None:
        report = self.fx.analyze()
        overall = report["overall"]
        self.assertEqual(overall["n"], 6)
        self.assertEqual(overall["wrong_to_right"], 2)
        self.assertEqual(overall["right_to_wrong"], 1)
        self.assertEqual(overall["unchanged_right"], 2)
        self.assertEqual(overall["unchanged_wrong"], 1)
        self.assertEqual(overall["net_corrections"], 1)
        self.assertAlmostEqual(overall["wrong_to_right_rate"], 2 / 6)
        self.assertAlmostEqual(overall["right_to_wrong_rate"], 1 / 6)

    def test_formal_conclusion_unchanged(self) -> None:
        report = self.fx.analyze()
        self.assertTrue(report["formal_result_unchanged"])
        self.assertEqual(report["formal_outcome"], analysis.FORMAL_OUTCOME)
        self.assertTrue(
            report["formal_report_reproduction"]["matches_frozen_report"]
        )

    def test_by_identity_bypass(self) -> None:
        report = self.fx.analyze()
        rows = {
            str(row["stratum"]): row for row in report["by_identity_bypass"]
        }
        self.assertEqual(rows["False"]["n"], 5)
        self.assertEqual(rows["False"]["wrong_to_right"], 2)
        self.assertEqual(rows["False"]["right_to_wrong"], 1)
        self.assertEqual(rows["True"]["n"], 1)
        self.assertEqual(rows["True"]["wrong_to_right"], 0)
        self.assertEqual(rows["True"]["right_to_wrong"], 0)

    def test_by_task_counts(self) -> None:
        report = self.fx.analyze()
        rows = {row["stratum"]: row for row in report["by_task"]}
        self.assertEqual(rows["Count"]["net_corrections"], -1)
        self.assertEqual(rows["Depth"]["net_corrections"], 0)
        self.assertEqual(rows["Distance"]["net_corrections"], 1)
        self.assertEqual(rows["Relation"]["net_corrections"], 1)

    def test_pre_selector_bins(self) -> None:
        report = self.fx.analyze()
        rows = {
            row["stratum"]: row
            for row in report["by_pre_selector_real_group_count_bin"]
        }
        self.assertEqual(set(rows), {"<=96", "97-256", "257-512", ">1024"})
        self.assertEqual(rows["<=96"]["n"], 1)
        self.assertEqual(rows["97-256"]["n"], 3)
        self.assertEqual(rows["257-512"]["n"], 1)
        self.assertEqual(rows[">1024"]["n"], 1)
        self.assertNotIn("513-1024", rows)

    def test_recorded_field_bin_is_degenerate(self) -> None:
        report = self.fx.analyze()
        recorded = report["by_recorded_real_group_count_bin"]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["stratum"], "<=96")
        self.assertEqual(recorded[0]["n"], 6)

    def test_selection_overlap_non_bypass(self) -> None:
        report = self.fx.analyze()
        overlap = report["selection_index_overlap_non_bypass"]
        self.assertEqual(overlap["n"], 5)
        self.assertEqual(overlap["exact_match_count"], 2)
        self.assertAlmostEqual(overlap["exact_match_rate"], 2 / 5)
        expected_mean = (1 + 95 / 97 + 95 / 97 + 86 / 106 + 1) / 5
        self.assertAlmostEqual(overlap["jaccard_mean"], expected_mean)
        self.assertAlmostEqual(overlap["jaccard_min"], 86 / 106)
        by_task = {row["task"]: row for row in overlap["by_task"]}
        self.assertEqual(by_task["Count"]["exact_match_count"], 1)
        self.assertEqual(by_task["Relation"]["exact_match_rate"], 1.0)

    def test_dense_descriptive_present(self) -> None:
        report = self.fx.analyze()
        dense = report["dense_descriptive"]
        self.assertIsNotNone(dense)
        self.assertEqual(dense["cv_bench_accuracy"], 0.75)
        self.assertIn("never a gate", dense["role"])

    def test_contract_flags(self) -> None:
        report = self.fx.analyze()
        contract = report["contract"]
        for key in (
            "sample_ids_identical",
            "record_sha256_verified",
            "manifest_sha256_verified",
            "selected_indices_strictly_ascending_unique_and_hashed",
            "checkpoint_hashes_verified",
            "cross_arm_config_fields_consistent",
            "inputs_unchanged_during_analysis",
        ):
            self.assertTrue(contract[key], key)

    def test_per_sample_rows(self) -> None:
        report = self.fx.analyze()
        self.assertEqual(len(report["per_sample"]), 6)
        by_id = {row["sample_id"]: row for row in report["per_sample"]}
        self.assertEqual(by_id["1"]["transition"], "unchanged_right")
        self.assertTrue(by_id["1"]["index_exact_match"])
        self.assertEqual(by_id["2"]["transition"], "right_to_wrong")
        self.assertEqual(by_id["5"]["transition"], "wrong_to_right")
        self.assertFalse(by_id["3"]["index_exact_match"])


class TestPureHelpers(unittest.TestCase):
    def test_pre_selector_bin_boundaries(self) -> None:
        cases = {
            1: "<=96",
            96: "<=96",
            97: "97-256",
            256: "97-256",
            257: "257-512",
            512: "257-512",
            513: "513-1024",
            1024: "513-1024",
            1025: ">1024",
        }
        for value, expected in cases.items():
            self.assertEqual(analysis.pre_selector_bin(value), expected, value)

    def test_transition_label(self) -> None:
        base = {
            f"{analysis.ARM_INTEGER}_correct": False,
            f"{analysis.ARM_PROMPT}_correct": False,
        }
        self.assertEqual(analysis.transition_label(base), "unchanged_wrong")
        self.assertEqual(
            analysis.transition_label({**base, f"{analysis.ARM_PROMPT}_correct": True}),
            "wrong_to_right",
        )
        self.assertEqual(
            analysis.transition_label({**base, f"{analysis.ARM_INTEGER}_correct": True}),
            "right_to_wrong",
        )
        self.assertEqual(
            analysis.transition_label(
                {
                    f"{analysis.ARM_INTEGER}_correct": True,
                    f"{analysis.ARM_PROMPT}_correct": True,
                }
            ),
            "unchanged_right",
        )

    def test_record_sha256_matches_canonical_json(self) -> None:
        record = {"b": [2, 3], "a": 1}
        payload = {"a": 1, "b": [2, 3]}
        expected = hashlib.sha256(
            json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        self.assertEqual(analysis.record_sha256(record), expected)
        self.assertEqual(
            analysis.record_sha256({**record, "record_sha256": "ignored"}), expected
        )

    def test_array_sha256_dtype_shape_bytes(self) -> None:
        array = np.asarray([0, 1, 2], dtype=np.int64)
        contiguous = np.ascontiguousarray(array)
        header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
        expected = hashlib.sha256(
            len(header).to_bytes(8, "little")
            + header
            + contiguous.tobytes(order="C")
        ).hexdigest()
        self.assertEqual(analysis.array_sha256(array), expected)
        self.assertNotEqual(
            analysis.array_sha256(array.astype(np.int32)), expected
        )

    def test_bootstrap_is_deterministic(self) -> None:
        records = [
            {
                "task": "Count",
                "source_dataset": "A",
                f"{analysis.ARM_PROMPT}_correct": True,
                f"{analysis.ARM_INTEGER}_correct": False,
            },
            {
                "task": "Count",
                "source_dataset": "A",
                f"{analysis.ARM_PROMPT}_correct": False,
                f"{analysis.ARM_INTEGER}_correct": True,
            },
        ]
        first = analysis.bootstrap_unweighted_paired_delta(
            records, seed=20261004, repeats=500
        )
        second = analysis.bootstrap_unweighted_paired_delta(
            records, seed=20261004, repeats=500
        )
        self.assertEqual(first, second)
        self.assertLessEqual(first[0], first[1])


class TestFailClosed(BaseTest):
    def _indices(self):
        return (
            self.fx.manifest_index,
            self.fx.arm_indices,
            MANIFEST_SHA,
            CHECKPOINT_PINS,
        )

    def test_duplicate_sample_id(self) -> None:
        rows = list(self.fx.arm_rows[analysis.ARM_INTEGER])
        rows.append(dict(rows[0]))
        with self.assertRaises(analysis.InputContractError):
            analysis.index_unique(rows, "integer")

    def test_task_mismatch(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        arm_indices[analysis.ARM_INTEGER]["2"]["task"] = "Depth"
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_record_hash_mismatch(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        arm_indices[analysis.ARM_INTEGER]["2"]["record_sha256"] = "0" * 64
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_manifest_sha_mismatch(self) -> None:
        manifest_index, arm_indices, _sha, pins = self._indices()
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256="other-sha",
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_non_ascending_indices(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        row = arm_indices[analysis.ARM_INTEGER]["3"]
        row["selected_group_indices"] = list(reversed(row["selected_group_indices"]))
        row["selected_group_indices_sha256"] = analysis.array_sha256(
            np.asarray(row["selected_group_indices"], dtype=np.int64)
        )
        row["record_sha256"] = analysis.record_sha256(row)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_index_hash_mismatch(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        row = arm_indices[analysis.ARM_INTEGER]["3"]
        row["selected_group_indices_sha256"] = "f" * 64
        row["record_sha256"] = analysis.record_sha256(row)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_prompt_checkpoint_missing(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        row = arm_indices[analysis.ARM_PROMPT]["2"]
        row["checkpoint_sha256"] = None
        row["record_sha256"] = analysis.record_sha256(row)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_cross_arm_field_mismatch(self) -> None:
        manifest_index, arm_indices, sha, pins = self._indices()
        row = arm_indices[analysis.ARM_PROMPT]["2"]
        row["context_group_count"] = 8
        row["record_sha256"] = analysis.record_sha256(row)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_contract(
                manifest_index=manifest_index,
                arm_indices=arm_indices,
                manifest_sha256=sha,
                checkpoint_hashes=pins,
                expected_sample_count=6,
            )

    def test_wrong_formal_outcome(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["decision"]["outcome"] = "SOMETHING_ELSE"
        self.fx.write_report(report)
        with self.assertRaises(analysis.InputContractError):
            self.fx.analyze()

    def test_wrong_formal_sample_count(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["fixed_configuration"]["sample_count"] = 999
        self.fx.write_report(report)
        with self.assertRaises(analysis.InputContractError):
            self.fx.analyze()

    def test_formal_report_arm_sha_mismatch(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["inputs_sha256"]["corrected_l2_integer_jsonl"]["sha256"] = "0" * 64
        self.fx.write_report(report)
        with self.assertRaises(analysis.InputContractError):
            self.fx.analyze()

    def test_tampered_arm_file_fails_contract(self) -> None:
        rows = [
            json.loads(line)
            for line in self.fx.integer_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[1]["task"] = "Depth"
        _write_jsonl(self.fx.integer_path, rows)
        self.fx.refresh_arm_shas()
        with self.assertRaises(analysis.InputContractError):
            self.fx.analyze()

    def test_verify_pins_wrong_expected(self) -> None:
        with self.assertRaises(analysis.InputContractError):
            analysis.verify_pins(
                self.fx.paths(), {"full_manifest": "0" * 64}
            )


class TestOutputs(BaseTest):
    def test_write_outputs_creates_files(self) -> None:
        report = self.fx.analyze()
        root = self.fx.root
        outputs = analysis.OutputPaths(
            json=root / "out.json",
            csv=root / "out.csv",
            md=root / "out.md",
        )
        analysis.write_outputs(report, outputs)
        self.assertTrue(outputs.json.is_file())
        self.assertTrue(outputs.csv.is_file())
        self.assertTrue(outputs.md.is_file())
        loaded = json.loads(outputs.json.read_text(encoding="utf-8"))
        self.assertEqual(loaded["formal_outcome"], analysis.FORMAL_OUTCOME)
        csv_lines = outputs.csv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), 7)
        self.assertEqual(csv_lines[0].split(",")[0], "sample_id")
        markdown = outputs.md.read_text(encoding="utf-8")
        self.assertIn(analysis.FORMAL_OUTCOME, markdown)
        self.assertIn("探索", markdown)
        self.assertIn("不作为正式门", markdown)

    def test_render_markdown_mentions_exploratory(self) -> None:
        report = self.fx.analyze()
        markdown = analysis.render_markdown(report)
        self.assertIn(analysis.EXPLORATORY_NOTE, markdown)
        self.assertIn("非 bypass", markdown)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
