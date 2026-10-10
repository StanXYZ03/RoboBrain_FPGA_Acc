#!/usr/bin/env python3
"""Focused tests for the out-of-fold G-gated Prompt diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import analyze_full2638_oof_gated_prompt as oof
from evaluation import analyze_full2638_prompt_gain_heterogeneity as het
from evaluation.build_cv_bench_pruning_report import score

MANIFEST_SHA = "test-manifest-sha"
CHECKPOINT_PINS = {
    "checkpoint_sha256": "c" * 64,
    "wv_sha256": "w" * 64,
    "wv_scale_sha256": "s" * 64,
}

#: 3 strata x 10 samples = 30; sources ADE20K / COCO / Omni3D all present.
STRATA = (
    ("Count", "ADE20K", "ADE20K Validation Set"),
    ("Count", "COCO", "COCO 2017 Validation Set"),
    ("Depth", "Omni3D", "Omni3D_Hypersim"),
)
GROUP_COUNTS = (64, 200, 300, 768, 1400)
#: (never_text, always_text) by position within a stratum.
ANSWER_PATTERNS = (
    ("(A)", "(A)"),  # both right (ground truth is A)
    ("(A)", "(B)"),  # never right, always wrong
    ("(B)", "(A)"),  # never wrong, always right
    ("(B)", "(A)"),
    ("(B)", "(A)"),
)


def _indices_sample(kind: str, g: int) -> list[int]:
    if kind == "bypass":
        return list(range(g))
    return list(range(96))


def _build_specs() -> list[dict]:
    specs: list[dict] = []
    sample_id = 0
    for task, source, source_dataset in STRATA:
        for position in range(10):
            group_count = GROUP_COUNTS[position % 5]
            never_text, always_text = ANSWER_PATTERNS[position % 5]
            bypass = group_count <= 96
            kind = "bypass" if bypass else "range"
            specs.append(
                {
                    "sample_id": str(sample_id),
                    "task": task,
                    "source": source,
                    "source_dataset": source_dataset,
                    "answer": "(A)",
                    "G": group_count,
                    "never_text": never_text,
                    "always_text": always_text,
                    "kind": kind,
                }
            )
            sample_id += 1
    return specs


def _manifest_row(spec: dict) -> dict:
    return {
        "sample_id": spec["sample_id"],
        "task": spec["task"],
        "type": "3D" if spec["source"] == "Omni3D" else "2D",
        "source": spec["source"],
        "source_dataset": spec["source_dataset"],
        "choices": ["zero", "one"],
        "answer": spec["answer"],
        "max_new_tokens": 16,
    }


def _arm_row(spec: dict, output_text: str, arm: str) -> dict:
    g = spec["G"]
    selected = _indices_sample(spec["kind"], g)
    bypass = g <= 96
    full_patch = g * het.MERGE_UNIT
    row = {
        "sample_id": spec["sample_id"],
        "task": spec["task"],
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
        "image_grid_thw": [1, het.MERGE_UNIT, g],
        "full_llm_visual_tokens": g,
        "output_group_count": len(selected),
        "real_group_count": g if bypass else 87,
        "context_group_count": 0 if bypass else 9,
        "identity_bypass": bypass,
        "selected_group_indices": selected,
        "selected_group_indices_sha256": het.array_sha256(
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
    if arm == het.ARM_PROMPT:
        row.update(CHECKPOINT_PINS)
    row["record_sha256"] = het.record_sha256(row)
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class Fixture:
    def __init__(self, tmp: str) -> None:
        self.root = Path(tmp)
        self.specs = _build_specs()
        self.manifest_rows = [_manifest_row(spec) for spec in self.specs]
        self.arm_rows = {
            het.ARM_INTEGER: [
                _arm_row(spec, spec["never_text"], het.ARM_INTEGER)
                for spec in self.specs
            ],
            het.ARM_PROMPT: [
                _arm_row(spec, spec["always_text"], het.ARM_PROMPT)
                for spec in self.specs
            ],
        }
        self.manifest_path = self.root / "manifest.jsonl"
        self.integer_path = self.root / "integer.jsonl"
        self.prompt_path = self.root / "prompt.jsonl"
        self.report_path = self.root / "formal_report.json"
        _write_jsonl(self.manifest_path, self.manifest_rows)
        _write_jsonl(self.integer_path, self.arm_rows[het.ARM_INTEGER])
        _write_jsonl(self.prompt_path, self.arm_rows[het.ARM_PROMPT])
        self.manifest_index = {row["sample_id"]: row for row in self.manifest_rows}
        self.arm_indices = {
            arm: {row["sample_id"]: row for row in rows}
            for arm, rows in self.arm_rows.items()
        }
        self.report = self._build_report()
        self.write_report(self.report)

    def _build_report(self) -> dict:
        joined = het.build_joined(
            manifest_index=self.manifest_index,
            arm_indices=self.arm_indices,
            formal_joined_index=None,
        )
        frozen_joined = []
        for row in joined:
            frozen_joined.append(
                {
                    "sample_id": row["sample_id"],
                    "task": row["task"],
                    "source": row["source"],
                    "source_dataset": row["source_dataset"],
                    "ground_truth": row["ground_truth"],
                    f"{het.ARM_INTEGER}_correct": row[
                        f"{het.ARM_INTEGER}_correct"
                    ],
                    f"{het.ARM_INTEGER}_answer": row[f"{het.ARM_INTEGER}_answer"],
                    f"{het.ARM_PROMPT}_correct": row[f"{het.ARM_PROMPT}_correct"],
                    f"{het.ARM_PROMPT}_answer": row[f"{het.ARM_PROMPT}_answer"],
                    "dense_correct": True,
                    "dense_answer": "A",
                }
            )

        def _metric(correct_key: str, answer_key: str) -> float:
            rows = [
                {
                    "task": row["task"],
                    "source": row["source"],
                    "source_dataset": row["source_dataset"],
                    "correct": row[correct_key],
                    "answer": row[answer_key],
                }
                for row in frozen_joined
            ]
            return float(score(rows, "correct")["cv_bench_accuracy"])

        never_metric = _metric(
            f"{het.ARM_INTEGER}_correct", f"{het.ARM_INTEGER}_answer"
        )
        always_metric = _metric(
            f"{het.ARM_PROMPT}_correct", f"{het.ARM_PROMPT}_answer"
        )
        return {
            "decision": {"outcome": het.FORMAL_OUTCOME},
            "fixed_configuration": {
                "manifest": str(self.manifest_path),
                "sample_count": len(self.specs),
                "budget_K": 96,
                "region_grid": 3,
                "local_candidate_cap": 16,
                "integration_score_mode": "integer",
                "bootstrap_seed": het.BOOTSTRAP_SEED,
                "bootstrap_repeats": het.BOOTSTRAP_REPEATS,
            },
            "comparisons": {
                "prompt_minus_integer_l2": {
                    "metric": het.PRIMARY_METRIC,
                    "delta_point_estimate": always_metric - never_metric,
                    "delta_ci95": [0.0, 0.0],
                }
            },
            "inputs_sha256": {
                "full_manifest": {"sha256": MANIFEST_SHA},
                "corrected_l2_integer_jsonl": {
                    "sha256": het.sha256_file(self.integer_path)
                },
                "prompt_plus_corrected_l2_integer_jsonl": {
                    "sha256": het.sha256_file(self.prompt_path)
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
            "metrics": {
                "corrected_l2_integer": {"cv_bench_accuracy": never_metric},
                "prompt_plus_corrected_l2_integer": {
                    "cv_bench_accuracy": always_metric
                },
                "dense": {"cv_bench_accuracy": 1.0},
            },
            "joined": frozen_joined,
        }

    def write_report(self, report: dict) -> None:
        self.report = report
        self.report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def refresh_arm_shas(self) -> None:
        report = json.loads(json.dumps(self.report))
        report["inputs_sha256"]["corrected_l2_integer_jsonl"] = {
            "sha256": het.sha256_file(self.integer_path)
        }
        report["inputs_sha256"]["prompt_plus_corrected_l2_integer_jsonl"] = {
            "sha256": het.sha256_file(self.prompt_path)
        }
        self.write_report(report)

    def paths(self) -> oof.InputPaths:
        return oof.InputPaths(
            formal_report=self.report_path,
            full_manifest=self.manifest_path,
            arm_integer_jsonl=self.integer_path,
            arm_prompt_jsonl=self.prompt_path,
        )

    def analyze(self) -> dict:
        return oof.analyze(
            self.paths(),
            expected_pins=None,
            expected_sample_count=len(self.specs),
            manifest_sha256=MANIFEST_SHA,
        )


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)


class TestPolicyHelpers(unittest.TestCase):
    def test_never_and_always(self) -> None:
        self.assertFalse(oof.policy_uses_prompt("never_prompt", 1400))
        self.assertTrue(oof.policy_uses_prompt("always_prompt", 64))

    def test_gated_policy_strictly_greater(self) -> None:
        self.assertFalse(oof.policy_uses_prompt("prompt_if_G_gt_96", 96))
        self.assertTrue(oof.policy_uses_prompt("prompt_if_G_gt_96", 97))
        self.assertFalse(oof.policy_uses_prompt("prompt_if_G_gt_256", 256))
        self.assertTrue(oof.policy_uses_prompt("prompt_if_G_gt_256", 257))
        self.assertFalse(oof.policy_uses_prompt("prompt_if_G_gt_512", 512))
        self.assertTrue(oof.policy_uses_prompt("prompt_if_G_gt_512", 513))
        self.assertFalse(oof.policy_uses_prompt("prompt_if_G_gt_1024", 1024))
        self.assertTrue(oof.policy_uses_prompt("prompt_if_G_gt_1024", 1025))

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            oof.policy_uses_prompt("prompt_if_task_is_count", 1400)

    def _candidate(self, policy: str, accuracy: float, rate: float) -> dict:
        return {
            "policy": policy,
            "train_cv_bench_accuracy": accuracy,
            "train_prompt_use_rate": rate,
        }

    def test_higher_accuracy_wins(self) -> None:
        candidates = [
            self._candidate("never_prompt", 0.80, 0.0),
            self._candidate("always_prompt", 0.81, 1.0),
        ]
        self.assertEqual(oof.select_policy(candidates), "always_prompt")

    def test_accuracy_tie_prefers_lower_prompt_use(self) -> None:
        candidates = [
            self._candidate("never_prompt", 0.80, 0.0),
            self._candidate("always_prompt", 0.80, 1.0),
            self._candidate("prompt_if_G_gt_96", 0.80, 0.9),
        ]
        self.assertEqual(oof.select_policy(candidates), "never_prompt")

    def test_full_tie_keeps_fixed_order(self) -> None:
        candidates = [
            self._candidate("never_prompt", 0.80, 0.5),
            self._candidate("always_prompt", 0.80, 0.5),
        ]
        self.assertEqual(oof.select_policy(candidates), "never_prompt")

    def test_transition_label(self) -> None:
        self.assertEqual(oof.transition_label(True, False), "wrong_to_right")
        self.assertEqual(oof.transition_label(False, True), "right_to_wrong")
        self.assertEqual(oof.transition_label(True, True), "unchanged_right")
        self.assertEqual(oof.transition_label(False, False), "unchanged_wrong")


class TestFoldAssignment(unittest.TestCase):
    def _records(self) -> list[dict]:
        return [
            {
                "sample_id": str(index),
                "task": STRATA[index % len(STRATA)][0],
                "source_dataset": STRATA[index % len(STRATA)][2],
            }
            for index in range(30)
        ]

    def test_every_sample_in_one_fold(self) -> None:
        records = self._records()
        fold_of = oof.assign_folds(records, folds=5, seed=oof.FOLD_SEED)
        self.assertEqual(set(fold_of), {r["sample_id"] for r in records})
        self.assertTrue(all(0 <= fold < 5 for fold in fold_of.values()))

    def test_deterministic(self) -> None:
        records = self._records()
        first = oof.assign_folds(records, folds=5, seed=oof.FOLD_SEED)
        second = oof.assign_folds(records, folds=5, seed=oof.FOLD_SEED)
        self.assertEqual(first, second)

    def test_stratified_balanced(self) -> None:
        records = self._records()
        fold_of = oof.assign_folds(records, folds=5, seed=oof.FOLD_SEED)
        counts: dict[str, list[int]] = {}
        for record in records:
            counts.setdefault(record["source_dataset"], [0] * 5)[
                fold_of[record["sample_id"]]
            ] += 1
        for per_fold in counts.values():
            self.assertLessEqual(max(per_fold) - min(per_fold), 1)

    def test_invalid_fold_count(self) -> None:
        with self.assertRaises(ValueError):
            oof.assign_folds(self._records(), folds=1, seed=oof.FOLD_SEED)


class TestEndToEnd(BaseTest):
    def test_overall_shape(self) -> None:
        report = self.fx.analyze()
        self.assertEqual(report["oof"]["n"], len(self.fx.specs))
        self.assertEqual(len(report["per_sample"]), len(self.fx.specs))
        self.assertEqual(len(report["fold_selections"]), 5)
        self.assertTrue(report["exploratory_only"])
        self.assertTrue(report["not_a_formal_claim"])
        self.assertTrue(report["formal_result_unchanged"])

    def test_reproduces_frozen_baselines(self) -> None:
        report = self.fx.analyze()
        self.assertTrue(report["reproduction"]["never_matches_formal_integer"])
        self.assertTrue(report["reproduction"]["always_matches_formal_prompt"])
        self.assertTrue(report["reproduction"]["dense_matches_formal_dense"])

    def test_fold_selections_valid(self) -> None:
        report = self.fx.analyze()
        for fold in report["fold_selections"]:
            self.assertIn(fold["selected_policy"], oof.POLICY_ORDER)
            self.assertEqual(len(fold["candidates"]), len(oof.POLICY_ORDER))
            self.assertEqual(
                [c["policy"] for c in fold["candidates"]], list(oof.POLICY_ORDER)
            )
            self.assertGreaterEqual(fold["held_out_sample_count"], 1)

    def test_selection_is_deterministic_optimum(self) -> None:
        report = self.fx.analyze()
        for fold in report["fold_selections"]:
            expected = oof.select_policy(fold["candidates"])
            self.assertEqual(fold["selected_policy"], expected)

    def test_folds_partition_samples(self) -> None:
        report = self.fx.analyze()
        by_fold: dict[int, int] = {}
        for row in report["per_sample"]:
            by_fold[row["fold"]] = by_fold.get(row["fold"], 0) + 1
        self.assertEqual(sum(by_fold.values()), len(self.fx.specs))
        self.assertEqual(set(by_fold), {0, 1, 2, 3, 4})

    def test_prompt_use_rate_matches_selected_policies(self) -> None:
        report = self.fx.analyze()
        expected = sum(
            oof.policy_uses_prompt(row["selected_policy"], row["G"])
            for row in report["per_sample"]
        ) / len(report["per_sample"])
        self.assertAlmostEqual(report["oof"]["prompt_use_rate"], expected)

    def test_transitions_sum_to_n(self) -> None:
        report = self.fx.analyze()
        for key in ("oof_vs_never", "oof_vs_always"):
            counts = report["transitions"][key]
            total = sum(counts[name] for name in oof.TRANSITIONS)
            self.assertEqual(total, len(self.fx.specs))

    def test_comparisons_present(self) -> None:
        report = self.fx.analyze()
        for name in ("oof_minus_never", "oof_minus_always"):
            comparison = report["comparisons"][name]
            self.assertEqual(comparison["metric"], oof.PRIMARY_METRIC)
            self.assertEqual(len(comparison["delta_ci95"]), 2)
            self.assertLessEqual(
                comparison["delta_ci95"][0], comparison["delta_ci95"][1]
            )
        dense = report["comparisons"]["oof_minus_dense_descriptive"]
        self.assertIsNone(dense["delta_ci95"])

    def test_diagnostics_cover_all_strata(self) -> None:
        report = self.fx.analyze()
        self.assertEqual(
            {row["stratum"] for row in report["by_task"]},
            {stratum[0] for stratum in STRATA},
        )
        self.assertEqual(
            {row["stratum"] for row in report["by_source"]},
            {stratum[1] for stratum in STRATA},
        )
        self.assertTrue(report["by_G_bin"])

    def test_fold_assignment_recorded(self) -> None:
        report = self.fx.analyze()
        self.assertEqual(len(report["fold_assignments"]), len(self.fx.specs))
        self.assertEqual(
            set(report["fold_assignments"].values()), {0, 1, 2, 3, 4}
        )

    def test_selection_stability_flag(self) -> None:
        report = self.fx.analyze()
        unique = set(report["selection_counts"])
        self.assertEqual(report["selection_stable_across_folds"], len(unique) == 1)


class TestOutputs(BaseTest):
    def test_write_outputs_creates_files(self) -> None:
        report = self.fx.analyze()
        root = self.fx.root
        outputs = oof.OutputPaths(
            json=root / "out.json",
            csv=root / "out.csv",
            md=root / "out.md",
        )
        oof.write_outputs(report, outputs)
        self.assertTrue(outputs.json.is_file())
        self.assertTrue(outputs.csv.is_file())
        self.assertTrue(outputs.md.is_file())
        loaded = json.loads(outputs.json.read_text(encoding="utf-8"))
        self.assertEqual(loaded["analysis"], "full2638_oof_gated_prompt")
        csv_lines = outputs.csv.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(csv_lines), len(self.fx.specs) + 1)
        self.assertEqual(csv_lines[0].split(",")[0], "sample_id")
        markdown = outputs.md.read_text(encoding="utf-8")
        self.assertIn("探索", markdown)
        self.assertIn("外部冻结数据集", markdown)

    def test_markdown_mentions_caveat(self) -> None:
        report = self.fx.analyze()
        markdown = oof.render_markdown(report)
        self.assertIn(oof.EXTERNAL_VALIDATION_CAVEAT, markdown)
        self.assertIn("OOF", markdown)

    def _group_row(self, **overrides: object) -> dict:
        row: dict[str, object] = {
            "stratum": "97-256",
            "n": 781,
            "prompt_use_rate": 0.0,
            "oof_unweighted_accuracy": 0.878361,
            "never_unweighted_accuracy": 0.878361,
            "always_unweighted_accuracy": 0.860435,
            "oof_minus_never_unweighted": 0.0,
            "oof_minus_always_unweighted": 0.017926,
            "oof_official_cv_bench_accuracy": 0.914510,
            "transitions_vs_never": {"net_corrections": 0},
        }
        row.update(overrides)
        return row

    def test_summary_table_labels_subset_oof_official_without_sign(self) -> None:
        """Regression: the last group-table column is the subset OOF official
        score, not a delta, so it must be labelled ``OOF official`` and must
        never render with a leading ``+`` sign."""

        lines: list[str] = []
        oof._summary_table(lines, "按 pre-selector G 桶", [self._group_row()])
        header = next(line for line in lines if line.startswith("| 层 |"))
        self.assertIn("OOF official", header)
        self.assertNotIn("official Δ", header)
        data = next(line for line in lines if line.startswith("| 97-256 |"))
        self.assertIn("| 0.914510 |", data)
        self.assertNotIn("+0.914510", data)

    def test_summary_table_missing_subset_oof_official_is_na(self) -> None:
        lines: list[str] = []
        oof._summary_table(
            lines,
            "按 task",
            [
                self._group_row(
                    stratum="Count", oof_official_cv_bench_accuracy=None
                )
            ],
        )
        data = next(line for line in lines if line.startswith("| Count |"))
        self.assertTrue(data.endswith("| n/a |"))
        self.assertNotIn("+n/a", data)

    def test_markdown_group_tables_do_not_label_subset_oof_as_delta(self) -> None:
        report = self.fx.analyze()
        markdown = oof.render_markdown(report)
        self.assertNotIn("official Δ", markdown)
        self.assertIn("OOF official", markdown)
        # The by_G_bin table is the one whose subset OOF official scores are
        # defined on this fixture; inspect that column cell-by-cell so a genuine
        # signed delta in a neighbouring column cannot mask the regression.
        section = markdown.split("## 9. 按 pre-selector G 桶", 1)[1]
        section = section.split("## 10.", 1)[0]
        table_lines = [line for line in section.splitlines() if line.startswith("|")]
        self.assertTrue(table_lines[0].endswith("| OOF official |"))
        data_lines = table_lines[2:]  # skip header + alignment separator
        self.assertEqual(len(data_lines), len(report["by_G_bin"]))
        for row, line in zip(report["by_G_bin"], data_lines):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            self.assertEqual(
                cells[-1], oof._fmt(row["oof_official_cv_bench_accuracy"])
            )
            self.assertFalse(cells[-1].startswith("+"))


class TestFailClosed(BaseTest):
    def test_pin_mismatch(self) -> None:
        with self.assertRaises(het.InputContractError):
            oof.analyze(
                self.fx.paths(),
                expected_pins={"full_manifest": "0" * 64},
                expected_sample_count=len(self.fx.specs),
                manifest_sha256=MANIFEST_SHA,
            )

    def test_duplicate_sample_id(self) -> None:
        rows = list(self.fx.arm_rows[het.ARM_INTEGER])
        rows.append(dict(rows[0]))
        with self.assertRaises(het.InputContractError):
            het.index_unique(rows, "integer")

    def test_task_mismatch(self) -> None:
        self.fx.arm_indices[het.ARM_INTEGER]["0"]["task"] = "Depth"
        with self.assertRaises(het.InputContractError):
            het.validate_contract(
                manifest_index=self.fx.manifest_index,
                arm_indices=self.fx.arm_indices,
                manifest_sha256=MANIFEST_SHA,
                checkpoint_hashes=CHECKPOINT_PINS,
                expected_sample_count=len(self.fx.specs),
            )

    def test_record_hash_mismatch(self) -> None:
        self.fx.arm_indices[het.ARM_INTEGER]["0"]["record_sha256"] = "0" * 64
        with self.assertRaises(het.InputContractError):
            het.validate_contract(
                manifest_index=self.fx.manifest_index,
                arm_indices=self.fx.arm_indices,
                manifest_sha256=MANIFEST_SHA,
                checkpoint_hashes=CHECKPOINT_PINS,
                expected_sample_count=len(self.fx.specs),
            )

    def test_tampered_arm_file_fails_analysis(self) -> None:
        rows = [
            json.loads(line)
            for line in self.fx.integer_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows[0]["task"] = "Depth"
        _write_jsonl(self.fx.integer_path, rows)
        self.fx.refresh_arm_shas()
        with self.assertRaises(het.InputContractError):
            self.fx.analyze()

    def test_missing_dense_fails_closed(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        for row in report["joined"]:
            row.pop("dense_correct", None)
        self.fx.write_report(report)
        with self.assertRaises(oof.InputContractError):
            self.fx.analyze()

    def test_metric_mismatch_fails_closed(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["metrics"]["corrected_l2_integer"]["cv_bench_accuracy"] = 0.123
        self.fx.write_report(report)
        with self.assertRaises(oof.InputContractError):
            self.fx.analyze()

    def test_wrong_outcome_fails_closed(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["decision"]["outcome"] = "SOMETHING_ELSE"
        self.fx.write_report(report)
        with self.assertRaises(het.InputContractError):
            self.fx.analyze()

    def test_wrong_sample_count_fails_closed(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["fixed_configuration"]["sample_count"] = 999
        self.fx.write_report(report)
        with self.assertRaises(het.InputContractError):
            self.fx.analyze()

    def test_formal_report_arm_sha_mismatch(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["inputs_sha256"]["corrected_l2_integer_jsonl"]["sha256"] = "0" * 64
        self.fx.write_report(report)
        with self.assertRaises(het.InputContractError):
            self.fx.analyze()

    def test_verify_pins_wrong_expected(self) -> None:
        with self.assertRaises(het.InputContractError):
            het.verify_pins(self.fx.paths(), {"full_manifest": "0" * 64})

    def test_no_write_on_failure(self) -> None:
        report = json.loads(json.dumps(self.fx.report))
        report["metrics"]["corrected_l2_integer"]["cv_bench_accuracy"] = 0.123
        self.fx.write_report(report)
        root = self.fx.root
        outputs = oof.OutputPaths(
            json=root / "fail.json",
            csv=root / "fail.csv",
            md=root / "fail.md",
        )
        code, result = oof.run(
            self.fx.paths(),
            outputs,
            expected_pins=None,
            expected_sample_count=len(self.fx.specs),
            manifest_sha256=MANIFEST_SHA,
        )
        self.assertEqual(code, 2)
        self.assertIsNone(result)
        self.assertFalse(outputs.json.exists())
        self.assertFalse(outputs.csv.exists())
        self.assertFalse(outputs.md.exists())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
