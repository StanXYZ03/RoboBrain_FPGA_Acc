#!/usr/bin/env python3
"""Focused tests for the adaptive-vs-r25 screening100 diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import analyze_adaptive_vs_r25_screening100 as analysis
from evaluation.build_cv_bench_pruning_report import score as official_score
from evaluation.layer5_adaptive_budget import compute_adaptive_budget


def _manifest_row(sample_id: str, task: str, source: str, answer: str) -> dict:
    return {
        "sample_id": sample_id,
        "task": task,
        "type": "3D" if source == "Omni3D" else "2D",
        "source": source,
        "source_dataset": source + " set",
        "choices": ["zero", "one"],
        "answer": answer,
        "max_new_tokens": 16,
    }


def _arm_row(
    sample_id: str,
    task: str,
    indices: list[int],
    output_text: str,
    *,
    arm: str,
    g: int,
    k: int,
) -> dict:
    row = {
        "sample_id": sample_id,
        "task": task,
        "prune_after_layer": 5,
        "output_text": output_text,
        "raw_output_text": output_text,
        "selected_group_indices": list(indices),
        "selected_group_indices_sha256": analysis.selection_sha256(indices),
        "identity_bypass": k >= g,
    }
    if arm == analysis.ARM_A:
        row.update(
            {
                "budget_mode": "adaptive_layer5_unbounded_real_only",
                "full_llm_visual_tokens": g,
                "raw_budget": max(96, (g + 3) // 4),
                "effective_budget": k,
                "adaptive_retention": k / g,
            }
        )
    elif arm == analysis.ARM_B:
        row.update(
            {
                "full_llm_visual_tokens": g,
                "requested_budget": 96,
            }
        )
    else:
        row.update(
            {
                "baseline_llm_visual_tokens": g,
                "llm_visual_tokens": k,
                "actual_token_retention": k / g,
            }
        )
    return row


SAMPLE_SPECS = (
    # (sample_id, task, source, answer, G, C_K, A_text, B_text, C_text)
    ("1", "Count", "ADE20K", "(A)", 64, 16, "(A) zero", "(A) zero", "(A) zero"),
    ("2", "Depth", "COCO", "(B)", 200, 50, "(B) one", "(A) zero", "(B) one"),
    ("3", "Distance", "Omni3D", "(A)", 1200, 300, "(A) zero", "(B) one", "(B) one"),
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class FixtureMixin:
    def _build(self, tmp: str) -> analysis.InputPaths:
        root = Path(tmp)
        manifest = [ _manifest_row(sid, task, source, answer)
                     for sid, task, source, answer, *_ in SAMPLE_SPECS ]
        arm_rows = {arm: [] for arm in analysis.ARMS}
        for sid, task, _source, _answer, g, c_k, a_text, b_text, c_text in SAMPLE_SPECS:
            a_k = compute_adaptive_budget(g).effective_budget
            b_k = min(g, 96)
            arm_rows[analysis.ARM_A].append(
                _arm_row(sid, task, list(range(a_k)), a_text, arm=analysis.ARM_A, g=g, k=a_k)
            )
            arm_rows[analysis.ARM_B].append(
                _arm_row(sid, task, list(range(b_k)), b_text, arm=analysis.ARM_B, g=g, k=b_k)
            )
            arm_rows[analysis.ARM_C].append(
                _arm_row(sid, task, list(range(c_k)), c_text, arm=analysis.ARM_C, g=g, k=c_k)
            )
        manifest_path = root / "manifest.jsonl"
        _write_jsonl(manifest_path, manifest)
        a_path, b_path, c_path = (root / name for name in ("a.jsonl", "b.jsonl", "c.jsonl"))
        _write_jsonl(a_path, arm_rows[analysis.ARM_A])
        _write_jsonl(b_path, arm_rows[analysis.ARM_B])
        _write_jsonl(c_path, arm_rows[analysis.ARM_C])
        prereg = {"inputs_sha256": {}}
        prereg_a = root / "prereg_a.json"
        prereg_b = root / "prereg_b.json"
        prereg_a.write_text(json.dumps(prereg), encoding="utf-8")
        prereg_b.write_text(json.dumps(prereg), encoding="utf-8")
        return analysis.InputPaths(
            screening_manifest=manifest_path,
            adaptive_arm=a_path,
            fixed_k96_arm=b_path,
            r25_arm=c_path,
            adaptive_prereg=prereg_a,
            unbounded_prereg=prereg_b,
        )


class ContractTests(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = self._build(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_valid_fixture_passes(self) -> None:
        validated = analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))
        self.assertEqual(set(validated["indexes"][analysis.ARM_A]), {"1", "2", "3"})

    def test_duplicate_sample_id_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.adaptive_arm.read_text().splitlines()]
        rows.append(dict(rows[0]))
        _write_jsonl(self.paths.adaptive_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))

    def test_sample_set_mismatch_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.r25_arm.read_text().splitlines()]
        rows[0]["sample_id"] = "999"
        _write_jsonl(self.paths.r25_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))

    def test_prune_after_layer_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.fixed_k96_arm.read_text().splitlines()]
        rows[0]["prune_after_layer"] = 4
        _write_jsonl(self.paths.fixed_k96_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))

    def test_task_mismatch_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.adaptive_arm.read_text().splitlines()]
        rows[0]["task"] = "Relation"
        _write_jsonl(self.paths.adaptive_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))

    def test_non_ascending_indices_fail(self) -> None:
        row = _arm_row("1", "Count", [5, 4, 3], "(A) zero", arm=analysis.ARM_B, g=200, k=96)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_selected_indices(row, "test")

    def test_duplicate_indices_fail(self) -> None:
        row = _arm_row("1", "Count", [1, 1, 2], "(A) zero", arm=analysis.ARM_B, g=200, k=96)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_selected_indices(row, "test")

    def test_sha256_mismatch_fails(self) -> None:
        row = _arm_row("1", "Count", [1, 2, 3], "(A) zero", arm=analysis.ARM_B, g=200, k=96)
        row["selected_group_indices_sha256"] = "0" * 64
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_selected_indices(row, "test")

    def test_missing_file_fails(self) -> None:
        paths = analysis.InputPaths(
            screening_manifest=self.paths.screening_manifest,
            adaptive_arm=self.paths.adaptive_arm,
            fixed_k96_arm=self.paths.fixed_k96_arm,
            r25_arm=self.paths.r25_arm,
            adaptive_prereg=self.paths.adaptive_prereg,
            unbounded_prereg=self.paths.unbounded_prereg,
        )
        Path(paths.adaptive_arm).unlink()
        with self.assertRaises(analysis.InputContractError):
            analysis.verify_pins(paths, None)

    def test_pin_mismatch_fails(self) -> None:
        with self.assertRaises(analysis.InputContractError):
            analysis.verify_pins(self.paths, {"adaptive_arm": "deadbeef"})

    def test_g_inconsistent_across_arms_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.adaptive_arm.read_text().splitlines()]
        rows[0]["full_llm_visual_tokens"] = 129
        rows[0]["effective_budget"] = compute_adaptive_budget(129).effective_budget
        rows[0]["selected_group_indices"] = list(range(compute_adaptive_budget(129).effective_budget))
        rows[0]["selected_group_indices_sha256"] = analysis.selection_sha256(
            rows[0]["selected_group_indices"]
        )
        _write_jsonl(self.paths.adaptive_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))

    def test_adaptive_budget_field_mismatch_fails(self) -> None:
        rows = [json.loads(line) for line in self.paths.adaptive_arm.read_text().splitlines()]
        rows[0]["effective_budget"] = 12345
        _write_jsonl(self.paths.adaptive_arm, rows)
        with self.assertRaises(analysis.InputContractError):
            analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))


class ScoreTests(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = self._build(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_official_scores_use_official_function(self) -> None:
        report = analysis.analyze(self.paths, None, len(SAMPLE_SPECS))
        validated = analysis.validate_inputs(self.paths, None, len(SAMPLE_SPECS))
        joined = analysis.build_joined(
            validated["manifest"], validated["indexes"], validated["budgets"]
        )
        for arm in analysis.ARMS:
            expected = official_score(joined, f"{arm}_correct")
            self.assertAlmostEqual(
                report["official_scores"][arm]["cv_bench_accuracy"],
                expected["cv_bench_accuracy"],
            )
            self.assertAlmostEqual(
                report["official_scores"][arm]["unweighted_sample_accuracy_secondary"],
                expected["accuracy"],
            )
        self.assertEqual(report["primary_metric"], analysis.PRIMARY_METRIC)

    def test_unweighted_is_labelled_secondary(self) -> None:
        report = analysis.analyze(self.paths, None, len(SAMPLE_SPECS))
        self.assertIn("SECONDARY", report["unweighted_note"])
        for arm in analysis.ARMS:
            self.assertIn(
                "unweighted_sample_accuracy_secondary",
                report["official_scores"][arm],
            )

    def test_expected_two_way_comparisons_present(self) -> None:
        report = analysis.analyze(self.paths, None, len(SAMPLE_SPECS))
        self.assertIn("A_vs_C", report["transitions"])
        self.assertIn("A_vs_B", report["transitions"])
        self.assertIn("A_vs_C", report["selection_sets"])
        self.assertIn("A_vs_B", report["selection_sets"])


class SetStatisticsTests(unittest.TestCase):
    def test_jaccard_and_supersets(self) -> None:
        stats = analysis._set_stats([1, 2, 3], [2, 3, 4])
        self.assertEqual(stats["intersection_count"], 2)
        self.assertEqual(stats["union_count"], 4)
        self.assertAlmostEqual(stats["jaccard"], 0.5)
        self.assertFalse(stats["left_superset_right"])
        self.assertFalse(stats["right_superset_left"])

    def test_superset_flags(self) -> None:
        stats = analysis._set_stats([1, 2, 3], [1, 2])
        self.assertTrue(stats["left_superset_right"])
        self.assertFalse(stats["right_superset_left"])
        self.assertFalse(stats["equal"])

    def test_aggregate_set_stats(self) -> None:
        joined = [
            {
                "A_C_jaccard": 0.5,
                "A_C_intersection_count": 2,
                "A_C_union_count": 4,
                "A_C_left_superset_right": True,
                "A_C_right_superset_left": False,
                "A_C_equal": False,
            },
            {
                "A_C_jaccard": 1.0,
                "A_C_intersection_count": 3,
                "A_C_union_count": 3,
                "A_C_left_superset_right": True,
                "A_C_right_superset_left": True,
                "A_C_equal": True,
            },
        ]
        result = analysis.aggregate_set_stats(joined, "A_C")
        self.assertAlmostEqual(result["mean_jaccard"], 0.75)
        self.assertEqual(result["left_superset_right_count"], 2)
        self.assertEqual(result["right_superset_left_count"], 1)
        self.assertEqual(result["equal_count"], 1)


class TransitionTests(unittest.TestCase):
    def test_transition_counts_and_text_change(self) -> None:
        joined = [
            {"ref_correct": True, "treat_correct": True, "ref_output_text": "x", "treat_output_text": "x", "ref_answer": "A", "treat_answer": "A"},
            {"ref_correct": True, "treat_correct": False, "ref_output_text": "x", "treat_output_text": "y", "ref_answer": "A", "treat_answer": "B"},
            {"ref_correct": False, "treat_correct": True, "ref_output_text": "x", "treat_output_text": "x", "ref_answer": "A", "treat_answer": "A"},
            {"ref_correct": False, "treat_correct": False, "ref_output_text": "x", "treat_output_text": "z", "ref_answer": "B", "treat_answer": "A"},
        ]
        counts = analysis.transition_counts(joined, "ref", "treat")
        self.assertEqual(counts["reference_correct_treatment_correct"], 1)
        self.assertEqual(counts["reference_correct_treatment_incorrect"], 1)
        self.assertEqual(counts["reference_incorrect_treatment_correct"], 1)
        self.assertEqual(counts["reference_incorrect_treatment_incorrect"], 1)
        self.assertEqual(counts["output_text_changed"], 2)
        self.assertEqual(counts["answer_changed"], 2)
        self.assertAlmostEqual(counts["output_text_changed_rate"], 0.5)

    def test_bucket_labels(self) -> None:
        self.assertEqual(analysis._bucket_label(0, analysis.KDIFF_BUCKETS), "0")
        self.assertEqual(analysis._bucket_label(15, analysis.KDIFF_BUCKETS), "1-15")
        self.assertEqual(analysis._bucket_label(16, analysis.KDIFF_BUCKETS), "16-31")
        self.assertEqual(analysis._bucket_label(99, analysis.KDIFF_BUCKETS), "32+")
        self.assertEqual(analysis._bucket_label(96, analysis.G_BUCKETS), "G<=96")
        self.assertEqual(analysis._bucket_label(200, analysis.G_BUCKETS), "97-256")
        self.assertEqual(analysis._bucket_label(400, analysis.G_BUCKETS), "257-768")
        self.assertEqual(analysis._bucket_label(4000, analysis.G_BUCKETS), "G>768")


class AttributionTests(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = self._build(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_attribution_is_non_causal_when_k_ge_and_non_nested(self) -> None:
        scores = {
            analysis.ARM_A: {"cv_bench_accuracy": 0.70},
            analysis.ARM_B: {"cv_bench_accuracy": 0.72},
            analysis.ARM_C: {"cv_bench_accuracy": 0.80},
        }
        budget = {"A_minus_C_K_min": 0, "A_minus_C_K_max": 4}
        sets_ac = {"left_superset_right_count": 9, "right_superset_left_count": 0}
        verdict = analysis.attribution(scores, budget, sets_ac)
        self.assertEqual(verdict["verdict"], "NOT_DIRECT_EVIDENCE_OF_TOKEN_INSUFFICIENCY")
        self.assertIn("不是 token 数量不足的直接证据", verdict["statement_zh"])

    def test_attribution_falls_back_when_nested(self) -> None:
        scores = {
            analysis.ARM_A: {"cv_bench_accuracy": 0.70},
            analysis.ARM_B: {"cv_bench_accuracy": 0.72},
            analysis.ARM_C: {"cv_bench_accuracy": 0.80},
        }
        budget = {"A_minus_C_K_min": 0, "A_minus_C_K_max": 4}
        sets_ac = {
            "left_superset_right_count": analysis.SAMPLE_COUNT,
            "right_superset_left_count": 0,
        }
        verdict = analysis.attribution(scores, budget, sets_ac)
        self.assertEqual(verdict["verdict"], "BUDGET_ATTRIBUTION_NOT_ESTABLISHED")

    def test_minimal_experiments_same_budget_same_implementation(self) -> None:
        experiments = analysis.minimal_causal_experiments()
        self.assertEqual(len(experiments), 1)
        for item in experiments:
            self.assertFalse(item["executed"])
            self.assertEqual(item["only_variable"], "selected_group_indices")
            self.assertEqual(
                item["implementation"],
                "evaluation/hardware_regular_layer5_unbounded_l2.py",
            )
        self.assertEqual(experiments[0]["only_variable"], "selected_group_indices")
        self.assertIn("同一实现", experiments[0]["design"])
        self.assertIn("同一预算", experiments[0]["design"])


class ReportRenderTests(FixtureMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.paths = self._build(self._tmp.name)
        self.out = Path(self._tmp.name) / "out"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_write_outputs_produces_json_csv_md(self) -> None:
        report = analysis.analyze(self.paths, None, len(SAMPLE_SPECS))
        outputs = analysis.OutputPaths(
            json=self.out / "r.json",
            csv=self.out / "r.csv",
            md=self.out / "r.md",
        )
        analysis.write_outputs(report, outputs)
        self.assertTrue(outputs.json.is_file())
        self.assertTrue(outputs.csv.is_file())
        self.assertTrue(outputs.md.is_file())
        payload = json.loads(outputs.json.read_text())
        self.assertEqual(payload["contract"]["sample_count"], len(SAMPLE_SPECS))
        header = outputs.csv.read_text().splitlines()[0].split(",")
        self.assertEqual(header, list(analysis.CSV_COLUMNS))
        self.assertEqual(len(outputs.csv.read_text().splitlines()), len(SAMPLE_SPECS) + 1)
        md = outputs.md.read_text()
        self.assertIn("## 8. 严格归因判断", md)
        self.assertIn(payload["attribution"]["verdict"], md)


if __name__ == "__main__":
    unittest.main()
