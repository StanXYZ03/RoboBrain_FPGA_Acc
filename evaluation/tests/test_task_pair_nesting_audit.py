#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.task_pair_nesting_audit`.

Covers the prompt-required surface:

* geometry boundaries (inclusive edges, tolerance, clipped area, invalid bbox),
* per-image three-seed median aggregation and stratification,
* the frozen three-condition decision rule,
* input isolation (no NPZ / image / weight / GPU / network / internal test),
* preregistration identity and input-pin immutability,
* the JSON / Markdown / records-CSV report schema.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation import task_pair_nesting_audit as audit


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------
def _record(
    *,
    sample_id: str,
    image: str,
    task: str,
    seed: int,
    gap: float,
    jaccard: float,
    arm: str = audit.PRIMARY_ARM,
    budget: int = audit.PRIMARY_BUDGET,
) -> dict[str, str]:
    return {
        "arm": arm,
        "seed": str(seed),
        "sample_id": sample_id,
        "image_sha256": image,
        "task": task,
        "group_count": "81",
        "grid_h": "9",
        "grid_w": "9",
        "budget": str(budget),
        "correct_retained_spatial_target_mass": "0",
        "wrong_retained_spatial_target_mass": "0",
        "gap_retained_spatial_target_mass": repr(float(gap)),
        "correct_wrong_selected_jaccard": repr(float(jaccard)),
    }


def _three_seed_rows(
    *,
    sample_id: str,
    image: str,
    task: str,
    gaps,
    jaccards,
) -> list[dict[str, str]]:
    return [
        _record(
            sample_id=sample_id,
            image=image,
            task=task,
            seed=seed,
            gap=gap,
            jaccard=jaccard,
        )
        for seed, gap, jaccard in zip(audit.SEEDS, gaps, jaccards)
    ]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
class GeometryBoundaryTests(unittest.TestCase):
    def test_point_on_every_edge_is_inside(self):
        bbox = [0.0, 0.0, 10.0, 10.0]
        self.assertTrue(audit.record_point_inside([0.0, 0.0], bbox))
        self.assertTrue(audit.record_point_inside([10.0, 10.0], bbox))
        self.assertTrue(audit.record_point_inside([0.0, 10.0], bbox))
        self.assertTrue(audit.record_point_inside([10.0, 0.0], bbox))

    def test_point_outside_is_outside(self):
        bbox = [0.0, 0.0, 10.0, 10.0]
        self.assertFalse(audit.record_point_inside([10.001, 5.0], bbox))
        self.assertFalse(audit.record_point_inside([5.0, -0.001], bbox))

    def test_tolerance_allows_sub_nano_boundary(self):
        bbox = [0.0, 0.0, 10.0, 10.0]
        self.assertTrue(audit.record_point_inside([10.0 + 5e-10, 5.0], bbox))
        self.assertFalse(audit.record_point_inside([10.0 + 5e-8, 5.0], bbox))

    def test_bbox_area_norm_exact_and_clipped(self):
        self.assertAlmostEqual(audit.bbox_area_norm([0.0, 0.0, 500.0, 500.0]), 0.25)
        self.assertAlmostEqual(audit.bbox_area_norm([-100.0, -100.0, 500.0, 500.0]), 0.25)
        self.assertAlmostEqual(audit.bbox_area_norm([900.0, 900.0, 1500.0, 1500.0]), 0.01)

    def test_bbox_area_rejects_degenerate(self):
        with self.assertRaises(audit.AuditContractError):
            audit.bbox_area_norm([0.0, 0.0, 0.0, 10.0])
        with self.assertRaises(audit.AuditContractError):
            audit.bbox_area_norm([0.0, 0.0, 10.0, 10.0, 20.0])

    def test_percentile_interpolates(self):
        values = [0.0, 1.0, 2.0, 3.0]
        self.assertAlmostEqual(audit._percentile(values, 0.25), 0.75)
        self.assertAlmostEqual(audit._percentile(values, 0.5), 1.5)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
class AggregationTests(unittest.TestCase):
    def test_per_image_task_median_across_three_seeds(self):
        rows = _three_seed_rows(
            sample_id="g1",
            image="img1",
            task="grounding",
            gaps=[-0.3, 0.1, 0.2],
            jaccards=[0.1, 0.2, 0.3],
        )
        medians = audit.per_image_task_medians(rows)
        self.assertAlmostEqual(medians["img1"]["grounding"], 0.1)
        self.assertAlmostEqual(medians["img1"]["grounding__jaccard"], 0.2)

    def test_per_image_task_median_rejects_ragged_seed_series(self):
        rows = _three_seed_rows(
            sample_id="g1",
            image="img1",
            task="grounding",
            gaps=[0.1, 0.2, 0.3],
            jaccards=[0.1, 0.2, 0.3],
        )[:2]
        with self.assertRaises(audit.AuditContractError):
            audit.per_image_task_medians(rows)

    def test_wrong_arm_and_budget_are_filtered(self):
        rows = _three_seed_rows(
            sample_id="g1",
            image="img1",
            task="grounding",
            gaps=[0.1, 0.2, 0.3],
            jaccards=[0.1, 0.2, 0.3],
        )
        rows += [
            _record(
                sample_id="g1",
                image="img1",
                task="grounding",
                seed=seed,
                gap=999.0,
                jaccard=999.0,
                arm=audit.CONTROL_ARM,
            )
            for seed in audit.SEEDS
        ]
        rows += [
            _record(
                sample_id="g1",
                image="img1",
                task="grounding",
                seed=seed,
                gap=777.0,
                jaccard=777.0,
                budget=64,
            )
            for seed in audit.SEEDS
        ]
        medians = audit.per_image_task_medians(rows)
        self.assertAlmostEqual(medians["img1"]["grounding"], 0.2)

    def test_aggregate_gaps_mean_and_median(self):
        medians = {
            "i1": {"grounding": -0.2, "pointing": 0.4, "grounding__jaccard": 0.1, "pointing__jaccard": 0.2},
            "i2": {"grounding": 0.2, "pointing": 0.6, "grounding__jaccard": 0.3, "pointing__jaccard": 0.4},
        }
        gaps = audit.aggregate_gaps(medians, ["i1", "i2"])
        self.assertAlmostEqual(gaps["grounding"]["mean"], 0.0)
        self.assertAlmostEqual(gaps["pointing"]["mean"], 0.5)
        self.assertAlmostEqual(gaps["overall"]["mean"], 0.25)
        self.assertEqual(gaps["grounding"]["n_images"], 2)

    def test_aggregate_jaccard(self):
        medians = {
            "i1": {"grounding": 0.0, "pointing": 0.0, "grounding__jaccard": 0.2, "pointing__jaccard": 0.6},
            "i2": {"grounding": 0.0, "pointing": 0.0, "grounding__jaccard": 0.4, "pointing__jaccard": 0.8},
        }
        jaccard = audit.aggregate_jaccard(medians, ["i1", "i2"])
        self.assertAlmostEqual(jaccard["grounding"]["mean"], 0.3)
        self.assertAlmostEqual(jaccard["pointing"]["mean"], 0.7)

    def test_stratify_both_and_single_stratum(self):
        medians = {
            "i1": {"grounding": -0.1, "pointing": 0.3, "grounding__jaccard": 0.2, "pointing__jaccard": 0.2},
            "i2": {"grounding": -0.3, "pointing": 0.5, "grounding__jaccard": 0.4, "pointing__jaccard": 0.4},
        }
        both = audit.stratify_by_point_inside(medians, ["i1", "i2"], {"i1": True, "i2": False})
        self.assertEqual(both["point_inside_true"]["n_images"], 1)
        self.assertEqual(both["point_inside_false"]["n_images"], 1)
        single = audit.stratify_by_point_inside(medians, ["i1", "i2"], {"i1": True, "i2": True})
        self.assertEqual(single["point_inside_false"]["n_images"], 0)
        self.assertIsNone(single["point_inside_false"]["grounding_gap_mean"])
        self.assertIn("empty", single["note"])

    def test_spearman_area_vs_grounding_gap(self):
        medians = {
            "i1": {"grounding": -0.3, "pointing": 0.0, "grounding__jaccard": 0.0, "pointing__jaccard": 0.0},
            "i2": {"grounding": 0.1, "pointing": 0.0, "grounding__jaccard": 0.0, "pointing__jaccard": 0.0},
            "i3": {"grounding": 0.5, "pointing": 0.0, "grounding__jaccard": 0.0, "pointing__jaccard": 0.0},
        }
        areas = {"i1": 0.1, "i2": 0.2, "i3": 0.9}
        result = audit.spearman_area_vs_grounding_gap(medians, ["i1", "i2", "i3"], areas)
        self.assertAlmostEqual(result["bbox_area_vs_grounding_gap_k48"], 1.0)
        self.assertEqual(result["n_images"], 3)


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------
class DecisionRuleTests(unittest.TestCase):
    def test_nested_threshold_boundary(self):
        self.assertTrue(audit.decide_task_pair_nesting(0.80, -0.1, 0.1)["same_image_task_targets_nested"])
        self.assertFalse(audit.decide_task_pair_nesting(0.799999, -0.1, 0.1)["same_image_task_targets_nested"])

    def test_symmetric_negative_invalid_requires_asymmetric_gap(self):
        result = audit.decide_task_pair_nesting(0.9, -0.001, 0.2)
        self.assertEqual(result["decision"], audit.DECISION_SYMMETRIC_INVALID)
        self.assertTrue(result["grounding_gap_negative_or_zero"])
        self.assertTrue(result["pointing_gap_positive"])

    def test_zero_grounding_gap_counts_as_non_positive(self):
        result = audit.decide_task_pair_nesting(0.9, 0.0, 0.2)
        self.assertEqual(result["decision"], audit.DECISION_SYMMETRIC_INVALID)

    def test_positive_grounding_gap_is_not_established(self):
        self.assertEqual(
            audit.decide_task_pair_nesting(0.9, 0.01, 0.2)["decision"],
            audit.DECISION_NOT_ESTABLISHED,
        )

    def test_non_positive_pointing_gap_is_not_established(self):
        self.assertEqual(
            audit.decide_task_pair_nesting(0.9, -0.1, 0.0)["decision"],
            audit.DECISION_NOT_ESTABLISHED,
        )

    def test_not_nested_is_not_established_even_with_asymmetric_gap(self):
        self.assertEqual(
            audit.decide_task_pair_nesting(0.79, -0.1, 0.2)["decision"],
            audit.DECISION_NOT_ESTABLISHED,
        )


# ---------------------------------------------------------------------------
# Input isolation
# ---------------------------------------------------------------------------
class InputIsolationTests(unittest.TestCase):
    def test_read_text_rejects_non_allow_listed_path(self):
        audit.reset_read_log()
        forbidden = audit.ROOT / "artifacts/token_selector/secret.npz"
        with self.assertRaises(audit.AuditContractError):
            audit.read_text(forbidden, purpose="npz")

    def test_allowed_paths_contain_no_npz_or_images(self):
        allowed = audit.allowed_read_paths()
        self.assertTrue(allowed)
        for path in allowed:
            lowered = path.lower()
            self.assertFalse(lowered.endswith(".npz"), path)
            self.assertNotIn("/coco2017/", lowered, path)
            self.assertNotIn(".gguf", lowered, path)

    def test_build_audit_never_calls_numpy_load_and_reads_only_allowed(self):
        audit.reset_read_log()
        with mock.patch("numpy.load", side_effect=AssertionError("numpy.load must not be called")):
            result = audit.build_audit()
        isolation = result["bundle"]["isolation"]
        self.assertEqual(isolation["npz_opened"], [])
        self.assertFalse(isolation["raw_images_opened"])
        self.assertFalse(isolation["model_weights_opened"])
        self.assertFalse(isolation["internal_test_access"])
        self.assertFalse(isolation["gpu_used"])
        self.assertFalse(isolation["network_used"])
        self.assertFalse(isolation["training"])
        self.assertTrue(set(isolation["read_paths"]) <= set(isolation["allowed_read_paths"]))

    def test_read_log_holds_only_text_inputs(self):
        audit.reset_read_log()
        audit.build_audit()
        observed = set(audit.read_log())
        self.assertEqual(observed, set(audit.allowed_read_paths()))
        self.assertFalse(any(path.endswith(".npz") for path in observed))


# ---------------------------------------------------------------------------
# Preregistration identity and immutability
# ---------------------------------------------------------------------------
class PreregImmutabilityTests(unittest.TestCase):
    def test_identity_config_and_prompt_match_report(self):
        prereg = audit.load_prereg()
        self.assertEqual(prereg["config_sha256"], prereg["report"]["config"]["sha256"])
        self.assertEqual(prereg["prompt_sha256"], prereg["config"]["authority_prompt"]["sha256"])
        self.assertEqual(prereg["prompt_sha256"], prereg["report"]["authority_prompt"]["sha256"])

    def test_real_pins_verify(self):
        prereg = audit.load_prereg()
        pins = audit.verify_input_pins(prereg)
        self.assertEqual(pins["mismatches"], {})
        self.assertEqual(set(pins["actual"]), set(pins["pinned"]))

    def test_verify_pins_detects_tampered_expected_hash(self):
        with self.assertRaises(audit.AuditContractError):
            audit.verify_input_pins(
                {"report": {"frozen_inputs": {str(audit.MANIFEST_PATH): "0" * 64}}}
            )

    def test_protected_files_unchanged_across_run(self):
        before = audit.snapshot_protected_hashes()
        bundle = audit.build_audit()["bundle"]
        after = audit.snapshot_protected_hashes()
        self.assertEqual(before, after)
        self.assertTrue(bundle["provenance"]["protected_files_unchanged"])
        self.assertTrue(bundle["provenance"]["hashes_before"] == bundle["provenance"]["hashes_after"])


# ---------------------------------------------------------------------------
# Schema / real-data regression
# ---------------------------------------------------------------------------
class RealDataRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = audit.build_audit()
        cls.bundle = cls.result["bundle"]

    def test_geometry_and_decision(self):
        geometry = self.bundle["point_inside_bbox"]
        self.assertEqual(geometry["record_count"], 76)
        self.assertEqual(geometry["image_count"], 38)
        self.assertAlmostEqual(geometry["overall_ratio_primary"], 1.0)
        self.assertAlmostEqual(geometry["image_ratio_secondary"], 1.0)
        decision = self.bundle["decision"]
        self.assertTrue(decision["same_image_task_targets_nested"])
        self.assertEqual(decision["decision"], audit.DECISION_SYMMETRIC_INVALID)
        self.assertIn(decision["decision"], audit.ALLOWED_DECISIONS)

    def test_gap_values_match_operational_json(self):
        gaps = self.bundle["gaps"]
        self.assertAlmostEqual(gaps["grounding"]["mean"], -0.0019252304813808796, places=12)
        self.assertAlmostEqual(gaps["pointing"]["mean"], 0.2877651809180278, places=12)
        self.assertEqual(gaps["grounding"]["n_images"], 38)
        self.assertEqual(gaps["pointing"]["n_images"], 38)
        self.assertTrue(self.bundle["cross_check"]["matches_within_tolerance"])

    def test_stratification_is_constant(self):
        strata = self.bundle["stratification"]
        self.assertEqual(strata["point_inside_true"]["n_images"], 38)
        self.assertEqual(strata["point_inside_false"]["n_images"], 0)

    def test_required_schema_keys_and_json_serializable(self):
        required = {
            "schema_version",
            "report_stem",
            "date",
            "status",
            "decision",
            "point_inside_bbox",
            "bbox_area_norm",
            "gaps",
            "stratification",
            "spearman",
            "jaccard",
            "cross_check",
            "provenance",
            "isolation",
            "split_counts",
            "limitations",
        }
        self.assertTrue(required <= set(self.bundle))
        json.dumps(self.bundle)

    def test_record_rows_schema_and_count(self):
        rows = self.result["record_rows"]
        self.assertEqual(len(rows), 76)
        for row in rows:
            self.assertEqual(set(row), set(audit.RECORDS_CSV_FIELDS))
        self.assertEqual({row["task"] for row in rows}, set(audit.TASKS))
        self.assertEqual(len({row["image_sha256"] for row in rows}), 38)


class WritersTests(unittest.TestCase):
    def test_write_outputs_produces_expected_files(self):
        result = audit.build_audit()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = audit.write_outputs(result["bundle"], result["record_rows"], root=root)
            json_path = Path(files["json"])
            md_path = Path(files["md"])
            csv_path = Path(files["csv"])
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            self.assertTrue(csv_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], result["bundle"]["status"])
            with csv_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(audit.RECORDS_CSV_FIELDS))
                csv_rows = list(reader)
            self.assertEqual(len(csv_rows), 76)
            markdown = md_path.read_text(encoding="utf-8")
            self.assertIn(result["bundle"]["decision"]["decision"], markdown)


if __name__ == "__main__":  # pragma: no cover - direct execution
    unittest.main()
