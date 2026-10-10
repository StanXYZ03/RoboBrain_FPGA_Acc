"""CPU-only focused tests for the B1 Oracle STOP postmortem.

These tests build a fully synthetic 24-sample world (six cells x two strata, the
frozen 10/3/9/2 transition split) to exercise fail-closed verification, the
index/source-type consistency checks and the metric formulas.  A second class
runs the real frozen inputs when they are present and pins the frozen aggregate
values.  No model weights, no CUDA, no network.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from evaluation.analyze_b1_answer_margin_oracle_stop_postmortem import (
    CSV_FIELDS,
    FROZEN_INPUT_SHA256,
    MANIFEST,
    ORACLE_JSONL,
    SUMMARY_JSON,
    CONTROL_JSONL,
    PostmortemInputError,
    analyze,
    build_report,
    csv_rows,
    render_markdown,
    signed_transition,
    spearman_rho,
)
from evaluation.b1_answer_margin_oracle import (
    ARM_ORACLE,
    array_sha256,
    record_sha256,
    retained_mass,
    selection_overlap,
)
from evaluation.b1_answer_margin_oracle_challenge_control import (
    CELLS,
    CHALLENGE,
    CONTROL,
    STRATA,
)

A_SHA = "a" * 64
B_SHA = "b" * 64
FULL_GROUPS = 256
CONTROL_ARM = "corrected_l2_integer"


def _patch_indices(selected: list[int]) -> list[int]:
    return [
        group * 4 + offset
        for group in selected
        for offset in range(4)
    ]


def _region_ids(selected: list[int]) -> list[int]:
    return [index % 9 for index in range(len(selected))]


def _selection_fields(selected: list[int], source_type: list[int]) -> dict:
    return {
        "selected_group_indices": list(selected),
        "selected_patch_indices": _patch_indices(selected),
        "source_type": list(source_type),
        "region_id": _region_ids(selected),
        "valid": [True] * len(selected),
    }


def _refresh_hashes(record: dict) -> dict:
    record["selected_group_indices_sha256"] = array_sha256(
        np.asarray(record["selected_group_indices"], dtype=np.int64)
    )
    record["selected_patch_indices_sha256"] = array_sha256(
        np.asarray(record["selected_patch_indices"], dtype=np.int64)
    )
    record["source_type_sha256"] = array_sha256(
        np.asarray(record["source_type"], dtype=np.int64)
    )
    record["region_id_sha256"] = array_sha256(
        np.asarray(record["region_id"], dtype=np.int64)
    )
    record["valid_sha256"] = array_sha256(
        np.asarray(record["valid"], dtype=bool)
    )
    if "saliency_group_values" in record:
        values = np.asarray(record["saliency_group_values"], dtype=np.float64)
        record["saliency_sha256"] = array_sha256(values)
    record["record_sha256"] = record_sha256(record)
    return record


def _base_record(sample_id: str, selected: list[int], source_type: list[int]) -> dict:
    return {
        "sample_id": sample_id,
        "cutpoint": "B1_after_vision_block_0",
        "prune_after_layer": 0,
        "requested_budget": 96,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "output_group_count": 96,
        "real_group_count": 87,
        "context_group_count": 9,
        "identity_bypass": False,
        "do_sample": False,
        "max_new_tokens": 16,
        "full_patch_tokens": FULL_GROUPS * 4,
        "image_grid_thw": [1, 16, FULL_GROUPS * 4 // 16],
        "full_llm_visual_tokens": FULL_GROUPS,
        "output_text": "(A)",
        **_selection_fields(selected, source_type),
    }


def oracle_record(sample_id: str, *, selected=None, source_type=None, **overrides) -> dict:
    if selected is None:
        selected = list(range(96))
    if source_type is None:
        source_type = [0] * 87 + [1] * 9
    values = np.linspace(0.5, 2.5, FULL_GROUPS, dtype=np.float64)
    record = {
        **_base_record(sample_id, selected, source_type),
        "arm": ARM_ORACLE,
        "score_mode": "oracle_b1_answer_margin_saliency",
        "manifest_sha256": A_SHA,
        "config_sha256": B_SHA,
        "margin": {"value": 1.5},
        "saliency_group_values": [float(value) for value in values],
        "saliency_sha256": array_sha256(values),
        "model_parameter_grads_all_none": True,
        "layout": {
            "passed": True,
            "output_group_count": 96,
            "real_group_count": 87,
            "context_group_count": 9,
            "identity_bypass": False,
        },
    }
    record.update(overrides)
    return _refresh_hashes(record)


def control_record(sample_id: str, *, selected=None, source_type=None, **overrides) -> dict:
    if selected is None:
        selected = list(range(10, 106))
    if source_type is None:
        source_type = [0] * 87 + [1] * 9
    record = {
        **_base_record(sample_id, selected, source_type),
        "arm": CONTROL_ARM,
        "score_mode": "integer",
    }
    record.update(overrides)
    return _refresh_hashes(record)


def _sample_id(cell, stratum: str, rank: int) -> str:
    return f"{cell.label()}|{stratum}|{rank}"


def _subdataset(cell, stratum: str) -> str:
    if cell.source != "Omni3D":
        return (
            "ADE20K Validation Set"
            if cell.source == "ADE20K"
            else "COCO 2017 Validation Set"
        )
    return "Omni3D_nuScenes"


def build_world() -> tuple[list[dict], list[dict], list[dict], dict]:
    """Synthetic world reproducing the frozen 10/3/9/2 transition split."""

    manifest: list[dict] = []
    oracle: list[dict] = []
    control: list[dict] = []
    summary_rows: list[dict] = []
    challenger_ordinal = 0
    control_ordinal = 0
    for cell in CELLS:
        for stratum in STRATA:
            for rank in range(2):
                sample_id = _sample_id(cell, stratum, rank)
                if stratum == CHALLENGE:
                    control_correct = False
                    oracle_correct = challenger_ordinal < 10
                    challenger_ordinal += 1
                else:
                    control_correct = True
                    oracle_correct = control_ordinal >= 3
                    control_ordinal += 1
                if not control_correct and oracle_correct:
                    transition = "W->R"
                elif control_correct and not oracle_correct:
                    transition = "R->W"
                elif control_correct:
                    transition = "unchanged_correct"
                else:
                    transition = "unchanged_wrong"
                manifest.append(
                    {
                        "sample_id": sample_id,
                        "task": cell.task,
                        "source": cell.source,
                        "source_dataset": _subdataset(cell, stratum),
                        "choices": ["0", "1", "2", "3"],
                        "answer": "(A)",
                        "oracle_pilot": {
                            "cell": cell.label(),
                            "stratum": stratum,
                        },
                    }
                )
                oracle.append(
                    oracle_record(
                        sample_id,
                        output_text="(A)" if oracle_correct else "(B)",
                    )
                )
                control.append(
                    control_record(
                        sample_id,
                        output_text="(A)" if control_correct else "(B)",
                    )
                )
                summary_rows.append(
                    {
                        "sample_id": sample_id,
                        "cell": cell.label(),
                        "stratum": stratum,
                        "ground_truth": "A",
                        "control_correct": control_correct,
                        "oracle_correct": oracle_correct,
                        "transition": transition,
                        "oracle_margin": 1.5,
                    }
                )
    summary = {
        "sample_count": 24,
        "per_sample": summary_rows,
        "transitions": {
            "W->R": 10,
            "R->W": 3,
            "unchanged_correct": 9,
            "unchanged_wrong": 2,
        },
        "decision": {
            "status": "PILOT_STOP",
            "challenge_repairs": 10,
            "control_regressions": 3,
            "oracle_parse_count": 24,
            "layout_contract_pass_count": 24,
        },
    }
    return manifest, oracle, control, summary


def _analyze(world) -> dict:
    manifest, oracle, control, summary = world
    return analyze(
        manifest_rows=manifest,
        oracle_rows=oracle,
        control_rows=control,
        summary=summary,
        manifest_sha256=A_SHA,
        config_sha256=B_SHA,
    )


class MetricFormulaTest(unittest.TestCase):
    def test_retained_mass_formula(self) -> None:
        self.assertAlmostEqual(retained_mass([1.0, 2.0, 3.0, 4.0], [0, 2]), 0.4)
        self.assertAlmostEqual(retained_mass([1.0, 2.0, 3.0, 4.0], []), 0.0)

    def test_selection_overlap_formula(self) -> None:
        overlap = selection_overlap([1, 2, 3], [2, 3, 4])
        self.assertEqual(overlap["intersection"], 2)
        self.assertEqual(overlap["union"], 4)
        self.assertAlmostEqual(overlap["jaccard"], 0.5)

    def test_real_context_split_formula(self) -> None:
        result = _analyze(build_world())
        row = result["per_sample"][0]
        # oracle real 0..86, control real 10..96 -> 77 shared of 97
        self.assertEqual(row["oracle_real_count"], 87)
        self.assertEqual(row["l2_real_count"], 87)
        self.assertEqual(row["real_slot_intersection"], 77)
        self.assertEqual(row["real_slot_union"], 97)
        self.assertAlmostEqual(row["real_slot_jaccard"], 77 / 97)
        # oracle all 0..95, control all 10..105 -> 86 shared of 106
        self.assertEqual(row["all_slot_intersection"], 86)
        self.assertAlmostEqual(row["all_slot_jaccard"], 86 / 106)
        # context representatives: oracle 87..95, control 97..105 -> disjoint
        self.assertEqual(row["context_slot_intersection"], 0)
        self.assertAlmostEqual(row["context_slot_jaccard"], 0.0)

    def test_spearman_rank_correlation(self) -> None:
        self.assertAlmostEqual(
            spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]), 1.0
        )
        self.assertAlmostEqual(
            spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]), -1.0
        )
        self.assertEqual(spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]), 0.0)
        # tie-averaged ranks: x ties on the first two -> rho = 0.948683...
        rho = spearman_rho([1, 1, 2, 3], [1, 2, 3, 4])
        self.assertAlmostEqual(rho, 0.9486832980505138, places=12)

    def test_signed_transition_definition(self) -> None:
        self.assertEqual(signed_transition(False, True), 1)
        self.assertEqual(signed_transition(True, False), -1)
        self.assertEqual(signed_transition(True, True), 0)
        self.assertEqual(signed_transition(False, False), 0)


class FrozenSplitTest(unittest.TestCase):
    def test_synthetic_world_reproduces_frozen_split(self) -> None:
        result = _analyze(build_world())
        self.assertEqual(
            result["verification"]["frozen_transitions"],
            {"W->R": 10, "R->W": 3, "unchanged_correct": 9, "unchanged_wrong": 2},
        )
        self.assertEqual(len(result["regressions"]), 3)
        self.assertEqual(len(result["unrepaired_challenges"]), 2)
        self.assertTrue(
            all(row["stratum"] == CHALLENGE for row in result["unrepaired_challenges"])
        )


class ValidationFailClosedTest(unittest.TestCase):
    def test_duplicate_oracle_sample_fails_closed(self) -> None:
        world = build_world()
        world[1].append(dict(world[1][0]))
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_missing_sample_fails_closed(self) -> None:
        world = build_world()
        world = (world[0], world[1][:-1], world[2], world[3])
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_tampered_oracle_record_hash_fails_closed(self) -> None:
        world = build_world()
        world[1][0]["output_text"] = "(B)"
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_tampered_control_record_hash_fails_closed(self) -> None:
        world = build_world()
        world[2][0]["region_id"][0] = 3
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_wrong_manifest_sha_fails_closed(self) -> None:
        world = build_world()
        world[1][0]["manifest_sha256"] = "c" * 64
        _refresh_hashes(world[1][0])
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_layout_context_count_fails_closed(self) -> None:
        world = build_world()
        world[1][0]["layout"]["context_group_count"] = 8
        _refresh_hashes(world[1][0])
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_non_ascending_indices_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["selected_group_indices"][0], record["selected_group_indices"][1] = (
            record["selected_group_indices"][1],
            record["selected_group_indices"][0],
        )
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_patch_indices_not_original_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["selected_patch_indices"][0] += 1
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_source_type_length_mismatch_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["source_type"] = record["source_type"][:-1]
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_unknown_source_type_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["source_type"][0] = 7
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_real_count_mismatch_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["source_type"] = [0] * 96
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_saliency_length_mismatch_fails_closed(self) -> None:
        world = build_world()
        record = world[1][0]
        record["saliency_group_values"] = list(
            np.linspace(0.5, 2.5, FULL_GROUPS - 1, dtype=np.float64)
        )
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_g_mismatch_between_arms_fails_closed(self) -> None:
        world = build_world()
        record = world[2][0]
        record["full_llm_visual_tokens"] = 260
        record["full_patch_tokens"] = 1040
        record["image_grid_thw"] = [1, 16, 65]
        _refresh_hashes(record)
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_transition_label_mismatch_fails_closed(self) -> None:
        world = build_world()
        world[3]["per_sample"][0]["transition"] = "R->W"
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_summary_status_not_stop_fails_closed(self) -> None:
        world = build_world()
        world[3]["decision"]["status"] = "PILOT_GO"
        with self.assertRaises(PostmortemInputError):
            _analyze(world)

    def test_frozen_decision_count_mismatch_fails_closed(self) -> None:
        world = build_world()
        world[3]["decision"]["challenge_repairs"] = 9
        with self.assertRaises(PostmortemInputError):
            _analyze(world)


class RenderingTest(unittest.TestCase):
    def test_csv_rows_and_markdown(self) -> None:
        result = _analyze(build_world())
        report = dict(result)
        report["inputs_sha256"] = {
            label: {"path": "x", "sha256": "0" * 64}
            for label in ("config", "manifest", "oracle", "summary", "control")
        }
        report["forbidden_claims"] = ["paper accuracy"]
        rows = csv_rows(report)
        self.assertEqual(len(rows), 24)
        self.assertEqual(set(rows[0]), set(CSV_FIELDS))
        markdown = render_markdown(report)
        self.assertIn("PILOT_STOP", markdown)
        self.assertIn("决策边界", markdown)


class FrozenAggregateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = [MANIFEST, ORACLE_JSONL, SUMMARY_JSON, CONTROL_JSONL]
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("frozen pilot inputs are not present")
        cls.report = build_report()

    def test_frozen_input_hashes(self) -> None:
        for label, expected in FROZEN_INPUT_SHA256.items():
            self.assertEqual(self.report["inputs_sha256"][label]["sha256"], expected)
        self.assertTrue(self.report["pins"]["pins_verified"])

    def test_frozen_decision(self) -> None:
        self.assertEqual(self.report["status"], "PILOT_STOP_POSTMORTEM_COMPLETE")
        self.assertEqual(self.report["frozen_status"], "PILOT_STOP")
        verification = self.report["verification"]
        self.assertEqual(
            verification["frozen_transitions"],
            {"W->R": 10, "R->W": 3, "unchanged_correct": 9, "unchanged_wrong": 2},
        )
        self.assertEqual(
            verification["frozen_decision"]["challenge_repairs"], 10
        )
        self.assertEqual(
            verification["frozen_decision"]["control_regressions"], 3
        )

    def test_frozen_regression_and_unrepaired_ids(self) -> None:
        self.assertEqual(
            [row["sample_id"] for row in self.report["regressions"]],
            ["334", "471", "2518"],
        )
        self.assertEqual(
            [row["sample_id"] for row in self.report["unrepaired_challenges"]],
            ["425", "969"],
        )

    def test_frozen_metric_aggregates(self) -> None:
        rows = self.report["per_sample"]
        self.assertAlmostEqual(
            float(np.mean([row["all_slot_jaccard"] for row in rows])),
            0.23859199418794108,
            places=9,
        )
        self.assertAlmostEqual(
            float(np.mean([row["real_slot_jaccard"] for row in rows])),
            0.20612423206922378,
            places=9,
        )
        self.assertAlmostEqual(
            float(np.mean([row["retained_mass_advantage"] for row in rows])),
            0.3135936102912272,
            places=9,
        )
        self.assertTrue(
            all(row["retained_mass_advantage"] > 0 for row in rows)
        )
        self.assertTrue(
            all(row["real_slot_intersection"] >= 0 for row in rows)
        )
        groups = self.report["transition_groups"]
        self.assertAlmostEqual(
            groups["W->R"]["oracle_margin"]["mean"], 2.8375, places=9
        )
        self.assertAlmostEqual(
            groups["unchanged_correct"]["oracle_margin"]["mean"],
            4.513888888888889,
            places=9,
        )
        self.assertEqual(groups["unchanged_wrong"]["n"], 2)

    def test_frozen_spearman(self) -> None:
        self.assertEqual(
            self.report["associations"]["exploratory"],
            True,
        )
        self.assertEqual(
            self.report["associations"]["uses_p_value_or_significance"],
            False,
        )
        spearman = self.report["associations"]["spearman"]
        self.assertAlmostEqual(
            spearman["oracle_margin"], -0.10027613959936446, places=10
        )
        self.assertAlmostEqual(
            spearman["retained_mass_advantage"], 0.08347257041177221, places=10
        )

    def test_frozen_decision_boundary(self) -> None:
        boundary = self.report["decision_boundary"]
        self.assertEqual(
            boundary["direct_absolute_gradient_full_replacement_teacher"],
            "STOPPED",
        )
        self.assertEqual(boundary["preregistered_stop_overturned"], False)
        self.assertEqual(
            boundary["thresholds_K_anchors_or_mixtures_tuned_on_24_samples"],
            False,
        )
        self.assertEqual(boundary["numeric_mixture_specified"], False)
        self.assertEqual(
            boundary["recommended_next_mechanism_class"],
            "conservative_l2_backbone_residual_task_aware_reranking",
        )
        self.assertEqual(
            boundary["recommendation_status"],
            "unproven_hypothesis_for_next_falsifiable_test",
        )
        self.assertFalse(self.report["is_official_cv_bench_accuracy"])
        self.assertEqual(self.report["gpu_runs"], 0)


if __name__ == "__main__":
    unittest.main()
