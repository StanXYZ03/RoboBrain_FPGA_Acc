#!/usr/bin/env python3
"""Focused tests for the G>512 gate / source-dataset confounding diagnostic."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import analyze_gate_source_confounding as analysis

ARM_INTEGER = analysis.ARM_INTEGER
ARM_PROMPT = analysis.ARM_PROMPT

#: Real report computed once for the writer / integration assertions.
_REAL_REPORT: dict | None = None


def setUpModule() -> None:  # noqa: N802 (unittest hook name)
    global _REAL_REPORT
    _REAL_REPORT = analysis.analyze()


def real_report() -> dict:
    if _REAL_REPORT is None:  # pragma: no cover - defensive
        setUpModule()
    assert _REAL_REPORT is not None
    return _REAL_REPORT


def transition(int_ok: bool, prompt_ok: bool) -> str:
    if int_ok and prompt_ok:
        return "unchanged_right"
    if int_ok and not prompt_ok:
        return "right_to_wrong"
    if not int_ok and prompt_ok:
        return "wrong_to_right"
    return "unchanged_wrong"


def make_row(
    sid: str,
    task: str,
    source: str,
    source_dataset: str,
    g: int,
    int_ok: bool,
    prompt_ok: bool,
) -> dict:
    return {
        "sample_id": str(sid),
        "task": task,
        "source": source,
        "source_dataset": source_dataset,
        "G": int(g),
        "G_bin": analysis.g_bin(int(g)),
        "gate_label": analysis.gate_label(int(g)),
        "transition": transition(int_ok, prompt_ok),
        f"{ARM_INTEGER}_correct": bool(int_ok),
        f"{ARM_PROMPT}_correct": bool(prompt_ok),
        f"{ARM_INTEGER}_answer": "(A)",
        f"{ARM_PROMPT}_answer": "(A)",
        "identity_bypass": int(g) <= 96,
    }


def synthetic_joined() -> list[dict]:
    rows: list[dict] = []
    sid = 0
    # ADE20K low: 10 rows, 5 W2R and 5 R2W -> delta 0.
    for index in range(10):
        rows.append(
            make_row(sid, "Count", "ADE20K", "ADE20K Validation Set", 100, index < 5, index >= 5)
        )
        sid += 1
    # ADE20K high: 10 rows, all W2R -> delta +1.
    for _ in range(10):
        rows.append(
            make_row(sid, "Count", "ADE20K", "ADE20K Validation Set", 600, False, True)
        )
        sid += 1
    # COCO: 10 rows, 2 W2R 1 R2W.
    specs = [(False, True), (False, True), (True, False)] + [(True, True)] * 7
    for int_ok, prompt_ok in specs:
        rows.append(
            make_row(sid, "Depth", "COCO", "COCO 2017 Validation Set", 300, int_ok, prompt_ok)
        )
        sid += 1
    # Omni3D three sub-datasets, one per gate region.
    for int_ok, prompt_ok in [(False, True), (True, False)] * 5:
        rows.append(make_row(sid, "Distance", "Omni3D", "Omni3D_Hypersim", 700, int_ok, prompt_ok))
        sid += 1
    for _ in range(10):
        rows.append(make_row(sid, "Distance", "Omni3D", "Omni3D_SUNRGBD", 200, True, False))
        sid += 1
    for _ in range(10):
        rows.append(make_row(sid, "Relation", "Omni3D", "Omni3D_nuScenes", 1200, False, True))
        sid += 1
    return rows


class TestHelpers(unittest.TestCase):
    def test_gate_label_boundary(self) -> None:
        self.assertEqual(analysis.gate_label(512), "G<=512")
        self.assertEqual(analysis.gate_label(513), "G>512")
        self.assertFalse(analysis.gate_selected(512))
        self.assertTrue(analysis.gate_selected(513))

    def test_g_bin_boundaries(self) -> None:
        self.assertEqual(analysis.g_bin(96), "<=96")
        self.assertEqual(analysis.g_bin(97), "97-256")
        self.assertEqual(analysis.g_bin(256), "97-256")
        self.assertEqual(analysis.g_bin(257), "257-512")
        self.assertEqual(analysis.g_bin(512), "257-512")
        self.assertEqual(analysis.g_bin(513), "513-1024")
        self.assertEqual(analysis.g_bin(1024), "513-1024")
        self.assertEqual(analysis.g_bin(1025), ">1024")

    def test_sample_sort_key_numeric(self) -> None:
        ids = ["10", "2", "1"]
        self.assertEqual(sorted(ids, key=analysis._sample_sort_key), ["1", "2", "10"])


class TestAssociation(unittest.TestCase):
    def test_cramers_v_perfect(self) -> None:
        self.assertAlmostEqual(analysis.cramers_v([[10, 0], [0, 10]]), 1.0)

    def test_cramers_v_independent(self) -> None:
        self.assertAlmostEqual(analysis.cramers_v([[5, 5], [5, 5]]), 0.0)

    def test_cramers_v_multi_column(self) -> None:
        value = analysis.cramers_v([[10, 0, 0], [0, 10, 10]])
        self.assertGreater(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_cramers_v_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            analysis.cramers_v([])

    def test_mutual_information_perfect_is_one_bit(self) -> None:
        self.assertAlmostEqual(
            analysis.mutual_information_bits([[10, 0], [0, 10]]), 1.0
        )

    def test_mutual_information_independent_zero(self) -> None:
        self.assertAlmostEqual(
            analysis.mutual_information_bits([[5, 5], [5, 5]]), 0.0
        )


class TestBootstrapInteraction(unittest.TestCase):
    def test_deterministic_interaction(self) -> None:
        rows = [
            make_row(
                str(index),
                "Count",
                "ADE20K",
                "ADE20K Validation Set",
                100,
                True,
                True,
            )
            for index in range(10)
        ] + [
            make_row(
                str(100 + index),
                "Count",
                "ADE20K",
                "ADE20K Validation Set",
                600,
                False,
                True,
            )
            for index in range(10)
        ]
        ci = analysis.bootstrap_interaction_delta(
            rows,
            bin_key=lambda row: analysis.gate_label(int(row["G"])),
            seed=20261004,
            repeats=200,
        )
        # high gain = +1.0, low gain = 0.0 -> deterministic +1.0
        self.assertAlmostEqual(ci[0], 1.0)
        self.assertAlmostEqual(ci[1], 1.0)

    def test_missing_bin_raises(self) -> None:
        rows = [make_row("1", "Count", "ADE20K", "ADE20K Validation Set", 600, False, True)]
        with self.assertRaises(analysis.InputContractError):
            analysis.bootstrap_interaction_delta(
                rows,
                bin_key=lambda row: analysis.gate_label(int(row["G"])),
                repeats=10,
            )


class TestTables(unittest.TestCase):
    def test_source_g_bin_table_counts(self) -> None:
        table = analysis.source_g_bin_table(synthetic_joined())
        by_name = {row["source_dataset"]: row for row in table}
        self.assertEqual(by_name["ADE20K Validation Set"]["n"], 20)
        self.assertEqual(by_name["ADE20K Validation Set"]["counts"]["97-256"], 10)
        self.assertEqual(by_name["ADE20K Validation Set"]["counts"]["513-1024"], 10)
        self.assertEqual(by_name["ADE20K Validation Set"]["gate_G_gt_512_count"], 10)
        self.assertEqual(by_name["Omni3D_nuScenes"]["gate_G_gt_512_count"], 10)
        self.assertTrue(by_name["Omni3D_nuScenes"]["fully_inside_gate"])
        self.assertTrue(by_name["COCO 2017 Validation Set"]["fully_outside_gate"])

    def test_gate_composition(self) -> None:
        composition = analysis.gate_composition(synthetic_joined())
        self.assertEqual(composition["gate_on_count"], 30)
        self.assertEqual(composition["gate_off_count"], 30)
        self.assertAlmostEqual(composition["gate_on_fraction"], 0.5)
        self.assertEqual(
            composition["gate_on_by_source_dataset"]["Omni3D_nuScenes"], 10
        )


class TestEffects(unittest.TestCase):
    def test_source_dataset_effects_transitions(self) -> None:
        effects = analysis.source_dataset_effects(synthetic_joined())
        by_name = {row["stratum"]: row for row in effects}
        ade = by_name["ADE20K Validation Set"]
        self.assertEqual(ade["wrong_to_right"], 15)
        self.assertEqual(ade["right_to_wrong"], 5)
        self.assertEqual(ade["net_corrections"], 10)
        self.assertAlmostEqual(ade["unweighted_paired_delta"], 0.5)
        self.assertEqual(len(ade["bootstrap_ci95_unweighted_paired_delta"]), 2)

    def test_ade_interaction(self) -> None:
        interaction = analysis.ade_interaction(synthetic_joined())
        self.assertTrue(interaction["defined"])
        self.assertEqual(interaction["n_G_le_512"], 10)
        self.assertEqual(interaction["n_G_gt_512"], 10)
        self.assertAlmostEqual(interaction["effect_G_le_512"], 0.0)
        self.assertAlmostEqual(interaction["effect_G_gt_512"], 1.0)
        self.assertAlmostEqual(
            interaction["interaction_G_gt_512_minus_G_le_512"], 1.0
        )
        low, high = interaction["bootstrap_ci95_interaction"]
        self.assertLessEqual(low, 1.0)
        self.assertGreaterEqual(high, 1.0)

    def test_conclusion_resolution_rule(self) -> None:
        self.assertEqual(
            analysis.conclusion_for(
                {"defined": True, "bootstrap_ci95_interaction": [0.01, 0.2]}
            ),
            analysis.CONCLUSION_RESOLVED,
        )
        self.assertEqual(
            analysis.conclusion_for(
                {"defined": True, "bootstrap_ci95_interaction": [-0.01, 0.2]}
            ),
            analysis.CONCLUSION_NOT_RESOLVED,
        )
        self.assertEqual(
            analysis.conclusion_for(
                {"defined": True, "bootstrap_ci95_interaction": [0.0, 0.2]}
            ),
            analysis.CONCLUSION_NOT_RESOLVED,
        )
        self.assertEqual(
            analysis.conclusion_for({"defined": False}),
            analysis.CONCLUSION_UNDEFINED,
        )

    def test_gate_policy_effect(self) -> None:
        effect = analysis.gate_policy_effect(synthetic_joined())
        self.assertIsNotNone(effect["gated_official_cv_bench_accuracy"])
        self.assertIsNotNone(effect["gated_minus_integer_official"])
        self.assertGreater(effect["gated_unweighted_accuracy"], 0.0)


class TestRealIntegration(unittest.TestCase):
    def test_conclusion_and_formal_result(self) -> None:
        report = real_report()
        self.assertEqual(report["conclusion"], analysis.CONCLUSION_NOT_RESOLVED)
        self.assertTrue(report["formal_result_unchanged"])
        self.assertEqual(report["formal_outcome"], analysis.FORMAL_OUTCOME)
        self.assertEqual(report["analysis"], analysis.ANALYSIS)

    def test_ade_interaction_ci_not_all_positive(self) -> None:
        ade = real_report()["ade_interaction"]
        self.assertTrue(ade["defined"])
        self.assertLessEqual(ade["bootstrap_ci95_interaction"][0], 0.0)
        self.assertEqual(ade["n_G_le_512"], 425)
        self.assertEqual(ade["n_G_gt_512"], 208)

    def test_contract_checks_all_ok(self) -> None:
        report = real_report()
        self.assertTrue(all(check["ok"] for check in report["contract"]["gate_contract_checks"]))
        self.assertEqual(report["contract"]["expected_sample_count"], 2638)
        self.assertEqual(len(report["per_sample"]), 2638)

    def test_dependency_is_exploratory(self) -> None:
        dependency = real_report()["dependency"]
        self.assertTrue(dependency["exploratory_only"])
        self.assertTrue(dependency["not_a_gate"])
        self.assertGreater(dependency["cramers_v_gate_vs_source_dataset"], 0.5)
        self.assertGreater(
            dependency["mutual_information_bits_gate_vs_source_dataset"], 0.0
        )

    def test_frozen_delta_reproduced(self) -> None:
        frozen = real_report()["frozen_reproduction"]
        self.assertTrue(frozen["reproduced"])
        self.assertAlmostEqual(
            frozen["frozen_delta_point_estimate"],
            frozen["recomputed_delta_point_estimate"],
        )


class TestOutputs(unittest.TestCase):
    def test_write_outputs(self) -> None:
        report = real_report()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
            self.assertEqual(loaded["conclusion"], analysis.CONCLUSION_NOT_RESOLVED)
            csv_lines = outputs.csv.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(csv_lines), 2639)
            self.assertEqual(csv_lines[0].split(",")[0], "sample_id")
            markdown = outputs.md.read_text(encoding="utf-8")
            self.assertIn(analysis.CONCLUSION_NOT_RESOLVED, markdown)
            self.assertIn("探索", markdown)
            self.assertIn("不作为门", markdown)

    def test_render_markdown_mentions_exploratory(self) -> None:
        markdown = analysis.render_markdown(real_report())
        self.assertIn(analysis.EXPLORATORY_NOTE_ZH, markdown)
        self.assertIn("ADE20K", markdown)

    def test_csv_columns_match_per_sample(self) -> None:
        report = real_report()
        for row in report["per_sample"]:
            self.assertEqual(set(row), set(analysis.CSV_COLUMNS))


class TestFailClosed(unittest.TestCase):
    def test_pin_mismatch_raises(self) -> None:
        from evaluation import analyze_full2638_prompt_gain_heterogeneity as het

        with self.assertRaises(het.InputContractError):
            analysis.analyze(expected_pins={"full_manifest": "0" * 64})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
