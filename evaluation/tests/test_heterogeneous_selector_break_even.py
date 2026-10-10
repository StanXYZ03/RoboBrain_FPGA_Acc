"""Unit tests for the baseline-corrected heterogeneous selector requirement model.

CPU only: no GPU, no board, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "analyze_heterogeneous_selector_break_even",
    ROOT / "scripts" / "analyze_heterogeneous_selector_break_even.py",
)
assert _SPEC is not None and _SPEC.loader is not None
model = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = model
_SPEC.loader.exec_module(model)

from evaluation import hardware_regular_selector_fixedpoint as fp  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "heterogeneous_selector_break_even.json"


class ByteContractTest(unittest.TestCase):
    def test_dense_input_is_groups_times_4096_plus_scores(self) -> None:
        self.assertEqual(model.dense_input_bytes(256), 256 * 4096 + 256 * 2)
        self.assertEqual(model.dense_input_bytes(576), 576 * 4096 + 576 * 2)
        self.assertEqual(model.dense_input_bytes(1024), 1024 * 4096 + 1024 * 2)

    def test_selected_output_matches_real_layout(self) -> None:
        for budget in (48, 64, 96):
            self.assertEqual(
                model.selected_output_bytes(budget),
                fp.fpga_transfer_layout(budget).total_bytes,
            )
            self.assertEqual(
                model.selected_output_bytes(budget) % fp.FPGA_LAYOUT_ALIGNMENT, 0
            )
            self.assertGreaterEqual(
                model.selected_output_bytes(budget), budget * fp.GROUP_ELEMENTS
            )

    def test_contract_rows_cover_requested_grids_and_budgets(self) -> None:
        rows = model.byte_contract_rows([256, 576, 1024], [48, 64, 96])
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertEqual(
                row["round_trip_bytes"],
                row["dense_input_bytes"] + row["selected_output_bytes"],
            )
            self.assertEqual(
                set(row["segment_sizes"]),
                {
                    "tokens_int8",
                    "group_index_uint16",
                    "source_type_uint8",
                    "region_id_uint8",
                    "valid_uint8",
                },
            )
            self.assertEqual(
                row["segment_sizes"]["tokens_int8"], row["budget"] * 4096
            )


class TimingHelperTest(unittest.TestCase):
    def test_link_ms_uses_decimal_megabytes(self) -> None:
        self.assertAlmostEqual(model.link_ms(1_000_000, 1000.0, 0.0), 1.0)
        self.assertAlmostEqual(model.link_ms(1_000_000, 2000.0, 0.5), 1.0)

    def test_cycles_to_ms_at_200mhz(self) -> None:
        self.assertAlmostEqual(model.cycles_to_ms(1_708_124, 200_000_000), 8.54062)
        self.assertAlmostEqual(model.cycles_to_ms(5_668, 200_000_000), 0.02834)


class PlacementBaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())

    def test_four_placements_and_two_baselines(self) -> None:
        self.assertEqual(
            set(self.config["placements"]),
            {"J_DENSE", "J_PRUNED", "J_F_J", "J_F_CONTINUE"},
        )
        self.assertEqual(
            set(self.config["baseline_confusion_fix"]),
            {"superseded", "fix", "consequence"},
        )
        self.assertEqual(tuple(model.BASELINES), ("J_DENSE", "J_PRUNED"))
        self.assertEqual(set(model.PLACEMENTS), {"J_DENSE", "J_PRUNED", "J_F_J", "J_F_CONTINUE"})

    def test_j_dense_has_no_score_no_selector(self) -> None:
        dense = self.config["placements"]["J_DENSE"]
        self.assertFalse(dense["uses_score"])
        self.assertFalse(dense["uses_software_selector"])
        self.assertEqual(dense["total_ms"], "T_vision_dense")

    def test_j_pruned_uses_score_and_software_selector(self) -> None:
        pruned = self.config["placements"]["J_PRUNED"]
        self.assertTrue(pruned["uses_score"])
        self.assertTrue(pruned["uses_software_selector"])
        self.assertFalse(pruned["fpga_selector"])

    def test_j_f_j_shares_k_token_vision_with_j_pruned(self) -> None:
        jfj = self.config["placements"]["J_F_J"]
        self.assertIn("T_vision_pruned", jfj["total_ms"])
        self.assertEqual(jfj["fixed_overheads"], 2)
        self.assertTrue(jfj["return_to_jetson"])

    def test_j_f_continue_requires_real_downstream_core(self) -> None:
        cont = self.config["placements"]["J_F_CONTINUE"]
        self.assertTrue(cont["requires_real_downstream_fpga_core"])
        self.assertFalse(cont["return_to_jetson"])
        self.assertEqual(cont["fixed_overheads"], 1)


class ComparisonFormulaTest(unittest.TestCase):
    def test_j_f_j_vs_j_pruned_contains_no_vision_saving(self) -> None:
        spec = model.requirement_spec("B_vs_J_PRUNED", "J_F_J", 9.196)
        self.assertNotIn("T_vision", spec["requirement"])
        self.assertNotIn("S_prune", spec["requirement"])
        self.assertEqual(spec["symbolic_missing_terms"], ["T_sel_sw"])
        self.assertTrue(spec["is_exact_threshold"])
        self.assertAlmostEqual(spec["numeric_threshold_ms"], 9.196)

    def test_j_f_j_vs_dense_contains_T_score_symbolic_term(self) -> None:
        spec = model.requirement_spec("A_vs_J_DENSE", "J_F_J", 9.196)
        self.assertIn("T_score", spec["requirement"])
        self.assertIn("T_score", spec["symbolic_missing_terms"])
        self.assertFalse(spec["is_exact_threshold"])
        self.assertIsNone(spec["numeric_threshold_ms"])

    def test_j_f_continue_vs_dense_contains_T_score_and_downstream(self) -> None:
        spec = model.requirement_spec("A_vs_J_DENSE", "J_F_CONTINUE", 9.0)
        self.assertIn("T_score", spec["requirement"])
        self.assertIn("T_fpga_downstream", spec["requirement"])
        self.assertIn("T_score", spec["symbolic_missing_terms"])

    def test_j_pruned_vs_dense_is_fully_symbolic(self) -> None:
        spec = model.requirement_spec("A_vs_J_DENSE", "J_PRUNED", 0.0)
        self.assertEqual(spec["requirement"], "S_prune > T_score + T_sel_sw")
        self.assertNotIn("T_link", spec["requirement"])
        self.assertIsNone(spec["numeric_threshold_ms"])

    def test_j_f_continue_vs_pruned_keeps_vision_and_downstream(self) -> None:
        spec = model.requirement_spec("B_vs_J_PRUNED", "J_F_CONTINUE", 9.0)
        self.assertIn("T_vision_pruned", spec["requirement"])
        self.assertIn("T_fpga_downstream", spec["requirement"])
        self.assertIn("T_sel_sw", spec["requirement"])

    def test_known_lower_bound_is_zero_for_non_fpga_placements(self) -> None:
        for placement in ("J_DENSE", "J_PRUNED"):
            known = model.known_lower_bound_ms(
                placement=placement,
                dense_bytes=4_000_000,
                selected_bytes=400_000,
                selector_cycles=1_708_124,
                clock_hz=200_000_000,
                throughput_mbps=2000.0,
                fixed_overhead_ms=0.5,
            )
            self.assertEqual(known["known_lower_bound_ms"], 0.0)

    def test_j_f_j_known_lower_bound_math(self) -> None:
        known = model.known_lower_bound_ms(
            placement="J_F_J",
            dense_bytes=4_000_000,
            selected_bytes=400_000,
            selector_cycles=440_899,
            clock_hz=200_000_000,
            throughput_mbps=2000.0,
            fixed_overhead_ms=0.5,
        )
        self.assertEqual(known["fixed_overheads"], 2.0)
        expected = (
            4_000_000 / 2_000_000
            + model.cycles_to_ms(440_899, 200_000_000)
            + 400_000 / 2_000_000
            + 1.0
        )
        self.assertAlmostEqual(known["known_lower_bound_ms"], expected)

    def test_j_f_continue_known_lower_bound_has_one_overhead_and_no_return(self) -> None:
        known = model.known_lower_bound_ms(
            placement="J_F_CONTINUE",
            dense_bytes=4_000_000,
            selected_bytes=400_000,
            selector_cycles=440_899,
            clock_hz=200_000_000,
            throughput_mbps=2000.0,
            fixed_overhead_ms=0.5,
        )
        self.assertEqual(known["fixed_overheads"], 1.0)
        self.assertEqual(known["t_link_return_ms"], 0.0)

    def test_placement_is_not_a_candidate_of_the_wrong_comparison(self) -> None:
        with self.assertRaises(ValueError):
            model.requirement_spec("B_vs_J_PRUNED", "J_PRUNED", 0.0)


class EvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.evidence = model.load_selector_evidence(cls.config)

    def test_cycle_counts_read_from_artifacts(self) -> None:
        self.assertEqual(self.evidence["csynth_worst_cycles"], 1_708_124)
        self.assertEqual(self.evidence["cosim_latency_cycles"]["min"], 5_668)
        self.assertEqual(self.evidence["cosim_latency_cycles"]["max"], 440_899)
        self.assertEqual(len(self.evidence["cosim_cases"]), 3)

    def test_three_cosim_cases_are_the_frozen_roster(self) -> None:
        roster = {(c["group_count"], c["budget"]) for c in self.evidence["cosim_cases"]}
        self.assertEqual(roster, {(16, 48), (256, 64), (1024, 96)})

    def test_scenarios_map_to_evidence(self) -> None:
        scenarios = model.selector_scenarios(self.evidence, self.config)
        self.assertEqual(
            [s["cycles"] for s in scenarios],
            [1_708_124, 440_899, 209_518, 5_668],
        )

    def test_score_generation_is_not_in_fpga_kernel_cycles(self) -> None:
        self.assertFalse(self.config["score_generation"]["inside_selector_ip"])
        self.assertFalse(self.config["score_generation"]["counted_in_fpga_selector_cycles"])

    def test_unmeasured_terms_are_declared(self) -> None:
        self.assertEqual(
            set(self.config["symbolic_missing_terms"]),
            {
                "T_score",
                "T_sel_sw",
                "T_vision_dense",
                "T_vision_pruned",
                "T_fpga_downstream",
            },
        )
        self.assertFalse(self.config["unmeasured_terms"]["T_score"]["measured"])
        self.assertFalse(self.config["unmeasured_terms"]["T_sel_sw"]["measured"])


class SweepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text())
        cls.evidence = model.load_selector_evidence(cls.config)
        cls.rows = model.build_sweep(cls.config, cls.evidence)
        cls.key = model.key_thresholds(cls.rows, cls.config)

    def test_row_count(self) -> None:
        # (A: 3 candidates + B: 2 candidates) * 4 scenarios * 3 grids * 3 budgets
        # * 6 links * 3 overheads
        self.assertEqual(len(self.rows), (3 + 2) * 4 * 3 * 3 * 6 * 3)

    def test_every_row_is_a_lower_bound_with_missing_terms(self) -> None:
        for row in self.rows:
            self.assertTrue(row["lower_bound_only"])
            self.assertGreater(row["dense_input_bytes"], 0)
            self.assertGreater(row["selected_output_bytes"], 0)
            self.assertTrue(row["symbolic_missing_terms"])
            self.assertIn(row["comparison"], model.COMPARISONS)
            self.assertEqual(row["baseline"], model.COMPARISON_BASELINE[row["comparison"]])

    def test_j_f_j_vs_j_pruned_rows_have_no_vision_saving(self) -> None:
        rows = [r for r in self.rows if r["comparison"] == "B_vs_J_PRUNED" and r["placement"] == "J_F_J"]
        self.assertTrue(rows)
        for row in rows:
            self.assertNotIn("T_vision", row["required_expression"])
            self.assertNotIn("S_prune", row["required_expression"])
            self.assertEqual(row["symbolic_missing_terms"], "T_sel_sw")
            self.assertEqual(row["threshold_quantity"], "T_sel_sw")
            self.assertTrue(row["is_exact_threshold"])

    def test_j_f_j_vs_dense_rows_keep_T_score(self) -> None:
        rows = [r for r in self.rows if r["comparison"] == "A_vs_J_DENSE" and r["placement"] == "J_F_J"]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("T_score", row["required_expression"])
            self.assertIn("T_score", row["symbolic_missing_terms"])
            self.assertFalse(row["is_exact_threshold"])
            self.assertIsNone(row["numeric_threshold_ms"])

    def test_only_b_vs_pruned_j_f_j_is_numerically_exact(self) -> None:
        for row in self.rows:
            exact = (
                row["comparison"] == "B_vs_J_PRUNED" and row["placement"] == "J_F_J"
            )
            self.assertEqual(row["is_exact_threshold"], exact)
            if exact:
                self.assertAlmostEqual(
                    row["numeric_threshold_ms"], row["known_lower_bound_ms"]
                )
            else:
                self.assertIsNone(row["numeric_threshold_ms"])

    def test_fastest_link_lowers_known_lower_bound(self) -> None:
        def pick(throughput: float) -> float:
            return next(
                r["known_lower_bound_ms"]
                for r in self.rows
                if r["comparison"] == "B_vs_J_PRUNED"
                and r["placement"] == "J_F_J"
                and r["group_count"] == 1024
                and r["budget"] == 96
                and r["selector_cycle_scenario"] == "csynth_worst_bound"
                and r["link_throughput_MBps"] == throughput
                and r["fixed_overhead_ms"] == 0.5
            )

        self.assertGreater(pick(100.0), pick(3000.0))

    def test_key_thresholds_cover_all_candidates(self) -> None:
        self.assertEqual(len(self.key["points"]), 2 * (3 + 2))
        expressions = {p["required_expression"] for p in self.key["points"]}
        self.assertTrue(any("T_sel_sw >" in e for e in expressions))
        self.assertTrue(any("T_score" in e for e in expressions))

    def test_key_thresholds_forbid_full_break_even_claim(self) -> None:
        self.assertIn("lower bound", self.key["forbidden_claim"].lower())


class EndToEndTest(unittest.TestCase):
    def test_main_writes_reports_and_harness_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness = tmp_path / "harness.json"
            code = model.main(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--report-dir",
                    str(tmp_path),
                    "--report-stem",
                    "break-even",
                    "--harness-result",
                    str(harness),
                    "--tests-status",
                    "SELFTEST_OK",
                ]
            )
            self.assertEqual(code, 0)
            for name in ("break-even.md", "break-even.csv", "break-even.json"):
                self.assertTrue((tmp_path / name).exists(), name)
            result = json.loads(harness.read_text())
            self.assertEqual(
                set(result),
                {
                    "status",
                    "baselines",
                    "contracts",
                    "evidence",
                    "placements",
                    "sweep",
                    "key_thresholds",
                    "tests",
                    "warnings",
                    "files_changed",
                    "next_gate",
                },
            )
            self.assertEqual(result["tests"], "SELFTEST_OK")
            self.assertEqual(set(result["baselines"]), {"J_DENSE", "J_PRUNED", "superseded", "fix"})
            self.assertEqual(
                set(result["placements"]),
                {"J_DENSE", "J_PRUNED", "J_F_J", "J_F_CONTINUE"},
            )
            self.assertNotIn("key_break_even", result)

    def test_markdown_does_not_claim_full_break_even(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            model.main(
                [
                    "--config",
                    str(CONFIG_PATH),
                    "--report-dir",
                    str(tmp_path),
                    "--report-stem",
                    "break-even",
                    "--skip-harness-result",
                ]
            )
            markdown = (tmp_path / "break-even.md").read_text()
            self.assertIn("`J_F_J` vs `J_PRUNED`", markdown)
            self.assertIn("不含任何 vision saving", markdown)
            self.assertIn("+T_score", markdown)
            self.assertIn("known lower bound", markdown)


if __name__ == "__main__":
    unittest.main()
