#!/usr/bin/env python3
"""CPU-only focused tests for the v2 full2638 development-result audit.

The tests pin the eight audit items, the independently recomputed scores and
stratified paired bootstrap, the 90%-of-dense floor and shortfall, the frozen
DEVELOPMENT_STOP decision, the label-aware postmortem counts and the
fail-closed behaviour on tampered records/pins/leakage flags.

No CUDA, no model weights, no network and no dependency installation are
touched; the real frozen artifacts are only read.
"""

from __future__ import annotations

import copy
import functools
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    audit_conservative_l2_guarded_rerank_v2_full2638 as audit,
)

ITEM_2 = audit.ITEM_2
ITEM_3 = audit.ITEM_3
ITEM_4 = audit.ITEM_4
ITEM_5 = audit.ITEM_5


@functools.lru_cache(maxsize=1)
def _audit_bundle() -> dict:
    """Run the real full2638 audit exactly once for the whole test module."""

    return audit.run_audit(audit.DEFAULT_PATHS)


def _loaded_bundle() -> dict:
    """Load the real frozen inputs once for the whole test class."""

    config = json.loads(audit.PREREGISTERED_CONFIG.read_text(encoding="utf-8"))
    manifest_rows = audit.load_jsonl(audit.FULL_MANIFEST)
    references = audit._load_references(audit.DEFAULT_PATHS, config)
    raw_rows = audit.load_jsonl(audit.RAW_FULL2638)
    return {
        "raw_sha256": audit.sha256_file(audit.RAW_FULL2638),
        "raw_rows": raw_rows,
        "config": config,
        "config_sha256": audit.sha256_file(audit.PREREGISTERED_CONFIG),
        "manifest_rows": manifest_rows,
        "manifest_sha256": audit.sha256_file(audit.FULL_MANIFEST),
        "manifest_index": audit.unique_index(manifest_rows, label="full_manifest"),
        "references": references,
        "bypass_row": next(
            row for row in raw_rows if bool(row.get("identity_bypass"))
        ),
        "compressed_row": next(
            row for row in raw_rows if not bool(row.get("identity_bypass"))
        ),
    }


class RealFull2638AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = _audit_bundle()
        cls.report = bundle["report"]
        cls.postmortem = bundle["postmortem"]

    def test_audit_decision_pass_and_development_stop(self) -> None:
        self.assertTrue(self.report["contract_all_passed"])
        self.assertEqual(self.report["audit_decision"], audit.AUDIT_PASS)
        self.assertEqual(self.report["failed_checks"], [])
        self.assertEqual(self.report["development_decision"]["outcome"], audit.OUTCOME_STOP)
        self.assertEqual(
            self.report["development_decision"]["failed_gates"],
            ["score_at_least_90pct_dense"],
        )

    def test_eight_contract_items_all_passed(self) -> None:
        grouped = self.report["contract_checks"]
        self.assertEqual(set(grouped), set(audit.CONTRACT_ITEMS))
        for item, group in grouped.items():
            with self.subTest(item=item):
                self.assertTrue(group["passed"], item)
                self.assertGreater(len(group["checks"]), 0)

    def test_raw_sha_and_2638_unique_ids(self) -> None:
        raw = self.report["raw_full2638"]
        self.assertEqual(raw["sha256_observed"], audit.EXPECTED_RAW_SHA256)
        self.assertTrue(raw["sha256_match"])
        self.assertEqual(raw["records"], 2638)
        self.assertEqual(raw["unique_sample_ids"], 2638)
        self.assertEqual(raw["missing_ids"], [])
        self.assertEqual(raw["extra_ids"], [])

    def test_bypass_and_compressed_counts(self) -> None:
        layout = self.report["layout"]
        self.assertEqual(layout["bypass_count"], 143)
        self.assertEqual(layout["compressed_count"], 2495)
        self.assertEqual(audit.EXPECTED_BYPASS + audit.EXPECTED_COMPRESSED, 2638)

    def test_scores_match_claim(self) -> None:
        scores = self.report["scores"]
        self.assertAlmostEqual(
            scores["new"]["cv_bench_accuracy"], audit.EXPECTED_V2_SCORE, places=15
        )
        self.assertAlmostEqual(
            scores["dense"]["cv_bench_accuracy"], audit.EXPECTED_DENSE_SCORE, places=15
        )
        self.assertAlmostEqual(
            scores["corrected_l2_integer"]["cv_bench_accuracy"],
            audit.EXPECTED_L2_SCORE,
            places=15,
        )
        self.assertAlmostEqual(
            scores["prompt_plus_corrected_l2_integer"]["cv_bench_accuracy"],
            audit.EXPECTED_PROMPT_SCORE,
            places=15,
        )

    def test_bootstrap_seed_repeats_and_ci(self) -> None:
        boot = self.report["paired_bootstrap"]
        self.assertEqual(boot["seed"], 20261006)
        self.assertEqual(boot["repeats"], 5000)
        self.assertAlmostEqual(
            boot["delta_point_estimate"], audit.EXPECTED_DELTA, places=15
        )
        self.assertAlmostEqual(boot["delta_ci95"][0], audit.EXPECTED_CI[0], places=15)
        self.assertAlmostEqual(boot["delta_ci95"][1], audit.EXPECTED_CI[1], places=15)

    def test_floor_and_shortfall(self) -> None:
        boot = self.report["paired_bootstrap"]
        self.assertAlmostEqual(boot["floor"], audit.EXPECTED_FLOOR, places=18)
        self.assertAlmostEqual(
            boot["shortfall_absolute"],
            audit.EXPECTED_FLOOR - audit.EXPECTED_V2_SCORE,
            places=18,
        )
        self.assertGreater(boot["shortfall_absolute"], 0.0)
        self.assertGreater(boot["retained_dense_fraction"], 0.0)
        self.assertLess(boot["retained_dense_fraction"], 1.0)
        self.assertEqual(boot["per_repeat_shortfall"], boot["shortfall_absolute"] * 5000)

    def test_gate_values_recomputed(self) -> None:
        gates = self.report["development_decision"]["gates"]
        self.assertFalse(gates["score_at_least_90pct_dense"])
        self.assertTrue(gates["paired_ci_lower_nonnegative"])
        self.assertTrue(gates["exact_k96_layout_contract"])
        self.assertTrue(gates["no_forbidden_leakage"])
        self.assertTrue(gates["all_2638_complete_parseable"])

    def test_frozen_pins_verified(self) -> None:
        frozen = self.report["frozen_inputs"]
        self.assertEqual(frozen["mismatched_pins"], [])
        self.assertEqual(frozen["null_pins"], ["new_arm_jsonl"])
        self.assertGreaterEqual(frozen["verified_count"], 36)

    def test_findings_record_parse_gate_defect_and_model_pin(self) -> None:
        ids = {finding["id"] for finding in self.report["findings"]}
        self.assertIn("parse_gate_implementation_defect", ids)
        self.assertIn("model_weight_sha_pin_absent", ids)
        parse = self.report["true_parse"]
        self.assertEqual(parse["new"]["failures"], 43)
        self.assertEqual(parse["dense"]["failures"], 40)
        self.assertEqual(parse["corrected_l2_integer"]["failures"], 42)

    def test_scope_cpu_only_no_gpu_runs(self) -> None:
        self.assertEqual(self.report["gpu_runs"], 0)
        self.assertEqual(self.report["audited_gpu_runs"], 1)
        self.assertEqual(self.report["audited_samples"], 2638)
        self.assertTrue(self.report["cpu_only"])
        self.assertTrue(self.report["read_only"])
        self.assertFalse(self.report["network_used"])
        self.assertFalse(self.report["training"])

    def test_postmortem_marker_and_migrations(self) -> None:
        postmortem = self.postmortem
        self.assertEqual(postmortem["marker"], "DIAGNOSTIC_DEVELOPMENT_ONLY")
        self.assertTrue(postmortem["development_only"])
        self.assertFalse(postmortem["is_a_gate"])
        self.assertTrue(postmortem["do_not_tune_or_select_v3"])
        migration = postmortem["v2_vs_corrected_l2_integer"]
        self.assertEqual(migration["repairs"], 117)
        self.assertEqual(migration["regressions"], 73)
        self.assertEqual(migration["net"], 117 - 73)
        self.assertEqual(migration["unchanged_correct"], 1941)
        self.assertEqual(migration["unchanged_wrong"], 507)
        dense_migration = postmortem["v2_vs_dense"]
        self.assertEqual(dense_migration["repairs"], 55)
        self.assertEqual(dense_migration["regressions"], 287)

    def test_postmortem_breakdowns_cover_required_dimensions(self) -> None:
        postmortem = self.postmortem
        for arm in (
            "new",
            "dense",
            "corrected_l2_integer",
            "prompt_plus_corrected_l2_integer",
        ):
            with self.subTest(arm=arm):
                self.assertIn(arm, postmortem["breakdown_by_arm"])
                self.assertIn("per_task", postmortem["breakdown_by_arm"][arm])
                self.assertIn(
                    "per_source_dataset", postmortem["breakdown_by_arm"][arm]
                )
                self.assertIn(
                    "per_original_G_bin", postmortem["breakdown_by_arm"][arm]
                )
        for label in audit.PRE_SELECTOR_BIN_ORDER:
            self.assertIn(
                label, postmortem["breakdown_by_arm"]["new"]["per_original_G_bin"]
            )

    def test_postmortem_swap_and_bypass_breakdown(self) -> None:
        postmortem = self.postmortem
        swaps = postmortem["swap_count_breakdown"]
        self.assertEqual(swaps["bypass_count"], 143)
        self.assertEqual(swaps["compressed_count"], 2495)
        self.assertEqual(swaps["compressed_zero_swap"], 0)
        self.assertEqual(swaps["histogram"]["0"], 143)
        self.assertEqual(swaps["histogram"]["16"], 2406)
        bypass = postmortem["bypass_breakdown"]
        self.assertEqual(bypass["count"], 143)
        self.assertGreaterEqual(bypass["G_min"], 1)
        self.assertLessEqual(bypass["G_max"], 96)

    def test_limitations_cover_required_disclaimers(self) -> None:
        limitations = " ".join(self.report["limitations"])
        for fragment in (
            "repeatedly observed",
            "not an untouched confirmation set",
            "gpu_runs=0",
            "FPGA latency",
            "external-validation",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, limitations)
        post_limitations = " ".join(self.postmortem["limitations"])
        for fragment in ("诊断/发展", "禁止", "未触碰确认集", "FPGA"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, post_limitations)


class DecisionLogicTest(unittest.TestCase):
    def test_failed_score_gate_forces_stop(self) -> None:
        decision = audit._decide(
            frozen_parse_rate=1.0,
            layout_ok=True,
            new_score=0.0,
            ci_lower=0.5,
            leakage_ok=True,
        )
        self.assertEqual(decision["outcome"], audit.OUTCOME_STOP)
        self.assertEqual(decision["failed_gates"], ["score_at_least_90pct_dense"])

    def test_all_gates_pass_gives_go(self) -> None:
        decision = audit._decide(
            frozen_parse_rate=1.0,
            layout_ok=True,
            new_score=1.0,
            ci_lower=0.0,
            leakage_ok=True,
        )
        self.assertEqual(decision["outcome"], audit.OUTCOME_GO)
        self.assertEqual(decision["failed_gates"], [])

    def test_negative_ci_lower_forces_stop(self) -> None:
        decision = audit._decide(
            frozen_parse_rate=1.0,
            layout_ok=True,
            new_score=1.0,
            ci_lower=-1e-12,
            leakage_ok=True,
        )
        self.assertEqual(decision["outcome"], audit.OUTCOME_STOP)
        self.assertIn("paired_ci_lower_nonnegative", decision["failed_gates"])

    def test_layout_failure_forces_stop(self) -> None:
        decision = audit._decide(
            frozen_parse_rate=1.0,
            layout_ok=False,
            new_score=1.0,
            ci_lower=0.0,
            leakage_ok=True,
        )
        self.assertEqual(decision["outcome"], audit.OUTCOME_STOP)
        self.assertIn("exact_k96_layout_contract", decision["failed_gates"])

    def test_leakage_forces_stop(self) -> None:
        decision = audit._decide(
            frozen_parse_rate=1.0,
            layout_ok=True,
            new_score=1.0,
            ci_lower=0.0,
            leakage_ok=False,
        )
        self.assertEqual(decision["outcome"], audit.OUTCOME_STOP)
        self.assertIn("no_forbidden_leakage", decision["failed_gates"])


class FailClosedTamperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = _loaded_bundle()
        # A cheap two-row slice still exercises every per-record contract path.
        cls.small_rows = [
            cls.bundle["bypass_row"],
            cls.bundle["compressed_row"],
        ]
        cls.bypass_id = str(cls.bundle["bypass_row"]["sample_id"])
        cls.compressed_id = str(cls.bundle["compressed_row"]["sample_id"])

    def _record_contract(self, rows):
        recorder = audit.CheckRecorder()
        audit._audit_record_contract(
            recorder,
            records=rows,
            manifest_index=self.bundle["manifest_index"],
            config=self.bundle["config"],
            manifest_sha256=self.bundle["manifest_sha256"],
        )
        return recorder

    def _assert_fails(self, recorder, key):
        self.assertFalse(recorder.all_passed)
        self.assertIn(key, recorder.failed_keys)

    def _mutated(self):
        return copy.deepcopy(self.small_rows)

    def test_record_sha256_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["record_sha256"] = "0" * 64
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_2}::record_sha256_recomputes"
        )

    def test_real_count_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["real_group_count"] = 88
        self._assert_fails(
            self._record_contract(rows),
            f"{ITEM_3}::compressed_exact_k96_87_real_9_context",
        )

    def test_bypass_illegality_fails(self) -> None:
        rows = self._mutated()
        rows[0]["output_group_count"] = 96
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_3}::bypass_exact_contract"
        )

    def test_region_id_map_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["region_id"][0] = (int(rows[1]["region_id"][0]) + 1) % 9
        self._assert_fails(
            self._record_contract(rows),
            f"{ITEM_3}::region_id_matches_frozen_grid_map",
        )

    def test_swap_legal_gain_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_legal_gain_q16"][0] += 1
        self._assert_fails(
            self._record_contract(rows),
            f"{ITEM_5}::legal_gain_identity_and_positive",
        )

    def test_swap_count_over_S_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_count"] = 17
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_5}::swap_count_within_S"
        )

    def test_anchor_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["baseline_anchors"] = rows[1]["baseline_anchors"][:-1]
        self._assert_fails(
            self._record_contract(rows),
            f"{ITEM_4}::anchors_frozen_one_per_region",
        )

    def test_region_mismatch_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_region_ids"][0] = (int(rows[1]["swap_region_ids"][0]) + 1) % 9
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_4}::same_region_one_for_one"
        )

    def test_leakage_flag_fails(self) -> None:
        rows = self._mutated()
        rows[1]["uses_labels"] = True
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_2}::leakage_flags_all_false"
        )

    def test_manifest_pin_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["manifest_sha256"] = "0" * 64
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_2}::manifest_sha256_pin"
        )

    def test_checkpoint_pin_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["wv_sha256"] = "0" * 64
        self._assert_fails(self._record_contract(rows), f"{ITEM_2}::checkpoint_pins")

    def test_selected_index_hash_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["selected_group_indices_sha256"] = "0" * 64
        self._assert_fails(
            self._record_contract(rows), f"{ITEM_5}::recomputable_record_hashes"
        )

    def test_missing_required_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = audit.InputPaths(
                raw_full2638=Path(tmp) / "missing.jsonl",
                report_json=Path(tmp) / "r.json",
                report_md=Path(tmp) / "r.md",
                postmortem_json=Path(tmp) / "p.json",
                postmortem_md=Path(tmp) / "p.md",
            )
            with self.assertRaises(audit.AuditError):
                audit.run_audit(paths)


class ConfigPinRecheckTest(unittest.TestCase):
    def test_recheck_pins_all_verified(self) -> None:
        result = audit.verify_config_pins()
        self.assertEqual(result["mismatched"], [])
        self.assertEqual(result["null_pins"], ["new_arm_jsonl"])
        self.assertGreaterEqual(len(result["verified"]), 36)
        self.assertEqual(result["config_sha256"], audit.EXPECTED_CONFIG_SHA256)

    def test_recheck_pins_detects_tamper(self) -> None:
        config = json.loads(audit.PREREGISTERED_CONFIG.read_text(encoding="utf-8"))
        config["inputs_sha256"]["full_manifest"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as tmp:
            tampered = Path(tmp) / "tampered.json"
            tampered.write_text(json.dumps(config), encoding="utf-8")
            result = audit.verify_config_pins(tampered)
            self.assertTrue(result["mismatched"])
            self.assertEqual(result["mismatched"][0]["name"], "full_manifest")


class ReportRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = _audit_bundle()
        cls.report = bundle["report"]
        cls.postmortem = bundle["postmortem"]

    def test_write_reports_to_temp_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = audit.InputPaths(
                report_json=Path(tmp) / "audit.json",
                report_md=Path(tmp) / "audit.md",
                postmortem_json=Path(tmp) / "post.json",
                postmortem_md=Path(tmp) / "post.md",
            )
            audit.write_reports(self.report, self.postmortem, paths)
            payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["audit_decision"], audit.AUDIT_PASS)
            self.assertEqual(payload["development_decision"]["outcome"], "DEVELOPMENT_STOP")
            markdown = paths.report_md.read_text(encoding="utf-8")
            self.assertIn("DEVELOPMENT_STOP", markdown)
            self.assertIn("CPU-only", markdown)
            self.assertIn("FPGA", markdown)
            post_markdown = paths.postmortem_md.read_text(encoding="utf-8")
            self.assertIn("诊断/发展专用", post_markdown)
            self.assertIn("不是门", post_markdown)

    def test_main_recheck_pins_returns_zero(self) -> None:
        code = audit.main(["--recheck-pins"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
