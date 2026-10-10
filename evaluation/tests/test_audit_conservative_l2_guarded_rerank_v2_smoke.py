#!/usr/bin/env python3
"""CPU-only focused tests for the v2 six-cell GPU smoke contract audit.

The tests pin the six contract items, the deterministic smoke-id recomputation,
the fail-closed behaviour on tampered records/pins/leakage flags, the
descriptive reference comparison, the report rendering and the exact audit
scope (``gpu_runs=0`` / ``audited_gpu_runs=1`` / ``audited_samples=6``).

No CUDA, no model weights, no network and no dependency installation are
touched; the real frozen artifacts are only read.
"""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    audit_conservative_l2_guarded_rerank_v2_smoke as audit,
)

ITEM_1 = audit.ITEM_1
ITEM_2 = audit.ITEM_2
ITEM_3 = audit.ITEM_3
ITEM_4 = audit.ITEM_4
ITEM_5 = audit.ITEM_5
ITEM_6 = audit.ITEM_6


def _loaded_bundle() -> dict:
    """Load the real frozen inputs once for the whole test class."""

    config = json.loads(
        audit.PREREGISTERED_CONFIG.read_text(encoding="utf-8")
    )
    manifest_rows = audit.load_jsonl(audit.FULL_MANIFEST)
    references = audit._load_references(audit.DEFAULT_PATHS, config)
    return {
        "raw_sha256": audit.sha256_file(audit.RAW_SMOKE),
        "raw_rows": audit.load_jsonl(audit.RAW_SMOKE),
        "config": config,
        "config_sha256": audit.sha256_file(audit.PREREGISTERED_CONFIG),
        "manifest_rows": manifest_rows,
        "manifest_sha256": audit.sha256_file(audit.FULL_MANIFEST),
        "manifest_index": audit.unique_index(manifest_rows, label="full_manifest"),
        "references": references,
    }


class RealSmokeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = _loaded_bundle()
        cls.report = audit.run_audit(audit.DEFAULT_PATHS)

    def test_decision_is_contract_smoke_pass(self) -> None:
        self.assertTrue(self.report["contract_all_passed"])
        self.assertEqual(self.report["decision"], audit.DECISION_PASS)
        self.assertEqual(self.report["failed_checks"], [])
        self.assertEqual(
            self.report["next_decision"], audit.NEXT_DECISION_PERMITTED
        )

    def test_six_contract_items_all_present_and_passed(self) -> None:
        grouped = self.report["contract_checks"]
        self.assertEqual(set(grouped), set(audit.CONTRACT_ITEMS))
        for item, group in grouped.items():
            with self.subTest(item=item):
                self.assertTrue(group["passed"], item)
                self.assertGreater(len(group["checks"]), 0)

    def test_raw_sha256_matches_claim(self) -> None:
        self.assertEqual(
            self.report["raw_smoke"]["sha256_observed"],
            audit.EXPECTED_RAW_SHA256,
        )
        self.assertEqual(
            self.bundle["raw_sha256"], audit.EXPECTED_RAW_SHA256
        )
        self.assertTrue(self.report["raw_smoke"]["sha256_match"])

    def test_sample_ids_recomputed_not_trusted(self) -> None:
        from evaluation.run_hf_conservative_l2_residual_rerank import (
            select_smoke_samples,
        )

        recomputed = [
            str(sample["sample_id"])
            for sample in select_smoke_samples(self.bundle["manifest_rows"])
        ]
        self.assertEqual(recomputed, list(audit.EXPECTED_SMOKE_IDS))
        self.assertEqual(self.report["sample_ids"], recomputed)

    def test_observed_shape(self) -> None:
        shape = self.report["observed_shape"]
        self.assertEqual(shape["sample_ids"], list(audit.EXPECTED_SMOKE_IDS))
        self.assertEqual(shape["bypass_ids"], ["286"])
        self.assertEqual(len(shape["compressed_ids"]), 5)
        self.assertEqual(
            shape["swap_counts"],
            {"286": 0, "374": 16, "1039": 16, "1392": 16, "1467": 16, "2566": 16},
        )
        self.assertEqual(
            shape["correct_counts"], audit.EXPECTED_CORRECT_COUNTS
        )

    def test_layout_counts(self) -> None:
        layout = self.report["layout"]
        self.assertEqual(layout["bypass_ids"], ["286"])
        self.assertEqual(len(layout["compressed_ids"]), 5)
        for row in layout["per_sample"]:
            with self.subTest(sample=row["sample_id"]):
                self.assertTrue(row["grid_ok"])
                self.assertTrue(row["selected_ok"])
                self.assertTrue(row["branch_ok"])

    def test_swap_audit_legality(self) -> None:
        for row in self.report["swaps"]:
            with self.subTest(sample=row["sample_id"]):
                self.assertLessEqual(row["swap_count"], audit.MAX_SWAPS)
                for index in range(row["swap_count"]):
                    self.assertGreater(row["swap_legal_gain_q16"][index], 0)
                    self.assertEqual(
                        row["swap_legal_gain_q16"][index],
                        row["swap_semantic_gain_q15"][index]
                        - row["swap_l2_loss_q15"][index],
                    )

    def test_descriptive_counts_and_agreement(self) -> None:
        quality = self.report["descriptive_quality"]
        self.assertEqual(
            quality["correct_counts"], audit.EXPECTED_CORRECT_COUNTS
        )
        self.assertTrue(quality["counts_match_expected"])
        # v2 and dense agree on every one of the six smoke samples.
        self.assertEqual(
            quality["parsed_answer_agreement_counts"]["v2_equals_dense"], 6
        )

    def test_scope_is_cpu_only_no_gpu_runs(self) -> None:
        self.assertEqual(self.report["gpu_runs"], 0)
        self.assertEqual(self.report["audited_gpu_runs"], 1)
        self.assertEqual(self.report["audited_samples"], 6)
        self.assertTrue(self.report["cpu_only"])
        self.assertTrue(self.report["read_only"])
        self.assertFalse(self.report["network_used"])
        self.assertFalse(self.report["training"])

    def test_frozen_pins_verified(self) -> None:
        frozen = self.report["frozen_inputs"]
        self.assertEqual(frozen["mismatched_pins"], [])
        self.assertGreaterEqual(len(frozen["verified_pins"]), 36)
        self.assertEqual(frozen["null_pins"], ["new_arm_jsonl"])

    def test_timing_scope_disclaims_latency(self) -> None:
        timing = self.report["timing_scope"]
        self.assertTrue(timing["first_sample_includes_first_execution_cost"])
        self.assertFalse(
            timing["any_of_six_is_steady_state_latency_evidence"]
        )
        self.assertFalse(timing["selector_cpu_roundtrip_is_fpga_latency"])
        self.assertFalse(
            timing["selector_cpu_roundtrip_is_acceleration_claim"]
        )
        self.assertAlmostEqual(
            timing["first_sample_model_inference_seconds"], 3.8936627849470824
        )


class FailClosedTamperTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = _loaded_bundle()

    def _audit(self, rows):
        return audit.audit_loaded(
            raw_sha256=self.bundle["raw_sha256"],
            raw_rows=rows,
            config=self.bundle["config"],
            config_sha256=self.bundle["config_sha256"],
            manifest_rows=self.bundle["manifest_rows"],
            manifest_sha256=self.bundle["manifest_sha256"],
            manifest_index=self.bundle["manifest_index"],
            references=self.bundle["references"],
            paths=audit.DEFAULT_PATHS,
        )

    def _mutated(self):
        return copy.deepcopy(self.bundle["raw_rows"])

    def _assert_fails(self, report, key):
        self.assertFalse(report["contract_all_passed"])
        self.assertEqual(report["decision"], audit.DECISION_FAIL)
        self.assertIn(key, report["failed_checks"])
        self.assertEqual(report["next_decision"], audit.NEXT_DECISION_STOP)

    def test_record_sha256_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["record_sha256"] = "0" * 64
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_2}::record_sha256_recomputes")

    def test_layout_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["real_group_count"] = 88
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_3}::bypass_or_exact_k96_87_9")

    def test_bypass_illegality_fails(self) -> None:
        rows = self._mutated()
        rows[0]["output_group_count"] = 96  # G=64 bypass must stay G
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_3}::bypass_or_exact_k96_87_9")

    def test_context_count_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["context_group_count"] = 8
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_3}::bypass_or_exact_k96_87_9")

    def test_swap_legal_gain_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_legal_gain_q16"][0] += 1
        report = self._audit(rows)
        self._assert_fails(
            report, f"{ITEM_5}::legal_gain_identity_and_positive[374]"
        )

    def test_swap_count_over_S_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_count"] = 17
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_5}::swap_count_within_S[374]")

    def test_anchor_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["baseline_anchors"] = rows[1]["baseline_anchors"][:-1]
        report = self._audit(rows)
        self._assert_fails(
            report, f"{ITEM_4}::anchors_frozen_and_one_per_region[374]"
        )

    def test_region_mismatch_fails(self) -> None:
        rows = self._mutated()
        rows[1]["swap_region_ids"][0] = (
            int(rows[1]["swap_region_ids"][0]) + 1
        ) % 9
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_4}::same_region_one_for_one[374]")

    def test_leakage_flag_fails(self) -> None:
        rows = self._mutated()
        rows[1]["uses_labels"] = True
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_2}::leakage_flags_all_false")

    def test_manifest_pin_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["manifest_sha256"] = "0" * 64
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_2}::manifest_sha256_pin")

    def test_checkpoint_pin_tamper_fails(self) -> None:
        rows = self._mutated()
        rows[1]["wv_sha256"] = "0" * 64
        report = self._audit(rows)
        self._assert_fails(report, f"{ITEM_2}::checkpoint_pins")

    def test_missing_required_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = audit.InputPaths(
                raw_smoke=Path(tmp) / "missing.jsonl",
                preregistered=audit.PREREGISTERED_CONFIG,
                full_manifest=audit.FULL_MANIFEST,
                dense_full_jsonl=audit.DENSE_FULL_JSONL,
                corrected_l2_integer_jsonl=audit.CORRECTED_L2_INTEGER_JSONL,
                prompt_plus_corrected_l2_integer_jsonl=(
                    audit.PROMPT_PLUS_CORRECTED_L2_INTEGER_JSONL
                ),
                report_json=Path(tmp) / "r.json",
                report_md=Path(tmp) / "r.md",
            )
            with self.assertRaises(audit.AuditError):
                audit.run_audit(paths)


class ReportRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = audit.run_audit(audit.DEFAULT_PATHS)

    def test_write_reports_to_temp_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = audit.InputPaths(
                report_json=Path(tmp) / "smoke.json",
                report_md=Path(tmp) / "smoke.md",
            )
            audit.write_reports(self.report, paths)
            payload = json.loads(paths.report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], audit.DECISION_PASS)
            self.assertEqual(payload["contract_all_passed"], True)
            markdown = paths.report_md.read_text(encoding="utf-8")
            self.assertIn("CONTRACT_SMOKE_PASS", markdown)
            self.assertIn("4/6 不是精度估计", markdown)
            self.assertIn("首次执行开销", markdown)
            self.assertIn("FPGA 时延", markdown)
            self.assertIn("论文/外部验证", markdown)

    def test_main_no_write_returns_zero(self) -> None:
        code = audit.main(["--no-write"])
        self.assertEqual(code, 0)

    def test_limitations_cover_required_disclaimers(self) -> None:
        limitations = " ".join(self.report["limitations"])
        for fragment in (
            "4/6",
            "首次执行开销",
            "稳态时延",
            "FPGA 时延",
            "论文/外部验证",
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, limitations)


if __name__ == "__main__":
    unittest.main()
