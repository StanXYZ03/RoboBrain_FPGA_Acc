#!/usr/bin/env python3
"""CPU-only focused tests for evaluation/cv_bench_metric_v2.py and protocol v2.

Covers: frozen-weighting equivalence, honest parse statistics, fail-closed
validation, the explicit parse gate, lifecycle states and lifecycle-safe
probe-path handling, and the pins declared by configs/evaluation_protocol_v2.json.

No CUDA, no model weights, no network; only temp paths and existing read-only
artifacts are touched.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import cv_bench_metric_v2 as metric  # noqa: E402
from evaluation.build_cv_bench_pruning_report import score as frozen_score  # noqa: E402

PROTOCOL_CONFIG = ROOT / "configs" / "evaluation_protocol_v2.json"
MODEL_MANIFEST = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"
PROTOCOL_REPORT_JSON = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-06-future-evaluation-protocol-v2.json"
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _records() -> list[dict]:
    """Synthetic CV-Bench records with known honest parse failures."""

    rows = []
    # ADE20K: 3 records, 1 unparseable, 2 correct.
    rows.append({"sample_id": 1, "source": "ADE20K", "task": "Count", "correct": True, "answer": "A"})
    rows.append({"sample_id": 2, "source": "ADE20K", "task": "Count", "correct": True, "answer": None})
    rows.append({"sample_id": 3, "source": "ADE20K", "task": "Relation", "correct": False, "answer": "B"})
    # COCO: 2 records, 1 unparseable, 1 correct.
    rows.append({"sample_id": 4, "source": "COCO", "task": "Count", "correct": True, "answer": "C"})
    rows.append({"sample_id": 5, "source": "COCO", "task": "Relation", "correct": False, "answer": None})
    # Omni3D: 2 records, all parsed, 1 correct.
    rows.append({"sample_id": 6, "source": "Omni3D", "task": "Depth", "correct": True, "answer": "A"})
    rows.append({"sample_id": 7, "source": "Omni3D", "task": "Distance", "correct": False, "answer": "D"})
    return rows


class FrozenWeightingEquivalenceTest(unittest.TestCase):
    def test_primary_score_matches_frozen_weighting(self) -> None:
        rows = _records()
        frozen = frozen_score(rows, "correct")
        v2 = metric.compute_cv_bench_metric(
            rows, correct_key="correct", parsed_answer_key="answer"
        )
        for key in ("sample_count", "accuracy", "source_accuracy", "task_accuracy"):
            self.assertEqual(v2[key], frozen[key], key)
        for key in ("accuracy_2d", "accuracy_3d", "cv_bench_accuracy"):
            self.assertAlmostEqual(v2[key], frozen[key], places=15)

    def test_honest_parse_fixes_aliased_boolean_key(self) -> None:
        rows = _records()
        frozen = frozen_score(rows, "correct")
        v2 = metric.compute_cv_bench_metric(
            rows, correct_key="correct", parsed_answer_key="answer"
        )
        # The frozen function aliases the boolean correctness field and claims 1.0.
        self.assertEqual(frozen["parse_success_rate"], 1.0)
        # The v2 metric honestly reports the two unparseable answers.
        self.assertEqual(v2["parse_failure_count"], 2)
        self.assertEqual(v2["parse_success_count"], 5)
        self.assertAlmostEqual(v2["parse_success_rate"], 5 / 7, places=15)
        self.assertEqual(v2["correct_key"], "correct")
        self.assertEqual(v2["parsed_answer_key"], "answer")


class FailClosedTest(unittest.TestCase):
    def test_empty_records(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                [], correct_key="correct", parsed_answer_key="answer"
            )

    def test_identical_keys_rejected(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                _records(), correct_key="correct", parsed_answer_key="correct"
            )

    def test_missing_keys(self) -> None:
        rows = _records()
        for row in rows:
            row.pop("answer")
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )

    def test_wrong_types(self) -> None:
        rows = _records()
        rows[0]["correct"] = 1
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )
        rows = _records()
        rows[0]["answer"] = 3
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )

    def test_expected_count_mismatch(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                _records(),
                correct_key="correct",
                parsed_answer_key="answer",
                expected_count=999,
            )

    def test_unknown_source(self) -> None:
        rows = _records()
        rows[0]["source"] = "NotASource"
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )

    def test_duplicate_sample_id(self) -> None:
        rows = _records()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )

    def test_missing_required_source(self) -> None:
        rows = [row for row in _records() if row["source"] != "Omni3D"]
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                rows, correct_key="correct", parsed_answer_key="answer"
            )

    def test_source_override_missing_official_source_rejected(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.compute_cv_bench_metric(
                _records(),
                correct_key="correct",
                parsed_answer_key="answer",
                sources=("ADE20K", "COCO"),
            )

    def test_parsed_key_is_required_positionally(self) -> None:
        with self.assertRaises(TypeError):
            metric.compute_cv_bench_metric(_records(), correct_key="correct")  # type: ignore[call-arg]


class ParseGateTest(unittest.TestCase):
    def test_require_all_parsed_fails_on_honest_failures(self) -> None:
        v2 = metric.compute_cv_bench_metric(
            _records(), correct_key="correct", parsed_answer_key="answer"
        )
        gate = metric.parse_gate(v2, require_all_parsed=True)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["parse_failure_count"], 2)
        self.assertTrue(gate["distinct"])

    def test_relaxed_gate_still_reports_failures(self) -> None:
        v2 = metric.compute_cv_bench_metric(
            _records(), correct_key="correct", parsed_answer_key="answer"
        )
        gate = metric.parse_gate(v2, require_all_parsed=False)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["parse_failure_count"], 2)

    def test_gate_rejects_reused_correctness_field(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.parse_gate(
                {
                    "correct_key": "correct",
                    "parsed_answer_key": "correct",
                    "parse_failure_count": 0,
                    "parse_success_count": 1,
                    "parse_success_rate": 1.0,
                }
            )


class LifecycleTest(unittest.TestCase):
    def test_states_are_explicit(self) -> None:
        self.assertEqual(
            metric.LIFECYCLE_STATES,
            (
                "PREREGISTERED_NOT_EXECUTED",
                "EXECUTED_UNAUDITED",
                "AUDITED_STOP",
                "AUDITED_GO_EXTERNAL_VALIDATION",
            ),
        )
        for state in metric.LIFECYCLE_STATES:
            self.assertEqual(metric.validate_lifecycle_state(state), state)
        self.assertEqual(
            set(metric.TERMINAL_LIFECYCLE_STATES),
            {"AUDITED_STOP", "AUDITED_GO_EXTERNAL_VALIDATION"},
        )

    def test_unknown_state_rejected(self) -> None:
        with self.assertRaises(metric.MetricV2Error):
            metric.validate_lifecycle_state("EXECUTED_AND_AUDITED")

    def test_probe_inside_temp_root_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            probe = Path(td) / "missing_output.jsonl"
            resolved = metric.require_lifecycle_safe_probe_path(
                probe,
                default_output_paths=[ROOT / "artifacts" / "real-output.jsonl"],
                temp_root=td,
            )
            self.assertTrue(metric.is_temporary_path(resolved, temp_root=td))

    def test_probe_equal_to_real_default_rejected(self) -> None:
        real_default = (
            ROOT
            / "artifacts"
            / "token_selector"
            / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
            / "conservative_l2_guarded_rerank_v2_k96_full2638.jsonl"
        )
        with self.assertRaises(metric.MetricV2Error):
            metric.require_lifecycle_safe_probe_path(
                real_default, default_output_paths=[real_default]
            )

    def test_probe_outside_temp_root_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            outside = ROOT / "artifacts" / "provenance" / "scratch.jsonl"
            with self.assertRaises(metric.MetricV2Error):
                metric.require_lifecycle_safe_probe_path(
                    outside,
                    default_output_paths=[ROOT / "artifacts" / "real-output.jsonl"],
                    temp_root=td,
                )

    def test_probe_nested_under_real_default_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            real_default = Path(td) / "real-output-dir"
            nested = real_default / "probe.jsonl"
            with self.assertRaises(metric.MetricV2Error):
                metric.require_lifecycle_safe_probe_path(
                    nested,
                    default_output_paths=[real_default],
                    temp_root=td,
                )


class ProtocolConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(PROTOCOL_CONFIG.read_text(encoding="utf-8"))

    def test_future_only_not_retroactive(self) -> None:
        self.assertTrue(self.config["future_only"])
        self.assertFalse(self.config["retroactive"])

    def test_pins_match_actual_files(self) -> None:
        pins = self.config["pins"]
        self.assertEqual(
            pins["cv_bench_metric_v2_source"]["sha256"],
            _sha256(ROOT / pins["cv_bench_metric_v2_source"]["path"]),
        )
        self.assertEqual(
            pins["model_identity_manifest"]["sha256"],
            _sha256(ROOT / pins["model_identity_manifest"]["path"]),
        )
        self.assertEqual(
            pins["model_identity_builder"]["sha256"],
            _sha256(ROOT / pins["model_identity_builder"]["path"]),
        )

    def test_lifecycle_states_declared(self) -> None:
        self.assertEqual(
            tuple(self.config["lifecycle"]["states"]), metric.LIFECYCLE_STATES
        )
        self.assertEqual(
            self.config["lifecycle"]["initial_state"], "PREREGISTERED_NOT_EXECUTED"
        )

    def test_path_policy_forbids_real_default_probe(self) -> None:
        policy = self.config["lifecycle"]["path_policy"]
        self.assertTrue(policy["missing_output_tests_must_use_temporary_paths"])
        self.assertTrue(policy["never_probe_real_default_output"])
        self.assertTrue(policy["forbidden_probe_paths"])

    def test_true_parse_gate_required(self) -> None:
        scoring = self.config["scoring_contract"]
        self.assertTrue(scoring["explicit_correct_key_required"])
        self.assertTrue(scoring["explicit_parsed_answer_key_required"])
        self.assertTrue(scoring["keys_must_be_distinct"])
        self.assertTrue(scoring["never_infer_parsed_answer_key_from_correct_key"])

    def test_measurement_classes_distinguished(self) -> None:
        self.assertEqual(
            set(self.config["measurement_classes"]),
            {
                "development",
                "untouched_confirmation",
                "software_latency",
                "fpga_estimate",
                "board_measurement",
            },
        )


class ProtocolReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(PROTOCOL_REPORT_JSON.read_text(encoding="utf-8"))

    def test_honest_parse_counts_match_frozen_jsonls(self) -> None:
        counts = self.report["honest_parse_counts"]
        self.assertEqual(counts["dense"]["parse_failure_count"], 40)
        self.assertEqual(counts["corrected_l2_integer"]["parse_failure_count"], 42)
        self.assertEqual(
            counts["prompt_plus_corrected_l2_integer"]["parse_failure_count"], 42
        )
        self.assertEqual(counts["v2"]["parse_failure_count"], 43)
        for arm in counts.values():
            self.assertEqual(arm["parse_success_count"], 2638 - arm["parse_failure_count"])

    def test_findings_not_retroactive(self) -> None:
        self.assertFalse(self.report["scope"]["retroactive"])
        self.assertTrue(self.report["scope"]["future_only"])
        self.assertEqual(self.report["development_decision_unchanged"], "DEVELOPMENT_STOP")
        self.assertTrue(self.report["primary_scores_unchanged"])
        self.assertEqual(self.report["scope"]["gpu_runs"], 0)


if __name__ == "__main__":
    unittest.main()
