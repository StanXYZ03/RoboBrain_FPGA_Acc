"""CPU-only tests for the fail-closed oracle-pilot summariser.

They cover the official three-arm accuracy scoring, the oracle-minus-control
W->R / R->W / net accounting, the pre-registered GO/STOP gate (including a
parse-rate stop), diagnostic-only overlap / retained mass, the absence of any
confidence interval, and fail-closed behaviour for missing / duplicate / wrong
SHA inputs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.b1_answer_margin_oracle import (
    ARM_ORACLE,
    BUDGET,
    array_sha256,
    record_sha256,
)
from evaluation.summarize_b1_answer_margin_oracle_pilot import (
    ARM_ORDER,
    InputContractError,
    InputPaths,
    _verify_pins,
    analyze,
    build_report,
    render_markdown,
)

SOURCES = (
    ("ADE20K Validation Set", "ADE20K"),
    ("COCO 2017 Validation Set", "COCO"),
    ("Omni3D_Hypersim", "Omni3D"),
)
SAMPLE_IDS = [str(index) for index in range(24)]
MANIFEST_SHA = "1" * 64
CONFIG_SHA = "2" * 64


def manifest_rows() -> list[dict]:
    rows: list[dict] = []
    for source_index, (source_dataset, source) in enumerate(SOURCES):
        for offset in range(8):
            rows.append(
                {
                    "sample_id": str(source_index * 8 + offset),
                    "task": "Count",
                    "source": source,
                    "source_dataset": source_dataset,
                    "choices": ["x", "y", "z", "w"],
                    "answer": "(A)",
                }
            )
    return rows


def oracle_record(
    sample_id: str,
    output: str,
    g: int = 256,
    *,
    manifest_sha: str = MANIFEST_SHA,
    config_sha: str = CONFIG_SHA,
) -> dict:
    values = np.linspace(0.1, 1.0, g, dtype=np.float64)
    record = {
        "sample_id": sample_id,
        "arm": ARM_ORACLE,
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "requested_budget": BUDGET,
        "region_grid": 3,
        "local_candidate_cap": 16,
        "output_text": output,
        "output_token_ids": [1],
        "full_patch_tokens": g * 4,
        "image_grid_thw": [1, 16, 64],
        "full_llm_visual_tokens": g,
        "output_group_count": BUDGET,
        "context_group_count": 9,
        "real_group_count": 87,
        "identity_bypass": False,
        "selected_group_indices": list(range(BUDGET)),
        "label_tokenization": {"contract_ok": True},
        "margin": {"value": 1.5},
        "saliency": {"definition": "x"},
        "saliency_sha256": array_sha256(values),
        "saliency_group_values": [float(value) for value in values],
        "model_parameter_grads_all_none": True,
        "layout": {"passed": True},
    }
    record["record_sha256"] = record_sha256(record)
    return record


def _arm_rows(output_for) -> list[dict]:
    return [
        {
            "sample_id": sample_id,
            "output_text": output_for(sample_id),
            "selected_group_indices": list(range(96)),
        }
        for sample_id in SAMPLE_IDS
    ]


def run_analyze(control_wrong=(), oracle_wrong=(), oracle_unparsed=()):
    dense = _arm_rows(lambda sample_id: "(A)")
    control = _arm_rows(
        lambda sample_id: "(B)" if int(sample_id) in control_wrong else "(A)"
    )
    oracle = []
    for sample_id in SAMPLE_IDS:
        if int(sample_id) in oracle_unparsed:
            text = "no letter here"
        elif int(sample_id) in oracle_wrong:
            text = "(B)"
        else:
            text = "(A)"
        oracle.append(oracle_record(sample_id, text))
    return analyze(
        manifest_rows=manifest_rows(),
        dense_rows=dense,
        control_rows=control,
        oracle_rows=oracle,
        manifest_sha256=MANIFEST_SHA,
        config_sha256=CONFIG_SHA,
        config={},
    )


class AnalyzeTest(unittest.TestCase):
    def test_go_when_all_gate_conditions_hold(self) -> None:
        result = run_analyze(
            control_wrong={0, 1, 2, 3, 4}, oracle_wrong={4}
        )
        gate = result["gate"]
        self.assertEqual(gate["w_to_r"], 4)
        self.assertEqual(gate["r_to_w"], 0)
        self.assertEqual(gate["oracle_minus_control_net"], 4)
        self.assertEqual(gate["oracle_parse_count"], 24)
        self.assertEqual(gate["status"], "PILOT_GO")

    def test_stop_when_w_to_r_below_gate(self) -> None:
        result = run_analyze(control_wrong={0, 1, 2}, oracle_wrong=set())
        self.assertEqual(result["gate"]["w_to_r"], 3)
        self.assertEqual(result["gate"]["status"], "PILOT_STOP")

    def test_stop_when_r_to_w_above_gate(self) -> None:
        result = run_analyze(
            control_wrong={0, 1, 2, 3, 4}, oracle_wrong={4, 5}
        )
        self.assertEqual(result["gate"]["r_to_w"], 1)
        self.assertEqual(result["gate"]["status"], "PILOT_GO")
        result2 = run_analyze(
            control_wrong=set(), oracle_wrong={0, 1}
        )
        self.assertEqual(result2["gate"]["r_to_w"], 2)
        self.assertEqual(result2["gate"]["status"], "PILOT_STOP")

    def test_stop_when_parse_rate_below_gate(self) -> None:
        result = run_analyze(
            control_wrong={0, 1, 2, 3}, oracle_unparsed={10, 11, 12}
        )
        self.assertEqual(result["gate"]["oracle_parse_count"], 21)
        self.assertEqual(result["gate"]["status"], "PILOT_STOP")

    def test_absolute_accuracies_and_dense_headroom(self) -> None:
        result = run_analyze(control_wrong={0, 1, 2, 3, 4}, oracle_wrong={4})
        self.assertEqual(result["metrics"]["dense"]["cv_bench_accuracy"], 1.0)
        self.assertAlmostEqual(
            result["metrics"]["dense"]["cv_bench_accuracy"]
            - result["metrics"]["oracle"]["cv_bench_accuracy"],
            result["comparisons"]["dense_minus_oracle"],
        )
        self.assertFalse(result["gate"]["uses_confidence_interval"])
        self.assertNotIn("ci", json.dumps(result["comparisons"]).lower())

    def test_diagnostics_are_present_and_non_decision(self) -> None:
        result = run_analyze()
        self.assertIn("mean_jaccard", result["diagnostics"]["oracle_control_overlap"])
        self.assertIn("mean", result["diagnostics"]["saliency_retained_mass"])
        self.assertNotIn("retained_mass", result["gate"]["checks"])

    def test_source_counts(self) -> None:
        result = run_analyze()
        self.assertEqual(set(result["source_counts"].values()), {8})

    def test_missing_sample_fails_closed(self) -> None:
        oracle = [oracle_record(sample_id, "(A)") for sample_id in SAMPLE_IDS[:-1]]
        with self.assertRaises(InputContractError):
            analyze(
                manifest_rows=manifest_rows(),
                dense_rows=_arm_rows(lambda _: "(A)"),
                control_rows=_arm_rows(lambda _: "(A)"),
                oracle_rows=oracle,
                manifest_sha256=MANIFEST_SHA,
                config_sha256=CONFIG_SHA,
                config={},
            )

    def test_duplicate_sample_fails_closed(self) -> None:
        oracle = [oracle_record(sample_id, "(A)") for sample_id in SAMPLE_IDS]
        oracle.append(oracle_record(SAMPLE_IDS[0], "(A)"))
        with self.assertRaises(InputContractError):
            analyze(
                manifest_rows=manifest_rows(),
                dense_rows=_arm_rows(lambda _: "(A)"),
                control_rows=_arm_rows(lambda _: "(A)"),
                oracle_rows=oracle,
                manifest_sha256=MANIFEST_SHA,
                config_sha256=CONFIG_SHA,
                config={},
            )

    def test_wrong_layout_fails_closed(self) -> None:
        oracle = [oracle_record(sample_id, "(A)") for sample_id in SAMPLE_IDS]
        broken = dict(oracle[0])
        broken["layout"] = {"passed": False}
        broken["record_sha256"] = record_sha256(broken)
        oracle[0] = broken
        with self.assertRaises(InputContractError):
            analyze(
                manifest_rows=manifest_rows(),
                dense_rows=_arm_rows(lambda _: "(A)"),
                control_rows=_arm_rows(lambda _: "(A)"),
                oracle_rows=oracle,
                manifest_sha256=MANIFEST_SHA,
                config_sha256=CONFIG_SHA,
                config={},
            )

    def test_wrong_saliency_sha_fails_closed(self) -> None:
        oracle = [oracle_record(sample_id, "(A)") for sample_id in SAMPLE_IDS]
        broken = dict(oracle[0])
        broken["saliency_group_values"] = [0.0] * 256
        broken["record_sha256"] = record_sha256(broken)
        oracle[0] = broken
        with self.assertRaises(InputContractError):
            analyze(
                manifest_rows=manifest_rows(),
                dense_rows=_arm_rows(lambda _: "(A)"),
                control_rows=_arm_rows(lambda _: "(A)"),
                oracle_rows=oracle,
                manifest_sha256=MANIFEST_SHA,
                config_sha256=CONFIG_SHA,
                config={},
            )


class BuildReportFailClosedTest(unittest.TestCase):
    def _write_inputs(self, tmp: str, *, oracle: bool = True) -> InputPaths:
        import hashlib

        root = Path(tmp)
        manifest = root / "manifest.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest_rows()),
            encoding="utf-8",
        )
        manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
        dense = root / "dense.jsonl"
        dense.write_text(
            "".join(json.dumps(row) + "\n" for row in _arm_rows(lambda _: "(A)")),
            encoding="utf-8",
        )
        control = root / "control.jsonl"
        control.write_text(
            "".join(json.dumps(row) + "\n" for row in _arm_rows(lambda _: "(A)")),
            encoding="utf-8",
        )
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "design": "test",
                    "manifest": {"sha256": manifest_sha},
                    "inputs_sha256": {},
                    "summarizer_inputs": {},
                }
            ),
            encoding="utf-8",
        )
        config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
        oracle_path = root / "oracle.jsonl"
        if oracle:
            oracle_path.write_text(
                "".join(
                    json.dumps(
                        oracle_record(
                            sample_id,
                            "(A)",
                            manifest_sha=manifest_sha,
                            config_sha=config_sha,
                        )
                    )
                    + "\n"
                    for sample_id in SAMPLE_IDS
                ),
                encoding="utf-8",
            )
        return InputPaths(
            config=config, manifest=manifest, dense=dense, control=control, oracle=oracle_path
        )

    def test_build_report_missing_oracle_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp, oracle=False)
            with self.assertRaises(InputContractError):
                build_report(paths, root=Path(tmp))

    def test_build_report_success_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp, oracle=True)
            report = build_report(paths, root=Path(tmp))
            self.assertEqual(report["status"], "PILOT_STOP")
            self.assertFalse(report["accuracy_improvement_confirmed"])
            self.assertFalse(report["uses_confidence_interval"])
            markdown = render_markdown(report)
            self.assertIn("不可部署", markdown)
            self.assertIn("不是论文精度", markdown)

    def test_config_manifest_pin_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._write_inputs(tmp, oracle=True)
            paths.config.write_text(
                json.dumps(
                    {
                        "design": "test",
                        "manifest": {"sha256": "0" * 64},
                        "inputs_sha256": {},
                        "summarizer_inputs": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(InputContractError):
                build_report(paths, root=Path(tmp))

    def test_verify_pins_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("payload", encoding="utf-8")
            with self.assertRaises(InputContractError):
                _verify_pins(
                    {"inputs_sha256": {"a": {"path": "a.txt", "sha256": "0" * 64}}},
                    root,
                )


if __name__ == "__main__":
    unittest.main()
