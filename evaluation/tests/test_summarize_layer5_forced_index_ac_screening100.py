from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation import summarize_layer5_forced_index_ac_screening100 as summary
from evaluation.build_cv_bench_pruning_report import (
    paired_bootstrap_delta as official_paired_bootstrap_delta,
)
from evaluation.build_cv_bench_pruning_report import score as official_score


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection_sha(indices) -> str:
    return hashlib.sha256(
        ",".join(str(int(index)) for index in indices).encode()
    ).hexdigest()


TASKS = ("Count", "Depth", "Distance", "Relation")
SOURCES = ("ADE20K", "COCO", "Omni3D")


def make_manifest(count: int) -> list[dict]:
    rows = []
    for index in range(count):
        source = SOURCES[index % len(SOURCES)]
        task = TASKS[index % len(TASKS)]
        rows.append(
            {
                "sample_id": str(index),
                "task": task,
                "type": "3D" if source == "Omni3D" else "2D",
                "source": source,
                "source_dataset": source + " Val",
                "choices": ["zero", "one"],
                "answer": "(A)",
            }
        )
    return rows


def reference_for(meta: dict) -> dict:
    index = int(meta["sample_id"])
    group_count = 80 + 8 * index
    k_c = max(2, group_count // 8)
    indices = list(range(0, 2 * k_c, 2))
    return {
        "sample_id": meta["sample_id"],
        "task": meta["task"],
        "prune_after_layer": 5,
        "baseline_llm_visual_tokens": group_count,
        "llm_visual_tokens": len(indices),
        "selected_group_indices": indices,
        "selected_group_indices_sha256": selection_sha(indices),
    }


def control_indices(reference: dict) -> list[int]:
    k_c = len(reference["selected_group_indices"])
    return list(range(1, 2 * k_c, 2))


def make_run(arm: str, manifest_sha: str, reference_sha: str) -> dict:
    rule = (
        summary.SELECTION_RULE_CONTROL
        if arm == summary.ARM_CONTROL
        else summary.SELECTION_RULE_FORCED
    )
    return {
        "dtype": "bf16",
        "attn_implementation": "sdpa",
        "do_sample": False,
        "custom_greedy_decode": True,
        "arm": arm,
        "mode": arm,
        "selection_rule": rule,
        "prune_after_layer": 5,
        "region_grid": 3,
        "manifest_sha256": manifest_sha,
        "reference_jsonl_sha256": reference_sha,
        "reference_selector": summary.REFERENCE_SELECTOR,
        "has_fixed_budget_knob": False,
        "local_candidate_cap": None,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "projection_npz": None,
        "training": False,
        "K_C_is_per_sample_historical_length": True,
    }


def make_arm_record(
    meta: dict,
    reference: dict,
    *,
    arm: str,
    selected: list[int],
    output_text: str,
    manifest_sha: str,
    reference_sha: str,
    dtype: str = "bf16",
    attn: str = "sdpa",
    decoding: str = "greedy_argmax",
    do_sample: bool = False,
    fpga_deployable_candidate: bool = False,
    manifest_sha_override: str | None = None,
) -> dict:
    rule = (
        summary.SELECTION_RULE_CONTROL
        if arm == summary.ARM_CONTROL
        else summary.SELECTION_RULE_FORCED
    )
    k_c = len(selected)
    group_count = int(reference["baseline_llm_visual_tokens"])
    record = {
        "sample_id": meta["sample_id"],
        "task": meta["task"],
        "arm": arm,
        "mode": arm,
        "selection_rule": rule,
        "prune_after_layer": 5,
        "region_grid": 3,
        "output_text": output_text,
        "raw_output_text": output_text,
        "K_C": k_c,
        "reference_K_C": int(reference["llm_visual_tokens"]),
        "reference_G": group_count,
        "reference_selected_group_indices": list(
            reference["selected_group_indices"]
        ),
        "reference_selected_group_indices_sha256": reference[
            "selected_group_indices_sha256"
        ],
        "reference_llm_visual_tokens": int(reference["llm_visual_tokens"]),
        "reference_baseline_llm_visual_tokens": group_count,
        "reference_selector": summary.REFERENCE_SELECTOR,
        "reference_jsonl_sha256": reference_sha,
        "manifest_sha256": manifest_sha_override or manifest_sha,
        "uses_b1_trained_wv": False,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "local_candidate_cap": None,
        "uses_pooled_context_group_values": False,
        "do_sample": do_sample,
        "decoding": decoding,
        "full_patch_tokens": group_count * 4,
        "image_grid_thw": [1, 2, 2 * group_count],
        "full_llm_visual_tokens": group_count,
        "baseline_llm_visual_tokens": group_count,
        "output_group_count": k_c,
        "context_group_count": 0,
        "real_group_count": k_c,
        "source_types_all_real": True,
        "identity_bypass": False,
        "requested_K_C": k_c,
        "selected_group_indices": list(selected),
        "selected_group_indices_sha256": selection_sha(selected),
        "source_type": [0] * k_c,
        "region_id": [0] * k_c,
        "valid": [True] * k_c,
        "llm_visual_tokens": k_c,
        "hardware_alignment_relaxed_for_diagnostic": True,
        "causal_diagnostic_only": True,
        "fpga_deployable_candidate": fpga_deployable_candidate,
        "only_variable": "selected_group_indices",
        "model_inference_seconds": 0.10,
        "application_end_to_end_seconds": 0.20,
        "vision_total_seconds": 0.05,
        "llm_prefill_seconds": 0.02,
        "decoder_seconds": 0.03,
        "selector_cpu_roundtrip_seconds": 0.001,
        "run": make_run(arm, manifest_sha, reference_sha),
    }
    record["run"]["dtype"] = dtype
    record["run"]["attn_implementation"] = attn
    return record


class Case:
    def __init__(self, root: Path):
        self.root = root
        self.manifest = root / "manifest.jsonl"
        self.reference = root / "reference.jsonl"
        self.control = root / "control.jsonl"
        self.forced = root / "forced.jsonl"
        self.config = root / "prereg.json"
        self.out_json = root / "report.json"
        self.out_csv = root / "report.csv"
        self.out_md = root / "report.md"
        self.manifest_rows: list[dict] = []
        self.reference_rows: list[dict] = []
        self.control_rows: list[dict] = []
        self.forced_rows: list[dict] = []
        self.manifest_sha = ""
        self.reference_sha = ""

    def write(self) -> "Case":
        self.manifest.write_text(
            "\n".join(json.dumps(row) for row in self.manifest_rows) + "\n",
            encoding="utf-8",
        )
        self.reference.write_text(
            "\n".join(json.dumps(row) for row in self.reference_rows) + "\n",
            encoding="utf-8",
        )
        self.manifest_sha = sha256_file(self.manifest)
        self.reference_sha = sha256_file(self.reference)
        self.control.write_text(
            "\n".join(json.dumps(row) for row in self.control_rows) + "\n",
            encoding="utf-8",
        )
        self.forced.write_text(
            "\n".join(json.dumps(row) for row in self.forced_rows) + "\n",
            encoding="utf-8",
        )
        pins = {
            "screening_manifest": {
                "path": str(self.manifest),
                "sha256": self.manifest_sha,
            },
            "reference_jsonl": {
                "path": str(self.reference),
                "sha256": self.reference_sha,
            },
            "control_arm": {
                "path": str(self.control),
                "sha256": sha256_file(self.control),
            },
            "forced_arm": {
                "path": str(self.forced),
                "sha256": sha256_file(self.forced),
            },
        }
        self.config.write_text(
            json.dumps({"inputs_sha256": pins}, indent=2), encoding="utf-8"
        )
        return self

    def paths(self) -> summary.InputPaths:
        return summary.InputPaths(
            preregistered=self.config,
            screening_manifest=self.manifest,
            reference_jsonl=self.reference,
            control_arm=self.control,
            forced_arm=self.forced,
        )

    def outputs(self) -> summary.OutputPaths:
        return summary.OutputPaths(
            json=self.out_json, csv=self.out_csv, md=self.out_md
        )

    def cli_args(self) -> list[str]:
        return [
            "--preregistered",
            str(self.config),
            "--manifest",
            str(self.manifest),
            "--reference-jsonl",
            str(self.reference),
            "--control-jsonl",
            str(self.control),
            "--forced-jsonl",
            str(self.forced),
            "--output-json",
            str(self.out_json),
            "--output-csv",
            str(self.out_csv),
            "--output-md",
            str(self.out_md),
        ]


def build_case(
    root: Path,
    *,
    count: int = 4,
    control_text: str = "(A) zero",
    forced_text: str = "(A) zero",
) -> Case:
    case = Case(root)
    manifest_rows = make_manifest(count)
    reference_rows = [reference_for(meta) for meta in manifest_rows]
    # Write manifest/reference first so the arm records can pin their hashes.
    case.manifest.write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n", encoding="utf-8"
    )
    case.reference.write_text(
        "\n".join(json.dumps(row) for row in reference_rows) + "\n",
        encoding="utf-8",
    )
    manifest_sha = sha256_file(case.manifest)
    reference_sha = sha256_file(case.reference)

    control_rows = []
    forced_rows = []
    for meta, reference in zip(manifest_rows, reference_rows):
        control_rows.append(
            make_arm_record(
                meta,
                reference,
                arm=summary.ARM_CONTROL,
                selected=control_indices(reference),
                output_text=control_text,
                manifest_sha=manifest_sha,
                reference_sha=reference_sha,
            )
        )
        forced_rows.append(
            make_arm_record(
                meta,
                reference,
                arm=summary.ARM_FORCED,
                selected=list(reference["selected_group_indices"]),
                output_text=forced_text,
                manifest_sha=manifest_sha,
                reference_sha=reference_sha,
            )
        )
    case.manifest_rows = manifest_rows
    case.reference_rows = reference_rows
    case.control_rows = control_rows
    case.forced_rows = forced_rows
    return case.write()


def analyze_case(case: Case, *, count: int | None = None):
    return summary.analyze(
        screening_rows=case.manifest_rows,
        reference_rows=case.reference_rows,
        control_rows=case.control_rows,
        forced_rows=case.forced_rows,
        manifest_sha256=case.manifest_sha,
        reference_sha256=case.reference_sha,
        expected_screening_count=count if count is not None else len(case.manifest_rows),
    )


class ForcedIndexSummaryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- official scorer / decision -----------------------------------------
    def test_uses_official_cv_bench_scorer(self) -> None:
        case = build_case(self.root, count=4)
        with mock.patch.object(summary, "score", wraps=official_score) as scorer:
            result = analyze_case(case)
        self.assertGreater(scorer.call_count, 0)
        self.assertEqual(result["metrics"][summary.ARM_CONTROL]["primary_metric"],
                         'score(...)["cv_bench_accuracy"]')
        self.assertAlmostEqual(
            result["metrics"][summary.ARM_CONTROL]["cv_bench_accuracy"], 1.0
        )
        self.assertAlmostEqual(
            result["metrics"][summary.ARM_FORCED]["cv_bench_accuracy"], 1.0
        )

    def test_frozen_bootstrap_contract(self) -> None:
        self.assertEqual(summary.BOOTSTRAP_SEED, 20261004)
        self.assertEqual(summary.BOOTSTRAP_REPEATS, 5000)
        self.assertEqual(summary.STRATIFIED_BY, "task::source_dataset")
        case = build_case(self.root, count=4)
        captured = {}

        def fake_paired(records, left, right, seed, repeats=5000):
            captured.update(
                {"left": left, "right": right, "seed": seed, "repeats": repeats}
            )
            return [0.0, 0.0]

        with mock.patch.object(summary, "paired_bootstrap_delta", fake_paired):
            result = analyze_case(case)
        self.assertEqual(captured["seed"], 20261004)
        self.assertEqual(captured["left"], "forced_correct")
        self.assertEqual(captured["right"], "control_correct")
        self.assertEqual(result["comparison"]["stratified_by"], "task::source_dataset")

    def test_decision_confirmed_branch_forced_better(self) -> None:
        case = build_case(
            self.root, count=4, control_text="(B) one", forced_text="(A) zero"
        )
        result = analyze_case(case)
        self.assertEqual(
            result["decision"]["outcome"], summary.DECISION_CONFIRMED
        )
        self.assertTrue(result["decision"]["historical_index_set_better"])
        self.assertGreater(result["comparison"]["delta_ci95"][0], 0.0)

    def test_decision_not_confirmed_branch_equal(self) -> None:
        case = build_case(self.root, count=4)
        result = analyze_case(case)
        self.assertEqual(
            result["decision"]["outcome"], summary.DECISION_NOT_CONFIRMED
        )
        self.assertFalse(result["decision"]["historical_index_set_better"])

    def test_decide_boundary_is_strict(self) -> None:
        below = summary.decide(
            {"delta_point_estimate": 0.1, "delta_ci95": [-0.0, 0.4]}
        )
        self.assertEqual(below["outcome"], summary.DECISION_NOT_CONFIRMED)
        above = summary.decide(
            {"delta_point_estimate": 0.1, "delta_ci95": [0.0001, 0.4]}
        )
        self.assertEqual(above["outcome"], summary.DECISION_CONFIRMED)

    # --- statistics / descriptive -------------------------------------------
    def test_reports_transitions_budget_and_selection(self) -> None:
        result = analyze_case(build_case(self.root, count=6))
        transitions = result["transitions"]
        self.assertEqual(transitions["both_correct"], 6)
        self.assertEqual(transitions["forced_only_correct"], 0)
        budget = result["budget_overview"]
        self.assertEqual(budget["sample_count"], 6)
        self.assertGreaterEqual(budget["K_C_min"], 1)
        selection = result["selection_overview"]
        self.assertEqual(selection["forced_equals_reference_count"], 6)
        self.assertEqual(selection["equal_count"], 0)
        self.assertFalse(result["timing_overview"]["used_for_decision"])

    def test_secondary_delta_records_reference_metric(self) -> None:
        result = analyze_case(build_case(self.root, count=4))
        self.assertEqual(
            result["metrics"][summary.ARM_CONTROL]["unweighted_metric"],
            'score(...)["accuracy"]',
        )
        self.assertEqual(
            result["comparison"]["metric"], 'score(...)["cv_bench_accuracy"]'
        )

    # --- fail-closed contract ------------------------------------------------
    def test_rejects_forced_not_equal_reference(self) -> None:
        case = build_case(self.root, count=4)
        row = copy.deepcopy(case.forced_rows[0])
        replacement = list(row["selected_group_indices"])
        replacement[-1] = replacement[-1] + 1
        row["selected_group_indices"] = replacement
        row["selected_group_indices_sha256"] = selection_sha(replacement)
        case.forced_rows[0] = row
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_forced_K_mismatch(self) -> None:
        case = build_case(self.root, count=4)
        row = copy.deepcopy(case.forced_rows[0])
        row["K_C"] = row["K_C"] + 1
        case.forced_rows[0] = row
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_duplicate_sample_id(self) -> None:
        case = build_case(self.root, count=4)
        case.control_rows.append(copy.deepcopy(case.control_rows[0]))
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_mismatched_sample_id_sets(self) -> None:
        case = build_case(self.root, count=4)
        case.forced_rows = case.forced_rows[:-1]
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_bad_run_metadata_dtype(self) -> None:
        case = build_case(self.root, count=4)
        case.forced_rows[0]["run"]["dtype"] = "fp16"
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_bad_run_metadata_attention(self) -> None:
        case = build_case(self.root, count=4)
        case.control_rows[0]["run"]["attn_implementation"] = "eager"
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_non_greedy_decoding(self) -> None:
        case = build_case(self.root, count=4)
        case.forced_rows[0]["decoding"] = "sampling"
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_fpga_deployable_claim(self) -> None:
        case = build_case(self.root, count=4)
        case.forced_rows[0]["fpga_deployable_candidate"] = True
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_record_manifest_hash_mismatch(self) -> None:
        case = build_case(self.root, count=4)
        case.control_rows[0]["manifest_sha256"] = "0" * 64
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_context_group(self) -> None:
        case = build_case(self.root, count=4)
        case.control_rows[0]["uses_context_pooling"] = True
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_control_arm_field_swap(self) -> None:
        case = build_case(self.root, count=4)
        case.control_rows[0]["arm"] = summary.ARM_FORCED
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    def test_rejects_reference_task_mismatch(self) -> None:
        case = build_case(self.root, count=4)
        case.reference_rows[0]["task"] = "Relation-other"
        with self.assertRaises(summary.InputContractError):
            analyze_case(case)

    # --- pins / report assembly ---------------------------------------------
    def test_build_report_verifies_pins(self) -> None:
        case = build_case(self.root, count=4)
        report = summary.build_report(
            case.paths(), expected_screening_count=4
        )
        self.assertTrue(report["config_pin_verification"]["pins_verified"])
        self.assertEqual(
            len(report["config_pin_verification"]["checks"]), 4
        )

    def test_build_report_rejects_tampered_pin(self) -> None:
        case = build_case(self.root, count=4)
        config = json.loads(case.config.read_text(encoding="utf-8"))
        config["inputs_sha256"]["screening_manifest"]["sha256"] = "0" * 64
        case.config.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaises(summary.InputContractError):
            summary.build_report(case.paths(), expected_screening_count=4)

    def test_null_pin_is_skipped_but_file_required(self) -> None:
        case = build_case(self.root, count=4)
        config = json.loads(case.config.read_text(encoding="utf-8"))
        config["inputs_sha256"]["control_arm"]["sha256"] = None
        config["inputs_sha256"]["control_arm"]["status"] = "TO_BE_GENERATED"
        case.config.write_text(json.dumps(config), encoding="utf-8")
        report = summary.build_report(case.paths(), expected_screening_count=4)
        self.assertIn(
            "control_arm", report["config_pin_verification"]["null_pins_skipped"]
        )
        case.control.unlink()
        with self.assertRaises(summary.InputContractError):
            summary.build_report(case.paths(), expected_screening_count=4)

    def test_csv_has_required_columns(self) -> None:
        case = build_case(self.root, count=4)
        report = summary.build_report(case.paths(), expected_screening_count=4)
        text = summary.render_csv(report)
        header = text.splitlines()[0].split(",")
        self.assertEqual(header, list(summary.CSV_COLUMNS))
        for column in (
            "source",
            "source_dataset",
            "task",
            "ground_truth",
            "control_answer",
            "forced_answer",
            "control_correct",
            "forced_correct",
            "K_C",
            "control_indices_sha256",
            "forced_indices_sha256",
            "output_text_changed",
        ):
            self.assertIn(column, header)

    def test_markdown_mentions_screening_scope(self) -> None:
        case = build_case(self.root, count=4)
        report = summary.build_report(case.paths(), expected_screening_count=4)
        markdown = summary.render_markdown(report)
        self.assertIn("screening100", markdown)
        self.assertIn("不用于决策", markdown)
        self.assertIn("forced − control", markdown)

    # --- CLI: no report written on failure / written on success -------------
    def test_main_writes_no_report_on_failure(self) -> None:
        case = build_case(self.root, count=100)
        row = copy.deepcopy(case.forced_rows[0])
        replacement = list(row["selected_group_indices"])
        replacement[-1] = replacement[-1] + 1
        row["selected_group_indices"] = replacement
        row["selected_group_indices_sha256"] = selection_sha(replacement)
        case.forced_rows[0] = row
        case.write()
        code = summary.main(case.cli_args())
        self.assertEqual(code, 2)
        self.assertFalse(case.out_json.exists())
        self.assertFalse(case.out_csv.exists())
        self.assertFalse(case.out_md.exists())

    def test_main_writes_all_outputs_on_success(self) -> None:
        case = build_case(self.root, count=100)
        code = summary.main(case.cli_args())
        self.assertEqual(code, 0)
        self.assertTrue(case.out_json.exists())
        self.assertTrue(case.out_csv.exists())
        self.assertTrue(case.out_md.exists())
        payload = json.loads(case.out_json.read_text(encoding="utf-8"))
        self.assertIn(payload["decision"]["outcome"],
                      {summary.DECISION_CONFIRMED, summary.DECISION_NOT_CONFIRMED})
        self.assertEqual(len(payload["per_sample"]), 100)
        csv_lines = case.out_csv.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(csv_lines), 101)

    # --- config / integration consistency -----------------------------------
    def test_real_config_pins_and_pending_arms(self) -> None:
        self.assertTrue(summary.PREREGISTERED_CONFIG.is_file())
        config = json.loads(
            summary.PREREGISTERED_CONFIG.read_text(encoding="utf-8")
        )
        pins = config["inputs_sha256"]
        for arm_name in ("control_arm", "forced_arm"):
            self.assertIn(arm_name, pins)
            self.assertIsNone(pins[arm_name]["sha256"])
            self.assertIn("TO_BE_GENERATED", pins[arm_name]["status"].upper())
        resolved = {
            "screening_manifest": summary.SCREENING_MANIFEST,
            "reference_jsonl": summary.REFERENCE_JSONL,
            "control_arm": summary.CONTROL_ARM,
            "forced_arm": summary.FORCED_ARM,
        }
        for name, spec in pins.items():
            expected = spec.get("sha256")
            if expected is None:
                continue
            path = resolved.get(name)
            if path is None:
                path = summary.ROOT / spec["path"]
            self.assertTrue(path.is_file(), name)
            self.assertEqual(sha256_file(path), expected, name)

    def test_integration_frozen_constants_match(self) -> None:
        from evaluation import hardware_regular_layer5_forced_index_ablation as integ

        self.assertEqual(summary.SELECTION_RULE_CONTROL, integ.SELECTION_RULE_CONTROL)
        self.assertEqual(summary.SELECTION_RULE_FORCED, integ.SELECTION_RULE_FORCED)
        self.assertEqual(summary.ONLY_VARIABLE, "selected_group_indices")
        self.assertEqual(
            summary.HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC,
            integ.HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC,
        )
        self.assertEqual(
            summary.CAUSAL_DIAGNOSTIC_ONLY, integ.CAUSAL_DIAGNOSTIC_ONLY
        )
        self.assertEqual(
            summary.FPGA_DEPLOYABLE_CANDIDATE, integ.FPGA_DEPLOYABLE_CANDIDATE
        )
        self.assertEqual(summary.PRUNE_AFTER_LAYER, integ.PRUNE_AFTER_LAYER)

    def test_official_paired_helper_is_reused(self) -> None:
        case = build_case(self.root, count=8)
        with mock.patch.object(
            summary,
            "paired_bootstrap_delta",
            wraps=official_paired_bootstrap_delta,
        ) as helper:
            analyze_case(case)
        self.assertGreater(helper.call_count, 0)


if __name__ == "__main__":
    unittest.main()
