#!/usr/bin/env python3
"""CPU-only tests for the unbounded real-only vs raw-representative summariser.

Read-only against the frozen real artifacts: the tests never rewrite the raw
JSONL, the formal quality report or ``configs/evaluation_protocol.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import (  # noqa: E402
    summarize_layer5_unbounded_vs_representative_screening100 as summ,
)

# Frozen real-artifact representative arm values (read-only).
REPRESENTATIVE_SHA256 = (
    "f09f5a539141d389d362af1fd2284fe4ab8cc5edc384c1a1904408ae491e796c"
)
DENSE_SHA256 = "3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63"


def _row(
    sample_id: str,
    *,
    task: str,
    source: str,
    source_dataset: str,
    answer: str,
    output_text: str,
    image_grid_thw: list[int] | None = None,
    full_patch_tokens: int | None = None,
    full_llm_visual_tokens: int | None = None,
    score_mode: str | None = None,
    requested_budget: int | None = None,
    prune_after_layer: int | None = None,
    output_group_count: int | None = None,
    identity_bypass: bool | None = None,
    real_group_count: int | None = None,
    context_group_count: int | None = None,
    source_type: list[int] | None = None,
    representative: bool = False,
    unbounded: bool = False,
) -> dict:
    row: dict = {
        "sample_id": sample_id,
        "task": task,
        "type": "2D" if source in {"ADE20K", "COCO"} else "3D",
        "source": source,
        "source_dataset": source_dataset,
        "choices": ["0", "1"],
        "answer": answer,
        "output_text": output_text,
    }
    for key, value in (
        ("image_grid_thw", image_grid_thw),
        ("full_patch_tokens", full_patch_tokens),
        ("full_llm_visual_tokens", full_llm_visual_tokens),
        ("score_mode", score_mode),
        ("requested_budget", requested_budget),
        ("prune_after_layer", prune_after_layer),
        ("output_group_count", output_group_count),
        ("identity_bypass", identity_bypass),
        ("real_group_count", real_group_count),
        ("context_group_count", context_group_count),
        ("source_type", source_type),
    ):
        if value is not None:
            row[key] = value
    if representative:
        row["context_group_values_source"] = "representative_raw_group"
        row["uses_pooled_context_group_values"] = False
        row["context_slots_replaced"] = int(context_group_count or 0)
    if unbounded:
        row["selection_rule"] = summ.SELECTION_RULE
        row["uses_context_pooling"] = False
        row["uses_context_slot_reservation"] = False
        row["uses_local_candidate_cap"] = False
        row["local_candidate_cap"] = None
        row["uses_pooled_context_group_values"] = False
    return row


_GRIDS = {
    "s0": ([1, 16, 16], 256, 64),
    "s1": ([1, 32, 32], 1024, 256),
    "s2": ([1, 48, 48], 2304, 576),
}
_SOURCE = {"s0": "ADE20K", "s1": "COCO", "s2": "Omni3D"}
_TASK = {"s0": "Count", "s1": "Depth", "s2": "Distance"}
_DATASET = {
    "s0": "ADE20K Validation Set",
    "s1": "COCO",
    "s2": "Omni3D",
}
_ANSWER = {"s0": "(A)", "s1": "(B)", "s2": "(A)"}


def _synthetic_rows() -> dict[str, list[dict]]:
    """Three samples, one per official source, so ``score`` is well defined."""

    screening = [
        _row(
            sid,
            task=_TASK[sid],
            source=_SOURCE[sid],
            source_dataset=_DATASET[sid],
            answer=_ANSWER[sid],
            output_text="",
        )
        for sid in _GRIDS
    ]
    full = [dict(row) for row in screening]
    dense = [
        _row(
            sid,
            task=_TASK[sid],
            source=_SOURCE[sid],
            source_dataset=_DATASET[sid],
            answer=_ANSWER[sid],
            output_text=_ANSWER[sid],
        )
        for sid in _GRIDS
    ]
    representative = []
    unbounded = []
    for sid, (grid, patches, groups) in _GRIDS.items():
        # Representative is wrong on s2; unbounded is correct everywhere, so the
        # unbounded-minus-representative delta is positive and well defined.
        representative.append(
            _row(
                sid,
                task=_TASK[sid],
                source=_SOURCE[sid],
                source_dataset=_DATASET[sid],
                answer=_ANSWER[sid],
                output_text=_ANSWER[sid] if sid != "s2" else "(B)",
                image_grid_thw=grid,
                full_patch_tokens=patches,
                full_llm_visual_tokens=groups,
                score_mode="l2",
                requested_budget=96,
                prune_after_layer=5,
                output_group_count=96,
                identity_bypass=False,
                real_group_count=86,
                context_group_count=10,
                source_type=[0] * 86 + [1] * 10,
                representative=True,
            )
        )
        unbounded.append(
            _row(
                sid,
                task=_TASK[sid],
                source=_SOURCE[sid],
                source_dataset=_DATASET[sid],
                answer=_ANSWER[sid],
                output_text=_ANSWER[sid],
                image_grid_thw=grid,
                full_patch_tokens=patches,
                full_llm_visual_tokens=groups,
                score_mode="l2",
                requested_budget=96,
                prune_after_layer=5,
                output_group_count=96,
                identity_bypass=False,
                real_group_count=96,
                context_group_count=0,
                source_type=[0] * 96,
                unbounded=True,
            )
        )
    return {
        "screening": screening,
        "full": full,
        "dense": dense,
        "representative": representative,
        "unbounded": unbounded,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_workspace(
    tmp: Path,
    rows: dict[str, list[dict]] | None = None,
    *,
    pins: dict | None = None,
):
    rows = rows or _synthetic_rows()
    tmp.mkdir(parents=True, exist_ok=True)
    screening = tmp / "screening.jsonl"
    full = tmp / "full.jsonl"
    dense = tmp / "dense.jsonl"
    representative = tmp / "representative.jsonl"
    unbounded = tmp / "unbounded.jsonl"
    _write_jsonl(screening, rows["screening"])
    _write_jsonl(full, rows["full"])
    _write_jsonl(dense, rows["dense"])
    _write_jsonl(representative, rows["representative"])
    _write_jsonl(unbounded, rows["unbounded"])
    prereg = tmp / "prereg.json"
    prereg.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_IMMUTABLE",
                "inputs_sha256": pins if pins is not None else {},
            }
        ),
        encoding="utf-8",
    )
    paths = summ.InputPaths(
        preregistered=prereg,
        screening_manifest=screening,
        full_manifest=full,
        dense_full_jsonl=dense,
        representative_layer5_l2_screening_jsonl=representative,
        unbounded_layer5_l2_screening_jsonl=unbounded,
    )
    return paths, rows


def _indices(rows=None):
    rows = rows or _synthetic_rows()
    return {name: summ.index_unique(rows[name], name) for name in rows}


def _validate(idx):
    return summ.validate_arms(
        screening_index=idx["screening"],
        dense_index=idx["dense"],
        representative_index=idx["representative"],
        unbounded_index=idx["unbounded"],
        expected_screening_count=3,
        expected_full_count=None,
    )


class DecisionRuleTest(unittest.TestCase):
    def test_confirmed_when_ci_lower_above_threshold(self) -> None:
        d = summ.decide([0.0201, 0.10])
        self.assertEqual(
            d["outcome"], summ.HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED
        )

    def test_confirmed_threshold_is_strict(self) -> None:
        d = summ.decide([0.02, 0.10])
        self.assertEqual(
            d["outcome"], summ.HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED
        )

    def test_not_confirmed_when_ci_lower_below_or_equal(self) -> None:
        for ci in ([-0.05, 0.01], [0.0, 0.03], [-0.10, -0.02]):
            with self.subTest(ci=ci):
                d = summ.decide(ci)
                self.assertEqual(
                    d["outcome"],
                    summ.HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED,
                )

    def test_decision_uses_primary_metric_only_and_ignores_dense(self) -> None:
        d = summ.decide([0.03, 0.04])
        self.assertTrue(d["uses_only_primary_metric"])
        self.assertFalse(d["dense_drives_decision"])
        self.assertEqual(d["dense_role"], summ.DENSE_ROLE)
        self.assertEqual(d["metric"], summ.PRIMARY_METRIC)
        self.assertEqual(d["threshold"], 0.02)
        self.assertEqual(d["ordered_rules"], list(summ.DECISION_ORDER))
        self.assertEqual(d["comparison"], "unbounded_minus_representative")


class IndexAndGridTest(unittest.TestCase):
    def test_index_unique_detects_duplicates(self) -> None:
        rows = [
            {"sample_id": "a", "output_text": "(A)"},
            {"sample_id": "a", "output_text": "(B)"},
        ]
        with self.assertRaises(summ.InputContractError):
            summ.index_unique(rows, "demo")

    def test_index_unique_requires_sample_id(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ.index_unique([{"output_text": "(A)"}], "demo")

    def test_grid_signature_detects_missing_field(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ.grid_signature({"image_grid_thw": [1, 2, 2]})


class ValidationTest(unittest.TestCase):
    def test_happy_path(self) -> None:
        v = _validate(_indices())
        self.assertTrue(v["sample_ids_identical"])
        self.assertTrue(v["all_score_mode_l2"])
        self.assertTrue(v["all_requested_budget_96"])
        self.assertTrue(v["all_prune_after_layer_5"])
        self.assertTrue(v["input_grid_consistent_representative_vs_unbounded"])
        self.assertTrue(v["representative_declaration_ok"])
        self.assertTrue(v["unbounded_declaration_ok"])
        self.assertTrue(v["unbounded_all_real"])
        self.assertTrue(v["unbounded_no_context"])
        self.assertTrue(v["unbounded_no_context_slot_reservation"])
        self.assertTrue(v["unbounded_no_local_candidate_cap"])
        self.assertTrue(v["unbounded_exact_k_or_identity"])
        self.assertFalse(v["selection_metadata_identity_required"])

    def test_wrong_score_mode_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][1]["score_mode"] = "integer"
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_wrong_budget_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["requested_budget"] = 64
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_non_bypass_output_group_count_not_96_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][2]["output_group_count"] = 95
        rows["unbounded"][2]["real_group_count"] = 95
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_wrong_cutpoint_fails(self) -> None:
        for cut in (0, 4, 6):
            with self.subTest(cut=cut):
                rows = _synthetic_rows()
                rows["representative"][0]["prune_after_layer"] = cut
                with self.assertRaises(summ.InputContractError):
                    _validate(_indices(rows))

    def test_grid_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][2]["full_patch_tokens"] = 999
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_missing_representative_declaration_fails(self) -> None:
        rows = _synthetic_rows()
        del rows["representative"][0]["uses_pooled_context_group_values"]
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_representative_pooled_values_true_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][0]["uses_pooled_context_group_values"] = True
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_missing_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"] = rows["unbounded"][:2]
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_extra_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"].append(
            dict(rows["unbounded"][0], sample_id="extra")
        )
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_wrong_screening_count_fails(self) -> None:
        idx = _indices()
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                representative_index=idx["representative"],
                unbounded_index=idx["unbounded"],
                expected_screening_count=100,
                expected_full_count=None,
            )


class UnboundedDeclarationTest(unittest.TestCase):
    def test_context_pooling_true_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["uses_context_pooling"] = True
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_context_slot_reservation_true_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["uses_context_slot_reservation"] = True
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_local_candidate_cap_true_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["uses_local_candidate_cap"] = True
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_positive_local_candidate_cap_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["local_candidate_cap"] = 16
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_wrong_selection_rule_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][1]["selection_rule"] = "bounded_context_pooling"
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_non_real_source_type_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][2]["source_type"] = [0] * 95 + [1]
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_context_group_count_nonzero_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["context_group_count"] = 1
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_real_group_count_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["unbounded"][0]["real_group_count"] = 90
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))

    def test_identity_bypass_is_accepted(self) -> None:
        rows = _synthetic_rows()
        for position, sid in enumerate(_GRIDS):
            rows["unbounded"][position]["identity_bypass"] = True
            rows["unbounded"][position]["output_group_count"] = _GRIDS[sid][2]
            rows["unbounded"][position]["real_group_count"] = _GRIDS[sid][2]
            rows["unbounded"][position]["source_type"] = [0] * _GRIDS[sid][2]
        v = _validate(_indices(rows))
        self.assertTrue(v["unbounded_exact_k_or_identity"])

    def test_identity_bypass_with_non_real_slot_fails(self) -> None:
        rows = _synthetic_rows()
        row = rows["unbounded"][0]
        row["identity_bypass"] = True
        row["output_group_count"] = _GRIDS["s0"][2]
        row["real_group_count"] = _GRIDS["s0"][2]
        row["source_type"] = [0] * (_GRIDS["s0"][2] - 1) + [1]
        with self.assertRaises(summ.InputContractError):
            _validate(_indices(rows))


class MetricNamingTest(unittest.TestCase):
    def _joined(self):
        idx = _indices()
        return summ.build_joined(
            list(idx["screening"]),
            idx["screening"],
            {
                "dense": idx["dense"],
                "representative": idx["representative"],
                "unbounded": idx["unbounded"],
            },
        )

    def test_arm_metrics_primary_and_secondary(self) -> None:
        metrics = summ.arm_metrics(self._joined(), "dense", "dense")
        self.assertEqual(metrics["primary_metric"], summ.PRIMARY_METRIC)
        self.assertAlmostEqual(metrics["cv_bench_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["unweighted_sample_accuracy"], 1.0)
        self.assertNotIn("accuracy", metrics)
        self.assertEqual(metrics["unweighted_metric"], summ.SECONDARY_METRIC)

    def test_join_recomputes_correctness_from_output_text(self) -> None:
        rows = _synthetic_rows()
        rows["dense"][0]["output_text"] = "(B)"  # wrong for answer (A)
        rows["dense"][0]["dense_correct"] = True  # stale flag must be ignored
        idx = _indices(rows)
        joined = summ.build_joined(
            list(idx["screening"]), idx["screening"], {"dense": idx["dense"]}
        )
        by_id = {row["sample_id"]: row for row in joined}
        self.assertFalse(by_id["s0"]["dense_correct"])
        self.assertTrue(by_id["s1"]["dense_correct"])
        self.assertTrue(by_id["s2"]["dense_correct"])

    def test_no_forbidden_metric_keys_in_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            report = summ.build_report(
                paths,
                expected_screening_count=3,
                expected_full_count=None,
                verify_pins=True,
            )
            for arm in report["metrics"].values():
                self.assertIn("cv_bench_accuracy", arm)
                self.assertIn("unweighted_sample_accuracy", arm)
                self.assertNotIn("accuracy", arm)
                for name in summ.FORBIDDEN_PRIMARY_METRICS:
                    self.assertNotIn(name, arm)
            self.assertEqual(
                tuple(report["forbidden_primary_metrics"]),
                summ.FORBIDDEN_PRIMARY_METRICS,
            )


class SyntheticReportTest(unittest.TestCase):
    def test_build_report_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            report = summ.build_report(
                paths,
                expected_screening_count=3,
                expected_full_count=None,
                verify_pins=True,
            )
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["formal_full2638_conclusion"])
            self.assertEqual(report["primary_metric"]["identifier"], summ.PRIMARY_METRIC)
            self.assertEqual(
                set(report["metrics"]), {"dense", "representative", "unbounded"}
            )
            self.assertEqual(
                set(report["comparisons"]),
                {
                    "unbounded_minus_representative",
                    "unbounded_minus_dense",
                    "representative_minus_dense",
                },
            )
            for comp in report["comparisons"].values():
                self.assertEqual(comp["metric"], summ.PRIMARY_METRIC)
                self.assertEqual(comp["seed"], summ.BOOTSTRAP_SEED)
                self.assertEqual(comp["repeats"], summ.BOOTSTRAP_REPEATS)
                self.assertEqual(comp["stratified_by"], "task::source_dataset")
            self.assertEqual(report["arms"]["dense"]["role"], summ.DENSE_ROLE)
            self.assertFalse(report["decision"]["dense_drives_decision"])
            self.assertTrue(report["config_pin_verification"]["pins_verified"])
            self.assertFalse(
                report["validation"]["selection_metadata_identity_required"]
            )
            self.assertEqual(
                report["decision"]["outcome"],
                summ.HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED,
            )
            md = summ.render_markdown(report)
            self.assertIn("unbounded", md)
            self.assertIn("CV-Bench 主分", md)
            self.assertIn("unweighted_sample_accuracy", md)
            self.assertIn("不得写作 CV-Bench 主分", md)
            self.assertIn("NOT a formal full2638", md)
            self.assertIn(
                summ.HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED, md
            )

    def test_build_report_hashes_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            report = summ.build_report(
                paths,
                expected_screening_count=3,
                expected_full_count=None,
                verify_pins=True,
            )
            for name, item in report["inputs_sha256"].items():
                self.assertEqual(
                    item["sha256"],
                    hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(),
                    name,
                )

    def test_pin_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            paths.preregistered.write_text(
                json.dumps(
                    {
                        "status": "PREREGISTERED_IMMUTABLE",
                        "inputs_sha256": {
                            "representative_layer5_l2_screening_jsonl": {
                                "path": "ignored.jsonl",
                                "sha256": "0" * 64,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths, expected_screening_count=3, expected_full_count=None
                )

    def test_matching_pin_passes_and_null_unbounded_pin_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            representative_sha = hashlib.sha256(
                paths.representative_layer5_l2_screening_jsonl.read_bytes()
            ).hexdigest()
            paths.preregistered.write_text(
                json.dumps(
                    {
                        "status": "PREREGISTERED_IMMUTABLE",
                        "inputs_sha256": {
                            "representative_layer5_l2_screening_jsonl": {
                                "path": "representative.jsonl",
                                "sha256": representative_sha,
                            },
                            "unbounded_layer5_l2_screening_jsonl": {
                                "path": "unbounded.jsonl",
                                "sha256": None,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = summ.build_report(
                paths, expected_screening_count=3, expected_full_count=None
            )
            self.assertTrue(report["config_pin_verification"]["pins_verified"])
            paths.unbounded_layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths, expected_screening_count=3, expected_full_count=None
                )


class FailClosedTest(unittest.TestCase):
    def test_missing_unbounded_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.unbounded_layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths,
                    expected_screening_count=3,
                    expected_full_count=None,
                    verify_pins=False,
                )

    def test_missing_representative_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.representative_layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths,
                    expected_screening_count=3,
                    expected_full_count=None,
                    verify_pins=False,
                )

    def test_missing_prereg_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.preregistered.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths,
                    expected_screening_count=3,
                    expected_full_count=None,
                    verify_pins=False,
                )

    def test_cli_returns_two_and_writes_nothing_when_unbounded_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            paths.unbounded_layer5_l2_screening_jsonl.unlink()
            stderr = io.StringIO()
            original_default = summ.DEFAULT_PATHS
            summ.DEFAULT_PATHS = paths
            try:
                out_json = tmp_path / "out.json"
                out_md = tmp_path / "out.md"
                with redirect_stderr(stderr):
                    code = summ.main(
                        ["--output-json", str(out_json), "--output-md", str(out_md)]
                    )
                self.assertEqual(code, 2)
                self.assertIn("FAIL_CLOSED", stderr.getvalue())
                self.assertFalse(out_json.exists())
                self.assertFalse(out_md.exists())
            finally:
                summ.DEFAULT_PATHS = original_default


class PreregisteredConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json"
            ).read_text(encoding="utf-8")
        )

    def test_status_and_scope(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_IMMUTABLE")
        self.assertTrue(self.config["diagnostic_only"])
        self.assertIn("not a formal full2638", self.config["diagnostic_scope"].lower())

    def test_primary_metric_is_official_only(self) -> None:
        self.assertEqual(
            self.config["primary_metric"]["identifier"], summ.PRIMARY_METRIC
        )
        self.assertEqual(
            self.config["unweighted_metric"]["identifier"], summ.UNWEIGHTED_METRIC
        )
        self.assertIn("cv_bench_accuracy", summ.PRIMARY_METRIC)

    def test_fixed_configuration_has_no_local_cap(self) -> None:
        cfg = self.config["fixed_configuration"]
        self.assertEqual(cfg["budget_K"], 96)
        self.assertEqual(cfg["region_grid"], 3)
        self.assertIsNone(cfg["local_candidate_cap"])
        self.assertEqual(cfg["dtype"], "bfloat16")
        self.assertEqual(cfg["attention_implementation"], "sdpa")
        self.assertEqual(cfg["decoding"], "greedy_argmax")
        self.assertEqual(cfg["score_mode"], "l2")
        self.assertFalse(cfg["do_sample"])
        self.assertEqual(cfg["cut_point"], 5)
        self.assertFalse(cfg["b1_trained_wv_used"])

    def test_decision_rule_threshold_and_outcomes(self) -> None:
        rule = self.config["decision_rule"]
        self.assertEqual(rule["threshold"], 0.02)
        outcomes = [item["outcome"] for item in rule["ordered_rules"]]
        self.assertEqual(
            outcomes,
            [
                summ.HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED,
                summ.HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED,
            ],
        )
        self.assertIn("reference only", rule["dense_role"])
        self.assertIn("unbounded_minus_representative", rule["uses"])

    def test_bootstrap_pins(self) -> None:
        boot = self.config["bootstrap"]
        self.assertEqual(boot["seed"], summ.BOOTSTRAP_SEED)
        self.assertEqual(boot["repeats"], summ.BOOTSTRAP_REPEATS)
        self.assertEqual(
            boot["method"], "paired bootstrap, stratified by task::source_dataset"
        )

    def test_representative_arm_is_the_frozen_file(self) -> None:
        arm = self.config["arms"]["representative_layer5_l2"]
        self.assertEqual(arm["sha256"], REPRESENTATIVE_SHA256)
        self.assertEqual(arm["required_score_mode"], "l2")
        self.assertEqual(arm["required_requested_budget"], 96)
        self.assertEqual(arm["required_prune_after_layer"], 5)
        self.assertEqual(
            arm["required_context_group_values_source"], "representative_raw_group"
        )
        self.assertFalse(arm["required_uses_pooled_context_group_values"])
        self.assertTrue(arm["forbids_b1_trained_wv"])

    def test_unbounded_arm_is_pending_and_declares_real_only(self) -> None:
        arm = self.config["arms"]["unbounded_layer5_l2"]
        self.assertIsNone(arm["sha256"])
        self.assertEqual(arm["status"], "TO_BE_GENERATED_AFTER_PREREGISTRATION")
        self.assertEqual(arm["required_score_mode"], "l2")
        self.assertEqual(arm["required_requested_budget"], 96)
        self.assertEqual(arm["required_prune_after_layer"], 5)
        self.assertEqual(arm["required_selection_rule"], summ.SELECTION_RULE)
        self.assertFalse(arm["required_uses_context_pooling"])
        self.assertFalse(arm["required_uses_context_slot_reservation"])
        self.assertFalse(arm["required_uses_local_candidate_cap"])
        self.assertIsNone(arm["required_local_candidate_cap"])
        self.assertEqual(arm["required_context_group_count"], 0)
        self.assertTrue(arm["required_all_real"])
        self.assertTrue(arm["required_exact_k_or_identity"])
        self.assertEqual(
            arm["runner"],
            "evaluation/run_hf_hardware_regular_layer5_unbounded_l2.py",
        )

    def test_validation_requirements_do_not_require_cross_arm_metadata(self) -> None:
        req = self.config["validation_requirements"]
        self.assertIn("unbounded_declaration", req)
        self.assertIn("representative_declaration", req)
        self.assertIn("fail_closed", req)
        self.assertIn("NOT required", req["selection_metadata_identity"])

    def test_dense_is_reference_only(self) -> None:
        self.assertIn("REFERENCE ONLY", self.config["arms"]["dense"]["role"])

    def test_report_paths(self) -> None:
        self.assertEqual(
            self.config["report_paths"]["json"],
            "reports/cv_bench/"
            "2026-10-04-layer5-unbounded-vs-representative-screening100.json",
        )
        self.assertEqual(
            self.config["report_paths"]["md"],
            "reports/cv_bench/"
            "2026-10-04-layer5-unbounded-vs-representative-screening100.md",
        )

    def test_pinned_input_hashes_match_files(self) -> None:
        pins = dict(self.config["inputs_sha256"])
        for name, spec in pins.items():
            expected = spec.get("sha256") if isinstance(spec, dict) else spec
            if expected is None:
                # The unbounded artifact is generated after preregistration.
                continue
            relative = spec.get("path") if isinstance(spec, dict) else name
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_pinned_new_arm_modules_match_files(self) -> None:
        pins = self.config["inputs_sha256"]
        for name in (
            "evaluation/hardware_regular_layer5_unbounded_l2.py",
            "evaluation/run_hf_hardware_regular_layer5_unbounded_l2.py",
        ):
            self.assertIn(name, pins)
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                pins[name]["sha256"],
                name,
            )


class RealFrozenRepresentativeSubsetTest(unittest.TestCase):
    """Read-only checks against the real frozen representative screening100 arm."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.screening = summ.index_unique(
            summ.load_rows(summ.SCREENING_MANIFEST, "screening"), "screening"
        )
        cls.representative = summ.index_unique(
            summ.load_rows(
                summ.REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL, "representative"
            ),
            "representative",
        )

    def test_screening_has_100_unique_ids(self) -> None:
        self.assertEqual(len(self.screening), 100)

    def test_representative_has_100_unique_ids(self) -> None:
        self.assertEqual(len(self.representative), 100)

    def test_every_screening_id_present_in_representative(self) -> None:
        self.assertTrue(set(self.screening).issubset(self.representative))

    def test_representative_records_are_l2_k96_layer5(self) -> None:
        for row in self.representative.values():
            self.assertEqual(row["score_mode"], "l2")
            self.assertEqual(int(row["requested_budget"]), 96)
            self.assertEqual(int(row["prune_after_layer"]), 5)
            expected = (
                int(row["full_llm_visual_tokens"])
                if row["identity_bypass"]
                else 96
            )
            self.assertEqual(int(row["output_group_count"]), expected)

    def test_representative_declares_context_ablation(self) -> None:
        for row in self.representative.values():
            self.assertEqual(
                row["context_group_values_source"], "representative_raw_group"
            )
            self.assertFalse(row["uses_pooled_context_group_values"])

    def test_representative_sha_matches_pin(self) -> None:
        self.assertEqual(
            summ.sha256_file(summ.REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL),
            REPRESENTATIVE_SHA256,
        )
        self.assertEqual(summ.sha256_file(summ.DENSE_FULL_JSONL), DENSE_SHA256)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
