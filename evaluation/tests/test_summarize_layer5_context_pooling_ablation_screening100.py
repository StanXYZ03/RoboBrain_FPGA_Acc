#!/usr/bin/env python3
"""CPU-only tests for the layer-5 context-pooling ablation summariser.

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
    summarize_layer5_context_pooling_ablation_screening100 as summ,
)
from evaluation.build_cv_bench_pruning_report import score  # noqa: E402

# Frozen real-artifact screening100 pooled layer-5 values (read-only, official scorer).
POOLED_SCREENING100_CV_BENCH_ACCURACY = 0.7416666666666667
POOLED_SCREENING100_UNWEIGHTED = 0.75
POOLED_SHA256 = "5395d5e2846022a75f8bb0317d4fa8ccd8e62771582437b55811bd08c64a571e"
DENSE_SHA256 = "3ff3b77ea4bddae96a8231c653bf08db1ce396a9c125fb414466417b1ea69f63"


def _metadata() -> dict:
    return {
        "selected_group_indices": [0, 1, 2, 3, 4, 5],
        "source_type": [0, 0, 0, 0, 1, 1],
        "region_id": [0, 1, 2, 3, 0, 1],
        "valid": [True, True, True, True, True, True],
        "context_group_count": 2,
        "real_group_count": 4,
    }


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
    metadata: bool = False,
    ablation: bool = False,
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
    if image_grid_thw is not None:
        row["image_grid_thw"] = list(image_grid_thw)
    if full_patch_tokens is not None:
        row["full_patch_tokens"] = full_patch_tokens
    if full_llm_visual_tokens is not None:
        row["full_llm_visual_tokens"] = full_llm_visual_tokens
    if score_mode is not None:
        row["score_mode"] = score_mode
    if requested_budget is not None:
        row["requested_budget"] = requested_budget
    if prune_after_layer is not None:
        row["prune_after_layer"] = prune_after_layer
    if output_group_count is not None:
        row["output_group_count"] = output_group_count
    if identity_bypass is not None:
        row["identity_bypass"] = identity_bypass
    if metadata:
        row.update(_metadata())
    if ablation:
        row["context_group_values_source"] = "representative_raw_group"
        row["uses_pooled_context_group_values"] = False
        row["context_slots_replaced"] = 2
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
    pooled = []
    representative = []
    for sid, (grid, patches, groups) in _GRIDS.items():
        # Pooled is correct on s0/s1 and wrong on s2; the representative arm
        # flips s2 to correct, so the context-pooling delta is well defined.
        pooled.append(
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
                metadata=True,
            )
        )
        representative.append(
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
                metadata=True,
                ablation=True,
            )
        )
    return {
        "screening": screening,
        "full": full,
        "dense": dense,
        "pooled": pooled,
        "representative": representative,
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
    pooled = tmp / "pooled.jsonl"
    representative = tmp / "representative.jsonl"
    _write_jsonl(screening, rows["screening"])
    _write_jsonl(full, rows["full"])
    _write_jsonl(dense, rows["dense"])
    _write_jsonl(pooled, rows["pooled"])
    _write_jsonl(representative, rows["representative"])
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
        pooled_layer5_l2_screening_jsonl=pooled,
        representative_layer5_l2_screening_jsonl=representative,
    )
    return paths, rows


class DecisionRuleTest(unittest.TestCase):
    def test_confirmed_when_ci_lower_above_threshold(self) -> None:
        d = summ.decide([0.0201, 0.10])
        self.assertEqual(d["outcome"], summ.CONTEXT_POOLING_HARM_CONFIRMED)

    def test_confirmed_threshold_is_strict(self) -> None:
        d = summ.decide([0.02, 0.10])
        self.assertEqual(d["outcome"], summ.CONTEXT_POOLING_HARM_NOT_CONFIRMED)

    def test_not_confirmed_when_ci_lower_below_or_equal(self) -> None:
        for ci in ([-0.05, 0.01], [0.0, 0.03], [-0.10, -0.02]):
            with self.subTest(ci=ci):
                d = summ.decide(ci)
                self.assertEqual(
                    d["outcome"], summ.CONTEXT_POOLING_HARM_NOT_CONFIRMED
                )

    def test_decision_uses_primary_metric_only_and_ignores_dense(self) -> None:
        d = summ.decide([0.03, 0.04])
        self.assertTrue(d["uses_only_primary_metric"])
        self.assertFalse(d["dense_drives_decision"])
        self.assertEqual(d["dense_role"], summ.DENSE_ROLE)
        self.assertEqual(d["metric"], summ.PRIMARY_METRIC)
        self.assertEqual(d["threshold"], 0.02)
        self.assertEqual(d["ordered_rules"], list(summ.DECISION_ORDER))
        self.assertEqual(d["comparison"], summ.DECISION_COMPARISON)


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

    def test_metadata_signature_detects_missing_field(self) -> None:
        with self.assertRaises(summ.InputContractError):
            summ.metadata_signature({"selected_group_indices": [1]})


class ValidationTest(unittest.TestCase):
    def _indices(self, rows=None):
        rows = rows or _synthetic_rows()
        return {name: summ.index_unique(rows[name], name) for name in rows}

    def _validate(self, idx):
        return summ.validate_arms(
            screening_index=idx["screening"],
            dense_index=idx["dense"],
            pooled_index=idx["pooled"],
            representative_index=idx["representative"],
            expected_screening_count=3,
            expected_full_count=None,
        )

    def test_happy_path(self) -> None:
        v = self._validate(self._indices())
        self.assertTrue(v["sample_ids_identical"])
        self.assertTrue(v["all_score_mode_l2"])
        self.assertTrue(v["all_requested_budget_96"])
        self.assertTrue(v["all_prune_after_layer_5"])
        self.assertTrue(v["selection_metadata_identical"])
        self.assertTrue(v["input_grid_consistent_pooled_vs_representative"])
        self.assertTrue(v["ablation_declaration_ok"])
        self.assertEqual(
            v["selection_metadata_fields"], list(summ.SELECTION_METADATA_FIELDS)
        )

    def test_wrong_score_mode_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][1]["score_mode"] = "integer"
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_wrong_budget_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][0]["requested_budget"] = 64
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_non_bypass_output_group_count_not_96_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][2]["output_group_count"] = 95
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_wrong_cutpoint_fails(self) -> None:
        for cut in (0, 4, 6):
            with self.subTest(cut=cut):
                rows = _synthetic_rows()
                rows["pooled"][0]["prune_after_layer"] = cut
                with self.assertRaises(summ.InputContractError):
                    self._validate(self._indices(rows))

    def test_grid_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][2]["full_patch_tokens"] = 999
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_selection_metadata_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][0]["selected_group_indices"] = [9, 8, 7, 6, 5, 4]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_source_type_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][1]["source_type"] = [1, 1, 1, 1, 0, 0]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_region_id_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][2]["region_id"] = [8, 8, 8, 8, 8, 8]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_context_count_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][0]["context_group_count"] = 3
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_missing_ablation_declaration_fails(self) -> None:
        rows = _synthetic_rows()
        del rows["representative"][0]["uses_pooled_context_group_values"]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_pooled_values_declared_true_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][0]["uses_pooled_context_group_values"] = True
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_context_slots_replaced_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"][1]["context_slots_replaced"] = 1
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_missing_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"] = rows["representative"][:2]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_extra_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["representative"].append(
            dict(rows["representative"][0], sample_id="extra")
        )
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_wrong_screening_count_fails(self) -> None:
        idx = self._indices()
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                pooled_index=idx["pooled"],
                representative_index=idx["representative"],
                expected_screening_count=100,
                expected_full_count=None,
            )


class MetricNamingTest(unittest.TestCase):
    def _joined(self):
        rows = _synthetic_rows()
        screening = summ.index_unique(rows["screening"], "screening")
        dense = summ.index_unique(rows["dense"], "dense")
        pooled = summ.index_unique(rows["pooled"], "pooled")
        representative = summ.index_unique(
            rows["representative"], "representative"
        )
        return summ.build_joined(
            list(screening),
            screening,
            {
                "dense": dense,
                "pooled": pooled,
                "representative": representative,
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
        screening = summ.index_unique(rows["screening"], "screening")
        dense = summ.index_unique(rows["dense"], "dense")
        joined = summ.build_joined(list(screening), screening, {"dense": dense})
        by_id = {row["sample_id"]: row for row in joined}
        self.assertFalse(by_id["s0"]["dense_correct"])
        self.assertTrue(by_id["s1"]["dense_correct"])
        self.assertTrue(by_id["s2"]["dense_correct"])
        self.assertAlmostEqual(
            summ.arm_metrics(joined, "dense", "dense")["cv_bench_accuracy"], 0.75
        )

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
                set(report["metrics"]), {"dense", "pooled", "representative"}
            )
            self.assertEqual(
                set(report["comparisons"]),
                {
                    "representative_minus_pooled",
                    "representative_minus_dense",
                    "pooled_minus_dense",
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
            self.assertTrue(
                report["validation"]["selection_metadata_identical"]
            )
            md = summ.render_markdown(report)
            self.assertIn("context-pooling", md)
            self.assertIn("CV-Bench 主分", md)
            self.assertIn("unweighted_sample_accuracy", md)
            self.assertIn("不得写作 CV-Bench 主分", md)
            self.assertIn("NOT a formal full2638", md)
            self.assertIn("selection metadata", md)
            self.assertIn(summ.CONTEXT_POOLING_HARM_CONFIRMED, md)

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
                            "pooled_layer5_l2_screening_jsonl": {
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

    def test_matching_pin_passes_and_null_representative_pin_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            pooled_sha = hashlib.sha256(
                paths.pooled_layer5_l2_screening_jsonl.read_bytes()
            ).hexdigest()
            paths.preregistered.write_text(
                json.dumps(
                    {
                        "status": "PREREGISTERED_IMMUTABLE",
                        "inputs_sha256": {
                            "pooled_layer5_l2_screening_jsonl": {
                                "path": "pooled.jsonl",
                                "sha256": pooled_sha,
                            },
                            "representative_layer5_l2_screening_jsonl": {
                                "path": "representative.jsonl",
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
            paths.representative_layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths, expected_screening_count=3, expected_full_count=None
                )


class FailClosedTest(unittest.TestCase):
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

    def test_missing_pooled_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.pooled_layer5_l2_screening_jsonl.unlink()
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

    def test_cli_returns_two_and_writes_nothing_when_representative_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            paths.representative_layer5_l2_screening_jsonl.unlink()
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
                / "2026-10-04-layer5-context-pooling-ablation-screening100.preregistered.json"
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

    def test_fixed_configuration(self) -> None:
        cfg = self.config["fixed_configuration"]
        self.assertEqual(cfg["budget_K"], 96)
        self.assertEqual(cfg["region_grid"], 3)
        self.assertEqual(cfg["local_candidate_cap"], 16)
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
                summ.CONTEXT_POOLING_HARM_CONFIRMED,
                summ.CONTEXT_POOLING_HARM_NOT_CONFIRMED,
            ],
        )
        self.assertIn("reference only", rule["dense_role"])
        self.assertIn("representative_minus_pooled", rule["uses"])

    def test_bootstrap_pins(self) -> None:
        boot = self.config["bootstrap"]
        self.assertEqual(boot["seed"], summ.BOOTSTRAP_SEED)
        self.assertEqual(boot["repeats"], summ.BOOTSTRAP_REPEATS)
        self.assertEqual(
            boot["method"], "paired bootstrap, stratified by task::source_dataset"
        )

    def test_pooled_arm_is_the_frozen_existing_file(self) -> None:
        pooled = self.config["arms"]["pooled_layer5_l2"]
        self.assertEqual(pooled["sha256"], POOLED_SHA256)
        self.assertEqual(pooled["required_score_mode"], "l2")
        self.assertEqual(pooled["required_requested_budget"], 96)
        self.assertEqual(pooled["required_prune_after_layer"], 5)
        self.assertTrue(pooled["uses_pooled_context_group_values"])
        self.assertEqual(
            pooled["path"],
            "artifacts/token_selector/2026-10-04-layer5-l2-diagnostic/"
            "layer5_l2_k96_screening100.jsonl",
        )

    def test_representative_arm_is_pending_and_declares_ablation(self) -> None:
        arm = self.config["arms"]["representative_layer5_l2"]
        self.assertIsNone(arm["sha256"])
        self.assertEqual(arm["status"], "TO_BE_GENERATED_AFTER_PREREGISTRATION")
        self.assertEqual(arm["required_score_mode"], "l2")
        self.assertEqual(arm["required_requested_budget"], 96)
        self.assertEqual(arm["required_prune_after_layer"], 5)
        self.assertEqual(
            arm["required_context_group_values_source"], "representative_raw_group"
        )
        self.assertFalse(arm["required_uses_pooled_context_group_values"])
        self.assertTrue(arm["forbids_b1_trained_wv"])
        self.assertEqual(
            arm["runner"],
            "evaluation/run_hf_hardware_regular_layer5_context_ablation.py",
        )

    def test_single_variable_contract(self) -> None:
        contract = self.config["single_variable_contract"]
        self.assertIn("only_change", contract)
        self.assertIn("SOURCE_CONTEXT", contract["only_change"])
        self.assertIn("SOURCE_REAL", contract["only_change"])
        self.assertIn("metadata", contract["verification"])

    def test_validation_requires_metadata_identity(self) -> None:
        req = self.config["validation_requirements"]
        self.assertIn("selection_metadata_identity", req)
        self.assertIn("ablation_declaration", req)
        self.assertIn("fail_closed", req)

    def test_dense_is_reference_only(self) -> None:
        self.assertIn("REFERENCE ONLY", self.config["arms"]["dense"]["role"])

    def test_report_paths(self) -> None:
        self.assertEqual(
            self.config["report_paths"]["json"],
            "reports/cv_bench/2026-10-04-layer5-context-pooling-ablation-screening100.json",
        )
        self.assertEqual(
            self.config["report_paths"]["md"],
            "reports/cv_bench/2026-10-04-layer5-context-pooling-ablation-screening100.md",
        )

    def test_pinned_input_hashes_match_files(self) -> None:
        pins = dict(self.config["inputs_sha256"])
        pins["datasets/cv_bench_screening100.jsonl"] = self.config["dataset_pins"][
            "screening_manifest"
        ]["sha256"]
        pins["datasets/cv_bench_full2638.jsonl"] = self.config["dataset_pins"][
            "full_manifest"
        ]["sha256"]
        pins[
            "evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl"
        ] = self.config["arms"]["dense"]["sha256"]
        pins[
            "artifacts/token_selector/2026-10-04-layer5-l2-diagnostic/"
            "layer5_l2_k96_screening100.jsonl"
        ] = self.config["arms"]["pooled_layer5_l2"]["sha256"]
        for name, spec in pins.items():
            expected = spec.get("sha256") if isinstance(spec, dict) else spec
            if expected is None:
                # The representative artifact is generated after preregistration.
                continue
            relative = spec.get("path") if isinstance(spec, dict) else name
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


class RealFrozenPooledSubsetTest(unittest.TestCase):
    """Read-only checks against the real frozen screening100 pooled baseline."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.screening = summ.index_unique(
            summ.load_rows(summ.SCREENING_MANIFEST, "screening"), "screening"
        )
        cls.dense = summ.index_unique(
            summ.load_rows(summ.DENSE_FULL_JSONL, "dense"), "dense"
        )
        cls.pooled = summ.index_unique(
            summ.load_rows(summ.POOLED_LAYER5_L2_SCREENING_JSONL, "pooled"),
            "pooled",
        )
        cls.joined = summ.build_joined(
            list(cls.screening),
            cls.screening,
            {"dense": cls.dense, "pooled": cls.pooled},
        )

    def test_screening_has_100_unique_ids(self) -> None:
        self.assertEqual(len(self.screening), 100)

    def test_every_screening_id_present_in_both_full_sets(self) -> None:
        ids = set(self.screening)
        self.assertTrue(ids.issubset(self.dense))
        self.assertTrue(ids.issubset(self.pooled))

    def test_pooled_official_scores(self) -> None:
        pooled = summ.arm_metrics(self.joined, "pooled", "pooled")
        self.assertAlmostEqual(
            pooled["cv_bench_accuracy"],
            POOLED_SCREENING100_CV_BENCH_ACCURACY,
            places=12,
        )
        self.assertAlmostEqual(
            pooled["unweighted_sample_accuracy"],
            POOLED_SCREENING100_UNWEIGHTED,
            places=12,
        )
        # The two metrics genuinely differ, so mixing them is detectable.
        self.assertGreater(
            abs(pooled["cv_bench_accuracy"] - pooled["unweighted_sample_accuracy"]),
            1e-6,
        )

    def test_pooled_records_are_l2_k96_layer5_with_metadata(self) -> None:
        for row in self.pooled.values():
            self.assertEqual(row["score_mode"], "l2")
            self.assertEqual(int(row["requested_budget"]), 96)
            self.assertEqual(int(row["prune_after_layer"]), 5)
            expected = (
                int(row["full_llm_visual_tokens"])
                if row["identity_bypass"]
                else 96
            )
            self.assertEqual(int(row["output_group_count"]), expected)
            for field in summ.SELECTION_METADATA_FIELDS:
                self.assertIn(field, row)
                self.assertEqual(len(row[field]), expected)

    def test_pooled_sha_matches_pin(self) -> None:
        self.assertEqual(
            summ.sha256_file(summ.POOLED_LAYER5_L2_SCREENING_JSONL), POOLED_SHA256
        )
        self.assertEqual(summ.sha256_file(summ.DENSE_FULL_JSONL), DENSE_SHA256)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
