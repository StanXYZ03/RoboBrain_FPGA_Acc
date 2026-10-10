#!/usr/bin/env python3
"""CPU-only tests for the layer-5 vs B1 L2 screening100 summariser.

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

from evaluation import summarize_layer5_vs_b1_l2_screening100 as summ  # noqa: E402
from evaluation.build_cv_bench_pruning_report import score  # noqa: E402

# Frozen real-artifact screening100 values (read-only, official scorer).
DENSE_SCREENING100_CV_BENCH_ACCURACY = 0.8916666666666666
B1_SCREENING100_CV_BENCH_ACCURACY = 0.6625000000000001
DENSE_SCREENING100_UNWEIGHTED = 0.9
B1_SCREENING100_UNWEIGHTED = 0.66
B1_MINUS_DENSE_CI = [-0.30833333333333346, -0.1499999999999999]
B1_SHA256 = "983300c1816a1d54a2f89354c14606cdc69b7fc4528fa299ce875c07d13849c6"


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
    return row


_GRIDS = {
    "s0": ([1, 16, 16], 256, 64),
    "s1": ([1, 32, 32], 1024, 256),
    "s2": ([1, 48, 48], 2304, 576),
}


def _synthetic_rows() -> dict[str, list[dict]]:
    """Three samples, one per official source, so ``score`` is well defined."""

    screening = [
        _row("s0", task="Count", source="ADE20K", source_dataset="ADE20K Validation Set", answer="(A)", output_text=""),
        _row("s1", task="Depth", source="COCO", source_dataset="COCO", answer="(B)", output_text=""),
        _row("s2", task="Distance", source="Omni3D", source_dataset="Omni3D", answer="(A)", output_text=""),
    ]
    full = [
        _row("s0", task="Count", source="ADE20K", source_dataset="ADE20K Validation Set", answer="(A)", output_text=""),
        _row("s1", task="Depth", source="COCO", source_dataset="COCO", answer="(B)", output_text=""),
        _row("s2", task="Distance", source="Omni3D", source_dataset="Omni3D", answer="(A)", output_text=""),
    ]
    dense = [
        _row("s0", task="Count", source="ADE20K", source_dataset="ADE20K Validation Set", answer="(A)", output_text="(A)"),
        _row("s1", task="Depth", source="COCO", source_dataset="COCO", answer="(B)", output_text="(B)"),
        _row("s2", task="Distance", source="Omni3D", source_dataset="Omni3D", answer="(A)", output_text="(A)"),
    ]
    b1 = []
    layer5 = []
    for sid, (grid, patches, groups) in _GRIDS.items():
        answer = {"s0": "(A)", "s1": "(B)", "s2": "(A)"}[sid]
        source = {"s0": "ADE20K", "s1": "COCO", "s2": "Omni3D"}[sid]
        task = {"s0": "Count", "s1": "Depth", "s2": "Distance"}[sid]
        source_dataset = {
            "s0": "ADE20K Validation Set",
            "s1": "COCO",
            "s2": "Omni3D",
        }[sid]
        # B1 is correct on s0/s1, wrong on s2; layer5 flips s2 to correct so the
        # cut-point delta is well defined and non-zero.
        b1.append(
            _row(
                sid,
                task=task,
                source=source,
                source_dataset=source_dataset,
                answer=answer,
                output_text=answer if sid != "s2" else "(B)",
                image_grid_thw=grid,
                full_patch_tokens=patches,
                full_llm_visual_tokens=groups,
                score_mode="l2",
                requested_budget=96,
                output_group_count=96,
                identity_bypass=False,
            )
        )
        layer5.append(
            _row(
                sid,
                task=task,
                source=source,
                source_dataset=source_dataset,
                answer=answer,
                output_text=answer,
                image_grid_thw=grid,
                full_patch_tokens=patches,
                full_llm_visual_tokens=groups,
                score_mode="l2",
                requested_budget=96,
                prune_after_layer=5,
                output_group_count=96,
                identity_bypass=False,
            )
        )
    return {
        "screening": screening,
        "full": full,
        "dense": dense,
        "b1": b1,
        "layer5": layer5,
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
    b1 = tmp / "b1.jsonl"
    layer5 = tmp / "layer5.jsonl"
    _write_jsonl(screening, rows["screening"])
    _write_jsonl(full, rows["full"])
    _write_jsonl(dense, rows["dense"])
    _write_jsonl(b1, rows["b1"])
    _write_jsonl(layer5, rows["layer5"])
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
        b1_l2_screening_jsonl=b1,
        layer5_l2_screening_jsonl=layer5,
    )
    return paths, rows


class DecisionRuleTest(unittest.TestCase):
    def test_confirmed_when_ci_lower_above_threshold(self) -> None:
        d = summ.decide([0.0201, 0.10])
        self.assertEqual(d["outcome"], summ.CUTPOINT_EFFECT_CONFIRMED)

    def test_confirmed_threshold_is_strict(self) -> None:
        d = summ.decide([0.02, 0.10])
        self.assertEqual(d["outcome"], summ.CUTPOINT_EFFECT_NOT_CONFIRMED)

    def test_not_confirmed_when_ci_lower_below_or_equal(self) -> None:
        for ci in ([-0.05, 0.01], [0.0, 0.03], [-0.10, -0.02]):
            with self.subTest(ci=ci):
                d = summ.decide(ci)
                self.assertEqual(d["outcome"], summ.CUTPOINT_EFFECT_NOT_CONFIRMED)

    def test_decision_uses_primary_metric_only_and_ignores_dense(self) -> None:
        d = summ.decide([0.03, 0.04])
        self.assertTrue(d["uses_only_primary_metric"])
        self.assertFalse(d["dense_drives_decision"])
        self.assertEqual(d["dense_role"], summ.DENSE_ROLE)
        self.assertEqual(d["metric"], summ.PRIMARY_METRIC)
        self.assertEqual(d["threshold"], 0.02)
        self.assertEqual(d["ordered_rules"], list(summ.DECISION_ORDER))


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

    def test_grid_signature_tuple(self) -> None:
        sig = summ.grid_signature(
            {
                "image_grid_thw": [1, 4, 4],
                "full_patch_tokens": 16,
                "full_llm_visual_tokens": 4,
            }
        )
        self.assertEqual(sig, ((1, 4, 4), 16, 4))


class ValidationTest(unittest.TestCase):
    def _indices(self, rows=None):
        rows = rows or _synthetic_rows()
        return {name: summ.index_unique(rows[name], name) for name in rows}

    def _validate(self, idx):
        return summ.validate_arms(
            screening_index=idx["screening"],
            dense_index=idx["dense"],
            b1_index=idx["b1"],
            layer5_index=idx["layer5"],
            expected_screening_count=3,
            expected_full_count=None,
        )

    def test_happy_path(self) -> None:
        v = self._validate(self._indices())
        self.assertTrue(v["sample_ids_identical"])
        self.assertTrue(v["all_score_mode_l2"])
        self.assertTrue(v["all_requested_budget_96"])
        self.assertTrue(v["layer5_prune_after_layer_all_5"])
        self.assertTrue(v["b1_prune_after_layer_absent_or_0"])
        self.assertTrue(v["input_grid_consistent_b1_vs_layer5"])

    def test_b1_explicit_layer0_is_accepted(self) -> None:
        rows = _synthetic_rows()
        for row in rows["b1"]:
            row["prune_after_layer"] = 0
        self._validate(self._indices(rows))

    def test_wrong_score_mode_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"][1]["score_mode"] = "integer"
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_wrong_budget_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"][0]["requested_budget"] = 64
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_non_bypass_output_group_count_not_96_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"][2]["output_group_count"] = 95
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_identity_bypass_small_output_group_count_is_accepted(self) -> None:
        rows = _synthetic_rows()
        rows["b1"][0]["identity_bypass"] = True
        rows["b1"][0]["output_group_count"] = 16
        rows["layer5"][0]["identity_bypass"] = True
        rows["layer5"][0]["output_group_count"] = 16
        self._validate(self._indices(rows))

    def test_layer5_missing_cutpoint_fails(self) -> None:
        rows = _synthetic_rows()
        del rows["layer5"][0]["prune_after_layer"]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_layer5_wrong_cutpoint_fails(self) -> None:
        for cut in (0, 4, 6):
            with self.subTest(cut=cut):
                rows = _synthetic_rows()
                rows["layer5"][0]["prune_after_layer"] = cut
                with self.assertRaises(summ.InputContractError):
                    self._validate(self._indices(rows))

    def test_b1_cutpoint_5_fails(self) -> None:
        rows = _synthetic_rows()
        rows["b1"][0]["prune_after_layer"] = 5
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_missing_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"] = rows["layer5"][:2]
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_extra_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"].append(dict(rows["layer5"][0], sample_id="extra"))
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_grid_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["layer5"][2]["full_patch_tokens"] = 999
        with self.assertRaises(summ.InputContractError):
            self._validate(self._indices(rows))

    def test_wrong_screening_count_fails(self) -> None:
        idx = self._indices()
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                b1_index=idx["b1"],
                layer5_index=idx["layer5"],
                expected_screening_count=100,
                expected_full_count=None,
            )


class MetricNamingTest(unittest.TestCase):
    def _joined(self):
        rows = _synthetic_rows()
        screening = summ.index_unique(rows["screening"], "screening")
        dense = summ.index_unique(rows["dense"], "dense")
        b1 = summ.index_unique(rows["b1"], "b1")
        layer5 = summ.index_unique(rows["layer5"], "layer5")
        return summ.build_joined(
            list(screening),
            screening,
            {"dense": dense, "b1": b1, "layer5": layer5},
        )

    def test_arm_metrics_primary_and_secondary(self) -> None:
        metrics = summ.arm_metrics(self._joined(), "dense", "dense")
        self.assertEqual(metrics["primary_metric"], summ.PRIMARY_METRIC)
        self.assertAlmostEqual(metrics["cv_bench_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["unweighted_sample_accuracy"], 1.0)
        # The simple mean must never be exposed under a plain "accuracy" key.
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
            self.assertEqual(set(report["metrics"]), {"dense", "b1", "layer5"})
            self.assertEqual(
                set(report["comparisons"]),
                {"layer5_minus_b1", "layer5_minus_dense", "b1_minus_dense"},
            )
            for comp in report["comparisons"].values():
                self.assertEqual(comp["metric"], summ.PRIMARY_METRIC)
                self.assertEqual(comp["seed"], summ.BOOTSTRAP_SEED)
                self.assertEqual(comp["repeats"], summ.BOOTSTRAP_REPEATS)
                self.assertEqual(comp["stratified_by"], "task::source_dataset")
            # dense is reference-only and never drives the decision.
            self.assertEqual(report["arms"]["dense"]["role"], summ.DENSE_ROLE)
            self.assertFalse(report["decision"]["dense_drives_decision"])
            self.assertTrue(
                report["config_pin_verification"]["pins_verified"]
            )
            md = summ.render_markdown(report)
            self.assertIn("screening100", md)
            self.assertIn("CV-Bench 主分", md)
            self.assertIn("unweighted_sample_accuracy", md)
            self.assertIn("不得写作 CV-Bench 主分", md)
            self.assertIn("NOT a formal full2638", md)
            self.assertIn(summ.CUTPOINT_EFFECT_CONFIRMED, md)

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
                            "b1_l2_screening_jsonl": {
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

    def test_matching_pin_passes_and_null_layer5_pin_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            b1_sha = hashlib.sha256(paths.b1_l2_screening_jsonl.read_bytes()).hexdigest()
            paths.preregistered.write_text(
                json.dumps(
                    {
                        "status": "PREREGISTERED_IMMUTABLE",
                        "inputs_sha256": {
                            "b1_l2_screening_jsonl": {
                                "path": "b1.jsonl",
                                "sha256": b1_sha,
                            },
                            "layer5_l2_screening_jsonl": {
                                "path": "layer5.jsonl",
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
            paths.layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths, expected_screening_count=3, expected_full_count=None
                )


class FailClosedTest(unittest.TestCase):
    def test_missing_layer5_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.layer5_l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths,
                    expected_screening_count=3,
                    expected_full_count=None,
                    verify_pins=False,
                )

    def test_missing_b1_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.b1_l2_screening_jsonl.unlink()
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

    def test_missing_dense_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.dense_full_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(
                    paths,
                    expected_screening_count=3,
                    expected_full_count=None,
                    verify_pins=False,
                )

    def test_cli_returns_two_and_writes_nothing_when_layer5_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            paths.layer5_l2_screening_jsonl.unlink()
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
                / "2026-10-04-layer5-vs-b1-l2-screening100.preregistered.json"
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
        self.assertFalse(cfg["b1_trained_wv_used_for_layer5"])
        self.assertEqual(cfg["cut_points"], {"b1": 0, "layer5": 5})
        self.assertEqual(cfg["manifest"], "datasets/cv_bench_screening100.jsonl")

    def test_decision_rule_threshold_and_outcomes(self) -> None:
        rule = self.config["decision_rule"]
        self.assertEqual(rule["threshold"], 0.02)
        outcomes = [item["outcome"] for item in rule["ordered_rules"]]
        self.assertEqual(
            outcomes,
            [
                summ.CUTPOINT_EFFECT_CONFIRMED,
                summ.CUTPOINT_EFFECT_NOT_CONFIRMED,
            ],
        )
        self.assertIn("reference only", rule["dense_role"])

    def test_bootstrap_pins(self) -> None:
        boot = self.config["bootstrap"]
        self.assertEqual(boot["seed"], summ.BOOTSTRAP_SEED)
        self.assertEqual(boot["repeats"], summ.BOOTSTRAP_REPEATS)
        self.assertEqual(
            boot["method"], "paired bootstrap, stratified by task::source_dataset"
        )

    def test_b1_arm_is_the_frozen_existing_file(self) -> None:
        b1 = self.config["arms"]["b1_l2"]
        self.assertEqual(b1["sha256"], B1_SHA256)
        self.assertEqual(b1["required_score_mode"], "l2")
        self.assertEqual(b1["required_requested_budget"], 96)
        self.assertEqual(b1["required_prune_after_layer"], 0)
        self.assertTrue(b1["prune_after_layer_may_be_absent"])
        self.assertEqual(
            b1["path"],
            "artifacts/token_selector/2026-10-04-b1-l2-diagnostic/b1_l2_k96_screening100.jsonl",
        )

    def test_layer5_arm_is_pending_layer5_l2(self) -> None:
        layer5 = self.config["arms"]["layer5_l2"]
        self.assertIsNone(layer5["sha256"])
        self.assertEqual(layer5["status"], "TO_BE_GENERATED_AFTER_PREREGISTRATION")
        self.assertEqual(layer5["required_score_mode"], "l2")
        self.assertEqual(layer5["required_requested_budget"], 96)
        self.assertEqual(layer5["required_prune_after_layer"], 5)
        self.assertTrue(layer5["forbids_b1_trained_wv"])
        self.assertEqual(
            layer5["runner"],
            "evaluation/run_hf_hardware_regular_layer5_l2.py",
        )
        self.assertEqual(
            layer5["path"],
            "artifacts/token_selector/2026-10-04-layer5-l2-diagnostic/layer5_l2_k96_screening100.jsonl",
        )

    def test_dense_is_reference_only(self) -> None:
        self.assertIn("REFERENCE ONLY", self.config["arms"]["dense"]["role"])

    def test_report_paths(self) -> None:
        self.assertEqual(
            self.config["report_paths"]["json"],
            "reports/cv_bench/2026-10-04-layer5-vs-b1-l2-screening100.json",
        )
        self.assertEqual(
            self.config["report_paths"]["md"],
            "reports/cv_bench/2026-10-04-layer5-vs-b1-l2-screening100.md",
        )

    def test_pinned_input_hashes_match_files(self) -> None:
        pins = dict(self.config["inputs_sha256"])
        pins["datasets/cv_bench_screening100.jsonl"] = self.config["dataset_pins"][
            "screening_manifest"
        ]["sha256"]
        pins["datasets/cv_bench_full2638.jsonl"] = self.config["dataset_pins"][
            "full_manifest"
        ]["sha256"]
        pins["evaluation/results/cv_bench_l2_0875_full2638_20260916/shallow_r1000.jsonl"] = self.config[
            "arms"
        ]["dense"]["sha256"]
        pins["artifacts/token_selector/2026-10-04-b1-l2-diagnostic/b1_l2_k96_screening100.jsonl"] = self.config[
            "arms"
        ]["b1_l2"]["sha256"]
        for name, spec in pins.items():
            expected = spec.get("sha256") if isinstance(spec, dict) else spec
            if expected is None:
                # The layer5 artifact is generated after preregistration.
                continue
            relative = spec.get("path") if isinstance(spec, dict) else name
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


class RealFrozenSubsetTest(unittest.TestCase):
    """Read-only checks against the real frozen screening100 B1/dense subsets.

    The layer5 artifact does not exist yet, so these tests only exercise the
    dense and B1 arms and the paired bootstrap between them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.screening = summ.index_unique(
            summ.load_rows(summ.SCREENING_MANIFEST, "screening"), "screening"
        )
        cls.dense = summ.index_unique(
            summ.load_rows(summ.DENSE_FULL_JSONL, "dense"), "dense"
        )
        cls.b1 = summ.index_unique(
            summ.load_rows(summ.B1_L2_SCREENING_JSONL, "b1"), "b1"
        )
        cls.joined = summ.build_joined(
            list(cls.screening),
            cls.screening,
            {"dense": cls.dense, "b1": cls.b1},
        )

    def test_screening_has_100_unique_ids(self) -> None:
        self.assertEqual(len(self.screening), 100)

    def test_every_screening_id_present_in_both_full_sets(self) -> None:
        ids = set(self.screening)
        self.assertTrue(ids.issubset(self.dense))
        self.assertTrue(ids.issubset(self.b1))

    def test_b1_and_dense_official_scores(self) -> None:
        dense = summ.arm_metrics(self.joined, "dense", "dense")
        b1 = summ.arm_metrics(self.joined, "b1", "b1")
        self.assertAlmostEqual(
            dense["cv_bench_accuracy"], DENSE_SCREENING100_CV_BENCH_ACCURACY, places=12
        )
        self.assertAlmostEqual(
            b1["cv_bench_accuracy"], B1_SCREENING100_CV_BENCH_ACCURACY, places=12
        )
        self.assertAlmostEqual(
            dense["unweighted_sample_accuracy"], DENSE_SCREENING100_UNWEIGHTED, places=12
        )
        self.assertAlmostEqual(
            b1["unweighted_sample_accuracy"], B1_SCREENING100_UNWEIGHTED, places=12
        )
        # The two metrics genuinely differ, so mixing them is detectable.
        self.assertGreater(
            abs(b1["cv_bench_accuracy"] - b1["unweighted_sample_accuracy"]), 1e-6
        )

    def test_b1_records_are_l2_k96_and_have_no_cutpoint(self) -> None:
        for row in self.b1.values():
            self.assertEqual(row["score_mode"], "l2")
            self.assertEqual(int(row["requested_budget"]), 96)
            self.assertNotIn("prune_after_layer", row)

    def test_b1_minus_dense_bootstrap_ci(self) -> None:
        comp = summ.paired_comparison(self.joined, "b1", "dense")
        self.assertEqual(comp["seed"], 20261004)
        self.assertEqual(comp["repeats"], 5000)
        self.assertAlmostEqual(comp["delta_ci95"][0], B1_MINUS_DENSE_CI[0], places=12)
        self.assertAlmostEqual(comp["delta_ci95"][1], B1_MINUS_DENSE_CI[1], places=12)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
