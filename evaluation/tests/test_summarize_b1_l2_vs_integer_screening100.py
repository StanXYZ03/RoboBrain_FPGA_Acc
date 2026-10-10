#!/usr/bin/env python3
"""CPU-only tests for the B1 L2 vs integer screening100 summariser.

Read-only against the frozen real artifacts: the tests never rewrite the raw
JSONL, the formal quality report or ``configs/evaluation_protocol.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import summarize_b1_l2_vs_integer_screening100 as summ  # noqa: E402
from evaluation.build_cv_bench_pruning_report import score  # noqa: E402

# Frozen real-artifact screening100 values (read-only, computed with the
# official scorer over the screening100 subset).
DENSE_SCREENING100_CV_BENCH_ACCURACY = 0.8916666666666666
CURRENT_SCREENING100_CV_BENCH_ACCURACY = 0.8125
DENSE_SCREENING100_UNWEIGHTED = 0.9
CURRENT_SCREENING100_UNWEIGHTED = 0.82
CURRENT_MINUS_DENSE_CI = [-0.15000000000000002, -0.008333333333333526]
CURRENT_RELATIVE_RETENTION = 0.9112149532710281


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
    return row


def _synthetic_rows() -> dict[str, list[dict]]:
    """Three samples, one per official source, so ``score`` is well defined."""

    screening = [
        _row(
            "s0",
            task="Count",
            source="ADE20K",
            source_dataset="ADE20K Validation Set",
            answer="(A)",
            output_text="",
        ),
        _row(
            "s1",
            task="Depth",
            source="COCO",
            source_dataset="COCO",
            answer="(B)",
            output_text="",
        ),
        _row(
            "s2",
            task="Distance",
            source="Omni3D",
            source_dataset="Omni3D",
            answer="(A)",
            output_text="",
        ),
    ]
    full = [
        _row(
            "s0",
            task="Count",
            source="ADE20K",
            source_dataset="ADE20K Validation Set",
            answer="(A)",
            output_text="",
        ),
        _row(
            "s1",
            task="Depth",
            source="COCO",
            source_dataset="COCO",
            answer="(B)",
            output_text="",
        ),
        _row(
            "s2",
            task="Distance",
            source="Omni3D",
            source_dataset="Omni3D",
            answer="(A)",
            output_text="",
        ),
    ]
    dense = [
        _row(
            "s0",
            task="Count",
            source="ADE20K",
            source_dataset="ADE20K Validation Set",
            answer="(A)",
            output_text="(A)",
        ),
        _row(
            "s1",
            task="Depth",
            source="COCO",
            source_dataset="COCO",
            answer="(B)",
            output_text="(B)",
        ),
        _row(
            "s2",
            task="Distance",
            source="Omni3D",
            source_dataset="Omni3D",
            answer="(A)",
            output_text="(A)",
        ),
    ]
    current = [
        _row(
            "s0",
            task="Count",
            source="ADE20K",
            source_dataset="ADE20K Validation Set",
            answer="(A)",
            output_text="(B)",
            image_grid_thw=[1, 16, 16],
            full_patch_tokens=256,
            full_llm_visual_tokens=64,
            score_mode="integer",
            requested_budget=96,
        ),
        _row(
            "s1",
            task="Depth",
            source="COCO",
            source_dataset="COCO",
            answer="(B)",
            output_text="(B)",
            image_grid_thw=[1, 32, 32],
            full_patch_tokens=1024,
            full_llm_visual_tokens=256,
            score_mode="integer",
            requested_budget=96,
        ),
        _row(
            "s2",
            task="Distance",
            source="Omni3D",
            source_dataset="Omni3D",
            answer="(A)",
            output_text="(A)",
            image_grid_thw=[1, 48, 48],
            full_patch_tokens=2304,
            full_llm_visual_tokens=576,
            score_mode="integer",
            requested_budget=96,
        ),
    ]
    l2 = [
        _row(
            "s0",
            task="Count",
            source="ADE20K",
            source_dataset="ADE20K Validation Set",
            answer="(A)",
            output_text="(A)",
            image_grid_thw=[1, 16, 16],
            full_patch_tokens=256,
            full_llm_visual_tokens=64,
            score_mode="l2",
            requested_budget=96,
        ),
        _row(
            "s1",
            task="Depth",
            source="COCO",
            source_dataset="COCO",
            answer="(B)",
            output_text="(B)",
            image_grid_thw=[1, 32, 32],
            full_patch_tokens=1024,
            full_llm_visual_tokens=256,
            score_mode="l2",
            requested_budget=96,
        ),
        _row(
            "s2",
            task="Distance",
            source="Omni3D",
            source_dataset="Omni3D",
            answer="(A)",
            output_text="(A)",
            image_grid_thw=[1, 48, 48],
            full_patch_tokens=2304,
            full_llm_visual_tokens=576,
            score_mode="l2",
            requested_budget=96,
        ),
    ]
    return {
        "screening": screening,
        "full": full,
        "dense": dense,
        "current": current,
        "l2": l2,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_workspace(tmp: Path, rows: dict[str, list[dict]] | None = None):
    rows = rows or _synthetic_rows()
    tmp.mkdir(parents=True, exist_ok=True)
    prereg = tmp / "prereg.json"
    prereg.write_text(json.dumps({"status": "PREREGISTERED_IMMUTABLE"}), encoding="utf-8")
    screening = tmp / "screening.jsonl"
    full = tmp / "full.jsonl"
    dense = tmp / "dense.jsonl"
    current = tmp / "current.jsonl"
    l2 = tmp / "l2.jsonl"
    _write_jsonl(screening, rows["screening"])
    _write_jsonl(full, rows["full"])
    _write_jsonl(dense, rows["dense"])
    _write_jsonl(current, rows["current"])
    _write_jsonl(l2, rows["l2"])
    paths = summ.InputPaths(
        preregistered=prereg,
        screening_manifest=screening,
        full_manifest=full,
        dense_full_jsonl=dense,
        current_full_jsonl=current,
        l2_screening_jsonl=l2,
    )
    return paths, rows


class DecisionRuleTest(unittest.TestCase):
    def test_ranker_defect_when_ci_lower_above_threshold(self) -> None:
        d = summ.decide([0.0201, 0.10], l2_relative_retention=0.90, current_relative_retention=0.88)
        self.assertEqual(d["outcome"], summ.RANKER_DEFECT_PRIMARY)

    def test_ranker_threshold_is_strict(self) -> None:
        d = summ.decide([0.02, 0.10], l2_relative_retention=0.90, current_relative_retention=0.88)
        self.assertNotEqual(d["outcome"], summ.RANKER_DEFECT_PRIMARY)

    def test_b1_premature_when_upper_below_threshold_and_both_retention_low(self) -> None:
        d = summ.decide([-0.10, 0.02], l2_relative_retention=0.91, current_relative_retention=0.90)
        self.assertEqual(d["outcome"], summ.B1_PREMATURE_PRUNING_PRIMARY)

    def test_b1_premature_requires_both_retentions_below(self) -> None:
        d = summ.decide([-0.10, 0.01], l2_relative_retention=0.93, current_relative_retention=0.90)
        self.assertEqual(d["outcome"], summ.MIXED_INCONCLUSIVE)

    def test_b1_premature_retention_threshold_is_strict(self) -> None:
        d = summ.decide([-0.10, 0.01], l2_relative_retention=0.92, current_relative_retention=0.90)
        self.assertEqual(d["outcome"], summ.MIXED_INCONCLUSIVE)

    def test_upper_threshold_is_inclusive(self) -> None:
        d = summ.decide([-0.10, 0.02], l2_relative_retention=0.91, current_relative_retention=0.91)
        self.assertEqual(d["outcome"], summ.B1_PREMATURE_PRUNING_PRIMARY)

    def test_mixed_when_ci_upper_above_threshold(self) -> None:
        d = summ.decide([-0.01, 0.03], l2_relative_retention=0.80, current_relative_retention=0.80)
        self.assertEqual(d["outcome"], summ.MIXED_INCONCLUSIVE)

    def test_decision_uses_primary_metric_only(self) -> None:
        d = summ.decide([0.03, 0.04], 0.9, 0.9)
        self.assertTrue(d["uses_only_primary_metric"])


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
    def _indices(self):
        rows = _synthetic_rows()
        return {name: summ.index_unique(rows[name], name) for name in rows}

    def test_happy_path(self) -> None:
        idx = self._indices()
        v = summ.validate_arms(
            screening_index=idx["screening"],
            dense_index=idx["dense"],
            current_index=idx["current"],
            l2_index=idx["l2"],
            expected_screening_count=3,
            expected_full_count=None,
        )
        self.assertTrue(v["sample_ids_identical"])
        self.assertTrue(v["l2_score_mode_all_l2"])
        self.assertTrue(v["l2_requested_budget_all_96"])
        self.assertTrue(v["input_grid_consistent_with_current"])

    def test_l2_wrong_score_mode_fails(self) -> None:
        rows = _synthetic_rows()
        rows["l2"][1]["score_mode"] = "integer"
        idx = {name: summ.index_unique(rows[name], name) for name in rows}
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_l2_wrong_budget_fails(self) -> None:
        rows = _synthetic_rows()
        rows["l2"][0]["requested_budget"] = 64
        idx = {name: summ.index_unique(rows[name], name) for name in rows}
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_l2_missing_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["l2"] = rows["l2"][:2]
        idx = {name: summ.index_unique(rows[name], name) for name in rows}
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_l2_extra_id_fails(self) -> None:
        rows = _synthetic_rows()
        rows["l2"].append(dict(rows["l2"][0], sample_id="extra"))
        idx = {name: summ.index_unique(rows[name], name) for name in rows}
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_grid_mismatch_fails(self) -> None:
        rows = _synthetic_rows()
        rows["l2"][2]["full_patch_tokens"] = 999
        idx = {name: summ.index_unique(rows[name], name) for name in rows}
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=3,
                expected_full_count=None,
            )

    def test_wrong_screening_count_fails(self) -> None:
        idx = self._indices()
        with self.assertRaises(summ.InputContractError):
            summ.validate_arms(
                screening_index=idx["screening"],
                dense_index=idx["dense"],
                current_index=idx["current"],
                l2_index=idx["l2"],
                expected_screening_count=100,
                expected_full_count=None,
            )


class MetricNamingTest(unittest.TestCase):
    def test_arm_metrics_primary_and_secondary(self) -> None:
        rows = _synthetic_rows()
        screening = summ.index_unique(rows["screening"], "screening")
        dense = summ.index_unique(rows["dense"], "dense")
        current = summ.index_unique(rows["current"], "current")
        l2 = summ.index_unique(rows["l2"], "l2")
        joined = summ.build_joined(
            list(screening), screening, {"dense": dense, "current": current, "l2": l2}
        )
        metrics = summ.arm_metrics(joined, "dense", "dense")
        self.assertEqual(metrics["primary_metric"], summ.PRIMARY_METRIC)
        self.assertAlmostEqual(metrics["cv_bench_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["unweighted_sample_accuracy"], 1.0)
        # The simple mean must never be exposed under a plain "accuracy" key.
        self.assertNotIn("accuracy", metrics)
        self.assertEqual(metrics["unweighted_metric"], summ.SECONDARY_METRIC)

    def test_join_recomputes_correctness_from_output_text(self) -> None:
        """Correctness must come from the output text, not a copied flag."""
        rows = _synthetic_rows()
        rows["dense"][0]["output_text"] = "(B)"  # wrong for answer (A)
        rows["dense"][0]["dense_correct"] = True  # stale flag must be ignored
        rows["dense"][0]["correct"] = True  # an even older alias, also ignored
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
                paths, expected_screening_count=3, expected_full_count=None
            )
            for arm in report["metrics"].values():
                self.assertIn("cv_bench_accuracy", arm)
                self.assertIn("unweighted_sample_accuracy", arm)
                self.assertNotIn("accuracy", arm)
                for name in summ.FORBIDDEN_PRIMARY_METRICS:
                    self.assertNotIn(name, arm)
            # Forbidden names may appear only in the explicit forbidden list.
            self.assertEqual(
                tuple(report["forbidden_primary_metrics"]),
                summ.FORBIDDEN_PRIMARY_METRICS,
            )


class SyntheticReportTest(unittest.TestCase):
    def test_build_report_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            report = summ.build_report(
                paths, expected_screening_count=3, expected_full_count=None
            )
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["formal_full2638_conclusion"])
            self.assertEqual(report["primary_metric"]["identifier"], summ.PRIMARY_METRIC)
            self.assertEqual(
                set(report["metrics"]), {"dense", "current", "l2"}
            )
            self.assertEqual(
                set(report["comparisons"]),
                {"l2_minus_current", "l2_minus_dense", "current_minus_dense"},
            )
            for name, comp in report["comparisons"].items():
                self.assertEqual(comp["metric"], summ.PRIMARY_METRIC)
                self.assertEqual(comp["seed"], summ.BOOTSTRAP_SEED)
                self.assertEqual(comp["repeats"], summ.BOOTSTRAP_REPEATS)
                self.assertEqual(comp["stratified_by"], "task::source_dataset")
            md = summ.render_markdown(report)
            self.assertIn("screening100", md)
            self.assertIn("CV-Bench 主分", md)
            self.assertIn("unweighted_sample_accuracy", md)
            self.assertIn("不得写作 CV-Bench 主分", md)
            self.assertIn("NOT a formal full2638", md)

    def test_build_report_hashes_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            report = summ.build_report(
                paths, expected_screening_count=3, expected_full_count=None
            )
            for name, item in report["inputs_sha256"].items():
                self.assertEqual(
                    item["sha256"], hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest(), name
                )


class FailClosedTest(unittest.TestCase):
    def test_missing_l2_input_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.l2_screening_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(paths, expected_screening_count=3, expected_full_count=None)

    def test_missing_prereg_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.preregistered.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(paths, expected_screening_count=3, expected_full_count=None)

    def test_missing_dense_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths, _rows = _synthetic_workspace(Path(tmp))
            paths.dense_full_jsonl.unlink()
            with self.assertRaises(summ.InputContractError):
                summ.build_report(paths, expected_screening_count=3, expected_full_count=None)

    def test_cli_returns_two_and_writes_nothing_when_l2_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths, _rows = _synthetic_workspace(tmp_path)
            paths.l2_screening_jsonl.unlink()
            import io
            from contextlib import redirect_stderr

            stderr = io.StringIO()
            original_default = summ.DEFAULT_PATHS
            summ.DEFAULT_PATHS = paths
            try:
                out_json = tmp_path / "out.json"
                out_md = tmp_path / "out.md"
                with redirect_stderr(stderr):
                    code = summ.main(["--output-json", str(out_json), "--output-md", str(out_md)])
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
            (ROOT / "configs" / "2026-10-04-b1-l2-vs-integer-screening100.preregistered.json").read_text(
                encoding="utf-8"
            )
        )

    def test_status_and_scope(self) -> None:
        self.assertEqual(self.config["status"], "PREREGISTERED_IMMUTABLE")
        self.assertTrue(self.config["diagnostic_only"])
        statement = self.config["diagnostic_scope"].lower()
        self.assertIn("not a formal full2638", statement)

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
        self.assertFalse(cfg["do_sample"])
        self.assertEqual(cfg["manifest"], "datasets/cv_bench_screening100.jsonl")

    def test_decision_rule_thresholds(self) -> None:
        rule = self.config["decision_rule"]
        self.assertEqual(rule["thresholds"]["ranker_defect_ci_lower"], 0.02)
        self.assertEqual(rule["thresholds"]["b1_premature_ci_upper"], 0.02)
        self.assertEqual(rule["thresholds"]["min_relative_retention"], 0.92)
        outcomes = [item["outcome"] for item in rule["ordered_rules"]]
        self.assertEqual(
            outcomes,
            [
                summ.RANKER_DEFECT_PRIMARY,
                summ.B1_PREMATURE_PRUNING_PRIMARY,
                summ.MIXED_INCONCLUSIVE,
            ],
        )

    def test_bootstrap_pins(self) -> None:
        boot = self.config["bootstrap"]
        self.assertEqual(boot["seed"], summ.BOOTSTRAP_SEED)
        self.assertEqual(boot["repeats"], summ.BOOTSTRAP_REPEATS)
        self.assertEqual(boot["method"], "paired bootstrap, stratified by task::source_dataset")

    def test_l2_arm_is_pending_and_pinned_to_runner(self) -> None:
        l2 = self.config["arms"]["l2"]
        self.assertIsNone(l2["sha256"])
        self.assertEqual(l2["status"], "TO_BE_GENERATED_AFTER_PREREGISTRATION")
        self.assertEqual(l2["required_score_mode"], "l2")
        self.assertEqual(l2["required_requested_budget"], 96)
        self.assertEqual(l2["path"], "artifacts/token_selector/2026-10-04-b1-l2-diagnostic/b1_l2_k96_screening100.jsonl")

    def test_report_paths(self) -> None:
        self.assertEqual(
            self.config["report_paths"]["json"],
            "reports/cv_bench/2026-10-04-b1-l2-vs-integer-screening100.json",
        )
        self.assertEqual(
            self.config["report_paths"]["md"],
            "reports/cv_bench/2026-10-04-b1-l2-vs-integer-screening100.md",
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
        pins["artifacts/token_selector/2026-10-04-current-selector-cvbench/current_selector_k96.jsonl"] = self.config[
            "arms"
        ]["current_integer"]["sha256"]
        for relative, expected in pins.items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


class RealFrozenSubsetTest(unittest.TestCase):
    """Read-only checks against the real frozen screening100 full2638 subsets.

    The L2 artifact does not exist yet, so these tests only exercise the dense
    and current arms and the paired bootstrap between them.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.screening = summ.index_unique(
            summ.load_rows(summ.SCREENING_MANIFEST, "screening"), "screening"
        )
        cls.dense = summ.index_unique(
            summ.load_rows(summ.DENSE_FULL_JSONL, "dense"), "dense"
        )
        cls.current = summ.index_unique(
            summ.load_rows(summ.CURRENT_FULL_JSONL, "current"), "current"
        )
        cls.joined = summ.build_joined(
            list(cls.screening),
            cls.screening,
            {"dense": cls.dense, "current": cls.current},
        )

    def test_screening_has_100_unique_ids(self) -> None:
        self.assertEqual(len(self.screening), 100)

    def test_every_screening_id_present_in_both_full_sets(self) -> None:
        ids = set(self.screening)
        self.assertTrue(ids.issubset(self.dense))
        self.assertTrue(ids.issubset(self.current))

    def test_dense_and_current_official_scores(self) -> None:
        dense = summ.arm_metrics(self.joined, "dense", "dense")
        current = summ.arm_metrics(self.joined, "current", "current")
        self.assertAlmostEqual(
            dense["cv_bench_accuracy"], DENSE_SCREENING100_CV_BENCH_ACCURACY, places=12
        )
        self.assertAlmostEqual(
            current["cv_bench_accuracy"],
            CURRENT_SCREENING100_CV_BENCH_ACCURACY,
            places=12,
        )
        self.assertAlmostEqual(
            dense["unweighted_sample_accuracy"], DENSE_SCREENING100_UNWEIGHTED, places=12
        )
        self.assertAlmostEqual(
            current["unweighted_sample_accuracy"],
            CURRENT_SCREENING100_UNWEIGHTED,
            places=12,
        )
        # The two metrics genuinely differ, so mixing them is detectable.
        self.assertGreater(
            abs(current["cv_bench_accuracy"] - current["unweighted_sample_accuracy"]),
            1e-6,
        )

    def test_current_minus_dense_bootstrap_ci(self) -> None:
        comp = summ.paired_comparison(self.joined, "current", "dense")
        self.assertEqual(comp["seed"], 20261004)
        self.assertEqual(comp["repeats"], 5000)
        self.assertAlmostEqual(comp["delta_ci95"][0], CURRENT_MINUS_DENSE_CI[0], places=12)
        self.assertAlmostEqual(comp["delta_ci95"][1], CURRENT_MINUS_DENSE_CI[1], places=12)

    def test_current_relative_retention(self) -> None:
        dense = score(list(self.joined), "dense_correct")["cv_bench_accuracy"]
        current = score(list(self.joined), "current_correct")["cv_bench_accuracy"]
        self.assertAlmostEqual(
            current / dense, CURRENT_RELATIVE_RETENTION, places=12
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
