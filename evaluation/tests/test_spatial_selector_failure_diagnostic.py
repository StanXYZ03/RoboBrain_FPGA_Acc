#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.spatial_selector_failure_diagnostic`.

Covers the contract's required test surface:

* internal test NPZ isolation (no test cache is ever opened/hashed/scored),
* the 7 frozen mode families / 8 lambda sweep values,
* corrected selected-set NDCG for anchor modes,
* three-seed per-image median aggregation,
* per-task and per-size / per-grid summarisation,
* every frozen diagnosis rule,
* preregistration hash pinning,
* the JSON/CSV report schema.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation import spatial_selector_failure_diagnostic as diag
from evaluation.prompt_projection_distillation import (
    ndcg_at_k,
    ndcg_for_selected_indices,
    save_projection_npz,
)
from evaluation.train_pairwise_prompt_selector import sha256_file


FAMILIES = {
    "l2_plain",
    "l2_anchor",
    "semantic_plain",
    "semantic_anchor",
    "combined_anchor",
    "lambda_sweep_anchor",
    "oracle_anchor",
}


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _cache_arrays(group_count: int = 81, tokens: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    group_mean = rng.normal(size=(group_count, 1024)).astype(np.float64)
    prompt = rng.normal(size=(tokens, 2560)).astype(np.float64)
    raw = rng.random(group_count).astype(np.float64)
    normalized = (raw - raw.min()) / (raw.max() - raw.min())
    soft = np.exp(normalized)
    soft = soft / soft.sum()
    return {
        "group_mean": group_mean,
        "prompt_embeddings": prompt,
        "teacher_scores": raw,
        "teacher_scores_normalized": normalized,
        "teacher_soft_target": soft,
    }


class ModeRegistryTests(unittest.TestCase):
    def test_seven_families_and_eight_lambdas(self):
        specs = diag.mode_specs()
        families = {spec["family"] for spec in specs}
        self.assertEqual(families, FAMILIES)
        self.assertEqual(len(specs), 13)
        self.assertEqual(len(diag.LAMBDAS), 8)
        self.assertEqual(
            list(diag.LAMBDAS), [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        )
        ids = [spec["mode_id"] for spec in specs]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(diag.combined_mode_id(0.5), "combined_anchor_lambda_0.50")
        self.assertIn("combined_anchor_lambda_0.50", ids)
        self.assertIn("oracle_anchor", ids)
        combined = [s for s in specs if s["family"] == "combined_anchor"]
        sweep = [s for s in specs if s["family"] == "lambda_sweep_anchor"]
        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["lambda"], 0.5)
        self.assertEqual(len(sweep), 7)
        # The sweep family carries the other seven frozen lambda values (it
        # includes lambda=0, which by construction ranks like the L2 baseline).
        self.assertEqual(
            {spec["lambda"] for spec in sweep}, set(diag.LAMBDAS) - {0.5}
        )


class SelectedSetNdcgTests(unittest.TestCase):
    def _record(self):
        group_count = 81
        grid_h = grid_w = 9
        target = np.zeros(group_count)
        # Relevance sits on groups the anchor rule forces in (54/57/60) and on
        # groups both rules include, so restricting DCG to the anchor-selected set
        # produces a genuinely different NDCG from the unrestricted top-K NDCG.
        target[[0, 1, 40, 54, 57, 60]] = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        # Descending scores with one low region so the fixed 3x3 anchor rule must
        # force that region's anchor into the selection, which the plain top-K
        # would never do. This makes the corrected selected-set NDCG observable.
        l2 = 1000.0 - np.arange(group_count, dtype=float)
        l2[[60, 61, 62, 69, 70, 71, 78, 79, 80]] = -1000.0
        semantic = np.linspace(0.0, 1.0, group_count)
        return target, l2, semantic, grid_h, grid_w

    def test_plain_ndcg_matches_ndcg_at_k(self):
        target, l2, semantic, gh, gw = self._record()
        metrics = diag.record_mode_metrics(
            semantic_scores=semantic,
            l2_scores=l2,
            target=target,
            grid_h=gh,
            grid_w=gw,
            budgets=(48,),
            specs=diag.mode_specs(),
        )
        selected = diag.topk_indices(l2, 48)
        expected = ndcg_at_k(l2, target, 48)
        self.assertAlmostEqual(
            metrics["l2_plain"][48]["ndcg"], expected, places=12
        )
        self.assertEqual(metrics["l2_plain"][48]["selected_count"], selected.size)

    def test_anchor_ndcg_restricted_to_anchor_selected_set(self):
        target, l2, semantic, gh, gw = self._record()
        metrics = diag.record_mode_metrics(
            semantic_scores=semantic,
            l2_scores=l2,
            target=target,
            grid_h=gh,
            grid_w=gw,
            budgets=(48,),
            specs=diag.mode_specs(),
        )
        anchor_selected = diag.select_anchor_groups(l2, gh, gw, 48)
        expected = ndcg_for_selected_indices(anchor_selected, l2, target, 48)
        self.assertAlmostEqual(metrics["l2_anchor"][48]["ndcg"], expected, places=12)
        # The anchor rule must pick a different set from a plain top-K here, so
        # the corrected selected-set NDCG is genuinely different from the naive
        # un-restricted NDCG (this is the regression the helper exists for).
        self.assertNotEqual(set(anchor_selected.tolist()), set(diag.topk_indices(l2, 48).tolist()))
        self.assertNotAlmostEqual(expected, ndcg_at_k(l2, target, 48), places=12)

    def test_jaccard_and_overlap_against_l2_anchor(self):
        target, l2, semantic, gh, gw = self._record()
        metrics = diag.record_mode_metrics(
            semantic_scores=semantic,
            l2_scores=l2,
            target=target,
            grid_h=gh,
            grid_w=gw,
            budgets=(48,),
            specs=diag.mode_specs(),
        )
        self.assertAlmostEqual(metrics["l2_anchor"][48]["selected_jaccard_vs_l2_anchor"], 1.0)
        self.assertEqual(
            metrics["l2_anchor"][48]["selected_overlap_count_vs_l2_anchor"],
            metrics["l2_anchor"][48]["selected_count"],
        )
        self.assertAlmostEqual(metrics["l2_anchor"][48]["delta_mass_vs_l2_anchor"], 0.0)


class AggregationTests(unittest.TestCase):
    def test_per_image_median_across_three_seeds(self):
        values = {20261003: [1.0, 5.0], 20261004: [2.0, 4.0], 20261005: [3.0, 6.0]}
        self.assertEqual(diag.per_image_median(values), [2.0, 5.0])

    def test_per_image_median_rejects_ragged_series(self):
        with self.assertRaises(Exception):
            diag.per_image_median({1: [1.0], 2: [1.0, 2.0]})

    def test_summarize_better_equal_worse(self):
        values = {"a": 0.3, "b": 0.1, "c": 0.2}
        reference = {"a": 0.2, "b": 0.1, "c": 0.3}
        out = diag.summarize(values, reference=reference)
        self.assertEqual(out["n_images"], 3)
        self.assertEqual(out["better"], 1)
        self.assertEqual(out["equal"], 1)
        self.assertEqual(out["worse"], 1)
        self.assertAlmostEqual(out["mean"], 0.2)
        self.assertAlmostEqual(out["median"], 0.2)
        self.assertLessEqual(out["ci_low"], out["mean"])
        self.assertGreaterEqual(out["ci_high"], out["mean"])

    def test_series_delta(self):
        delta = diag.series_delta({"a": 0.5, "b": 0.2}, {"a": 0.1, "b": 0.2})
        self.assertAlmostEqual(delta["a"], 0.4)
        self.assertAlmostEqual(delta["b"], 0.0)


def _build_diagnosis_inputs(
    *,
    overall=None,
    grounding=None,
    pointing=None,
    images=("i1", "i2"),
):
    overall = overall or {}
    grounding = grounding or {}
    pointing = pointing or {}
    budget = 48
    series = {}
    series_task = {}
    for spec in diag.mode_specs():
        mode = spec["mode_id"]
        base = overall.get(mode, 0.5)
        base_values = base if isinstance(base, (list, tuple)) else [base] * len(images)
        series[mode] = {
            budget: {"retained_mass": dict(zip(images, [float(v) for v in base_values]))}
        }
        ground = grounding.get(mode, base_values)
        ground_values = ground if isinstance(ground, (list, tuple)) else [ground] * len(images)
        point = pointing.get(mode, base_values)
        point_values = point if isinstance(point, (list, tuple)) else [point] * len(images)
        series_task[mode] = {
            budget: {
                "retained_mass": {
                    "grounding": dict(zip(images, [float(v) for v in ground_values])),
                    "pointing": dict(zip(images, [float(v) for v in point_values])),
                }
            }
        }
    return series, series_task


class DiagnosisRuleTests(unittest.TestCase):
    def _diagnose(self, **kwargs):
        series, series_task = _build_diagnosis_inputs(**kwargs)
        images = ["i1", "i2"]
        task_images = {"grounding": images, "pointing": images}
        return diag.diagnose(
            series=series, series_task=series_task, images=images, task_images=task_images
        )

    def test_fusion_damage(self):
        out = self._diagnose(
            overall={
                "semantic_anchor": 0.6,
                diag.combined_mode_id(0.5): 0.4,
            }
        )
        self.assertEqual(out["labels"], ["FUSION_DAMAGE"])

    def test_head_signal_weak(self):
        out = self._diagnose(overall={"semantic_anchor": 0.4})
        self.assertEqual(out["labels"], ["HEAD_SIGNAL_WEAK"])

    def test_anchor_budget_damage(self):
        out = self._diagnose(
            overall={
                "semantic_plain": 0.6,
                "l2_plain": 0.5,
                "semantic_anchor": 0.4,
                diag.combined_mode_id(0.5): 0.55,
            }
        )
        self.assertEqual(out["labels"], ["ANCHOR_BUDGET_DAMAGE"])

    def test_task_specific(self):
        out = self._diagnose(
            overall={
                "semantic_anchor": 0.5,
                diag.combined_mode_id(0.5): 0.5,
            },
            grounding={diag.combined_mode_id(0.5): 0.6},
            pointing={diag.combined_mode_id(0.5): 0.4},
        )
        self.assertEqual(out["labels"], ["TASK_SPECIFIC"])
        self.assertEqual(out["evaluations"]["TASK_SPECIFIC"]["grounding_sign"], 1)
        self.assertEqual(out["evaluations"]["TASK_SPECIFIC"]["pointing_sign"], -1)

    def test_headroom_present(self):
        out = self._diagnose(
            overall={
                "oracle_anchor": 0.7,
                "l2_anchor": 0.5,
                "semantic_anchor": 0.6,
                diag.combined_mode_id(0.5): 0.55,
            }
        )
        self.assertEqual(out["labels"], ["HEADROOM_PRESENT"])
        self.assertGreater(out["evaluations"]["HEADROOM_PRESENT"]["d_oracle_anchor"], 0.10)

    def test_no_label_when_deltas_are_zero(self):
        out = self._diagnose()
        self.assertEqual(out["labels"], ["HEAD_SIGNAL_WEAK"])


class PreregPinTests(unittest.TestCase):
    def test_pins_unchanged_then_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "pinned.txt"
            target.write_text("frozen", encoding="utf-8")
            expected = sha256_file(target)
            prereg = {
                "frozen_inputs": {
                    "thing": {"path": "pinned.txt", "sha256": expected}
                }
            }
            report = diag.verify_prereg_pins(prereg, root=root)
            self.assertTrue(report["pins_unchanged"])
            self.assertEqual(report["mismatches"], [])
            target.write_text("changed", encoding="utf-8")
            with self.assertRaises(Exception):
                diag.verify_prereg_pins(prereg, root=root, strict=True)
            lenient = diag.verify_prereg_pins(prereg, root=root, strict=False)
            self.assertFalse(lenient["pins_unchanged"])

    def test_verify_prereg_pins_hashes_but_never_parses_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "val_a.npz"
            raw.write_bytes(b"not-a-real-npz")
            expected = sha256_file(raw)
            prereg = {
                "frozen_inputs": {
                    "val_npz_sha256": {"val_a.npz": expected}
                }
            }
            report = diag.verify_prereg_pins(prereg, root=root)
            self.assertTrue(report["pins_unchanged"])


class EndToEndIsolationTests(unittest.TestCase):
    GROUP_COUNT = 81
    GRID_H = 9
    GRID_W = 9
    TOKENS = 4

    def _fixture(self, root: Path):
        features = root / "artifacts" / "token_selector" / "prompt_projection_features"
        features.mkdir(parents=True, exist_ok=True)
        index_records = []
        manifest_records = []
        val_ids = []
        for image_index, image_name in enumerate(("imgA", "imgB"), start=1):
            image_sha = f"{image_index:064x}"
            for task in ("grounding", "pointing"):
                sample_id = f"val-{image_name}-{task}"
                npz_name = f"{sample_id}.npz"
                (features / npz_name).write_bytes(b"dummy-bytes")
                index_records.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "npz": npz_name,
                        "group_count": self.GROUP_COUNT,
                        "grid_h": self.GRID_H,
                        "grid_w": self.GRID_W,
                        "prompt_token_count": self.TOKENS,
                        "source_image_sha256": image_sha,
                        "npz_sha256": "0" * 64,
                    }
                )
                manifest_records.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "ground_truth": {
                            "bbox_1000": [10.0, 10.0, 50.0, 50.0],
                            "point_1000": [30.0, 30.0],
                        },
                    }
                )
                val_ids.append(sample_id)
        (features / "test_x.npz").write_bytes(b"dummy-test-bytes")
        index_records.append(
            {
                "sample_id": "test-1",
                "task": "grounding",
                "npz": "test_x.npz",
                "group_count": self.GROUP_COUNT,
                "grid_h": self.GRID_H,
                "grid_w": self.GRID_W,
                "prompt_token_count": self.TOKENS,
                "source_image_sha256": "f" * 64,
                "npz_sha256": "0" * 64,
            }
        )
        _write_jsonl(features / "index.jsonl", index_records)
        _write_jsonl(root / "datasets" / "manifest.jsonl", manifest_records)

        prereg_path = root / "configs" / "prereg.json"
        prereg_path.parent.mkdir(parents=True, exist_ok=True)
        prereg_path.write_text(
            json.dumps(
                {
                    "frozen_split": {
                        "seed": 1,
                        "ratios": {"train": 0.0, "val": 0.8, "test": 0.2},
                        "salt": "unit-test",
                    },
                    "frozen_inputs": {},
                }
            ),
            encoding="utf-8",
        )
        report_path = root / "reports" / "prereg-report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"config": {"sha256": sha256_file(prereg_path)}}),
            encoding="utf-8",
        )

        seed = 20261003
        ckpt_dir = (
            root
            / "artifacts"
            / "token_selector"
            / "spatial_student_trainval"
            / diag.BEST_ARM
            / str(seed)
        )
        rng = np.random.default_rng(11)
        save_projection_npz(
            ckpt_dir / "best.npz",
            rng.normal(size=(128, 2560)).astype(np.float32),
            rng.normal(size=(128, 1024)).astype(np.float32),
            {"checkpoint": "best"},
        )
        return {
            "features": features,
            "index_path": Path("artifacts/token_selector/prompt_projection_features/index.jsonl"),
            "manifest_path": Path("datasets/manifest.jsonl"),
            "artifacts_dir": Path("artifacts/token_selector/spatial_student_trainval"),
            "prereg_config_path": Path("configs/prereg.json"),
            "prereg_report_path": Path("reports/prereg-report.json"),
            "val_ids": val_ids,
            "seed": seed,
        }

    def test_test_npz_never_opened_and_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            opened_by_loader: list[str] = []
            arrays = _cache_arrays(self.GROUP_COUNT, self.TOKENS)

            def spy(path):
                opened_by_loader.append(str(path))
                return arrays

            def fake_split(index_records, *, seed, ratios, salt):
                return {"train": [], "val": list(fx["val_ids"]), "test": ["test-1"]}

            with mock.patch.object(diag, "split_sample_ids", side_effect=fake_split):
                result = diag.run_diagnostic(
                    root=root,
                    prereg_config_path=fx["prereg_config_path"],
                    prereg_report_path=fx["prereg_report_path"],
                    index_path=fx["index_path"],
                    manifest_path=fx["manifest_path"],
                    artifacts_dir=fx["artifacts_dir"],
                    features_dir=fx["features"],
                    seeds=(fx["seed"],),
                    budgets=(48,),
                    cache_loader=spy,
                    verify_pins=False,
                )
            bundle = result["bundle"]
            csv_rows = result["csv_rows"]

            # Isolation: only the four val caches were resolved/opened.
            self.assertEqual(len(opened_by_loader), 4)
            self.assertTrue(all("test_x.npz" not in path for path in opened_by_loader))
            self.assertEqual(bundle["isolation"]["test_npz_opened"], [])
            self.assertEqual(bundle["isolation"]["train_npz_opened"], [])
            self.assertTrue(bundle["isolation"]["opened_only_val_npz"])
            self.assertFalse(bundle["isolation"]["test_npz_resolved_hashed_or_opened"])
            self.assertFalse(bundle["isolation"]["restricted_val_report_files_parsed"])
            self.assertEqual(bundle["split_counts"]["val_records"], 4)
            self.assertEqual(bundle["split_counts"]["val_images"], 2)

            # CSV row schema and cardinality: seeds x records x budgets x modes.
            self.assertEqual(len(csv_rows), 1 * 4 * 1 * len(diag.mode_specs()))
            self.assertEqual(set(csv_rows[0]), set(diag.CSV_FIELDS))

            # JSON/MD/CSV writers produce a parseable, schema-complete report.
            files = diag.write_outputs(bundle, csv_rows, reports_dir=root / "reports_out")
            payload = json.loads(Path(files["json"]).read_text(encoding="utf-8"))
            for key in (
                "schema_version",
                "report_stem",
                "status",
                "exploratory",
                "provenance",
                "isolation",
                "split_counts",
                "config",
                "modes",
                "lambda_sweep",
                "diagnosis",
                "headroom",
                "size_strata",
                "limitations",
            ):
                self.assertIn(key, payload)
            self.assertIn("labels", payload["diagnosis"])
            self.assertEqual(len(payload["lambda_sweep"]), len(diag.LAMBDAS))
            for mode_id in (spec["mode_id"] for spec in diag.mode_specs()):
                self.assertIn(mode_id, payload["modes"])
                self.assertIn("overall", payload["modes"][mode_id])
                self.assertIn("k48", payload["modes"][mode_id]["overall"])
            with Path(files["csv"]).open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(diag.CSV_FIELDS))
                self.assertEqual(len(list(reader)), len(csv_rows))
            self.assertTrue(Path(files["md"]).read_text(encoding="utf-8").startswith("#"))

    def test_harness_payload_contains_decision_fields(self):
        primary = diag.combined_mode_id(0.5)

        def leaf(mass: float, delta: float):
            return {
                "k48": {
                    "retained_spatial_target_mass": {"mean": mass},
                    "delta_mass_vs_l2_anchor": {"mean": delta},
                }
            }

        payload = diag.harness_payload(
            {
                "status": "SPATIAL_SELECTOR_FAILURE_DIAGNOSED",
                "diagnosis": {"labels": ["HEADROOM_PRESENT"], "k48_image_level_mean_deltas": {}},
                "headroom": {"present": True},
                "isolation": {"test_npz_opened": []},
                "modes": {
                    primary: {
                        "overall": leaf(0.46, 0.25),
                        "grounding": leaf(0.4, 0.12),
                        "pointing": leaf(0.5, 0.37),
                    },
                    "l2_anchor": {"overall": leaf(0.21, 0.0)},
                },
                "lambda_sweep": {
                    "0.50": {
                        "combined_anchor_k48_overall": {
                            "retained_spatial_target_mass": {"mean": 0.46},
                            "delta_mass_vs_l2_anchor": {"mean": 0.25},
                        }
                    },
                    "1.00": {
                        "combined_anchor_k48_overall": {
                            "retained_spatial_target_mass": {"mean": 0.30},
                            "delta_mass_vs_l2_anchor": {"mean": 0.09},
                        }
                    },
                },
                "limitations": ["exploratory only"],
                "provenance": {
                    "prereg_config": {"unchanged": True},
                    "prereg_report": {"unchanged": True},
                    "pins": {"pins_unchanged": True},
                    "val_npz_aggregate_matches_prereg": True,
                },
            },
            {"json": "x.json"},
        )
        self.assertEqual(payload["diagnosis_labels"], ["HEADROOM_PRESENT"])
        self.assertTrue(payload["headroom_present"])
        self.assertTrue(payload["provenance"]["prereg_config_unchanged"])
        # Contract item 10 decision fields are present and derived.
        self.assertEqual(payload["task_stratification"]["grounding"][primary], 0.12)
        self.assertEqual(payload["task_stratification"]["pointing"][primary], 0.37)
        self.assertEqual(payload["lambda_sweep_best"]["lambda"], "0.50")
        self.assertAlmostEqual(payload["lambda_sweep_best"]["retained_mass_mean"], 0.46)
        self.assertEqual(payload["limitations"], ["exploratory only"])


class RealDataRegressionTests(unittest.TestCase):
    """End-to-end regression against the frozen val CSV and pinned inputs."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cfg = cls.root / "configs/spatial_selector_failure_diagnostic.preregistered.json"
        rep = (
            cls.root
            / "reports/token_selector/2026-10-03-spatial-selector-failure-diagnostic-preregistration.json"
        )
        ckpt = (
            cls.root
            / "artifacts/token_selector/spatial_student_trainval"
            / diag.BEST_ARM
            / str(diag.SEEDS[0])
            / "best.npz"
        )
        if not (cfg.is_file() and rep.is_file() and ckpt.is_file()):
            raise unittest.SkipTest("real diagnostic inputs unavailable")
        cls.result = diag.run_diagnostic(root=cls.root, verify_pins=True)

    def test_split_is_frozen_val_and_test_isolated(self):
        bundle = self.result["bundle"]
        self.assertEqual(bundle["split_counts"]["val_records"], 76)
        self.assertEqual(bundle["split_counts"]["val_images"], 38)
        self.assertTrue(bundle["isolation"]["opened_only_val_npz"])
        self.assertEqual(bundle["isolation"]["test_npz_opened"], [])
        self.assertEqual(bundle["isolation"]["train_npz_opened"], [])
        self.assertFalse(bundle["isolation"]["test_npz_resolved_hashed_or_opened"])

    def test_prereg_hashes_unchanged(self):
        bundle = self.result["bundle"]
        self.assertTrue(bundle["provenance"]["prereg_config"]["unchanged"])
        self.assertTrue(bundle["provenance"]["prereg_report"]["unchanged"])
        self.assertTrue(bundle["provenance"]["pins"]["pins_unchanged"])
        self.assertTrue(bundle["provenance"]["val_npz_aggregate_matches_prereg"])

    def test_three_seed_image_median_matches_frozen_csv(self):
        bundle = self.result["bundle"]
        csv_path = (
            self.root
            / "reports/token_selector/2026-10-03-spatial-student-trainval-val-samples.csv"
        )
        with csv_path.open(encoding="utf-8") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["arm"] == "B_SPATIAL_KL_PRIMARY"
            ]
        # Mirror the frozen aggregation independently: per seed average the two
        # task records of an image, median across seeds, then the mean image delta.
        selector_by_image: dict[str, dict[int, list[float]]] = {}
        anchor_by_image: dict[str, list[float]] = {}
        for row in rows:
            image = row["image_sha256"]
            selector_by_image.setdefault(image, {}).setdefault(
                int(row["seed"]), []
            ).append(float(row["selector_mass_k48"]))
            anchor_by_image.setdefault(image, []).append(
                float(row["l2_anchor_mass_k48"])
            )
        deltas = []
        for image, per_seed in selector_by_image.items():
            per_seed_mean = [float(np.mean(values)) for values in per_seed.values()]
            selector_median = float(np.median(per_seed_mean))
            anchor_mean = float(np.mean(anchor_by_image[image]))
            deltas.append(selector_median - anchor_mean)
        expected = float(np.mean(deltas))
        actual = bundle["diagnosis"]["k48_image_level_mean_deltas"][
            "combined_lambda_0.50_minus_l2_anchor"
        ]
        self.assertAlmostEqual(actual, expected, places=9)

    def test_diagnosis_labels_and_task_size_strata(self):
        bundle = self.result["bundle"]
        self.assertEqual(bundle["diagnosis"]["labels"], ["HEADROOM_PRESENT"])
        combo = bundle["modes"][diag.combined_mode_id(0.5)]
        for stratum in ("overall", "grounding", "pointing"):
            self.assertEqual(
                combo[stratum]["k48"]["retained_spatial_target_mass"]["n_images"], 38
            )
        self.assertTrue(any(key.startswith("group_count=") for key in combo))
        self.assertTrue(any(key.startswith("grid=") for key in combo))
        self.assertTrue(any(key.startswith("size_band=") for key in combo))
        self.assertEqual(len(bundle["lambda_sweep"]), 8)
        prior = bundle["prior_report_consistency"]
        self.assertTrue(prior["available"])
        self.assertFalse(prior["sign_consistent_with_stated_definition"])
        # The narrative interpretation must be derived from the frozen numbers,
        # never hand-written.
        dm = combo["overall"]["k48"]["delta_mass_vs_l2_anchor"]
        self.assertIn(
            f"{dm['better']}/{dm['n_images']}",
            bundle["interpretation"]["within_task_heterogeneity"],
        )
        self.assertIn(
            f"{dm['worse']}/{dm['n_images']}",
            bundle["interpretation"]["within_task_heterogeneity"],
        )
        self.assertIn(
            bundle["diagnosis"]["labels"][0], bundle["interpretation"]["summary"]
        )
        # The restricted prior report JSON is parsed only for the sign
        # cross-check; the isolation flags must say so truthfully.
        self.assertEqual(
            bundle["isolation"]["restricted_val_report_files_parsed"],
            prior["available"],
        )
        self.assertFalse(
            bundle["isolation"][
                "restricted_val_samples_csv_or_json_parsed_as_metric_source"
            ]
        )


if __name__ == "__main__":
    unittest.main()

