#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.operational_prompt_utility_diagnostic`.

Covers the authority-prompt-required surface:

* correct/wrong prompt exchange,
* the identical deployment selection chain (combine + 3x3 anchor),
* P1/P2 direction, seed-sign rule and the exactly-two-test Holm family,
* three-seed per-image median aggregation,
* per-task / per-grid / per-group-count stratification,
* correct-vs-wrong selected Jaccard/overlap,
* internal test NPZ isolation,
* preregistration identity / pins unchanged,
* the prior contrastive report gate + SHA reference (S4),
* the JSON/MD/CSV report schema.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from evaluation import operational_prompt_utility_diagnostic as diag
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.prompt_projection_distillation import (
    jaccard,
    ndcg_for_selected_indices,
    save_projection_npz,
)
from evaluation.train_pairwise_prompt_selector import sha256_file
from evaluation.train_spatial_prompt_selector import _select_groups

GROUP_COUNT = 81
GRID_H = 9
GRID_W = 9
TOKENS = 4


def _write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _cache_arrays(seed: int = 0, tokens: int = TOKENS):
    rng = np.random.default_rng(seed)
    group_mean = rng.normal(size=(GROUP_COUNT, 1024)).astype(np.float64)
    prompt = rng.normal(size=(tokens, 2560)).astype(np.float64)
    raw = rng.random(GROUP_COUNT).astype(np.float64)
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


class RecordUtilityMetricTests(unittest.TestCase):
    def _record(self):
        rng = np.random.default_rng(7)
        target = np.zeros(GROUP_COUNT)
        target[[0, 1, 40, 54, 57, 60]] = [0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
        l2 = 1000.0 - np.arange(GROUP_COUNT, dtype=float)
        l2[[60, 61, 62, 69, 70, 71, 78, 79, 80]] = -1000.0
        correct = np.linspace(0.0, 1.0, GROUP_COUNT) + rng.normal(scale=1e-6, size=GROUP_COUNT)
        wrong = np.linspace(1.0, 0.0, GROUP_COUNT) + rng.normal(scale=1e-6, size=GROUP_COUNT)
        return target, l2, correct, wrong

    def test_correct_wrong_prompt_exchange(self):
        target, l2, correct, wrong = self._record()
        forward = diag.record_utility_metrics(
            correct_scores=correct, wrong_scores=wrong, l2_scores=l2,
            target=target, grid_h=GRID_H, grid_w=GRID_W, budgets=(48,),
        )[48]
        swapped = diag.record_utility_metrics(
            correct_scores=wrong, wrong_scores=correct, l2_scores=l2,
            target=target, grid_h=GRID_H, grid_w=GRID_W, budgets=(48,),
        )[48]
        self.assertAlmostEqual(forward["correct_mass"], swapped["wrong_mass"], places=12)
        self.assertAlmostEqual(forward["wrong_mass"], swapped["correct_mass"], places=12)
        self.assertAlmostEqual(forward["gap_mass"], -swapped["gap_mass"], places=12)
        self.assertAlmostEqual(forward["correct_ndcg"], swapped["wrong_ndcg"], places=12)
        self.assertAlmostEqual(forward["jaccard"], swapped["jaccard"], places=12)
        self.assertEqual(forward["overlap_count"], swapped["overlap_count"])

    def test_same_deployment_selection_chain(self):
        target, l2, correct, wrong = self._record()
        metrics = diag.record_utility_metrics(
            correct_scores=correct, wrong_scores=wrong, l2_scores=l2,
            target=target, grid_h=GRID_H, grid_w=GRID_W, budgets=(48,),
        )[48]
        combined = combine_scores(correct, l2, diag.LAMBDA_SEMANTIC)
        expected_selection = _select_groups(combined, GRID_H, GRID_W, 48)
        expected_ndcg = ndcg_for_selected_indices(
            expected_selection, combined, target, 48
        )
        self.assertEqual(metrics["correct_selected_count"], expected_selection.size)
        self.assertAlmostEqual(metrics["correct_ndcg"], expected_ndcg, places=12)

    def test_jaccard_and_overlap_match_sets(self):
        target, l2, correct, wrong = self._record()
        metrics = diag.record_utility_metrics(
            correct_scores=correct, wrong_scores=wrong, l2_scores=l2,
            target=target, grid_h=GRID_H, grid_w=GRID_W, budgets=(48,),
        )[48]
        correct_sel = _select_groups(
            combine_scores(correct, l2, diag.LAMBDA_SEMANTIC), GRID_H, GRID_W, 48
        )
        wrong_sel = _select_groups(
            combine_scores(wrong, l2, diag.LAMBDA_SEMANTIC), GRID_H, GRID_W, 48
        )
        self.assertAlmostEqual(
            metrics["jaccard"],
            jaccard(sorted(int(v) for v in correct_sel), sorted(int(v) for v in wrong_sel)),
            places=12,
        )
        self.assertEqual(
            metrics["overlap_count"],
            len(set(int(v) for v in correct_sel) & set(int(v) for v in wrong_sel)),
        )


class AggregationTests(unittest.TestCase):
    def test_per_image_median_across_three_seeds(self):
        values = {20261003: [1.0, 5.0], 20261004: [2.0, 4.0], 20261005: [3.0, 6.0]}
        self.assertEqual(diag.per_image_median(values), [2.0, 5.0])

    def test_per_image_median_rejects_ragged_series(self):
        with self.assertRaises(Exception):
            diag.per_image_median({1: [1.0], 2: [1.0, 2.0]})

    def test_summarize_better_equal_worse(self):
        values = {"a": 0.3, "b": -0.2, "c": 0.0}
        out = diag.summarize(values)
        self.assertEqual(out["n_images"], 3)
        self.assertEqual(out["better"], 1)
        self.assertEqual(out["worse"], 1)
        self.assertEqual(out["equal"], 1)
        self.assertAlmostEqual(out["mean"], 1.0 / 30.0, places=12)


class PrimaryHolmTests(unittest.TestCase):
    def _series(self, base):
        return {f"img{i}": float(base + 0.001 * ((i % 5) - 2)) for i in range(38)}

    def test_positive_series_passes_and_holm_family_is_two(self):
        p1 = self._series(0.10)
        p2 = self._series(0.05)
        out = diag.evaluate_primary_tests(
            p1, p2,
            p1_seed_means={20261003: 0.10, 20261004: 0.10, 20261005: 0.10},
            p2_seed_means={20261003: 0.05, 20261004: 0.06, 20261005: 0.04},
        )
        self.assertEqual(out["family_size"], 2)
        self.assertEqual(out["family"], [diag.P1_KEY, diag.P2_KEY])
        self.assertTrue(out["all_pass"])
        for key in (diag.P1_KEY, diag.P2_KEY):
            entry = out["tests"][key]
            self.assertTrue(entry["mean_positive"])
            self.assertTrue(entry["ci_low_positive"])
            self.assertTrue(entry["seed_sign_ok"])
            self.assertEqual(entry["positive_seed_count"], 3)
            self.assertLess(entry["holm_adjusted_p"], diag.ALPHA)
        # Holm step-down over a two-test family: the smallest raw p is doubled.
        raw = {k: out["tests"][k]["raw_p_value"] for k in (diag.P1_KEY, diag.P2_KEY)}
        smallest = min(raw, key=raw.get)
        self.assertAlmostEqual(
            out["tests"][smallest]["holm_adjusted_p"],
            min(1.0, 2.0 * raw[smallest]),
            places=12,
        )

    def test_negative_direction_fails(self):
        p1 = self._series(-0.10)
        p2 = self._series(-0.05)
        out = diag.evaluate_primary_tests(
            p1, p2,
            p1_seed_means={20261003: -0.1, 20261004: -0.1, 20261005: -0.1},
            p2_seed_means={20261003: -0.05, 20261004: -0.05, 20261005: -0.05},
        )
        self.assertFalse(out["all_pass"])
        self.assertFalse(out["tests"][diag.P1_KEY]["mean_positive"])
        self.assertFalse(out["tests"][diag.P2_KEY]["mean_positive"])

    def test_seed_sign_rule_requires_two_of_three(self):
        out = diag.evaluate_primary_tests(
            self._series(0.10), self._series(0.05),
            p1_seed_means={20261003: 0.10, 20261004: -0.01, 20261005: -0.01},
            p2_seed_means={20261003: 0.05, 20261004: 0.05, 20261005: 0.05},
        )
        self.assertFalse(out["tests"][diag.P1_KEY]["seed_sign_ok"])
        self.assertFalse(out["tests"][diag.P1_KEY]["pass"])
        self.assertTrue(out["tests"][diag.P2_KEY]["seed_sign_ok"])


class AuxiliaryGateTests(unittest.TestCase):
    def _series(self, value=0.2):
        return {f"img{i}": float(value) for i in range(3)}

    def test_s1_requires_both_tasks_positive(self):
        series = {48: {"gap_mass": {f"i{i}": 0.1 for i in range(3)},
                       "gap_ndcg": {f"i{i}": 0.1 for i in range(3)},
                       "jaccard": {f"i{i}": 0.2 for i in range(3)}}}
        task_series = {48: {"gap_mass": {
            "grounding": {f"i{i}": -0.1 for i in range(3)},
            "pointing": {f"i{i}": 0.2 for i in range(3)},
        }}}
        out = diag.evaluate_auxiliary_gates(
            primary_series=series, primary_task_series=task_series,
            images=[f"i{i}" for i in range(3)],
            task_images={"grounding": [f"i{i}" for i in range(3)],
                         "pointing": [f"i{i}" for i in range(3)]},
            prior={"all_pass": True, "gates": {}}, pins_unchanged=True,
            identity_consistent=True,
        )
        self.assertFalse(out["S1"]["pass"])
        self.assertFalse(out["all_pass"])

    def test_s3_threshold_and_s4_dependency(self):
        series = {48: {"gap_mass": {f"i{i}": 0.1 for i in range(3)},
                       "gap_ndcg": {f"i{i}": 0.1 for i in range(3)},
                       "jaccard": {f"i{i}": 0.99 for i in range(3)}}}
        task_series = {48: {"gap_mass": {
            "grounding": {f"i{i}": 0.1 for i in range(3)},
            "pointing": {f"i{i}": 0.1 for i in range(3)},
        }}}
        images = [f"i{i}" for i in range(3)]
        task_images = {"grounding": images, "pointing": images}
        out = diag.evaluate_auxiliary_gates(
            primary_series=series, primary_task_series=task_series,
            images=images, task_images=task_images,
            prior={"all_pass": True, "gates": {}}, pins_unchanged=True,
            identity_consistent=True,
        )
        self.assertFalse(out["S3"]["pass"])
        failing_s4 = diag.evaluate_auxiliary_gates(
            primary_series=series, primary_task_series=task_series,
            images=images, task_images=task_images,
            prior={"all_pass": False, "gates": {}}, pins_unchanged=True,
            identity_consistent=True,
        )
        self.assertFalse(failing_s4["S4"]["pass"])

    def test_decide_requires_both_families(self):
        self.assertEqual(
            diag.decide({"all_pass": True}, {"all_pass": True})["decision"],
            "OPERATIONAL_PROMPT_UTILITY_SUPPORTED",
        )
        self.assertEqual(
            diag.decide({"all_pass": True}, {"all_pass": False})["decision"],
            "OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED",
        )


class PriorGateReferenceTests(unittest.TestCase):
    def test_prior_reference_reads_only_gates_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "reports" / "prior.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "primary_gate": {
                            "G1_correct_alignment_non_inferiority": {"pass": True},
                            "G4_deployed_K48_utility": {"pass": True},
                            "G5_collapse_removed": {"pass": True},
                            "G7_hardware_contract": {"pass": False},
                        },
                        "decision": {"decision": "PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED"},
                    }
                ),
                encoding="utf-8",
            )
            out = diag.prior_gate_reference(
                root=root, prior_report_path=Path("reports/prior.json")
            )
            self.assertTrue(out["available"])
            self.assertEqual(out["prior_report_sha256"], sha256_file(path))
            self.assertFalse(out["all_pass"])
            self.assertFalse(out["gates"]["G7"]["pass"])
            self.assertTrue(out["gates"]["G1"]["pass"])

    def test_real_prior_report_gates_pass(self):
        root = Path(__file__).resolve().parents[2]
        if not (root / diag.PRIOR_REPORT_PATH).is_file():
            raise unittest.SkipTest("prior contrastive report unavailable")
        out = diag.prior_gate_reference(root=root)
        self.assertTrue(out["available"])
        for short in ("G1", "G4", "G5", "G7"):
            self.assertTrue(out["gates"][short]["pass"], short)
        self.assertTrue(out["all_pass"])


class PreregIdentityAndPinTests(unittest.TestCase):
    def _write_identity_fixture(self, root: Path):
        authority = root / "prompts" / "authority.md"
        authority.parent.mkdir(parents=True, exist_ok=True)
        authority.write_text("authority", encoding="utf-8")
        authority_sha = sha256_file(authority)
        config = root / "configs" / "cfg.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps({"authority_prompt": {"path": "prompts/authority.md", "sha256": authority_sha}}),
            encoding="utf-8",
        )
        cfg_sha = sha256_file(config)
        report = root / "reports" / "report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "config": {"path": "configs/cfg.json", "sha256": cfg_sha},
                    "authority_prompt": {"path": "prompts/authority.md", "sha256": authority_sha},
                    "frozen_inputs": {},
                    "checkpoints": {},
                }
            ),
            encoding="utf-8",
        )
        return config, report

    def test_identity_consistent_then_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, report = self._write_identity_fixture(root)
            prereg = json.loads(config.read_text(encoding="utf-8"))
            prereg_report = json.loads(report.read_text(encoding="utf-8"))
            out = diag.verify_prereg_identity(
                prereg, prereg_report, root=root,
                config_path=Path("configs/cfg.json"), report_path=Path("reports/report.json"),
            )
            self.assertTrue(out["consistent"])
            (root / "prompts" / "authority.md").write_text("changed", encoding="utf-8")
            drifted = diag.verify_prereg_identity(
                prereg, prereg_report, root=root,
                config_path=Path("configs/cfg.json"), report_path=Path("reports/report.json"),
            )
            self.assertFalse(drifted["consistent"])
            self.assertFalse(drifted["authority_prompt_sha256_matches_report"])

    def test_pins_hash_but_never_parse_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = (
                root / "artifacts" / "token_selector" / "x"
                / "B_SPATIAL_KL_CONTRASTIVE" / "20261003" / "best.npz"
            )
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(b"not-a-real-npz")
            rel = "artifacts/token_selector/prompt_projection_features/index.jsonl"
            pinned = root / rel
            pinned.parent.mkdir(parents=True, exist_ok=True)
            pinned.write_text("index", encoding="utf-8")
            prereg_report = {
                "frozen_inputs": {rel: sha256_file(pinned)},
                "checkpoints": {
                    "B_SPATIAL_KL_CONTRASTIVE/20261003": sha256_file(ckpt)
                },
            }
            out = diag.verify_prereg_pins(
                prereg_report, root=root,
                artifacts_dir=Path("artifacts/token_selector/x"),
            )
            self.assertTrue(out["pins_unchanged"])
            ckpt.write_bytes(b"changed")
            drifted = diag.verify_prereg_pins(
                prereg_report, root=root,
                artifacts_dir=Path("artifacts/token_selector/x"), strict=False,
            )
            self.assertFalse(drifted["pins_unchanged"])


class EndToEndIsolationTests(unittest.TestCase):
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
                (features / f"{sample_id}.npz").write_bytes(b"dummy-bytes")
                index_records.append(
                    {
                        "sample_id": sample_id,
                        "task": task,
                        "npz": f"{sample_id}.npz",
                        "group_count": GROUP_COUNT,
                        "grid_h": GRID_H,
                        "grid_w": GRID_W,
                        "prompt_token_count": TOKENS,
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
        (features / "test_x.npz").write_bytes(b"corrupted-test-bytes")
        index_records.append(
            {
                "sample_id": "test-1",
                "task": "grounding",
                "npz": "test_x.npz",
                "group_count": GROUP_COUNT,
                "grid_h": GRID_H,
                "grid_w": GRID_W,
                "prompt_token_count": TOKENS,
                "source_image_sha256": "f" * 64,
                "npz_sha256": "0" * 64,
            }
        )
        _write_jsonl(features / "index.jsonl", index_records)
        _write_jsonl(root / "datasets" / "manifest.jsonl", manifest_records)

        authority = root / "prompts" / "authority.md"
        authority.parent.mkdir(parents=True, exist_ok=True)
        authority.write_text("authority", encoding="utf-8")
        authority_sha = sha256_file(authority)
        config = root / "configs" / "prereg.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(
            json.dumps(
                {
                    "data": {"val_npz_aggregate_sha256": "irrelevant-in-test", "expected_records": 4, "expected_images": 2},
                    "authority_prompt": {"path": "prompts/authority.md", "sha256": authority_sha},
                }
            ),
            encoding="utf-8",
        )
        report = root / "reports" / "prereg-report.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "config": {"path": "configs/prereg.json", "sha256": sha256_file(config)},
                    "authority_prompt": {"path": "prompts/authority.md", "sha256": authority_sha},
                    "frozen_inputs": {},
                    "checkpoints": {},
                }
            ),
            encoding="utf-8",
        )
        prior = root / "reports" / "prior.json"
        prior.write_text(
            json.dumps(
                {
                    "primary_gate": {
                        "G1_correct_alignment_non_inferiority": {"pass": True},
                        "G4_deployed_K48_utility": {"pass": True},
                        "G5_collapse_removed": {"pass": True},
                        "G7_hardware_contract": {"pass": True},
                    },
                    "decision": {"decision": "PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED"},
                }
            ),
            encoding="utf-8",
        )

        seed = 20261003
        for arm in diag.ARMS:
            ckpt_dir = (
                root / "artifacts" / "token_selector" / "spatial_prompt_contrastive_trainval"
                / arm / str(seed)
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
            "artifacts_dir": Path("artifacts/token_selector/spatial_prompt_contrastive_trainval"),
            "prereg_config_path": Path("configs/prereg.json"),
            "prereg_report_path": Path("reports/prereg-report.json"),
            "prior_report_path": Path("reports/prior.json"),
            "val_ids": val_ids,
            "seed": seed,
        }

    def test_test_npz_never_opened_and_report_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fx = self._fixture(root)
            opened_by_loader: list[str] = []
            arrays = _cache_arrays()
            loader_state = {"n": 0}

            def spy(path):
                opened_by_loader.append(str(path))
                loader_state["n"] += 1
                return _cache_arrays(seed=loader_state["n"])

            def fake_split(index_records, *, seed, ratios, salt):
                return {"train": [], "val": list(fx["val_ids"]), "test": ["test-1"]}

            with mock.patch.object(diag, "split_sample_ids", side_effect=fake_split):
                result = diag.run_diagnostic(
                    root=root,
                    prereg_config_path=fx["prereg_config_path"],
                    prereg_report_path=fx["prereg_report_path"],
                    prior_report_path=fx["prior_report_path"],
                    index_path=fx["index_path"],
                    manifest_path=fx["manifest_path"],
                    artifacts_dir=fx["artifacts_dir"],
                    features_dir=fx["features"],
                    arms=diag.ARMS,
                    seeds=(fx["seed"],),
                    budgets=(48,),
                    split_rule={"seed": 1, "ratios": {"train": 0.0, "val": 0.8, "test": 0.2}, "salt": "unit"},
                    cache_loader=spy,
                    verify_pins=False,
                )
            del arrays
            bundle = result["bundle"]
            csv_rows = result["csv_rows"]

            self.assertEqual(len(opened_by_loader), 4)
            self.assertTrue(all("test_x.npz" not in path for path in opened_by_loader))
            self.assertEqual(bundle["isolation"]["test_npz_opened"], [])
            self.assertEqual(bundle["isolation"]["train_npz_opened"], [])
            self.assertTrue(bundle["isolation"]["opened_only_val_npz"])
            self.assertFalse(bundle["isolation"]["test_npz_resolved_hashed_or_opened"])
            self.assertEqual(bundle["split_counts"]["val_records"], 4)
            self.assertEqual(bundle["split_counts"]["val_images"], 2)
            self.assertTrue(
                bundle["isolation"]["prior_report_json_parsed_for_gate_reference_only"]
            )
            self.assertFalse(
                bundle["isolation"]["restricted_val_samples_csv_or_json_parsed_as_metric_source"]
            )
            self.assertTrue(bundle["auxiliary_gates"]["S4"]["pass"])
            self.assertTrue(bundle["provenance"]["prereg_config"]["unchanged"])
            self.assertTrue(bundle["provenance"]["prereg_report"]["unchanged"])

            # task / grid / group_count stratification is present.
            for arm in diag.ARMS:
                block = bundle["arms"][arm]
                for key in ("overall", "grounding", "pointing", "grid=9x9", "group_count=81"):
                    self.assertIn(key, block)
                    self.assertIn("k48", block[key])

            # CSV cardinality: arms x seeds x records x budgets.
            self.assertEqual(len(csv_rows), 2 * 1 * 4 * 1)
            self.assertEqual(set(csv_rows[0]), set(diag.CSV_FIELDS))

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
                "arms",
                "primary_holm_family",
                "auxiliary_gates",
                "decision",
                "task_k48",
                "k48_jaccard",
                "limitations",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["primary_holm_family"]["family_size"], 2)
            self.assertIn("tests", payload["primary_holm_family"])
            with Path(files["csv"]).open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(diag.CSV_FIELDS))
                self.assertEqual(len(list(reader)), len(csv_rows))
            self.assertTrue(Path(files["md"]).read_text(encoding="utf-8").startswith("#"))

            harness = diag.harness_payload(bundle, files)
            self.assertEqual(harness["decision"]["decision"], bundle["status"])
            for key in ("P1", "P2", "S1", "S2", "S3", "S4", "task_k48", "isolation", "files", "limitations", "next_decision"):
                self.assertIn(key, harness)
            self.assertEqual(set(harness["S4"]), {"pass", "prior_gates_all_pass", "pins_unchanged", "identity_consistent"})


class RealDataRegressionTests(unittest.TestCase):
    """End-to-end regression against the real pinned checkpoints and val split."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cfg = cls.root / diag.PREREG_CONFIG_PATH
        rep = cls.root / diag.PREREG_REPORT_PATH
        ckpt = (
            cls.root
            / diag.ARTIFACTS_DIR
            / diag.PRIMARY_ARM
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

    def test_prereg_identity_and_pins_unchanged(self):
        bundle = self.result["bundle"]
        self.assertTrue(bundle["provenance"]["identity"]["consistent"])
        self.assertTrue(bundle["provenance"]["prereg_config"]["unchanged"])
        self.assertTrue(bundle["provenance"]["prereg_report"]["unchanged"])
        self.assertTrue(bundle["provenance"]["pins"]["pins_unchanged"])
        self.assertTrue(bundle["provenance"]["val_npz_aggregate_matches_prereg"])

    def test_prior_gates_and_primary_family_shape(self):
        bundle = self.result["bundle"]
        prior = bundle["prior_report_reference"]
        self.assertTrue(prior["available"])
        for short in ("G1", "G4", "G5", "G7"):
            self.assertTrue(prior["gates"][short]["pass"], short)
        family = bundle["primary_holm_family"]
        self.assertEqual(family["family_size"], 2)
        for key in (diag.P1_KEY, diag.P2_KEY):
            entry = family["tests"][key]
            self.assertEqual(entry["n_images"], 38)
            self.assertEqual(sorted(entry["seed_means"]), ["20261003", "20261004", "20261005"])
            self.assertIn("holm_adjusted_p", entry)
        self.assertIn(
            bundle["status"],
            {"OPERATIONAL_PROMPT_UTILITY_SUPPORTED", "OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED"},
        )

    def test_every_budget_and_task_reported(self):
        bundle = self.result["bundle"]
        for budget in ("k48", "k64", "k96"):
            self.assertIn(budget, bundle["budget_summary"])
        for task in diag.TASKS:
            self.assertIn(task, bundle["task_k48"])
            self.assertIn("overall", bundle["arms"][diag.PRIMARY_ARM])
            self.assertIn(task, bundle["arms"][diag.PRIMARY_ARM])


if __name__ == "__main__":
    unittest.main()
