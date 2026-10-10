#!/usr/bin/env python3
"""Focused CPU tests for the parameter-free spatial supervision feasibility audit.

Coverage:
* exact bbox/grid-cell overlap and normalization;
* bilinear point mass, including corners / edges / interior / boundary clamping;
* join hard failures (missing, duplicate, task mismatch);
* grid mismatch hard failure;
* train-only access with spy loader and corrupted val/test NPZ;
* decision logic (signal present vs not established);
* report schema and CSV fields;
* deterministic paired resampling;
* frozen-file immutability of the preregistration inputs.

These tests never train, never load a model, never use CUDA and never touch the
network.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluation import spatial_teacher_feasibility_audit as audit
from evaluation.prompt_projection_distillation import (
    FEATURE_DIM,
    PROMPT_DIM,
    assign_split,
    load_distillation_config,
    l2_baseline_scores,
    ndcg_at_k,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    save_feature_cache,
    topk_indices,
)
from evaluation.hardware_regular_token_selector import select_real_groups

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / "configs" / "prompt_projection_distillation.json"


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class SpatialTargetTests(unittest.TestCase):
    def test_bbox_exact_cell_overlap(self) -> None:
        target = audit.grounding_area_target([500.0, 0.0, 1000.0, 500.0], 2, 2)
        np.testing.assert_allclose(target, [0.0, 1.0, 0.0, 0.0], atol=1e-12)

    def test_bbox_half_overlap_two_cells_normalized(self) -> None:
        target = audit.grounding_area_target([0.0, 0.0, 250.0, 1000.0], 2, 2)
        np.testing.assert_allclose(target, [0.5, 0.0, 0.5, 0.0], atol=1e-12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)

    def test_bbox_partial_fraction(self) -> None:
        # 4x4 grid, cell width 250; bbox covers 125 of the 250 of cells (0,0),(1,0)
        target = audit.grounding_area_target([0.0, 0.0, 125.0, 1000.0], 4, 4)
        self.assertAlmostEqual(float(target[0]), 0.25, places=12)
        self.assertAlmostEqual(float(target[4]), 0.25, places=12)
        self.assertAlmostEqual(float(target[8]), 0.25, places=12)
        self.assertAlmostEqual(float(target[12]), 0.25, places=12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)

    def test_invalid_zero_area_bbox_raises(self) -> None:
        with self.assertRaises(audit.SpatialContractError):
            audit.grounding_area_target([10.0, 10.0, 10.0, 20.0], 4, 4)

    def test_non_finite_bbox_raises(self) -> None:
        with self.assertRaises(audit.SpatialContractError):
            audit.grounding_area_target([0.0, 0.0, float("nan"), 20.0], 4, 4)

    def test_bilinear_center_distributes_quarter_mass(self) -> None:
        target = audit.pointing_bilinear_target([500.0, 500.0], 4, 4).reshape(4, 4)
        np.testing.assert_allclose(target[1:3, 1:3], 0.25, atol=1e-12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)
        self.assertEqual(int(np.count_nonzero(target)), 4)

    def test_bilinear_corner_all_mass_first_cell(self) -> None:
        target = audit.pointing_bilinear_target([0.0, 0.0], 4, 4)
        self.assertAlmostEqual(float(target[0]), 1.0, places=12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)

    def test_bilinear_far_corner_all_mass_last_cell(self) -> None:
        target = audit.pointing_bilinear_target([1000.0, 1000.0], 4, 4).reshape(4, 4)
        self.assertAlmostEqual(float(target[3, 3]), 1.0, places=12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)

    def test_bilinear_combines_duplicate_boundary_index(self) -> None:
        # x at the right edge clamps u to W-1 so col0 == col1; mass must combine.
        target = audit.pointing_bilinear_target([1000.0, 500.0], 4, 4).reshape(4, 4)
        self.assertAlmostEqual(float(target[1:3, 3].sum()), 1.0, places=12)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)

    def test_bilinear_left_boundary_split(self) -> None:
        # x = W/2 * 1000/W => u = 0.5 between col0 and col1
        target = audit.pointing_bilinear_target([250.0, 0.0], 4, 4).reshape(4, 4)
        self.assertAlmostEqual(float(target[0, 0]), 0.5, places=12)
        self.assertAlmostEqual(float(target[0, 1]), 0.5, places=12)

    def test_point_outside_range_raises(self) -> None:
        with self.assertRaises(audit.SpatialContractError):
            audit.pointing_bilinear_target([1001.0, 500.0], 4, 4)

    def test_point_inside_bbox(self) -> None:
        self.assertTrue(audit.point_inside_bbox([50.0, 50.0], [0.0, 0.0, 100.0, 100.0]))
        self.assertFalse(audit.point_inside_bbox([150.0, 50.0], [0.0, 0.0, 100.0, 100.0]))

    def test_target_diagnostics_values(self) -> None:
        diagnostics = audit.target_diagnostics(np.array([0.25, 0.25, 0.25, 0.25]))
        self.assertTrue(diagnostics["finite"])
        self.assertTrue(diagnostics["non_negative"])
        self.assertAlmostEqual(diagnostics["sum_error"], 0.0, places=12)
        self.assertEqual(diagnostics["support_size"], 4)
        self.assertAlmostEqual(diagnostics["entropy_nats"], math.log(4.0), places=12)
        self.assertAlmostEqual(diagnostics["effective_support"], 4.0, places=12)
        self.assertAlmostEqual(diagnostics["max_mass"], 0.25, places=12)

    def test_grid_validation(self) -> None:
        with self.assertRaises(audit.SpatialContractError):
            audit.grounding_area_target([0.0, 0.0, 10.0, 10.0], 0, 4)


class JoinTests(unittest.TestCase):
    def _index(self, sample_id: str, task: str, image_hash: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "task": task,
            "npz": f"{sample_id}.npz",
            "group_count": 16,
            "grid_h": 4,
            "grid_w": 4,
            "source_image_sha256": image_hash,
        }

    def _manifest(self, sample_id: str, task: str, coco_id: int = 1) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "task": task,
            "coco_image_id": coco_id,
            "image": "coco2017/x.jpg",
            "ground_truth": {"bbox_1000": [0.0, 0.0, 1.0, 1.0], "point_1000": [0.0, 0.0]},
        }

    def test_join_missing_sample_raises(self) -> None:
        index = [self._index("a", "grounding", "0" * 64)]
        manifest = {"b": self._manifest("b", "grounding")}
        with self.assertRaises(audit.SpatialContractError):
            audit.join_index_manifest(index, manifest)

    def test_join_duplicate_index_raises(self) -> None:
        index = [
            self._index("a", "grounding", "0" * 64),
            self._index("a", "pointing", "0" * 64),
        ]
        manifest = {"a": self._manifest("a", "grounding")}
        with self.assertRaises(audit.SpatialContractError):
            audit.join_index_manifest(index, manifest)

    def test_join_task_mismatch_raises(self) -> None:
        index = [self._index("a", "grounding", "0" * 64)]
        manifest = {"a": self._manifest("a", "pointing")}
        with self.assertRaises(audit.SpatialContractError):
            audit.join_index_manifest(index, manifest)

    def test_join_image_identity_mismatch_raises(self) -> None:
        index = [
            self._index("a", "grounding", "0" * 64),
            self._index("b", "pointing", "0" * 64),
        ]
        manifest = {
            "a": self._manifest("a", "grounding", coco_id=1),
            "b": self._manifest("b", "pointing", coco_id=2),
        }
        with self.assertRaises(audit.SpatialContractError):
            audit.join_index_manifest(index, manifest)

    def test_duplicate_manifest_sample_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.jsonl"
            record = self._manifest("a", "grounding")
            path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaises(audit.SpatialContractError):
                audit.load_clean_manifest(path)

    def test_valid_join_passes(self) -> None:
        index = [
            self._index("a", "grounding", "0" * 64),
            self._index("b", "pointing", "0" * 64),
        ]
        manifest = {"a": self._manifest("a", "grounding"), "b": self._manifest("b", "pointing")}
        index_by_id, groups, diagnostics = audit.join_index_manifest(index, manifest)
        self.assertEqual(set(index_by_id), {"a", "b"})
        self.assertEqual(diagnostics["image_group_count"], 1)
        self.assertEqual(sorted(groups["0" * 64]), ["a", "b"])


class SyntheticSpatialDataset:
    """Tiny train/val/test index + manifest + feature cache."""

    def __init__(self, root: Path, per_split: Mapping[str, int], mode: str = "present") -> None:
        self.root = Path(root)
        self.features_dir = self.root / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        config = load_distillation_config(REAL_CONFIG)
        split = config["split"]
        self.salt = str(split["salt"])
        self.split_seed = int(split["seed"])
        self.ratios = split["ratios"]
        self.grid_h = 4
        self.grid_w = 4
        self.group_count = self.grid_h * self.grid_w
        self.assignments: dict[str, str] = {}
        self.train_npz: set[str] = set()
        self.val_test_npz: set[str] = set()
        self.index_rows: list[dict[str, Any]] = []
        self.manifest_rows: list[dict[str, Any]] = []
        self._build(per_split)

    def _pick_hashes(self, per_split: Mapping[str, int]) -> dict[str, list[str]]:
        wanted = dict(per_split)
        chosen: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        counter = 0
        while any(len(chosen[name]) < wanted[name] for name in wanted):
            image_hash = _hash(f"spatial-audit-test-image-{counter}")
            counter += 1
            name = assign_split(
                image_hash, seed=self.split_seed, ratios=self.ratios, salt=self.salt
            )
            if len(chosen[name]) < wanted[name] and image_hash not in self.assignments:
                chosen[name].append(image_hash)
                self.assignments[image_hash] = name
            if counter > 20000:  # pragma: no cover - defensive
                raise RuntimeError("could not find enough image hashes")
        return chosen

    def _teacher(self, task: str) -> tuple[np.ndarray, np.ndarray]:
        if self.mode == "present":
            soft = np.full(self.group_count, 1.0 / self.group_count, dtype=np.float64)
        else:
            soft = np.zeros(self.group_count, dtype=np.float64)
            soft[0 if task == "grounding" else self.group_count - 1] = 1.0
        raw = np.linspace(0.0, 1.0, self.group_count)
        normalized = np.zeros(self.group_count)
        return raw.astype(np.float32), soft.astype(np.float32)

    def _build(self, per_split: Mapping[str, int]) -> None:
        chosen = self._pick_hashes(per_split)
        rng = np.random.default_rng(23)
        for split_name, image_hashes in chosen.items():
            for order, image_hash in enumerate(image_hashes):
                coco_id = 100000 + order
                image_path = f"coco2017/visual_refinement_original/{coco_id:012d}.jpg"
                # geometry per mode
                if self.mode == "present":
                    point = [937.5, 937.5]
                else:
                    point = [125.0, 125.0]
                bbox = [0.0, 0.0, 250.0, 250.0]
                for task in ("grounding", "pointing"):
                    sample_id = f"spa-{split_name}-{order:03d}-{task}"
                    raw, soft = self._teacher(task)
                    normalized = ((raw - raw.min()) / max(raw.max() - raw.min(), 1e-9)).astype(np.float32)
                    group_mean = rng.random((self.group_count, FEATURE_DIM)).astype(np.float32)
                    prompt = rng.random((3, PROMPT_DIM)).astype(np.float32)
                    npz_name = f"{sample_id}.npz"
                    npz_path = self.features_dir / npz_name
                    save_feature_cache(
                        npz_path,
                        group_mean=group_mean,
                        prompt_embeddings=prompt,
                        teacher_raw=raw,
                        teacher_normalized=normalized,
                        teacher_soft=soft,
                        metadata={
                            "sample_id": sample_id,
                            "task": task,
                            "grid_h": self.grid_h,
                            "grid_w": self.grid_w,
                            "group_count": self.group_count,
                            "source_image_sha256": image_hash,
                        },
                    )
                    if split_name == "train":
                        self.train_npz.add(str(npz_path))
                    else:
                        self.val_test_npz.add(str(npz_path))
                    self.index_rows.append(
                        {
                            "sample_id": sample_id,
                            "task": task,
                            "npz": npz_name,
                            "group_count": self.group_count
                            if self.mode != "grid_mismatch"
                            else self.group_count - 1,
                            "prompt_token_count": 3,
                            "grid_h": self.grid_h,
                            "grid_w": self.grid_w,
                            "source_image_sha256": image_hash,
                        }
                    )
                    self.manifest_rows.append(
                        {
                            "sample_id": sample_id,
                            "task": task,
                            "coco_image_id": coco_id,
                            "image": image_path,
                            "coordinate_system": "normalized_0_1000",
                            "ground_truth": {"bbox_1000": bbox, "point_1000": point},
                        }
                    )
        self.index_path = self.root / "index.jsonl"
        self.manifest_path = self.root / "manifest.jsonl"
        with self.index_path.open("w", encoding="utf-8") as stream:
            for row in self.index_rows:
                stream.write(json.dumps(row) + "\n")
        with self.manifest_path.open("w", encoding="utf-8") as stream:
            for row in self.manifest_rows:
                stream.write(json.dumps(row) + "\n")

    def drop_manifest_record(self, sample_id: str) -> None:
        self.manifest_rows = [r for r in self.manifest_rows if r["sample_id"] != sample_id]
        with self.manifest_path.open("w", encoding="utf-8") as stream:
            for row in self.manifest_rows:
                stream.write(json.dumps(row) + "\n")


def _run(dataset: SyntheticSpatialDataset):
    return audit.compute_report(
        config_path=REAL_CONFIG,
        index_path=dataset.index_path,
        manifest_path=dataset.manifest_path,
        features_dir=dataset.features_dir,
    )


class AuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_signal_present_decision(self) -> None:
        dataset = SyntheticSpatialDataset(self.root, {"train": 4, "val": 1, "test": 1}, "present")
        report, rows = _run(dataset)
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["decision_label"], audit.DECISION_PRESENT)
        self.assertEqual(len(rows), 4)
        self.assertGreater(report["paired_differences"]["diff_js_nats"]["ci_low"], 0.0)
        self.assertGreater(report["paired_differences"]["diff_tv"]["ci_low"], 0.0)

    def test_signal_not_established_decision(self) -> None:
        dataset = SyntheticSpatialDataset(
            self.root, {"train": 4, "val": 1, "test": 1}, "not_established"
        )
        report, rows = _run(dataset)
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["decision_label"], audit.DECISION_NOT_ESTABLISHED)
        self.assertLess(report["paired_differences"]["diff_js_nats"]["ci_high"], 0.0)

    def test_grid_mismatch_blocks(self) -> None:
        dataset = SyntheticSpatialDataset(
            self.root, {"train": 2, "val": 1, "test": 1}, "grid_mismatch"
        )
        report, rows = _run(dataset)
        self.assertEqual(report["decision_label"], audit.DECISION_BLOCKED)
        self.assertEqual(rows, [])
        self.assertTrue(report["contract_violations"])

    def test_missing_manifest_join_blocks(self) -> None:
        dataset = SyntheticSpatialDataset(self.root, {"train": 2, "val": 1, "test": 1}, "present")
        dataset.drop_manifest_record(dataset.manifest_rows[0]["sample_id"])
        report, rows = _run(dataset)
        self.assertEqual(report["decision_label"], audit.DECISION_BLOCKED)
        self.assertEqual(rows, [])

    def test_only_train_feature_cache_is_opened(self) -> None:
        import unittest.mock as mock

        dataset = SyntheticSpatialDataset(self.root, {"train": 3, "val": 1, "test": 1}, "present")
        seen: list[str] = []
        real_loader = audit.load_feature_cache

        def spy(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(str(path))
            return real_loader(path, *args, **kwargs)

        with mock.patch.object(audit, "load_feature_cache", spy):
            report, _ = _run(dataset)
        self.assertEqual(set(seen), dataset.train_npz)
        self.assertEqual(set(seen) & dataset.val_test_npz, set())
        self.assertTrue(report["isolation"]["opened_only_train_npz"])
        self.assertEqual(report["isolation"]["val_test_npz_opened"], [])
        self.assertEqual(report["isolation"]["opened_npz_count"], len(dataset.train_npz))

    def test_corrupting_val_test_npz_cannot_move_train_metrics(self) -> None:
        dataset = SyntheticSpatialDataset(self.root, {"train": 3, "val": 1, "test": 1}, "present")
        report_before, rows_before = _run(dataset)
        for path in dataset.val_test_npz:
            Path(path).write_bytes(b"this is not a valid npz file")
        report_after, rows_after = _run(dataset)
        self.assertEqual(rows_before, rows_after)
        self.assertEqual(
            report_before["provenance"]["input_npz_aggregate_sha256"],
            report_after["provenance"]["input_npz_aggregate_sha256"],
        )
        self.assertEqual(report_before["spatial_pair_signal"], report_after["spatial_pair_signal"])

    def test_report_schema_and_csv_fields(self) -> None:
        dataset = SyntheticSpatialDataset(self.root, {"train": 3, "val": 1, "test": 1}, "present")
        report, rows = _run(dataset)
        for key in (
            "split_counts",
            "isolation",
            "integrity",
            "geometry",
            "semantic_pair_signal",
            "spatial_pair_signal",
            "paired_differences",
            "semantic_spatial_compatibility",
            "l2_baseline",
            "anchor_baseline",
            "correction",
            "provenance",
        ):
            self.assertIn(key, report)
        self.assertIn(
            report["decision_label"],
            {
                audit.DECISION_PRESENT,
                audit.DECISION_NOT_ESTABLISHED,
                audit.DECISION_BLOCKED,
            },
        )
        for key in audit.DIFF_KEYS:
            stats = report["paired_differences"][key]
            for field in (
                "mean",
                "median",
                "std",
                "ci_low",
                "ci_high",
                "p_value_two_sided",
                "cohens_dz",
                "n",
            ):
                self.assertIn(field, stats)
            self.assertEqual(stats["bootstrap_resamples"], 2000)
            self.assertEqual(stats["permutation_resamples"], 10000)
        for row in rows:
            self.assertIn("spatial_js_nats", row)
            self.assertIn("semantic_js_nats", row)
            self.assertAlmostEqual(
                row["diff_js_nats"],
                row["spatial_js_nats"] - row["semantic_js_nats"],
                places=12,
            )
        csv_path = self.root / "out.csv"
        audit.write_csv(csv_path, rows)
        header = csv_path.read_text(encoding="utf-8").splitlines()[0]
        for field in audit.CSV_FIELDS:
            self.assertIn(field, header)
        markdown = audit.render_markdown(report)
        self.assertIn("决策", markdown)
        self.assertIn("L2", markdown)
        self.assertIn("bootstrap", markdown)

    def test_l2_and_anchor_baseline_shapes(self) -> None:
        dataset = SyntheticSpatialDataset(self.root, {"train": 3, "val": 1, "test": 1}, "present")
        report, _ = _run(dataset)
        for name in ("l2_baseline", "anchor_baseline"):
            baseline = report[name]
            self.assertTrue(baseline["available"], msg=str(baseline))
            for k in ("48", "64", "96"):
                for scope in ("overall", "grounding", "pointing"):
                    budget = baseline["per_budget"][k][scope]
                    self.assertEqual(budget["count"] > 0, True)
                    for metric in ("ndcg", "mass"):
                        mean = budget[metric]["mean"]
                        self.assertGreaterEqual(mean, -1e-9)
                        self.assertLessEqual(mean, 1.0 + 1e-9)

    def test_deterministic_paired_resampling(self) -> None:
        values = [0.4, 0.5, 0.6, 0.7, 0.8]
        first = audit.paired_stats(values, seed=20261003)
        second = audit.paired_stats(values, seed=20261003)
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 5)
        self.assertAlmostEqual(first["median"], 0.6, places=12)


class AnchorNdcgCorrectionTests(unittest.TestCase):
    """Focused coverage for the anchor-baseline NDCG selection-set correction.

    These tests fail under the pre-correction code, where
    ``anchor_baseline_record`` computed ``ndcg_at_k(scores, target, effective)``
    and therefore ignored the anchor-selected set.
    """

    def _group_mean_from_scores(self, scores: Any) -> np.ndarray:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        group_mean = np.zeros((values.size, FEATURE_DIM), dtype=np.float32)
        for index, value in enumerate(values):
            group_mean[index, index % FEATURE_DIM] = value
        return group_mean

    def _anchor_case(self):
        grid_h = 8
        grid_w = 8
        group_count = grid_h * grid_w
        rng = np.random.default_rng(1030)
        raw = rng.random(group_count)
        target = rng.random(group_count)
        target = target / target.sum()
        group_mean = self._group_mean_from_scores(raw)
        scores = l2_baseline_scores(group_mean)
        return grid_h, grid_w, group_count, group_mean, scores, target

    def test_helper_ordering_descending_l2_low_index_ties(self) -> None:
        scores = np.array([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        relevance = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # indices 2 and 5 tie on score; low index 2 must be ranked first, so the
        # relevant index 5 lands at position 2 (discount 1/log2(3)).
        discount1 = 1.0 / math.log2(2.0)
        discount2 = 1.0 / math.log2(3.0)
        idcg = (1.0 * discount1)  # single positive relevance, k=2 ideal
        expected = (0.0 * discount1 + 1.0 * discount2) / idcg
        value = ndcg_for_selected_indices([5, 2], scores, relevance, 2)
        self.assertAlmostEqual(value, expected, places=12)
        # Input order must not matter; low-index tie policy is deterministic.
        self.assertEqual(
            value,
            ndcg_for_selected_indices([2, 5], scores, relevance, 2),
        )

    def test_helper_ignores_unselected_high_relevance(self) -> None:
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        relevance = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        selected = [0, 1, 2]
        # The only relevant group (index 4) is unselected and must not enter DCG.
        self.assertAlmostEqual(
            ndcg_for_selected_indices(selected, scores, relevance, 3), 0.0, places=12
        )
        # The global metric would score the unselected group and return 1.0.
        self.assertAlmostEqual(ndcg_at_k(scores, relevance, 3), 1.0, places=12)

    def test_helper_identity_bypass_equals_global_ndcg(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.random(40)
        relevance = rng.random(40)
        selected = np.arange(40)
        for k in (1, 7, 40, 96):
            self.assertAlmostEqual(
                ndcg_for_selected_indices(selected, scores, relevance, k),
                ndcg_at_k(scores, relevance, k),
                places=12,
                msg=f"k={k}",
            )

    def test_helper_explicit_validation(self) -> None:
        scores = np.array([0.1, 0.2, 0.3])
        relevance = np.array([0.0, 0.5, 1.0])
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([0, 1], scores, relevance, 0)
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([], scores, relevance, 2)
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([0, 0], scores, relevance, 2)
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([0, 3], scores, relevance, 2)
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([0, 1], scores, relevance[:2], 2)
        with self.assertRaises(audit.DistillationContractError):
            ndcg_for_selected_indices([0, 1], scores, [-1.0, 1.0, 0.0], 2)

    def test_anchor_record_restricts_ndcg_to_selected_groups(self) -> None:
        grid_h, grid_w, group_count, group_mean, scores, target = self._anchor_case()
        record = audit.anchor_baseline_record(
            group_mean, target, grid_h, grid_w, ks=(48,)
        )
        budget = record["per_budget"]["48"]
        self.assertTrue(budget["available"], msg=str(record))
        self.assertEqual(budget["kind"], "select_real_groups")
        selected = np.asarray(
            select_real_groups(scores, grid_h, grid_w, 48).selected, dtype=np.int64
        )
        self.assertNotEqual(
            set(int(value) for value in selected),
            set(int(value) for value in topk_indices(scores, 48)),
        )
        corrected = ndcg_for_selected_indices(selected, scores, target, 48)
        pure_l2 = ndcg_at_k(scores, target, 48)
        self.assertAlmostEqual(budget["ndcg"], corrected, places=12)
        self.assertNotAlmostEqual(budget["ndcg"], pure_l2, places=9)
        # Old code returned exactly the pure-L2 NDCG.
        self.assertNotAlmostEqual(corrected, pure_l2, places=9)

    def test_anchor_mass_is_unchanged_by_the_correction(self) -> None:
        grid_h, grid_w, group_count, group_mean, scores, target = self._anchor_case()
        record = audit.anchor_baseline_record(
            group_mean, target, grid_h, grid_w, ks=(48,)
        )
        budget = record["per_budget"]["48"]
        selected = np.asarray(
            select_real_groups(scores, grid_h, grid_w, 48).selected, dtype=np.int64
        )
        # Mass already used the selected set before the correction and must be
        # identical to the reference mass computed directly from the selection.
        self.assertAlmostEqual(
            budget["mass"],
            saliency_mass_retained(selected, target),
            places=12,
        )
        # It must not coincide with the top-K L2 mass when selection differs.
        self.assertNotAlmostEqual(
            budget["mass"],
            saliency_mass_retained(topk_indices(scores, 48), target),
            places=9,
        )

    def test_report_correction_block_identity_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticSpatialDataset(
                Path(tmp), {"train": 3, "val": 1, "test": 1}, "present"
            )
            report, _ = _run(dataset)
        correction = report["correction"]
        self.assertEqual(correction["label"], audit.CORRECTION_LABEL)
        self.assertTrue(correction["primary_metrics_unchanged"])
        self.assertIn("anchor_mass_unchanged", correction)
        self.assertIsInstance(correction["anchor_mass_max_abs_diff"], float)
        self.assertGreaterEqual(correction["anchor_mass_max_abs_diff"], 0.0)
        # The synthetic grid has 16 groups, so K=48/64/96 are identity bypass:
        # corrected anchor NDCG must equal the pre-correction (pure-L2) NDCG.
        for k in ("48", "64", "96"):
            for scope in ("overall", "grounding", "pointing"):
                self.assertAlmostEqual(
                    correction["anchor_ndcg_corrected"][k][scope],
                    correction["anchor_ndcg_old"][k][scope],
                    places=12,
                )
                self.assertAlmostEqual(
                    correction["anchor_ndcg_old"][k][scope],
                    report["l2_baseline"]["per_budget"][k][scope]["ndcg"]["mean"],
                    places=12,
                )
        markdown = audit.render_markdown(report)
        self.assertIn("更正记录", markdown)
        self.assertIn("anchor NDCG", markdown)


class PreregistrationTests(unittest.TestCase):
    def test_preregistered_inputs_match_and_are_immutable(self) -> None:
        config_path = PROJECT_ROOT / audit.PREREG_CONFIG_REL
        report_path = PROJECT_ROOT / audit.PREREG_REPORT_REL
        self.assertTrue(config_path.is_file())
        self.assertTrue(report_path.is_file())
        first = audit.verify_preregistration(config_path, report_path)
        self.assertTrue(first["all_match"], msg=json.dumps(first, sort_keys=True))
        second = audit.verify_preregistration(config_path, report_path)
        self.assertEqual(
            first["preregistration_config_sha256"], second["preregistration_config_sha256"]
        )
        self.assertEqual(
            first["preregistration_report_sha256"], second["preregistration_report_sha256"]
        )
        for name, entry in first["pinned_inputs"].items():
            self.assertEqual(
                entry["actual_sha256"], second["pinned_inputs"][name]["actual_sha256"]
            )

    def test_run_audit_requires_preregistration_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticSpatialDataset(Path(tmp), {"train": 1, "val": 1, "test": 1}, "present")
            with self.assertRaises(audit.SpatialContractError):
                audit.run_audit(
                    config_path=REAL_CONFIG,
                    index_path=dataset.index_path,
                    manifest_path=dataset.manifest_path,
                    features_dir=dataset.features_dir,
                    verify_preregistration_hashes=True,
                )


if __name__ == "__main__":
    unittest.main()
