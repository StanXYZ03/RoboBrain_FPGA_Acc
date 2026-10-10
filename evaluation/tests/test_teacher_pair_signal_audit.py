#!/usr/bin/env python3
"""Focused CPU tests for the train-only teacher paired-signal audit.

The tests cover the new distribution primitives, grid distances, aggregation,
top-K min(K, G) handling, and -- most importantly -- the hard isolation
contract: the split comes from index metadata only and no val/test feature
cache is ever opened.  Corrupting the val/test npz files must not move any
train metric or the input aggregate hash.

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
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation import teacher_pair_signal_audit as audit
from evaluation.prompt_projection_distillation import (
    PROMPT_DIM,
    FEATURE_DIM,
    assign_split,
    load_distillation_config,
    save_feature_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / "configs" / "prompt_projection_distillation.json"


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _soft_peak(group_count: int, peak: int) -> np.ndarray:
    values = np.zeros(group_count, dtype=np.float64)
    values[peak] = 1.0
    return values.astype(np.float32)


def _norm_from_raw(raw: np.ndarray) -> np.ndarray:
    low = float(raw.min())
    high = float(raw.max())
    span = high - low if high > low else 1.0
    return ((raw - low) / span).astype(np.float32)


class DistributionPrimitiveTests(unittest.TestCase):
    def test_js_zero_for_identical(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        self.assertEqual(audit.js_divergence_nats(p, p), 0.0)

    def test_js_ln2_for_disjoint(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        self.assertAlmostEqual(audit.js_divergence_nats(p, q), math.log(2.0), places=12)

    def test_js_symmetric_and_matches_base2_times_ln2(self) -> None:
        from evaluation.prompt_identifiability_diagnostic import js_divergence

        p = np.array([0.1, 0.2, 0.7])
        q = np.array([0.3, 0.3, 0.4])
        nats = audit.js_divergence_nats(p, q)
        self.assertAlmostEqual(nats, audit.js_divergence_nats(q, p), places=12)
        self.assertAlmostEqual(nats, js_divergence(p, q) * math.log(2.0), places=12)

    def test_js_range_bounded(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.random(10)
            q = rng.random(10)
            value = audit.js_divergence_nats(p, q)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, math.log(2.0) + 1e-12)

    def test_kl_asymmetry_and_identity(self) -> None:
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        self.assertAlmostEqual(audit.kl_divergence_nats(p, p), 0.0, places=12)
        self.assertGreater(audit.kl_divergence_nats(p, q), 0.0)
        self.assertGreater(audit.kl_divergence_nats(q, p), 0.0)
        self.assertNotAlmostEqual(
            audit.kl_divergence_nats(p, q), audit.kl_divergence_nats(q, p), places=12
        )

    def test_symmetric_kl_is_mean_of_directions(self) -> None:
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.2, 0.3, 0.5])
        expected = 0.5 * (
            audit.kl_divergence_nats(p, q) + audit.kl_divergence_nats(q, p)
        )
        self.assertAlmostEqual(audit.symmetric_kl_nats(p, q), expected, places=12)

    def test_total_variation_known_value(self) -> None:
        p = np.array([0.75, 0.25])
        q = np.array([0.25, 0.75])
        self.assertAlmostEqual(audit.total_variation_distance(p, q), 0.5, places=12)
        self.assertEqual(audit.total_variation_distance(p, p), 0.0)

    def test_kl_epsilon_handles_zero_support(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([1.0, 0.0])
        self.assertEqual(audit.kl_divergence_nats(p, q), 0.0)
        # q has a zero where p has mass -> finite (clamped) value, never nan/inf
        value = audit.kl_divergence_nats(np.array([0.5, 0.5]), np.array([1.0, 0.0]))
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)


class GridDistanceTests(unittest.TestCase):
    def test_manhattan_and_euclidean(self) -> None:
        # grid 4 rows x 5 cols; index 0 = (0,0), index 12 = (2,2)
        distance = audit.grid_distance(0, 12, grid_h=4, grid_w=5)
        self.assertEqual(distance["manhattan"], 4.0)
        self.assertAlmostEqual(distance["euclidean"], math.hypot(2, 2), places=12)

    def test_same_index_zero(self) -> None:
        distance = audit.grid_distance(7, 7, grid_h=4, grid_w=5)
        self.assertEqual(distance["manhattan"], 0.0)
        self.assertEqual(distance["euclidean"], 0.0)

    def test_out_of_range_rejected(self) -> None:
        from evaluation.prompt_projection_distillation import DistillationContractError

        with self.assertRaises(DistillationContractError):
            audit.grid_distance(0, 20, grid_h=4, grid_w=5)


class DescribeTests(unittest.TestCase):
    def test_full_stats(self) -> None:
        stats = audit.describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4)
        self.assertAlmostEqual(stats["mean"], 2.5)
        self.assertAlmostEqual(stats["median"], 2.5)
        self.assertAlmostEqual(stats["min"], 1.0)
        self.assertAlmostEqual(stats["max"], 4.0)
        self.assertAlmostEqual(stats["p25"], 1.75)
        self.assertAlmostEqual(stats["p95"], 3.85)

    def test_empty_stats(self) -> None:
        stats = audit.describe([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["mean"])


class SyntheticDataset:
    """Build a tiny train/val/test feature cache from the frozen split salt."""

    def __init__(self, root: Path, per_split: Mapping[str, int]) -> None:
        self.root = Path(root)
        self.features_dir = self.root / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        config = load_distillation_config(REAL_CONFIG)
        split = config["split"]
        self.salt = str(split["salt"])
        self.split_seed = int(split["seed"])
        self.ratios = split["ratios"]
        self.assignments: dict[str, str] = {}
        self.image_hashes: list[str] = []
        self.records: list[dict[str, Any]] = []
        self.train_npz: set[str] = set()
        self.val_test_npz: set[str] = set()
        self._build(per_split)

    def _pick_hashes(self, per_split: Mapping[str, int]) -> dict[str, list[str]]:
        wanted = dict(per_split)
        chosen: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        counter = 0
        while any(len(chosen[name]) < wanted[name] for name in wanted):
            image_hash = _hash(f"teacher-pair-audit-test-image-{counter}")
            counter += 1
            name = assign_split(
                image_hash, seed=self.split_seed, ratios=self.ratios, salt=self.salt
            )
            if len(chosen[name]) < wanted[name] and image_hash not in self.assignments:
                chosen[name].append(image_hash)
                self.assignments[image_hash] = name
            if counter > 5000:  # pragma: no cover - defensive
                raise RuntimeError("could not find enough image hashes")
        return chosen

    def _build(self, per_split: Mapping[str, int]) -> None:
        chosen = self._pick_hashes(per_split)
        rng = np.random.default_rng(11)
        self.index_rows: list[dict[str, Any]] = []
        for split_name, image_hashes in chosen.items():
            for order, image_hash in enumerate(image_hashes):
                group_count = 8 if order % 2 == 0 else 16
                grid_w = 4
                grid_h = group_count // grid_w
                identical = split_name == "train" and order == 0
                for task in ("grounding", "pointing"):
                    sample_id = f"tpa-{split_name}-{order:03d}-{task}"
                    if identical:
                        peak = 0
                    elif task == "grounding":
                        peak = 1
                    else:
                        peak = group_count - 1
                    raw = np.linspace(0.0, 1.0, group_count)
                    raw[peak] = 2.0
                    raw = raw.astype(np.float32)
                    normalized = _norm_from_raw(raw)
                    soft = _soft_peak(group_count, peak)
                    group_mean = rng.random((group_count, FEATURE_DIM)).astype(np.float32)
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
                            "grid_h": grid_h,
                            "grid_w": grid_w,
                            "group_count": group_count,
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
                            "group_count": group_count,
                            "prompt_token_count": 3,
                            "grid_h": grid_h,
                            "grid_w": grid_w,
                            "source_image_sha256": image_hash,
                        }
                    )
        self.index_path = self.root / "index.jsonl"
        with self.index_path.open("w", encoding="utf-8") as stream:
            for row in self.index_rows:
                stream.write(json.dumps(row) + "\n")


class AuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = SyntheticDataset(Path(self._tmp.name), {"train": 3, "val": 2, "test": 2})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return audit.run_audit(
            config_path=REAL_CONFIG,
            index_path=self.dataset.index_path,
            features_dir=self.dataset.features_dir,
        )

    def test_run_isolates_and_reports_train_images(self) -> None:
        report, rows = self._run()
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["split_counts"]["train"], 6)
        self.assertEqual(report["split_counts"]["val"], 4)
        self.assertEqual(report["split_counts"]["test"], 4)
        self.assertEqual(len(rows), 3)
        self.assertEqual(report["aggregate"]["independent_image_units"], 3)
        self.assertEqual(report["aggregate"]["sample_record_count"], 6)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])
        self.assertEqual(report["isolation"]["val_test_npz_opened"], [])
        self.assertEqual(report["aggregate"]["fractions"]["exact_identical_fraction"], 1.0 / 3.0)
        self.assertEqual(report["aggregate"]["fractions"]["js_positive_fraction"], 2.0 / 3.0)
        self.assertEqual(report["aggregate"]["fractions"]["argmax_different_fraction"], 2.0 / 3.0)

    def test_only_train_feature_cache_is_opened(self) -> None:
        seen: list[str] = []
        real_loader = audit.load_feature_cache

        def spy(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(str(path))
            return real_loader(path, *args, **kwargs)

        import unittest.mock as mock

        with mock.patch.object(audit, "load_feature_cache", spy):
            report, _ = self._run()
        self.assertEqual(set(seen), self.dataset.train_npz)
        self.assertEqual(set(seen) & self.dataset.val_test_npz, set())
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_corrupting_val_test_npz_cannot_move_train_metrics_or_hash(self) -> None:
        report_before, rows_before = self._run()
        for path in self.dataset.val_test_npz:
            Path(path).write_bytes(b"this is not a valid npz file")
        report_after, rows_after = self._run()
        self.assertEqual(rows_before, rows_after)
        self.assertEqual(
            report_before["provenance"]["input_npz_aggregate_sha256"],
            report_after["provenance"]["input_npz_aggregate_sha256"],
        )
        self.assertEqual(report_before["aggregate"], report_after["aggregate"])
        self.assertEqual(
            report_before["provenance"]["index_sha256"],
            report_after["provenance"]["index_sha256"],
        )

    def test_loader_that_raises_on_non_train_paths(self) -> None:
        allowed = self.dataset.train_npz.copy()

        def strict(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if str(path) not in allowed:
                raise AssertionError(f"non-train cache opened: {path}")
            return audit.load_feature_cache(path, *args, **kwargs)

        report, _ = audit.run_audit(
            config_path=REAL_CONFIG,
            index_path=self.dataset.index_path,
            features_dir=self.dataset.features_dir,
            cache_loader=strict,
        )
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_index_metadata_load_opens_no_npz(self) -> None:
        import unittest.mock as mock

        with mock.patch.object(
            audit, "load_feature_cache", side_effect=AssertionError("npz opened")
        ):
            metadata = audit.load_index_metadata(self.dataset.index_path)
        self.assertEqual(len(metadata), 14)

    def test_grid_stratification_present(self) -> None:
        report, rows = self._run()
        group_counts = sorted(int(row["group_count"]) for row in rows)
        self.assertEqual(set(group_counts), {8, 16})
        strata = report["aggregate"]["strata_by_group_count"]
        self.assertEqual(set(strata), {"8", "16"})
        self.assertEqual(strata["8"]["image_count"] + strata["16"]["image_count"], 3)


class PerImageMetricTests(unittest.TestCase):
    def _sample(self, sample_id: str, task: str, group_count: int, peak: int) -> dict[str, Any]:
        raw = np.linspace(0.0, 1.0, group_count)
        raw[peak] = 2.0
        raw = raw.astype(np.float32)
        return {
            "sample_id": sample_id,
            "task": task,
            "source_image_sha256": "a" * 64,
            "grid_h": 4,
            "grid_w": 4,
            "group_count": group_count,
            "teacher_raw": raw,
            "teacher_normalized": _norm_from_raw(raw),
            "teacher_soft": _soft_peak(group_count, peak),
        }

    def test_topk_min_k_g_when_k_exceeds_g(self) -> None:
        # G=8 < K=48: effective K is 8, so both top-K sets cover every group and
        # the disagreement is exactly 0 (an important descriptive edge case).
        grounding = self._sample("g", "grounding", 8, 0)
        pointing = self._sample("p", "pointing", 8, 7)
        row = audit.per_image_metrics(grounding, pointing)
        self.assertEqual(row["k48_effective"], 8.0)
        self.assertTrue(row["k48_exceeds_g"])
        self.assertEqual(row["k96_effective"], 8.0)
        self.assertTrue(row["k96_exceeds_g"])
        self.assertEqual(row["k48_jaccard"], 1.0)
        self.assertEqual(row["k48_disagreement"], 0.0)
        self.assertEqual(row["argmax_same"], 0.0)
        self.assertGreater(row["js_nats"], 0.0)
        self.assertEqual(row["exact_equal"], 0.0)

    def test_topk_at_48_on_large_g(self) -> None:
        rng = np.random.default_rng(5)
        grounding = self._sample("g", "grounding", 200, 3)
        pointing = self._sample("p", "pointing", 200, 190)
        grounding["teacher_raw"] = rng.random(200).astype(np.float32)
        pointing["teacher_raw"] = rng.random(200).astype(np.float32)
        row = audit.per_image_metrics(grounding, pointing)
        self.assertEqual(row["k48_effective"], 48.0)
        self.assertFalse(row["k48_exceeds_g"])
        self.assertGreaterEqual(row["k48_jaccard"], 0.0)
        self.assertLessEqual(row["k48_jaccard"], 1.0)
        self.assertAlmostEqual(
            row["k48_disagreement"], 1.0 - row["k48_jaccard"], places=12
        )

    def test_grid_distances_use_flat_row_major_index(self) -> None:
        grounding = self._sample("g", "grounding", 16, 0)  # grid 4x4 -> (0,0)
        pointing = self._sample("p", "pointing", 16, 10)  # (2,2)
        row = audit.per_image_metrics(grounding, pointing)
        self.assertEqual(row["argmax_manhattan"], 4.0)
        self.assertAlmostEqual(row["argmax_euclidean"], math.hypot(2, 2), places=12)

    def test_identical_soft_gives_zero_divergences(self) -> None:
        grounding = self._sample("g", "grounding", 16, 3)
        pointing = self._sample("p", "pointing", 16, 3)
        row = audit.per_image_metrics(grounding, pointing)
        self.assertEqual(row["js_nats"], 0.0)
        self.assertEqual(row["symmetric_kl"], 0.0)
        self.assertEqual(row["tv"], 0.0)
        self.assertEqual(row["exact_equal"], 1.0)
        self.assertEqual(row["tolerance_equal"], 1.0)
        self.assertEqual(row["argmax_same"], 1.0)

    def test_oracle_swap_direction_positive(self) -> None:
        grounding = self._sample("g", "grounding", 16, 0)
        pointing = self._sample("p", "pointing", 16, 15)
        row = audit.per_image_metrics(grounding, pointing)
        self.assertGreater(row["kl_grounding_pointing"], 0.0)
        self.assertGreater(row["kl_pointing_grounding"], 0.0)


class MarkdownAndAggregateTests(unittest.TestCase):
    def test_markdown_has_required_sections_and_no_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            report, rows = audit.run_audit(
                config_path=REAL_CONFIG,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
            markdown = audit.render_markdown(report)
            self.assertIn("自然对数", markdown)
            self.assertIn("oracle", markdown.lower())
            self.assertIn("max-over-prompt-token", markdown)
            self.assertIn("未自创通过阈值", markdown)
            self.assertIn("不能**据此声称", markdown)
            self.assertIn("有效独立单位", markdown)
            self.assertEqual(report["js_base"], "natural_log")
            self.assertFalse(report["isolation"]["val_test_npz_opened"])


if __name__ == "__main__":
    unittest.main()
