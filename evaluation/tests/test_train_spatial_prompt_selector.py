#!/usr/bin/env python3
"""Focused tests for the preregistered spatial-supervision trainer.

Covers: target choice by arm/task, shared epoch-0 state and deterministic batch
order, exact KL reference, val-only best selection (tie earliest), corrupted
internal-test NPZ isolation, the five-test Holm family, G1-G6 gates, the
corrected selected-set NDCG helper, checkpoint round-trip, preregistration
immutability, and report schemas.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from evaluation.prompt_identifiability_diagnostic import (
    derange_images,
    sample_donor_map,
)
from evaluation.prompt_projection_distillation import (
    load_feature_cache,
    ndcg_at_k,
    ndcg_for_selected_indices,
    save_feature_cache,
    topk_indices,
)
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.spatial_teacher_feasibility_audit import (
    grounding_area_target,
    pointing_bilinear_target,
)
import evaluation.train_spatial_prompt_selector as trainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PREREG = PROJECT_ROOT / "configs/spatial_student_trainval.preregistered.json"
REAL_PREREG_REPORT = (
    PROJECT_ROOT
    / "reports/token_selector/2026-10-03-spatial-student-trainval-preregistration.json"
)

GRID = 8


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _softmax(x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def _kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    a = p / p.sum()
    b = q / q.sum()
    mask = a > 0
    # no 1e-12 clip: q is strictly positive here and clipping would distort the
    # true KL for very small probabilities (the trainer computes the exact KL).
    return float(
        np.sum(a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], 1e-300, None))))
    )


def build_synthetic_dataset(
    root: Path, *, n_images: int = 20, seed: int = 0, grid: int = GRID
) -> dict:
    """Tiny width-correct feature caches plus index, manifest and prereg."""
    rng = np.random.default_rng(seed)
    records = []
    manifest = []
    for index in range(n_images):
        image_sha = "%064x" % (index + 1)
        for task in ("grounding", "pointing"):
            sample_id = f"syn-{index:03d}-{task}"
            group_mean = rng.standard_normal((grid * grid, 1024)).astype(np.float32)
            prompt_embeddings = rng.standard_normal((4, 2560)).astype(np.float32)
            raw = rng.standard_normal(grid * grid).astype(np.float32)
            normalized = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            soft = _softmax(normalized)
            save_feature_cache(
                root / f"{sample_id}.npz",
                group_mean=group_mean,
                prompt_embeddings=prompt_embeddings,
                teacher_raw=raw,
                teacher_normalized=normalized.astype(np.float32),
                teacher_soft=soft.astype(np.float32),
                metadata={"source_image_sha256": image_sha, "task": task},
            )
            records.append(
                {
                    "sample_id": sample_id,
                    "npz": f"{sample_id}.npz",
                    "task": task,
                    "source_image_sha256": image_sha,
                    "grid_h": grid,
                    "grid_w": grid,
                    "group_count": grid * grid,
                    "prompt_token_count": 4,
                    "npz_sha256": "",
                }
            )
            manifest.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "ground_truth": {
                        "bbox_1000": [50.0, 50.0, 500.0, 500.0],
                        "point_1000": [300.0, 300.0],
                    },
                }
            )
    index_path = root / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(record) for record in manifest) + "\n", encoding="utf-8"
    )
    prereg_path = root / "prereg.json"
    prereg_path.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_TRAIN_VAL_ONLY",
                "split": {
                    "seed": 20260927,
                    "salt": "synthetic_spatial_test",
                    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": root,
        "index_path": index_path,
        "manifest_path": manifest_path,
        "prereg_path": prereg_path,
        "records": records,
        "manifest": manifest,
    }


class SpyLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: Path):
        self.calls.append(str(path))
        return load_feature_cache(path)


def _run(dataset: dict, *, seeds, arm_names, epochs, write_outputs, spy=None):
    root = dataset["root"]
    return trainer.run_experiment(
        prereg_config=dataset["prereg_path"],
        prereg_report=dataset["prereg_path"],
        index_path=dataset["index_path"],
        manifest_path=dataset["manifest_path"],
        features_dir=root,
        artifacts_dir=root / "artifacts",
        reports_dir=root / "reports",
        seeds=seeds,
        arm_names=arm_names,
        epochs=epochs,
        enforce_frozen_inputs=False,
        enforce_expected_counts=False,
        cache_loader=spy,
        write_outputs=write_outputs,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class SpatialTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="spatial_tests_")
        cls.root = Path(cls._tmp.name)
        cls.data = build_synthetic_dataset(cls.root)
        cls.spy = SpyLoader()
        cls.bundle = _run(
            cls.data,
            seeds=[20261003],
            arm_names=trainer.ARM_ORDER,
            epochs=2,
            write_outputs=True,
            spy=cls.spy,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- target choice by arm/task ----------------------------------------
    def test_target_choice_by_arm_and_task(self) -> None:
        manifest_by_id = {record["sample_id"]: record for record in self.data["manifest"]}
        grounding = {
            "sample_id": "g0",
            "task": "grounding",
            "group_count": GRID * GRID,
            "grid_h": GRID,
            "grid_w": GRID,
            "teacher_soft": np.full(GRID * GRID, 1.0 / (GRID * GRID)),
        }
        pointing = {
            "sample_id": "p0",
            "task": "pointing",
            "group_count": GRID * GRID,
            "grid_h": GRID,
            "grid_w": GRID,
            "teacher_soft": np.full(GRID * GRID, 1.0 / (GRID * GRID)),
        }
        manifest_by_id["g0"] = {
            "sample_id": "g0",
            "task": "grounding",
            "ground_truth": {"bbox_1000": [0.0, 0.0, 500.0, 1000.0], "point_1000": [1.0, 1.0]},
        }
        manifest_by_id["p0"] = {
            "sample_id": "p0",
            "task": "pointing",
            "ground_truth": {"bbox_1000": [0.0, 0.0, 500.0, 1000.0], "point_1000": [250.0, 750.0]},
        }
        attached = trainer.attach_spatial_targets([grounding, pointing], manifest_by_id)
        g_sample, p_sample = attached
        # arm A always uses the cached semantic teacher target
        np.testing.assert_allclose(
            trainer.record_target(g_sample, trainer.CONTROL_ARM),
            g_sample["teacher_soft"],
        )
        # arm B uses the frozen spatial targets per task
        np.testing.assert_allclose(
            trainer.record_target(g_sample, trainer.PRIMARY_ARM),
            grounding_area_target([0.0, 0.0, 500.0, 1000.0], GRID, GRID),
        )
        np.testing.assert_allclose(
            trainer.record_target(p_sample, trainer.PRIMARY_ARM),
            pointing_bilinear_target([250.0, 750.0], GRID, GRID),
        )
        self.assertFalse(
            np.allclose(
                trainer.record_target(g_sample, trainer.PRIMARY_ARM),
                trainer.record_target(p_sample, trainer.PRIMARY_ARM),
            )
        )
        # spatial target never equals the semantic teacher target in general
        self.assertEqual(g_sample["spatial_target"].shape, (GRID * GRID,))

    def test_attach_spatial_targets_grid_contract(self) -> None:
        sample = {
            "sample_id": "bad",
            "task": "grounding",
            "group_count": 63,
            "grid_h": GRID,
            "grid_w": GRID,
            "teacher_soft": np.zeros(63),
        }
        with self.assertRaises(Exception):
            trainer.attach_spatial_targets(
                [sample],
                {"bad": {"sample_id": "bad", "ground_truth": {"bbox_1000": [0, 0, 1, 1]}}},
            )

    # -- shared epoch-0 state and deterministic batches --------------------
    def test_shared_epoch0_state_and_batch_order(self) -> None:
        hashes = self.bundle["epoch0_hashes"]["20261003"]
        self.assertEqual(len(set(hashes.values())), 1)
        self.assertTrue(
            all(self.bundle["epoch0_hashes_consistent_across_arms"].values())
        )
        images = [f"img-{i}" for i in range(10)]
        first = trainer.epoch_batch_order(20261003, 3, images, 4)
        self.assertEqual(first, trainer.epoch_batch_order(20261003, 3, images, 4))
        flat = sorted(index for batch in first for index in batch)
        self.assertEqual(flat, list(range(10)))
        self.assertNotEqual(first, trainer.epoch_batch_order(20261003, 4, images, 4))

    # -- exact KL reference -------------------------------------------------
    def test_exact_kl_reference(self) -> None:
        torch_mod = torch
        rng = np.random.default_rng(11)
        for _ in range(5):
            target = _softmax(rng.standard_normal(9))
            scores = rng.standard_normal(9).astype(np.float32)
            value = trainer.record_kl(
                torch_mod.as_tensor(scores),
                torch_mod.as_tensor(target.astype(np.float32)),
                temperature=0.1,
            )
            reference = _kl_nats(target, _softmax(scores, 0.1))
            self.assertAlmostEqual(float(value), reference, delta=5e-3)

    def test_pair_loss_is_mean_of_grounding_and_pointing(self) -> None:
        torch_mod = torch
        rng = np.random.default_rng(12)
        size = 9
        target_g = _softmax(rng.standard_normal(size))
        target_p = _softmax(rng.standard_normal(size))
        scores_g = rng.standard_normal(size).astype(np.float32)
        scores_p = rng.standard_normal(size).astype(np.float32)
        grounding = {
            "group_mean": torch_mod.as_tensor(rng.standard_normal((size, 1024)).astype(np.float32)),
            "prompt_embeddings": torch_mod.as_tensor(rng.standard_normal((4, 2560)).astype(np.float32)),
        }
        pointing = {
            "group_mean": torch_mod.as_tensor(rng.standard_normal((size, 1024)).astype(np.float32)),
            "prompt_embeddings": torch_mod.as_tensor(rng.standard_normal((4, 2560)).astype(np.float32)),
        }
        tensors = {
            "g": {**grounding, "teacher_soft": torch_mod.as_tensor(target_g.astype(np.float32)), "spatial_target": torch_mod.as_tensor(target_g.astype(np.float32))},
            "p": {**pointing, "teacher_soft": torch_mod.as_tensor(target_p.astype(np.float32)), "spatial_target": torch_mod.as_tensor(target_p.astype(np.float32))},
        }
        pair = {"grounding_id": "g", "pointing_id": "p"}
        Wt = torch_mod.nn.Parameter(torch_mod.zeros((128, 2560)))
        Wv = torch_mod.nn.Parameter(torch_mod.zeros((128, 1024)))
        # zeros make every score 0 -> uniform student distribution; the KL then
        # depends only on the target, so the mean-of-two reference is exact.
        loss = trainer.pair_kl_loss(tensors, pair, Wt, Wv, trainer.CONTROL_ARM)
        uniform = np.full(size, 1.0 / size)
        reference = 0.5 * (
            _kl_nats(target_g, uniform) + _kl_nats(target_p, uniform)
        )
        self.assertAlmostEqual(float(loss), reference, delta=5e-4)

    # -- val-only best selection -------------------------------------------
    def test_best_epoch_uses_own_val_kl_tie_earliest(self) -> None:
        self.assertEqual(trainer.select_best_epoch([1.0, 0.5, 0.5, 2.0]), 1)
        self.assertEqual(trainer.select_best_epoch([0.3, 0.3, 0.3]), 0)
        for arm_name, training in self.bundle["training"].items():
            for entry in training["per_seed"]:
                val_kls = [row["val"]["kl"] for row in entry["epoch_metrics"]]
                self.assertEqual(entry["best_epoch"], trainer.select_best_epoch(val_kls))
                self.assertAlmostEqual(entry["best_val_kl"], min(val_kls), places=9)
                self.assertEqual(entry["epoch_metrics"][0]["phase"], "pre_update")
        # arm A selection uses semantic KL, arm B selection uses spatial KL
        self.assertEqual(
            self.bundle["arms"][trainer.CONTROL_ARM]["target"], "semantic_teacher"
        )
        self.assertEqual(
            self.bundle["arms"][trainer.PRIMARY_ARM]["target"], "spatial"
        )

    # -- internal-test NPZ isolation ---------------------------------------
    def test_isolation_opened_only_train_val(self) -> None:
        self.assertEqual(self.bundle["isolation"]["test_npz_opened"], [])
        self.assertEqual(self.bundle["isolation"]["test_npz_opened_count"], 0)
        self.assertTrue(self.bundle["isolation"]["opened_only_train_val_npz"])
        self.assertFalse(self.bundle["isolation"]["test_used_for_selection"])
        test_ids = set(self.bundle["split"]["test"])
        test_names = {
            record["sample_id"]
            for record in self.data["records"]
            if record["sample_id"] in test_ids
        }
        called_names = {Path(path).stem for path in self.spy.calls}
        self.assertTrue(test_names.isdisjoint(called_names))

    def test_corrupt_test_npz_never_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spatial_corrupt_") as tmp_name:
            tmp = Path(tmp_name)
            data = build_synthetic_dataset(tmp, n_images=20, seed=3)
            splits = data.get("splits")
            if splits is None:
                from evaluation.prompt_projection_distillation import split_sample_ids

                splits = split_sample_ids(
                    [
                        {
                            "sample_id": record["sample_id"],
                            "source_image_sha256": record["source_image_sha256"],
                        }
                        for record in data["records"]
                    ],
                    seed=20260927,
                    ratios={"train": 0.7, "val": 0.15, "test": 0.15},
                    salt="synthetic_spatial_test",
                )
            self.assertGreaterEqual(len(splits["test"]), 2)
            for record in data["records"]:
                if record["sample_id"] in set(splits["test"]):
                    (tmp / record["npz"]).write_bytes(b"not-a-valid-npz")
            spy = SpyLoader()
            bundle = _run(
                data,
                seeds=[20261004],
                arm_names=(trainer.CONTROL_ARM,),
                epochs=1,
                write_outputs=False,
                spy=spy,
            )
            self.assertEqual(bundle["isolation"]["test_npz_opened"], [])
            test_paths = {
                str(tmp / record["npz"])
                for record in data["records"]
                if record["sample_id"] in set(splits["test"])
            }
            self.assertTrue(test_paths.isdisjoint(set(spy.calls)))

    # -- five-test Holm family ---------------------------------------------
    def test_holm_family_is_exactly_five(self) -> None:
        self.assertEqual(trainer.HOLM_FAMILY_SIZE, 5)
        self.assertEqual(len(trainer.PRIMARY_TESTS), 5)
        primary = self.bundle["validation_statistics"]["primary_tests"]
        self.assertEqual(len(primary), 5)
        for entry in primary.values():
            self.assertEqual(entry["holm_family_size"], 5)
            self.assertIsNotNone(entry["holm_adjusted_p"])
            self.assertGreaterEqual(
                float(entry["holm_adjusted_p"]) + 1e-12,
                float(entry["p_value_two_sided"]) - 1e-12,
            )
            self.assertEqual(entry["independent_unit"], "image")

    # -- gates --------------------------------------------------------------
    def test_all_gates_and_decision(self) -> None:
        def passing_entry(value: float = 0.1) -> dict:
            return {
                "mean": value,
                "ci_low": value / 2.0,
                "ci_high": value * 2.0,
                "cohens_dz": 0.6,
                "p_value_two_sided": 0.001,
                "holm_adjusted_p": 0.004,
                "positive_seed_count": 3,
                "seed_count": 3,
            }

        validation = {
            "arms": {
                trainer.PRIMARY_ARM: {
                    "collapse": {"max_exact_prompt_invariance_fraction": 0.0}
                }
            },
            "primary_tests": {
                "B_minus_A_correct_spearman": passing_entry(),
                "B_minus_A_correct_ndcg_full": passing_entry(),
                "B_correct_minus_same_spearman": passing_entry(),
                "B_correct_minus_same_ndcg_full": passing_entry(),
                "B_selector_minus_L2_anchor_mass_k48": passing_entry(),
            },
        }
        training = {
            "per_seed": [
                {"best_epoch": 1, "best_val_kl": 0.5, "epoch0_val_kl": 0.6},
                {"best_epoch": 2, "best_val_kl": 0.4, "epoch0_val_kl": 0.5},
                {"best_epoch": 0, "best_val_kl": 0.6, "epoch0_val_kl": 0.5},
            ]
        }
        gate = trainer.evaluate_gates(validation=validation, training=training)
        self.assertTrue(gate["all_pass"])
        decision = trainer.decide(gate)
        self.assertEqual(decision["decision"], "SPATIAL_STUDENT_LEARNABLE")
        # flipping G1 fails the whole arm
        validation["primary_tests"]["B_minus_A_correct_spearman"]["mean"] = -0.1
        gate = trainer.evaluate_gates(validation=validation, training=training)
        self.assertFalse(gate["G1_spatial_supervision_gain"]["pass"])
        self.assertFalse(gate["all_pass"])
        self.assertEqual(
            trainer.decide(gate)["decision"], "SPATIAL_STUDENT_NOT_ESTABLISHED"
        )
        # collapse above threshold fails G4
        validation["primary_tests"]["B_minus_A_correct_spearman"]["mean"] = 0.1
        validation["arms"][trainer.PRIMARY_ARM]["collapse"][
            "max_exact_prompt_invariance_fraction"
        ] = 0.5
        gate = trainer.evaluate_gates(validation=validation, training=training)
        self.assertFalse(gate["G4_collapse_removed"]["pass"])
        # a missing primary arm never wins
        self.assertEqual(
            trainer.decide(None)["decision"], "SPATIAL_STUDENT_NOT_ESTABLISHED"
        )

    def test_control_arm_can_never_win(self) -> None:
        self.assertFalse(trainer.ARM_SPEC[trainer.CONTROL_ARM]["can_win"])
        self.assertTrue(trainer.ARM_SPEC[trainer.PRIMARY_ARM]["can_win"])
        self.assertNotEqual(trainer.PRIMARY_ARM, trainer.CONTROL_ARM)

    # -- corrected selected-set NDCG ---------------------------------------
    def test_corrected_selected_set_ndcg_used(self) -> None:
        # group 4 is both the highest-scored and the highest-relevance group;
        # the restricted selected set {0, 1} must therefore give a lower NDCG
        # than the unrestricted top-K.
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
        selected = np.array([0, 1])
        full = ndcg_at_k(scores, target, 2)
        restricted = ndcg_for_selected_indices(selected, scores, target, 2)
        self.assertLess(restricted, full)
        ideal = target[4] * (1.0 / np.log2(2.0)) + target[3] * (1.0 / np.log2(3.0))
        manual = (
            target[1] * (1.0 / np.log2(2.0)) + target[0] * (1.0 / np.log2(3.0))
        ) / ideal
        self.assertAlmostEqual(restricted, manual, places=9)
        # identity selection reproduces ndcg_at_k exactly
        identity = np.arange(target.size)
        self.assertAlmostEqual(
            ndcg_for_selected_indices(identity, scores, target, target.size),
            ndcg_at_k(scores, target, target.size),
            places=12,
        )

    def test_deployed_selector_uses_corrected_ndcg_and_anchor_rule(self) -> None:
        rng = np.random.default_rng(5)
        group_count = GRID * GRID
        target = np.abs(rng.standard_normal(group_count))
        target = target / target.sum()
        semantic = rng.standard_normal(group_count)
        l2 = np.abs(rng.standard_normal(group_count))
        combined = combine_scores(semantic, l2, trainer.LAMBDA_SEMANTIC)
        metrics = trainer.deployed_selection_metrics(
            combined, l2, target, GRID, GRID, budgets=(48,)
        )
        selected = trainer._select_groups(combined, GRID, GRID, 48)
        baseline = trainer._select_groups(l2, GRID, GRID, 48)
        self.assertEqual(len(selected), 48)
        self.assertEqual(len(baseline), 48)
        self.assertAlmostEqual(
            metrics["selector_ndcg_k48"],
            ndcg_for_selected_indices(selected, combined, target, 48),
            places=12,
        )
        self.assertAlmostEqual(
            metrics["l2_anchor_ndcg_k48"],
            ndcg_for_selected_indices(baseline, l2, target, 48),
            places=12,
        )
        # the baseline is the L2+anchor path, not a second anchor invention
        self.assertEqual(trainer.ANCHOR_REGION_GRID, 3)

    # -- checkpoints --------------------------------------------------------
    def test_checkpoint_roundtrip(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import load_projection

        for arm_name, training in self.bundle["training"].items():
            entry = training["per_seed"][0]
            best = Path(entry["best_checkpoint"])
            final = Path(entry["final_checkpoint"])
            metadata = Path(entry["metadata_path"])
            self.assertTrue(best.is_file())
            self.assertTrue(final.is_file())
            self.assertTrue(metadata.is_file())
            projection, file_sha = load_projection(best)
            self.assertEqual(tuple(projection.Wt.shape), (128, 2560))
            self.assertEqual(tuple(projection.Wv.shape), (128, 1024))
            self.assertEqual(file_sha, _sha256(best))
            with np.load(best, allow_pickle=False) as archive:
                self.assertIn("Wt", archive)
                self.assertIn("Wv", archive)
                self.assertIn("metadata", archive)
                self.assertEqual(
                    str(np.asarray(archive["weights_sha256"]).reshape(-1)[0]),
                    entry["best_weights_sha256"],
                )
        # no per-epoch weight files were written
        arm_dir = Path(self.bundle["training"][trainer.PRIMARY_ARM]["per_seed"][0]["best_checkpoint"]).parent
        self.assertEqual(
            sorted(path.name for path in arm_dir.iterdir()),
            ["best.npz", "final.npz", "metadata.json"],
        )

    # -- prereg immutability -------------------------------------------------
    def test_prereg_immutability(self) -> None:
        before = (_sha256(REAL_PREREG), _sha256(REAL_PREREG_REPORT))
        verification = trainer.verify_frozen_inputs()
        after = (_sha256(REAL_PREREG), _sha256(REAL_PREREG_REPORT))
        self.assertEqual(before, after)
        self.assertEqual(verification["checks"]["prereg_report_pins_config"], True)
        report = json.loads(REAL_PREREG_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(
            report["preregistered_config"]["sha256"], before[0]
        )

    # -- report schemas ------------------------------------------------------
    def test_report_schemas(self) -> None:
        paths = self.bundle["report_paths"]
        for key in ("epochs_csv", "val_samples_csv", "json", "md"):
            self.assertTrue(Path(paths[key]).is_file(), key)
        payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        for key in (
            "status",
            "decision",
            "validation_statistics",
            "primary_gate",
            "training",
            "isolation",
            "split_group_summary",
            "hashes_unchanged",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["validation_statistics"]["holm_family_size"], 5)
        header = Path(paths["val_samples_csv"]).read_text(encoding="utf-8").splitlines()[0]
        for column in (
            "arm",
            "seed",
            "sample_id",
            "correct_spearman",
            "correct_ndcg_full",
            "same_spearman",
            "selector_mass_k48",
            "l2_anchor_mass_k48",
            "exact_prompt_invariant",
        ):
            self.assertIn(column, header)
        epoch_header = (
            Path(paths["epochs_csv"]).read_text(encoding="utf-8").splitlines()[0]
        )
        for column in ("arm", "seed", "epoch", "split", "kl"):
            self.assertIn(column, epoch_header)
        self.assertIn("G1", Path(paths["md"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
