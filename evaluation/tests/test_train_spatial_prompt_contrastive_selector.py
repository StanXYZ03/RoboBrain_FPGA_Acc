#!/usr/bin/env python3
"""Focused tests for the preregistered prompt-contrastive spatial trainer.

Covers: both arms' exact loss references, the same-image wrong-prompt exchange,
shared epoch-0 state and deterministic batch order, val-only best selection,
the five-test Holm family, G1-G7 gates, the correct P5 (selector minus baseline)
sign, internal-test NPZ isolation, preregistration immutability, and
checkpoint/report schemas.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import torch

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from evaluation.prompt_projection_distillation import (
    load_feature_cache,
    save_feature_cache,
)
from evaluation.spatial_teacher_feasibility_audit import (
    grounding_area_target,
    pointing_bilinear_target,
)
import evaluation.train_spatial_prompt_contrastive_selector as trainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_PREREG = (
    PROJECT_ROOT / "configs/spatial_prompt_contrastive_trainval.preregistered.json"
)
REAL_PREREG_REPORT = (
    PROJECT_ROOT
    / "reports/token_selector/2026-10-03-spatial-prompt-contrastive-trainval-preregistration.json"
)
REAL_PROMPT = (
    PROJECT_ROOT
    / "reports/token_selector/prompts/2026-10-03-spatial-prompt-contrastive-trainval-harness-prompt.md"
)
EXISTING_SPATIAL_TRAINER = PROJECT_ROOT / "evaluation/train_spatial_prompt_selector.py"

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
                    "salt": "synthetic_contrastive_test",
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
        harness_prompt=dataset["prereg_path"],
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


def _tensor_bundle(size: int = 9, seed: int = 5):
    """A minimal two-record tensor bundle for direct loss tests."""
    t = torch
    rng = np.random.default_rng(seed)
    target_g = _softmax(rng.standard_normal(size))
    target_p = _softmax(rng.standard_normal(size))
    grounding = {
        "group_mean": t.as_tensor(rng.standard_normal((size, 1024)).astype(np.float32)),
        "prompt_embeddings": t.as_tensor(
            rng.standard_normal((4, 2560)).astype(np.float32)
        ),
        "spatial_target": t.as_tensor(target_g.astype(np.float32)),
        "teacher_soft": t.as_tensor(target_g.astype(np.float32)),
    }
    pointing = {
        "group_mean": t.as_tensor(rng.standard_normal((size, 1024)).astype(np.float32)),
        "prompt_embeddings": t.as_tensor(
            rng.standard_normal((4, 2560)).astype(np.float32)
        ),
        "spatial_target": t.as_tensor(target_p.astype(np.float32)),
        "teacher_soft": t.as_tensor(target_p.astype(np.float32)),
    }
    return {
        "g": grounding,
        "p": pointing,
        "target_g": target_g,
        "target_p": target_p,
    }


def _fake_arm_stats(seed: int, images, values: dict) -> dict:
    """Minimal val_statistics-shaped dict for aggregate_validation tests."""
    metric_keys = (
        "correct_spearman",
        "correct_ndcg_full",
        "same_spearman",
        "same_ndcg_full",
        "cross_spearman",
        "cross_ndcg_full",
        "selector_mass_k48",
        "l2_anchor_mass_k48",
    )
    image_metrics = {
        key: {image: float(values[key]) for image in images} for key in metric_keys
    }
    return {
        "count": len(images) * 2,
        "independent_image_units": len(images),
        "correct_spearman": float(values["correct_spearman"]),
        "correct_ndcg_full": float(values["correct_ndcg_full"]),
        "same_spearman": float(values["same_spearman"]),
        "same_ndcg_full": float(values["same_ndcg_full"]),
        "cross_spearman": float(values["cross_spearman"]),
        "cross_ndcg_full": float(values["cross_ndcg_full"]),
        "exact_prompt_invariance_fraction": 0.0,
        "tolerance_prompt_invariance_fraction": 0.0,
        "image_metrics": image_metrics,
        "rows": [],
    }


def _pass_delta(value: float = 0.1) -> dict:
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


def _pass_noninferior(value: float = 0.0) -> dict:
    return {
        "mean": value,
        "ci_low": -0.01,
        "ci_high": 0.01,
        "cohens_dz": 0.05,
        "p_value_two_sided": 0.5,
        "holm_adjusted_p": None,
        "positive_seed_count": 2,
        "seed_count": 3,
        "noninferior_seed_count": 3,
        "noninferior_seed_min": 2,
    }


def _passing_validation() -> dict:
    return {
        "arms": {
            trainer.PRIMARY_ARM: {
                "collapse": {"max_exact_prompt_invariance_fraction": 0.0}
            }
        },
        "primary_tests": {
            "P1_B_minus_A_correct_minus_same_spearman": _pass_delta(),
            "P2_B_minus_A_correct_minus_same_ndcg_full": _pass_delta(),
            "P3_B_correct_minus_same_spearman": _pass_delta(),
            "P4_B_correct_minus_same_ndcg_full": _pass_delta(),
            "P5_B_selector_minus_L2_anchor_mass_k48": _pass_delta(),
        },
        "correct_alignment_noninferiority": {
            "correct_spearman": _pass_noninferior(),
            "correct_ndcg_full": _pass_noninferior(),
        },
    }


def _passing_training() -> dict:
    return {
        "per_seed": [
            {"best_epoch": 1, "best_val_total": 0.5, "epoch0_val_total": 0.6},
            {"best_epoch": 2, "best_val_total": 0.4, "epoch0_val_total": 0.5},
            {"best_epoch": 0, "best_val_total": 0.6, "epoch0_val_total": 0.5},
        ]
    }


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class ContrastiveTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="contrastive_tests_")
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

    # -- arm targets and exact loss references -----------------------------
    def test_both_arms_use_the_frozen_spatial_target(self) -> None:
        manifest_by_id = {
            record["sample_id"]: record for record in self.data["manifest"]
        }
        sample = {
            "sample_id": "g0",
            "task": "grounding",
            "group_count": GRID * GRID,
            "grid_h": GRID,
            "grid_w": GRID,
            "teacher_soft": np.full(GRID * GRID, 1.0 / (GRID * GRID)),
        }
        manifest_by_id["g0"] = {
            "sample_id": "g0",
            "task": "grounding",
            "ground_truth": {
                "bbox_1000": [0.0, 0.0, 500.0, 1000.0],
                "point_1000": [1.0, 1.0],
            },
        }
        attached = trainer.attach_spatial_targets([sample], manifest_by_id)[0]
        expected = grounding_area_target([0.0, 0.0, 500.0, 1000.0], GRID, GRID)
        np.testing.assert_allclose(attached["spatial_target"], expected)
        self.assertFalse(
            np.allclose(attached["spatial_target"], sample["teacher_soft"])
        )
        # Both arms train on the spatial target, never the semantic teacher target.
        t = torch
        rng = np.random.default_rng(101)
        spatial_g = _softmax(rng.standard_normal(GRID * GRID))
        spatial_p = _softmax(rng.standard_normal(GRID * GRID))
        uniform = np.full(GRID * GRID, 1.0 / (GRID * GRID))
        tensors = {
            "g": {
                "group_mean": t.as_tensor(
                    rng.standard_normal((GRID * GRID, 1024)).astype(np.float32)
                ),
                "prompt_embeddings": t.as_tensor(
                    rng.standard_normal((4, 2560)).astype(np.float32)
                ),
                "spatial_target": t.as_tensor(spatial_g.astype(np.float32)),
                "teacher_soft": t.as_tensor(uniform.astype(np.float32)),
            },
            "p": {
                "group_mean": t.as_tensor(
                    rng.standard_normal((GRID * GRID, 1024)).astype(np.float32)
                ),
                "prompt_embeddings": t.as_tensor(
                    rng.standard_normal((4, 2560)).astype(np.float32)
                ),
                "spatial_target": t.as_tensor(spatial_p.astype(np.float32)),
                "teacher_soft": t.as_tensor(uniform.astype(np.float32)),
            },
        }
        Wt = t.nn.Parameter(t.zeros((128, 2560)))
        Wv = t.nn.Parameter(t.zeros((128, 1024)))
        components = trainer.pair_loss_components(
            tensors, {"grounding_id": "g", "pointing_id": "p"}, Wt, Wv
        )
        expected_base = 0.5 * (
            _kl_nats(spatial_g, uniform) + _kl_nats(spatial_p, uniform)
        )
        # the semantic teacher target would give base = 0 here
        self.assertAlmostEqual(float(components["base"]), expected_base, delta=1e-4)
        self.assertGreater(expected_base, 1e-3)
        for arm in (trainer.CONTROL_ARM, trainer.PRIMARY_ARM):
            total = float(trainer.arm_total(components, arm))
            self.assertGreater(total, 1e-3)

    def test_arm_a_loss_is_base_reference(self) -> None:
        t = torch
        bundle = _tensor_bundle(size=9, seed=11)
        tensors = {"g": bundle["g"], "p": bundle["p"]}
        pair = {"grounding_id": "g", "pointing_id": "p"}
        Wt = t.nn.Parameter(t.zeros((128, 2560)))
        Wv = t.nn.Parameter(t.zeros((128, 1024)))
        components = trainer.pair_loss_components(tensors, pair, Wt, Wv)
        uniform = np.full(9, 1.0 / 9)
        expected_base = 0.5 * (
            _kl_nats(bundle["target_g"], uniform)
            + _kl_nats(bundle["target_p"], uniform)
        )
        self.assertAlmostEqual(float(components["base"]), expected_base, delta=5e-4)
        # zero scores make correct == wrong, so softplus(0)=ln2 each.
        self.assertAlmostEqual(
            float(components["contrastive"]), float(np.log(2.0)), delta=5e-5
        )
        self.assertAlmostEqual(
            float(trainer.arm_total(components, trainer.CONTROL_ARM)),
            expected_base,
            delta=5e-4,
        )
        self.assertAlmostEqual(
            float(trainer.arm_total(components, trainer.PRIMARY_ARM)),
            expected_base + float(np.log(2.0)),
            delta=5e-4,
        )

    def test_arm_b_total_is_base_plus_weighted_contrastive(self) -> None:
        t = torch
        rng = np.random.default_rng(21)
        Wt = t.nn.Parameter(
            t.as_tensor(rng.standard_normal((128, 2560)).astype(np.float32)) * 0.02
        )
        Wv = t.nn.Parameter(
            t.as_tensor(rng.standard_normal((128, 1024)).astype(np.float32)) * 0.02
        )
        bundle = _tensor_bundle(size=9, seed=7)
        tensors = {"g": bundle["g"], "p": bundle["p"]}
        pair = {"grounding_id": "g", "pointing_id": "p"}
        components = trainer.pair_loss_components(tensors, pair, Wt, Wv)
        self.assertAlmostEqual(trainer.CONTRASTIVE_WEIGHT, 1.0)
        self.assertAlmostEqual(trainer.CONTRASTIVE_MARGIN, 0.0)
        self.assertAlmostEqual(trainer.SOFTPLUS_TEMPERATURE, 1.0)
        self.assertAlmostEqual(
            float(trainer.arm_total(components, trainer.PRIMARY_ARM)),
            float(components["base"] + components["contrastive"]),
            delta=1e-6,
        )
        self.assertAlmostEqual(
            float(trainer.arm_total(components, trainer.CONTROL_ARM)),
            float(components["base"]),
            delta=1e-6,
        )
        # the contrastive standard reference recomputed independently
        reference = 0.5 * (
            float(
                t.logaddexp(
                    components["kl_g_correct"] - components["kl_g_wrong"],
                    t.zeros(()),
                )
            )
            + float(
                t.logaddexp(
                    components["kl_p_correct"] - components["kl_p_wrong"],
                    t.zeros(()),
                )
            )
        )
        self.assertAlmostEqual(
            float(components["contrastive"]), reference, delta=1e-6
        )

    def test_wrong_prompt_is_same_image_other_task(self) -> None:
        t = torch
        rng = np.random.default_rng(31)
        Wt = t.nn.Parameter(
            t.as_tensor(rng.standard_normal((128, 2560)).astype(np.float32)) * 0.05
        )
        Wv = t.nn.Parameter(
            t.as_tensor(rng.standard_normal((128, 1024)).astype(np.float32)) * 0.05
        )
        bundle = _tensor_bundle(size=9, seed=17)
        # add a decoy third prompt that must never be used by the exchange
        decoy_prompt = t.as_tensor(
            rng.standard_normal((4, 2560)).astype(np.float32)
        )
        tensors = {"g": bundle["g"], "p": bundle["p"]}
        pair = {"grounding_id": "g", "pointing_id": "p"}
        components = trainer.pair_loss_components(tensors, pair, Wt, Wv)
        from evaluation.pair_loss_feasibility_audit import student_scores

        expected_g_wrong = trainer.record_kl(
            student_scores(
                bundle["g"]["group_mean"],
                bundle["p"]["prompt_embeddings"],
                Wt,
                Wv,
            ),
            bundle["g"]["spatial_target"],
        )
        expected_p_wrong = trainer.record_kl(
            student_scores(
                bundle["p"]["group_mean"],
                bundle["g"]["prompt_embeddings"],
                Wt,
                Wv,
            ),
            bundle["p"]["spatial_target"],
        )
        expected_decoy = trainer.record_kl(
            student_scores(bundle["g"]["group_mean"], decoy_prompt, Wt, Wv),
            bundle["g"]["spatial_target"],
        )
        self.assertAlmostEqual(
            float(components["kl_g_wrong"]), float(expected_g_wrong), places=6
        )
        self.assertAlmostEqual(
            float(components["kl_p_wrong"]), float(expected_p_wrong), places=6
        )
        # the exchange is a real same-image swap, not a no-op and not a donor
        self.assertGreater(
            abs(float(components["kl_g_correct"]) - float(components["kl_g_wrong"])),
            1e-8,
        )
        self.assertGreater(
            abs(float(components["kl_g_wrong"]) - float(expected_decoy)), 1e-8
        )

    # -- shared epoch-0 state and deterministic batches --------------------
    def test_shared_epoch0_state_and_batch_order(self) -> None:
        hashes = self.bundle["epoch0_hashes"]["20261003"]
        self.assertEqual(len(hashes), 2)
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

    # -- val-only checkpoint selection -------------------------------------
    def test_val_only_checkpoint_selection(self) -> None:
        for arm_name, training in self.bundle["training"].items():
            for entry in training["per_seed"]:
                values = [
                    trainer.arm_selection_value(row["val"], arm_name)
                    for row in entry["epoch_metrics"]
                ]
                self.assertEqual(entry["best_epoch"], trainer.select_best_epoch(values))
                self.assertAlmostEqual(
                    entry["best_val_selection"], min(values), places=9
                )
                self.assertEqual(entry["epoch_metrics"][0]["phase"], "pre_update")
        self.assertEqual(
            self.bundle["training"][trainer.CONTROL_ARM]["selection_metrics"],
            ["val_base"],
        )
        self.assertEqual(
            self.bundle["training"][trainer.PRIMARY_ARM]["selection_metrics"],
            ["val_total"],
        )

    # -- five-test Holm family and P5 sign ---------------------------------
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

    def test_p5_uses_selector_minus_baseline_sign(self) -> None:
        images = ["a", "b", "c"]
        good = {
            "correct_spearman": 0.5,
            "correct_ndcg_full": 0.5,
            "same_spearman": 0.1,
            "same_ndcg_full": 0.1,
            "cross_spearman": 0.0,
            "cross_ndcg_full": 0.0,
            "selector_mass_k48": 0.8,
            "l2_anchor_mass_k48": 0.2,
        }
        per_arm = {
            trainer.CONTROL_ARM: {20261003: _fake_arm_stats(20261003, images, good)},
            trainer.PRIMARY_ARM: {20261003: _fake_arm_stats(20261003, images, good)},
        }
        aggregated = trainer.aggregate_validation(
            per_arm, arm_names=trainer.ARM_ORDER, seeds=[20261003]
        )
        p5 = aggregated["primary_tests"]["P5_B_selector_minus_L2_anchor_mass_k48"]
        self.assertAlmostEqual(float(p5["mean"]), 0.6, places=9)
        # direct helper check: the delta is metric minus base, never the reverse
        test = {
            "kind": "selector_minus_baseline",
            "metric": "selector_mass_k48",
            "base": "l2_anchor_mass_k48",
        }
        deltas = trainer._per_seed_deltas(
            test,
            per_arm[trainer.CONTROL_ARM],
            per_arm[trainer.PRIMARY_ARM],
            20261003,
            images,
        )
        self.assertAlmostEqual(deltas["a"], 0.6, places=12)
        # flipping the two masses flips the sign of P5
        bad = dict(good)
        bad["selector_mass_k48"] = 0.2
        bad["l2_anchor_mass_k48"] = 0.8
        per_arm_bad = {
            trainer.CONTROL_ARM: {20261003: _fake_arm_stats(20261003, images, bad)},
            trainer.PRIMARY_ARM: {20261003: _fake_arm_stats(20261003, images, bad)},
        }
        aggregated_bad = trainer.aggregate_validation(
            per_arm_bad, arm_names=trainer.ARM_ORDER, seeds=[20261003]
        )
        self.assertLess(
            float(
                aggregated_bad["primary_tests"][
                    "P5_B_selector_minus_L2_anchor_mass_k48"
                ]["mean"]
            ),
            0.0,
        )

    def test_val_sample_p5_column_matches_selector_minus_baseline(self) -> None:
        by_key = {
            (row["arm"], row["seed"], row["sample_id"]): row
            for row in self.bundle["val_sample_rows"]
        }
        self.assertTrue(by_key)
        for row in by_key.values():
            self.assertAlmostEqual(
                row["selector_minus_l2_anchor_mass_k48"],
                row["selector_mass_k48"] - row["l2_anchor_mass_k48"],
                places=12,
            )

    # -- gates G1-G7 --------------------------------------------------------
    def test_all_gates_pass_and_decision(self) -> None:
        gate = trainer.evaluate_gates(
            validation=_passing_validation(), training=_passing_training()
        )
        self.assertTrue(gate["all_pass"], gate)
        for key in (
            "G1_correct_alignment_non_inferiority",
            "G2_prompt_discrimination_gain",
            "G3_prompt_discrimination_absolute",
            "G4_deployed_K48_utility",
            "G5_collapse_removed",
            "G6_validation_learning",
            "G7_hardware_contract",
        ):
            self.assertTrue(gate[key]["pass"], key)
        decision = trainer.decide(gate)
        self.assertEqual(
            decision["decision"], "PROMPT_CONTRASTIVE_STUDENT_ESTABLISHED"
        )

    def test_each_gate_failure_flips_the_decision(self) -> None:
        # G1
        validation = _passing_validation()
        validation["correct_alignment_noninferiority"]["correct_spearman"][
            "mean"
        ] = -0.03
        gate = trainer.evaluate_gates(
            validation=validation, training=_passing_training()
        )
        self.assertFalse(gate["G1_correct_alignment_non_inferiority"]["pass"])
        self.assertFalse(gate["all_pass"])
        # G2
        validation = _passing_validation()
        validation["primary_tests"]["P1_B_minus_A_correct_minus_same_spearman"][
            "mean"
        ] = -0.1
        gate = trainer.evaluate_gates(
            validation=validation, training=_passing_training()
        )
        self.assertFalse(gate["G2_prompt_discrimination_gain"]["pass"])
        # G3
        validation = _passing_validation()
        validation["primary_tests"]["P3_B_correct_minus_same_spearman"][
            "ci_low"
        ] = 0.0
        gate = trainer.evaluate_gates(
            validation=validation, training=_passing_training()
        )
        self.assertFalse(gate["G3_prompt_discrimination_absolute"]["pass"])
        # G4
        validation = _passing_validation()
        validation["primary_tests"]["P5_B_selector_minus_L2_anchor_mass_k48"][
            "holm_adjusted_p"
        ] = 0.5
        gate = trainer.evaluate_gates(
            validation=validation, training=_passing_training()
        )
        self.assertFalse(gate["G4_deployed_K48_utility"]["pass"])
        # G5
        validation = _passing_validation()
        validation["arms"][trainer.PRIMARY_ARM]["collapse"][
            "max_exact_prompt_invariance_fraction"
        ] = 0.5
        gate = trainer.evaluate_gates(
            validation=validation, training=_passing_training()
        )
        self.assertFalse(gate["G5_collapse_removed"]["pass"])
        # G6: no seed improves
        validation = _passing_validation()
        training = {
            "per_seed": [
                {"best_epoch": 0, "best_val_total": 0.6, "epoch0_val_total": 0.5},
                {"best_epoch": 0, "best_val_total": 0.6, "epoch0_val_total": 0.5},
                {"best_epoch": 0, "best_val_total": 0.6, "epoch0_val_total": 0.5},
            ]
        }
        gate = trainer.evaluate_gates(validation=validation, training=training)
        self.assertFalse(gate["G6_validation_learning"]["pass"])
        # G7: force a broken hardware contract
        broken = dict(trainer.hardware_contract())
        broken["query_shape_ok"] = False
        with mock.patch.object(
            trainer, "hardware_contract", return_value=broken
        ):
            gate = trainer.evaluate_gates(
                validation=_passing_validation(), training=_passing_training()
            )
        self.assertFalse(gate["G7_hardware_contract"]["pass"])
        # missing primary arm never wins
        self.assertEqual(
            trainer.decide(None)["decision"],
            "PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED",
        )

    def test_control_arm_can_never_win(self) -> None:
        self.assertFalse(trainer.ARM_SPEC[trainer.CONTROL_ARM]["can_win"])
        self.assertTrue(trainer.ARM_SPEC[trainer.PRIMARY_ARM]["can_win"])
        self.assertNotEqual(trainer.PRIMARY_ARM, trainer.CONTROL_ARM)

    def test_hardware_contract_is_frozen(self) -> None:
        hardware = trainer.hardware_contract()
        self.assertEqual(hardware["query_shape"], [1, 128])
        self.assertEqual(hardware["fpga_scorer"], "Gx128 dot")
        self.assertEqual(hardware["fpga_scorer_dot_mac_per_group"], 128)
        self.assertEqual(hardware["fpga_scorer_compare_per_group"], 0)
        self.assertEqual(hardware["fpga_scorer_exp_per_group"], 0)
        self.assertFalse(hardware["token_axis_max_softmax_exp"])
        self.assertEqual(trainer.ANCHOR_REGION_GRID, 3)

    # -- internal-test NPZ isolation ---------------------------------------
    def test_isolation_opened_only_train_val(self) -> None:
        self.assertEqual(self.bundle["isolation"]["test_npz_opened"], [])
        self.assertEqual(self.bundle["isolation"]["test_npz_opened_count"], 0)
        self.assertEqual(self.bundle["isolation"]["test_npz_hashed_count"], 0)
        self.assertFalse(self.bundle["isolation"]["test_npz_resolved_or_opened"])
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
        with tempfile.TemporaryDirectory(prefix="contrastive_corrupt_") as tmp_name:
            tmp = Path(tmp_name)
            data = build_synthetic_dataset(tmp, n_images=20, seed=3)
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
                salt="synthetic_contrastive_test",
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

    # -- prereg immutability -------------------------------------------------
    def test_prereg_immutability(self) -> None:
        before = (
            _sha256(REAL_PREREG),
            _sha256(REAL_PREREG_REPORT),
            _sha256(REAL_PROMPT),
            _sha256(EXISTING_SPATIAL_TRAINER),
        )
        verification = trainer.verify_frozen_inputs()
        after = (
            _sha256(REAL_PREREG),
            _sha256(REAL_PREREG_REPORT),
            _sha256(REAL_PROMPT),
            _sha256(EXISTING_SPATIAL_TRAINER),
        )
        self.assertEqual(before, after)
        self.assertTrue(all(verification["checks"].values()), verification["checks"])
        self.assertEqual(verification["checks"]["prereg_report_pins_config"], True)
        self.assertEqual(
            verification["checks"]["prereg_report_pins_harness_prompt"], True
        )
        report = json.loads(REAL_PREREG_REPORT.read_text(encoding="utf-8"))
        self.assertEqual(report["preregistered_config"]["sha256"], before[0])

    # -- checkpoint / report schemas ---------------------------------------
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
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(payload["arm"], arm_name)
            self.assertEqual(payload["loss"], trainer.ARM_SPEC[arm_name]["loss"])
            self.assertIn("best_val_base", payload)
            self.assertIn("best_val_contrastive", payload)
            self.assertIn("best_val_total", payload)
        arm_dir = Path(
            self.bundle["training"][trainer.PRIMARY_ARM]["per_seed"][0][
                "best_checkpoint"
            ]
        ).parent
        self.assertEqual(
            sorted(path.name for path in arm_dir.iterdir()),
            ["best.npz", "final.npz", "metadata.json"],
        )

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
            "contrastive",
            "development_validation_only",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["validation_statistics"]["holm_family_size"], 5)
        self.assertEqual(payload["contrastive"]["weight"], 1.0)
        self.assertEqual(payload["contrastive"]["margin"], 0.0)
        self.assertEqual(payload["contrastive"]["softplus_temperature"], 1.0)
        header = (
            Path(paths["val_samples_csv"]).read_text(encoding="utf-8").splitlines()[0]
        )
        for column in (
            "arm",
            "seed",
            "sample_id",
            "correct_spearman",
            "correct_ndcg_full",
            "same_spearman",
            "selector_mass_k48",
            "l2_anchor_mass_k48",
            "selector_minus_l2_anchor_mass_k48",
            "exact_prompt_invariant",
        ):
            self.assertIn(column, header)
        epoch_header = (
            Path(paths["epochs_csv"]).read_text(encoding="utf-8").splitlines()[0]
        )
        for column in ("arm", "seed", "epoch", "split", "base", "contrastive", "total"):
            self.assertIn(column, epoch_header)
        markdown = Path(paths["md"]).read_text(encoding="utf-8")
        self.assertIn("G1", markdown)
        self.assertIn("development validation only", markdown)


if __name__ == "__main__":
    unittest.main()
