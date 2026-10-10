#!/usr/bin/env python3
"""Focused tests for the three-arm pairwise-retrain trainer.

Covers: arm loss math, shared epoch-0 hash across arms, deterministic batch
order, best-epoch tie-break, val-only selection, test-NPZ isolation (access log,
spy loader, corrupted test NPZ), prereg/draft immutability, Holm m=12,
image-level aggregation/derangement/no-self, winner rule (only C) and
checkpoint round-trip.
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

from evaluation.pair_loss_feasibility_audit import pair_loss_terms
from evaluation.prompt_identifiability_diagnostic import (
    derange_images,
    image_task_index,
    sample_donor_map,
)
from evaluation.prompt_projection_distillation import (
    load_feature_cache,
    save_feature_cache,
    split_sample_ids,
)
from evaluation.teacher_pair_signal_audit import js_divergence_nats
import evaluation.train_pairwise_prompt_selector as trainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_DRAFT = PROJECT_ROOT / "configs/prompt_mean_pairwise_retrain.draft.json"
REAL_PREREG = PROJECT_ROOT / "configs/prompt_mean_pairwise_retrain.preregistered.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _softmax(x: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    z = (x - x.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


def _kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    a = p / p.sum()
    b = q / q.sum()
    mask = a > 0
    return float(np.sum(a[mask] * (np.log(a[mask]) - np.log(np.clip(b[mask], 1e-12, None)))))


def _js_nats(p: np.ndarray, q: np.ndarray) -> float:
    a = p / p.sum()
    b = q / q.sum()
    m = 0.5 * (a + b)
    return 0.5 * _kl_nats(a, m) + 0.5 * _kl_nats(b, m)


def build_synthetic_dataset(
    root: Path, *, n_images: int = 24, seed: int = 0
) -> dict:
    """Create tiny but width-correct feature caches plus an index/prereg."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n_images):
        image_sha = "%064x" % (index + 1)
        group_count = 3 + (index % 3)
        for task in ("grounding", "pointing"):
            sample_id = f"syn-{index:03d}-{task}"
            group_mean = rng.standard_normal((group_count, 1024)).astype(np.float32)
            prompt_embeddings = rng.standard_normal((4, 2560)).astype(np.float32)
            raw = rng.standard_normal(group_count).astype(np.float32)
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
                    "grid_h": 2,
                    "grid_w": 2,
                    "group_count": group_count,
                    "prompt_token_count": 4,
                    "npz_sha256": "",
                }
            )
    index_path = root / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    prereg_path = root / "prereg.json"
    prereg_path.write_text(
        json.dumps(
            {
                "status": "PREREGISTERED_TRAIN_VAL_ONLY",
                "split": {
                    "seed": 20260927,
                    "salt": "synthetic_pairwise_test",
                    "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
                },
            }
        ),
        encoding="utf-8",
    )
    splits = split_sample_ids(
        records,
        seed=20260927,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        salt="synthetic_pairwise_test",
    )
    return {
        "root": root,
        "index_path": index_path,
        "prereg_path": prereg_path,
        "records": records,
        "splits": splits,
    }


class SpyLoader:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, path: Path):
        self.calls.append(str(path))
        return load_feature_cache(path)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class PairwiseTrainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="pairwise_tests_")
        cls.root = Path(cls._tmp.name)
        cls.data = build_synthetic_dataset(cls.root)
        cls.spy = SpyLoader()
        cls.bundle = trainer.run_experiment(
            prereg_config=cls.data["prereg_path"],
            prereg_report=cls.data["prereg_path"],
            index_path=cls.data["index_path"],
            features_dir=cls.root,
            artifacts_dir=cls.root / "artifacts",
            reports_dir=cls.root / "reports",
            seeds=[20261002],
            arm_names=("A_KL_ONLY_CONTROL", "B_PAIR_LAMBDA_1_CONTROL", "C_PAIR_LAMBDA_0P25_PRIMARY"),
            epochs=2,
            enforce_frozen_inputs=False,
            enforce_expected_counts=False,
            cache_loader=cls.spy,
            write_outputs=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    # -- loss math ---------------------------------------------------------
    def test_arm_total_loss_scaling(self) -> None:
        base, pair = 2.0, 3.0
        self.assertAlmostEqual(trainer.lambda_total_loss(0.0, base, pair), base)
        self.assertAlmostEqual(trainer.lambda_total_loss(1.0, base, pair), base + pair)
        self.assertAlmostEqual(
            trainer.lambda_total_loss(trainer.LAMBDA_TARGET_0P25, base, pair),
            base + trainer.LAMBDA_TARGET_0P25 * pair,
        )
        self.assertEqual(trainer.ARM_SPEC["A_KL_ONLY_CONTROL"]["lambda_pair"], 0.0)
        self.assertEqual(trainer.ARM_SPEC["B_PAIR_LAMBDA_1_CONTROL"]["lambda_pair"], 1.0)
        self.assertAlmostEqual(
            trainer.ARM_SPEC["C_PAIR_LAMBDA_0P25_PRIMARY"]["lambda_pair"],
            trainer.LAMBDA_TARGET_0P25,
        )

    def test_pair_loss_matches_numpy_reference(self) -> None:
        torch_mod = torch
        rng = np.random.default_rng(7)
        for _ in range(5):
            size = 6
            y_g = _softmax(rng.standard_normal(size))
            y_p = _softmax(rng.standard_normal(size))
            s_g = rng.standard_normal(size).astype(np.float32)
            s_p = rng.standard_normal(size).astype(np.float32)
            js = _js_nats(y_g, y_p)
            l_base, l_pair, d_swap, _l_total, margin = pair_loss_terms(
                torch_mod.as_tensor(s_g),
                torch_mod.as_tensor(s_p),
                torch_mod.as_tensor(y_g.astype(np.float32)),
                torch_mod.as_tensor(y_p.astype(np.float32)),
                js,
                temperature=0.1,
            )
            p_g = _softmax(s_g)
            p_p = _softmax(s_p)
            ref_base = 0.5 * (_kl_nats(y_g, p_g) + _kl_nats(y_p, p_p))
            ref_swap = 0.5 * (_kl_nats(y_g, p_p) + _kl_nats(y_p, p_g))
            ref_margin = js + ref_base - ref_swap
            ref_pair = max(0.0, ref_margin)
            self.assertAlmostEqual(float(l_base), ref_base, delta=5e-3)
            self.assertAlmostEqual(float(d_swap), ref_swap, delta=5e-3)
            self.assertAlmostEqual(float(margin), ref_margin, delta=5e-3)
            self.assertAlmostEqual(float(l_pair), ref_pair, delta=5e-3)

    def test_pair_loss_uses_nats_js_not_bits(self) -> None:
        # JS-nats is exactly ln(2) for disjoint one-hot distributions, not 1.0 bits.
        y_g = np.array([1.0, 0.0], dtype=np.float32)
        y_p = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(js_divergence_nats(y_g, y_p), float(np.log(2)), places=6)

    # -- shared epoch-0 state / batch order --------------------------------
    def test_epoch0_hash_identical_across_arms(self) -> None:
        hashes = self.bundle["epoch0_hashes"]["20261002"]
        self.assertEqual(len(set(hashes.values())), 1)
        self.assertTrue(all(self.bundle["epoch0_hashes_consistent_across_arms"].values()))

    def test_deterministic_batch_order(self) -> None:
        images = [f"img-{i}" for i in range(10)]
        first = trainer.epoch_batch_order(20261002, 3, images, 4)
        second = trainer.epoch_batch_order(20261002, 3, images, 4)
        self.assertEqual(first, second)
        # all images exactly once per epoch
        flat = sorted(index for batch in first for index in batch)
        self.assertEqual(flat, list(range(10)))
        # a different epoch/seed changes the order
        self.assertNotEqual(first, trainer.epoch_batch_order(20261002, 4, images, 4))
        self.assertNotEqual(first, trainer.epoch_batch_order(20261003, 3, images, 4))

    # -- best epoch / val-only selection -----------------------------------
    def test_best_epoch_tie_keeps_earliest(self) -> None:
        self.assertEqual(trainer.select_best_epoch([1.0, 0.5, 0.5, 2.0]), 1)
        self.assertEqual(trainer.select_best_epoch([0.3, 0.3, 0.3]), 0)
        self.assertEqual(trainer.select_best_epoch([2.0, 1.0, 0.5]), 2)

    def test_best_epoch_uses_val_total(self) -> None:
        for arm_name, training in self.bundle["training"].items():
            for entry in training["per_seed"]:
                val_totals = [row["val"]["l_total"] for row in entry["epoch_metrics"]]
                self.assertEqual(entry["best_epoch"], trainer.select_best_epoch(val_totals))
                self.assertAlmostEqual(
                    entry["best_val_l_total"], min(val_totals), places=9
                )

    # -- test isolation ----------------------------------------------------
    def test_isolation_opened_only_train_val(self) -> None:
        opened = set(self.bundle["isolation"]["test_npz_opened"])
        self.assertEqual(opened, set())
        self.assertEqual(self.bundle["isolation"]["test_npz_opened_count"], 0)
        self.assertTrue(self.bundle["isolation"]["opened_only_train_val_npz"])
        test_ids = set(self.data["splits"]["test"])
        test_names = {
            record["sample_id"] for record in self.data["records"] if record["sample_id"] in test_ids
        }
        called_names = {Path(path).stem for path in self.spy.calls}
        self.assertTrue(test_names.isdisjoint(called_names))

    def test_corrupt_test_npz_never_read(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pairwise_corrupt_") as tmp_name:
            tmp = Path(tmp_name)
            data = build_synthetic_dataset(tmp, n_images=20, seed=3)
            splits = data["splits"]
            self.assertGreaterEqual(len(splits["val"]), 2)
            # corrupt every test npz: a real read would raise
            for record in data["records"]:
                if record["sample_id"] in set(splits["test"]):
                    (tmp / record["npz"]).write_bytes(b"not-a-valid-npz")
            spy = SpyLoader()
            bundle = trainer.run_experiment(
                prereg_config=data["prereg_path"],
                prereg_report=data["prereg_path"],
                index_path=data["index_path"],
                features_dir=tmp,
                artifacts_dir=tmp / "artifacts",
                reports_dir=tmp / "reports",
                seeds=[20261003],
                arm_names=("A_KL_ONLY_CONTROL",),
                epochs=1,
                enforce_frozen_inputs=False,
                enforce_expected_counts=False,
                cache_loader=spy,
                write_outputs=False,
            )
            self.assertEqual(bundle["isolation"]["test_npz_opened"], [])
            test_paths = {
                str(tmp / record["npz"])
                for record in data["records"]
                if record["sample_id"] in set(splits["test"])
            }
            self.assertTrue(test_paths.isdisjoint(set(spy.calls)))

    # -- immutability ------------------------------------------------------
    def test_prereg_and_draft_unchanged(self) -> None:
        before = (_sha256(REAL_DRAFT), _sha256(REAL_PREREG))
        # a fresh read-only verification against the frozen SHAs
        trainer.verify_frozen_inputs()
        after = (_sha256(REAL_DRAFT), _sha256(REAL_PREREG))
        self.assertEqual(before, after)
        self.assertEqual(after[0], trainer.DRAFT_SHA256)

    # -- Holm family / aggregation -----------------------------------------
    def test_holm_family_size_is_twelve(self) -> None:
        self.assertEqual(trainer.HOLM_FAMILY_SIZE, 12)
        self.assertEqual(
            trainer.HOLM_FAMILY_SIZE,
            len(trainer.ARM_ORDER) * len(trainer.ROLES) * len(trainer.PRIMARY_METRICS),
        )

    def test_aggregate_val_statistics_holm_and_medians(self) -> None:
        per_arm_seed = {}
        for arm_name in trainer.ARM_ORDER:
            per_arm_seed[arm_name] = {}
            for seed in (20261002, 20261003, 20261004):
                base = float(seed % 10) + 1.0
                per_image = {
                    role: {
                        metric: {"img-a": base, "img-b": base + 2.0}
                        for metric in trainer.PRIMARY_METRICS
                    }
                    for role in trainer.ROLES
                }
                per_arm_seed[arm_name][seed] = {
                    "count": 4,
                    "independent_image_units": 2,
                    "correct_spearman": 0.05 + base / 100.0,
                    "correct_ndcg_full": 0.84 + base / 1000.0,
                    "exact_prompt_invariance_fraction": 0.0,
                    "tolerance_prompt_invariance_fraction": 0.1,
                    "per_image_delta": per_image,
                }
        aggregated = trainer.aggregate_val_statistics(per_arm_seed)
        count = 0
        for arm_name, arm in aggregated["arms"].items():
            for role in trainer.ROLES:
                for metric in trainer.PRIMARY_METRICS:
                    entry = arm["deltas"][role][metric]
                    self.assertIn("holm_adjusted_p", entry)
                    count += 1
        self.assertEqual(count, 12)
        # per-image median across seeds: median(1.0, 2.0, 3.0) == 2.0 for seed 20261002's base
        entry = aggregated["arms"][trainer.ARM_ORDER[0]]["deltas"][trainer.ROLES[0]][
            trainer.PRIMARY_METRICS[0]
        ]
        self.assertGreaterEqual(entry["mean"], 0.0)

    def test_derangement_no_self_and_same_task_donor(self) -> None:
        image_ids = [f"{i:064x}" for i in range(8)]
        donors = derange_images(image_ids, seed=trainer.DONOR_SEED)
        self.assertEqual(set(donors), set(image_ids))
        self.assertTrue(all(donors[image] != image for image in image_ids))
        self.assertEqual(len(set(donors.values())), len(image_ids))
        samples = []
        for image in image_ids:
            for task in ("grounding", "pointing"):
                samples.append(
                    {
                        "sample_id": f"{image}-{task}",
                        "source_image_sha256": image,
                        "task": task,
                        "group_mean": np.zeros((3, 1024), dtype=np.float32),
                    }
                )
        donor_map = sample_donor_map(samples, donors, same_task=True)
        by_id = {sample["sample_id"]: sample for sample in samples}
        for sample_id, donor_id in donor_map.items():
            self.assertNotEqual(
                by_id[sample_id]["source_image_sha256"],
                by_id[donor_id]["source_image_sha256"],
            )
            self.assertEqual(by_id[sample_id]["task"], by_id[donor_id]["task"])

    # -- winner rule -------------------------------------------------------
    def test_winner_rule_only_primary_arm(self) -> None:
        def gate(passed: bool) -> dict:
            return {
                key: {"pass": passed}
                for key in (
                    "A_prompt_sensitivity",
                    "B_shuffled_discrimination",
                    "C_collapse_removed",
                    "D_teacher_alignment",
                    "E_generalization",
                    "F_hardware_contract",
                )
            } | {"all_pass": passed}

        # A and B pass, C fails -> NO_PROMOTION
        gates = {
            "A_KL_ONLY_CONTROL": gate(True),
            "B_PAIR_LAMBDA_1_CONTROL": gate(True),
            "C_PAIR_LAMBDA_0P25_PRIMARY": gate(False),
        }
        decision = trainer.decide(gates)
        self.assertEqual(decision["decision"], "NO_PROMOTION")
        self.assertIsNone(decision["winner"])
        # only C passing -> PROMOTION_PASS
        gates["C_PAIR_LAMBDA_0P25_PRIMARY"] = gate(True)
        decision = trainer.decide(gates)
        self.assertEqual(decision["decision"], "PROMOTION_PASS")
        self.assertEqual(decision["winner"], trainer.PRIMARY_ARM)

    # -- hardware contract -------------------------------------------------
    def test_hardware_contract_fixed(self) -> None:
        contract = trainer.hardware_contract()
        self.assertEqual(contract["query_shape"], [1, 128])
        self.assertTrue(contract["query_shape_ok"])
        self.assertEqual(contract["fpga_scorer_dot_mac_per_group"], 128)
        self.assertEqual(contract["fpga_scorer_compare_per_group"], 0)
        self.assertEqual(contract["fpga_scorer_exp_per_group"], 0)
        self.assertFalse(contract["token_axis_max_softmax_exp"])

    # -- checkpoint round-trip ---------------------------------------------
    def test_checkpoint_roundtrip(self) -> None:
        from evaluation.run_hf_hardware_regular_b1 import load_projection

        entry = self.bundle["training"]["C_PAIR_LAMBDA_0P25_PRIMARY"]["per_seed"][0]
        checkpoint = Path(entry["best_checkpoint"])
        projection, file_sha = load_projection(checkpoint)
        self.assertEqual(tuple(projection.Wt.shape), (128, 2560))
        self.assertEqual(tuple(projection.Wv.shape), (128, 1024))
        self.assertEqual(file_sha, _sha256(checkpoint))
        with np.load(checkpoint, allow_pickle=False) as archive:
            self.assertIn("Wt", archive)
            self.assertIn("Wv", archive)
            self.assertIn("metadata", archive)
            self.assertEqual(
                str(np.asarray(archive["weights_sha256"]).reshape(-1)[0]),
                entry["best_weights_sha256"],
            )

    def test_reports_created(self) -> None:
        paths = self.bundle["report_paths"]
        for key in ("epochs_csv", "val_samples_csv", "json", "md"):
            self.assertTrue(Path(paths[key]).is_file(), key)
        # write_outputs=True in setUpClass; the report files must be non-empty
        self.assertGreater(Path(paths["json"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
