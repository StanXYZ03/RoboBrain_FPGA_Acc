"""Focused CPU tests for the prompt-identifiability diagnostic.

These tests never load a model, never use CUDA and never touch the network.
They cover the new diagnostic statistics (Jensen-Shannon divergence, mean-pooled
cosine, paired bootstrap CI / sign-flip permutation, signal-ratio bootstrap, the
hierarchical variance decomposition), the integrity gate, and one end-to-end
synthetic run through ``run_diagnostic`` building a tiny feature cache with the
frozen ``save_feature_cache`` / ``save_projection_npz`` schema.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.prompt_identifiability_diagnostic import (
    ALPHA,
    BOOTSTRAP_SEED,
    CSV_FIELDS,
    EFFECT_SIZE_FLOOR,
    PROMPT_SIGNAL_RATIO_FLOOR,
    THRESHOLDS,
    bootstrap_ratio_ci,
    cosine_similarity,
    cross_image_donor_map,
    derange_images,
    derange_images_by_stratum,
    image_task_index,
    integrity_report,
    js_divergence,
    mark_v1_superseded,
    mean_pooled,
    paired_effect,
    projection_spectrum,
    render_markdown,
    run_diagnostic,
    same_image_partner_map,
    sample_donor_map,
    score_invariance_stats,
    score_reduction_collapse,
    variance_components,
    write_csv,
)
from evaluation.prompt_projection_distillation import (
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    load_distillation_config,
    load_feature_cache,
    save_feature_cache,
    save_projection_npz,
    split_sample_ids,
    teacher_target,
)
from evaluation.train_prompt_projection import load_index_samples


class JsDivergenceTests(unittest.TestCase):
    def test_identical_distribution_is_zero(self) -> None:
        p = np.array([0.2, 0.3, 0.5])
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=12)

    def test_disjoint_support_is_one_bit(self) -> None:
        p = np.array([1.0, 0.0])
        q = np.array([0.0, 1.0])
        self.assertAlmostEqual(js_divergence(p, q), 1.0, places=12)

    def test_symmetric_and_bounded(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.random(7)
            q = rng.random(7)
            self.assertAlmostEqual(js_divergence(p, q), js_divergence(q, p), places=12)
            self.assertGreaterEqual(js_divergence(p, q), 0.0)
            self.assertLessEqual(js_divergence(p, q), 1.0 + 1e-12)

    def test_unormalised_inputs_are_normalised(self) -> None:
        p = np.array([2.0, 2.0])
        q = np.array([1.0, 1.0])
        self.assertAlmostEqual(js_divergence(p, q), 0.0, places=12)


class CosineAndPoolingTests(unittest.TestCase):
    def test_cosine_of_identical_and_orthogonal(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0, places=12)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0, places=12)

    def test_zero_norm_is_zero(self) -> None:
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_mean_pooled_averages_tokens(self) -> None:
        prompts = np.array([[1.0, 3.0], [3.0, 5.0]])
        np.testing.assert_allclose(mean_pooled(prompts), [2.0, 4.0])


class PairedEffectTests(unittest.TestCase):
    def test_positive_shift_is_detected(self) -> None:
        deltas = np.linspace(0.1, 0.3, 40)
        result = paired_effect(deltas, seed=1, n_bootstrap=500, n_permutation=1000)
        self.assertGreater(result["ci_low"], 0.0)
        self.assertGreaterEqual(result["cohens_dz"], EFFECT_SIZE_FLOOR)
        self.assertLess(result["p_value_two_sided"], ALPHA)
        self.assertTrue(result["signal_present"])

    def test_zero_centred_noise_is_not_detected(self) -> None:
        rng = np.random.default_rng(3)
        deltas = rng.normal(0.0, 1.0, 60)
        result = paired_effect(deltas, seed=2, n_bootstrap=500, n_permutation=2000)
        self.assertFalse(result["signal_present"])

    def test_deterministic_for_a_fixed_seed(self) -> None:
        deltas = [0.1, -0.2, 0.3, 0.05, 0.4]
        first = paired_effect(deltas, seed=7, n_bootstrap=200, n_permutation=300)
        second = paired_effect(deltas, seed=7, n_bootstrap=200, n_permutation=300)
        self.assertEqual(first, second)

    def test_empty_input_is_unavailable(self) -> None:
        result = paired_effect([], seed=0, n_bootstrap=10, n_permutation=10)
        self.assertIsNone(result["mean"])
        self.assertFalse(result["signal_present"])


class BootstrapRatioTests(unittest.TestCase):
    def test_ratio_of_equal_series_is_one(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8]
        result = bootstrap_ratio_ci(values, values, seed=0, n_bootstrap=200)
        self.assertAlmostEqual(result["ratio_of_means"], 1.0, places=12)
        self.assertLessEqual(result["ci_low"], 1.0)
        self.assertGreaterEqual(result["ci_high"], 1.0)

    def test_double_series_has_ratio_two(self) -> None:
        same = [0.2, 0.4, 0.6]
        cross = [0.1, 0.2, 0.3]
        result = bootstrap_ratio_ci(same, cross, seed=0, n_bootstrap=200)
        self.assertAlmostEqual(result["ratio_of_means"], 2.0, places=12)


class VarianceDecompositionTests(unittest.TestCase):
    def test_identity_and_manual_values(self) -> None:
        # two images, two prompts, two groups each.
        values = {
            "imgA": {"grounding": np.array([0.0, 2.0]), "pointing": np.array([2.0, 4.0])},
            "imgB": {"grounding": np.array([10.0, 12.0]), "pointing": np.array([10.0, 12.0])},
        }
        result = variance_components(values)
        self.assertAlmostEqual(
            result["v_total"],
            result["v_image"] + result["v_prompt"] + result["v_group"],
            places=12,
        )
        # image means are 2 and 11; grand mean 6.5; weights equal
        self.assertAlmostEqual(result["v_image"], 0.5 * (2.0 - 6.5) ** 2 + 0.5 * (11.0 - 6.5) ** 2, places=12)
        # imgA prompt means 1 and 3 -> prompt variance contribution 1 each; imgB 0
        self.assertAlmostEqual(result["v_prompt"], 0.25 * 1.0 + 0.25 * 1.0, places=12)
        # within-prompt across-group variance: imgA 2+2, imgB 2+2 -> 8/8 = 1.0
        self.assertAlmostEqual(result["v_group"], 1.0, places=12)

    def test_prompt_only_variation_has_image_variance_zero(self) -> None:
        values = {
            f"img{i}": {"grounding": np.array([0.0]), "pointing": np.array([1.0])}
            for i in range(5)
        }
        result = variance_components(values)
        self.assertAlmostEqual(result["v_image"], 0.0, places=12)
        self.assertIsNone(result["prompt_signal_ratio"])
        self.assertAlmostEqual(result["prompt_signal_fraction"], 1.0, places=12)


class PairingMapTests(unittest.TestCase):
    def _samples(self) -> list[dict]:
        out = []
        for image_index in range(4):
            for task in ("grounding", "pointing"):
                out.append(
                    {
                        "sample_id": f"img{image_index}-{task}",
                        "task": task,
                        "source_image_sha256": f"{image_index:064x}",
                        "group_mean": np.zeros((4, FEATURE_DIM), dtype=np.float32),
                    }
                )
        return out

    def test_partner_map_is_involution(self) -> None:
        partner = same_image_partner_map(self._samples())
        for sample_id, other in partner.items():
            self.assertNotEqual(sample_id, other)
            self.assertEqual(partner[other], sample_id)

    def test_cross_image_donor_never_same_image(self) -> None:
        samples = self._samples()
        donors = cross_image_donor_map(samples, seed=5, same_grid=False)
        by_id = {s["sample_id"]: s for s in samples}
        for sample_id, donor in donors.items():
            self.assertNotEqual(
                by_id[sample_id]["source_image_sha256"],
                by_id[donor]["source_image_sha256"],
            )
        # deterministic
        self.assertEqual(donors, cross_image_donor_map(samples, seed=5, same_grid=False))


class IntegrityTests(unittest.TestCase):
    def _sample(self, sample_id: str, task: str, image_sha: str, group_mean: np.ndarray) -> dict:
        size = group_mean.shape[0]
        return {
            "sample_id": sample_id,
            "task": task,
            "source_image_sha256": image_sha,
            "group_mean": group_mean,
            "grid_h": 2,
            "grid_w": size // 2,
            "teacher_soft": np.full(size, 1.0 / size, dtype=np.float32),
            "teacher_normalized": np.linspace(0.0, 1.0, size, dtype=np.float32),
            "prompt_embeddings": np.zeros((3, PROMPT_DIM), dtype=np.float32),
        }

    def test_matching_group_mean_passes(self) -> None:
        image_sha = "a" * 64
        gm = np.ones((4, FEATURE_DIM), dtype=np.float32)
        samples = [
            self._sample("g", "grounding", image_sha, gm),
            self._sample("p", "pointing", image_sha, gm.copy()),
        ]
        report = integrity_report(samples, np.zeros((SCORE_DIM, PROMPT_DIM)), np.zeros((SCORE_DIM, FEATURE_DIM)))
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["same_image_group_mean_equal"])

    def test_differing_group_mean_fails(self) -> None:
        image_sha = "b" * 64
        samples = [
            self._sample("g", "grounding", image_sha, np.ones((4, FEATURE_DIM), dtype=np.float32)),
            self._sample("p", "pointing", image_sha, np.zeros((4, FEATURE_DIM), dtype=np.float32)),
        ]
        report = integrity_report(samples, np.zeros((SCORE_DIM, PROMPT_DIM)), np.zeros((SCORE_DIM, FEATURE_DIM)))
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["same_image_group_mean_equal"])
        self.assertTrue(any("group_mean bytes differ" in item for item in report["failures"]))

    def test_bad_soft_sum_fails(self) -> None:
        image_sha = "c" * 64
        gm = np.ones((4, FEATURE_DIM), dtype=np.float32)
        good = self._sample("g", "grounding", image_sha, gm)
        bad = self._sample("p", "pointing", image_sha, gm.copy())
        bad["teacher_soft"] = np.full(4, 0.5, dtype=np.float32)
        report = integrity_report([good, bad], np.zeros((SCORE_DIM, PROMPT_DIM)), np.zeros((SCORE_DIM, FEATURE_DIM)))
        self.assertEqual(report["status"], "FAIL")


class ProjectionCollapseTests(unittest.TestCase):
    def test_spectrum_shapes_and_effective_rank(self) -> None:
        Wt = np.eye(SCORE_DIM, PROMPT_DIM, dtype=np.float32)
        Wv = np.eye(SCORE_DIM, FEATURE_DIM, dtype=np.float32)
        spectrum = projection_spectrum(Wt, Wv)
        self.assertEqual(spectrum["Wt"]["numerical_rank_singular_gt_1e-6_max"], SCORE_DIM)
        self.assertAlmostEqual(spectrum["Wt"]["effective_rank_participation_ratio"], SCORE_DIM, places=6)

    def test_rank_one_wt_is_reported_as_low_effective_rank(self) -> None:
        Wt = np.outer(np.ones(SCORE_DIM), np.ones(PROMPT_DIM)).astype(np.float32)
        spectrum = projection_spectrum(Wt, np.zeros((SCORE_DIM, FEATURE_DIM), dtype=np.float32))
        self.assertAlmostEqual(spectrum["Wt"]["effective_rank_participation_ratio"], 1.0, places=6)

    def test_score_reduction_collapse_detects_identical_prompts(self) -> None:
        samples = []
        for image_index in range(3):
            group_mean = np.random.default_rng(image_index).normal(size=(4, FEATURE_DIM)).astype(np.float32)
            prompt = np.random.default_rng(100 + image_index).normal(size=(5, PROMPT_DIM)).astype(np.float32)
            for task in ("grounding", "pointing"):
                samples.append(
                    {
                        "sample_id": f"img{image_index}-{task}",
                        "task": task,
                        "source_image_sha256": f"{image_index:064x}",
                        "group_mean": group_mean,
                        # both tasks share the very same prompt -> exact invariance
                        "prompt_embeddings": prompt,
                    }
                )
        Wt = np.random.default_rng(0).normal(size=(SCORE_DIM, PROMPT_DIM)).astype(np.float32) * 0.02
        Wv = np.random.default_rng(1).normal(size=(SCORE_DIM, FEATURE_DIM)).astype(np.float32) * 0.02
        partner = same_image_partner_map(samples)
        result = score_reduction_collapse(samples, Wt, Wv, partner)
        self.assertEqual(result["sample_count"], 6)
        self.assertAlmostEqual(result["exact_prompt_invariance_fraction"], 1.0, places=12)
        self.assertGreaterEqual(result["single_argmax_token_fraction"], 0.0)
        self.assertLessEqual(result["single_argmax_token_fraction"], 1.0)


class ThresholdTests(unittest.TestCase):
    def test_thresholds_are_frozen_normalised_effects(self) -> None:
        self.assertEqual(THRESHOLDS["bootstrap_resamples"], 2000)
        self.assertEqual(THRESHOLDS["permutation_resamples"], 10000)
        self.assertGreaterEqual(THRESHOLDS["bootstrap_resamples"], 2000)
        self.assertGreaterEqual(THRESHOLDS["permutation_resamples"], 10000)
        self.assertEqual(THRESHOLDS["alpha"], ALPHA)
        self.assertEqual(THRESHOLDS["effect_size_floor_cohens_dz"], EFFECT_SIZE_FLOOR)
        self.assertEqual(THRESHOLDS["prompt_signal_ratio_floor"], PROMPT_SIGNAL_RATIO_FLOOR)
        for label in (
            "TEACHER_PROMPT_SIGNAL_WEAK",
            "STUDENT_PROMPT_COLLAPSE",
            "GENERALIZATION_OVERFIT",
            "DATA_OR_SCHEMA_INVALID",
        ):
            self.assertIn(label, THRESHOLDS)
        self.assertTrue(THRESHOLDS["fixed_before_test"])


def _seed_all(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_synthetic_case(root: Path, image_count: int = 80) -> dict[str, Path]:
    """Tiny but schema-exact feature cache + projection + training report."""
    rng = _seed_all(11)
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    index_records = []
    image_hashes = []
    for image_index in range(image_count):
        image_hash = hashlib.sha256(f"image-{image_index}".encode("utf-8")).hexdigest()
        image_hashes.append(image_hash)
        group_count = 4
        merged = rng.normal(size=(group_count, PROMPT_DIM)).astype(np.float32)
        group_mean = rng.normal(size=(group_count, FEATURE_DIM)).astype(np.float32)
        for task in ("grounding", "pointing"):
            prompt = rng.normal(size=(3, PROMPT_DIM)).astype(np.float32)
            target = teacher_target(merged, prompt, 0.1)
            sample_id = f"syn-{image_index:04d}-{task}"
            npz_name = f"{sample_id}.npz"
            save_feature_cache(
                features_dir / npz_name,
                group_mean=group_mean,
                prompt_embeddings=prompt,
                teacher_raw=target["raw"],
                teacher_normalized=target["normalized"],
                teacher_soft=target["soft"],
                metadata={
                    "sample_id": sample_id,
                    "task": task,
                    "grid_h": 2,
                    "grid_w": 2,
                    "temperature": 0.1,
                    "source_image_sha256": image_hash,
                },
            )
            index_records.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "npz": npz_name,
                    "grid_h": 2,
                    "grid_w": 2,
                    "source_image_sha256": image_hash,
                }
            )
    index_path = features_dir / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(record) for record in index_records) + "\n",
        encoding="utf-8",
    )
    projection_path = root / "projection.npz"
    save_projection_npz(
        projection_path,
        rng.normal(size=(SCORE_DIM, PROMPT_DIM)).astype(np.float32) * 0.02,
        rng.normal(size=(SCORE_DIM, FEATURE_DIM)).astype(np.float32) * 0.02,
    )
    training_report_path = root / "training.json"
    training_report_path.write_text(
        json.dumps(
            {
                "status": "TRAINED",
                "split_counts": {"train": len(index_records) * 7 // 10, "val": 1, "test": 1},
                "best_epoch": 0,
                "best_val_listwise_kl": 0.03,
                "epoch_train_losses": [0.04, 0.02],
                "epoch_val_losses": [0.03, 0.031],
                "metadata": {
                    "Wt_shape": [SCORE_DIM, PROMPT_DIM],
                    "Wv_shape": [SCORE_DIM, FEATURE_DIM],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = root / "config.json"
    repo_config = Path(__file__).resolve().parents[2] / "configs" / "prompt_projection_distillation.json"
    config_path.write_text(repo_config.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "config": config_path,
        "index": index_path,
        "features_dir": features_dir,
        "projection": projection_path,
        "training_report": training_report_path,
    }


class EndToEndSyntheticTests(unittest.TestCase):
    def test_run_diagnostic_on_synthetic_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_case(Path(tmp), image_count=80)
            report, rows = run_diagnostic(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                training_report_path=paths["training_report"],
            )
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["integrity"]["status"], "PASS")
        self.assertEqual(len(rows), 160)
        self.assertEqual(set(report["split_counts"]), {"train", "val", "test"})
        self.assertEqual(sum(report["split_counts"].values()), 160)
        for name in ("train", "val", "test"):
            self.assertGreater(report["split_counts"][name], 0)
        for label in (
            "DATA_OR_SCHEMA_INVALID",
            "TEACHER_PROMPT_SIGNAL_WEAK",
            "STUDENT_PROMPT_COLLAPSE",
            "GENERALIZATION_OVERFIT",
        ):
            self.assertIn(label, report["diagnosis"])
        # variance decomposition identity holds on every split and target
        for name in ("train", "val", "test"):
            for key, entry in report["splits"][name]["variance"].items():
                if not isinstance(entry, dict) or "v_total" not in entry:
                    continue
                self.assertAlmostEqual(
                    entry["v_total"],
                    entry["v_image"] + entry["v_prompt"] + entry["v_group"],
                    places=10,
                )
        # every CSV field is populated for every row and finite where numeric
        for row in rows:
            self.assertEqual(row["split"] in {"train", "val", "test"}, True)
            for field in CSV_FIELDS:
                self.assertIn(field, row)
        # writers produce the three report artefacts
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = Path(tmp) / "samples.csv"
            write_csv(out_csv, rows, CSV_FIELDS)
            self.assertTrue(out_csv.is_file())
            text = render_markdown(report)
            self.assertIn("TEACHER_PROMPT_SIGNAL_WEAK", text)
            self.assertIn("TEACHER_PROMPT_SIGNAL_WEAK", text)

    def test_end_to_end_generalization_overfit_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_case(Path(tmp), image_count=80)
            report, _ = run_diagnostic(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                training_report_path=paths["training_report"],
            )
        # synthetic report has best_epoch=0 and decreasing train loss
        self.assertTrue(report["generalization"]["train_loss_decreased"])
        self.assertTrue(report["diagnosis"]["GENERALIZATION_OVERFIT"])


def synthetic_split(paths: dict[str, Path]) -> dict[str, list[str]]:
    """Recompute the frozen group-hash split for a synthetic case."""
    records = [
        json.loads(line)
        for line in Path(paths["index"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stubs = [
        {"sample_id": record["sample_id"], "source_image_sha256": record["source_image_sha256"]}
        for record in records
    ]
    config = load_distillation_config(paths["config"])
    return split_sample_ids(
        stubs,
        seed=int(config["split"]["seed"]),
        ratios=config["split"]["ratios"],
        salt=str(config["split"]["salt"]),
    )


class DerangementTests(unittest.TestCase):
    def test_derangement_is_bijective_and_self_pair_free(self) -> None:
        images = [f"img{i}" for i in range(9)]
        for seed in (0, 5, BOOTSTRAP_SEED):
            mapping = derange_images(images, seed=seed)
            self.assertEqual(set(mapping), set(images))
            self.assertEqual(set(mapping.values()), set(images))
            for source, donor in mapping.items():
                self.assertNotEqual(source, donor)
            self.assertEqual(mapping, derange_images(images, seed=seed))

    def test_singleton_stratum_is_excluded_and_reported(self) -> None:
        image_to_group = {"img-a": 4, "img-b": 4, "img-c": 7}
        donors, excluded = derange_images_by_stratum(image_to_group, seed=1)
        self.assertEqual(excluded, ["img-c"])
        self.assertEqual(set(donors), {"img-a", "img-b"})
        self.assertEqual(donors["img-a"], "img-b")
        self.assertEqual(donors["img-b"], "img-a")

    def test_same_G_strata_never_mix(self) -> None:
        image_to_group = {f"g4-{i}": 4 for i in range(3)}
        image_to_group.update({f"g7-{i}": 7 for i in range(2)})
        donors, excluded = derange_images_by_stratum(image_to_group, seed=3)
        self.assertEqual(excluded, [])
        for source, donor in donors.items():
            self.assertEqual(image_to_group[source], image_to_group[donor])

    def test_sample_donor_map_is_same_task(self) -> None:
        samples = []
        for index in range(4):
            for task in ("grounding", "pointing"):
                samples.append(
                    {
                        "sample_id": f"img{index}-{task}",
                        "task": task,
                        "source_image_sha256": f"{index:064x}",
                    }
                )
        donor_by_image = derange_images([f"{i:064x}" for i in range(4)], seed=7)
        donors = sample_donor_map(samples, donor_by_image)
        by_id = {sample["sample_id"]: sample for sample in samples}
        for sample_id, donor_id in donors.items():
            self.assertEqual(by_id[sample_id]["task"], by_id[donor_id]["task"])
            self.assertNotEqual(
                by_id[sample_id]["source_image_sha256"],
                by_id[donor_id]["source_image_sha256"],
            )


class ScoreInvarianceStatsTests(unittest.TestCase):
    def test_exact_uses_array_equal_and_tolerance_is_named(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        exact = score_invariance_stats(values, values.copy())
        self.assertTrue(exact["exact"])
        self.assertEqual(exact["max_abs_diff"], 0.0)
        tiny = score_invariance_stats(values, values + 1e-12)
        self.assertFalse(tiny["exact"])
        self.assertTrue(tiny["tolerance"])
        self.assertAlmostEqual(tiny["max_abs_diff"], 1e-12, places=15)
        big = score_invariance_stats(values, values + 1e-3)
        self.assertFalse(big["exact"])
        self.assertFalse(big["tolerance"])

    def test_collapse_reports_exact_equality_and_max_abs_stats(self) -> None:
        samples = []
        for image_index in range(3):
            group_mean = np.random.default_rng(image_index).normal(size=(4, FEATURE_DIM)).astype(np.float32)
            prompt = np.random.default_rng(100 + image_index).normal(size=(5, PROMPT_DIM)).astype(np.float32)
            for task in ("grounding", "pointing"):
                samples.append(
                    {
                        "sample_id": f"img{image_index}-{task}",
                        "task": task,
                        "source_image_sha256": f"{image_index:064x}",
                        "group_mean": group_mean,
                        "prompt_embeddings": prompt,
                    }
                )
        Wt = np.random.default_rng(0).normal(size=(SCORE_DIM, PROMPT_DIM)).astype(np.float32) * 0.02
        Wv = np.random.default_rng(1).normal(size=(SCORE_DIM, FEATURE_DIM)).astype(np.float32) * 0.02
        result = score_reduction_collapse(samples, Wt, Wv, same_image_partner_map(samples))
        self.assertTrue(result["exact_invariance_uses_np_array_equal"])
        self.assertEqual(result["exact_prompt_invariance_fraction"], 1.0)
        self.assertEqual(result["pairwise_max_abs_diff"]["max"], 0.0)
        self.assertIn("tolerance_invariance_fraction_1e-9", result)


class V2SplitLocalDesignTests(unittest.TestCase):
    """End-to-end design checks on one synthetic diagnostic run."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.mkdtemp()
        cls.paths = build_synthetic_case(Path(cls._tmp), image_count=48)
        cls.report, cls.rows = run_diagnostic(
            config_path=cls.paths["config"],
            index_path=cls.paths["index"],
            features_dir=cls.paths["features_dir"],
            projection_path=cls.paths["projection"],
            training_report_path=cls.paths["training_report"],
        )
        cls.split = synthetic_split(cls.paths)
        cls.samples = load_index_samples(cls.paths["index"], cls.paths["features_dir"])
        cls.by_id = {str(sample["sample_id"]): sample for sample in cls.samples}

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def test_donors_are_strictly_split_local(self) -> None:
        for name in ("train", "val", "test"):
            split_ids = set(self.split[name])
            others = set().union(
                *[set(value) for other, value in self.split.items() if other != name]
            )
            pairing = self.report["splits"][name]["pairing"]
            self.assertEqual(pairing["scope"], "split_local")
            for mapping in ("student_donors", "teacher_donors"):
                for recipient, donor in pairing[mapping].items():
                    self.assertIn(recipient, split_ids)
                    self.assertIn(donor, split_ids)
                    self.assertNotIn(donor, others)

    def test_student_donor_is_same_task_different_image(self) -> None:
        for name in ("train", "val", "test"):
            pairing = self.report["splits"][name]["pairing"]
            for recipient, donor in pairing["student_donors"].items():
                self.assertEqual(self.by_id[recipient]["task"], self.by_id[donor]["task"])
                self.assertNotEqual(
                    self.by_id[recipient]["source_image_sha256"],
                    self.by_id[donor]["source_image_sha256"],
                )

    def test_teacher_donor_is_same_G_and_same_task(self) -> None:
        for name in ("train", "val", "test"):
            pairing = self.report["splits"][name]["pairing"]
            groups = pairing["image_to_group"]
            for source, donor in pairing["teacher_image_derangement"].items():
                self.assertEqual(groups[source], groups[donor])
            for recipient, donor in pairing["teacher_donors"].items():
                self.assertEqual(groups[self.by_id[recipient]["source_image_sha256"]],
                                 groups[self.by_id[donor]["source_image_sha256"]])
                self.assertEqual(self.by_id[recipient]["task"], self.by_id[donor]["task"])
            for image in pairing["teacher_excluded_image_hashes"]:
                stratum = groups[image]
                self.assertEqual(sum(1 for value in groups.values() if value == stratum), 1)

    def test_image_derangement_is_bijective_and_self_pair_free(self) -> None:
        for name in ("train", "val", "test"):
            mapping = self.report["splits"][name]["pairing"]["image_derangement"]
            self.assertEqual(set(mapping), set(mapping.values()))
            for source, donor in mapping.items():
                self.assertNotEqual(source, donor)

    def test_student_paired_delta_is_image_clustered(self) -> None:
        for name in ("train", "val", "test"):
            entry = self.report["splits"][name]
            self.assertEqual(entry["student"]["independent_image_units"], entry["image_count"])
            self.assertEqual(entry["count"], len(self.split[name]))
            for condition in ("same", "cross"):
                for statistic in entry["student"]["paired_deltas"][condition].values():
                    self.assertEqual(statistic["independent_image_units"], entry["image_count"])
                    self.assertEqual(statistic["n"], statistic["independent_image_units"])
                    self.assertEqual(statistic["sample_count"], entry["count"])

    def test_prompt_paired_control_is_per_image_not_global_mean(self) -> None:
        for name in ("train", "val", "test"):
            entry = self.report["splits"][name]
            pairing = entry["pairing"]
            split_samples = [self.by_id[sid] for sid in self.split[name]]
            index = image_task_index(split_samples)
            deltas = []
            same_values = []
            cross_values = []
            for image, tasks in index.items():
                if set(tasks) != {"grounding", "pointing"}:
                    continue
                donor = pairing["image_derangement"].get(image)
                donor_tasks = index.get(donor, {}) if donor else {}
                if set(donor_tasks) != {"grounding", "pointing"}:
                    continue
                same = cosine_similarity(
                    mean_pooled(tasks["grounding"]["prompt_embeddings"]),
                    mean_pooled(tasks["pointing"]["prompt_embeddings"]),
                )
                cross = 0.5 * (
                    cosine_similarity(
                        mean_pooled(tasks["grounding"]["prompt_embeddings"]),
                        mean_pooled(donor_tasks["pointing"]["prompt_embeddings"]),
                    )
                    + cosine_similarity(
                        mean_pooled(tasks["pointing"]["prompt_embeddings"]),
                        mean_pooled(donor_tasks["grounding"]["prompt_embeddings"]),
                    )
                )
                same_values.append(same)
                cross_values.append(cross)
                deltas.append(same - cross)
            reported = entry["prompt_embedding"]["paired_same_minus_cross"]
            self.assertEqual(reported["n"], len(deltas))
            self.assertAlmostEqual(reported["mean"], float(np.mean(deltas)), places=10)
            self.assertEqual(reported["independent_image_units"], entry["image_count"])
            # The paired statistic must use the per-image cross term: the delta
            # dispersion differs from the v1 "same_i - global cross mean" form.
            per_image_std = float(np.std(deltas, ddof=1))
            global_delta = [same - float(np.mean(cross_values)) for same in same_values]
            global_std = float(np.std(global_delta, ddof=1))
            self.assertAlmostEqual(reported["std"], per_image_std, places=10)
            self.assertGreater(abs(per_image_std - global_std), 1e-6)

    def test_design_metadata_tracks_the_correction(self) -> None:
        self.assertEqual(self.report["report_version"], "2.0")
        self.assertTrue(self.report["gate"]["fixed_before_test"])
        self.assertTrue(self.report["gate"]["test_reported_once_after_gate"])
        self.assertTrue(THRESHOLDS["prompt_signal_ratio_floor_is_engineering_gate"])
        self.assertEqual(
            self.report["diagnosis"]["STUDENT_PROMPT_COLLAPSE"],
            self.report["primary_blocker"] == "STUDENT_PROMPT_COLLAPSE",
        )
        gen = self.report["generalization"]
        self.assertEqual(gen["train_records"], len(self.split["train"]))
        self.assertGreater(gen["train_group_targets"], 0)
        self.assertEqual(gen["parameters_per_record"], gen["trainable_parameters"] / gen["train_records"])


class V2TestIsolationTests(unittest.TestCase):
    def test_changing_test_prompts_does_not_change_val_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_synthetic_case(Path(tmp), image_count=48)
            split = synthetic_split(paths)
            before, _ = run_diagnostic(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                training_report_path=paths["training_report"],
            )
            records = {
                json.loads(line)["sample_id"]: json.loads(line)
                for line in Path(paths["index"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            for sample_id in split["test"]:
                npz_path = paths["features_dir"] / records[sample_id]["npz"]
                cache = load_feature_cache(npz_path)
                metadata = {
                    key: value
                    for key, value in cache["metadata"].items()
                    if key != "sample_sha256"
                }
                altered = (cache["prompt_embeddings"].astype(np.float64) * 3.0 + 0.5).astype(
                    np.float32
                )
                save_feature_cache(
                    npz_path,
                    group_mean=cache["group_mean"],
                    prompt_embeddings=altered,
                    teacher_raw=cache["teacher_scores"],
                    teacher_normalized=cache["teacher_scores_normalized"],
                    teacher_soft=cache["teacher_soft_target"],
                    metadata=metadata,
                )
            after, _ = run_diagnostic(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                training_report_path=paths["training_report"],
            )
        self.assertEqual(before["splits"]["val"], after["splits"]["val"])
        self.assertNotEqual(before["splits"]["test"], after["splits"]["test"])


class V1SupersessionMarkerTests(unittest.TestCase):
    def test_mark_v1_superseded_is_idempotent_and_preserves_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "v1.md"
            json_path = Path(tmp) / "v1.json"
            md_path.write_text("# v1\n\nvalue = 0.7304\n", encoding="utf-8")
            json_path.write_text(json.dumps({"number": 0.7304}) + "\n", encoding="utf-8")
            first = mark_v1_superseded(md_path, json_path)
            second = mark_v1_superseded(md_path, json_path)
            self.assertTrue(first["markdown"] and first["json"])
            self.assertEqual(first["markdown_sha256"], second["markdown_sha256"])
            self.assertIn("SUPERSEDED_FOR_INFERENCE_BY_V2", md_path.read_text(encoding="utf-8"))
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["number"], 0.7304)
            self.assertIn("SUPERSEDED_FOR_INFERENCE_BY_V2", payload)


if __name__ == "__main__":
    unittest.main()
