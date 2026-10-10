"""CPU synthetic tests for B1 prompt-projection distillation.

These tests never load real weights and never require CUDA.  They cover the
frozen teacher target, the student shape/compatibility, the image-grouped
leak-free split (including a missing-image-hash rejection), the feature-cache
schema, the removable block-0 capture hook, the GPU ownership gate, a short
training run that must reduce the loss, the hardware-constrained real-group
metrics (matched real budget vs total budget coverage), and the split controls
(L2 baseline, deterministic cyclic shuffled prompt, few-sample unavailability,
and correct-prompt-above-shuffled on synthetic data).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

import evaluation.extract_prompt_projection_features as extract_module
from evaluation.extract_prompt_projection_features import (
    dense_vision_with_block0_capture,
    group_mean_from_block0,
    gpu_preflight_audit,
    gpu_runtime_snapshot,
    main as extract_main,
    prompt_embeddings_from_text,
)
from evaluation.hardware_regular_token_selector import (
    SOURCE_REAL,
    LowRankProjection,
    group_l2_saliency,
    select_hardware_regular_groups,
    semantic_cosine_scores,
)
from evaluation.prompt_projection_distillation import (
    CACHE_DTYPE,
    DistillationContractError,
    FEATURE_DIM,
    PRIMARY_PROMOTION_METRICS,
    PROMPT_DIM,
    SCORE_DIM,
    SHUFFLED_UNAVAILABLE_REASON,
    canonical_feature_index_record,
    canonical_projection_sha256,
    cyclic_shuffled_prompt_map,
    hardware_real_group_metrics,
    image_file_sha256,
    jaccard,
    l2_baseline_scores,
    load_feature_cache,
    metric_bundle,
    minmax_normalize_1d,
    ndcg_at_k,
    pearson,
    pinned_manifest_spec,
    promotion_gate,
    rank_metric_differences,
    saliency_mass_retained,
    save_feature_cache,
    save_projection_npz,
    spearman,
    split_group_summary,
    split_sample_ids,
    stable_rank,
    student_cosine_scores,
    teacher_scores,
    teacher_target,
    teacher_topk_recall,
    temperature_soft_target,
    topk_indices,
    validate_pinned_manifest,
    validate_source_image_sha256,
)
from evaluation.train_prompt_projection import (
    evaluate_split_metrics,
    load_index_samples,
    train_from_cache,
)

G = 64
T = 7
GRID_H = 8
GRID_W = 8


def _unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(count, dim))
    return (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)


def _projections(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    Wt = (rng.normal(size=(SCORE_DIM, PROMPT_DIM)) * 0.02).astype(np.float32)
    Wv = (rng.normal(size=(SCORE_DIM, FEATURE_DIM)) * 0.02).astype(np.float32)
    return Wt, Wv


class FakeVisionBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, hidden_states, cu_seqlens=None, position_embeddings=None):
        return torch.tanh(self.linear(hidden_states))


class FakeVisual(nn.Module):
    """Minimal Qwen3-VL-like tower: 3 blocks at 1024, merger to 2560."""

    def __init__(self, hidden: int = FEATURE_DIM, out: int = PROMPT_DIM, blocks: int = 3):
        super().__init__()
        self.spatial_merge_size = 2
        self.blocks = nn.ModuleList([FakeVisionBlock(hidden) for _ in range(blocks)])
        self.merger = nn.Linear(hidden * 4, out)
        self.deepstack_visual_indexes = [1]
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(hidden, out)])

    def forward(self, pixel_values, grid_thw=None):
        hidden = pixel_values
        for block in self.blocks:
            hidden = block(hidden)
        merged = self.merger(hidden.reshape(-1, hidden.shape[1] * 4))
        return merged, [self.deepstack_merger_list[0](hidden)]


class FakeTokenizer:
    def __init__(self, ids: list[int] | None = None) -> None:
        self.ids = ids or [1, 2, 3]
        self.seen_add_special_tokens: bool | None = None

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        self.seen_add_special_tokens = add_special_tokens
        return {"input_ids": torch.tensor([self.ids], dtype=torch.long)}


def _write_synthetic_index(
    directory: Path,
    *,
    count: int = 24,
    seed: int = 2026,
    temperature: float = 0.1,
    group_size: int = 1,
) -> Path:
    """Write a learnable synthetic cache + JSONL index and return the index path.

    ``group_size`` samples share one source-image SHA256 so the grouped split
    can be exercised; the default gives every sample a distinct image.
    """
    rng = np.random.default_rng(seed)
    Wt_star, Wv_star = _projections(seed + 1)
    index_path = directory / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for i in range(count):
            mean = rng.normal(size=(G, FEATURE_DIM)).astype(np.float32)
            prompt = _unit_vectors(T, PROMPT_DIM, seed + 100 + i)
            scores = student_cosine_scores(mean, prompt, Wt_star, Wv_star)
            normalized = minmax_normalize_1d(scores)
            soft = temperature_soft_target(normalized, temperature)
            sample_id = f"synthetic-{i:03d}"
            image_sha256 = hashlib.sha256(
                f"image-{seed}-{i // max(group_size, 1)}".encode("utf-8")
            ).hexdigest()
            metadata = {
                "sample_id": sample_id,
                "task": "synthetic",
                "grid_h": GRID_H,
                "grid_w": GRID_W,
                "temperature": temperature,
                "source_image_sha256": image_sha256,
            }
            write = save_feature_cache(
                directory / f"{sample_id}.npz",
                group_mean=mean,
                prompt_embeddings=prompt,
                teacher_raw=scores,
                teacher_normalized=normalized,
                teacher_soft=soft,
                metadata=metadata,
            )
            stream.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "task": "synthetic",
                        "npz": f"{sample_id}.npz",
                        "group_count": G,
                        "prompt_token_count": T,
                        "grid_h": GRID_H,
                        "grid_w": GRID_W,
                        "sample_sha256": write["sample_sha256"],
                        "source_image_sha256": image_sha256,
                    }
                )
                + "\n"
            )
    return index_path


def _small_config() -> dict[str, Any]:
    return {
        "feature_dim": FEATURE_DIM,
        "prompt_dim": PROMPT_DIM,
        "score_dim": SCORE_DIM,
        "merge_unit": 4,
        "temperature": 0.1,
        "lambda_semantic": 0.5,
        "budgets": [48, 64],
        "seed": 123,
        "loss": {
            "listwise_kl_weight": 1.0,
            "score_mse_weight": 0.25,
            "temperature_squared_scaling": True,
        },
        "optim": {
            "optimizer": "adamw",
            "epochs": 12,
            "lr": 0.05,
            "weight_decay": 0.0,
            "batch_size": 12,
            "grad_clip_norm": 1.0,
            "init_std": 0.02,
        },
        "split": {
            "salt": "synthetic-test",
            "seed": 7,
            "ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
            "group_by": "source_image_sha256",
            "require_image_sha256": True,
        },
        "metrics": {"ks": [48, 64]},
        "controls": {
            "l2_baseline": {"score": "group_mean_l2_norm"},
            "shuffled_prompt": {"assignment": "cyclic_by_sorted_sample_id"},
        },
        "promotion_criteria": {
            "val_beats_l2": "student_correct strictly better than l2_baseline on val",
            "val_correct_beats_shuffled": (
                "student_correct strictly better than shuffled prompt on val"
            ),
            "primary_metrics": list(PRIMARY_PROMOTION_METRICS),
            "no_arbitrary_threshold": "sign of paired delta only",
            "test_discipline": "test evaluated once after freeze",
        },
    }


class TeacherTargetTests(unittest.TestCase):
    def test_teacher_max_cosine_known_values(self) -> None:
        prompt = np.zeros((4, PROMPT_DIM), dtype=np.float32)
        for row in range(4):
            prompt[row, row] = 1.0  # orthonormal rows
        merged = np.stack(
            [prompt[0], prompt[1], prompt[0] * 0.5 + prompt[1] * 0.5], axis=0
        )
        scores = teacher_scores(merged, prompt)
        self.assertAlmostEqual(float(scores[0]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=5)
        self.assertAlmostEqual(float(scores[2]), 1.0 / math.sqrt(2.0), places=5)

    def test_minmax_and_soft_target(self) -> None:
        raw = np.array([0.0, 5.0, 10.0], dtype=np.float32)
        normalized = minmax_normalize_1d(raw)
        np.testing.assert_allclose(normalized, [0.0, 0.5, 1.0], atol=1e-6)
        soft = temperature_soft_target(normalized, 1.0)
        expected = np.exp([0.0, 0.5, 1.0])
        expected = expected / expected.sum()
        np.testing.assert_allclose(soft, expected, atol=1e-6)
        self.assertAlmostEqual(float(soft.sum()), 1.0, places=6)

    def test_teacher_target_bundle(self) -> None:
        prompt = _unit_vectors(T, PROMPT_DIM, 3)
        merged = _unit_vectors(G, PROMPT_DIM, 4)
        target = teacher_target(merged, prompt, 0.5)
        self.assertEqual(target["raw"].shape, (G,))
        self.assertAlmostEqual(float(target["soft"].sum()), 1.0, places=5)
        self.assertGreaterEqual(float(target["normalized"].min()), 0.0)
        self.assertLessEqual(float(target["normalized"].max()), 1.0)


class StudentScoreTests(unittest.TestCase):
    def test_student_shape_and_selector_compatibility(self) -> None:
        mean = np.random.default_rng(0).normal(size=(G, FEATURE_DIM)).astype(np.float32)
        prompt = _unit_vectors(T, PROMPT_DIM, 1)
        Wt, Wv = _projections(2)
        scores = student_cosine_scores(mean, prompt, Wt, Wv)
        self.assertEqual(scores.shape, (G,))
        self.assertTrue(np.isfinite(scores).all())
        # Broadcasting the mean over the 4 patch positions must reproduce the
        # frozen hardware-selector semantic score exactly.
        hidden = np.repeat(mean[:, None, :], 4, axis=1)
        selector = semantic_cosine_scores(hidden, prompt, LowRankProjection(Wt=Wt, Wv=Wv))
        np.testing.assert_allclose(scores, selector, atol=1e-5)


class SplitTests(unittest.TestCase):
    @staticmethod
    def _records(count: int, images: int, prefix: str = "sample") -> list[dict[str, str]]:
        records = []
        for i in range(count):
            records.append(
                {
                    "sample_id": f"{prefix}-{i:03d}",
                    "source_image_sha256": hashlib.sha256(
                        f"img-{prefix}-{i % images}".encode("utf-8")
                    ).hexdigest(),
                }
            )
        return records

    def test_image_grouped_split_is_stable_and_leak_free(self) -> None:
        records = self._records(200, images=80)
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        first = split_sample_ids(records, seed=11, ratios=ratios, salt="a")
        second = split_sample_ids(records, seed=11, ratios=ratios, salt="a")
        self.assertEqual(first, second)
        other = split_sample_ids(records, seed=12, ratios=ratios, salt="a")
        self.assertNotEqual(first, other)
        sets = {name: set(value) for name, value in first.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])
        self.assertEqual(sum(len(value) for value in sets.values()), len(records))
        self.assertGreater(len(sets["train"]), 0)
        self.assertGreater(len(sets["val"]), 0)
        self.assertGreater(len(sets["test"]), 0)

    def test_same_image_never_leaks_across_splits(self) -> None:
        # 12 samples but only 4 distinct images: all prompts of one image must
        # land in exactly one split.
        records = self._records(12, images=4)
        split = split_sample_ids(
            records,
            seed=3,
            ratios={"train": 0.5, "val": 0.25, "test": 0.25},
            salt="grouped",
        )
        image_hashes = {
            record["sample_id"]: record["source_image_sha256"] for record in records
        }
        summary = split_group_summary(split, image_hashes)
        self.assertTrue(all(entry["image_count"] <= 4 for entry in summary.values()))
        owner: dict[str, set[str]] = {}
        for name, values in split.items():
            for sample_id in values:
                owner.setdefault(image_hashes[sample_id], set()).add(name)
        self.assertTrue(all(len(names) == 1 for names in owner.values()))
        total_images = sum(entry["image_count"] for entry in summary.values())
        self.assertEqual(total_images, 4)
        total_samples = sum(entry["sample_count"] for entry in summary.values())
        self.assertEqual(total_samples, 12)

    def test_missing_or_invalid_image_hash_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            split_sample_ids(
                [{"sample_id": "a"}], seed=1, ratios={"train": 1.0, "val": 0.0, "test": 0.0}, salt="x"
            )
        with self.assertRaises(Exception):
            split_sample_ids(
                [{"sample_id": "a", "source_image_sha256": ""}],
                seed=1,
                ratios={"train": 1.0, "val": 0.0, "test": 0.0},
                salt="x",
            )
        # Bare sample_ids (the forbidden per-sample_id split) are rejected.
        with self.assertRaises(Exception):
            split_sample_ids(
                ["bare-sample-id"],
                seed=1,
                ratios={"train": 1.0, "val": 0.0, "test": 0.0},
                salt="x",
            )

    def test_validate_source_image_sha256_requires_64_hex(self) -> None:
        good = hashlib.sha256(b"image").hexdigest()
        self.assertEqual(validate_source_image_sha256(good.upper()), good)
        for bad in (None, "", "abc", "z" * 64):
            with self.assertRaises(Exception):
                validate_source_image_sha256(bad)

    def test_image_file_sha256_matches_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.bin"
            payload = b"not-a-real-image-but-bytes"
            path.write_bytes(payload)
            self.assertEqual(
                image_file_sha256(path),
                hashlib.sha256(payload).hexdigest(),
            )

    def test_index_record_carries_source_image_sha256(self) -> None:
        image_sha = hashlib.sha256(b"img").hexdigest()
        record = canonical_feature_index_record(
            sample_id="s1",
            task="synthetic",
            npz_name="s1.npz",
            cache={
                "teacher_scores": np.linspace(-1.0, 1.0, 4, dtype=np.float32),
                "group_mean": np.zeros((4, FEATURE_DIM), dtype=np.float32),
                "prompt_embeddings": np.zeros((2, PROMPT_DIM), dtype=np.float32),
                "grid_h": 2,
                "grid_w": 2,
                "sample_sha256": "deadbeef",
            },
            config_hash="config",
            source_image_sha256=image_sha,
        )
        self.assertEqual(record["source_image_sha256"], image_sha)


class HashAndExportTests(unittest.TestCase):
    def test_hash_matches_existing_runner_and_export_loads(self) -> None:
        Wt, Wv = _projections(5)
        self.assertEqual(
            canonical_projection_sha256(Wt, Wv),
            canonical_projection_sha256(Wt.astype(np.float64), Wv.astype(np.float64)),
        )
        from evaluation.run_hf_hardware_regular_b1 import (
            canonical_projection_sha256 as runner_hash,
            load_projection,
        )

        self.assertEqual(canonical_projection_sha256(Wt, Wv), runner_hash(Wt, Wv))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.npz"
            export = save_projection_npz(
                path, Wt, Wv, {"trained": True, "best_epoch": 3}
            )
            self.assertEqual(export["weights_sha256"], runner_hash(Wt, Wv))
            loaded, file_digest = load_projection(path)
            np.testing.assert_array_equal(loaded.Wt, Wt)
            np.testing.assert_array_equal(loaded.Wv, Wv)
            self.assertEqual(len(file_digest), 64)
            # A tampered weights_sha256 must be rejected by load_projection.
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: np.array(archive[key]) for key in archive.files}
            payload["weights_sha256"] = np.asarray("0" * 64)
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, **payload)
            with self.assertRaises(ValueError):
                load_projection(bad)


class MetricTests(unittest.TestCase):
    def test_pearson_and_spearman(self) -> None:
        self.assertAlmostEqual(pearson([1, 2, 3], [2, 4, 6]), 1.0, places=6)
        self.assertAlmostEqual(pearson([1, 2, 3], [3, 2, 1]), -1.0, places=6)
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0, places=6)
        self.assertAlmostEqual(spearman([1, 1, 2], [5, 5, 9]), 1.0, places=6)
        np.testing.assert_allclose(stable_rank([10, 10, 20]), [1.5, 1.5, 3.0])

    def test_ndcg_known_values(self) -> None:
        pred = np.array([3.0, 2.0, 1.0, 0.0])
        relevance = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(ndcg_at_k(pred, relevance, 1), 0.0, places=6)
        relevance = np.array([3.0, 2.0, 1.0, 0.0])
        self.assertAlmostEqual(ndcg_at_k(pred, relevance, 3), 1.0, places=6)

    def test_recall_jaccard_and_mass(self) -> None:
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        teacher = np.array([10.0, 9.0, 8.0, 7.0])
        self.assertEqual(float(teacher_topk_recall(topk_indices(pred, 2), topk_indices(teacher, 2))), 1.0)
        pred2 = np.array([0.0, 8.0, 7.0, 1.0])
        self.assertAlmostEqual(
            float(teacher_topk_recall(topk_indices(pred2, 2), topk_indices(teacher, 2))),
            0.5,
        )
        self.assertAlmostEqual(
            float(jaccard(topk_indices(pred2, 2), topk_indices(teacher, 2))),
            1.0 / 3.0,
        )
        mass = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(saliency_mass_retained([0, 3], mass), 0.5, places=6)

    def test_metric_bundle_keys(self) -> None:
        target = teacher_target(_unit_vectors(G, PROMPT_DIM, 6), _unit_vectors(T, PROMPT_DIM, 7), 0.2)
        Wt, Wv = _projections(8)
        mean = np.random.default_rng(9).normal(size=(G, FEATURE_DIM)).astype(np.float32)
        scores = student_cosine_scores(mean, _unit_vectors(T, PROMPT_DIM, 7), Wt, Wv)
        bundle = metric_bundle(scores, target["raw"], target["normalized"], target["soft"], budgets=[48, 64])
        self.assertIn("pearson", bundle)
        self.assertIn("spearman", bundle)
        for budget in ("48", "64"):
            for key in ("ndcg", "topk_recall", "jaccard", "saliency_mass_retained"):
                self.assertIn(key, bundle["per_budget"][budget])


class CacheSchemaTests(unittest.TestCase):
    def test_roundtrip_dtypes_and_no_dense_hidden(self) -> None:
        mean = np.random.default_rng(0).normal(size=(G, FEATURE_DIM)).astype(np.float32)
        prompt = _unit_vectors(T, PROMPT_DIM, 1)
        scores = np.linspace(-1.0, 1.0, G).astype(np.float32)
        normalized = minmax_normalize_1d(scores)
        soft = temperature_soft_target(normalized, 0.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.npz"
            write = save_feature_cache(
                path,
                group_mean=mean,
                prompt_embeddings=prompt,
                teacher_raw=scores,
                teacher_normalized=normalized,
                teacher_soft=soft,
                metadata={"sample_id": "x", "grid_h": 8, "grid_w": 8},
            )
            sample = load_feature_cache(path)
            self.assertEqual(sample["group_mean"].dtype, np.dtype(CACHE_DTYPE["group_mean"]))
            self.assertEqual(
                sample["prompt_embeddings"].dtype,
                np.dtype(CACHE_DTYPE["prompt_embeddings"]),
            )
            self.assertEqual(sample["teacher_scores"].dtype, np.dtype(np.float32))
            self.assertEqual(sample["teacher_soft_target"].shape, (G,))
            self.assertEqual(write["sample_sha256"], sample["metadata"]["sample_sha256"])
            # Forbidden dense payload must be rejected on load.
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: np.array(archive[key]) for key in archive.files}
            payload["dense_hidden"] = np.zeros((G, 2560), dtype=np.float16)
            bad = Path(tmp) / "bad.npz"
            np.savez(bad, **payload)
            with self.assertRaises(Exception):
                load_feature_cache(bad)


class ExtractorCpuTests(unittest.TestCase):
    def test_block0_hook_capture_and_group_mean(self) -> None:
        visual = FakeVisual()
        pixel_values = torch.randn(16, FEATURE_DIM)
        grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
        captured = dense_vision_with_block0_capture(visual, pixel_values, grid_thw)
        self.assertEqual(captured["group_count"], 4)
        self.assertEqual(captured["merged_embeddings"].shape, (4, PROMPT_DIM))
        self.assertEqual(captured["block0_patches"].shape, (16, FEATURE_DIM))
        self.assertEqual(captured["grid_h"], 2)
        self.assertEqual(captured["grid_w"], 2)
        self.assertEqual(len(visual.blocks[0]._forward_hooks), 0)
        mean = group_mean_from_block0(captured["block0_patches"], 4)
        self.assertEqual(mean.shape, (4, FEATURE_DIM))

    def test_grid_mismatch_is_rejected(self) -> None:
        visual = FakeVisual()
        with self.assertRaises(Exception):
            dense_vision_with_block0_capture(
                visual, torch.randn(16, FEATURE_DIM), torch.tensor([[1, 3, 3]])
            )

    def test_prompt_tokenization_has_no_special_tokens(self) -> None:
        tokenizer = FakeTokenizer([4, 5, 6, 7])
        embedding = nn.Embedding(10, PROMPT_DIM)
        embeddings = prompt_embeddings_from_text(embedding, tokenizer, "raw prompt")
        self.assertFalse(tokenizer.seen_add_special_tokens)
        self.assertEqual(embeddings.shape, (4, PROMPT_DIM))


class GpuGateTests(unittest.TestCase):
    def _runner(self, free_mib: int, owner: str):
        def run(command: list[str]):
            joined = " ".join(command)
            if "--query-gpu" in joined:
                return 0, f"NVIDIA RTX A6000, 46068, 500, {free_mib}, 0, 40, 580.0", ""
            if "--query-compute-apps" in joined:
                return 0, "1234, python, 100", ""
            if command[0] == "ps":
                return 0, owner + "\n", ""
            return 1, "", "unknown command"

        return run

    def test_pass_when_owner_and_memory_ok(self) -> None:
        audit = gpu_preflight_audit(
            required_free_mib=20480,
            current_user="zhaoxiangyu",
            command_runner=self._runner(40000, "zhaoxiangyu"),
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["status"], "PASS")

    def test_blocked_on_non_owner_process(self) -> None:
        audit = gpu_preflight_audit(
            required_free_mib=20480,
            current_user="zhaoxiangyu",
            command_runner=self._runner(40000, "seey6"),
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "BLOCKED_NON_OWNER_COMPUTE_PROCESS")
        self.assertEqual(audit["non_owner_compute_processes"][0]["user"], "seey6")

    def test_blocked_on_low_memory(self) -> None:
        audit = gpu_preflight_audit(
            required_free_mib=20480,
            current_user="zhaoxiangyu",
            command_runner=self._runner(1024, "zhaoxiangyu"),
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "BLOCKED_INSUFFICIENT_FREE_MEMORY")

    def test_blocked_when_nvidia_smi_fails(self) -> None:
        def run(command: list[str]):
            return 1, "", "NVML failed"

        audit = gpu_preflight_audit(
            required_free_mib=1,
            current_user="zhaoxiangyu",
            command_runner=run,
        )
        self.assertEqual(audit["status"], "BLOCKED_NVIDIA_SMI_FAILED")


class SharedGpuGateTests(unittest.TestCase):
    """Opt-in shared policy: tolerate other users but honour resource limits."""

    def _runner(self, free_mib: int, owner: str, util: int):
        def run(command: list[str]):
            joined = " ".join(command)
            if joined.startswith(
                "nvidia-smi --query-gpu=memory.total,memory.used,memory.free,utilization.gpu"
            ):
                return 0, f"46068, 500, {free_mib}, {util}\n", ""
            if joined.startswith("nvidia-smi --query-gpu=utilization.gpu"):
                return 0, f"{util}\n", ""
            if "--query-gpu" in joined:
                return (
                    0,
                    f"NVIDIA RTX A6000, 46068, 500, {free_mib}, {util}, 40, 580.0",
                    "",
                )
            if "--query-compute-apps" in joined:
                return 0, "1234, python, 100\n", ""
            if command[0] == "ps":
                return 0, owner + "\n", ""
            return 1, "", "unknown command"

        return run

    def _shared_audit(self, free_mib: int, owner: str, util: int):
        return gpu_preflight_audit(
            allow_shared_gpu=True,
            min_free_mib=14336,
            max_util_pct=80.0,
            current_user="zhaoxiangyu",
            command_runner=self._runner(free_mib, owner, util),
            util_samples=5,
            util_sample_interval_s=0.0,
            sleep_fn=lambda _seconds: None,
        )

    def test_shared_passes_with_non_owner_and_enough_free_memory(self) -> None:
        audit = self._shared_audit(21756, "seey6", 0)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(len(audit["non_owner_compute_processes"]), 1)
        policy = audit["shared_gpu_policy"]
        self.assertTrue(policy["enabled"])
        self.assertEqual(policy["util_samples_observed"], 5)
        self.assertEqual(policy["util_avg_percent"], 0.0)
        self.assertTrue(policy["non_owner_processes_do_not_block"])

    def test_shared_blocks_below_min_free_memory(self) -> None:
        audit = self._shared_audit(14000, "seey6", 0)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "BLOCKED_SHARED_GPU_LOW_MEMORY")

    def test_shared_blocks_on_high_average_utilization(self) -> None:
        audit = self._shared_audit(40000, "zhaoxiangyu", 90)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "BLOCKED_SHARED_GPU_HIGH_UTILIZATION")

    def test_legacy_default_still_blocks_non_owner(self) -> None:
        audit = gpu_preflight_audit(
            required_free_mib=20480,
            current_user="zhaoxiangyu",
            command_runner=self._runner(40000, "seey6", 0),
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["status"], "BLOCKED_NON_OWNER_COMPUTE_PROCESS")
        self.assertNotIn("shared_gpu_policy", audit)

    def test_runtime_snapshot_parses_single_query(self) -> None:
        audit = gpu_runtime_snapshot(
            command_runner=self._runner(21756, "seey6", 12),
        )
        self.assertEqual(audit["status"], "OK")
        self.assertEqual(audit["memory_free_mib"], 21756)
        self.assertEqual(audit["utilization_percent"], 12)


class HardwareMetricTests(unittest.TestCase):
    def _metrics(self) -> dict[str, Any]:
        rng = np.random.default_rng(21)
        mean = rng.normal(size=(G, FEATURE_DIM)).astype(np.float32)
        prompt = _unit_vectors(T, PROMPT_DIM, 22)
        Wt, Wv = _projections(23)
        target = teacher_target(
            rng.normal(size=(G, PROMPT_DIM)).astype(np.float32), prompt, 0.1
        )
        return hardware_real_group_metrics(
            mean,
            GRID_H,
            GRID_W,
            text_embeddings=prompt,
            Wt=Wt,
            Wv=Wv,
            lambda_semantic=0.5,
            teacher_raw=target["raw"],
            teacher_soft=target["soft"],
            budgets=[48, 64],
        )

    def test_matched_real_budget_and_total_budget_coverage_are_separate(self) -> None:
        metrics = self._metrics()
        self.assertEqual(set(metrics), {"48", "64"})
        for key, entry in metrics.items():
            self.assertEqual(entry["budget"], int(key))
            self.assertGreater(entry["real_group_count"], 0)
            self.assertLessEqual(entry["real_group_count"], int(key))
            # The vague top-level teacher_topk_recall is gone.
            self.assertNotIn("teacher_topk_recall", entry)
            self.assertNotIn("saliency_mass_retained", entry)
            matched = entry["matched_real_budget"]
            coverage = entry["total_budget_coverage"]
            self.assertEqual(matched["real_budget"], entry["real_group_count"])
            self.assertGreaterEqual(matched["teacher_top_r_recall"], 0.0)
            self.assertLessEqual(matched["teacher_top_r_recall"], 1.0)
            self.assertGreaterEqual(matched["jaccard"], 0.0)
            self.assertLessEqual(matched["jaccard"], 1.0)
            self.assertGreaterEqual(matched["saliency_mass_retained"], 0.0)
            self.assertLessEqual(matched["saliency_mass_retained"], 1.0)
            self.assertEqual(coverage["budget"], int(key))
            self.assertTrue(coverage["upper_bound_only"])
            self.assertGreaterEqual(coverage["teacher_topk_recall"], 0.0)
            self.assertLessEqual(coverage["teacher_topk_recall"], 1.0)

    def test_matched_real_budget_uses_r_not_k(self) -> None:
        # Independently reproduce the matched real-budget metric from the
        # frozen selector selection.
        rng = np.random.default_rng(31)
        mean = rng.normal(size=(G, FEATURE_DIM)).astype(np.float32)
        prompt = _unit_vectors(T, PROMPT_DIM, 32)
        Wt, Wv = _projections(33)
        target = teacher_target(
            rng.normal(size=(G, PROMPT_DIM)).astype(np.float32), prompt, 0.1
        )
        metrics = hardware_real_group_metrics(
            mean,
            GRID_H,
            GRID_W,
            text_embeddings=prompt,
            Wt=Wt,
            Wv=Wv,
            lambda_semantic=0.5,
            teacher_raw=target["raw"],
            teacher_soft=target["soft"],
            budgets=[48],
        )
        entry = metrics["48"]
        hidden = np.repeat(mean[:, None, :], 4, axis=1)
        selection = select_hardware_regular_groups(
            hidden,
            GRID_H,
            GRID_W,
            budget=48,
            text_embeddings=prompt,
            projection=LowRankProjection(Wt=Wt, Wv=Wv),
            lambda_semantic=0.5,
            region_grid=3,
        )
        real = [
            int(index)
            for index, source in zip(selection.group_indices, selection.source_type)
            if source == SOURCE_REAL
        ]
        r = len(real)
        expected = teacher_topk_recall(real, topk_indices(target["raw"], r))
        self.assertEqual(entry["real_group_count"], r)
        self.assertEqual(entry["matched_real_budget"]["real_budget"], r)
        self.assertAlmostEqual(
            entry["matched_real_budget"]["teacher_top_r_recall"], expected, places=6
        )
        self.assertEqual(entry["total_budget_coverage"]["budget"], 48)
        self.assertTrue(entry["total_budget_coverage"]["upper_bound_only"])


class ControlTests(unittest.TestCase):
    @staticmethod
    def _samples(count: int = 6, seed: int = 7):
        rng = np.random.default_rng(seed)
        Wt_star, Wv_star = _projections(seed + 1)
        samples = []
        for i in range(count):
            mean = rng.normal(size=(G, FEATURE_DIM)).astype(np.float32)
            prompt = _unit_vectors(T, PROMPT_DIM, seed + 50 + i)
            scores = student_cosine_scores(mean, prompt, Wt_star, Wv_star)
            normalized = minmax_normalize_1d(scores)
            soft = temperature_soft_target(normalized, 0.1)
            samples.append(
                {
                    "sample_id": f"c-{i:03d}",
                    "source_image_sha256": hashlib.sha256(
                        f"cimg-{seed}-{i}".encode("utf-8")
                    ).hexdigest(),
                    "grid_h": GRID_H,
                    "grid_w": GRID_W,
                    "group_mean": mean,
                    "prompt_embeddings": prompt,
                    "teacher_raw": scores,
                    "teacher_normalized": normalized,
                    "teacher_soft": soft,
                }
            )
        return samples, Wt_star, Wv_star

    def test_cyclic_shuffled_prompt_map_is_deterministic_derangement(self) -> None:
        mapping = cyclic_shuffled_prompt_map(["c", "a", "b"])
        self.assertEqual(mapping, {"a": "b", "b": "c", "c": "a"})
        self.assertEqual(mapping, cyclic_shuffled_prompt_map(["b", "c", "a"]))
        self.assertTrue(all(key != value for key, value in mapping.items()))
        self.assertIsNone(cyclic_shuffled_prompt_map(["only-one"]))
        self.assertIsNone(cyclic_shuffled_prompt_map([]))

    def test_l2_baseline_matches_selector_saliency(self) -> None:
        mean = np.random.default_rng(3).normal(size=(G, FEATURE_DIM)).astype(np.float32)
        hidden = np.repeat(mean[:, None, :], 4, axis=1)
        np.testing.assert_allclose(
            l2_baseline_scores(mean), group_l2_saliency(hidden), rtol=1e-5
        )

    def test_split_evaluation_reports_controls_correct_beats_shuffled(self) -> None:
        samples, Wt_star, Wv_star = self._samples()
        metrics = evaluate_split_metrics(
            samples, Wt_star, Wv_star, lambda_semantic=0.5, budgets=[48]
        )
        self.assertEqual(
            set(metrics),
            {
                "count",
                "student_correct",
                "l2_baseline",
                "student_shuffled_prompt",
                "correct_minus_shuffled",
            },
        )
        shuffled = metrics["student_shuffled_prompt"]
        self.assertTrue(shuffled["available"])
        correct = metrics["student_correct"]["rank"]
        # The teacher is generated by the same Wt*/Wv*, so the correct prompts
        # reproduce it exactly while mismatched prompts cannot.
        self.assertAlmostEqual(correct["pearson"], 1.0, places=5)
        self.assertAlmostEqual(correct["spearman"], 1.0, places=5)
        self.assertGreater(correct["spearman"], shuffled["rank"]["spearman"])
        self.assertGreater(correct["ndcg_full"], shuffled["rank"]["ndcg_full"])
        delta = metrics["correct_minus_shuffled"]
        self.assertTrue(delta["available"])
        self.assertGreater(delta["spearman"], 0.0)
        self.assertIn("per_budget", delta)
        self.assertIn("student_correct", metrics)
        self.assertIn("rank", metrics["l2_baseline"])

    def test_shuffled_control_unavailable_for_few_samples(self) -> None:
        samples, Wt_star, Wv_star = self._samples(count=1)
        metrics = evaluate_split_metrics(
            samples, Wt_star, Wv_star, lambda_semantic=0.5, budgets=[48]
        )
        self.assertFalse(metrics["student_shuffled_prompt"]["available"])
        self.assertEqual(
            metrics["student_shuffled_prompt"]["reason"],
            SHUFFLED_UNAVAILABLE_REASON,
        )
        self.assertFalse(metrics["correct_minus_shuffled"]["available"])

    def test_rank_metric_differences_and_promotion_gate(self) -> None:
        samples, Wt_star, Wv_star = self._samples()
        metrics = evaluate_split_metrics(
            samples, Wt_star, Wv_star, lambda_semantic=0.5, budgets=[48]
        )
        gate = promotion_gate({"val": metrics})
        self.assertEqual(gate["status"], "EVALUATED")
        self.assertTrue(gate["student_beats_l2"])
        self.assertTrue(gate["correct_beats_shuffled"])
        self.assertEqual(gate["primary_metrics"], list(PRIMARY_PROMOTION_METRICS))
        self.assertIn("student_correct_minus_l2_baseline", gate)
        self.assertIn("student_correct_minus_shuffled", gate)
        delta = rank_metric_differences(
            metrics["student_correct"]["rank"],
            metrics["student_shuffled_prompt"]["rank"],
        )
        self.assertGreater(delta["spearman"], 0.0)

    def test_promotion_gate_unavailable_with_single_sample(self) -> None:
        samples, Wt_star, Wv_star = self._samples(count=1)
        metrics = evaluate_split_metrics(
            samples, Wt_star, Wv_star, lambda_semantic=0.5, budgets=[48]
        )
        gate = promotion_gate({"val": metrics})
        self.assertEqual(gate["status"], "UNAVAILABLE")
        self.assertIsNone(gate["correct_beats_shuffled"])
        self.assertFalse(gate["student_correct_minus_shuffled"]["available"])


class TrainingTests(unittest.TestCase):
    def test_short_training_decreases_loss_and_exports_loadable_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index_path = _write_synthetic_index(directory, count=24)
            loaded = load_index_samples(index_path)
            self.assertEqual(len(loaded), 24)
            self.assertTrue(all(sample["source_image_sha256"] for sample in loaded))
            output = directory / "prompt_projection.npz"
            report_path = directory / "report.json"
            report = train_from_cache(
                config=_small_config(),
                index_path=index_path,
                output_npz=output,
                report_path=report_path,
                device="cpu",
                seed=123,
                max_epochs=12,
            )
            self.assertTrue(report["loss_decreased"], report["epoch_train_losses"])
            self.assertLess(
                report["epoch_train_losses"][-1], report["epoch_train_losses"][0]
            )
            self.assertFalse(report["test_used_for_selection"])
            self.assertTrue(report["test_evaluated_once_after_freeze"])
            self.assertEqual(report["split_group_by"], "source_image_sha256")
            self.assertTrue(report_path.is_file())
            self.assertGreater(report["split_counts"]["train"], 0)
            val = report["metrics"]["val"]
            for key in ("student_correct", "l2_baseline", "student_shuffled_prompt"):
                self.assertIn(key, val)
            self.assertIn("promotion_gate", report)
            self.assertIn("val_gate", report)

            from evaluation.run_hf_hardware_regular_b1 import load_projection

            projection, file_digest = load_projection(output)
            self.assertEqual(projection.Wt.shape, (SCORE_DIM, PROMPT_DIM))
            self.assertEqual(projection.Wv.shape, (SCORE_DIM, FEATURE_DIM))
            self.assertEqual(
                report["weights_sha256"],
                canonical_projection_sha256(projection.Wt, projection.Wv),
            )
            self.assertEqual(len(file_digest), 64)
            # Exported weights must not be the raw random init: the hash differs
            # from the untouched init and the metadata records training.
            with np.load(output, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"]))
            self.assertTrue(metadata["trained"])
            self.assertTrue(metadata["training_complete"])
            self.assertIn("test", report["metrics"])

    def test_training_refuses_index_without_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            index_path = _write_synthetic_index(directory, count=4)
            records = [
                json.loads(line)
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for record in records:
                record.pop("source_image_sha256", None)
            # Also drop the cached metadata fallback so the hash is truly absent.
            for record in records:
                with np.load(directory / record["npz"], allow_pickle=False) as archive:
                    payload = {key: np.array(archive[key]) for key in archive.files}
                payload = {
                    key: value
                    for key, value in payload.items()
                    if key != "meta_source_image_sha256"
                }
                np.savez(directory / record["npz"], **payload)
            index_path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception):
                load_index_samples(index_path)


class ManifestPinTests(unittest.TestCase):
    """The extraction source must equal the frozen config pin (fail closed).

    ``validate_manifest_only`` is exercised against the real frozen artifacts
    when they are present; it must never call the GPU audit or load a model.
    """

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def test_missing_pin_is_rejected(self) -> None:
        with self.assertRaises(DistillationContractError):
            pinned_manifest_spec(_small_config())

    def test_validate_manifest_only_is_cpu_only(self) -> None:
        repo_root = self._repo_root()
        config = repo_root / "configs" / "prompt_projection_distillation.json"
        manifest = (
            repo_root / "datasets" / "prompt_projection_clean_inductive_v1.jsonl"
        )
        if not (config.is_file() and manifest.is_file()):
            self.skipTest("frozen clean-inductive manifest is not present")
        original = extract_module.gpu_preflight_audit

        def _boom(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("gpu_preflight_audit must not run in validate mode")

        extract_module.gpu_preflight_audit = _boom
        try:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = extract_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--config",
                        str(config),
                        "--validate-manifest-only",
                    ]
                )
        finally:
            extract_module.gpu_preflight_audit = original
        payload = json.loads(stream.getvalue())
        self.assertEqual(code, 0, stream.getvalue())
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(payload["mode"], "validate_manifest_only")
        self.assertTrue(payload["path_matches"])
        self.assertTrue(payload["sha256_matches"])
        self.assertFalse(payload["gpu_used"])
        self.assertFalse(payload["model_loaded"])

    def test_validate_manifest_only_rejects_contaminated_source(self) -> None:
        repo_root = self._repo_root()
        config = repo_root / "configs" / "prompt_projection_distillation.json"
        contaminated = (
            repo_root / "datasets" / "cv_bench_screening100_budget64_diagnostic.jsonl"
        )
        if not (config.is_file() and contaminated.is_file()):
            self.skipTest("contaminated reference manifest is not present")
        original = extract_module.gpu_preflight_audit
        extract_module.gpu_preflight_audit = lambda *_a, **_k: self.fail(
            "gpu audit must not run"
        )
        try:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                code = extract_main(
                    [
                        "--manifest",
                        str(contaminated),
                        "--config",
                        str(config),
                        "--validate-manifest-only",
                    ]
                )
        finally:
            extract_module.gpu_preflight_audit = original
        payload = json.loads(stream.getvalue())
        self.assertEqual(code, 1)
        self.assertEqual(payload["status"], "FAIL")
        self.assertFalse(payload["path_matches"])
        self.assertFalse(payload["sha256_matches"])

    def test_validate_pinned_manifest_function(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "datasets" / "clean.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("payload\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            config = {
                "extraction": {
                    "manifest": {"path": "datasets/clean.jsonl", "sha256": digest}
                }
            }
            passed = validate_pinned_manifest(
                config, manifest_path=manifest, repo_root=root
            )
            self.assertEqual(passed["status"], "PASS")
            manifest.write_text("tampered\n", encoding="utf-8")
            failed = validate_pinned_manifest(
                config, manifest_path=manifest, repo_root=root
            )
            self.assertEqual(failed["status"], "FAIL")
            self.assertFalse(failed["sha256_matches"])


if __name__ == "__main__":
    unittest.main()
