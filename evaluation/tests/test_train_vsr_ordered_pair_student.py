from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from evaluation.ordered_pair_scorer import PROMPT_DIM, OrderedPairScorer, grid_positions
from evaluation.train_vsr_ordered_pair_student import (
    METRIC_KEYS,
    RestrictedFeatureLoader,
    TestSplitOpenedError,
    cross_image_donor_map,
    image_equal_stat,
    load_ordered_pair_npz,
    maybe_evaluate_internal_test,
    ordered_pair_loss_terms,
    ordered_pair_state_sha256,
    resolve_parameters,
    same_image_donor_map,
    save_ordered_pair_npz,
    slice_caption_prompt,
    validation_gate,
)

FEATURE_DIM = 1024


def _image_hash(char: str) -> str:
    return char * 64


def _write_cache(path: Path, sample_id: str, *, group_count: int = 4, tokens: int = 3, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    np.savez(
        path,
        group_mean=rng.normal(size=(group_count, FEATURE_DIM)).astype(np.float16),
        prompt_embeddings=rng.normal(size=(tokens, PROMPT_DIM)).astype(np.float16),
        teacher_scores=rng.normal(size=group_count).astype(np.float32),
        teacher_scores_normalized=np.zeros(group_count, dtype=np.float32),
        teacher_soft_target=np.full(group_count, 1.0 / group_count, dtype=np.float32),
        meta_sample_id=np.asarray(sample_id),
        meta_grid_h=np.asarray(2),
        meta_grid_w=np.asarray(2),
        meta_group_count=np.asarray(group_count),
        meta_prompt_token_count=np.asarray(tokens),
    )


def _synthetic_sample(sample_id: str, image: str, *, group_count: int = 4, tokens: int = 3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "sample_id": sample_id,
        "source_image_sha256": image,
        "group_mean": rng.normal(size=(group_count, FEATURE_DIM)).astype(np.float32),
        "prompt_embeddings": rng.normal(size=(tokens, PROMPT_DIM)).astype(np.float32),
        "positions": grid_positions(2, 2).numpy().astype(np.float32),
        "teacher_raw": rng.normal(size=group_count).astype(np.float32),
        "teacher_normalized": np.zeros(group_count, dtype=np.float32),
        "teacher_soft": np.full(group_count, 1.0 / group_count, dtype=np.float32),
    }


def _variant(*, available: bool = True, mean: float = 1.0, ci_low: float = 1.0) -> dict:
    entry = {"mean": mean, "ci_low": ci_low, "ci_high": mean + 0.5, "image_count": 2, "sample_count": 4}
    return {
        "available": available,
        "sample_count": 4 if available else 0,
        "image_count": 2 if available else 0,
        **{name: dict(entry) for name in METRIC_KEYS},
    }


class SameDonorTest(unittest.TestCase):
    def test_same_donor_cycles_within_one_image_only(self):
        rows = [
            {"sample_id": "c", "source_image_sha256": _image_hash("1")},
            {"sample_id": "a", "source_image_sha256": _image_hash("1")},
            {"sample_id": "b", "source_image_sha256": _image_hash("1")},
            {"sample_id": "d", "source_image_sha256": _image_hash("2")},
        ]
        donors = same_image_donor_map(rows)
        self.assertEqual(donors, {"a": "b", "b": "c", "c": "a"})
        by_id = {row["sample_id"]: row for row in rows}
        for sample_id, donor_id in donors.items():
            self.assertEqual(
                by_id[sample_id]["source_image_sha256"],
                by_id[donor_id]["source_image_sha256"],
            )

    def test_single_prompt_image_has_no_same_donor(self):
        rows = [{"sample_id": "solo", "source_image_sha256": _image_hash("a")}]
        self.assertEqual(same_image_donor_map(rows), {})

    def test_cross_donor_never_shares_an_image(self):
        rows = [
            {"sample_id": "a", "source_image_sha256": _image_hash("1")},
            {"sample_id": "b", "source_image_sha256": _image_hash("1")},
            {"sample_id": "c", "source_image_sha256": _image_hash("2")},
        ]
        donors = cross_image_donor_map(rows)
        by_id = {row["sample_id"]: row for row in rows}
        self.assertEqual(set(donors), {"a", "b", "c"})
        for sample_id, donor_id in donors.items():
            self.assertNotEqual(
                by_id[sample_id]["source_image_sha256"],
                by_id[donor_id]["source_image_sha256"],
            )
        self.assertEqual(cross_image_donor_map(rows[:1]), {})


class ImageEqualEstimateTest(unittest.TestCase):
    def test_image_equal_estimate_differs_from_sample_weighting(self):
        values = [0.0, 1.0, 1.0, 1.0]
        images = ["a", "b", "b", "b"]
        stat = image_equal_stat(values, images, seed=0, reps=0)
        self.assertAlmostEqual(stat["mean"], 0.5)
        self.assertAlmostEqual(float(np.mean(values)), 0.75)
        self.assertGreater(abs(stat["mean"] - float(np.mean(values))), 1e-9)
        self.assertEqual(stat["image_count"], 2)
        self.assertEqual(stat["sample_count"], 4)


class RestrictedLoaderTest(unittest.TestCase):
    def test_restricted_loader_never_opens_test_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_cache(root / "train.npz", "train-1", seed=0)
            records = [
                {
                    "sample_id": "train-1",
                    "npz": "train.npz",
                    "grid_h": 2,
                    "grid_w": 2,
                    "group_count": 4,
                    "source_image_sha256": _image_hash("a"),
                },
                {
                    "sample_id": "test-1",
                    "npz": "missing-test.npz",
                    "grid_h": 2,
                    "grid_w": 2,
                    "group_count": 4,
                    "source_image_sha256": _image_hash("b"),
                },
            ]
            loader = RestrictedFeatureLoader(records, root, allowed_sample_ids=["train-1"])
            loader.load("train-1")
            self.assertEqual(loader.opened_sample_ids, ["train-1"])
            with self.assertRaises(TestSplitOpenedError):
                loader.load("test-1")
            self.assertEqual(loader.opened_sample_ids, ["train-1"])
            self.assertEqual(loader.blocked_sample_ids, ["test-1"])


class InternalTestGateTest(unittest.TestCase):
    def test_failed_gate_never_builds_the_test_loader(self):
        calls: list[str] = []

        def factory():
            calls.append("built")
            raise AssertionError("test loader must not be constructed")

        result = maybe_evaluate_internal_test(
            gate_pass=False,
            test_loader_factory=factory,
            evaluate_fn=lambda loader: {"evaluated": True},
        )
        self.assertEqual(calls, [])
        self.assertIsNone(result["internal_test"])
        self.assertEqual(result["test_npz_opened"], [])

    def test_passing_gate_records_opened_test_ids(self):
        class FakeLoader:
            opened_sample_ids = ["test-1", "test-2"]

        result = maybe_evaluate_internal_test(
            gate_pass=True,
            test_loader_factory=FakeLoader,
            evaluate_fn=lambda loader: {"evaluated": True},
        )
        self.assertEqual(result["internal_test"], {"evaluated": True})
        self.assertEqual(result["test_npz_opened"], ["test-1", "test-2"])

    def test_validation_gate_requires_cross_and_same_positive(self):
        val = {
            "correct_minus_l2": _variant(mean=1.0),
            "correct_minus_cross": _variant(ci_low=1.0),
            "correct_minus_same": _variant(ci_low=1.0),
        }
        self.assertTrue(validation_gate(val)["pass"])
        bad_same = dict(val, correct_minus_same=_variant(ci_low=-0.1))
        self.assertFalse(validation_gate(bad_same)["pass"])
        bad_l2 = dict(val, correct_minus_l2=_variant(mean=0.0))
        self.assertFalse(validation_gate(bad_l2)["pass"])


class CaptionOverflowTest(unittest.TestCase):
    def test_caption_overflow_fails_closed(self):
        prompt = np.zeros((20, PROMPT_DIM), dtype=np.float16)
        overflow = {"start": 2, "end_exclusive": 19, "full_token_count": 20, "caption_token_count": 17}
        with self.assertRaisesRegex(ValueError, "outside"):
            slice_caption_prompt(prompt, overflow, 16)
        empty = {"start": 2, "end_exclusive": 2, "full_token_count": 20}
        with self.assertRaises(ValueError):
            slice_caption_prompt(prompt, empty, 16)
        mismatch = {"start": 0, "end_exclusive": 5, "full_token_count": 19}
        with self.assertRaisesRegex(ValueError, "mismatch"):
            slice_caption_prompt(prompt, mismatch, 16)
        valid = {"start": 4, "end_exclusive": 20, "full_token_count": 20, "caption_token_count": 16}
        self.assertEqual(slice_caption_prompt(prompt, valid, 16).shape, (16, PROMPT_DIM))


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_roundtrip_and_content_hash(self):
        model = OrderedPairScorer(seed=3, max_prompt_tokens=4, partner_k=2)
        state = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed-3.npz"
            info = save_ordered_pair_npz(path, state, {"seed": 3})
            loaded = load_ordered_pair_npz(path)
        self.assertEqual(info["state_sha256"], ordered_pair_state_sha256(state))
        self.assertEqual(set(loaded), set(state))
        for name, tensor in state.items():
            np.testing.assert_array_equal(loaded[name], tensor.detach().cpu().numpy())
        self.assertEqual(ordered_pair_state_sha256(loaded), info["state_sha256"])


class TrainingLossTest(unittest.TestCase):
    def test_small_training_loss_backward_is_finite(self):
        cfg = resolve_parameters({})
        model = OrderedPairScorer(seed=5, init_std=0.02, max_prompt_tokens=16, partner_k=4)
        sample = _synthetic_sample("s1", _image_hash("a"), seed=1)
        donor_prompt = _synthetic_sample("s2", _image_hash("a"), seed=2)["prompt_embeddings"]

        no_donor_loss, no_donor_info = ordered_pair_loss_terms(model, sample, None, cfg, "cpu")
        self.assertTrue(torch.isfinite(no_donor_loss))
        self.assertEqual(no_donor_info["contrast"], 0.0)

        loss, info = ordered_pair_loss_terms(model, sample, donor_prompt, cfg, "cpu")
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(info["contrast"], 0.0)
        loss.backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in grads))


if __name__ == "__main__":
    unittest.main()
