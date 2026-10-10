#!/usr/bin/env python3
"""Focused CPU tests for the train-only pair-loss feasibility / gradient audit.

Coverage:

* the draft pair-loss arithmetic (``L_base`` / ``D_swap`` / ``L_pair`` /
  ``L_total`` / adaptive-margin slack) on known distributions;
* the teacher-optimum identity (``L_base = 0``, ``D_swap = symmetric_KL``,
  ``L_pair = 0``) verified with the actual torch implementation;
* a float64 ``torch.autograd.gradcheck`` of a small synthetic pair loss;
* the scorer convention equals the frozen ablation ``prompt_mean_vector``;
* the hard isolation contract (split from index metadata only, only train npz
  opened, corrupted val/test npz cannot move any train metric or hash);
* a static source test forbidding any parameter-update / weight-save construct.

No training, no model, no CUDA, no network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from evaluation import pair_loss_feasibility_audit as audit
from evaluation.frozen_scorer_reduction_ablation import (
    prompt_cosine_matrix,
    reduce_cosine_matrix,
)
from evaluation.prompt_projection_distillation import (
    FEATURE_DIM,
    PROMPT_DIM,
    assign_split,
    load_distillation_config,
    save_feature_cache,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / "configs" / "prompt_projection_distillation.json"
REAL_DRAFT = PROJECT_ROOT / audit.DRAFT_RELATIVE_PATH
SOURCE_PATH = PROJECT_ROOT / "evaluation" / "pair_loss_feasibility_audit.py"
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")


def _hash(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


class SourceSafetyTests(unittest.TestCase):
    """The audit must not be able to update or persist any parameter."""

    def test_no_forbidden_update_or_optimizer_tokens(self) -> None:
        for token in audit.FORBIDDEN_SOURCE_TOKENS:
            self.assertNotIn(token, SOURCE_TEXT, f"forbidden token {token!r} in audit source")
        self.assertIsNone(re.search(r"\.step\s*\(", SOURCE_TEXT))
        self.assertIsNone(
            re.search(r"(torch\.optim|optim\.(Adam|AdamW|SGD)|AdamW\s*\(|SGD\s*\()", SOURCE_TEXT)
        )

    def test_only_autograd_grad_and_no_backward(self) -> None:
        self.assertIn("autograd.grad", SOURCE_TEXT)
        self.assertNotIn(".backward(", SOURCE_TEXT)


class DraftContractTests(unittest.TestCase):
    def test_draft_status_and_constants(self) -> None:
        draft, digest = audit.load_draft_config(REAL_DRAFT)
        checks = audit.draft_contract_checks(draft)
        self.assertTrue(all(checks.values()), msg=str(checks))
        self.assertEqual(draft["status"], "DRAFT_REQUIRES_PI_APPROVAL")
        self.assertEqual(len(digest), 64)
        self.assertEqual(len(draft["optimization"]["seeds"]), 3)

    def test_run_audit_does_not_modify_the_draft(self) -> None:
        before = REAL_DRAFT.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
        self.assertEqual(before, REAL_DRAFT.read_bytes())


class TeacherOptimumFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.torch = audit._require_torch()
        self.y_g = np.array([0.6, 0.3, 0.1], dtype=np.float64)
        self.y_p = np.array([0.1, 0.2, 0.7], dtype=np.float64)

    def test_slack_is_symmetric_kl_minus_js_and_nonnegative(self) -> None:
        js = audit.js_divergence_nats(self.y_g, self.y_p)
        skl = audit.symmetric_kl_nats(self.y_g, self.y_p)
        self.assertAlmostEqual(skl, 0.5 * (
            audit.kl_divergence_nats(self.y_g, self.y_p)
            + audit.kl_divergence_nats(self.y_p, self.y_g)
        ), places=12)
        self.assertGreaterEqual(skl - js, 0.0)
        self.assertGreaterEqual(skl / js, 2.0)

    def test_slack_identity_over_random_pairs(self) -> None:
        rng = np.random.default_rng(20261002)
        for _ in range(50):
            p = rng.random(9)
            q = rng.random(9)
            js = audit.js_divergence_nats(p, q)
            skl = audit.symmetric_kl_nats(p, q)
            self.assertGreaterEqual(skl - js, -1e-12)
            if js > 0.0:
                self.assertGreaterEqual(skl / js, 2.0 - 1e-9)

    def test_teacher_optimum_hinge_is_zero(self) -> None:
        y_g, y_p = self.y_g, self.y_p
        js = audit.js_divergence_nats(y_g, y_p)
        skl = audit.symmetric_kl_nats(y_g, y_p)
        temperature = audit.STUDENT_TEMPERATURE
        # scores = T * log(y) makes softmax(scores / T) == y exactly.
        scores_g = self.torch.tensor(np.log(y_g) * temperature, dtype=self.torch.float64)
        scores_p = self.torch.tensor(np.log(y_p) * temperature, dtype=self.torch.float64)
        teacher_g = self.torch.tensor(y_g, dtype=self.torch.float64)
        teacher_p = self.torch.tensor(y_p, dtype=self.torch.float64)
        l_base, l_pair, d_swap, l_total, margin = audit.pair_loss_terms(
            scores_g, scores_p, teacher_g, teacher_p, js, temperature=temperature
        )
        self.assertAlmostEqual(float(l_base.item()), 0.0, places=10)
        self.assertAlmostEqual(float(d_swap.item()), skl, places=10)
        self.assertEqual(float(l_pair.item()), 0.0)
        self.assertAlmostEqual(float(l_total.item()), 0.0, places=10)
        self.assertLessEqual(float(margin.item()), 0.0)

    def test_teacher_math_statistics_flags_negative_slack(self) -> None:
        rows = [
            {
                "group_count": 3,
                "slack": -0.25,
                "slack_nonnegative": False,
                "js_nats": 0.5,
                "symmetric_kl": 0.25,
                "js_positive": True,
                "d_swap_over_js": 0.5,
            },
            {
                "group_count": 4,
                "slack": 0.75,
                "slack_nonnegative": True,
                "js_nats": 0.5,
                "symmetric_kl": 1.25,
                "js_positive": True,
                "d_swap_over_js": 2.5,
            },
        ]
        stats = audit.teacher_math_statistics(rows)
        self.assertEqual(stats["slack_negative_image_count"], 1)
        self.assertEqual(stats["slack_negative_group_counts"], [3])
        self.assertAlmostEqual(stats["slack_nonnegative_fraction"], 0.5)
        self.assertFalse(stats["min_slack_ge_0"])
        self.assertFalse(stats["min_ratio_ge_2"])


class ScorerConventionTests(unittest.TestCase):
    def test_student_distribution_sums_to_one(self) -> None:
        torch = audit._require_torch()
        scores = torch.linspace(-3.0, 3.0, 11)
        soft = audit.student_distribution(scores)
        self.assertAlmostEqual(float(soft.sum().item()), 1.0, places=6)

    def test_matches_frozen_prompt_mean_vector_reducer(self) -> None:
        torch = audit._require_torch()
        rng = np.random.default_rng(7)
        group_mean = rng.random((6, FEATURE_DIM)).astype(np.float64)
        prompt = rng.random((4, PROMPT_DIM)).astype(np.float64)
        Wt = rng.normal(0.0, 0.02, (128, PROMPT_DIM))
        Wv = rng.normal(0.0, 0.02, (128, FEATURE_DIM))
        got = audit.student_scores(
            torch.as_tensor(group_mean, dtype=torch.float64),
            torch.as_tensor(prompt, dtype=torch.float64),
            torch.as_tensor(Wt, dtype=torch.float64),
            torch.as_tensor(Wv, dtype=torch.float64),
        ).numpy()
        # Compare against the float64 internals of the frozen reducer (the public
        # score_candidate casts to float32, which would only test rounding).
        cosines, visual_unit, prompt_unit = prompt_cosine_matrix(
            group_mean, prompt, Wt, Wv
        )
        expected = reduce_cosine_matrix(
            "prompt_mean_vector", cosines, visual_unit, prompt_unit
        )
        np.testing.assert_allclose(got, expected, atol=1e-9, rtol=1e-9)


class GradcheckTests(unittest.TestCase):
    def _problem(self) -> tuple[Any, ...]:
        torch = audit._require_torch()
        generator = torch.Generator().manual_seed(1234)
        G, T, DP, DF, D = 5, 3, 7, 4, 6
        Wt = (torch.randn(D, DP, generator=generator, dtype=torch.float64) * 0.2).requires_grad_(True)
        Wv = (torch.randn(D, DF, generator=generator, dtype=torch.float64) * 0.2).requires_grad_(True)
        group_mean = torch.randn(G, DF, generator=generator, dtype=torch.float64)
        prompt_g = torch.randn(T, DP, generator=generator, dtype=torch.float64)
        prompt_p = torch.randn(T, DP, generator=generator, dtype=torch.float64)
        y_g = torch.softmax(torch.randn(G, generator=generator, dtype=torch.float64), dim=-1)
        y_p = torch.softmax(torch.randn(G, generator=generator, dtype=torch.float64), dim=-1)
        return Wt, Wv, group_mean, prompt_g, prompt_p, y_g, y_p

    def test_gradcheck_active_hinge_float64(self) -> None:
        torch = audit._require_torch()
        Wt, Wv, group_mean, prompt_g, prompt_p, y_g, y_p = self._problem()

        def func(wt: Any, wv: Any) -> Any:
            scores_g = audit.student_scores(group_mean, prompt_g, wt, wv)
            scores_p = audit.student_scores(group_mean, prompt_p, wt, wv)
            _, _, _, l_total, margin = audit.pair_loss_terms(scores_g, scores_p, y_g, y_p, 5.0)
            self.assertGreater(float(margin.item()), 0.0)
            return l_total

        self.assertTrue(
            torch.autograd.gradcheck(func, (Wt, Wv), eps=1e-6, atol=1e-5, rtol=1e-3)
        )

    def test_gradcheck_inactive_hinge_float64(self) -> None:
        torch = audit._require_torch()
        Wt, Wv, group_mean, prompt_g, prompt_p, y_g, y_p = self._problem()

        def func(wt: Any, wv: Any) -> Any:
            scores_g = audit.student_scores(group_mean, prompt_g, wt, wv)
            scores_p = audit.student_scores(group_mean, prompt_p, wt, wv)
            _, _, _, l_total, margin = audit.pair_loss_terms(scores_g, scores_p, y_g, y_p, -5.0)
            self.assertLess(float(margin.item()), 0.0)
            return l_total

        self.assertTrue(
            torch.autograd.gradcheck(func, (Wt, Wv), eps=1e-6, atol=1e-5, rtol=1e-3)
        )

    def test_l_total_equals_l_base_plus_l_pair(self) -> None:
        torch = audit._require_torch()
        _, _, group_mean, prompt_g, prompt_p, y_g, y_p = self._problem()
        scores_g = audit.student_scores(group_mean, prompt_g, torch.eye(6, 7, dtype=torch.float64), torch.eye(6, 4, dtype=torch.float64))
        scores_p = audit.student_scores(group_mean, prompt_p, torch.eye(6, 7, dtype=torch.float64), torch.eye(6, 4, dtype=torch.float64))
        l_base, l_pair, _, l_total, _ = audit.pair_loss_terms(scores_g, scores_p, y_g, y_p, 0.3)
        self.assertAlmostEqual(
            float(l_total.item()), float(l_base.item()) + float(l_pair.item()), places=12
        )


class SyntheticDataset:
    """Tiny train/val/test feature cache assigned by the frozen split salt."""

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
        self.records: list[dict[str, Any]] = []
        self.train_npz: set[str] = set()
        self.val_test_npz: set[str] = set()
        self.index_rows: list[dict[str, Any]] = []
        self._build(per_split)

    def _pick_hashes(self, per_split: Mapping[str, int]) -> dict[str, list[str]]:
        wanted = dict(per_split)
        chosen: dict[str, list[str]] = {"train": [], "val": [], "test": []}
        counter = 0
        while any(len(chosen[name]) < wanted[name] for name in wanted):
            image_hash = _hash(f"pair-loss-audit-test-image-{counter}")
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
        for split_name, image_hashes in chosen.items():
            for order, image_hash in enumerate(image_hashes):
                group_count = 8 if order % 2 == 0 else 16
                grid_w = 4
                grid_h = group_count // grid_w
                for task in ("grounding", "pointing"):
                    sample_id = f"plf-{split_name}-{order:03d}-{task}"
                    raw = np.linspace(0.0, 1.0, group_count).astype(np.float32)
                    raw[(1 if task == "grounding" else group_count - 1)] = 2.0
                    normalized = ((raw - raw.min()) / (raw.max() - raw.min())).astype(np.float32)
                    soft = (np.exp(normalized / 0.1) / np.exp(normalized / 0.1).sum()).astype(np.float32)
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

    def _run(self, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        return audit.run_audit(
            config_path=REAL_CONFIG,
            draft_path=REAL_DRAFT,
            index_path=self.dataset.index_path,
            features_dir=self.dataset.features_dir,
            **kwargs,
        )

    def test_run_isolates_and_reports_train_images(self) -> None:
        report, teacher_rows, init_rows = self._run()
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["split_counts"]["train"], 6)
        self.assertEqual(report["split_counts"]["val"], 4)
        self.assertEqual(report["split_counts"]["test"], 4)
        self.assertEqual(len(teacher_rows), 3)
        self.assertEqual(len(init_rows), 9)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])
        self.assertEqual(report["isolation"]["val_test_npz_opened"], [])
        self.assertTrue(all(report["draft_contract_checks"].values()))
        # every seed has finite, non-zero gradient diagnostics and a non-negative slack
        for seed_metrics in report["initialization_audit"]["per_seed"]:
            grad = seed_metrics["gradient"]["concat"]
            self.assertTrue(grad["l_base_finite"])
            self.assertTrue(grad["l_base_nonzero"])
            self.assertTrue(grad["l_total_finite"])
        self.assertFalse(report["teacher_math_audit"]["min_slack_ge_0"] is None)

    def test_only_train_feature_cache_is_opened(self) -> None:
        seen: list[str] = []
        real_loader = audit.load_feature_cache

        def spy(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(str(path))
            return real_loader(path, *args, **kwargs)

        report, _, _ = self._run(cache_loader=spy)
        self.assertEqual(set(seen), self.dataset.train_npz)
        self.assertEqual(set(seen) & self.dataset.val_test_npz, set())
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_loader_that_raises_on_non_train_paths(self) -> None:
        allowed = self.dataset.train_npz.copy()

        def strict(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if str(path) not in allowed:
                raise AssertionError(f"non-train cache opened: {path}")
            return audit.load_feature_cache(path, *args, **kwargs)

        report, _, _ = self._run(cache_loader=strict)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_corrupting_val_test_npz_cannot_move_train_metrics_or_hash(self) -> None:
        report_before, teacher_before, init_before = self._run()
        for path in self.dataset.val_test_npz:
            Path(path).write_bytes(b"this is not a valid npz file")
        report_after, teacher_after, init_after = self._run()
        self.assertEqual(teacher_before, teacher_after)
        self.assertEqual(init_before, init_after)
        self.assertEqual(
            report_before["provenance"]["input_npz_aggregate_sha256"],
            report_after["provenance"]["input_npz_aggregate_sha256"],
        )
        self.assertEqual(
            report_before["teacher_math_audit"], report_after["teacher_math_audit"]
        )
        self.assertEqual(
            report_before["initialization_audit"], report_after["initialization_audit"]
        )
        self.assertEqual(
            report_before["provenance"]["index_sha256"],
            report_after["provenance"]["index_sha256"],
        )

    def test_index_metadata_load_opens_no_npz(self) -> None:
        import unittest.mock as mock

        with mock.patch.object(
            audit, "load_feature_cache", side_effect=AssertionError("npz opened")
        ):
            metadata = audit.load_index_metadata(self.dataset.index_path)
        self.assertEqual(len(metadata), 14)

    def test_hardware_contract_is_declared(self) -> None:
        report, _, _ = self._run()
        contract = report["initialization_audit"]["hardware_contract"]
        self.assertEqual(contract["query_shape"], [1, 128])
        self.assertEqual(contract["scorer"], "Gx128 dot")


class MarkdownAndWriterTests(unittest.TestCase):
    def test_markdown_and_csv_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            report, teacher_rows, init_rows = audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
            markdown = audit.render_markdown(report)
            self.assertIn("train-only", markdown)
            self.assertIn("teacher optimum", markdown)
            self.assertIn("不训练", markdown)
            self.assertIn("不是重训成功的证据", markdown)
            self.assertIn("slack", markdown)
            teacher_csv = Path(tmp) / "teacher.csv"
            init_csv = Path(tmp) / "init.csv"
            audit.write_teacher_csv(teacher_csv, teacher_rows)
            audit.write_init_csv(init_csv, init_rows)
            self.assertEqual(
                teacher_csv.read_text(encoding="utf-8").splitlines()[0].split(","),
                list(audit.TEACHER_CSV_FIELDS),
            )
            self.assertEqual(
                init_csv.read_text(encoding="utf-8").splitlines()[0].split(","),
                list(audit.INIT_CSV_FIELDS),
            )
            self.assertEqual(
                len(init_csv.read_text(encoding="utf-8").strip().splitlines()),
                len(init_rows) + 1,
            )


if __name__ == "__main__":
    unittest.main()
