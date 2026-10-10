#!/usr/bin/env python3
"""CPU-only tests for the 2026-10-04 current-Selector CV-Bench preflight.

These tests never load the 4B model, never touch a GPU and never read a
proposed CV-Bench output.  They exercise the frozen integer bridge, the minimal
B1 integer score mode plumbing, the runner CLI contract and the official
summariser's complete/incomplete gating.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import current_selector_integer as csi  # noqa: E402
from evaluation import selector_integer_scorer_audit as audit  # noqa: E402
from evaluation import selector_int8_numerical_audit as base  # noqa: E402
from evaluation.hardware_regular_b1_integration import (  # noqa: E402
    ALLOWED_SCORE_MODES,
    SCORE_MODE_INTEGER,
)
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    select_hardware_regular_groups,
)
from evaluation.run_hf_current_selector_cvbench import (  # noqa: E402
    INT32_BOUND_KEYS,
    build_parser,
    int32_bounds_present,
    measurement_schedule,
    validate_args,
)
from evaluation.summarize_current_selector_cvbench import (  # noqa: E402
    FORBIDDEN_PRIMARY_METRICS,
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    summarise_one,
)

SEED = 20261004


def synthetic_group_mean(groups: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((groups, csi.audit.EMBED_DIM))


def synthetic_prompt(tokens: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((tokens, 2560))


class DeploymentFrozenTest(unittest.TestCase):
    def test_deployment_projection_and_wv_pin(self) -> None:
        projection, meta = csi.load_deployment_projection(ROOT)
        self.assertEqual(meta["seed"], csi.DEPLOYMENT_SEED)
        self.assertEqual(meta["sha256"], csi.DEPLOYMENT_CHECKPOINT_SHA256)
        self.assertEqual(tuple(np.asarray(projection.Wt).shape), (128, 2560))
        self.assertEqual(tuple(np.asarray(projection.Wv).shape), (128, 1024))
        bundle = csi.load_deployment_wv_bundle(ROOT)
        self.assertEqual(bundle["wv_sha256"], csi.DEPLOYMENT_WV_SHA256)
        self.assertEqual(bundle["scale_sha256"], csi.DEPLOYMENT_WV_SCALE_SHA256)
        self.assertEqual(tuple(bundle["wv_q"].shape), (128, 1024))
        self.assertTrue(np.issubdtype(bundle["wv_q"].dtype, np.integer))
        self.assertLessEqual(int(np.max(np.abs(bundle["wv_q"]))), 127)
        self.assertGreater(bundle["wv_scale"], 0.0)

    def test_seed_20261003_is_the_hls_referenced_bundle(self) -> None:
        fixture = (
            ROOT
            / "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls"
            / "fixtures/manifest.json"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        self.assertEqual(payload["wv"]["source_sha256"], csi.DEPLOYMENT_WV_SHA256)
        self.assertIn("20261003", payload["wv"]["source_path"])


class IntegerChainTest(unittest.TestCase):
    def test_integer_scores_matches_audit_chain(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        group_mean = synthetic_group_mean(50, rng)
        prompt = synthetic_prompt(7, rng)
        result = csi.integer_scores(
            group_mean, prompt, projection.Wt, bundle["wv_q"], 10, 5, budgets=[48]
        )
        mean_q, scale = base.quantize_activation(group_mean)
        query = base.primary_quantized_query(prompt, projection.Wt)
        reference = audit.integer_chain(
            mean_q, query["query_q"], bundle["wv_q"], 10, 5, budgets=[48]
        )
        np.testing.assert_array_equal(result["score_q"], reference["score_q"])
        self.assertEqual(int(result["score_q"].max()), int(reference["score_q"].max()))
        self.assertGreaterEqual(int(result["score_q"].min()), 0)
        self.assertLessEqual(int(result["score_q"].max()), 32767)
        self.assertIn("projection_int_absmax", result["bounds"])

    def test_integer_scores_rejects_grid_mismatch(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        with self.assertRaises(csi.CurrentSelectorError):
            csi.integer_scores(
                synthetic_group_mean(50, rng),
                synthetic_prompt(7, rng),
                projection.Wt,
                bundle["wv_q"],
                7,
                7,
                budgets=[48],
            )

    def test_selection_is_deterministic_and_budget_exact(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        hidden = rng.standard_normal((100, 4, 1024))
        group_mean = hidden.mean(axis=1)
        payload = csi.integer_scores(
            group_mean,
            synthetic_prompt(6, rng),
            projection.Wt,
            bundle["wv_q"],
            10,
            10,
            budgets=[64],
        )
        first = csi.select_from_score_q(hidden, payload["score_q"], 10, 10, 64)
        second = csi.select_from_score_q(hidden, payload["score_q"], 10, 10, 64)
        self.assertEqual(first.output_group_count, 64)
        np.testing.assert_array_equal(first.group_indices, second.group_indices)
        np.testing.assert_array_equal(first.groups, second.groups)
        self.assertTrue(np.all(np.diff(first.group_indices) > 0))
        self.assertTrue(np.all(first.valid))

    def test_selection_matches_explicit_combined(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        hidden = rng.standard_normal((100, 4, 1024))
        group_mean = hidden.mean(axis=1)
        payload = csi.integer_scores(
            group_mean,
            synthetic_prompt(6, rng),
            projection.Wt,
            bundle["wv_q"],
            10,
            10,
            budgets=[96],
        )
        combined = csi.score_q_to_combined(payload["score_q"])
        reference = select_hardware_regular_groups(
            hidden, 10, 10, budget=96, combined_scores=combined
        )
        actual = csi.select_from_score_q(hidden, payload["score_q"], 10, 10, 96)
        np.testing.assert_array_equal(actual.group_indices, reference.group_indices)
        np.testing.assert_array_equal(actual.groups, reference.groups)

    def test_identity_bypass_when_groups_at_most_budget(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        hidden = rng.standard_normal((9, 4, 1024))
        payload = csi.integer_scores(
            hidden.mean(axis=1),
            synthetic_prompt(4, rng),
            projection.Wt,
            bundle["wv_q"],
            3,
            3,
            budgets=[48],
        )
        selection = csi.select_from_score_q(hidden, payload["score_q"], 3, 3, 48)
        self.assertTrue(selection.bypass)
        self.assertEqual(selection.output_group_count, 9)
        np.testing.assert_array_equal(selection.groups, hidden)

    def test_integer_scorer_callback_records_payload(self) -> None:
        rng = np.random.default_rng(SEED)
        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        scorer = csi.IntegerSelectorScorer(
            projection.Wt, bundle["wv_q"], budgets=[48, 64]
        )
        hidden = rng.standard_normal((100, 4, 1024))
        payload = scorer(hidden, 10, 10, 64, synthetic_prompt(6, rng))
        self.assertEqual(payload["combined_scores"].shape, (100,))
        self.assertIs(scorer.last_payload, payload)
        self.assertEqual(SCORE_MODE_INTEGER in ALLOWED_SCORE_MODES, True)


class RunnerCliTest(unittest.TestCase):
    def _base_args(self, mode: str, **extra: object) -> list[str]:
        argv = [
            "--model",
            "/tmp/model",
            "--model-label",
            "m",
            "--manifest",
            "/tmp/m.jsonl",
            "--output",
            "/tmp/out.jsonl",
            "--mode",
            mode,
        ]
        for key, value in extra.items():
            argv.extend([f"--{key.replace('_', '-')}", str(value)])
        return argv

    def test_budget_required_for_full(self) -> None:
        parser = build_parser()
        args = parser.parse_args(self._base_args("full"))
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

    def test_identity_rejects_budget(self) -> None:
        parser = build_parser()
        args = parser.parse_args(self._base_args("identity", budget=64))
        self.assertEqual(args.mode, "identity")
        with self.assertRaises(SystemExit):
            validate_args(parser, args)

    def test_identity_without_budget_is_valid(self) -> None:
        parser = build_parser()
        args = parser.parse_args(self._base_args("identity"))
        validate_args(parser, args)
        self.assertIsNone(args.budget)

    def test_latency_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(self._base_args("latency", budget=64))
        validate_args(parser, args)
        self.assertEqual(args.repeats, 3)
        self.assertEqual(args.warmup, 1)

    def test_int32_bounds_are_not_required_for_identity_bypass(self) -> None:
        self.assertTrue(int32_bounds_present({}, bypass=True))

    def test_int32_bounds_are_required_when_selector_runs(self) -> None:
        self.assertFalse(int32_bounds_present({}, bypass=False))
        bounds = {key: 0 for key in INT32_BOUND_KEYS}
        self.assertTrue(int32_bounds_present(bounds, bypass=False))


class MeasurementScheduleTest(unittest.TestCase):
    def test_latency_warms_up_once_then_records_three_repeats(self) -> None:
        self.assertEqual(
            measurement_schedule("latency", 1, 3),
            {"warmup": 1, "repeats": 3},
        )

    def test_latency_without_warmup_still_records_all_repeats(self) -> None:
        self.assertEqual(
            measurement_schedule("latency", 0, 3),
            {"warmup": 0, "repeats": 3},
        )

    def test_full_never_warms_up_and_runs_once(self) -> None:
        self.assertEqual(
            measurement_schedule("full", 1, 3),
            {"warmup": 0, "repeats": 1},
        )

    def test_non_latency_modes_ignore_warmup_and_repeats(self) -> None:
        for mode in ("identity", "smoke"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    measurement_schedule(mode, 5, 7),
                    {"warmup": 0, "repeats": 1},
                )


class ResumeStateTest(unittest.TestCase):
    def _record(self, sample_id: str, *, mode: str, budget: int, seed: int = 20261003) -> dict:
        return {
            "sample_id": sample_id,
            "mode": mode,
            "requested_budget": budget,
            "deployment_seed": seed,
            "score_mode": SCORE_MODE_INTEGER,
        }

    def test_resume_collects_ids_and_rejects_config_drift(self) -> None:
        from evaluation.run_hf_current_selector_cvbench import resume_state

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.jsonl"
            rows = [
                self._record("0", mode="full", budget=64),
                self._record("1", mode="full", budget=64),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            done = resume_state(path, mode="full", budget=64)
            self.assertEqual(done, {"0", "1"})
            with self.assertRaises(RuntimeError):
                resume_state(path, mode="full", budget=48)

            path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in [
                        self._record("0", mode="full", budget=64),
                        self._record("0", mode="full", budget=64),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                resume_state(path, mode="full", budget=64)


def _tiny_manifest() -> list[dict]:
    return [
        {
            "sample_id": "0",
            "task": "Count",
            "type": "2D",
            "source": "ADE20K",
            "source_dataset": "ADE20K Validation Set",
            "choices": ["1", "2"],
            "answer": "(A)",
        },
        {
            "sample_id": "1",
            "task": "Relation",
            "type": "2D",
            "source": "COCO",
            "source_dataset": "COCO",
            "choices": ["1", "2"],
            "answer": "(B)",
        },
        {
            "sample_id": "2",
            "task": "Depth",
            "type": "3D",
            "source": "Omni3D",
            "source_dataset": "Omni3D",
            "choices": ["1", "2"],
            "answer": "(A)",
        },
        {
            "sample_id": "3",
            "task": "Distance",
            "type": "3D",
            "source": "Omni3D",
            "source_dataset": "Omni3D",
            "choices": ["1", "2"],
            "answer": "(B)",
        },
    ]


class SummarizerTest(unittest.TestCase):
    def _native(self) -> dict:
        return {
            "0": {"output_text": "(A)"},
            "1": {"output_text": "(B)"},
            "2": {"output_text": "(A)"},
            "3": {"output_text": "(A)"},
        }

    def _dense(self) -> dict:
        return {
            "0": {"output_text": "(A)"},
            "1": {"output_text": "(B)"},
            "2": {"output_text": "(A)"},
            "3": {"output_text": "(B)"},
        }

    def test_partial_candidate_is_in_progress_without_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.jsonl"
            path.write_text(
                json.dumps({"sample_id": "0", "output_text": "(A)"}) + "\n",
                encoding="utf-8",
            )
            result = summarise_one(
                label="K48",
                candidate_jsonl=path,
                manifest=_tiny_manifest(),
                native=self._native(),
                dense=self._dense(),
                seed=1,
                bootstrap_repeats=20,
            )
        self.assertEqual(result["status"], STATUS_IN_PROGRESS)
        self.assertNotIn("candidate", result)
        self.assertEqual(result["completed"], 1)

    def test_complete_candidate_uses_official_scorer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.jsonl"
            rows = [
                {"sample_id": "0", "output_text": "(A)"},
                {"sample_id": "1", "output_text": "(B)"},
                {"sample_id": "2", "output_text": "(A)"},
                {"sample_id": "3", "output_text": "(B)"},
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
            )
            result = summarise_one(
                label="K48",
                candidate_jsonl=path,
                manifest=_tiny_manifest(),
                native=self._native(),
                dense=self._dense(),
                seed=1,
                bootstrap_repeats=20,
            )
        self.assertEqual(result["status"], STATUS_COMPLETE)
        self.assertEqual(result["candidate"]["sample_count"], 4)
        self.assertAlmostEqual(result["candidate"]["cv_bench_accuracy"], 1.0)
        self.assertAlmostEqual(result["dense_bf16_reference"]["cv_bench_accuracy"], 1.0)
        self.assertIn("paired_bootstrap_ci95_vs_dense", result)
        score_block = json.dumps(
            {
                "candidate": result["candidate"],
                "delta": result["absolute_delta_vs_dense"],
                "retention": result["relative_retention_vs_dense"],
            }
        ).lower()
        for token in FORBIDDEN_PRIMARY_METRICS:
            self.assertNotIn(token, score_block)


class IntegerIntegrationTest(unittest.TestCase):
    def test_b1_integer_mode_wires_through_and_matches_selection(self) -> None:
        from evaluation.hardware_regular_b1_integration import (
            forward_hardware_regular_b1_vision,
        )
        from evaluation.tests.test_hardware_regular_b1_integration import (
            _FakeVisual,
            _grid,
            _pixels,
            _text_embeddings,
        )

        projection, _ = csi.load_deployment_projection(ROOT)
        bundle = csi.load_deployment_wv_bundle(ROOT)
        grid, budget = 18, 48  # G = 81 > 48 -> compression after block 0
        scorer = csi.IntegerSelectorScorer(projection.Wt, bundle["wv_q"])
        result = forward_hardware_regular_b1_vision(
            _FakeVisual(),
            _pixels(grid),
            _grid(grid),
            budget=budget,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=_text_embeddings(5),
            integer_scorer=scorer,
        )
        self.assertIsNotNone(result.integer_payload)
        self.assertEqual(result.integer_payload["combined_scores"].shape, (81,))
        self.assertEqual(result.image_embeds.shape[0], budget)
        reference = csi.select_from_score_q(
            result.block0_hidden.detach().numpy().reshape(81, 4, 1024),
            result.integer_payload["score_q"],
            9,
            9,
            budget,
        )
        np.testing.assert_array_equal(
            result.selection.group_indices, reference.group_indices
        )

    def test_b1_integer_mode_requires_callback(self) -> None:
        from evaluation.hardware_regular_b1_integration import (
            forward_hardware_regular_b1_vision,
        )
        from evaluation.tests.test_hardware_regular_b1_integration import (
            _FakeVisual,
            _grid,
            _pixels,
            _text_embeddings,
        )

        with self.assertRaises(Exception):
            forward_hardware_regular_b1_vision(
                _FakeVisual(),
                _pixels(18),
                _grid(18),
                budget=48,
                score_mode=SCORE_MODE_INTEGER,
                text_embeddings=_text_embeddings(5),
            )


if __name__ == "__main__":
    unittest.main()
