#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.selector_int8_numerical_audit`.

Covers the authority-prompt-required surface:

* round-half-away-from-zero activation / weight / Q0.15 quantisation,
* exact INT32 accumulation and the analytic INT32 length bounds,
* the two deterministic quantised query chains (primary / descriptive control),
* selected-set rank equivalence with the hardware ``score_q`` code,
* per-record metrics, cross-seed median aggregation and image-level summaries,
* the frozen ``INT8_NUMERICS_PASS`` decision rule,
* the compact hardware bundle and its deterministic fixtures (no ``Wt`` export),
* pinned-input verification and report rendering.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import selector_int8_numerical_audit as audit
from evaluation.frozen_scorer_reduction_ablation import (
    prompt_cosine_matrix,
    score_candidate,
)
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.prompt_projection_distillation import l2_baseline_scores

GROUP_COUNT = 144
GRID_H = 12
GRID_W = 12
TOKENS = 7


def _projections(seed: int = 0):
    rng = np.random.default_rng(seed)
    Wt = rng.normal(size=(128, 2560)).astype(np.float64)
    Wv = rng.normal(size=(128, 1024)).astype(np.float64)
    return Wt, Wv


def _record_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    group_mean = rng.normal(size=(GROUP_COUNT, 1024)).astype(np.float64)
    prompt_correct = rng.normal(size=(TOKENS, 2560)).astype(np.float64)
    prompt_wrong = rng.normal(size=(TOKENS + 2, 2560)).astype(np.float64)
    target = rng.random(GROUP_COUNT)
    target = target / target.sum()
    return group_mean, prompt_correct, prompt_wrong, target


def _raw_by_seed(seeds=(20261003, 20261004, 20261005)):
    """Synthetic frozen aggregation input with 2 images x 2 tasks."""
    sample_meta = {
        "g1": {"task": "grounding", "image_sha256": "img1", "group_count": GROUP_COUNT,
               "grid_h": GRID_H, "grid_w": GRID_W},
        "p1": {"task": "pointing", "image_sha256": "img1", "group_count": GROUP_COUNT,
               "grid_h": GRID_H, "grid_w": GRID_W},
        "g2": {"task": "grounding", "image_sha256": "img2", "group_count": GROUP_COUNT,
               "grid_h": GRID_H, "grid_w": GRID_W},
        "p2": {"task": "pointing", "image_sha256": "img2", "group_count": GROUP_COUNT,
               "grid_h": GRID_H, "grid_w": GRID_W},
    }
    raw = {}
    for offset, seed in enumerate(seeds):
        raw[seed] = {}
        for arm in (audit.PRIMARY_ARM, audit.CONTROL_ARM, audit.FLOAT_ARM):
            raw[seed][arm] = {}
            for index, sid in enumerate(sorted(sample_meta)):
                base = 0.0 if arm == audit.FLOAT_ARM else float(offset) * 0.001
                role_map = {}
                for role in audit.ROLES:
                    role_map[role] = {
                        budget: {
                            "metrics": {
                                "mass": 0.5 + base,
                                "ndcg": 0.6 + base,
                                "selected_count": 48.0,
                                "jaccard": 1.0 if arm == audit.FLOAT_ARM else 0.98,
                                "exact_match": 1.0,
                                "mass_delta": 0.0 if arm == audit.FLOAT_ARM else 0.01,
                                "ndcg_delta": 0.0 if arm == audit.FLOAT_ARM else 0.01,
                            }
                        }
                        for budget in audit.BUDGETS
                    }
                role_map["spearman_correct"] = 0.99
                role_map["spearman_wrong"] = 0.97
                raw[seed][arm][sid] = role_map
    return raw, sample_meta


class QuantisationPrimitiveTests(unittest.TestCase):
    def test_round_half_away_from_zero(self):
        values = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 0.4999999, -0.4999999])
        rounded = audit.round_half_away_from_zero(values)
        np.testing.assert_array_equal(rounded, [1, 2, 3, -1, -2, -3, 0, 0])

    def test_activation_quantisation_scale_and_bounds(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=(20, 30)) * 3.7
        quantized, scale = audit.quantize_activation(values)
        self.assertAlmostEqual(scale, 128.0 / float(np.abs(values).max()))
        self.assertGreaterEqual(int(quantized.min()), audit.INT8_MIN)
        self.assertLessEqual(int(quantized.max()), audit.INT8_MAX)
        reconstructed = quantized.astype(np.float64) / scale
        self.assertTrue(np.all(np.abs(reconstructed - values) <= scale))

    def test_activation_zero_tensor(self):
        quantized, scale = audit.quantize_activation(np.zeros((4, 5)))
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(quantized, np.zeros((4, 5)))

    def test_weight_row_quantisation(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(8, 16)) * 0.5
        quantized, scales = audit.quantize_weight_rows(matrix)
        np.testing.assert_allclose(scales, np.abs(matrix).max(axis=1) / 127.0)
        self.assertGreaterEqual(int(quantized.min()), audit.WEIGHT_MIN)
        self.assertLessEqual(int(quantized.max()), audit.WEIGHT_MAX)
        reconstructed = quantized.astype(np.float64) * scales[:, None]
        self.assertTrue(np.all(np.abs(reconstructed - matrix) <= scales[:, None]))

    def test_weight_zero_row_keeps_scale_one(self):
        matrix = np.zeros((2, 5))
        quantized, scales = audit.quantize_weight_rows(matrix)
        np.testing.assert_array_equal(scales, [1.0, 1.0])
        np.testing.assert_array_equal(quantized, np.zeros((2, 5)))

    def test_score_q_pack_and_clamp(self):
        combined = np.array([0.0, 0.5, 1.0, 1.5, -0.5, 1.0 / 65536.0])
        packed = audit.quantize_score_q(combined)
        expected = np.array([0, 16384, 32767, 32767, 0, 1], dtype=np.uint16)
        np.testing.assert_array_equal(packed, expected)
        self.assertEqual(packed.dtype, np.uint16)
        np.testing.assert_allclose(audit.dequantize_score_q(packed), packed / 32768.0)


class Int32Tests(unittest.TestCase):
    def test_analytical_bounds_fit_int32(self):
        bounds = audit.analytical_int32_bounds()
        self.assertTrue(bounds["all_fit_int32"])
        self.assertEqual(bounds["lengths"]["group_mean_dot_wv_row"]["bound"], 1024 * 128 * 127)
        self.assertEqual(bounds["lengths"]["prompt_dot_wt_row"]["bound"], 2560 * 128 * 127)
        self.assertEqual(bounds["lengths"]["query_dot"]["bound"], 128 * 128 * 127)
        for entry in bounds["lengths"].values():
            self.assertLess(entry["bound"], audit.INT32_MAX)

    def test_exact_int_matmul_matches_numpy(self):
        rng = np.random.default_rng(4)
        left = rng.integers(-128, 128, size=(9, 1024))
        right = rng.integers(-127, 128, size=(6, 1024))
        exact = audit.exact_int_matmul(left, right.T)
        reference = left.astype(np.int64) @ right.astype(np.int64).T
        np.testing.assert_array_equal(exact, reference)
        self.assertEqual(exact.dtype, np.int64)


class QueryChainTests(unittest.TestCase):
    def test_primary_query_matches_frozen_float_query(self):
        Wt, Wv = _projections(5)
        group_mean, prompt, _, _ = _record_data(6)
        query = audit.float_prompt_query(prompt, Wt)
        # Reproduce the frozen prompt_mean_vector query via the shared reducer.
        _, _, prompt_unit = prompt_cosine_matrix(group_mean, prompt, Wt, Wv)
        expected = prompt_unit.mean(axis=0)
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(query, expected, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(query)), 1.0, places=12)

    def test_primary_query_quantisation_roundtrip(self):
        Wt, _ = _projections(7)
        prompt = np.random.default_rng(8).normal(size=(TOKENS, 2560))
        entry = audit.primary_quantized_query(prompt, Wt)
        self.assertEqual(entry["query_q"].shape, (128,))
        self.assertEqual(entry["query_q"].dtype, np.int8)
        self.assertGreaterEqual(int(entry["query_q"].min()), audit.INT8_MIN)
        self.assertLessEqual(int(entry["query_q"].max()), audit.INT8_MAX)
        self.assertAlmostEqual(float(np.linalg.norm(entry["query_hat"])), 1.0, places=12)

    def test_control_query_uses_int32_projection(self):
        Wt, _ = _projections(9)
        prompt = np.random.default_rng(10).normal(size=(TOKENS, 2560))
        wt_q, wt_scales = audit.quantize_weight_rows(Wt)
        entry = audit.control_quantized_query(prompt, Wt, wt_q, wt_scales)
        self.assertEqual(entry["projection_int"].shape, (TOKENS, 128))
        self.assertAlmostEqual(float(np.linalg.norm(entry["query_hat"])), 1.0, places=12)
        self.assertLess(audit.max_abs_int32(entry["projection_int"]), audit.INT32_MAX)

    def test_analytical_bound_dominates_observed(self):
        Wt, Wv = _projections(11)
        group_mean, prompt, _, _ = _record_data(12)
        gm_q, gm_scale = audit.quantize_activation(group_mean)
        wv_q, wv_scales = audit.quantize_weight_rows(Wv)
        projection = audit.quantized_visual_projection(gm_q, gm_scale, wv_q, wv_scales)
        self.assertLessEqual(
            audit.max_abs_int32(projection["projection_int"]),
            audit.analytical_int32_bound(1024)["bound"],
        )


class SelectionEquivalenceTests(unittest.TestCase):
    def test_selection_on_score_q_equals_score_hat(self):
        rng = np.random.default_rng(13)
        combined = rng.random(GROUP_COUNT)
        score_q = audit.quantize_score_q(combined)
        score_hat = audit.dequantize_score_q(score_q)
        for budget in audit.BUDGETS:
            left = audit.selected_indices(score_q.astype(np.float64), GRID_H, GRID_W, budget)
            right = audit.selected_indices(score_hat, GRID_H, GRID_W, budget)
            self.assertEqual(left, right)

    def test_quantized_combined_scores_shapes(self):
        rng = np.random.default_rng(14)
        visual = rng.normal(size=(GROUP_COUNT, 128))
        query = rng.normal(size=128)
        l2 = rng.random(GROUP_COUNT)
        combined, score_q, score_hat = audit.quantized_combined_scores(visual, query, l2)
        self.assertEqual(combined.shape, (GROUP_COUNT,))
        self.assertEqual(score_q.dtype, np.uint16)
        np.testing.assert_allclose(score_hat, score_q / 32768.0)


class RecordAndAggregationTests(unittest.TestCase):
    def _evaluate(self):
        Wt, Wv = _projections(15)
        group_mean, prompt, wrong, target = _record_data(16)
        wt_q, wt_scales = audit.quantize_weight_rows(Wt)
        wv_q, wv_scales = audit.quantize_weight_rows(Wv)
        return audit.evaluate_record(
            group_mean=group_mean,
            prompt_correct=prompt,
            prompt_wrong=wrong,
            Wt=Wt,
            Wv=Wv,
            wt_q=wt_q,
            wt_scales=wt_scales,
            wv_q=wv_q,
            wv_scales=wv_scales,
            target=target,
            grid_h=GRID_H,
            grid_w=GRID_W,
        )

    def test_evaluate_record_metric_contract(self):
        record = self._evaluate()
        for arm in audit.QUANT_ARMS:
            for role in audit.ROLES:
                for budget in audit.BUDGETS:
                    entry = record["arms"][arm]["by_role"][role][budget]
                    for key in ("mass", "ndcg", "jaccard", "exact_match",
                                "mass_delta", "ndcg_delta", "selected_count"):
                        self.assertIn(key, entry)
                    self.assertGreaterEqual(entry["jaccard"], 0.0)
                    self.assertLessEqual(entry["jaccard"], 1.0)
                    self.assertIn(entry["exact_match"], (0.0, 1.0))
        self.assertLessEqual(record["projection_int_max_abs"], audit.INT32_MAX)

    def test_float_reference_matches_shared_scorer(self):
        Wt, Wv = _projections(17)
        group_mean, prompt, _, _ = _record_data(18)
        _, _, combined = audit.float_reference_scores(group_mean, prompt, Wt, Wv)
        semantic = score_candidate("prompt_mean_vector", group_mean, prompt, Wt, Wv)
        l2 = l2_baseline_scores(group_mean)
        expected = combine_scores(semantic, l2, audit.LAMBDA_SEMANTIC)
        np.testing.assert_allclose(combined, expected, atol=1e-12)

    def test_aggregate_medians_and_image_summary(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        self.assertEqual(summary["n_images"], 2)
        self.assertEqual(summary["n_records"], 4)
        # Per-record median across the three seeds.
        self.assertIn("SPLIT_WV_Q8_A8_Q8QUERY|correct|48|jaccard", summary["record_medians"])
        # Two images -> each overall CI has n=2.
        self.assertEqual(summary["overall"][audit.PRIMARY_ARM]["48"]["jaccard"]["n"], 2)
        self.assertEqual(
            summary["per_task"][audit.PRIMARY_ARM]["pointing"]["48"]["mass"]["n"], 2
        )
        gap = summary["prompt_utility_gap_delta"][audit.PRIMARY_ARM]["48"]["mass"]
        self.assertIn("pointing", gap["per_task"])
        self.assertIn("overall", gap)
        # Both prompt roles carry an explicit quant-vs-float summary.
        for role in audit.ROLES:
            self.assertEqual(
                summary["overall_by_role"][role][audit.PRIMARY_ARM]["48"]["jaccard"]["n"], 2
            )

    def test_decision_pass_and_fail(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        passing = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True
        )
        self.assertEqual(passing["decision"], "INT8_NUMERICS_PASS")
        self.assertTrue(all(passing["checks"].values()))

        failing = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=False
        )
        self.assertEqual(failing["decision"], "INT8_NUMERICS_NOT_ESTABLISHED")
        self.assertFalse(failing["checks"]["int32_worst_case_bounds_below_2147483647"])

    def test_decision_fails_on_seed_jaccard(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        for seed in audit.SEEDS:
            summary["seed_overall"][audit.PRIMARY_ARM][seed]["48"] = 0.5
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True
        )
        self.assertFalse(decision["checks"]["each_seed_k48_mean_jaccard_at_least_0.95"])


class BundleTests(unittest.TestCase):
    def test_build_seed_bundle_no_wt_and_fixture(self):
        Wt, Wv = _projections(19)
        group_mean, prompt, wrong, target = _record_data(20)
        fixtures = [
            {
                "task": "grounding",
                "sample_id": "g1",
                "group_count": GROUP_COUNT,
                "grid_h": GRID_H,
                "grid_w": GRID_W,
                "group_mean": group_mean,
                "prompt_embeddings": prompt,
                "wrong_prompt_embeddings": wrong,
                "spatial_target": target,
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed"
            metadata = audit.build_seed_bundle(
                20261003,
                Wt=Wt,
                Wv=Wv,
                fixtures=fixtures,
                checkpoint={"path": "x/best.npz", "sha256": "abc",
                            "canonical_weights_sha256": "def"},
                out_dir=out,
            )
            self.assertFalse(metadata["wt_exported"])
            self.assertTrue((out / "Wv_int8_rowmajor.bin").is_file())
            self.assertTrue((out / "Wv_row_scales_f32le.bin").is_file())
            self.assertTrue((out / "metadata.json").is_file())
            self.assertEqual(metadata["wv"]["shape"], [128, 1024])
            self.assertEqual(
                (out / "Wv_int8_rowmajor.bin").stat().st_size, 128 * 1024
            )
            self.assertEqual(
                (out / "Wv_row_scales_f32le.bin").stat().st_size, 128 * 4
            )
            with np.load(out / "fixture_grounding.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["group_mean_q"].dtype, np.int8)
                self.assertEqual(archive["query_q_correct"].shape, (128,))
                self.assertEqual(archive["score_q_correct"].dtype, np.uint16)
                for budget in audit.BUDGETS:
                    self.assertIn(f"selected_correct_k{budget}", archive.files)


class PinsAndRenderTests(unittest.TestCase):
    def test_verify_pins_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data.bin"
            target.write_bytes(b"hello")
            prereg = {"frozen_inputs": {"data.bin": "0" * 64}, "checkpoints": []}
            report = audit.verify_pins(prereg, root=root)
            self.assertFalse(report["pins_unchanged"])
            self.assertEqual(report["mismatches"], ["data.bin"])

    def test_verify_pins_accepts_correct_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data.bin"
            target.write_bytes(b"hello")
            digest = audit.sha256_file(target)
            prereg = {"frozen_inputs": {"data.bin": digest}, "checkpoints": []}
            report = audit.verify_pins(prereg, root=root)
            self.assertTrue(report["pins_unchanged"])

    def test_markdown_and_harness_render(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True
        )
        bundle = {
            "decision": decision,
            "metrics": {
                "overall": summary["overall"],
                "prompt_utility_gap_delta": summary["prompt_utility_gap_delta"],
            },
            "aggregation": {"n_images": summary["n_images"]},
            "int32": {
                "analytical": audit.analytical_int32_bounds(),
                "observed_max_abs": {"gm_wv_max_abs": 1, "prompt_wt_max_abs": 2},
                "no_overflow": True,
            },
            "isolation": {
                "opened_npz_count": 76,
                "opened_only_val_npz": True,
                "test_npz_opened": [],
                "train_npz_opened": [],
                "val_npz_aggregate_matches_prereg": True,
            },
            "pins_unchanged": True,
            "pins_after": {"checked": 15},
            "limitations": ["test limitation"],
        }
        markdown = audit.render_markdown(bundle)
        self.assertIn("INT8_NUMERICS_PASS", markdown)
        self.assertIn("software golden", markdown)
        harness = audit.harness_payload(bundle, {"json": "x"})
        self.assertEqual(harness["decision"], "INT8_NUMERICS_PASS")
        json.dumps(harness)

    def test_real_config_identity_and_pins(self):
        # Integration check against the frozen repository inputs (read-only).
        prereg = audit.load_prereg(audit.ROOT / audit.PREREG_CONFIG_PATH)
        identity = audit.verify_prereguration_identity(prereg, root=audit.ROOT)
        self.assertTrue(identity["consistent"])
        pins = audit.verify_pins(prereg, root=audit.ROOT)
        self.assertTrue(pins["pins_unchanged"])
        self.assertGreaterEqual(pins["checked"], 15)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
