#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.selector_wv_scale_regularization`.

Covers the authority-prompt-required surface:

* the new per-tensor Wv quantiser (scale, bounds, zero tensor),
* the shared per-row reference / per-tensor candidate / float record chain,
* candidate-vs-reference and candidate-vs-float metrics, replacement count and
  score Spearman,
* cross-seed median aggregation, image-level mean, 2000-resample bootstrap and
  the K48 prompt-utility gap,
* the frozen ``PER_TENSOR_WV_ACCEPTABLE_FOR_HLS`` decision rule,
* the compact per-seed bundle (single float32 scale, no ``Wt`` export) and its
  byte-exact fixture round-trip,
* pinned-input verification, Markdown and harness rendering.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import selector_int8_numerical_audit as base
from evaluation import selector_wv_scale_regularization as audit

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


def _evaluate(Wt=None, Wv=None, seed=16):
    if Wt is None or Wv is None:
        Wt, Wv = _projections(15)
    group_mean, prompt, wrong, target = _record_data(seed)
    wv_q_ref, wv_scales_ref = base.quantize_weight_rows(Wv)
    wv_q_cand, wv_scale_cand = audit.quantize_weight_tensor(Wv)
    return audit.evaluate_record(
        group_mean=group_mean,
        prompt_correct=prompt,
        prompt_wrong=wrong,
        Wt=Wt,
        Wv=Wv,
        wv_q_ref=wv_q_ref,
        wv_scales_ref=wv_scales_ref,
        wv_q_cand=wv_q_cand,
        wv_scale_cand=wv_scale_cand,
        target=target,
        grid_h=GRID_H,
        grid_w=GRID_W,
    )


def _metric_value(name: str, seed_offset: int = 0) -> float:
    values = {
        "cand_vs_ref_jaccard": 0.99 + 0.001 * seed_offset,
        "cand_vs_ref_exact_match": 1.0,
        "cand_vs_ref_replacement_count": 0.0,
        "cand_vs_ref_mass_delta": 0.0,
        "cand_vs_ref_ndcg_delta": 0.0,
        "cand_vs_float_jaccard": 0.98,
        "cand_vs_float_mass_delta": 0.0,
        "cand_vs_float_ndcg_delta": 0.0,
    }
    if name in values:
        return values[name]
    if name.endswith("_mass") or name.endswith("_ndcg"):
        return 0.6
    return 48.0


def _raw_by_seed(seeds=(20261003, 20261004, 20261005), jaccard_by_seed=None):
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
        for sid in sorted(sample_meta):
            by_role = {}
            for role in audit.ROLES:
                by_role[role] = {}
                for budget in audit.BUDGETS:
                    metrics = {
                        name: _metric_value(name, offset) for name in audit.RECORD_METRICS
                    }
                    if jaccard_by_seed is not None:
                        metrics["cand_vs_ref_jaccard"] = float(jaccard_by_seed[offset])
                    by_role[role][int(budget)] = metrics
            raw[seed][sid] = {
                "by_role": by_role,
                "spearman": {
                    role: {
                        "candidate_vs_per_row": 0.999,
                        "per_row_vs_float": 0.998,
                        "candidate_vs_float": 0.997,
                    }
                    for role in audit.ROLES
                },
            }
    return raw, sample_meta


class WeightTensorQuantisationTests(unittest.TestCase):
    def test_scale_is_global_amax_over_127(self):
        rng = np.random.default_rng(3)
        matrix = rng.normal(size=(8, 16)) * 0.5
        quantized, scale = audit.quantize_weight_tensor(matrix)
        self.assertAlmostEqual(scale, float(np.abs(matrix).max()) / 127.0)
        self.assertGreaterEqual(int(quantized.min()), audit.WEIGHT_MIN)
        self.assertLessEqual(int(quantized.max()), audit.WEIGHT_MAX)
        # The global amax maps to +127, not just a per-row amax.
        self.assertEqual(int(np.max(np.abs(quantized))), 127)
        reconstructed = quantized.astype(np.float64) * scale
        self.assertTrue(np.all(np.abs(reconstructed - matrix) <= scale))

    def test_zero_tensor_keeps_scale_one(self):
        quantized, scale = audit.quantize_weight_tensor(np.zeros((4, 5)))
        self.assertEqual(scale, 1.0)
        np.testing.assert_array_equal(quantized, np.zeros((4, 5)))

    def test_rejects_non_2d_and_nonfinite(self):
        with self.assertRaises(audit.WvScaleAuditError):
            audit.quantize_weight_tensor(np.zeros(4))
        with self.assertRaises(audit.WvScaleAuditError):
            audit.quantize_weight_tensor(np.array([[0.0, np.nan]]))

    def test_per_tensor_scale_vector(self):
        vector = audit.per_tensor_scale_vector(0.25, 128)
        np.testing.assert_array_equal(vector, np.full(128, 0.25))
        self.assertEqual(vector.shape, (128,))

    def test_round_half_away_shared_with_reference(self):
        values = np.array([0.5, 1.5, -0.5, -1.5])
        np.testing.assert_array_equal(
            audit.base.round_half_away_from_zero(values), [1, 2, -1, -2]
        )


class RecordEvaluationTests(unittest.TestCase):
    def test_evaluate_record_metric_contract(self):
        record = _evaluate()
        for role in audit.ROLES:
            for budget in audit.BUDGETS:
                entry = record["by_role"][role][budget]
                for key in (
                    "cand_mass",
                    "cand_ndcg",
                    "cand_selected_count",
                    "ref_mass",
                    "ref_ndcg",
                    "cand_vs_ref_jaccard",
                    "cand_vs_ref_exact_match",
                    "cand_vs_ref_replacement_count",
                    "cand_vs_ref_mass_delta",
                    "cand_vs_ref_ndcg_delta",
                    "cand_vs_float_jaccard",
                    "cand_vs_float_mass_delta",
                    "cand_vs_float_ndcg_delta",
                ):
                    self.assertIn(key, entry)
                self.assertGreaterEqual(entry["cand_vs_ref_jaccard"], 0.0)
                self.assertLessEqual(entry["cand_vs_ref_jaccard"], 1.0)
                self.assertIn(entry["cand_vs_ref_exact_match"], (0.0, 1.0))
                self.assertIsInstance(entry["cand_vs_ref_replacement_count"], int)
                self.assertGreaterEqual(entry["cand_vs_ref_replacement_count"], 0)
                self.assertEqual(
                    entry["cand_vs_ref_mass_delta"], entry["cand_mass"] - entry["ref_mass"]
                )
                self.assertEqual(
                    entry["cand_vs_float_mass_delta"],
                    entry["cand_mass"] - entry["float_mass"],
                )
        self.assertLessEqual(record["projection_int_max_abs_per_row"], audit.INT32_MAX)
        self.assertLessEqual(
            record["projection_int_max_abs_per_tensor"], audit.INT32_MAX
        )
        # The candidate scale is the single global value.
        self.assertEqual(record["cand_wv_scale"], audit.quantize_weight_tensor(
            _projections(15)[1]
        )[1])

    def test_replacement_count_is_candidate_only_members(self):
        record = _evaluate()
        for role in audit.ROLES:
            for budget in audit.BUDGETS:
                entry = record["by_role"][role][budget]
                expected = len(set(entry["cand_selected"]) - set(entry["ref_selected"]))
                self.assertEqual(entry["cand_vs_ref_replacement_count"], expected)

    def test_spearman_comparisons_present_and_bounded(self):
        record = _evaluate()
        for role in audit.ROLES:
            for comparison in audit.SPEARMAN_COMPARISONS:
                value = record["spearman"][role][comparison]
                self.assertGreaterEqual(value, -1.0000001)
                self.assertLessEqual(value, 1.0000001)

    def test_per_tensor_and_per_row_share_activation_scale(self):
        Wt, Wv = _projections(21)
        group_mean, prompt, wrong, target = _record_data(22)
        ref_record = audit.evaluate_record(
            group_mean=group_mean, prompt_correct=prompt, prompt_wrong=wrong,
            Wt=Wt, Wv=Wv,
            wv_q_ref=base.quantize_weight_rows(Wv)[0],
            wv_scales_ref=base.quantize_weight_rows(Wv)[1],
            wv_q_cand=audit.quantize_weight_tensor(Wv)[0],
            wv_scale_cand=audit.quantize_weight_tensor(Wv)[1],
            target=target, grid_h=GRID_H, grid_w=GRID_W,
        )
        # Activation scale is computed once for the whole image and identical.
        gm_q, gm_scale = base.quantize_activation(np.asarray(group_mean))
        self.assertEqual(ref_record["group_mean_scale"], float(gm_scale))

    def test_analytical_int32_bound_dominates_per_tensor(self):
        record = _evaluate()
        self.assertLessEqual(
            record["projection_int_max_abs_per_tensor"],
            base.analytical_int32_bound(1024)["bound"],
        )


class AggregationTests(unittest.TestCase):
    def test_medians_and_image_summary(self):
        raw, sample_meta = _raw_by_seed(jaccard_by_seed=(0.90, 0.99, 1.00))
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        self.assertEqual(summary["n_images"], 2)
        self.assertEqual(summary["n_records"], 4)
        # Per-record median across the three seeds.
        self.assertIn("correct|48|cand_vs_ref_jaccard", summary["record_medians"])
        median = summary["record_medians"]["correct|48|cand_vs_ref_jaccard"]["g1"]
        self.assertAlmostEqual(median, 0.99)
        # Two images -> each overall CI has n=2.
        self.assertEqual(
            summary["overall"]["correct"]["48"]["cand_vs_ref_jaccard"]["n"], 2
        )
        self.assertEqual(
            summary["per_task"]["correct"]["pointing"]["48"]["cand_vs_float_mass_delta"]["n"],
            2,
        )
        self.assertIn(
            "candidate_vs_per_row", summary["score_rank_spearman"]
        )
        self.assertEqual(
            summary["score_rank_spearman"]["candidate_vs_per_row"]["correct"]["overall"]["n"],
            2,
        )
        gap = summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]
        self.assertIn("pointing", gap["per_task"])
        self.assertIn("overall", gap)
        self.assertEqual(
            gap["per_task"]["pointing"]["mean"], 0.0
        )
        self.assertEqual(
            set(summary["seed_k48_candidate_vs_per_row_jaccard"]),
            {"20261003", "20261004", "20261005"},
        )

    def test_image_mean_averages_two_task_records(self):
        raw, sample_meta = _raw_by_seed(jaccard_by_seed=(0.99, 0.99, 0.99))
        # Perturb only the pointing record of image 1 so its two-task mean moves.
        for seed in (20261003, 20261004, 20261005):
            raw[seed]["p1"]["by_role"]["correct"][48]["cand_vs_ref_jaccard"] = 0.0
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        # img1 = mean(grounding 0.99, pointing 0.0); img2 = mean(0.99, 0.99);
        # the overall image-level mean is then the mean of the two image means.
        image1 = (0.99 + 0.0) / 2.0
        image2 = (0.99 + 0.99) / 2.0
        expected = (image1 + image2) / 2.0
        entry = summary["overall"]["correct"]["48"]["cand_vs_ref_jaccard"]
        self.assertAlmostEqual(entry["mean"], expected)


class DecisionTests(unittest.TestCase):
    def _decision(self, **overrides):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        params = dict(pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True)
        params.update(overrides)
        return summary, audit.evaluate_decision(summary, **params)

    def test_pass_rule(self):
        _, decision = self._decision()
        self.assertEqual(decision["decision"], "PER_TENSOR_WV_ACCEPTABLE_FOR_HLS")
        self.assertTrue(all(decision["checks"].values()))

    def test_fails_on_isolation_int32_fixture(self):
        _, decision = self._decision(int32_ok=False)
        self.assertEqual(decision["decision"], "PER_TENSOR_WV_NOT_ESTABLISHED")
        self.assertFalse(decision["checks"]["pins_isolation_int32_fixture_pass"])
        _, decision = self._decision(fixture_ok=False)
        self.assertFalse(decision["checks"]["pins_isolation_int32_fixture_pass"])

    def test_fails_on_jaccard(self):
        summary, _ = self._decision()
        for budget in audit.BUDGETS:
            summary["overall"]["correct"][str(budget)]["cand_vs_ref_jaccard"]["mean"] = 0.97
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_row_mean_jaccard_each_budget_at_least_0.98"
            ]
        )

    def test_fails_on_replacement_count(self):
        summary, _ = self._decision()
        for budget in audit.BUDGETS:
            summary["overall"]["correct"][str(budget)][
                "cand_vs_ref_replacement_count"
            ]["mean"] = 1.5
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_row_mean_replacement_count_each_budget_at_most_1.0"
            ]
        )

    def test_fails_on_spearman(self):
        summary, _ = self._decision()
        summary["score_rank_spearman"]["candidate_vs_per_row"]["correct"]["overall"][
            "mean"
        ] = 0.99
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_row_mean_score_spearman_at_least_0.995"
            ]
        )

    def test_fails_on_float_deltas(self):
        summary, _ = self._decision()
        summary["overall"]["correct"]["48"]["cand_vs_float_mass_delta"]["mean"] = -0.02
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_float_mass_delta_each_budget_at_least_-0.01"
            ]
        )
        summary, _ = self._decision()
        summary["overall"]["correct"]["96"]["cand_vs_float_ndcg_delta"]["mean"] = -0.02
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_float_ndcg_delta_each_budget_at_least_-0.01"
            ]
        )

    def test_fails_on_pointing_gap(self):
        summary, _ = self._decision()
        summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]["per_task"][
            "pointing"
        ]["mean"] = -0.02
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "pointing_k48_prompt_utility_gap_delta_candidate_vs_float_at_least_-0.01"
            ]
        )

    def test_fails_on_seed_k48(self):
        summary, _ = self._decision()
        summary["seed_k48_candidate_vs_per_row_jaccard"]["20261004"] = 0.5
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "each_seed_k48_candidate_vs_per_row_mean_jaccard_at_least_0.97"
            ]
        )


class BundleTests(unittest.TestCase):
    def _fixtures(self, Wt, Wv):
        group_mean, prompt, wrong, target = _record_data(23)
        return [
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

    def test_build_seed_bundle_no_wt_and_roundtrip(self):
        Wt, Wv = _projections(19)
        fixtures = self._fixtures(Wt, Wv)
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
            self.assertTrue(metadata["roundtrip"]["ok"])
            self.assertTrue((out / "Wv_int8_rowmajor.bin").is_file())
            self.assertTrue((out / "Wv_per_tensor_scale_f32le.bin").is_file())
            self.assertTrue((out / "metadata.json").is_file())
            self.assertEqual(metadata["wv"]["shape"], [128, 1024])
            self.assertEqual((out / "Wv_int8_rowmajor.bin").stat().st_size, 128 * 1024)
            # A single little-endian float32 per tensor, not 128 row scales.
            self.assertEqual((out / "Wv_per_tensor_scale_f32le.bin").stat().st_size, 4)
            scale_bytes = (out / "Wv_per_tensor_scale_f32le.bin").read_bytes()
            loaded = float(np.frombuffer(scale_bytes, dtype="<f4")[0])
            self.assertAlmostEqual(loaded, metadata["wv"]["scale_value_float32"])
            with np.load(out / "fixture_grounding.npz", allow_pickle=False) as archive:
                self.assertEqual(archive["group_mean_q"].dtype, np.int8)
                self.assertEqual(archive["query_q_correct"].shape, (128,))
                self.assertEqual(archive["score_q_correct"].dtype, np.uint16)
                for budget in audit.BUDGETS:
                    self.assertIn(f"selected_correct_k{budget}", archive.files)
            self.assertNotIn("Wt", metadata["files"])
            # No Wt-like payload exists in the exported files.
            for path in out.iterdir():
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        for key in archive.files:
                            self.assertFalse(key.lower().startswith("wt"))

    def test_roundtrip_detects_corruption(self):
        Wt, Wv = _projections(24)
        fixtures = self._fixtures(Wt, Wv)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed"
            metadata = audit.build_seed_bundle(
                20261003, Wt=Wt, Wv=Wv, fixtures=fixtures,
                checkpoint={"path": "x", "sha256": "a"}, out_dir=out,
            )
            self.assertTrue(metadata["roundtrip"]["ok"])
            wv_path = out / "Wv_int8_rowmajor.bin"
            data = bytearray(wv_path.read_bytes())
            data[0] = data[0] ^ 0x7F
            wv_path.write_bytes(bytes(data))
            corrupted = audit.validate_fixture_roundtrip(
                out, Wv=Wv, fixtures=fixtures
            )
            self.assertFalse(corrupted["wv_q_matches_quantizer"])
            self.assertFalse(corrupted["ok"])

    def test_roundtrip_detects_scale_corruption(self):
        Wt, Wv = _projections(25)
        fixtures = self._fixtures(Wt, Wv)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "seed"
            audit.build_seed_bundle(
                20261003, Wt=Wt, Wv=Wv, fixtures=fixtures,
                checkpoint={"path": "x", "sha256": "a"}, out_dir=out,
            )
            scale_path = out / "Wv_per_tensor_scale_f32le.bin"
            scale_path.write_bytes(np.asarray([123.0], dtype="<f4").tobytes())
            corrupted = audit.validate_fixture_roundtrip(out, Wv=Wv, fixtures=fixtures)
            self.assertFalse(corrupted["wv_scale_matches_float32"])
            self.assertFalse(corrupted["ok"])


class PinsAndRenderTests(unittest.TestCase):
    def test_verify_pins_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data.bin"
            target.write_bytes(b"hello")
            prereg = {"frozen_inputs": {"data.bin": "0" * 64}, "checkpoints": []}
            report = base.verify_pins(prereg, root=root)
            self.assertFalse(report["pins_unchanged"])
            self.assertEqual(report["mismatches"], ["data.bin"])

    def _bundle(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, int32_ok=True, fixture_ok=True
        )
        return {
            "decision": decision,
            "metrics": {
                "overall": summary["overall"],
                "per_task": summary["per_task"],
                "score_rank_spearman": summary["score_rank_spearman"],
                "prompt_utility_gap_delta": summary["prompt_utility_gap_delta"],
                "seed_k48_candidate_vs_per_row_jaccard": summary[
                    "seed_k48_candidate_vs_per_row_jaccard"
                ],
            },
            "aggregation": {"n_images": summary["n_images"]},
            "int32": {
                "analytical": base.analytical_int32_bounds(),
                "observed_max_abs": {
                    "gm_wv_per_row_max_abs": 1,
                    "gm_wv_per_tensor_max_abs": 2,
                },
                "no_overflow": True,
            },
            "fixtures_validated": True,
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

    def test_markdown_and_harness_render(self):
        bundle = self._bundle()
        markdown = audit.render_markdown(bundle)
        self.assertIn("PER_TENSOR_WV_ACCEPTABLE_FOR_HLS", markdown)
        self.assertIn("software golden", markdown)
        harness = audit.harness_payload(bundle, {"json": "x"})
        self.assertEqual(harness["decision"], "PER_TENSOR_WV_ACCEPTABLE_FOR_HLS")
        json.dumps(harness)

    def test_write_reports_flat_and_directory_mirror(self):
        bundle = self._bundle()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = audit._write_reports(bundle, [], root=root)
            flat = root / audit.REPORTS_DIR / f"{audit.REPORT_STEM}.json"
            mirror = root / audit.REPORT_DIR / f"{audit.REPORT_STEM}.json"
            self.assertTrue(flat.is_file())
            self.assertTrue(mirror.is_file())
            self.assertEqual(flat.read_bytes(), mirror.read_bytes())
            self.assertEqual(files["json"], str(flat))
            self.assertEqual(files["json_dir_mirror"], str(mirror))
            self.assertTrue(
                (root / audit.REPORTS_DIR / f"{audit.REPORT_STEM}.md").is_file()
            )
            self.assertTrue(
                (root / audit.REPORTS_DIR / f"{audit.REPORT_STEM}-records.csv").is_file()
            )

    def test_real_config_identity_and_pins(self):
        # Integration check against the frozen repository inputs (read-only).
        prereg = base.load_prereg(audit.ROOT / audit.PREREG_CONFIG_PATH)
        identity = base.verify_prereguration_identity(
            prereg, root=audit.ROOT, config_path=audit.PREREG_CONFIG_PATH
        )
        self.assertTrue(identity["consistent"])
        pins = base.verify_pins(prereg, root=audit.ROOT)
        self.assertTrue(pins["pins_unchanged"])
        self.assertGreaterEqual(pins["checked"], 14)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
