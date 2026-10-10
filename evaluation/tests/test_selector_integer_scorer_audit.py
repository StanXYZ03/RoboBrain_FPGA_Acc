#!/usr/bin/env python3
"""Focused tests for :mod:`evaluation.selector_integer_scorer_audit`.

Covers the frozen surface:

* exact integer ``floor_sqrt`` and signed-rational round-half-away primitives,
* the exact min-max Q15 map,
* the candidate ``INTEGER_SCORER_Q16`` chain (widths, zero norm, selections),
* the per-record candidate / per-tensor / float comparison metrics,
* cross-seed median aggregation and image-level bootstrap summaries,
* the frozen ``INTEGER_SCORER_NUMERICS_PASS`` decision rule,
* the machine-readable width/operation contract,
* pinned-input verification (frozen inputs and accepted Wv bundles),
* the compact per-seed fixture bundle, its reference-only Wv handling and its
  byte-exact round-trip, and the Markdown/harness rendering.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation import selector_wv_scale_regularization as wv_audit
from evaluation import selector_integer_scorer_audit as audit

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
    wv_q_ref, wv_scale_ref = wv_audit.quantize_weight_tensor(Wv)
    return audit.evaluate_record(
        group_mean=group_mean,
        prompt_correct=prompt,
        prompt_wrong=wrong,
        Wt=Wt,
        Wv=Wv,
        wv_q_ref=wv_q_ref,
        wv_scale_ref=wv_scale_ref,
        target=target,
        grid_h=GRID_H,
        grid_w=GRID_W,
    )


def _fixture_sample(task: str, seed: int = 5):
    rng = np.random.default_rng(seed)
    return {
        "sample_id": f"{task}-sample",
        "task": task,
        "group_count": GROUP_COUNT,
        "grid_h": GRID_H,
        "grid_w": GRID_W,
        "group_mean": rng.normal(size=(GROUP_COUNT, 1024)).astype(np.float64),
        "prompt_embeddings": rng.normal(size=(TOKENS, 2560)).astype(np.float64),
        "wrong_prompt_embeddings": rng.normal(size=(TOKENS + 1, 2560)).astype(np.float64),
    }


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
                        "candidate_vs_per_tensor": 0.999,
                        "per_tensor_vs_float": 0.998,
                        "candidate_vs_float": 0.997,
                    }
                    for role in audit.ROLES
                },
            }
    return raw, sample_meta


class IntegerPrimitiveTests(unittest.TestCase):
    def test_floor_sqrt_is_exact(self):
        for value in (0, 1, 2, 3, 4, 8, 9, 10, 99, 100, 101, 10**12, 2**61):
            root = audit.floor_sqrt_int64(value)
            self.assertLessEqual(root * root, value)
            self.assertGreater((root + 1) * (root + 1), value)

    def test_floor_sqrt_rejects_negative_and_overflow(self):
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.floor_sqrt_int64(-1)
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.floor_sqrt_int64(audit.INT64_MAX + 1)

    def test_round_half_away_div_examples(self):
        self.assertEqual(audit.round_half_away_div(0, 3), 0)
        self.assertEqual(audit.round_half_away_div(1, 2), 1)
        self.assertEqual(audit.round_half_away_div(3, 2), 2)
        self.assertEqual(audit.round_half_away_div(-1, 2), -1)
        self.assertEqual(audit.round_half_away_div(-3, 2), -2)
        self.assertEqual(audit.round_half_away_div(5, 2), 3)
        self.assertEqual(audit.round_half_away_div(1 << 16, 1), 1 << 16)

    def test_round_half_away_div_rejects_bad_denominator(self):
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.round_half_away_div(1, 0)
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.round_half_away_div(1, -2)

    def test_minmax_q15_endpoints_and_all_equal(self):
        np.testing.assert_array_equal(
            audit.minmax_normalize_q15(np.array([5, 5, 5])), np.zeros(3, dtype=np.int64)
        )
        np.testing.assert_array_equal(
            audit.minmax_normalize_q15(np.array([3, 4])), np.array([0, 32767])
        )
        # half-away: midpoint (1 of span 2) rounds up to 16384.
        np.testing.assert_array_equal(
            audit.minmax_normalize_q15(np.array([0, 1, 2])),
            np.array([0, 16384, 32767]),
        )


class IntegerChainTests(unittest.TestCase):
    def test_projection_is_exact_int32_and_bounded(self):
        _Wt, Wv = _projections(31)
        wv_q, _ = wv_audit.quantize_weight_tensor(Wv)
        mean_q, _ = audit.base.quantize_activation(_record_data(32)[0])
        query_q = np.asarray(audit.base.primary_quantized_query(
            _record_data(32)[1], _projections(31)[0]
        )["query_q"], dtype=np.int64)
        chain = audit.integer_chain(mean_q, query_q, wv_q, GRID_H, GRID_W)
        expected = mean_q.astype(np.int64) @ wv_q.T.astype(np.int64)
        np.testing.assert_array_equal(chain["projection_int"], expected)
        self.assertLessEqual(
            chain["bounds"]["projection_int_absmax"], audit.PROJECTION_BOUND
        )
        self.assertLessEqual(
            chain["bounds"]["projection_int_absmax"], audit.INT32_MAX
        )
        self.assertTrue(
            np.all(chain["score_q"] >= 0) and np.all(chain["score_q"] <= audit.SCORE_Q_MAX)
        )
        for budget in audit.BUDGETS:
            self.assertTrue(
                np.array_equal(
                    np.asarray(chain["selections"][budget], dtype=np.int64),
                    np.asarray(
                        audit.base.selected_indices(
                            chain["score_q"].astype(np.float64), GRID_H, GRID_W, budget
                        ),
                        dtype=np.int64,
                    ),
                )
            )

    def test_zero_norm_maps_to_zero_semantic(self):
        wv_q = np.zeros((128, 1024), dtype=np.int64)
        mean_q = np.zeros((GROUP_COUNT, 1024), dtype=np.int64)
        query_q = np.ones(128, dtype=np.int64)
        chain = audit.integer_chain(mean_q, query_q, wv_q, GRID_H, GRID_W)
        np.testing.assert_array_equal(chain["semantic_q16"], np.zeros(GROUP_COUNT))
        np.testing.assert_array_equal(chain["l2_int"], np.zeros(GROUP_COUNT))
        # All-equal semantic/L2 terms normalise to zero and combine to zero.
        np.testing.assert_array_equal(chain["score_q"], np.zeros(GROUP_COUNT))

    def test_chain_rejects_shape_and_magnitude_violations(self):
        wv_q, _ = wv_audit.quantize_weight_tensor(_projections(4)[1])
        mean_q, _ = audit.base.quantize_activation(_record_data(4)[0])
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.integer_chain(mean_q, np.ones(3, dtype=np.int64), wv_q, GRID_H, GRID_W)
        bad = mean_q.copy()
        bad[0, 0] = 129
        with self.assertRaises(audit.IntegerScorerAuditError):
            audit.integer_chain(bad, np.ones(128, dtype=np.int64), wv_q, GRID_H, GRID_W)


class RecordEvaluationTests(unittest.TestCase):
    def test_metric_contract_and_deltas(self):
        record = _evaluate()
        for role in audit.ROLES:
            for budget in audit.BUDGETS:
                entry = record["by_role"][role][budget]
                for key in audit.RECORD_METRICS:
                    self.assertIn(key, entry)
                self.assertGreaterEqual(entry["cand_vs_ref_jaccard"], 0.0)
                self.assertLessEqual(entry["cand_vs_ref_jaccard"], 1.0)
                self.assertIn(entry["cand_vs_ref_exact_match"], (0.0, 1.0))
                self.assertIsInstance(entry["cand_vs_ref_replacement_count"], int)
                self.assertEqual(
                    entry["cand_vs_ref_mass_delta"],
                    entry["cand_mass"] - entry["ref_mass"],
                )
                self.assertEqual(
                    entry["cand_vs_float_mass_delta"],
                    entry["cand_mass"] - entry["float_mass"],
                )
                self.assertEqual(
                    entry["cand_vs_ref_replacement_count"],
                    len(set(entry["cand_selected"]) - set(entry["ref_selected"])),
                )

    def test_spearman_comparisons_bounded(self):
        record = _evaluate()
        for role in audit.ROLES:
            for comparison in audit.SPEARMAN_COMPARISONS:
                value = record["spearman"][role][comparison]
                self.assertGreaterEqual(value, -1.0000001)
                self.assertLessEqual(value, 1.0000001)

    def test_candidate_is_close_to_reference(self):
        record = _evaluate()
        for role in audit.ROLES:
            for budget in audit.BUDGETS:
                self.assertGreaterEqual(
                    record["by_role"][role][budget]["cand_vs_ref_jaccard"], 0.8
                )
            self.assertGreaterEqual(record["spearman"][role]["candidate_vs_per_tensor"], 0.95)


class AggregationTests(unittest.TestCase):
    def test_medians_and_image_summary(self):
        raw, sample_meta = _raw_by_seed(jaccard_by_seed=(0.90, 0.99, 1.00))
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        self.assertEqual(summary["n_images"], 2)
        self.assertEqual(summary["n_records"], 4)
        self.assertAlmostEqual(
            summary["record_medians"]["correct|48|cand_vs_ref_jaccard"]["g1"], 0.99
        )
        self.assertEqual(
            summary["overall"]["correct"]["48"]["cand_vs_ref_jaccard"]["n"], 2
        )
        self.assertEqual(
            summary["score_rank_spearman"]["candidate_vs_per_tensor"]["correct"][
                "overall"
            ]["n"],
            2,
        )
        self.assertEqual(
            set(summary["seed_k48_candidate_vs_per_tensor_jaccard"]),
            {"20261003", "20261004", "20261005"},
        )
        gap = summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]
        self.assertIn("pointing", gap["per_task"])

    def test_image_mean_averages_two_task_records(self):
        raw, sample_meta = _raw_by_seed(jaccard_by_seed=(0.99, 0.99, 0.99))
        for seed in (20261003, 20261004, 20261005):
            raw[seed]["p1"]["by_role"]["correct"][48]["cand_vs_ref_jaccard"] = 0.0
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        image1 = (0.99 + 0.0) / 2.0
        image2 = (0.99 + 0.99) / 2.0
        self.assertAlmostEqual(
            summary["overall"]["correct"]["48"]["cand_vs_ref_jaccard"]["mean"],
            (image1 + image2) / 2.0,
        )


class DecisionTests(unittest.TestCase):
    def _decision(self, **overrides):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        params = dict(
            pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        params.update(overrides)
        return summary, audit.evaluate_decision(summary, **params)

    def test_pass_rule(self):
        _, decision = self._decision()
        self.assertEqual(decision["decision"], "INTEGER_SCORER_NUMERICS_PASS")
        self.assertTrue(all(decision["checks"].values()))

    def test_fails_on_pin_fixture_or_width(self):
        for key in ("pins_ok", "isolation_ok", "fixture_ok", "width_ok"):
            _, decision = self._decision(**{key: False})
            self.assertEqual(
                decision["decision"], "INTEGER_SCORER_NUMERICS_NOT_ESTABLISHED"
            )
            self.assertFalse(decision["checks"]["pins_isolation_fixture_width_pass"])

    def test_fails_on_jaccard_and_replacement(self):
        summary, _ = self._decision()
        for budget in audit.BUDGETS:
            summary["overall"]["correct"][str(budget)]["cand_vs_ref_jaccard"][
                "mean"
            ] = 0.97
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_tensor_mean_jaccard_each_budget_at_least_0.98"
            ]
        )
        summary, _ = self._decision()
        for budget in audit.BUDGETS:
            summary["overall"]["correct"][str(budget)][
                "cand_vs_ref_replacement_count"
            ]["mean"] = 2.0
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_tensor_mean_replacement_count_each_budget_at_most_1.0"
            ]
        )

    def test_fails_on_spearman_and_float_deltas(self):
        summary, _ = self._decision()
        summary["score_rank_spearman"]["candidate_vs_per_tensor"]["correct"][
            "overall"
        ]["mean"] = 0.99
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "candidate_vs_per_tensor_mean_score_spearman_at_least_0.995"
            ]
        )
        summary, _ = self._decision()
        summary["overall"]["correct"]["48"]["cand_vs_float_mass_delta"]["mean"] = -0.02
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"]["candidate_vs_float_mass_delta_each_budget_at_least_-0.01"]
        )

    def test_fails_on_pointing_gap_and_seed_k48(self):
        summary, _ = self._decision()
        summary["prompt_utility_gap_delta"]["float"]["48"]["mass"]["per_task"][
            "pointing"
        ]["mean"] = -0.02
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "pointing_k48_prompt_utility_gap_delta_candidate_vs_float_at_least_-0.01"
            ]
        )
        summary, _ = self._decision()
        summary["seed_k48_candidate_vs_per_tensor_jaccard"]["20261003"] = 0.96
        decision = audit.evaluate_decision(
            summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
        )
        self.assertFalse(
            decision["checks"][
                "each_seed_k48_candidate_vs_per_tensor_mean_jaccard_at_least_0.97"
            ]
        )


class WidthContractTests(unittest.TestCase):
    def test_analytical_contract_fits(self):
        contract = audit.declared_width_contract()
        self.assertTrue(contract["analytical_all_fit"])
        self.assertFalse(contract["float_after_input_quantization"])
        for entry in contract["operations"]:
            self.assertTrue(entry["fits"])
            self.assertLessEqual(entry["analytical_bound"], entry["max_value"])

    def test_finalize_detects_observed_overflow(self):
        contract = audit.declared_width_contract()
        good = audit.finalize_width_contract(contract, {"score_q_max": 32767})
        self.assertTrue(good["all_fit_declared_widths"])
        bad = audit.finalize_width_contract(contract, {"score_q_max": 40000})
        self.assertFalse(bad["all_fit_declared_widths"])
        self.assertFalse(bad["observed_all_fit"])


class PinsTests(unittest.TestCase):
    def test_verify_pins_checks_frozen_inputs_and_bundles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "frozen.txt").write_text("frozen", encoding="utf-8")
            (root / "Wv.bin").write_bytes(b"\x01\x02")
            (root / "scale.bin").write_bytes(b"\x00\x00\x80\x3f")
            prereg = {
                "frozen_inputs": {
                    "frozen.txt": audit.sha256_file(root / "frozen.txt"),
                },
                "checkpoints": [],
                "accepted_wv_bundles": [
                    {
                        "seed": 1,
                        "wv_path": "Wv.bin",
                        "wv_sha256": audit.sha256_file(root / "Wv.bin"),
                        "scale_path": "scale.bin",
                        "scale_sha256": audit.sha256_file(root / "scale.bin"),
                    }
                ],
            }
            result = audit.verify_pins(prereg, root=root)
            self.assertTrue(result["pins_unchanged"])
            self.assertEqual(result["checked"], 3)
            (root / "Wv.bin").write_bytes(b"\xff")
            drifted = audit.verify_pins(prereg, root=root)
            self.assertFalse(drifted["pins_unchanged"])
            self.assertIn("Wv.bin", drifted["mismatches"])

    def test_real_config_identity_and_pins(self):
        prereg = audit.load_prereg(audit.ROOT / audit.PREREG_CONFIG_PATH)
        identity = audit.base.verify_prereguration_identity(
            prereg, root=audit.ROOT, config_path=audit.PREREG_CONFIG_PATH
        )
        self.assertTrue(identity["consistent"])
        pins = audit.verify_pins(prereg, root=audit.ROOT)
        self.assertTrue(pins["pins_unchanged"])
        # 8 frozen inputs + 3 checkpoints + 3 accepted Wv bundles x 2 files.
        self.assertEqual(pins["checked"], 17)


class BundleTests(unittest.TestCase):
    def _bundle_inputs(self):
        Wt, Wv = _projections(61)
        wv_q, wv_scale = wv_audit.quantize_weight_tensor(Wv)
        accepted = {
            "wv_path": "accepted/Wv_int8_rowmajor.bin",
            "wv_sha256": "deadbeef",
            "scale_path": "accepted/Wv_per_tensor_scale_f32le.bin",
            "scale_sha256": "cafebabe",
            "_wv_q": wv_q,
            "_scale_float32": float(np.float32(wv_scale)),
        }
        return Wt, Wv, accepted

    def test_build_seed_bundle_references_wv_and_roundtrips(self):
        Wt, Wv, accepted = self._bundle_inputs()
        fixtures = [_fixture_sample("grounding"), _fixture_sample("pointing", seed=6)]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "1"
            meta = audit.build_seed_bundle(
                1,
                Wt=Wt,
                Wv=Wv,
                fixtures=fixtures,
                checkpoint={"path": "ckpt.npz", "sha256": "abc"},
                accepted_bundle=accepted,
                out_dir=out_dir,
            )
            self.assertTrue(meta["roundtrip"]["ok"])
            # Wv is referenced, never copied into the seed directory.
            names = {path.name for path in out_dir.iterdir()}
            self.assertNotIn("Wv_int8_rowmajor.bin", names)
            self.assertNotIn("Wv_per_tensor_scale_f32le.bin", names)
            self.assertEqual(meta["wv_reference"]["sha256"], "deadbeef")
            self.assertFalse(meta["wv_reference"]["path"].startswith(str(out_dir)))
            written = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                written["wv_reference"]["scale_sha256"], "cafebabe"
            )

    def test_roundtrip_detects_corruption(self):
        Wt, Wv, accepted = self._bundle_inputs()
        fixtures = [_fixture_sample("grounding")]
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "1"
            audit.build_seed_bundle(
                1, Wt=Wt, Wv=Wv, fixtures=fixtures,
                checkpoint={"path": "ckpt.npz", "sha256": "abc"},
                accepted_bundle=accepted, out_dir=out_dir,
            )
            name = out_dir / "fixture_grounding.npz"
            with np.load(name, allow_pickle=False) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["score_q_correct"] = payload["score_q_correct"].copy()
            payload["score_q_correct"][0] = (
                int(payload["score_q_correct"][0]) + 1
            ) % 32768
            np.savez(name, **payload)
            failed = audit.validate_fixture_roundtrip(
                out_dir, wv_q=accepted["_wv_q"], fixtures=fixtures
            )
            self.assertFalse(failed["ok"])


class RenderTests(unittest.TestCase):
    def _bundle(self):
        raw, sample_meta = _raw_by_seed()
        summary = audit.aggregate_metrics(raw, sample_meta=sample_meta)
        contract = audit.finalize_width_contract(
            audit.declared_width_contract(), {"score_q_max": 100}
        )
        return {
            "decision": audit.evaluate_decision(
                summary, pins_ok=True, isolation_ok=True, fixture_ok=True, width_ok=True
            ),
            "metrics": {
                "overall": summary["overall"],
                "per_task": summary["per_task"],
                "score_rank_spearman": summary["score_rank_spearman"],
                "prompt_utility_gap_delta": summary["prompt_utility_gap_delta"],
            },
            "aggregation": {"n_images": 2},
            "integer_width_contract": contract,
            "fixtures_validated": True,
            "pins_unchanged": True,
            "pins_after": {"checked": 3},
            "isolation": {
                "opened_npz_count": 4,
                "opened_only_val_npz": True,
                "test_npz_opened": [],
                "train_npz_opened": [],
                "val_npz_aggregate_matches_prereg": True,
            },
            "limitations": ["limitation"],
        }

    def test_markdown_and_harness_render(self):
        bundle = self._bundle()
        markdown = audit.render_markdown(bundle)
        self.assertIn("INTEGER_SCORER_NUMERICS_PASS", markdown)
        self.assertIn("score_q", markdown)
        harness = audit.harness_payload(bundle, {"json": "x"})
        self.assertEqual(harness["decision"], "INTEGER_SCORER_NUMERICS_PASS")
        self.assertTrue(harness["integer_width_contract"]["all_fit_declared_widths"])
        self.assertIn("candidate_vs_per_tensor", harness)


if __name__ == "__main__":
    unittest.main()
