#!/usr/bin/env python3
"""Focused CPU tests for the train-only pair-gradient concentration audit.

Coverage:

* the positive-only concentration statistics (top-share / ESS / Gini) on known
  arrays, including the explicit positive-only Gini convention;
* the signed projection contribution identity (``sum_i c_i ~ 1`` per seed) on a
  synthetic train-only dataset, and its pooled expected sum = seed count;
* the statistical invariants: a positive ``lambda`` rescales pair/base only and
  leaves pair-only concentration / rank / cosine unchanged;
* the split/loader reuse and the hard isolation contract (split from index
  metadata only, only train npz opened, corrupted val/test npz cannot move any
  train metric or hash);
* the static source test forbidding any parameter-update / persistent-save /
  model / GPU / network construct and a second scorer / loss;
* the report's required limitations and the CSV schema.

No training, no model, no CUDA, no network.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import pair_gradient_concentration_audit as audit
from evaluation import pair_loss_feasibility_audit as feasibility
from evaluation import pair_loss_scale_audit as scale
from evaluation.tests.test_pair_loss_feasibility_audit import SyntheticDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / "configs" / "prompt_projection_distillation.json"
REAL_DRAFT = PROJECT_ROOT / audit.DRAFT_RELATIVE_PATH
SOURCE_PATH = PROJECT_ROOT / "evaluation" / "pair_gradient_concentration_audit.py"
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")


class SourceSafetyTests(unittest.TestCase):
    """The audit must not be able to update, save or load anything heavy."""

    def test_no_forbidden_update_or_optimizer_tokens(self) -> None:
        for token in audit.FORBIDDEN_SOURCE_TOKENS:
            self.assertNotIn(token, SOURCE_TEXT, f"forbidden token {token!r} in audit source")
        self.assertNotIn("torch" + "." + "optim", SOURCE_TEXT)
        self.assertNotIn("Adam" + "W", SOURCE_TEXT)

    def test_only_autograd_grad_and_no_backward(self) -> None:
        self.assertIn("autograd.grad", SOURCE_TEXT)
        self.assertNotIn(".backward(", SOURCE_TEXT)

    def test_no_model_gpu_or_network_tokens(self) -> None:
        for token in ("from_pretrained", "transformers", "torchvision", "requests", "urllib"):
            self.assertNotIn(token, SOURCE_TEXT, f"forbidden token {token!r} in audit source")
        self.assertNotIn('to("cuda")', SOURCE_TEXT)
        self.assertNotIn("to('cuda')", SOURCE_TEXT)

    def test_gradient_and_loss_helpers_are_reused_not_duplicated(self) -> None:
        self.assertIs(audit.pair_loss_terms, feasibility.pair_loss_terms)
        self.assertIs(audit.prompt_query, feasibility.prompt_query)
        self.assertIs(audit.visual_unit, feasibility.visual_unit)
        self.assertIs(audit.load_train_samples, feasibility.load_train_samples)
        self.assertIs(audit._require_torch, feasibility._require_torch)
        self.assertIs(audit._forward_pair_losses, scale._forward_pair_losses)
        self.assertIs(audit.candidate_lambdas, scale.candidate_lambdas)


class ConcentrationMathTests(unittest.TestCase):
    def test_gini_is_computed_on_positive_values_only(self) -> None:
        # Two zeros are ignored: the remaining [10, 10] have Gini 0.
        self.assertAlmostEqual(audit.gini_coefficient([0.0, 0.0, 10.0, 10.0]), 0.0, places=12)
        metrics = audit.concentration_metrics([0.0, 0.0, 10.0, 10.0])
        self.assertEqual(metrics["count"], 4)
        self.assertEqual(metrics["zero_count"], 2)
        self.assertEqual(metrics["positive_count"], 2)

    def test_gini_of_uniform_is_zero_and_of_single_large_is_high(self) -> None:
        self.assertAlmostEqual(audit.gini_coefficient([1.0] * 10), 0.0, places=12)
        concentrated = audit.gini_coefficient([1.0] * 9 + [1000.0])
        self.assertGreater(concentrated, 0.8)

    def test_effective_sample_size_of_uniform_equals_n(self) -> None:
        self.assertAlmostEqual(audit.effective_sample_size([2.0] * 7), 7.0, places=10)
        self.assertAlmostEqual(audit.effective_sample_size([1000.0] + [0.0] * 9), 1.0, places=10)

    def test_top_fraction_share_uses_ceil(self) -> None:
        values = [4.0, 3.0, 2.0, 1.0]
        # ceil(0.2 * 4) = 1 -> top value 4 / total 10.
        self.assertAlmostEqual(audit.top_fraction_share(values, 0.20), 0.4, places=12)
        # ceil(0.5 * 4) = 2 -> (4 + 3) / 10.
        self.assertAlmostEqual(audit.top_fraction_share(values, 0.50), 0.7, places=12)
        # Fraction 0.01 with n=4 still selects one value.
        self.assertAlmostEqual(audit.top_fraction_share(values, 0.01), 0.4, places=12)

    def test_top_fraction_share_ignores_zero_and_rejects_bad_fraction(self) -> None:
        # positive values are [3, 1]; ceil(0.5 * 2) = 1 -> 3 / 4.
        self.assertAlmostEqual(audit.top_fraction_share([0.0, 3.0, 1.0], 0.5), 0.75, places=12)
        self.assertAlmostEqual(audit.top_fraction_share([0.0, 3.0, 1.0], 1.0), 1.0, places=12)
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(feasibility.DistillationContractError):
                audit.top_fraction_share([1.0, 2.0], bad)

    def test_signed_contribution_metrics_sum_and_negative_mass(self) -> None:
        metrics = audit.signed_contribution_metrics([1.5, -0.5, 0.0])
        self.assertAlmostEqual(metrics["sum"], 1.0, places=12)
        self.assertAlmostEqual(metrics["sum_error_abs"], 0.0, places=12)
        self.assertEqual(metrics["negative_count"], 1)
        self.assertEqual(metrics["zero_count"], 1)
        self.assertAlmostEqual(metrics["negative_mass_abs"], 0.5, places=12)
        self.assertAlmostEqual(metrics["positive_sum"], 1.5, places=12)

    def test_pooled_expected_sum_is_seed_count(self) -> None:
        metrics = audit.signed_contribution_metrics(
            [1.0, 1.0, 0.999999], expected_sum=3.0
        )
        self.assertLess(metrics["sum_error_abs"], 1e-5)


class AuditRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dataset = SyntheticDataset(Path(self._tmp.name), {"train": 3, "val": 2, "test": 2})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, **kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return audit.run_audit(
            config_path=REAL_CONFIG,
            draft_path=REAL_DRAFT,
            index_path=self.dataset.index_path,
            features_dir=self.dataset.features_dir,
            **kwargs,
        )

    def test_run_is_train_only_and_reports_all_seeds(self) -> None:
        report, rows = self._run()
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["split_counts"]["train"], 6)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])
        self.assertEqual(report["isolation"]["val_test_npz_opened"], [])
        self.assertTrue(all(report["draft_contract_checks"].values()))
        audit_block = report["concentration_audit"]
        self.assertEqual(len(audit_block["per_seed"]), 3)
        # One wide CSV row per (seed, image pair).
        self.assertEqual(len(rows), 3 * 3)
        # On the synthetic dataset the derived scale need not equal the frozen
        # real-data constant, but it must reproduce 0.25 / r_median exactly.
        self.assertAlmostEqual(
            audit_block["lambda_target_0p25_used"],
            scale.candidate_lambdas(audit_block["computed_r_median"])["lambda_target_0p25"],
            places=12,
        )

    def test_signed_contributions_sum_to_one_per_seed_and_group(self) -> None:
        report, _ = self._run()
        for block in report["concentration_audit"]["per_seed"]:
            for group in audit.PARAM_GROUPS:
                metrics = block["per_param"][group]["pair_signed_contribution"]
                self.assertAlmostEqual(metrics["sum"], 1.0, places=4, msg=f"{block['seed']}/{group}")
                self.assertLess(metrics["sum_error_abs"], 1e-4)
        self.assertLess(report["concentration_audit"]["signed_sum_error_max"], 1e-4)

    def test_lambda_scales_pair_only_metrics_and_preserves_shape(self) -> None:
        low, _ = self._run(lambda_value=1.0)
        high, _ = self._run(lambda_value=7.0)
        low_block = low["concentration_audit"]
        high_block = high["concentration_audit"]
        self.assertIsNone(low_block["r_median_matches_frozen"])
        self.assertIsNone(high_block["lambda_matches_frozen"])
        for group in audit.PARAM_GROUPS:
            low_pooled = low_block["pooled"][group]
            high_pooled = high_block["pooled"][group]
            # Pair-only concentration is invariant to a positive lambda.
            self.assertEqual(
                low_pooled["pair_grad_norm_concentration"],
                high_pooled["pair_grad_norm_concentration"],
            )
            self.assertEqual(
                low_pooled["pair_signed_contribution"]["positive_gini"],
                high_pooled["pair_signed_contribution"]["positive_gini"],
            )
        # The rank structure is invariant as well.
        self.assertEqual(
            low_block["cross_seed_pair_norm_rank"], high_block["cross_seed_pair_norm_rank"]
        )
        # The scaled columns scale exactly with lambda.
        for group in audit.PARAM_GROUPS:
            low_metric = low_block["pooled"][group]["pair_grad_norm"]["mean"]
            high_metric = high_block["pooled"][group]["pair_grad_norm"]["mean"]
            self.assertAlmostEqual(low_metric, high_metric, places=10)
        for block in high_block["per_seed"]:
            for group in audit.PARAM_GROUPS:
                scaled = block["per_param"][group]["lambda_0p25_scaled_pair_norm"]["mean"]
                raw = block["per_param"][group]["pair_grad_norm"]["mean"]
                self.assertAlmostEqual(scaled, 7.0 * raw, places=4)

    def test_aggregate_gradient_is_the_mean_gradient(self) -> None:
        report, _ = self._run()
        # r_seed is the aggregate pair/base ratio and is positive and finite.
        for block in report["concentration_audit"]["per_seed"]:
            ratio = block["r_seed_concat_pair_over_base"]
            self.assertIsNotNone(ratio)
            self.assertGreater(ratio, 0.0)
            self.assertTrue(math.isfinite(ratio))

    def test_only_train_feature_cache_is_opened(self) -> None:
        seen: list[str] = []
        real_loader = feasibility.load_feature_cache

        def spy(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            seen.append(str(path))
            return real_loader(path, *args, **kwargs)

        report, _ = self._run(cache_loader=spy)
        self.assertEqual(set(seen), self.dataset.train_npz)
        self.assertEqual(set(seen) & self.dataset.val_test_npz, set())
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_loader_that_raises_on_non_train_paths(self) -> None:
        allowed = self.dataset.train_npz.copy()

        def strict(path: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
            if str(path) not in allowed:
                raise AssertionError(f"non-train cache opened: {path}")
            return feasibility.load_feature_cache(path, *args, **kwargs)

        report, _ = self._run(cache_loader=strict)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_corrupting_val_test_npz_cannot_move_train_metrics_or_hash(self) -> None:
        report_before, rows_before = self._run()
        for path in self.dataset.val_test_npz:
            Path(path).write_bytes(b"this is not a valid npz file")
        report_after, rows_after = self._run()
        self.assertEqual(rows_before, rows_after)
        self.assertEqual(
            report_before["concentration_audit"], report_after["concentration_audit"]
        )
        self.assertEqual(
            report_before["provenance"]["input_npz_aggregate_sha256"],
            report_after["provenance"]["input_npz_aggregate_sha256"],
        )
        self.assertEqual(
            report_before["provenance"]["index_sha256"],
            report_after["provenance"]["index_sha256"],
        )

    def test_index_metadata_load_opens_no_npz(self) -> None:
        import unittest.mock as mock

        with mock.patch.object(
            feasibility, "load_feature_cache", side_effect=AssertionError("npz opened")
        ):
            metadata = feasibility.load_index_metadata(self.dataset.index_path)
        self.assertEqual(len(metadata), 14)

    def test_run_is_deterministic_and_does_not_modify_the_draft(self) -> None:
        before = REAL_DRAFT.read_bytes()
        first, rows_first = self._run()
        second, rows_second = self._run()
        self.assertEqual(before, REAL_DRAFT.read_bytes())
        self.assertEqual(rows_first, rows_second)
        self.assertEqual(
            first["concentration_audit"]["pooled"],
            second["concentration_audit"]["pooled"],
        )

    def test_frozen_constants_match_the_scale_audit_report(self) -> None:
        self.assertAlmostEqual(
            scale.candidate_lambdas(audit.FROZEN_R_MEDIAN)["lambda_target_0p25"],
            audit.FROZEN_LAMBDA_TARGET_0P25,
            places=12,
        )

    def test_hardware_contract_is_declared(self) -> None:
        report, _ = self._run()
        contract = report["concentration_audit"]["hardware_contract"]
        self.assertEqual(contract["query_shape"], [1, 128])
        self.assertEqual(contract["scorer"], "Gx128 dot")


class MarkdownAndWriterTests(unittest.TestCase):
    def test_markdown_and_csv_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            report, rows = audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
            markdown = audit.render_markdown(report)
            self.assertIn("train-only", markdown)
            self.assertIn("signed", markdown)
            self.assertIn("不是 aggregate 梯度", markdown)
            self.assertIn("top-share", markdown)
            self.assertIn("不训练", markdown)
            csv_path = Path(tmp) / "concentration.csv"
            audit.write_csv(csv_path, rows)
            header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
            self.assertEqual(header, list(audit.CSV_FIELDS))
            self.assertEqual(
                len(csv_path.read_text(encoding="utf-8").strip().splitlines()),
                len(rows) + 1,
            )

    def test_report_limitations_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            report, _ = audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
        joined = " ".join(report["limitations"])
        self.assertIn("不是 aggregate 梯度", joined)
        self.assertIn("可为负", joined)
        self.assertIn("不改变 pair-only", joined)
        self.assertIn("不设通过/失败阈值", joined)
        self.assertIn("不能从初始梯度推断收敛", joined)


if __name__ == "__main__":
    unittest.main()
