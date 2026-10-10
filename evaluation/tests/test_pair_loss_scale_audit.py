#!/usr/bin/env python3
"""Focused CPU tests for the train-only pair-loss scale-stability audit.

Coverage:

* the candidate ``lambda`` set is exactly the four frozen scales and is derived
  from ``r_med`` without inventing any other scale;
* the per-parameter gradient diagnostics (scaled pair/base and total/base norm
  ratios, base/pair/total cosines) match a direct ``autograd.grad`` computation;
* the split/loader reuse and the hard isolation contract (split from index
  metadata only, only train npz opened, corrupted val/test npz cannot move any
  train metric or hash);
* the static source test forbidding any parameter-update / persistent-save
  construct;
* the report's required limitations and the CSV schema.

No training, no model, no CUDA, no network.
"""

from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from evaluation import pair_loss_feasibility_audit as feasibility
from evaluation import pair_loss_scale_audit as audit
from evaluation.tests.test_pair_loss_feasibility_audit import SyntheticDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / "configs" / "prompt_projection_distillation.json"
REAL_DRAFT = PROJECT_ROOT / audit.DRAFT_RELATIVE_PATH
SOURCE_PATH = PROJECT_ROOT / "evaluation" / "pair_loss_scale_audit.py"
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")


class SourceSafetyTests(unittest.TestCase):
    """The audit must not be able to update, save or load anything heavy."""

    def test_no_forbidden_update_or_optimizer_tokens(self) -> None:
        for token in audit.FORBIDDEN_SOURCE_TOKENS:
            self.assertNotIn(token, SOURCE_TEXT, f"forbidden token {token!r} in audit source")

    def test_only_autograd_grad_and_no_backward(self) -> None:
        self.assertIn("autograd.grad", SOURCE_TEXT)
        self.assertNotIn(".backward(", SOURCE_TEXT)

    def test_no_model_gpu_or_network_tokens(self) -> None:
        for token in ("from_pretrained", "transformers", "torchvision", "requests", "urllib"):
            self.assertNotIn(token, SOURCE_TEXT, f"forbidden token {token!r} in audit source")
        self.assertNotIn('to("cuda")', SOURCE_TEXT)
        self.assertNotIn("to('cuda')", SOURCE_TEXT)

    def test_gradient_and_loss_helpers_are_reused_not_duplicated(self) -> None:
        # Reuse, not a second algorithm.
        self.assertIs(audit.pair_loss_terms, feasibility.pair_loss_terms)
        self.assertIs(audit._grad_diagnostics, feasibility._grad_diagnostics)
        self.assertIs(audit._cosine, feasibility._cosine)
        self.assertIs(audit._require_torch, feasibility._require_torch)
        self.assertIs(audit.load_train_samples, feasibility.load_train_samples)


class CandidateLambdaTests(unittest.TestCase):
    def test_candidate_set_is_exactly_the_four_frozen_scales(self) -> None:
        lambdas = audit.candidate_lambdas(0.02)
        self.assertEqual(tuple(lambdas), audit.CANDIDATE_LAMBDA_NAMES)
        self.assertEqual(len(lambdas), 4)
        self.assertEqual(set(audit.TARGET_LAMBDA_RATIOS), {
            "lambda_target_0p1",
            "lambda_target_0p25",
            "lambda_target_1p0",
        })

    def test_candidate_values_are_target_over_r_median(self) -> None:
        r_median = 0.017507
        lambdas = audit.candidate_lambdas(r_median)
        self.assertAlmostEqual(lambdas["lambda_current"], 1.0)
        self.assertAlmostEqual(lambdas["lambda_target_0p1"], 0.1 / r_median, places=12)
        self.assertAlmostEqual(lambdas["lambda_target_0p25"], 0.25 / r_median, places=12)
        self.assertAlmostEqual(lambdas["lambda_target_1p0"], 1.0 / r_median, places=12)

    def test_candidate_rejects_nonpositive_or_nonfinite_r_median(self) -> None:
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(feasibility.DistillationContractError):
                audit.candidate_lambdas(bad)


class DiagnosticsMathTests(unittest.TestCase):
    def _problem(self) -> tuple[Any, ...]:
        torch = audit._require_torch()
        generator = torch.Generator().manual_seed(20261002)
        Wt = (torch.randn(4, 3, generator=generator, dtype=torch.float64) * 0.3).requires_grad_(True)
        Wv = (torch.randn(4, 5, generator=generator, dtype=torch.float64) * 0.3).requires_grad_(True)
        group_mean = torch.randn(6, 5, generator=generator, dtype=torch.float64)
        prompt_g = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        prompt_p = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        y_g = torch.softmax(torch.randn(6, generator=generator, dtype=torch.float64), dim=-1)
        y_p = torch.softmax(torch.randn(6, generator=generator, dtype=torch.float64), dim=-1)
        return Wt, Wv, group_mean, prompt_g, prompt_p, y_g, y_p

    def _losses(self) -> tuple[Any, Any, Any, Any]:
        torch = audit._require_torch()
        Wt, Wv, group_mean, prompt_g, prompt_p, y_g, y_p = self._problem()
        scores_g = feasibility.student_scores(group_mean, prompt_g, Wt, Wv)
        scores_p = feasibility.student_scores(group_mean, prompt_p, Wt, Wv)
        loss_base, loss_pair, _d_swap, _l_total, _margin = audit.pair_loss_terms(
            scores_g, scores_p, y_g, y_p, 5.0
        )
        return Wt, Wv, loss_base, loss_pair

    def test_total_gradient_equals_base_plus_lambda_pair(self) -> None:
        torch = audit._require_torch()
        Wt, Wv, loss_base, loss_pair = self._losses()
        for lam in (0.1, 0.25, 1.0, 7.0):
            base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
            pair_grads = torch.autograd.grad(loss_pair, [Wt, Wv], retain_graph=True)
            total_grads = torch.autograd.grad(loss_base + lam * loss_pair, [Wt, Wv], retain_graph=True)
            for index in (0, 1):
                expected = base_grads[index] + lam * pair_grads[index]
                self.assertTrue(torch.allclose(total_grads[index], expected, atol=1e-10, rtol=1e-8))

    def test_scaled_pair_ratio_scales_linearly_with_lambda(self) -> None:
        torch = audit._require_torch()
        Wt, Wv, loss_base, loss_pair = self._losses()
        base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
        pair_grads = torch.autograd.grad(loss_pair, [Wt, Wv], retain_graph=True)
        raw = feasibility._grad_diagnostics(base_grads, pair_grads, base_grads)["concat"][
            "pair_over_base_norm_ratio"
        ]
        for lam in (0.1, 0.25, 1.0, 7.0):
            scaled = torch.autograd.grad(lam * loss_pair, [Wt, Wv], retain_graph=True)
            total = torch.autograd.grad(loss_base + lam * loss_pair, [Wt, Wv], retain_graph=True)
            diag = audit._scale_grad_diagnostics(base_grads, scaled, total)
            self.assertAlmostEqual(
                diag["concat"]["scaled_pair_over_base_norm_ratio"], lam * raw, places=10
            )
            self.assertAlmostEqual(
                diag["concat"]["cos_base_pair"],
                feasibility._grad_diagnostics(base_grads, pair_grads, base_grads)["concat"][
                    "base_pair_cosine"
                ],
                places=10,
            )

    def test_diagnostics_match_direct_tensor_math(self) -> None:
        torch = audit._require_torch()
        Wt, Wv, loss_base, loss_pair = self._losses()
        lam = 3.0
        base_grads = torch.autograd.grad(loss_base, [Wt, Wv], retain_graph=True)
        scaled = torch.autograd.grad(lam * loss_pair, [Wt, Wv], retain_graph=True)
        total = torch.autograd.grad(loss_base + lam * loss_pair, [Wt, Wv], retain_graph=True)
        diag = audit._scale_grad_diagnostics(base_grads, scaled, total)
        for group, index in (("Wt", 0), ("Wv", 1)):
            base_norm = float(base_grads[index].norm())
            self.assertAlmostEqual(diag[group]["l_base_norm"], base_norm, places=8)
            self.assertAlmostEqual(
                diag[group]["scaled_pair_over_base_norm_ratio"],
                float(scaled[index].norm()) / base_norm,
                places=8,
            )
            self.assertAlmostEqual(
                diag[group]["total_over_base_norm_ratio"],
                float(total[index].norm()) / base_norm,
                places=8,
            )
            self.assertAlmostEqual(
                diag[group]["cos_total_base"],
                feasibility._cosine(total[index], base_grads[index]),
                places=8,
            )
            self.assertAlmostEqual(
                diag[group]["cos_total_pair"],
                feasibility._cosine(total[index], scaled[index]),
                places=8,
            )
            self.assertTrue(diag[group]["l_base_finite"])
            self.assertTrue(diag[group]["l_total_nonzero"])


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

    def test_run_reports_four_lambdas_and_train_only(self) -> None:
        report, per_seed, rows = self._run()
        self.assertEqual(report["status"], "COMPUTED")
        self.assertEqual(report["split_counts"]["train"], 6)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])
        self.assertEqual(report["isolation"]["val_test_npz_opened"], [])
        self.assertTrue(all(report["draft_contract_checks"].values()))
        self.assertEqual(tuple(report["scale_audit"]["candidate_lambdas"]), audit.CANDIDATE_LAMBDA_NAMES)
        self.assertEqual(len(per_seed), 3)
        self.assertEqual(len(rows), 3 * 4 * 3)

    def test_r_median_is_median_of_seed_concat_ratios(self) -> None:
        report, _, _ = self._run()
        ratios = [
            float(value)
            for value in report["scale_audit"]["r_seed_concat_pair_over_base"].values()
        ]
        self.assertEqual(len(ratios), 3)
        self.assertAlmostEqual(
            report["scale_audit"]["r_median"], float(np.median(ratios)), places=12
        )
        self.assertGreater(report["scale_audit"]["r_median"], 0.0)

    def test_lambda_current_concat_ratio_equals_raw_pair_over_base(self) -> None:
        report, _, _ = self._run()
        per_seed = report["scale_audit"]["per_seed"]
        base_ratios = report["scale_audit"]["r_seed_concat_pair_over_base"]
        for seed_block in per_seed:
            current = seed_block["lambdas"]["lambda_current"]
            self.assertAlmostEqual(
                current["gradient"]["concat"]["scaled_pair_over_base_norm_ratio"],
                float(base_ratios[str(seed_block["seed"])]),
                places=10,
            )

    def test_all_scaled_metrics_are_finite_and_nonzero(self) -> None:
        report, _, _ = self._run()
        for name, agg in report["scale_audit"]["cross_seed"].items():
            self.assertTrue(agg["all_finite"], msg=name)
            self.assertTrue(agg["all_nonzero"], msg=name)
            for group in audit.PARAM_GROUPS:
                for field in (
                    "scaled_pair_over_base_norm_ratio",
                    "total_over_base_norm_ratio",
                    "cos_total_base",
                    "cos_total_pair",
                    "cos_base_pair",
                ):
                    stats = agg["per_param"][group][field]
                    self.assertIsNotNone(stats["mean"], msg=f"{name}/{group}/{field}")
                    self.assertIsNotNone(stats["cv"], msg=f"{name}/{group}/{field}")

    def test_target_lambdas_hit_target_median_ratio(self) -> None:
        report, _, _ = self._run()
        for name, target in audit.TARGET_LAMBDA_RATIOS.items():
            agg = report["scale_audit"]["cross_seed"][name]
            self.assertAlmostEqual(agg["target_ratio"], target)
            # The seed-median realised concat ratio must reproduce the target by
            # construction: lambda * r_seed median == (target / r_med) * r_med.
            median = agg["per_param"]["concat"]["scaled_pair_over_base_norm_ratio"]["median"]
            self.assertAlmostEqual(median, target, places=6)

    def test_only_train_feature_cache_is_opened(self) -> None:
        seen: list[str] = []
        real_loader = feasibility.load_feature_cache

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
            return feasibility.load_feature_cache(path, *args, **kwargs)

        report, _, _ = self._run(cache_loader=strict)
        self.assertTrue(report["isolation"]["opened_only_train_npz"])

    def test_corrupting_val_test_npz_cannot_move_train_metrics_or_hash(self) -> None:
        report_before, per_seed_before, rows_before = self._run()
        for path in self.dataset.val_test_npz:
            Path(path).write_bytes(b"this is not a valid npz file")
        report_after, per_seed_after, rows_after = self._run()
        self.assertEqual(per_seed_before, per_seed_after)
        self.assertEqual(rows_before, rows_after)
        self.assertEqual(
            report_before["provenance"]["input_npz_aggregate_sha256"],
            report_after["provenance"]["input_npz_aggregate_sha256"],
        )
        self.assertEqual(
            report_before["scale_audit"]["r_median"], report_after["scale_audit"]["r_median"]
        )
        self.assertEqual(
            report_before["scale_audit"]["cross_seed"], report_after["scale_audit"]["cross_seed"]
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
        first, _, _ = self._run()
        second, _, _ = self._run()
        self.assertEqual(before, REAL_DRAFT.read_bytes())
        self.assertEqual(first["scale_audit"]["r_median"], second["scale_audit"]["r_median"])
        self.assertEqual(first["scale_audit"]["cross_seed"], second["scale_audit"]["cross_seed"])

    def test_hardware_contract_is_declared(self) -> None:
        report, _, _ = self._run()
        contract = report["scale_audit"]["hardware_contract"]
        self.assertEqual(contract["query_shape"], [1, 128])
        self.assertEqual(contract["scorer"], "Gx128 dot")


class MarkdownAndWriterTests(unittest.TestCase):
    def test_markdown_and_csv_writers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset = SyntheticDataset(Path(tmp), {"train": 2, "val": 1, "test": 1})
            report, _per_seed, rows = audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
            markdown = audit.render_markdown(report)
            self.assertIn("train-only", markdown)
            self.assertIn("r_med", markdown)
            self.assertIn("不足以说明", markdown)
            self.assertIn("不是文献标准", markdown)
            self.assertIn("不能从零更新初始梯度推断收敛", markdown)
            self.assertIn("不训练", markdown)
            self.assertIn("promotion", markdown)
            csv_path = Path(tmp) / "scale.csv"
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
            report, _, _ = audit.run_audit(
                config_path=REAL_CONFIG,
                draft_path=REAL_DRAFT,
                index_path=dataset.index_path,
                features_dir=dataset.features_dir,
            )
        joined = " ".join(report["limitations"])
        self.assertIn("1–2%", joined)
        self.assertIn("不是文献标准", joined)
        self.assertIn("不能从零更新初始梯度推断收敛", joined)
        self.assertIn("不修改 DRAFT", joined)
        self.assertIn("promotion", joined)


if __name__ == "__main__":
    unittest.main()
