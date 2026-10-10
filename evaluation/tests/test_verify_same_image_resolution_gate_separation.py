#!/usr/bin/env python3
"""Focused tests for the processor-only G-separation dry run."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from evaluation import verify_same_image_resolution_gate_separation as verify

REAL_CONFIG = verify.CONFIG_JSON


class TestSmartResize(unittest.TestCase):
    def test_group_count_matches_recorded_for_tiny_image(self) -> None:
        # 256x256 is exactly min_pixels=65536 at the default budget.
        self.assertEqual(
            verify.predict_group_count(
                256, 256, min_pixels=65536, max_pixels=16777216
            ),
            64,
        )

    def test_low_max_pixels_caps_group_count(self) -> None:
        g = verify.predict_group_count(
            2048, 1536, min_pixels=65536, max_pixels=524288
        )
        self.assertLessEqual(g, 512)

    def test_high_min_pixels_forces_above_gate(self) -> None:
        g = verify.predict_group_count(
            64, 64, min_pixels=525312, max_pixels=16777216
        )
        self.assertGreater(g, 512)

    def test_real_recorded_g_is_reproduced(self) -> None:
        from evaluation import analyze_full2638_prompt_gain_heterogeneity as het

        manifest = het.index_unique(
            het.load_rows(het.FULL_MANIFEST, "manifest"), "manifest"
        )
        arm = het.index_unique(
            het.load_rows(het.ARM_INTEGER_JSONL, "arm"), "arm"
        )
        checked = 0
        for sample_id, meta in manifest.items():
            if not str(meta["source_dataset"]).startswith("ADE20K"):
                continue
            with Image.open(meta["image"]) as image:
                width, height = image.size
            predicted = verify.predict_group_count(
                height, width, min_pixels=65536, max_pixels=16777216
            )
            self.assertEqual(predicted, int(arm[sample_id]["full_llm_visual_tokens"]))
            checked += 1
        self.assertEqual(checked, 633)

    def test_small_image_raises(self) -> None:
        with self.assertRaises(verify.GateSeparationError):
            verify.smart_resize(8, 8, min_pixels=65536, max_pixels=16777216)


class TestSeparationStatus(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertTrue(verify.separation_status(512, 513))
        self.assertTrue(verify.separation_status(64, 513))
        self.assertFalse(verify.separation_status(513, 513))
        self.assertFalse(verify.separation_status(512, 512))
        self.assertFalse(verify.separation_status(512, 400))


class TestPlanReplacements(unittest.TestCase):
    def _measurement(self, sid: str, stratum: str, g_low: int, g_high: int) -> dict:
        return {
            "sample_id": sid,
            "stratum": stratum,
            "G_low": g_low,
            "G_high": g_high,
            "separated": verify.separation_status(g_low, g_high),
        }

    def _reserve(self, sid: str, stratum: str, rank: int, g_low: int, g_high: int) -> dict:
        return {
            "sample_id": sid,
            "resolution_counterfactual": {"stratum": stratum, "selection_rank": rank},
            "_g_low": g_low,
            "_g_high": g_high,
        }

    def test_replacement_uses_same_stratum(self) -> None:
        stratum = "G<=512::Count::integer_correct"
        measurements = [self._measurement("a", stratum, 600, 700)]
        reserve = [
            self._reserve("b", stratum, 0, 600, 700),
            self._reserve("c", stratum, 1, 100, 600),
        ]

        def measure(candidate: dict) -> dict:
            return {
                "sample_id": candidate["sample_id"],
                "stratum": candidate["resolution_counterfactual"]["stratum"],
                "G_low": candidate["_g_low"],
                "G_high": candidate["_g_high"],
                "separated": verify.separation_status(
                    candidate["_g_low"], candidate["_g_high"]
                ),
            }

        plan = verify.plan_replacements(
            measurements, reserve, measure_candidate=measure
        )
        self.assertEqual(plan["replaced"], [
            {
                "dropped": "a",
                "replacement": "c",
                "stratum": stratum,
                "replacement_G_low": 100,
                "replacement_G_high": 600,
            }
        ])
        self.assertEqual(plan["accepted"], ["c"])
        self.assertTrue(plan["resolved"])

    def test_exhausted_reserve_is_unresolved(self) -> None:
        stratum = "G<=512::Count::integer_correct"
        measurements = [self._measurement("a", stratum, 600, 700)]
        reserve = [self._reserve("b", stratum, 0, 600, 700)]

        def measure(candidate: dict) -> dict:
            return {
                "sample_id": candidate["sample_id"],
                "stratum": candidate["resolution_counterfactual"]["stratum"],
                "G_low": candidate["_g_low"],
                "G_high": candidate["_g_high"],
                "separated": verify.separation_status(
                    candidate["_g_low"], candidate["_g_high"]
                ),
            }

        plan = verify.plan_replacements(
            measurements, reserve, measure_candidate=measure
        )
        self.assertEqual(plan["unresolved_count"], 1)
        self.assertFalse(plan["resolved"])

    def test_all_separated_never_touches_reserve(self) -> None:
        stratum = "G<=512::Count::integer_correct"
        measurements = [self._measurement("a", stratum, 100, 600)]
        calls: list[str] = []

        def measure(candidate: dict) -> dict:  # pragma: no cover - must not run
            calls.append(candidate["sample_id"])
            raise AssertionError("reserve must not be measured")

        plan = verify.plan_replacements(
            measurements,
            [self._reserve("b", stratum, 0, 100, 600)],
            measure_candidate=measure,
        )
        self.assertEqual(plan["accepted"], ["a"])
        self.assertEqual(calls, [])


class TestBudgets(unittest.TestCase):
    def test_real_config_budgets(self) -> None:
        config = verify.load_config(REAL_CONFIG)
        budgets = verify.budgets_from_config(config)
        self.assertEqual(budgets["low"]["max_pixels"], 524288)
        self.assertEqual(budgets["high"]["min_pixels"], 525312)

    def test_bad_budgets_raise(self) -> None:
        with self.assertRaises(verify.GateSeparationError):
            verify.budgets_from_config({"processor_pixel_budgets": {"low": {}}})
        with self.assertRaises(verify.GateSeparationError):
            verify.budgets_from_config(
                {
                    "processor_pixel_budgets": {
                        "low": {"min_pixels": 100, "max_pixels": 50},
                        "high": {"min_pixels": 200, "max_pixels": 400},
                    }
                }
            )


class TestAnalyzeAnalytic(unittest.TestCase):
    def _write_image(self, path: Path, size: tuple[int, int] = (64, 64)) -> None:
        Image.new("RGB", size, (10, 20, 30)).save(path)

    def _fixture(self, tmp: Path, *, separated: bool = True) -> dict:
        image_low = tmp / "low.png"
        image_low_reserve = tmp / "reserve.png"
        self._write_image(image_low, (64, 64))
        self._write_image(image_low_reserve, (64, 64))
        stratum = "G<=512::Count::integer_correct"
        manifest = [
            {
                "sample_id": "1",
                "task": "Count",
                "source": "ADE20K",
                "source_dataset": "ADE20K Validation Set",
                "image": str(image_low),
                "prompt": "how many?",
                "choices": ["a", "b"],
                "answer": "(A)",
                "resolution_counterfactual": {
                    "design": verify.DESIGN,
                    "split": "selected",
                    "stratum": stratum,
                    "selection_rank": 0,
                },
            }
        ]
        reserve = [
            {
                "sample_id": "2",
                "task": "Count",
                "source": "ADE20K",
                "source_dataset": "ADE20K Validation Set",
                "image": str(image_low_reserve),
                "prompt": "how many?",
                "choices": ["a", "b"],
                "answer": "(A)",
                "resolution_counterfactual": {
                    "design": verify.DESIGN,
                    "split": "reserve",
                    "stratum": stratum,
                    "selection_rank": 0,
                },
            }
        ]
        manifest_path = tmp / "manifest.jsonl"
        reserve_path = tmp / "reserve.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
        )
        reserve_path.write_text(
            "".join(json.dumps(row) + "\n" for row in reserve), encoding="utf-8"
        )
        budgets = verify.budgets_from_config(verify.load_config(REAL_CONFIG))
        if not separated:
            budgets["high"] = dict(budgets["low"])
        config_path = tmp / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "frozen_sample_count": 1,
                    "processor_pixel_budgets": budgets,
                }
            ),
            encoding="utf-8",
        )
        return {
            "manifest_path": manifest_path,
            "reserve_path": reserve_path,
            "config_path": config_path,
        }

    def test_analytic_separates_and_writes_no_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = self._fixture(tmp_path)
            report = verify.analyze(mode="analytic", **fixture)
            self.assertFalse(report["authoritative"])
            self.assertTrue(report["plan"]["resolved"])
            self.assertEqual(report["plan"]["accepted_count"], 1)
            self.assertIsNone(report["final_manifest"])
            self.assertTrue(all(check["ok"] for check in report["contract_checks"]))

    def test_analytic_replaces_unseparated_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = self._fixture(tmp_path, separated=False)
            report = verify.analyze(mode="analytic", **fixture)
            # Both selected and reserve are non-separating, so the run is unresolved.
            self.assertFalse(report["plan"]["resolved"])
            self.assertEqual(report["plan"]["rejected_count"], 1)
            self.assertEqual(report["plan"]["unresolved_count"], 1)

    def test_processor_mode_requires_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = self._fixture(tmp_path)
            with self.assertRaises(verify.GateSeparationError):
                verify.analyze(
                    mode="processor", model_path=tmp_path / "does-not-exist", **fixture
                )

    def test_frozen_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fixture = self._fixture(tmp_path)
            config = json.loads(fixture["config_path"].read_text())
            config["frozen_sample_count"] = 99
            fixture["config_path"].write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(verify.GateSeparationError):
                verify.analyze(mode="analytic", **fixture)


class TestFrozenPrediction(unittest.TestCase):
    def test_frozen_100_all_separate_analytic(self) -> None:
        report = verify.analyze(mode="analytic")
        self.assertEqual(len(report["measurements"]), 100)
        self.assertTrue(all(m["separated"] for m in report["measurements"]))
        self.assertEqual(report["plan"]["rejected_count"], 0)
        self.assertTrue(report["plan"]["resolved"])
        self.assertFalse(report["authoritative"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
