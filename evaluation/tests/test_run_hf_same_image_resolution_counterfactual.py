#!/usr/bin/env python3
"""Focused tests for the same-image resolution counterfactual runner scaffold."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import run_hf_same_image_resolution_counterfactual as runner


class TestPixelBudget(unittest.TestCase):
    def test_metadata(self) -> None:
        meta = runner.pixel_budget_metadata("low", 65536, 524288)
        self.assertEqual(meta["design"], runner.DESIGN)
        self.assertEqual(meta["label"], "low")
        self.assertEqual(meta["min_pixels"], 65536)

    def test_unknown_label_raises(self) -> None:
        with self.assertRaises(runner.CounterfactualRunnerError):
            runner.pixel_budget_metadata("medium", 1, 2)

    def test_resolve_defaults(self) -> None:
        self.assertEqual(runner.resolve_pixel_budget("low", None, None), (65536, 524288))
        self.assertEqual(
            runner.resolve_pixel_budget("high", None, None), (525312, 16777216)
        )

    def test_resolve_explicit(self) -> None:
        self.assertEqual(runner.resolve_pixel_budget("low", 100, 200), (100, 200))

    def test_resolve_partial_raises(self) -> None:
        with self.assertRaises(runner.CounterfactualRunnerError):
            runner.resolve_pixel_budget("low", 100, None)

    def test_resolve_invalid_range_raises(self) -> None:
        with self.assertRaises(runner.CounterfactualRunnerError):
            runner.resolve_pixel_budget("low", 200, 100)


class TestGateClearance(unittest.TestCase):
    def _write(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_missing_report_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(runner.CounterfactualRunnerError):
                runner.load_gate_clearance(Path(tmp) / "missing.json")

    def test_non_authoritative_report_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            self._write(
                path,
                {
                    "mode": "analytic",
                    "authoritative": False,
                    "plan": {"resolved": True},
                    "final_manifest": {"path": "x", "sha256": "y"},
                },
            )
            with self.assertRaises(runner.CounterfactualRunnerError):
                runner.load_gate_clearance(path)

    def test_unresolved_report_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            self._write(
                path,
                {
                    "mode": "processor",
                    "authoritative": True,
                    "plan": {"resolved": False},
                    "final_manifest": {"path": "x", "sha256": "y"},
                },
            )
            with self.assertRaises(runner.CounterfactualRunnerError):
                runner.load_gate_clearance(path)

    def test_valid_report_returns_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            self._write(
                path,
                {
                    "mode": "processor",
                    "authoritative": True,
                    "plan": {"resolved": True},
                    "final_manifest": {"path": "f.jsonl", "sha256": "a" * 64},
                    "gate_threshold": 512,
                },
            )
            clearance = runner.load_gate_clearance(path)
            self.assertEqual(clearance["final_manifest_sha256"], "a" * 64)
            self.assertEqual(clearance["gate_threshold"], 512)


class TestResumeContract(unittest.TestCase):
    def test_matching_rows_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "1",
                        "pixel_budget_label": "low",
                        "min_pixels": 65536,
                        "max_pixels": 524288,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runner.validate_pixel_budget_resume(
                path, label="low", min_pixels=65536, max_pixels=524288
            )

    def test_mixed_rows_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "1",
                        "pixel_budget_label": "high",
                        "min_pixels": 525312,
                        "max_pixels": 16777216,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(runner.CounterfactualRunnerError):
                runner.validate_pixel_budget_resume(
                    path, label="low", min_pixels=65536, max_pixels=524288
                )

    def test_missing_file_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runner.validate_pixel_budget_resume(
                Path(tmp) / "missing.jsonl",
                label="low",
                min_pixels=65536,
                max_pixels=524288,
            )


class TestParser(unittest.TestCase):
    def test_parser_defaults(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--model",
                "m",
                "--model-label",
                "label",
                "--manifest",
                "manifest.jsonl",
                "--output",
                "out.jsonl",
                "--arm",
                "corrected_l2_integer",
                "--pixel-budget",
                "low",
            ]
        )
        self.assertEqual(args.budget, 96)
        self.assertEqual(args.region_grid, 3)
        self.assertEqual(args.local_candidate_cap, 16)
        self.assertEqual(args.dtype, "bf16")

    def test_parser_requires_pixel_budget(self) -> None:
        import contextlib
        import io

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.build_parser().parse_args(
                    [
                        "--model",
                        "m",
                        "--model-label",
                        "label",
                        "--manifest",
                        "manifest.jsonl",
                        "--output",
                        "out.jsonl",
                        "--arm",
                        "corrected_l2_integer",
                    ]
                )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
