#!/usr/bin/env python3
"""CPU-only contract tests for the residual-reranker GPU runner.

The runner is only executed on a GPU host; these tests pin the deterministic
six-cell smoke selection, the frozen CLI constants, the extra fail-closed resume
checks, the audit-field mapping and the no-write-to-full2638 rule.  No CUDA or
model is touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import run_hf_conservative_l2_residual_rerank as runner  # noqa: E402
from evaluation.conservative_l2_residual_reranker import (  # noqa: E402
    BUDGET,
    MAX_SWAPS,
    MECHANISM,
    select_conservative_residual_rerank,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    corrected_patch_l2_integer,
)


def _samples() -> list[dict]:
    rows = []
    for source, task in runner.SMOKE_CELLS:
        for index in range(3):
            sample_id = f"{source}-{task}-{index}"
            rows.append({"sample_id": sample_id, "source": source, "task": task})
    rows.append({"sample_id": "other", "source": "ADE20K", "task": "Depth"})
    return rows


def _hash_order(sample: dict) -> str:
    return hashlib.sha256(str(sample["sample_id"]).encode("utf-8")).hexdigest()


class SmokeSelectionTest(unittest.TestCase):
    def test_six_cells_deterministic(self) -> None:
        samples = _samples()
        chosen = runner.select_smoke_samples(samples)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(
            {(row["source"], row["task"]) for row in chosen}, set(runner.SMOKE_CELLS)
        )
        expected = set()
        for source, task in runner.SMOKE_CELLS:
            cell = [
                row
                for row in samples
                if row["source"] == source and row["task"] == task
            ]
            expected.add(min(cell, key=_hash_order)["sample_id"])
        self.assertEqual({row["sample_id"] for row in chosen}, expected)
        self.assertEqual(
            [row["sample_id"] for row in chosen],
            [row["sample_id"] for row in runner.select_smoke_samples(samples)],
        )

    def test_full_output_path_is_pinned(self) -> None:
        self.assertEqual(
            runner.FULL2638_OUTPUT.name,
            "conservative_l2_residual_rerank_v1_k96_full2638.jsonl",
        )
        self.assertIn(
            "2026-10-05-conservative-l2-residual-reranker-full2638",
            str(runner.FULL2638_OUTPUT),
        )

    def test_cli_defaults_are_frozen(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--model",
                "model",
                "--model-label",
                "label",
                "--manifest",
                "manifest.jsonl",
                "--output",
                "out.jsonl",
            ]
        )
        self.assertEqual(args.budget, BUDGET)
        self.assertEqual(args.max_swaps, MAX_SWAPS)
        self.assertFalse(args.smoke)


class ResumeExtraTest(unittest.TestCase):
    def _record(self, **overrides) -> dict:
        record = {
            "sample_id": "s0",
            "arm": MECHANISM,
            "mechanism": MECHANISM,
            "run_mode": "full2638",
            "residual_formula": "semantic_norm_q15 - corrected_l2_norm_q15",
            "max_swaps": MAX_SWAPS,
            "swap_count": 0,
            "context_pooling_basis": "corrected_integer_l2_norm_q15",
            "baseline_anchors": [],
            "removed_group_indices": [],
            "added_group_indices": [],
            "swap_region_ids": [],
            "swap_gain_q15": [],
            "per_region_selected_counts_before": [],
            "per_region_selected_counts_after": [],
            "uses_labels": False,
            "uses_answer_tokens": False,
            "uses_gradients": False,
            "uses_floating_lambda": False,
            "uses_learned_new_weights": False,
            "uses_task_source_ids": False,
        }
        record.update(overrides)
        return record

    def _write(self, path: Path, record: dict) -> None:
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    def test_accepts_valid_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            self._write(path, self._record())
            self.assertEqual(
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS),
                {"s0"},
            )

    def test_missing_or_empty_file_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            self.assertEqual(
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS), set()
            )
            path.write_text("", encoding="utf-8")
            self.assertEqual(
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS), set()
            )

    def test_missing_field_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            record = self._record()
            del record["swap_count"]
            self._write(path, record)
            with self.assertRaises(runner.ResidualResumeContractError):
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS)

    def test_forbidden_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            self._write(path, self._record(uses_labels=True))
            with self.assertRaises(runner.ResidualResumeContractError):
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS)

    def test_max_swaps_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            self._write(path, self._record(max_swaps=8))
            with self.assertRaises(runner.ResidualResumeContractError):
                runner.validate_residual_resume_extra(path, max_swaps=MAX_SWAPS)


class AuditFieldsTest(unittest.TestCase):
    def test_bypass_payload_maps_to_zeros(self) -> None:
        hidden = np.zeros((64, 4, 1024), dtype=np.float64)
        payload = select_conservative_residual_rerank(
            hidden,
            8,
            8,
            semantic_norm_q15=np.zeros(64, dtype=np.int64),
            l2_norm_q15=np.zeros(64, dtype=np.int64),
        )
        audit = runner._residual_audit_fields(payload)
        self.assertEqual(audit["mechanism"], MECHANISM)
        self.assertEqual(audit["swap_count"], 0)
        self.assertIsNone(audit["residual_q15_sha256"])
        self.assertEqual(audit["baseline_anchors"], [])

    def test_compressed_payload_hashes_and_lists(self) -> None:
        rng = np.random.default_rng(21)
        grid_h, grid_w = 15, 15
        hidden = rng.normal(
            size=(grid_h * grid_w, 4, 1024)
        ) * rng.normal(size=(grid_h * grid_w, 1, 1))
        l2 = corrected_patch_l2_integer(hidden)["l2_norm_q15"]
        semantic = rng.integers(0, 32768, size=grid_h * grid_w)
        payload = select_conservative_residual_rerank(
            hidden, grid_h, grid_w, semantic_norm_q15=semantic, l2_norm_q15=l2
        )
        audit = runner._residual_audit_fields(payload)
        self.assertFalse(payload["bypass"])
        self.assertEqual(audit["swap_count"], payload["swap_count"])
        self.assertEqual(
            len(audit["removed_group_indices"]), payload["swap_count"]
        )
        self.assertEqual(
            len(audit["added_group_indices"]), payload["swap_count"]
        )
        for key in (
            "residual_q15_sha256",
            "semantic_norm_q15_sha256",
            "l2_norm_q15_sha256",
            "l2_combined_sha256",
            "baseline_selected_real_sha256",
            "reranked_selected_real_sha256",
        ):
            self.assertRegex(audit[key], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
