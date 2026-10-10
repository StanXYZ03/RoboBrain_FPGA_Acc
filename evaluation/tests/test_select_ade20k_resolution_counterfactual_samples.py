#!/usr/bin/env python3
"""Focused tests for the frozen ADE20K resolution-counterfactual selection."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import analyze_full2638_prompt_gain_heterogeneity as het
from evaluation import select_ade20k_resolution_counterfactual_samples as selection


def candidate(sid: str, gate: str, task: str, correct: bool, rank: int = 0) -> dict:
    return {
        "sample_id": str(sid),
        "task": task,
        "source": "ADE20K",
        "source_dataset": selection.ADE_SOURCE_DATASET,
        "G": 600 if gate == "G>512" else 100,
        "integer_correct": correct,
    }


class TestStratification(unittest.TestCase):
    def test_stratum_order_has_eight_entries(self) -> None:
        order = selection.stratum_order()
        self.assertEqual(len(order), 8)
        self.assertEqual(len(set(order)), 8)
        self.assertIn("G<=512::Count::integer_correct", order)
        self.assertIn("G>512::Relation::integer_wrong", order)

    def test_stratum_key(self) -> None:
        row = candidate("1", "G>512", "Relation", False)
        self.assertEqual(selection.stratum_key(row), "G>512::Relation::integer_wrong")


class TestAllocation(unittest.TestCase):
    def test_allocation_sums_to_target(self) -> None:
        order = selection.stratum_order()
        counts = {label: 100 for label in order}
        allocation = selection.largest_remainder_allocation(
            counts, total=100, minimum=4, order=order
        )
        self.assertEqual(sum(allocation.values()), 100)
        self.assertTrue(all(value >= 4 for value in allocation.values()))

    def test_allocation_respects_availability(self) -> None:
        order = selection.stratum_order()
        counts = {label: 100 for label in order}
        counts[order[0]] = 2
        allocation = selection.largest_remainder_allocation(
            counts, total=100, minimum=4, order=order
        )
        self.assertEqual(allocation[order[0]], 2)
        self.assertEqual(sum(allocation.values()), 100)

    def test_minimum_above_target_raises(self) -> None:
        order = selection.stratum_order()
        counts = {label: 100 for label in order}
        with self.assertRaises(selection.SelectionError):
            selection.largest_remainder_allocation(
                counts, total=10, minimum=4, order=order
            )


class TestSelect(unittest.TestCase):
    def _candidates(self) -> list[dict]:
        rows: list[dict] = []
        sid = 0
        for label in selection.stratum_order():
            gate, task, correctness = label.split("::")
            for _ in range(20):
                rows.append(
                    candidate(
                        str(sid),
                        gate,
                        task,
                        correctness == "integer_correct",
                    )
                )
                sid += 1
        return rows

    def test_select_is_deterministic_and_disjoint(self) -> None:
        rows = self._candidates()
        first = selection.select(rows)
        second = selection.select(rows)
        self.assertEqual(
            [row["sample_id"] for row in first[0]],
            [row["sample_id"] for row in second[0]],
        )
        self.assertEqual(len(first[0]), selection.TARGET_TOTAL)
        selected_ids = {row["sample_id"] for row in first[0]}
        reserve_ids = {row["sample_id"] for row in first[1]}
        self.assertFalse(selected_ids & reserve_ids)
        self.assertEqual(len(selected_ids), 100)

    def test_select_covers_all_strata(self) -> None:
        selected, _reserve, allocation = selection.select(self._candidates())
        self.assertEqual(
            {row["stratum"] for row in selected}, set(selection.stratum_order())
        )
        self.assertEqual(allocation["target_total"], 100)

    def test_manifest_record_shape(self) -> None:
        selected, _reserve, _allocation = selection.select(self._candidates())
        meta = {
            "sample_id": selected[0]["sample_id"],
            "task": selected[0]["task"],
            "source": "ADE20K",
            "source_dataset": selection.ADE_SOURCE_DATASET,
            "choices": ["a"],
            "answer": "(A)",
        }
        record = selection._manifest_record(meta, selected[0], reserve=False)
        self.assertEqual(record["sample_id"], meta["sample_id"])
        self.assertEqual(
            record["resolution_counterfactual"]["design"], selection.DESIGN
        )
        self.assertEqual(record["resolution_counterfactual"]["split"], "selected")


class TestFrozenArtifacts(unittest.TestCase):
    def test_on_disk_manifest_matches_config(self) -> None:
        report = json.loads(
            selection.SELECTION_REPORT_JSON.read_text(encoding="utf-8")
        )
        self.assertEqual(report["selected_count"], 100)
        self.assertEqual(report["reserve_count"], 533)
        self.assertEqual(report["gate_report"]["conclusion"], "SOURCE_CONFOUNDING_NOT_RESOLVED")
        self.assertTrue(all(check["ok"] for check in report["contract_checks"]))
        config = json.loads(
            (
                selection.ROOT
                / "configs"
                / "2026-10-04-same-image-resolution-counterfactual.preregistered.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(report["manifest"]["sha256"], config["manifest"]["sha256"])
        self.assertEqual(report["reserve"]["sha256"], config["reserve"]["sha256"])

    def test_selection_reproduces_frozen_ids(self) -> None:
        manifest_index = het.index_unique(
            het.load_rows(het.FULL_MANIFEST, "full_manifest"), "full_manifest"
        )
        arm_indices = {
            het.ARM_INTEGER: het.index_unique(
                het.load_rows(het.ARM_INTEGER_JSONL, "integer"), "integer"
            ),
            het.ARM_PROMPT: het.index_unique(
                het.load_rows(het.ARM_PROMPT_JSONL, "prompt"), "prompt"
            ),
        }
        candidates = selection.build_candidates(manifest_index, arm_indices)
        selected, _reserve, _allocation = selection.select(candidates)
        frozen = [
            json.loads(line)
            for line in selection.MANIFEST_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            sorted(row["sample_id"] for row in selected),
            sorted(row["sample_id"] for row in frozen),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
