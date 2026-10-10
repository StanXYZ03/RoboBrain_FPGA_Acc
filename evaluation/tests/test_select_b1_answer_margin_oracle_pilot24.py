"""CPU-only tests for the 24-sample oracle-pilot selection contract.

These tests never load model weights, never touch CUDA and never read any
dense/L2 correctness.  They cover deterministic per-source stratified selection,
the frozen exclusion set, the label-branch eligibility flag, failure when a
source has fewer than eight eligible candidates, duplicate detection and the
"selection ignores correctness" property.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.b1_answer_margin_oracle import (
    PILOT_PER_SOURCE,
    SOURCE_DATASETS,
    OraclePilotError,
    stable_sha256_rank,
)
from evaluation.select_b1_answer_margin_oracle_pilot24 import (
    SelectionError,
    _pin,
    build_config,
    load_exclusions,
    select_from_candidates,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def make_candidate(
    sample_id: str,
    source: str,
    g: int = 256,
    branch_ok: bool = True,
    **extra,
) -> dict:
    candidate = {
        "sample_id": str(sample_id),
        "source_dataset": source,
        "full_llm_visual_tokens": int(g),
        "label_branch_ok": bool(branch_ok),
    }
    candidate.update(extra)
    return candidate


def full_pool(per_source_extra: int = 12) -> list[dict]:
    candidates: list[dict] = []
    for source_index, source in enumerate(SOURCE_DATASETS):
        for offset in range(per_source_extra):
            candidates.append(
                make_candidate(f"{source_index}-{offset}", source, g=256 + offset)
            )
    return candidates


class SelectFromCandidatesTest(unittest.TestCase):
    def test_selects_exactly_eight_per_source(self) -> None:
        selected, metadata = select_from_candidates(
            full_pool(), exclusions=set()
        )
        self.assertEqual(len(selected), len(SOURCE_DATASETS) * PILOT_PER_SOURCE)
        counts = {}
        for row in selected:
            counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
        self.assertEqual(set(counts), set(SOURCE_DATASETS))
        self.assertTrue(all(value == PILOT_PER_SOURCE for value in counts.values()))
        self.assertEqual(metadata["reads_correctness"], False)

    def test_deterministic_and_order_independent(self) -> None:
        pool = full_pool()
        first, _ = select_from_candidates(pool, exclusions=set())
        second, _ = select_from_candidates(list(reversed(pool)), exclusions=set())
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )

    def test_takes_lowest_sha256_rank_first(self) -> None:
        pool = full_pool(per_source_extra=12)
        selected, _ = select_from_candidates(pool, exclusions=set())
        for source in SOURCE_DATASETS:
            ranks = sorted(
                stable_sha256_rank(row["sample_id"])
                for row in pool
                if row["source_dataset"] == source
            )
            chosen = sorted(
                stable_sha256_rank(row["sample_id"])
                for row in selected
                if row["source_dataset"] == source
            )
            self.assertEqual(chosen, ranks[:PILOT_PER_SOURCE])

    def test_ignores_correctness_field(self) -> None:
        """The selection has no correctness input: adding one changes nothing."""

        pool_a = [dict(row) for row in full_pool()]
        pool_b = [dict(row, correct=index % 2 == 0) for index, row in enumerate(full_pool())]
        selected_a, _ = select_from_candidates(pool_a, exclusions=set())
        selected_b, _ = select_from_candidates(pool_b, exclusions=set())
        self.assertEqual(
            [row["sample_id"] for row in selected_a],
            [row["sample_id"] for row in selected_b],
        )

    def test_exclusions_are_honoured(self) -> None:
        pool = full_pool()
        excluded = {row["sample_id"] for row in pool if row["source_dataset"] == SOURCE_DATASETS[0]}
        excluded = set(sorted(excluded)[:4])
        selected, _ = select_from_candidates(pool, exclusions=excluded)
        chosen = {row["sample_id"] for row in selected}
        self.assertFalse(chosen & excluded)

    def test_g_le_96_and_branch_flag_filter(self) -> None:
        pool = full_pool()
        pool[0]["full_llm_visual_tokens"] = 96
        pool[1]["label_branch_ok"] = False
        selected, _ = select_from_candidates(pool, exclusions=set())
        chosen = {row["sample_id"] for row in selected}
        self.assertNotIn(pool[0]["sample_id"], chosen)
        self.assertNotIn(pool[1]["sample_id"], chosen)

    def test_fails_closed_when_a_source_is_short(self) -> None:
        pool = [
            row
            for row in full_pool()
            if not (
                row["source_dataset"] == SOURCE_DATASETS[0]
                and row["sample_id"].endswith(("7", "8", "9", "10", "11"))
            )
        ]
        with self.assertRaises(OraclePilotError):
            select_from_candidates(pool, exclusions=set())

    def test_duplicate_candidate_id_fails_closed(self) -> None:
        pool = full_pool()
        pool.append(dict(pool[0]))
        with self.assertRaises(OraclePilotError):
            select_from_candidates(pool, exclusions=set())

    def test_missing_field_fails_closed(self) -> None:
        pool = full_pool()
        del pool[0]["label_branch_ok"]
        with self.assertRaises(OraclePilotError):
            select_from_candidates(pool, exclusions=set())


class ExclusionContractTest(unittest.TestCase):
    def test_real_exclusion_union(self) -> None:
        exclusions = load_exclusions(WORKSPACE_ROOT)
        self.assertIn("1", exclusions["all"])
        self.assertEqual(len(exclusions["screening100"]), 100)
        self.assertEqual(len(exclusions["counterfactual100"]), 100)
        self.assertEqual(exclusions["count"], len(exclusions["all"]))


class PinContractTest(unittest.TestCase):
    def test_pin_hashes_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("hello", encoding="utf-8")
            pin = _pin(target, root=root)
            self.assertEqual(pin["path"], "a.txt")
            self.assertEqual(len(pin["sha256"]), 64)

    def test_pin_missing_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SelectionError):
                _pin(root / "missing.txt", root=root)


class BuildConfigTest(unittest.TestCase):
    def test_config_carries_leakage_and_gate_contract(self) -> None:
        selected = [
            {"sample_id": str(index), "source_dataset": source}
            for source in SOURCE_DATASETS
            for index in range(8)
        ]
        # build_config only needs specific keys; fill them for the manifest view.
        records = [
            {"sample_id": row["sample_id"], "source_dataset": row["source_dataset"]}
            for row in selected
        ]
        config = build_config(
            selected_records=records,
            selection_metadata={
                "criteria": ["x"],
                "per_source": 8,
                "source_datasets": list(SOURCE_DATASETS),
                "reads_correctness": False,
            },
            exclusions={"all": ["1"], "count": 1},
            manifest_sha256="0" * 64,
        )
        self.assertTrue(config["ground_truth_answer_leakage"])
        self.assertFalse(config["deployable_selector"])
        self.assertFalse(config["gpu_executed"])
        self.assertEqual(config["pilot_gate"]["on_pass"], "GO")
        self.assertFalse(config["pilot_gate"]["uses_confidence_interval"])
        self.assertIn("gpu_runner", config["inputs_sha256"])


if __name__ == "__main__":
    unittest.main()
