"""CPU-only tests for the B1 oracle challenge/control shared contracts.

Covers the constrained Omni3D sub-dataset coverage selection (feasibility,
lexicographic minimality, fail-closed), the cell/stratum helpers, the leakage
declarations and the frozen exclusion union.  No model weights, no CUDA.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.b1_answer_margin_oracle_challenge_control import (
    CELLS,
    CHALLENGE,
    CONTROL,
    DESIGN,
    GATE_CHALLENGE_REPAIR_MIN,
    GATE_CONTROL_REGRESSION_MAX,
    GATE_ORACLE_PARSE_MIN,
    OMNI3D_MIN_PER_SUBDATASET,
    OMNI3D_SUBDATASETS,
    PER_CELL,
    PER_STRATUM,
    PILOT_SAMPLE_COUNT,
    STRATA,
    OracleChallengeControlError,
    cell_counts,
    challenge_control_label,
    constrained_omni3d_selection,
    load_exclusions,
    omni3d_coverage,
    omni3d_coverage_ok,
    rank_key,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _candidate(order: int, subdataset: str) -> dict:
    return {"sample_id": f"id{order:02d}", "order": f"{order:02d}", "subdataset": subdataset}


def _key(candidate: dict) -> tuple[str, str]:
    return (str(candidate["order"]), str(candidate["sample_id"]))


class CellContractTest(unittest.TestCase):
    def test_six_cells_and_two_strata(self) -> None:
        self.assertEqual(len(CELLS), 6)
        self.assertEqual(PER_CELL, 4)
        self.assertEqual(PER_STRATUM, 2)
        self.assertEqual(PILOT_SAMPLE_COUNT, 24)
        self.assertEqual(set(STRATA), {CHALLENGE, CONTROL})
        labels = [cell.label() for cell in CELLS]
        self.assertEqual(len(set(labels)), 6)
        self.assertEqual(
            labels,
            [
                "ADE20K x Count",
                "ADE20K x Relation",
                "COCO x Count",
                "COCO x Relation",
                "Omni3D x Depth",
                "Omni3D x Distance",
            ],
        )
        self.assertEqual(DESIGN, "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24")

    def test_gate_constants(self) -> None:
        self.assertEqual(GATE_CHALLENGE_REPAIR_MIN, 8)
        self.assertEqual(GATE_CONTROL_REGRESSION_MAX, 1)
        self.assertEqual(GATE_ORACLE_PARSE_MIN, 24)
        self.assertEqual(OMNI3D_MIN_PER_SUBDATASET, 2)
        self.assertEqual(
            OMNI3D_SUBDATASETS,
            ("Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_nuScenes"),
        )

    def test_challenge_control_label(self) -> None:
        self.assertIsNone(challenge_control_label(False, False))
        self.assertIsNone(challenge_control_label(False, True))
        self.assertEqual(challenge_control_label(True, False), CHALLENGE)
        self.assertEqual(challenge_control_label(True, True), CONTROL)

    def test_rank_key_is_stable_and_id_tiebreak(self) -> None:
        import hashlib

        self.assertEqual(rank_key("7")[0], hashlib.sha256(b"7").hexdigest())
        self.assertEqual(rank_key("7")[1], "7")

    def test_cell_counts_and_coverage(self) -> None:
        records = [
            {"cell": "ADE20K x Count", "source_dataset": "ADE20K Validation Set"},
            {"cell": "ADE20K x Count", "source_dataset": "ADE20K Validation Set"},
            {"cell": "Omni3D x Depth", "source_dataset": "Omni3D_Hypersim"},
        ]
        self.assertEqual(
            cell_counts(records), {"ADE20K x Count": 2, "Omni3D x Depth": 1}
        )
        coverage = omni3d_coverage(records)
        self.assertEqual(coverage["Omni3D_Hypersim"], 1)
        self.assertFalse(omni3d_coverage_ok(records))
        self.assertTrue(omni3d_coverage_ok(records, min_per_subdataset=0))


class ConstrainedSelectionTest(unittest.TestCase):
    def _buckets(self) -> dict:
        return {
            "b0": [_candidate(0, "H"), _candidate(1, "H"), _candidate(2, "S"), _candidate(3, "S")],
            "b1": [_candidate(4, "H"), _candidate(5, "H"), _candidate(6, "S"), _candidate(7, "S")],
            "b2": [_candidate(8, "N"), _candidate(9, "N")],
            "b3": [_candidate(10, "N"), _candidate(11, "N")],
        }

    def test_covers_all_subdatasets_with_minimum_rank_tuple(self) -> None:
        chosen, metadata = constrained_omni3d_selection(
            self._buckets(), subdatasets=("H", "S", "N"), key=_key
        )
        selected_ids = [
            candidate["sample_id"] for bucket in chosen.values() for candidate in bucket
        ]
        self.assertEqual(
            selected_ids,
            ["id00", "id01", "id06", "id07", "id08", "id09", "id10", "id11"],
        )
        coverage = {
            sub: sum(
                1
                for bucket in chosen.values()
                for candidate in bucket
                if candidate["subdataset"] == sub
            )
            for sub in ("H", "S", "N")
        }
        self.assertTrue(all(value >= 2 for value in coverage.values()))
        self.assertEqual(metadata["coverage"], coverage)

    def test_input_order_does_not_change_the_selection(self) -> None:
        buckets = self._buckets()
        reversed_buckets = {
            key: list(reversed(value)) for key, value in reversed(list(buckets.items()))
        }
        first, _ = constrained_omni3d_selection(buckets, subdatasets=("H", "S", "N"), key=_key)
        second, _ = constrained_omni3d_selection(
            reversed_buckets, subdatasets=("H", "S", "N"), key=_key
        )
        self.assertEqual(
            [c["sample_id"] for c in first["b0"]], [c["sample_id"] for c in second["b0"]]
        )
        self.assertEqual(
            [c["sample_id"] for c in first["b1"]], [c["sample_id"] for c in second["b1"]]
        )

    def test_infeasible_coverage_fails_closed(self) -> None:
        buckets = {
            "b0": [_candidate(0, "H"), _candidate(1, "H")],
            "b1": [_candidate(2, "H"), _candidate(3, "H")],
            "b2": [_candidate(4, "H"), _candidate(5, "H")],
            "b3": [_candidate(6, "H"), _candidate(7, "H")],
        }
        with self.assertRaises(OracleChallengeControlError):
            constrained_omni3d_selection(
                buckets, subdatasets=("H", "S", "N"), key=_key
            )

    def test_unknown_subdataset_fails_closed(self) -> None:
        buckets = {
            "b0": [_candidate(0, "H"), _candidate(1, "X")],
            "b1": [_candidate(2, "H"), _candidate(3, "H")],
            "b2": [_candidate(4, "N"), _candidate(5, "N")],
            "b3": [_candidate(6, "N"), _candidate(7, "N")],
        }
        with self.assertRaises(OracleChallengeControlError):
            constrained_omni3d_selection(buckets, subdatasets=("H", "S", "N"), key=_key)


class ExclusionContractTest(unittest.TestCase):
    def test_real_exclusion_union(self) -> None:
        exclusions = load_exclusions(WORKSPACE_ROOT)
        self.assertIn("1", exclusions["all"])
        self.assertEqual(len(exclusions["screening100"]), 100)
        self.assertEqual(len(exclusions["counterfactual100"]), 100)
        self.assertEqual(len(exclusions["random_v1_pilot24"]), 24)
        self.assertEqual(exclusions["count"], len(exclusions["all"]))
        self.assertGreaterEqual(exclusions["count"], 223)
        for sample_id in exclusions["random_v1_pilot24"]:
            self.assertIn(sample_id, exclusions["all"])

    def test_missing_exclusion_manifest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(OracleChallengeControlError):
                load_exclusions(Path(tmp))


class LeakageDeclarationTest(unittest.TestCase):
    def test_module_docstring_declares_leakage(self) -> None:
        import evaluation.b1_answer_margin_oracle_challenge_control as module

        text = module.__doc__ or ""
        self.assertIn("outcome/label leakage", text)
        self.assertIn("non-deployable", text.lower().replace("non_deployable", "non-deployable"))


if __name__ == "__main__":
    unittest.main()
