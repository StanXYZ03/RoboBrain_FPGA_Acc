"""CPU-only tests for the challenge/control pilot24 v2 selection contract.

Covers deterministic per-cell / per-stratum selection, the constrained Omni3D
sub-dataset coverage path, the outcome-leakage declarations, the exclusion
union and the fail-closed behaviour.  No model weights, no CUDA, no network.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.b1_answer_margin_oracle_challenge_control import (
    CELLS,
    CHALLENGE,
    CONTROL,
    OMNI3D_SOURCE,
    PER_CELL,
    PER_STRATUM,
    PILOT_SAMPLE_COUNT,
    STRATA,
    OracleChallengeControlError,
    cell_counts,
    load_exclusions,
    omni3d_coverage_ok,
    rank_key,
)
from evaluation.select_b1_answer_margin_oracle_challenge_control_pilot24 import (
    SelectionError,
    _pin,
    build_config,
    official_correct,
    select_pilot_cells,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _simple_pool(cell, *, count: int = 3) -> dict:
    pool = {CHALLENGE: [], CONTROL: []}
    for stratum in STRATA:
        for index in range(count):
            pool[stratum].append({"sample_id": f"{cell.label()}|{stratum}|{index}"})
    return pool


def _omni_bucket(prefix: str, subdatasets_by_position: list[str]) -> list[dict]:
    """Assign sub-datasets to the *rank-ordered* candidate ids on purpose."""

    ids = sorted(
        (f"{prefix}-{index}" for index in range(len(subdatasets_by_position))),
        key=rank_key,
    )
    return [
        {"sample_id": sample_id, "source_dataset": subdataset}
        for sample_id, subdataset in zip(ids, subdatasets_by_position)
    ]


def _all_cells_pool() -> dict:
    pools: dict = {}
    for cell in CELLS:
        if cell.source != OMNI3D_SOURCE:
            pools[cell.key] = _simple_pool(cell)
    return pools


class SelectPilotCellsTest(unittest.TestCase):
    def test_selects_two_plus_two_per_cell(self) -> None:
        pools = _all_cells_pool()
        pools[("Omni3D", "Depth")] = {
            CHALLENGE: _omni_bucket("dc", ["Omni3D_Hypersim"] * 2),
            CONTROL: _omni_bucket("dk", ["Omni3D_SUNRGBD"] * 2),
        }
        pools[("Omni3D", "Distance")] = {
            CHALLENGE: _omni_bucket("tc", ["Omni3D_nuScenes"] * 2),
            CONTROL: _omni_bucket("tk", ["Omni3D_nuScenes"] * 2),
        }
        selected, metadata = select_pilot_cells(pools)
        self.assertEqual(
            sum(len(selected[cell.key][s]) for cell in CELLS for s in STRATA),
            PILOT_SAMPLE_COUNT,
        )
        for cell in CELLS:
            self.assertEqual(len(selected[cell.key][CHALLENGE]), PER_STRATUM)
            self.assertEqual(len(selected[cell.key][CONTROL]), PER_STRATUM)
        self.assertEqual(metadata["strategy"], "naive_first_two")

    def test_takes_lowest_sha256_rank_first(self) -> None:
        pools = _all_cells_pool()
        pools[("Omni3D", "Depth")] = {
            CHALLENGE: _omni_bucket("dc", ["Omni3D_Hypersim"] * 2),
            CONTROL: _omni_bucket("dk", ["Omni3D_SUNRGBD"] * 2),
        }
        pools[("Omni3D", "Distance")] = {
            CHALLENGE: _omni_bucket("tc", ["Omni3D_nuScenes"] * 2),
            CONTROL: _omni_bucket("tk", ["Omni3D_nuScenes"] * 2),
        }
        selected, _ = select_pilot_cells(pools)
        chosen = {candidate["sample_id"] for candidate in selected[CELLS[0].key][CHALLENGE]}
        expected = {
            sorted(
                (row["sample_id"] for row in pools[CELLS[0].key][CHALLENGE]),
                key=rank_key,
            )[index]
            for index in range(PER_STRATUM)
        }
        self.assertEqual(chosen, expected)

    def test_order_independent(self) -> None:
        pools = _all_cells_pool()
        pools[("Omni3D", "Depth")] = {
            CHALLENGE: _omni_bucket("dc", ["Omni3D_Hypersim"] * 2),
            CONTROL: _omni_bucket("dk", ["Omni3D_SUNRGBD"] * 2),
        }
        pools[("Omni3D", "Distance")] = {
            CHALLENGE: _omni_bucket("tc", ["Omni3D_nuScenes"] * 2),
            CONTROL: _omni_bucket("tk", ["Omni3D_nuScenes"] * 2),
        }
        reversed_pools = {
            key: {stratum: list(reversed(value)) for stratum, value in entry.items()}
            for key, entry in reversed(list(pools.items()))
        }
        first, _ = select_pilot_cells(pools)
        second, _ = select_pilot_cells(reversed_pools)
        for cell in CELLS:
            for stratum in STRATA:
                self.assertEqual(
                    [c["sample_id"] for c in first[cell.key][stratum]],
                    [c["sample_id"] for c in second[cell.key][stratum]],
                )

    def test_constrained_path_when_naive_coverage_fails(self) -> None:
        pools = _all_cells_pool()
        pools[("Omni3D", "Depth")] = {
            CHALLENGE: _omni_bucket(
                "dc", ["Omni3D_Hypersim", "Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_SUNRGBD"]
            ),
            CONTROL: _omni_bucket(
                "dk", ["Omni3D_Hypersim", "Omni3D_Hypersim", "Omni3D_SUNRGBD", "Omni3D_SUNRGBD"]
            ),
        }
        pools[("Omni3D", "Distance")] = {
            CHALLENGE: _omni_bucket(
                "tc", ["Omni3D_Hypersim", "Omni3D_Hypersim", "Omni3D_nuScenes", "Omni3D_nuScenes"]
            ),
            CONTROL: _omni_bucket(
                "tk", ["Omni3D_Hypersim", "Omni3D_Hypersim", "Omni3D_nuScenes", "Omni3D_nuScenes"]
            ),
        }
        selected, metadata = select_pilot_cells(pools)
        self.assertEqual(metadata["strategy"], "constrained_lexicographic_rank_tuple")
        omni = [
            candidate
            for cell in CELLS
            if cell.source == OMNI3D_SOURCE
            for stratum in STRATA
            for candidate in selected[cell.key][stratum]
        ]
        self.assertEqual(len(omni), 8)
        self.assertTrue(omni3d_coverage_ok(omni))

    def test_short_stratum_fails_closed(self) -> None:
        pools = _all_cells_pool()
        pools[("Omni3D", "Depth")] = {
            CHALLENGE: _omni_bucket("dc", ["Omni3D_Hypersim"] * 2),
            CONTROL: _omni_bucket("dk", ["Omni3D_SUNRGBD"] * 2),
        }
        pools[("Omni3D", "Distance")] = {
            CHALLENGE: _omni_bucket("tc", ["Omni3D_nuScenes"] * 2),
            CONTROL: _omni_bucket("tk", ["Omni3D_nuScenes"] * 2),
        }
        pools[CELLS[0].key][CHALLENGE] = pools[CELLS[0].key][CHALLENGE][:1]
        with self.assertRaises(OracleChallengeControlError):
            select_pilot_cells(pools)


class OfficialCorrectTest(unittest.TestCase):
    def test_official_parse_and_truth(self) -> None:
        manifest_row = {"choices": ["0", "1", "2", "3"], "answer": "(A)"}
        correct, parsed = official_correct(manifest_row, {"output_text": "(A)"})
        self.assertTrue(correct)
        self.assertEqual(parsed, "A")
        wrong, parsed = official_correct(manifest_row, {"output_text": "(B)"})
        self.assertFalse(wrong)
        self.assertEqual(parsed, "B")
        unparsed, parsed = official_correct(manifest_row, {"output_text": "no idea"})
        self.assertFalse(unparsed)
        self.assertIsNone(parsed)


class ExclusionAndPinTest(unittest.TestCase):
    def test_real_exclusion_union(self) -> None:
        exclusions = load_exclusions(WORKSPACE_ROOT)
        self.assertIn("1", exclusions["all"])
        self.assertGreaterEqual(exclusions["count"], 223)
        self.assertEqual(len(exclusions["random_v1_pilot24"]), 24)

    def test_pin_missing_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(SelectionError):
                _pin(root / "missing.json", root=root)


class BuildConfigTest(unittest.TestCase):
    def _selected(self) -> list[dict]:
        records = []
        for cell in CELLS:
            for stratum in STRATA:
                for index in range(PER_STRATUM):
                    records.append(
                        {
                            "sample_id": f"{cell.label()}|{stratum}|{index}",
                            "oracle_pilot": {
                                "cell": cell.label(),
                                "stratum": stratum,
                            },
                        }
                    )
        return records

    def test_config_declares_outcome_leakage_and_gate(self) -> None:
        config = build_config(
            selected_records=self._selected(),
            selection_metadata={
                "criteria": ["x"],
                "per_cell": PER_CELL,
                "per_stratum": PER_STRATUM,
                "reads_correctness": True,
            },
            exclusions={"all": ["1"], "count": 1},
            manifest_sha256="0" * 64,
            manifest_pin={"path": "datasets/synthetic.jsonl", "sha256": "0" * 64},
        )
        self.assertTrue(config["outcome_enriched_diagnostic"])
        self.assertTrue(config["outcome_label_leakage"])
        self.assertTrue(config["non_deployable"])
        self.assertTrue(config["non_representative"])
        self.assertTrue(config["forbidden_for_official_absolute_accuracy"])
        self.assertTrue(config["ground_truth_answer_leakage"])
        self.assertFalse(config["deployable_selector"])
        self.assertEqual(config["gpu_runs"], 0)
        self.assertFalse(config["pilot_gate"]["uses_confidence_interval"])
        self.assertTrue(config["pilot_gate"]["dense_is_only_eligibility_reference"])
        self.assertEqual(config["pilot_gate"]["challenge_repair_min"], 8)
        self.assertEqual(config["manifest"]["cell_counts"][CELLS[0].label()], 4)
        self.assertEqual(config["manifest"]["outcome_counts"][CHALLENGE], 12)
        self.assertIn("selection_script", config["inputs_sha256"])
        self.assertIn("gpu_runner", config["inputs_sha256"])
        self.assertIn("pilot_manifest", config["inputs_sha256"])

    def test_config_pins_existing_files(self) -> None:
        config = build_config(
            selected_records=self._selected(),
            selection_metadata={"criteria": ["x"]},
            exclusions={"all": ["1"], "count": 1},
            manifest_sha256="a" * 64,
            manifest_pin={"path": "datasets/synthetic.jsonl", "sha256": "a" * 64},
        )
        for name, spec in config["inputs_sha256"].items():
            if name == "pilot_manifest":
                continue
            self.assertEqual(len(spec["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
