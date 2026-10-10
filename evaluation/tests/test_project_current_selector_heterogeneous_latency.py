#!/usr/bin/env python3
"""CPU-only tests for the heterogeneous-latency projection.

Read-only: the tests hash/read the frozen latency-100 JSONL and the accepted HLS
report; they never write to the original artifacts.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import project_current_selector_heterogeneous_latency as proj  # noqa: E402


def _sample(
    sample_id: str = "s0",
    group_count: int = 80,
    grid_h: int = 8,
    grid_w: int = 10,
    inference: float = 0.150,
    software: float = 0.045,
    application: float = 0.170,
) -> dict:
    return {
        "sample_id": sample_id,
        "task": "Count",
        "source": "ADE20K",
        "group_count": group_count,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "model_inference_seconds": inference,
        "selector_software_seconds": software,
        "application_end_to_end_seconds": application,
        "repeats": 3,
    }


class AcceptedFormulaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = proj.load_accepted_cycle_model()

    def test_scorer_constants(self) -> None:
        self.assertEqual(self.model["wv_load_cycles"], 65539.0)
        self.assertEqual(self.model["per_group_cycles"], 1842.0)
        self.assertEqual(self.model["target_fmax_mhz"], 200.0)

    def test_production_cases_match_accepted_report(self) -> None:
        cases = (
            (80, 48, 8, 10, 328698.8),
            (260, 64, 13, 20, 752341.8),
            (400, 96, 20, 20, 1097801.8),
        )
        for group_count, budget, grid_h, grid_w, accepted in cases:
            scorer = self.model["wv_load_cycles"] + group_count * self.model["per_group_cycles"]
            selector = self.model["selector_cycles"](group_count, budget, grid_h, grid_w)
            overhead = self.model["compact_overhead_cycles"](budget)
            self.assertAlmostEqual(scorer + selector + overhead, accepted, places=3)

    def test_identity_bypass_selector_cycles(self) -> None:
        # Accepted report G80_K96 selector_cycles_estimate == 80800.0.
        self.assertEqual(self.model["selector_cycles"](80, 96, 8, 10), 80800.0)
        self.assertEqual(self.model["selector_cycles"](64, 96, 8, 8), 64 * 260.0 + 60000.0)

    def test_compact_overhead_budgets(self) -> None:
        self.assertEqual(self.model["compact_overhead_cycles"](48), 12355.0)
        self.assertEqual(self.model["compact_overhead_cycles"](64), 16456.0)
        self.assertEqual(self.model["compact_overhead_cycles"](96), 24674.0)


class TransferReuseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.transfer = proj.load_transfer_model()

    def test_input_arena_formula(self) -> None:
        self.assertEqual(self.transfer.input_arena_bytes(80), 80 * 5120 + 128)

    def test_fixed_arena_bytes(self) -> None:
        self.assertEqual(int(self.transfer.fpga_transfer_layout(48).total_bytes), 196928)
        self.assertEqual(int(self.transfer.fpga_transfer_layout(64).total_bytes), 262464)
        self.assertEqual(int(self.transfer.fpga_transfer_layout(96).total_bytes), 393792)


class ProjectionMathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = proj.load_accepted_cycle_model()
        cls.transfer = proj.load_transfer_model()

    def _project(self, sample, budget):
        return proj.project_sample(budget, sample, self.model, self.transfer)

    def test_formula_is_serial_and_resident(self) -> None:
        sample = _sample(group_count=80, inference=0.150, software=0.045)
        row = self._project(sample, 48)
        expected = (
            0.150
            - 0.045
            + row["fpga_compute_seconds"]
            + row["transfer_1gbe_110MBps_seconds"]
        )
        self.assertAlmostEqual(row["projected_model_e2e_1gbe_110MBps_seconds"], expected)
        self.assertEqual(row["jetson_to_fpga_bytes"], 80 * 5120 + 128)
        self.assertEqual(row["fpga_to_jetson_arena_bytes"], 196928)
        self.assertFalse(row["identity_bypass"])
        self.assertEqual(row["output_group_count"], 48)

    def test_10gbe_is_faster(self) -> None:
        row = self._project(_sample(), 64)
        self.assertLess(
            row["transfer_10gbe_1p1GBps_assumed_seconds"],
            row["transfer_1gbe_110MBps_seconds"],
        )
        self.assertLess(
            row["projected_model_e2e_10gbe_1p1GBps_assumed_seconds"],
            row["projected_model_e2e_1gbe_110MBps_seconds"],
        )

    def test_identity_bypass_uses_identity_branch(self) -> None:
        row = self._project(_sample(group_count=64, grid_h=8, grid_w=8), 96)
        self.assertTrue(row["identity_bypass"])
        self.assertEqual(row["output_group_count"], 64)
        self.assertEqual(row["selector_cycles_estimate"], 64 * 260.0 + 60000.0)
        self.assertEqual(row["fpga_to_jetson_arena_bytes"], 393792)


class LoaderValidationTest(unittest.TestCase):
    def _write(self, rows):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for row in rows:
            tmp.write(json.dumps(row) + "\n")
        tmp.close()
        return Path(tmp.name)

    def _base_row(self, sample_id, repeat):
        return {
            "sample_id": sample_id,
            "repeat": repeat,
            "mode": "latency",
            "warmup": False,
            "task": "Count",
            "source": "ADE20K",
            "full_llm_visual_tokens": 80,
            "image_grid_thw": [1, 16, 20],
            "model_inference_seconds": 0.1,
            "selector_software_seconds": 0.01,
            "application_end_to_end_seconds": 0.12,
        }

    def test_rejects_wrong_record_count(self) -> None:
        path = self._write([self._base_row("0", 0)])
        with self.assertRaises(ValueError):
            proj.load_latency_samples(path)

    def test_rejects_duplicate_run(self) -> None:
        rows = [
            self._base_row(str(index), repeat)
            for index in range(100)
            for repeat in (0, 1, 2)
        ]
        rows[1] = self._base_row("0", 0)  # duplicate (sample, repeat) pair
        path = self._write(rows)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            proj.load_latency_samples(path)


class RealLatencyReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = proj.build_report()

    def test_measurement_status_and_disclaimers(self) -> None:
        self.assertEqual(self.report["measurement_status"], "not_measured")
        self.assertTrue(self.report["not_a_measurement"])
        self.assertIn("PURE ANALYSIS", self.report["scope"])
        self.assertIn("not a ZCU102", self.report["scope"])

    def test_per_budget_identity_counts(self) -> None:
        expected = {"K48": 0, "K64": 2, "K96": 2}
        for label, count in expected.items():
            budget = self.report["per_budget"][label]
            self.assertEqual(budget["sample_count"], 100, label)
            self.assertEqual(budget["identity_bypass_count"], count, label)

    def test_inputs_hashed(self) -> None:
        inputs = self.report["inputs_sha256"]
        self.assertEqual(
            set(inputs),
            {
                "accepted_hls_report",
                "accepted_csynth_model",
                "transfer_audit_module",
                "latency100_K48",
                "latency100_K64",
                "latency100_K96",
            },
        )
        for name, item in inputs.items():
            self.assertEqual(len(item["sha256"]), 64, name)

    def test_transfer_arena_fixed(self) -> None:
        self.assertEqual(
            self.report["per_budget"]["K48"]["bytes"]["fpga_to_jetson_arena_fixed"], 196928
        )
        self.assertEqual(
            self.report["per_budget"]["K64"]["bytes"]["fpga_to_jetson_arena_fixed"], 262464
        )
        self.assertEqual(
            self.report["per_budget"]["K96"]["bytes"]["fpga_to_jetson_arena_fixed"], 393792
        )

    def test_10gbe_projection_faster_than_1gbe(self) -> None:
        for label in proj.CANDIDATE_ORDER:
            budget = self.report["per_budget"][label]
            self.assertLess(
                budget["projected_model_e2e_ms"]["10gbe_1p1GBps_assumed"]["mean"],
                budget["projected_model_e2e_ms"]["1gbe_110MBps"]["mean"],
                label,
            )

    def test_markdown_warns_not_measured(self) -> None:
        md = proj.render_markdown(self.report)
        self.assertIn("纯分析估算", md)
        self.assertIn("ZCU102", md)
        self.assertIn("SHA256", md)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
