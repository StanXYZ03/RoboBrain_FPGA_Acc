#!/usr/bin/env python3
"""Focused tests for the fused integer Selector pipeline HLS module.

They check, without any model/GPU/network:

* the fixture ledger is internally consistent (SHA256 / FNV hashes, coverage);
* every stored expected file can be re-derived from the **frozen goldens**
  (``integer_chain`` + ``select_fixedpoint``) over the stored scene bytes;
* the production top has no ``score_q`` data port (source-level contract) and
  the intermediate is a local on-chip buffer;
* the wrapper does not introduce a second copy of the arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
MODULE_DIR = TEST_DIR.parent
ROOT = MODULE_DIR.parents[1]
for path in (str(ROOT), str(MODULE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.hardware_regular_selector_fixedpoint import select_fixedpoint  # noqa: E402
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    describe_fpga_layout,
    fpga_transfer_layout,
)
from evaluation.selector_integer_scorer_audit import integer_chain  # noqa: E402
from tools import generate_fixtures as gf  # noqa: E402
from tools.transfer_audit import build_transfer_audit, image_bytes  # noqa: E402

FIXTURE_DIR = (
    ROOT / "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls/fixtures"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scene(path: Path) -> gf.Scene:
    data = path.read_bytes()
    assert data[:8] == b"ISPSCN01"
    group_count, grid_h, grid_w, _flags = struct.unpack_from("<IIII", data, 8)
    payload = struct.unpack_from("<Q", data, 48)[0]
    assert payload == len(data) - 64
    cursor = 64
    token_bytes = group_count * 4096
    gm_bytes = group_count * 1024
    tokens = np.frombuffer(data, dtype=np.int8, count=token_bytes, offset=cursor).copy()
    cursor += token_bytes
    gm = np.frombuffer(data, dtype=np.int8, count=gm_bytes, offset=cursor).copy()
    cursor += gm_bytes
    query = np.frombuffer(data, dtype=np.int8, count=128, offset=cursor).copy()
    return gf.Scene(
        name=path.stem,
        group_count=group_count,
        grid_h=grid_h,
        grid_w=grid_w,
        tokens_q=tokens.reshape(group_count, 4, 1024),
        group_mean_q=gm.reshape(group_count, 1024),
        query_q=query,
    )


class FixtureLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
        cls.wv = gf.load_wv_int8()
        gf.WV_CACHE["wv_q"] = cls.wv.astype(np.int64)

    def test_wv_copy_is_accepted_bundle(self) -> None:
        self.assertEqual(self.manifest["wv"]["source_sha256"], gf.WV_SHA256)
        self.assertEqual(
            _sha256(FIXTURE_DIR / "wv_q.bin"), self.manifest["wv"]["copied_sha256"]
        )

    def test_scene_and_case_hashes(self) -> None:
        for record in self.manifest["scenes"].values():
            self.assertEqual(_sha256(FIXTURE_DIR / record["file"]), record["sha256"])
        for record in self.manifest["cases"].values():
            self.assertEqual(_sha256(FIXTURE_DIR / record["file"]), record["sha256"])

    def test_coverage_requirements(self) -> None:
        coverage = self.manifest["coverage"]
        self.assertEqual(set(coverage["budgets"]), {48, 64, 96})
        for group_count in (80, 260, 400):
            self.assertIn(group_count, coverage["group_counts"])
        # grounding + pointing correct prompts present.
        scene_names = set(self.manifest["scenes"])
        self.assertIn("real_grounding_correct", scene_names)
        self.assertIn("real_pointing_correct", scene_names)
        # identity bypass / empty-region / flat-score cases present.
        case_names = " ".join(self.manifest["cases"])
        self.assertIn("identity_g16", case_names)
        self.assertIn("empty_2x25", case_names)
        self.assertIn("zero_g128", case_names)
        self.assertIn("all_equal_g128", case_names)

    def test_every_case_rederives_from_frozen_goldens(self) -> None:
        for name, record in self.manifest["cases"].items():
            scene = _load_scene(FIXTURE_DIR / f"scene_{record['scene']}.bin")
            case = gf.Case(record["scene"], record["budget"], record["local_cap"],
                           record["scale"])
            data, meta = gf.expected_bytes(case, scene)
            stored = (FIXTURE_DIR / record["file"]).read_bytes()
            self.assertEqual(data, stored, name)
            self.assertEqual(meta["status"], record["status"], name)
            self.assertEqual(meta["output_count"], record["output_count"], name)
            self.assertEqual(meta["context_count"], record["context_count"], name)


class InterfaceContractTest(unittest.TestCase):
    def test_top_has_no_score_q_port(self) -> None:
        header = (MODULE_DIR / "include/integer_selector_pipeline.h").read_text()
        # The production top declaration must not mention score_q.
        start = header.index("int integer_selector_pipeline(")
        end = header.index(");", start)
        top = header[start:end]
        self.assertNotIn("score", top.lower())
        # The debug twin may expose it (host only), guarded by __SYNTHESIS__.
        self.assertIn("integer_selector_pipeline_debug", header)
        self.assertIn("#ifndef __SYNTHESIS__", header)

    def test_intermediate_is_local_and_never_an_interface(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline.cpp").read_text()
        self.assertIn("uint16_t score_local[IS_MAX_GROUPS];", source)
        # No interface pragma may bind a score buffer.
        for line in source.splitlines():
            if "HLS INTERFACE" in line:
                self.assertNotIn("score", line.lower())
        # The intermediate must be handed to both cores.
        self.assertIn("is_scorer_core<", source)
        self.assertIn("token_selector_core_packed(", source)

    def test_wrapper_does_not_duplicate_upstream_maths(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline.cpp").read_text()
        # The two accepted translation units are compiled in, not reimplemented.
        self.assertIn(
            '#include "../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp"',
            source,
        )
        self.assertIn(
            '#include "../../token_selector_hls/src/token_selector_hls.cpp"', source
        )
        # No arithmetic primitives re-declared in the wrapper.
        for forbidden in ("floor_sqrt", "round_half_away", "minmax", "projection["):
            self.assertNotIn(forbidden, source)


class TransferLayoutAuditTest(unittest.TestCase):
    """Regression guards for the corrected 5-segment FPGA -> Jetson accounting.

    The pre-correction audit summed ``selected_tokens + metadata`` and applied
    one trailing ``align_up``; that misses the frozen contract's per-segment
    offset alignment and underestimates K=48/K=96.  These tests fail on the old
    implementation and pin the frozen offsets/sizes/total exactly.
    """

    BUDGETS = (48, 64, 96)
    EXPECTED_ARENA = {48: 196928, 64: 262464, 96: 393792}
    EXPECTED_SEGMENTS = {
        48: [
            ("tokens_int8", 0, 196608),
            ("group_index_uint16", 196608, 96),
            ("source_type_uint8", 196736, 48),
            ("region_id_uint8", 196800, 48),
            ("valid_uint8", 196864, 48),
        ],
        64: [
            ("tokens_int8", 0, 262144),
            ("group_index_uint16", 262144, 128),
            ("source_type_uint8", 262272, 64),
            ("region_id_uint8", 262336, 64),
            ("valid_uint8", 262400, 64),
        ],
        96: [
            ("tokens_int8", 0, 393216),
            ("group_index_uint16", 393216, 192),
            ("source_type_uint8", 393408, 96),
            ("region_id_uint8", 393536, 96),
            ("valid_uint8", 393664, 96),
        ],
    }

    @staticmethod
    def _legacy_single_align_total(output_group_count: int) -> int:
        """The deprecated pre-correction FPGA -> Jetson total for one (G, K)."""
        selected = output_group_count * 4096
        metadata = output_group_count * (2 + 1 + 1 + 1)
        raw = selected + metadata
        return ((raw + 63) // 64) * 64

    def test_arena_offsets_sizes_total_match_frozen_function(self) -> None:
        for budget in self.BUDGETS:
            frozen = describe_fpga_layout(budget)
            layout = fpga_transfer_layout(budget)
            self.assertEqual(frozen["total_bytes"], self.EXPECTED_ARENA[budget])
            self.assertEqual(layout.total_bytes, self.EXPECTED_ARENA[budget])
            self.assertEqual(frozen["alignment"], 64)
            self.assertEqual(len(frozen["segments"]), 5)
            self.assertEqual(
                [(s["name"], s["offset"], s["size"]) for s in frozen["segments"]],
                self.EXPECTED_SEGMENTS[budget],
            )
            for segment in frozen["segments"]:
                self.assertEqual(segment["offset"] % 64, 0)
            self.assertEqual(frozen["segments"][0]["offset"], 0)
            self.assertEqual(
                frozen["total_bytes"],
                ((frozen["segments"][-1]["offset"] + frozen["segments"][-1]["size"]
                  + 63) // 64) * 64,
            )

    def test_every_config_uses_frozen_arena_verbatim(self) -> None:
        audit = build_transfer_audit()
        self.assertEqual(len(audit["configs"]), 18)
        self.assertEqual(audit["transport_time_basis"], "fixed_budget_arena_bytes")
        for key, record in audit["configs"].items():
            budget = record["budget"]
            frozen = describe_fpga_layout(budget)
            fj = record["fpga_to_jetson"]
            self.assertEqual(fj["fixed_budget_arena_bytes"], frozen["total_bytes"], key)
            self.assertEqual(fj["fixed_budget_arena"]["total_bytes"], frozen["total_bytes"], key)
            self.assertEqual(fj["fixed_budget_arena"]["segments"], frozen["segments"], key)

    def test_k48_k96_regression_guard(self) -> None:
        # K=48: G=80 output_count=48; the old single align_up gave 196864 < arena.
        record = image_bytes(80, 48, False)
        out_groups = record["fpga_to_jetson"]["output_group_count"]
        self.assertEqual(out_groups, 48)
        arena = record["fpga_to_jetson"]["fixed_budget_arena_bytes"]
        self.assertEqual(arena, fpga_transfer_layout(48).total_bytes)
        self.assertEqual(arena, 196928)
        self.assertGreater(arena, self._legacy_single_align_total(out_groups))

        # K=96: G=80 identity; output_count=80 but the arena is the K=96 buffer.
        record = image_bytes(80, 96, False)
        out_groups = record["fpga_to_jetson"]["output_group_count"]
        self.assertEqual(out_groups, 80)
        arena = record["fpga_to_jetson"]["fixed_budget_arena_bytes"]
        self.assertEqual(arena, fpga_transfer_layout(96).total_bytes)
        self.assertEqual(arena, 393792)
        self.assertGreater(arena, self._legacy_single_align_total(out_groups))

        # K=64 happens to coincide with the old single align_up (documented).
        record = image_bytes(80, 64, False)
        out_groups = record["fpga_to_jetson"]["output_group_count"]
        self.assertEqual(out_groups, 64)
        arena = record["fpga_to_jetson"]["fixed_budget_arena_bytes"]
        self.assertEqual(arena, 262464)
        self.assertEqual(arena, self._legacy_single_align_total(out_groups))

    def test_identity_g80_k96_reports_payload_and_arena_separately(self) -> None:
        record = image_bytes(80, 96, False)
        fj = record["fpga_to_jetson"]
        self.assertEqual(record["output_group_count"], 80)
        self.assertEqual(fj["output_group_count"], 80)
        self.assertEqual(fj["selected_tokens"], 80 * 4096)
        self.assertEqual(fj["logical_payload_bytes"], 80 * 4096 + 80 * 5)
        self.assertEqual(fj["logical_payload_bytes"], 328080)
        self.assertEqual(fj["fixed_budget_arena_bytes"], 393792)
        self.assertGreater(fj["fixed_budget_arena_bytes"], fj["logical_payload_bytes"])
        # The arena is the runtime fixed budget K, never the identity output count.
        self.assertEqual(fj["fixed_budget_arena"]["budget"], 96)

    def test_network_and_ddr_times_are_arena_based(self) -> None:
        for key, record in build_transfer_audit()["configs"].items():
            fj = record["fpga_to_jetson"]
            arena = fj["fixed_budget_arena_bytes"]
            payload = fj["logical_payload_bytes"]
            times = record["times"]
            self.assertAlmostEqual(
                times["fpga_to_jetson_1gbe_arena"]["central_us"],
                arena / (110.0e6) * 1e6, places=9, msg=key,
            )
            self.assertAlmostEqual(
                times["fpga_to_jetson_local_ddr_6gbs_us"],
                arena / 6.0e9 * 1e6, places=9, msg=key,
            )
            lower_bound = times["fpga_to_jetson_1gbe_payload_scatter_gather_lower_bound"]
            self.assertEqual(lower_bound["basis"], "logical_payload_bytes", key)
            self.assertIn("lower bound", lower_bound["note"], key)
            self.assertAlmostEqual(
                lower_bound["central_us"], payload / (110.0e6) * 1e6, places=9, msg=key,
            )
            self.assertLessEqual(
                lower_bound["central_us"],
                times["fpga_to_jetson_1gbe_arena"]["central_us"],
                key,
            )

    def test_jetson_to_fpga_and_score_q_unchanged(self) -> None:
        record = image_bytes(80, 48, False)
        jf = record["jetson_to_fpga"]
        self.assertEqual(jf["tokens_q"], 327680)
        self.assertEqual(jf["group_mean_q"], 81920)
        self.assertEqual(jf["query_q"], 128)
        self.assertEqual(jf["wv_q"], 131072)
        self.assertEqual(jf["total"], 540800)
        self.assertEqual(record["baseline_score_q_ddr_round_trip"]["total"], 320)
        self.assertEqual(record["on_chip_fusion_saving"]["ddr_bytes"], 320)
        resident = image_bytes(80, 48, True)
        self.assertEqual(resident["jetson_to_fpga"]["wv_q"], 0)
        self.assertEqual(
            resident["jetson_to_fpga"]["total"],
            540800 - 131072,
        )

    def test_no_second_offset_algorithm_in_audit_source(self) -> None:
        source = (MODULE_DIR / "tools/transfer_audit.py").read_text()
        self.assertIn("fpga_transfer_layout", source)
        self.assertIn("describe_fpga_layout", source)
        self.assertNotIn("def _align_up", source)
        self.assertNotIn("for name, dtype, element_bytes", source)


if __name__ == "__main__":
    unittest.main()
