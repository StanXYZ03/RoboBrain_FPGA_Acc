#!/usr/bin/env python3
"""Focused tests for the compact integer Selector pipeline HLS module.

No model / GPU / network is touched:

* the input/output arena offsets, sizes and totals are checked against explicit
  formulas and against the frozen authority
  ``evaluation.hardware_regular_token_selector`` for K = 16/32/48/64/80/96;
* the C++ on-chip layout mirrors are compiled and compared against the same
  authority so the hardware implementation cannot silently drift;
* every stored expected canonical arena is re-derived from the frozen scorer and
  selector goldens over the stored input arena, and its padding/unused bytes are
  proven zero;
* the wrapper's source interface is checked: exactly three ``m_axi`` bundles,
  exactly one AXI-Lite ``control`` bundle, and no ``score_q`` port.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

TEST_DIR = Path(__file__).resolve().parent
MODULE_DIR = TEST_DIR.parent
ROOT = MODULE_DIR.parents[1]
for path in (str(ROOT), str(MODULE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.hardware_regular_token_selector import (  # noqa: E402
    describe_fpga_layout,
    fpga_transfer_layout,
)
from tools import generate_fixtures as gf  # noqa: E402
from tools.generate_fixtures import FIXTURE_DIR  # noqa: E402

CXX = os.environ.get("CXX", "g++")
INCLUDES = [
    "-Iinclude",
    "-I../integer_selector_scorer_hls/include",
    "-I../token_selector_hls/include",
    "-I../integer_selector_pipeline_hls/include",
]

LAYOUT_BUDGETS = (16, 32, 48, 64, 80, 96)
ARENA_TOTALS = {48: 196928, 64: 262464, 96: 393792}
ARENA_SEGMENTS = {
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
GROUP_COUNTS = (9, 16, 50, 80, 128, 260, 400, 4096)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _align_up(value: int, alignment: int = 64) -> int:
    return ((value + alignment - 1) // alignment) * alignment


_CXX_DUMP = None


def _layout_dump() -> Path:
    """Compile the C++ layout mirror once (host g++, no Vitis)."""
    global _CXX_DUMP
    if _CXX_DUMP is not None and Path(_CXX_DUMP).is_file():
        return Path(_CXX_DUMP)
    build_dir = Path(tempfile.mkdtemp(prefix="ispc_layout_dump_"))
    out = build_dir / "compact_layout_dump"
    cmd = [CXX, "-std=c++14", "-O1", "-Wno-unknown-pragmas", *INCLUDES,
           "tools/compact_layout_dump.cpp", "-o", str(out)]
    subprocess.run(cmd, cwd=MODULE_DIR, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    _CXX_DUMP = str(out)
    return out


def _dump_out(budget: int) -> dict:
    proc = subprocess.run([str(_layout_dump()), "out", str(budget)],
                          cwd=MODULE_DIR, check=True, stdout=subprocess.PIPE,
                          text=True)
    return _parse_out(proc.stdout)


def _parse_out(text: str) -> dict:
    status = None
    offsets: dict[int, tuple[int, int]] = {}
    total = None
    for line in text.splitlines():
        parts = line.split()
        if parts[:1] == ["status"]:
            status = int(parts[1])
        elif parts[:1] == ["seg"]:
            offsets[int(parts[1])] = (int(parts[2]), int(parts[3]))
        elif parts[:1] == ["total"]:
            total = int(parts[1])
    return {"status": status, "segments": offsets, "total": total}


def _dump_in(group_count: int) -> dict:
    proc = subprocess.run([str(_layout_dump()), "in", str(group_count)],
                          cwd=MODULE_DIR, check=True, stdout=subprocess.PIPE,
                          text=True)
    out = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = int(parts[1])
    return out


class LayoutAuthorityTest(unittest.TestCase):
    def test_output_formula_and_frozen_authority(self) -> None:
        for budget in LAYOUT_BUDGETS:
            frozen = describe_fpga_layout(budget)
            layout = fpga_transfer_layout(budget)
            expected_sizes = [budget * 4096, budget * 2, budget, budget, budget]
            self.assertEqual(frozen["alignment"], 64)
            self.assertEqual(len(frozen["segments"]), 5)
            offset = 0
            for segment, size in zip(frozen["segments"], expected_sizes):
                offset = _align_up(offset)
                self.assertEqual(segment["offset"], offset, (budget, segment))
                self.assertEqual(segment["size"], size, (budget, segment))
                self.assertEqual(segment["offset"] % 64, 0)
                offset += size
            self.assertEqual(frozen["total_bytes"], _align_up(offset))
            self.assertEqual(layout.total_bytes, frozen["total_bytes"])

    def test_known_totals(self) -> None:
        for budget, total in ARENA_TOTALS.items():
            self.assertEqual(fpga_transfer_layout(budget).total_bytes, total)
            self.assertEqual(describe_fpga_layout(budget)["total_bytes"], total)

    def test_known_segment_table(self) -> None:
        for budget, table in ARENA_SEGMENTS.items():
            frozen = describe_fpga_layout(budget)
            got = [(s["name"], s["offset"], s["size"]) for s in frozen["segments"]]
            self.assertEqual(got, table)

    def test_cpp_mirror_matches_authority(self) -> None:
        for budget in LAYOUT_BUDGETS:
            dumped = _dump_out(budget)
            frozen = describe_fpga_layout(budget)
            self.assertEqual(dumped["status"], 0)
            self.assertEqual(dumped["total"], frozen["total_bytes"])
            for index, segment in enumerate(frozen["segments"]):
                self.assertEqual(dumped["segments"][index],
                                 (segment["offset"], segment["size"]))

    def test_cpp_mirror_rejects_bad_budget(self) -> None:
        for bad in (0, 15, 47, 97, -16):
            proc = subprocess.run([str(_layout_dump()), "out", str(bad)],
                                  cwd=MODULE_DIR, stdout=subprocess.PIPE,
                                  text=True)
            self.assertEqual(proc.returncode, 1, bad)

    def test_input_arena_formula(self) -> None:
        for group_count in GROUP_COUNTS:
            expected_total = group_count * 5120 + 128
            self.assertEqual(group_count * 4096, group_count * 256 * 16)
            self.assertEqual(group_count * 1024, group_count * 64 * 16)
            for offset in (0, group_count * 4096, group_count * 5120):
                self.assertEqual(offset % 64, 0)
            dumped = _dump_in(group_count)
            self.assertEqual(dumped["status"], 0)
            self.assertEqual(dumped["tokens"], 0)
            self.assertEqual(dumped["group_mean"], group_count * 4096)
            self.assertEqual(dumped["query"], group_count * 5120)
            self.assertEqual(dumped["total"], expected_total)
            # These are the exact words the compact top derives its pointers
            # from (G*256, G*320) -- keep them in lockstep.
            self.assertEqual(dumped["tokens"] // 16, 0)
            self.assertEqual(dumped["group_mean"] // 16, group_count * 256)
            self.assertEqual(
                dumped["query"] // 16, group_count * 320
            )


class FixtureLedgerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text())
        cls.mod = gf.load_accepted_generator()
        wv = cls.mod.load_wv_int8()
        cls.mod.WV_CACHE["wv_q"] = wv.astype(np.int64)
        cls.scenes = {scene.name: scene for scene in cls.mod.build_scenes()}

    def test_wv_copy_is_accepted_bundle(self) -> None:
        self.assertEqual(self.manifest["wv"]["source_sha256"], self.mod.WV_SHA256)
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
        scene_names = set(self.manifest["scenes"])
        self.assertIn("real_grounding_correct", scene_names)
        self.assertIn("real_pointing_correct", scene_names)
        case_names = " ".join(self.manifest["cases"])
        self.assertIn("identity_g16", case_names)
        self.assertIn("identity_g9", case_names)
        self.assertIn("empty_2x25", case_names)
        self.assertIn("zero_g128", case_names)
        self.assertIn("all_equal_g128", case_names)

    def test_every_case_rederives_from_frozen_goldens(self) -> None:
        cases = {case.name: case for case in self.mod.build_cases()}
        for name, record in self.manifest["cases"].items():
            scene = self.scenes[record["scene"]]
            case = cases[name]
            data, meta = gf.expected_bytes(case, scene, self.mod)
            stored = (FIXTURE_DIR / record["file"]).read_bytes()
            self.assertEqual(data, stored, name)
            self.assertEqual(meta["status"], record["status"], name)
            self.assertEqual(meta["output_count"], record["output_count"], name)
            self.assertEqual(meta["total_bytes"], record["total_bytes"], name)

    def test_expected_arena_padding_is_zero(self) -> None:
        header = gf.EXP_HEADER
        for record in self.manifest["cases"].values():
            blob = (FIXTURE_DIR / record["file"]).read_bytes()
            fields = header.unpack_from(blob, 0)
            (
                magic,
                _status,
                output_count,
                _real,
                _ctx,
                _scale,
                budget,
                total_bytes,
                _fnv,
                payload,
            ) = fields
            self.assertEqual(magic, gf.EXP_MAGIC)
            self.assertEqual(payload, total_bytes)
            self.assertEqual(len(blob), header.size + total_bytes)
            arena = blob[header.size:]
            layout = fpga_transfer_layout(budget)
            off = layout.offsets()
            valid = bytearray(total_bytes)
            valid[0 : output_count * 4096] = b"\x01" * (output_count * 4096)
            for key, size in (
                ("group_index_uint16", output_count * 2),
                ("source_type_uint8", output_count),
                ("region_id_uint8", output_count),
                ("valid_uint8", output_count),
            ):
                start = off[key]
                valid[start : start + size] = b"\x01" * size
            for index, byte in enumerate(arena):
                if valid[index] == 0:
                    self.assertEqual(byte, 0, (record["case"], index))
            # identity cases leave the token tail and unused metadata zero
            if output_count < budget:
                self.assertEqual(
                    arena[output_count * 4096 : budget * 4096], b"\x00" * ((budget - output_count) * 4096)
                )


class InterfaceContractTest(unittest.TestCase):
    def test_top_has_no_score_q_port(self) -> None:
        header = (MODULE_DIR / "include/integer_selector_pipeline_compact.h").read_text()
        start = header.index("int integer_selector_pipeline_compact(")
        end = header.index(");", start)
        top = header[start:end]
        self.assertNotIn("score", top.lower())

    def test_source_has_exactly_three_m_axi_and_one_control(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline_compact.cpp").read_text()
        m_axi_bundles = []
        s_axilite_bundles = []
        for line in source.splitlines():
            if "HLS INTERFACE" not in line:
                continue
            if "m_axi" in line:
                self.assertIn("bundle=", line)
                bundle = line.split("bundle=", 1)[1].split()[0]
                m_axi_bundles.append(bundle)
            elif "s_axilite" in line:
                self.assertIn("bundle=control", line)
                s_axilite_bundles.append(line.split("bundle=", 1)[1].split()[0])
        self.assertEqual(sorted(m_axi_bundles),
                         ["input_arena", "output_arena", "wv_q"])
        self.assertEqual(set(s_axilite_bundles), {"control"})
        # Exactly one m_axi + one explicit s_axilite control per pointer so HLS
        # merges all offset registers into `control` (no `control_r`).
        for bundle in ("input_arena", "wv_q", "output_arena"):
            self.assertEqual(m_axi_bundles.count(bundle), 1)
            self.assertIn(f"s_axilite port={bundle} bundle=control", source)
        # No interface pragma may bind a score buffer.
        for line in source.splitlines():
            if "HLS INTERFACE" in line:
                self.assertNotIn("score", line.lower())

    def test_intermediate_is_local_and_never_an_interface(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline_compact.cpp").read_text()
        self.assertIn("uint16_t score_local[IS_MAX_GROUPS];", source)
        self.assertIn("is_scorer_core<", source)
        self.assertIn("token_selector_core_packed(", source)

    def test_wrapper_does_not_duplicate_upstream_maths(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline_compact.cpp").read_text()
        self.assertIn(
            '#include "../../integer_selector_scorer_hls/src/integer_selector_scorer.cpp"',
            source,
        )
        self.assertIn(
            '#include "../../token_selector_hls/src/token_selector_hls.cpp"', source
        )
        for forbidden in ("floor_sqrt", "round_half_away", "minmax", "projection["):
            self.assertNotIn(forbidden, source)

    def test_layout_mirror_has_no_second_float_maths(self) -> None:
        source = (MODULE_DIR / "src/integer_selector_pipeline_compact.cpp").read_text()
        header = (MODULE_DIR / "include/integer_selector_pipeline_compact.h").read_text()
        self.assertIn("ispc_fpga_layout", source)
        self.assertIn("ispc_input_layout", header)
        for forbidden in ("sqrt", "pow(", "floor("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
