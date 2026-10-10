#!/usr/bin/env python3
"""Focused tests for the standalone integer Selector scorer HLS kernel.

These tests exercise the Python fixture layer (never a second copy of the
golden maths) plus the static HLS-source contract.  Run with:

    python -m unittest discover -s fpga_accel/integer_selector_scorer_hls/tests
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "tools"))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import generate_fixtures as gf  # noqa: E402
from evaluation.selector_integer_scorer_audit import (  # noqa: E402
    integer_chain,
    round_half_away_div,
)

DUT_SOURCE = PROJECT / "src/integer_selector_scorer.cpp"
DUT_HEADER = PROJECT / "include/integer_selector_scorer.h"
TB_SOURCE = PROJECT / "tb/tb_integer_selector_scorer.cpp"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def parse_fixture_header(data: bytes) -> dict:
    fields = gf.FIXTURE_HEADER.unpack_from(data, 0)
    return {
        "magic": fields[0],
        "version": fields[1],
        "kind": fields[2],
        "group_count": fields[3],
        "grid_h": fields[4],
        "grid_w": fields[5],
        "role_count": fields[6],
        "generator": fields[7],
        "generator_seed": fields[8],
        "score_fnv1a64": fields[9],
        "semantic_fnv1a64": fields[10],
        "l2_fnv1a64": fields[11],
        "group_mean_fnv1a64": fields[12],
        "payload_bytes": fields[13],
    }


class WvBundleTests(unittest.TestCase):
    def test_wv_sha_is_pinned(self) -> None:
        self.assertEqual(gf.sha256_file(gf.WV_PATH), gf.WV_SHA256)
        self.assertEqual(gf.sha256_file(gf.WV_SCALE_PATH), gf.WV_SCALE_SHA256)

    def test_wv_shape_and_dtype(self) -> None:
        tensor = gf.load_wv_int8()
        self.assertEqual(tensor.shape, (128, 1024))
        self.assertEqual(tensor.dtype, np.int8)


class RealFixtureGoldenTests(unittest.TestCase):
    def setUp(self) -> None:
        gf.WV_CACHE["wv_q"] = gf.load_wv_int8().astype(np.int64)

    def test_grounding_and_pointing_golden_matches_stored_npz(self) -> None:
        for task in gf.REAL_TASKS:
            with np.load(
                gf.REAL_FIXTURE_DIR / f"fixture_{task}.npz", allow_pickle=False
            ) as archive:
                group_mean_q = archive["group_mean_q"].astype(np.int8)
                grid_h = int(archive["grid_h"])
                grid_w = int(archive["grid_w"])
                for role in ("correct", "wrong"):
                    query_q = archive[f"query_q_{role}"].astype(np.int8)
                    golden = gf.golden_chain(group_mean_q, query_q, grid_h, grid_w)
                    for key in (
                        "semantic_q16",
                        "l2_int",
                        "semantic_norm_q15",
                        "l2_norm_q15",
                    ):
                        np.testing.assert_array_equal(
                            golden[key], archive[f"{key}_{role}"].astype(np.int64)
                        )
                    np.testing.assert_array_equal(
                        golden["score_q"], archive[f"score_q_{role}"].astype(np.uint16)
                    )

    def test_real_fixture_bytes_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            gf.generate(Path(first))
            gf.generate(Path(second))
            for task in gf.REAL_TASKS:
                name = f"full_{task}.bin"
                self.assertEqual(
                    (Path(first) / name).read_bytes(),
                    (Path(second) / name).read_bytes(),
                )


class SyntheticCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        gf.WV_CACHE["wv_q"] = gf.load_wv_int8().astype(np.int64)

    def _role_score(self, data: bytes, role: int) -> np.ndarray:
        header = parse_fixture_header(data)
        g_count = header["group_count"]
        role_stride = 128 + 12 * g_count
        base = 128 + (g_count * 1024 if header["kind"] in (0, 1) else 0)
        score_off = base + role * role_stride + 128
        return np.frombuffer(data, dtype="<u2", count=g_count, offset=score_off)

    def test_zero_maps_to_zero(self) -> None:
        _, data, _ = gf.build_synthetic_zero()
        self.assertTrue(np.all(self._role_score(data, 0) == 0))

    def test_all_equal_maps_to_zero(self) -> None:
        _, data, _ = gf.build_synthetic_all_equal()
        self.assertTrue(np.all(self._role_score(data, 0) == 0))

    def test_g1_maps_to_zero(self) -> None:
        _, data, _ = gf.build_synthetic_g1()
        header = parse_fixture_header(data)
        self.assertEqual(header["group_count"], 1)
        self.assertTrue(np.all(self._role_score(data, 0) == 0))

    def test_mixed_sign_exercises_both_semantic_signs(self) -> None:
        _, _, meta = gf.build_synthetic_mixed_sign()
        self.assertGreater(meta["dot_positive_groups"], 0)
        self.assertGreater(meta["dot_negative_groups"], 0)
        self.assertGreater(meta["semantic_negative_groups"], 0)

    def test_stress_g400_is_streamed_not_embedded(self) -> None:
        _, data, meta = gf.build_stress_g400()
        header = parse_fixture_header(data)
        self.assertEqual(header["kind"], 2)
        self.assertEqual(header["group_count"], 400)
        self.assertEqual(header["generator"], 1)
        self.assertLess(len(data), 400 * 1024)
        self.assertFalse(meta["full_rows_embedded"])

    def test_stress_generator_matches_vectorised_version(self) -> None:
        small = gf.xorshift_bytes(gf.SEED, 4096)
        vectorised = gf.xorshift_bytes_np(gf.SEED, 4096)
        self.assertEqual(small, vectorised.tobytes())

    def test_stress_score_hash_header_matches_payload(self) -> None:
        _, data, _ = gf.build_stress_g400()
        header = parse_fixture_header(data)
        g_count = header["group_count"]
        base = 128  # kind 2: no embedded group_mean
        score = np.frombuffer(data, dtype="<u2", count=g_count, offset=base + 128)
        self.assertEqual(gf.fnv1a64(score.tobytes()), header["score_fnv1a64"])


class PrimitiveVectorTests(unittest.TestCase):
    def test_positive_and_negative_half_ties(self) -> None:
        data = gf.build_primitives()
        n_sqrt, n_div, _n_minmax, _total = gf.PRIMITIVE_HEADER.unpack_from(data, 0)[2:6]
        cursor = 64 + n_sqrt * 16
        positive_ties = 0
        negative_ties = 0
        for _ in range(n_div):
            numerator, denominator, expected = struct.unpack_from("<qqq", data, cursor)
            cursor += 24
            self.assertEqual(expected, round_half_away_div(numerator, denominator))
            if 2 * abs(numerator) == denominator:
                if numerator > 0:
                    positive_ties += 1
                    self.assertEqual(expected, 1)
                elif numerator < 0:
                    negative_ties += 1
                    self.assertEqual(expected, -1)
        self.assertGreaterEqual(positive_ties, 1)
        self.assertGreaterEqual(negative_ties, 1)

    def test_floor_sqrt_vectors_match_frozen_golden(self) -> None:
        from evaluation.selector_integer_scorer_audit import floor_sqrt_int64

        data = gf.build_primitives()
        n_sqrt = gf.PRIMITIVE_HEADER.unpack_from(data, 0)[2]
        cursor = 64
        for _ in range(n_sqrt):
            value, expected = struct.unpack_from("<QQ", data, cursor)
            cursor += 16
            self.assertEqual(expected, floor_sqrt_int64(value))


class PackingContractTests(unittest.TestCase):
    @staticmethod
    def _pack(raw: bytes, lanes: int = 16):
        words = []
        for w in range((len(raw) + lanes - 1) // lanes):
            word = 0
            for lane in range(lanes):
                index = w * lanes + lane
                byte = raw[index] if index < len(raw) else 0
                word |= (byte & 0xFF) << (8 * lane)
            words.append(word)
        return words

    def test_little_endian_lane_layout(self) -> None:
        raw = bytes(range(32))
        words = self._pack(raw)
        self.assertEqual(words[0] & 0xFF, 0)
        self.assertEqual((words[0] >> 8) & 0xFF, 1)
        self.assertEqual((words[1] >> (8 * 15)) & 0xFF, 31)

    def test_fixture_header_sizes(self) -> None:
        self.assertEqual(gf.FIXTURE_HEADER.size, 128)
        self.assertEqual(gf.PRIMITIVE_HEADER.size, 64)


class SourceContractTests(unittest.TestCase):
    def test_no_float_or_double_in_dut_datapath(self) -> None:
        code = strip_comments(DUT_SOURCE.read_text(encoding="utf-8"))
        code = re.sub(r'"([^"\\]|\\.)*"', '""', code)
        self.assertIsNone(re.search(r"\b(float|double)\b", code))
        self.assertNotIn("pow(", code)
        self.assertNotIn("sqrt(", code)

    def test_shifted_numerator_is_a_multiplication(self) -> None:
        code = strip_comments(DUT_SOURCE.read_text(encoding="utf-8"))
        self.assertIn("dot * (int64_t)((int64_t)1 <<", code)
        self.assertNotIn("dot <<", code)

    def test_packed_axi_interface_and_status_enum(self) -> None:
        header = DUT_HEADER.read_text(encoding="utf-8")
        self.assertIn("typedef ap_uint<128> is_word_t;", header)
        self.assertIn("IS_STATUS_OK = 0", header)
        self.assertIn("IS_STATUS_ERR_NULL = 1", header)
        self.assertIn("IS_STATUS_ERR_GROUP_COUNT = 2", header)
        code = DUT_SOURCE.read_text(encoding="utf-8")
        for port in ("group_mean_q", "wv_q", "query_q", "score_q"):
            self.assertIn(f"m_axi port={port}", code)

    def test_generator_reuses_frozen_golden(self) -> None:
        source = (PROJECT / "tools/generate_fixtures.py").read_text(encoding="utf-8")
        self.assertIn("from evaluation.selector_integer_scorer_audit import", source)
        self.assertIn("integer_chain", source)
        # The generator must not define its own sqrt / rounding maths.
        self.assertNotIn("def floor_sqrt", source)
        self.assertNotIn("def round_half_away", source)
        self.assertNotIn("def minmax", source)

    def test_testbench_checks_every_golden_field(self) -> None:
        source = TB_SOURCE.read_text(encoding="utf-8")
        for field in (
            "debug score_q bytes",
            "top score_q bytes",
            "semantic_q16 bytes",
            "l2_int bytes",
            "semantic_norm_q15 bytes",
            "l2_norm_q15 bytes",
            "floor_sqrt vectors",
            "round_half_away vectors",
            "round_half_away exact ties",
            "minmax vectors",
        ):
            self.assertIn(field, source)


class CommittedArtifactTests(unittest.TestCase):
    def test_manifest_matches_committed_binaries(self) -> None:
        fixture_dir = gf.FIXTURE_DIR
        manifest_path = fixture_dir / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("fixtures not generated in this checkout")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["wv"]["source_sha256"], gf.WV_SHA256)
        for name, meta in manifest["fixtures"].items():
            path = fixture_dir / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(meta["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
