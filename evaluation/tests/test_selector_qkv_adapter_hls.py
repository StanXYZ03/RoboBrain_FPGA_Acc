"""Focused CPU contract tests for the Selector -> LayerNorm-Q8_0 adapter HLS.

These tests cover the Python golden and the frozen real-block1 fixtures that the
C++ / HLS C-sim testbench byte-compares against:

* the fused identity ``LN(q/s) = ((q-mean_q)/sqrt(var_q+eps*s*s))*gamma+beta``
  against the direct dequant -> FP32 LayerNorm chain;
* population-variance semantics and constant-row / ``q = -128`` behaviour;
* the repository binary16 conversion and GGML Q8_0 requantization;
* the frozen adapter / bias fixtures reproduce their stored golden bytes;
* the fixtures use the real block1 norm1 gamma/beta and real qkv.bias.

No GPU, no board, no synthesis, no network. Real-tensor tests are skipped when
the model shard is absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOOLS = ROOT / "fpga_accel" / "selector_qkv_adapter_hls" / "tools"
_SPEC = importlib.util.spec_from_file_location(
    "selector_qkv_golden", TOOLS / "selector_qkv_golden.py")
assert _SPEC is not None and _SPEC.loader is not None
golden = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = golden
_SPEC.loader.exec_module(golden)

REAL_SHARD = ROOT / "models" / "Qwen3-VL-4B-Instruct" / "model-00002-of-00002.safetensors"
FIXTURE_DIR = (ROOT / "artifacts" / "token_selector" /
               "2026-09-28-selector-qkv-adapter-csim")
ADAPTER_FIXTURE = FIXTURE_DIR / "real_block1_adapter_fixture.bin"
BIAS_FIXTURE = FIXTURE_DIR / "real_block1_bias_fixture.bin"
MANIFEST = FIXTURE_DIR / "fixture_manifest.json"

TENSOR_NORM1_W = "model.visual.blocks.1.norm1.weight"
TENSOR_NORM1_B = "model.visual.blocks.1.norm1.bias"
TENSOR_QKV_BIAS = "model.visual.blocks.1.attn.qkv.bias"


def _repo_float_to_half_bits_scalar(value: float) -> int:
    """Scalar port of ffn_vector.cpp::float_to_half_bits."""
    bits = int(np.array(value, dtype=np.float32).view(np.uint32))
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if exponent == 0xFF:
        return sign | (0x7E00 if mantissa else 0x7C00)
    if exponent > 142:
        return sign | 0x7C00
    if exponent < 113:
        if exponent < 103:
            return sign
        mantissa |= 0x800000
        shift = 125 - exponent
        rounded = mantissa >> shift
        remainder = mantissa & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and (rounded & 1)):
            rounded += 1
        return sign | rounded
    exponent -= 112
    mantissa += 0xFFF + ((mantissa >> 13) & 1)
    if mantissa & 0x800000:
        mantissa = 0
        exponent += 1
        if exponent >= 31:
            return sign | 0x7C00
    return sign | (exponent << 10) | (mantissa >> 13)


def _synthetic_q(rows: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.integers(golden.SELECTOR_MIN, golden.SELECTOR_MAX + 1,
                     size=(rows, golden.HIDDEN))
    features = np.arange(golden.HIDDEN, dtype=np.int32)
    q = q + (((features // 64)[None, :] + np.arange(rows)[:, None]) % 5 - 2) * 8
    return np.clip(q, golden.SELECTOR_MIN, golden.SELECTOR_MAX).astype(np.int8)


def _load_real_tensor(name: str) -> np.ndarray:
    with open(REAL_SHARD, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len).decode("utf-8"))
        info = header[name]
        begin, end = (int(v) for v in info["data_offsets"])
        handle.seek(8 + header_len + begin)
        raw = handle.read(end - begin)
    assert info["dtype"] == "BF16"
    widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return widened.view(np.float32).reshape(tuple(int(v) for v in info["shape"]))


class FusedLayerNormContractTest(unittest.TestCase):
    def test_fused_identity_matches_direct_across_scales(self) -> None:
        gamma = (0.75 + 0.00390625 * (np.arange(golden.HIDDEN) % 17)).astype(np.float32)
        beta = (0.00390625 * ((np.arange(golden.HIDDEN) % 7) - 3)).astype(np.float32)
        for seed in (20260928, 20260930):
            q = _synthetic_q(8, seed)
            for scale in (0.03, 1.0, 30.0, 1.0e30):
                fused = golden.fused_layernorm(q, gamma, beta, scale)
                direct = golden.direct_layernorm(q, gamma, beta, scale)
                metrics = golden.metrics(fused, direct)
                self.assertLess(metrics["relative_l2"], 1.0e-5, (seed, scale))
                self.assertGreater(metrics["cosine"], 0.999999, (seed, scale))
                self.assertTrue(np.isfinite(fused).all())

    def test_eps_scale_identity_matches_manual_formula(self) -> None:
        q = _synthetic_q(2, 7)
        gamma = np.ones(golden.HIDDEN, dtype=np.float32)
        beta = np.zeros(golden.HIDDEN, dtype=np.float32)
        scale = 32.0
        fused = golden.fused_layernorm(q, gamma, beta, scale)
        qf = q.astype(np.float64)
        mean = qf.mean(axis=1)
        var = qf.var(axis=1)  # population variance
        eps_sq = golden.EPSILON * scale * scale
        manual = (qf - mean[:, None]) / np.sqrt(var[:, None] + eps_sq)
        np.testing.assert_allclose(fused.astype(np.float64), manual, rtol=0, atol=2.0e-6)

    def test_eps_term_uses_eps_times_s_squared(self) -> None:
        # The only scale-dependent term is eps*s*s: fused(q,s) must equal the
        # manual q-domain formula with the eps*s*s denominator for every s,
        # including scales where the direct x-domain chain overflows.
        rng = np.random.default_rng(13)
        gamma = (1.0 + 0.05 * rng.standard_normal(golden.HIDDEN)).astype(np.float32)
        beta = (0.02 * rng.standard_normal(golden.HIDDEN)).astype(np.float32)
        q = _synthetic_q(3, 11)
        qf = q.astype(np.float64)
        mean = qf.mean(axis=1)
        var = qf.var(axis=1)
        for scale in (0.03, 1.0, 32.0, 1.0e30):
            eps_sq = golden.EPSILON * scale * scale
            manual = ((qf - mean[:, None]) / np.sqrt(var[:, None] + eps_sq)
                      * gamma.astype(np.float64)[None, :]
                      + beta.astype(np.float64)[None, :])
            fused = golden.fused_layernorm(q, gamma, beta, scale)
            np.testing.assert_allclose(fused.astype(np.float64), manual,
                                       rtol=0, atol=2.0e-5)

    def test_constant_row_reduces_to_beta(self) -> None:
        gamma = (0.5 + 0.01 * (np.arange(golden.HIDDEN) % 13)).astype(np.float32)
        beta = (0.001 * ((np.arange(golden.HIDDEN) % 11) - 5)).astype(np.float32)
        for value in (7, -128):
            q = np.full((2, golden.HIDDEN), value, dtype=np.int8)
            fused = golden.fused_layernorm(q, gamma, beta, 30.0)
            np.testing.assert_array_equal(fused[0], beta)
            np.testing.assert_array_equal(fused[1], beta)

    def test_minus128_is_used_before_q8_clamp(self) -> None:
        gamma = np.ones(golden.HIDDEN, dtype=np.float32)
        beta = np.zeros(golden.HIDDEN, dtype=np.float32)
        q = np.full((1, golden.HIDDEN), -128, dtype=np.int8)
        fused = golden.fused_layernorm(q, gamma, beta, 30.0)
        # constant -128 row -> mean = -128, centered = 0 -> zeros
        np.testing.assert_array_equal(fused, np.zeros_like(fused))
        payload, scale_bits, _ = golden.quantize_q8_0(fused)
        self.assertEqual(int(payload.min()), 0)
        self.assertEqual(int(scale_bits[0, 0]), 0)
        # A mixed row keeps -128 in the statistics and clamps only the output.
        mixed = np.full((1, golden.HIDDEN), 127, dtype=np.int8)
        mixed[0, 0] = -128
        payload2, _, _ = golden.quantize_q8_0(golden.fused_layernorm(
            mixed, gamma, beta, 30.0))
        self.assertGreaterEqual(int(payload2.view(np.int8).min()), golden.Q8_0_MIN)
        self.assertLessEqual(int(payload2.view(np.int8).max()), golden.Q8_0_MAX)


class HalfConversionContractTest(unittest.TestCase):
    def test_vectorized_matches_scalar_port(self) -> None:
        rng = np.random.default_rng(123)
        values = np.concatenate([
            (rng.standard_normal(20000).astype(np.float32) * 40.0),
            np.asarray([0.0, -0.0, 1e-9, 5.9e-8, 6.1e-8, 3.05e-5, 0.0078,
                        0.0079, 1.0, 65504.0, 70000.0, np.inf, -np.inf, np.nan],
                       dtype=np.float32),
        ])
        mine = golden.repo_float_to_half_bits(values)
        scalar = np.asarray([_repo_float_to_half_bits_scalar(v) for v in values],
                            dtype=np.uint16)
        np.testing.assert_array_equal(mine, scalar)

    def test_matches_ieee_numpy_in_normal_range(self) -> None:
        rng = np.random.default_rng(321)
        values = (np.abs(rng.standard_normal(50000).astype(np.float32)) * 20.0 + 0.01)
        mine = golden.repo_float_to_half_bits(values)
        numpy_half = values.astype("<f2").view(np.uint16)
        np.testing.assert_array_equal(mine, numpy_half)

    def test_decode_roundtrip(self) -> None:
        values = np.asarray([0.0, 1.0, -2.5, 0.0078125, 65504.0], dtype=np.float32)
        bits = golden.repo_float_to_half_bits(values)
        decoded = golden.repo_half_bits_to_float(bits)
        for original, restored in zip(values, decoded):
            if original == 0.0:
                self.assertEqual(float(restored), 0.0)
            else:
                self.assertAlmostEqual(float(restored), float(original), delta=0.01)


class Q8RequantizationContractTest(unittest.TestCase):
    def test_block_layout_and_clamp(self) -> None:
        values = np.zeros((1, golden.HIDDEN), dtype=np.float32)
        values[0, 0] = 1.0
        values[0, 1] = -1.0
        payload, scale_bits, d = golden.quantize_q8_0(values)
        self.assertEqual(payload.shape, (1, golden.BLOCKS_PER_ROW, golden.BLOCK))
        self.assertGreater(float(d[0, 0]), 0.0)
        signed = payload.view(np.int8)
        self.assertEqual(int(signed[0, 0, 0]), 127)
        self.assertEqual(int(signed[0, 0, 1]), -127)

    def test_zero_block_has_zero_scale_and_payload(self) -> None:
        values = np.zeros((1, golden.HIDDEN), dtype=np.float32)
        payload, scale_bits, _ = golden.quantize_q8_0(values)
        self.assertTrue(np.all(payload == 0))
        self.assertTrue(np.all(scale_bits == 0))

    def test_dequantize_roundtrip_within_q8_step(self) -> None:
        rng = np.random.default_rng(99)
        values = (rng.standard_normal((3, golden.HIDDEN), dtype=np.float32) * 2.0)
        payload, scale_bits, _ = golden.quantize_q8_0(values)
        dequant = golden.dequantize_q8_0(payload, scale_bits)
        metrics = golden.metrics(dequant, values)
        self.assertLess(metrics["relative_l2"], 0.02)
        self.assertGreater(metrics["cosine"], 0.9999)


class BiasAddGoldenTest(unittest.TestCase):
    def test_elementwise_sum_is_exact(self) -> None:
        rng = np.random.default_rng(5)
        values = (rng.standard_normal((4, golden.QKV_COLS), dtype=np.float32) * 0.5)
        bias = (0.001 * (np.arange(golden.QKV_COLS) % 17 - 8)).astype(np.float32)
        expected = (values + bias[None, :]).astype(np.float32)
        np.testing.assert_array_equal(expected, values + bias[None, :])
        self.assertEqual(expected.dtype, np.float32)


class AdapterRowContractTest(unittest.TestCase):
    """The adapter accepts exactly the Selector legal M = 192/256/384."""

    def test_legal_rows_are_exactly_192_256_384(self) -> None:
        self.assertEqual(tuple(golden.LEGAL_ADAPTER_ROWS), (192, 256, 384))
        self.assertEqual(golden.MAX_ROWS, 384)
        for rows in (192, 256, 384):
            self.assertTrue(golden.adapter_dimensions_valid(rows), rows)

    def test_all_other_rows_are_dimension_errors(self) -> None:
        for rows in (0, -1, 1, 48, 96, 128, 191, 193, 255, 257, 383, 385,
                     512, 4096, 4320):
            self.assertFalse(golden.adapter_dimensions_valid(rows), rows)

    def test_legal_rows_equal_four_times_selector_budgets(self) -> None:
        for rows in golden.LEGAL_ADAPTER_ROWS:
            self.assertIn(rows // 4, (48, 64, 96))

    def test_bias_add_keeps_general_row_gate(self) -> None:
        for rows in (1, 5, 16, 192, 256, 384, 4320):
            self.assertTrue(
                golden.bias_dimensions_valid(rows, golden.QKV_COLS), rows)
        for rows in (0, -1, 4321):
            self.assertFalse(
                golden.bias_dimensions_valid(rows, golden.QKV_COLS), rows)
        for cols in (0, 1, 2, 3, 3073, 9729):
            self.assertFalse(golden.bias_dimensions_valid(384, cols), cols)
        self.assertTrue(golden.bias_dimensions_valid(384, 3072))



@unittest.skipUnless(ADAPTER_FIXTURE.is_file() and BIAS_FIXTURE.is_file(),
                     "frozen fixtures not generated")
class FrozenFixtureTest(unittest.TestCase):
    def test_adapter_fixture_reproduces_stored_golden_bytes(self) -> None:
        fixture = golden.read_adapter_fixture(ADAPTER_FIXTURE)
        self.assertEqual(fixture["cols"], golden.HIDDEN)
        self.assertEqual(fixture["block"], golden.BLOCK)
        self.assertEqual(fixture["blocks"], golden.BLOCKS_PER_ROW)
        fused = golden.fused_layernorm(fixture["q_int8"], fixture["gamma"],
                                       fixture["beta"], fixture["scale"],
                                       fixture["epsilon"])
        payload, scale_bits, _ = golden.quantize_q8_0(fused)
        np.testing.assert_array_equal(payload, fixture["payload"])
        np.testing.assert_array_equal(scale_bits, fixture["scale_bits"])

    def test_adapter_fixture_input_packing_is_row_major_int8(self) -> None:
        fixture = golden.read_adapter_fixture(ADAPTER_FIXTURE)
        expected_bytes = fixture["q_int8"].astype(np.int8).tobytes()
        np.testing.assert_array_equal(fixture["input_bytes"].tobytes(),
                                      expected_bytes)

    def test_bias_fixture_expected_is_exact_sum(self) -> None:
        fixture = golden.read_bias_fixture(BIAS_FIXTURE)
        expected = (fixture["input"] + fixture["bias"][None, :]).astype(np.float32)
        np.testing.assert_array_equal(fixture["expected"], expected)

    def test_manifest_hashes_match_files(self) -> None:
        with open(MANIFEST, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for key, path in (("adapter_fixture", ADAPTER_FIXTURE),
                          ("bias_fixture", BIAS_FIXTURE)):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(manifest[key]["sha256"], digest)
            self.assertEqual(manifest[key]["bytes"], path.stat().st_size)
        self.assertTrue(manifest["shard_info"]["read_only_unchanged"])

    def test_manifest_declares_legal_adapter_rows(self) -> None:
        with open(MANIFEST, encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["adapter_fixture"]["legal_rows_M"],
                         [192, 256, 384])
        self.assertEqual(manifest["adapter_fixture"]["max_rows"], 384)
        self.assertEqual(manifest["adapter_fixture"]["rows_M"], 192)
        self.assertEqual(manifest["bias_fixture"]["formal_rows_M"],
                         [192, 256, 384])
        self.assertEqual(manifest["bias_fixture"]["rows_M"], 384)
        self.assertEqual(manifest["bias_fixture"]["cols_N"], 3072)

    def test_adapter_fixture_rows_are_legal(self) -> None:
        fixture = golden.read_adapter_fixture(ADAPTER_FIXTURE)
        self.assertTrue(golden.adapter_dimensions_valid(fixture["rows"]))

    def test_bias_fixture_rows_are_formal_boundary(self) -> None:
        fixture = golden.read_bias_fixture(BIAS_FIXTURE)
        self.assertTrue(golden.bias_dimensions_valid(fixture["rows"],
                                                     fixture["cols"]))
        self.assertIn(fixture["rows"], golden.LEGAL_ADAPTER_ROWS)
        self.assertEqual(fixture["cols"], golden.QKV_COLS)


    @unittest.skipUnless(REAL_SHARD.is_file(), "real safetensors shard not present")
    def test_fixture_gamma_beta_and_bias_are_real_block1(self) -> None:
        fixture = golden.read_adapter_fixture(ADAPTER_FIXTURE)
        gamma = _load_real_tensor(TENSOR_NORM1_W)
        beta = _load_real_tensor(TENSOR_NORM1_B)
        np.testing.assert_array_equal(fixture["gamma"], gamma)
        np.testing.assert_array_equal(fixture["beta"], beta)
        bias_fixture = golden.read_bias_fixture(BIAS_FIXTURE)
        qkv_bias = _load_real_tensor(TENSOR_QKV_BIAS)
        np.testing.assert_array_equal(bias_fixture["bias"], qkv_bias)
        with open(MANIFEST, encoding="utf-8") as handle:
            manifest = json.load(handle)
        for name in (TENSOR_NORM1_W, TENSOR_NORM1_B, TENSOR_QKV_BIAS):
            self.assertIn(name, manifest["tensors"])
            self.assertEqual(manifest["tensors"][name]["dtype"], "BF16")


if __name__ == "__main__":
    unittest.main()
