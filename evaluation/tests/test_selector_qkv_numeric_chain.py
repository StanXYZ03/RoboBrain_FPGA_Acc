"""CPU-only tests for the Selector -> real block1 fused QKV numeric chain.

These tests cover the *implementation* of the numeric chain only: the selector
global INT8 round/clamp/dequant, LayerNorm ordering, the generalized Q8_0
block-32 (including a partial tail), the mandatory bias add, the fused-QKV
three-segment layout, determinism, read-only tensor loading, and the equivalence
of blocked / INT32 accumulation with direct FP32.

No GPU, no board, no network. The real-weight test is skipped when the model
shard is absent.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "validate_selector_qkv_numeric_chain",
    ROOT / "scripts" / "validate_selector_qkv_numeric_chain.py",
)
assert _SPEC is not None and _SPEC.loader is not None
chain = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = chain
_SPEC.loader.exec_module(chain)

REAL_SHARD = ROOT / "models" / "Qwen3-VL-4B-Instruct" / "model-00002-of-00002.safetensors"


def _synthetic_weights(rows: int = 3072, hidden: int = 1024) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(4242)
    return {
        "norm1.weight": (1.0 + 0.1 * rng.standard_normal(hidden)).astype(np.float32),
        "norm1.bias": (0.05 * rng.standard_normal(hidden)).astype(np.float32),
        "qkv.weight": (0.03 * rng.standard_normal((rows, hidden))).astype(np.float32),
        "qkv.bias": (0.1 * rng.standard_normal(rows)).astype(np.float32),
    }


class SelectorInt8ContractTest(unittest.TestCase):
    def test_round_half_away_from_zero(self) -> None:
        values = np.asarray([0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.4999], dtype=np.float64)
        rounded = chain._round_half_away_from_zero(values)
        np.testing.assert_array_equal(rounded, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 0.0])

    def test_round_clamp_and_dequant(self) -> None:
        scale = 2.0
        values = np.asarray([[0.0, 0.25, -0.25, 0.5, -0.5]], dtype=np.float32)
        quantized, saturated = chain.selector_quantize(values, scale)
        self.assertEqual(saturated, 0)
        np.testing.assert_array_equal(quantized, [[0, 1, -1, 1, -1]])
        dequantized = chain.selector_dequantize(quantized, scale)
        np.testing.assert_allclose(dequantized, [[0.0, 0.5, -0.5, 0.5, -0.5]], rtol=0, atol=0)

    def test_saturation_counts_clamp(self) -> None:
        scale = 128.0
        values = np.asarray([[1.0, -1.0, 0.5]], dtype=np.float32)
        quantized, saturated = chain.selector_quantize(values, scale)
        # +1.0*128 -> +128 clamps to 127; -1.0*128 -> -128 is kept.
        np.testing.assert_array_equal(quantized, [[127, -128, 64]])
        self.assertEqual(saturated, 1)
        self.assertEqual(chain.SELECTOR_CLAMP, (-128, 127))

    def test_minus128_selector_vs_minus127_q8_0(self) -> None:
        values = np.asarray([[-1.0, -0.999, 1.0, 0.5]], dtype=np.float32)
        selector_q, _ = chain.selector_quantize(values, scale=128.0)
        payload, _ = chain.q8_0_quantize_rows(values)
        self.assertEqual(int(selector_q[0, 0]), -128)
        self.assertEqual(int(payload[0, 0, 0]), -127)
        self.assertEqual(chain.Q8_0_CLAMP, (-127, 127))


class LayerNormOrderTest(unittest.TestCase):
    def test_matches_manual_formula(self) -> None:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((3, 8), dtype=np.float32).astype(np.float32)
        gamma = (1.0 + 0.2 * rng.standard_normal(8)).astype(np.float32)
        beta = (0.1 * rng.standard_normal(8)).astype(np.float32)
        epsilon = 1.0e-6
        got = chain.layer_norm(x, gamma, beta, epsilon)
        mean = x.mean(axis=1, dtype=np.float32)
        variance = ((x - mean[:, None]) ** 2).mean(axis=1, dtype=np.float32)
        expected = (
            (x - mean[:, None]) / np.sqrt(variance[:, None] + epsilon) * gamma[None, :] + beta[None, :]
        ).astype(np.float32)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_center_then_scale_order_matters(self) -> None:
        x = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        gamma = np.ones(4, dtype=np.float32)
        beta = np.zeros(4, dtype=np.float32)
        correct = chain.layer_norm(x, gamma, beta, 1.0e-6)
        # Wrong order: normalize raw x by RMS first, then center -> not LayerNorm.
        wrong_rms = (x / np.sqrt((x * x).mean(axis=1, keepdims=True) + 1.0e-6)).astype(np.float32)
        wrong = (wrong_rms - wrong_rms.mean(axis=1, keepdims=True)).astype(np.float32)
        self.assertGreater(float(np.max(np.abs(correct - wrong))), 1.0e-3)

    def test_gamma_and_beta_applied_last(self) -> None:
        x = np.asarray([[0.5, -0.5, 1.5, -1.5]], dtype=np.float32)
        gamma = np.full(4, 2.0, dtype=np.float32)
        beta = np.full(4, 0.25, dtype=np.float32)
        normalized = chain.layer_norm(x, np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32), 0.0)
        scaled = chain.layer_norm(x, gamma, beta, 0.0)
        np.testing.assert_allclose(scaled, normalized * 2.0 + 0.25, rtol=1e-5, atol=1e-6)


class Q8_0BlockTailTest(unittest.TestCase):
    def test_divisible_width_matches_existing_reference(self) -> None:
        rng = np.random.default_rng(11)
        for width in (32, 64, 96):
            values = (rng.standard_normal((5, width), dtype=np.float32) * 2.5).astype(np.float32)
            payload, scales = chain.q8_0_quantize_rows(values)
            ref_payload, ref_scales = chain._Q8_REFERENCE.q8_quantize(values)
            np.testing.assert_array_equal(payload, ref_payload)
            np.testing.assert_array_equal(scales, ref_scales)

    def test_partial_tail_block_is_general(self) -> None:
        rng = np.random.default_rng(13)
        for width in (1, 31, 33, 63, 100):
            values = (rng.standard_normal((4, width), dtype=np.float32) * 1.7).astype(np.float32)
            payload, scales = chain.q8_0_quantize_rows(values)
            blocks = (width + chain.QK - 1) // chain.QK
            self.assertEqual(payload.shape, (4, blocks, chain.QK))
            self.assertEqual(scales.shape, (4, blocks))
            tail_start = (blocks - 1) * chain.QK
            if width % chain.QK != 0:
                # lanes past the real width must be exactly zero padding
                self.assertTrue(np.all(payload[:, blocks - 1, width - tail_start:] == 0))
            dequantized = chain.q8_0_dequantize_rows(payload, scales, width)
            self.assertEqual(dequantized.shape, (4, width))
            for row in range(4):
                for block in range(blocks):
                    start = block * chain.QK
                    stop = min(start + chain.QK, width)
                    d = float(scales[row, block])
                    error = np.abs(dequantized[row, start:stop] - values[row, start:stop])
                    if d > 0:
                        # 0.5*d rounding plus the binary16 stored-scale mismatch
                        # (|q| <= 127, fp16 relative error <= 2**-11).
                        self.assertTrue(np.all(error <= d * 0.6 + 1.0e-6))
                    else:
                        self.assertTrue(np.all(values[row, start:stop] == 0.0))

    def test_dequantize_roundtrip_divisible(self) -> None:
        rng = np.random.default_rng(17)
        values = (rng.standard_normal((6, 128), dtype=np.float32) * 3.0).astype(np.float32)
        payload, scales = chain.q8_0_quantize_rows(values)
        back = chain.q8_0_dequantize_rows(payload, scales, 128)
        self.assertLess(float(np.max(np.abs(back - values))), 0.06)


class LinearAndBiasTest(unittest.TestCase):
    def test_bias_is_added(self) -> None:
        rng = np.random.default_rng(19)
        x = rng.standard_normal((3, 4), dtype=np.float32)
        weight = rng.standard_normal((5, 4), dtype=np.float32)
        bias = rng.standard_normal((5,), dtype=np.float32)
        with_bias = chain.fp32_linear(x, weight, bias)
        without_bias = chain.fp32_linear(x, weight, np.zeros((5,), dtype=np.float32))
        np.testing.assert_allclose(
            with_bias - without_bias, np.broadcast_to(bias, with_bias.shape), rtol=1e-5, atol=1e-6
        )
        self.assertGreater(float(np.max(np.abs(bias))), 1.0e-3)

    def test_blocked_matches_direct_fp32_small(self) -> None:
        rng = np.random.default_rng(23)
        x = rng.standard_normal((4, 64), dtype=np.float32)
        weight = (0.1 * rng.standard_normal((8, 64))).astype(np.float32)
        bias = rng.standard_normal((8,), dtype=np.float32)
        direct = chain.fp32_linear(x, weight, bias)
        blocked = chain.fp32_linear_blocked(x, weight, bias, block_k=32)
        self.assertLess(float(np.max(np.abs(blocked - direct))), 1.0e-5)

    def test_int32_accumulation_matches_dequantized_fp32_small(self) -> None:
        rng = np.random.default_rng(29)
        activation = rng.standard_normal((4, 64), dtype=np.float32)
        weight = (0.1 * rng.standard_normal((8, 64))).astype(np.float32)
        bias = rng.standard_normal((8,), dtype=np.float32)
        a_payload, a_scales = chain.q8_0_quantize_rows(activation)
        w_payload, w_scales = chain.q8_0_quantize_rows(weight)
        a_dequant = chain.q8_0_dequantize_rows(a_payload, a_scales, 64)
        w_dequant = chain.q8_0_dequantize_rows(w_payload, w_scales, 64)
        int32_out = chain.q8_0_int32_linear(a_payload, a_scales, w_payload, w_scales, bias)
        fp32_out = chain.fp32_linear(a_dequant, w_dequant, bias)
        self.assertLess(chain.metrics(int32_out, fp32_out)["relative_l2"], 1.0e-5)


class FusedQkvLayoutTest(unittest.TestCase):
    def test_three_segments_are_contiguous_and_equal_three_gemms(self) -> None:
        rng = np.random.default_rng(31)
        hidden = chain.HIDDEN
        weight = (0.05 * rng.standard_normal((chain.QKV_OUT, hidden))).astype(np.float32)
        bias = (0.1 * rng.standard_normal(chain.QKV_OUT)).astype(np.float32)
        x = rng.standard_normal((3, hidden), dtype=np.float32)
        fused = chain.fp32_linear(x, weight, bias)
        segments = chain.fused_qkv_segments(fused)
        self.assertEqual(set(segments), {"q", "k", "v"})
        for name in segments:
            self.assertEqual(segments[name].shape, (3, hidden))
        # Contiguous, non-overlapping concatenation in Q|K|V order.
        np.testing.assert_array_equal(
            np.concatenate([segments["q"], segments["k"], segments["v"]], axis=1), fused
        )
        for index, name in enumerate(("q", "k", "v")):
            start = index * hidden
            separate = chain.fp32_linear(x, weight[start:start + hidden], bias[start:start + hidden])
            np.testing.assert_array_equal(segments[name], separate)

    def test_wrong_width_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chain.fused_qkv_segments(np.zeros((2, 1024), dtype=np.float32))


class DeterminismTest(unittest.TestCase):
    def test_synthetic_activation_is_deterministic(self) -> None:
        first = chain.synth_block0_output(20260928, 16)
        second = chain.synth_block0_output(20260928, 16)
        third = chain.synth_block0_output(20260929, 16)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, third))
        self.assertEqual(chain.tensor_sha256(first), chain.tensor_sha256(second))

    def test_full_case_is_deterministic(self) -> None:
        weights = _synthetic_weights()
        weight_q8 = chain.quantify_weight_q8(weights["qkv.weight"])
        case = {"case_id": "det", "seed": 5, "rows": 4, "note": ""}
        first = chain.run_case(case, weights, weight_q8, do_timing=False)
        second = chain.run_case(case, weights, weight_q8, do_timing=False)
        self.assertEqual(first["x_sha256_fp32"], second["x_sha256_fp32"])
        self.assertEqual(first["selector_scale_s"], second["selector_scale_s"])
        self.assertEqual(first["levels"], second["levels"])
        self.assertEqual(first["diagnostics"], second["diagnostics"])


class ReadOnlyWeightLoadTest(unittest.TestCase):
    def _write_bf16_shard(self, path: Path) -> None:
        values = np.asarray([[1.0, -2.0, 0.5], [0.0, 3.0, -0.25]], dtype=np.float32)
        bf16 = (values.view(np.uint32) >> 16).astype("<u2").tobytes()
        header = {
            "toy.weight": {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, len(bf16)]},
        }
        header_bytes = json.dumps(header).encode("utf-8")
        padding = (8 - (len(header_bytes) % 8)) % 8
        header_bytes += b" " * padding
        with path.open("wb") as handle:
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            handle.write(bf16)

    def test_synthetic_shard_is_read_only_and_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "toy.safetensors"
            self._write_bf16_shard(shard)
            before = hashlib.sha256(shard.read_bytes()).hexdigest()
            tensors, info = chain.load_safetensors_tensors(shard, ("toy.weight",))
            after = hashlib.sha256(shard.read_bytes()).hexdigest()
            self.assertEqual(before, after)
            self.assertTrue(info["read_only_unchanged"])
            self.assertEqual(info["requested_tensors"], ["toy.weight"])
            self.assertEqual(tensors["toy.weight"]["dtype"], "BF16")
            self.assertEqual(tensors["toy.weight"]["shape"], (2, 3))
            np.testing.assert_array_equal(
                tensors["toy.weight"]["array"],
                np.asarray([[1.0, -2.0, 0.5], [0.0, 3.0, -0.25]], dtype=np.float32),
            )

    def test_missing_tensor_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "toy.safetensors"
            self._write_bf16_shard(shard)
            with self.assertRaises(KeyError):
                chain.load_safetensors_tensors(shard, ("does.not.exist",))

    @unittest.skipUnless(REAL_SHARD.is_file(), "real RoboBrain safetensors shard not present")
    def test_real_block1_tensor_shapes_and_read_only(self) -> None:
        tensors, info = chain.load_safetensors_tensors(REAL_SHARD, chain.TENSOR_NAMES)
        self.assertTrue(info["read_only_unchanged"])
        self.assertEqual(tensors["model.visual.blocks.1.attn.qkv.weight"]["shape"], (3072, 1024))
        self.assertEqual(tensors["model.visual.blocks.1.attn.qkv.bias"]["shape"], (3072,))
        self.assertEqual(tensors["model.visual.blocks.1.norm1.weight"]["shape"], (1024,))
        self.assertEqual(tensors["model.visual.blocks.1.norm1.bias"]["shape"], (1024,))
        for name in chain.TENSOR_NAMES:
            self.assertEqual(tensors[name]["dtype"], "BF16")
            self.assertEqual(len(tensors[name]["raw_sha256"]), 64)
            self.assertGreater(tensors[name]["raw_bytes"], 0)
        # Known first four BF16 values of block1 attn.qkv.bias from the audit.
        np.testing.assert_allclose(
            tensors["model.visual.blocks.1.attn.qkv.bias"]["array"][:4],
            [-0.05712890625, 0.8828125, -0.99609375, -0.1689453125],
            rtol=0, atol=0,
        )


if __name__ == "__main__":
    unittest.main()
