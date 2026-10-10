#!/usr/bin/env python3
"""Bit-exact Python golden for the Selector -> LayerNorm-Q8_0 adapter.

This module mirrors, operation for operation, the arithmetic of
``fpga_accel/selector_qkv_adapter_hls/src/selector_qkv_adapter.cpp``:

* statistics are integer-exact in the int8 domain:
  ``mean_q = sum(q)/1024``, ``var_q = E[q^2] - E[q]^2`` (population variance);
* the LayerNorm is evaluated through the algebraic identity
  ``LN(q/s) = ((q - mean_q) / sqrt(var_q + eps*s*s)) * gamma + beta``;
* the GGML Q8_0 requantization is ``d = amax/127``,
  ``q8 = clamp(round_half_away(v/d), -127, 127)`` and the stored scale is the
  repository's ``float_to_half_bits`` binary16 value.

The direct dequant -> FP32 LayerNorm chain is provided separately so the fused
kernel output can be compared against it (max_abs / RMSE / relative_L2 /
cosine). It is *not* used to produce the frozen golden bytes.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

import numpy as np

HIDDEN = 1024
BLOCK = 32
BLOCKS_PER_ROW = HIDDEN // BLOCK
INT8_LANES = 16
F32_LANES = 4
QKV_COLS = 3072
# Adapter row contract: Selector legal budgets are K = 48/64/96 only, so the
# adapter accepts exactly M = 4K = 192/256/384. SQA_MAX_ROWS is the adapter-only
# upper bound. The general bias-add operator keeps the historical row gate.
LEGAL_ADAPTER_ROWS = (192, 256, 384)
MAX_ROWS = LEGAL_ADAPTER_ROWS[-1]
BIAS_MAX_ROWS = 4320
MAX_COLS = 9728
EPSILON = 1.0e-6
SELECTOR_MIN = -128
SELECTOR_MAX = 127
Q8_0_MIN = -127
Q8_0_MAX = 127


def adapter_dimensions_valid(rows: int) -> bool:
    """The adapter accepts exactly the Selector legal M = 192/256/384."""
    return int(rows) in LEGAL_ADAPTER_ROWS


def bias_dimensions_valid(rows: int, cols: int) -> bool:
    """General bias-add row/col gate (rows 1..4320, cols 1..9728, cols%4==0)."""
    rows = int(rows)
    cols = int(cols)
    return 1 <= rows <= BIAS_MAX_ROWS and 1 <= cols <= MAX_COLS and cols % F32_LANES == 0

ADAPTER_FIXTURE_MAGIC = b"SQALNQ01"
BIAS_FIXTURE_MAGIC = b"SQABIA01"


# ---------------------------------------------------------------------------
# Repository binary16 conversion (exact copy of ffn_vector.cpp semantics)
# ---------------------------------------------------------------------------
def repo_float_to_half_bits(values: np.ndarray) -> np.ndarray:
    """Vectorized exact copy of ``ffn_vector.cpp::float_to_half_bits``."""
    bits = np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)
    bits = bits.astype(np.int64)
    sign = (bits >> 16) & 0x8000
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF

    out = np.zeros(bits.shape, dtype=np.int64)

    is_special = exponent == 0xFF
    out = np.where(is_special, sign | np.where(mantissa != 0, 0x7E00, 0x7C00), out)

    overflow = (exponent > 142) & (~is_special)
    out = np.where(overflow, sign | 0x7C00, out)

    subnormal = (exponent < 113) & (~is_special) & (~overflow)
    tiny = subnormal & (exponent < 103)
    out = np.where(tiny, sign, out)

    sub = subnormal & (~tiny)
    shifted_mantissa = mantissa | 0x800000
    shift = 125 - exponent
    rounded = shifted_mantissa >> shift
    remainder = shifted_mantissa & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    round_up = (remainder > halfway) | ((remainder == halfway) & ((rounded & 1) == 1))
    out = np.where(sub, sign | (rounded + round_up.astype(np.int64)), out)

    normal = ~(is_special | overflow | subnormal)
    normal_exponent = exponent - 112
    normal_mantissa = mantissa + 0xFFF + ((mantissa >> 13) & 1)
    carry = (normal_mantissa & 0x800000) != 0
    normal_mantissa = np.where(carry, 0, normal_mantissa)
    normal_exponent = normal_exponent + carry.astype(np.int64)
    normal_overflow = normal_exponent >= 31
    normal_value = (sign | (normal_exponent << 10) | (normal_mantissa >> 13))
    normal_value = np.where(normal_overflow, sign | 0x7C00, normal_value)
    out = np.where(normal, normal_value, out)

    return out.astype(np.uint16)


def repo_half_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """Decode repo binary16 bits back to float32 (used for dequantization)."""
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32)
    sign = (b >> 15) & 1
    exponent = (b >> 10) & 0x1F
    mantissa = b & 0x3FF
    value = np.zeros(b.shape, dtype=np.float64)
    subnormal = exponent == 0
    value = np.where(subnormal, np.ldexp(mantissa.astype(np.float64), -24), value)
    normal = (exponent > 0) & (exponent < 31)
    value = np.where(normal,
                     np.ldexp(1024.0 + mantissa, exponent.astype(np.int32) - 25),
                     value)
    special = exponent == 31
    value = np.where(special, np.where(mantissa != 0, np.nan, np.inf), value)
    return np.where(sign == 1, -value, value).astype(np.float32)


# ---------------------------------------------------------------------------
# Selector contract helpers
# ---------------------------------------------------------------------------
def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float32)
    return np.where(v >= np.float32(0.0),
                    np.floor(v + np.float32(0.5)),
                    np.ceil(v - np.float32(0.5))).astype(np.float32)


def selector_quantize(values: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    """q = clamp(round_half_away(x*s), -128, 127); returns (int8, saturated)."""
    scaled = np.asarray(values, dtype=np.float32) * np.float32(scale)
    rounded = round_half_away_from_zero(scaled)
    saturated = int(np.count_nonzero((rounded > SELECTOR_MAX) |
                                     (rounded < SELECTOR_MIN)))
    q = np.clip(rounded, SELECTOR_MIN, SELECTOR_MAX).astype(np.int8)
    return q, saturated


def selector_scale_from_amax(amax: float, levels: float = 128.0) -> float:
    amax = float(amax)
    if not np.isfinite(amax) or amax <= 0.0:
        return 1.0
    return float(levels / amax)


# ---------------------------------------------------------------------------
# LayerNorm references
# ---------------------------------------------------------------------------
def fused_layernorm(q_int8: np.ndarray,
                    gamma: np.ndarray,
                    beta: np.ndarray,
                    scale: float,
                    epsilon: float = EPSILON) -> np.ndarray:
    """Algebrically equivalent LayerNorm evaluated in the int8 domain."""
    q = np.asarray(q_int8, dtype=np.int64)
    if q.ndim != 2 or q.shape[1] != HIDDEN:
        raise ValueError(f"expected [rows,{HIDDEN}] int8, got {q.shape}")
    sum_q = q.sum(axis=1)
    sum_sq = (q * q).sum(axis=1)
    mean = (sum_q.astype(np.float32) / np.float32(HIDDEN)).astype(np.float32)
    mean_square = (sum_sq.astype(np.float32) / np.float32(HIDDEN)).astype(np.float32)
    variance = (mean_square - mean * mean).astype(np.float32)
    variance = np.maximum(variance, np.float32(0.0))
    with np.errstate(over="ignore", invalid="ignore"):
        eps_sq = (np.float32(epsilon) * np.float32(scale)) * np.float32(scale)
    denominator = (variance + np.float32(eps_sq)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = np.where(denominator > np.float32(0.0),
                           np.float32(1.0) / np.sqrt(denominator),
                           np.float32(0.0)).astype(np.float32)
    centered = (q.astype(np.float32) - mean[:, None]).astype(np.float32)
    normalized = (centered * inverse[:, None]).astype(np.float32)
    linear = (normalized * np.asarray(gamma, dtype=np.float32)[None, :]).astype(np.float32)
    return (linear + np.asarray(beta, dtype=np.float32)[None, :]).astype(np.float32)


def direct_layernorm(q_int8: np.ndarray,
                     gamma: np.ndarray,
                     beta: np.ndarray,
                     scale: float,
                     epsilon: float = EPSILON) -> np.ndarray:
    """Direct reference: dequantize x = q/s, then FP32 population LayerNorm."""
    q = np.asarray(q_int8, dtype=np.float32)
    x = (q / np.float32(scale)).astype(np.float32)
    mean = (x.mean(axis=1, dtype=np.float32)).astype(np.float32)
    mean_square = (np.mean(x * x, axis=1, dtype=np.float32)).astype(np.float32)
    variance = (mean_square - mean * mean).astype(np.float32)
    variance = np.maximum(variance, np.float32(0.0))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inverse = (np.float32(1.0) /
                   np.sqrt((variance + np.float32(epsilon)).astype(np.float32))
                   ).astype(np.float32)
    centered = (x - mean[:, None]).astype(np.float32)
    normalized = (centered * inverse[:, None]).astype(np.float32)
    linear = (normalized * np.asarray(gamma, dtype=np.float32)[None, :]).astype(np.float32)
    return (linear + np.asarray(beta, dtype=np.float32)[None, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Q8_0 requantization and 128-bit packing
# ---------------------------------------------------------------------------
def quantize_q8_0(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (payload[M,blocks,32] uint8, scale_bits[M,blocks] uint16, d[M,blocks] fp32)."""
    v = np.asarray(values, dtype=np.float32)
    rows, cols = v.shape
    if cols % BLOCK != 0:
        raise ValueError(f"cols={cols} is not a multiple of {BLOCK}")
    blocks = cols // BLOCK
    tiled = v.reshape(rows, blocks, BLOCK)
    amax = np.max(np.abs(tiled), axis=2).astype(np.float32)
    d = (amax / np.float32(127.0)).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(d == np.float32(0.0),
                           np.float32(0.0),
                           np.float32(1.0) / d).astype(np.float32)
    scaled = (tiled * inverse[:, :, None]).astype(np.float32)
    rounded = round_half_away_from_zero(scaled)
    q = np.clip(rounded, Q8_0_MIN, Q8_0_MAX).astype(np.int8)
    payload = q.view(np.uint8).reshape(rows, blocks, BLOCK)
    scale_bits = repo_float_to_half_bits(d)
    return payload, scale_bits, d


def dequantize_q8_0(payload: np.ndarray,
                    scale_bits: np.ndarray,
                    cols: int = HIDDEN) -> np.ndarray:
    rows, blocks, block = payload.shape
    scale = repo_half_bits_to_float(scale_bits.astype(np.uint16))
    values = payload.astype(np.int8).astype(np.float32) * scale[:, :, None]
    return values.reshape(rows, blocks * block)[:, :cols]


# ---------------------------------------------------------------------------
# Fixture containers
# ---------------------------------------------------------------------------
def write_adapter_fixture(path: Path,
                          q_int8: np.ndarray,
                          gamma: np.ndarray,
                          beta: np.ndarray,
                          scale: float,
                          epsilon: float,
                          payload: np.ndarray,
                          scale_bits: np.ndarray) -> None:
    rows = int(q_int8.shape[0])
    header = struct.pack("<8sIIIIIffII", ADAPTER_FIXTURE_MAGIC, 1, rows, HIDDEN,
                         BLOCK, BLOCKS_PER_ROW, float(scale), float(epsilon), 0, 0)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(gamma, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(beta, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(q_int8, dtype=np.int8).tobytes())
        handle.write(np.ascontiguousarray(payload, dtype=np.uint8).tobytes())
        handle.write(np.ascontiguousarray(scale_bits, dtype="<u2").tobytes())


def write_bias_fixture(path: Path,
                       input_values: np.ndarray,
                       bias: np.ndarray,
                       expected: np.ndarray) -> None:
    rows, cols = input_values.shape
    header = struct.pack("<8sIIIII", BIAS_FIXTURE_MAGIC, 1, rows, cols, 0, 0)
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(np.ascontiguousarray(input_values, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(bias, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(expected, dtype="<f4").tobytes())


def read_adapter_fixture(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    fmt = "<8sIIIIIffII"
    (magic, version, rows, cols, block, blocks, scale, epsilon,
     reserved0, reserved1) = struct.unpack_from(fmt, raw, 0)
    if magic != ADAPTER_FIXTURE_MAGIC or version != 1:
        raise ValueError(f"bad adapter fixture header: {magic!r} v{version}")
    offset = struct.calcsize(fmt)
    gamma = np.frombuffer(raw, dtype="<f4", count=cols, offset=offset).copy()
    offset += cols * 4
    beta = np.frombuffer(raw, dtype="<f4", count=cols, offset=offset).copy()
    offset += cols * 4
    input_bytes = np.frombuffer(raw, dtype=np.uint8,
                                count=rows * cols, offset=offset).copy()
    offset += rows * cols
    payload = np.frombuffer(raw, dtype=np.uint8,
                            count=rows * blocks * block, offset=offset).copy()
    offset += rows * blocks * block
    scale_bits = np.frombuffer(raw, dtype="<u2",
                               count=rows * blocks, offset=offset).copy()
    offset += rows * blocks * 2
    if offset != len(raw):
        raise ValueError("adapter fixture has trailing bytes")
    q_int8 = input_bytes.view(np.int8).reshape(rows, cols)
    return {
        "magic": magic,
        "version": version,
        "rows": rows,
        "cols": cols,
        "block": block,
        "blocks": blocks,
        "scale": scale,
        "epsilon": epsilon,
        "reserved": (reserved0, reserved1),
        "gamma": gamma,
        "beta": beta,
        "input_bytes": input_bytes,
        "q_int8": q_int8,
        "payload": payload.reshape(rows, blocks, block),
        "scale_bits": scale_bits.reshape(rows, blocks),
        "trailing_bytes": 0,
    }


def read_bias_fixture(path: Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    fmt = "<8sIIIII"
    magic, version, rows, cols, reserved0, reserved1 = struct.unpack_from(fmt, raw, 0)
    if magic != BIAS_FIXTURE_MAGIC or version != 1:
        raise ValueError(f"bad bias fixture header: {magic!r} v{version}")
    offset = struct.calcsize(fmt)
    input_values = np.frombuffer(raw, dtype="<f4", count=rows * cols,
                                 offset=offset).copy()
    offset += rows * cols * 4
    bias = np.frombuffer(raw, dtype="<f4", count=cols, offset=offset).copy()
    offset += cols * 4
    expected = np.frombuffer(raw, dtype="<f4", count=rows * cols,
                             offset=offset).copy()
    offset += rows * cols * 4
    if offset != len(raw):
        raise ValueError("bias fixture has trailing bytes")
    return {
        "magic": magic,
        "version": version,
        "rows": rows,
        "cols": cols,
        "reserved": (reserved0, reserved1),
        "input": input_values.reshape(rows, cols),
        "bias": bias,
        "expected": expected.reshape(rows, cols),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(reference, dtype=np.float64).reshape(-1)
    error = a - b
    max_abs = float(np.max(np.abs(error))) if error.size else 0.0
    rmse = float(np.sqrt(np.mean(error * error))) if error.size else 0.0
    reference_norm = float(np.linalg.norm(b))
    relative_l2 = float(np.linalg.norm(error) / reference_norm) if reference_norm else 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom else 0.0
    return {
        "max_abs": max_abs,
        "rmse": rmse,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(values: np.ndarray, dtype: str = "<f4") -> str:
    return sha256_bytes(np.ascontiguousarray(values, dtype=dtype).tobytes())
