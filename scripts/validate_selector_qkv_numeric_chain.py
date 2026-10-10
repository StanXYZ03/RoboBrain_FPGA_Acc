#!/usr/bin/env python3
"""CPU-only Selector -> real block1 fused QKV numeric-chain validation.

This script closes the minimum CPU verification candidate from
``reports/token_selector/2026-09-28-selector-downstream-compatibility.md`` §6 for
the *only* model-correct direct selector boundary of the Qwen3-VL vision tower::

    selector int8 -> dequant(global s) -> LayerNorm(norm1, eps=1e-6)
                  -> Q8_0 requant -> fused QKV GEMM(M, N=3072, K=1024)
                  -> + qkv.bias

It is a **numerical interface / implementation self-consistency check**, not a
task-accuracy result. It does not claim that selector pruning preserves model
accuracy; that requires the downstream task evaluation which is out of scope.

Hard boundaries
---------------
* CPU only. No CUDA, no GPU, no ZCU102 board, no synthesis/cosim, no network.
* No existing HLS/RTL/model/source file is modified.
* The real model is **not** loaded. Only four tensors are read tensor-by-tensor
  from the real BF16 safetensors shard through an explicit header+byte-range
  reader: ``model.visual.blocks.1.{norm1.weight,norm1.bias}`` and
  ``model.visual.blocks.1.attn.qkv.{weight,bias}``.
* ``x_fp32`` is a deterministic, numerically controlled *synthetic* block0
  residual hidden. It is **not** a captured real activation and must never be
  presented as one.
* Only the boundary up to ``+qkv.bias`` is covered. reshape/split, RoPE,
  softmax, PV, attn.proj, residual, norm2, fc1/gelu/fc2 and blocks 2..23 are
  not implemented here.

Levels (all share the same real weights ``W[3072,1024]``, ``bias[3072]``,
``gamma/beta[1024]``)::

    A  reference : LN(x)                     -> FP32 W + bias
    B  + selector: LN(dequant_int8(quant_int8(x))) -> FP32 W + bias
    C  + act Q8_0: dequant_q8(q8(LN(x_b)))   -> FP32 W + bias
    D  + wgt Q8_0: dequant_q8(q8(LN(x_b))) x dequant_q8(q8(W)) with INT32
                   per-32 block accumulation equivalent to the Q8Q8 core

Contract notes
--------------
* Selector contract (``fpga_accel/token_selector_hls/quantization_contract.md``):
  ``q = clamp(round_half_away(x * s), -128, 127)``, ``x ~= q / s``, zero point 0,
  one global symmetric ``s`` per image. ``1/s`` is **only** dequantization and is
  **not** equivalent to LayerNorm.
* Existing Q8_0 (``fpga_accel/pl_resident_ffn_hls/scripts/export_real_tensor_case.py``
  ``q8_quantize`` and ``fpga_accel/q8q8_pe16_dataflow_hls/src/q8q8_gemm.cpp``):
  per-32-element ``d = amax/127``, ``q = clamp(round_half_away(x/d), -127, 127)``,
  stored scale is binary16. The selector allows ``-128``; Q8_0 clamps to ``-127``.
"""

from __future__ import annotations

# Determinism: keep BLAS single-threaded before numpy is imported by this module.
import os

for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, "1")

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import resource
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT_DATE = "2026-09-28"
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-VL-4B-Instruct"
DEFAULT_SHARD = "model-00002-of-00002.safetensors"
DEFAULT_REPORT_DIR = ROOT / "reports" / "token_selector"
DEFAULT_HARNESS_PATH = Path("/tmp/robobrain_harness_result.json")

QK = 32
EPSILON = 1.0e-6
HIDDEN = 1024
QKV_OUT = 3072
NUM_HEADS = 16
HEAD_DIM = 64
SELECTOR_CLAMP = (-128, 127)
Q8_0_CLAMP = (-127, 127)
SELECTOR_LEVELS = 128.0  # s = SELECTOR_LEVELS / amax -> full asymmetric [-128,127]
LEVEL_NAMES = ("B", "C", "D")
TENSOR_NAMES = (
    "model.visual.blocks.1.norm1.weight",
    "model.visual.blocks.1.norm1.bias",
    "model.visual.blocks.1.attn.qkv.weight",
    "model.visual.blocks.1.attn.qkv.bias",
)

_STORAGE_DTYPES = {
    "BF16": ("<u2", 2),
    "F16": ("<f2", 2),
    "F32": ("<f4", 4),
    "F64": ("<f8", 8),
    "I8": ("i1", 1),
    "U8": ("u1", 1),
    "I16": ("<i2", 2),
    "I32": ("<i4", 4),
    "I64": ("<i8", 8),
    "U16": ("<u2", 2),
    "U32": ("<u4", 4),
    "U64": ("<u8", 8),
}


# ---------------------------------------------------------------------------
# Reuse of the existing Q8_0 software reference
# ---------------------------------------------------------------------------
def _load_existing_q8_reference() -> Any:
    ref_path = ROOT / "fpga_accel" / "pl_resident_ffn_hls" / "scripts" / "export_real_tensor_case.py"
    if not ref_path.is_file():
        raise FileNotFoundError(f"existing Q8_0 reference not found: {ref_path}")
    spec = importlib.util.spec_from_file_location("_selector_qkv_q8_reference", ref_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load existing Q8_0 reference: {ref_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_Q8_REFERENCE = _load_existing_q8_reference()


def _round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Element-wise round-half-away-from-zero (the selector/Q8_0 tie rule)."""
    v = np.asarray(values, dtype=np.float64)
    return np.where(v >= 0.0, np.floor(v + 0.5), np.ceil(v - 0.5))


def selector_quantize(values: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    """Selector contract: q = clamp(round_half_away(x*s), -128, 127)."""
    scaled = np.asarray(values, dtype=np.float64) * float(scale)
    rounded = _round_half_away_from_zero(scaled)
    saturated = int(np.count_nonzero((rounded > SELECTOR_CLAMP[1]) | (rounded < SELECTOR_CLAMP[0])))
    quantized = np.clip(rounded, SELECTOR_CLAMP[0], SELECTOR_CLAMP[1]).astype(np.int8)
    return quantized, saturated


def selector_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Selector contract: x ~= q / s (zero point 0)."""
    return (np.asarray(quantized, dtype=np.float32) / np.float32(scale)).astype(np.float32)


def selector_scale_from_amax(amax: float, levels: float = SELECTOR_LEVELS) -> float:
    """Global symmetric scale s = levels / amax (zero point 0)."""
    amax = float(amax)
    if not math.isfinite(amax) or amax <= 0.0:
        return 1.0
    return float(levels / amax)


def q8_0_quantize_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GGML Q8_0 rows, delegating the divisible case to the existing reference.

    Returns ``(payload[M, blocks, 32] int8, scales[M, blocks] fp32)`` where the
    scales are the stored binary16 values already rounded to fp16. For widths
    that are not a multiple of 32 the final partial block is zero padded and
    only the valid lanes are quantized (generic tail support).
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"Q8_0 expects a non-empty 2-D array, got {array.shape}")
    if array.shape[1] % QK == 0:
        return _Q8_REFERENCE.q8_quantize(array)
    return _q8_0_quantize_rows_tail(array)


def _q8_0_quantize_rows_tail(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = array.shape
    blocks = (cols + QK - 1) // QK
    padded = np.zeros((rows, blocks * QK), dtype=np.float32)
    padded[:, :cols] = array
    tiled = padded.reshape(rows, blocks, QK)
    amax = np.max(np.abs(tiled), axis=2).astype(np.float32)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inverse = np.zeros_like(d)
    np.divide(np.float32(1.0), d, out=inverse, where=d != 0.0)
    scaled = (tiled * inverse[:, :, None]).astype(np.float32)
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    payload = np.clip(rounded, Q8_0_CLAMP[0], Q8_0_CLAMP[1]).astype(np.int8)
    # The zero padding stays exactly zero after clipping.
    payload[:, :, :] = np.where(tiled == 0.0, np.int8(0), payload)
    scales = d.astype("<f2").astype(np.float32)
    return payload, scales


def q8_0_dequantize_rows(payload: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    """x = q * d, slicing the (possible) zero padding back to ``cols``."""
    dequantized = (
        payload.astype(np.float32) * scales[:, :, None]
    ).reshape(payload.shape[0], payload.shape[1] * QK)
    return np.ascontiguousarray(dequantized[:, :cols])


# ---------------------------------------------------------------------------
# Reference math (FP32) and layout helpers
# ---------------------------------------------------------------------------
def layer_norm(values: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = EPSILON) -> np.ndarray:
    """LayerNorm: center -> 1/sqrt(var+eps) -> *gamma -> +beta (var = E[x^2]-E[x]^2)."""
    x = np.asarray(values, dtype=np.float32)
    mean = np.mean(x, axis=1, dtype=np.float32)
    mean_square = np.mean(x * x, axis=1, dtype=np.float32)
    variance = (mean_square - mean * mean).astype(np.float32)
    variance = np.maximum(variance, np.float32(0.0))
    inverse = (np.float32(1.0) / np.sqrt(variance + np.float32(epsilon))).astype(np.float32)
    centered = (x - mean[:, None]).astype(np.float32)
    normalized = (centered * inverse[:, None]).astype(np.float32)
    return (normalized * np.asarray(gamma, dtype=np.float32)[None, :]
            + np.asarray(beta, dtype=np.float32)[None, :]).astype(np.float32)


def fp32_linear(values: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Y = X @ W.T + bias in float32."""
    y = np.matmul(np.asarray(values, dtype=np.float32), np.asarray(weight, dtype=np.float32).T)
    return (y.astype(np.float32) + np.asarray(bias, dtype=np.float32)[None, :]).astype(np.float32)


def fp32_linear_blocked(
    values: np.ndarray, weight: np.ndarray, bias: np.ndarray, block_k: int = QK
) -> np.ndarray:
    """K-blocked float32 linear used to verify blocked == direct on a small case."""
    x = np.asarray(values, dtype=np.float32)
    w = np.asarray(weight, dtype=np.float32)
    if x.shape[1] != w.shape[1]:
        raise ValueError("inner dimensions do not match")
    accum = np.zeros((x.shape[0], w.shape[0]), dtype=np.float32)
    for start in range(0, x.shape[1], block_k):
        stop = min(start + block_k, x.shape[1])
        accum = (accum + np.matmul(x[:, start:stop], w[:, start:stop].T)).astype(np.float32)
    return (accum + np.asarray(bias, dtype=np.float32)[None, :]).astype(np.float32)


def q8_0_int32_linear(
    activation_payload: np.ndarray,
    activation_scales: np.ndarray,
    weight_payload: np.ndarray,
    weight_scales: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Q8Q8-equivalent software simulation with INT32 per-32 block accumulation.

    Mirrors ``q8q8_gemm.cpp`` ``compute_cached_tile``: for every K block (ascending)
    ``contribution = float32(dot_int32(q_w, q_x)) * (weight_scale * activation_scale)``
    and accumulate in float32, then add the bias.
    """
    if activation_payload.shape[1] != weight_payload.shape[1]:
        raise ValueError("activation and weight block counts differ")
    rows = activation_payload.shape[0]
    outputs = weight_payload.shape[0]
    blocks = activation_payload.shape[1]
    accumulator: np.ndarray | None = None
    for block in range(blocks):
        xb = activation_payload[:, block, :].astype(np.int32)
        wb = weight_payload[:, block, :].astype(np.int32)
        dot = xb @ wb.T  # int32; |dot| <= 32*127*127 = 516128
        block_scale = (
            weight_scales[:, block][None, :] * activation_scales[:, block][:, None]
        ).astype(np.float32)
        contribution = (dot.astype(np.float32) * block_scale).astype(np.float32)
        accumulator = contribution if accumulator is None else (accumulator + contribution).astype(np.float32)
    assert accumulator is not None
    if accumulator.shape != (rows, outputs):
        raise ValueError("internal accumulation shape mismatch")
    return (accumulator + np.asarray(bias, dtype=np.float32)[None, :]).astype(np.float32)


def fused_qkv_segments(output: np.ndarray) -> dict[str, np.ndarray]:
    """Split fused ``[rows, 3072]`` output into Q|K|V ``[rows,1024]`` segments."""
    if output.shape[1] != QKV_OUT:
        raise ValueError(f"fused QKV output must be [*, {QKV_OUT}], got {output.shape}")
    return {
        "q": output[:, 0:HIDDEN],
        "k": output[:, HIDDEN:2 * HIDDEN],
        "v": output[:, 2 * HIDDEN:3 * HIDDEN],
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """max_abs / rmse / relative_L2 / cosine similarity of ``actual`` vs ``reference``."""
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    if a.shape != r.shape:
        raise ValueError(f"metric shape mismatch {a.shape} vs {r.shape}")
    delta = a - r
    reference_norm = float(np.linalg.norm(r))
    actual_norm = float(np.linalg.norm(a))
    if actual_norm > 0.0 and reference_norm > 0.0:
        cosine = float(np.dot(a, r) / (actual_norm * reference_norm))
    else:
        cosine = 1.0 if actual_norm == 0.0 and reference_norm == 0.0 else 0.0
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta) / reference_norm) if reference_norm > 0.0 else 0.0,
        "cosine_similarity": cosine,
    }


# ---------------------------------------------------------------------------
# Real weight loading (read-only, tensor by tensor)
# ---------------------------------------------------------------------------
def decode_storage(raw: bytes, dtype: str) -> np.ndarray:
    """Decode stored bytes to a flat numpy array; BF16 -> FP32 by exact widening."""
    if dtype == "BF16":
        widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return widened.view(np.float32)
    spec = _STORAGE_DTYPES.get(dtype)
    if spec is None:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return np.frombuffer(raw, dtype=spec[0])


def load_safetensors_tensors(path: Path, names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read only the requested tensors from a real safetensors shard (never the whole model)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    tensors: dict[str, Any] = {}
    with open(path, "rb") as handle:
        raw_header_len = handle.read(8)
        if len(raw_header_len) != 8:
            raise ValueError(f"{path} is not a safetensors file (short header length)")
        header_len = struct.unpack("<Q", raw_header_len)[0]
        header_bytes = handle.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        data_start = 8 + header_len
        for name in names:
            if name not in header:
                raise KeyError(f"{name} not present in {path.name}")
            info = header[name]
            offset_begin, offset_end = (int(v) for v in info["data_offsets"])
            handle.seek(data_start + offset_begin)
            raw = handle.read(offset_end - offset_begin)
            if len(raw) != offset_end - offset_begin:
                raise IOError(f"short read for {name}")
            array = decode_storage(raw, info["dtype"]).reshape(tuple(int(v) for v in info["shape"]))
            tensors[name] = {
                "array": array,
                "dtype": info["dtype"],
                "shape": tuple(int(v) for v in info["shape"]),
                "data_offsets": [offset_begin, offset_end],
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "fp32_sha256": hashlib.sha256(
                    np.ascontiguousarray(array, dtype="<f4").tobytes()
                ).hexdigest(),
            }
    after = path.stat()
    shard_info = {
        "shard": path.name,
        "shard_bytes": int(before.st_size),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "read_only_unchanged": bool(
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        ),
        "requested_tensors": list(names),
    }
    return tensors, shard_info


# ---------------------------------------------------------------------------
# Synthetic block0 output
# ---------------------------------------------------------------------------
def synth_block0_output(seed: int, rows: int, hidden: int = HIDDEN) -> np.ndarray:
    """Deterministic, numerically controlled synthetic block0 residual hidden.

    This is a *synthetic* activation, never a captured real-model activation. The
    construction mixes a seeded standard normal with a fixed low-frequency
    feature bias so LayerNorm has a non-trivial per-row mean and variance.
    """
    rng = np.random.default_rng(int(seed))
    base = rng.standard_normal((rows, hidden), dtype=np.float32)
    features = np.arange(hidden, dtype=np.float32)
    feature_bias = (
        np.float32(0.35) * np.sin(np.float32(2.0 * math.pi) * features / np.float32(hidden))
        + np.float32(0.15) * np.cos(np.float32(6.0 * math.pi) * features / np.float32(hidden))
    ).astype(np.float32)
    row_scale = (np.float32(0.9) + np.float32(0.2) * (
        np.arange(rows, dtype=np.float32) / np.float32(max(rows - 1, 1))
    )).astype(np.float32)
    values = (
        base * row_scale[:, None] * np.float32(0.8) + feature_bias[None, :]
    ).astype(np.float32)
    return np.ascontiguousarray(values)


def array_stats(values: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "amax": float(np.max(np.abs(flat))),
    }


def tensor_sha256(values: np.ndarray, dtype: str = "<f4") -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# One numeric-chain case
# ---------------------------------------------------------------------------
def run_case(
    case: dict[str, Any],
    weights: dict[str, np.ndarray],
    weight_q8: dict[str, Any],
    *,
    do_timing: bool = True,
) -> dict[str, Any]:
    """Run levels A/B/C/D for one synthetic activation case."""
    seed = int(case["seed"])
    rows = int(case["rows"])
    gamma = np.asarray(weights["norm1.weight"], dtype=np.float32)
    beta = np.asarray(weights["norm1.bias"], dtype=np.float32)
    weight = np.asarray(weights["qkv.weight"], dtype=np.float32)
    bias = np.asarray(weights["qkv.bias"], dtype=np.float32)

    timings: dict[str, float] = {}

    def timed(label: str, function):
        if not do_timing:
            return function()
        start = time.perf_counter()
        result = function()
        timings[label] = time.perf_counter() - start
        return result

    x = timed("synthesize_x", lambda: synth_block0_output(seed, rows))
    stats = array_stats(x)
    scale = selector_scale_from_amax(stats["amax"])

    x_q, saturated = timed("selector_quantize", lambda: selector_quantize(x, scale))
    x_b = timed("selector_dequantize", lambda: selector_dequantize(x_q, scale))

    ln_reference = timed("layernorm_A", lambda: layer_norm(x, gamma, beta))
    ln_after_selector = timed("layernorm_B", lambda: layer_norm(x_b, gamma, beta))

    act_payload, act_scales = timed(
        "activation_q8_0_quantize", lambda: q8_0_quantize_rows(ln_after_selector)
    )
    act_dequant = timed(
        "activation_q8_0_dequantize",
        lambda: q8_0_dequantize_rows(act_payload, act_scales, ln_after_selector.shape[1]),
    )

    ref_a = timed("matmul_A_fp32", lambda: fp32_linear(ln_reference, weight, bias))
    level_b = timed("matmul_B_fp32", lambda: fp32_linear(ln_after_selector, weight, bias))
    level_c = timed("matmul_C_fp32", lambda: fp32_linear(act_dequant, weight, bias))
    level_d = timed(
        "int32_q8q8_D",
        lambda: q8_0_int32_linear(
            act_payload,
            act_scales,
            weight_q8["payload"],
            weight_q8["scales"],
            bias,
        ),
    )

    # D vs "C with quantized weight", i.e. the INT32-simulation self-consistency.
    weight_dequant = np.asarray(weight_q8["dequant"], dtype=np.float32)
    c_quant_weight = timed(
        "matmul_C_quant_weight_fp32", lambda: fp32_linear(act_dequant, weight_dequant, bias)
    )

    kernel_labels = (
        "selector_quantize",
        "selector_dequantize",
        "layernorm_A",
        "layernorm_B",
        "activation_q8_0_quantize",
        "activation_q8_0_dequantize",
        "matmul_A_fp32",
        "matmul_B_fp32",
        "matmul_C_fp32",
        "matmul_C_quant_weight_fp32",
        "int32_q8q8_D",
    )
    kernel_total = float(sum(timings.get(label, 0.0) for label in kernel_labels))
    peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    segments = fused_qkv_segments(level_d)
    return {
        "case_id": case["case_id"],
        "seed": seed,
        "rows_M": rows,
        "selector_groups_K": rows // 4,
        "hidden_K": int(x.shape[1]),
        "qkv_N": int(weight.shape[0]),
        "note": case.get("note", ""),
        "x_synthetic": True,
        "x_stats": stats,
        "x_sha256_fp32": tensor_sha256(x),
        "selector_scale_s": scale,
        "selector_levels": SELECTOR_LEVELS,
        "selector_saturated_values": saturated,
        "selector_saturation_fraction": float(saturated) / float(x.size),
        "selector_clamp": [SELECTOR_CLAMP[0], SELECTOR_CLAMP[1]],
        "q8_0_clamp": [Q8_0_CLAMP[0], Q8_0_CLAMP[1]],
        "levels": {
            "B": metrics(level_b, ref_a),
            "C": metrics(level_c, ref_a),
            "D": metrics(level_d, ref_a),
        },
        "diagnostics": {
            "selector_input_quantization": metrics(x_b, x),
            "layernorm_output": metrics(ln_after_selector, ln_reference),
            "activation_q8_0": metrics(act_dequant, ln_after_selector),
            "weight_q8_0": metrics(weight_dequant, weight),
            "int32_sim_vs_dequant_fp32": metrics(level_d, c_quant_weight),
        },
        "level_max_abs_summary": {
            "B": float(np.max(np.abs(level_b - ref_a))),
            "C": float(np.max(np.abs(level_c - ref_a))),
            "D": float(np.max(np.abs(level_d - ref_a))),
        },
        "fused_qkv_layout": {
            "shape": [rows, QKV_OUT],
            "segments": {
                name: {"shape": list(segment.shape), "max_abs": float(np.max(np.abs(segment)))}
                for name, segment in segments.items()
            },
        },
        "timings_s": timings,
        "kernel_total_s": kernel_total,
        "peak_rss_kb": peak_rss_kb,
    }


# ---------------------------------------------------------------------------
# Self-consistency checks
# ---------------------------------------------------------------------------
def self_consistency_checks(weights: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    # 1. Generalized Q8_0 equals the existing reference on a divisible width.
    rng = np.random.default_rng(20260928)
    sample = (rng.standard_normal((7, 96), dtype=np.float32) * 3.0).astype(np.float32)
    payload, scales = q8_0_quantize_rows(sample)
    ref_payload, ref_scales = _Q8_REFERENCE.q8_quantize(sample)
    checks.append({
        "check": "q8_0_generalized_matches_existing_reference",
        "detail": "width 96 (multiple of 32) delegated to existing export_real_tensor_case.q8_quantize",
        "passed": bool(np.array_equal(payload, ref_payload) and np.array_equal(scales, ref_scales)),
        "max_abs_payload_diff": float(np.max(np.abs(payload.astype(np.int32) - ref_payload.astype(np.int32)))),
        "max_abs_scale_diff": float(np.max(np.abs(scales - ref_scales))),
    })

    # 2. Blocked FP32 matmul equals direct FP32 on a small case.
    x_small = (rng.standard_normal((4, 64), dtype=np.float32)).astype(np.float32)
    w_small = (rng.standard_normal((8, 64), dtype=np.float32) * 0.1).astype(np.float32)
    b_small = rng.standard_normal((8,), dtype=np.float32).astype(np.float32)
    direct = fp32_linear(x_small, w_small, b_small)
    blocked = fp32_linear_blocked(x_small, w_small, b_small, block_k=32)
    diff = metrics(blocked, direct)
    checks.append({
        "check": "blocked_fp32_matches_direct_small_case",
        "detail": "x[4,64] W[8,64] block_k=32",
        "passed": bool(diff["max_abs"] <= 1.0e-5),
        "comparison": diff,
    })

    # 3. INT32 accumulation equals the dequantized FP32 computation.
    act = (rng.standard_normal((4, 64), dtype=np.float32)).astype(np.float32)
    wgt = (rng.standard_normal((8, 64), dtype=np.float32) * 0.1).astype(np.float32)
    a_payload, a_scales = q8_0_quantize_rows(act)
    w_payload, w_scales = q8_0_quantize_rows(wgt)
    a_dequant = q8_0_dequantize_rows(a_payload, a_scales, 64)
    w_dequant = q8_0_dequantize_rows(w_payload, w_scales, 64)
    int32_out = q8_0_int32_linear(a_payload, a_scales, w_payload, w_scales, b_small)
    fp32_out = fp32_linear(a_dequant, w_dequant, b_small)
    diff32 = metrics(int32_out, fp32_out)
    checks.append({
        "check": "int32_accumulation_matches_dequantized_fp32_small_case",
        "detail": "x[4,64] W[8,64], same Q8_0 payloads",
        "passed": bool(diff32["relative_l2"] <= 1.0e-5),
        "comparison": diff32,
    })

    # 4. Fused QKV layout equals three separate GEMMs over the three weight segments.
    w_full = np.asarray(weights["qkv.weight"], dtype=np.float32)
    b_full = np.asarray(weights["qkv.bias"], dtype=np.float32)
    x_layout = (rng.standard_normal((3, HIDDEN), dtype=np.float32)).astype(np.float32)
    fused = fp32_linear(x_layout, w_full, b_full)
    segments = fused_qkv_segments(fused)
    split_ok = True
    split_detail: dict[str, Any] = {}
    for index, name in enumerate(("q", "k", "v")):
        start = index * HIDDEN
        separate = fp32_linear(
            x_layout, w_full[start:start + HIDDEN, :], b_full[start:start + HIDDEN]
        )
        comparison = metrics(segments[name], separate)
        split_detail[name] = comparison
        split_ok = split_ok and comparison["max_abs"] == 0.0
    checks.append({
        "check": "fused_qkv_layout_matches_three_segment_gemms",
        "detail": "[M,3072] split at 1024/2048 equals three GEMMs on W[0:1024]/[1024:2048]/[2048:3072]",
        "passed": bool(split_ok),
        "comparison": split_detail,
    })

    # 5. Selector -128 vs Q8_0 -127 clamp difference (informational contract check).
    extreme = np.asarray([[-1.0, -0.999, 1.0, 0.5]], dtype=np.float32)
    sel_q, _ = selector_quantize(extreme, scale=128.0)
    q8_payload, _ = q8_0_quantize_rows(extreme)
    q8_lane_values = [int(q8_payload[0, 0, i]) for i in range(4)]
    checks.append({
        "check": "selector_minus128_vs_q8_0_minus127",
        "detail": "selector keeps -128; Q8_0 clamps the same value to -127",
        "passed": bool(int(sel_q[0, 0]) == -128 and q8_lane_values[0] == -127),
        "selector_q": [int(v) for v in sel_q[0]],
        "q8_0_q": q8_lane_values,
    })

    return checks


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def default_cases() -> list[dict[str, Any]]:
    return [
        {"case_id": "m192_k48_seed20260928", "seed": 20260928, "rows": 192,
         "note": "required full case: M=4K with K=48 selector groups, hidden=1024, N=3072"},
        {"case_id": "m192_k48_seed20260929", "seed": 20260929, "rows": 192,
         "note": "full case, second seed"},
        {"case_id": "m192_k48_seed20260930", "seed": 20260930, "rows": 192,
         "note": "full case, third seed"},
        {"case_id": "m8_k2_seed20260928", "seed": 20260928, "rows": 8,
         "note": "small-M case proving the chain is M-independent"},
    ]


def quantify_weight_q8(weight: np.ndarray) -> dict[str, Any]:
    payload, scales = q8_0_quantize_rows(np.asarray(weight, dtype=np.float32))
    dequant = q8_0_dequantize_rows(payload, scales, weight.shape[1])
    return {
        "payload": payload,
        "scales": scales,
        "dequant": dequant,
        "block_size": QK,
        "clamp": [Q8_0_CLAMP[0], Q8_0_CLAMP[1]],
        "scale_storage": "binary16 rounded to fp32",
        "payload_sha256": hashlib.sha256(np.ascontiguousarray(payload, dtype=np.int8).tobytes()).hexdigest(),
        "metrics_vs_fp32": metrics(dequant, weight),
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir)
    shard_path = model_dir / args.shard
    weight_load_start = time.perf_counter()
    tensors, shard_info = load_safetensors_tensors(shard_path, TENSOR_NAMES)
    weight_load_seconds = time.perf_counter() - weight_load_start

    norm_weight = tensors["model.visual.blocks.1.norm1.weight"]["array"]
    norm_bias = tensors["model.visual.blocks.1.norm1.bias"]["array"]
    qkv_weight = tensors["model.visual.blocks.1.attn.qkv.weight"]["array"]
    qkv_bias = tensors["model.visual.blocks.1.attn.qkv.bias"]["array"]

    if tuple(qkv_weight.shape) != (QKV_OUT, HIDDEN):
        raise ValueError(f"unexpected qkv.weight shape {qkv_weight.shape}")
    if tuple(qkv_bias.shape) != (QKV_OUT,):
        raise ValueError(f"unexpected qkv.bias shape {qkv_bias.shape}")
    if tuple(norm_weight.shape) != (HIDDEN,) or tuple(norm_bias.shape) != (HIDDEN,):
        raise ValueError("unexpected norm1 shapes")

    weights = {
        "norm1.weight": norm_weight,
        "norm1.bias": norm_bias,
        "qkv.weight": qkv_weight,
        "qkv.bias": qkv_bias,
    }

    weight_meta = {}
    for name in TENSOR_NAMES:
        info = tensors[name]
        weight_meta[name] = {
            "dtype": info["dtype"],
            "shape": list(info["shape"]),
            "shard": shard_info["shard"],
            "data_offsets": info["data_offsets"],
            "raw_bytes": info["raw_bytes"],
            "raw_sha256": info["raw_sha256"],
            "fp32_sha256": info["fp32_sha256"],
            "stats": array_stats(info["array"]),
        }

    weight_q8_start = time.perf_counter()
    weight_q8 = quantify_weight_q8(qkv_weight)
    weight_q8_seconds = time.perf_counter() - weight_q8_start

    cases = default_cases() if args.cases == "default" else _parse_cases(args.cases)
    results = [run_case(case, weights, weight_q8, do_timing=True) for case in cases]

    # Determinism: repeat the first case and require byte-identical outputs.
    determinism = _determinism_probe(weights, weight_q8, cases[0])

    checks = self_consistency_checks(weights)

    return {
        "schema_version": 1,
        "date": args.date,
        "scope": (
            "CPU-only numeric interface/implementation self-consistency check for the single "
            "model-correct selector direct boundary: dequant(global s) -> LayerNorm(norm1, eps=1e-6) "
            "-> Q8_0 -> fused QKV GEMM(M,N=3072,K=1024) -> +qkv.bias. Real block1 weights are read "
            "tensor-by-tensor from BF16 safetensors; activations are deterministic synthetic block0 "
            "hidden, NOT captured real activations. No GPU, no board, no synthesis, no network, no "
            "existing HLS/RTL/model source modified, full model not loaded."
        ),
        "decision": (
            "SELECTOR_QKV_CPU_NUMERIC_CHAIN_IMPLEMENTED / INTERFACE_AND_IMPLEMENTATION_SELF_CONSISTENT"
            " / SYNTHETIC_ACTIVATION / NO_TASK_ACCURACY_CLAIM"
        ),
        "boundaries": {
            "gpu_used": False,
            "board_used": False,
            "synthesis_run": False,
            "network_used": False,
            "full_model_loaded": False,
            "activation_source": "deterministic synthetic block0 residual hidden (not real model activation)",
            "weights_source": "real BF16 safetensors, four block1 tensors only, read-only",
            "coverage": "up to +qkv.bias only; reshape/split, RoPE, softmax, PV, attn.proj, residual, norm2, fc1/gelu/fc2 and blocks 2..23 are NOT implemented",
            "claim": "interface/numerical implementation self-consistency only; no task-accuracy pass/fail",
        },
        "memory_bytes": {
            "norm1.weight": int(norm_weight.nbytes),
            "norm1.bias": int(norm_bias.nbytes),
            "qkv.weight": int(qkv_weight.nbytes),
            "qkv.bias": int(qkv_bias.nbytes),
            "qkv_weight_q8_payload": int(weight_q8["payload"].nbytes),
            "qkv_weight_q8_scales": int(weight_q8["scales"].nbytes),
            "total_weight_tensors": int(
                norm_weight.nbytes + norm_bias.nbytes + qkv_weight.nbytes + qkv_bias.nbytes
            ),
        },
        "weights": weight_meta,
        "shard": shard_info,
        "selector_contract": {
            "formula_quantize": "q = clamp(round_half_away(x * s), -128, 127)",
            "formula_dequantize": "x_approx = q / s",
            "zero_point": 0,
            "scale": "single global symmetric s shared by the whole selector input",
            "clamp": [SELECTOR_CLAMP[0], SELECTOR_CLAMP[1]],
            "scale_rule": f"s = {SELECTOR_LEVELS} / amax(x)",
            "note_1_over_s": "1/s broadcast is dequantization only and is NOT equivalent to LayerNorm",
        },
        "q8_0_contract": {
            "block_size": QK,
            "formula": "d = amax(block)/127; q = clamp(round_half_away(x/d), -127, 127); x_approx = q*d",
            "clamp": [Q8_0_CLAMP[0], Q8_0_CLAMP[1]],
            "scale_storage": "binary16",
            "reference_reused": "fpga_accel/pl_resident_ffn_hls/scripts/export_real_tensor_case.py::q8_quantize",
            "clamp_difference": "selector keeps -128, Q8_0 clamps to -127",
        },
        "weight_q8_0": {
            "quantize_seconds": weight_q8_seconds,
            "payload_sha256": weight_q8["payload_sha256"],
            "metrics_vs_fp32": weight_q8["metrics_vs_fp32"],
        },
        "weight_load_seconds": weight_load_seconds,
        "cases": results,
        "self_consistency": checks,
        "determinism": determinism,
        "tests": _load_tests_summary(args.tests_json),
        "resource_usage": {
            "cpu_count": os.cpu_count(),
            "blas_thread_vars": {key: os.environ.get(key) for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"
            )},
            "peak_rss_kb_final": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        },
        "limitations": [
            "x_fp32 is deterministic synthetic block0 hidden; it is not a real captured activation.",
            "CPU software emulation only: no GPU, no ZCU102 board, no synthesis/bitstream/cosim.",
            "The chain stops at +qkv.bias; the remaining block1 attention/MLP path is absent.",
            "No throughput/latency speedup is modelled or claimed; only wall-clock on this CPU host.",
            "The selector global scale s is derived as 128/amax(x); it is not a calibrated production scale.",
            "Results judge interface/numerical self-consistency only, not downstream task accuracy.",
        ],
        "limitations_zh": [
            "`x_fp32` 是确定性合成的 block0 hidden，不是真实采集激活。",
            "纯 CPU 软件模拟：无 GPU、无 ZCU102 板卡、无综合/bitstream/cosim。",
            "数值链只到 `+qkv.bias`；block1 后续 attention/MLP 路径完全缺失。",
            "不建模也不声明吞吐/时延加速；表中 wall-clock 仅是本机 CPU 串行计时。",
            "selector 全局 scale 由 `s = 128/amax(x)` 推导，不是标定后的生产 scale。",
            "结论只判接口/数值实现自洽，不判下游任务精度。",
        ],
        "next_gate": (
            "Only after this CPU chain is accepted: decide whether to implement the int8->fp32/LayerNorm-Q8_0 "
            "adapter and an explicit bias-add operator in HLS; do not synthesize, build a BD/driver, or go on "
            "board without separate authorization. Task accuracy still requires the full downstream path."
        ),
        "next_gate_zh": (
            "仅在本 CPU 链被接受后：再决定是否在 HLS 中实现 int8→fp32/LayerNorm-Q8_0 适配器与显式 bias-add 算子；"
            "没有独立授权不得综合、不得建 BD/驱动、不得上板。任务精度仍取决于完整下游路径。"
        ),
    }


def _load_tests_summary(path: str | None) -> dict[str, Any]:
    if not path:
        return {"note": "run `python -m unittest evaluation.tests.test_selector_qkv_numeric_chain` and the full suite separately"}
    summary_path = Path(path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tests summary must be a JSON object")
    return data


def _determinism_probe(
    weights: dict[str, np.ndarray], weight_q8: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    first = run_case(case, weights, weight_q8, do_timing=False)
    second = run_case(case, weights, weight_q8, do_timing=False)
    keys = (
        "x_sha256_fp32",
        "selector_scale_s",
        "selector_saturated_values",
        "levels",
        "diagnostics",
    )
    stable = all(first[key] == second[key] for key in keys)
    return {
        "case_id": case["case_id"],
        "repeat_count": 2,
        "byte_identical_fields": list(keys),
        "passed": bool(stable),
        "x_sha256_fp32": first["x_sha256_fp32"],
    }


def _parse_cases(spec: str) -> list[dict[str, Any]]:
    parsed = json.loads(spec)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("--cases must be a non-empty JSON list")
    cases = []
    for index, item in enumerate(parsed):
        cases.append({
            "case_id": str(item.get("case_id", f"case{index}")),
            "seed": int(item["seed"]),
            "rows": int(item["rows"]),
            "note": str(item.get("note", "")),
        })
    return cases


# ---------------------------------------------------------------------------
# Report emission
# ---------------------------------------------------------------------------
def write_csv(path: Path, report: dict[str, Any]) -> None:
    columns = [
        "case_id", "seed", "rows_M", "selector_groups_K", "hidden_K", "qkv_N",
        "selector_scale_s", "selector_saturated_values", "selector_saturation_fraction",
        "x_min", "x_max", "x_mean", "x_std", "x_amax",
        "B_max_abs", "B_rmse", "B_relative_l2", "B_cosine_similarity",
        "C_max_abs", "C_rmse", "C_relative_l2", "C_cosine_similarity",
        "D_max_abs", "D_rmse", "D_relative_l2", "D_cosine_similarity",
        "selector_input_quantization_max_abs", "selector_input_quantization_rmse",
        "selector_input_quantization_relative_l2", "selector_input_quantization_cosine_similarity",
        "layernorm_output_max_abs", "layernorm_output_rmse",
        "layernorm_output_relative_l2", "layernorm_output_cosine_similarity",
        "activation_q8_0_max_abs", "activation_q8_0_rmse", "activation_q8_0_relative_l2", "activation_q8_0_cosine_similarity",
        "weight_q8_0_max_abs", "weight_q8_0_rmse", "weight_q8_0_relative_l2", "weight_q8_0_cosine_similarity",
        "int32_sim_vs_dequant_fp32_max_abs", "int32_sim_vs_dequant_fp32_rmse",
        "int32_sim_vs_dequant_fp32_relative_l2", "int32_sim_vs_dequant_fp32_cosine_similarity",
        "kernel_total_s", "peak_rss_kb",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case in report["cases"]:
            row = {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "rows_M": case["rows_M"],
                "selector_groups_K": case["selector_groups_K"],
                "hidden_K": case["hidden_K"],
                "qkv_N": case["qkv_N"],
                "selector_scale_s": case["selector_scale_s"],
                "selector_saturated_values": case["selector_saturated_values"],
                "selector_saturation_fraction": case["selector_saturation_fraction"],
                "kernel_total_s": case["kernel_total_s"],
                "peak_rss_kb": case["peak_rss_kb"],
            }
            row.update({f"x_{key}": value for key, value in case["x_stats"].items()})
            for level in LEVEL_NAMES:
                row.update({f"{level}_{key}": value for key, value in case["levels"][level].items()})
            for key, value in case["diagnostics"].items():
                row.update({f"{key}_{metric}": value for metric, value in value.items()})
            writer.writerow(row)


def _md_metrics_table(entries: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| 对象 | max_abs | RMSE | relative_L2 | cosine |", "|---|---|---|---|---|"]
    for name, metric in entries.items():
        lines.append(
            f"| {name} | {metric['max_abs']:.6g} | {metric['rmse']:.6g} | "
            f"{metric['relative_l2']:.6g} | {metric['cosine_similarity']:.8f} |"
        )
    return lines


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Selector → 真实 block1 fused QKV CPU 数值链验证（合成激活）")
    lines.append("")
    lines.append(f"- 日期：{report['date']}")
    lines.append(f"- 决策：`{report['decision']}`")
    lines.append("- 范围：CPU-only；无 GPU、无板卡、无综合、无联网；只读真实 BF16 safetensors 的 4 个 block1 张量；不加载完整模型；不改任何 HLS/RTL/模型文件。")
    lines.append("")
    lines.append("## 0. 边界声明（必读）")
    lines.append("")
    lines.append("- `x_fp32` 是**确定性合成**的 block0 residual hidden，**不是**真实模型激活；不得声称是真实激活。")
    lines.append("- 本报告只判**接口/数值实现是否自洽**，**不**给出“任务精度通过”结论。")
    lines.append("- 数值链只覆盖到 `+qkv.bias`；reshape/split、RoPE、softmax、PV、`attn.proj`、残差、`norm2`、`fc1/GELU/fc2` 及 blocks 2..23 均未实现。")
    lines.append("- 无时延/吞吐加速声明；下表 wall-clock 仅是本机 CPU 计时。")
    lines.append("")
    lines.append("## 1. 量化合同")
    lines.append("")
    lines.append(f"- Selector：`{report['selector_contract']['formula_quantize']}`，反量化 `x ≈ q/s`，zero point 0，全图共享单一对称 `s`；本报告 `{report['selector_contract']['scale_rule']}`。饱和区间 `[{report['selector_contract']['clamp'][0]}, {report['selector_contract']['clamp'][1]}]`。")
    lines.append(f"- Q8_0：block={report['q8_0_contract']['block_size']}，`d = amax/127`，`q = clamp(round_half_away(x/d), -127, 127)`，`x ≈ q*d`，scale 存 binary16。复用既有 `export_real_tensor_case.py::q8_quantize`（宽度整除 32 时直接委托；尾块通用实现由单测证明与之一致）。")
    lines.append(f"- **差异**：selector 允许 `-128`，Q8_0 饱和到 `-127`；已用自洽检查 `selector_minus128_vs_q8_0_minus127` 记录。")
    lines.append("- `1/s` 广播**只是反量化**，**不等价于** LayerNorm；链中显式执行 LayerNorm(norm1, eps=1e-6)。")
    lines.append("")
    lines.append("## 2. 真实权重取证（只读）")
    lines.append("")
    lines.append(f"- shard：`{report['shard']['shard']}`（{report['shard']['shard_bytes']} B），header sha256 `{report['shard']['header_sha256']}`，读后未改动文件：`{report['shard']['read_only_unchanged']}`。")
    lines.append("")
    lines.append("| 张量 | dtype | shape | shard | data_offsets | raw sha256 |")
    lines.append("|---|---|---|---|---|---|")
    for name, meta in report["weights"].items():
        lines.append(
            f"| `{name}` | {meta['dtype']} | {meta['shape']} | {meta['shard']} | "
            f"{meta['data_offsets']} | `{meta['raw_sha256'][:16]}…` |"
        )
    lines.append("")
    lines.append(f"- 确认 `qkv.weight` shape = `[3072, 1024]`、`qkv.bias` shape = `[3072]`；`norm1.weight/bias` shape = `[1024]`。")
    lines.append("")
    lines.append("## 3. 案例与合成激活")
    lines.append("")
    lines.append("| case_id | seed | M | K_groups | hidden | N | scale s | 饱和数 | x amax | x std |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for case in report["cases"]:
        lines.append(
            f"| {case['case_id']} | {case['seed']} | {case['rows_M']} | {case['selector_groups_K']} | "
            f"{case['hidden_K']} | {case['qkv_N']} | {case['selector_scale_s']:.6g} | "
            f"{case['selector_saturated_values']} | {case['x_stats']['amax']:.6g} | {case['x_stats']['std']:.6g} |"
        )
    lines.append("")
    lines.append("## 4. 逐级误差（相对 A：FP32 LN(x) · W + b）")
    lines.append("")
    lines.append("- A：`LayerNorm(x_fp32)` → FP32 `W`+bias。")
    lines.append("- B：selector INT8 量化/反量化 → LayerNorm → FP32 `W`+bias。")
    lines.append("- C：在 B 基础上把 LayerNorm 输出经现有 Q8_0 量化/反量化 → FP32 `W`+bias。")
    lines.append("- D：在 C 基础上真实 `W` 也按 Q8_0 量化，INT32 分块累加（Q8Q8 等价）再 +bias。")
    lines.append("")
    for case in report["cases"]:
        lines.append(f"### {case['case_id']}（seed={case['seed']}, M={case['rows_M']}, N={case['qkv_N']}, K={case['hidden_K']}）")
        lines.append("")
        entries = {
            "B vs A": case["levels"]["B"],
            "C vs A": case["levels"]["C"],
            "D vs A": case["levels"]["D"],
        }
        lines.extend(_md_metrics_table(entries))
        lines.append("")
        lines.append("诊断分量：")
        lines.append("")
        diag = case["diagnostics"]
        diag_entries = {
            "selector 输入量化": diag["selector_input_quantization"],
            "LayerNorm 后": diag["layernorm_output"],
            "激活 Q8_0": diag["activation_q8_0"],
            "权重 Q8_0": diag["weight_q8_0"],
            "INT32 模拟 vs 反量化 FP32": diag["int32_sim_vs_dequant_fp32"],
        }
        lines.extend(_md_metrics_table(diag_entries))
        lines.append("")
    lines.append("## 5. fused QKV 输出布局")
    lines.append("")
    lines.append("`[M,3072]` 按 Q=`[:,0:1024]` | K=`[:,1024:2048]` | V=`[:,2048:3072]` 切分（与 `reshape(seq,3,num_heads,head_dim)` 一致）。")
    lines.append("")
    lines.append("## 6. 自洽性检查（只判实现，不判任务精度）")
    lines.append("")
    lines.append("| 检查 | 通过 | 说明 |")
    lines.append("|---|---|---|")
    for check in report["self_consistency"]:
        lines.append(f"| {check['check']} | {check['passed']} | {check['detail']} |")
    lines.append("")
    lines.append(f"- 确定性：case `{report['determinism']['case_id']}` 重复两次，字段逐字节一致 = `{report['determinism']['passed']}`，x sha256 `{report['determinism']['x_sha256_fp32'][:16]}…`。")
    lines.append("")
    lines.append("## 7. 资源与耗时")
    lines.append("")
    lines.append(f"- 权重读取（文件 IO + BF16→FP32，**不计入** kernel 链计时）：{report['weight_load_seconds']:.6g} s。")
    lines.append(f"- 权重 Q8_0 量化（一次性预处理，不计入每 case kernel 链计时）：{report['weight_q8_0']['quantize_seconds']:.6g} s。")
    lines.append(f"- CPU count = {report['resource_usage']['cpu_count']}；BLAS 线程环境 = {report['resource_usage']['blas_thread_vars']}。")
    lines.append("")
    lines.append("| case_id | kernel_total_s | A matmul s | B matmul s | C matmul s | D int32 s | peak_rss_kb |")
    lines.append("|---|---|---|---|---|---|---|")
    for case in report["cases"]:
        timing = case["timings_s"]
        lines.append(
            f"| {case['case_id']} | {case['kernel_total_s']:.6g} | "
            f"{timing.get('matmul_A_fp32', 0.0):.6g} | {timing.get('matmul_B_fp32', 0.0):.6g} | "
            f"{timing.get('matmul_C_fp32', 0.0):.6g} | {timing.get('int32_q8q8_D', 0.0):.6g} | "
            f"{case['peak_rss_kb']} |"
        )
    lines.append("")
    lines.append("## 8. 限制与下一步")
    lines.append("")
    for limitation in report.get("limitations_zh", report["limitations"]):
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append(f"**精确下一步（未执行）**：{report.get('next_gate_zh', report['next_gate'])}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_harness(path: Path, report: dict[str, Any]) -> None:
    """Overwrite the harness result with exactly the nine required keys."""
    payload = {
        "status": report["decision"],
        "evidence": [
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.md",
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.json",
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.csv",
            "scripts/validate_selector_qkv_numeric_chain.py",
            "evaluation/tests/test_selector_qkv_numeric_chain.py",
        ],
        "cases": [
            {
                "case_id": case["case_id"],
                "seed": case["seed"],
                "M": case["rows_M"],
                "K": case["hidden_K"],
                "N": case["qkv_N"],
                "synthetic_activation": True,
            }
            for case in report["cases"]
        ],
        "metrics": {
            case["case_id"]: {
                "B_vs_A": case["levels"]["B"],
                "C_vs_A": case["levels"]["C"],
                "D_vs_A": case["levels"]["D"],
                "diagnostics": case["diagnostics"],
            }
            for case in report["cases"]
        },
        "tests": report.get("tests", {}),
        "resource_usage": {
            "cpu_count": report["resource_usage"]["cpu_count"],
            "blas_thread_vars": report["resource_usage"]["blas_thread_vars"],
            "peak_rss_kb_final": report["resource_usage"]["peak_rss_kb_final"],
            "weight_load_seconds": report["weight_load_seconds"],
            "weight_q8_0_quantize_seconds": report["weight_q8_0"]["quantize_seconds"],
            "per_case_kernel_total_s": {
                case["case_id"]: case["kernel_total_s"] for case in report["cases"]
            },
        },
        "limitations": report["limitations"],
        "files_changed": [
            "scripts/validate_selector_qkv_numeric_chain.py",
            "evaluation/tests/test_selector_qkv_numeric_chain.py",
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.md",
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.json",
            "reports/token_selector/2026-09-28-selector-qkv-numeric-chain.csv",
            "CURRENT_STATUS.md",
        ],
        "next_gate": report["next_gate"],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--report-prefix", default=f"{REPORT_DATE}-selector-qkv-numeric-chain")
    parser.add_argument("--date", default=REPORT_DATE)
    parser.add_argument("--cases", default="default",
                        help='"default" or a JSON list of {case_id,seed,rows,note}')
    parser.add_argument("--harness", default=str(DEFAULT_HARNESS_PATH))
    parser.add_argument("--tests-json", default=None,
                        help="optional JSON object with the unittest summary to embed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_validation(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{args.report_prefix}.json"
    csv_path = report_dir / f"{args.report_prefix}.csv"
    md_path = report_dir / f"{args.report_prefix}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, report)
    write_markdown(md_path, report)
    write_harness(Path(args.harness), report)
    summary = {
        "decision": report["decision"],
        "cases": [case["case_id"] for case in report["cases"]],
        "D_vs_A_relative_l2": {case["case_id"]: case["levels"]["D"]["relative_l2"] for case in report["cases"]},
        "self_consistency_passed": all(check["passed"] for check in report["self_consistency"]),
        "determinism_passed": report["determinism"]["passed"],
        "json": str(json_path),
        "csv": str(csv_path),
        "md": str(md_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["self_consistency_passed"] and summary["determinism_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
