#!/usr/bin/env python3
"""Build a frozen real-RoboBrain-weight FFN numerical case without CUDA.

The input rows are dequantized token embeddings.  They are model-derived proxy
activations, not captured block-0 runtime activations; the JSON records this
distinction explicitly.  The full proxy gate compares the candidate Q8/PWL
data path with a dequantized-weight FP32 reference.  A compact binary slice is
also emitted for the HLS C simulation gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np


MAGIC = b"FFNVRW01"
VERSION = 1
QK = 32
GGUF_Q8_BLOCK_BYTES = 34
DEFAULT_TOKEN_IDS = (0, 1, 42, 1000)

SILU_LUT = np.asarray([
    -0.00268280104, -0.00333681738, -0.00414583978, -0.00514511039,
    -0.00637735836, -0.00789419429, -0.00975768467, -0.0120420916,
    -0.0148357389, -0.0182429263, -0.0223857574, -0.0274056599,
    -0.0334642546, -0.0407430557, -0.0494412418, -0.0597704149,
    -0.0719448398, -0.0861651372, -0.102592808, -0.121312384,
    -0.14227762, -0.165238288, -0.18964545, -0.214536296,
    -0.238405844, -0.259082597, -0.273638286, -0.278375174,
    -0.268941421, -0.240615976, -0.188770334, -0.109455875,
    0.0, 0.140544125, 0.311229666, 0.509384024,
    0.731058579, 0.971624826, 1.22636171, 1.4909174,
    1.76159416, 2.0354637, 2.31035455, 2.58476171,
    2.85772238, 3.12868762, 3.39740719, 3.66383486,
    3.92805516, 4.19022959, 4.45055876, 4.70925694,
    4.96653575, 5.22259434, 5.47761424, 5.73175707,
    5.98516426, 6.23795791, 6.49024232, 6.74210581,
    6.99362264, 7.24485489, 7.49585416, 7.74666318,
    7.9973172,
], dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q8_planes(raw_rows: np.ndarray, inputs: int) -> tuple[np.ndarray, np.ndarray]:
    if inputs % QK:
        raise ValueError(f"Q8 input dimension {inputs} is not divisible by {QK}")
    blocks = inputs // QK
    expected = blocks * GGUF_Q8_BLOCK_BYTES
    rows = np.ascontiguousarray(raw_rows)
    if rows.dtype != np.uint8 or rows.ndim != 2 or rows.shape[1] != expected:
        raise ValueError(f"unexpected Q8 bytes {rows.dtype} {rows.shape}; expected (*,{expected})")
    shaped = rows.reshape(rows.shape[0], blocks, GGUF_Q8_BLOCK_BYTES)
    scales = np.ascontiguousarray(shaped[:, :, :2]).view("<f2").reshape(rows.shape[0], blocks)
    payload = np.ascontiguousarray(shaped[:, :, 2:]).view(np.int8)
    return payload, scales.astype(np.float32)


def q8_quantize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = values.shape
    if cols % QK:
        raise ValueError("quantized width must be a multiple of 32")
    blocks = values.astype(np.float32, copy=False).reshape(rows, cols // QK, QK)
    amax = np.max(np.abs(blocks), axis=2).astype(np.float32)
    d = (amax / np.float32(127.0)).astype(np.float32)
    inverse = np.zeros_like(d)
    np.divide(np.float32(1.0), d, out=inverse, where=d != 0)
    scaled = blocks * inverse[:, :, None]
    rounded = np.where(scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    payload = np.clip(rounded, -127, 127).astype(np.int8)
    # The GEMM consumes the stored FP16 scale, while q values are computed from
    # the unrounded FP32 scale, matching GGML quantize_row_q8_0_ref.
    return payload, d.astype("<f2").astype(np.float32)


def q8_linear(payload: np.ndarray, scales: np.ndarray,
              weight_payload: np.ndarray, weight_scales: np.ndarray) -> np.ndarray:
    dots = np.einsum(
        "rbi,nbi->rnb",
        payload.astype(np.int32),
        weight_payload.astype(np.int32),
        dtype=np.int32,
        optimize=True,
    )
    terms = dots.astype(np.float32) * scales[:, None, :] * weight_scales[None, :, :]
    return np.sum(terms, axis=2, dtype=np.float32)


def float_linear(values: np.ndarray, weight_payload: np.ndarray,
                 weight_scales: np.ndarray) -> np.ndarray:
    blocks = weight_payload.shape[1]
    shaped = values.astype(np.float32, copy=False).reshape(values.shape[0], blocks, QK)
    dots = np.einsum(
        "rbi,nbi->rnb",
        shaped,
        weight_payload.astype(np.float32),
        dtype=np.float32,
        optimize=True,
    )
    return np.sum(dots * weight_scales[None, :, :], axis=2, dtype=np.float32)


def rms_norm(values: np.ndarray, gamma: np.ndarray, epsilon: float) -> np.ndarray:
    mean_square = np.mean(values * values, axis=1, dtype=np.float32)
    inverse = np.float32(1.0) / np.sqrt(mean_square + np.float32(epsilon), dtype=np.float32)
    return values * inverse[:, None] * gamma[None, :]


def silu_pwl(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32, copy=False)
    result = np.empty_like(x)
    low = x <= -8.0
    high = x >= 8.0
    middle = ~(low | high)
    result[low] = 0.0
    result[high] = x[high]
    positions = (x[middle] + np.float32(8.0)) * np.float32(4.0)
    segments = np.clip(positions.astype(np.int32), 0, 63)
    fractions = positions - segments.astype(np.float32)
    result[middle] = SILU_LUT[segments] + (
        SILU_LUT[segments + 1] - SILU_LUT[segments]
    ) * fractions
    return result


def exact_silu(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32, copy=False)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = np.linalg.norm(reference.astype(np.float64).reshape(-1))
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / denominator if denominator else 0.0),
        "reference_max_abs": float(np.max(np.abs(reference))),
    }


def select_tensors(reader, layer: int):
    wanted = {
        "embedding": "token_embd.weight",
        "norm": f"blk.{layer}.ffn_norm.weight",
        "gate": f"blk.{layer}.ffn_gate.weight",
        "up": f"blk.{layer}.ffn_up.weight",
        "down": f"blk.{layer}.ffn_down.weight",
    }
    by_name = {tensor.name: tensor for tensor in reader.tensors}
    missing = [name for name in wanted.values() if name not in by_name]
    if missing:
        raise ValueError(f"missing tensors: {missing}")
    return wanted, {role: by_name[name] for role, name in wanted.items()}


def tensor_shape(tensor) -> tuple[int, ...]:
    return tuple(int(item) for item in tensor.shape.tolist())


def write_slice(path: Path, token_ids: tuple[int, ...], epsilon: float,
                input_rows: np.ndarray, gamma: np.ndarray,
                gate_raw: np.ndarray, up_raw: np.ndarray, down_raw: np.ndarray,
                expanded: int, outputs: int, reference: np.ndarray) -> None:
    header = struct.pack(
        "<8s6If",
        MAGIC,
        VERSION,
        len(token_ids),
        input_rows.shape[1],
        expanded,
        outputs,
        len(token_ids),
        epsilon,
    )
    with path.open("xb") as handle:
        handle.write(header)
        handle.write(np.asarray(token_ids, dtype="<u4").tobytes())
        handle.write(np.asarray(input_rows, dtype="<f4").tobytes())
        handle.write(np.asarray(gamma, dtype="<f4").tobytes())
        handle.write(np.ascontiguousarray(gate_raw, dtype=np.uint8).tobytes())
        handle.write(np.ascontiguousarray(up_raw, dtype=np.uint8).tobytes())
        handle.write(np.ascontiguousarray(down_raw, dtype=np.uint8).tobytes())
        handle.write(np.asarray(reference, dtype="<f4").tobytes())


def run(args: argparse.Namespace) -> dict:
    model = args.model.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        from gguf import GGMLQuantizationType, GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "gguf-py is unavailable; set PYTHONPATH=third_party/llama.cpp/gguf-py "
            "and use a Python environment with numpy"
        ) from exc

    reader = GGUFReader(model, "r")
    names, tensors = select_tensors(reader, args.layer)
    for role in ("embedding", "gate", "up", "down"):
        if tensors[role].tensor_type != GGMLQuantizationType.Q8_0:
            raise ValueError(f"{names[role]} is not Q8_0: {tensors[role].tensor_type}")
    if tensors["norm"].tensor_type != GGMLQuantizationType.F32:
        raise ValueError(f"{names['norm']} is not F32: {tensors['norm'].tensor_type}")

    hidden = tensor_shape(tensors["norm"])[0]
    gate_shape = tensor_shape(tensors["gate"])
    up_shape = tensor_shape(tensors["up"])
    down_shape = tensor_shape(tensors["down"])
    if gate_shape != up_shape or gate_shape[0] != hidden or down_shape != (gate_shape[1], hidden):
        raise ValueError(f"unexpected FFN shapes norm={hidden} gate={gate_shape} up={up_shape} down={down_shape}")
    expanded = gate_shape[1]
    token_ids = tuple(args.token_ids)
    vocabulary = tensor_shape(tensors["embedding"])[1]
    if any(token < 0 or token >= vocabulary for token in token_ids):
        raise ValueError(f"token ids must be within [0,{vocabulary})")
    if args.slice_expanded % QK or args.slice_outputs % QK:
        raise ValueError("slice dimensions must be multiples of 32")
    if args.slice_expanded > expanded or args.slice_outputs > hidden:
        raise ValueError("slice dimensions exceed model dimensions")

    embedding_raw = tensors["embedding"].data[np.asarray(token_ids)]
    embedding_q, embedding_s = q8_planes(embedding_raw, hidden)
    input_rows = (embedding_q.astype(np.float32) * embedding_s[:, :, None]).reshape(len(token_ids), hidden)
    gamma = np.asarray(tensors["norm"].data, dtype=np.float32).copy()
    gate_q, gate_s = q8_planes(tensors["gate"].data, hidden)
    up_q, up_s = q8_planes(tensors["up"].data, hidden)
    down_q, down_s = q8_planes(tensors["down"].data, expanded)

    normalized = rms_norm(input_rows, gamma, args.epsilon)
    norm_q, norm_s = q8_quantize(normalized)
    gate_candidate = q8_linear(norm_q, norm_s, gate_q, gate_s)
    up_candidate = q8_linear(norm_q, norm_s, up_q, up_s)
    activated_candidate = silu_pwl(gate_candidate) * up_candidate
    activation_q, activation_s = q8_quantize(activated_candidate)
    candidate = q8_linear(activation_q, activation_s, down_q, down_s) + input_rows

    gate_reference = float_linear(normalized, gate_q, gate_s)
    up_reference = float_linear(normalized, up_q, up_s)
    activated_reference = exact_silu(gate_reference) * up_reference
    reference = float_linear(activated_reference, down_q, down_s) + input_rows
    full_metrics = metrics(candidate, reference)

    se = args.slice_expanded
    so = args.slice_outputs
    gate_raw_slice = np.ascontiguousarray(tensors["gate"].data[:se])
    up_raw_slice = np.ascontiguousarray(tensors["up"].data[:se])
    down_row_bytes = (se // QK) * GGUF_Q8_BLOCK_BYTES
    down_raw_slice = np.ascontiguousarray(tensors["down"].data[:so, :down_row_bytes])
    gate_sq, gate_ss = q8_planes(gate_raw_slice, hidden)
    up_sq, up_ss = q8_planes(up_raw_slice, hidden)
    down_sq, down_ss = q8_planes(down_raw_slice, se)
    gate_slice_ref = float_linear(normalized, gate_sq, gate_ss)
    up_slice_ref = float_linear(normalized, up_sq, up_ss)
    activation_slice_ref = exact_silu(gate_slice_ref) * up_slice_ref
    slice_reference = float_linear(activation_slice_ref, down_sq, down_ss) + input_rows[:, :so]
    gate_slice_candidate = q8_linear(norm_q, norm_s, gate_sq, gate_ss)
    up_slice_candidate = q8_linear(norm_q, norm_s, up_sq, up_ss)
    activation_slice_candidate = silu_pwl(gate_slice_candidate) * up_slice_candidate
    activation_slice_q, activation_slice_s = q8_quantize(activation_slice_candidate)
    slice_candidate = q8_linear(
        activation_slice_q, activation_slice_s, down_sq, down_ss
    ) + input_rows[:, :so]
    slice_metrics = metrics(slice_candidate, slice_reference)

    binary_path = output_dir / "real_weight_proxy_slice.bin"
    write_slice(
        binary_path, token_ids, args.epsilon, input_rows, gamma,
        gate_raw_slice, up_raw_slice, down_raw_slice, se, so, slice_reference,
    )
    source_hash = args.model_sha256 or sha256_file(model)
    summary = {
        "schema_version": 1,
        "decision": "REAL_WEIGHT_PROXY_NUMERIC_PASS" if (
            math.isfinite(full_metrics["relative_l2"]) and
            full_metrics["relative_l2"] <= args.max_relative_l2
        ) else "REAL_WEIGHT_PROXY_NUMERIC_FAIL",
        "scope": {
            "weights": "real RoboBrain2.5-4B GGUF Q8_0 tensors",
            "activations": "dequantized real token-embedding rows used as deterministic proxy inputs",
            "runtime_activation_capture": False,
            "cuda_loaded": False,
            "board_accessed": False,
            "bitstream_generated": False,
            "sd_modified": False,
        },
        "source": {
            "model_path": str(model),
            "model_sha256": source_hash,
            "layer": args.layer,
            "tensor_names": names,
            "tensor_shapes_gguf": {role: tensor_shape(tensor) for role, tensor in tensors.items()},
        },
        "case": {
            "token_ids": token_ids,
            "rows": len(token_ids),
            "hidden": hidden,
            "expanded": expanded,
            "epsilon": args.epsilon,
            "reference": "dequantized Q8 weights + FP32 activations + exact SiLU",
            "candidate": "RMSNorm/Q8 + Q8xQ8 GEMM + 0.25-step SiLU PWL/Q8 + Q8xQ8 GEMM + residual",
        },
        "full_proxy_metrics": full_metrics,
        "gate": {
            "max_relative_l2": args.max_relative_l2,
            "pass": full_metrics["relative_l2"] <= args.max_relative_l2,
        },
        "hls_slice": {
            "binary": binary_path.name,
            "binary_sha256": sha256_file(binary_path),
            "expanded": se,
            "outputs": so,
            "proxy_metrics": slice_metrics,
            "binary_format": "FFNVRW01 little-endian; header, token_ids, input_f32, gamma_f32, gate/up/down GGUF-Q8 rows, reference_f32",
        },
        "limitations": [
            "The proxy input is a real model embedding, not a captured activation at the selected FFN layer.",
            "Passing this gate validates real-weight numerical feasibility, not end-to-end model accuracy.",
            "The compact HLS case uses a real-weight slice; the full proxy metric is computed in Python/NumPy.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="read-only RoboBrain Q8_0 GGUF")
    parser.add_argument("--output-dir", type=Path, required=True, help="new append-only artifact directory")
    parser.add_argument("--model-sha256", help="trusted source hash; omit to hash the whole model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token-ids", type=lambda text: tuple(int(x) for x in text.split(",")), default=DEFAULT_TOKEN_IDS)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--slice-expanded", type=int, default=64)
    parser.add_argument("--slice-outputs", type=int, default=64)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except Exception as exc:
        print(f"REAL_WEIGHT_PROXY_ERROR {exc}", file=sys.stderr)
        raise
    print(json.dumps({
        "decision": result["decision"],
        "metrics": result["full_proxy_metrics"],
        "slice_sha256": result["hls_slice"]["binary_sha256"],
        "summary_sha256": result["summary_sha256"],
    }, ensure_ascii=False))
    raise SystemExit(0 if result["gate"]["pass"] else 1)
