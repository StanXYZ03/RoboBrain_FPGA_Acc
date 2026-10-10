#!/usr/bin/env python3
"""Evaluate a captured pre-norm FFN input with real RoboBrain Q8_0 weights.

The reference path dequantizes the stored GGUF Q8_0 weights and evaluates the
layer in FP32 with exact SiLU.  The candidate mirrors the proposed PL data path:
RMSNorm, GGML-compatible Q8_0 activation quantization, Q8xQ8 matrix products,
piecewise-linear SiLU, activation requantization, down projection, and residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

from parse_capture import HEADER, parse_capture


SCRIPT_DIR = Path(__file__).resolve().parent
FFN_SCRIPT_DIR = SCRIPT_DIR.parent / "pl_resident_ffn_hls" / "scripts"
sys.path.insert(0, str(FFN_SCRIPT_DIR))

from export_real_tensor_case import (  # noqa: E402
    exact_silu,
    float_linear,
    metrics,
    q8_linear,
    q8_planes,
    q8_quantize,
    rms_norm,
    select_tensors,
    silu_pwl,
    tensor_shape,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> tuple[dict, np.ndarray]:
    metadata = parse_capture(path)
    if metadata["tensor_name"] != "ffn_inp-0":
        raise ValueError(
            f"expected pre-norm layer-0 tensor ffn_inp-0, got {metadata['tensor_name']}"
        )
    if metadata["ne"][2:] != (1, 1):
        raise ValueError(f"unsupported capture rank/dimensions {metadata['ne']}")
    hidden, rows = metadata["ne"][:2]
    with path.open("rb") as handle:
        handle.seek(HEADER.size)
        values = np.fromfile(handle, dtype="<f4", count=hidden * rows)
    if values.size != hidden * rows:
        raise ValueError("truncated capture payload")
    return metadata, values.reshape(rows, hidden).copy()


def per_row_metrics(actual: np.ndarray, reference: np.ndarray) -> list[dict]:
    return [
        {"row": row, **metrics(actual[row], reference[row])}
        for row in range(actual.shape[0])
    ]


def run(args: argparse.Namespace) -> dict:
    model = args.model.resolve()
    capture = args.capture.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    try:
        from gguf import GGMLQuantizationType, GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "gguf-py is unavailable; set PYTHONPATH to llama.cpp/gguf-py"
        ) from exc

    capture_metadata, input_rows = load_rows(capture)
    reader = GGUFReader(model, "r")
    names, tensors = select_tensors(reader, args.layer)
    for role in ("gate", "up", "down"):
        if tensors[role].tensor_type != GGMLQuantizationType.Q8_0:
            raise ValueError(f"{names[role]} is not Q8_0: {tensors[role].tensor_type}")
    if tensors["norm"].tensor_type != GGMLQuantizationType.F32:
        raise ValueError(f"{names['norm']} is not F32: {tensors['norm'].tensor_type}")

    hidden = tensor_shape(tensors["norm"])[0]
    if input_rows.shape[1] != hidden:
        raise ValueError(f"capture hidden={input_rows.shape[1]}, layer hidden={hidden}")
    gate_shape = tensor_shape(tensors["gate"])
    up_shape = tensor_shape(tensors["up"])
    down_shape = tensor_shape(tensors["down"])
    if gate_shape != up_shape or gate_shape[0] != hidden or down_shape != (gate_shape[1], hidden):
        raise ValueError(
            f"unexpected FFN shapes norm={hidden} gate={gate_shape} "
            f"up={up_shape} down={down_shape}"
        )
    expanded = gate_shape[1]

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

    aggregate = metrics(candidate, reference)
    passed = math.isfinite(aggregate["relative_l2"]) and (
        aggregate["relative_l2"] <= args.max_relative_l2
    )
    source_hash = args.model_sha256 or sha256_file(model)
    summary = {
        "schema_version": 1,
        "decision": "RUNTIME_FFN_NUMERIC_PASS" if passed else "RUNTIME_FFN_NUMERIC_FAIL",
        "scope": {
            "weights": "real RoboBrain2.5-4B layer FFN GGUF Q8_0 tensors",
            "activations": "captured llama.cpp runtime ffn_inp-0 before RMSNorm",
            "candidate": "RMSNorm/Q8 + Q8xQ8 GEMM + SiLU-PWL/Q8 + Q8xQ8 GEMM + residual",
            "reference": "dequantized Q8 weights + FP32 activations + exact SiLU + residual",
            "end_to_end_accuracy_claim": False,
            "cuda_loaded": False,
            "board_accessed": False,
            "bitstream_generated": False,
            "sd_modified": False,
        },
        "source": {
            "model_path": str(model),
            "model_sha256": source_hash,
            "capture": capture_metadata,
            "layer": args.layer,
            "tensor_names": names,
            "tensor_shapes_gguf": {
                role: tensor_shape(tensor) for role, tensor in tensors.items()
            },
        },
        "case": {
            "rows": int(input_rows.shape[0]),
            "hidden": hidden,
            "expanded": expanded,
            "epsilon": args.epsilon,
        },
        "aggregate_metrics": aggregate,
        "per_row_metrics": per_row_metrics(candidate, reference),
        "gate": {
            "max_relative_l2": args.max_relative_l2,
            "pass": passed,
        },
        "limitations": [
            "This validates one captured layer-0 FFN input with two runtime rows.",
            "It does not yet validate all layers, prompts, vision inputs, or end-to-end task accuracy.",
            "The candidate is a NumPy mirror of the proposed PL path, not a board measurement.",
        ],
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-sha256", help="trusted model SHA-256; omit to hash model")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        result = run(parse_args())
    except Exception as exc:
        print(f"RUNTIME_FFN_NUMERIC_ERROR {exc}", file=sys.stderr)
        raise
    print(json.dumps({
        "decision": result["decision"],
        "metrics": result["aggregate_metrics"],
    }, ensure_ascii=False))
    raise SystemExit(0 if result["gate"]["pass"] else 1)
