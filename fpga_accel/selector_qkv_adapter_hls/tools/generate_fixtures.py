#!/usr/bin/env python3
"""Generate the frozen real-block1 fixtures for the adapter / bias-add kernels.

Reads only three real BF16 tensors from the real Qwen3-VL safetensors shard
through an explicit header + byte-range reader (the full model is never loaded):

    model.visual.blocks.1.norm1.weight   [1024]
    model.visual.blocks.1.norm1.bias     [1024]
    model.visual.blocks.1.attn.qkv.bias  [3072]

The adapter fixture stores a deterministic synthetic Selector segment-0 int8
payload plus the Python golden Q8_0 payload/scale bytes; the C++ testbench
byte-compares its kernel output against them. The bias fixture stores a
synthetic Q8Q8-output-like FP32 tensor plus the real `qkv.bias` and the exact
FP32 sum. Nothing here is a captured runtime activation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import selector_qkv_golden as golden  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / "models" / "Qwen3-VL-4B-Instruct"
DEFAULT_SHARD = "model-00002-of-00002.safetensors"
ADAPTER_ROWS = 192  # K = 48 selector groups (legal M in {192, 256, 384})
# The formal qkv bias-add case this round is M = 192/256/384, N = 3072; the
# frozen fixture uses the largest legal M = 384 as the boundary case.
BIAS_FIXTURE_ROWS = 384
ADAPTER_SEED = 20260928

TENSOR_NAMES = (
    "model.visual.blocks.1.norm1.weight",
    "model.visual.blocks.1.norm1.bias",
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


def decode_storage(raw: bytes, dtype: str) -> np.ndarray:
    if dtype == "BF16":
        widened = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return widened.view(np.float32)
    spec = _STORAGE_DTYPES.get(dtype)
    if spec is None:
        raise ValueError(f"unsupported safetensors dtype: {dtype}")
    return np.frombuffer(raw, dtype=spec[0])


def load_safetensors_tensors(path: Path,
                             names: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    before = path.stat()
    tensors: dict[str, Any] = {}
    with open(path, "rb") as handle:
        raw_header_len = handle.read(8)
        if len(raw_header_len) != 8:
            raise ValueError(f"{path} is not a safetensors file")
        header_len = struct.unpack("<Q", raw_header_len)[0]
        header_bytes = handle.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        data_start = 8 + header_len
        for name in names:
            if name not in header:
                raise KeyError(f"{name} not present in {path.name}")
            info = header[name]
            begin, end = (int(v) for v in info["data_offsets"])
            handle.seek(data_start + begin)
            raw = handle.read(end - begin)
            if len(raw) != end - begin:
                raise IOError(f"short read for {name}")
            array = decode_storage(raw, info["dtype"]).reshape(
                tuple(int(v) for v in info["shape"]))
            tensors[name] = {
                "array": array,
                "dtype": info["dtype"],
                "shape": [int(v) for v in info["shape"]],
                "data_offsets": [begin, end],
                "raw_bytes": len(raw),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "fp32_sha256": golden.tensor_sha256(array, "<f4"),
                "min": float(np.min(array.astype(np.float64))),
                "max": float(np.max(array.astype(np.float64))),
            }
    after = path.stat()
    shard_info = {
        "shard": path.name,
        "shard_bytes": int(before.st_size),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
        "read_only_unchanged": bool(
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns),
        "requested_tensors": list(names),
    }
    return tensors, shard_info


def synthetic_block0_hidden(seed: int, rows: int) -> np.ndarray:
    """Deterministic synthetic block0 residual hidden (never a real activation)."""
    rng = np.random.default_rng(int(seed))
    base = rng.standard_normal((rows, golden.HIDDEN), dtype=np.float32)
    features = np.arange(golden.HIDDEN, dtype=np.float32)
    feature_bias = (
        np.float32(0.35) * np.sin(np.float32(2.0 * math.pi) * features / np.float32(golden.HIDDEN))
        + np.float32(0.15) * np.cos(np.float32(6.0 * math.pi) * features / np.float32(golden.HIDDEN))
    ).astype(np.float32)
    row_scale = (np.float32(0.9) + np.float32(0.2) * (
        np.arange(rows, dtype=np.float32) / np.float32(max(rows - 1, 1)))).astype(np.float32)
    values = (base * row_scale[:, None] * np.float32(0.8) + feature_bias[None, :]).astype(np.float32)
    return np.ascontiguousarray(values)


def synthetic_q8q8_output(seed: int, rows: int, cols: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    values = rng.standard_normal((rows, cols), dtype=np.float32) * np.float32(0.5)
    return np.ascontiguousarray(values.astype(np.float32))


def build_fixtures(out_dir: Path,
                   model_dir: Path = DEFAULT_MODEL_DIR,
                   shard_name: str = DEFAULT_SHARD) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = model_dir / shard_name
    tensors, shard_info = load_safetensors_tensors(shard_path, TENSOR_NAMES)
    gamma = np.asarray(tensors["model.visual.blocks.1.norm1.weight"]["array"], dtype=np.float32)
    beta = np.asarray(tensors["model.visual.blocks.1.norm1.bias"]["array"], dtype=np.float32)
    qkv_bias = np.asarray(tensors["model.visual.blocks.1.attn.qkv.bias"]["array"], dtype=np.float32)
    if gamma.shape != (golden.HIDDEN,) or beta.shape != (golden.HIDDEN,):
        raise ValueError("unexpected norm1 shape")
    if qkv_bias.shape != (golden.QKV_COLS,):
        raise ValueError("unexpected qkv.bias shape")

    # --- adapter fixture (real norm1 gamma/beta, synthetic selector int8) ---
    hidden = synthetic_block0_hidden(ADAPTER_SEED, ADAPTER_ROWS)
    amax = float(np.max(np.abs(hidden.astype(np.float64))))
    scale = golden.selector_scale_from_amax(amax)
    q_int8, saturated = golden.selector_quantize(hidden, scale)
    fused = golden.fused_layernorm(q_int8, gamma, beta, scale)
    direct = golden.direct_layernorm(q_int8, gamma, beta, scale)
    payload, scale_bits, d_values = golden.quantize_q8_0(fused)
    direct_payload, direct_scale_bits, _ = golden.quantize_q8_0(direct)
    dequant = golden.dequantize_q8_0(payload, scale_bits)

    adapter_path = out_dir / "real_block1_adapter_fixture.bin"
    golden.write_adapter_fixture(adapter_path, q_int8, gamma, beta, scale,
                                 golden.EPSILON, payload, scale_bits)

    adapter_metrics = {
        "fused_vs_direct": golden.metrics(fused, direct),
        "dequant_vs_direct": golden.metrics(dequant, direct),
        "fused_golden_payload_vs_direct_quant_payload": {
            "payload_byte_mismatch": int(np.count_nonzero(
                payload.astype(np.int16) - direct_payload.astype(np.int16))),
            "scale_value_mismatch": int(np.count_nonzero(
                scale_bits.astype(np.uint16) != direct_scale_bits.astype(np.uint16))),
            "payload_bytes": int(payload.size),
            "scale_values": int(scale_bits.size),
        },
    }

    # --- bias fixture (real qkv.bias, synthetic Q8Q8-like FP32 output) ---
    bias_input = synthetic_q8q8_output(ADAPTER_SEED + 1, BIAS_FIXTURE_ROWS,
                                       golden.QKV_COLS)
    bias_expected = (bias_input + qkv_bias[None, :]).astype(np.float32)
    bias_path = out_dir / "real_block1_bias_fixture.bin"
    golden.write_bias_fixture(bias_path, bias_input, qkv_bias, bias_expected)

    manifest = {
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "model_dir": str(model_dir),
        "shard_info": shard_info,
        "tensors": {
            name: {key: value for key, value in info.items() if key != "array"}
            for name, info in tensors.items()
        },
        "adapter_fixture": {
            "path": adapter_path.name,
            "sha256": golden.sha256_file(adapter_path),
            "bytes": adapter_path.stat().st_size,
            "rows_M": ADAPTER_ROWS,
            "legal_rows_M": list(golden.LEGAL_ADAPTER_ROWS),
            "max_rows": golden.MAX_ROWS,
            "selector_groups_K": ADAPTER_ROWS // 4,
            "hidden": golden.HIDDEN,
            "block": golden.BLOCK,
            "blocks_per_row": golden.BLOCKS_PER_ROW,
            "global_scale_s": scale,
            "epsilon": golden.EPSILON,
            "selector_levels": 128.0,
            "selector_saturated_values": saturated,
            "hidden_amax": amax,
            "input_is_synthetic_activation": True,
            "gamma_beta_source": "real model.visual.blocks.1.norm1.weight/bias (BF16->FP32)",
            "expected_source": "Python fused golden (selector_qkv_golden.py)",
            "metrics": adapter_metrics,
        },
        "bias_fixture": {
            "path": bias_path.name,
            "sha256": golden.sha256_file(bias_path),
            "bytes": bias_path.stat().st_size,
            "rows_M": BIAS_FIXTURE_ROWS,
            "formal_rows_M": list(golden.LEGAL_ADAPTER_ROWS),
            "cols_N": golden.QKV_COLS,
            "bias_source": "real model.visual.blocks.1.attn.qkv.bias (BF16->FP32)",
            "input_is_synthetic_q8q8_like_output": True,
            "expected": "element-wise FP32 input + real qkv.bias",
        },
        "case_metrics": {
            "adapter_rows_M": ADAPTER_ROWS,
            "adapter_legal_rows_M": list(golden.LEGAL_ADAPTER_ROWS),
            "bias_formal_rows_M": list(golden.LEGAL_ADAPTER_ROWS),
            "bias_formal_cols_N": golden.QKV_COLS,
            "adapter_selector_groups_K": ADAPTER_ROWS // 4,
            "adapter_fused_vs_direct": adapter_metrics["fused_vs_direct"],
            "adapter_dequant_vs_direct": adapter_metrics["dequant_vs_direct"],
        },
    }
    manifest_path = out_dir / "fixture_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "artifacts" / "token_selector" /
                        "2026-09-28-selector-qkv-adapter-csim")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--shard", default=DEFAULT_SHARD)
    args = parser.parse_args(argv)
    manifest = build_fixtures(args.out_dir, args.model_dir, args.shard)
    print(json.dumps({
        "adapter_fixture": manifest["adapter_fixture"]["path"],
        "adapter_fixture_sha256": manifest["adapter_fixture"]["sha256"],
        "bias_fixture": manifest["bias_fixture"]["path"],
        "bias_fixture_sha256": manifest["bias_fixture"]["sha256"],
        "shard_read_only_unchanged": manifest["shard_info"]["read_only_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
