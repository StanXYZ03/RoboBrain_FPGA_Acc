#!/usr/bin/env python3
"""Generate compact C++-readable fixtures for the ``integer_selector_scorer`` HLS kernel.

The golden is **re-used, never copied**: every ``score_q`` / ``semantic_q16`` /
``l2_int`` / min-max value written here comes from
``evaluation.selector_integer_scorer_audit.integer_chain`` (and its frozen
``floor_sqrt_int64`` / ``round_half_away_div`` / ``minmax_normalize_q15``
primitives).  This tool only serialises those values into the documented binary
format (see ``tools/fixture_format.md``).

Fixtures
--------
* ``wv_q.bin``                 byte copy of the accepted seed-20261003 Wv bundle
* ``full_grounding.bin``       real fixture (correct + wrong prompt), kind 0
* ``full_pointing.bin``        real fixture (correct + wrong prompt), kind 0
* ``synthetic_zero.bin``       all-zero group means, kind 1
* ``synthetic_all_equal.bin``  all-equal nonzero group means, kind 1
* ``synthetic_g1.bin``         G = 1, kind 1
* ``synthetic_mixed_sign.bin`` mixed-sign deterministic rows, kind 1
* ``stress_g400.bin``          streaming xorshift64* deterministic G = 400, kind 2
* ``primitives.bin``           floor-sqrt / round-half-away ties / min-max, kind 3

No network, no GPU, no training, no model load.  CPU-only and read-only w.r.t.
every accepted upstream artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.selector_integer_scorer_audit import (
    floor_sqrt_int64,
    integer_chain,
    minmax_normalize_q15,
    round_half_away_div,
)

DATE = "2026-10-03"
ARTIFACT_DIR = ROOT / "artifacts/token_selector/2026-10-03-integer-selector-scorer-hls"
FIXTURE_DIR = ARTIFACT_DIR / "fixtures"
REAL_FIXTURE_DIR = (
    ROOT / "artifacts/token_selector/selector_integer_scorer_audit/20261003"
)
WV_PATH = (
    ROOT
    / "artifacts/token_selector/selector_wv_scale_regularization/20261003/Wv_int8_rowmajor.bin"
)
WV_SCALE_PATH = (
    ROOT
    / "artifacts/token_selector/selector_wv_scale_regularization/20261003/Wv_per_tensor_scale_f32le.bin"
)
WV_SHA256 = "126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b"
WV_SCALE_SHA256 = "93585a3bd2d584a612b40d827f8d15f43c87acdc764d7e217d99df473428b13f"
SEED = 20261003
EMBED_DIM = 1024
QUERY_DIM = 128
WV_ROWS = 128
REAL_TASKS = ("grounding", "pointing")

FIXTURE_HEADER = struct.Struct("<4sIIIIIIIQQQQQQ6Q")
PRIMITIVE_HEADER = struct.Struct("<4sIIIIIQ4Q")

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
XORSHIFT_MUL = 0x2545F4914F6CDD1D


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fnv1a64(data: bytes) -> int:
    value = FNV_OFFSET
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & MASK64
    return value


def _combine_hash(parts: Sequence[bytes]) -> int:
    return fnv1a64(b"".join(parts))


def load_wv_int8() -> np.ndarray:
    actual = sha256_file(WV_PATH)
    if actual != WV_SHA256:
        raise SystemExit(f"accepted Wv SHA mismatch: {actual} != {WV_SHA256}")
    tensor = np.frombuffer(WV_PATH.read_bytes(), dtype=np.int8)
    if tensor.size != WV_ROWS * EMBED_DIM:
        raise SystemExit(f"unexpected Wv size {tensor.size}")
    return tensor.reshape(WV_ROWS, EMBED_DIM).copy()


def golden_chain(group_mean_q: np.ndarray, query_q: np.ndarray, grid_h: int,
                 grid_w: int) -> dict[str, np.ndarray]:
    """Run the frozen integer golden and return the arrays this tool stores."""
    chain = integer_chain(
        group_mean_q.astype(np.int64),
        query_q.astype(np.int64),
        WV_CACHE["wv_q"],
        int(grid_h),
        int(grid_w),
        budgets=(),
    )
    return {
        "semantic_q16": chain["semantic_q16"].astype(np.int64),
        "l2_int": chain["l2_int"].astype(np.int64),
        "semantic_norm_q15": chain["semantic_norm_q15"].astype(np.int64),
        "l2_norm_q15": chain["l2_norm_q15"].astype(np.int64),
        "score_q": chain["score_q"].astype(np.uint16),
        "dot": chain["dot"].astype(np.int64),
    }


def role_payload(query_q: np.ndarray, golden: Mapping[str, np.ndarray]) -> bytes:
    return b"".join(
        [
            np.asarray(query_q, dtype=np.int8).tobytes(),
            np.asarray(golden["score_q"], dtype="<u2").tobytes(),
            np.asarray(golden["semantic_q16"], dtype="<i4").tobytes(),
            np.asarray(golden["l2_int"], dtype="<u2").tobytes(),
            np.asarray(golden["semantic_norm_q15"], dtype="<u2").tobytes(),
            np.asarray(golden["l2_norm_q15"], dtype="<u2").tobytes(),
        ]
    )


def build_fixture(
    *,
    kind: int,
    group_count: int,
    grid_h: int,
    grid_w: int,
    group_mean_bytes: bytes,
    roles: Sequence[tuple[np.ndarray, Mapping[str, np.ndarray]]],
    generator: int = 0,
    generator_seed: int = 0,
) -> bytes:
    score_parts = [np.asarray(g["score_q"], dtype="<u2").tobytes() for _, g in roles]
    semantic_parts = [
        np.asarray(g["semantic_q16"], dtype="<i4").tobytes() for _, g in roles
    ]
    l2_parts = [np.asarray(g["l2_int"], dtype="<u2").tobytes() for _, g in roles]
    payload = bytearray()
    if kind in (0, 1):
        payload += group_mean_bytes
    for query_q, golden in roles:
        payload += role_payload(query_q, golden)
    header = FIXTURE_HEADER.pack(
        b"ISF1",
        1,
        int(kind),
        int(group_count),
        int(grid_h),
        int(grid_w),
        len(roles),
        int(generator),
        int(generator_seed),
        _combine_hash(score_parts),
        _combine_hash(semantic_parts),
        _combine_hash(l2_parts),
        fnv1a64(group_mean_bytes),
        len(payload),
        0,
        0,
        0,
        0,
        0,
        0,
    )
    return bytes(header) + bytes(payload)


def build_primitives() -> bytes:
    sqrt_inputs = [
        0,
        1,
        2,
        3,
        4,
        5,
        8,
        15,
        16,
        17,
        24,
        25,
        26,
        10**6,
        (10**6) ** 2,
        (10**6) ** 2 + 1,
        (1 << 62) - 1,
        1 << 62,
        (1 << 63) - 1,
        35468046088798208,
        9223372036854775807 // 4,
    ]
    sqrt_inputs += [int(v) for v in np.random.default_rng(SEED).integers(
        0, 1 << 62, size=24, dtype=np.uint64)]

    # Positive and negative exact .5 ties plus ordinary and large rationals.
    div_pairs = [
        (1, 2), (-1, 2), (3, 2), (-3, 2), (5, 2), (-5, 2), (7, 2), (-7, 2),
        (3, 6), (-3, 6), (9, 6), (-9, 6), (1, 6), (-1, 6), (5, 6), (-5, 6),
        (1, 3), (-1, 3), (2, 3), (-2, 3), (4, 3), (-4, 3), (5, 3), (-5, 3),
        (6, 3), (-6, 3), (0, 5), (0, 1),
        (17873661021126656, 201326592), (-17873661021126656, 201326592),
        (10249239396352, 23640732), (-10249239396352, 23640732),
        (1, 1), (-1, 1), (32767, 2), (-32767, 2),
    ]
    rng = np.random.default_rng(SEED + 1)
    for _ in range(16):
        den = int(rng.integers(1, 1 << 28))
        num = int(rng.integers(-(1 << 55), 1 << 55))
        div_pairs.append((num, den))

    minmax_arrays = [
        [5, 5, 5, 5],
        [0, 1, 2, 3],
        [-7, -3, 0, 9],
        [-128, 127],
        [10**9, 10**9 + 1],
        [-(10**12), 0, 10**12],
    ]
    minmax_arrays += [
        [int(v) for v in rng.integers(-(10**9), 10**9, size=8)]
        for _ in range(6)
    ]

    sqrt_entries = b"".join(
        struct.pack("<QQ", int(v), int(floor_sqrt_int64(int(v)))) for v in sqrt_inputs
    )
    div_entries = b"".join(
        struct.pack("<qqq", int(n), int(d), int(round_half_away_div(int(n), int(d))))
        for n, d in div_pairs
    )
    minmax_entries = bytearray()
    total_values = 0
    for values in minmax_arrays:
        array = np.asarray(values, dtype=np.int64)
        low = int(array.min())
        high = int(array.max())
        expected = minmax_normalize_q15(array)
        minmax_entries += struct.pack("<qqI", low, high, array.size)
        minmax_entries += array.astype("<i8").tobytes()
        minmax_entries += expected.astype("<u2").tobytes()
        total_values += int(array.size)
    payload = sqrt_entries + div_entries + bytes(minmax_entries)
    header = PRIMITIVE_HEADER.pack(
        b"ISP1",
        1,
        len(sqrt_inputs),
        len(div_pairs),
        len(minmax_arrays),
        total_values,
        len(payload),
        0,
        0,
        0,
        0,
    )
    return bytes(header) + payload


# ---------------------------------------------------------------------------
# deterministic xorshift64* generator (must match the C++ testbench)
# ---------------------------------------------------------------------------
def xorshift_bytes(seed: int, count: int) -> bytes:
    state = seed & MASK64
    out = bytearray(count)
    for index in range(count):
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        product = (state * XORSHIFT_MUL) & MASK64
        out[index] = (product >> 56) & 0xFF
    return bytes(out)


def xorshift_bytes_np(seed: int, count: int) -> np.ndarray:
    """Vectorised-identical generator used for the large G=400 fixture."""
    state = np.uint64(seed & MASK64)
    values = np.empty(count, dtype=np.uint8)
    mask = np.uint64(MASK64)
    mul = np.uint64(XORSHIFT_MUL)
    for index in range(count):
        state ^= (state << np.uint64(13)) & mask
        state ^= state >> np.uint64(7)
        state ^= (state << np.uint64(17)) & mask
        with np.errstate(over="ignore"):
            product = (state * mul) & mask
        values[index] = np.uint8((product >> np.uint64(56)) & np.uint64(0xFF))
    return values


def signed_int8(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int16).astype(np.int8)


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------
WV_CACHE: dict[str, np.ndarray] = {}


def build_real(task: str) -> tuple[str, bytes, dict[str, Any]]:
    with np.load(REAL_FIXTURE_DIR / f"fixture_{task}.npz", allow_pickle=False) as archive:
        group_mean_q = archive["group_mean_q"].astype(np.int8)
        grid_h = int(archive["grid_h"])
        grid_w = int(archive["grid_w"])
        group_count = int(group_mean_q.shape[0])
        roles = []
        for role in ("correct", "wrong"):
            query_q = archive[f"query_q_{role}"].astype(np.int8)
            golden = golden_chain(group_mean_q, query_q, grid_h, grid_w)
            for key in (
                "semantic_q16",
                "l2_int",
                "semantic_norm_q15",
                "l2_norm_q15",
            ):
                stored = archive[f"{key}_{role}"].astype(np.int64)
                if not np.array_equal(golden[key], stored):
                    raise SystemExit(f"{task}/{role}: regenerated {key} != stored NPZ")
            if not np.array_equal(
                golden["score_q"], archive[f"score_q_{role}"].astype(np.uint16)
            ):
                raise SystemExit(f"{task}/{role}: regenerated score_q != stored NPZ")
            roles.append((query_q, golden))
    data = build_fixture(
        kind=0,
        group_count=group_count,
        grid_h=grid_h,
        grid_w=grid_w,
        group_mean_bytes=group_mean_q.tobytes(),
        roles=roles,
    )
    meta = {
        "kind": 0,
        "task": task,
        "group_count": group_count,
        "grid": [grid_h, grid_w],
        "role_count": 2,
        "source_npz": f"artifacts/token_selector/selector_integer_scorer_audit/20261003/fixture_{task}.npz",
    }
    return f"full_{task}.bin", data, meta


def build_synthetic_zero() -> tuple[str, bytes, dict[str, Any]]:
    group_count = 8
    group_mean_q = np.zeros((group_count, EMBED_DIM), dtype=np.int8)
    query_q = np.asarray([((d * 7) % 255) - 127 for d in range(QUERY_DIM)], dtype=np.int8)
    golden = golden_chain(group_mean_q, query_q, 2, 4)
    data = build_fixture(
        kind=1, group_count=group_count, grid_h=2, grid_w=4,
        group_mean_bytes=group_mean_q.tobytes(), roles=[(query_q, golden)],
    )
    return "synthetic_zero.bin", data, {
        "kind": 1, "case": "zero", "group_count": group_count, "expected": "all score_q = 0",
    }


def build_synthetic_all_equal() -> tuple[str, bytes, dict[str, Any]]:
    group_count = 8
    group_mean_q = np.full((group_count, EMBED_DIM), 37, dtype=np.int8)
    query_q = np.asarray([((d * 11) % 255) - 127 for d in range(QUERY_DIM)], dtype=np.int8)
    golden = golden_chain(group_mean_q, query_q, 2, 4)
    data = build_fixture(
        kind=1, group_count=group_count, grid_h=2, grid_w=4,
        group_mean_bytes=group_mean_q.tobytes(), roles=[(query_q, golden)],
    )
    return "synthetic_all_equal.bin", data, {
        "kind": 1, "case": "all_equal", "group_count": group_count, "expected": "all score_q = 0",
    }


def build_synthetic_g1() -> tuple[str, bytes, dict[str, Any]]:
    group_count = 1
    row = np.asarray([((d * 29) % 255) - 127 for d in range(EMBED_DIM)], dtype=np.int8)
    group_mean_q = row.reshape(1, EMBED_DIM)
    query_q = np.asarray([((d * 5) % 255) - 127 for d in range(QUERY_DIM)], dtype=np.int8)
    golden = golden_chain(group_mean_q, query_q, 1, 1)
    data = build_fixture(
        kind=1, group_count=group_count, grid_h=1, grid_w=1,
        group_mean_bytes=group_mean_q.tobytes(), roles=[(query_q, golden)],
    )
    return "synthetic_g1.bin", data, {
        "kind": 1, "case": "g1", "group_count": group_count, "expected": "single group, score_q = 0",
    }


def build_synthetic_mixed_sign() -> tuple[str, bytes, dict[str, Any]]:
    group_count = 16
    rows = [
        [(((g * 131 + d * 17) % 255) - 127) for d in range(EMBED_DIM)]
        for g in range(group_count)
    ]
    group_mean_q = np.asarray(rows, dtype=np.int8)
    query_q = np.asarray([((d * 13) % 255) - 127 for d in range(QUERY_DIM)], dtype=np.int8)
    golden = golden_chain(group_mean_q, query_q, 4, 4)
    dot = golden["dot"]
    data = build_fixture(
        kind=1, group_count=group_count, grid_h=4, grid_w=4,
        group_mean_bytes=group_mean_q.tobytes(), roles=[(query_q, golden)],
    )
    return "synthetic_mixed_sign.bin", data, {
        "kind": 1,
        "case": "mixed_sign",
        "group_count": group_count,
        "dot_positive_groups": int(np.count_nonzero(dot > 0)),
        "dot_negative_groups": int(np.count_nonzero(dot < 0)),
        "semantic_negative_groups": int(np.count_nonzero(golden["semantic_q16"] < 0)),
        "semantic_positive_groups": int(np.count_nonzero(golden["semantic_q16"] > 0)),
    }


def build_stress_g400() -> tuple[str, bytes, dict[str, Any]]:
    group_count = 400
    generator_seed = SEED
    raw = xorshift_bytes_np(generator_seed, group_count * EMBED_DIM + QUERY_DIM)
    group_mean_q = signed_int8(raw[: group_count * EMBED_DIM].tobytes()).reshape(
        group_count, EMBED_DIM
    )
    query_q = signed_int8(raw[group_count * EMBED_DIM :].tobytes())
    golden = golden_chain(group_mean_q, query_q, 20, 20)
    data = build_fixture(
        kind=2,
        group_count=group_count,
        grid_h=20,
        grid_w=20,
        group_mean_bytes=group_mean_q.tobytes(),
        roles=[(query_q, golden)],
        generator=1,
        generator_seed=generator_seed,
    )
    return "stress_g400.bin", data, {
        "kind": 2,
        "case": "stress_g400",
        "group_count": group_count,
        "generator": "xorshift64*",
        "generator_seed": generator_seed,
        "dot_positive_groups": int(np.count_nonzero(golden["dot"] > 0)),
        "dot_negative_groups": int(np.count_nonzero(golden["dot"] < 0)),
        "full_rows_embedded": False,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def generate(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wv_q = load_wv_int8()
    WV_CACHE["wv_q"] = wv_q.astype(np.int64)

    wv_copy = out_dir / "wv_q.bin"
    wv_copy.write_bytes(wv_q.tobytes())

    (out_dir / "FORMAT.md").write_text(
        (Path(__file__).resolve().parent / "fixture_format.md").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )

    builders = [
        lambda: build_real("grounding"),
        lambda: build_real("pointing"),
        build_synthetic_zero,
        build_synthetic_all_equal,
        build_synthetic_g1,
        build_synthetic_mixed_sign,
        build_stress_g400,
    ]
    records: dict[str, Any] = {}
    for builder in builders:
        name, data, meta = builder()
        path = out_dir / name
        path.write_bytes(data)
        meta = dict(meta)
        meta["file"] = name
        meta["bytes"] = len(data)
        meta["sha256"] = hashlib.sha256(data).hexdigest()
        meta["score_fnv1a64"] = FIXTURE_HEADER.unpack_from(data, 0)[9]
        meta["group_mean_fnv1a64"] = FIXTURE_HEADER.unpack_from(data, 0)[12]
        records[name] = meta

    primitives = build_primitives()
    prim_path = out_dir / "primitives.bin"
    prim_path.write_bytes(primitives)
    records["primitives.bin"] = {
        "file": "primitives.bin",
        "kind": 3,
        "bytes": len(primitives),
        "sha256": hashlib.sha256(primitives).hexdigest(),
        "n_sqrt": PRIMITIVE_HEADER.unpack_from(primitives, 0)[2],
        "n_div": PRIMITIVE_HEADER.unpack_from(primitives, 0)[3],
        "n_minmax": PRIMITIVE_HEADER.unpack_from(primitives, 0)[4],
        "minmax_total_values": PRIMITIVE_HEADER.unpack_from(primitives, 0)[5],
    }

    chain_source = ROOT / "evaluation/selector_integer_scorer_audit.py"
    manifest = {
        "schema_version": "1.0",
        "date": DATE,
        "generator": "fpga_accel/integer_selector_scorer_hls/tools/generate_fixtures.py",
        "format_doc": "fixtures/FORMAT.md",
        "golden": {
            "module": "evaluation/selector_integer_scorer_audit.py",
            "entry": "integer_chain",
            "primitives": [
                "floor_sqrt_int64",
                "round_half_away_div",
                "minmax_normalize_q15",
            ],
            "sha256": sha256_file(chain_source),
        },
        "wv": {
            "source_path": str(WV_PATH.relative_to(ROOT)),
            "source_sha256": WV_SHA256,
            "copied_file": "wv_q.bin",
            "copied_sha256": sha256_file(wv_copy),
            "shape": [WV_ROWS, EMBED_DIM],
            "dtype": "int8 row-major",
            "scale_path": str(WV_SCALE_PATH.relative_to(ROOT)),
            "scale_sha256": WV_SCALE_SHA256,
            "note": "reused by SHA, never retrained; scale is a positive common factor and is omitted",
        },
        "generator_spec": {
            "name": "xorshift64*",
            "shift": [13, 7, 17],
            "multiplier": hex(XORSHIFT_MUL),
            "byte_lane": "(state*multiplier mod 2^64) >> 56, reinterpreted as signed int8",
        },
        "fixtures": records,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    started = time.time()
    manifest = generate(args.out_dir)
    print(f"fixtures -> {args.out_dir} ({time.time() - started:.2f}s)")
    for name, meta in manifest["fixtures"].items():
        print(
            f"  {name:28s} bytes={meta['bytes']:8d} sha256={meta['sha256'][:16]} "
            f"score_fnv=0x{meta.get('score_fnv1a64', 0):016x}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
