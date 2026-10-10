#!/usr/bin/env python3
"""Generate the 3 dedicated C/RTL co-simulation fixtures for selector v4.

This is intentionally SEPARATE from ``generate_fixtures.py`` (184 cases): the
RTL co-simulation must not be handed the whole g++ differential set.  Only three
representative cases are produced, and every ``.exp.bin`` golden comes from the
frozen reference ``evaluation/hardware_regular_selector_fixedpoint.py``.

Cases
-----
``cosim_identity_4x4_K48``
    small identity bypass (``G=16 <= K=48``): no context, all-real output.
``cosim_16x16_K64_alltie``
    ``G=256``, ``K=64``: context path with every score tied (12345), so the
    anchor / candidate / centroid tie rules are all exercised.
``cosim_32x32_K96_mixedextreme``
    ``G=1024``, ``K=96``: context path with saturated int8 tokens (+127 / -128)
    and a score mix of the Q0.15 extremes 0 / 32767 plus random values.

Output format is the frozen ``TSIN``/``TSEX`` frame of ``generate_fixtures.py``
plus a 3-line ``manifest.txt`` (``name G grid_h grid_w budget local_cap``).
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import hardware_regular_selector_fixedpoint as fp  # noqa: E402

DEFAULT_OUT_DIR = Path("/tmp/robobrain_token_selector_cosim_fixtures_v4")


def standard_tokens(group_count: int) -> np.ndarray:
    """Deterministic mixed-pattern int8 tokens (same shape as g++ fixtures)."""
    group = np.arange(group_count, dtype=np.int64)[:, None, None]
    patch = np.arange(fp.MERGE_UNIT, dtype=np.int64)[None, :, None]
    feature = np.arange(fp.FEATURE_DIM, dtype=np.int64)[None, None, :]
    values = ((group * 7 + patch * 3 + feature * 13) % 256) - 128
    return values.astype(np.int8)


def extreme_tokens(group_count: int) -> np.ndarray:
    """Tokens saturated at both int8 extremes to stress the pooling datapath."""
    group = np.arange(group_count, dtype=np.int64)[:, None, None]
    patch = np.arange(fp.MERGE_UNIT, dtype=np.int64)[None, :, None]
    feature = np.arange(fp.FEATURE_DIM, dtype=np.int64)[None, None, :]
    index = (group * 13 + patch * 5 + feature * 7) % 251
    values = (index - 125).astype(np.int64)
    values = np.where(index % 11 == 0, 127, values)
    values = np.where(index % 13 == 0, -128, values)
    return values.astype(np.int8)


def random_scores(group_count: int, seed: int) -> np.ndarray:
    values = (
        np.arange(group_count, dtype=np.uint64) * 1103515245 + seed
    ) % 32768
    return values.astype(np.uint16)


def alltie_scores(group_count: int) -> np.ndarray:
    return np.full((group_count,), 12345, dtype=np.uint16)


def mixed_extreme_scores(group_count: int) -> np.ndarray:
    index = np.arange(group_count, dtype=np.int64)
    scores = np.where(index % 2 == 0, 32767, 0).astype(np.uint16)
    # Sprinkle ~14% full-range random values while keeping the extremes heavy.
    random_mask = (index % 7) == 3
    scores = np.where(
        random_mask,
        ((index * 48271 + 12345) % 32768).astype(np.uint16),
        scores,
    )
    return scores.astype(np.uint16)


def build_cases() -> list[dict]:
    return [
        {
            "name": "cosim_identity_4x4_K48",
            "grid_h": 4,
            "grid_w": 4,
            "budget": 48,
            "local_cap": 16,
            "scale": 1.0,
            "tokens": standard_tokens,
            "scores": lambda n: random_scores(n, 24680),
        },
        {
            "name": "cosim_16x16_K64_alltie",
            "grid_h": 16,
            "grid_w": 16,
            "budget": 64,
            "local_cap": 16,
            "scale": 0.5,
            "tokens": standard_tokens,
            "scores": alltie_scores,
        },
        {
            "name": "cosim_32x32_K96_mixedextreme",
            "grid_h": 32,
            "grid_w": 32,
            "budget": 96,
            "local_cap": 16,
            "scale": 2.0,
            "tokens": extreme_tokens,
            "scores": mixed_extreme_scores,
        },
    ]


def write_case(out_dir: Path, case: dict) -> str:
    group_count = case["grid_h"] * case["grid_w"]
    tokens = case["tokens"](group_count)
    scores = case["scores"](group_count)
    result = fp.select_fixedpoint(
        tokens,
        scores,
        case["grid_h"],
        case["grid_w"],
        case["budget"],
        local_cap=case["local_cap"],
        scale=case["scale"],
    )
    if result.status != fp.STATUS_OK:
        raise RuntimeError(f"{case['name']} golden status={result.status}")

    in_path = out_dir / f"{case['name']}.in.bin"
    with in_path.open("wb") as handle:
        handle.write(b"TSIN")
        handle.write(
            struct.pack(
                "<5i",
                group_count,
                case["grid_h"],
                case["grid_w"],
                case["budget"],
                case["local_cap"],
            )
        )
        handle.write(struct.pack("<f", case["scale"]))
        handle.write(np.ascontiguousarray(tokens, dtype=np.int8).tobytes())
        handle.write(np.ascontiguousarray(scores, dtype="<u2").tobytes())

    exp_path = out_dir / f"{case['name']}.exp.bin"
    with exp_path.open("wb") as handle:
        handle.write(b"TSEX")
        handle.write(
            struct.pack(
                "<4i",
                result.status,
                result.output_count,
                result.real_count,
                result.context_count,
            )
        )
        handle.write(struct.pack("<f", result.scale_out))
        handle.write(np.ascontiguousarray(result.tokens_out, dtype=np.int8).tobytes())
        handle.write(np.ascontiguousarray(result.group_index, dtype="<u2").tobytes())
        handle.write(np.ascontiguousarray(result.source_type, dtype=np.uint8).tobytes())
        handle.write(np.ascontiguousarray(result.region_id, dtype=np.uint8).tobytes())
        handle.write(np.ascontiguousarray(result.valid, dtype=np.uint8).tobytes())

    print(
        f"{case['name']}: G={group_count} K={case['budget']} "
        f"out={result.output_count} real={result.real_count} "
        f"context={result.context_count} scale={result.scale_out}"
    )
    return (
        f"{case['name']} {group_count} {case['grid_h']} {case['grid_w']} "
        f"{case['budget']} {case['local_cap']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [write_case(out_dir, case) for case in build_cases()]
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} cosim fixtures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
