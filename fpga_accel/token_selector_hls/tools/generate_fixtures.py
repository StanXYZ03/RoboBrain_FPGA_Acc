#!/usr/bin/env python3
"""Generate frozen binary fixtures for the fixed-point selector C-simulation.

The golden data comes exclusively from
``evaluation/hardware_regular_selector_fixedpoint.py`` (CPU, no model, no GPU).

Two families are produced into the same directory:

* hand-picked boundary/compression/degeneracy/rejection cases (``Case`` table),
* ``RANDOM_CASE_COUNT`` deterministic random cases (``randNNN``) used as a
  differential test of the refactored HLS kernel against the frozen Python
  golden.

Binary formats (little endian)
------------------------------
``<name>.in.bin``
    magic  "TSIN" (4 bytes)
    int32  group_count, grid_h, grid_w, budget, local_cap
    float  scale
    int8   tokens_q[group_count * 4 * 1024]
    uint16 score_q[group_count]              (Q0.15, [0, 32767])

``<name>.exp.bin``
    magic  "TSEX" (4 bytes)
    int32  status, output_count, real_count, context_count
    float  scale_out
    if status == 0:
        int8   tokens_out[output_count * 4 * 1024]
        uint16 group_index[output_count]
        uint8  source_type[output_count]
        uint8  region_id[output_count]
        uint8  valid[output_count]

``manifest.txt``    one ``name G grid_h grid_w budget local_cap`` line per case.
``layout.txt``      one ``budget total o0..o4 s0..s4`` line per layout budget.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import hardware_regular_selector_fixedpoint as fp  # noqa: E402

DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "token_selector" / "2026-09-27-fixedpoint-selector-fixtures"

# Frozen deterministic random differential cases (>= 100 required).
RANDOM_CASE_COUNT = 150
RANDOM_SEED = 20260927


@dataclass(frozen=True)
class Case:
    name: str
    grid_h: int
    grid_w: int
    budget: int
    score_mode: str = "rand"
    group_override: int | None = None
    scale: float = 0.03125
    local_cap: int = fp.LOCAL_CANDIDATE_CAP
    seed: int = 12345

    @property
    def group_count(self) -> int:
        if self.group_override is not None:
            return int(self.group_override)
        return self.grid_h * self.grid_w


def make_tokens(group_count: int) -> np.ndarray:
    group = np.arange(group_count, dtype=np.int64)[:, None, None]
    patch = np.arange(fp.MERGE_UNIT, dtype=np.int64)[None, :, None]
    feature = np.arange(fp.FEATURE_DIM, dtype=np.int64)[None, None, :]
    values = ((group * 7 + patch * 3 + feature * 13) % 256) - 128
    return values.astype(np.int8)


def make_scores(group_count: int, mode: str, seed: int = 12345) -> np.ndarray:
    if mode == "zero":
        return np.zeros((group_count,), dtype=np.uint16)
    if mode == "same":
        return np.full((group_count,), 12345, dtype=np.uint16)
    base = (np.arange(group_count, dtype=np.uint64) * 1103515245 + seed) % 32768
    return base.astype(np.uint16)


def build_cases() -> list[Case]:
    return [
        # identity bypass (G <= K)
        Case("identity_4x4_K48", 4, 4, 48),
        Case("identity_8x8_K64", 8, 8, 64),
        Case("identity_9x9_K96", 9, 9, 96),
        Case("identity_2x9_K48", 2, 9, 48),
        Case("identity_5x4_K48", 5, 4, 48),
        Case("identity_1x96_K96", 1, 96, 96),
        # compression at the production budgets
        Case("grid9x9_K48", 9, 9, 48),
        Case("grid9x9_K64", 9, 9, 64),
        Case("grid10x12_K48", 10, 12, 48),
        Case("grid10x12_K64", 10, 12, 64),
        Case("grid10x12_K96", 10, 12, 96),
        # empty regions / dynamic context / single-member regions
        Case("grid2x25_K48", 2, 25, 48),
        Case("grid2x9_K16", 2, 9, 16, scale=0.5),
        Case("grid5x4_K16", 5, 4, 16, scale=2.0),
        # MAX_GRID_DIM boundary (128 accepted, 129 rejected) and extremes
        Case("grid1x128_K48", 1, 128, 48),
        Case("grid128x1_K48", 128, 1, 48),
        Case("grid2x128_K64", 2, 128, 64),
        Case("grid1x129_K48", 1, 129, 48),
        Case("grid2x129_K48", 2, 129, 48),
        Case("grid129x2_K48", 129, 2, 48),
        # score degeneracies
        Case("allsame_9x9_K48", 9, 9, 48, score_mode="same"),
        Case("allsame_10x12_K64", 10, 12, 64, score_mode="same"),
        Case("allzero_9x9_K48", 9, 9, 48, score_mode="zero"),
        Case("allzero_10x12_K64", 10, 12, 64, score_mode="zero"),
        Case("allzero_5x4_K16", 5, 4, 16, score_mode="zero"),
        # rejection paths
        Case("illegal_K0_9x9", 9, 9, 0),
        Case("illegal_K47_9x9", 9, 9, 47),
        Case("illegal_K81_9x9", 9, 9, 81),
        Case("illegal_K100_10x12", 10, 12, 100),
        Case("illegal_K112_10x12", 10, 12, 112),  # 16-multiple above TS_MAX_BUDGET
        Case("illegal_K128_1x128", 1, 128, 128),  # above TS_MAX_BUDGET
        Case("badgrid_G18_9x9", 9, 9, 48, group_override=18),
        Case("localcap0_9x9_K48", 9, 9, 48, local_cap=0),
        Case("localcap17_9x9_K48", 9, 9, 48, local_cap=17),  # above the frozen cap
    ]


def build_random_cases(count: int = RANDOM_CASE_COUNT, seed: int = RANDOM_SEED) -> list[Case]:
    """Deterministic random differential cases (valid + a few rejection paths).

    Group counts are bounded so the frozen binaries stay small; most cases use
    ``G <= 512`` and a few reach the full ``G = 4096`` stress shape.
    """
    rng = np.random.default_rng(seed)
    budgets = (16, 32, 48, 64, 80, 96)
    budgets = np.asarray(budgets, dtype=np.int64)
    cases: list[Case] = []
    for index in range(count):
        # 85% small (<= 512 groups), 10% medium (<= 1024), 5% full (<= 4096).
        draw = float(rng.random())
        if draw < 0.85:
            cap = 512
        elif draw < 0.95:
            cap = 1024
        else:
            cap = fp.MAX_GROUPS
        budget = int(budgets[int(rng.integers(0, len(budgets)))])
        if index % 7 == 0 and cap <= 1024:
            # extreme aspect ratio, still inside MAX_GRID_DIM
            grid_h = 1
            grid_w = int(rng.integers(1, min(128, cap) + 1))
        else:
            grid_h = int(rng.integers(1, min(64, cap) + 1))
            max_w = min(fp.MAX_GRID_DIM, cap // max(1, grid_h))
            if max_w < 1:
                grid_h = 1
                max_w = min(fp.MAX_GRID_DIM, cap)
            grid_w = int(rng.integers(1, max_w + 1))
        group_count = grid_h * grid_w
        if group_count > fp.MAX_GROUPS:
            group_count = fp.MAX_GROUPS
            grid_h, grid_w = 64, 64
        mode = ("rand", "rand", "rand", "same", "zero")[int(rng.integers(0, 5))]
        local_cap = int(rng.integers(1, fp.MAX_LOCAL_CANDIDATE_CAP + 1))
        scale = float(rng.choice([0.03125, 0.5, 1.0, 2.0, 8.0]))
        case_seed = int(rng.integers(0, 1 << 31))
        name = f"rand{index:03d}"
        if index % 23 == 22:
            # deterministic rejection path: grid product does not match G
            case = Case(
                name,
                grid_h,
                grid_w,
                budget,
                score_mode=mode,
                group_override=group_count + 1,
                scale=scale,
                local_cap=local_cap,
                seed=case_seed,
            )
        elif index % 29 == 28:
            case = Case(
                name,
                grid_h,
                grid_w,
                112,  # 16-multiple above TS_MAX_BUDGET -> ERR_BUDGET
                score_mode=mode,
                local_cap=local_cap,
                seed=case_seed,
                scale=scale,
            )
        else:
            case = Case(
                name,
                grid_h,
                grid_w,
                budget,
                score_mode=mode,
                local_cap=local_cap,
                seed=case_seed,
                scale=scale,
            )
        cases.append(case)
    return cases


def write_case(out_dir: Path, case: Case) -> str:
    group_count = case.group_count
    tokens = make_tokens(group_count)
    scores = make_scores(group_count, case.score_mode, case.seed)
    local_cap = case.local_cap
    result = fp.select_fixedpoint(
        tokens, scores, case.grid_h, case.grid_w, case.budget,
        local_cap=local_cap, scale=case.scale,
    )

    in_path = out_dir / f"{case.name}.in.bin"
    with in_path.open("wb") as handle:
        handle.write(b"TSIN")
        handle.write(struct.pack("<5i", group_count, case.grid_h, case.grid_w, case.budget, local_cap))
        handle.write(struct.pack("<f", case.scale))
        handle.write(np.ascontiguousarray(tokens, dtype=np.int8).tobytes())
        handle.write(np.ascontiguousarray(scores, dtype="<u2").tobytes())

    exp_path = out_dir / f"{case.name}.exp.bin"
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
        if result.status == fp.STATUS_OK:
            handle.write(np.ascontiguousarray(result.tokens_out, dtype=np.int8).tobytes())
            handle.write(np.ascontiguousarray(result.group_index, dtype="<u2").tobytes())
            handle.write(np.ascontiguousarray(result.source_type, dtype=np.uint8).tobytes())
            handle.write(np.ascontiguousarray(result.region_id, dtype=np.uint8).tobytes())
            handle.write(np.ascontiguousarray(result.valid, dtype=np.uint8).tobytes())

    return (
        f"{case.name} {group_count} {case.grid_h} {case.grid_w} "
        f"{case.budget} {local_cap}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-count", type=int, default=RANDOM_CASE_COUNT)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases() + build_random_cases(args.random_count)
    lines = [write_case(out_dir, case) for case in cases]
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    layout_lines = []
    for budget in (48, 64, 96):
        layout = fp.fpga_transfer_layout(budget)
        offsets = " ".join(str(value) for value in layout.offsets().values())
        sizes = " ".join(str(value) for value in layout.sizes().values())
        layout_lines.append(f"{budget} {layout.total_bytes} {offsets} {sizes}")
    (out_dir / "layout.txt").write_text("\n".join(layout_lines) + "\n", encoding="utf-8")

    fixed_count = len(cases) - args.random_count
    print(
        f"wrote {len(cases)} cases ({fixed_count} fixed + {args.random_count} random) "
        f"to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
