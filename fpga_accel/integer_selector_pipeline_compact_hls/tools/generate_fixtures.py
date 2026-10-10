#!/usr/bin/env python3
"""Generate compact C++-readable fixtures for ``integer_selector_pipeline_compact``.

The golden is **re-used, never copied**:

* the accepted fixture scene table, the deterministic xorshift64* generator and
  the accepted seed-20261003 Wv bundle come from the already-accepted module
  ``fpga_accel/integer_selector_pipeline_hls/tools/generate_fixtures.py``
  (imported read-only by file path);
* the accepted scorer golden is ``evaluation.selector_integer_scorer_audit.integer_chain``;
* the accepted selector golden is
  ``evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint``;
* the canonical output-arena offsets/sizes/total come *live* from
  ``evaluation.hardware_regular_token_selector.fpga_transfer_layout``.

For every accepted case this tool serialises the exact expected canonical
output arena (tokens at offset 0, the four metadata segments at their frozen
64-byte-aligned offsets, every gap and unused entry zero) plus the accepted
status/counts/scale.  Nothing here re-implements the scorer, the selector or the
freeze-frozen offset algorithm.

No network, no GPU, no training, no model load.  CPU-only and read-only w.r.t.
every accepted upstream artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.hardware_regular_token_selector import (  # noqa: E402
    describe_fpga_layout,
    fpga_transfer_layout,
)

DATE = "2026-10-03"
MODULE_DIR = Path(__file__).resolve().parents[1]
ACCEPTED_TOOLS = ROOT / "fpga_accel/integer_selector_pipeline_hls/tools"
ACCEPTED_GENERATOR = ACCEPTED_TOOLS / "generate_fixtures.py"
ARTIFACT_DIR = (
    ROOT / "artifacts/token_selector/2026-10-03-integer-selector-compact-interface-hls"
)
FIXTURE_DIR = ARTIFACT_DIR / "fixtures"

MASK64 = (1 << 64) - 1
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3

#: ``<`` = native standard sizes, no padding: 8 + 4*4 + 4 + 4 + 4 + 8 + 8 = 52.
EXP_HEADER = struct.Struct("<8siiiifiiQQ")
EXP_MAGIC = b"ISPCMP01"


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


def load_accepted_generator():
    """Import the accepted pipeline fixture generator read-only by path."""
    spec = importlib.util.spec_from_file_location(
        "accepted_pipeline_generate_fixtures", ACCEPTED_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def input_arena_bytes(scene) -> bytes:
    """tokens_int8 | group_mean_int8 | query_int8, already 64-byte aligned."""
    tokens = np.ascontiguousarray(scene.tokens_q.reshape(-1), dtype=np.int8)
    gm = np.ascontiguousarray(scene.group_mean_q, dtype=np.int8)
    query = np.ascontiguousarray(scene.query_q, dtype=np.int8)
    return tokens.tobytes() + gm.tobytes() + query.tobytes()


def expected_arena(selection, budget: int) -> bytes:
    """Canonical fixed-budget output arena from the frozen selector golden."""
    layout = fpga_transfer_layout(budget)
    arena = bytearray(layout.total_bytes)
    if selection.status == 0:
        off = layout.offsets()
        count = int(selection.output_count)
        tokens = np.ascontiguousarray(selection.tokens_out, dtype=np.int8).tobytes()
        assert len(tokens) == count * 4096
        arena[0 : count * 4096] = tokens
        arena[off["group_index_uint16"] : off["group_index_uint16"] + count * 2] = (
            np.ascontiguousarray(selection.group_index, dtype="<u2").tobytes()
        )
        arena[off["source_type_uint8"] : off["source_type_uint8"] + count] = (
            np.ascontiguousarray(selection.source_type, dtype=np.uint8).tobytes()
        )
        arena[off["region_id_uint8"] : off["region_id_uint8"] + count] = (
            np.ascontiguousarray(selection.region_id, dtype=np.uint8).tobytes()
        )
        arena[off["valid_uint8"] : off["valid_uint8"] + count] = (
            np.ascontiguousarray(selection.valid, dtype=np.uint8).tobytes()
        )
    return bytes(arena)


def expected_bytes(case, scene, mod) -> tuple[bytes, dict[str, Any]]:
    chain = mod.integer_chain(
        scene.group_mean_q, scene.query_q, mod.WV_CACHE["wv_q"], scene.grid_h,
        scene.grid_w, budgets=(),
    )
    score_q = chain["score_q"].astype(np.uint16)
    selection = mod.select_fixedpoint(
        scene.tokens_q, score_q, scene.grid_h, scene.grid_w, case.budget,
        local_cap=case.local_cap, scale=case.scale,
    )
    arena = expected_arena(selection, case.budget)
    layout = fpga_transfer_layout(case.budget)
    header = EXP_HEADER.pack(
        EXP_MAGIC,
        int(selection.status),
        int(selection.output_count),
        int(selection.real_count),
        int(selection.context_count),
        float(selection.scale_out),
        int(case.budget),
        int(layout.total_bytes),
        fnv1a64(arena),
        int(layout.total_bytes),
    )
    meta = {
        "case": case.name,
        "scene": case.scene,
        "group_count": scene.group_count,
        "grid": [scene.grid_h, scene.grid_w],
        "budget": case.budget,
        "local_cap": case.local_cap,
        "scale": case.scale,
        "status": int(selection.status),
        "output_count": int(selection.output_count),
        "real_count": int(selection.real_count),
        "context_count": int(selection.context_count),
        "total_bytes": int(layout.total_bytes),
        "score_q_min": int(score_q.min()) if score_q.size else 0,
        "score_q_max": int(score_q.max()) if score_q.size else 0,
    }
    return bytes(header) + arena, meta


def generate(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mod = load_accepted_generator()

    wv = mod.load_wv_int8()
    mod.WV_CACHE["wv_q"] = wv.astype(np.int64)
    wv_copy = out_dir / "wv_q.bin"
    wv_copy.write_bytes(wv.tobytes())

    (out_dir / "FORMAT.md").write_text(
        (MODULE_DIR / "tools/fixture_format.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    scenes = mod.build_scenes()
    scene_records: dict[str, Any] = {}
    scene_by_name: dict[str, Any] = {}
    for scene in scenes:
        data = input_arena_bytes(scene)
        name = f"arena_{scene.name}.bin"
        (out_dir / name).write_bytes(data)
        scene_by_name[scene.name] = scene
        expected_total = scene.group_count * 5120 + 128
        if len(data) != expected_total:
            raise SystemExit(
                f"{scene.name}: arena {len(data)} != formula {expected_total}"
            )
        scene_records[scene.name] = {
            "file": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "group_count": scene.group_count,
            "grid": [scene.grid_h, scene.grid_w],
            "tokens_offset": 0,
            "tokens_bytes": scene.group_count * 4096,
            "group_mean_offset": scene.group_count * 4096,
            "group_mean_bytes": scene.group_count * 1024,
            "query_offset": scene.group_count * 5120,
            "query_bytes": 128,
            "total_bytes": len(data),
        }

    cases = mod.build_cases()
    case_records: dict[str, Any] = {}
    manifest_lines: list[str] = []
    for case in cases:
        scene = scene_by_name[case.scene]
        data, meta = expected_bytes(case, scene, mod)
        path = out_dir / f"{case.name}.exp.bin"
        path.write_bytes(data)
        meta = dict(meta)
        meta.update(
            {
                "file": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        case_records[case.name] = meta
        manifest_lines.append(
            f"{case.name} arena_{case.scene}.bin {case.budget} {case.local_cap} "
            f"{case.scale} {scene.grid_h} {scene.grid_w}"
        )

    (out_dir / "manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    budgets = sorted({case.budget for case in cases})
    layout_records = {
        str(budget): describe_fpga_layout(budget) for budget in budgets
    }
    manifest = {
        "schema_version": "1.0",
        "date": DATE,
        "generator": (
            "fpga_accel/integer_selector_pipeline_compact_hls/tools/generate_fixtures.py"
        ),
        "format_doc": "fixtures/FORMAT.md",
        "golden": {
            "scorer_entry": "evaluation.selector_integer_scorer_audit.integer_chain",
            "selector_entry": (
                "evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint"
            ),
            "layout_entry": (
                "evaluation.hardware_regular_token_selector.fpga_transfer_layout"
            ),
            "scorer_sha256": sha256_file(
                ROOT / "evaluation/selector_integer_scorer_audit.py"
            ),
            "selector_sha256": sha256_file(
                ROOT / "evaluation/hardware_regular_selector_fixedpoint.py"
            ),
            "layout_sha256": sha256_file(
                ROOT / "evaluation/hardware_regular_token_selector.py"
            ),
            "accepted_generator_sha256": sha256_file(ACCEPTED_GENERATOR),
        },
        "wv": {
            "source_path": str(mod.WV_PATH.relative_to(ROOT)),
            "source_sha256": mod.WV_SHA256,
            "copied_file": "wv_q.bin",
            "copied_sha256": sha256_file(wv_copy),
            "note": "reused by SHA, never retrained; Wv scale is a positive common factor",
        },
        "input_arena_formula": {
            "tokens_int8": "offset 0, size G*4096",
            "group_mean_int8": "offset G*4096, size G*1024",
            "query_int8": "offset G*5120, size 128",
            "total_bytes": "G*5120 + 128",
            "alignment": 64,
        },
        "output_arena_authority": {
            "module": "evaluation.hardware_regular_token_selector",
            "functions": ["fpga_transfer_layout", "describe_fpga_layout"],
            "layouts": layout_records,
        },
        "scenes": scene_records,
        "cases": case_records,
        "counts": {"scenes": len(scenes), "cases": len(cases)},
        "coverage": {
            "budgets": sorted({case.budget for case in cases}),
            "group_counts": sorted(
                {scene_by_name[case.scene].group_count for case in cases}
            ),
            "grids": sorted(
                {
                    (scene_by_name[case.scene].grid_h, scene_by_name[case.scene].grid_w)
                    for case in cases
                }
            ),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    started = time.time()
    manifest = generate(args.out_dir)
    print(f"fixtures -> {args.out_dir} ({time.time() - started:.2f}s)")
    print(
        f"  scenes={manifest['counts']['scenes']} "
        f"cases={manifest['counts']['cases']}"
    )
    print(
        f"  budgets={manifest['coverage']['budgets']} "
        f"group_counts={manifest['coverage']['group_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
