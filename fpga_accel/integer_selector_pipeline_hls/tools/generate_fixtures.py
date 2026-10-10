#!/usr/bin/env python3
"""Generate compact C++-readable fixtures for the fused ``integer_selector_pipeline``.

The golden is **re-used, never copied**:

* the intermediate ``score_q`` and the scorer intermediates come from
  ``evaluation.selector_integer_scorer_audit.integer_chain``;
* the selected tokens / metadata come from
  ``evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint``.

Both are the exact frozen Python goldens already accepted for the two upstream
HLS modules.  This tool only serialises their values into the documented binary
format (see ``tools/fixture_format.md``).

Fixture families
----------------
* ``wv_q.bin``            byte copy of the accepted seed-20261003 Wv bundle
* ``scene_<name>.bin``    one shared input scene: G, grid, tokens_q, group_mean_q, query_q
* ``<case>.exp.bin``      expected ``score_q`` + selection for one (scene, K, local_cap, scale)
* ``manifest.txt``        ``case scene budget local_cap scale`` lines
* ``manifest.json``       SHA256 / hash / coverage ledger

Real grounding/pointing scenes reuse the accepted validation NPZ
``artifacts/token_selector/selector_integer_scorer_audit/20261003/fixture_<task>.npz``
for ``group_mean_q`` / grid / query; the accepted fixture has no raw ``tokens_q``
(only the pooled group means), so the raw 4x1024 token bytes are drawn from the
documented deterministic xorshift64* generator.  That does not affect the
scorer-side golden (which never sees the tokens) and the selector-side golden is
computed over exactly these bytes.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from evaluation.hardware_regular_selector_fixedpoint import select_fixedpoint
from evaluation.selector_integer_scorer_audit import integer_chain

DATE = "2026-10-03"
ARTIFACT_DIR = ROOT / "artifacts/token_selector/2026-10-03-integer-selector-pipeline-hls"
FIXTURE_DIR = ARTIFACT_DIR / "fixtures"
REAL_FIXTURE_DIR = ROOT / "artifacts/token_selector/selector_integer_scorer_audit/20261003"
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
GROUP_ELEMENTS = 4 * EMBED_DIM

SCENE_HEADER = struct.Struct("<8sIIIIQQQQQ")
EXP_HEADER = struct.Struct("<8siiiifIQQQQQQ")

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


def load_wv_int8() -> np.ndarray:
    actual = sha256_file(WV_PATH)
    if actual != WV_SHA256:
        raise SystemExit(f"accepted Wv SHA mismatch: {actual} != {WV_SHA256}")
    tensor = np.frombuffer(WV_PATH.read_bytes(), dtype=np.int8)
    if tensor.size != WV_ROWS * EMBED_DIM:
        raise SystemExit(f"unexpected Wv size {tensor.size}")
    return tensor.reshape(WV_ROWS, EMBED_DIM).copy()


def xorshift_bytes(seed: int, count: int) -> bytes:
    """Pure-python documented xorshift64* byte stream (matches the C++ TB)."""
    state = seed & MASK64
    out = bytearray(count)
    for index in range(count):
        state ^= (state << 13) & MASK64
        state ^= state >> 7
        state ^= (state << 17) & MASK64
        product = (state * XORSHIFT_MUL) & MASK64
        out[index] = (product >> 56) & 0xFF
    return bytes(out)


def signed_int8(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int16).astype(np.int8)


def pack_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype=np.int8).tobytes()


# ---------------------------------------------------------------------------
# scenes
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    name: str
    group_count: int
    grid_h: int
    grid_w: int
    tokens_q: np.ndarray  # [G,4,1024] int8
    group_mean_q: np.ndarray  # [G,1024] int8
    query_q: np.ndarray  # [128] int8
    meta: dict[str, Any] = field(default_factory=dict)


def scene_payload(scene: Scene) -> bytes:
    return (
        pack_bytes(scene.tokens_q.reshape(-1))
        + pack_bytes(scene.group_mean_q)
        + pack_bytes(scene.query_q)
    )


def scene_bytes(scene: Scene) -> bytes:
    payload = scene_payload(scene)
    header = SCENE_HEADER.pack(
        b"ISPSCN01",
        int(scene.group_count),
        int(scene.grid_h),
        int(scene.grid_w),
        0,
        fnv1a64(pack_bytes(scene.tokens_q.reshape(-1))),
        fnv1a64(pack_bytes(scene.group_mean_q)),
        fnv1a64(pack_bytes(scene.query_q)),
        len(payload),
        0,
    )
    return bytes(header) + payload


def synth_tokens(group_count: int, seed: int) -> np.ndarray:
    raw = xorshift_bytes(seed, group_count * GROUP_ELEMENTS)
    return signed_int8(raw).reshape(group_count, 4, EMBED_DIM)


def synth_group_mean(group_count: int, seed: int) -> np.ndarray:
    raw = xorshift_bytes(seed, group_count * EMBED_DIM)
    return signed_int8(raw).reshape(group_count, EMBED_DIM)


def synth_query(seed: int) -> np.ndarray:
    raw = xorshift_bytes(seed, QUERY_DIM)
    return signed_int8(raw)


def build_real_scene(task: str, role: str) -> Scene:
    with np.load(REAL_FIXTURE_DIR / f"fixture_{task}.npz", allow_pickle=False) as archive:
        group_mean_q = archive["group_mean_q"].astype(np.int8)
        grid_h = int(archive["grid_h"])
        grid_w = int(archive["grid_w"])
        group_count = int(group_mean_q.shape[0])
        query_q = archive[f"query_q_{role}"].astype(np.int8)
        stored_score = archive[f"score_q_{role}"].astype(np.uint16)
    # Re-derive the frozen scorer golden and require a byte-for-byte match with
    # the accepted NPZ before serialising anything.
    chain = integer_chain(group_mean_q, query_q, WV_CACHE["wv_q"], grid_h, grid_w, budgets=())
    if not np.array_equal(chain["score_q"].astype(np.uint16), stored_score):
        raise SystemExit(f"{task}/{role}: regenerated score_q != accepted NPZ")
    tokens_q = synth_tokens(group_count, SEED + 1000 + (0 if task == "grounding" else 1))
    return Scene(
        name=f"real_{task}_{role}",
        group_count=group_count,
        grid_h=grid_h,
        grid_w=grid_w,
        tokens_q=tokens_q,
        group_mean_q=group_mean_q,
        query_q=query_q,
        meta={
            "family": "real",
            "task": task,
            "role": role,
            "source_npz": f"artifacts/token_selector/selector_integer_scorer_audit/20261003/fixture_{task}.npz",
            "score_matches_accepted_npz": True,
            "tokens_source": "xorshift64* (accepted NPZ has no raw tokens)",
        },
    )


def build_synth_scene(name: str, grid_h: int, grid_w: int, mode: str,
                      seed: int) -> Scene:
    group_count = grid_h * grid_w
    tokens_q = synth_tokens(group_count, seed)
    query_q = synth_query(seed + 7)
    if mode == "random":
        group_mean_q = synth_group_mean(group_count, seed + 1)
    elif mode == "zero":
        group_mean_q = np.zeros((group_count, EMBED_DIM), dtype=np.int8)
    elif mode == "all_equal":
        group_mean_q = np.full((group_count, EMBED_DIM), 37, dtype=np.int8)
    else:  # pragma: no cover - guarded by the scene table
        raise SystemExit(f"unknown scene mode {mode}")
    return Scene(
        name=name,
        group_count=group_count,
        grid_h=grid_h,
        grid_w=grid_w,
        tokens_q=tokens_q,
        group_mean_q=group_mean_q,
        query_q=query_q,
        meta={"family": "synthetic", "mode": mode, "generator": "xorshift64*",
              "generator_seed": seed},
    )


def build_scenes() -> list[Scene]:
    scenes = [
        build_real_scene("grounding", "correct"),
        build_real_scene("grounding", "wrong"),
        build_real_scene("pointing", "correct"),
        build_real_scene("pointing", "wrong"),
        build_synth_scene("synth_g80", 8, 10, "random", SEED + 2001),
        build_synth_scene("synth_g400", 20, 20, "random", SEED + 2002),
        build_synth_scene("identity_g16", 4, 4, "random", SEED + 2003),
        build_synth_scene("identity_g9", 3, 3, "random", SEED + 2004),
        build_synth_scene("zero_g128", 8, 16, "zero", SEED + 2005),
        build_synth_scene("all_equal_g128", 8, 16, "all_equal", SEED + 2006),
        build_synth_scene("empty_2x25", 2, 25, "random", SEED + 2007),
    ]
    return scenes


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Case:
    scene: str
    budget: int
    local_cap: int = 16
    scale: float = 0.03125

    @property
    def name(self) -> str:
        return f"{self.scene}_K{self.budget}_cap{self.local_cap}"


def build_cases() -> list[Case]:
    cases: list[Case] = []
    for scene in ("real_grounding_correct", "real_pointing_correct"):
        for budget in (48, 64, 96):
            cases.append(Case(scene, budget))
    for scene in ("real_grounding_wrong", "real_pointing_wrong"):
        cases.append(Case(scene, 48))
    for budget in (48, 64, 96):
        cases.append(Case("synth_g80", budget))
        cases.append(Case("synth_g400", budget))
        cases.append(Case("zero_g128", budget))
        cases.append(Case("all_equal_g128", budget))
    cases.append(Case("identity_g16", 48))
    cases.append(Case("identity_g9", 96))
    cases.append(Case("empty_2x25", 48))
    return cases


def expected_bytes(case: Case, scene: Scene) -> tuple[bytes, dict[str, Any]]:
    chain = integer_chain(
        scene.group_mean_q, scene.query_q, WV_CACHE["wv_q"], scene.grid_h,
        scene.grid_w, budgets=(),
    )
    score_q = chain["score_q"].astype(np.uint16)
    selection = select_fixedpoint(
        scene.tokens_q, score_q, scene.grid_h, scene.grid_w, case.budget,
        local_cap=case.local_cap, scale=case.scale,
    )

    score_bytes = np.ascontiguousarray(score_q, dtype="<u2").tobytes()
    if selection.status == 0:
        tokens_bytes = np.ascontiguousarray(selection.tokens_out, dtype=np.int8).tobytes()
        group_index_bytes = np.ascontiguousarray(selection.group_index, dtype="<u2").tobytes()
        source_bytes = np.ascontiguousarray(selection.source_type, dtype=np.uint8).tobytes()
        region_bytes = np.ascontiguousarray(selection.region_id, dtype=np.uint8).tobytes()
        valid_bytes = np.ascontiguousarray(selection.valid, dtype=np.uint8).tobytes()
    else:
        tokens_bytes = group_index_bytes = source_bytes = region_bytes = valid_bytes = b""

    payload = (
        score_bytes + tokens_bytes + group_index_bytes + source_bytes
        + region_bytes + valid_bytes
    )
    metadata_bytes = source_bytes + region_bytes + valid_bytes
    header = EXP_HEADER.pack(
        b"ISPEX001",
        int(selection.status),
        int(selection.output_count),
        int(selection.real_count),
        int(selection.context_count),
        float(selection.scale_out),
        0,
        fnv1a64(score_bytes),
        fnv1a64(tokens_bytes),
        fnv1a64(group_index_bytes),
        fnv1a64(metadata_bytes),
        len(payload),
        0,
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
        "score_q_max": int(score_q.max()) if score_q.size else 0,
        "score_q_min": int(score_q.min()) if score_q.size else 0,
    }
    return bytes(header) + payload, meta


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
WV_CACHE: dict[str, np.ndarray] = {}


def generate(out_dir: Path) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wv_q = load_wv_int8()
    WV_CACHE["wv_q"] = wv_q.astype(np.int64)
    wv_copy = out_dir / "wv_q.bin"
    wv_copy.write_bytes(wv_q.tobytes())

    (out_dir / "FORMAT.md").write_text(
        (Path(__file__).resolve().parent / "fixture_format.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    scenes = build_scenes()
    scene_records: dict[str, Any] = {}
    scene_by_name: dict[str, Scene] = {}
    for scene in scenes:
        data = scene_bytes(scene)
        (out_dir / f"scene_{scene.name}.bin").write_bytes(data)
        scene_by_name[scene.name] = scene
        record = dict(scene.meta)
        record.update(
            {
                "file": f"scene_{scene.name}.bin",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "group_count": scene.group_count,
                "grid": [scene.grid_h, scene.grid_w],
                "tokens_fnv1a64": fnv1a64(pack_bytes(scene.tokens_q.reshape(-1))),
                "group_mean_fnv1a64": fnv1a64(pack_bytes(scene.group_mean_q)),
                "query_fnv1a64": fnv1a64(pack_bytes(scene.query_q)),
            }
        )
        scene_records[scene.name] = record

    cases = build_cases()
    case_records: dict[str, Any] = {}
    manifest_lines: list[str] = []
    for case in cases:
        scene = scene_by_name[case.scene]
        data, meta = expected_bytes(case, scene)
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
            f"{case.name} scene_{case.scene}.bin {case.budget} {case.local_cap} {case.scale}"
        )

    (out_dir / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    chain_source = ROOT / "evaluation/selector_integer_scorer_audit.py"
    selector_source = ROOT / "evaluation/hardware_regular_selector_fixedpoint.py"
    manifest = {
        "schema_version": "1.0",
        "date": DATE,
        "generator": "fpga_accel/integer_selector_pipeline_hls/tools/generate_fixtures.py",
        "format_doc": "fixtures/FORMAT.md",
        "golden": {
            "scorer_entry": "evaluation.selector_integer_scorer_audit.integer_chain",
            "selector_entry": "evaluation.hardware_regular_selector_fixedpoint.select_fixedpoint",
            "scorer_sha256": sha256_file(chain_source),
            "selector_sha256": sha256_file(selector_source),
        },
        "wv": {
            "source_path": str(WV_PATH.relative_to(ROOT)),
            "source_sha256": WV_SHA256,
            "copied_file": "wv_q.bin",
            "copied_sha256": sha256_file(wv_copy),
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
        "scenes": scene_records,
        "cases": case_records,
        "counts": {"scenes": len(scenes), "cases": len(cases)},
        "coverage": {
            "budgets": sorted({case.budget for case in cases}),
            "group_counts": sorted({scene_by_name[case.scene].group_count for case in cases}),
            "grids": sorted({(scene_by_name[case.scene].grid_h, scene_by_name[case.scene].grid_w)
                             for case in cases}),
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
    print(f"  scenes={manifest['counts']['scenes']} cases={manifest['counts']['cases']}")
    print(f"  budgets={manifest['coverage']['budgets']} "
          f"group_counts={manifest['coverage']['group_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
