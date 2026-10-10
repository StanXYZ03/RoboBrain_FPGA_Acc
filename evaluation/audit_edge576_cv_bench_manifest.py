#!/usr/bin/env python3
"""Independent read-only auditor for the 2026-10-07 edge576 CV-Bench manifest.

The auditor re-derives *every* edge576 manifest contract from first principles;
it deliberately does **not** import the builder so that a builder bug cannot
silently validate itself.  It never writes to the manifest or the summary; it
only emits an audit JSON report.

Contracts re-checked (fail closed on the first violation):

1. the edge manifest is valid JSONL with exactly ``--expected-count`` records
   and unique ``sample_id`` values;
2. the ``sample_id`` set and the frozen label fields (``source``,
   ``source_dataset``, ``task``, ``type``, ``prompt``, ``choices``,
   ``answer``, ``max_new_tokens``) are identical to the frozen full manifest;
3. every record's ``record_sha256`` re-hashes correctly;
4. each image exists, its SHA256 matches, and its original width/height match
   the record;
5. ``full_patch_tokens == product(image_grid_thw)`` and
   ``full_llm_visual_tokens == full_patch_tokens // merge_size**2`` with exact
   divisibility, and every ``full_llm_visual_tokens <= budget_g``;
6. the recorded processing parameters equal the pinned edge576 parameters and
   ``max_pixels == budget_g * (patch_size*merge_size)**2``;
7. the summary JSON and report JSON agree with a freshly recomputed summary
   (counts, G statistics, histogram, source/task counts) and with the physical
   manifest SHA256 and byte size;
8. (optional, ``--verify-processor``) re-running the local processor on every
   image reproduces the recorded ``image_grid_thw`` exactly.  The processor is
   always opened from the **explicit** ``--model``/``model_path`` argument, and
   the audit report records that exact path; the auditor never silently falls
   back to ``DEFAULT_MODEL``.

CPU-only.  Only the processor is opened (never the model weights).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.jsonl"
DEFAULT_SUMMARY = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.summary.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DEFAULT_REPORT = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cv-bench-manifest-audit.json"
)
DEFAULT_MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")

EXPECTED_COUNT = 2638
BUDGET_G = 576
PATCH_SIZE = 16
MERGE_SIZE = 2
MERGE_UNIT = MERGE_SIZE * MERGE_SIZE
RESIZE_FACTOR = PATCH_SIZE * MERGE_SIZE
MIN_PIXELS = 65536
MAX_PIXELS = BUDGET_G * RESIZE_FACTOR * RESIZE_FACTOR  # 589824

COPIED_FIELDS: tuple[str, ...] = (
    "sample_id",
    "source",
    "source_dataset",
    "task",
    "type",
    "prompt",
    "choices",
    "answer",
    "max_new_tokens",
)

G_DEFINITION = (
    "full_llm_visual_tokens = (image_grid_thw[0]*image_grid_thw[1]*"
    "image_grid_thw[2]) // (spatial_merge_size**2)"
)


class Edge576AuditError(RuntimeError):
    """Raised when any edge576 manifest contract is violated (fail closed)."""


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Edge576AuditError(message)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require(path.is_file(), f"missing required input {path} (fail closed)")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise Edge576AuditError(f"{path}:{line_number} invalid JSON: {exc}") from exc
    return rows


def load_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"missing required input {path} (fail closed)")
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def recompute_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = sorted(int(record["full_llm_visual_tokens"]) for record in records)
    source_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for record in records:
        source_counts[str(record["source"])] = source_counts.get(str(record["source"]), 0) + 1
        task_counts[str(record["task"])] = task_counts.get(str(record["task"]), 0) + 1
    histogram = {"g_le_96": 0, "g_97_256": 0, "g_257_512": 0, "g_513_576": 0}
    for value in values:
        if value <= 96:
            histogram["g_le_96"] += 1
        elif value <= 256:
            histogram["g_97_256"] += 1
        elif value <= 512:
            histogram["g_257_512"] += 1
        else:
            histogram["g_513_576"] += 1
    return {
        "sample_count": len(records),
        "unique_sample_id_count": len({str(r["sample_id"]) for r in records}),
        "g_min": values[0],
        "g_max": values[-1],
        "g_mean": statistics.fmean(values),
        "g_median": _percentile(values, 0.5),
        "g_p90": _percentile(values, 0.90),
        "g_p95": _percentile(values, 0.95),
        "g_count_at_budget": sum(1 for value in values if value == BUDGET_G),
        "g_histogram": histogram,
        "identity_bypass_count_g_le_96": sum(1 for value in values if value <= 96),
        "compressed_count_g_gt_96": sum(1 for value in values if value > 96),
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }


def _compare_field(name: str, observed: Any, expected: Any, *, sample_id: Any = None) -> None:
    _require(
        observed == expected,
        f"summary field {name!r} mismatch (sample {sample_id}): "
        f"observed={observed!r} expected={expected!r}",
    )


def audit_records(
    records: Sequence[Mapping[str, Any]],
    full_records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = EXPECTED_COUNT,
    budget_g: int = BUDGET_G,
    check_images: bool = True,
) -> dict[str, Any]:
    """Recompute all per-record contracts; return descriptive statistics."""

    from PIL import Image  # noqa: PLC0415

    _require(
        len(records) == expected_count,
        f"edge manifest has {len(records)} records, expected {expected_count}",
    )
    edge_ids = [str(record["sample_id"]) for record in records]
    _require(len(set(edge_ids)) == len(edge_ids), "edge manifest has duplicate sample_id")
    full_by_id = {str(record["sample_id"]): record for record in full_records}
    _require(
        len(full_by_id) == len(full_records),
        "frozen full manifest has duplicate sample_id (fail closed)",
    )
    _require(
        set(edge_ids) == set(full_by_id),
        "edge sample_id set differs from the frozen full manifest",
    )
    _require(
        sorted(edge_ids, key=int) == edge_ids,
        "edge manifest is not in ascending numeric sample_id order",
    )

    image_hashes_checked = 0
    dims_checked = 0
    grids: list[list[int]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        for field in COPIED_FIELDS + (
            "image",
            "image_sha256",
            "original_width",
            "original_height",
            "image_grid_thw",
            "full_patch_tokens",
            "full_llm_visual_tokens",
            "processing",
            "record_sha256",
        ):
            _require(field in record, f"sample {sample_id} missing field {field!r}")
        for field in COPIED_FIELDS:
            _compare_field(field, record[field], full_by_id[sample_id][field], sample_id=sample_id)

        grid = record["image_grid_thw"]
        _require(
            isinstance(grid, list) and len(grid) == 3,
            f"sample {sample_id} image_grid_thw must be a length-3 list, got {grid!r}",
        )
        _require(
            all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in grid),
            f"sample {sample_id} image_grid_thw must be positive ints, got {grid!r}",
        )
        _require(
            grid[0] == 1,
            f"sample {sample_id} image_grid_thw[0] must be 1 for a single image, "
            f"got {grid!r}",
        )
        patch_tokens = grid[0] * grid[1] * grid[2]
        _require(
            patch_tokens % MERGE_UNIT == 0,
            f"sample {sample_id} patch tokens {patch_tokens} not divisible by {MERGE_UNIT}",
        )
        g = patch_tokens // MERGE_UNIT
        _require(
            int(record["full_patch_tokens"]) == patch_tokens,
            f"sample {sample_id} full_patch_tokens {record['full_patch_tokens']} != "
            f"product(grid)={patch_tokens}",
        )
        _require(
            int(record["full_llm_visual_tokens"]) == g,
            f"sample {sample_id} G {record['full_llm_visual_tokens']} != "
            f"full_patch_tokens//{MERGE_UNIT}={g}",
        )
        _require(
            g <= budget_g,
            f"sample {sample_id} G={g} exceeds edge576 budget {budget_g} "
            "(fail closed; no sample may be deleted)",
        )
        grids.append([int(v) for v in grid])

        processing = record["processing"]
        _require(isinstance(processing, Mapping), f"sample {sample_id} processing not a mapping")
        _compare_field("processing.keep_aspect_ratio", processing.get("keep_aspect_ratio"), True, sample_id=sample_id)
        _compare_field("processing.edge576", processing.get("edge576"), True, sample_id=sample_id)
        _compare_field("processing.patch_size", processing.get("patch_size"), PATCH_SIZE, sample_id=sample_id)
        _compare_field("processing.merge_size", processing.get("merge_size"), MERGE_SIZE, sample_id=sample_id)
        _compare_field("processing.merge_unit", processing.get("merge_unit"), MERGE_UNIT, sample_id=sample_id)
        _compare_field("processing.resize_factor", processing.get("resize_factor"), RESIZE_FACTOR, sample_id=sample_id)
        _compare_field("processing.min_pixels", processing.get("min_pixels"), MIN_PIXELS, sample_id=sample_id)
        _compare_field("processing.max_pixels", processing.get("max_pixels"), MAX_PIXELS, sample_id=sample_id)
        _compare_field("processing.budget_g", processing.get("budget_g"), budget_g, sample_id=sample_id)
        _compare_field("processing.g_definition", processing.get("g_definition"), G_DEFINITION, sample_id=sample_id)
        _require(
            int(processing["max_pixels"]) == budget_g * RESIZE_FACTOR * RESIZE_FACTOR,
            f"sample {sample_id} max_pixels is not budget_g*(patch*merge)**2",
        )

        digest = recompute_record_sha256(record)
        _require(
            digest == record["record_sha256"],
            f"sample {sample_id} record_sha256 mismatch: "
            f"observed={record['record_sha256']} recomputed={digest}",
        )

        if check_images:
            image_path = Path(str(record["image"]))
            _require(image_path.is_file(), f"sample {sample_id} image missing: {image_path}")
            observed_hash = sha256_file(image_path)
            _require(
                observed_hash == record["image_sha256"],
                f"sample {sample_id} image_sha256 mismatch: "
                f"observed={observed_hash} recorded={record['image_sha256']}",
            )
            image_hashes_checked += 1
            with Image.open(image_path) as opened:
                width, height = opened.size
            _require(
                int(record["original_width"]) == width and int(record["original_height"]) == height,
                f"sample {sample_id} original size mismatch: recorded="
                f"({record['original_width']},{record['original_height']}) observed=({width},{height})",
            )
            dims_checked += 1

    return {
        "records_checked": len(records),
        "image_hashes_checked": image_hashes_checked,
        "original_dims_checked": dims_checked,
        "g_min": min(grids, key=lambda g: g[1] * g[2]),
        "unique_sample_ids": len(set(edge_ids)),
    }


def load_local_processor(model_path: Path) -> Any:
    """Open *only* the local processor from an explicit path.

    Never loads model weights, never touches the network.  Extracted into a
    named helper so the ``--model`` argument is a testable, explicit input
    rather than an implicit ``DEFAULT_MODEL`` fallback.
    """

    from transformers import AutoProcessor  # noqa: PLC0415

    return AutoProcessor.from_pretrained(str(model_path), local_files_only=True)


def verify_processor_grids(
    processor: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> int:
    """Re-run the authoritative processor and compare every recorded grid."""

    from evaluation.run_hf_token_reduction import (  # noqa: PLC0415
        make_messages,
        processor_inputs,
    )
    from PIL import Image  # noqa: PLC0415

    checked = 0
    for index, record in enumerate(records, start=1):
        image_path = Path(str(record["image"]))
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            messages = make_messages(image_path, str(record["prompt"]))
            processed = processor_inputs(
                processor,
                messages,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        grid = [int(value) for value in processed.image_grid_thw[0].tolist()]
        _require(
            grid == [int(v) for v in record["image_grid_thw"]],
            f"sample {record['sample_id']} processor grid drift: "
            f"reobserved={grid} recorded={record['image_grid_thw']}",
        )
        checked += 1
        if index % 250 == 0:
            print(f"  re-verified {index}/{len(records)}", flush=True)
    return checked


def audit(
    *,
    manifest: Path = DEFAULT_MANIFEST,
    summary_path: Path = DEFAULT_SUMMARY,
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST,
    expected_count: int = EXPECTED_COUNT,
    budget_g: int = BUDGET_G,
    check_images: bool = True,
    verify_processor: bool = False,
    processor: Any | None = None,
    model_path: Path = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run the complete read-only audit and return the audit report.

    ``model_path`` is the explicit processor location.  It is recorded verbatim
    in the report and is the only path used when ``verify_processor`` opens the
    processor, so the CLI ``--model`` value can never be silently dropped.
    """

    records = load_jsonl(manifest)
    full_records = load_jsonl(source_manifest)
    summary = load_json(summary_path)

    stats = audit_records(
        records,
        full_records,
        expected_count=expected_count,
        budget_g=budget_g,
        check_images=check_images,
    )

    manifest_sha = sha256_file(manifest)
    manifest_bytes = manifest.stat().st_size
    _compare_field("summary.manifest.sha256", summary["manifest"]["sha256"], manifest_sha)
    _compare_field("summary.manifest.bytes", summary["manifest"]["bytes"], manifest_bytes)
    _compare_field("summary.manifest.line_count", summary["manifest"]["line_count"], len(records))
    _compare_field(
        "summary.source_manifest.sha256",
        summary["source_manifest"]["sha256"],
        sha256_file(source_manifest),
    )
    _compare_field(
        "summary.source_manifest.path",
        Path(summary["source_manifest"]["path"]).resolve(),
        source_manifest.resolve(),
    )
    _compare_field("summary.budget_g", summary["budget_g"], budget_g)
    _compare_field("summary.all_g_within_budget", summary["all_g_within_budget"], True)
    _compare_field("summary.edge576", summary["edge576"], True)

    recomputed = recompute_summary(records)
    for field, expected in recomputed.items():
        _compare_field(f"summary.{field}", summary[field], expected)

    processing = summary["processing_parameters"]
    _compare_field("processing_parameters.min_pixels", processing["min_pixels"], MIN_PIXELS)
    _compare_field("processing_parameters.max_pixels", processing["max_pixels"], MAX_PIXELS)
    _compare_field("processing_parameters.patch_size", processing["patch_size"], PATCH_SIZE)
    _compare_field("processing_parameters.merge_size", processing["merge_size"], MERGE_SIZE)
    _compare_field("processing_parameters.keep_aspect_ratio", processing["keep_aspect_ratio"], True)

    processor_checked = 0
    if verify_processor:
        if processor is None:
            processor = load_local_processor(model_path)
        processor_checked = verify_processor_grids(processor, records)

    return {
        "status": "EDGE576_MANIFEST_AUDIT_PASS",
        "date": "2026-10-07",
        "edge576": True,
        "gpu_runs": 0,
        "weights_loaded": False,
        "processor_model_path": str(model_path),
        "processor_grid_verification_requested": bool(verify_processor),
        "manifest": {"path": str(manifest), "sha256": manifest_sha, "bytes": manifest_bytes},
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "auditor": "evaluation/audit_edge576_cv_bench_manifest.py",
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "contracts": {
            "exact_record_count": len(records) == expected_count,
            "unique_sample_ids": stats["unique_sample_ids"] == expected_count,
            "id_set_matches_frozen_full_manifest": True,
            "frozen_labels_identical": True,
            "record_sha256_recomputed": True,
            "image_sha256_recomputed": bool(stats["image_hashes_checked"] == len(records)),
            "original_dims_recomputed": bool(stats["original_dims_checked"] == len(records)),
            "g_formula_consistent": True,
            "all_g_within_budget": True,
            "processing_parameters_pinned": True,
            "summary_matches_recomputation": True,
            "processor_grid_reverified": bool(verify_processor),
        },
        "records_checked": stats["records_checked"],
        "image_hashes_checked": stats["image_hashes_checked"],
        "original_dims_checked": stats["original_dims_checked"],
        "processor_grids_reverified": processor_checked,
        "g_statistics": recomputed,
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    parser.add_argument("--budget-g", type=int, default=BUDGET_G)
    parser.add_argument("--skip-image-checks", action="store_true")
    parser.add_argument(
        "--verify-processor",
        action="store_true",
        help="re-run the local processor on every image and compare image_grid_thw",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    report = audit(
        manifest=args.manifest,
        summary_path=args.summary,
        source_manifest=args.source_manifest,
        expected_count=args.expected_count,
        budget_g=args.budget_g,
        check_images=not args.skip_image_checks,
        verify_processor=args.verify_processor,
        model_path=args.model,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "EDGE576_MANIFEST_AUDIT_PASS "
        f"n={report['records_checked']} "
        f"images={report['image_hashes_checked']} "
        f"processor_grids={report['processor_grids_reverified']} "
        f"model={report['processor_model_path']} "
        f"manifest_sha256={report['manifest']['sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Edge576AuditError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
