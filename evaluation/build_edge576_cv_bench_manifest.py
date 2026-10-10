#!/usr/bin/env python3
"""Deterministic edge576 CV-Bench manifest builder (CPU-only, no model weights).

This is the frozen infrastructure for the 2026-10-07 edge deployment runtime
domain.  It re-preprocesses the *complete* frozen CV-Bench 2638-sample manifest
with the local RoboBrain2.5-4B processor under a single aspect-ratio-preserving
pixel cap:

    max_pixels = budget_g * (patch_size * merge_size)**2 = 576 * 32**2 = 589824

so that, for every sample, the number of vision tokens actually entering the
LLM is ``G <= 576``.  ``G`` is *not* trusted from the pixel parameter: it is
measured from the authoritative ``image_grid_thw`` returned by the processor
with ``spatial_merge_size = 2``::

    full_patch_tokens      = image_grid_thw[0] * image_grid_thw[1] * image_grid_thw[2]
    full_llm_visual_tokens = full_patch_tokens // (merge_size ** 2)

Contract:

* every one of the 2638 frozen samples is processed; nothing is dropped after
  the fact for being too large;
* ``sample_id`` / ``source`` / ``source_dataset`` / ``task`` / ``type`` /
  ``prompt`` / ``choices`` / ``answer`` / ``max_new_tokens`` are copied
  byte-for-byte from the frozen full manifest;
* each record carries the image path, image SHA256, original width/height, the
  authoritative ``image_grid_thw``, patch/LLM tokens, the full processing
  parameters and a canonical ``record_sha256``;
* if *any* sample yields ``G > budget_g`` (including a rounding surprise),
  the builder fails closed and writes no manifest at all; samples are never
  removed to make the contract hold.

CPU-only.  ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` are set by the
caller; only the processor is opened, never the model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
DEFAULT_SOURCE_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DEFAULT_MANIFEST = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.jsonl"
DEFAULT_SUMMARY = ROOT / "datasets" / "2026-10-07-cv-bench-edge576-full2638.summary.json"
DEFAULT_REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cv-bench-manifest.json"
)
DEFAULT_REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-07-edge576-cv-bench-manifest.md"
)

BUILDER_NAME = "evaluation/build_edge576_cv_bench_manifest.py"
DEFAULT_MAX_PIXELS = 589824
DEFAULT_MIN_PIXELS = 65536
DEFAULT_BUDGET_G = 576
DEFAULT_PATCH_SIZE = 16
DEFAULT_MERGE_SIZE = 2

#: frozen label fields copied verbatim from the full manifest.
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


class Edge576ManifestError(RuntimeError):
    """Raised when the edge576 manifest contract is violated (fail closed)."""


# ---------------------------------------------------------------------------
# Pure helpers (importable without transformers / PIL / torch)
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA256 of a file (never loads the whole file into RAM)."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> str:
    """Deterministic, compact JSON used for record hashing and writing."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_group_count(
    image_grid_thw: Sequence[int],
    *,
    merge_size: int = DEFAULT_MERGE_SIZE,
) -> tuple[int, int]:
    """Return ``(full_patch_tokens, full_llm_visual_tokens)`` from a grid.

    Fails closed on a malformed grid or a merger misalignment; never rounds.
    """

    if not isinstance(image_grid_thw, (list, tuple)) or len(image_grid_thw) != 3:
        raise Edge576ManifestError(
            f"image_grid_thw must be a length-3 sequence, got {image_grid_thw!r}"
        )
    grid: list[int] = []
    for value in image_grid_thw:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Edge576ManifestError(
                f"image_grid_thw entries must be positive ints, got {image_grid_thw!r}"
            )
        grid.append(int(value))
    if not isinstance(merge_size, int) or isinstance(merge_size, bool) or merge_size <= 0:
        raise Edge576ManifestError(f"merge_size must be a positive int, got {merge_size!r}")
    merge_unit = merge_size * merge_size
    full_patch_tokens = grid[0] * grid[1] * grid[2]
    if full_patch_tokens % merge_unit:
        raise Edge576ManifestError(
            f"grid {grid} patch tokens {full_patch_tokens} are not divisible by "
            f"merge_unit {merge_unit} (fail closed)"
        )
    return full_patch_tokens, full_patch_tokens // merge_unit


def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical SHA256 over a record, excluding its own ``record_sha256``."""

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def assert_g_within_budget(records: Iterable[Mapping[str, Any]], *, budget_g: int) -> None:
    """Fail closed if any record has ``G > budget_g`` (no sample is dropped)."""

    offenders = [
        (record.get("sample_id"), record.get("full_llm_visual_tokens"))
        for record in records
        if int(record.get("full_llm_visual_tokens", budget_g + 1)) > budget_g
    ]
    if offenders:
        raise Edge576ManifestError(
            f"{len(offenders)} sample(s) exceed the edge576 budget G<={budget_g} "
            f"after authoritative processing: {offenders[:10]} (fail closed; "
            "samples are never deleted to satisfy the contract)"
        )


def label_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the frozen label fields that must match the full manifest."""

    missing = [key for key in COPIED_FIELDS if key not in record]
    if missing:
        raise Edge576ManifestError(f"record missing frozen label fields {missing}")
    return {key: record[key] for key in COPIED_FIELDS}


def validate_label_identity(
    edge_records: Sequence[Mapping[str, Any]],
    full_records: Sequence[Mapping[str, Any]],
) -> None:
    """Every edge record's frozen label fields must equal the full manifest."""

    full_by_id = {str(record["sample_id"]): record for record in full_records}
    if len(full_by_id) != len(full_records):
        raise Edge576ManifestError("full manifest has duplicate sample_id values")
    edge_ids = [str(record["sample_id"]) for record in edge_records]
    if len(set(edge_ids)) != len(edge_ids):
        raise Edge576ManifestError("edge manifest has duplicate sample_id values")
    if set(edge_ids) != set(full_by_id):
        missing = sorted(set(full_by_id) - set(edge_ids), key=int)
        extra = sorted(set(edge_ids) - set(full_by_id), key=int)
        raise Edge576ManifestError(
            f"edge/full sample_id sets differ: missing={missing[:10]} extra={extra[:10]}"
        )
    for record in edge_records:
        sample_id = str(record["sample_id"])
        expected = label_fields(full_by_id[sample_id])
        observed = label_fields(record)
        if observed != expected:
            raise Edge576ManifestError(
                f"sample {sample_id} label drift vs frozen full manifest: "
                f"observed={observed!r} expected={expected!r}"
            )


def build_record(
    sample: Mapping[str, Any],
    *,
    image_grid_thw: Sequence[int],
    image_sha256: str,
    original_width: int,
    original_height: int,
    model_path: str,
    processor_class: str,
    image_processor_class: str,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    budget_g: int,
) -> dict[str, Any]:
    """Assemble one edge576 record (deterministic field order)."""

    full_patch_tokens, full_llm_visual_tokens = compute_group_count(
        image_grid_thw, merge_size=merge_size
    )
    record: dict[str, Any] = {
        "sample_id": str(sample["sample_id"]),
        "source": sample["source"],
        "source_dataset": sample["source_dataset"],
        "task": sample["task"],
        "type": sample["type"],
        "image": str(sample["image"]),
        "image_sha256": str(image_sha256),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "image_grid_thw": [int(value) for value in image_grid_thw],
        "full_patch_tokens": int(full_patch_tokens),
        "full_llm_visual_tokens": int(full_llm_visual_tokens),
        "processing": {
            "model_path": str(model_path),
            "processor_class": str(processor_class),
            "image_processor_class": str(image_processor_class),
            "patch_size": int(patch_size),
            "merge_size": int(merge_size),
            "merge_unit": int(merge_size) * int(merge_size),
            "resize_factor": int(patch_size) * int(merge_size),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "keep_aspect_ratio": True,
            "edge576": True,
            "budget_g": int(budget_g),
            "g_definition": G_DEFINITION,
        },
        "prompt": sample["prompt"],
        "choices": sample["choices"],
        "answer": sample["answer"],
        "max_new_tokens": sample["max_new_tokens"],
    }
    record["record_sha256"] = record_sha256(record)
    return record


def _percentile(sorted_values: Sequence[int], fraction: float) -> float:
    """Linear-interpolation percentile on a pre-sorted sequence."""

    if not sorted_values:
        raise Edge576ManifestError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def g_histogram(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Deterministic G bins used by the edge-domain report."""

    bins = {"g_le_96": 0, "g_97_256": 0, "g_257_512": 0, "g_513_576": 0}
    for record in records:
        g = int(record["full_llm_visual_tokens"])
        if g <= 96:
            bins["g_le_96"] += 1
        elif g <= 256:
            bins["g_97_256"] += 1
        elif g <= 512:
            bins["g_257_512"] += 1
        else:
            bins["g_513_576"] += 1
    return bins


def build_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    source_manifest: Path,
    manifest_path: Path,
    manifest_bytes: int,
    manifest_sha256: str,
    model_path: str,
    model_identity_manifest: Path | None,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    budget_g: int,
) -> dict[str, Any]:
    """Summary statistics for the written manifest (fail closed on drift)."""

    if not records:
        raise Edge576ManifestError("cannot summarise an empty edge576 manifest")
    assert_g_within_budget(records, budget_g=budget_g)
    values = sorted(int(record["full_llm_visual_tokens"]) for record in records)
    source_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    for record in records:
        source_counts[str(record["source"])] = source_counts.get(str(record["source"]), 0) + 1
        task_counts[str(record["task"])] = task_counts.get(str(record["task"]), 0) + 1
    summary: dict[str, Any] = {
        "date": "2026-10-07",
        "name": "2026-10-07-cv-bench-edge576-full2638",
        "edge576": True,
        "budget_g": int(budget_g),
        "source_manifest": {
            "path": str(source_manifest),
            "sha256": sha256_file(source_manifest),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": str(manifest_sha256),
            "bytes": int(manifest_bytes),
            "line_count": len(records),
        },
        "sample_count": len(records),
        "unique_sample_id_count": len({str(record["sample_id"]) for record in records}),
        "all_g_within_budget": True,
        "g_min": values[0],
        "g_max": values[-1],
        "g_mean": statistics.fmean(values),
        "g_median": _percentile(values, 0.5),
        "g_p90": _percentile(values, 0.90),
        "g_p95": _percentile(values, 0.95),
        "g_count_at_budget": sum(1 for value in values if value == budget_g),
        "g_histogram": g_histogram(records),
        "identity_bypass_count_g_le_96": sum(1 for value in values if value <= 96),
        "compressed_count_g_gt_96": sum(1 for value in values if value > 96),
        "source_counts": dict(sorted(source_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "processing_parameters": {
            "model_path": str(model_path),
            "patch_size": int(patch_size),
            "merge_size": int(merge_size),
            "merge_unit": int(merge_size) * int(merge_size),
            "resize_factor": int(patch_size) * int(merge_size),
            "min_pixels": int(min_pixels),
            "max_pixels": int(max_pixels),
            "keep_aspect_ratio": True,
            "g_definition": G_DEFINITION,
            "g_budget_rule": f"full_llm_visual_tokens <= {int(budget_g)}",
        },
        "model_identity_manifest": (
            {
                "path": str(model_identity_manifest),
                "sha256": sha256_file(model_identity_manifest),
            }
            if model_identity_manifest is not None and Path(model_identity_manifest).is_file()
            else None
        ),
        "builder": BUILDER_NAME,
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "gpu_runs": 0,
        "weights_loaded": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return summary


# ---------------------------------------------------------------------------
# Authoritative processor loop
# ---------------------------------------------------------------------------
def _load_processor(model_path: Path) -> Any:
    from transformers import AutoProcessor  # noqa: PLC0415

    return AutoProcessor.from_pretrained(str(model_path), local_files_only=True)


def _process_one(
    processor: Any,
    image_path: Path,
    prompt: str,
    *,
    min_pixels: int,
    max_pixels: int,
) -> tuple[list[int], int, int, str]:
    from PIL import Image  # noqa: PLC0415

    from evaluation.run_hf_token_reduction import (  # noqa: PLC0415
        make_messages,
        processor_inputs,
    )

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        width, height = image.size
        messages = make_messages(image_path, prompt)
        processed = processor_inputs(
            processor,
            messages,
            image,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    grid = [int(value) for value in processed.image_grid_thw[0].tolist()]
    if len(grid) != 3:
        raise Edge576ManifestError(
            f"processor returned a non length-3 grid for {image_path}: {grid}"
        )
    return grid, width, height, sha256_file(image_path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Edge576ManifestError(f"missing required input {path} (fail closed)")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise Edge576ManifestError(
                f"{path}:{line_number} is not valid JSON: {exc}"
            ) from exc
    return rows


def build_manifest_records(
    processor: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    model_path: Path,
    patch_size: int,
    merge_size: int,
    min_pixels: int,
    max_pixels: int,
    budget_g: int,
) -> list[dict[str, Any]]:
    """Process every sample in frozen order; fail closed before any write."""

    processor_class = type(processor).__name__
    image_processor = getattr(processor, "image_processor", None)
    image_processor_class = type(image_processor).__name__ if image_processor else "unknown"
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        image_path = Path(str(sample["image"]))
        if not image_path.is_file():
            raise Edge576ManifestError(
                f"sample {sample.get('sample_id')!r} image is missing: {image_path} "
                "(fail closed)"
            )
        grid, width, height, image_digest = _process_one(
            processor,
            image_path,
            str(sample["prompt"]),
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        record = build_record(
            sample,
            image_grid_thw=grid,
            image_sha256=image_digest,
            original_width=width,
            original_height=height,
            model_path=str(model_path),
            processor_class=processor_class,
            image_processor_class=image_processor_class,
            patch_size=patch_size,
            merge_size=merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            budget_g=budget_g,
        )
        if int(record["full_llm_visual_tokens"]) > budget_g:
            raise Edge576ManifestError(
                f"sample {record['sample_id']} produced G="
                f"{record['full_llm_visual_tokens']} > budget {budget_g} with "
                f"grid {grid} (fail closed; the sample is never deleted)"
            )
        records.append(record)
        if index % 250 == 0:
            print(f"  processed {index}/{len(samples)}", flush=True)
    return records


def write_manifest(
    records: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    budget_g: int,
) -> tuple[str, int]:
    """Write JSONL deterministically; return ``(sha256, bytes)``."""

    assert_g_within_budget(records, budget_g=budget_g)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(canonical_json(record) for record in records) + "\n"
    output.write_text(payload, encoding="utf-8")
    return sha256_file(output), output.stat().st_size


def render_report_markdown(summary: Mapping[str, Any]) -> str:
    """Concise Chinese report for the manifest freeze."""

    hist = summary["g_histogram"]
    processing = summary["processing_parameters"]
    lines = [
        "# edge576 CV-Bench manifest 构建报告（2026-10-07）",
        "",
        f"- 状态：`EDGE576_MANIFEST_BUILT`（CPU-only、`gpu_runs=0`、未加载模型权重）",
        f"- 样本数：{summary['sample_count']}（唯一 `sample_id` "
        f"{summary['unique_sample_id_count']}）",
        f"- 输入冻结 manifest：`{summary['source_manifest']['path']}` "
        f"SHA `{summary['source_manifest']['sha256']}`",
        f"- 输出 manifest：`{summary['manifest']['path']}` "
        f"SHA `{summary['manifest']['sha256']}`（{summary['manifest']['bytes']} B）",
        f"- 处理参数：保持长宽比、`min_pixels={processing['min_pixels']}`、"
        f"`max_pixels={processing['max_pixels']}`、`patch_size={processing['patch_size']}`、"
        f"`merge_size={processing['merge_size']}`、`G<= {summary['budget_g']}`",
        f"- 公式：{processing['g_definition']}",
        "",
        "## G 分布（full_llm_visual_tokens）",
        "",
        f"- min/median/mean/p90/p95/max = {summary['g_min']}/"
        f"{summary['g_median']:.1f}/{summary['g_mean']:.2f}/"
        f"{summary['g_p90']:.1f}/{summary['g_p95']:.1f}/{summary['g_max']}",
        f"- 恰好等于 {summary['budget_g']} 的样本数：{summary['g_count_at_budget']}",
        f"- `G<=96`（identity bypass）：{summary['identity_bypass_count_g_le_96']}；"
        f"`96<G<=576`（压缩）：{summary['compressed_count_g_gt_96']}",
        f"- 分桶：<=96={hist['g_le_96']}、97-256={hist['g_97_256']}、"
        f"257-512={hist['g_257_512']}、513-576={hist['g_513_576']}",
        f"- 源分布：{summary['source_counts']}",
        f"- 任务分布：{summary['task_counts']}",
        "",
        "## 合同",
        "",
        "- 全部 2638 条均以权威 `image_grid_thw` 实测，未事后删除任何高 G 样本；",
        "- ID/标签/源/任务与冻结 full manifest 完全一致；",
        "- 每条记录含图片 SHA256、原始尺寸、`image_grid_thw`、patch/LLM token、"
        "处理参数与 `record_sha256`；",
        "- 任意 `G>576` 即 fail closed 且不写出 manifest。",
        "",
        "## 声明",
        "",
        "- 本构建只打开 processor，未加载模型权重、未使用 GPU、未联网、未训练；",
        "- full2638 是发展集，不是未触碰确认集；",
        "- 本 manifest 冻结不代表任何精度/部署/加速主张。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--model-identity-manifest", type=Path,
                        default=ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--merge-size", type=int, default=DEFAULT_MERGE_SIZE)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--budget-g", type=int, default=DEFAULT_BUDGET_G)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke only: process the first N samples; refuses the real default output path",
    )
    args = parser.parse_args(argv)

    samples = load_jsonl(args.manifest)
    if args.limit is not None:
        if args.output.resolve() == DEFAULT_MANIFEST.resolve():
            raise Edge576ManifestError(
                "--limit may not write the real default edge576 manifest (fail closed)"
            )
        samples = samples[: args.limit]
    expected_max_pixels = args.budget_g * (args.patch_size * args.merge_size) ** 2
    if args.max_pixels != expected_max_pixels:
        raise Edge576ManifestError(
            f"max_pixels {args.max_pixels} != budget_g*(patch_size*merge_size)**2 "
            f"= {expected_max_pixels} (fail closed)"
        )
    if not args.model.is_dir():
        raise Edge576ManifestError(f"model directory missing: {args.model} (fail closed)")

    print(f"loading processor from {args.model} (CPU-only, offline)")
    processor = _load_processor(args.model)
    records = build_manifest_records(
        processor,
        samples,
        model_path=args.model,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        budget_g=args.budget_g,
    )
    validate_label_identity(records, load_jsonl(args.manifest))

    manifest_sha, manifest_bytes = write_manifest(
        records, args.output, budget_g=args.budget_g
    )
    summary = build_summary(
        records,
        source_manifest=args.manifest,
        manifest_path=args.output,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha,
        model_path=str(args.model),
        model_identity_manifest=args.model_identity_manifest,
        patch_size=args.patch_size,
        merge_size=args.merge_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        budget_g=args.budget_g,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report_md.write_text(render_report_markdown(summary), encoding="utf-8")

    print(
        "EDGE576_MANIFEST_BUILT "
        f"n={len(records)} G=[{summary['g_min']},{summary['g_max']}] "
        f"manifest_sha256={manifest_sha}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Edge576ManifestError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
