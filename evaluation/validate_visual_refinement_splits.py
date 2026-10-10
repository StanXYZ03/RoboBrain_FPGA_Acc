#!/usr/bin/env python3
"""Independently validate frozen visual-refinement splits and source images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_visual_refinement_dataset import manifest_image_ids


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return records


def aggregate_image_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("ascii"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def validate_split(
    name: str,
    manifest: Path,
    expected_items: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[int], list[Path], list[str]]:
    records = load_jsonl(manifest)
    errors: list[str] = []
    image_ids = {int(record["coco_image_id"]) for record in records}
    expected_ids = {int(item["coco_image_id"]) for item in expected_items}
    if image_ids != expected_ids:
        errors.append(f"{name}: manifest image IDs differ from lock")
    if len(records) != 2 * len(expected_items):
        errors.append(f"{name}: expected {2 * len(expected_items)} records, got {len(records)}")
    sample_ids = [str(record.get("sample_id")) for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append(f"{name}: duplicate sample_id")

    paths_by_id: dict[int, Path] = {}
    records_by_id: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        image_id = int(record["coco_image_id"])
        records_by_id.setdefault(image_id, []).append(record)
        image_path = (manifest.parent / str(record["image"])).resolve()
        paths_by_id[image_id] = image_path

    category_counts: Counter[str] = Counter()
    stratum_counts: Counter[str] = Counter()
    image_bytes = 0
    for item in expected_items:
        image_id = int(item["coco_image_id"])
        category_counts[str(item["category"])] += 1
        stratum_counts["/".join(str(value) for value in item["stratum"])] += 1
        item_records = records_by_id.get(image_id, [])
        if {str(record.get("task")) for record in item_records} != {"pointing", "grounding"}:
            errors.append(f"{name}:{image_id}: expected pointing and grounding records")
            continue
        path = paths_by_id[image_id]
        if not path.is_file():
            errors.append(f"{name}:{image_id}: missing image {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.load()
        except Exception as error:  # PIL exposes several decoder-specific exceptions.
            errors.append(f"{name}:{image_id}: image decode failed: {error}")
            continue
        expected_sizes = {
            (int(record["source_image_width"]), int(record["source_image_height"]))
            for record in item_records
        }
        if expected_sizes != {(width, height)}:
            errors.append(
                f"{name}:{image_id}: decoded size {(width, height)} != manifest {sorted(expected_sizes)}"
            )
        if not all(bool(record.get("source_is_original_resolution")) for record in item_records):
            errors.append(f"{name}:{image_id}: original-resolution flag is not true")
        image_bytes += path.stat().st_size

    summary = {
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "records": len(records),
        "unique_images": len(image_ids),
        "tasks": dict(sorted(Counter(str(record.get("task")) for record in records).items())),
        "categories": dict(sorted(category_counts.items())),
        "strata": dict(sorted(stratum_counts.items())),
        "image_bytes": image_bytes,
    }
    return summary, image_ids, list(paths_by_id.values()), errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest-prefix", type=Path, required=True)
    parser.add_argument("--historical-manifest-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    errors: list[str] = []
    if sha256(args.annotations) != str(lock.get("annotations_sha256")):
        errors.append("COCO annotation SHA256 differs from lock")

    split_summaries: dict[str, Any] = {}
    split_ids: dict[str, set[int]] = {}
    all_paths: list[Path] = []
    for name in ("development", "calibration", "test"):
        manifest = Path(f"{args.manifest_prefix}_{name}.jsonl")
        summary, image_ids, paths, split_errors = validate_split(
            name, manifest, lock["splits"][name]
        )
        split_summaries[name] = summary
        split_ids[name] = image_ids
        all_paths.extend(paths)
        errors.extend(split_errors)

    names = list(split_ids)
    overlap_counts: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            overlap_counts[f"{left}__{right}"] = len(overlap)
            if overlap:
                errors.append(f"{left}/{right}: {len(overlap)} overlapping image IDs")

    historical_paths = [
        path
        for path in args.historical_manifest_dir.glob("*.jsonl")
        if not path.name.startswith("visual_refinement_")
    ]
    historical_ids, historical_counts = manifest_image_ids(historical_paths)
    selected_ids = set().union(*split_ids.values())
    historical_overlap = selected_ids & historical_ids
    if historical_overlap:
        errors.append(f"selected set overlaps {len(historical_overlap)} historical images")

    expected_files = {
        str(item["file_name"])
        for items in lock["splits"].values()
        for item in items
    }
    actual_files = {path.name for path in args.image_dir.glob("*.jpg")}
    extra_files = sorted(actual_files - expected_files)
    missing_files = sorted(expected_files - actual_files)
    part_files = sorted(path.name for path in args.image_dir.glob("*.part"))
    if extra_files:
        errors.append(f"image directory contains {len(extra_files)} extra JPEG files")
    if missing_files:
        errors.append(f"image directory misses {len(missing_files)} selected JPEG files")
    if part_files:
        errors.append(f"image directory contains {len(part_files)} incomplete .part files")

    unique_paths = sorted(set(all_paths), key=lambda item: item.name)
    report = {
        "status": "VISUAL_REFINEMENT_SPLIT_LOCK_PASS" if not errors else "VISUAL_REFINEMENT_SPLIT_LOCK_FAIL",
        "lock": str(args.lock),
        "lock_sha256": sha256(args.lock),
        "annotations_sha256": sha256(args.annotations),
        "split_summaries": split_summaries,
        "split_overlap_counts": overlap_counts,
        "historical_manifest_record_counts": historical_counts,
        "historical_unique_images": len(historical_ids),
        "historical_overlap_images": len(historical_overlap),
        "selected_unique_images": len(selected_ids),
        "image_directory_jpegs": len(actual_files),
        "image_set_sha256": aggregate_image_digest(unique_paths) if not missing_files else None,
        "total_image_bytes": sum(path.stat().st_size for path in unique_paths if path.is_file()),
        "extra_jpegs": extra_files,
        "missing_jpegs": missing_files,
        "part_files": part_files,
        "errors": errors,
        "evidence_boundary": (
            "The split is frozen and leak-checked, but COCO bounding-box centers are not "
            "affordance points and bbox overlap is not a human occlusion annotation."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 按需视觉细化数据集冻结验收",
        "",
        f"状态：`{report['status']}`。",
        "",
        "| 划分 | 图像 | Pointing记录 | Grounding记录 | 图像字节 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("development", "calibration", "test"):
        item = split_summaries[name]
        lines.append(
            f"| {name} | {item['unique_images']} | {item['tasks'].get('pointing', 0)} | "
            f"{item['tasks'].get('grounding', 0)} | {item['image_bytes']} |"
        )
    lines.extend(
        [
            "",
            f"- 三个划分共 `{len(selected_ids)}` 张唯一原始分辨率JPEG，目录实有 `{len(actual_files)}` 张。",
            f"- 划分间重叠：`{sum(overlap_counts.values())}`；与历史 `{len(historical_ids)}` 张图像重叠：`{len(historical_overlap)}`。",
            f"- 图像集合内容摘要：`{report['image_set_sha256']}`。",
            f"- 冻结文件摘要：`{report['lock_sha256']}`。",
            "- 每张图均完成PIL解码、实际尺寸与manifest尺寸核对，并核验Pointing/Grounding成对存在。",
            "",
            "证据边界：该集合可用于Grounding和目标框中心Pointing；COCO框中心不是可供性点，框重叠也不是真实人工遮挡标注。测试集在策略冻结前不得用于调参。",
        ]
    )
    if errors:
        lines.extend(["", "## 错误", ""] + [f"- {error}" for error in errors])
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": len(errors)}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
