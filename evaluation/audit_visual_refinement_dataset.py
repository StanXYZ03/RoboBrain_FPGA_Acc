#!/usr/bin/env python3
"""Audit whether a leak-free original-resolution COCO split can be built locally."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_coco_spatial_smoke import DEFAULT_CATEGORIES


def manifest_image_ids(paths: Iterable[Path]) -> tuple[set[int], dict[str, int]]:
    used: set[int] = set()
    counts: dict[str, int] = {}
    for path in sorted(set(paths)):
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if "coco_image_id" in record:
                    used.add(int(record["coco_image_id"]))
                    count += 1
        if count:
            counts[str(path)] = count
    return used, counts


def eligible_image_ids(payload: dict[str, Any]) -> set[int]:
    images = {int(item["id"]): item for item in payload["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    eligible: set[int] = set()
    for image_id, annotations in by_image.items():
        image = images[image_id]
        category_counts = Counter(int(item["category_id"]) for item in annotations)
        image_area = float(image["width"] * image["height"])
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            category = categories[category_id]
            _, _, width, height = (float(value) for value in annotation["bbox"])
            area_fraction = float(annotation.get("area", width * height)) / image_area
            if (
                category in DEFAULT_CATEGORIES
                and category_counts[category_id] == 1
                and not int(annotation.get("iscrowd", 0))
                and width >= 32
                and height >= 32
                and 0.015 <= area_fraction <= 0.45
            ):
                eligible.add(image_id)
                break
    return eligible


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--development-images", type=int, default=192)
    parser.add_argument("--calibration-images", type=int, default=96)
    parser.add_argument("--test-images", type=int, default=512)
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    annotation_ids = {int(item["id"]) for item in payload["images"]}
    manifest_paths = list(args.manifest_dir.glob("*.jsonl"))
    used_ids, manifest_counts = manifest_image_ids(manifest_paths)
    local_ids = {
        int(path.stem)
        for path in args.original_image_dir.glob("*.jpg")
        if path.stem.isdigit()
    }
    eligible_ids = eligible_image_ids(payload)
    requested = args.development_images + args.calibration_images + args.test_images
    local_unused = local_ids - used_ids
    eligible_unused = eligible_ids - used_ids
    local_eligible_unused = local_unused & eligible_ids
    status = (
        "LOCAL_ORIGINALS_SUFFICIENT"
        if len(local_eligible_unused) >= requested
        else "BLOCKED_NEEDS_ADDITIONAL_ORIGINAL_IMAGES"
    )
    report = {
        "status": status,
        "policy": "all historical COCO image IDs are excluded; resized copies do not count as new original-resolution images",
        "requested_split_images": {
            "development": args.development_images,
            "calibration": args.calibration_images,
            "test": args.test_images,
            "total": requested,
        },
        "counts": {
            "annotation_images": len(annotation_ids),
            "eligible_annotation_images": len(eligible_ids),
            "historically_used_unique_images": len(used_ids),
            "local_original_images": len(local_ids),
            "local_unused_original_images": len(local_unused),
            "eligible_unused_annotation_images": len(eligible_unused),
            "local_eligible_unused_original_images": len(local_eligible_unused),
            "additional_originals_required_at_least": max(0, requested - len(local_eligible_unused)),
        },
        "manifest_record_counts": manifest_counts,
        "missing_local_candidate_example_ids": sorted(eligible_unused - local_ids)[:20],
        "output_manifests_created": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts = report["counts"]
    lines = [
        "# 按需视觉细化数据隔离预检",
        "",
        f"状态：`{status}`。",
        "",
        "隔离规则：排除 `datasets/` 顶层全部历史JSONL中出现过的COCO图像ID；同一图像的不同尺寸副本视为同一图像；只有原始分辨率JPEG可用于高分辨率恢复实验。",
        "",
        "| 项目 | 数量 |",
        "|---|---:|",
        f"| COCO val2017标注图像 | {counts['annotation_images']} |",
        f"| 满足当前空间任务筛选条件的图像 | {counts['eligible_annotation_images']} |",
        f"| 历史已使用唯一图像 | {counts['historically_used_unique_images']} |",
        f"| 本地原始分辨率图像 | {counts['local_original_images']} |",
        f"| 本地未使用原图 | {counts['local_unused_original_images']} |",
        f"| 本地、未使用且满足筛选条件的原图 | {counts['local_eligible_unused_original_images']} |",
        f"| 计划需要的开发/校准/测试图像 | {requested} |",
        f"| 至少仍需补齐原图 | {counts['additional_originals_required_at_least']} |",
        "",
        "结论：当前不创建新manifest，因为本地原图不足。标注中仍有可选且未历史使用的图像ID，但需先获取对应官方原始JPEG；不得用已有512像素副本冒充高分辨率数据。",
        "",
    ]
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, **counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
