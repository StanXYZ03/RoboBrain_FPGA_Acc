#!/usr/bin/env python3
"""Build deterministic leak-free COCO splits for visual-refinement research."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_visual_refinement_dataset import manifest_image_ids
from evaluation.build_coco_spatial_smoke import (
    DEFAULT_CATEGORIES,
    download_image,
    image_is_valid,
    normalize_bbox,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def intersection_fraction(target: list[float], other: list[float]) -> float:
    tx, ty, tw, th = target
    ox, oy, ow, oh = other
    intersection = max(0.0, min(tx + tw, ox + ow) - max(tx, ox)) * max(
        0.0, min(ty + th, oy + oh) - max(ty, oy)
    )
    return intersection / (tw * th) if tw > 0 and th > 0 else 0.0


def make_candidates(payload: dict[str, Any], excluded: set[int]) -> list[dict[str, Any]]:
    images = {int(item["id"]): item for item in payload["images"]}
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)
    candidates: list[dict[str, Any]] = []
    for image_id, annotations in by_image.items():
        if image_id in excluded:
            continue
        image = images[image_id]
        image_area = float(image["width"] * image["height"])
        category_counts = Counter(int(item["category_id"]) for item in annotations)
        eligible: list[dict[str, Any]] = []
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            category = categories[category_id]
            x, y, width, height = (float(value) for value in annotation["bbox"])
            area_fraction = float(annotation.get("area", width * height)) / image_area
            if (
                category in DEFAULT_CATEGORIES
                and category_counts[category_id] == 1
                and not int(annotation.get("iscrowd", 0))
                and width >= 32
                and height >= 32
                and 0.015 <= area_fraction <= 0.45
            ):
                overlap = max(
                    (
                        intersection_fraction(annotation["bbox"], other["bbox"])
                        for other in annotations
                        if int(other["id"]) != int(annotation["id"])
                    ),
                    default=0.0,
                )
                normalized = normalize_bbox(annotation, int(image["width"]), int(image["height"]))
                edge = normalized[0] <= 100 or normalized[1] <= 100 or normalized[2] >= 900 or normalized[3] >= 900
                eligible.append(
                    {
                        "image": image,
                        "annotation": annotation,
                        "category": category,
                        "area_fraction": area_fraction,
                        "area_bin": "small" if area_fraction < 0.05 else "medium" if area_fraction < 0.15 else "large",
                        "edge_bin": "edge" if edge else "interior",
                        "overlap_proxy": overlap,
                        "overlap_bin": "overlap" if overlap >= 0.2 else "clear",
                        "complexity_bin": "sparse" if len(annotations) <= 5 else "medium" if len(annotations) <= 12 else "dense",
                    }
                )
        if eligible:
            eligible.sort(key=lambda item: int(item["annotation"]["id"]))
            candidates.append(eligible[0])
    return candidates


def proportional_sample(candidates: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if count > len(candidates):
        raise ValueError(f"requested {count} candidates from {len(candidates)}")
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        key = (item["area_bin"], item["edge_bin"], item["overlap_bin"], item["complexity_bin"])
        buckets[key].append(item)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    exact = {key: count * len(bucket) / len(candidates) for key, bucket in buckets.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    order = sorted(buckets, key=lambda key: (exact[key] - quotas[key], str(key)), reverse=True)
    for key in order[:remaining]:
        quotas[key] += 1
    selected = [item for key, bucket in buckets.items() for item in bucket[: quotas[key]]]
    rng.shuffle(selected)
    return selected


def assign_splits(
    selected: list[dict[str, Any]], split_sizes: dict[str, int], seed: int
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed + 1)
    by_stratum: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        key = (item["area_bin"], item["edge_bin"], item["overlap_bin"], item["complexity_bin"])
        by_stratum[key].append(item)
    result = {name: [] for name in split_sizes}
    remaining = dict(split_sizes)
    for key in sorted(by_stratum):
        bucket = by_stratum[key]
        rng.shuffle(bucket)
        for item in bucket:
            available = [name for name, count in remaining.items() if count > 0]
            chosen = max(available, key=lambda name: (remaining[name] / split_sizes[name], name))
            result[chosen].append(item)
            remaining[chosen] -= 1
    if any(remaining.values()):
        raise RuntimeError(f"split assignment incomplete: {remaining}")
    return result


def manifest_records(
    items: list[dict[str, Any]], role: str, image_dir: Path, manifest_parent: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        image = item["image"]
        annotation = item["annotation"]
        image_path = image_dir / str(image["file_name"])
        bbox = normalize_bbox(annotation, int(image["width"]), int(image["height"]))
        point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        common = {
            "source": "COCO 2017 val instances",
            "source_url": "https://cocodataset.org/#download",
            "dataset_role": f"visual_refinement_{role}",
            "sealed_test": role == "test",
            "image": str(image_path.relative_to(manifest_parent)),
            "image_width": int(image["width"]),
            "image_height": int(image["height"]),
            "source_image_width": int(image["width"]),
            "source_image_height": int(image["height"]),
            "source_is_original_resolution": True,
            "coco_image_id": int(image["id"]),
            "coco_annotation_id": int(annotation["id"]),
            "category": item["category"],
            "coordinate_system": "normalized_0_1000",
            "ground_truth": {"point_1000": point, "bbox_1000": bbox},
            "stratification": {
                "target_area_fraction": item["area_fraction"],
                "target_area_bin": item["area_bin"],
                "edge_bin": item["edge_bin"],
                "bbox_overlap_proxy": item["overlap_proxy"],
                "overlap_bin": item["overlap_bin"],
                "scene_complexity_bin": item["complexity_bin"],
                "occlusion_note": "bbox overlap is a proxy, not a human occlusion label",
            },
        }
        image_id = int(image["id"])
        annotation_id = int(annotation["id"])
        category = item["category"]
        records.extend(
            [
                {
                    **common,
                    "sample_id": f"coco-val-{image_id:012d}-{annotation_id}-pointing",
                    "task": "pointing",
                    "prompt": f"Point to the center of the only {category} in the image. Return exactly one point as [(x, y)], where x and y are integers normalized to 0-1000.",
                    "max_new_tokens": 24,
                },
                {
                    **common,
                    "sample_id": f"coco-val-{image_id:012d}-{annotation_id}-grounding",
                    "task": "grounding",
                    "prompt": f"Locate the only {category} in the image. Return exactly one bounding box as [x1, y1, x2, y2], where all coordinates are integers normalized to 0-1000.",
                    "max_new_tokens": 32,
                },
            ]
        )
    return sorted(records, key=lambda item: (item["task"], item["sample_id"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--development-images", type=int, default=192)
    parser.add_argument("--calibration-images", type=int, default=96)
    parser.add_argument("--test-images", type=int, default=512)
    parser.add_argument("--download-images", action="store_true")
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()

    historical = [
        path for path in args.manifest_dir.glob("*.jsonl")
        if not path.name.startswith("visual_refinement_")
    ]
    excluded, manifest_counts = manifest_image_ids(historical)
    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    candidates = make_candidates(payload, excluded)
    split_sizes = {
        "development": args.development_images,
        "calibration": args.calibration_images,
        "test": args.test_images,
    }
    selected = proportional_sample(candidates, sum(split_sizes.values()), args.seed)
    splits = assign_splits(selected, split_sizes, args.seed)
    lock = {
        "schema_version": "1.0",
        "seed": args.seed,
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "historical_manifest_record_counts": manifest_counts,
        "historically_excluded_unique_images": len(excluded),
        "eligible_unused_images": len(candidates),
        "split_sizes": split_sizes,
        "stratification": ["target_area_bin", "edge_bin", "bbox_overlap_proxy", "scene_complexity_bin"],
        "occlusion_limitation": "COCO has no complete occlusion label; bbox overlap is only a proxy",
        "splits": {
            name: [
                {
                    "coco_image_id": int(item["image"]["id"]),
                    "coco_annotation_id": int(item["annotation"]["id"]),
                    "file_name": item["image"]["file_name"],
                    "category": item["category"],
                    "stratum": [item["area_bin"], item["edge_bin"], item["overlap_bin"], item["complexity_bin"]],
                }
                for item in items
            ]
            for name, items in splits.items()
        },
    }
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    args.lock.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    downloaded = 0
    if args.download_images:
        if not 1 <= args.download_workers <= 16:
            parser.error("--download-workers must be between 1 and 16")
        args.image_dir.mkdir(parents=True, exist_ok=True)
        total = sum(split_sizes.values())
        destinations = [args.image_dir / str(item["image"]["file_name"]) for item in selected]
        missing = [destination for destination in destinations if not image_is_valid(destination)]

        def fetch(destination: Path) -> Path:
            download_image(
                f"http://images.cocodataset.org/val2017/{destination.name}", destination
            )
            return destination

        completed_count = total - len(missing)
        with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
            futures = [pool.submit(fetch, destination) for destination in missing]
            for future in as_completed(futures):
                future.result()
                downloaded += 1
                completed_count += 1
                if completed_count % 25 == 0 or completed_count == total:
                    print(
                        f"IMAGES {completed_count}/{total} downloaded_this_run={downloaded}",
                        flush=True,
                    )
        invalid = [str(destination) for destination in destinations if not image_is_valid(destination)]
        if invalid:
            raise RuntimeError(f"download validation failed for {len(invalid)} images")
        for name, items in splits.items():
            path = Path(f"{args.output_prefix}_{name}.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            records = manifest_records(items, name, args.image_dir, path.parent)
            with path.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "lock": str(args.lock),
                "eligible_unused_images": len(candidates),
                "split_sizes": split_sizes,
                "download_images": args.download_images,
                "downloaded": downloaded,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
