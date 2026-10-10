#!/usr/bin/env python3
"""Build a deterministic labeled Pointing/Grounding smoke set from COCO val2017.

This is a pipeline gate, not the final publication benchmark. Pointing targets
are defined as centers of labeled COCO boxes so that every prediction has a
real, reproducible annotation rather than a model-generated pseudo-label.
"""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_CATEGORIES = {
    "backpack", "bed", "bench", "bicycle", "book", "bottle", "bowl",
    "cell phone", "chair", "clock", "couch", "cup", "dining table",
    "fork", "handbag", "keyboard", "knife", "laptop", "microwave",
    "mouse", "oven", "potted plant", "refrigerator", "remote", "scissors",
    "sink", "spoon", "suitcase", "teddy bear", "toaster", "toilet", "vase",
}


def normalize_bbox(annotation: dict[str, Any], width: int, height: int) -> list[float]:
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    return [
        1000.0 * x / width,
        1000.0 * y / height,
        1000.0 * (x + box_width) / width,
        1000.0 * (y + box_height) / height,
    ]


def resize_image(source: Path, destination: Path, max_side: int) -> tuple[int, int]:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="JPEG", quality=95, subsampling=0)
        return image.size


def image_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError):
        return False


def download_image(url: str, destination: Path, attempts: int = 6) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            print(f"DOWNLOAD attempt={attempt + 1}/{attempts} {url}", flush=True)
            urllib.request.urlretrieve(url, temporary)
            if not image_is_valid(temporary):
                raise OSError(f"downloaded file is not a valid image: {temporary}")
            temporary.replace(destination)
            return
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"failed to download {url} after {attempts} attempts") from last_error


def choose_annotations(
    payload: dict[str, Any],
    count: int,
    seed: int,
    *,
    max_per_category: int = 2,
    excluded_image_ids: set[int] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    images = {int(image["id"]): image for image in payload["images"]}
    categories = {int(category["id"]): str(category["name"]) for category in payload["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    excluded_image_ids = excluded_image_ids or set()
    candidates = []
    for image_id, annotations in by_image.items():
        if image_id in excluded_image_ids:
            continue
        image = images[image_id]
        category_counts = Counter(int(annotation["category_id"]) for annotation in annotations)
        image_area = float(image["width"] * image["height"])
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            category = categories[category_id]
            x, y, width, height = (float(value) for value in annotation["bbox"])
            area_fraction = float(annotation.get("area", width * height)) / image_area
            if category not in DEFAULT_CATEGORIES:
                continue
            if category_counts[category_id] != 1 or int(annotation.get("iscrowd", 0)):
                continue
            if width < 32 or height < 32 or not (0.015 <= area_fraction <= 0.45):
                continue
            candidates.append((image, annotation, category))

    random.Random(seed).shuffle(candidates)
    selected = []
    category_use: Counter[str] = Counter()
    image_use: set[int] = set()
    for candidate in candidates:
        image, _, category = candidate
        if int(image["id"]) in image_use or category_use[category] >= max_per_category:
            continue
        selected.append(candidate)
        image_use.add(int(image["id"]))
        category_use[category] += 1
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"requested {count} images but only selected {len(selected)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--eval-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--max-per-category", type=int, default=2)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--dataset-role",
        choices=(
            "smoke",
            "expanded_eval",
            "selector_validation",
            "routing_test",
            "cutpoint_oracle_dev",
        ),
        default="smoke",
    )
    parser.add_argument("--download-images", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    excluded_image_ids: set[int] = set()
    for exclude_manifest in args.exclude_manifest:
        with exclude_manifest.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    excluded_image_ids.add(int(json.loads(line)["coco_image_id"]))
    selected = choose_annotations(
        payload,
        args.count,
        args.seed,
        max_per_category=args.max_per_category,
        excluded_image_ids=excluded_image_ids,
    )
    records = []
    for image, annotation, category in selected:
        filename = str(image["file_name"])
        original_path = args.original_image_dir / filename
        if not image_is_valid(original_path):
            if not args.download_images:
                raise FileNotFoundError(original_path)
            # COCO also serves this endpoint over HTTP.  Some institutional
            # HTTPS proxies present a mismatched certificate for this host.
            url = f"http://images.cocodataset.org/val2017/{filename}"
            download_image(url, original_path)
        eval_path = args.eval_image_dir / filename
        resized_width, resized_height = resize_image(original_path, eval_path, args.max_side)
        bbox = normalize_bbox(annotation, int(image["width"]), int(image["height"]))
        point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        common = {
            "source": "COCO 2017 val instances",
            "source_url": "https://cocodataset.org/#download",
            "smoke_only": args.dataset_role == "smoke",
            "dataset_role": args.dataset_role,
            "image": str(eval_path.relative_to(args.output.parent)),
            "image_width": resized_width,
            "image_height": resized_height,
            "source_image_width": int(image["width"]),
            "source_image_height": int(image["height"]),
            "coco_image_id": int(image["id"]),
            "coco_annotation_id": int(annotation["id"]),
            "category": category,
            "coordinate_system": "normalized_0_1000",
            "ground_truth": {"point_1000": point, "bbox_1000": bbox},
        }
        records.append(
            {
                **common,
                "sample_id": f"coco-val-{int(image['id']):012d}-{int(annotation['id'])}-pointing",
                "task": "pointing",
                "prompt": (
                    f"Point to the center of the only {category} in the image. "
                    "Return exactly one point as [(x, y)], where x and y are integers "
                    "normalized to 0-1000."
                ),
                "max_new_tokens": 24,
            }
        )
        records.append(
            {
                **common,
                "sample_id": f"coco-val-{int(image['id']):012d}-{int(annotation['id'])}-grounding",
                "task": "grounding",
                "prompt": (
                    f"Locate the only {category} in the image. Return exactly one bounding "
                    "box as [x1, y1, x2, y2], where all coordinates are integers normalized "
                    "to 0-1000."
                ),
                "max_new_tokens": 32,
            }
        )

    records.sort(key=lambda record: (record["task"], record["sample_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "images": len(selected),
                "samples": len(records),
                "tasks": dict(Counter(record["task"] for record in records)),
                "max_side": args.max_side,
                "seed": args.seed,
                "max_per_category": args.max_per_category,
                "excluded_images": len(excluded_image_ids),
                "dataset_role": args.dataset_role,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
