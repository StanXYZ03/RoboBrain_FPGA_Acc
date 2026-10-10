#!/usr/bin/env python3
"""Build a leakage-controlled VSR zero-shot train manifest without opening test."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.vsr_zeroshot_dev_contract import (
    DATA_ROOT, DEV_METADATA, DEV_METADATA_SHA256, PROMPT_TEMPLATE, sha256_file,
)

TRAIN_METADATA = DATA_ROOT / "splits/zeroshot/train.jsonl"
TRAIN_METADATA_SHA256 = "62daf2c5d573667038f88feaf1d26a1154c892083a72d83ed5c7551666ed3066"
TRAIN_IMAGES = DATA_ROOT / "train_images_manifest.jsonl"
TRAIN_IMAGES_SHA256 = "52cf8086ed2ee2cf7633698ae5a4d281ec3c70911eac5b4885e9cde2d6e470cb"
OUTPUT = DATA_ROOT / "vsr_zeroshot_train_no_dev_image_overlap.jsonl"
REPORT_JSON = Path(__file__).resolve().parents[1] / "reports/token_selector/2026-10-10-vsr-zeroshot-train-manifest.json"
REPORT_MD = Path(__file__).resolve().parents[1] / "reports/token_selector/2026-10-10-vsr-zeroshot-train-manifest.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build() -> dict[str, Any]:
    expected = {
        TRAIN_METADATA: TRAIN_METADATA_SHA256,
        DEV_METADATA: DEV_METADATA_SHA256,
        TRAIN_IMAGES: TRAIN_IMAGES_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"pinned input mismatch: {path}")
    train = load_jsonl(TRAIN_METADATA)
    dev = load_jsonl(DEV_METADATA)
    images = {str(row["image"]): row for row in load_jsonl(TRAIN_IMAGES)}
    dev_images = {str(row["image"]) for row in dev}
    overlap_images = sorted({str(row["image"]) for row in train} & dev_images)
    retained_source = [row for row in train if str(row["image"]) not in dev_images]
    removed = [row for row in train if str(row["image"]) in dev_images]
    output = []
    for source_index, row in enumerate(train):
        image_name = str(row["image"])
        if image_name in dev_images:
            continue
        image = images[image_name]
        path = Path(str(image["path"]))
        if not path.is_file() or sha256_file(path) != image["sha256"]:
            raise ValueError(f"image mismatch: {image_name}")
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"invalid label at train row {source_index}")
        caption = str(row["caption"])
        output.append(
            {
                "sample_id": f"vsr-zs-train-{source_index:04d}",
                "source_row_index": source_index,
                "task": "vsr_binary_spatial_reasoning",
                "type": "binary_vqa",
                "source": "VSR",
                "source_dataset": "cambridgeltl/visual-spatial-reasoning",
                "official_split": "zero-shot-train",
                "image": str(path.resolve()),
                "image_name": image_name,
                "image_sha256": str(image["sha256"]),
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
                "caption": caption,
                "relation": str(row["relation"]),
                "subject": str(row["subj"]),
                "object": str(row["obj"]),
                "official_label": label,
                "prompt": PROMPT_TEMPLATE.format(caption=caption),
                "max_new_tokens": 4,
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output)
    OUTPUT.write_text(payload, encoding="utf-8")
    retained_images = {row["image_name"] for row in output}
    report = {
        "status": "VSR_ZEROSHOT_TRAIN_MANIFEST_COMPLETE",
        "source_train_records": len(train),
        "source_train_unique_images": len({str(row["image"]) for row in train}),
        "dev_unique_images": len(dev_images),
        "overlap_images_removed": overlap_images,
        "overlap_image_count": len(overlap_images),
        "records_removed_for_image_overlap": len(removed),
        "retained_records": len(output),
        "retained_unique_images": len(retained_images),
        "label_counts": dict(sorted(Counter(row["official_label"] for row in output).items())),
        "relation_counts": dict(sorted(Counter(row["relation"] for row in output).items())),
        "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT),
        "inputs": {str(path): sha256_file(path) for path in expected},
        "test_metadata_opened": False,
        "label_policy": "Train labels may define teacher margin targets only; labels, relation names, subject and object annotations are forbidden at selector inference.",
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(
        "\n".join(
            [
                "# VSR Zero-Shot Train 防泄漏清单", "",
                f"- 官方 train：{len(train)} 条 / {report['source_train_unique_images']} 张图。",
                f"- 与 dev 重叠：{len(overlap_images)} 张图，对应删除 {len(removed)} 条 train 记录。",
                f"- 最终训练域：{len(output)} 条 / {len(retained_images)} 张图。",
                f"- 标签：False={report['label_counts'].get(0, 0)}，True={report['label_counts'].get(1, 0)}。",
                f"- 清单 SHA256：`{report['output_sha256']}`。", "",
                "训练标签只允许构造教师 margin/蒸馏目标；正式 Selector 输入不得包含标签、relation、subject 或 object。zero-shot test 未打开。", "",
            ]
        ),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps(result, ensure_ascii=False, indent=2))
