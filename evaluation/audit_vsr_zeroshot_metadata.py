#!/usr/bin/env python3
"""Audit pinned VSR zero-shot train/dev while leaving test JSON unparsed."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.audit_vsr_development_metadata import (
    COCO_CACHE_ROOT,
    COMMIT,
    ROOT,
    load_development_split,
    line_count_without_json_parse,
    sha256_file,
    summarize_split,
)


DATA_ROOT = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/splits/zeroshot"
EXPECTED_SHA256 = {
    "train": "62daf2c5d573667038f88feaf1d26a1154c892083a72d83ed5c7551666ed3066",
    "dev": "0abda42fe9333ab6a8671b32886738b5cfa6d37a8c277b56d8368c339c8dffce",
    "test": "914d9156723b912d8f794b179248af865a68c5f81c9d8e3a21d007778cacaad4",
}
EXPECTED_LINES = {"train": 3489, "dev": 340, "test": 1222}
REPORT_JSON = ROOT / "reports/token_selector/2026-10-09-vsr-zeroshot-data-audit.json"
REPORT_MD = ROOT / "reports/token_selector/2026-10-09-vsr-zeroshot-data-audit.md"
IMAGE_MANIFEST = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/development_image_requirements.jsonl"


def audit() -> dict[str, Any]:
    paths = {split: DATA_ROOT / f"{split}.jsonl" for split in EXPECTED_LINES}
    file_audit: dict[str, Any] = {}
    for split, path in paths.items():
        actual_sha = sha256_file(path)
        actual_lines = line_count_without_json_parse(path)
        if actual_sha != EXPECTED_SHA256[split] or actual_lines != EXPECTED_LINES[split]:
            raise ValueError(f"pinned {split} metadata mismatch")
        file_audit[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "lines": actual_lines,
        }

    train = load_development_split(paths["train"], EXPECTED_LINES["train"])
    dev = load_development_split(paths["dev"], EXPECTED_LINES["dev"])
    rows_by_split = {"train": train, "dev": dev}
    concepts = {
        split: {
            str(row[field]).strip().lower()
            for row in rows
            for field in ("subj", "obj")
        }
        for split, rows in rows_by_split.items()
    }

    existing_by_name: dict[str, list[str]] = defaultdict(list)
    for path in COCO_CACHE_ROOT.rglob("*"):
        if path.is_file():
            existing_by_name[path.name].append(str(path))

    memberships: dict[str, set[str]] = defaultdict(set)
    urls: dict[str, str] = {}
    sources: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            name = str(row["image"])
            url = str(row["image_link"])
            if name in urls and urls[name] != url:
                raise ValueError(f"conflicting URLs for {name}")
            memberships[name].add(split)
            urls[name] = url
            sources[name] = "train2017" if "/train2017/" in url else "val2017"

    manifest = [
        {
            "image": name,
            "url": urls[name],
            "source": sources[name],
            "splits": sorted(memberships[name]),
            "cached": name in existing_by_name,
            "cached_paths": sorted(existing_by_name.get(name, [])),
        }
        for name in sorted(memberships)
    ]
    IMAGE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )

    coverage = {}
    for split, rows in rows_by_split.items():
        names = {str(row["image"]) for row in rows}
        missing = names - set(existing_by_name)
        coverage[split] = {
            "unique_images": len(names),
            "cached_unique_images": len(names) - len(missing),
            "missing_unique_images": len(missing),
            "coverage_rate": (len(names) - len(missing)) / len(names),
            "missing_by_source": dict(sorted(Counter(sources[name] for name in missing).items())),
        }

    train_images = {str(row["image"]) for row in train}
    dev_images = {str(row["image"]) for row in dev}
    return {
        "status": "VSR_ZEROSHOT_METADATA_AUDIT_PASS_IMAGES_INCOMPLETE",
        "official_commit": COMMIT,
        "files": file_audit,
        "test_split_policy": {
            "json_parsed": False,
            "labels_captions_relations_inspected": False,
            "only_hash_bytes_and_line_count_checked": True,
        },
        "development_splits": {
            "train": summarize_split(train),
            "dev": summarize_split(dev),
            "train_dev_unique_image_overlap": len(train_images & dev_images),
            "train_concept_count": len(concepts["train"]),
            "dev_concept_count": len(concepts["dev"]),
            "train_dev_concept_overlap": len(concepts["train"] & concepts["dev"]),
        },
        "local_cache": {"root": str(COCO_CACHE_ROOT), "coverage": coverage},
        "image_requirements_manifest": {
            "path": str(IMAGE_MANIFEST),
            "sha256": sha256_file(IMAGE_MANIFEST),
            "unique_images": len(manifest),
            "cached_unique_images": sum(bool(row["cached"]) for row in manifest),
            "missing_unique_images": sum(not row["cached"] for row in manifest),
        },
        "decision": {
            "preferred_over_random_split": True,
            "gpu_run_ready": False,
            "next_safe_action": "Download only the 222 missing zero-shot dev images after traffic approval.",
            "test_remains_sealed": True,
        },
    }


def markdown(result: Mapping[str, Any]) -> str:
    d = result["development_splits"]
    tc = result["local_cache"]["coverage"]["train"]
    dc = result["local_cache"]["coverage"]["dev"]
    return "\n".join(
        [
            "# VSR Zero-Shot 数据审计",
            "",
            f"- 状态：`{result['status']}`。",
            f"- 官方提交：`{result['official_commit']}`。",
            "- test 只核对 SHA、字节数和行数；未解析 JSON，未读取标签、caption 或 relation。",
            "",
            "## 隔离情况",
            "",
            f"- train：{d['train']['records']} 条、{d['train']['unique_images']} 张图、{d['train']['relations']} 种关系。",
            f"- dev：{d['dev']['records']} 条、{d['dev']['unique_images']} 张图、{d['dev']['relations']} 种关系。",
            f"- train/dev 概念重叠：{d['train_dev_concept_overlap']}；图像重叠：{d['train_dev_unique_image_overlap']}。",
            "- 相比 random split 的 558 张 train/dev 图像重叠，zero-shot 更适合关系保持 Selector 的发展验证。",
            "",
            "## 本机覆盖",
            "",
            f"- train：缓存 {tc['cached_unique_images']}/{tc['unique_images']}，缺失 {tc['missing_unique_images']}。",
            f"- dev：缓存 {dc['cached_unique_images']}/{dc['unique_images']}，缺失 {dc['missing_unique_images']}。",
            "",
            "## 决策",
            "",
            "建议使用 zero-shot train/dev 发展，继续封存 zero-shot test。先只补齐 222 张 dev 图像并运行短输出预算曲线；",
            "只有关系保持候选需要训练时，才补齐 2204 张 train 图像。",
            "",
        ]
    )


def main() -> int:
    result = audit()
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_MD.write_text(markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(REPORT_JSON), "markdown": str(REPORT_MD)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
