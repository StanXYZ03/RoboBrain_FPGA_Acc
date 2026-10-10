#!/usr/bin/env python3
"""Audit pinned VSR train/dev metadata without parsing the sealed test split."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b27a0af0ee1462d2b6b92c8c83e869d9254a241a"
DATA_ROOT = ROOT / "datasets/external/vsr-random-b27a0af0/splits/random"
COCO_CACHE_ROOT = ROOT / "datasets/coco2017"
EXPECTED_SHA256 = {
    "train": "8b725a8621f2d0f94745f1fefdcf3ff8a7eb96fd358676296ff9c9a4ff463c6e",
    "dev": "90de24b811597a913310d043758f1119a1ba9ebf1b1ca8f1e27c4907f01980d4",
    "test": "8ade82a0b93ac9dc1e53f6cf1f11e9d5536776a3715102b6b4d27d4f81d551cc",
}
EXPECTED_LINES = {"train": 7680, "dev": 1097, "test": 2195}
REQUIRED_FIELDS = {"image", "image_link", "caption", "label", "relation"}

REPORT_JSON = ROOT / "reports/token_selector/2026-10-09-vsr-development-data-audit.json"
REPORT_MD = ROOT / "reports/token_selector/2026-10-09-vsr-development-data-audit.md"
IMAGE_MANIFEST = ROOT / "datasets/external/vsr-random-b27a0af0/development_image_requirements.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_count_without_json_parse(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def load_development_split(path: Path, expected_lines: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_lines:
        raise ValueError(f"{path} has {len(rows)} rows, expected {expected_lines}")
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"{path}:{index + 1} missing {sorted(missing)}")
        if int(row["label"]) not in (0, 1):
            raise ValueError(f"{path}:{index + 1} has non-binary label")
        if Path(str(row["image_link"])).name != str(row["image"]):
            raise ValueError(f"{path}:{index + 1} image/link basename mismatch")
    return rows


def summarize_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    images = {str(row["image"]) for row in rows}
    return {
        "records": len(rows),
        "unique_images": len(images),
        "relations": len({str(row["relation"]) for row in rows}),
        "label_counts": dict(sorted(Counter(str(int(row["label"])) for row in rows).items())),
        "source_record_counts": dict(
            sorted(
                Counter(
                    "train2017" if "/train2017/" in str(row["image_link"]) else "val2017"
                    for row in rows
                ).items()
            )
        ),
    }


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
    existing_by_name: dict[str, list[str]] = defaultdict(list)
    for path in COCO_CACHE_ROOT.rglob("*"):
        if path.is_file():
            existing_by_name[path.name].append(str(path))

    memberships: dict[str, set[str]] = defaultdict(set)
    image_urls: dict[str, str] = {}
    image_sources: dict[str, str] = {}
    for split, rows in rows_by_split.items():
        for row in rows:
            name = str(row["image"])
            url = str(row["image_link"])
            if name in image_urls and image_urls[name] != url:
                raise ValueError(f"conflicting URLs for {name}")
            memberships[name].add(split)
            image_urls[name] = url
            image_sources[name] = "train2017" if "/train2017/" in url else "val2017"

    manifest_rows = []
    for name in sorted(memberships):
        manifest_rows.append(
            {
                "image": name,
                "url": image_urls[name],
                "source": image_sources[name],
                "splits": sorted(memberships[name]),
                "cached": name in existing_by_name,
                "cached_paths": sorted(existing_by_name.get(name, [])),
            }
        )
    IMAGE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )

    coverage = {}
    for split, rows in rows_by_split.items():
        names = {str(row["image"]) for row in rows}
        cached = names & set(existing_by_name)
        missing = names - set(existing_by_name)
        coverage[split] = {
            "unique_images": len(names),
            "cached_unique_images": len(cached),
            "missing_unique_images": len(missing),
            "coverage_rate": len(cached) / len(names),
            "missing_by_source": dict(sorted(Counter(image_sources[name] for name in missing).items())),
        }

    train_images = {str(row["image"]) for row in train}
    dev_images = {str(row["image"]) for row in dev}
    return {
        "status": "VSR_METADATA_AUDIT_PASS_IMAGES_INCOMPLETE",
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
        },
        "local_cache": {
            "root": str(COCO_CACHE_ROOT),
            "files_scanned": sum(len(paths) for paths in existing_by_name.values()),
            "coverage": coverage,
        },
        "image_requirements_manifest": {
            "path": str(IMAGE_MANIFEST),
            "sha256": sha256_file(IMAGE_MANIFEST),
            "unique_images": len(manifest_rows),
            "cached_unique_images": sum(bool(row["cached"]) for row in manifest_rows),
            "missing_unique_images": sum(not row["cached"] for row in manifest_rows),
        },
        "decision": {
            "gpu_run_ready": False,
            "reason": "Only about two percent of required train/dev COCO images are cached.",
            "next_safe_action": "Download only the 997 missing dev images first, after explicit traffic approval.",
        },
    }


def markdown(result: Mapping[str, Any]) -> str:
    train = result["development_splits"]["train"]
    dev = result["development_splits"]["dev"]
    tc = result["local_cache"]["coverage"]["train"]
    dc = result["local_cache"]["coverage"]["dev"]
    return "\n".join(
        [
            "# VSR 发展数据审计",
            "",
            f"- 状态：`{result['status']}`。",
            f"- 官方提交：`{result['official_commit']}`。",
            "- test 文件只核对 SHA、字节数和行数；未解析 JSON，未读取标签、caption 或 relation。",
            "",
            "## 元数据",
            "",
            f"- train：{train['records']} 条，{train['unique_images']} 张图，{train['relations']} 种关系。",
            f"- dev：{dev['records']} 条，{dev['unique_images']} 张图，{dev['relations']} 种关系。",
            f"- train/dev 唯一图像重叠：{result['development_splits']['train_dev_unique_image_overlap']} 张。",
            "",
            "## 本机覆盖",
            "",
            f"- train：缓存 {tc['cached_unique_images']}/{tc['unique_images']}，缺失 {tc['missing_unique_images']}，覆盖率 {100*tc['coverage_rate']:.2f}%。",
            f"- dev：缓存 {dc['cached_unique_images']}/{dc['unique_images']}，缺失 {dc['missing_unique_images']}，覆盖率 {100*dc['coverage_rate']:.2f}%。",
            f"- 去重后的 train+dev 缺失图像总数：{result['image_requirements_manifest']['missing_unique_images']}。",
            "",
            "## 决策",
            "",
            "当前不能启动 VSR GPU 实验。建议先只下载 997 张缺失 dev 图像，完成 Dense/L2/v2 的短输出预算曲线；",
            "是否再下载 train 图像，取决于 dev 诊断是否支持继续训练关系保持 Selector。",
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
