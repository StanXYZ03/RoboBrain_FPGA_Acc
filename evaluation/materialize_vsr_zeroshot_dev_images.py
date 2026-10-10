#!/usr/bin/env python3
"""Materialize and validate only the pinned VSR zero-shot dev images."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/development_image_requirements.jsonl"
OUTPUT_DIR = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/dev_images"
OUTPUT_MANIFEST = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/dev_images_manifest.jsonl"
REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-dev-images.json"
EXPECTED_DEV_IMAGES = 227
WORKERS = 8
RETRIES = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
        image_format = image.format
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions for {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "width": width,
        "height": height,
        "mode": mode,
        "format": image_format,
    }


def choose_original_cache(paths: list[str]) -> Path | None:
    candidates = [
        Path(path)
        for path in paths
        if "original" in Path(path).parent.name.lower() and Path(path).is_file()
    ]
    return sorted(candidates)[0] if candidates else None


def materialize_one(requirement: Mapping[str, Any]) -> dict[str, Any]:
    name = str(requirement["image"])
    destination = OUTPUT_DIR / name
    if destination.is_file():
        metadata = validate_image(destination)
        return {**requirement, "path": str(destination), "materialization": "resume", **metadata}

    cached = choose_original_cache(list(requirement.get("cached_paths", [])))
    if cached is not None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(cached, temporary)
        validate_image(temporary)
        temporary.replace(destination)
        metadata = validate_image(destination)
        return {
            **requirement,
            "path": str(destination),
            "materialization": "copied_local_original",
            "source_path": str(cached),
            **metadata,
        }

    url = str(requirement["url"])
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        temporary = destination.with_suffix(destination.suffix + f".part.{attempt}")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "RoboBrain-VSR-audit/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            validate_image(temporary)
            temporary.replace(destination)
            metadata = validate_image(destination)
            return {
                **requirement,
                "path": str(destination),
                "materialization": "downloaded_official_url",
                "attempt": attempt,
                **metadata,
            }
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(attempt)
    raise RuntimeError(f"failed to materialize {name} from {url}: {last_error}")


def main() -> int:
    requirements = [
        json.loads(line)
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev = [row for row in requirements if "dev" in row["splits"]]
    if len(dev) != EXPECTED_DEV_IMAGES or len({row["image"] for row in dev}) != EXPECTED_DEV_IMAGES:
        raise ValueError("VSR zero-shot dev image requirement count mismatch")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(materialize_one, row): str(row["image"]) for row in dev}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"image": name, "error": f"{type(exc).__name__}: {exc}"})
            if completed % 25 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures), "failures": len(failures)}), flush=True)

    results.sort(key=lambda row: str(row["image"]))
    OUTPUT_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    report = {
        "status": "VSR_ZEROSHOT_DEV_IMAGES_COMPLETE" if not failures and len(results) == EXPECTED_DEV_IMAGES else "VSR_ZEROSHOT_DEV_IMAGES_INCOMPLETE",
        "requirements": str(REQUIREMENTS),
        "requirements_sha256": sha256_file(REQUIREMENTS),
        "output_dir": str(OUTPUT_DIR),
        "output_manifest": str(OUTPUT_MANIFEST),
        "output_manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "expected": EXPECTED_DEV_IMAGES,
        "materialized": len(results),
        "total_bytes": sum(int(row["bytes"]) for row in results),
        "materialization_counts": dict(sorted(Counter(row["materialization"] for row in results).items())),
        "failures": failures,
        "test_metadata_opened": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "VSR_ZEROSHOT_DEV_IMAGES_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
