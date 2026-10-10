#!/usr/bin/env python3
"""Materialize and validate only the pinned VSR zero-shot train images."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.materialize_vsr_zeroshot_dev_images import (
    REQUIREMENTS,
    RETRIES,
    ROOT,
    WORKERS,
    choose_original_cache,
    sha256_file,
    validate_image,
)

OUTPUT_DIR = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/train_images"
DEV_DIR = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/dev_images"
OUTPUT_MANIFEST = ROOT / "datasets/external/vsr-zeroshot-b27a0af0/train_images_manifest.jsonl"
REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-zeroshot-train-images.json"
EXPECTED_TRAIN_IMAGES = 2249


def materialize_one(requirement: Mapping[str, Any]) -> dict[str, Any]:
    name = str(requirement["image"])
    destination = OUTPUT_DIR / name
    if destination.is_file():
        return {**requirement, "path": str(destination), "materialization": "resume", **validate_image(destination)}

    local_candidates = [DEV_DIR / name]
    cached = choose_original_cache(list(requirement.get("cached_paths", [])))
    if cached is not None:
        local_candidates.append(cached)
    source = next((path for path in local_candidates if path.is_file()), None)
    if source is not None:
        temporary = destination.with_suffix(destination.suffix + ".part")
        shutil.copyfile(source, temporary)
        validate_image(temporary)
        temporary.replace(destination)
        return {
            **requirement, "path": str(destination),
            "materialization": "copied_local_original", "source_path": str(source),
            **validate_image(destination),
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
            return {
                **requirement, "path": str(destination),
                "materialization": "downloaded_official_url", "attempt": attempt,
                **validate_image(destination),
            }
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            time.sleep(attempt)
    raise RuntimeError(f"failed to materialize {name} from {url}: {last_error}")


def main() -> int:
    requirements = [
        json.loads(line) for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train = [row for row in requirements if "train" in row["splits"]]
    if len(train) != EXPECTED_TRAIN_IMAGES or len({row["image"] for row in train}) != EXPECTED_TRAIN_IMAGES:
        raise ValueError("VSR zero-shot train image requirement count mismatch")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(materialize_one, row): str(row["image"]) for row in train}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"image": name, "error": f"{type(exc).__name__}: {exc}"})
            if completed % 100 == 0 or completed == len(futures):
                print(json.dumps({"completed": completed, "total": len(futures), "failures": len(failures)}), flush=True)

    results.sort(key=lambda row: str(row["image"]))
    OUTPUT_MANIFEST.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    complete = not failures and len(results) == EXPECTED_TRAIN_IMAGES
    report = {
        "status": "VSR_ZEROSHOT_TRAIN_IMAGES_COMPLETE" if complete else "VSR_ZEROSHOT_TRAIN_IMAGES_INCOMPLETE",
        "requirements": str(REQUIREMENTS), "requirements_sha256": sha256_file(REQUIREMENTS),
        "output_dir": str(OUTPUT_DIR), "output_manifest": str(OUTPUT_MANIFEST),
        "output_manifest_sha256": sha256_file(OUTPUT_MANIFEST),
        "expected": EXPECTED_TRAIN_IMAGES, "materialized": len(results),
        "total_bytes": sum(int(row["bytes"]) for row in results),
        "materialization_counts": dict(sorted(Counter(row["materialization"] for row in results).items())),
        "failures": failures, "test_metadata_opened": False,
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
