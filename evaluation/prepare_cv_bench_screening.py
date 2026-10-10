#!/usr/bin/env python3
"""Freeze the CV-Bench paired-screening manifest from an existing BF16 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(dataset_dir: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    columns = [
        "idx", "type", "task", "question", "choices", "answer", "prompt",
        "filename", "source", "source_dataset", "image",
    ]
    for filename in ("test_2d.parquet", "test_3d.parquet"):
        parquet = pq.ParquetFile(dataset_dir / filename)
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(group_index, columns=columns)
            for row in table.to_pylist():
                sample_id = int(row["idx"])
                if sample_id in wanted:
                    row["parquet_file"] = filename
                    rows[sample_id] = row
    missing = sorted(wanted - set(rows))
    if missing:
        raise ValueError(f"CV-Bench rows missing from parquet files: {missing[:5]}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    baseline_records = baseline["records"]
    ordered_ids = [int(record["idx"]) for record in baseline_records]
    if len(ordered_ids) != len(set(ordered_ids)):
        raise ValueError("baseline contains duplicate sample IDs")

    rows = load_rows(args.dataset_dir, set(ordered_ids))
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    image_hashes: dict[str, str] = {}
    for sample_id in ordered_ids:
        row = rows[sample_id]
        image_bytes = row["image"]["bytes"]
        if not image_bytes:
            raise ValueError(f"CV-Bench sample {sample_id} has no embedded image")
        image_path = args.image_dir / f"{sample_id}.image"
        if not image_path.is_file() or image_path.read_bytes() != image_bytes:
            image_path.write_bytes(image_bytes)
        image_hashes[str(sample_id)] = hashlib.sha256(image_bytes).hexdigest()
        manifest_records.append(
            {
                "sample_id": str(sample_id),
                "task": str(row["task"]),
                "type": str(row["type"]),
                "source": str(row["source"]),
                "source_dataset": str(row["source_dataset"]),
                "image": str(image_path.resolve()),
                "prompt": str(row["prompt"]),
                "choices": list(row["choices"]),
                "answer": str(row["answer"]),
                "max_new_tokens": int(baseline["metadata"]["max_new_tokens"]),
            }
        )

    with args.output_manifest.open("w", encoding="utf-8") as stream:
        for record in manifest_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    lock = {
        "format": "cv-bench-paired-screening-lock-v1",
        "sample_count": len(manifest_records),
        "ordered_sample_ids": ordered_ids,
        "sampling": baseline["metadata"].get("sampling"),
        "seed": baseline["metadata"].get("seed"),
        "max_new_tokens": baseline["metadata"]["max_new_tokens"],
        "baseline_json": str(args.baseline_json.resolve()),
        "baseline_json_sha256": sha256(args.baseline_json),
        "dataset_files": {
            filename: sha256(args.dataset_dir / filename)
            for filename in ("test_2d.parquet", "test_3d.parquet")
        },
        "image_sha256": image_hashes,
        "manifest": str(args.output_manifest.resolve()),
        "manifest_sha256": sha256(args.output_manifest),
        "strategy": {
            "selector": "l2_anchor",
            "retention": 0.875,
            "prune_after_layer": 5,
            "anchor_grid": 3,
            "grouping": "Qwen3-VL spatial merger 2x2 patch groups",
            "position_policy": "preserve original MRoPE indices",
        },
    }
    args.lock_file.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE manifest={args.output_manifest} samples={len(manifest_records)}")
    print(f"WROTE lock={args.lock_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
