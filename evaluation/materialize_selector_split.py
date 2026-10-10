#!/usr/bin/env python3
"""Materialize image-level train/calibration manifests from a selector summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl


def image_id(sample_id: str) -> str:
    return sample_id.rsplit("-", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    key = f"{args.split}_image_ids"
    selected_images = set(summary[key])
    samples = [
        sample for sample in load_jsonl(args.manifest)
        if image_id(str(sample["sample_id"])) in selected_images
    ]
    observed_images = {image_id(str(sample["sample_id"])) for sample in samples}
    if observed_images != selected_images:
        raise ValueError("training summary references images absent from the source manifest")
    task_counts: dict[str, int] = {}
    for sample in samples:
        task = str(sample["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "images": len(selected_images),
                "records": len(samples),
                "tasks": task_counts,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
