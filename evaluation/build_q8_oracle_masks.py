#!/usr/bin/env python3
"""Build development-only geometric-oracle group masks for Q8 cut-point gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl
from evaluation.shallow_block_pruning import geometric_oracle_group_indices


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-count", type=int, default=8)
    parser.add_argument(
        "--baseline-records",
        type=Path,
        default=ROOT / "evaluation/results/cutpoint_oracle_dev/oracle_post_merger_r1000.jsonl",
    )
    args = parser.parse_args()
    if not (0.0 < args.retention < 1.0):
        parser.error("--retention must be in (0, 1)")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    manifest = ROOT / "datasets/dac_cutpoint_oracle_dev.jsonl"
    all_samples = load_jsonl(manifest)
    baseline_by_id = {
        record["sample_id"]: record for record in load_jsonl(args.baseline_records)
    }
    samples = [sample for sample in all_samples if sample["sample_id"] in baseline_by_id]
    if len(samples) != args.require_count:
        raise SystemExit(
            f"expected {args.require_count} samples with baseline records, found {len(samples)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for sample in samples:
            baseline = baseline_by_id[sample["sample_id"]]
            _, patch_h, patch_w = (int(value) for value in baseline["image_grid_thw"])
            if patch_h % 2 or patch_w % 2:
                raise ValueError(f"non-merge-aligned grid for {sample['sample_id']}")
            grid_h, grid_w = patch_h // 2, patch_w // 2
            selected, _ = geometric_oracle_group_indices(
                grid_h,
                grid_w,
                args.retention,
                sample["ground_truth"]["bbox_1000"],
                anchor_grid=3,
                device="cpu",
            )
            full_tokens = grid_h * grid_w
            expected = math.ceil(full_tokens * args.retention)
            if selected.numel() != expected:
                raise RuntimeError("oracle selector returned an unexpected token count")
            record = {
                "format": "q8-geometric-oracle-mask-v1",
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "requested_retention": args.retention,
                "baseline_llm_visual_tokens": full_tokens,
                "selected_group_indices": [int(value) for value in selected.tolist()],
                "selected_group_indices_sha256": hashlib.sha256(
                    selected.numpy().astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "oracle_uses_ground_truth_bbox": True,
                "not_deployable": True,
                "source_manifest_sha256": sha256(manifest),
                "source_baseline_records_sha256": sha256(args.baseline_records),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(samples), "sha256": sha256(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
