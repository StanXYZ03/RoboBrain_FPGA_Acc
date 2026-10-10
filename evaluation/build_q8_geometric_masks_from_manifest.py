#!/usr/bin/env python3
"""Build geometric-oracle Q8 masks directly from a labeled image manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from transformers import AutoProcessor

from evaluation.metrics import load_jsonl
from evaluation.shallow_block_pruning import geometric_oracle_group_indices


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention", type=float, required=True)
    parser.add_argument("--processor", type=Path, default=Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B"))
    args = parser.parse_args()
    if not 0.0 < args.retention < 1.0:
        parser.error("--retention must be in (0, 1)")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")

    samples = load_jsonl(args.manifest)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    grid_cache: dict[str, tuple[int, int, int]] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for ordinal, sample in enumerate(samples, 1):
            image_key = str(sample["image"])
            if image_key not in grid_cache:
                image_path = args.manifest.parent / image_key
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(sample["prompt"])},
                ]}]
                prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
                grid_cache[image_key] = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
            temporal, patch_h, patch_w = grid_cache[image_key]
            if temporal != 1 or patch_h % 2 or patch_w % 2:
                raise ValueError(f"unsupported grid for {sample['sample_id']}: {(temporal, patch_h, patch_w)}")
            grid_h, grid_w = patch_h // 2, patch_w // 2
            selected, _ = geometric_oracle_group_indices(
                grid_h, grid_w, args.retention,
                sample["ground_truth"]["bbox_1000"],
                anchor_grid=3, device="cpu",
            )
            full_tokens = grid_h * grid_w
            if selected.numel() != math.ceil(full_tokens * args.retention):
                raise RuntimeError("unexpected retained-token count")
            record = {
                "format": "q8-geometric-oracle-mask-v2",
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "image": image_key,
                "image_grid_thw": [temporal, patch_h, patch_w],
                "requested_retention": args.retention,
                "baseline_llm_visual_tokens": full_tokens,
                "selected_group_indices": [int(value) for value in selected.tolist()],
                "selected_group_indices_sha256": hashlib.sha256(
                    selected.numpy().astype("<i8", copy=False).tobytes()
                ).hexdigest(),
                "oracle_uses_ground_truth_bbox": True,
                "not_deployable": True,
                "source_manifest_sha256": sha256(args.manifest),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            if ordinal % 32 == 0 or ordinal == len(samples):
                print(f"MASKS {ordinal}/{len(samples)} images_cached={len(grid_cache)}", flush=True)
    print(json.dumps({
        "output": str(args.output), "records": len(samples),
        "unique_images": len(grid_cache), "sha256": sha256(args.output),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
