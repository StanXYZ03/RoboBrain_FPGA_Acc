#!/usr/bin/env python3
"""CPU-only preflight for Qwen3-VL global-image plus ROI input contracts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from transformers import AutoProcessor

from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import resolve_image
from evaluation.token_reduction import expanded_bbox_1000
from evaluation.visual_refinement import (
    crop_image_exact_1000,
    grid_token_counts,
    make_global_roi_messages,
    resize_long_edge,
)


def choose_samples(records: list[dict[str, Any]], limit_per_task: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for record in records:
        task = str(record["task"])
        if counts.get(task, 0) >= limit_per_task:
            continue
        chosen.append(record)
        counts[task] = counts.get(task, 0) + 1
    return chosen


def original_image_path(manifest: Path, record: dict[str, Any], original_dir: Path) -> Path:
    image_id = record.get("coco_image_id")
    if image_id is not None:
        candidate = original_dir / f"{int(image_id):012d}.jpg"
        if candidate.is_file():
            return candidate
    return resolve_image(manifest, str(record["image"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--limit-per-task", type=int, default=2)
    parser.add_argument("--global-long-edge", type=int, default=256)
    parser.add_argument("--roi-long-edge", type=int, default=384)
    parser.add_argument("--oracle-margin", type=float, default=0.5)
    args = parser.parse_args()

    if args.limit_per_task <= 0:
        parser.error("--limit-per-task must be positive")
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    merge_size = int(processor.image_processor.merge_size)
    minimum_pixels = int(processor.image_processor.patch_size * merge_size) ** 2
    maximum_pixels = int(processor.image_processor.size["longest_edge"])
    samples = choose_samples(load_jsonl(args.manifest), args.limit_per_task)
    records: list[dict[str, Any]] = []

    for sample in samples:
        source = original_image_path(args.manifest, sample, args.original_image_dir)
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        requested_crop = expanded_bbox_1000(
            sample["ground_truth"]["bbox_1000"], args.oracle_margin
        )
        roi, effective_crop, pixel_crop = crop_image_exact_1000(original, requested_crop)
        global_image = resize_long_edge(original, args.global_long_edge)
        roi_image = resize_long_edge(roi, args.roi_long_edge)
        artifact_paths = None
        if args.artifact_dir is not None:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            stem = str(sample["sample_id"])
            global_path = args.artifact_dir / f"{stem}-global.jpg"
            roi_path = args.artifact_dir / f"{stem}-roi.jpg"
            global_image.save(global_path, format="JPEG", quality=95, subsampling=0)
            roi_image.save(roi_path, format="JPEG", quality=95, subsampling=0)
            artifact_paths = {
                "global": str(global_path.resolve()),
                "roi": str(roi_path.resolve()),
            }
        messages = make_global_roi_messages(str(sample["prompt"]))
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[chat_text],
            images=[global_image, roi_image],
            padding=True,
            return_tensors="pt",
            images_kwargs={"min_pixels": minimum_pixels, "max_pixels": maximum_pixels},
        )
        token_counts = grid_token_counts(
            inputs.image_grid_thw.tolist(), merge_size=merge_size
        )
        placeholders = int((inputs.input_ids == int(processor.image_token_id)).sum().item())
        records.append(
            {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "source_image": str(source.resolve()),
                "source_is_original_resolution": source.parent.resolve()
                == args.original_image_dir.resolve(),
                "source_size": list(original.size),
                "global_prepared_size": list(global_image.size),
                "roi_source_size": list(roi.size),
                "roi_prepared_size": list(roi_image.size),
                "requested_crop_box_1000": list(requested_crop),
                "effective_crop_box_1000": list(effective_crop),
                "pixel_crop_box": list(pixel_crop),
                **token_counts,
                "image_token_placeholders": placeholders,
                "placeholder_count_matches_merged_tokens": placeholders
                == token_counts["llm_visual_tokens_total"],
                "input_sequence_tokens": int(inputs.input_ids.shape[1]),
                "coordinate_contract": "model outputs ROI-relative 0-1000; host remaps through effective_crop_box_1000",
                "prepared_image_artifacts": artifact_paths,
            }
        )

    passed = bool(records) and all(
        record["placeholder_count_matches_merged_tokens"]
        and len(record["image_grid_thw"]) == 2
        for record in records
    )
    report = {
        "status": "PROCESSOR_MULTI_IMAGE_CONTRACT_PASS" if passed else "PROCESSOR_MULTI_IMAGE_CONTRACT_FAIL",
        "scope": "CPU processor and coordinate-contract preflight only; no model forward, MRoPE, DeepStack, accuracy or latency claim",
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "global_long_edge": args.global_long_edge,
        "roi_long_edge": args.roi_long_edge,
        "oracle_margin": args.oracle_margin,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 全局图加ROI多图接口预检",
        "",
        f"状态：`{report['status']}`。",
        "",
        "本报告只验证CPU处理器、双图占位符数量和精确坐标映射契约；尚未验证模型前向、MRoPE、DeepStack、精度或延迟。ROI来自真值框，仅用于接口诊断。",
        "",
        "| 样本 | 任务 | 原图尺寸 | 全局/ROI准备尺寸 | 两图grid | Encoder Patch | LLM视觉Token | 占位符一致 |",
        "|---|---|---:|---:|---|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {record['sample_id']} | {record['task']} | "
            f"{record['source_size'][0]}x{record['source_size'][1]} | "
            f"{record['global_prepared_size'][0]}x{record['global_prepared_size'][1]} / "
            f"{record['roi_prepared_size'][0]}x{record['roi_prepared_size'][1]} | "
            f"`{record['image_grid_thw']}` | {record['encoder_tokens_total']} | "
            f"{record['llm_visual_tokens_total']} | "
            f"{'是' if record['placeholder_count_matches_merged_tokens'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "坐标规则：模型只输出相对第二张ROI的0-1000坐标，主机按实际像素裁剪边界对应的 `effective_crop_box_1000` 映射回原图；不使用理想浮点框代替实际裁剪框。",
            "",
        ]
    )
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(records)}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
