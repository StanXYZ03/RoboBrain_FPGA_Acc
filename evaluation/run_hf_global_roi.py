#!/usr/bin/env python3
"""Run a revealed HF pilot for dense, global, ROI and global-plus-ROI inputs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.build_visual_refinement_splits import proportional_sample
from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import CudaCallTimer, make_messages, resolve_image
from evaluation.token_reduction import expanded_bbox_1000, remap_output_text
from evaluation.visual_refinement import (
    crop_image_exact_1000,
    grid_token_counts,
    make_global_roi_messages,
    make_labeled_global_roi_messages,
    make_roi_only_messages,
    resize_long_edge,
)


STRATEGIES = (
    "dense",
    "global_only",
    "roi_only",
    "global_roi",
    "global_roi_v2",
    "roi_global_v2",
)


def choose_samples(
    records: list[dict[str, Any]], limit_per_task: int, seed: int
) -> list[dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_image.setdefault(int(record["coco_image_id"]), []).append(record)
    if limit_per_task > len(by_image):
        raise ValueError(f"requested {limit_per_task} images from {len(by_image)}")
    candidates: list[dict[str, Any]] = []
    for image_id, image_records in by_image.items():
        tasks = {str(record["task"]) for record in image_records}
        if tasks != {"grounding", "pointing"}:
            raise ValueError(f"image {image_id} does not have a paired task set: {sorted(tasks)}")
        stratification = image_records[0]["stratification"]
        candidates.append(
            {
                "image_id": image_id,
                "area_bin": str(stratification["target_area_bin"]),
                "edge_bin": str(stratification["edge_bin"]),
                "overlap_bin": str(stratification["overlap_bin"]),
                "complexity_bin": str(stratification["scene_complexity_bin"]),
            }
        )
    selected_ids = {
        int(item["image_id"])
        for item in proportional_sample(candidates, limit_per_task, seed)
    }
    chosen = [record for record in records if int(record["coco_image_id"]) in selected_ids]
    task_counts = {
        task: sum(str(record["task"]) == task for record in chosen)
        for task in ("grounding", "pointing")
    }
    if task_counts != {"grounding": limit_per_task, "pointing": limit_per_task}:
        raise RuntimeError(f"stratified task counts are inconsistent: {task_counts}")
    return chosen


def original_image_path(manifest: Path, sample: dict[str, Any], original_dir: Path) -> Path:
    image_id = sample.get("coco_image_id")
    if image_id is not None:
        candidate = original_dir / f"{int(image_id):012d}.jpg"
        if candidate.is_file():
            return candidate
    return resolve_image(manifest, str(sample["image"]))


def tensor_bytes(inputs: Any) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for value in inputs.values()
        if isinstance(value, torch.Tensor)
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))]


def prepare_strategy(
    sample: dict[str, Any],
    image_path: Path,
    original: Image.Image,
    strategy: str,
    *,
    global_long_edge: int,
    roi_long_edge: int,
    oracle_margin: float,
    roi_upscale: bool,
) -> dict[str, Any]:
    prepared: dict[str, Any] = {
        "requested_crop": None,
        "effective_crop": None,
        "pixel_crop": None,
        "roi": None,
        "global_image": None,
        "roi_image": None,
        "crop_for_remap": None,
    }
    if strategy == "dense":
        prepared.update(
            images=[original],
            messages=make_messages(image_path, str(sample["prompt"])),
        )
        return prepared
    if strategy == "global_only":
        global_image = resize_long_edge(original, global_long_edge)
        prepared.update(
            global_image=global_image,
            images=[global_image],
            messages=make_messages(image_path, str(sample["prompt"])),
        )
        return prepared

    requested_crop = expanded_bbox_1000(sample["ground_truth"]["bbox_1000"], oracle_margin)
    roi, effective_crop, pixel_crop = crop_image_exact_1000(original, requested_crop)
    roi_image = resize_long_edge(roi, roi_long_edge, allow_upscale=roi_upscale)
    prepared.update(
        requested_crop=requested_crop,
        effective_crop=effective_crop,
        pixel_crop=pixel_crop,
        roi=roi,
        roi_image=roi_image,
        crop_for_remap=effective_crop,
    )
    if strategy == "roi_only":
        prepared.update(
            images=[roi_image],
            messages=make_roi_only_messages(str(sample["prompt"])),
        )
        return prepared

    global_image = resize_long_edge(original, global_long_edge)
    prepared["global_image"] = global_image
    if strategy == "global_roi":
        prepared.update(
            images=[global_image, roi_image],
            messages=make_global_roi_messages(str(sample["prompt"])),
        )
    elif strategy == "global_roi_v2":
        prepared.update(
            images=[global_image, roi_image],
            messages=make_labeled_global_roi_messages(str(sample["prompt"]), roi_first=False),
        )
    else:
        prepared.update(
            images=[roi_image, global_image],
            messages=make_labeled_global_roi_messages(str(sample["prompt"]), roi_first=True),
        )
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-task", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=20260912)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--global-long-edge", type=int, default=256)
    parser.add_argument("--roi-long-edge", type=int, default=384)
    parser.add_argument(
        "--disable-roi-upscale",
        action="store_true",
        help="Keep small crops at native pixel dimensions (diagnostic only).",
    )
    parser.add_argument("--oracle-margin", type=float, default=0.5)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument(
        "--strategy-label",
        help="Override the stored strategy name when exactly one strategy is requested.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if args.limit_per_task <= 0 or args.repeats <= 0:
        parser.error("sample limit and repeats must be positive")
    if args.strategy_label and len(args.strategies) != 1:
        parser.error("--strategy-label requires exactly one --strategies value")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this pilot requires CUDA")
    samples = choose_samples(load_jsonl(args.manifest), args.limit_per_task, args.sample_seed)
    done: set[tuple[str, str, int]] = set()
    if args.output.is_file():
        done = {
            (str(item["sample_id"]), str(item["strategy"]), int(item["repeat"]))
            for item in load_jsonl(args.output)
        }
    pending = sum(
        (
            sample["sample_id"],
            args.strategy_label if args.strategy_label else strategy,
            repeat,
        )
        not in done
        for strategy in args.strategies
        for sample in samples
        for repeat in range(args.repeats)
    )
    print(f"LOAD samples={len(samples)} configs={len(args.strategies)} pending={pending}", flush=True)
    if pending == 0:
        return 0

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=dtype,
        device_map=args.device,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    visual_timer = CudaCallTimer(model.model.visual)
    language_timer = CudaCallTimer(model.model.language_model)
    patch_size = int(model.config.vision_config.patch_size)
    merge_size = int(model.config.vision_config.spatial_merge_size)
    minimum_pixels = (patch_size * merge_size) ** 2
    maximum_pixels = int(processor.image_processor.size["longest_edge"])
    run_metadata = {
        "status": "REVEALED_ORACLE_MECHANICS_PILOT_NOT_VALIDATION",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "deterministic_decode": True,
        "oracle_uses_ground_truth_bbox": True,
        "coordinate_contract": "ROI-relative output remapped through effective pixel crop",
        "sampling": "paired deterministic proportional stratification",
        "sample_seed": args.sample_seed,
    }

    # Warm every requested representation path before measured calls.
    warmup = samples[0]
    warmup_path = original_image_path(args.manifest, warmup, args.original_image_dir)
    with Image.open(warmup_path) as opened:
        warmup_original = opened.convert("RGB")
    for strategy in args.strategies:
        warmup_prepared = prepare_strategy(
            warmup,
            warmup_path,
            warmup_original,
            strategy,
            global_long_edge=args.global_long_edge,
            roi_long_edge=args.roi_long_edge,
            oracle_margin=args.oracle_margin,
            roi_upscale=not args.disable_roi_upscale,
        )
        warmup_text = processor.apply_chat_template(
            warmup_prepared["messages"],
            tokenize=False,
            add_generation_prompt=True,
            add_vision_id=strategy in {"global_roi_v2", "roi_global_v2"},
        )
        warmup_inputs = processor(
            text=[warmup_text],
            images=warmup_prepared["images"],
            padding=True,
            return_tensors="pt",
            images_kwargs={"min_pixels": minimum_pixels, "max_pixels": maximum_pixels},
        ).to(args.device)
        with torch.inference_mode():
            warmup_generated = model.generate(
                **warmup_inputs, do_sample=False, max_new_tokens=1
            )
        torch.cuda.synchronize()
        print(f"WARMUP strategy={strategy} decode_tokens=1 result=PASS", flush=True)
        del warmup_inputs, warmup_generated, warmup_prepared
    del warmup_original
    gc.collect()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("a", encoding="utf-8", buffering=1) as stream:
            for strategy in args.strategies:
                record_strategy = args.strategy_label if args.strategy_label else strategy
                print(
                    f"CONFIG strategy={strategy} record_strategy={record_strategy}", flush=True
                )
                for sample_index, sample in enumerate(samples, 1):
                    for repeat in range(args.repeats):
                        key = (sample["sample_id"], record_strategy, repeat)
                        if key in done:
                            continue
                        application_started = time.perf_counter()
                        image_path = original_image_path(args.manifest, sample, args.original_image_dir)
                        decode_started = time.perf_counter()
                        with Image.open(image_path) as opened:
                            original = opened.convert("RGB")
                        image_read_decode_ms = 1000.0 * (time.perf_counter() - decode_started)

                        prepare_started = time.perf_counter()
                        prepared = prepare_strategy(
                            sample,
                            image_path,
                            original,
                            strategy,
                            global_long_edge=args.global_long_edge,
                            roi_long_edge=args.roi_long_edge,
                            oracle_margin=args.oracle_margin,
                            roi_upscale=not args.disable_roi_upscale,
                        )
                        images = prepared["images"]
                        messages = prepared["messages"]
                        requested_crop = prepared["requested_crop"]
                        effective_crop = prepared["effective_crop"]
                        pixel_crop = prepared["pixel_crop"]
                        roi = prepared["roi"]
                        global_image = prepared["global_image"]
                        roi_image = prepared["roi_image"]
                        crop_for_remap = prepared["crop_for_remap"]
                        roi_fetch_resize_ms = 1000.0 * (time.perf_counter() - prepare_started)

                        processor_started = time.perf_counter()
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            add_vision_id=strategy in {"global_roi_v2", "roi_global_v2"},
                        )
                        inputs = processor(
                            text=[text], images=images, padding=True, return_tensors="pt",
                            images_kwargs={"min_pixels": minimum_pixels, "max_pixels": maximum_pixels},
                        )
                        final_preprocess_ms = 1000.0 * (time.perf_counter() - processor_started)
                        counts = grid_token_counts(inputs.image_grid_thw.tolist(), merge_size=merge_size)
                        placeholders = int((inputs.input_ids == int(processor.image_token_id)).sum().item())
                        if placeholders != counts["llm_visual_tokens_total"]:
                            raise RuntimeError("image placeholder count does not match merged visual tokens")
                        transfer_bytes = tensor_bytes(inputs)
                        torch.cuda.synchronize()
                        transfer_started = time.perf_counter()
                        inputs = inputs.to(args.device)
                        torch.cuda.synchronize()
                        host_to_device_ms = 1000.0 * (time.perf_counter() - transfer_started)

                        visual_timer.reset()
                        language_timer.reset()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        model_started = time.perf_counter()
                        with torch.inference_mode():
                            generated = model.generate(
                                **inputs,
                                do_sample=False,
                                max_new_tokens=int(sample["max_new_tokens"]),
                            )
                        torch.cuda.synchronize()
                        generate_ms = 1000.0 * (time.perf_counter() - model_started)
                        vision_calls = visual_timer.milliseconds()
                        language_calls = language_timer.milliseconds()
                        trimmed = generated[:, inputs.input_ids.shape[1] :]
                        post_started = time.perf_counter()
                        raw_text = processor.batch_decode(
                            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        output_text = (
                            remap_output_text(sample["task"], raw_text, crop_for_remap)
                            if crop_for_remap is not None
                            else raw_text
                        )
                        postprocess_ms = 1000.0 * (time.perf_counter() - post_started)
                        application_end_to_end_ms = 1000.0 * (
                            time.perf_counter() - application_started
                        )
                        record = {
                            "sample_id": sample["sample_id"],
                            "image_id": sample.get("coco_image_id"),
                            "task": sample["task"],
                            "stratification": sample.get("stratification"),
                            "strategy": record_strategy,
                            "base_strategy": strategy,
                            "model_revision": args.model_label,
                            "processor_revision": args.model_label,
                            "repeat": repeat,
                            "output_text": output_text,
                            "raw_output_text": raw_text,
                            "output_tokens": int(trimmed.shape[1]),
                            "source_image": str(image_path.resolve()),
                            "source_size": list(original.size),
                            "source_image_bytes": image_path.stat().st_size,
                            "roi_rgb_bytes": int(roi.width * roi.height * 3) if roi else 0,
                            "global_prepared_size": list(global_image.size) if global_image else None,
                            "roi_prepared_size": list(roi_image.size) if roi_image else None,
                            "roi_upscaled": bool(
                                roi is not None
                                and roi_image is not None
                                and (roi_image.width > roi.width or roi_image.height > roi.height)
                            ),
                            "requested_crop_box_1000": (
                                list(requested_crop) if requested_crop is not None else None
                            ),
                            "effective_crop_box_1000": (
                                list(effective_crop) if effective_crop is not None else None
                            ),
                            "pixel_crop_box": list(pixel_crop) if pixel_crop is not None else None,
                            **counts,
                            "global_encoder_tokens": (
                                counts["encoder_tokens_per_image"][-1]
                                if strategy == "roi_global_v2"
                                else counts["encoder_tokens_per_image"][0]
                                if strategy
                                in {"dense", "global_only", "global_roi", "global_roi_v2"}
                                else 0
                            ),
                            "roi_encoder_tokens": (
                                counts["encoder_tokens_per_image"][0]
                                if strategy in {"roi_only", "roi_global_v2"}
                                else counts["encoder_tokens_per_image"][-1]
                                if strategy in {"global_roi", "global_roi_v2"}
                                else 0
                            ),
                            "final_context_visual_tokens": counts["llm_visual_tokens_total"],
                            "cumulative_processed_visual_tokens": counts["llm_visual_tokens_total"],
                            "image_token_placeholders": placeholders,
                            "input_tokens": int(inputs.input_ids.shape[1]),
                            "image_read_decode_ms": image_read_decode_ms,
                            "scout_preprocess_ms": 0.0,
                            "scout_vision_ms": 0.0,
                            "scout_prefill_ms": 0.0,
                            "scout_decode_ms": 0.0,
                            "policy_ms": 0.0,
                            "roi_fetch_resize_ms": roi_fetch_resize_ms,
                            "final_preprocess_ms": final_preprocess_ms,
                            "host_to_device_ms": host_to_device_ms,
                            "final_vision_ms": sum(vision_calls),
                            "final_prefill_ms": language_calls[0] if language_calls else None,
                            "final_decode_ms": sum(language_calls[1:]) if len(language_calls) > 1 else 0.0,
                            "generate_ms": generate_ms,
                            "postprocess_ms": postprocess_ms,
                            "model_end_to_end_ms": (
                                roi_fetch_resize_ms
                                + final_preprocess_ms
                                + host_to_device_ms
                                + generate_ms
                                + postprocess_ms
                            ),
                            "application_end_to_end_ms": application_end_to_end_ms,
                            "host_to_device_bytes": transfer_bytes,
                            "cross_device_bytes": 0,
                            "estimated_weight_bytes_read": None,
                            "prediction_original_coordinates": output_text,
                            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                            "run": run_metadata,
                        }
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        print(
                            f"RESULT {sample_index}/{len(samples)} repeat={repeat + 1}/{args.repeats} "
                            f"id={sample['sample_id']} strategy={record_strategy} "
                            f"visual_tokens={counts['llm_visual_tokens_total']} "
                            f"total={application_end_to_end_ms:.1f}ms output={output_text!r}",
                            flush=True,
                        )
                        del inputs, generated, trimmed, images, prepared, roi_image, global_image, roi, original
                        gc.collect()
    finally:
        visual_timer.close()
        language_timer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
