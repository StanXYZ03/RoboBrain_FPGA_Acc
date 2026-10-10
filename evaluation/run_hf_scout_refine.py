#!/usr/bin/env python3
"""Evaluate a no-ground-truth, two-pass structured crop selector.

The first pass (scout) is loaded from a completed uniform low-resolution run.
Its predicted point/box selects a regular crop for a second high-detail pass.
The final coordinates are mapped back to the original image.  Total token,
compute-proxy and timing fields include both passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import (
    CudaCallTimer,
    make_messages,
    processor_inputs,
    resolve_image,
)
from evaluation.token_reduction import (
    crop_image_1000,
    qwen3_vl_compute_proxy,
    remap_output_text,
    retention_label,
    scout_crop_box_1000,
)


PROFILES = {
    "tight": {"grounding_margin": 0.5, "point_half_extent": 250.0},
    "wide": {"grounding_margin": 1.0, "point_half_extent": 375.0},
}


def unique_by_id(path: Path) -> tuple[dict[str, dict[str, Any]], float]:
    records = load_jsonl(path)
    by_id = {record["sample_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path}: duplicate sample IDs")
    if not records:
        raise ValueError(f"{path}: empty scout prediction file")
    if any(record.get("strategy") != "uniform" for record in records):
        raise ValueError(f"{path}: scout inputs must come from uniform runs")
    retentions = {float(record["requested_retention"]) for record in records}
    if len(retentions) != 1:
        raise ValueError(f"{path}: mixed scout retentions")
    return by_id, retentions.pop()


def completed_ids(path: Path) -> set[str]:
    return {record["sample_id"] for record in load_jsonl(path)} if path.is_file() else set()


def limit_each_task(samples: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    if count is None:
        return samples
    selected: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for sample in samples:
        task = str(sample["task"])
        if seen.get(task, 0) < count:
            selected.append(sample)
            seen[task] = seen.get(task, 0) + 1
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scout-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=("tight", "wide"))
    parser.add_argument("--refine-retentions", type=float, nargs="+", default=(0.125, 0.25))
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")
    if args.limit_per_task is not None and args.limit_per_task <= 0:
        raise ValueError("--limit-per-task must be positive")
    for retention in args.refine_retentions:
        retention_label(retention)

    samples = limit_each_task(load_jsonl(args.manifest), args.limit_per_task)
    scout_sets: list[tuple[Path, dict[str, dict[str, Any]], float]] = []
    sample_ids = {sample["sample_id"] for sample in samples}
    for path in args.scout_predictions:
        by_id, retention = unique_by_id(path)
        missing = sample_ids - by_id.keys()
        if missing:
            raise ValueError(f"{path}: missing scout output for {next(iter(missing))}")
        scout_sets.append((path, by_id, retention))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[int, float, str], tuple[Path, set[str]]] = {}
    for scout_index, (_, _, scout_retention) in enumerate(scout_sets):
        for refine_retention in args.refine_retentions:
            for profile in args.profiles:
                filename = (
                    f"scout_{retention_label(scout_retention)}_"
                    f"refine_{retention_label(refine_retention)}_{profile}.jsonl"
                )
                path = args.output_dir / filename
                outputs[(scout_index, refine_retention, profile)] = path, completed_ids(path)

    pending_total = sum(
        sum(sample["sample_id"] not in done for sample in samples)
        for _, done in outputs.values()
    )
    print(
        f"LOAD model={args.model} samples={len(samples)} configs={len(outputs)} pending={pending_total}",
        flush=True,
    )
    if pending_total == 0:
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
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)

    warmup_sample = samples[0]
    warmup_path = resolve_image(args.manifest, warmup_sample["image"])
    warmup_messages = make_messages(warmup_path, warmup_sample["prompt"])
    with Image.open(warmup_path) as opened:
        warmup_image = opened.convert("RGB")
    warmup_inputs = processor_inputs(processor, warmup_messages, warmup_image).to(args.device)
    with torch.inference_mode():
        warmup_generated = model.generate(**warmup_inputs, do_sample=False, max_new_tokens=1)
    torch.cuda.synchronize()
    print("WARMUP result=PASS multimodal_prefill=1 decode_tokens=1", flush=True)
    del warmup_inputs, warmup_generated, warmup_image
    gc.collect()

    vision_timer = CudaCallTimer(model.model.visual)
    language_timer = CudaCallTimer(model.model.language_model)
    patch_size = int(model.config.vision_config.patch_size)
    merge_size = int(model.config.vision_config.spatial_merge_size)
    min_pixels = (patch_size * merge_size) ** 2
    baseline_cache: dict[str, tuple[list[int], int]] = {}
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "deterministic_decode": True,
        "selector": "no-ground-truth two-pass scout-refine structured crop",
    }

    try:
        for config_number, ((scout_index, refine_retention, profile), (output_path, done)) in enumerate(
            outputs.items(), 1
        ):
            scout_path, scout_by_id, scout_retention = scout_sets[scout_index]
            profile_values = PROFILES[profile]
            pending = [sample for sample in samples if sample["sample_id"] not in done]
            print(
                f"CONFIG {config_number}/{len(outputs)} scout={scout_retention:.3f} "
                f"refine={refine_retention:.3f} profile={profile} pending={len(pending)}",
                flush=True,
            )
            if not pending:
                continue
            with output_path.open("a", encoding="utf-8", buffering=1) as output:
                for sample_number, sample in enumerate(pending, 1):
                    scout = scout_by_id[sample["sample_id"]]
                    image_path = resolve_image(args.manifest, sample["image"])
                    messages = make_messages(image_path, sample["prompt"])
                    with Image.open(image_path) as opened:
                        original = opened.convert("RGB")

                    cache_key = str(image_path)
                    if cache_key not in baseline_cache:
                        base_inputs = processor_inputs(processor, messages, original)
                        base_grid = [int(value) for value in base_inputs.image_grid_thw[0].tolist()]
                        baseline_pixels = base_grid[1] * base_grid[2] * patch_size * patch_size
                        baseline_cache[cache_key] = base_grid, baseline_pixels
                        del base_inputs
                    base_grid, baseline_pixels = baseline_cache[cache_key]
                    base_tokens = base_grid[0] * base_grid[1] * base_grid[2]
                    base_proxy = qwen3_vl_compute_proxy(base_tokens, merge_size=merge_size)

                    crop_box = scout_crop_box_1000(
                        sample["task"],
                        str(scout["output_text"]),
                        grounding_margin=float(profile_values["grounding_margin"]),
                        point_half_extent=float(profile_values["point_half_extent"]),
                    )
                    refine_image = crop_image_1000(original, crop_box) if crop_box else original
                    target_pixels = max(min_pixels, int(round(baseline_pixels * refine_retention)))

                    preprocess_started = time.perf_counter()
                    inputs = processor_inputs(
                        processor,
                        messages,
                        refine_image,
                        min_pixels=target_pixels,
                        max_pixels=target_pixels,
                    ).to(args.device)
                    refine_preprocess_seconds = time.perf_counter() - preprocess_started
                    grid = [int(value) for value in inputs.image_grid_thw[0].tolist()]
                    refine_encoder_tokens = grid[0] * grid[1] * grid[2]
                    refine_proxy = qwen3_vl_compute_proxy(refine_encoder_tokens, merge_size=merge_size)

                    vision_timer.reset()
                    language_timer.reset()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            do_sample=False,
                            max_new_tokens=int(sample["max_new_tokens"]),
                        )
                    torch.cuda.synchronize()
                    refine_elapsed = time.perf_counter() - started
                    vision_ms = vision_timer.milliseconds()
                    language_ms = language_timer.milliseconds()
                    trimmed = generated[:, inputs.input_ids.shape[1] :]
                    raw_text = processor.batch_decode(
                        trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    output_text = (
                        remap_output_text(sample["task"], raw_text, crop_box)
                        if crop_box is not None
                        else raw_text
                    )

                    scout_encoder_tokens = int(scout["encoder_tokens"])
                    total_encoder_tokens = scout_encoder_tokens + refine_encoder_tokens
                    scout_encoder_macs = int(scout["encoder_macs_proxy"])
                    scout_llm_macs = int(scout["llm_prefill_macs_proxy"])
                    total_encoder_macs = scout_encoder_macs + refine_proxy.encoder_macs
                    total_llm_macs = scout_llm_macs + refine_proxy.llm_prefill_macs
                    scout_elapsed = float(scout["elapsed_seconds"])
                    scout_preprocess = float(scout.get("preprocess_seconds", 0.0))
                    scout_vision = float(scout["vision_encoder_seconds"])
                    scout_prefill = float(scout["llm_prefill_seconds"])
                    scout_decoder = float(scout["decoder_seconds"])
                    refine_vision = sum(vision_ms) / 1000.0
                    refine_prefill = (language_ms[0] / 1000.0) if language_ms else 0.0
                    refine_decoder = (sum(language_ms[1:]) / 1000.0) if len(language_ms) > 1 else 0.0

                    record: dict[str, Any] = {
                        "sample_id": sample["sample_id"],
                        "task": sample["task"],
                        "output_text": output_text,
                        "raw_output_text": raw_text,
                        "scout_output_text": scout["output_text"],
                        "elapsed_seconds": scout_elapsed + refine_elapsed,
                        "pipeline_seconds": scout_elapsed
                        + scout_preprocess
                        + refine_elapsed
                        + refine_preprocess_seconds,
                        "scout_elapsed_seconds": scout_elapsed,
                        "refine_elapsed_seconds": refine_elapsed,
                        "preprocess_seconds": scout_preprocess + refine_preprocess_seconds,
                        "vision_encoder_seconds": scout_vision + refine_vision,
                        "llm_prefill_seconds": scout_prefill + refine_prefill,
                        "decoder_seconds": scout_decoder + refine_decoder,
                        "refine_image_grid_thw": grid,
                        "baseline_image_grid_thw": base_grid,
                        "scout_encoder_tokens": scout_encoder_tokens,
                        "refine_encoder_tokens": refine_encoder_tokens,
                        "encoder_tokens": total_encoder_tokens,
                        "baseline_encoder_tokens": base_tokens,
                        "actual_token_retention": total_encoder_tokens / base_tokens,
                        "encoder_macs_proxy": total_encoder_macs,
                        "baseline_encoder_macs_proxy": base_proxy.encoder_macs,
                        "llm_prefill_macs_proxy": total_llm_macs,
                        "baseline_llm_prefill_macs_proxy": base_proxy.llm_prefill_macs,
                        "scout_requested_retention": scout_retention,
                        "refine_requested_retention": refine_retention,
                        "profile": profile,
                        "scout_crop_box_1000": list(crop_box) if crop_box is not None else None,
                        "selector_parse_success": crop_box is not None,
                        "grounding_margin": profile_values["grounding_margin"],
                        "point_half_extent": profile_values["point_half_extent"],
                        "target_pixels": target_pixels,
                        "peak_cuda_bytes": max(
                            int(scout["peak_cuda_bytes"]), int(torch.cuda.max_memory_allocated())
                        ),
                        "scout_file": str(scout_path),
                        "run": metadata,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        "RESULT "
                        f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                        f"total_retention={record['actual_token_retention']:.3f} "
                        f"crop={'yes' if crop_box else 'fallback'} total={record['elapsed_seconds']:.4f}s "
                        f"output={raw_text!r}",
                        flush=True,
                    )
                    del inputs, generated, trimmed, refine_image, original
                    gc.collect()
    finally:
        vision_timer.close()
        language_timer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
