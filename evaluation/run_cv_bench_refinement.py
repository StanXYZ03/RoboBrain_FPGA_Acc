#!/usr/bin/env python3
"""Evaluate a frozen training-free scout and global-plus-crop representation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.evaluate_task_conditioned_roi_proposals import heatmap_center_1000, shifted_box
from evaluation.extract_roi_vision_prefix_features import fixed_projection
from evaluation.extract_task_conditioned_roi_scout import spatial_scout_features
from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import CudaCallTimer, make_messages, processor_inputs, resolve_image
from evaluation.visual_refinement import crop_image_exact_1000, grid_token_counts, resize_long_edge

STRATEGIES = ("dense", "global256", "cosine_global_roi")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def detail_messages(prompt: str, crop: list[float]) -> list[dict]:
    bounds = ", ".join(f"{value:.2f}" for value in crop)
    return [{"role": "user", "content": [
        {"type": "image", "image": "global"},
        {"type": "image", "image": "detail"},
        {"type": "text", "text": (
            "The first image is the complete scene. The second image is a close-up "
            "of the same scene, not another scene. Its crop bounds (x1, y1, x2, y2) "
            f"in the complete image on a 0-1000 coordinate scale are ({bounds}). "
            "Answer the following question about the complete scene:\n" + prompt
        )},
    ]}]


def rotated_strategies(values: list[str], sample_index: int, repeat: int) -> list[str]:
    offset = (sample_index + repeat) % len(values)
    return values[offset:] + values[:offset]


def validate_resume(rows: list[dict], signature: dict, expected: set[tuple]) -> set[tuple]:
    seen = set()
    for row in rows:
        key = (str(row["sample_id"]), row["strategy"], row["repeat"])
        if key in seen or key not in expected or row["run"] != signature:
            raise ValueError("duplicate, foreign or incompatible resume record")
        seen.add(key)
    return seen


def run_one(model, processor, sample: dict, manifest: Path, strategy: str,
            projection: torch.Tensor, visual_timer: CudaCallTimer,
            language_timer: CudaCallTimer) -> dict:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    stage = time.perf_counter()
    path = resolve_image(manifest, sample["image"])
    with Image.open(path) as opened:
        original = opened.convert("RGB")
    image_ms = (time.perf_counter() - stage) * 1000
    stage = time.perf_counter()
    global_image = original if strategy == "dense" else resize_long_edge(original, 256, allow_upscale=False)
    resize_ms = (time.perf_counter() - stage) * 1000
    scout_processor_ms = scout_transfer_ms = scout_ms = policy_ms = 0.0
    scout_tokens = 0
    crop = None
    if strategy == "cosine_global_roi":
        stage = time.perf_counter()
        scout = processor_inputs(processor, make_messages(path, sample["prompt"]),
                                 global_image, min_pixels=1024,
                                 max_pixels=processor.image_processor.size["longest_edge"])
        scout_processor_ms = (time.perf_counter() - stage) * 1000
        torch.cuda.synchronize()
        stage = time.perf_counter()
        scout = scout.to("cuda:0")
        torch.cuda.synchronize()
        scout_transfer_ms = (time.perf_counter() - stage) * 1000
        stage = time.perf_counter()
        mask = (scout.attention_mask[0] != 0) & (scout.input_ids[0] != model.config.image_token_id)
        with torch.inference_mode():
            embeddings = model.get_input_embeddings()(scout.input_ids[0, mask])
            compact, relevance, grid_h, grid_w = spatial_scout_features(
                model.model.visual, scout.pixel_values, scout.image_grid_thw,
                embeddings, completed_blocks=6, projection=projection)
        torch.cuda.synchronize()
        scout_ms = (time.perf_counter() - stage) * 1000
        scout_tokens = grid_h * grid_w
        stage = time.perf_counter()
        center = heatmap_center_1000(np.asarray(relevance, dtype=np.float64).reshape(grid_h, grid_w))
        proposal = shifted_box(*center, 600.0, 600.0)
        roi, effective, pixel_crop = crop_image_exact_1000(original, proposal)
        crop = list(effective)
        roi_image = resize_long_edge(roi, 384, allow_upscale=True)
        policy_ms = (time.perf_counter() - stage) * 1000
        images, messages = [global_image, roi_image], detail_messages(sample["prompt"], crop)
        del scout, embeddings, compact, relevance
    else:
        images = [original if strategy == "dense" else global_image]
        messages = make_messages(path, sample["prompt"])
    stage = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {} if strategy == "dense" else {"images_kwargs": {
        "min_pixels": 1024, "max_pixels": processor.image_processor.size["longest_edge"]}}
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True, **kwargs)
    processor_ms = (time.perf_counter() - stage) * 1000
    counts = grid_token_counts(inputs.image_grid_thw.tolist(), merge_size=2)
    assert int((inputs.input_ids == model.config.image_token_id).sum()) == counts["llm_visual_tokens_total"]
    torch.cuda.synchronize()
    stage = time.perf_counter()
    inputs = inputs.to("cuda:0")
    torch.cuda.synchronize()
    transfer_ms = (time.perf_counter() - stage) * 1000
    visual_timer.reset()
    language_timer.reset()
    stage = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=sample["max_new_tokens"])
    torch.cuda.synchronize()
    generate_ms = (time.perf_counter() - stage) * 1000
    vision = visual_timer.milliseconds()
    language = language_timer.milliseconds()
    stage = time.perf_counter()
    trimmed = output[:, inputs.input_ids.shape[1]:]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    post_ms = (time.perf_counter() - stage) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "sample_id": str(sample["sample_id"]), "strategy": strategy,
        "output_text": answer, "output_tokens": int(trimmed.shape[1]),
        "image_grid_thw": inputs.image_grid_thw.tolist(),
        "final_visual_tokens": counts["llm_visual_tokens_total"],
        "encoder_tokens_per_image": counts["encoder_tokens_per_image"],
        "scout_visual_tokens": scout_tokens, "effective_crop_1000": crop,
        "image_decode_ms": image_ms, "resize_ms": resize_ms,
        "scout_processor_ms": scout_processor_ms, "scout_h2d_ms": scout_transfer_ms,
        "scout_forward_and_feature_d2h_ms": scout_ms,
        "policy_crop_resize_ms": policy_ms, "processor_ms": processor_ms,
        "h2d_ms": transfer_ms, "generate_ms": generate_ms,
        "vision_ms": sum(vision), "prefill_ms": language[0] if language else None,
        "decode_ms": sum(language[1:]), "postprocess_ms": post_ms,
        "request_end_to_end_ms": total_ms,
        "model_end_to_end_ms": total_ms - image_ms - post_ms,
        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=["global256", "cosine_global_roi"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1 or len(set(args.strategies)) != len(args.strategies):
        parser.error("positive repeats/warmup and unique strategies required")
    samples = load_jsonl(args.manifest)
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise ValueError("duplicate manifest ID")
    signature = {
        "model": str(args.model.resolve()), "manifest_sha256": sha256(args.manifest),
        "runner_sha256": sha256(Path(__file__)),
        "scout_sha256": sha256(Path("evaluation/extract_task_conditioned_roi_scout.py")),
        "policy_sha256": sha256(Path("evaluation/evaluate_task_conditioned_roi_proposals.py")),
        "strategies": args.strategies, "repeats": args.repeats, "warmup": args.warmup,
        "global_long_edge": 256, "roi_long_edge": 384, "box_extent_1000": 600,
        "completed_blocks": 6, "dtype": "bf16", "attention": "sdpa",
        "ground_truth_at_inference": False, "adaptive_escalation": False,
        "learned_ridge_used": False, "seed": 0,
    }
    expected = {(str(row["sample_id"]), strategy, repeat) for row in samples
                for strategy in args.strategies for repeat in range(args.repeats)}
    seen = validate_resume(load_jsonl(args.output) if args.output.exists() else [], signature, expected)
    print(f"STAGE samples={len(samples)} pending={len(expected-seen)}; frozen full sweep, no early quality gate", flush=True)
    if seen == expected:
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), dtype=torch.bfloat16,
        device_map="cuda:0", attn_implementation="sdpa", low_cpu_mem_usage=True,
        local_files_only=True).eval()
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    projection = fixed_projection(model.config.text_config.hidden_size, 8, 20260915)
    timers = CudaCallTimer(model.model.visual), CudaCallTimer(model.model.language_model)
    try:
        for strategy in args.strategies:
            for index in range(args.warmup):
                run_one(model, processor, samples[index % len(samples)], args.manifest,
                        strategy, projection, *timers)
            print(f"WARMUP strategy={strategy} runs={args.warmup} done", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8", buffering=1) as stream:
            completed = 0
            for index, sample in enumerate(samples):
                for repeat in range(args.repeats):
                    for strategy in rotated_strategies(args.strategies, index, repeat):
                        key = (str(sample["sample_id"]), strategy, repeat)
                        if key in seen:
                            continue
                        record = run_one(model, processor, sample, args.manifest,
                                         strategy, projection, *timers)
                        record.update(repeat=repeat, run=signature)
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        seen.add(key)
                        completed += 1
                        if completed % 100 == 0:
                            print(f"PROGRESS records={len(seen)}/{len(expected)} last={sample['sample_id']} strategy={strategy}", flush=True)
                gc.collect()
    finally:
        for timer in timers:
            timer.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
