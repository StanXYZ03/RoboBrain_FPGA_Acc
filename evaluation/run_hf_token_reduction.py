#!/usr/bin/env python3
"""Run real-token reduction scans on a local Qwen3-VL/RoboBrain checkpoint."""

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
from evaluation.token_reduction import (
    crop_image_1000,
    expanded_bbox_1000,
    qwen3_vl_compute_proxy,
    remap_output_text,
    retention_label,
)


class CudaCallTimer:
    def __init__(self, module: torch.nn.Module) -> None:
        self.calls: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None
        self._pre = module.register_forward_pre_hook(self._before)
        self._post = module.register_forward_hook(self._after)

    def _before(self, _module: torch.nn.Module, _inputs: Any) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def _after(self, _module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        if self._start is None:
            raise RuntimeError("timer post-hook observed without pre-hook")
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.calls.append((self._start, end))
        self._start = None

    def reset(self) -> None:
        self.calls.clear()
        self._start = None

    def milliseconds(self) -> list[float]:
        return [float(start.elapsed_time(end)) for start, end in self.calls]

    def close(self) -> None:
        self._pre.remove()
        self._post.remove()


def resolve_image(manifest: Path, image: str) -> Path:
    path = Path(image)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def completed_ids(path: Path) -> set[str]:
    return {record["sample_id"] for record in load_jsonl(path)} if path.is_file() else set()


def make_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def processor_inputs(
    processor: Any,
    messages: list[dict[str, Any]],
    image: Image.Image,
    *,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
) -> Any:
    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs: dict[str, Any] = {}
    if min_pixels is not None and max_pixels is not None:
        kwargs["images_kwargs"] = {"min_pixels": min_pixels, "max_pixels": max_pixels}
    return processor(
        text=[chat_text],
        images=[image],
        padding=True,
        return_tensors="pt",
        **kwargs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25, 0.125))
    parser.add_argument("--strategies", nargs="+", choices=("uniform", "oracle_crop"), default=("uniform", "oracle_crop"))
    parser.add_argument("--oracle-margin", type=float, default=0.5)
    parser.add_argument("--task", choices=("pointing", "grounding"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this timing experiment requires a CUDA device")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    samples = load_jsonl(args.manifest)
    if args.task:
        samples = [sample for sample in samples if sample["task"] == args.task]
    if args.limit is not None:
        samples = samples[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[tuple[str, float], tuple[Path, set[str]]] = {}
    for strategy in args.strategies:
        for retention in args.retentions:
            label = retention_label(retention)
            path = args.output_dir / f"{strategy}_{label}.jsonl"
            outputs[(strategy, retention)] = path, completed_ids(path)
    pending_total = sum(
        sum(sample["sample_id"] not in done for sample in samples)
        for _, done in outputs.values()
    )
    print(f"LOAD model={args.model} samples={len(samples)} pending={pending_total}", flush=True)
    if pending_total == 0:
        return 0

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
    # Pay one-time CUDA context, allocator and kernel-initialization costs
    # before any configuration is measured. The warm-up is intentionally a
    # real multimodal prefill so the first r1000 sample is comparable with the
    # later retention points.
    warmup_sample = samples[0]
    warmup_path = resolve_image(args.manifest, warmup_sample["image"])
    warmup_messages = make_messages(warmup_path, warmup_sample["prompt"])
    with Image.open(warmup_path) as opened:
        warmup_image = opened.convert("RGB")
    warmup_inputs = processor_inputs(processor, warmup_messages, warmup_image).to(args.device)
    with torch.inference_mode():
        warmup_generated = model.generate(
            **warmup_inputs,
            do_sample=False,
            max_new_tokens=1,
        )
    torch.cuda.synchronize()
    print("WARMUP result=PASS multimodal_prefill=1 decode_tokens=1", flush=True)
    del warmup_inputs, warmup_generated, warmup_image
    gc.collect()
    visual = model.model.visual
    language_model = model.model.language_model
    vision_timer = CudaCallTimer(visual)
    language_timer = CudaCallTimer(language_model)
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
        "compute_proxy_scope": "dominant vision/text block MACs; excludes patch embed, merger, DeepStack projections and text tokens",
    }

    try:
        for config_index, ((strategy, retention), (output_path, done)) in enumerate(outputs.items(), 1):
            pending = [sample for sample in samples if sample["sample_id"] not in done]
            print(
                f"CONFIG {config_index}/{len(outputs)} strategy={strategy} retention={retention} pending={len(pending)}",
                flush=True,
            )
            if not pending:
                continue
            with output_path.open("a", encoding="utf-8", buffering=1) as output:
                for sample_index, sample in enumerate(pending, 1):
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
                    target_pixels = max(min_pixels, int(round(baseline_pixels * retention)))
                    crop_box = None
                    experiment_image = original
                    if strategy == "oracle_crop":
                        crop_box = expanded_bbox_1000(
                            sample["ground_truth"]["bbox_1000"], args.oracle_margin
                        )
                        experiment_image = crop_image_1000(original, crop_box)

                    preprocess_started = time.perf_counter()
                    inputs = processor_inputs(
                        processor,
                        messages,
                        experiment_image,
                        # Equal lower/upper budgets make a small oracle crop
                        # consume approximately the same token budget as the
                        # corresponding uniform configuration. Smart-resize
                        # may still differ by one grid row/column because it
                        # must preserve aspect ratio and merge alignment.
                        min_pixels=target_pixels,
                        max_pixels=target_pixels,
                    ).to(args.device)
                    preprocess_seconds = time.perf_counter() - preprocess_started
                    grid = [int(value) for value in inputs.image_grid_thw[0].tolist()]
                    encoder_tokens = grid[0] * grid[1] * grid[2]
                    base_tokens = base_grid[0] * base_grid[1] * base_grid[2]
                    proxy = qwen3_vl_compute_proxy(encoder_tokens, merge_size=merge_size)
                    base_proxy = qwen3_vl_compute_proxy(base_tokens, merge_size=merge_size)

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
                    elapsed = time.perf_counter() - started
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
                    record: dict[str, Any] = {
                        "sample_id": sample["sample_id"],
                        "task": sample["task"],
                        "output_text": output_text,
                        "raw_output_text": raw_text,
                        "elapsed_seconds": elapsed,
                        "preprocess_seconds": preprocess_seconds,
                        "vision_encoder_seconds": sum(vision_ms) / 1000.0,
                        "llm_prefill_seconds": (language_ms[0] / 1000.0) if language_ms else None,
                        "decoder_seconds": (sum(language_ms[1:]) / 1000.0) if len(language_ms) > 1 else 0.0,
                        "language_forward_calls": len(language_ms),
                        "input_tokens": int(inputs.input_ids.shape[1]),
                        "output_tokens": int(trimmed.shape[1]),
                        "image_grid_thw": grid,
                        "baseline_image_grid_thw": base_grid,
                        "encoder_tokens": encoder_tokens,
                        "baseline_encoder_tokens": base_tokens,
                        "llm_visual_tokens": proxy.llm_visual_tokens,
                        "actual_token_retention": encoder_tokens / base_tokens,
                        "encoder_macs_proxy": proxy.encoder_macs,
                        "baseline_encoder_macs_proxy": base_proxy.encoder_macs,
                        "llm_prefill_macs_proxy": proxy.llm_prefill_macs,
                        "baseline_llm_prefill_macs_proxy": base_proxy.llm_prefill_macs,
                        "target_pixels": target_pixels,
                        "strategy": strategy,
                        "requested_retention": retention,
                        "oracle_crop_box_1000": list(crop_box) if crop_box is not None else None,
                        "oracle_margin": args.oracle_margin if crop_box is not None else None,
                        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
                        "run": metadata,
                    }
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        "RESULT "
                        f"{sample_index}/{len(pending)} id={sample['sample_id']} "
                        f"strategy={strategy} requested={retention:.3f} actual={record['actual_token_retention']:.3f} "
                        f"encoder_tokens={encoder_tokens} vision={record['vision_encoder_seconds']:.4f}s "
                        f"prefill={record['llm_prefill_seconds']:.4f}s total={elapsed:.4f}s output={raw_text!r}",
                        flush=True,
                    )
                    del inputs, generated, trimmed, experiment_image, original
                    gc.collect()
    finally:
        vision_timer.close()
        language_timer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
