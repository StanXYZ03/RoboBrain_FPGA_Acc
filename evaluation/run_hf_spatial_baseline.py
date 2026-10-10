#!/usr/bin/env python3
"""Run deterministic Hugging Face BF16 spatial inference with resumable JSONL output."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from evaluation.metrics import load_jsonl


def git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_image(manifest: Path, image: str) -> Path:
    path = Path(image)
    return path if path.is_absolute() else (manifest.parent / path).resolve()


def completed_ids(path: Path) -> set[str]:
    return {record["sample_id"] for record in load_jsonl(path)} if path.is_file() else set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task", choices=("pointing", "grounding"))
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    samples = load_jsonl(args.manifest)
    if args.task:
        samples = [sample for sample in samples if sample["task"] == args.task]
    if args.limit is not None:
        samples = samples[: args.limit]
    done = completed_ids(args.output)
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(f"LOAD model={args.model} label={args.model_label} pending={len(pending)} done={len(done)}", flush=True)
    if not pending:
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_git_revision": git_revision(args.model),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "deterministic_decode": True,
    }
    with args.output.open("a", encoding="utf-8", buffering=1) as output:
        for index, sample in enumerate(pending, 1):
            image_path = resolve_image(args.manifest, sample["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": sample["prompt"]},
                    ],
                }
            ]
            chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(args.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=int(sample["max_new_tokens"]),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            trimmed = generated[:, inputs.input_ids.shape[1] :]
            text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            record: dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "task": sample["task"],
                "output_text": text,
                "elapsed_seconds": elapsed,
                "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(trimmed.shape[1]),
                "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "run": metadata,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(pending)} id={sample['sample_id']} elapsed={elapsed:.3f}s output={text!r}",
                flush=True,
            )
            del inputs, generated, trimmed
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
