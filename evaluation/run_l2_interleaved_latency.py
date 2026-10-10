#!/usr/bin/env python3
"""Interleave repeated latency measurements without changing the frozen selector."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.metrics import load_jsonl
from evaluation.run_hf_shallow_block_pruning import run_one
from evaluation.run_hf_token_reduction import CudaCallTimer


def configuration_order(count: int, sample_index: int, repeat_index: int) -> list[int]:
    if count < 1:
        raise ValueError("configuration count must be positive")
    offset = (sample_index + repeat_index) % count
    return [(offset + position) % count for position in range(count)]


def completed_keys(records: list[dict], metadata: dict) -> set[tuple[str, int, float]]:
    result = set()
    for record in records:
        for field in ("manifest_sha256", "model_path", "selector_source_sha256", "repeats", "retentions"):
            if record["run"].get(field) != metadata[field]:
                raise ValueError(f"resume metadata mismatch: {field}")
        key = (str(record["sample_id"]), int(record["repeat_index"]), float(record["requested_retention"]))
        if key in result:
            raise ValueError(f"duplicate latency record: {key}")
        result.add(key)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=[1.0, 0.875, 0.75, 0.5, 0.25])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1:
        parser.error("formal latency runs need at least 3 repeats and 1 warmup")
    if len(set(args.retentions)) != len(args.retentions) or any(not 0 < value <= 1 for value in args.retentions):
        parser.error("retentions must be unique and in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; inspect GPU before retrying")
    samples = load_jsonl(args.manifest)
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("manifest contains duplicate sample IDs")
    metadata = {
        "model_path": str(args.model.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "selector_source_sha256": hashlib.sha256(
            Path(__file__).with_name("shallow_block_pruning.py").read_bytes()
        ).hexdigest(),
        "repeats": args.repeats,
        "warmup_per_configuration": args.warmup,
        "retentions": args.retentions,
        "order": "rotate configurations by sample and repeat index",
        "gpu": torch.cuda.get_device_name(0),
        "precision": "BF16",
        "do_sample": False,
        "timing_boundary": "run_one inference core; processor and transfer reported separately as combined preprocess",
    }
    prior = load_jsonl(args.output_jsonl) if args.output_jsonl.is_file() else []
    done = completed_keys(prior, metadata)
    expected = {
        (str(sample["sample_id"]), repeat, retention)
        for sample in samples for repeat in range(args.repeats) for retention in args.retentions
    }
    if not done <= expected:
        raise ValueError("output contains records outside the requested experiment")
    print(f"LATENCY samples={len(samples)} configs={len(args.retentions)} repeats={args.repeats} pending={len(expected - done)}", flush=True)
    if done == expected:
        return 0
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa", low_cpu_mem_usage=True, local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    timers = [CudaCallTimer(block) for block in model.model.visual.blocks]

    def measure(sample: dict, retention: float) -> dict:
        torch.cuda.synchronize()
        record = run_one(
            model, processor, sample, args.manifest, retention=retention,
            prune_after_layer=5, anchor_grid=3, selector="l2_anchor",
            residual_energy_weight=0.25, text_weight=0.5, selector_head=None,
            selector_feature_mask=(0, 1, 2, 3, 4), joint_candidate_config=None,
            candidate_task_conditioned=False, block_timers=timers,
        )
        torch.cuda.synchronize()
        return record

    try:
        for retention in args.retentions:
            for index in range(args.warmup):
                measure(samples[index % len(samples)], retention)
            print(f"WARMUP retention={retention} count={args.warmup} PASS", flush=True)
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("a", encoding="utf-8", buffering=1) as stream:
            for repeat in range(args.repeats):
                for sample_index, sample in enumerate(samples):
                    for config_index in configuration_order(len(args.retentions), sample_index, repeat):
                        retention = args.retentions[config_index]
                        key = (str(sample["sample_id"]), repeat, retention)
                        if key in done:
                            continue
                        record = measure(sample, retention)
                        record["repeat_index"] = repeat
                        record["run"] = metadata
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        done.add(key)
                    if (sample_index + 1) % 10 == 0:
                        print(f"MEASURE repeat={repeat + 1}/{args.repeats} sample={sample_index + 1}/{len(samples)} saved={len(done)}", flush=True)
    finally:
        for timer in timers:
            timer.close()
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
