#!/usr/bin/env python3
"""Measure coordinate NLL for region-preserving joint visual-token deletions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.extract_coordinate_saliency import (
    target_text,
    teacher_forced_coordinate_nll,
)
from evaluation.joint_deletion import (
    region_preserving_swap_candidates,
    require_development_manifest,
    stable_candidate_seed,
)
from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import make_messages, processor_inputs, resolve_image
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_per_task(samples: list[dict[str, Any]], per_task_limit: int) -> list[dict[str, Any]]:
    if per_task_limit <= 0:
        raise ValueError("per-task limit must be positive")
    counts: Counter[str] = Counter()
    selected = []
    for sample in samples:
        task = str(sample["task"])
        if counts[task] >= per_task_limit:
            continue
        selected.append(sample)
        counts[task] += 1
    expected_tasks = {str(sample["task"]) for sample in samples}
    if any(counts[task] != per_task_limit for task in expected_tasks):
        raise ValueError("manifest does not contain enough examples for every task")
    return selected




def evaluate_mask(
    model: torch.nn.Module,
    processor: Any,
    batch: Any,
    answer: str,
    *,
    retention: float,
    selector: str,
    fixed_indices: torch.Tensor | None = None,
) -> tuple[Any, float, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        vision = forward_shallow_pruned_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            retention=retention,
            prune_after_layer=5,
            anchor_grid=3,
            selector=selector,
            fixed_group_indices=fixed_indices,
        )
        prepared = prepare_pruned_multimodal_inputs(model, batch, vision)
        loss = teacher_forced_coordinate_nll(model, processor, prepared, answer)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    value = float(loss.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("joint-deletion coordinate NLL is not finite")
    return vision, value, elapsed


def mask_payload(
    candidate_id: str,
    swap_count: int,
    selected_indices: list[int],
    coordinate_nll: float,
    elapsed_seconds: float,
    baseline_nll: float,
) -> dict[str, Any]:
    encoded = ",".join(str(index) for index in selected_indices).encode()
    return {
        "candidate_id": candidate_id,
        "swap_count": swap_count,
        "selected_group_count": len(selected_indices),
        "selected_group_indices": selected_indices,
        "mask_sha256": hashlib.sha256(encoded).hexdigest(),
        "coordinate_nll": coordinate_nll,
        "delta_nll_vs_l2": coordinate_nll - baseline_nll,
        "elapsed_seconds": elapsed_seconds,
    }


def extract_sample(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    retention: float,
    swap_counts: tuple[int, ...],
    candidates_per_swap: int,
    region_grid: int,
    seed: int,
) -> dict[str, Any]:
    image_path = resolve_image(manifest, sample["image"])
    messages = make_messages(image_path, sample["prompt"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    batch = processor_inputs(processor, messages, image).to(model.device)
    answer = target_text(sample)

    baseline_vision, baseline_nll, baseline_elapsed = evaluate_mask(
        model,
        processor,
        batch,
        answer,
        retention=retention,
        selector="l2_anchor",
    )
    merge_size = int(model.model.visual.spatial_merge_size)
    grid_h = int(batch.image_grid_thw[0, 1].item()) // merge_size
    grid_w = int(batch.image_grid_thw[0, 2].item()) // merge_size
    candidates = region_preserving_swap_candidates(
        baseline_vision.selected_group_indices,
        grid_h,
        grid_w,
        swap_counts=swap_counts,
        candidates_per_swap=candidates_per_swap,
        region_grid=region_grid,
        seed=stable_candidate_seed(seed, str(sample["image"])),
    )
    expected_keep = math.ceil(grid_h * grid_w * retention)
    if len(candidates[0].selected_group_indices) != expected_keep:
        raise RuntimeError("L2 baseline does not match the requested token budget")

    measured = [
        mask_payload(
            "baseline",
            0,
            baseline_vision.selected_group_indices.detach().cpu().tolist(),
            baseline_nll,
            baseline_elapsed,
            baseline_nll,
        )
    ]
    for candidate in candidates[1:]:
        fixed = torch.tensor(candidate.selected_group_indices, device=model.device)
        vision, coordinate_nll, elapsed = evaluate_mask(
            model,
            processor,
            batch,
            answer,
            retention=retention,
            selector="fixed_indices",
            fixed_indices=fixed,
        )
        actual = vision.selected_group_indices.detach().cpu().tolist()
        if actual != list(candidate.selected_group_indices):
            raise RuntimeError("fixed selector changed the requested group mask")
        measured.append(
            mask_payload(
                candidate.candidate_id,
                candidate.swap_count,
                actual,
                coordinate_nll,
                elapsed,
                baseline_nll,
            )
        )

    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "image": sample["image"],
        "target_text": answer,
        "grid_hw": [grid_h, grid_w],
        "full_group_count": grid_h * grid_w,
        "retention": retention,
        "retained_group_count": expected_keep,
        "candidate_count": len(measured),
        "candidates": measured,
    }


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record["sample_id"])
        if sample_id in records:
            raise ValueError(f"duplicate sample_id at output line {line_number}: {sample_id}")
        records[sample_id] = record
    return records


def write_summary(
    path: Path,
    records: list[dict[str, Any]],
    *,
    manifest: Path,
    output: Path,
) -> None:
    tasks = Counter(str(record["task"]) for record in records)
    deltas = [
        float(candidate["delta_nll_vs_l2"])
        for record in records
        for candidate in record["candidates"][1:]
    ]
    summary = {
        "format": "joint-deletion-coordinate-nll-summary-v1",
        "status": "PILOT_ONLY_NOT_ACCURACY_CLAIM",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "raw_jsonl": str(output.resolve()),
        "raw_jsonl_sha256": file_sha256(output),
        "sample_count": len(records),
        "task_counts": dict(sorted(tasks.items())),
        "candidate_count": sum(int(record["candidate_count"]) for record in records),
        "all_losses_finite": all(
            math.isfinite(float(candidate["coordinate_nll"]))
            for record in records
            for candidate in record["candidates"]
        ),
        "nonbaseline_delta_nll_min": min(deltas) if deltas else None,
        "nonbaseline_delta_nll_max": max(deltas) if deltas else None,
        "nonbaseline_better_than_l2_count": sum(delta < 0 for delta in deltas),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-task-limit", type=int, default=1)
    parser.add_argument("--retention", type=float, default=0.875)
    parser.add_argument("--swap-counts", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--candidates-per-swap", type=int, default=2)
    parser.add_argument("--region-grid", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()
    try:
        require_development_manifest(str(args.manifest))
    except ValueError as error:
        parser.error(str(error))
    if not (0.0 < args.retention < 1.0):
        parser.error("retention must be in (0, 1) for joint deletion")
    if not torch.cuda.is_available():
        raise RuntimeError("joint-deletion teacher extraction requires CUDA")

    samples = select_per_task(load_jsonl(args.manifest), args.per_task_limit)
    existing = load_existing(args.output)
    pending = [sample for sample in samples if str(sample["sample_id"]) not in existing]
    print(
        f"LOAD selected={len(samples)} existing={len(existing)} pending={len(pending)}",
        flush=True,
    )
    if pending:
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model),
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as stream:
            for index, sample in enumerate(pending, 1):
                record = extract_sample(
                    model,
                    processor,
                    sample,
                    args.manifest,
                    retention=args.retention,
                    swap_counts=tuple(args.swap_counts),
                    candidates_per_swap=args.candidates_per_swap,
                    region_grid=args.region_grid,
                    seed=args.seed,
                )
                record["run"] = {
                    "model": str(args.model.resolve()),
                    "torch": torch.__version__,
                    "transformers": __import__("transformers").__version__,
                    "python": platform.python_version(),
                    "dtype": "bf16",
                    "objective": "teacher-forced coordinate-token NLL",
                    "selector_baseline": "layer-5 L2 anchors",
                    "candidate_policy": "same-budget within-region multi-group swaps",
                    "swap_counts": args.swap_counts,
                    "candidates_per_swap": args.candidates_per_swap,
                    "region_grid": args.region_grid,
                    "seed": args.seed,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                best_delta = min(
                    candidate["delta_nll_vs_l2"] for candidate in record["candidates"]
                )
                print(
                    f"RESULT {index}/{len(pending)} id={record['sample_id']} "
                    f"task={record['task']} groups={record['retained_group_count']}/"
                    f"{record['full_group_count']} candidates={record['candidate_count']} "
                    f"best_delta={best_delta:+.6f}",
                    flush=True,
                )

    records = list(load_existing(args.output).values())
    write_summary(args.summary, records, manifest=args.manifest, output=args.output)
    print(f"SUMMARY records={len(records)} output={args.output} summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

