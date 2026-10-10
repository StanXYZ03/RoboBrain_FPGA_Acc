#!/usr/bin/env python3
"""Extract fixed-size ROI vision-prefix features for resolution policy studies."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl
from evaluation.run_hf_global_roi import original_image_path, prepare_strategy
from evaluation.shallow_block_pruning import text_relevance_group_scores
from evaluation.visual_refinement import grid_token_counts


def fixed_projection(hidden_size: int, output_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (hidden_size, output_size), generator=generator)
    return (signs.float() * 2.0 - 1.0) / math.sqrt(hidden_size)


def prefix_feature_vector(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    completed_blocks: int,
    projection: torch.Tensor,
    text_embeddings: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[str]]:
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("one still image is required")
    if not 0 <= completed_blocks <= len(visual.blocks):
        raise ValueError("completed_blocks is outside the vision stack")
    hidden = visual.patch_embed(pixel_values.type(visual.dtype))
    hidden = hidden + visual.fast_pos_embed_interpolate(grid_thw)
    rotary = visual.rot_pos_emb(grid_thw).reshape(hidden.shape[0], -1)
    rotary = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (rotary.cos(), rotary.sin())
    cu_seqlens = torch.tensor(
        [0, hidden.shape[0]], dtype=torch.int32, device=hidden.device
    )
    for block in visual.blocks[:completed_blocks]:
        hidden = block(
            hidden,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    merge = int(visual.spatial_merge_size)
    merge_unit = merge * merge
    grid_h = int(grid_thw[0, 1].item()) // merge
    grid_w = int(grid_thw[0, 2].item()) // merge
    groups = hidden.reshape(grid_h * grid_w, merge_unit, -1).float().mean(dim=1)
    projected = groups @ projection.to(groups.device)
    feature_values = [
        projected.mean(dim=0),
        projected.std(dim=0, unbiased=False),
        projected.amin(dim=0),
        projected.amax(dim=0),
    ]
    feature_names = [
        f"projected_{stat}_{index}"
        for stat in ("mean", "std", "min", "max")
        for index in range(projected.shape[1])
    ]
    spatial = projected.reshape(grid_h, grid_w, -1).permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(spatial, (4, 4)).flatten()
    feature_values.append(pooled)
    feature_names.extend(
        f"spatial_{row}_{column}_{channel}"
        for channel in range(projected.shape[1])
        for row in range(4)
        for column in range(4)
    )
    norms = groups.norm(dim=1)
    norm_quantiles = torch.quantile(
        norms, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=norms.device)
    )
    feature_values.extend(
        (
            norm_quantiles,
            torch.tensor(
                [grid_h, grid_w, grid_h / grid_w, groups.shape[0]],
                dtype=torch.float32,
                device=groups.device,
            ),
        )
    )
    feature_names.extend(
        ("norm_min", "norm_q10", "norm_q25", "norm_q50", "norm_q75", "norm_q90", "norm_max")
    )
    feature_names.extend(("grid_h", "grid_w", "grid_aspect", "group_count"))
    if text_embeddings is not None:
        layer_index = completed_blocks - 1
        if layer_index not in visual.deepstack_visual_indexes:
            raise ValueError("task relevance requires a completed DeepStack extraction layer")
        merger_index = visual.deepstack_visual_indexes.index(layer_index)
        projected_groups = visual.deepstack_merger_list[merger_index](hidden)
        relevance = text_relevance_group_scores(projected_groups, text_embeddings)
        relevance_map = relevance.reshape(1, 1, grid_h, grid_w)
        pooled_relevance = F.adaptive_avg_pool2d(relevance_map, (4, 4)).flatten()
        relevance_quantiles = torch.quantile(
            relevance,
            torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=relevance.device),
        )
        top_count = min(8, relevance.numel())
        top_values, top_indices = torch.topk(relevance, top_count, sorted=True)
        top_rows = torch.div(top_indices, grid_w, rounding_mode="floor").float() / max(grid_h - 1, 1)
        top_columns = (top_indices % grid_w).float() / max(grid_w - 1, 1)
        feature_values.extend((pooled_relevance, relevance_quantiles, top_values, top_rows, top_columns))
        feature_names.extend(f"text_relevance_grid_{row}_{column}" for row in range(4) for column in range(4))
        feature_names.extend(
            ("text_relevance_min", "text_relevance_q10", "text_relevance_q25", "text_relevance_q50", "text_relevance_q75", "text_relevance_q90", "text_relevance_max")
        )
        feature_names.extend(f"text_relevance_top_value_{index}" for index in range(top_count))
        feature_names.extend(f"text_relevance_top_row_{index}" for index in range(top_count))
        feature_names.extend(f"text_relevance_top_column_{index}" for index in range(top_count))
    vector = torch.cat([value.flatten() for value in feature_values]).detach().cpu()
    if vector.numel() != len(feature_names) or not torch.isfinite(vector).all():
        raise RuntimeError("invalid prefix feature vector")
    return vector, feature_names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi-long-edge", type=int, default=384)
    parser.add_argument("--oracle-margin", type=float, default=0.5)
    parser.add_argument("--completed-blocks", type=int, default=1)
    parser.add_argument("--projection-dim", type=int, default=16)
    parser.add_argument("--projection-seed", type=int, default=20260914)
    parser.add_argument("--task-conditioned-relevance", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("ROI vision-prefix extraction requires CUDA")

    all_samples = load_jsonl(args.manifest)
    if args.task_conditioned_relevance:
        samples = all_samples
        completed: set[Any] = {
            str(record["sample_id"]) for record in load_jsonl(args.output)
        } if args.output.is_file() else set()
    else:
        by_image: dict[int, dict[str, Any]] = {}
        for sample in all_samples:
            image_id = int(sample["coco_image_id"])
            if image_id not in by_image or sample["task"] == "grounding":
                by_image[image_id] = sample
        samples = [by_image[image_id] for image_id in sorted(by_image)]
        completed = {
            int(record["coco_image_id"]) for record in load_jsonl(args.output)
        } if args.output.is_file() else set()

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
    visual = model.model.visual
    projection = fixed_projection(
        int(model.config.vision_config.hidden_size), args.projection_dim, args.projection_seed
    )
    projection_sha256 = hashlib.sha256(projection.numpy().tobytes()).hexdigest()
    minimum_pixels = (
        int(model.config.vision_config.patch_size)
        * int(model.config.vision_config.spatial_merge_size)
    ) ** 2
    maximum_pixels = int(processor.image_processor.size["longest_edge"])

    def prepare(sample: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        path = original_image_path(args.manifest, sample, args.original_image_dir)
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        prepared = prepare_strategy(
            sample,
            path,
            original,
            "roi_only",
            global_long_edge=256,
            roi_long_edge=args.roi_long_edge,
            oracle_margin=args.oracle_margin,
            roi_upscale=True,
        )
        text = processor.apply_chat_template(
            prepared["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=prepared["images"],
            padding=True,
            return_tensors="pt",
            images_kwargs={"min_pixels": minimum_pixels, "max_pixels": maximum_pixels},
        ).to("cuda:0")
        return inputs, prepared

    def text_embeddings(inputs: Any) -> torch.Tensor:
        mask = (inputs.attention_mask[0] != 0) & (
            inputs.input_ids[0] != int(model.config.image_token_id)
        )
        return model.get_input_embeddings()(inputs.input_ids[0, mask])

    warmup_inputs, _ = prepare(samples[0])
    with torch.inference_mode():
        prefix_feature_vector(
            visual,
            warmup_inputs.pixel_values,
            warmup_inputs.image_grid_thw,
            completed_blocks=args.completed_blocks,
            projection=projection,
            text_embeddings=(
                text_embeddings(warmup_inputs) if args.task_conditioned_relevance else None
            ),
        )
    torch.cuda.synchronize()
    del warmup_inputs

    metadata = {
        "model": str(args.model.resolve()),
        "dtype": "bf16",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "roi_long_edge": args.roi_long_edge,
        "oracle_margin": args.oracle_margin,
        "completed_vision_blocks": args.completed_blocks,
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "projection_sha256": projection_sha256,
        "oracle_roi_conditional": True,
        "task_conditioned_relevance": args.task_conditioned_relevance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for index, sample in enumerate(samples, 1):
            image_id = int(sample["coco_image_id"])
            completion_key: Any = str(sample["sample_id"]) if args.task_conditioned_relevance else image_id
            if completion_key in completed:
                continue
            inputs, prepared = prepare(sample)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                vector, names = prefix_feature_vector(
                    visual,
                    inputs.pixel_values,
                    inputs.image_grid_thw,
                    completed_blocks=args.completed_blocks,
                    projection=projection,
                    text_embeddings=(
                        text_embeddings(inputs) if args.task_conditioned_relevance else None
                    ),
                )
            torch.cuda.synchronize()
            prefix_ms = 1000.0 * (time.perf_counter() - started)
            counts = grid_token_counts(
                inputs.image_grid_thw.tolist(),
                merge_size=int(model.config.vision_config.spatial_merge_size),
            )
            record = {
                "coco_image_id": image_id,
                "sample_id": sample["sample_id"] if args.task_conditioned_relevance else None,
                "task": sample["task"] if args.task_conditioned_relevance else None,
                "source_sample_id": sample["sample_id"],
                "pixel_crop_box": list(prepared["pixel_crop"]),
                "effective_crop_box_1000": list(prepared["effective_crop"]),
                "llm_visual_tokens": counts["llm_visual_tokens_total"],
                "image_grid_thw": inputs.image_grid_thw.tolist(),
                "prefix_ms": prefix_ms,
                "feature_names": names,
                "feature_vector": vector.tolist(),
                "run": metadata,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(samples)} image={image_id} task={record['task']} tokens={record['llm_visual_tokens']} "
                f"features={len(names)} prefix={prefix_ms:.3f}ms",
                flush=True,
            )
            del inputs, prepared, vector
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
