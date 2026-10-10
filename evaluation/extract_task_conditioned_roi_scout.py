#!/usr/bin/env python3
"""Extract task-conditioned low-resolution spatial features for ROI proposals."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.extract_roi_vision_prefix_features import fixed_projection
from evaluation.metrics import load_jsonl
from evaluation.run_hf_global_roi import original_image_path
from evaluation.run_hf_token_reduction import make_messages, processor_inputs
from evaluation.shallow_block_pruning import text_relevance_group_scores
from evaluation.visual_refinement import grid_token_counts, resize_long_edge


def spatial_scout_features(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    completed_blocks: int,
    projection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return one compact feature vector and one text score per merged cell."""
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise ValueError("one still image is required")
    if completed_blocks - 1 not in visual.deepstack_visual_indexes:
        raise ValueError("completed_blocks must end at a DeepStack extraction layer")

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

    merger_index = visual.deepstack_visual_indexes.index(completed_blocks - 1)
    projected_groups = visual.deepstack_merger_list[merger_index](hidden).float()
    compact = projected_groups @ projection.to(projected_groups.device)
    relevance = text_relevance_group_scores(projected_groups, text_embeddings)
    merge = int(visual.spatial_merge_size)
    grid_h = int(grid_thw[0, 1].item()) // merge
    grid_w = int(grid_thw[0, 2].item()) // merge
    if compact.shape[0] != grid_h * grid_w or relevance.shape != (grid_h * grid_w,):
        raise RuntimeError("merged feature count does not match the spatial grid")
    if not torch.isfinite(compact).all() or not torch.isfinite(relevance).all():
        raise RuntimeError("non-finite scout feature")
    return compact.detach().cpu(), relevance.detach().cpu(), grid_h, grid_w


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--original-image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-long-edge", type=int, default=256)
    parser.add_argument("--completed-blocks", type=int, default=6)
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-seed", type=int, default=20260915)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scout feature extraction requires CUDA")

    samples = load_jsonl(args.manifest)
    if len({str(sample["sample_id"]) for sample in samples}) != len(samples):
        raise ValueError("manifest sample IDs are not unique")
    completed = (
        {str(record["sample_id"]) for record in load_jsonl(args.output)}
        if args.output.is_file()
        else set()
    )

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
    text_hidden = int(model.get_input_embeddings().weight.shape[1])
    projection = fixed_projection(text_hidden, args.projection_dim, args.projection_seed)
    projection_sha256 = hashlib.sha256(projection.numpy().tobytes()).hexdigest()
    minimum_pixels = (
        int(model.config.vision_config.patch_size)
        * int(model.config.vision_config.spatial_merge_size)
    ) ** 2
    maximum_pixels = int(processor.image_processor.size["longest_edge"])

    def prepare(sample: dict[str, Any]) -> tuple[Any, tuple[int, int], Path]:
        path = original_image_path(args.manifest, sample, args.original_image_dir)
        with Image.open(path) as opened:
            original_size = opened.size
            image = resize_long_edge(
                opened.convert("RGB"), args.global_long_edge, allow_upscale=False
            )
        inputs = processor_inputs(
            processor,
            make_messages(path, str(sample["prompt"])),
            image,
            min_pixels=minimum_pixels,
            max_pixels=maximum_pixels,
        ).to("cuda:0")
        return inputs, original_size, path

    def prompt_embeddings(inputs: Any) -> torch.Tensor:
        mask = (inputs.attention_mask[0] != 0) & (
            inputs.input_ids[0] != int(model.config.image_token_id)
        )
        embeddings = model.get_input_embeddings()(inputs.input_ids[0, mask])
        if embeddings.shape[0] == 0:
            raise RuntimeError("prompt produced no non-image embeddings")
        return embeddings

    warmup_inputs, _, _ = prepare(samples[0])
    with torch.inference_mode():
        spatial_scout_features(
            visual,
            warmup_inputs.pixel_values,
            warmup_inputs.image_grid_thw,
            prompt_embeddings(warmup_inputs),
            completed_blocks=args.completed_blocks,
            projection=projection,
        )
    torch.cuda.synchronize()
    del warmup_inputs

    metadata = {
        "model": str(args.model.resolve()),
        "dtype": "bf16",
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "global_long_edge": args.global_long_edge,
        "completed_vision_blocks": args.completed_blocks,
        "deepstack_layer_zero_based": args.completed_blocks - 1,
        "projection_dim": args.projection_dim,
        "projection_seed": args.projection_seed,
        "projection_sha256": projection_sha256,
        "source_is_full_image": True,
        "generation_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for index, sample in enumerate(samples, 1):
            sample_id = str(sample["sample_id"])
            if sample_id in completed:
                continue
            inputs, original_size, path = prepare(sample)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                compact, relevance, grid_h, grid_w = spatial_scout_features(
                    visual,
                    inputs.pixel_values,
                    inputs.image_grid_thw,
                    prompt_embeddings(inputs),
                    completed_blocks=args.completed_blocks,
                    projection=projection,
                )
            torch.cuda.synchronize()
            prefix_ms = 1000.0 * (time.perf_counter() - started)
            counts = grid_token_counts(
                inputs.image_grid_thw.tolist(),
                merge_size=int(model.config.vision_config.spatial_merge_size),
            )
            record = {
                "sample_id": sample_id,
                "coco_image_id": int(sample["coco_image_id"]),
                "task": str(sample["task"]),
                "image": str(sample["image"]),
                "resolved_image": str(path),
                "original_width": int(original_size[0]),
                "original_height": int(original_size[1]),
                "merged_grid_h": grid_h,
                "merged_grid_w": grid_w,
                "image_grid_thw": inputs.image_grid_thw.tolist(),
                "encoder_tokens": counts["encoder_tokens_total"],
                "llm_visual_tokens": counts["llm_visual_tokens_total"],
                "prefix_ms": prefix_ms,
                "projected_cell_features": compact.tolist(),
                "text_relevance": relevance.tolist(),
                "run": metadata,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(samples)} sample={sample_id} "
                f"grid={grid_h}x{grid_w} prefix={prefix_ms:.3f}ms",
                flush=True,
            )
            del inputs, compact, relevance
            gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
