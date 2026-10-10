#!/usr/bin/env python3
"""Extract hardware-computable layer-5 features for selector distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.learned_selector import normalize_per_image, scalar_selector_features
from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import make_messages, processor_inputs, resolve_image
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    text_relevance_group_scores,
)


def prompt_embeddings(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    text_mask = (batch.attention_mask[0] != 0) & (
        batch.input_ids[0] != int(model.config.image_token_id)
    )
    return model.get_input_embeddings()(batch.input_ids[0, text_mask])


def feature_record(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
) -> dict[str, Any]:
    image_path = resolve_image(manifest, sample["image"])
    messages = make_messages(image_path, sample["prompt"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    batch = processor_inputs(processor, messages, image).to(model.device)
    text_embeddings = prompt_embeddings(model, batch)
    vision = forward_shallow_pruned_vision(
        model.model.visual,
        batch.pixel_values,
        batch.image_grid_thw,
        retention=1.0,
        prune_after_layer=5,
        selector="residual_quota",
        residual_energy_weight=0.0,
        capture_selector_features=True,
    )
    if (
        vision.group_residual is None
        or vision.group_energy is None
        or vision.group_l2 is None
    ):
        raise RuntimeError("selector scalar features were not captured")
    projected = vision.deepstack_embeds[0]
    relevance = text_relevance_group_scores(projected, text_embeddings)
    merge_size = int(model.model.visual.spatial_merge_size)
    grid_h = int(batch.image_grid_thw[0, 1].item()) // merge_size
    grid_w = int(batch.image_grid_thw[0, 2].item()) // merge_size
    features = scalar_selector_features(
        vision.group_residual,
        vision.group_energy,
        relevance,
        grid_h,
        grid_w,
    ).detach().cpu()
    normalized_l2 = normalize_per_image(vision.group_l2[:, None]).squeeze(1).detach().cpu()
    payload = ",".join(f"{value:.8e}" for value in features.flatten().tolist()).encode()
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "group_count": int(features.shape[0]),
        "grid_hw": [grid_h, grid_w],
        "feature_channels": ["residual", "energy", "text_relevance", "x", "y"],
        "features_finite": bool(torch.isfinite(features).all()),
        "features_sha256": hashlib.sha256(payload).hexdigest(),
        "selector_features": features.tolist(),
        "normalized_l2": normalized_l2.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("feature extraction requires CUDA")
    samples = load_jsonl(args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": str(args.model.resolve()),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "dtype": "bf16",
        "feature_point": "vision layer 5 before pruning",
    }
    with args.output.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for index, sample in enumerate(samples, 1):
            record = feature_record(model, processor, sample, args.manifest)
            record["run"] = metadata
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(samples)} id={record['sample_id']} "
                f"groups={record['group_count']} finite={record['features_finite']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
