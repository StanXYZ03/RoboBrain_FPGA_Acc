#!/usr/bin/env python3
"""Extract one or more layer-5 activation-gradient saliency teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import make_messages, processor_inputs, resolve_image
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)


def target_text(sample: dict[str, Any]) -> str:
    if sample["task"] == "pointing":
        x, y = sample["ground_truth"]["point_1000"]
        return f"[({round(x)}, {round(y)})]"
    x1, y1, x2, y2 = sample["ground_truth"]["bbox_1000"]
    return f"[{round(x1)}, {round(y1)}, {round(x2)}, {round(y2)}]"

def teacher_forced_coordinate_nll(
    model: torch.nn.Module,
    processor: Any,
    prepared: Any,
    answer: str,
) -> torch.Tensor:
    """Return coordinate-token NLL for one prepared multimodal prompt."""
    answer_ids = processor.tokenizer(
        answer,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(model.device)
    answer_embeds = model.get_input_embeddings()(answer_ids)
    prompt_length = prepared.inputs_embeds.shape[1]
    answer_length = answer_ids.shape[1]
    inputs_embeds = torch.cat((prepared.inputs_embeds, answer_embeds), dim=1)
    attention_mask = torch.ones(
        (1, prompt_length + answer_length),
        dtype=prepared.attention_mask.dtype,
        device=model.device,
    )
    next_position = int(prepared.position_ids.max().item()) + 1
    answer_positions = torch.arange(
        next_position,
        next_position + answer_length,
        dtype=prepared.position_ids.dtype,
        device=model.device,
    ).view(1, 1, -1).expand(3, 1, -1)
    position_ids = torch.cat((prepared.position_ids, answer_positions), dim=2)
    visual_mask = torch.cat(
        (
            prepared.visual_pos_masks,
            torch.zeros((1, answer_length), dtype=torch.bool, device=model.device),
        ),
        dim=1,
    )
    outputs = model.model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        visual_pos_masks=visual_mask,
        deepstack_visual_embeds=prepared.deepstack_visual_embeds,
    )
    logits = model.lm_head(outputs.last_hidden_state)
    prediction_logits = logits[:, prompt_length - 1 : prompt_length - 1 + answer_length]
    return F.cross_entropy(
        prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
        answer_ids.reshape(-1),
    )



def extract_one(
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

    vision = forward_shallow_pruned_vision(
        model.model.visual,
        batch.pixel_values,
        batch.image_grid_thw,
        retention=1.0,
        prune_after_layer=5,
        anchor_grid=3,
        selector="residual_quota",
        residual_energy_weight=0.0,
        capture_prune_gradient=True,
    )
    if vision.prune_hidden_states is None:
        raise RuntimeError("vision pruning activation was not captured")
    prepared = prepare_pruned_multimodal_inputs(model, batch, vision)
    answer = target_text(sample)
    loss = teacher_forced_coordinate_nll(model, processor, prepared, answer)
    loss.backward()
    activation = vision.prune_hidden_states
    gradient = activation.grad
    if gradient is None:
        raise RuntimeError("no gradient reached the captured vision activation")
    merge_unit = int(model.model.visual.spatial_merge_size) ** 2
    group_saliency = (activation.float() * gradient.float()).abs().mean(dim=-1)
    group_saliency = group_saliency.view(-1, merge_unit).mean(dim=1)
    values = group_saliency.detach().cpu()
    payload = ",".join(f"{value:.9e}" for value in values.tolist()).encode()
    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "target_text": answer,
        "teacher_loss": float(loss.detach().cpu()),
        "group_count": values.numel(),
        "grid_hw": [
            int(batch.image_grid_thw[0, 1].item()) // int(model.model.visual.spatial_merge_size),
            int(batch.image_grid_thw[0, 2].item()) // int(model.model.visual.spatial_merge_size),
        ],
        "saliency_min": float(values.min()),
        "saliency_mean": float(values.mean()),
        "saliency_max": float(values.max()),
        "saliency_finite": bool(torch.isfinite(values).all()),
        "saliency_sha256": hashlib.sha256(payload).hexdigest(),
        "group_saliency": values.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("teacher extraction requires CUDA")
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
    samples = load_jsonl(args.manifest)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(samples, 1):
            model.zero_grad(set_to_none=True)
            record = extract_one(model, processor, sample, args.manifest)
            record["run"] = {
                "model": str(args.model.resolve()),
                "torch": torch.__version__,
                "transformers": __import__("transformers").__version__,
                "python": platform.python_version(),
                "dtype": "bf16",
                "teacher": "ground-truth coordinate NLL activation-times-gradient at vision layer 5",
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"RESULT {index}/{len(samples)} id={record['sample_id']} "
                f"loss={record['teacher_loss']:.6f} groups={record['group_count']} "
                f"finite={record['saliency_finite']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
