#!/usr/bin/env python3
"""Run single-pass shallow 2x2-group pruning on RoboBrain/Qwen3-VL."""

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
from typing import Any, Callable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, DynamicCache

from evaluation.joint_deletion import stable_candidate_seed
from evaluation.metrics import load_jsonl
from evaluation.learned_selector import (
    CandidateInteractionHead,
    CoordinateSelectorHead,
    L2ResidualSelectorHead,
)
from evaluation.run_hf_token_reduction import (
    CudaCallTimer,
    make_messages,
    processor_inputs,
    resolve_image,
)
from evaluation.shallow_block_pruning import (
    PrunedMultimodalInputs,
    PrunedVisionOutput,
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)
from evaluation.token_reduction import (
    qwen3_vl_compute_proxy,
    qwen3_vl_shallow_pruned_compute_proxy,
    retention_label,
)


def completed_ids(path: Path) -> set[str]:
    return {record["sample_id"] for record in load_jsonl(path)} if path.is_file() else set()


def matched_visual_token_targets(
    path: Path,
    strategy: str,
    *,
    expected_repeats: int,
) -> dict[str, int]:
    """Load one deterministic final visual-token target per sample."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in load_jsonl(path):
        if str(record.get("strategy")) == strategy:
            grouped.setdefault(str(record["sample_id"]), []).append(record)
    if not grouped:
        raise ValueError(f"no {strategy!r} records found in {path}")

    targets: dict[str, int] = {}
    for sample_id, records in grouped.items():
        if len(records) != expected_repeats:
            raise ValueError(
                f"{sample_id}: expected {expected_repeats} target repeats, got {len(records)}"
            )
        values = {int(record["final_context_visual_tokens"]) for record in records}
        if len(values) != 1:
            raise ValueError(f"{sample_id}: target visual-token count changed across repeats")
        target = values.pop()
        if target <= 0:
            raise ValueError(f"{sample_id}: target visual-token count must be positive")
        targets[sample_id] = target
    return targets


def limit_each_task(samples: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    if count is None:
        return samples
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for sample in samples:
        task = str(sample["task"])
        if counts.get(task, 0) < count:
            selected.append(sample)
            counts[task] = counts.get(task, 0) + 1
    return selected


def cuda_seconds(call: Callable[[], Any]) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = call()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end)) / 1000.0


def eos_ids(model: torch.nn.Module) -> set[int]:
    value = model.generation_config.eos_token_id
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(token) for token in value}


def greedy_generate_from_pruned(
    model: torch.nn.Module,
    prepared: PrunedMultimodalInputs,
    max_new_tokens: int,
) -> tuple[torch.Tensor, float, float, int]:
    cache = DynamicCache(config=model.config.text_config)

    def prefill() -> Any:
        return model.model.language_model(
            input_ids=None,
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            position_ids=prepared.position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(
                prepared.inputs_embeds.shape[1], device=prepared.inputs_embeds.device
            ),
            visual_pos_masks=prepared.visual_pos_masks,
            deepstack_visual_embeds=prepared.deepstack_visual_embeds,
        )

    outputs, prefill_seconds = cuda_seconds(prefill)
    logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    stop_ids = eos_ids(model)
    generated: list[int] = []
    decoder_seconds = 0.0
    physical_prefix_length = prepared.inputs_embeds.shape[1]
    next_mrope_position = int(prepared.position_ids.max().item()) + 1

    for step in range(max_new_tokens):
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated.append(next_token)
        if next_token in stop_ids or step + 1 == max_new_tokens:
            break
        token = torch.tensor([[next_token]], dtype=torch.long, device=prepared.input_ids.device)
        token_embed = model.get_input_embeddings()(token)
        position_ids = torch.full(
            (3, 1, 1),
            next_mrope_position + step,
            dtype=prepared.position_ids.dtype,
            device=prepared.position_ids.device,
        )
        attention_mask = torch.ones(
            (1, physical_prefix_length + step + 1),
            dtype=prepared.attention_mask.dtype,
            device=prepared.attention_mask.device,
        )

        def decode() -> Any:
            return model.model.language_model(
                input_ids=None,
                inputs_embeds=token_embed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor(
                    [physical_prefix_length + step],
                    dtype=torch.long,
                    device=prepared.input_ids.device,
                ),
            )

        outputs, one_decode = cuda_seconds(decode)
        decoder_seconds += one_decode
        logits = model.lm_head(outputs.last_hidden_state[:, -1, :])

    return (
        torch.tensor([generated], dtype=torch.long, device=prepared.input_ids.device),
        prefill_seconds,
        decoder_seconds,
        len(generated),
    )


def selection_sha256(indices: torch.Tensor) -> str:
    payload = ",".join(str(int(index)) for index in indices.tolist()).encode()
    return hashlib.sha256(payload).hexdigest()


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    retention: float | None,
    prune_after_layer: int,
    anchor_grid: int,
    selector: str,
    residual_energy_weight: float,
    text_weight: float,
    selector_head: torch.nn.Module | None,
    selector_feature_mask: tuple[int, ...],
    joint_candidate_config: dict[str, Any] | None,
    candidate_task_conditioned: bool,
    block_timers: list[CudaCallTimer],
    oracle_bbox_1000: tuple[float, float, float, float] | list[float] | None = None,
    target_llm_visual_tokens: int | None = None,
    processor_min_pixels: int | None = None,
    processor_max_pixels: int | None = None,
) -> dict[str, Any]:
    effective_selector = (
        "candidate_interaction_quota"
        if selector == "task_routed_candidate_interaction"
        and str(sample["task"]) == "grounding"
        else "l2_anchor"
        if selector == "task_routed_candidate_interaction"
        else selector
    )
    effective_oracle_bbox = oracle_bbox_1000
    if effective_selector == "geometric_oracle" and effective_oracle_bbox is None:
        effective_oracle_bbox = list(sample["ground_truth"]["bbox_1000"])
    image_path = resolve_image(manifest, sample["image"])
    messages = make_messages(image_path, sample["prompt"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    preprocess_started = time.perf_counter()
    batch = processor_inputs(
        processor,
        messages,
        image,
        min_pixels=processor_min_pixels,
        max_pixels=processor_max_pixels,
    ).to(model.device)
    preprocess_seconds = time.perf_counter() - preprocess_started
    full_patch_tokens = int(batch.image_grid_thw.prod().item())
    merge_unit = int(model.config.vision_config.spatial_merge_size) ** 2
    if full_patch_tokens % merge_unit:
        raise ValueError("processor grid is not aligned to the visual merger")
    full_llm_visual_tokens = full_patch_tokens // merge_unit
    requested_target = target_llm_visual_tokens
    if requested_target is not None:
        effective_target = min(requested_target, full_llm_visual_tokens)
        # Stay infinitesimally below the exact fraction because downstream
        # selectors use ceil(full_groups * retention).
        retention = (
            1.0
            if effective_target == full_llm_visual_tokens
            else (effective_target - 1e-6) / full_llm_visual_tokens
        )
    else:
        effective_target = None
    if retention is None:
        raise ValueError("retention is required without a matched token target")
    torch.cuda.reset_peak_memory_stats()
    for timer in block_timers:
        timer.reset()

    selector_text_embeddings = None
    if effective_selector == "text_residual_quota" or (
        effective_selector
        in {
            "learned_quota",
            "l2_residual_quota",
            "joint_set_quota",
            "candidate_interaction_quota",
        }
        and 2 in selector_feature_mask
    ):
        text_mask = (batch.attention_mask[0] != 0) & (
            batch.input_ids[0] != int(model.config.image_token_id)
        )
        with torch.inference_mode():
            selector_text_embeddings = model.get_input_embeddings()(
                batch.input_ids[0, text_mask]
            )

    def vision_call() -> PrunedVisionOutput:
        return forward_shallow_pruned_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            retention=retention,
            prune_after_layer=prune_after_layer,
            anchor_grid=anchor_grid,
            selector=effective_selector,
            residual_energy_weight=residual_energy_weight,
            text_embeddings=selector_text_embeddings,
            text_weight=text_weight,
            selector_head=selector_head,
            selector_feature_mask=selector_feature_mask,
            joint_candidate_seed=(
                stable_candidate_seed(
                    int(joint_candidate_config["base_seed"]), str(sample["image"])
                )
                if joint_candidate_config is not None
                else None
            ),
            joint_swap_counts=(
                tuple(int(value) for value in joint_candidate_config["swap_counts"])
                if joint_candidate_config is not None
                else (1, 2, 4)
            ),
            joint_candidates_per_swap=(
                int(joint_candidate_config["candidates_per_swap"])
                if joint_candidate_config is not None
                else 2
            ),
            candidate_task_value=(
                {"grounding": -1.0, "pointing": 1.0}[str(sample["task"])]
                if candidate_task_conditioned
                else 0.0
            )
            if effective_selector == "candidate_interaction_quota"
            else None,
            oracle_bbox_1000=effective_oracle_bbox,
        )

    inference_started = time.perf_counter()
    with torch.inference_mode():
        vision, vision_seconds = cuda_seconds(vision_call)
        if (
            effective_target is not None
            and int(vision.image_embeds.shape[0]) != effective_target
        ):
            raise RuntimeError(
                "matched token selector produced "
                f"{vision.image_embeds.shape[0]} groups, expected {effective_target}"
            )
        prepared = prepare_pruned_multimodal_inputs(model, batch, vision)
        generated, prefill_seconds, decoder_seconds, output_tokens = greedy_generate_from_pruned(
            model,
            prepared,
            int(sample["max_new_tokens"]),
        )
    inference_seconds = time.perf_counter() - inference_started
    raw_text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    block_seconds = [sum(timer.milliseconds()) / 1000.0 for timer in block_timers]
    full_proxy = qwen3_vl_compute_proxy(vision.full_patch_tokens)
    proxy = qwen3_vl_shallow_pruned_compute_proxy(
        vision.full_patch_tokens,
        vision.image_embeds.shape[0],
        prune_after_layer=prune_after_layer,
    )
    selected = vision.selected_group_indices.detach().cpu()
    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "output_text": raw_text,
        "raw_output_text": raw_text,
        "elapsed_seconds": inference_seconds,
        "timed_stage_seconds": vision_seconds + prefill_seconds + decoder_seconds,
        "preprocess_seconds": preprocess_seconds,
        "vision_encoder_seconds": vision_seconds,
        "vision_prefix_block_seconds": sum(block_seconds[: prune_after_layer + 1]),
        "vision_pruned_block_seconds": sum(block_seconds[prune_after_layer + 1 :]),
        "vision_other_seconds": max(0.0, vision_seconds - sum(block_seconds)),
        "llm_prefill_seconds": prefill_seconds,
        "decoder_seconds": decoder_seconds,
        "input_tokens": int(prepared.input_ids.shape[1]),
        "original_input_tokens": int(prepared.original_sequence_length),
        "output_tokens": output_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_encoder_tokens": vision.full_patch_tokens,
        "encoder_tokens_after_prune": proxy.encoder_tokens,
        "baseline_encoder_tokens": vision.full_patch_tokens,
        "llm_visual_tokens": proxy.llm_visual_tokens,
        "baseline_llm_visual_tokens": vision.full_llm_visual_tokens,
        "actual_token_retention": proxy.llm_visual_tokens / vision.full_llm_visual_tokens,
        "encoder_macs_proxy": proxy.encoder_macs,
        "baseline_encoder_macs_proxy": full_proxy.encoder_macs,
        "llm_prefill_macs_proxy": proxy.llm_prefill_macs,
        "baseline_llm_prefill_macs_proxy": full_proxy.llm_prefill_macs,
        "requested_retention": retention,
        "matched_requested_llm_visual_tokens": requested_target,
        "matched_effective_llm_visual_tokens": effective_target,
        "matched_target_capped_to_dense": (
            requested_target is not None and requested_target > full_llm_visual_tokens
        ),
        "prune_after_layer": prune_after_layer,
        "anchor_grid": anchor_grid,
        "selector": selector,
        "effective_selector": effective_selector,
        "residual_energy_weight": residual_energy_weight,
        "text_weight": text_weight,
        "selected_group_indices": selected.tolist(),
        "selected_group_indices_sha256": selection_sha256(selected),
        "joint_candidate_id": vision.joint_candidate_id,
        "joint_candidate_utility": vision.joint_candidate_utility,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del batch, vision, prepared, generated, image
    gc.collect()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--retentions", type=float, nargs="+", default=(1.0, 0.5, 0.25, 0.125))
    parser.add_argument("--prune-after-layer", type=int, default=5)
    parser.add_argument("--anchor-grid", type=int, default=3)
    parser.add_argument(
        "--selector",
        choices=(
            "l2_anchor",
            "residual_quota",
            "text_residual_quota",
            "learned_quota",
            "l2_residual_quota",
            "joint_set_quota",
            "candidate_interaction_quota",
            "task_routed_candidate_interaction",
            "geometric_oracle",
        ),
        default="l2_anchor",
    )
    parser.add_argument("--residual-energy-weight", type=float, default=0.25)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--selector-artifact", type=Path)
    parser.add_argument("--selector-variant")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--matched-token-results",
        type=Path,
        help="Result JSONL containing per-sample final visual-token targets.",
    )
    parser.add_argument("--matched-token-strategy", default="roi_only")
    parser.add_argument("--matched-token-expected-repeats", type=int, default=3)
    parser.add_argument("--matched-token-output-name", default="shallow_matched_tokens")
    parser.add_argument(
        "--matched-only",
        action="store_true",
        help="Run only the per-sample matched-token configuration.",
    )
    parser.add_argument(
        "--match-global-roi-processor-bounds",
        action="store_true",
        help=(
            "Use the explicit min/max pixel bounds from run_hf_global_roi.py "
            "so the dense processor grid is identical."
        ),
    )
    args = parser.parse_args()
    learned_selectors = {
        "learned_quota",
        "l2_residual_quota",
        "joint_set_quota",
        "candidate_interaction_quota",
        "task_routed_candidate_interaction",
    }
    if args.selector in learned_selectors and args.selector_artifact is None:
        parser.error("--selector-artifact is required for learned selectors")
    if args.matched_only and args.matched_token_results is None:
        parser.error("--matched-only requires --matched-token-results")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")
    for retention in args.retentions:
        retention_label(retention)
    samples = limit_each_task(load_jsonl(args.manifest), args.limit_per_task)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_targets: dict[str, int] | None = None
    matched_source_sha256: str | None = None
    if args.matched_token_results is not None:
        matched_targets = matched_visual_token_targets(
            args.matched_token_results,
            args.matched_token_strategy,
            expected_repeats=args.matched_token_expected_repeats,
        )
        missing = [str(sample["sample_id"]) for sample in samples if str(sample["sample_id"]) not in matched_targets]
        if missing:
            raise ValueError(f"matched token targets missing sample {missing[0]}")
        matched_source_sha256 = hashlib.sha256(
            args.matched_token_results.read_bytes()
        ).hexdigest()
    output_prefix = {
        "l2_anchor": "shallow",
        "residual_quota": "residual",
        "text_residual_quota": "textual",
        "learned_quota": f"learned_{args.selector_variant or 'selected'}",
        "l2_residual_quota": f"l2res_{args.selector_variant or 'selected'}",
        "joint_set_quota": f"joint_{args.selector_variant or 'selected'}",
        "candidate_interaction_quota": (
            f"interaction_{args.selector_variant or 'selected'}"
        ),
        "task_routed_candidate_interaction": (
            f"routed_interaction_{args.selector_variant or 'selected'}"
        ),
        "geometric_oracle": "geometric_oracle",
    }[args.selector]
    outputs: list[tuple[str, float | None, Path, set[str]]] = []
    if not args.matched_only:
        for retention in args.retentions:
            path = args.output_dir / f"{output_prefix}_{retention_label(retention)}.jsonl"
            outputs.append((f"retention={retention:.3f}", retention, path, completed_ids(path)))
    if matched_targets is not None:
        path = args.output_dir / f"{args.matched_token_output_name}.jsonl"
        outputs.append(("matched-token-targets", None, path, completed_ids(path)))
    pending_total = sum(
        sum(sample["sample_id"] not in done for sample in samples)
        for _, _, _, done in outputs
    )
    print(f"LOAD samples={len(samples)} configs={len(outputs)} pending={pending_total}", flush=True)
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
    processor_min_pixels: int | None = None
    processor_max_pixels: int | None = None
    if args.match_global_roi_processor_bounds:
        patch_size = int(model.config.vision_config.patch_size)
        merge_size = int(model.config.vision_config.spatial_merge_size)
        processor_min_pixels = (patch_size * merge_size) ** 2
        processor_max_pixels = int(processor.image_processor.size["longest_edge"])
    selector_head: torch.nn.Module | None = None
    selector_feature_mask = (0, 1, 2, 3, 4)
    selected_variant: str | None = None
    selector_artifact_sha256: str | None = None
    joint_candidate_config: dict[str, Any] | None = None
    candidate_task_conditioned = False
    if args.selector in learned_selectors:
        artifact = torch.load(args.selector_artifact, map_location="cpu", weights_only=True)
        selected_variant = args.selector_variant or str(artifact["selected_variant"])
        states = artifact["variant_state_dicts"]
        if selected_variant not in states:
            raise ValueError(f"selector variant not found: {selected_variant}")
        if args.selector in {
            "candidate_interaction_quota",
            "task_routed_candidate_interaction",
        }:
            if artifact.get("format") != "candidate-interaction-selector-v1":
                raise ValueError(
                    "candidate_interaction_quota requires a "
                    "candidate-interaction-selector-v1 artifact"
                )
            protocol = artifact["variants"][selected_variant]
            selector_feature_mask = tuple(
                int(value) for value in protocol["feature_mask"]
            )
            candidate_task_conditioned = bool(protocol["task_conditioned"])
            selector_head = CandidateInteractionHead(
                input_dim=int(artifact["input_dim"]),
                hidden=int(artifact["hidden"]),
                zero_initialize=False,
            ).to(args.device).eval()
        else:
            variant_masks = artifact.get(
                "variant_feature_masks",
                {
                    "full": [0, 1, 2, 3, 4],
                    "no_text": [0, 1, 3, 4],
                    "no_coordinates": [0, 1, 2],
                    "no_region_loss": [0, 1, 2, 3, 4],
                },
            )
            selector_feature_mask = tuple(
                int(value)
                for value in variant_masks.get(
                    selected_variant, [0, 1, 2, 3, 4]
                )
            )
            selector_head = (
                L2ResidualSelectorHead(zero_initialize=False)
                if args.selector in {"l2_residual_quota", "joint_set_quota"}
                else CoordinateSelectorHead()
            ).to(args.device).eval()
        selector_head.load_state_dict(states[selected_variant])
        selector_artifact_sha256 = hashlib.sha256(args.selector_artifact.read_bytes()).hexdigest()
        if args.selector in {
            "joint_set_quota",
            "candidate_interaction_quota",
            "task_routed_candidate_interaction",
        }:
            expected_format = {
                "joint_set_quota": "joint-deletion-selector-v1",
                "candidate_interaction_quota": "candidate-interaction-selector-v1",
                "task_routed_candidate_interaction": (
                    "candidate-interaction-selector-v1"
                ),
            }[args.selector]
            if artifact.get("format") != expected_format:
                raise ValueError(
                    f"{args.selector} requires a {expected_format} artifact"
                )
            joint_candidate_config = dict(artifact["candidate_config"])
            if int(joint_candidate_config["region_grid"]) != args.anchor_grid:
                raise ValueError("artifact region grid differs from --anchor-grid")
            artifact_retention = float(artifact["retention"])
            if any(abs(retention - artifact_retention) > 1e-12 for retention in args.retentions):
                raise ValueError("joint selector may only run at its trained retention")
    block_timers = [CudaCallTimer(block) for block in model.model.visual.blocks]
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "selector": args.selector,
        "residual_energy_weight": args.residual_energy_weight,
        "text_weight": args.text_weight,
        "selector_variant": selected_variant,
        "selector_feature_mask": list(selector_feature_mask),
        "selector_artifact": str(args.selector_artifact.resolve()) if args.selector_artifact else None,
        "selector_artifact_sha256": selector_artifact_sha256,
        "joint_candidate_config": joint_candidate_config,
        "candidate_task_conditioned": candidate_task_conditioned,
        "original_position_ids_preserved": True,
        "custom_greedy_decode": True,
        "matched_token_results": (
            str(args.matched_token_results.resolve())
            if args.matched_token_results is not None
            else None
        ),
        "matched_token_results_sha256": matched_source_sha256,
        "matched_token_strategy": (
            args.matched_token_strategy if matched_targets is not None else None
        ),
        "processor_bounds": {
            "mode": (
                "global_roi_explicit"
                if args.match_global_roi_processor_bounds
                else "processor_default"
            ),
            "min_pixels": processor_min_pixels,
            "max_pixels": processor_max_pixels,
        },
    }
    try:
        # Warm every pruning shape through the exact custom path.  Otherwise
        # the first sample at a new retention includes one-time SDPA/kernel
        # selection overhead and is not comparable with later samples.
        for config_label, retention, _, _ in outputs:
            warmup_target = (
                matched_targets[str(samples[0]["sample_id"])]
                if retention is None and matched_targets is not None
                else None
            )
            warmup = run_one(
                model,
                processor,
                samples[0],
                args.manifest,
                retention=retention,
                prune_after_layer=args.prune_after_layer,
                anchor_grid=args.anchor_grid,
                selector=args.selector,
                residual_energy_weight=args.residual_energy_weight,
                text_weight=args.text_weight,
                selector_head=selector_head,
                selector_feature_mask=selector_feature_mask,
                joint_candidate_config=joint_candidate_config,
                candidate_task_conditioned=candidate_task_conditioned,
                block_timers=block_timers,
                target_llm_visual_tokens=warmup_target,
                processor_min_pixels=processor_min_pixels,
                processor_max_pixels=processor_max_pixels,
            )
            print(
                f"WARMUP result=PASS config={config_label} output={warmup['output_text']!r}",
                flush=True,
            )
        for config_number, (config_label, retention, path, done) in enumerate(outputs, 1):
            pending = [sample for sample in samples if sample["sample_id"] not in done]
            print(
                f"CONFIG {config_number}/{len(outputs)} {config_label} pending={len(pending)}",
                flush=True,
            )
            if not pending:
                continue
            with path.open("a", encoding="utf-8", buffering=1) as stream:
                for sample_number, sample in enumerate(pending, 1):
                    target = (
                        matched_targets[str(sample["sample_id"])]
                        if retention is None and matched_targets is not None
                        else None
                    )
                    record = run_one(
                        model,
                        processor,
                        sample,
                        args.manifest,
                        retention=retention,
                        prune_after_layer=args.prune_after_layer,
                        anchor_grid=args.anchor_grid,
                        selector=args.selector,
                        residual_energy_weight=args.residual_energy_weight,
                        text_weight=args.text_weight,
                        selector_head=selector_head,
                        selector_feature_mask=selector_feature_mask,
                        joint_candidate_config=joint_candidate_config,
                        candidate_task_conditioned=candidate_task_conditioned,
                        block_timers=block_timers,
                        target_llm_visual_tokens=target,
                        processor_min_pixels=processor_min_pixels,
                        processor_max_pixels=processor_max_pixels,
                    )
                    record["run"] = metadata
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(
                        "RESULT "
                        f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                        f"actual={record['actual_token_retention']:.3f} "
                        f"vision={record['vision_encoder_seconds']:.4f}s "
                        f"prefill={record['llm_prefill_seconds']:.4f}s "
                        f"total={record['elapsed_seconds']:.4f}s output={record['output_text']!r}",
                        flush=True,
                    )
    finally:
        for timer in block_timers:
            timer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
