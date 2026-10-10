#!/usr/bin/env python3
"""Run the layer-5 unbounded real-only L2 + 3x3 anchor arm at K=96.

Independent diagnostic arm (B) for the preregistered
``configs/2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json``
comparison against the frozen layer-5 raw-representative artifact:

* identical screening100 manifest, cut point ``prune_after_layer=5``,
  ``score_mode="l2"``, ``budget=96``, ``region_grid=3``, BF16 + SDPA + greedy
  decoding;
* **no** context pooling, **no** context slot reservation and **no**
  ``local_candidate_cap`` (there is deliberately no ``--local-candidate-cap``
  knob);
* one anchor per non-empty 3x3 region plus the **global** L2 ranking for the
  remaining slots; every emitted slot is a real original 2x2 merge group;
* ``G <= K`` is an element-wise all-real identity bypass.

No training and no B1-trained ``Wv`` projection (there is deliberately no
``--projection-npz`` knob) and no other score mode.  This runner is only
executed on a GPU host; it is CPU-import-safe.
"""

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
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.hardware_regular_layer5_unbounded_l2 import (
    ARM_LABEL,
    LOCAL_CANDIDATE_CAP,
    PRUNE_AFTER_LAYER,
    SCORE_MODE_L2,
    SELECTION_RULE,
    forward_hardware_regular_layer5_unbounded_l2_vision,
    prepare_layer5_unbounded_multimodal_inputs,
    verify_real_only_selection,
)
from evaluation.run_hf_shallow_block_pruning import (
    completed_ids,
    greedy_generate_from_pruned,
    limit_each_task,
    selection_sha256,
)
from evaluation.run_hf_token_reduction import (
    make_messages,
    processor_inputs,
    resolve_image,
)
from evaluation.token_reduction import (
    qwen3_vl_compute_proxy,
    qwen3_vl_shallow_pruned_compute_proxy,
)

#: Re-exported for the frozen runner contract and tests.
SCORE_MODE = SCORE_MODE_L2
CUTPOINT_PREFIX_BLOCKS = tuple(range(0, PRUNE_AFTER_LAYER + 1))
#: Frozen representative arm this unbounded arm is paired against.
BASELINE_REPRESENTATIVE_JSONL = (
    "artifacts/token_selector/2026-10-04-layer5-context-pooling-ablation/"
    "layer5_context_ablation_k96_screening100.jsonl"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sync_clock() -> float:
    """Drain the GPU then read the wall clock: a clean stage boundary."""
    torch.cuda.synchronize()
    return time.perf_counter()


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    budget: int,
    score_mode: str,
    region_grid: int,
) -> dict[str, Any]:
    if score_mode != SCORE_MODE_L2:
        raise ValueError(
            f"the layer-5 unbounded runner only supports score_mode='{SCORE_MODE_L2}'"
        )
    if LOCAL_CANDIDATE_CAP is not None:
        raise AssertionError(
            "the unbounded real-only arm must not use a local_candidate_cap"
        )
    application_started = _sync_clock()
    image_path = resolve_image(manifest, sample["image"])
    messages = make_messages(image_path, sample["prompt"])

    image_started = _sync_clock()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image_open_decode_seconds = _sync_clock() - image_started

    processor_started = _sync_clock()
    processed = processor_inputs(processor, messages, image)
    processor_seconds = _sync_clock() - processor_started

    host_to_device_started = _sync_clock()
    batch = processed.to(model.device)
    host_to_device_seconds = _sync_clock() - host_to_device_started

    full_patch_tokens = int(batch.image_grid_thw.prod().item())
    merge_unit = int(model.config.vision_config.spatial_merge_size) ** 2
    if full_patch_tokens % merge_unit:
        raise ValueError("processor grid is not aligned to the visual merger")
    full_llm_visual_tokens = full_patch_tokens // merge_unit

    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    with torch.inference_mode():
        vision = forward_hardware_regular_layer5_unbounded_l2_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            budget=budget,
            score_mode=score_mode,
            region_grid=region_grid,
            prune_after_layer=PRUNE_AFTER_LAYER,
            synchronize=torch.cuda.synchronize,
        )
        llm_prepare_started = _sync_clock()
        prepared = prepare_layer5_unbounded_multimodal_inputs(
            model, batch, vision
        )
        llm_prepare_seconds = _sync_clock() - llm_prepare_started
        generated, prefill_seconds, decoder_seconds, output_tokens = (
            greedy_generate_from_pruned(
                model,
                prepared,
                int(sample["max_new_tokens"]),
            )
        )
    model_inference_seconds = _sync_clock() - model_inference_started

    postprocess_started = _sync_clock()
    raw_text = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    postprocess_seconds = _sync_clock() - postprocess_started
    application_end_to_end_seconds = _sync_clock() - application_started

    vision_total_seconds = vision.vision_total_seconds
    gpu_vision_seconds = vision.gpu_vision_seconds
    if vision_total_seconds != (
        vision.vision_preamble_seconds
        + vision.cut_blocks_seconds
        + vision.selector_cpu_seconds
        + vision.remaining_vision_seconds
    ):
        raise RuntimeError("vision timing segments do not sum to the vision total")
    # Fail closed unless the all-real / no-context / exact-K contract holds.
    verify_real_only_selection(vision.selection)
    if vision.uses_context_pooling:
        raise RuntimeError("unbounded real-only arm must not use context pooling")
    if vision.uses_context_slot_reservation:
        raise RuntimeError(
            "unbounded real-only arm must not use context slot reservation"
        )
    if vision.uses_local_candidate_cap:
        raise RuntimeError("unbounded real-only arm must not use a local candidate cap")
    proxy = qwen3_vl_shallow_pruned_compute_proxy(
        full_patch_tokens,
        int(vision.image_embeds.shape[0]),
        prune_after_layer=PRUNE_AFTER_LAYER,
    )
    full_proxy = qwen3_vl_compute_proxy(full_patch_tokens)
    selected = torch.as_tensor(vision.group_indices, dtype=torch.long)
    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "score_mode": score_mode,
        "prune_after_layer": int(vision.prune_after_layer),
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "uses_b1_trained_wv": False,
        # --- all-real / no-context declaration ---------------------------------
        "arm_label": ARM_LABEL,
        "selection_rule": SELECTION_RULE,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "local_candidate_cap": None,
        "uses_pooled_context_group_values": False,
        # --- frozen runner fields ------------------------------------------
        "output_text": raw_text,
        "raw_output_text": raw_text,
        "image_open_decode_seconds": image_open_decode_seconds,
        "processor_seconds": processor_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "prompt_text_embedding_seconds": 0.0,
        "vision_preamble_seconds": vision.vision_preamble_seconds,
        "block0_seconds": vision.cut_blocks_seconds,
        "cut_blocks_seconds": vision.cut_blocks_seconds,
        "selector_cpu_roundtrip_seconds": vision.selector_cpu_seconds,
        "remaining_vision_seconds": vision.remaining_vision_seconds,
        "vision_total_seconds": vision_total_seconds,
        "vision_encoder_seconds": vision_total_seconds,
        "gpu_vision_seconds": gpu_vision_seconds,
        "llm_prepare_seconds": llm_prepare_seconds,
        "llm_prefill_seconds": prefill_seconds,
        "decoder_seconds": decoder_seconds,
        "postprocess_seconds": postprocess_seconds,
        "model_inference_seconds": model_inference_seconds,
        "application_end_to_end_seconds": application_end_to_end_seconds,
        "elapsed_seconds": model_inference_seconds,
        "timed_stage_seconds": model_inference_seconds,
        "deprecated_alias": {
            "elapsed_seconds": "model_inference_seconds",
            "timed_stage_seconds": "model_inference_seconds",
            "vision_encoder_seconds": "vision_total_seconds",
        },
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        "selector_cpu_roundtrip_excluded_from_gpu_vision": True,
        "do_sample": False,
        "decoding": "greedy_argmax",
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "output_group_count": int(vision.image_embeds.shape[0]),
        "context_group_count": vision.context_group_count,
        "real_group_count": vision.real_group_count,
        "source_types_all_real": bool(
            np.all(np.asarray(vision.source_type) == 0)
        ),
        "identity_bypass": vision.identity_bypass,
        "requested_budget": budget,
        "block_input_shapes": vision.block_input_shapes,
        "input_tokens": int(prepared.input_ids.shape[1]),
        "original_input_tokens": int(prepared.original_sequence_length),
        "output_tokens": output_tokens,
        "selected_group_indices": [int(value) for value in selected.tolist()],
        "selected_group_indices_sha256": selection_sha256(selected),
        "source_type": [int(value) for value in vision.source_type.tolist()],
        "region_id": [int(value) for value in vision.region_id.tolist()],
        "valid": [bool(value) for value in vision.valid.tolist()],
        "deepstack_cut_merger_index": vision.deepstack_cut_merger_index,
        "deepstack_merger_indexes": list(vision.deepstack_merger_indexes),
        "llm_visual_tokens": proxy.llm_visual_tokens,
        "baseline_llm_visual_tokens": full_llm_visual_tokens,
        "encoder_tokens_after_prune": proxy.encoder_tokens,
        "encoder_macs_proxy": proxy.encoder_macs,
        "baseline_encoder_macs_proxy": full_proxy.encoder_macs,
        "llm_prefill_macs_proxy": proxy.llm_prefill_macs,
        "baseline_llm_prefill_macs_proxy": full_proxy.llm_prefill_macs,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    del batch, vision, prepared, generated, image
    gc.collect()
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-mode",
        choices=(SCORE_MODE_L2,),
        default=SCORE_MODE_L2,
        help="Only 'l2' is supported; training and every other score mode are banned.",
    )
    parser.add_argument(
        "--prune-after-layer",
        type=int,
        choices=(PRUNE_AFTER_LAYER,),
        default=PRUNE_AFTER_LAYER,
        help="Fixed cut point after vision block 5.",
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--region-grid", type=int, default=3)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.score_mode != SCORE_MODE_L2:
        parser.error(f"only --score-mode {SCORE_MODE_L2} is supported")
    if args.prune_after_layer != PRUNE_AFTER_LAYER:
        parser.error(f"only --prune-after-layer {PRUNE_AFTER_LAYER} is supported")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")

    samples = limit_each_task(
        [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        args.limit_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = completed_ids(args.output)
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(
        f"LOAD arm={ARM_LABEL} selection_rule={SELECTION_RULE} "
        f"score_mode={args.score_mode} prune_after_layer={args.prune_after_layer} "
        f"samples={len(samples)} budget={args.budget} pending={len(pending)}",
        flush=True,
    )
    if not pending:
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
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "score_mode": args.score_mode,
        "prune_after_layer": args.prune_after_layer,
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": None,
        "arm_label": ARM_LABEL,
        "selection_rule": SELECTION_RULE,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "projection_npz": None,
        "training": False,
        "do_sample": False,
        "custom_greedy_decode": True,
        "original_position_ids_preserved": True,
        "baseline_representative_jsonl": BASELINE_REPRESENTATIVE_JSONL,
        "single_variable_contract": (
            "same screening100 / K=96 / layer5 / region_grid=3 / BF16 / SDPA / "
            "greedy; only the selection constraint/indexing changes: this arm is "
            "global-L2 + 3x3 anchor real-only with no context pooling, no context "
            "slot reservation and no local candidate cap"
        ),
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        "timing_contract": {
            "vision_total_seconds": (
                "vision_preamble_seconds + cut_blocks_seconds + "
                "selector_cpu_roundtrip_seconds + remaining_vision_seconds"
            ),
            "gpu_vision_seconds": "vision_total_seconds - selector_cpu_roundtrip_seconds",
            "model_inference_seconds": (
                "vision_total_seconds + llm_prepare_seconds + llm_prefill_seconds "
                "+ decoder_seconds"
            ),
            "application_end_to_end_seconds": (
                "image_open_decode + processor + host_to_device + "
                "prompt_text_embedding + model_inference + postprocess"
            ),
            "gpu_boundaries_synchronized": True,
            "selector_cpu_roundtrip_is_gpu_compute": False,
        },
    }
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for sample_number, sample in enumerate(pending, 1):
            record = run_one(
                model,
                processor,
                sample,
                args.manifest,
                budget=args.budget,
                score_mode=args.score_mode,
                region_grid=args.region_grid,
            )
            record["run"] = metadata
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                f"G={record['full_llm_visual_tokens']} K={record['output_group_count']} "
                f"C={record['context_group_count']} "
                f"bypass={record['identity_bypass']} "
                f"preamble={record['vision_preamble_seconds']:.4f}s "
                f"cut_blocks={record['cut_blocks_seconds']:.4f}s "
                f"selector_cpu={record['selector_cpu_roundtrip_seconds']:.4f}s "
                f"remaining={record['remaining_vision_seconds']:.4f}s "
                f"vision={record['vision_total_seconds']:.4f}s "
                f"prepare={record['llm_prepare_seconds']:.4f}s "
                f"prefill={record['llm_prefill_seconds']:.4f}s "
                f"decode={record['decoder_seconds']:.4f}s "
                f"model={record['model_inference_seconds']:.4f}s "
                f"app_e2e={record['application_end_to_end_seconds']:.4f}s "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
