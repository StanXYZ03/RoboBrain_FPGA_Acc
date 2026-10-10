#!/usr/bin/env python3
"""Offline local BF16+SDPA runner for the layer-5 score-formula replay.

This runner answers one narrow causal question on the frozen CV-Bench
``screening100`` set: holding the layer-5 cut point, the 3x3 anchor/global-fill
policy, the per-sample ``K_C`` and the unified vision forward fixed, does
changing only the per-group score aggregation from the current control formula
``norm(mean(x_0..x_3))`` to the historical ``mean(norm(x_0)..norm(x_3))``
reproduce the frozen ``shallow_r0250`` index sets?

Contract
--------
* the model and processor are loaded once, offline/local-only;
* only the candidate layer-5 **vision** path runs
  (:func:`evaluation.layer5_score_formula_replay.
  forward_layer5_score_formula_replay_vision`); there is no LLM decode;
* for every sample ``K_C = len(reference.selected_group_indices)`` and ``G`` is
  the frozen reference ``baseline_llm_visual_tokens``;
* the *frozen control arm* is an input: the runner recomputes the control
  selection from the same hidden state and fails closed if it differs from the
  pinned control JSONL;
* manifest / reference / control are cross-validated fail closed before
  ``--limit-per-task`` is applied;
* an existing ``--output`` is revalidated fail closed (provenance hashes,
  diagnostic flags, index hashes, jaccards) before its ``sample_id`` set is used
  for resume: resume never silently drops or mixes work.

This runner is executed on a GPU host only; it is CPU-import-safe (no CUDA call
at import time) and never used by the harness.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.layer5_score_formula_replay import (
    ARM_LABEL,
    CAUSAL_DIAGNOSTIC_ONLY,
    CUTPOINT_PREFIX_BLOCKS,
    DEFAULT_REPLAY_JSONL,
    EXPECTED_SAMPLES,
    FORMULA_CANDIDATE,
    FORMULA_CONTROL,
    FPGA_DEPLOYABLE_CANDIDATE,
    HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC,
    MODE_CANDIDATE,
    ONLY_VARIABLE,
    REFERENCE_SELECTOR,
    SELECTION_RULE_CANDIDATE,
    SELECTION_RULE_CONTROL,
    ReplayInputs,
    diagnostic_flags,
    exact_index_match,
    forward_layer5_score_formula_replay_vision,
    indices_sha256,
    jaccard,
    load_replay_inputs,
    merged_grid_from_thw,
    resume_completed_ids,
    score_summary,
    validate_replay_record,
)
from evaluation.hardware_regular_layer5_forced_index_ablation import PRUNE_AFTER_LAYER
from evaluation.hardware_regular_token_selector import REGION_GRID
from evaluation.run_hf_shallow_block_pruning import limit_each_task
from evaluation.run_hf_token_reduction import (
    make_messages,
    processor_inputs,
    resolve_image,
)

ROOT = Path(__file__).resolve().parents[1]


def _sync_clock() -> float:
    """Drain the GPU then read the wall clock: a clean stage boundary."""
    torch.cuda.synchronize()
    return time.perf_counter()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument(
        "--control-jsonl",
        type=Path,
        required=True,
        help="Frozen control arm; the runner recomputes and fails closed on mismatch.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPLAY_JSONL)
    parser.add_argument("--region-grid", type=int, default=REGION_GRID)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def provenance_of(inputs: ReplayInputs) -> dict[str, str]:
    return {
        "manifest_sha256": inputs.manifest_sha256,
        "reference_jsonl_sha256": inputs.reference_sha256,
        "control_jsonl_sha256": inputs.control_sha256,
    }


def build_run_metadata(
    args: argparse.Namespace, *, provenance: dict[str, str]
) -> dict[str, Any]:
    return {
        "arm": ARM_LABEL,
        "mode": MODE_CANDIDATE,
        "selection_rule": SELECTION_RULE_CANDIDATE,
        "control_selection_rule": SELECTION_RULE_CONTROL,
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "attn_implementation": "sdpa",
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "region_grid": args.region_grid,
        "manifest": str(args.manifest),
        "reference_jsonl": str(args.reference_jsonl),
        "control_jsonl": str(args.control_jsonl),
        "reference_selector": REFERENCE_SELECTOR,
        "llm_decode": False,
        "vision_only_replay": True,
        "do_sample": False,
        "decoding": None,
        "local_candidate_cap": None,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "training": False,
        "K_C_is_per_sample_reference_length": True,
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        "timing_contract": {
            "vision_total_seconds": (
                "vision_preamble_seconds + cut_blocks_seconds + "
                "selector_cpu_roundtrip_seconds + remaining_vision_seconds"
            ),
            "gpu_vision_seconds": (
                "vision_total_seconds - selector_cpu_roundtrip_seconds"
            ),
            "model_inference_seconds": "vision_total_seconds (no LLM decode)",
            "gpu_boundaries_synchronized": True,
        },
        **provenance,
        **diagnostic_flags(),
    }


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    reference: dict[str, Any],
    control: dict[str, Any],
    *,
    region_grid: int,
    provenance: dict[str, str],
) -> dict[str, Any]:
    """Run the candidate vision path for one sample and build its record."""
    sample_id = str(sample["sample_id"])
    if str(reference["sample_id"]) != sample_id or str(control["sample_id"]) != sample_id:
        raise ValueError(f"input sample_id mismatch for {sample_id!r}")
    if str(reference["task"]) != str(sample["task"]):
        raise ValueError(
            f"reference task {reference['task']!r} != sample task {sample['task']!r}"
        )
    k_c = int(reference["K_C"])
    reference_g = int(reference["G"])
    reference_indices = [int(value) for value in reference["group_indices"]]
    frozen_control_indices = [int(value) for value in control["group_indices"]]

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
    merge_size = int(model.config.vision_config.spatial_merge_size)
    merge_unit = merge_size * merge_size
    if full_patch_tokens % merge_unit:
        raise ValueError("processor grid is not aligned to the visual merger")
    full_llm_visual_tokens = full_patch_tokens // merge_unit
    if full_llm_visual_tokens != reference_g:
        raise ValueError(
            f"G={full_llm_visual_tokens} != frozen reference G={reference_g}"
        )
    merged_h, merged_w = merged_grid_from_thw(
        batch.image_grid_thw, merge_size=merge_size
    )
    if merged_h * merged_w != full_llm_visual_tokens:
        raise ValueError("merged grid does not match the visual group count")

    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    with torch.inference_mode():
        replay = forward_layer5_score_formula_replay_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            budget=k_c,
            region_grid=region_grid,
            prune_after_layer=PRUNE_AFTER_LAYER,
            synchronize=torch.cuda.synchronize,
        )
    model_inference_seconds = _sync_clock() - model_inference_started
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated())

    forward = replay.forward
    candidate_indices = [int(value) for value in replay.candidate_selection.group_indices]
    recomputed_control_indices = [
        int(value) for value in replay.control_selection.group_indices
    ]
    if len(candidate_indices) != k_c or len(recomputed_control_indices) != k_c:
        raise RuntimeError(
            f"candidate/control did not emit exactly K_C={k_c} groups"
        )
    if int(forward.real_group_count) != k_c or int(forward.image_embeds.shape[0]) != k_c:
        raise RuntimeError(f"vision output groups != K_C={k_c}")
    if int(forward.full_llm_visual_tokens) != reference_g:
        raise RuntimeError("vision G != frozen reference G")
    if forward.identity_bypass:
        raise RuntimeError("unexpected identity bypass for a frozen reference sample")
    if recomputed_control_indices != frozen_control_indices:
        raise RuntimeError(
            "recomputed control selection differs from the pinned control arm "
            "(fail closed)"
        )

    vision_total_seconds = forward.vision_total_seconds
    gpu_vision_seconds = forward.gpu_vision_seconds
    if vision_total_seconds != (
        forward.vision_preamble_seconds
        + forward.cut_blocks_seconds
        + forward.selector_cpu_seconds
        + forward.remaining_vision_seconds
    ):
        raise RuntimeError("vision timing segments do not sum to the vision total")
    application_end_to_end_seconds = _sync_clock() - application_started

    candidate_equals_reference = exact_index_match(
        candidate_indices, reference_indices
    )
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "task": str(sample["task"]),
        "arm": ARM_LABEL,
        "mode": MODE_CANDIDATE,
        "selection_rule": SELECTION_RULE_CANDIDATE,
        "control_selection_rule": SELECTION_RULE_CONTROL,
        "prune_after_layer": int(forward.prune_after_layer),
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "region_grid": int(region_grid),
        "score_formula_control": FORMULA_CONTROL,
        "score_formula_candidate": FORMULA_CANDIDATE,
        # --- frozen reference / pinned control --------------------------------
        "K_C": k_c,
        "reference_K_C": k_c,
        "reference_G": reference_g,
        "reference_selected_group_indices": reference_indices,
        "reference_selected_group_indices_sha256": reference["group_indices_sha256"],
        "reference_selector": REFERENCE_SELECTOR,
        "candidate_selected_group_indices": candidate_indices,
        "candidate_selected_group_indices_sha256": indices_sha256(candidate_indices),
        "control_selected_group_indices": recomputed_control_indices,
        "control_selected_group_indices_sha256": indices_sha256(
            recomputed_control_indices
        ),
        "recomputed_control_equals_frozen_control": bool(
            recomputed_control_indices == frozen_control_indices
        ),
        "candidate_equals_reference": candidate_equals_reference,
        "candidate_vs_reference_exact": candidate_equals_reference,
        "candidate_vs_reference_jaccard": jaccard(
            candidate_indices, reference_indices
        ),
        "control_vs_reference_exact": exact_index_match(
            recomputed_control_indices, reference_indices
        ),
        "control_vs_reference_jaccard": jaccard(
            recomputed_control_indices, reference_indices
        ),
        "candidate_score_summary": score_summary(replay.candidate_scores),
        "control_score_summary": score_summary(replay.control_scores),
        # --- vision bookkeeping ----------------------------------------------
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "merged_grid": [int(merged_h), int(merged_w)],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "baseline_llm_visual_tokens": reference_g,
        "output_group_count": int(forward.real_group_count),
        "context_group_count": int(forward.context_group_count),
        "real_group_count": int(forward.real_group_count),
        "identity_bypass": bool(forward.identity_bypass),
        "source_types_all_real": bool(
            np.all(np.asarray(forward.source_type) == 0)
        ),
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "source_type": [int(value) for value in forward.source_type.tolist()],
        "region_id": [int(value) for value in forward.region_id.tolist()],
        "valid": [bool(value) for value in forward.valid.tolist()],
        "block_input_shapes": list(forward.block_input_shapes),
        "deepstack_cut_merger_index": forward.deepstack_cut_merger_index,
        "deepstack_merger_indexes": list(forward.deepstack_merger_indexes),
        # --- synchronized stage timings (descriptive only) --------------------
        "image_open_decode_seconds": image_open_decode_seconds,
        "processor_seconds": processor_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "vision_preamble_seconds": forward.vision_preamble_seconds,
        "cut_blocks_seconds": forward.cut_blocks_seconds,
        "selector_cpu_roundtrip_seconds": forward.selector_cpu_seconds,
        "remaining_vision_seconds": forward.remaining_vision_seconds,
        "vision_total_seconds": vision_total_seconds,
        "gpu_vision_seconds": gpu_vision_seconds,
        "model_inference_seconds": model_inference_seconds,
        "application_end_to_end_seconds": application_end_to_end_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
        "selector_cpu_roundtrip_is_acceleration_claim": False,
        "selector_cpu_roundtrip_excluded_from_gpu_vision": True,
        # --- provenance / flags -----------------------------------------------
        **provenance,
        **diagnostic_flags(),
    }
    del batch, replay, forward, image, processed
    gc.collect()
    return record


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this diagnostic experiment requires a CUDA device")

    inputs = load_replay_inputs(
        args.manifest,
        args.reference_jsonl,
        args.control_jsonl,
        expected_samples=EXPECTED_SAMPLES,
    )
    provenance = provenance_of(inputs)
    samples = limit_each_task(list(inputs.manifest_rows), args.limit_per_task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = resume_completed_ids(
        args.output,
        [str(sample["sample_id"]) for sample in samples],
        inputs=inputs,
        provenance=provenance,
    )
    pending = [sample for sample in samples if str(sample["sample_id"]) not in done]
    print(
        "LOAD "
        f"samples={len(samples)} pending={len(pending)} region_grid={args.region_grid} "
        f"reference={args.reference_jsonl} control={args.control_jsonl}",
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
    metadata = build_run_metadata(args, provenance=provenance)
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for sample_number, sample in enumerate(pending, 1):
            sample_id = str(sample["sample_id"])
            record = run_one(
                model,
                processor,
                sample,
                args.manifest,
                inputs.reference_map[sample_id],
                inputs.control_map[sample_id],
                region_grid=args.region_grid,
                provenance=provenance,
            )
            # Every written record is revalidated against the contract.
            validate_replay_record(
                record,
                inputs.reference_map[sample_id],
                inputs.control_map[sample_id],
                provenance=provenance,
            )
            record["run"] = metadata
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample_id} "
                f"G={record['full_llm_visual_tokens']} K={record['K_C']} "
                f"candidate_eq_reference={record['candidate_equals_reference']} "
                f"cand_jaccard={record['candidate_vs_reference_jaccard']:.4f} "
                f"ctrl_jaccard={record['control_vs_reference_jaccard']:.4f} "
                f"vision={record['vision_total_seconds']:.4f}s "
                f"peak_mem={record['peak_cuda_bytes']}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
