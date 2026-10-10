#!/usr/bin/env python3
"""GPU runner for the conservative L2-residual reranker (development arm).

One frozen arm only: ``conservative_l2_residual_rerank_v1``.  It reuses the
frozen corrected-L2 B1 runner's provenance/resume machinery and the accepted
integer semantic + corrected integer patch-L2 chain, but drives the additive
:mod:`evaluation.hardware_regular_b1_residual_rerank` integration so that
selection ranking (signed residual) stays separate from context pooling
(corrected integer-L2).

Fixed contract
--------------
* Local RoboBrain2.5-4B, BF16 / SDPA, offline (``HF_HUB_OFFLINE=1`` /
  ``TRANSFORMERS_OFFLINE=1``), deterministic greedy decode, ``max_new_tokens=16``.
* B1 = after vision block 0, ``K=96``, ``region_grid=3``,
  ``local_candidate_cap=16``, ``S=16``.
* ``torch.cuda.synchronize`` brackets every timed GPU segment; model-load time
  is never part of the per-sample steady timing.
* Append-only raw per-sample JSONL with fail-closed resume.
* ``--smoke`` runs the deterministic six-cell contract smoke (one sample per
  canonical source x task cell) and refuses to write to the preregistered
  full2638 path.

No training, no tuning, no labels/answer tokens/gradients/floating lambda and
no task/source IDs enter the mechanism.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
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

from evaluation.conservative_l2_residual_reranker import (
    BUDGET,
    CONTEXT_POOLING_BASIS,
    LOCAL_CANDIDATE_CAP,
    MAX_SWAPS,
    MECHANISM,
    RESIDUAL_FORMULA,
    array_sha256,
    build_residual_scorer,
)
from evaluation.hardware_regular_b1_integration import (
    prepare_b1_multimodal_inputs,
)
from evaluation.hardware_regular_b1_residual_rerank import (
    forward_conservative_residual_rerank_b1_vision,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT
from evaluation.run_hf_hardware_regular_b1_corrected_l2 import (
    _sync_clock,
    compute_text_embeddings,
    file_sha256,
    record_sha256,
    validate_resume,
)
from evaluation.run_hf_shallow_block_pruning import (
    greedy_generate_from_pruned,
    limit_each_task,
)
from evaluation.run_hf_token_reduction import (
    make_messages,
    processor_inputs,
    resolve_image,
)

DEFAULT_REGION_GRID = 3

#: One sample per canonical source x task cell for the contract-only smoke.
SMOKE_CELLS: tuple[tuple[str, str], ...] = (
    ("ADE20K", "Count"),
    ("ADE20K", "Relation"),
    ("COCO", "Count"),
    ("COCO", "Relation"),
    ("Omni3D", "Depth"),
    ("Omni3D", "Distance"),
)

#: The preregistered development output.  ``--smoke`` must never write here.
FULL2638_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "token_selector"
    / "2026-10-05-conservative-l2-residual-reranker-full2638"
    / "conservative_l2_residual_rerank_v1_k96_full2638.jsonl"
)

#: Extra audit fields every mechanism record must carry.
REQUIRED_RESIDUAL_RECORD_FIELDS = (
    "mechanism",
    "run_mode",
    "residual_formula",
    "max_swaps",
    "swap_count",
    "context_pooling_basis",
    "baseline_anchors",
    "removed_group_indices",
    "added_group_indices",
    "swap_region_ids",
    "swap_gain_q15",
    "per_region_selected_counts_before",
    "per_region_selected_counts_after",
    "uses_labels",
    "uses_answer_tokens",
    "uses_gradients",
    "uses_floating_lambda",
    "uses_learned_new_weights",
    "uses_task_source_ids",
)


class ResidualResumeContractError(RuntimeError):
    """Raised when an existing residual-reranker JSONL violates the contract."""


def load_residual_checkpoint() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the frozen seed-20261003 ``Wt`` / ``Wv_q`` bundle with hash checks."""

    from evaluation.current_selector_integer import (  # noqa: PLC0415
        load_deployment_projection,
        load_deployment_wv_bundle,
    )

    projection, metadata = load_deployment_projection()
    bundle = load_deployment_wv_bundle()
    Wt = np.asarray(projection.Wt)
    wv_q = np.asarray(bundle["wv_q"], dtype=np.int64)
    if Wt.shape != (128, 2560):
        raise ValueError(f"unexpected Wt shape {Wt.shape}")
    if wv_q.shape != (128, 1024):
        raise ValueError(f"unexpected Wv_q shape {wv_q.shape}")
    return Wt, wv_q, {
        "checkpoint_sha256": str(metadata["sha256"]),
        "wv_sha256": str(bundle["wv_sha256"]),
        "wv_scale_sha256": str(bundle["scale_sha256"]),
        "deployment_seed": int(metadata["seed"]),
        "projection_npz": str(metadata["path"]),
    }


def select_smoke_samples(
    samples: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically pick one sample per canonical source x task cell.

    Within a cell the sample whose ``sha256(sample_id)`` is smallest wins; ties
    break on the string sample id.  Cells absent from the manifest are skipped.
    """

    chosen: list[dict[str, Any]] = []
    for source, task in SMOKE_CELLS:
        cell = [
            sample
            for sample in samples
            if str(sample.get("source")) == source and str(sample.get("task")) == task
        ]
        if not cell:
            continue
        chosen.append(
            min(
                cell,
                key=lambda sample: (
                    hashlib.sha256(str(sample["sample_id"]).encode("utf-8")).hexdigest(),
                    str(sample["sample_id"]),
                ),
            )
        )
    return chosen


def _residual_audit_fields(payload: dict[str, Any]) -> dict[str, Any]:
    bypass = bool(payload.get("bypass", True))
    plan = payload.get("baseline_plan")
    if bypass or not isinstance(plan, dict):
        return {
            "mechanism": MECHANISM,
            "residual_formula": RESIDUAL_FORMULA,
            "max_swaps": int(MAX_SWAPS),
            "swap_count": 0,
            "context_pooling_basis": CONTEXT_POOLING_BASIS,
            "baseline_anchors": [],
            "baseline_selected_real": [],
            "reranked_selected_real": [],
            "removed_group_indices": [],
            "added_group_indices": [],
            "swap_region_ids": [],
            "swap_residual_removed_q15": [],
            "swap_residual_added_q15": [],
            "swap_gain_q15": [],
            "per_region_selected_counts_before": [],
            "per_region_selected_counts_after": [],
            "residual_q15_sha256": None,
            "semantic_norm_q15_sha256": None,
            "l2_norm_q15_sha256": None,
            "l2_combined_sha256": None,
            "baseline_selected_real_sha256": None,
            "reranked_selected_real_sha256": None,
        }
    return {
        "mechanism": MECHANISM,
        "residual_formula": RESIDUAL_FORMULA,
        "max_swaps": int(plan["max_swaps"]),
        "swap_count": int(plan["swap_count"]),
        "context_pooling_basis": CONTEXT_POOLING_BASIS,
        "baseline_anchors": [int(v) for v in plan["baseline_anchors"]],
        "baseline_selected_real": [int(v) for v in plan["baseline_selected_real"]],
        "reranked_selected_real": [int(v) for v in plan["reranked_selected_real"]],
        "removed_group_indices": [int(v) for v in plan["removed_group_indices"]],
        "added_group_indices": [int(v) for v in plan["added_group_indices"]],
        "swap_region_ids": [int(v) for v in plan["swap_region_ids"]],
        "swap_residual_removed_q15": [
            int(v) for v in plan["swap_residual_removed_q15"]
        ],
        "swap_residual_added_q15": [
            int(v) for v in plan["swap_residual_added_q15"]
        ],
        "swap_gain_q15": [int(v) for v in plan["swap_gain_q15"]],
        "per_region_selected_counts_before": [
            int(v) for v in plan["per_region_selected_counts_before"]
        ],
        "per_region_selected_counts_after": [
            int(v) for v in plan["per_region_selected_counts_after"]
        ],
        "residual_q15_sha256": array_sha256(
            np.asarray(payload["residual_q15"], dtype=np.int64)
        ),
        "semantic_norm_q15_sha256": array_sha256(
            np.asarray(payload["semantic_norm_q15"], dtype=np.int64)
        ),
        "l2_norm_q15_sha256": array_sha256(
            np.asarray(payload["l2_norm_q15"], dtype=np.int64)
        ),
        "l2_combined_sha256": array_sha256(
            np.asarray(payload["l2_combined"], dtype=np.float64)
        ),
        "baseline_selected_real_sha256": array_sha256(
            np.asarray(plan["baseline_selected_real"], dtype=np.int64)
        ),
        "reranked_selected_real_sha256": array_sha256(
            np.asarray(plan["reranked_selected_real"], dtype=np.int64)
        ),
    }


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    budget: int,
    region_grid: int,
    local_candidate_cap: int,
    max_swaps: int,
    scorer: Any,
    manifest_sha256: str,
    checkpoint_identity: dict[str, Any],
    run_mode: str,
) -> dict[str, Any]:
    """Run one sample through the additive residual-rerank B1 forward."""

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

    prompt_text_embedding_started = _sync_clock()
    text_embeddings = compute_text_embeddings(model, batch)
    prompt_text_embedding_seconds = _sync_clock() - prompt_text_embedding_started

    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    with torch.inference_mode():
        vision = forward_conservative_residual_rerank_b1_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            text_embeddings=text_embeddings,
            residual_scorer=scorer,
            budget=budget,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
            max_swaps=max_swaps,
            synchronize=torch.cuda.synchronize,
        )
        llm_prepare_started = _sync_clock()
        prepared = prepare_b1_multimodal_inputs(model, batch, vision)
        llm_prepare_seconds = _sync_clock() - llm_prepare_started
        generated, prefill_seconds, decoder_seconds, output_tokens = (
            greedy_generate_from_pruned(
                model, prepared, int(sample["max_new_tokens"])
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

    payload = vision.integer_payload or {}
    audit = _residual_audit_fields(payload)
    selected = np.asarray(vision.group_indices, dtype=np.int64)
    patch_indices = np.asarray(vision.patch_indices, dtype=np.int64).reshape(-1)
    source_type = np.asarray(vision.source_type, dtype=np.int64)
    region_id = np.asarray(vision.region_id, dtype=np.int64)
    valid = np.asarray(vision.valid, dtype=bool)
    combined_scores = (
        np.asarray(payload["combined_scores"], dtype=np.float64)
        if payload.get("combined_scores") is not None
        else None
    )

    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "arm": MECHANISM,
        "mechanism": MECHANISM,
        "run_mode": run_mode,
        "score_mode": "integer",
        "integration_score_mode": "integer",
        "prune_after_layer": 0,
        "cutpoint": "B1_after_vision_block_0",
        "output_text": raw_text,
        "raw_output_text": raw_text,
        "output_token_ids": [int(value) for value in generated[0].tolist()],
        "output_tokens": output_tokens,
        "image_open_decode_seconds": image_open_decode_seconds,
        "processor_seconds": processor_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "prompt_text_embedding_seconds": prompt_text_embedding_seconds,
        "vision_preamble_seconds": vision.vision_preamble_seconds,
        "block0_seconds": vision.block0_seconds,
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
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "max_new_tokens": int(sample["max_new_tokens"]),
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "output_group_count": int(vision.image_embeds.shape[0]),
        "context_group_count": vision.context_group_count,
        "real_group_count": vision.real_group_count,
        "identity_bypass": bool(vision.selection.bypass),
        "requested_budget": budget,
        "region_grid": region_grid,
        "local_candidate_cap": local_candidate_cap,
        "block_input_shapes": vision.block_input_shapes,
        "input_tokens": int(prepared.input_ids.shape[1]),
        "original_input_tokens": int(prepared.original_sequence_length),
        "selected_group_indices": [int(value) for value in selected.tolist()],
        "selected_group_indices_sha256": array_sha256(selected),
        "selected_patch_indices": [int(value) for value in patch_indices.tolist()],
        "selected_patch_indices_sha256": array_sha256(patch_indices),
        "source_type": [int(value) for value in source_type.tolist()],
        "source_type_sha256": array_sha256(source_type),
        "region_id": [int(value) for value in region_id.tolist()],
        "region_id_sha256": array_sha256(region_id),
        "valid": [bool(value) for value in valid.tolist()],
        "valid_sha256": array_sha256(valid),
        "combined_scores_sha256": (
            array_sha256(combined_scores) if combined_scores is not None else None
        ),
        "score_vector_length": (
            int(combined_scores.size) if combined_scores is not None else None
        ),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "wv_sha256": checkpoint_identity["wv_sha256"],
        "wv_scale_sha256": checkpoint_identity["wv_scale_sha256"],
        "deployment_seed": checkpoint_identity["deployment_seed"],
        "projection_npz": checkpoint_identity["projection_npz"],
        "manifest_sha256": manifest_sha256,
        "selector_software_is_acceleration_claim": False,
        "frozen_anchors": True,
        "same_region_one_for_one_swaps": True,
        "uses_labels": False,
        "uses_answer_tokens": False,
        "uses_gradients": False,
        "uses_floating_lambda": False,
        "uses_learned_new_weights": False,
        "uses_task_source_ids": False,
        "score_formula": RESIDUAL_FORMULA,
        **audit,
    }
    record["record_sha256"] = record_sha256(record)
    del batch, vision, prepared, generated, image
    gc.collect()
    return record


def validate_residual_resume_extra(
    path: Path,
    *,
    max_swaps: int,
    arm: str = MECHANISM,
) -> set[str]:
    """Fail-closed extra checks on an existing residual-reranker JSONL."""

    if not path.is_file() or path.stat().st_size == 0:
        return set()
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResidualResumeContractError(
                f"output line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        missing = [
            field for field in REQUIRED_RESIDUAL_RECORD_FIELDS if field not in record
        ]
        if missing:
            raise ResidualResumeContractError(
                f"output line {line_number} is missing residual fields {missing} "
                f"(fail closed)"
            )
        sample_id = str(record["sample_id"])
        if str(record.get("arm")) != arm or str(record.get("mechanism")) != arm:
            raise ResidualResumeContractError(
                f"output sample {sample_id!r} has arm/mechanism "
                f"{record.get('arm')!r}/{record.get('mechanism')!r} != {arm!r} "
                f"(fail closed)"
            )
        if int(record.get("max_swaps", -1)) != int(max_swaps):
            raise ResidualResumeContractError(
                f"output sample {sample_id!r} max_swaps "
                f"{record.get('max_swaps')!r} != {max_swaps} (fail closed)"
            )
        if int(record.get("swap_count", -1)) > int(max_swaps):
            raise ResidualResumeContractError(
                f"output sample {sample_id!r} swap_count exceeds max_swaps "
                f"(fail closed)"
            )
        for flag in (
            "uses_labels",
            "uses_answer_tokens",
            "uses_gradients",
            "uses_floating_lambda",
            "uses_learned_new_weights",
            "uses_task_source_ids",
        ):
            if bool(record.get(flag)):
                raise ResidualResumeContractError(
                    f"output sample {sample_id!r} declares forbidden {flag}=True "
                    f"(fail closed)"
                )
        done.add(sample_id)
    return done


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--region-grid", type=int, default=DEFAULT_REGION_GRID)
    parser.add_argument(
        "--local-candidate-cap", type=int, default=LOCAL_CANDIDATE_CAP
    )
    parser.add_argument("--max-swaps", type=int, default=MAX_SWAPS)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the deterministic six-cell contract smoke; never decides quality.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.budget != BUDGET:
        parser.error(f"--budget is frozen at {BUDGET}")
    if args.region_grid != DEFAULT_REGION_GRID:
        parser.error(f"--region-grid is frozen at {DEFAULT_REGION_GRID}")
    if args.local_candidate_cap != LOCAL_CANDIDATE_CAP:
        parser.error(f"--local-candidate-cap is frozen at {LOCAL_CANDIDATE_CAP}")
    if args.max_swaps != MAX_SWAPS:
        parser.error(f"--max-swaps is frozen at {MAX_SWAPS}")
    if args.smoke:
        try:
            same = args.output.resolve() == FULL2638_OUTPUT.resolve()
        except OSError:
            same = False
        if same:
            parser.error(
                "--smoke must not write to the preregistered full2638 output path"
            )
    if not args.model.is_dir():
        raise RuntimeError(
            f"--model must be a local directory (strict offline): {args.model}"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")

    all_samples = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.smoke:
        samples = select_smoke_samples(all_samples)
    else:
        samples = limit_each_task(all_samples, args.limit_per_task)
    run_mode = "smoke" if args.smoke else "full2638"

    manifest_sha256 = file_sha256(args.manifest)
    Wt, wv_q, checkpoint_identity = load_residual_checkpoint()
    scorer = build_residual_scorer(
        Wt,
        wv_q,
        budgets=(args.budget,),
        region_grid=args.region_grid,
        local_candidate_cap=args.local_candidate_cap,
        max_swaps=args.max_swaps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = validate_resume(
        args.output,
        samples,
        arm=MECHANISM,
        manifest_sha256=manifest_sha256,
        checkpoint_identity=checkpoint_identity,
        budget=args.budget,
        region_grid=args.region_grid,
        local_candidate_cap=args.local_candidate_cap,
    )
    validate_residual_resume_extra(args.output, max_swaps=args.max_swaps)
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(
        f"LOAD arm={MECHANISM} mode={run_mode} samples={len(samples)} "
        f"budget={args.budget} region_grid={args.region_grid} "
        f"local_cap={args.local_candidate_cap} max_swaps={args.max_swaps} "
        f"checkpoint={checkpoint_identity['checkpoint_sha256']} "
        f"done={len(done)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return 0

    dtype = {"bf16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)
    from transformers import (  # noqa: PLC0415
        AutoModelForImageTextToText,
        AutoProcessor,
    )

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
        "arm": MECHANISM,
        "mechanism": MECHANISM,
        "run_mode": run_mode,
        "integration": "evaluation.hardware_regular_b1_residual_rerank",
        "prune_after_layer": 0,
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": args.local_candidate_cap,
        "max_swaps": args.max_swaps,
        "residual_formula": RESIDUAL_FORMULA,
        "context_pooling_basis": CONTEXT_POOLING_BASIS,
        "manifest_sha256": manifest_sha256,
        "do_sample": False,
        "custom_greedy_decode": True,
        "original_position_ids_preserved": True,
        "training": False,
        "uses_labels": False,
        "uses_answer_tokens": False,
        "uses_gradients": False,
        "uses_floating_lambda": False,
        "uses_learned_new_weights": False,
        "uses_task_source_ids": False,
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        **checkpoint_identity,
        "timing_contract": {
            "vision_total_seconds": (
                "vision_preamble_seconds + block0_seconds + "
                "selector_cpu_roundtrip_seconds + remaining_vision_seconds"
            ),
            "gpu_vision_seconds": (
                "vision_total_seconds - selector_cpu_roundtrip_seconds"
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
                region_grid=args.region_grid,
                local_candidate_cap=args.local_candidate_cap,
                max_swaps=args.max_swaps,
                scorer=scorer,
                manifest_sha256=manifest_sha256,
                checkpoint_identity=checkpoint_identity,
                run_mode=run_mode,
            )
            record["run"] = metadata
            record["record_sha256"] = record_sha256(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                f"arm={MECHANISM} G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']} "
                f"C={record['context_group_count']} "
                f"swaps={record['swap_count']} "
                f"model={record['model_inference_seconds']:.4f}s "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
