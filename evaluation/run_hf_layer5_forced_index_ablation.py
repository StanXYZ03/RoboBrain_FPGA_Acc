#!/usr/bin/env python3
"""Run the layer-5 forced-index / control diagnostic ablation, one arm at a time.

This is the host runner for the minimal causal experiment identified by the
2026-10-04 ``adaptive_vs_r25`` diagnostic: one implementation, one budget, and
the *only* experimental variable is the resolved
``selected_group_indices``.  Each CLI invocation runs exactly one arm and
writes exactly one ``--output`` JSONL; callers choose a distinct output path per
arm so the two arms are never mixed in one file.

Contract
--------
* the vision forward is **only**
  :func:`evaluation.hardware_regular_layer5_forced_index_ablation.
  forward_hardware_regular_layer5_forced_index_ablation_vision`
  (``control`` and ``forced`` share that one unified forward);
* the old module is reused for exactly one thing: the common LLM input assembly
  :func:`evaluation.hardware_regular_layer5_unbounded_l2.
  prepare_layer5_unbounded_multimodal_inputs`.  The old *vision* forward is
  never imported or called;
* ``--reference-jsonl`` defaults to the frozen historical ``shallow_r0250``
  artifact and is mapped uniquely by ``sample_id``;
* for every sample ``K_C = len(selected_group_indices)`` of the historical
  record; both arms use that *same* ``K_C``;
* ``control`` passes only ``K_C``; ``forced`` additionally passes the historical
  indices.  There is deliberately **no** ``--budget`` knob, no projection knob
  and no local-candidate-cap knob;
* manifest/reference sample sets, task labels, the one-dimensional integer /
  unique / ascending-or-normalized-hash indices, ``K_C > 0`` and the index range
  (from the historical ``baseline_llm_visual_tokens``) are validated fail-closed
  **before** ``--limit-per-task`` is applied; missing, duplicate and extra
  reference records all abort;
* after the forward, ``G`` must equal the historical baseline and ``K`` must
  equal ``K_C``, otherwise the sample aborts;
* BF16 + SDPA + greedy, every GPU timing boundary is synchronized, the model is
  loaded offline/local-only; an existing ``--output`` is revalidated fail-closed
  (arm / input hashes / diagnostic flags / unique in-range ids) before its
  ``sample_id`` set is used for resume.

This runner is only executed on a GPU host; it is CPU-import-safe (no CUDA call
at import time).
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

from evaluation.hardware_regular_layer5_forced_index_ablation import (
    CAUSAL_DIAGNOSTIC_ONLY,
    FPGA_DEPLOYABLE_CANDIDATE,
    HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC,
    MODE_CONTROL,
    MODE_FORCED,
    PRUNE_AFTER_LAYER,
    SELECTION_RULE_CONTROL,
    SELECTION_RULE_FORCED,
    forward_hardware_regular_layer5_forced_index_ablation_vision,
    validate_diagnostic_budget,
    verify_diagnostic_selection,
)
from evaluation.hardware_regular_layer5_unbounded_l2 import (
    prepare_layer5_unbounded_multimodal_inputs,
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
from evaluation.token_reduction import (
    qwen3_vl_compute_proxy,
    qwen3_vl_shallow_pruned_compute_proxy,
)

ROOT = Path(__file__).resolve().parents[1]

#: Re-exported for the frozen runner contract and tests.
CUTPOINT_PREFIX_BLOCKS = tuple(range(0, PRUNE_AFTER_LAYER + 1))

#: The two arms.  The only difference between them is the resolved indices.
ARM_CONTROL = "control"
ARM_FORCED = "forced"
ALLOWED_ARMS = (ARM_CONTROL, ARM_FORCED)

#: The single experimental variable, spelled out in every record.
ONLY_VARIABLE = "selected_group_indices"
SINGLE_VARIABLE_CONTRACT = (
    "唯一变量=selected_group_indices：同一实现（"
    "evaluation/hardware_regular_layer5_forced_index_ablation.py 的 layer-5 "
    "cut=5 统一 forward）、每个样本同一 K_C=len(历史 selected_group_indices)、"
    "同一 region_grid / BF16 / SDPA / greedy；control 仅传 K_C，forced 额外传"
    "历史 selected_group_indices，其余输入完全一致。"
)

#: Frozen historical reference (mapped uniquely by ``sample_id``).
REFERENCE_SELECTOR = "shallow_r0250"
REFERENCE_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_budget64_pilot100_20260927"
    / "shallow_r0250.jsonl"
)


class ReferenceContractError(RuntimeError):
    """Raised when the frozen historical reference JSONL violates the contract."""


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indices_sha256(indices: Sequence[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def _sync_clock() -> float:
    """Drain the GPU then read the wall clock: a clean stage boundary."""
    torch.cuda.synchronize()
    return time.perf_counter()


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceContractError(f"{label} must be an int, got {value!r}")
    return int(value)


def diagnostic_flags() -> dict[str, Any]:
    """Frozen diagnostic metadata written into every output and run block."""
    return {
        "hardware_alignment_relaxed_for_diagnostic": (
            HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC
        ),
        "causal_diagnostic_only": CAUSAL_DIAGNOSTIC_ONLY,
        "fpga_deployable_candidate": FPGA_DEPLOYABLE_CANDIDATE,
        "only_variable": ONLY_VARIABLE,
        "single_variable_contract": SINGLE_VARIABLE_CONTRACT,
    }


# ---------------------------------------------------------------------------
# Frozen historical reference contract
# ---------------------------------------------------------------------------


def normalize_reference_indices(
    raw: Any,
    *,
    group_count: int,
    recorded_sha: Any,
    label: str,
) -> list[int]:
    """Validate the historical indices: 1-D integer, unique, range, normalized hash.

    The accepted forms are (a) already strictly ascending or (b) any order whose
    **normalized** (ascending) SHA-256 matches ``selected_group_indices_sha256``.
    """
    if not isinstance(raw, list) or not raw:
        raise ReferenceContractError(
            f"{label}.selected_group_indices must be a non-empty JSON list"
        )
    values = [_as_int(value, f"{label}.selected_group_indices") for value in raw]
    if len(set(values)) != len(values):
        raise ReferenceContractError(
            f"{label}.selected_group_indices must be unique"
        )
    if any(value < 0 for value in values):
        raise ReferenceContractError(
            f"{label}.selected_group_indices must be non-negative"
        )
    normalized = sorted(values)
    if normalized[-1] >= int(group_count):
        raise ReferenceContractError(
            f"{label}.selected_group_indices must lie in [0, "
            f"{int(group_count)}), got max={normalized[-1]}"
        )
    recomputed = indices_sha256(normalized)
    if not isinstance(recorded_sha, str) or recorded_sha != recomputed:
        raise ReferenceContractError(
            f"{label}.selected_group_indices_sha256 mismatch: "
            f"recorded={recorded_sha!r} recomputed={recomputed}"
        )
    return normalized


def validate_reference_record(
    row: dict[str, Any], manifest_row: dict[str, Any]
) -> dict[str, Any]:
    """Validate one historical record against the manifest and normalize it."""
    sample_id = str(row.get("sample_id"))
    label = f"reference/{sample_id}"
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise ReferenceContractError(
            f"{label} task {row.get('task')!r} != manifest "
            f"{manifest_row.get('task')!r}"
        )
    prune_after_layer = row.get("prune_after_layer")
    if (
        isinstance(prune_after_layer, bool)
        or not isinstance(prune_after_layer, int)
        or int(prune_after_layer) != PRUNE_AFTER_LAYER
    ):
        raise ReferenceContractError(
            f"{label} prune_after_layer must be {PRUNE_AFTER_LAYER}, "
            f"got {prune_after_layer!r}"
        )
    group_count = _as_int(
        row.get("baseline_llm_visual_tokens"),
        f"{label}.baseline_llm_visual_tokens",
    )
    if group_count <= 0:
        raise ReferenceContractError(
            f"{label}.baseline_llm_visual_tokens must be positive"
        )
    indices = normalize_reference_indices(
        row.get("selected_group_indices"),
        group_count=group_count,
        recorded_sha=row.get("selected_group_indices_sha256"),
        label=label,
    )
    k_c = len(indices)
    if k_c <= 0:  # pragma: no cover - guarded by the non-empty check above
        raise ReferenceContractError(f"{label} K_C must be positive")
    recorded_k = _as_int(
        row.get("llm_visual_tokens"), f"{label}.llm_visual_tokens"
    )
    if recorded_k != k_c:
        raise ReferenceContractError(
            f"{label}.llm_visual_tokens={recorded_k} != K_C={k_c}"
        )
    return {
        "sample_id": sample_id,
        "task": str(manifest_row.get("task")),
        "G": group_count,
        "K_C": k_c,
        "group_indices": indices,
        "group_indices_sha256": indices_sha256(indices),
        "llm_visual_tokens": recorded_k,
    }


def load_reference_records(
    path: Path, manifest_rows: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Load the full historical reference mapping fail-closed.

    The mapping is validated against the **full** manifest before any
    ``--limit-per-task`` filtering.  Missing, duplicate and extra records all
    abort.
    """
    if not path.is_file():
        raise ReferenceContractError(
            f"missing required reference JSONL: {path} (fail closed)"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReferenceContractError(
                f"reference line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ReferenceContractError(
                f"reference line {line_number} must be a JSON object"
            )
        rows.append(payload)

    reference_index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if "sample_id" not in row:
            raise ReferenceContractError(
                f"reference row {position} is missing sample_id (fail closed)"
            )
        sample_id = str(row["sample_id"])
        if sample_id in reference_index:
            raise ReferenceContractError(
                f"duplicate reference sample_id {sample_id!r} (fail closed)"
            )
        reference_index[sample_id] = row

    manifest_index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(manifest_rows):
        if "sample_id" not in row:
            raise ReferenceContractError(
                f"manifest row {position} is missing sample_id"
            )
        sample_id = str(row["sample_id"])
        if sample_id in manifest_index:
            raise ReferenceContractError(
                f"duplicate manifest sample_id {sample_id!r}"
            )
        manifest_index[sample_id] = row

    missing = sorted(set(manifest_index) - set(reference_index))
    extra = sorted(set(reference_index) - set(manifest_index))
    if missing:
        raise ReferenceContractError(
            f"reference is missing {len(missing)} manifest sample(s): "
            f"{missing[:10]} (fail closed)"
        )
    if extra:
        raise ReferenceContractError(
            f"reference has {len(extra)} extra sample(s) not in the manifest: "
            f"{extra[:10]} (fail closed)"
        )

    return {
        sample_id: validate_reference_record(
            reference_index[sample_id], manifest_index[sample_id]
        )
        for sample_id in manifest_index
    }


def reference_K(reference: dict[str, Any]) -> int:
    """``K_C = len(selected_group_indices)`` of the historical record."""
    return validate_diagnostic_budget(int(reference["K_C"]))


def arm_forward_kwargs(
    arm: str, *, budget: int, reference_indices: Sequence[int]
) -> dict[str, Any]:
    """Pure arm -> unified-forward mapping (the whole testable arm contract)."""
    budget = validate_diagnostic_budget(budget)
    if arm == ARM_CONTROL:
        return {
            "mode": MODE_CONTROL,
            "budget": budget,
            "forced_group_indices": None,
        }
    if arm == ARM_FORCED:
        return {
            "mode": MODE_FORCED,
            "budget": budget,
            "forced_group_indices": [int(index) for index in reference_indices],
        }
    raise ValueError(f"arm must be one of {ALLOWED_ARMS}, got {arm!r}")


def arm_selection_rule(arm: str) -> str:
    if arm == ARM_CONTROL:
        return SELECTION_RULE_CONTROL
    if arm == ARM_FORCED:
        return SELECTION_RULE_FORCED
    raise ValueError(f"arm must be one of {ALLOWED_ARMS}, got {arm!r}")


# ---------------------------------------------------------------------------
# Fail-closed resume validation
# ---------------------------------------------------------------------------


class ResumeContractError(RuntimeError):
    """Raised when an existing ``--output`` JSONL violates the resume contract."""


def _resume_expected_fields(
    *, arm: str, manifest_sha256: str, reference_jsonl_sha256: str
) -> tuple[dict[str, str], dict[str, bool]]:
    """Fields every existing output record must agree with (fail closed)."""
    expected_strings = {
        "arm": arm,
        "manifest_sha256": manifest_sha256,
        "reference_jsonl_sha256": reference_jsonl_sha256,
    }
    expected_booleans = {
        "hardware_alignment_relaxed_for_diagnostic": (
            HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC
        ),
        "causal_diagnostic_only": CAUSAL_DIAGNOSTIC_ONLY,
        "fpga_deployable_candidate": FPGA_DEPLOYABLE_CANDIDATE,
    }
    return expected_strings, expected_booleans


def load_resume_completed_ids(
    path: Path,
    samples: Sequence[dict[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
    reference_jsonl_sha256: str,
) -> set[str]:
    """Validate an existing output JSONL and return its completed sample ids.

    A missing or empty file yields the empty set.  Otherwise every non-blank
    line must parse to a JSON object with a unique ``sample_id`` drawn from the
    current post-``--limit-per-task`` ``samples``; every record must agree with
    the current run's ``arm``, ``manifest_sha256`` and
    ``reference_jsonl_sha256`` and carry the frozen diagnostic flags
    (``hardware_alignment_relaxed_for_diagnostic`` / ``causal_diagnostic_only``
    true, ``fpga_deployable_candidate`` false).  Invalid JSON, non-object lines,
    missing/duplicate/out-of-range ids, mixed control/forced arms and stale
    input hashes all abort: resume never silently drops or mixes work.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    allowed_ids = {str(sample["sample_id"]) for sample in samples}
    expected_strings, expected_booleans = _resume_expected_fields(
        arm=arm,
        manifest_sha256=manifest_sha256,
        reference_jsonl_sha256=reference_jsonl_sha256,
    )
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResumeContractError(
                f"output line {line_number} is not valid JSON: {exc} "
                f"(fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise ResumeContractError(
                f"output line {line_number} must be a JSON object (fail closed)"
            )
        if "sample_id" not in record:
            raise ResumeContractError(
                f"output line {line_number} is missing sample_id (fail closed)"
            )
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise ResumeContractError(
                f"duplicate output sample_id {sample_id!r} (fail closed)"
            )
        if sample_id not in allowed_ids:
            raise ResumeContractError(
                f"output sample_id {sample_id!r} is not in the current "
                f"post-limit samples (fail closed)"
            )
        for field, expected in expected_strings.items():
            if record.get(field) != expected:
                raise ResumeContractError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"!= current run {expected!r} (fail closed)"
                )
        for field, expected in expected_booleans.items():
            if record.get(field) is not expected:
                raise ResumeContractError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"must be {expected!r} for this diagnostic (fail closed)"
                )
        done.add(sample_id)
    return done


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    reference: dict[str, Any],
    *,
    arm: str,
    region_grid: int,
    manifest_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    """Run one sample for one arm; both arms share this whole body."""
    if arm not in ALLOWED_ARMS:
        raise ValueError(f"arm must be one of {ALLOWED_ARMS}, got {arm!r}")
    sample_id = str(sample["sample_id"])
    if str(reference.get("sample_id")) != sample_id:
        raise ValueError(
            f"reference sample_id {reference.get('sample_id')!r} != sample "
            f"sample_id {sample_id!r}"
        )
    if str(reference.get("task")) != str(sample["task"]):
        raise ValueError(
            f"reference task {reference.get('task')!r} != sample task "
            f"{sample['task']!r}"
        )
    k_c = validate_diagnostic_budget(int(reference["K_C"]))
    reference_indices = [int(index) for index in reference["group_indices"]]
    if k_c != len(reference_indices):
        raise ValueError(
            f"reference K_C={k_c} != len(group_indices)={len(reference_indices)}"
        )
    reference_g = int(reference["G"])
    kwargs = arm_forward_kwargs(
        arm, budget=k_c, reference_indices=reference_indices
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
        vision = forward_hardware_regular_layer5_forced_index_ablation_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            mode=kwargs["mode"],
            budget=kwargs["budget"],
            forced_group_indices=kwargs["forced_group_indices"],
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

    # Fail closed unless the all-real / no-context / exact-K_C contract holds.
    verify_diagnostic_selection(vision.selection)
    output_count = int(vision.image_embeds.shape[0])
    if output_count != int(vision.real_group_count):
        raise RuntimeError("image embeds are not aligned with the real group count")
    if full_llm_visual_tokens != reference_g:
        raise RuntimeError(
            f"G={full_llm_visual_tokens} != historical baseline G={reference_g}"
        )
    if int(vision.full_llm_visual_tokens) != reference_g:
        raise RuntimeError(
            f"vision G={vision.full_llm_visual_tokens} != historical baseline "
            f"G={reference_g}"
        )
    if int(vision.real_group_count) != k_c or output_count != k_c:
        raise RuntimeError(
            f"K={vision.real_group_count}/{output_count} != K_C={k_c}"
        )
    selected = torch.as_tensor(vision.group_indices, dtype=torch.long)
    if arm == ARM_FORCED:
        observed = [int(index) for index in selected.tolist()]
        if observed != reference_indices:
            raise RuntimeError(
                "forced arm selection differs from the historical indices"
            )

    proxy = qwen3_vl_shallow_pruned_compute_proxy(
        full_patch_tokens,
        output_count,
        prune_after_layer=PRUNE_AFTER_LAYER,
    )
    full_proxy = qwen3_vl_compute_proxy(full_patch_tokens)
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "task": str(sample["task"]),
        "arm": arm,
        "mode": kwargs["mode"],
        "selection_rule": arm_selection_rule(arm),
        "prune_after_layer": int(vision.prune_after_layer),
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "region_grid": int(region_grid),
        # --- official-scoring surface -----------------------------------------
        "output_text": raw_text,
        "raw_output_text": raw_text,
        # --- single-variable / diagnostic declaration -------------------------
        "K_C": k_c,
        "reference_K_C": k_c,
        "reference_G": reference_g,
        "reference_selected_group_indices": list(reference_indices),
        "reference_selected_group_indices_sha256": str(
            reference["group_indices_sha256"]
        ),
        "reference_llm_visual_tokens": int(reference["llm_visual_tokens"]),
        "reference_baseline_llm_visual_tokens": reference_g,
        "reference_selector": REFERENCE_SELECTOR,
        "reference_jsonl_sha256": reference_sha256,
        "manifest_sha256": manifest_sha256,
        "uses_b1_trained_wv": False,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "local_candidate_cap": None,
        "uses_pooled_context_group_values": False,
        # --- timings (all GPU boundaries synchronized) ------------------------
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
        # --- decoding / token bookkeeping -------------------------------------
        "do_sample": False,
        "decoding": "greedy_argmax",
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "baseline_llm_visual_tokens": reference_g,
        "output_group_count": output_count,
        "context_group_count": vision.context_group_count,
        "real_group_count": vision.real_group_count,
        "source_types_all_real": bool(
            np.all(np.asarray(vision.source_type) == 0)
        ),
        "identity_bypass": vision.identity_bypass,
        "requested_K_C": k_c,
        "block_input_shapes": vision.block_input_shapes,
        "input_tokens": int(prepared.input_ids.shape[1]),
        "original_input_tokens": int(prepared.original_sequence_length),
        "output_tokens": output_tokens,
        "selected_group_indices": [int(value) for value in selected.tolist()],
        "selected_group_indices_sha256": indices_sha256(selected.tolist()),
        "source_type": [int(value) for value in vision.source_type.tolist()],
        "region_id": [int(value) for value in vision.region_id.tolist()],
        "valid": [bool(value) for value in vision.valid.tolist()],
        "deepstack_cut_merger_index": vision.deepstack_cut_merger_index,
        "deepstack_merger_indexes": list(vision.deepstack_merger_indexes),
        "llm_visual_tokens": output_count,
        "encoder_tokens_after_prune": proxy.encoder_tokens,
        "encoder_macs_proxy": proxy.encoder_macs,
        "baseline_encoder_macs_proxy": full_proxy.encoder_macs,
        "llm_prefill_macs_proxy": proxy.llm_prefill_macs,
        "baseline_llm_prefill_macs_proxy": full_proxy.llm_prefill_macs,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    record.update(diagnostic_flags())
    del batch, vision, prepared, generated, image
    gc.collect()
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-jsonl",
        type=Path,
        default=REFERENCE_JSONL,
        help=(
            "Frozen historical reference; defaults to the shallow_r0250 artifact "
            "and is mapped uniquely by sample_id."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ALLOWED_ARMS, required=True)
    parser.add_argument("--region-grid", type=int, default=3)
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def build_run_metadata(
    args: argparse.Namespace,
    *,
    manifest_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    return {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "arm": args.arm,
        "mode": MODE_CONTROL if args.arm == ARM_CONTROL else MODE_FORCED,
        "selection_rule": arm_selection_rule(args.arm),
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "cutpoint_prefix_blocks": list(CUTPOINT_PREFIX_BLOCKS),
        "region_grid": args.region_grid,
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha256,
        "reference_jsonl": str(args.reference_jsonl),
        "reference_selector": REFERENCE_SELECTOR,
        "reference_jsonl_sha256": reference_sha256,
        "has_fixed_budget_knob": False,
        "local_candidate_cap": None,
        "uses_context_pooling": False,
        "uses_context_slot_reservation": False,
        "uses_local_candidate_cap": False,
        "uses_b1_trained_wv": False,
        "projection_npz": None,
        "training": False,
        "do_sample": False,
        "custom_greedy_decode": True,
        "attn_implementation": "sdpa",
        "original_position_ids_preserved": True,
        "K_C_is_per_sample_historical_length": True,
        "cpu_selector_roundtrip_is_acceleration_claim": False,
        "timing_contract": {
            "vision_total_seconds": (
                "vision_preamble_seconds + cut_blocks_seconds + "
                "selector_cpu_roundtrip_seconds + remaining_vision_seconds"
            ),
            "gpu_vision_seconds": (
                "vision_total_seconds - selector_cpu_roundtrip_seconds"
            ),
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
        **diagnostic_flags(),
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.arm not in ALLOWED_ARMS:  # pragma: no cover - guarded by choices
        parser.error(f"only --arm {ALLOWED_ARMS} is supported")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this diagnostic experiment requires a CUDA device")

    manifest_rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_sha256 = file_sha256(args.manifest)
    reference_sha256 = file_sha256(args.reference_jsonl)
    # Validate the full manifest/reference mapping *before* applying the limit.
    reference_map = load_reference_records(args.reference_jsonl, manifest_rows)
    samples = limit_each_task(manifest_rows, args.limit_per_task)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail-closed resume: only a fully consistent output contributes ids.
    done = load_resume_completed_ids(
        args.output,
        samples,
        arm=args.arm,
        manifest_sha256=manifest_sha256,
        reference_jsonl_sha256=reference_sha256,
    )
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(
        f"LOAD arm={args.arm} reference_selector={REFERENCE_SELECTOR} "
        f"reference={args.reference_jsonl} samples={len(samples)} "
        f"pending={len(pending)} region_grid={args.region_grid}",
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
    metadata = build_run_metadata(
        args,
        manifest_sha256=manifest_sha256,
        reference_sha256=reference_sha256,
    )
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for sample_number, sample in enumerate(pending, 1):
            record = run_one(
                model,
                processor,
                sample,
                args.manifest,
                reference_map[sample["sample_id"]],
                arm=args.arm,
                region_grid=args.region_grid,
                manifest_sha256=manifest_sha256,
                reference_sha256=reference_sha256,
            )
            record["run"] = metadata
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                f"arm={record['arm']} G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']} K_C={record['K_C']} "
                f"bypass={record['identity_bypass']} "
                f"vision={record['vision_total_seconds']:.4f}s "
                f"model={record['model_inference_seconds']:.4f}s "
                f"app_e2e={record['application_end_to_end_seconds']:.4f}s "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
