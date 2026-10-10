#!/usr/bin/env python3
"""GPU runner for the B1 answer-margin oracle K96 pilot (NOT executed here).

One arm only (``oracle_answer_margin_b1_saliency``) on the frozen 24-sample
manifest produced by
``evaluation/select_b_answer_margin_oracle_pilot24.py``.  This module is written
and unit-tested on CPU but **must not be executed in the pre-registration
round**: the report status is ``PRE-REGISTERED_NOT_EXECUTED`` and
``gpu_runs = 0``.

Per sample:

1. run the full (uncut) B1 visual sequence and the official correct-answer
   branch token to obtain the B1 ``|grad × activation|`` group saliency;
   the model is frozen (``requires_grad_(False)``), there is no optimizer and
   every parameter ``grad`` stays ``None``;
2. inject that saliency as the score through the existing
   ``integer_scorer`` hook of the *unmodified*
   :func:`evaluation.hardware_regular_b1_integration.forward_hardware_regular_b1_vision`,
   so the frozen :func:`select_hardware_regular_groups` anchor/context contract
   runs unchanged: ``G>96`` must emit ``output=96``, ``real=87``,
   ``context=9`` on the original positions (no fixed-indices shortcut);
3. greedy-decode the oracle arm with ``max_new_tokens=16``.

The output is an append-only raw JSONL with fail-closed safe resume: a stale,
partial, duplicated, mixed-arm or hash-inconsistent existing output aborts.
This is a label-aware oracle upper bound (ground-truth answer leakage), never a
deployable selector and never a paper/CV-Bench accuracy claim.
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

from evaluation.answer_margin_b1_preflight import choice_margin
from evaluation.b1_answer_margin_oracle import (
    ALLOWED_ARMS,
    ARM_ORACLE,
    BUDGET,
    CUTPOINT,
    LOCAL_CANDIDATE_CAP,
    MAX_NEW_TOKENS,
    PRUNE_AFTER_LAYER,
    REGION_GRID_FIXED,
    SALIENCY_DEFINITION,
    OraclePilotError,
    array_sha256,
    assert_layout_contract,
    build_oracle_scorer,
    file_sha256,
    group_saliency_from_grad_activation,
    record_sha256,
    saliency_summary,
)
from evaluation.hardware_regular_b1_integration import (
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
    prepare_b1_multimodal_inputs,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT
from evaluation.run_b1_answer_margin_gpu_smoke import (
    answer_index,
    build_option_labels,
    parse_label_branches,
    tokenize_labels,
)
from evaluation.run_hf_shallow_block_pruning import greedy_generate_from_pruned
from evaluation.run_hf_token_reduction import (
    make_messages,
    processor_inputs,
    resolve_image,
)
from evaluation.shallow_block_pruning import (
    forward_shallow_pruned_vision,
    prepare_pruned_multimodal_inputs,
)

__all__ = [
    "ARM_ORACLE",
    "OracleRunnerError",
    "REQUIRED_RECORD_FIELDS",
    "build_parser",
    "main",
    "record_sha256",
    "run_one",
    "validate_resume",
    "verify_config_pins",
]

DEFAULT_MANIFEST = Path("datasets/2026-10-05-b1-answer-margin-oracle-pilot24.jsonl")
DEFAULT_CONFIG = Path("configs/2026-10-05-b1-answer-margin-oracle-pilot24.preregistered.json")
DEFAULT_OUTPUT = Path(
    "artifacts/token_selector/2026-10-05-b1-answer-margin-oracle-pilot24/"
    "oracle_answer_margin_b1_saliency_k96.jsonl"
)
DEFAULT_MODEL = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
RETENTION_FULL = 1.0

REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "arm",
    "manifest_sha256",
    "config_sha256",
    "requested_budget",
    "region_grid",
    "local_candidate_cap",
    "output_text",
    "output_token_ids",
    "full_patch_tokens",
    "image_grid_thw",
    "full_llm_visual_tokens",
    "output_group_count",
    "context_group_count",
    "real_group_count",
    "identity_bypass",
    "selected_group_indices",
    "label_tokenization",
    "margin",
    "saliency",
    "saliency_sha256",
    "saliency_group_values",
    "timings",
    "peak_cuda_bytes",
    "model_parameter_grads_all_none",
    "all_model_parameters_requires_grad_false",
    "record_sha256",
)


class OracleRunnerError(RuntimeError):
    """Raised when the oracle runner or its resume contract is violated."""


def _sync_clock() -> float:
    """Drain the GPU then read the wall clock: a clean stage boundary."""

    torch.cuda.synchronize()
    return time.perf_counter()


def _all_parameters_frozen(model: torch.nn.Module) -> tuple[bool, int]:
    parameters = list(model.parameters())
    return all(not parameter.requires_grad for parameter in parameters), len(parameters)


def _all_parameter_grads_none(model: torch.nn.Module) -> tuple[bool, int]:
    parameters = list(model.parameters())
    return all(parameter.grad is None for parameter in parameters), len(parameters)


# ---------------------------------------------------------------------------
# Config pin verification
# ---------------------------------------------------------------------------
def verify_config_pins(
    config: dict[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Re-verify every non-null SHA pin declared in the pre-registered config."""

    checks: list[dict[str, Any]] = []
    for section in ("inputs_sha256", "summarizer_inputs"):
        specs = config.get(section, {})
        if not isinstance(specs, dict):
            raise OracleRunnerError(f"config {section} must be an object")
        for name, spec in specs.items():
            if not isinstance(spec, dict):
                raise OracleRunnerError(f"config pin {section}.{name} must be an object")
            expected = spec.get("sha256")
            relative = spec.get("path")
            if expected is None:
                continue
            if relative is None:
                raise OracleRunnerError(f"config pin {section}.{name} has no path")
            path = Path(str(relative))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise OracleRunnerError(
                    f"pinned file {path} is missing (fail closed)"
                )
            actual = file_sha256(path)
            if actual != str(expected):
                raise OracleRunnerError(
                    f"pin {section}.{name} sha256 mismatch: expected {expected}, "
                    f"got {actual} (fail closed)"
                )
            checks.append(
                {
                    "name": f"{section}.{name}",
                    "path": str(path),
                    "sha256": actual,
                    "matches": True,
                }
            )
    return {"pins_verified": True, "checks": checks}


# ---------------------------------------------------------------------------
# Fail-closed resume validation
# ---------------------------------------------------------------------------
def validate_resume(
    path: Path,
    samples: Sequence[dict[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
    config_sha256: str,
    budget: int,
    region_grid: int,
    local_candidate_cap: int,
) -> set[str]:
    """Validate an existing output JSONL and return its completed sample ids."""

    if not path.is_file() or path.stat().st_size == 0:
        return set()
    allowed_ids = {str(sample["sample_id"]) for sample in samples}
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OracleRunnerError(
                f"output line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise OracleRunnerError(
                f"output line {line_number} must be a JSON object (fail closed)"
            )
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise OracleRunnerError(
                f"output line {line_number} is missing fields {missing} "
                f"(partial record, fail closed)"
            )
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise OracleRunnerError(
                f"duplicate output sample_id {sample_id!r} (fail closed)"
            )
        if sample_id not in allowed_ids:
            raise OracleRunnerError(
                f"output sample_id {sample_id!r} is not in the current manifest "
                f"(stale output, fail closed)"
            )
        for field, expected in (
            ("arm", arm),
            ("manifest_sha256", manifest_sha256),
            ("config_sha256", config_sha256),
        ):
            if record.get(field) != expected:
                raise OracleRunnerError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"!= current run {expected!r} (mixed/stale output, fail closed)"
                )
        for field, expected in (
            ("requested_budget", int(budget)),
            ("region_grid", int(region_grid)),
            ("local_candidate_cap", int(local_candidate_cap)),
        ):
            try:
                actual_value = int(record.get(field))
            except (TypeError, ValueError) as exc:
                raise OracleRunnerError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"is not an integer (fail closed)"
                ) from exc
            if actual_value != expected:
                raise OracleRunnerError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"!= current run {expected!r} (fail closed)"
                )
        grid = [int(value) for value in record["image_grid_thw"]]
        full_patch_tokens = int(record["full_patch_tokens"])
        full_llm = int(record["full_llm_visual_tokens"])
        output_group_count = int(record["output_group_count"])
        selected = [int(value) for value in record["selected_group_indices"]]
        if len(grid) != 3 or grid[0] != 1:
            raise OracleRunnerError(
                f"output sample {sample_id!r} image_grid_thw={grid!r} is not a "
                f"single image (fail closed)"
            )
        if grid[1] * grid[2] != full_patch_tokens:
            raise OracleRunnerError(
                f"output sample {sample_id!r} grid {grid} does not match "
                f"full_patch_tokens {full_patch_tokens} (fail closed)"
            )
        if full_llm * MERGE_UNIT != full_patch_tokens:
            raise OracleRunnerError(
                f"output sample {sample_id!r} full_llm_visual_tokens {full_llm} "
                f"does not match full_patch_tokens {full_patch_tokens} (fail closed)"
            )
        if len(selected) != output_group_count:
            raise OracleRunnerError(
                f"output sample {sample_id!r} has {len(selected)} selected groups "
                f"but output_group_count {output_group_count} (fail closed)"
            )
        if selected and not all(
            later > earlier for earlier, later in zip(selected, selected[1:])
        ):
            raise OracleRunnerError(
                f"output sample {sample_id!r} selected_group_indices are not "
                f"strictly ascending (fail closed)"
            )
        if bool(record["identity_bypass"]):
            raise OracleRunnerError(
                f"output sample {sample_id!r} is an identity bypass; the pilot is "
                f"pre-registered for G>96 only (fail closed)"
            )
        if full_llm <= int(budget):
            raise OracleRunnerError(
                f"output sample {sample_id!r} has G={full_llm} <= K={budget}; the "
                f"pilot is pre-registered for G>96 only (fail closed)"
            )
        if output_group_count != int(budget):
            raise OracleRunnerError(
                f"output sample {sample_id!r} non-bypass must emit K={budget}, "
                f"got {output_group_count} (fail closed)"
            )
        if int(record["real_group_count"]) != 87 or int(
            record["context_group_count"]
        ) != 9:
            raise OracleRunnerError(
                f"output sample {sample_id!r} layout real/context="
                f"{record['real_group_count']}/{record['context_group_count']} "
                f"!= 87/9 (fail closed)"
            )
        if not bool(record["model_parameter_grads_all_none"]):
            raise OracleRunnerError(
                f"output sample {sample_id!r} has non-None parameter grads "
                f"(fail closed)"
            )
        values = np.asarray(record["saliency_group_values"], dtype=np.float64).reshape(-1)
        if values.size != full_llm:
            raise OracleRunnerError(
                f"output sample {sample_id!r} saliency has {values.size} groups "
                f"but G={full_llm} (fail closed)"
            )
        if str(record.get("saliency_sha256")) != array_sha256(values):
            raise OracleRunnerError(
                f"output sample {sample_id!r} saliency_sha256 mismatch (fail closed)"
            )
        expected_record_sha = record_sha256(record)
        if str(record.get("record_sha256")) != expected_record_sha:
            raise OracleRunnerError(
                f"output sample {sample_id!r} record_sha256 mismatch (stale or "
                f"corrupted record, fail closed)"
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
    *,
    budget: int,
    region_grid: int,
    local_candidate_cap: int,
    max_new_tokens: int,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Run one oracle sample through the saliency + injected-selection path."""

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
    if merge_unit != MERGE_UNIT or full_patch_tokens % merge_unit:
        raise OracleRunnerError("processor grid is not aligned to the visual merger")
    full_llm_visual_tokens = full_patch_tokens // merge_unit
    if full_llm_visual_tokens <= int(budget):
        raise OracleRunnerError(
            f"sample {sample['sample_id']!r} has G={full_llm_visual_tokens} <= "
            f"K={budget}; the pre-registered pilot requires G>96"
        )

    # --- Real-tokenizer label contract + manifest cross-check --------------
    choices = [str(choice) for choice in sample.get("choices", [])]
    labels = build_option_labels(len(choices))
    label_token_ids = tokenize_labels(processor.tokenizer, labels)
    branches = parse_label_branches(label_token_ids)
    correct_index = answer_index(str(sample["answer"]), labels)
    decoded_branch_tokens = [
        processor.tokenizer.decode([token]) for token in branches.branch_token_ids
    ]
    pilot = sample.get("oracle_pilot") or {}
    if pilot:
        if int(pilot.get("full_llm_visual_tokens", -1)) != full_llm_visual_tokens:
            raise OracleRunnerError(
                f"sample {sample['sample_id']!r} processor G="
                f"{full_llm_visual_tokens} != manifest selection G="
                f"{pilot.get('full_llm_visual_tokens')} (fail closed)"
            )
        if [int(v) for v in pilot.get("branch_token_ids", [])] != list(
            branches.branch_token_ids
        ):
            raise OracleRunnerError(
                f"sample {sample['sample_id']!r} branch token ids drifted from the "
                f"pre-registered manifest (fail closed)"
            )

    # --- Stage 1: full-sequence B1 saliency (grad × activation) ------------
    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    saliency_forward_started = _sync_clock()
    vision = forward_shallow_pruned_vision(
        model.model.visual,
        batch.pixel_values,
        batch.image_grid_thw,
        retention=RETENTION_FULL,
        prune_after_layer=PRUNE_AFTER_LAYER,
        capture_prune_gradient=True,
    )
    prepared = prepare_pruned_multimodal_inputs(model, batch, vision)
    inputs_embeds = prepared.inputs_embeds
    attention_mask = prepared.attention_mask
    position_ids = prepared.position_ids
    visual_pos_masks = prepared.visual_pos_masks
    if visual_pos_masks.ndim == 1:
        visual_pos_masks = visual_pos_masks.unsqueeze(0)
    prefix_length = branches.prefix_length
    if prefix_length:
        prefix_ids = torch.tensor(
            [branches.common_prefix],
            dtype=prepared.input_ids.dtype,
            device=prepared.input_ids.device,
        )
        inputs_embeds = torch.cat(
            [inputs_embeds, model.get_input_embeddings()(prefix_ids)], dim=1
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], prefix_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=1,
        )
        base_position = int(position_ids.max().item()) + 1
        extra_positions = (
            torch.arange(prefix_length, device=position_ids.device) + base_position
        ).view(1, 1, prefix_length)
        extra_positions = extra_positions.expand(position_ids.shape[0], 1, prefix_length)
        position_ids = torch.cat([position_ids, extra_positions], dim=2)
        visual_pos_masks = torch.cat(
            [
                visual_pos_masks,
                torch.zeros(
                    (visual_pos_masks.shape[0], prefix_length),
                    dtype=torch.bool,
                    device=visual_pos_masks.device,
                ),
            ],
            dim=1,
        )
    outputs = model.model.language_model(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=prepared.deepstack_visual_embeds,
    )
    branch_logits = model.lm_head(outputs.last_hidden_state[:, -1, :])
    margin = choice_margin(
        branch_logits, [branches.branch_token_ids], [correct_index]
    )
    margin_value = float(margin.detach().item())
    option_ids = torch.tensor(
        branches.branch_token_ids, dtype=torch.long, device=branch_logits.device
    )
    option_logits = branch_logits[0].index_select(0, option_ids)
    option_logits_list = [float(value) for value in option_logits.detach().tolist()]
    predicted_index = int(option_logits.detach().argmax().item())
    saliency_forward_seconds = _sync_clock() - saliency_forward_started

    saliency_backward_started = _sync_clock()
    margin.sum().backward()
    saliency_backward_seconds = _sync_clock() - saliency_backward_started

    b1_hidden = vision.prune_hidden_states
    if b1_hidden is None or b1_hidden.grad is None:
        raise OracleRunnerError(
            "full-sequence B1 hidden states received no gradient"
        )
    if int(b1_hidden.shape[0]) != full_patch_tokens:
        raise OracleRunnerError(
            f"B1 hidden rows {int(b1_hidden.shape[0])} != full patch tokens "
            f"{full_patch_tokens} (retention must be 1.0)"
        )
    saliency = group_saliency_from_grad_activation(b1_hidden, b1_hidden.grad, merge_unit)
    if len(saliency) != full_llm_visual_tokens:
        raise OracleRunnerError(
            f"saliency has {len(saliency)} groups but G={full_llm_visual_tokens}"
        )
    summary = saliency_summary(saliency)
    saliency_sha256 = array_sha256(np.asarray(saliency, dtype=np.float64))
    grads_none, parameter_count = _all_parameter_grads_none(model)
    frozen, frozen_count = _all_parameters_frozen(model)
    if not grads_none:
        raise OracleRunnerError(
            "model parameters received gradients; the model must stay frozen"
        )
    del outputs, branch_logits, margin, prepared, vision
    gc.collect()
    torch.cuda.empty_cache()

    # --- Stage 2: inject saliency into the frozen selection contract -------
    saliency_np = np.asarray(saliency, dtype=np.float64)
    scorer = build_oracle_scorer(saliency_np)
    with torch.inference_mode():
        hardware = forward_hardware_regular_b1_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            budget=budget,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=None,
            integer_scorer=scorer,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
            synchronize=torch.cuda.synchronize,
        )
        layout = assert_layout_contract(
            hardware.selection, full_groups=full_llm_visual_tokens, budget=budget
        )
        llm_prepare_started = _sync_clock()
        prepared2 = prepare_b1_multimodal_inputs(model, batch, hardware)
        llm_prepare_seconds = _sync_clock() - llm_prepare_started
        generated, prefill_seconds, decoder_seconds, output_tokens = (
            greedy_generate_from_pruned(model, prepared2, int(max_new_tokens))
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

    selected = np.asarray(hardware.group_indices, dtype=np.int64)
    patch_indices = np.asarray(hardware.patch_indices, dtype=np.int64).reshape(-1)
    source_type = np.asarray(hardware.source_type, dtype=np.int64)
    region_id = np.asarray(hardware.region_id, dtype=np.int64)
    valid = np.asarray(hardware.valid, dtype=bool)

    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "source": sample.get("source"),
        "source_dataset": sample.get("source_dataset"),
        "arm": ARM_ORACLE,
        "score_mode": "oracle_b1_answer_margin_saliency",
        "integration_score_mode": SCORE_MODE_INTEGER,
        "score_injection": "integer_scorer_hook_precomputed_combined_scores",
        "is_label_aware_oracle": True,
        "deployable_selector": False,
        "is_paper_accuracy_claim": False,
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "cutpoint": CUTPOINT,
        "output_text": raw_text,
        "raw_output_text": raw_text,
        "output_token_ids": [int(value) for value in generated[0].tolist()],
        "output_tokens": output_tokens,
        "label_tokenization": {
            "labels": labels,
            "label_token_ids": label_token_ids,
            "common_prefix": list(branches.common_prefix),
            "common_suffix": list(branches.common_suffix),
            "branch_token_ids": list(branches.branch_token_ids),
            "decoded_branch_tokens": decoded_branch_tokens,
            "prefix_length": int(branches.prefix_length),
            "suffix_length": int(branches.suffix_length),
            "single_branch_token_per_label": True,
            "branch_tokens_distinct": True,
            "contract_ok": True,
        },
        "margin": {
            "definition": "correct_option_logit - max(other_option_logits)",
            "value": margin_value,
            "correct_index": correct_index,
            "correct_label": labels[correct_index],
            "predicted_index": predicted_index,
            "predicted_label": labels[predicted_index],
            "option_logits": option_logits_list,
            "branch_token_ids": list(branches.branch_token_ids),
            "decoded_branch_tokens": decoded_branch_tokens,
        },
        "saliency": {
            "definition": SALIENCY_DEFINITION,
            "summary": summary,
            "sha256": saliency_sha256,
            "group_count": len(saliency),
            "full_sequence_retention": RETENTION_FULL,
        },
        "saliency_sha256": saliency_sha256,
        "saliency_group_values": [float(value) for value in saliency],
        "layout": layout,
        "image_open_decode_seconds": image_open_decode_seconds,
        "processor_seconds": processor_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "saliency_forward_seconds": saliency_forward_seconds,
        "saliency_backward_seconds": saliency_backward_seconds,
        "vision_preamble_seconds": hardware.vision_preamble_seconds,
        "block0_seconds": hardware.block0_seconds,
        "selector_cpu_roundtrip_seconds": hardware.selector_cpu_seconds,
        "remaining_vision_seconds": hardware.remaining_vision_seconds,
        "vision_total_seconds": hardware.vision_total_seconds,
        "gpu_vision_seconds": hardware.gpu_vision_seconds,
        "llm_prepare_seconds": llm_prepare_seconds,
        "llm_prefill_seconds": prefill_seconds,
        "decoder_seconds": decoder_seconds,
        "postprocess_seconds": postprocess_seconds,
        "model_inference_seconds": model_inference_seconds,
        "application_end_to_end_seconds": application_end_to_end_seconds,
        "timings": {
            "gpu_boundaries_synchronized": True,
            "excludes_model_load": True,
            "not_steady_state_latency": True,
            "not_selector_speedup": True,
            "saliency_forward_seconds": saliency_forward_seconds,
            "saliency_backward_seconds": saliency_backward_seconds,
            "vision_total_seconds": hardware.vision_total_seconds,
            "gpu_vision_seconds": hardware.gpu_vision_seconds,
            "llm_prepare_seconds": llm_prepare_seconds,
            "llm_prefill_seconds": prefill_seconds,
            "decoder_seconds": decoder_seconds,
            "model_inference_seconds": model_inference_seconds,
            "application_end_to_end_seconds": application_end_to_end_seconds,
        },
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "do_sample": False,
        "decoding": "greedy_argmax",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "max_new_tokens": int(max_new_tokens),
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "output_group_count": int(hardware.image_embeds.shape[0]),
        "context_group_count": hardware.context_group_count,
        "real_group_count": hardware.real_group_count,
        "identity_bypass": bool(hardware.selection.bypass),
        "requested_budget": budget,
        "region_grid": region_grid,
        "local_candidate_cap": local_candidate_cap,
        "block_input_shapes": hardware.block_input_shapes,
        "input_tokens": int(prepared2.input_ids.shape[1]),
        "original_input_tokens": int(prepared2.original_sequence_length),
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
        "model_parameter_grads_all_none": grads_none,
        "all_model_parameters_requires_grad_false": frozen,
        "model_parameter_count": parameter_count,
        "model_parameter_frozen_count": frozen_count,
        "optimizer_constructed": False,
        "training": False,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
    }
    record["record_sha256"] = record_sha256(record)
    del batch, hardware, prepared2, generated, image
    gc.collect()
    torch.cuda.empty_cache()
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default="RoboBrain2.5-4B-BF16")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--region-grid", type=int, default=REGION_GRID_FIXED)
    parser.add_argument("--local-candidate-cap", type=int, default=LOCAL_CANDIDATE_CAP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not args.model.is_dir():
        raise OracleRunnerError(
            f"--model must be a local directory (strict offline): {args.model}"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise OracleRunnerError("this experiment requires a CUDA device")

    config_path = args.config if args.config.is_absolute() else root / args.config
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    if not config_path.is_file():
        raise OracleRunnerError(f"missing pre-registered config {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_config_pins(config, root=root)
    config_sha256 = file_sha256(config_path)
    manifest_sha256 = file_sha256(manifest_path)
    pinned_manifest = config.get("manifest", {}).get("sha256")
    if pinned_manifest is not None and str(pinned_manifest) != manifest_sha256:
        raise OracleRunnerError(
            f"manifest sha256 {manifest_sha256} != config pin {pinned_manifest} "
            f"(fail closed)"
        )
    if config.get("arm") != ARM_ORACLE:
        raise OracleRunnerError(
            f"config arm {config.get('arm')!r} != {ARM_ORACLE!r} (fail closed)"
        )

    samples = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [str(sample["sample_id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise OracleRunnerError("manifest contains duplicate sample_id (fail closed)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = validate_resume(
        output_path,
        samples,
        arm=ARM_ORACLE,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
        budget=args.budget,
        region_grid=args.region_grid,
        local_candidate_cap=args.local_candidate_cap,
    )
    pending = [sample for sample in samples if str(sample["sample_id"]) not in done]
    print(
        f"LOAD arm={ARM_ORACLE} samples={len(samples)} budget={args.budget} "
        f"region_grid={args.region_grid} local_cap={args.local_candidate_cap} "
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
    model.requires_grad_(False)
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    metadata = {
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "arm": ARM_ORACLE,
        "integration_score_mode": SCORE_MODE_INTEGER,
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "cutpoint": CUTPOINT,
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": args.local_candidate_cap,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "custom_greedy_decode": True,
        "training": False,
        "optimizer_constructed": False,
        "model_parameters_frozen": True,
        "is_label_aware_oracle": True,
        "ground_truth_answer_leakage": True,
        "deployable_selector": False,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
    }
    with output_path.open("a", encoding="utf-8", buffering=1) as stream:
        for sample_number, sample in enumerate(pending, 1):
            record = run_one(
                model,
                processor,
                sample,
                manifest_path,
                budget=args.budget,
                region_grid=args.region_grid,
                local_candidate_cap=args.local_candidate_cap,
                max_new_tokens=MAX_NEW_TOKENS,
                manifest_sha256=manifest_sha256,
                config_sha256=config_sha256,
            )
            record["run"] = metadata
            record["record_sha256"] = record_sha256(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                f"G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']} "
                f"C={record['context_group_count']} "
                f"margin={record['margin']['value']:.4f} "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
