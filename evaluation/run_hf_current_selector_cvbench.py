#!/usr/bin/env python3
"""Real-model B1 runner for the 2026-10-03 current (integer) Selector.

Modes
-----
``identity``
    For at most a handful of samples, force the B1 integer selector into its
    ``G <= K`` identity bypass and compare the generated token ids, decoded
    text and dense vision ``image_embeds`` against the official dense path
    (``model.generate`` / ``model.model.visual``).  No quality is aggregated.
``smoke``
    Structural check only (shapes, K, placeholders / M-RoPE, finite timing,
    INT8/INT32 bounds).  Quality is recorded but never aggregated.
``full``
    One complete deterministic CV-Bench pass for one K.  Resumable from the
    raw JSONL: a ``sample_id`` already present with the same configuration is
    never recomputed.
``latency``
    Performance pass: fixed stratified subset, ``warmup`` then ``repeats`` raw
    records per sample.  Quality is not aggregated.

The selector is always labelled ``software``: the FPGA is not on the board in
this workspace, so no field may be read as board acceleration.

Timing follows ``configs/evaluation_protocol.json``: the application pipeline
covers image decode, processor, host-to-device, vision, selector software round
trip, LLM prepare/prefill/decode and postprocess; the model end-to-end covers
vision + prepare + prefill + decode.  Every GPU boundary is bracketed by
``torch.cuda.synchronize``.  Model loading is never part of the steady state.
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

from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice
from evaluation.current_selector_integer import (
    DEPLOYMENT_CHECKPOINT,
    DEPLOYMENT_CHECKPOINT_SHA256,
    DEPLOYMENT_SEED,
    DEPLOYMENT_WV_PATH,
    DEPLOYMENT_WV_SCALE_PATH,
    IntegerSelectorScorer,
    load_deployment_wv_bundle,
)
from evaluation.hardware_regular_b1_integration import (
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
    prepare_b1_multimodal_inputs,
)
from evaluation.run_hf_hardware_regular_b1 import file_sha256, load_projection
from evaluation.run_hf_shallow_block_pruning import (
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

MODES = ("identity", "smoke", "full", "latency")
SELECTOR_PATH = "software"
INT32_BOUND_KEYS = (
    "projection_int_absmax",
    "dot_absmax",
    "norm2_max",
    "semantic_q16_absmax",
)


def _sync_clock() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _finite_nonnegative(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    for key in keys:
        value = record.get(key)
        if value is None or not np.isfinite(value) or value < 0.0:
            return False
    return True


def int32_bounds_present(bounds: Any, *, bypass: bool) -> bool:
    """Require integer audit bounds only when the selector actually runs."""
    if bypass:
        return True
    return isinstance(bounds, dict) and all(key in bounds for key in INT32_BOUND_KEYS)


def resume_state(
    path: Path,
    *,
    mode: str,
    budget: int | None,
) -> set[str]:
    """Return already-completed sample ids, failing closed on config drift.

    A raw JSONL must contain at most one record per ``sample_id`` for a given
    mode/budget/seed; mixing configs in one file would silently change the
    measured population, so it is a hard error.
    """
    from evaluation.metrics import load_jsonl

    if not path.is_file():
        return set()
    done: set[str] = set()
    for record in load_jsonl(path):
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise RuntimeError(f"duplicate sample_id {sample_id} in {path}")
        if str(record.get("mode")) != mode:
            raise RuntimeError(
                f"{path} mixes modes: {record.get('mode')!r} != {mode!r}"
            )
        if budget is not None and int(record.get("requested_budget", -1)) != int(budget):
            raise RuntimeError(
                f"{path} mixes budgets: {record.get('requested_budget')!r} != {budget}"
            )
        if int(record.get("deployment_seed", -1)) != DEPLOYMENT_SEED:
            raise RuntimeError(
                f"{path} uses deployment_seed {record.get('deployment_seed')!r}"
            )
        if str(record.get("score_mode")) != SCORE_MODE_INTEGER:
            raise RuntimeError(
                f"{path} uses score_mode {record.get('score_mode')!r}"
            )
        done.add(sample_id)
    return done


def dense_generate(
    model: torch.nn.Module,
    processor: Any,
    batch: Any,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    """Official dense greedy path: ``model.generate(do_sample=False)``."""
    generated = model.generate(
        **batch,
        do_sample=False,
        max_new_tokens=int(max_new_tokens),
    )
    trimmed = generated[:, batch.input_ids.shape[1] :]
    text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return [int(value) for value in trimmed[0].tolist()], text


def dense_vision_embeds(model: torch.nn.Module, batch: Any) -> torch.Tensor:
    with torch.inference_mode():
        image_embeds, _ = model.model.visual(
            batch.pixel_values, batch.image_grid_thw
        )
    return image_embeds


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    mode: str,
    budget: int,
    selector: IntegerSelectorScorer,
    max_new_tokens: int | None,
    region_grid: int,
    local_candidate_cap: int,
) -> dict[str, Any]:
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

    effective_budget = int(full_llm_visual_tokens) if mode == "identity" else int(budget)
    if effective_budget % 16:
        raise ValueError("effective budget must be a multiple of 16")

    image_token_id = int(model.config.image_token_id)
    text_mask = (batch.attention_mask[0] != 0) & (batch.input_ids[0] != image_token_id)
    text_embedding_started = _sync_clock()
    with torch.inference_mode():
        selector_text = model.get_input_embeddings()(batch.input_ids[0, text_mask])
    prompt_text_embedding_seconds = _sync_clock() - text_embedding_started
    text_embeddings = selector_text.detach().to(torch.float32).cpu().numpy()

    new_tokens = (
        int(max_new_tokens)
        if max_new_tokens is not None
        else int(sample["max_new_tokens"])
    )

    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    with torch.inference_mode():
        vision = forward_hardware_regular_b1_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            budget=effective_budget,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=text_embeddings,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
            synchronize=torch.cuda.synchronize,
            integer_scorer=selector,
        )
        llm_prepare_started = _sync_clock()
        prepared = prepare_b1_multimodal_inputs(model, batch, vision)
        llm_prepare_seconds = _sync_clock() - llm_prepare_started
        generated, prefill_seconds, decoder_seconds, output_tokens = (
            greedy_generate_from_pruned(model, prepared, new_tokens)
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

    payload = vision.integer_payload or {}
    score_q = np.asarray(payload.get("score_q", []), dtype=np.int64)
    bounds = payload.get("bounds", {})
    selected = torch.as_tensor(vision.group_indices, dtype=torch.long)
    output_group_count = int(vision.image_embeds.shape[0])

    choices = list(sample["choices"])
    parsed = parse_choice(raw_text, choices)
    truth = answer_letter(str(sample["answer"]))

    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "type": sample["type"],
        "source": sample["source"],
        "source_dataset": sample["source_dataset"],
        "official_answer": sample["answer"],
        "parsed_answer": parsed,
        "parse_success": parsed is not None,
        "correct": parsed == truth,
        "mode": mode,
        "selector_path": SELECTOR_PATH,
        "score_mode": SCORE_MODE_INTEGER,
        "deployment_seed": DEPLOYMENT_SEED,
        "output_text": raw_text,
        "output_token_ids": [int(value) for value in generated[0].tolist()],
        "do_sample": False,
        "decoding": "greedy_argmax",
        "max_new_tokens": new_tokens,
        "requested_budget": int(budget),
        "effective_budget": effective_budget,
        "output_group_count": output_group_count,
        "real_group_count": int(vision.real_group_count),
        "context_group_count": int(vision.context_group_count),
        "identity_bypass": bool(vision.selection.bypass),
        "actual_token_retention": (
            output_group_count / full_llm_visual_tokens
            if full_llm_visual_tokens
            else 1.0
        ),
        "image_open_decode_seconds": image_open_decode_seconds,
        "processor_seconds": processor_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "prompt_text_embedding_seconds": prompt_text_embedding_seconds,
        "vision_preamble_seconds": vision.vision_preamble_seconds,
        "block0_seconds": vision.block0_seconds,
        "selector_software_seconds": vision.selector_cpu_seconds,
        "remaining_vision_seconds": vision.remaining_vision_seconds,
        "vision_total_seconds": vision.vision_total_seconds,
        "gpu_vision_seconds": vision.gpu_vision_seconds,
        "llm_prepare_seconds": llm_prepare_seconds,
        "llm_prefill_seconds": prefill_seconds,
        "decoder_seconds": decoder_seconds,
        "postprocess_seconds": postprocess_seconds,
        "model_inference_seconds": model_inference_seconds,
        "application_end_to_end_seconds": application_end_to_end_seconds,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "full_patch_tokens": full_patch_tokens,
        "image_grid_thw": [int(value) for value in batch.image_grid_thw[0].tolist()],
        "full_llm_visual_tokens": full_llm_visual_tokens,
        "input_tokens": int(prepared.input_ids.shape[1]),
        "original_input_tokens": int(prepared.original_sequence_length),
        "visual_placeholder_count": int(prepared.visual_pos_masks.sum()),
        "output_tokens": output_tokens,
        "selected_group_indices": [int(value) for value in selected.tolist()],
        "selected_group_indices_sha256": selection_sha256(selected),
        "source_type": [int(value) for value in vision.source_type.tolist()],
        "region_id": [int(value) for value in vision.region_id.tolist()],
        "valid": [bool(value) for value in vision.valid.tolist()],
        "block_input_shapes": vision.block_input_shapes,
        "score_q_sha256": file_sha256_of_bytes(score_q.astype("<u2").tobytes()),
        "score_q_min": int(score_q.min()) if score_q.size else None,
        "score_q_max": int(score_q.max()) if score_q.size else None,
        "integer_bounds": bounds,
        "score_q": [int(value) for value in score_q.tolist()],
        "selector_software_is_acceleration_claim": False,
        "timing_contract": {
            "vision_total_seconds": (
                "vision_preamble_seconds + block0_seconds + "
                "selector_software_seconds + remaining_vision_seconds"
            ),
            "gpu_vision_seconds": (
                "vision_total_seconds - selector_software_seconds"
            ),
            "model_inference_seconds": (
                "vision_total_seconds + llm_prepare_seconds + "
                "llm_prefill_seconds + decoder_seconds"
            ),
            "application_end_to_end_seconds": (
                "image_open_decode + processor + host_to_device + "
                "prompt_text_embedding + model_inference + postprocess"
            ),
            "gpu_boundaries_synchronized": True,
        },
        "proxy": {
            "llm_visual_tokens": qwen3_vl_shallow_pruned_compute_proxy(
                full_patch_tokens,
                int(vision.image_embeds.shape[0]),
                prune_after_layer=0,
            ).llm_visual_tokens,
            "baseline_llm_visual_tokens": full_llm_visual_tokens,
            "baseline_encoder_macs_proxy": qwen3_vl_compute_proxy(
                full_patch_tokens
            ).encoder_macs,
        },
    }

    timing_keys = (
        "image_open_decode_seconds",
        "processor_seconds",
        "host_to_device_seconds",
        "prompt_text_embedding_seconds",
        "vision_preamble_seconds",
        "block0_seconds",
        "selector_software_seconds",
        "remaining_vision_seconds",
        "vision_total_seconds",
        "gpu_vision_seconds",
        "llm_prepare_seconds",
        "llm_prefill_seconds",
        "decoder_seconds",
        "postprocess_seconds",
        "model_inference_seconds",
        "application_end_to_end_seconds",
    )
    checks = {
        "timings_finite_nonnegative": _finite_nonnegative(record, timing_keys),
        "placeholder_count_matches": (
            record["visual_placeholder_count"] == output_group_count
        ),
        "input_tokens_consistent": record["input_tokens"]
        == record["original_input_tokens"]
        - (full_llm_visual_tokens - output_group_count),
        "output_count_is_budget_or_identity": (
            output_group_count == effective_budget or bool(vision.selection.bypass)
        ),
        "group_indices_strictly_ascending": bool(
            np.all(np.diff(vision.group_indices) > 0)
        ),
        "image_embeds_rows_match": (
            int(vision.image_embeds.shape[0]) == output_group_count
        ),
        "score_q_within_uint16": bool(
            score_q.size == 0 or (score_q.min() >= 0 and score_q.max() <= 32767)
        ),
        "int32_bounds_present": int32_bounds_present(
            bounds,
            bypass=bool(vision.selection.bypass),
        ),
    }
    record["structural_checks"] = checks

    if mode == "identity":
        dense_ids, dense_text = dense_generate(model, processor, batch, new_tokens)
        dense_embeds = dense_vision_embeds(model, batch)
        record["identity_gate"] = {
            "dense_token_ids": dense_ids,
            "b1_token_ids": [int(value) for value in generated[0].tolist()],
            "token_ids_equal": dense_ids == [int(v) for v in generated[0].tolist()],
            "text_equal": dense_text == raw_text,
            "vision_embeds_equal": bool(
                torch.equal(dense_embeds, vision.image_embeds)
            ),
            "vision_embeds_max_abs_diff": float(
                (dense_embeds - vision.image_embeds).abs().max().item()
            ),
        }
        record["identity_gate"]["pass"] = bool(
            record["identity_gate"]["token_ids_equal"]
            and record["identity_gate"]["text_equal"]
            and record["identity_gate"]["vision_embeds_equal"]
        )

    del batch, vision, prepared, generated, image, selector_text
    gc.collect()
    return record


def file_sha256_of_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--budget", type=int, help="required for smoke/full/latency")
    parser.add_argument("--limit-per-task", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--region-grid", type=int, default=3)
    parser.add_argument("--local-candidate-cap", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject mode/flag combinations before any model is loaded."""
    if args.mode in ("smoke", "full", "latency") and args.budget is None:
        parser.error(f"--mode {args.mode} requires --budget")
    if args.mode == "latency" and (args.repeats < 1 or args.warmup < 0):
        parser.error("--repeats must be >= 1 and --warmup >= 0")
    if args.mode == "identity" and args.budget is not None:
        parser.error("--mode identity forces budget = G and must not receive --budget")


def measurement_schedule(mode: str, warmup: int, repeats: int) -> dict[str, int]:
    """Return the per-sample warmup / recorded-repeat schedule.

    Only ``latency`` performs ``warmup`` unrecorded ``run_one`` calls before
    the recorded repeats; those warmup records are never serialised.  Every
    other mode runs exactly once and never warms up, so ``--warmup`` and
    ``--repeats`` cannot change their behaviour.
    """
    if mode == "latency":
        return {"warmup": int(warmup), "repeats": int(repeats)}
    return {"warmup": 0, "repeats": 1}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")

    projection, checkpoint_meta = load_projection(
        Path(DEPLOYMENT_CHECKPOINT), expected_sha256=DEPLOYMENT_CHECKPOINT_SHA256
    )
    wv_bundle = load_deployment_wv_bundle()
    if file_sha256(Path(DEPLOYMENT_WV_PATH)) != wv_bundle["wv_sha256"]:
        raise RuntimeError("deployment Wv changed between load and use")
    if file_sha256(Path(DEPLOYMENT_WV_SCALE_PATH)) != wv_bundle["scale_sha256"]:
        raise RuntimeError("deployment Wv scale changed between load and use")
    selector = IntegerSelectorScorer(projection.Wt, wv_bundle["wv_q"])

    samples = limit_each_task(
        [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        args.limit_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = (
        resume_state(args.output, mode=args.mode, budget=args.budget)
        if args.mode != "latency"
        else set()
    )
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(
        f"LOAD mode={args.mode} budget={args.budget} samples={len(samples)} "
        f"pending={len(pending)}",
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
    schedule = measurement_schedule(args.mode, args.warmup, args.repeats)
    metadata = {
        "mode": args.mode,
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "dtype": args.dtype,
        "device": args.device,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "python": platform.python_version(),
        "selector_path": SELECTOR_PATH,
        "deployment_seed": DEPLOYMENT_SEED,
        "deployment_checkpoint": checkpoint_meta,
        "deployment_wv": {
            "wv_path": wv_bundle["wv_path"],
            "wv_sha256": wv_bundle["wv_sha256"],
            "scale_path": wv_bundle["scale_path"],
            "scale_sha256": wv_bundle["scale_sha256"],
            "wv_scale": wv_bundle["wv_scale"],
        },
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": args.local_candidate_cap,
        "warmup_per_sample": schedule["warmup"],
        "do_sample": False,
        "custom_greedy_decode": True,
        "selector_software_is_acceleration_claim": False,
    }

    failures = 0
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        for sample_number, sample in enumerate(pending, 1):
            for _ in range(schedule["warmup"]):
                run_one(
                    model,
                    processor,
                    sample,
                    args.manifest,
                    mode=args.mode,
                    budget=args.budget if args.budget is not None else 0,
                    selector=selector,
                    max_new_tokens=args.max_new_tokens,
                    region_grid=args.region_grid,
                    local_candidate_cap=args.local_candidate_cap,
                )
            for repeat in range(schedule["repeats"]):
                record = run_one(
                    model,
                    processor,
                    sample,
                    args.manifest,
                    mode=args.mode,
                    budget=args.budget if args.budget is not None else 0,
                    selector=selector,
                    max_new_tokens=args.max_new_tokens,
                    region_grid=args.region_grid,
                    local_candidate_cap=args.local_candidate_cap,
                )
                record["repeat"] = repeat
                record["warmup"] = False
                record["run"] = metadata
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if args.mode == "smoke":
                    if not all(record["structural_checks"].values()):
                        failures += 1
                if args.mode == "identity" and not record["identity_gate"]["pass"]:
                    failures += 1
                print(
                    f"RESULT mode={args.mode} {sample_number}/{len(pending)} "
                    f"repeat={repeat} id={sample['sample_id']} "
                    f"G={record['full_llm_visual_tokens']} "
                    f"K={record['output_group_count']} "
                    f"vision={record['vision_total_seconds']:.4f}s "
                    f"model={record['model_inference_seconds']:.4f}s "
                    f"output={record['output_text']!r}",
                    flush=True,
                )
    if failures:
        print(f"FAILURES={failures}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
