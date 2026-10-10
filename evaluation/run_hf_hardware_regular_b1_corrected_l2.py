#!/usr/bin/env python3
"""GPU runner for the B1 corrected-L2 / Prompt-ablation screening100 experiment.

One arm at a time (``--arm``), on the fixed B1 cut after vision block 0, K=96,
``region_grid=3``, ``local_candidate_cap=16``, BF16/SDPA, greedy decode,
``max_new_tokens=16`` on the frozen ``datasets/cv_bench_screening100.jsonl``
manifest.  It drives the **unmodified** frozen B1 forward
(:func:`evaluation.hardware_regular_b1_integration.forward_hardware_regular_b1_vision`)
through its existing ``integer_scorer`` hook, so all three arms share one
unified forward, the accepted ``select_hardware_regular_groups`` selection and
the same context behaviour/layout.

Arms
----
* ``corrected_l2_float``  - :func:`evaluation.hardware_regular_b1_corrected_l2.score_corrected_l2_float`
* ``corrected_l2_integer`` - :func:`evaluation.hardware_regular_b1_corrected_l2.score_corrected_l2_integer`
* ``prompt_plus_corrected_l2_integer`` - the accepted integer semantic chain with
  the corrected patch-wise L2, using the frozen seed-20261003 ``Wt``/``Wv_q``.

No training, no tuning, no other score mode, no projection knob for arms 1/2.
The model is loaded strictly offline from a local directory
(``local_files_only=True`` plus the ``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``
environment).  Output is an append-only raw JSONL with safe fail-closed resume:
a stale, partial, duplicated, mixed-arm or hash-inconsistent existing output
aborts instead of silently continuing.

Every timed GPU boundary is drained with ``torch.cuda.synchronize``.  The CPU
selector round trip is reported as its own overhead term and is excluded from
``gpu_vision_seconds``; it is never an acceleration claim.
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

from evaluation.hardware_regular_b1_corrected_l2 import (
    ALLOWED_ARMS,
    ARM_FLOAT,
    ARM_PROMPT_INTEGER,
    CorrectedL2Error,
    array_sha256,
    build_b1_scorer,
)
from evaluation.hardware_regular_b1_integration import (
    SCORE_MODE_INTEGER,
    forward_hardware_regular_b1_vision,
    prepare_b1_multimodal_inputs,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT
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
DEFAULT_LOCAL_CANDIDATE_CAP = 16
DEFAULT_BUDGET = 96
SCREENING_LIMIT_PER_TASK: int | None = None

#: Fields every existing output record must carry to count as complete.
REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "arm",
    "manifest_sha256",
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
    "record_sha256",
)


class ResumeContractError(RuntimeError):
    """Raised when an existing ``--output`` JSONL violates the resume contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself."""

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _sync_clock() -> float:
    """Drain the GPU then read the wall clock: a clean stage boundary."""

    torch.cuda.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# Provenance / checkpoint identity
# ---------------------------------------------------------------------------
def load_arm_checkpoint(arm: str) -> tuple[Any | None, Any | None, dict[str, Any]]:
    """Load the frozen checkpoint for ``arm``; verify pinned hashes.

    Arms 1/2 have no prompt term and therefore no checkpoint.  Arm 3 loads the
    frozen seed-20261003 ``Wt`` and the accepted per-tensor ``Wv_q`` bundle
    through the accepted :mod:`evaluation.current_selector_integer` helpers,
    which re-verify their pinned SHA256 values.
    """

    if arm != ARM_PROMPT_INTEGER:
        return None, None, {
            "checkpoint_sha256": None,
            "wv_sha256": None,
            "wv_scale_sha256": None,
            "deployment_seed": None,
            "projection_npz": None,
        }
    # Import lazily so arms 1/2 never touch the transformers-heavy checkpoint path.
    from evaluation.current_selector_integer import (  # noqa: PLC0415
        load_deployment_projection,
        load_deployment_wv_bundle,
    )

    projection, metadata = load_deployment_projection()
    bundle = load_deployment_wv_bundle()
    Wt = np.asarray(projection.Wt)
    wv_q = np.asarray(bundle["wv_q"], dtype=np.int64)
    if Wt.shape != (128, 2560):
        raise CorrectedL2Error(f"unexpected Wt shape {Wt.shape}")
    if wv_q.shape != (128, 1024):
        raise CorrectedL2Error(f"unexpected Wv_q shape {wv_q.shape}")
    return Wt, wv_q, {
        "checkpoint_sha256": str(metadata["sha256"]),
        "wv_sha256": str(bundle["wv_sha256"]),
        "wv_scale_sha256": str(bundle["scale_sha256"]),
        "deployment_seed": int(metadata["seed"]),
        "projection_npz": str(metadata["path"]),
    }


# ---------------------------------------------------------------------------
# Fail-closed resume validation
# ---------------------------------------------------------------------------
def _expected_resume_strings(
    *, arm: str, manifest_sha256: str, checkpoint_identity: dict[str, Any]
) -> dict[str, str | None]:
    return {
        "arm": arm,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": checkpoint_identity["checkpoint_sha256"],
        "wv_sha256": checkpoint_identity["wv_sha256"],
        "wv_scale_sha256": checkpoint_identity["wv_scale_sha256"],
    }


def validate_resume(
    path: Path,
    samples: Sequence[dict[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
    checkpoint_identity: dict[str, Any],
    budget: int,
    region_grid: int,
    local_candidate_cap: int,
) -> set[str]:
    """Validate an existing output JSONL and return its completed sample ids.

    A missing or empty file yields the empty set.  Otherwise every non-blank
    line must parse to a JSON object with a unique ``sample_id`` drawn from the
    current post-``--limit-per-task`` ``samples``; every record must agree with
    the current run's arm / manifest hash / checkpoint hash, carry the frozen
    K / grid / cap, be structurally complete, satisfy the per-record K=G/grid
    invariants and carry a valid ``record_sha256``.  Any violation aborts:
    resume never silently drops, mixes or continues stale work.
    """

    if not path.is_file() or path.stat().st_size == 0:
        return set()
    allowed_ids = {str(sample["sample_id"]) for sample in samples}
    expected = _expected_resume_strings(
        arm=arm, manifest_sha256=manifest_sha256, checkpoint_identity=checkpoint_identity
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
                f"output line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise ResumeContractError(
                f"output line {line_number} must be a JSON object (fail closed)"
            )
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ResumeContractError(
                f"output line {line_number} is missing fields {missing} "
                f"(partial record, fail closed)"
            )
        sample_id = str(record["sample_id"])
        if sample_id in done:
            raise ResumeContractError(
                f"duplicate output sample_id {sample_id!r} (fail closed)"
            )
        if sample_id not in allowed_ids:
            raise ResumeContractError(
                f"output sample_id {sample_id!r} is not in the current samples "
                f"(stale output, fail closed)"
            )
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                raise ResumeContractError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"!= current run {expected_value!r} (fail closed)"
                )
        for field, expected_value in (
            ("requested_budget", int(budget)),
            ("region_grid", int(region_grid)),
            ("local_candidate_cap", int(local_candidate_cap)),
        ):
            try:
                actual_value = int(record.get(field))
            except (TypeError, ValueError) as exc:
                raise ResumeContractError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"is not an integer (fail closed)"
                ) from exc
            if actual_value != expected_value:
                raise ResumeContractError(
                    f"output sample {sample_id!r} {field}={record.get(field)!r} "
                    f"!= current run {expected_value!r} (fail closed)"
                )
        grid = [int(value) for value in record["image_grid_thw"]]
        full_patch_tokens = int(record["full_patch_tokens"])
        full_llm = int(record["full_llm_visual_tokens"])
        output_group_count = int(record["output_group_count"])
        selected = [int(value) for value in record["selected_group_indices"]]
        if len(grid) != 3 or grid[0] != 1:
            raise ResumeContractError(
                f"output sample {sample_id!r} image_grid_thw={grid!r} is not a "
                f"single image (fail closed)"
            )
        if grid[1] * grid[2] != full_patch_tokens:
            raise ResumeContractError(
                f"output sample {sample_id!r} grid {grid} does not match "
                f"full_patch_tokens {full_patch_tokens} (fail closed)"
            )
        if full_llm * MERGE_UNIT != full_patch_tokens:
            raise ResumeContractError(
                f"output sample {sample_id!r} full_llm_visual_tokens {full_llm} "
                f"does not match full_patch_tokens {full_patch_tokens} (fail closed)"
            )
        if len(selected) != output_group_count:
            raise ResumeContractError(
                f"output sample {sample_id!r} has {len(selected)} selected groups "
                f"but output_group_count {output_group_count} (fail closed)"
            )
        if selected and not all(
            later > earlier for earlier, later in zip(selected, selected[1:])
        ):
            raise ResumeContractError(
                f"output sample {sample_id!r} selected_group_indices are not "
                f"strictly ascending (fail closed)"
            )
        if bool(record["identity_bypass"]):
            if output_group_count != full_llm:
                raise ResumeContractError(
                    f"output sample {sample_id!r} identity bypass must emit G="
                    f"{full_llm} groups, got {output_group_count} (fail closed)"
                )
        elif output_group_count != int(budget):
            raise ResumeContractError(
                f"output sample {sample_id!r} non-bypass must emit K={budget}, "
                f"got {output_group_count} (fail closed)"
            )
        expected_record_sha = record_sha256(record)
        if str(record.get("record_sha256")) != expected_record_sha:
            raise ResumeContractError(
                f"output sample {sample_id!r} record_sha256 mismatch (stale or "
                f"corrupted record, fail closed)"
            )
        done.add(sample_id)
    return done


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------
def compute_text_embeddings(model: torch.nn.Module, batch: Any) -> np.ndarray:
    """Frozen prompt text embedding for arm 3 (identical to the B1 runner)."""

    image_token_id = int(model.config.image_token_id)
    text_mask = (batch.attention_mask[0] != 0) & (batch.input_ids[0] != image_token_id)
    with torch.inference_mode():
        selector_text = model.get_input_embeddings()(batch.input_ids[0, text_mask])
    return selector_text.detach().to(torch.float32).cpu().numpy()


def _payload_hashes(payload: dict[str, Any] | None, arm: str) -> dict[str, Any]:
    """Every score/index hash a raw record must store for the given arm."""

    if payload is None:
        return {}
    hashes: dict[str, Any] = {}
    if arm == ARM_FLOAT:
        hashes["patch_norms_sha256"] = array_sha256(payload["patch_norms"])
        hashes["group_scores_sha256"] = array_sha256(payload["group_scores"])
    else:
        hashes["patch_norms_sha256"] = array_sha256(payload["patch_norms"])
        hashes["group_norm_sum_sha256"] = array_sha256(payload["group_norm_sum"])
        hashes["l2_norm_q15_sha256"] = array_sha256(payload["l2_norm_q15"])
        if arm == ARM_PROMPT_INTEGER:
            hashes["semantic_q16_sha256"] = array_sha256(payload["semantic_q16"])
            hashes["semantic_norm_q15_sha256"] = array_sha256(
                payload["semantic_norm_q15"]
            )
            hashes["old_group_mean_l2_norm_q15_sha256"] = array_sha256(
                payload["old_group_mean_l2_norm_q15"]
            )
            hashes["score_q_sha256"] = array_sha256(payload["score_q"])
    return hashes


def run_one(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    manifest: Path,
    *,
    arm: str,
    budget: int,
    region_grid: int,
    local_candidate_cap: int,
    scorer: Any,
    manifest_sha256: str,
    checkpoint_identity: dict[str, Any],
) -> dict[str, Any]:
    """Run one sample for one arm through the unified frozen B1 forward."""

    if arm not in ALLOWED_ARMS:
        raise ValueError(f"arm must be one of {ALLOWED_ARMS}, got {arm!r}")
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

    text_embeddings: np.ndarray | None = None
    prompt_text_embedding_seconds = 0.0
    if arm == ARM_PROMPT_INTEGER:
        prompt_text_embedding_started = _sync_clock()
        text_embeddings = compute_text_embeddings(model, batch)
        prompt_text_embedding_seconds = (
            _sync_clock() - prompt_text_embedding_started
        )

    torch.cuda.reset_peak_memory_stats()
    model_inference_started = _sync_clock()
    with torch.inference_mode():
        vision = forward_hardware_regular_b1_vision(
            model.model.visual,
            batch.pixel_values,
            batch.image_grid_thw,
            budget=budget,
            score_mode=SCORE_MODE_INTEGER,
            text_embeddings=text_embeddings,
            integer_scorer=scorer,
            region_grid=region_grid,
            local_candidate_cap=local_candidate_cap,
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
    if vision_total_seconds != (
        vision.vision_preamble_seconds
        + vision.block0_seconds
        + vision.selector_cpu_seconds
        + vision.remaining_vision_seconds
    ):
        raise RuntimeError("vision timing segments do not sum to the vision total")

    payload = vision.integer_payload
    payload_hashes = _payload_hashes(payload, arm)
    selected = np.asarray(vision.group_indices, dtype=np.int64)
    patch_indices = np.asarray(vision.patch_indices, dtype=np.int64).reshape(-1)
    source_type = np.asarray(vision.source_type, dtype=np.int64)
    region_id = np.asarray(vision.region_id, dtype=np.int64)
    valid = np.asarray(vision.valid, dtype=bool)
    combined_scores = (
        np.asarray(payload["combined_scores"], dtype=np.float64)
        if payload is not None
        else None
    )

    record: dict[str, Any] = {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "arm": arm,
        "score_mode": SCORE_MODE_INTEGER,
        "integration_score_mode": SCORE_MODE_INTEGER,
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
        "payload_score_hashes": payload_hashes,
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
    }
    record["record_sha256"] = record_sha256(record)
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
        "--arm",
        choices=ALLOWED_ARMS,
        required=True,
        help="One corrected-L2 arm; run exactly one arm per output file.",
    )
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--region-grid", type=int, default=DEFAULT_REGION_GRID)
    parser.add_argument(
        "--local-candidate-cap", type=int, default=DEFAULT_LOCAL_CANDIDATE_CAP
    )
    parser.add_argument("--limit-per-task", type=int, default=SCREENING_LIMIT_PER_TASK)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.model.is_dir():
        raise RuntimeError(
            f"--model must be a local directory (strict offline): {args.model}"
        )
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this experiment requires a CUDA device")

    manifest_sha256 = file_sha256(args.manifest)
    Wt, wv_q, checkpoint_identity = load_arm_checkpoint(args.arm)
    scorer = build_b1_scorer(args.arm, Wt=Wt, wv_q=wv_q, budgets=(args.budget,))

    samples = limit_each_task(
        [
            json.loads(line)
            for line in args.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        args.limit_per_task,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = validate_resume(
        args.output,
        samples,
        arm=args.arm,
        manifest_sha256=manifest_sha256,
        checkpoint_identity=checkpoint_identity,
        budget=args.budget,
        region_grid=args.region_grid,
        local_candidate_cap=args.local_candidate_cap,
    )
    pending = [sample for sample in samples if sample["sample_id"] not in done]
    print(
        f"LOAD arm={args.arm} samples={len(samples)} budget={args.budget} "
        f"region_grid={args.region_grid} local_cap={args.local_candidate_cap} "
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
        "arm": args.arm,
        "integration_score_mode": SCORE_MODE_INTEGER,
        "prune_after_layer": 0,
        "budget": args.budget,
        "region_grid": args.region_grid,
        "local_candidate_cap": args.local_candidate_cap,
        "manifest_sha256": manifest_sha256,
        "do_sample": False,
        "custom_greedy_decode": True,
        "original_position_ids_preserved": True,
        "training": False,
        "uses_b1_trained_wv": args.arm == ARM_PROMPT_INTEGER,
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
            "model_inference_seconds": (
                "vision_total_seconds + llm_prepare_seconds + "
                "llm_prefill_seconds + decoder_seconds"
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
                arm=args.arm,
                budget=args.budget,
                region_grid=args.region_grid,
                local_candidate_cap=args.local_candidate_cap,
                scorer=scorer,
                manifest_sha256=manifest_sha256,
                checkpoint_identity=checkpoint_identity,
            )
            record["run"] = metadata
            # The record digest is computed before attaching the run metadata;
            # recompute so the stored digest covers the full written record.
            record["record_sha256"] = record_sha256(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "RESULT "
                f"{sample_number}/{len(pending)} id={sample['sample_id']} "
                f"arm={args.arm} G={record['full_llm_visual_tokens']} "
                f"K={record['output_group_count']} "
                f"C={record['context_group_count']} "
                f"vision={record['vision_total_seconds']:.4f}s "
                f"model={record['model_inference_seconds']:.4f}s "
                f"output={record['output_text']!r}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
