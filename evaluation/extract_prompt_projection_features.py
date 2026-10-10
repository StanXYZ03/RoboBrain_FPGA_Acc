#!/usr/bin/env python3
"""Offline feature extractor for B1 prompt-projection distillation.

The extractor runs the **full dense vision tower** once per *image* to obtain
the teacher target, and uses a removable forward hook on ``visual.blocks[0]`` to
capture the student input.  It never runs the language model and never trains
anything.  Each sample is written to its own ``.npz`` cache file; a JSONL index
makes the run resumable.  The SHA256 of the source image file bytes is computed
here and written to both the npz metadata and the index record as
``source_image_sha256``; the trainer groups samples by that hash so every prompt
of one image stays in a single split.

Image reuse contract
--------------------
For one resolved image path + ``source_image_sha256`` the following outputs are
pure functions of the image and are therefore shared across every prompt of that
image:

* the processor image tensors (``pixel_values`` / ``image_grid_thw``);
* the full dense vision-tower + merger embedding ``merged_embeddings[G,2560]``;
* the block-0 capture and the per-group mean ``group_mean[G,1024]``;
* the merge grid (``grid_h``, ``grid_w``).

The prompt embedding ``prompt_embeddings[T,2560]`` and the teacher max-cosine
target are computed **per record** and are never reused.  The manifest is *not*
assumed to be image-contiguous (the frozen clean-inductive manifest lists 472
records for 236 images in two non-adjacent passes), so samples are grouped by
``(resolved path, source_image_sha256)`` across the whole manifest while the
emitted JSONL index records are kept in exactly manifest order by an ordered
writer.  Only one image's tensors are ever resident at a time: a group's capture
is dropped (and, on CUDA, the allocator cache released) before the next group is
touched, so tensors are never accumulated across all images.

Cache contract (one sample per npz, no full dense hidden is ever stored):

======  ==========================  ==========================================
key     dtype / shape               meaning
======  ==========================  ==========================================
group_mean               float16 [G,1024]   block-0 per-group 4-patch mean
prompt_embeddings        float16 [T,2560]   raw prompt token embeddings
teacher_scores           float32 [G]        max cosine vs prompt tokens
teacher_scores_normalized float32 [G]       min-max of teacher_scores
teacher_soft_target      float32 [G]        softmax(normalized / temperature)
meta_<key>               scalar             sample/grid/hash metadata
======  ==========================  ==========================================

Before any model load the extractor runs the mandatory GPU gate: it reads
``nvidia-smi`` and requires that every GPU compute process belongs to the
current user and that enough memory is free.  On failure it writes the audit
JSON and exits without loading a model.

The definitions live in ``evaluation.prompt_projection_distillation``; this
module only orchestrates I/O.  The execution layer
(:func:`run_prompt_projection_extraction`) takes injected callables for
resolution, hashing, image loading, vision capture and prompt embedding, so it
is fully testable on CPU with fake processors/models and no CUDA.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.metrics import load_jsonl
from evaluation.prompt_projection_distillation import (
    DEFAULT_CONFIG_FILENAME,
    FEATURE_DIM,
    PROMPT_DIM,
    DistillationContractError,
    canonical_feature_index_record,
    config_sha256,
    image_file_sha256,
    load_distillation_config,
    save_feature_cache,
    teacher_target,
    validate_pinned_manifest,
)

DEFAULT_REQUIRED_FREE_MIB = 20480
# Shared-GPU policy (opt-in via --allow-shared-gpu).  The strict legacy gate
# above stays the default: a non-owner compute process still blocks unless the
# operator explicitly accepts coexisting with other users' CUDA contexts.
DEFAULT_MIN_FREE_MIB = 14336
DEFAULT_MAX_UTIL_PCT = 80.0
DEFAULT_UTIL_SAMPLES = 5
DEFAULT_UTIL_SAMPLE_INTERVAL_S = 1.0
DEFAULT_POST_LOAD_MIN_FREE_MIB = 4096
DEFAULT_AUDIT_NAME = "gpu-audit.json"
DEFAULT_INDEX_NAME = "index.jsonl"

CommandRunner = Callable[[list[str]], tuple[int, str, str]]
ImageResolver = Callable[[Path, str], Path]
ImageHasher = Callable[[Path], str]
ImageLoader = Callable[[Path], Any]
VisionCaptureFn = Callable[[Mapping[str, Any], Any, Path], Mapping[str, Any]]
PromptEmbeddingFn = Callable[[Mapping[str, Any]], Any]
SaveCacheFn = Callable[..., Mapping[str, Any]]
IndexRecordFn = Callable[..., Mapping[str, Any]]


# ---------------------------------------------------------------------------
# GPU preflight gate (read-only)
# ---------------------------------------------------------------------------


def _default_command_runner(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, capture_output=True, text=True)
    return completed.returncode, completed.stdout, completed.stderr


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sample_gpu_utilization(
    runner: CommandRunner,
    *,
    samples: int,
    interval_s: float,
    sleep_fn: Callable[[float], None],
) -> list[int]:
    """Read-only GPU utilization samples (percent) for the shared policy."""
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    values: list[int] = []
    for index in range(max(int(samples), 1)):
        if index:
            sleep_fn(float(interval_s))
        code, out, _err = runner(command)
        if code != 0 or not out.strip():
            continue
        try:
            values.append(int(float(out.strip().splitlines()[0].strip())))
        except ValueError:
            continue
    return values


def gpu_preflight_audit(
    *,
    required_free_mib: int = DEFAULT_REQUIRED_FREE_MIB,
    current_user: str | None = None,
    command_runner: CommandRunner | None = None,
    allow_shared_gpu: bool = False,
    min_free_mib: int = DEFAULT_MIN_FREE_MIB,
    max_util_pct: float = DEFAULT_MAX_UTIL_PCT,
    util_samples: int = DEFAULT_UTIL_SAMPLES,
    util_sample_interval_s: float = DEFAULT_UTIL_SAMPLE_INTERVAL_S,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Read-only GPU audit returning a machine-readable report.

    Default (legacy strict) policy: ``status`` is ``PASS`` only when
    ``nvidia-smi`` works, at least ``required_free_mib`` MiB are free and every
    compute process is owned by ``current_user``.

    ``allow_shared_gpu=True`` opts into the shared-resource policy: other users'
    compute processes no longer block, but at least ``min_free_mib`` MiB must be
    free and the averaged utilization over ``util_samples`` samples spaced by
    ``util_sample_interval_s`` seconds must stay below ``max_util_pct``.  The
    non-owner process list is still reported.  Nothing is killed, paused or
    modified in either mode.
    """
    runner = command_runner or _default_command_runner
    user = current_user or getpass.getuser()
    audit: dict[str, Any] = {
        "checked_at_utc": _utc_now(),
        "current_user": user,
        "required_free_mib": int(required_free_mib),
        "allow_shared_gpu": bool(allow_shared_gpu),
        "min_free_mib": int(min_free_mib),
        "max_util_pct": float(max_util_pct),
        "commands": [],
        "passed": False,
    }

    gpu_cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,"
        "utilization.gpu,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = runner(gpu_cmd)
    audit["commands"].append({"cmd": gpu_cmd, "returncode": code})
    if code != 0 or not out.strip():
        audit["status"] = "BLOCKED_NVIDIA_SMI_FAILED"
        audit["nvidia_smi_stderr"] = err.strip()
        return audit
    fields = [value.strip() for value in out.strip().splitlines()[0].split(",")]
    if len(fields) < 7:
        audit["status"] = "BLOCKED_NVIDIA_SMI_UNPARSABLE"
        audit["nvidia_smi_stdout"] = out.strip()
        return audit
    gpu = {
        "name": fields[0],
        "memory_total_mib": int(float(fields[1])),
        "memory_used_mib": int(float(fields[2])),
        "memory_free_mib": int(float(fields[3])),
        "utilization_percent": int(float(fields[4])),
        "temperature_c": int(float(fields[5])),
        "driver_version": fields[6],
    }
    audit["gpu"] = gpu

    apps_cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = runner(apps_cmd)
    audit["commands"].append({"cmd": apps_cmd, "returncode": code})
    processes: list[dict[str, Any]] = []
    if code == 0:
        for line in out.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [value.strip() for value in line.split(",")]
            if len(parts) < 3:
                continue
            pid = int(parts[0])
            ps_cmd = ["ps", "-o", "user=", "-p", str(pid)]
            ps_code, ps_out, _ = runner(ps_cmd)
            owner = ps_out.strip().splitlines()[0].strip() if ps_out.strip() else "unknown"
            processes.append(
                {
                    "pid": pid,
                    "process_name": parts[1],
                    "used_memory_mib": int(float(parts[2])),
                    "user": owner,
                    "owner_lookup_returncode": ps_code,
                }
            )
    else:
        audit["compute_apps_stderr"] = err.strip()
    audit["compute_processes"] = processes

    non_owner = [proc for proc in processes if proc["user"] != user]
    audit["non_owner_compute_processes"] = non_owner

    if allow_shared_gpu:
        util_values = _sample_gpu_utilization(
            runner,
            samples=util_samples,
            interval_s=util_sample_interval_s,
            sleep_fn=sleep_fn,
        )
        util_avg = float(sum(util_values) / len(util_values)) if util_values else None
        audit["shared_gpu_policy"] = {
            "enabled": True,
            "min_free_mib": int(min_free_mib),
            "max_util_pct": float(max_util_pct),
            "util_samples_requested": int(util_samples),
            "util_samples_observed": len(util_values),
            "util_samples_percent": util_values,
            "util_avg_percent": util_avg,
            "non_owner_processes_do_not_block": True,
            "note": "other users' CUDA contexts are tolerated by operator opt-in",
        }
        if gpu["memory_free_mib"] < int(min_free_mib):
            audit["status"] = "BLOCKED_SHARED_GPU_LOW_MEMORY"
            return audit
        if util_avg is None:
            audit["status"] = "BLOCKED_SHARED_GPU_UTILIZATION_UNREADABLE"
            return audit
        if util_avg >= float(max_util_pct):
            audit["status"] = "BLOCKED_SHARED_GPU_HIGH_UTILIZATION"
            return audit
        audit["status"] = "PASS"
        audit["passed"] = True
        return audit

    if non_owner:
        audit["status"] = "BLOCKED_NON_OWNER_COMPUTE_PROCESS"
        return audit
    if gpu["memory_free_mib"] < int(required_free_mib):
        audit["status"] = "BLOCKED_INSUFFICIENT_FREE_MEMORY"
        return audit
    audit["status"] = "PASS"
    audit["passed"] = True
    return audit


def gpu_runtime_snapshot(
    *, command_runner: CommandRunner | None = None
) -> dict[str, Any]:
    """One read-only memory/utilization snapshot used after load and extraction.

    This is a single ``nvidia-smi`` call (no polling loop) so the heavy path
    never spins on the device while other users share it.
    """
    runner = command_runner or _default_command_runner
    snapshot: dict[str, Any] = {"checked_at_utc": _utc_now()}
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    code, out, err = runner(command)
    if code != 0 or not out.strip():
        snapshot["status"] = "AUDIT_FAILED"
        snapshot["stderr"] = err.strip()
        return snapshot
    fields = [value.strip() for value in out.strip().splitlines()[0].split(",")]
    if len(fields) < 4:
        snapshot["status"] = "AUDIT_UNPARSABLE"
        snapshot["stdout"] = out.strip()
        return snapshot
    snapshot.update(
        {
            "status": "OK",
            "memory_total_mib": int(float(fields[0])),
            "memory_used_mib": int(float(fields[1])),
            "memory_free_mib": int(float(fields[2])),
            "utilization_percent": int(float(fields[3])),
        }
    )
    return snapshot


# ---------------------------------------------------------------------------
# Vision capture and prompt embeddings
# ---------------------------------------------------------------------------


def _unwrap_module_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise DistillationContractError(
            f"hook captured a non-tensor output of type {type(output)!r}"
        )
    return output


def dense_vision_with_block0_capture(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
) -> dict[str, Any]:
    """Run the full vision tower once and capture block-0 output via a hook.

    The hook is always removed, including on failure.  ``group_count`` is
    verified to be consistent between the captured block-0 patch rows, the
    merged embedding rows and ``grid_h * grid_w``.
    """
    if grid_thw.shape != (1, 3) or int(grid_thw[0, 0].item()) != 1:
        raise DistillationContractError("exactly one still image is supported")
    if not hasattr(visual, "blocks") or len(visual.blocks) < 1:
        raise DistillationContractError("the visual tower must expose blocks")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["block0"] = _unwrap_module_output(output).detach()

    handle = visual.blocks[0].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            try:
                result = visual(pixel_values, grid_thw=grid_thw)
            except TypeError:
                result = visual(pixel_values)
    finally:
        handle.remove()
    if "block0" not in captured:
        raise DistillationContractError("block-0 forward hook never fired")
    if isinstance(result, (tuple, list)):
        merged = result[0]
    else:
        merged = result
    if not isinstance(merged, torch.Tensor):
        raise DistillationContractError("vision tower did not return merged embeddings")

    merge_size = int(visual.spatial_merge_size)
    merge_unit = merge_size * merge_size
    block0 = captured["block0"]
    if block0.ndim != 2:
        raise DistillationContractError(
            f"block-0 output must be 2-D, got shape {tuple(block0.shape)}"
        )
    if merged.ndim != 2:
        raise DistillationContractError(
            f"merged embeddings must be 2-D, got shape {tuple(merged.shape)}"
        )
    if block0.shape[0] % merge_unit != 0:
        raise DistillationContractError("block-0 patch rows are not merge-aligned")
    group_count = block0.shape[0] // merge_unit
    if merged.shape[0] != group_count:
        raise DistillationContractError(
            f"group mismatch: block0 implies {group_count}, merged has "
            f"{merged.shape[0]}"
        )
    grid_h = int(grid_thw[0, 1].item()) // merge_size
    grid_w = int(grid_thw[0, 2].item()) // merge_size
    if grid_h * grid_w != group_count:
        raise DistillationContractError(
            f"grid {grid_h}x{grid_w} does not match {group_count} groups"
        )
    return {
        "merged_embeddings": merged.float(),
        "block0_patches": block0.float(),
        "group_count": group_count,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "merge_unit": merge_unit,
    }


def group_mean_from_block0(block0_patches: torch.Tensor, merge_unit: int) -> torch.Tensor:
    """Per-group mean of the captured block-0 patches -> ``[G, 1024]``."""
    if block0_patches.ndim != 2 or block0_patches.shape[0] % merge_unit != 0:
        raise DistillationContractError("block-0 patches are not merge-aligned")
    groups = block0_patches.reshape(-1, merge_unit, block0_patches.shape[1])
    return groups.mean(dim=1)


def prompt_token_ids(tokenizer: Any, prompt_text: str) -> torch.Tensor:
    """Tokenise only the raw user prompt text, ``add_special_tokens=False``."""
    encoded = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    ids = encoded["input_ids"][0]
    if ids.numel() == 0:
        raise DistillationContractError("raw prompt produced zero tokens")
    return ids.long()


def prompt_embeddings_from_text(
    embedding_layer: Any, tokenizer: Any, prompt_text: str
) -> torch.Tensor:
    """Embed the raw prompt tokens through the model input-embedding table."""
    ids = prompt_token_ids(tokenizer, prompt_text).to(embedding_layer.weight.device)
    with torch.inference_mode():
        return embedding_layer(ids).detach().float()


# ---------------------------------------------------------------------------
# Image-group planning (one vision forward per image, many prompts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageGroup:
    """All manifest samples that share one resolved path and content hash."""

    resolved_image_path: str
    source_image_sha256: str
    manifest_indices: tuple[int, ...]
    samples: tuple[Mapping[str, Any], ...]
    source_image_paths: tuple[str, ...]


def validate_vision_capture(capture: Mapping[str, Any]) -> None:
    """Reject a capture whose shared image-dependent tensors are inconsistent."""
    required = ("merged_embeddings", "group_mean", "grid_h", "grid_w")
    missing = [key for key in required if key not in capture]
    if missing:
        raise DistillationContractError(f"vision capture missing keys: {missing}")
    merged = capture["merged_embeddings"]
    mean = capture["group_mean"]
    if merged.ndim != 2:
        raise DistillationContractError("merged_embeddings must be 2-D")
    if mean.ndim != 2:
        raise DistillationContractError("group_mean must be 2-D")
    if merged.shape[1] != PROMPT_DIM:
        raise DistillationContractError(
            f"merged embedding dim must be {PROMPT_DIM}, got {merged.shape[1]}"
        )
    if mean.shape[1] != FEATURE_DIM:
        raise DistillationContractError(
            f"group_mean dim must be {FEATURE_DIM}, got {mean.shape[1]}"
        )
    if merged.shape[0] != mean.shape[0]:
        raise DistillationContractError("merged and group_mean group counts differ")
    if int(capture["grid_h"]) * int(capture["grid_w"]) != mean.shape[0]:
        raise DistillationContractError("grid does not match group count")


def plan_image_groups(
    samples: Sequence[Mapping[str, Any]],
    *,
    resolve_image_fn: ImageResolver,
    image_sha256_fn: ImageHasher,
    completed_records: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_path: Path | str = "",
) -> list[ImageGroup]:
    """Group manifest samples by ``(resolved path, source_image_sha256)``.

    Groups appear in first-occurrence (manifest) order and each group keeps its
    samples in manifest order.  The current content hash is compared with the
    hash recorded for any already-completed sample: a change is a hard error, so
    stale vision features are never silently reused on resume.
    """
    completed = dict(completed_records or {})
    manifest = Path(manifest_path) if manifest_path else Path(".")
    groups: dict[tuple[str, str], dict[str, list[Any]]] = {}
    path_to_sha: dict[str, str] = {}
    for position, sample in enumerate(samples):
        sample_id = str(sample["sample_id"])
        image_path = resolve_image_fn(manifest, sample["image"])
        resolved = str(Path(image_path).resolve())
        digest = image_sha256_fn(image_path)
        prior = path_to_sha.get(resolved)
        if prior is not None and prior != digest:
            raise DistillationContractError(
                "source image content changed during extraction for "
                f"{resolved}: {prior} -> {digest}"
            )
        path_to_sha[resolved] = digest
        cached = completed.get(sample_id)
        if cached is not None:
            recorded = str(cached.get("source_image_sha256", "")).strip().lower()
            if recorded != digest:
                raise DistillationContractError(
                    "source image sha256 changed for image "
                    f"{resolved}: cached {recorded!r} != current {digest!r}; "
                    "refusing to reuse stale vision features"
                )
        key = (resolved, digest)
        entry = groups.get(key)
        if entry is None:
            entry = {"indices": [], "samples": [], "paths": []}
            groups[key] = entry
        entry["indices"].append(position)
        entry["samples"].append(dict(sample))
        entry["paths"].append(str(image_path))
    return [
        ImageGroup(
            resolved_image_path=key[0],
            source_image_sha256=key[1],
            manifest_indices=tuple(entry["indices"]),
            samples=tuple(entry["samples"]),
            source_image_paths=tuple(entry["paths"]),
        )
        for key, entry in groups.items()
    ]


class _OrderedIndexWriter:
    """Append index records in manifest order even when groups arrive out of order.

    Group processing follows first-occurrence order, so a group's later record
    (e.g. a ``pointing`` prompt far down the manifest) can be ready before an
    earlier position from another group.  Records are buffered and flushed in
    exactly the pre-computed pending-position order, keeping the JSONL index
    deterministic and manifest ordered without ever accumulating GPU tensors.
    """

    def __init__(self, stream: Any, ordered_positions: Sequence[int]) -> None:
        self._stream = stream
        self._order = list(ordered_positions)
        self._rank = {position: index for index, position in enumerate(self._order)}
        self._cursor = 0
        self._buffer: dict[int, Mapping[str, Any]] = {}

    def submit(self, position: int, record: Mapping[str, Any]) -> None:
        if position not in self._rank:
            raise DistillationContractError(
                f"index position {position} is not a pending manifest position"
            )
        self._buffer[position] = record
        self._flush()

    def _flush(self) -> None:
        while self._cursor < len(self._order):
            position = self._order[self._cursor]
            if position not in self._buffer:
                break
            self._stream.write(
                json.dumps(self._buffer.pop(position), ensure_ascii=False) + "\n"
            )
            self._cursor += 1

    def finish(self) -> None:
        self._flush()
        if self._cursor != len(self._order):
            missing = self._order[self._cursor]
            raise DistillationContractError(
                f"missing index record for manifest position {missing}"
            )


def _load_index_records(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.is_file():
        return {}
    return {
        str(record["sample_id"]): dict(record) for record in load_jsonl(index_path)
    }


def _sample_metadata(
    sample: Mapping[str, Any],
    *,
    image_path: str,
    image_sha256: str,
    grid_h: int,
    grid_w: int,
    temperature: float,
    config_hash: str,
    manifest_path: Path | str,
    extraction_meta: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": str(sample["sample_id"]),
        "task": str(sample["task"]),
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "temperature": temperature,
        "dense_vision_blocks": int(extraction_meta.get("dense_vision_blocks", 24)),
        "capture_layer": int(extraction_meta.get("capture_layer", 0)),
        "prompt_tokenization": str(
            extraction_meta.get("prompt_tokenization", "sample_prompt_only")
        ),
        "add_special_tokens": bool(extraction_meta.get("add_special_tokens", False)),
        "source_manifest": str(manifest_path),
        "source_image": str(image_path),
        "source_image_sha256": image_sha256,
        "config_sha256": config_hash,
    }


def run_prompt_projection_extraction(
    *,
    samples: Sequence[Mapping[str, Any]],
    manifest_path: Path | str,
    out_dir: Path | str,
    index_path: Path | str,
    temperature: float,
    config_hash: str,
    resolve_image_fn: ImageResolver,
    image_sha256_fn: ImageHasher,
    image_loader_fn: ImageLoader,
    vision_capture_fn: VisionCaptureFn,
    prompt_embeddings_fn: PromptEmbeddingFn,
    completed_records: Mapping[str, Mapping[str, Any]] | None = None,
    overwrite: bool = False,
    save_cache_fn: SaveCacheFn = save_feature_cache,
    index_record_fn: IndexRecordFn = canonical_feature_index_record,
    extraction_meta: Mapping[str, Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
    progress: Callable[[str], None] | None = None,
    release_fn: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """CPU-testable extraction core with injected model/processor callables.

    One :func:`vision_capture_fn` call is made per image that has at least one
    pending prompt; every pending record gets its own ``prompt_embeddings_fn``
    call and its own npz.  A group with no pending prompt performs no vision
    forward.  The capture is released after each group.
    """
    out_dir = Path(out_dir)
    index_path = Path(index_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = {} if overwrite else dict(completed_records or {})
    extraction = dict(extraction_meta or {})

    groups = plan_image_groups(
        samples,
        resolve_image_fn=resolve_image_fn,
        image_sha256_fn=image_sha256_fn,
        completed_records=completed,
        manifest_path=manifest_path,
    )
    ordered_pending = [
        position
        for position, sample in enumerate(samples)
        if str(sample["sample_id"]) not in completed
    ]

    vision_forwards = 0
    prompt_calls = 0
    records_written = 0
    images_processed = 0
    index_mode = "w" if overwrite else "a"
    started = clock()
    index_stream = index_path.open(index_mode, encoding="utf-8", buffering=1)
    writer = _OrderedIndexWriter(index_stream, ordered_pending)
    try:
        for group in groups:
            pending = [
                (position, sample, image_path)
                for position, sample, image_path in zip(
                    group.manifest_indices,
                    group.samples,
                    group.source_image_paths,
                )
                if str(sample["sample_id"]) not in completed
            ]
            if not pending:
                continue
            images_processed += 1
            image = image_loader_fn(Path(group.resolved_image_path))
            capture: Mapping[str, Any] | None = None
            try:
                _first_position, first_sample, first_path = pending[0]
                capture = vision_capture_fn(
                    first_sample, image, Path(first_path)
                )
                validate_vision_capture(capture)
                vision_forwards += 1
                merged = capture["merged_embeddings"]
                grid_h = int(capture["grid_h"])
                grid_w = int(capture["grid_w"])
                group_mean = capture["group_mean"]
                for position, sample, image_path in pending:
                    record_started = clock()
                    sample_id = str(sample["sample_id"])
                    prompt_embeddings = prompt_embeddings_fn(sample)
                    prompt_calls += 1
                    if prompt_embeddings.ndim != 2:
                        raise DistillationContractError(
                            "prompt embeddings must be 2-D"
                        )
                    if prompt_embeddings.shape[1] != merged.shape[1]:
                        raise DistillationContractError(
                            "prompt and merged embedding dimensions differ"
                        )
                    target = teacher_target(
                        merged.cpu().numpy(),
                        prompt_embeddings.cpu().numpy(),
                        temperature,
                    )
                    npz_name = f"{sample_id}.npz"
                    metadata = _sample_metadata(
                        sample,
                        image_path=image_path,
                        image_sha256=group.source_image_sha256,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        temperature=temperature,
                        config_hash=config_hash,
                        manifest_path=manifest_path,
                        extraction_meta=extraction,
                    )
                    write_result = save_cache_fn(
                        out_dir / npz_name,
                        group_mean=group_mean,
                        prompt_embeddings=prompt_embeddings,
                        teacher_raw=target["raw"],
                        teacher_normalized=target["normalized"],
                        teacher_soft=target["soft"],
                        metadata=metadata,
                    )
                    record = index_record_fn(
                        sample_id=sample_id,
                        task=str(sample["task"]),
                        npz_name=npz_name,
                        cache={
                            "teacher_scores": target["raw"],
                            "group_mean": group_mean,
                            "prompt_embeddings": prompt_embeddings,
                            "grid_h": grid_h,
                            "grid_w": grid_w,
                            "sample_sha256": write_result["sample_sha256"],
                        },
                        config_hash=config_hash,
                        source_image_sha256=group.source_image_sha256,
                    )
                    record["npz_sha256"] = write_result["sample_sha256"]
                    record["bytes"] = write_result["bytes"]
                    record["extract_seconds"] = clock() - record_started
                    writer.submit(position, record)
                    records_written += 1
                    if progress is not None:
                        progress(
                            f"RESULT {position + 1}/{len(samples)} {sample_id} "
                            f"G={record['group_count']} "
                            f"T={record['prompt_token_count']}"
                        )
            finally:
                # Drop this image's tensors immediately; nothing accumulates.
                capture = None
                image = None
                if release_fn is not None:
                    release_fn()
        writer.finish()
    finally:
        index_stream.close()

    return {
        "status": "DONE",
        "records_written": records_written,
        "vision_forwards": vision_forwards,
        "prompt_embeddings": prompt_calls,
        "images_total": len(groups),
        "images_processed": images_processed,
        "index_path": str(index_path),
        "extract_seconds": clock() - started,
    }


# ---------------------------------------------------------------------------
# Main extraction loop
# ---------------------------------------------------------------------------


def _limit_each_task(samples: list[dict[str, Any]], count: int | None) -> list[dict[str, Any]]:
    if count is None:
        return samples
    kept: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for sample in samples:
        task = str(sample["task"])
        if seen.get(task, 0) < count:
            kept.append(sample)
            seen[task] = seen.get(task, 0) + 1
    return kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / DEFAULT_CONFIG_FILENAME,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/token_selector/prompt_projection_features"),
    )
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--gpu-audit-json", type=Path, default=None)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--required-free-mib", type=int, default=DEFAULT_REQUIRED_FREE_MIB)
    parser.add_argument(
        "--allow-shared-gpu",
        action="store_true",
        help=(
            "opt in to the shared-GPU policy: tolerate other users' compute "
            "contexts, requiring only --min-free-mib free and a 5 s average "
            "utilization below --max-util-pct; without this flag the legacy "
            "strict owner gate is unchanged"
        ),
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="shared policy: minimum free memory in MiB (default 14336)",
    )
    parser.add_argument(
        "--max-util-pct",
        type=float,
        default=DEFAULT_MAX_UTIL_PCT,
        help="shared policy: maximum 5 s average GPU utilization percent (default 80)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu-audit-only", action="store_true")
    parser.add_argument(
        "--validate-manifest-only",
        action="store_true",
        help=(
            "CPU-only: check that --manifest resolves to the frozen config pin "
            "(relative path + sha256) and exit without any GPU/model access"
        ),
    )
    args = parser.parse_args(argv)

    config_path = args.config
    try:
        config = load_distillation_config(config_path)
    except DistillationContractError as exc:
        print(
            json.dumps(
                {"status": "FAIL", "stage": "config", "error": str(exc)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    # Fail-closed governance gate: the extraction source must equal the frozen
    # config pin before any GPU audit, model load or file write happens.
    try:
        pin_result = validate_pinned_manifest(
            config, manifest_path=args.manifest
        )
    except DistillationContractError as exc:
        pin_result = {
            "status": "FAIL",
            "stage": "manifest_pin",
            "error": str(exc),
            "manifest_path": str(Path(args.manifest).resolve()),
        }
    if args.validate_manifest_only:
        pin_result["mode"] = "validate_manifest_only"
        pin_result["gpu_used"] = False
        pin_result["model_loaded"] = False
        print(json.dumps(pin_result, ensure_ascii=False, indent=2), flush=True)
        return 0 if pin_result.get("status") == "PASS" else 1
    if pin_result.get("status") != "PASS":
        print(
            json.dumps(
                {"status": "FAIL", "stage": "manifest_pin", **pin_result},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    if args.model is None:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "stage": "args",
                    "error": "--model is required unless --validate-manifest-only is set",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 3

    config_hash = config_sha256(config_path)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index or (out_dir / DEFAULT_INDEX_NAME)
    audit_path = args.gpu_audit_json or (out_dir / DEFAULT_AUDIT_NAME)

    audit = gpu_preflight_audit(
        required_free_mib=args.required_free_mib,
        allow_shared_gpu=args.allow_shared_gpu,
        min_free_mib=args.min_free_mib,
        max_util_pct=args.max_util_pct,
    )
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"GPU_AUDIT {audit['status']} -> {audit_path}", flush=True)
    if not audit["passed"]:
        print(
            "BLOCKED before model load; no CUDA model was created and no sample "
            "was extracted.",
            flush=True,
        )
        return 2
    if args.gpu_audit_only:
        return 0

    # Heavy imports stay inside main so the module is importable without a full
    # model stack (and so the gate runs first).
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from evaluation.run_hf_token_reduction import (
        make_messages,
        processor_inputs,
        resolve_image,
    )

    temperature = float(config["temperature"])
    extraction = config["extraction"]
    samples = load_jsonl(args.manifest)
    samples = _limit_each_task(samples, args.limit_per_task)
    completed_records = {} if args.overwrite else _load_index_records(index_path)

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model),
            dtype=torch.bfloat16,
            device_map=args.device,
            local_files_only=True,
        ).eval()
        model.requires_grad_(False)
        processor = AutoProcessor.from_pretrained(
            str(args.model), local_files_only=True
        )
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # includes CUDA OOM
        stage = {
            "stage": "model_load_failed",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "checked_at_utc": _utc_now(),
            "shared_gpu_policy_active": bool(args.allow_shared_gpu),
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        (out_dir / "gpu-audit-model-load-failed.json").write_text(
            json.dumps(stage, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"MODEL_LOAD_FAILED {stage['error']}", flush=True)
        return 4

    after_load = gpu_runtime_snapshot()
    after_load["stage"] = "after_model_load"
    after_load["min_free_mib"] = DEFAULT_POST_LOAD_MIN_FREE_MIB
    if torch.cuda.is_available():
        after_load["torch_cuda_memory_allocated_mib"] = round(
            torch.cuda.memory_allocated() / (1024**2), 1
        )
        after_load["torch_cuda_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024**2), 1
        )
    after_load["passed"] = bool(
        after_load.get("status") == "OK"
        and int(after_load.get("memory_free_mib", 0)) >= DEFAULT_POST_LOAD_MIN_FREE_MIB
    )
    (out_dir / "gpu-audit-after-load.json").write_text(
        json.dumps(after_load, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"GPU_AFTER_LOAD free={after_load.get('memory_free_mib')}MiB "
        f"passed={after_load['passed']}",
        flush=True,
    )
    if not after_load["passed"]:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            "MODEL_RELEASED: fewer than "
            f"{DEFAULT_POST_LOAD_MIN_FREE_MIB} MiB free after load; no sample extracted.",
            flush=True,
        )
        return 4

    visual = model.model.visual
    tokenizer = processor.tokenizer
    embedding_layer = model.get_input_embeddings()

    def _load_image(path: Path) -> Any:
        with Image.open(path) as opened:
            return opened.convert("RGB")

    def _vision_capture(
        sample: Mapping[str, Any], image: Any, image_path: Path
    ) -> dict[str, Any]:
        messages = make_messages(image_path, sample["prompt"])
        batch = processor_inputs(processor, messages, image).to(args.device)
        captured = dense_vision_with_block0_capture(
            visual, batch.pixel_values, batch.image_grid_thw
        )
        # Device-neutral adapter: the shared core consumes numpy via ``.cpu()``
        # and the npz writer calls ``np.asarray`` directly, so materialise the
        # image-shared tensors on the host here.  Values/dtypes are unchanged.
        return {
            "merged_embeddings": captured["merged_embeddings"].detach().cpu(),
            "group_mean": group_mean_from_block0(
                captured["block0_patches"], int(captured["merge_unit"])
            )
            .detach()
            .cpu(),
            "group_count": captured["group_count"],
            "grid_h": captured["grid_h"],
            "grid_w": captured["grid_w"],
            "merge_unit": captured["merge_unit"],
        }

    def _prompt_embeddings(sample: Mapping[str, Any]) -> torch.Tensor:
        return (
            prompt_embeddings_from_text(
                embedding_layer, tokenizer, sample["prompt"]
            )
            .detach()
            .cpu()
        )

    def _release() -> None:
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    use_cuda = str(args.device).startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    extract_started = time.perf_counter()
    report = run_prompt_projection_extraction(
        samples=samples,
        manifest_path=args.manifest,
        out_dir=out_dir,
        index_path=index_path,
        temperature=temperature,
        config_hash=config_hash,
        completed_records=completed_records,
        overwrite=args.overwrite,
        resolve_image_fn=resolve_image,
        image_sha256_fn=image_file_sha256,
        image_loader_fn=_load_image,
        vision_capture_fn=_vision_capture,
        prompt_embeddings_fn=_prompt_embeddings,
        extraction_meta={
            "dense_vision_blocks": extraction.get("dense_vision_blocks", 24),
            "capture_layer": extraction.get("capture_layer", 0),
            "prompt_tokenization": config["teacher"].get(
                "prompt_tokenization", "sample_prompt_only"
            ),
            "add_special_tokens": config["teacher"].get(
                "add_special_tokens", False
            ),
        },
        release_fn=_release,
        progress=lambda message: print(message, flush=True),
    )
    if use_cuda:
        torch.cuda.synchronize()
    gpu_extract_seconds = time.perf_counter() - extract_started

    after_extract = gpu_runtime_snapshot()
    after_extract["stage"] = "after_extraction"
    after_extract["gpu_extract_seconds"] = gpu_extract_seconds
    after_extract["records_written"] = report["records_written"]
    after_extract["vision_forwards"] = report["vision_forwards"]
    after_extract["prompt_embeddings"] = report["prompt_embeddings"]
    if use_cuda:
        after_extract["torch_cuda_peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / (1024**2), 1
        )
        after_extract["torch_cuda_peak_reserved_mib"] = round(
            torch.cuda.max_memory_reserved() / (1024**2), 1
        )
    (out_dir / "gpu-audit-after-extraction.json").write_text(
        json.dumps(after_extract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["gpu_extract_seconds"] = gpu_extract_seconds
    report["peak_memory"] = after_extract
    print(
        f"DONE extracted={report['records_written']} "
        f"vision_forwards={report['vision_forwards']} index={index_path} "
        f"gpu_seconds={gpu_extract_seconds:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
