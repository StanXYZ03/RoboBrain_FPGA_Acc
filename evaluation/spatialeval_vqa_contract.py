#!/usr/bin/env python3
"""SpatialEval VQA 4500 external-confirmation contract (CPU-only, no model).

This module is the single source of truth for the untouched external
confirmation experiment frozen by
``configs/2026-10-09-spatialeval-vqa-external-confirmation.preregistered.json``:

* only the three tasks the official evaluator actually handles -
  ``spatialmap`` / ``mazenav`` / ``spatialgrid`` - with 1500 samples each
  (4500 total); ``spatialreal`` (135) is excluded from the main gate;
* the official ``w_reason`` prompt suffix, ``max_new_tokens=64``,
  ``do_sample=false``, BF16/SDPA and the pinned pixel budget
  ``min_pixels=65536`` / ``max_pixels=589824``;
* the three arms ``dense_edge576`` / ``corrected_l2_integer_b1_k96`` /
  ``v2_edge576_b1_k96`` reusing the frozen edge576 producers (no selection
  algorithm lives here);
* materialisation of the deduplicated images from the two pinned
  ``/tmp`` parquet shards by reference-manifest row index into
  ``/tmp/zhaoxiangyu-spatialeval-vqa/images``, verifying the parquet SHA256, the
  image SHA256 and the image width/height;
* a label-free selector projection and an append-only, fail-closed resume
  contract for the raw per-sample JSONL;
* SHA256 pin verification of the frozen dependencies and the same fail-closed
  ``--model`` identity verifier used by the edge576 contract.

Importing this module must never load model weights or touch CUDA.  The actual
Dense / corrected-integer-L2 / v2 producers stay in the frozen modules the
runner imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.edge576_cvbench_runner_contract import (  # noqa: F401
    Edge576ContractError,
    Edge576Processor,
    _validate_token_fields_in_record,
    compact_model_identity,
    grid_llm_tokens,
    prompt_sha256,
    record_sha256,
    resolve_output_token_fields,
    sha256_file,
    verify_model_identity,
)

ROOT = Path(__file__).resolve().parents[1]

#: The SpatialEval contract reuses the frozen edge576 contract error class so a
#: single ``except`` catches every contract violation in this family.
SpatialEvalContractError = Edge576ContractError


# ---------------------------------------------------------------------------
# Pinned dataset provenance
# ---------------------------------------------------------------------------
IMAGE_ROOT = Path("/tmp/zhaoxiangyu-spatialeval-vqa")
IMAGE_DIR = IMAGE_ROOT / "images"

REFERENCE_MANIFEST = (
    ROOT / "datasets" / "external" / "2026-10-09-spatialeval-vqa-test4635.reference.jsonl"
)
REFERENCE_MANIFEST_SHA256 = (
    "03649b9c2bae7e43cc72555b744396eb8205785e7df80c5a73a6ae1e455f4050"
)
REFERENCE_MANIFEST_BYTES = 5282303
REFERENCE_MANIFEST_ROWS = 4635

#: Fixed local parquet shards (offline).  Each pin is re-verified before any
#: image byte is read.
PARQUET_PINS = (
    {
        "path": "/tmp/zhaoxiangyu-spatialeval-vqa/test-00000-of-00002.parquet",
        "sha256": "624aad413e7eb5dbd4e234ac2a36945b9be2280a9d1c8972c1f2fc886b27b897",
        "bytes": 76143378,
        "row_count": 2318,
    },
    {
        "path": "/tmp/zhaoxiangyu-spatialeval-vqa/test-00001-of-00002.parquet",
        "sha256": "b9301e70fd79ec1f828ad04100bb6b546a76bcdf15bae71b9aded77b98d27add",
        "bytes": 635994517,
        "row_count": 2317,
    },
)

#: Run manifest written by materialisation; it references absolute image paths
#: and carries no label field at all, so labels cannot reach the selector.
RUN_MANIFEST = (
    ROOT / "datasets" / "external" / "2026-10-09-spatialeval-vqa-test4500.run.jsonl"
)

TASKS = ("spatialmap", "mazenav", "spatialgrid")
NON_HANDLER_TASKS = ("spatialreal",)
TASK_SAMPLE_COUNTS = {"spatialmap": 1500, "mazenav": 1500, "spatialgrid": 1500}
EXPECTED_SAMPLE_COUNT = 4500
REFERENCE_TASK_COUNTS = {
    "spatialmap": 1500,
    "mazenav": 1500,
    "spatialgrid": 1500,
    "spatialreal": 135,
}

#: Official ``--w_reason`` suffix for the non-bunny branch of the pinned
#: ``inference_vlm.py``.
W_REASON_SUFFIX = (
    "\nFirst, provide a concise answer in one sentence. Then, elaborate on the "
    "reasoning behind your answer in a detailed, step-by-step explanation."
)

# ---------------------------------------------------------------------------
# Frozen runtime domain
# ---------------------------------------------------------------------------
MIN_PIXELS = 65536
MAX_PIXELS = 589824
BUDGET_G = 576
PATCH_SIZE = 16
MERGE_SIZE = 2
MERGE_UNIT = 4
K_BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
MAX_SWAPS = 16
MAX_NEW_TOKENS = 64
DTYPE = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
DECODING = "greedy_argmax"
DO_SAMPLE = False
IDENTITY_BYPASS_G_LE = 96

#: Fail closed at import time if the reused frozen edge576 runtime constants
#: ever diverge from the SpatialEval contract.
from evaluation import edge576_cvbench_runner_contract as _edge  # noqa: E402

_REQUIRED_EQUAL = {
    "MIN_PIXELS": (MIN_PIXELS, _edge.MIN_PIXELS),
    "MAX_PIXELS": (MAX_PIXELS, _edge.MAX_PIXELS),
    "BUDGET_G": (BUDGET_G, _edge.BUDGET_G),
    "K_BUDGET": (K_BUDGET, _edge.K_BUDGET),
    "REGION_GRID": (REGION_GRID, _edge.REGION_GRID),
    "LOCAL_CANDIDATE_CAP": (LOCAL_CANDIDATE_CAP, _edge.LOCAL_CANDIDATE_CAP),
    "MAX_SWAPS": (MAX_SWAPS, _edge.MAX_SWAPS),
    "MERGE_UNIT": (MERGE_UNIT, _edge.MERGE_UNIT),
}
for _name, (_ours, _theirs) in _REQUIRED_EQUAL.items():
    if _ours != _theirs:
        raise SpatialEvalContractError(
            f"SpatialEval {_name}={_ours} != frozen edge576 {_theirs} (fail closed)"
        )

# ---------------------------------------------------------------------------
# Arms and output paths
# ---------------------------------------------------------------------------
ARM_DENSE = "dense_edge576"
ARM_L2 = "corrected_l2_integer_b1_k96"
ARM_V2 = "v2_edge576_b1_k96"
ARMS = (ARM_DENSE, ARM_L2, ARM_V2)

ARM_MECHANISM_IDS = {
    ARM_DENSE: "dense_edge576",
    ARM_L2: "corrected_l2_integer",
    ARM_V2: "conservative_l2_guarded_rerank_v2",
}
ARM_IS_NEW_MECHANISM = {ARM_DENSE: False, ARM_L2: False, ARM_V2: False}

ARTIFACT_DIR = ROOT / "artifacts" / "token_selector" / "2026-10-09-spatialeval-vqa"
ARM_OUTPUT_PATHS = {
    ARM_DENSE: ARTIFACT_DIR / "dense_edge576_spatialeval_vqa_raw.jsonl",
    ARM_L2: ARTIFACT_DIR / "corrected_l2_integer_k96_spatialeval_vqa_raw.jsonl",
    ARM_V2: ARTIFACT_DIR / "v2_spatialeval_vqa_raw.jsonl",
}

RUN_MODE = "external4500"

# ---------------------------------------------------------------------------
# Pre-registered config + pins
# ---------------------------------------------------------------------------
SPATIALEVAL_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-09-spatialeval-vqa-external-confirmation.preregistered.json"
)
SPATIALEVAL_CONFIG_VERSION = "2026-10-09-spatialeval-vqa-external-confirmation"

RUNNER_DEPENDENCY_PINS = (
    "reference_manifest",
    "official_evaluation_py",
    "official_inference_vlm_py",
    "official_inference_configs_py",
    "model_identity_manifest",
    "spatialeval_contract",
    "spatialeval_runner",
    "spatialeval_official_evaluator",
    "spatialeval_summarizer",
    "spatialeval_producer_runner",
    "spatialeval_producer_contract",
    "spatialeval_draft_config",
)

# ---------------------------------------------------------------------------
# Label-free selector projection
# ---------------------------------------------------------------------------
SELECTOR_SAMPLE_KEYS = (
    "sample_id",
    "task",
    "question_id",
    "image",
    "prompt",
    "max_new_tokens",
)

FORBIDDEN_SELECTOR_KEYS = (
    "oracle_answer",
    "oracle_option",
    "oracle_full_answer",
    "answer",
    "ground_truth",
    "correct",
    "parsed_answer",
    "official_answer",
    "parse_success",
    "label",
    "labels",
)

#: Fields every raw SpatialEval record must carry for a safe append-only resume.
REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "task",
    "question_id",
    "arm",
    "mechanism_id",
    "manifest_sha256",
    "min_pixels",
    "max_pixels",
    "image",
    "image_sha256",
    "width",
    "height",
    "image_grid_thw",
    "full_llm_visual_tokens",
    "output_text",
    "output_token_count",
    "output_token_ids_available",
    "output_group_count",
    "selected_group_indices",
    "identity_bypass",
    "prompt_sha256",
    "max_new_tokens",
    "do_sample",
    "decoding",
    "dtype",
    "attention_implementation",
    "run_mode",
    "record_sha256",
)

RUN_MANIFEST_REQUIRED_FIELDS = (
    "sample_id",
    "task",
    "question_id",
    "source",
    "text",
    "prompt",
    "max_new_tokens",
    "image",
    "image_sha256",
    "width",
    "height",
    "source_parquet_path",
    "source_parquet_sha256",
    "source_parquet_row_index",
    "reference_manifest_sha256",
    "record_sha256",
)

IMAGE_FORMAT_EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise SpatialEvalContractError(f"missing required input {path} (fail closed)")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalContractError(
                f"{path} line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise SpatialEvalContractError(
                f"{path} line {line_number} is not a JSON object (fail closed)"
            )
        records.append(record)
    return records


def _require_new_path(path: Path, label: str) -> None:
    if os.path.lexists(path):
        raise SpatialEvalContractError(f"{label} already exists: {path} (fail closed)")


def _staged_temp_file(directory: Path, prefix: str, suffix: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(name)


# ---------------------------------------------------------------------------
# Reference manifest
# ---------------------------------------------------------------------------
def load_reference_manifest(
    path: Path = REFERENCE_MANIFEST,
    *,
    expected_sha256: str | None = REFERENCE_MANIFEST_SHA256,
    expected_bytes: int | None = REFERENCE_MANIFEST_BYTES,
    expected_rows: int | None = REFERENCE_MANIFEST_ROWS,
) -> list[dict[str, Any]]:
    """Load and verify the pinned SpatialEval VQA test4635 reference manifest."""

    path = Path(path)
    if not path.is_file():
        raise SpatialEvalContractError(f"missing reference manifest {path} (fail closed)")
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise SpatialEvalContractError(
                f"reference manifest SHA256 {actual} != pinned {expected_sha256} "
                "(fail closed)"
            )
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise SpatialEvalContractError(
            f"reference manifest bytes {path.stat().st_size} != pinned "
            f"{expected_bytes} (fail closed)"
        )
    records = load_jsonl(path)
    if expected_rows is not None and len(records) != expected_rows:
        raise SpatialEvalContractError(
            f"reference manifest has {len(records)} rows != pinned {expected_rows} "
            "(fail closed)"
        )
    seen: set[str] = set()
    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SpatialEvalContractError(
                f"reference row {index} has no sample_id (fail closed)"
            )
        if sample_id in seen:
            raise SpatialEvalContractError(
                f"duplicate reference sample_id {sample_id!r} (fail closed)"
            )
        seen.add(sample_id)
        for field in (
            "text",
            "oracle_answer",
            "image_sha256",
            "width",
            "height",
            "source_parquet_path",
            "source_parquet_sha256",
            "source_parquet_row_index",
        ):
            if field not in record:
                raise SpatialEvalContractError(
                    f"reference row {sample_id!r} is missing {field!r} (fail closed)"
                )
    return records


def select_handler_records(
    reference_records: Sequence[Mapping[str, Any]],
    *,
    task_sample_counts: Mapping[str, int] = TASK_SAMPLE_COUNTS,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> list[dict[str, Any]]:
    """Return only the handler-task rows, fail-closed on any count drift."""

    counts: dict[str, int] = {task: 0 for task in TASKS}
    selected: list[dict[str, Any]] = []
    for record in reference_records:
        sample_id = str(record["sample_id"])
        task = sample_id.split(".")[0]
        if task in TASKS:
            counts[task] += 1
            selected.append(dict(record))
        elif task in NON_HANDLER_TASKS:
            continue
        else:
            raise SpatialEvalContractError(
                f"reference sample {sample_id!r} has unknown task {task!r} "
                "(fail closed)"
            )
    for task, expected in task_sample_counts.items():
        if counts[task] != expected:
            raise SpatialEvalContractError(
                f"handler task {task!r} has {counts[task]} rows != expected "
                f"{expected} (fail closed)"
            )
    if len(selected) != expected_count:
        raise SpatialEvalContractError(
            f"selected {len(selected)} rows != expected {expected_count} "
            "(fail closed)"
        )
    return selected


# ---------------------------------------------------------------------------
# Image materialisation from the pinned parquet shards
# ---------------------------------------------------------------------------
def _image_bytes(value: Any, parquet_dir: Path, row_index: int) -> bytes:
    if not isinstance(value, Mapping):
        raise SpatialEvalContractError(
            f"parquet row {row_index} image is not a struct (fail closed)"
        )
    raw = value.get("bytes")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes) and raw:
        return raw
    local_path = value.get("path")
    if isinstance(local_path, str) and local_path:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            candidate = parquet_dir / candidate
        if candidate.is_file():
            return candidate.read_bytes()
    raise SpatialEvalContractError(
        f"parquet row {row_index} image has no bytes or local file (fail closed)"
    )


def _validate_image(raw: bytes, row_index: int) -> tuple[int, int, str]:
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise SpatialEvalContractError(
            f"parquet row {row_index} contains a bad image: {exc} (fail closed)"
        ) from exc
    extension = IMAGE_FORMAT_EXTENSIONS.get(image_format)
    if extension is None:
        raise SpatialEvalContractError(
            f"parquet row {row_index} image format {image_format!r} is unsupported "
            "(fail closed)"
        )
    if width <= 0 or height <= 0:
        raise SpatialEvalContractError(
            f"parquet row {row_index} image size {width}x{height} is invalid "
            "(fail closed)"
        )
    return int(width), int(height), extension


def verify_parquet_pins(
    parquet_pins: Sequence[Mapping[str, Any]] = PARQUET_PINS,
) -> list[dict[str, Any]]:
    """Verify every pinned parquet shard's existence, bytes, rows and SHA256."""

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SpatialEvalContractError(
            "pyarrow is required for parquet materialisation (fail closed)"
        ) from exc
    verified: list[dict[str, Any]] = []
    for pin in parquet_pins:
        path = Path(str(pin["path"]))
        if not path.is_file():
            raise SpatialEvalContractError(
                f"missing parquet shard {path} (fail closed)"
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != int(pin["bytes"]):
            raise SpatialEvalContractError(
                f"parquet {path} bytes {actual_bytes} != pinned {pin['bytes']} "
                "(fail closed)"
            )
        actual_sha = sha256_file(path)
        if actual_sha != str(pin["sha256"]):
            raise SpatialEvalContractError(
                f"parquet {path} SHA256 {actual_sha} != pinned {pin['sha256']} "
                "(fail closed)"
            )
        rows = pq.ParquetFile(path).metadata.num_rows
        if rows != int(pin["row_count"]):
            raise SpatialEvalContractError(
                f"parquet {path} has {rows} rows != pinned {pin['row_count']} "
                "(fail closed)"
            )
        verified.append(
            {"path": str(path), "sha256": actual_sha, "bytes": actual_bytes,
             "row_count": rows}
        )
    return verified


def materialize_images(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    parquet_pins: Sequence[Mapping[str, Any]] = PARQUET_PINS,
    image_dir: Path = IMAGE_DIR,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Materialise the deduplicated images for ``selected_records``.

    Every selected row is streamed from its pinned parquet shard; its image
    SHA256 and width/height are checked against the reference manifest before
    the byte is written.  Files are named ``<image_sha256><ext>`` so the same
    image shared by several questions is written exactly once.
    """

    if batch_size <= 0:
        raise SpatialEvalContractError("batch_size must be positive (fail closed)")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SpatialEvalContractError(
            "pyarrow is required for parquet materialisation (fail closed)"
        ) from exc

    pin_by_path = {str(pin["path"]): pin for pin in parquet_pins}
    verified_pins = verify_parquet_pins(parquet_pins)

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    by_parquet: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in selected_records:
        parquet_path = str(record["source_parquet_path"])
        if parquet_path not in pin_by_path:
            raise SpatialEvalContractError(
                f"sample {record['sample_id']!r} references un-pinned parquet "
                f"{parquet_path!r} (fail closed)"
            )
        if str(record["source_parquet_sha256"]) != str(pin_by_path[parquet_path]["sha256"]):
            raise SpatialEvalContractError(
                f"sample {record['sample_id']!r} parquet SHA differs from the pin "
                "(fail closed)"
            )
        row_index = int(record["source_parquet_row_index"])
        by_parquet.setdefault(parquet_path, {})[row_index] = record

    written: dict[str, str] = {}
    checked_rows = 0
    for parquet_path, needed in by_parquet.items():
        parquet_file = pq.ParquetFile(Path(parquet_path))
        row_offset = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image"]):
            rows = batch.to_pylist()
            for local_index, row in enumerate(rows):
                global_index = row_offset + local_index
                record = needed.get(global_index)
                if record is None:
                    continue
                raw = _image_bytes(row["image"], Path(parquet_path).parent, global_index)
                actual_sha = sha256_bytes(raw)
                if actual_sha != str(record["image_sha256"]):
                    raise SpatialEvalContractError(
                        f"sample {record['sample_id']!r} image SHA256 {actual_sha} != "
                        f"reference {record['image_sha256']} (fail closed)"
                    )
                width, height, extension = _validate_image(raw, global_index)
                if (width, height) != (int(record["width"]), int(record["height"])):
                    raise SpatialEvalContractError(
                        f"sample {record['sample_id']!r} image size {width}x{height} != "
                        f"reference {record['width']}x{record['height']} (fail closed)"
                    )
                checked_rows += 1
                destination = image_dir / f"{actual_sha}{extension}"
                if actual_sha in written:
                    if written[actual_sha] != str(destination):
                        raise SpatialEvalContractError(
                            f"image SHA {actual_sha} mapped to two names (fail closed)"
                        )
                    continue
                if not destination.exists():
                    staged = _staged_temp_file(
                        image_dir, f".{actual_sha[:12]}-", extension
                    )
                    staged.write_bytes(raw)
                    os.replace(staged, destination)
                written[actual_sha] = str(destination)
            row_offset += len(rows)

    if checked_rows != len(selected_records):
        raise SpatialEvalContractError(
            f"checked {checked_rows} rows != selected {len(selected_records)} "
            "(fail closed)"
        )
    return {
        "status": "SPATIALEVAL_IMAGES_MATERIALIZED",
        "selected_rows": len(selected_records),
        "unique_images": len(written),
        "image_dir": str(image_dir),
        "parquet_pins": verified_pins,
        "images": dict(sorted(written.items())),
    }


def _resolve_materialized_image(image_dir: Path, image_sha256: str) -> Path:
    matches = sorted(Path(image_dir).glob(f"{image_sha256}.*"))
    if len(matches) != 1:
        raise SpatialEvalContractError(
            f"image {image_sha256} resolves to {len(matches)} files in {image_dir} "
            "(fail closed)"
        )
    return matches[0]


def write_run_manifest(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    output_path: Path = RUN_MANIFEST,
    image_dir: Path = IMAGE_DIR,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    task_sample_counts: Mapping[str, int] = TASK_SAMPLE_COUNTS,
) -> dict[str, Any]:
    """Write the label-free run manifest referencing absolute image paths."""

    output_path = Path(output_path)
    _require_new_path(output_path, "run manifest")
    if len(selected_records) != expected_count:
        raise SpatialEvalContractError(
            f"selected {len(selected_records)} rows != {expected_count} "
            "(fail closed)"
        )

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in selected_records:
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise SpatialEvalContractError(
                f"duplicate selected sample_id {sample_id!r} (fail closed)"
            )
        seen.add(sample_id)
        task = sample_id.split(".")[0]
        if task not in TASKS:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} task {task!r} is not a handler "
                "task (fail closed)"
            )
        question_id = int(sample_id.split(".")[-1])
        image = _resolve_materialized_image(image_dir, str(record["image_sha256"]))
        actual_sha = sha256_file(image)
        if actual_sha != str(record["image_sha256"]):
            raise SpatialEvalContractError(
                f"materialized image {image} SHA256 drifted (fail closed)"
            )
        entry: dict[str, Any] = {
            "sample_id": sample_id,
            "task": task,
            "question_id": question_id,
            "source": task,
            "text": str(record["text"]),
            "prompt": str(record["text"]) + W_REASON_SUFFIX,
            "max_new_tokens": MAX_NEW_TOKENS,
            "image": str(image.resolve()),
            "image_sha256": str(record["image_sha256"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "source_parquet_path": str(record["source_parquet_path"]),
            "source_parquet_sha256": str(record["source_parquet_sha256"]),
            "source_parquet_row_index": int(record["source_parquet_row_index"]),
            "reference_manifest_sha256": REFERENCE_MANIFEST_SHA256,
        }
        leaked = [key for key in FORBIDDEN_SELECTOR_KEYS if key in entry]
        if leaked:
            raise SpatialEvalContractError(
                f"run manifest entry {sample_id!r} would carry labels {leaked} "
                "(fail closed)"
            )
        if not Path(entry["image"]).is_absolute():
            raise SpatialEvalContractError(
                f"run manifest image for {sample_id!r} is not absolute (fail closed)"
            )
        entry["record_sha256"] = record_sha256(entry)
        records.append(entry)

    task_counts = {task: sum(1 for entry in records if entry["task"] == task) for task in TASKS}
    for task, expected in task_sample_counts.items():
        if task_counts[task] != expected:
            raise SpatialEvalContractError(
                f"run manifest task {task!r} has {task_counts[task]} rows != "
                f"{expected} (fail closed)"
            )

    staged = _staged_temp_file(output_path.parent, ".spatialeval-run-manifest-", ".jsonl")
    try:
        with staged.open("w", encoding="utf-8", newline="\n") as handle:
            for entry in records:
                handle.write(canonical_json(entry) + "\n")
        os.replace(staged, output_path)
    finally:
        if staged.exists():
            staged.unlink()
    return {
        "status": "SPATIALEVAL_RUN_MANIFEST_BUILT",
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "row_count": len(records),
        "task_counts": {
            task: sum(1 for entry in records if entry["task"] == task)
            for task in TASKS
        },
    }


def prepare_external_inputs(
    *,
    reference_manifest: Path = REFERENCE_MANIFEST,
    parquet_pins: Sequence[Mapping[str, Any]] = PARQUET_PINS,
    image_dir: Path = IMAGE_DIR,
    run_manifest: Path = RUN_MANIFEST,
) -> dict[str, Any]:
    """Materialise the images and write the 4500-row label-free run manifest."""

    reference_records = load_reference_manifest(reference_manifest)
    selected = select_handler_records(reference_records)
    materialised = materialize_images(
        selected, parquet_pins=parquet_pins, image_dir=image_dir
    )
    manifest_summary = write_run_manifest(
        selected, output_path=run_manifest, image_dir=image_dir
    )
    return {
        "status": "SPATIALEVAL_EXTERNAL_INPUTS_READY",
        "reference_manifest": str(reference_manifest),
        "reference_manifest_sha256": sha256_file(Path(reference_manifest)),
        "materialization": {k: v for k, v in materialised.items() if k != "images"},
        "run_manifest": manifest_summary,
    }


def load_run_manifest(
    path: Path = RUN_MANIFEST,
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    task_sample_counts: Mapping[str, int] = TASK_SAMPLE_COUNTS,
    verify_images: bool = True,
) -> list[dict[str, Any]]:
    """Load and fully validate the label-free SpatialEval run manifest."""

    path = Path(path)
    if not path.is_file():
        raise SpatialEvalContractError(
            f"missing run manifest {path}; run the materialiser first (fail closed)"
        )
    records = load_jsonl(path)
    if len(records) != expected_count:
        raise SpatialEvalContractError(
            f"run manifest has {len(records)} rows != expected {expected_count} "
            "(fail closed)"
        )
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SpatialEvalContractError(
            "Pillow is required to validate run-manifest images (fail closed)"
        ) from exc

    seen: set[str] = set()
    task_counts: dict[str, int] = {task: 0 for task in TASKS}
    for index, record in enumerate(records):
        for field in RUN_MANIFEST_REQUIRED_FIELDS:
            if field not in record:
                raise SpatialEvalContractError(
                    f"run manifest row {index} is missing {field!r} (fail closed)"
                )
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise SpatialEvalContractError(
                f"duplicate run manifest sample_id {sample_id!r} (fail closed)"
            )
        seen.add(sample_id)
        task = str(record["task"])
        if task not in TASKS:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} task {task!r} is not a handler "
                "task (fail closed)"
            )
        if str(sample_id).split(".")[0] != task:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} task field disagrees with its id "
                "(fail closed)"
            )
        task_counts[task] += 1
        if str(record["prompt"]) != str(record["text"]) + W_REASON_SUFFIX:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} prompt is not text + official "
                "w_reason suffix (fail closed)"
            )
        if int(record["max_new_tokens"]) != MAX_NEW_TOKENS:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} max_new_tokens != {MAX_NEW_TOKENS} "
                "(fail closed)"
            )
        leaked = [key for key in FORBIDDEN_SELECTOR_KEYS if key in record]
        if leaked:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} carries labels {leaked} "
                "(fail closed)"
            )
        image = Path(str(record["image"]))
        if not image.is_absolute():
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} image path is not absolute "
                "(fail closed)"
            )
        if not image.is_file():
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} image is missing: {image} "
                "(fail closed)"
            )
        if verify_images:
            actual_sha = sha256_file(image)
            if actual_sha != str(record["image_sha256"]):
                raise SpatialEvalContractError(
                    f"run manifest sample {sample_id!r} image SHA256 drifted "
                    "(fail closed)"
                )
            with Image.open(image) as opened:
                if (opened.size[0], opened.size[1]) != (
                    int(record["width"]),
                    int(record["height"]),
                ):
                    raise SpatialEvalContractError(
                        f"run manifest sample {sample_id!r} image size drifted "
                        "(fail closed)"
                    )
        if str(record.get("reference_manifest_sha256")) != REFERENCE_MANIFEST_SHA256:
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} reference SHA != pinned (fail closed)"
            )
        if str(record.get("record_sha256")) != record_sha256(record):
            raise SpatialEvalContractError(
                f"run manifest sample {sample_id!r} record_sha256 mismatch (fail closed)"
            )
    for task, expected in task_sample_counts.items():
        if task_counts[task] != expected:
            raise SpatialEvalContractError(
                f"run manifest task {task!r} has {task_counts[task]} rows != {expected} "
                "(fail closed)"
            )
    return records


# ---------------------------------------------------------------------------
# Processor wrapper + per-sample grid assertion
# ---------------------------------------------------------------------------
def wrap_processor(processor: Any) -> Edge576Processor:
    """Wrap a processor with the pinned pixel budget (bounds are identical)."""

    return Edge576Processor(processor, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)


def assert_spatialeval_grid(
    sample: Mapping[str, Any], actual_grid: Sequence[int] | None
) -> int:
    """Fail closed unless the actual grid is a single image within ``G<=576``.

    Returns ``full_llm_visual_tokens`` for the record.
    """

    if actual_grid is None:
        raise SpatialEvalContractError(
            f"sample {sample.get('sample_id')!r} produced no image_grid_thw "
            "(fail closed)"
        )
    grid = [int(value) for value in actual_grid]
    if len(grid) != 3 or grid[0] != 1:
        raise SpatialEvalContractError(
            f"sample {sample.get('sample_id')!r} image_grid_thw {grid} is not a "
            "single image (fail closed)"
        )
    full_llm = grid_llm_tokens(grid)
    if full_llm < 1 or full_llm > BUDGET_G:
        raise SpatialEvalContractError(
            f"sample {sample.get('sample_id')!r} G={full_llm} is outside the pinned "
            f"edge576 budget 1..{BUDGET_G} (fail closed)"
        )
    return full_llm


def sanitize_sample_for_selector(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Return the label-free projection handed to a frozen producer.

    Any label field present in the input aborts the run (fail closed) instead of
    being silently dropped, so a label can never be carried into a selector.
    """

    leaked = [key for key in FORBIDDEN_SELECTOR_KEYS if key in sample]
    if leaked:
        raise SpatialEvalContractError(
            f"selector input carries labels {leaked} (fail closed)"
        )
    return {key: sample[key] for key in SELECTOR_SAMPLE_KEYS if key in sample}


# ---------------------------------------------------------------------------
# Output path / append-only resume contract
# ---------------------------------------------------------------------------
def validate_arm(arm: str) -> str:
    if arm not in ARMS:
        raise SpatialEvalContractError(
            f"unknown arm {arm!r}; expected one of {ARMS} (fail closed)"
        )
    return arm


def ensure_output_path(arm: str, output: Path | None) -> Path:
    validate_arm(arm)
    pinned = ARM_OUTPUT_PATHS[arm]
    if output is None:
        return pinned
    candidate = Path(output)
    try:
        same = candidate.resolve() == pinned.resolve()
    except OSError:
        same = False
    if not same:
        raise SpatialEvalContractError(
            f"--output {candidate} != pinned path {pinned} for arm {arm!r} "
            "(fail closed)"
        )
    return pinned


def _validate_resume_record(
    record: Mapping[str, Any],
    *,
    line_number: int,
    allowed_ids: set[str],
    done: set[str],
    arm: str,
    manifest_sha256: str,
    samples_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    if missing:
        raise SpatialEvalContractError(
            f"output line {line_number} is missing fields {missing} "
            "(partial record, fail closed)"
        )
    _validate_token_fields_in_record(record, label=f"output line {line_number}")
    sample_id = str(record["sample_id"])
    if sample_id in done:
        raise SpatialEvalContractError(
            f"duplicate output sample_id {sample_id!r} (fail closed)"
        )
    if sample_id not in allowed_ids:
        raise SpatialEvalContractError(
            f"output sample_id {sample_id!r} is not in the current run manifest "
            "(stale output, fail closed)"
        )
    if str(record.get("arm")) != arm:
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} arm={record.get('arm')!r} != {arm!r} "
            "(fail closed)"
        )
    if str(record.get("mechanism_id")) != ARM_MECHANISM_IDS[arm]:
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} mechanism_id="
            f"{record.get('mechanism_id')!r} != {ARM_MECHANISM_IDS[arm]!r} "
            "(fail closed)"
        )
    if record.get("manifest_sha256") != manifest_sha256:
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} manifest_sha256 differs from the current "
            "run (fail closed)"
        )
    for field, expected in (
        ("min_pixels", MIN_PIXELS),
        ("max_pixels", MAX_PIXELS),
        ("max_new_tokens", MAX_NEW_TOKENS),
        ("do_sample", DO_SAMPLE),
        ("decoding", DECODING),
        ("dtype", DTYPE),
        ("attention_implementation", ATTENTION_IMPLEMENTATION),
        ("run_mode", RUN_MODE),
    ):
        if record.get(field) != expected:
            raise SpatialEvalContractError(
                f"output sample {sample_id!r} {field}={record.get(field)!r} != pinned "
                f"{expected!r} (fail closed)"
            )
    sample = samples_by_id[sample_id]
    if str(record.get("task")) != str(sample["task"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} task differs from the run manifest "
            "(fail closed)"
        )
    if int(record.get("question_id", -1)) != int(sample["question_id"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} question_id differs from the run manifest "
            "(fail closed)"
        )
    if str(record.get("image")) != str(sample["image"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} image differs from the run manifest "
            "(fail closed)"
        )
    if str(record.get("image_sha256")) != str(sample["image_sha256"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} image_sha256 differs from the run manifest "
            "(fail closed)"
        )
    if record.get("prompt_sha256") != prompt_sha256(sample["prompt"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} prompt_sha256 does not match the run "
            "manifest prompt (fail closed)"
        )
    full_llm = assert_spatialeval_grid(sample, record["image_grid_thw"])
    if full_llm != int(record["full_llm_visual_tokens"]):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} full_llm_visual_tokens "
            f"{record['full_llm_visual_tokens']} does not match its grid (fail closed)"
        )
    output_group_count = int(record["output_group_count"])
    selected = [int(value) for value in record["selected_group_indices"]]
    if len(selected) != output_group_count:
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} has {len(selected)} selected groups but "
            f"output_group_count {output_group_count} (fail closed)"
        )
    if selected and not all(
        later > earlier for earlier, later in zip(selected, selected[1:])
    ):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} selected_group_indices are not strictly "
            "ascending (fail closed)"
        )
    identity_bypass = bool(record["identity_bypass"])
    if arm == ARM_DENSE:
        if output_group_count != full_llm:
            raise SpatialEvalContractError(
                f"dense output sample {sample_id!r} must emit G={full_llm} groups, "
                f"got {output_group_count} (fail closed)"
            )
    else:
        if identity_bypass:
            if output_group_count != full_llm:
                raise SpatialEvalContractError(
                    f"output sample {sample_id!r} identity bypass must emit G="
                    f"{full_llm} groups, got {output_group_count} (fail closed)"
                )
        elif output_group_count != K_BUDGET:
            raise SpatialEvalContractError(
                f"output sample {sample_id!r} non-bypass must emit K={K_BUDGET}, got "
                f"{output_group_count} (fail closed)"
            )
    if str(record.get("record_sha256")) != record_sha256(record):
        raise SpatialEvalContractError(
            f"output sample {sample_id!r} record_sha256 mismatch (stale or corrupted "
            "record, fail closed)"
        )


def validate_resume(
    path: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    manifest_sha256: str,
) -> set[str]:
    """Validate an existing raw output JSONL; return its completed sample ids."""

    validate_arm(arm)
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    done: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalContractError(
                f"output line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise SpatialEvalContractError(
                f"output line {line_number} must be a JSON object (fail closed)"
            )
        _validate_resume_record(
            record,
            line_number=line_number,
            allowed_ids=set(samples_by_id),
            done=done,
            arm=arm,
            manifest_sha256=manifest_sha256,
            samples_by_id=samples_by_id,
        )
        done.add(str(record["sample_id"]))
    return done


# ---------------------------------------------------------------------------
# Config pins + mechanism identity
# ---------------------------------------------------------------------------
def load_config(path: Path = SPATIALEVAL_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise SpatialEvalContractError(
            f"missing SpatialEval config {path} (fail closed)"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SpatialEvalContractError(
            f"SpatialEval config {path} is not valid JSON: {exc} (fail closed)"
        ) from exc
    if config.get("version") != SPATIALEVAL_CONFIG_VERSION:
        raise SpatialEvalContractError(
            f"unexpected SpatialEval config version {config.get('version')!r} "
            "(fail closed)"
        )
    return config


def verify_dependency_pins(
    *,
    pin_names: Sequence[str] = RUNNER_DEPENDENCY_PINS,
    config_path: Path = SPATIALEVAL_CONFIG,
) -> dict[str, Any]:
    config = load_config(config_path)
    pins = config.get("pins")
    if not isinstance(pins, Mapping):
        raise SpatialEvalContractError(
            "SpatialEval config has no 'pins' object (fail closed)"
        )
    verified: dict[str, dict[str, Any]] = {}
    for name in pin_names:
        if name not in pins:
            raise SpatialEvalContractError(
                f"SpatialEval config has no dependency pin {name!r} (fail closed)"
            )
        entry = pins[name]
        path = ROOT / str(entry["path"])
        if not path.is_file():
            raise SpatialEvalContractError(
                f"pinned dependency {name!r} is missing: {path} (fail closed)"
            )
        actual = sha256_file(path)
        if actual != str(entry["sha256"]):
            raise SpatialEvalContractError(
                f"pinned dependency {name!r} SHA256 drifted: {actual} != "
                f"{entry['sha256']} (fail closed)"
            )
        verified[name] = {
            "path": str(entry["path"]),
            "sha256": actual,
            "role": entry.get("role"),
        }
    return {
        "verified_count": len(verified),
        "requested_count": len(pin_names),
        "all_match": True,
        "pins": verified,
    }


def mechanism_identity(
    arm: str, *, config_path: Path = SPATIALEVAL_CONFIG
) -> dict[str, Any]:
    validate_arm(arm)
    expected = ARM_MECHANISM_IDS[arm]
    config = load_config(config_path)
    config_arm = config.get("arms", {}).get(arm)
    if not isinstance(config_arm, Mapping):
        raise SpatialEvalContractError(
            f"SpatialEval config has no arm {arm!r} (fail closed)"
        )
    if str(config_arm.get("mechanism_id")) != expected:
        raise SpatialEvalContractError(
            f"config arm {arm!r} mechanism_id {config_arm.get('mechanism_id')!r} != "
            f"{expected!r} (fail closed)"
        )
    if bool(config_arm.get("is_new_mechanism")):
        raise SpatialEvalContractError(
            f"config arm {arm!r} declares is_new_mechanism=true (fail closed)"
        )
    if str(config_arm.get("output_path")) != str(
        ARM_OUTPUT_PATHS[arm].relative_to(ROOT)
    ):
        raise SpatialEvalContractError(
            f"config arm {arm!r} output_path does not match the pinned path "
            "(fail closed)"
        )
    return {
        "arm": arm,
        "mechanism_id": expected,
        "is_new_mechanism": False,
        "config_output_path": str(config_arm.get("output_path")),
        "ok": True,
    }


# ---------------------------------------------------------------------------
# CLI: materialise the external confirmation inputs
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--run-manifest", type=Path, default=RUN_MANIFEST)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate_only:
        records = load_run_manifest(args.run_manifest)
        summary = {
            "status": "SPATIALEVAL_RUN_MANIFEST_VALIDATE_OK",
            "path": str(args.run_manifest),
            "sha256": sha256_file(args.run_manifest),
            "row_count": len(records),
        }
    else:
        summary = prepare_external_inputs(
            reference_manifest=args.reference_manifest,
            image_dir=args.image_dir,
            run_manifest=args.run_manifest,
        )
    report = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out is not None:
        json_out = Path(args.json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
