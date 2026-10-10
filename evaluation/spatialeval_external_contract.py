#!/usr/bin/env python3
"""Frozen data/runtime contract for the SpatialEval external confirmation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.build_spatialeval_vqa_manifest import (
    _image_bytes,
    _validate_image,
    iter_parquet_rows,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT / "datasets/external/2026-10-09-spatialeval-vqa-test4635.reference.jsonl"
)
REFERENCE_MANIFEST_SHA256 = (
    "03649b9c2bae7e43cc72555b744396eb8205785e7df80c5a73a6ae1e455f4050"
)
PARQUET_SHARDS = (
    {
        "path": Path("/tmp/zhaoxiangyu-spatialeval-vqa/test-00000-of-00002.parquet"),
        "sha256": "624aad413e7eb5dbd4e234ac2a36945b9be2280a9d1c8972c1f2fc886b27b897",
        "bytes": 76143378,
        "rows": 2318,
    },
    {
        "path": Path("/tmp/zhaoxiangyu-spatialeval-vqa/test-00001-of-00002.parquet"),
        "sha256": "b9301e70fd79ec1f828ad04100bb6b546a76bcdf15bae71b9aded77b98d27add",
        "bytes": 635994517,
        "rows": 2317,
    },
)
PRIMARY_TASKS = ("spatialmap", "mazenav", "spatialgrid")
PRIMARY_SAMPLE_COUNT = 4500
SPATIALREAL_DIAGNOSTIC_COUNT = 135
OFFICIAL_REASON_SUFFIX = (
    "First, provide a concise answer in one sentence. Then, elaborate on the "
    "reasoning behind your answer in a detailed, step-by-step explanation."
)
MAX_NEW_TOKENS = 64
MIN_PIXELS = 65536
MAX_PIXELS = 589824
K_BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
MAX_SWAPS = 16
DTYPE = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
DECODING = "greedy_argmax"
DO_SAMPLE = False
IMAGE_CACHE = Path("/tmp/zhaoxiangyu-spatialeval-vqa/images")
PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-09-spatialeval-vqa-three-arm-external-confirmation-r2.preregistered.json"
)

ARM_DENSE = "dense_edge576"
ARM_L2 = "corrected_l2_integer_b1_k96"
ARM_V2 = "v2_edge576_b1_k96"
ARMS = (ARM_DENSE, ARM_L2, ARM_V2)
ARM_MECHANISM_IDS = {
    ARM_DENSE: "dense_edge576",
    ARM_L2: "corrected_l2_integer",
    ARM_V2: "conservative_l2_guarded_rerank_v2",
}
OUTPUT_DIR = ROOT / "artifacts/token_selector/2026-10-09-spatialeval-vqa-three-arm"
ARM_OUTPUT_PATHS = {
    ARM_DENSE: OUTPUT_DIR / "dense_edge576_test4500.jsonl",
    ARM_L2: OUTPUT_DIR / "corrected_l2_integer_k96_test4500.jsonl",
    ARM_V2: OUTPUT_DIR / "v2_edge576_k96_test4500.jsonl",
}

SELECTOR_SAMPLE_KEYS = (
    "sample_id",
    "task",
    "type",
    "source",
    "source_dataset",
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
)


class SpatialEvalExternalContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def task_and_qid(sample_id: str) -> tuple[str, int]:
    parts = str(sample_id).split(".")
    if len(parts) != 4:
        raise SpatialEvalExternalContractError(f"invalid SpatialEval id {sample_id!r}")
    task = parts[0]
    try:
        qid = int(parts[-1])
    except ValueError as exc:
        raise SpatialEvalExternalContractError(
            f"invalid question id in {sample_id!r}"
        ) from exc
    return task, qid


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalExternalContractError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise SpatialEvalExternalContractError(
                f"non-object JSON at {path}:{line_number}"
            )
        records.append(value)
    return records


def load_reference_manifest(
    path: Path = REFERENCE_MANIFEST,
    *,
    expected_sha256: str = REFERENCE_MANIFEST_SHA256,
    expected_total: int = 4635,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise SpatialEvalExternalContractError(f"missing reference manifest {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise SpatialEvalExternalContractError(
            f"reference manifest SHA256 {actual} != pinned {expected_sha256}"
        )
    records = _load_jsonl(path)
    if len(records) != expected_total:
        raise SpatialEvalExternalContractError(
            f"reference manifest count {len(records)} != {expected_total}"
        )
    seen: set[str] = set()
    task_counts: dict[str, int] = defaultdict(int)
    task_qid_counts: dict[str, int] = defaultdict(int)
    for record in records:
        required = (
            "sample_id",
            "text",
            "oracle_answer",
            "oracle_option",
            "oracle_full_answer",
            "width",
            "height",
            "image_sha256",
            "source_parquet_path",
            "source_parquet_sha256",
            "source_parquet_row_index",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise SpatialEvalExternalContractError(
                f"manifest record missing fields {missing}"
            )
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise SpatialEvalExternalContractError(f"duplicate sample id {sample_id!r}")
        seen.add(sample_id)
        task, qid = task_and_qid(sample_id)
        task_counts[task] += 1
        task_qid_counts[f"{task}::{qid}"] += 1
    expected_tasks = {
        "spatialmap": 1500,
        "mazenav": 1500,
        "spatialgrid": 1500,
        "spatialreal": 135,
    }
    if dict(task_counts) != expected_tasks:
        raise SpatialEvalExternalContractError(
            f"unexpected task counts {dict(task_counts)}"
        )
    for task in PRIMARY_TASKS:
        for qid in (0, 1, 2):
            if task_qid_counts[f"{task}::{qid}"] != 500:
                raise SpatialEvalExternalContractError(
                    f"unexpected count for {task}::{qid}: "
                    f"{task_qid_counts[f'{task}::{qid}']}"
                )
    return records


def verify_parquet_shards(
    shards: Sequence[Mapping[str, Any]] = PARQUET_SHARDS,
) -> None:
    for shard in shards:
        path = Path(shard["path"])
        if not path.is_file():
            raise SpatialEvalExternalContractError(f"missing parquet shard {path}")
        if path.stat().st_size != int(shard["bytes"]):
            raise SpatialEvalExternalContractError(
                f"parquet byte size mismatch for {path}"
            )
        actual = sha256_file(path)
        if actual != shard["sha256"]:
            raise SpatialEvalExternalContractError(
                f"parquet SHA256 {actual} != pinned {shard['sha256']} for {path}"
            )


def verify_preregistered_pins(path: Path = PREREGISTERED_CONFIG) -> dict[str, Any]:
    """Recompute every required input/code pin before preparation or inference."""

    path = Path(path)
    if not path.is_file():
        raise SpatialEvalExternalContractError(f"missing preregistered config {path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpatialEvalExternalContractError(
            f"invalid preregistered config JSON: {path}"
        ) from exc
    if config.get("status") not in {
        "PREREGISTERED_NOT_EXECUTED",
        "PREREGISTERED_AMENDMENT_BEFORE_SCORING",
    }:
        raise SpatialEvalExternalContractError(
            f"unexpected preregistration status {config.get('status')!r}"
        )
    pins = config.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise SpatialEvalExternalContractError("preregistered config has no pins")
    verified: dict[str, Any] = {}
    for name, pin in pins.items():
        if not isinstance(pin, dict) or "path" not in pin or "sha256" not in pin:
            raise SpatialEvalExternalContractError(f"malformed pin {name!r}")
        candidate = Path(pin["path"])
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            raise SpatialEvalExternalContractError(
                f"pinned file {name!r} is missing: {candidate}"
            )
        if "bytes" in pin and candidate.stat().st_size != int(pin["bytes"]):
            raise SpatialEvalExternalContractError(
                f"pinned file {name!r} byte size mismatch"
            )
        actual = sha256_file(candidate)
        if actual != str(pin["sha256"]):
            raise SpatialEvalExternalContractError(
                f"pinned file {name!r} SHA256 {actual} != {pin['sha256']}"
            )
        verified[name] = {"path": str(candidate), "sha256": actual}
    return verified


def _atomic_write_new_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or sha256_file(path) != sha256_bytes(payload):
            raise SpatialEvalExternalContractError(
                f"existing image cache entry differs: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            if sha256_file(path) != sha256_bytes(payload):
                raise SpatialEvalExternalContractError(
                    f"concurrent image cache entry differs: {path}"
                )
        else:
            os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def materialize_primary_samples(
    reference_records: Sequence[Mapping[str, Any]],
    *,
    image_cache: Path = IMAGE_CACHE,
    shards: Sequence[Mapping[str, Any]] = PARQUET_SHARDS,
    verify_shards: bool = True,
) -> list[dict[str, Any]]:
    if verify_shards:
        verify_parquet_shards(shards)
    by_source: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in reference_records:
        task, _ = task_and_qid(str(record["sample_id"]))
        if task not in PRIMARY_TASKS:
            continue
        key = (
            str(Path(record["source_parquet_path"])),
            int(record["source_parquet_row_index"]),
        )
        if key in by_source:
            raise SpatialEvalExternalContractError(
                f"duplicate parquet row reference {key}"
            )
        by_source[key] = record

    paths_by_sample: dict[str, str] = {}
    for shard in shards:
        path = Path(shard["path"])
        row_index = 0
        for rows in iter_parquet_rows(path):
            for row in rows:
                key = (str(path), row_index)
                reference = by_source.get(key)
                if reference is not None:
                    if str(reference["source_parquet_sha256"]) != str(shard["sha256"]):
                        raise SpatialEvalExternalContractError(
                            f"source parquet hash mismatch in {reference['sample_id']}"
                        )
                    raw = _image_bytes(row["image"], path.parent, row_index)
                    raw_sha = sha256_bytes(raw)
                    if raw_sha != reference["image_sha256"]:
                        raise SpatialEvalExternalContractError(
                            f"image SHA256 mismatch for {reference['sample_id']}"
                        )
                    width, height, extension = _validate_image(raw, row_index)
                    if (width, height) != (
                        int(reference["width"]),
                        int(reference["height"]),
                    ):
                        raise SpatialEvalExternalContractError(
                            f"image dimensions mismatch for {reference['sample_id']}"
                        )
                    destination = Path(image_cache) / f"{raw_sha}{extension}"
                    _atomic_write_new_or_equal(destination, raw)
                    paths_by_sample[str(reference["sample_id"])] = str(destination)
                row_index += 1
        if row_index != int(shard["rows"]):
            raise SpatialEvalExternalContractError(
                f"parquet row count {row_index} != pinned {shard['rows']} for {path}"
            )
    if len(paths_by_sample) != PRIMARY_SAMPLE_COUNT:
        raise SpatialEvalExternalContractError(
            f"materialized sample paths {len(paths_by_sample)} != {PRIMARY_SAMPLE_COUNT}"
        )

    samples: list[dict[str, Any]] = []
    for reference in reference_records:
        sample_id = str(reference["sample_id"])
        task, qid = task_and_qid(sample_id)
        if task not in PRIMARY_TASKS:
            continue
        prompt = f"{reference['text']}\n{OFFICIAL_REASON_SUFFIX}"
        samples.append(
            {
                "sample_id": sample_id,
                "task": task,
                "question_id": qid,
                "type": "vqa",
                "source": "SpatialEval",
                "source_dataset": "MilaWang/SpatialEval",
                "image": paths_by_sample[sample_id],
                "image_sha256": reference["image_sha256"],
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "oracle_answer": reference["oracle_answer"],
                "oracle_option": reference["oracle_option"],
                "oracle_full_answer": reference["oracle_full_answer"],
            }
        )
    if len(samples) != PRIMARY_SAMPLE_COUNT:
        raise SpatialEvalExternalContractError(
            f"primary sample count {len(samples)} != {PRIMARY_SAMPLE_COUNT}"
        )
    return samples


def sanitize_sample_for_selector(sample: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {key: sample[key] for key in SELECTOR_SAMPLE_KEYS if key in sample}
    leaked = [key for key in FORBIDDEN_SELECTOR_KEYS if key in sanitized]
    if leaked:
        raise SpatialEvalExternalContractError(f"selector input leaks labels: {leaked}")
    required = ("sample_id", "task", "image", "prompt", "max_new_tokens")
    missing = [key for key in required if key not in sanitized]
    if missing:
        raise SpatialEvalExternalContractError(
            f"selector input missing fields {missing}"
        )
    return sanitized


def ensure_output_path(arm: str, candidate: Path | None) -> Path:
    if arm not in ARMS:
        raise SpatialEvalExternalContractError(f"unknown arm {arm!r}")
    pinned = ARM_OUTPUT_PATHS[arm]
    if candidate is not None and Path(candidate).resolve() != pinned.resolve():
        raise SpatialEvalExternalContractError(
            f"output {candidate} != pinned path {pinned}"
        )
    return pinned
