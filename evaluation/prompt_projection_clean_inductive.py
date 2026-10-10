#!/usr/bin/env python3
"""Reproducible builder/validator for the strict-inductive prompt-projection corpus.

This tool materialises the audited *candidate A* clean corpus for the B1
prompt-projection distillation:

* source records are exactly ``visual_refinement_20260912_development.jsonl``
  and ``visual_refinement_20260912_calibration.jsonl`` (the sealed test manifest
  is never read);
* every record whose ``coco_image_id`` belongs to the COCO subset of the frozen
  CV-Bench parquet files (``test_2d.parquet`` + ``test_3d.parquet``) is removed;
* the remaining records get a new stable ``sample_id`` while the original source
  record id is preserved in ``source_sample_id`` for traceability.

The companion lock file freezes the two source manifests, the two CV-Bench
parquet files, the four formal CV-Bench manifests, the excluded and retained
COCO image ids, the per-image path/size/sha256, the record/task counts, the
final manifest hash, the construction-script hash and ``strict_inductive=true``.

The validator re-derives all of that and additionally proves the overlap gates
required by the governance task:

* zero retained ``coco_image_id`` overlap with the 805 frozen CV-Bench COCO ids;
* zero retained image SHA256 overlap with the formal evaluation image bytes;
* zero new ``sample_id`` overlap with any formal CV-Bench manifest;
* every referenced image exists and matches the frozen size/SHA256;
* all prompts of one image land in exactly one deterministic split;
* the train/val/test splits are all non-empty.

The claim that the prompt *text distribution* is identical to CV-Bench is
explicitly **not** made; grounding/pointing prompts differ from the CV-Bench
Count/Relation/Depth/Distance tasks.

This module only reads local files, never downloads anything, never imports a
model framework and never touches a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_projection_distillation import (  # noqa: E402
    DistillationContractError,
    assign_split,
    load_distillation_config,
    validate_source_image_sha256,
)

SCHEMA = "prompt-projection-clean-inductive-lock-v1"
CLEAN_VERSION = "v1"
SAMPLE_ID_PREFIX = "ppclean-v1"

DEFAULT_MANIFEST_REL = "datasets/prompt_projection_clean_inductive_v1.jsonl"
DEFAULT_LOCK_REL = "datasets/prompt_projection_clean_inductive_v1.lock.json"
DEFAULT_CONFIG_REL = "configs/prompt_projection_distillation.json"
DEFAULT_SCRIPT_REL = "evaluation/prompt_projection_clean_inductive.py"

DEFAULT_SOURCES: tuple[dict[str, str], ...] = (
    {
        "role": "development",
        "path": "datasets/visual_refinement_20260912_development.jsonl",
    },
    {
        "role": "calibration",
        "path": "datasets/visual_refinement_20260912_calibration.jsonl",
    },
)

DEFAULT_CVBENCH_PARQUETS: tuple[str, ...] = (
    "datasets/CV-Bench/test_2d.parquet",
    "datasets/CV-Bench/test_3d.parquet",
)

FORMAL_CVBENCH_MANIFESTS: tuple[dict[str, str], ...] = (
    {
        "id": "full2638_16tok",
        "path": "datasets/cv_bench_full2638.jsonl",
    },
    {
        "id": "screening100_16tok",
        "path": "datasets/cv_bench_screening100.jsonl",
    },
    {
        "id": "full2638_64tok",
        "path": "datasets/cv_bench_full2638_budget64_diagnostic.jsonl",
    },
    {
        "id": "screening100_64tok",
        "path": "datasets/cv_bench_screening100_budget64_diagnostic.jsonl",
    },
)

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SEALED = "visual_refinement_20260912_test.jsonl"


class CleanInductiveError(ValueError):
    """Raised when the clean-inductive corpus/lock contract is violated."""


# ---------------------------------------------------------------------------
# Hashing / I/O helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_jsonl_records(path: Path | str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise CleanInductiveError(
                    f"{path}:{line_number} is not a JSON object"
                )
            records.append(value)
    return records


def canonical_record_line(record: Mapping[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def canonical_manifest_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    text = "".join(canonical_record_line(record) + "\n" for record in records)
    return text.encode("utf-8")


def ids_digest(image_ids: Iterable[int]) -> str:
    joined = "\n".join(str(int(value)) for value in sorted(set(int(v) for v in image_ids)))
    return sha256_text(joined)


def sha_set_digest(values: Iterable[str]) -> str:
    joined = "\n".join(sorted(str(value) for value in set(values)))
    return sha256_text(joined)


def _relative_to(repo_root: Path, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


# ---------------------------------------------------------------------------
# CV-Bench COCO identity recovery
# ---------------------------------------------------------------------------


def recover_cvbench_coco_image_ids(
    parquet_paths: Sequence[Path | str],
) -> list[int]:
    """Recover the COCO ``coco_image_id`` set from the frozen CV-Bench parquet.

    Only rows with ``source == "COCO"`` are considered.  The COCO identity is
    parsed from the numeric stem of ``source_filename`` (for example
    ``/coco/val2017/000000135902.jpg`` -> ``135902``).
    """
    import pyarrow.parquet as pq  # local import: pyarrow is optional for tests

    found: set[int] = set()
    for raw_path in parquet_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise CleanInductiveError(f"CV-Bench parquet is missing: {path}")
        parquet = pq.ParquetFile(path)
        for group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                group_index, columns=["source", "source_filename"]
            )
            for row in table.to_pylist():
                if str(row["source"]) != "COCO":
                    continue
                name = Path(str(row["source_filename"])).name
                match = re.search(r"(\d+)", name)
                if match is None:
                    raise CleanInductiveError(
                        f"cannot recover COCO id from {row['source_filename']!r}"
                    )
                found.add(int(match.group(1)))
    return sorted(found)


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------


def make_sample_id(coco_image_id: int, coco_annotation_id: int, task: str) -> str:
    """Stable, filesystem-safe new sample id for the clean corpus."""
    safe_task = re.sub(r"[^0-9A-Za-z_-]", "-", str(task))
    return (
        f"{SAMPLE_ID_PREFIX}-{int(coco_image_id):012d}-"
        f"{int(coco_annotation_id)}-{safe_task}"
    )


def clean_record(
    record: Mapping[str, Any],
    *,
    source_path: str,
    source_index: int,
) -> dict[str, Any]:
    if "coco_image_id" not in record:
        raise CleanInductiveError(f"{source_path}:{source_index} has no coco_image_id")
    if "coco_annotation_id" not in record:
        raise CleanInductiveError(
            f"{source_path}:{source_index} has no coco_annotation_id"
        )
    if "sample_id" not in record or not str(record["sample_id"]).strip():
        raise CleanInductiveError(f"{source_path}:{source_index} has no sample_id")
    cleaned = dict(record)
    cleaned["sample_id"] = make_sample_id(
        int(record["coco_image_id"]),
        int(record["coco_annotation_id"]),
        str(record.get("task", "")),
    )
    cleaned["source_sample_id"] = str(record["sample_id"])
    cleaned["source_manifest"] = str(source_path)
    cleaned["source_record_index"] = int(source_index)
    cleaned["clean_inductive_version"] = CLEAN_VERSION
    return cleaned


def build_clean_records(
    sources: Sequence[Path | str],
    excluded_coco_image_ids: Iterable[int],
    *,
    repo_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded = {int(value) for value in excluded_coco_image_ids}
    root = Path(repo_root) if repo_root else Path(".")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    base_coco_image_ids: set[int] = set()
    source_stats: list[dict[str, Any]] = []
    for source in sources:
        given = Path(source)
        read_path = given if given.is_absolute() else (root / given)
        if not read_path.is_file():
            raise CleanInductiveError(f"source manifest is missing: {read_path}")
        source_records = load_jsonl_records(read_path)
        kept = 0
        skipped = 0
        for index, record in enumerate(source_records):
            base_coco_image_ids.add(int(record["coco_image_id"]))
            if int(record["coco_image_id"]) in excluded:
                skipped += 1
                continue
            cleaned = clean_record(
                record, source_path=given.as_posix(), source_index=index
            )
            sample_id = cleaned["sample_id"]
            if sample_id in seen:
                raise CleanInductiveError(f"duplicate clean sample_id {sample_id!r}")
            seen.add(sample_id)
            records.append(cleaned)
            kept += 1
        source_stats.append(
            {
                "path": given.as_posix(),
                "records_in": len(source_records),
                "records_kept": kept,
                "records_excluded": skipped,
            }
        )
    return records, {
        "sources": source_stats,
        "base_coco_image_ids": sorted(base_coco_image_ids),
    }


# ---------------------------------------------------------------------------
# Image facts and split preview
# ---------------------------------------------------------------------------


def per_image_facts(
    records: Sequence[Mapping[str, Any]], *, repo_root: Path
) -> dict[int, dict[str, Any]]:
    facts: dict[int, dict[str, Any]] = {}
    for record in records:
        coco_image_id = int(record["coco_image_id"])
        image_value = Path(str(record["image"]))
        source_manifest = Path(str(record["source_manifest"]))
        if image_value.is_absolute():
            absolute = image_value
        else:
            absolute = (repo_root / source_manifest).parent / image_value
        relative = Path(_relative_to(repo_root, absolute))
        entry = facts.get(coco_image_id)
        if entry is None:
            if not absolute.is_file():
                raise CleanInductiveError(f"source image is missing: {absolute}")
            facts[coco_image_id] = {
                "coco_image_id": coco_image_id,
                "path": relative.as_posix(),
                "bytes": absolute.stat().st_size,
                "sha256": sha256_file(absolute),
            }
        elif entry["path"] != relative.as_posix():
            raise CleanInductiveError(
                f"coco_image_id {coco_image_id} maps to multiple paths"
            )
    return facts


def aggregate_image_set_sha256(facts: Mapping[int, Mapping[str, Any]]) -> str:
    """Digest matching the prior audit: sorted ``"<sha256>  <basename>"`` lines."""
    lines = sorted(
        f"{entry['sha256']}  {Path(str(entry['path'])).name}"
        for entry in facts.values()
    )
    return sha256_text("\n".join(lines))


def split_preview(
    facts: Mapping[int, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    split = config["split"]
    ratios = split["ratios"]
    salt = str(split["salt"])
    seed = int(split["seed"])
    image_split = {
        coco_image_id: assign_split(
            str(entry["sha256"]), seed=seed, ratios=ratios, salt=salt
        )
        for coco_image_id, entry in facts.items()
    }
    preview: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        image_ids = sorted(
            coco_image_id
            for coco_image_id, assigned in image_split.items()
            if assigned == name
        )
        record_count = sum(
            1
            for record in records
            if image_split[int(record["coco_image_id"])] == name
        )
        preview[name] = {
            "images": len(image_ids),
            "records": record_count,
            "image_ids": image_ids,
        }
    return {
        "salt": salt,
        "seed": seed,
        "ratios": dict(ratios),
        "group_by": "source_image_sha256",
        "image_to_split": {str(key): value for key, value in sorted(image_split.items())},
        "splits": preview,
    }


# ---------------------------------------------------------------------------
# Lock assembly
# ---------------------------------------------------------------------------


def _manifest_facts(repo_root: Path, path: str) -> dict[str, Any]:
    absolute = repo_root / path
    if not absolute.is_file():
        raise CleanInductiveError(f"manifest is missing: {absolute}")
    records = load_jsonl_records(absolute)
    return {
        "path": path,
        "sha256": sha256_file(absolute),
        "records": len(records),
    }


def build_lock(
    *,
    repo_root: Path,
    manifest_path: Path,
    config_path: Path,
    records: Sequence[Mapping[str, Any]],
    facts: Mapping[int, Mapping[str, Any]],
    excluded_coco_image_ids: Sequence[int],
    retained_coco_image_ids: Sequence[int],
    cvbench_coco_image_ids: Sequence[int],
    source_stats: Mapping[str, Any],
    config: Mapping[str, Any],
    formal_image_sha256_count: int,
    formal_image_sha256_digest: str,
    formal_manifest_specs: Sequence[Mapping[str, str]] = FORMAL_CVBENCH_MANIFESTS,
    parquet_specs: Sequence[str] = DEFAULT_CVBENCH_PARQUETS,
) -> dict[str, Any]:
    script_path = repo_root / DEFAULT_SCRIPT_REL
    if not script_path.is_file():
        raise CleanInductiveError(f"construction script is missing: {script_path}")
    task_counts = Counter(str(record.get("task", "")) for record in records)
    image_ids = sorted(int(value) for value in facts)
    images = [dict(facts[coco_image_id]) for coco_image_id in image_ids]
    split = split_preview(facts, records, config=config)
    manifest_rel = _relative_to(repo_root, manifest_path)

    config_pin: dict[str, Any] | None = None
    extraction = config.get("extraction")
    if isinstance(extraction, Mapping) and isinstance(extraction.get("manifest"), Mapping):
        pin = extraction["manifest"]
        config_pin = {
            "path": str(pin.get("path", "")),
            "sha256": str(pin.get("sha256", "")),
            "matches_manifest": (
                str(pin.get("path", "")) == manifest_rel
                and str(pin.get("sha256", "")) == sha256_file(manifest_path)
            ),
        }

    return {
        "schema": SCHEMA,
        "strict_inductive": True,
        "clean_version": CLEAN_VERSION,
        "sample_id_prefix": SAMPLE_ID_PREFIX,
        "sealed_test_used": False,
        "manifest": {
            "path": manifest_rel,
            "records": len(records),
            "unique_images": len(facts),
            "tasks": {key: int(value) for key, value in sorted(task_counts.items())},
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "sources": [
            {**stat, "sha256": sha256_file(repo_root / stat["path"])}
            for stat in source_stats["sources"]
        ],
        "cvbench_parquet": [
            {"path": path, "sha256": sha256_file(repo_root / path)}
            for path in parquet_specs
        ],
        "cvbench_coco": {
            "coco_image_ids": len(cvbench_coco_image_ids),
            "coco_image_ids_sha256": ids_digest(cvbench_coco_image_ids),
            "excluded_image_ids": sorted(int(value) for value in excluded_coco_image_ids),
            "excluded_image_ids_sha256": ids_digest(excluded_coco_image_ids),
            "retained_image_ids": sorted(int(value) for value in retained_coco_image_ids),
            "retained_image_ids_sha256": ids_digest(retained_coco_image_ids),
        },
        "formal_cvbench_manifests": [
            _manifest_facts(repo_root, spec["path"]) | {"id": spec["id"]}
            for spec in formal_manifest_specs
        ],
        "formal_eval_images": {
            "unique_image_sha256": int(formal_image_sha256_count),
            "sha256_digest": str(formal_image_sha256_digest),
        },
        "images": images,
        "image_set_sha256": aggregate_image_set_sha256(facts),
        "total_image_bytes": int(sum(int(entry["bytes"]) for entry in images)),
        "split_preview": split,
        "config_pin": config_pin,
        "prompt_distribution_matches_cvbench": False,
        "warnings": [
            "grounding/pointing prompt text distribution differs from the CV-Bench "
            "Count/Relation/Depth/Distance tasks; distribution equality is NOT claimed",
            "the internal test split is a sanity check only and never a headline "
            "CV-Bench generalisation number",
            "COCO identity is matched by coco_image_id; image SHA256 differs across "
            "resolutions and must not be used as the COCO identity",
            "sealed visual_refinement_20260912_test.jsonl is never read by this builder",
        ],
        "construction_script": {
            "path": DEFAULT_SCRIPT_REL,
            "sha256": sha256_file(script_path),
        },
    }


# ---------------------------------------------------------------------------
# Formal-data collection and validation gates
# ---------------------------------------------------------------------------


def collect_formal_sample_ids(repo_root: Path) -> set[str]:
    sample_ids: set[str] = set()
    for spec in FORMAL_CVBENCH_MANIFESTS:
        path = repo_root / spec["path"]
        if not path.is_file():
            raise CleanInductiveError(f"formal CV-Bench manifest is missing: {path}")
        for record in load_jsonl_records(path):
            if "sample_id" in record:
                sample_ids.add(str(record["sample_id"]))
    return sample_ids


def collect_formal_eval_image_sha256(repo_root: Path) -> set[str]:
    """Hash every unique image file referenced by the four formal manifests.

    This is the byte-level ``zero formal eval image SHA overlap`` gate.  It is
    intentionally computed from the manifest-referenced image files, not from
    the COCO identity.
    """
    image_paths: set[Path] = set()
    for spec in FORMAL_CVBENCH_MANIFESTS:
        manifest = repo_root / spec["path"]
        if not manifest.is_file():
            raise CleanInductiveError(f"formal CV-Bench manifest is missing: {manifest}")
        for record in load_jsonl_records(manifest):
            image = record.get("image")
            if image is None:
                continue
            candidate = Path(str(image))
            absolute = candidate if candidate.is_absolute() else (repo_root / candidate)
            if absolute.is_file():
                image_paths.add(absolute)
    return {sha256_file(path) for path in sorted(image_paths)}


def validate_manifest_and_lock(
    *,
    repo_root: Path,
    manifest_path: Path,
    lock_path: Path,
    config: Mapping[str, Any],
    formal_sample_ids: Iterable[str],
    formal_eval_image_sha256: Iterable[str],
    recovered_cvbench_coco_ids: Sequence[int],
) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    records = load_jsonl_records(manifest_path)
    failures: list[str] = []
    gates: dict[str, Any] = {}
    manifest_rel = _relative_to(repo_root, manifest_path)

    if lock.get("schema") != SCHEMA:
        failures.append(f"lock schema mismatch: {lock.get('schema')!r}")
    if lock.get("strict_inductive") is not True:
        failures.append("lock strict_inductive must be true")
    if lock.get("sealed_test_used") is not False:
        failures.append("lock sealed_test_used must be false")

    # 1. frozen hashes.
    manifest_sha = sha256_file(manifest_path)
    gates["manifest_sha256"] = {
        "expected": lock.get("manifest", {}).get("sha256"),
        "actual": manifest_sha,
        "passed": lock.get("manifest", {}).get("sha256") == manifest_sha,
    }
    if not gates["manifest_sha256"]["passed"]:
        failures.append("manifest sha256 does not match the lock")

    sources = lock.get("sources", [])
    source_rows: list[dict[str, Any]] = []
    for stat in sources:
        path = repo_root / stat["path"]
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == stat.get("sha256")
        source_rows.append({**stat, "actual_sha256": actual, "passed": passed})
        if not passed:
            failures.append(f"source manifest hash mismatch: {stat['path']}")
    gates["source_manifests"] = source_rows
    for stat in sources:
        if _SEALED in str(stat.get("path", "")):
            failures.append("sealed test manifest must never be a source")

    parquet_rows: list[dict[str, Any]] = []
    for stat in lock.get("cvbench_parquet", []):
        path = repo_root / stat["path"]
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == stat.get("sha256")
        parquet_rows.append({**stat, "actual_sha256": actual, "passed": passed})
        if not passed:
            failures.append(f"CV-Bench parquet hash mismatch: {stat['path']}")
    gates["cvbench_parquet"] = parquet_rows

    formal_rows: list[dict[str, Any]] = []
    for stat in lock.get("formal_cvbench_manifests", []):
        path = repo_root / stat["path"]
        actual = sha256_file(path) if path.is_file() else None
        passed = actual == stat.get("sha256")
        formal_rows.append({**stat, "actual_sha256": actual, "passed": passed})
        if not passed:
            failures.append(f"formal CV-Bench manifest hash mismatch: {stat['path']}")
    gates["formal_cvbench_manifests"] = formal_rows

    script = lock.get("construction_script", {})
    script_path = repo_root / str(script.get("path", ""))
    script_actual = sha256_file(script_path) if script_path.is_file() else None
    script_expected = script.get("sha256")
    script_passed = (
        isinstance(script_expected, str)
        and bool(SHA256_HEX.match(script_expected))
        and script_actual == script_expected
    )
    gates["construction_script"] = {
        "path": script.get("path"),
        "expected": script_expected,
        "actual": script_actual,
        "passed": script_passed,
    }
    if not script_passed:
        failures.append("construction script hash does not match the lock")

    # 2. record/sample-id shape and provenance.
    new_ids = [str(record.get("sample_id", "")) for record in records]
    duplicated = sorted(
        value for value, count in Counter(new_ids).items() if count > 1
    )
    if duplicated:
        failures.append(f"duplicate sample_id in manifest: {duplicated[:5]}")
    bad_prefix = [value for value in new_ids if not value.startswith(SAMPLE_ID_PREFIX)]
    if bad_prefix:
        failures.append(f"sample_id missing prefix {SAMPLE_ID_PREFIX!r}: {bad_prefix[:5]}")
    missing_source = [
        value
        for value, record in zip(new_ids, records)
        if not str(record.get("source_sample_id", "")).strip()
    ]
    if missing_source:
        failures.append("some records lost their source_sample_id")
    gates["record_counts"] = {
        "records": len(records),
        "expected_records": lock.get("manifest", {}).get("records"),
        "passed": len(records) == lock.get("manifest", {}).get("records"),
    }
    if not gates["record_counts"]["passed"]:
        failures.append("record count does not match the lock")
    task_counts = Counter(str(record.get("task", "")) for record in records)
    gates["task_counts"] = {
        "actual": {key: int(value) for key, value in sorted(task_counts.items())},
        "expected": lock.get("manifest", {}).get("tasks"),
        "passed": {key: int(value) for key, value in sorted(task_counts.items())}
        == lock.get("manifest", {}).get("tasks"),
    }
    if not gates["task_counts"]["passed"]:
        failures.append("task counts do not match the lock")

    # 3. excluded / retained COCO ids and the 805 recovery.
    locked_cvbench = lock.get("cvbench_coco", {})
    recovered = sorted(int(value) for value in recovered_cvbench_coco_ids)
    gates["cvbench_coco_ids"] = {
        "recovered": len(recovered),
        "expected": locked_cvbench.get("coco_image_ids"),
        "sha256_expected": locked_cvbench.get("coco_image_ids_sha256"),
        "sha256_actual": ids_digest(recovered),
        "passed": (
            len(recovered) == int(locked_cvbench.get("coco_image_ids", -1))
            and ids_digest(recovered) == locked_cvbench.get("coco_image_ids_sha256")
        ),
    }
    if not gates["cvbench_coco_ids"]["passed"]:
        failures.append("recovered CV-Bench COCO id set does not match the lock")

    manifest_ids = sorted({int(record["coco_image_id"]) for record in records})
    locked_retained = sorted(
        int(value) for value in locked_cvbench.get("retained_image_ids", [])
    )
    if manifest_ids != locked_retained:
        failures.append("retained image id set does not match the lock")
    excluded = {int(value) for value in locked_cvbench.get("excluded_image_ids", [])}
    excluded_in_manifest = set(manifest_ids) & excluded
    if excluded_in_manifest:
        failures.append(
            f"excluded CV-Bench COCO ids reappear in the manifest: {sorted(excluded_in_manifest)[:5]}"
        )
    overlap = set(manifest_ids) & set(recovered)
    gates["zero_cvbench_coco_overlap"] = {
        "retained_unique_images": len(manifest_ids),
        "cvbench_coco_images": len(recovered),
        "overlap": sorted(overlap),
        "passed": not overlap,
    }
    if overlap:
        failures.append(
            f"retained COCO ids overlap CV-Bench: {sorted(overlap)[:5]}"
        )

    # 4. per-image path/size/sha256.
    locked_images = {int(entry["coco_image_id"]): entry for entry in lock.get("images", [])}
    image_problems: list[str] = []
    for coco_image_id in manifest_ids:
        entry = locked_images.get(coco_image_id)
        if entry is None:
            image_problems.append(f"missing lock entry for {coco_image_id}")
            continue
        path = repo_root / str(entry["path"])
        if not path.is_file():
            image_problems.append(f"missing image {entry['path']}")
            continue
        if path.stat().st_size != int(entry["bytes"]):
            image_problems.append(f"size mismatch for {entry['path']}")
            continue
        if sha256_file(path) != entry["sha256"]:
            image_problems.append(f"sha256 mismatch for {entry['path']}")
    gates["images_present_and_frozen"] = {
        "images": len(manifest_ids),
        "problems": image_problems[:10],
        "passed": not image_problems,
    }
    if image_problems:
        failures.append("per-image path/size/sha256 gate failed")

    # 5. zero formal sample_id overlap.
    formal_ids = {str(value) for value in formal_sample_ids}
    id_overlap = sorted(set(new_ids) & formal_ids)
    gates["zero_formal_sample_id_overlap"] = {
        "manifest_sample_ids": len(set(new_ids)),
        "formal_sample_ids": len(formal_ids),
        "overlap": id_overlap[:10],
        "passed": not id_overlap,
    }
    if id_overlap:
        failures.append(f"sample_id overlap with formal manifests: {id_overlap[:5]}")

    # 6. zero formal eval image SHA overlap.
    formal_hashes = {str(value) for value in formal_eval_image_sha256}
    retained_hashes = {str(entry["sha256"]) for entry in locked_images.values()}
    sha_overlap = sorted(retained_hashes & formal_hashes)
    gates["zero_formal_eval_image_sha_overlap"] = {
        "retained_images": len(retained_hashes),
        "formal_eval_images": len(formal_hashes),
        "overlap": sha_overlap[:10],
        "passed": not sha_overlap,
    }
    if sha_overlap:
        failures.append(
            f"image SHA overlap with formal eval images: {sha_overlap[:5]}"
        )

    # 7. split consistency: same image -> exactly one split; all non-empty.
    split = config["split"]
    image_hashes = {
        coco_image_id: validate_source_image_sha256(entry["sha256"])
        for coco_image_id, entry in locked_images.items()
    }
    assigned = {
        coco_image_id: assign_split(
            entry["sha256"],
            seed=int(split["seed"]),
            ratios=split["ratios"],
            salt=str(split["salt"]),
        )
        for coco_image_id, entry in locked_images.items()
    }
    split_violations: list[str] = []
    for record in records:
        coco_image_id = int(record["coco_image_id"])
        if coco_image_id not in assigned:
            split_violations.append(f"image {coco_image_id} not in lock")
    per_image_prompts: defaultdict[int, int] = defaultdict(int)
    for record in records:
        per_image_prompts[int(record["coco_image_id"])] += 1
    split_record_counts = Counter(
        assigned[int(record["coco_image_id"])] for record in records
    )
    nonempty = all(split_record_counts.get(name, 0) > 0 for name in ("train", "val", "test"))
    if not nonempty:
        failures.append("train/val/test must all be non-empty")
    if split_violations:
        failures.append("split assignment violations: " + "; ".join(split_violations[:5]))
    if sorted(image_hashes) != sorted(locked_images):  # pragma: no cover - defensive
        failures.append("image hash key set drift")
    gates["same_image_same_split"] = {
        "images": len(assigned),
        "splits": {
            name: {
                "images": sum(1 for value in assigned.values() if value == name),
                "records": int(split_record_counts.get(name, 0)),
            }
            for name in ("train", "val", "test")
        },
        "all_splits_nonempty": bool(nonempty),
        "passed": nonempty and not split_violations,
    }

    # 8. config pin matches the manifest.
    pin = lock.get("config_pin")
    config_pin_gate: dict[str, Any] = {"present": pin is not None}
    if isinstance(pin, Mapping):
        config_pin_gate.update(
            {
                "path": pin.get("path"),
                "sha256": pin.get("sha256"),
                "matches_manifest": (
                    str(pin.get("path", "")) == manifest_rel
                    and str(pin.get("sha256", "")) == manifest_sha
                ),
            }
        )
        if not config_pin_gate["matches_manifest"]:
            failures.append("lock config_pin does not match the manifest")
    gates["config_pin"] = config_pin_gate

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "gates": gates,
        "manifest": {"path": manifest_rel, "sha256": manifest_sha},
        "lock": {"path": Path(lock_path).as_posix(), "sha256": sha256_file(lock_path)},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> dict[str, Any]:
    return load_distillation_config(path)


def cmd_build(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    lock_path = Path(args.lock)
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    sources = [stat["path"] for stat in DEFAULT_SOURCES]
    config = _load_config(config_path)
    cvbench_ids = recover_cvbench_coco_image_ids(
        [repo_root / path for path in DEFAULT_CVBENCH_PARQUETS]
    )
    records, source_stats = build_clean_records(
        sources, cvbench_ids, repo_root=repo_root
    )
    facts = per_image_facts(records, repo_root=repo_root)
    retained = sorted(facts)
    base_ids = set(int(value) for value in source_stats["base_coco_image_ids"])
    excluded = sorted(base_ids & set(cvbench_ids))
    if set(retained) != (base_ids - set(cvbench_ids)):
        raise CleanInductiveError(
            "retained image ids are not exactly the base set minus CV-Bench COCO ids"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(canonical_manifest_bytes(records))

    formal_sample_ids = collect_formal_sample_ids(repo_root)
    formal_hashes = collect_formal_eval_image_sha256(repo_root)

    lock = build_lock(
        repo_root=repo_root,
        manifest_path=manifest_path,
        config_path=config_path,
        records=records,
        facts=facts,
        excluded_coco_image_ids=excluded,
        retained_coco_image_ids=retained,
        cvbench_coco_image_ids=cvbench_ids,
        source_stats=source_stats,
        config=config,
        formal_image_sha256_count=len(formal_hashes),
        formal_image_sha256_digest=sha_set_digest(formal_hashes),
    )
    lock_path.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = {
        "status": "BUILT",
        "manifest": {"path": manifest_path.as_posix(), "sha256": lock["manifest"]["sha256"]},
        "lock": {"path": lock_path.as_posix(), "sha256": sha256_file(lock_path)},
        "records": lock["manifest"]["records"],
        "unique_images": lock["manifest"]["unique_images"],
        "tasks": lock["manifest"]["tasks"],
        "excluded_image_ids": excluded,
        "retained_image_ids": retained,
        "image_set_sha256": lock["image_set_sha256"],
        "split_preview": lock["split_preview"]["splits"],
        "formal_sample_ids": len(formal_sample_ids),
        "formal_eval_unique_fallback_images": len(formal_hashes),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    lock_path = Path(args.lock)
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    config = _load_config(config_path)
    cvbench_ids = recover_cvbench_coco_image_ids(
        [repo_root / path for path in DEFAULT_CVBENCH_PARQUETS]
    )
    result = validate_manifest_and_lock(
        repo_root=repo_root,
        manifest_path=manifest_path,
        lock_path=lock_path,
        config=config,
        formal_sample_ids=collect_formal_sample_ids(repo_root),
        formal_eval_image_sha256=collect_formal_eval_image_sha256(repo_root),
        recovered_cvbench_coco_ids=cvbench_ids,
    )
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "failures": result["failures"],
                "manifest": result["manifest"],
                "lock": result["lock"],
                "gates": result["gates"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("build", cmd_build), ("validate", cmd_validate)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--repo-root", type=Path, default=None)
        sub.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_REL)
        sub.add_argument("--lock", type=Path, default=DEFAULT_LOCK_REL)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG_REL)
        sub.add_argument("--json-out", type=Path, default=None)
        sub.set_defaults(func=handler)
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (CleanInductiveError, DistillationContractError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
