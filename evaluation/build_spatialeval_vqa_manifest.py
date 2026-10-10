#!/usr/bin/env python3
"""Build an offline SpatialEval VQA manifest from explicit local parquet shards.

This adapter never downloads data. It streams every required column with
``pyarrow.parquet.ParquetFile.iter_batches`` so a multi-shard, hundreds-of-MB
input is never fully materialised in memory.

Two storage modes are supported:

``extract``
    Write the embedded (or explicitly local) image bytes to ``image_dir`` and
    reference the extracted copy from each record. This keeps the original
    single-parquet behaviour.

``parquet_reference``
    Do not create ``image_dir`` and do not copy image bytes. Each record still
    validates the embedded image with Pillow and records its hash and size, but
    references the source parquet/row instead of an extracted file. This avoids
    duplicating the official 712 MB VQA test shards on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


REQUIRED_FIELDS = (
    "id",
    "text",
    "image",
    "oracle_answer",
    "oracle_option",
    "oracle_full_answer",
)

STORAGE_MODES = ("extract", "parquet_reference")

DEFAULT_BATCH_SIZE = 256


class SpatialEvalManifestError(RuntimeError):
    """Raised when the offline manifest contract is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_new_path(path: Path, label: str) -> None:
    # ``lexists`` also refuses a broken symlink instead of silently replacing it.
    if os.path.lexists(path):
        raise SpatialEvalManifestError(f"{label} already exists: {path}")


def _reject_overlapping_destinations(destinations: Sequence[tuple[Path, str]]) -> None:
    """Refuse duplicate or nested destinations before any filesystem change.

    One output is a file and another is a directory; if they overlap, the later
    ``os.replace`` can turn the file path into a directory (or fail) and leave a
    partially published manifest behind.
    """

    resolved = [(path.resolve(), label) for path, label in destinations]
    for index, (path, label) in enumerate(resolved):
        for other_path, other_label in resolved[index + 1 :]:
            if path == other_path:
                raise SpatialEvalManifestError(
                    f"{label} and {other_label} resolve to the same path: {path}"
                )
            if path in other_path.parents or other_path in path.parents:
                raise SpatialEvalManifestError(
                    f"{label} and {other_label} must not be nested: {path} / {other_path}"
                )


def _staged_temp_file(directory: Path, prefix: str, suffix: str) -> Path:
    """Create an empty temp file in ``directory`` and return its path."""

    descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(name)


def _normalize_parquet_paths(
    parquet_paths: Sequence[Path] | Path | None,
    parquet_path: Path | None,
) -> tuple[Path, ...]:
    """Normalize the shard argument into an ordered tuple of paths.

    ``parquet_path`` is the legacy single-shard keyword kept for backward
    compatibility; a single ``Path`` passed as ``parquet_paths`` is accepted too.
    """

    if parquet_path is not None:
        if parquet_paths is not None:
            raise SpatialEvalManifestError(
                "pass parquet_paths (new) or parquet_path (legacy), not both"
            )
        parquet_paths = parquet_path
    if parquet_paths is None:
        raise SpatialEvalManifestError("parquet_paths is required")
    if isinstance(parquet_paths, (str, os.PathLike)):
        parquet_paths = [parquet_paths]
    try:
        paths = tuple(Path(path) for path in parquet_paths)
    except TypeError as exc:
        raise SpatialEvalManifestError(
            "parquet_paths must be a path or a sequence of paths"
        ) from exc
    if not paths:
        raise SpatialEvalManifestError("parquet_paths must contain at least one shard")
    return paths


def _nonempty_string(row: Mapping[str, Any], field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SpatialEvalManifestError(
            f"row {row_index} field {field!r} must be a non-empty string"
        )
    return value


def _image_bytes(value: Any, parquet_dir: Path, row_index: int) -> bytes:
    if not isinstance(value, Mapping):
        raise SpatialEvalManifestError(
            f"row {row_index} image must be one embedded image struct, got {type(value).__name__}"
        )
    raw = value.get("bytes")
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytearray):
        raw = bytes(raw)
    if isinstance(raw, bytes) and raw:
        return raw
    local_path = value.get("path")
    if not isinstance(local_path, str) or not local_path:
        raise SpatialEvalManifestError(f"row {row_index} image has no bytes or local path")
    candidate = Path(local_path)
    if not candidate.is_absolute():
        candidate = parquet_dir / candidate
    if not candidate.is_file():
        raise SpatialEvalManifestError(f"row {row_index} local image does not exist: {candidate}")
    return candidate.read_bytes()


def _validate_image(raw: bytes, row_index: int) -> tuple[int, int, str]:
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as image:
            image.verify()
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
            image_format = (image.format or "").upper()
    except Exception as exc:
        raise SpatialEvalManifestError(f"row {row_index} contains a bad image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise SpatialEvalManifestError(f"row {row_index} image has invalid size {width}x{height}")
    extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image_format)
    if extension is None:
        raise SpatialEvalManifestError(
            f"row {row_index} image format {image_format!r} is not supported"
        )
    return int(width), int(height), extension


def iter_parquet_rows(parquet_path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[dict[str, Any]]]:
    """Yield one list of row dicts per streamed parquet batch.

    Only ``iter_batches`` is used, so the full shard is never loaded at once and
    within-shard row order is preserved. Missing required columns surface as a
    ``SpatialEvalManifestError``.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SpatialEvalManifestError("pyarrow is required to read the local parquet") from exc
    if batch_size <= 0:
        raise SpatialEvalManifestError("batch_size must be positive")
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=list(REQUIRED_FIELDS)
        ):
            yield batch.to_pylist()
    except SpatialEvalManifestError:
        raise
    except Exception as exc:
        raise SpatialEvalManifestError(
            f"cannot stream required parquet fields from {parquet_path}: {exc}"
        ) from exc


def build_manifest(
    *,
    parquet_paths: Sequence[Path] | Path | None = None,
    parquet_path: Path | None = None,
    output_path: Path,
    summary_path: Path,
    image_dir: Path | None = None,
    storage_mode: str = "extract",
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    shard_paths = _normalize_parquet_paths(parquet_paths, parquet_path)
    if storage_mode not in STORAGE_MODES:
        raise SpatialEvalManifestError(
            f"storage_mode must be one of {STORAGE_MODES}, got {storage_mode!r}"
        )
    extract = storage_mode == "extract"
    if extract and image_dir is None:
        raise SpatialEvalManifestError("storage_mode 'extract' requires image_dir")

    if extract:
        destinations = (
            (output_path, "manifest output"),
            (summary_path, "summary output"),
            (image_dir, "image directory"),
        )
    else:
        # Reference mode neither creates nor requires an image directory.
        destinations = (
            (output_path, "manifest output"),
            (summary_path, "summary output"),
        )
    for path, label in destinations:
        _require_new_path(path, label)
    _reject_overlapping_destinations(destinations)

    # Freeze every shard's existence, SHA256 and byte size before any read so a
    # truncated or swapped shard cannot be mixed across the manifest.
    for shard_path in shard_paths:
        if not shard_path.is_file():
            raise SpatialEvalManifestError(f"parquet does not exist: {shard_path}")
    shards: list[dict[str, Any]] = [
        {
            "path": str(shard_path),
            "sha256": sha256_file(shard_path),
            "bytes": shard_path.stat().st_size,
            "row_count": 0,
        }
        for shard_path in shard_paths
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    stage_root: Path | None = None
    stage_images: Path | None = None
    if extract:
        image_dir.parent.mkdir(parents=True, exist_ok=True)
        stage_root = Path(tempfile.mkdtemp(prefix=".spatialeval-stage-", dir=image_dir.parent))
        stage_images = stage_root / "images"
        stage_images.mkdir()

    staged_files: list[Path] = []
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    try:
        global_index = 0
        for shard_path, shard in zip(shard_paths, shards):
            shard_row_count = 0
            for rows in iter_parquet_rows(shard_path, batch_size=batch_size):
                for row in rows:
                    missing = [field for field in REQUIRED_FIELDS if field not in row]
                    if missing:
                        raise SpatialEvalManifestError(
                            f"row {global_index} missing fields: {missing}"
                        )
                    sample_id = _nonempty_string(row, "id", global_index)
                    if sample_id in seen_ids:
                        raise SpatialEvalManifestError(f"duplicate id {sample_id!r}")
                    seen_ids.add(sample_id)
                    raw = _image_bytes(row["image"], shard_path.parent, global_index)
                    width, height, extension = _validate_image(raw, global_index)
                    image_sha = sha256_bytes(raw)
                    record = {
                        "sample_id": sample_id,
                        "text": _nonempty_string(row, "text", global_index),
                        "oracle_answer": _nonempty_string(row, "oracle_answer", global_index),
                        "oracle_option": _nonempty_string(row, "oracle_option", global_index),
                        "oracle_full_answer": _nonempty_string(
                            row, "oracle_full_answer", global_index
                        ),
                        "image_sha256": image_sha,
                        "width": width,
                        "height": height,
                    }
                    if extract:
                        assert stage_images is not None
                        image_name = f"{global_index:06d}-{image_sha[:16]}{extension}"
                        (stage_images / image_name).write_bytes(raw)
                        record["image_path"] = str(image_dir / image_name)
                    else:
                        record["source_parquet_path"] = str(shard_path)
                        record["source_parquet_sha256"] = shard["sha256"]
                        record["source_parquet_row_index"] = shard_row_count
                        record["image_storage"] = "parquet_embedded_or_local"
                    record["record_sha256"] = sha256_bytes(
                        canonical_json(record).encode("utf-8")
                    )
                    records.append(record)
                    shard_row_count += 1
                    global_index += 1
            shard["row_count"] = shard_row_count

        if not records:
            raise SpatialEvalManifestError("parquet contains no rows")

        # Stage every file in its own destination directory so each final
        # ``os.replace`` stays on one filesystem (a cross-device replace would
        # fail after the image directory was already published).
        manifest_tmp = _staged_temp_file(output_path.parent, ".spatialeval-manifest-", ".jsonl")
        staged_files.append(manifest_tmp)
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(canonical_json(record) + "\n")
        summary: dict[str, Any] = {
            "status": "MANIFEST_BUILT_OFFLINE",
            "dataset": "MilaWang/SpatialEval",
            "config": "vqa",
            "split": "test",
            "storage_mode": storage_mode,
            "row_count": len(records),
            "unique_id_count": len(seen_ids),
            "source_parquets": shards,
            "manifest": {
                "path": str(output_path),
                "sha256": sha256_file(manifest_tmp),
                "bytes": manifest_tmp.stat().st_size,
            },
        }
        if extract:
            summary["image_dir"] = str(image_dir)
        summary_tmp = _staged_temp_file(summary_path.parent, ".spatialeval-summary-", ".json")
        staged_files.append(summary_tmp)
        summary_tmp.write_text(canonical_json(summary) + "\n", encoding="utf-8")

        # Publish images first (extract mode), then the manifest that references
        # them, then the summary that references the manifest; a failure leaves
        # no manifest pointing at missing images. Reference mode only publishes
        # the manifest and summary.
        if extract:
            assert stage_images is not None
            os.replace(stage_images, image_dir)
        os.replace(manifest_tmp, output_path)
        staged_files.remove(manifest_tmp)
        os.replace(summary_tmp, summary_path)
        staged_files.remove(summary_tmp)
        return summary
    finally:
        if stage_root is not None:
            shutil.rmtree(stage_root, ignore_errors=True)
        for staged_file in staged_files:
            try:
                staged_file.unlink()
            except FileNotFoundError:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        dest="parquet_paths",
        type=Path,
        action="append",
        required=True,
        help="local parquet shard; repeat in order to stream multiple shards",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="extracted image directory; required for --storage-mode extract",
    )
    parser.add_argument(
        "--storage-mode",
        choices=STORAGE_MODES,
        default="extract",
        help="extract copies image bytes; parquet_reference only references them",
    )
    args = parser.parse_args(argv)
    summary = build_manifest(
        parquet_paths=args.parquet_paths,
        output_path=args.output,
        summary_path=args.summary,
        image_dir=args.image_dir,
        storage_mode=args.storage_mode,
    )
    print(canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
