#!/usr/bin/env python3
"""Stream-hash local RoboBrain2.5-4B model identity files into a deterministic JSON.

The frozen v2 preregistration pins code, datasets and deployment checkpoints,
but not the local model weight shards.  This builder closes that gap for future
experiments:

* it streams every file through ``hashlib.sha256`` in fixed-size blocks, so the
  ~9.7 GB ``model.safetensors`` is never loaded into RAM;
* it includes every local ``*.safetensors`` shard, every
  ``*.safetensors.index.json`` and every existing processor / tokenizer /
  chat-template / config file;
* when an index file exists, every shard it references must exist on disk or the
  build fails closed;
* it records relative path, byte size and SHA256 and writes deterministic,
  sorted JSON (no wall-clock timestamp), so two runs are byte-identical.

CPU-only and stdlib-only.  It never opens a model framework, never touches the
GPU and never writes into the model directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = Path("/home/zhaoxiangyu/RoboBrain2.5/RoboBrain2.5-4B")
DEFAULT_OUTPUT = ROOT / "artifacts" / "provenance" / "robobrain25-4b-model-identity.json"

#: Exact filenames that define processor / tokenizer / chat-template / config identity.
DEFAULT_IDENTITY_FILENAMES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "spiece.model",
    "sentencepiece.bpe.model",
)

CHUNK_SIZE = 1 << 20  # 1 MiB streaming blocks


class ModelIdentityError(RuntimeError):
    """Raised when the model identity contract is violated (fail closed)."""


def sha256_file(path: Path, *, chunk_size: int = CHUNK_SIZE) -> str:
    """Stream ``path`` through SHA256 without loading it into RAM."""

    if chunk_size <= 0:
        raise ModelIdentityError(f"chunk_size must be positive, got {chunk_size!r}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _relative_posix(path: Path, model_dir: Path) -> str:
    return path.relative_to(model_dir).as_posix()


def discover_weight_shards(model_dir: Path) -> list[Path]:
    return sorted(
        (path for path in model_dir.rglob("*.safetensors") if path.is_file()),
        key=lambda path: _relative_posix(path, model_dir),
    )


def discover_index_files(model_dir: Path) -> list[Path]:
    return sorted(
        (path for path in model_dir.rglob("*.safetensors.index.json") if path.is_file()),
        key=lambda path: _relative_posix(path, model_dir),
    )


def discover_named_identity_files(
    model_dir: Path, filenames: Sequence[str] = DEFAULT_IDENTITY_FILENAMES
) -> list[Path]:
    found: list[Path] = []
    for name in filenames:
        candidate = model_dir / name
        if candidate.is_file():
            found.append(candidate)
    return sorted(found, key=lambda path: _relative_posix(path, model_dir))


def referenced_shards(index_path: Path) -> list[str]:
    """Return the shard paths referenced by a safetensors index, failing closed."""

    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIdentityError(f"cannot parse index {index_path}: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ModelIdentityError(f"index {index_path} has no weight_map mapping")
    shards = sorted({str(value) for value in weight_map.values()})
    if not shards:
        raise ModelIdentityError(f"index {index_path} references no shards")
    return shards


def _role_for(relative_path: str, *, is_index: bool) -> str:
    if is_index:
        return "index"
    if relative_path.endswith(".safetensors"):
        return "weight_shard"
    name = Path(relative_path).name
    if "tokenizer" in name or name in {"vocab.json", "merges.txt", "added_tokens.json"}:
        return "tokenizer"
    if name in {"special_tokens_map.json", "spiece.model", "sentencepiece.bpe.model"}:
        return "tokenizer"
    if name.startswith("chat_template"):
        return "chat_template"
    if "processor" in name:
        return "processor"
    return "config"


def build_model_identity_manifest(
    model_dir: str | Path = DEFAULT_MODEL_PATH,
    *,
    identity_filenames: Sequence[str] = DEFAULT_IDENTITY_FILENAMES,
) -> dict[str, Any]:
    """Build the deterministic model identity manifest (no timestamp)."""

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ModelIdentityError(f"model directory does not exist: {model_dir}")

    index_files = discover_index_files(model_dir)
    shards = discover_weight_shards(model_dir)
    if not shards:
        raise ModelIdentityError(f"no *.safetensors shards found under {model_dir}")

    # Fail closed when an index references a shard that is not present.
    referenced: dict[str, list[str]] = {}
    for index_path in index_files:
        rel_index = _relative_posix(index_path, model_dir)
        refs = referenced_shards(index_path)
        referenced[rel_index] = refs
        for ref in refs:
            target = model_dir / ref
            if not target.is_file():
                raise ModelIdentityError(
                    f"index {rel_index} references missing shard {ref!r}"
                )

    named = discover_named_identity_files(model_dir, identity_filenames)

    # Deterministic union of all identity files, keyed by relative path.
    index_rel = {_relative_posix(path, model_dir) for path in index_files}
    unique: dict[str, Path] = {}
    for path in [*shards, *index_files, *named]:
        unique[_relative_posix(path, model_dir)] = path

    files: list[dict[str, Any]] = []
    for relative in sorted(unique):
        path = unique[relative]
        size = path.stat().st_size
        files.append(
            {
                "relative_path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
                "role": _role_for(relative, is_index=relative in index_rel),
            }
        )

    weight_rel = sorted(_relative_posix(path, model_dir) for path in shards)
    return {
        "schema_version": "1.0",
        "kind": "robobrain_model_identity",
        "hash_algorithm": "sha256",
        "streaming_chunk_bytes": CHUNK_SIZE,
        "model_dir": str(model_dir),
        "model_label": model_dir.name,
        "identity_filenames": list(identity_filenames),
        "file_count": len(files),
        "weight_shard_count": len(weight_rel),
        "index_file_count": len(index_files),
        "total_bytes": sum(entry["bytes"] for entry in files),
        "weight_shards": weight_rel,
        "index_files": sorted(index_rel),
        "referenced_shards": referenced,
        "files": files,
    }


def dumps_manifest(manifest: Mapping[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_manifest(manifest: Mapping[str, Any], output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_manifest(manifest), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"local model directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"manifest JSON output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and compare against the existing output; do not write",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_model_identity_manifest(args.model)
    rendered = dumps_manifest(manifest)
    if args.check:
        if not args.output.is_file():
            raise ModelIdentityError(f"manifest not found: {args.output}")
        existing = args.output.read_text(encoding="utf-8")
        if existing != rendered:
            raise ModelIdentityError(f"manifest mismatch: {args.output}")
        print(
            json.dumps(
                {
                    "check": "MATCH",
                    "output": str(args.output),
                    "file_count": manifest["file_count"],
                    "total_bytes": manifest["total_bytes"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    output = write_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(output),
                "model_dir": manifest["model_dir"],
                "file_count": manifest["file_count"],
                "weight_shard_count": manifest["weight_shard_count"],
                "total_bytes": manifest["total_bytes"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ModelIdentityError, OSError) as exc:  # fail closed
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        raise SystemExit(2)
