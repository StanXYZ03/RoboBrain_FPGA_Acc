#!/usr/bin/env python3
"""Download and validate the pinned RoboBrain GGUF artifacts.

The downloader is resumable and never overwrites a completed file. Validation
records SHA-256, byte size and safe GGUF metadata in a server-side lock file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from gguf_metadata import GGUFError, read_metadata


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = ROOT / "deployment/config/model.json"
CHUNK = 8 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    part = target.with_name(target.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "robobrain-zcu102/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists():
            os.replace(part, target)
            return
        raise
    status = getattr(response, "status", response.getcode())
    if offset and status != 206:
        raise RuntimeError("server ignored Range; refusing to overwrite partial download")
    mode = "ab" if offset else "xb"
    target.parent.mkdir(parents=True, exist_ok=True)
    with response, part.open(mode) as stream:
        shutil.copyfileobj(response, stream, CHUNK)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, target)


def validate_file(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    size = path.stat().st_size
    if "bytes" in spec:
        if size != spec["bytes"]:
            raise RuntimeError(f"{path.name}: size {size}, expected exactly {spec['bytes']}")
    elif not spec["min_bytes"] <= size <= spec["max_bytes"]:
        raise RuntimeError(f"{path.name}: size {size} is outside policy bounds")
    metadata = read_metadata(path)
    architecture = metadata.get("general.architecture")
    if architecture != spec["architecture"]:
        raise RuntimeError(
            f"{path.name}: architecture {architecture!r}, expected {spec['architecture']!r}"
        )
    if "file_type" in spec and metadata.get("general.file_type") != spec["file_type"]:
        raise RuntimeError(
            f"{path.name}: file type {metadata.get('general.file_type')!r}, "
            f"expected Q8_0 ({spec['file_type']})"
        )
    name = str(metadata.get("general.name", "")).lower()
    if "robobrain" not in name:
        raise RuntimeError(f"{path.name}: unexpected general.name {name!r}")
    actual_sha256 = sha256(path)
    if "sha256" in spec and actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            f"{path.name}: SHA-256 {actual_sha256}, expected {spec['sha256']}"
        )
    return {
        "role": spec["role"],
        "name": path.name,
        "bytes": size,
        "sha256": actual_sha256,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("download", "verify"))
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument(
        "--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face-compatible endpoint; repository and revision remain pinned",
    )
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    directory = args.directory.resolve()
    lock_path = args.lock or directory / "model.lock.json"
    results = []
    for spec in policy["files"]:
        target = directory / spec["name"]
        if args.command == "download" and not target.exists():
            endpoint = args.endpoint.rstrip("/")
            url = f"{endpoint}/{policy['repo_id']}/resolve/{policy['revision']}/{spec['name']}?download=true"
            print(f"download: {url}", flush=True)
            download(url, target)
        if not target.is_file():
            raise FileNotFoundError(target)
        print(f"verify: {target}", flush=True)
        results.append(validate_file(target, spec))
    lock = {
        "schema": 1,
        "repo_id": policy["repo_id"],
        "revision": policy["revision"],
        "source_model": policy["source_model"],
        "official": policy["official"],
        "files": results,
    }
    lock_path.write_text(json.dumps(lock, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"PASS: wrote {lock_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GGUFError, OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
