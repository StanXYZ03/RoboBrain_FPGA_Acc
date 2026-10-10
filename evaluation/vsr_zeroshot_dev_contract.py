#!/usr/bin/env python3
"""Fail-closed contract for the VSR zero-shot development budget curve."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/external/vsr-zeroshot-b27a0af0"
DEV_METADATA = DATA_ROOT / "splits/zeroshot/dev.jsonl"
DEV_METADATA_SHA256 = "0abda42fe9333ab6a8671b32886738b5cfa6d37a8c277b56d8368c339c8dffce"
DEV_IMAGE_MANIFEST = DATA_ROOT / "dev_images_manifest.jsonl"
DEV_IMAGE_MANIFEST_SHA256 = "10ec0f015baaea250245d69c0ea21aa539f6579f4534f993a37f9ad54e902fae"
EVAL_MANIFEST = DATA_ROOT / "vsr_zeroshot_dev_eval_manifest.jsonl"
PREREGISTERED_CONFIG = ROOT / "configs/2026-10-10-vsr-zeroshot-dev-budget-curve.preregistered.json"
OUTPUT_DIR = ROOT / "artifacts/token_selector/2026-10-10-vsr-zeroshot-dev-budget-curve"

SAMPLE_COUNT = 340
MIN_PIXELS = 65536
MAX_PIXELS = 589824
MAX_NEW_TOKENS = 4
BUDGETS = (96, 144, 192)
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
LOCAL_CANDIDATE_CAP_BY_BUDGET = {96: 16, 144: 16, 192: 32}
MAX_SWAPS = 16
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_REPETITIONS = 5000

ARM_DENSE = "dense_edge576"
ARM_L2_PREFIX = "corrected_l2_integer_b1_k"
ARM_V2_PREFIX = "v2_edge576_b1_k"
ARMS = (ARM_DENSE,) + tuple(f"{ARM_L2_PREFIX}{k}" for k in BUDGETS) + tuple(
    f"{ARM_V2_PREFIX}{k}" for k in BUDGETS
)

PROMPT_TEMPLATE = (
    "Statement: {caption}\n"
    "Based only on the image, is this statement true or false? "
    "Answer with exactly True or False."
)


class VSRContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VSRContractError(f"invalid JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise VSRContractError(f"non-object JSON at {path}:{line_number}")
        rows.append(row)
    return rows


def arm_kind_budget(arm: str) -> tuple[str, int | None]:
    if arm == ARM_DENSE:
        return "dense", None
    for prefix, kind in ((ARM_L2_PREFIX, "l2"), (ARM_V2_PREFIX, "v2")):
        if arm.startswith(prefix):
            try:
                budget = int(arm[len(prefix) :])
            except ValueError as exc:
                raise VSRContractError(f"invalid arm {arm!r}") from exc
            if budget not in BUDGETS:
                raise VSRContractError(f"unregistered budget K={budget}")
            return kind, budget
    raise VSRContractError(f"unknown arm {arm!r}")


def output_path(arm: str) -> Path:
    arm_kind_budget(arm)
    return OUTPUT_DIR / f"{arm}.jsonl"


def local_candidate_cap(budget: int) -> int:
    try:
        return LOCAL_CANDIDATE_CAP_BY_BUDGET[int(budget)]
    except KeyError as exc:
        raise VSRContractError(f"no preregistered local candidate cap for K={budget}") from exc


def parse_true_false(text: str) -> bool | None:
    match = re.fullmatch(r"\s*(true|false)\s*[.!]?\s*", str(text), flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group(1).lower() == "true"


def build_eval_manifest() -> list[dict[str, Any]]:
    if sha256_file(DEV_METADATA) != DEV_METADATA_SHA256:
        raise VSRContractError("VSR zero-shot dev metadata hash mismatch")
    if sha256_file(DEV_IMAGE_MANIFEST) != DEV_IMAGE_MANIFEST_SHA256:
        raise VSRContractError("VSR dev image manifest hash mismatch")
    metadata = _load_jsonl(DEV_METADATA)
    images = _load_jsonl(DEV_IMAGE_MANIFEST)
    if len(metadata) != SAMPLE_COUNT:
        raise VSRContractError(f"dev metadata count {len(metadata)} != {SAMPLE_COUNT}")
    image_by_name = {str(row["image"]): row for row in images}
    if len(image_by_name) != len(images):
        raise VSRContractError("duplicate image name in materialization manifest")

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(metadata):
        image_name = str(row["image"])
        image = image_by_name.get(image_name)
        if image is None:
            raise VSRContractError(f"missing materialized image metadata for {image_name}")
        image_path = Path(str(image["path"]))
        if not image_path.is_file() or sha256_file(image_path) != str(image["sha256"]):
            raise VSRContractError(f"materialized image mismatch for {image_name}")
        label = int(row["label"])
        if label not in (0, 1):
            raise VSRContractError(f"invalid binary label at dev row {index}")
        caption = str(row["caption"])
        samples.append(
            {
                "sample_id": f"vsr-zs-dev-{index:04d}",
                "task": "vsr_binary_spatial_reasoning",
                "type": "binary_vqa",
                "source": "VSR",
                "source_dataset": "cambridgeltl/visual-spatial-reasoning",
                "official_split": "zero-shot-dev",
                "image": str(image_path.resolve()),
                "image_name": image_name,
                "image_sha256": str(image["sha256"]),
                "image_width": int(image["width"]),
                "image_height": int(image["height"]),
                "caption": caption,
                "relation": str(row["relation"]),
                "subject": str(row["subj"]),
                "object": str(row["obj"]),
                "official_label": label,
                "prompt": PROMPT_TEMPLATE.format(caption=caption),
                "max_new_tokens": MAX_NEW_TOKENS,
            }
        )
    return samples


def write_eval_manifest() -> dict[str, Any]:
    rows = build_eval_manifest()
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    EVAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if EVAL_MANIFEST.exists() and EVAL_MANIFEST.read_text(encoding="utf-8") != payload:
        raise VSRContractError(f"existing eval manifest differs: {EVAL_MANIFEST}")
    EVAL_MANIFEST.write_text(payload, encoding="utf-8")
    return {"path": str(EVAL_MANIFEST), "sha256": sha256_file(EVAL_MANIFEST), "samples": len(rows)}


def load_eval_manifest(expected_sha256: str | None = None) -> list[dict[str, Any]]:
    if not EVAL_MANIFEST.is_file():
        raise VSRContractError("evaluation manifest is missing; run --prepare-manifest")
    if expected_sha256 is not None and sha256_file(EVAL_MANIFEST) != expected_sha256:
        raise VSRContractError("evaluation manifest hash differs from preregistration")
    rows = _load_jsonl(EVAL_MANIFEST)
    if len(rows) != SAMPLE_COUNT or len({row["sample_id"] for row in rows}) != SAMPLE_COUNT:
        raise VSRContractError("evaluation manifest count or identifiers are invalid")
    return rows


SELECTOR_KEYS = (
    "sample_id", "task", "type", "source", "source_dataset", "image", "prompt", "max_new_tokens"
)


def sanitize_for_selector(sample: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {key: sample[key] for key in SELECTOR_KEYS}
    forbidden = {"official_label", "relation", "caption", "subject", "object"}
    if forbidden.intersection(sanitized):
        raise VSRContractError("label or annotation leaked into selector input")
    return sanitized


def verify_preregistration() -> dict[str, Any]:
    if not PREREGISTERED_CONFIG.is_file():
        raise VSRContractError(f"missing preregistration {PREREGISTERED_CONFIG}")
    config = json.loads(PREREGISTERED_CONFIG.read_text(encoding="utf-8"))
    if config.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise VSRContractError("unexpected preregistration status")
    for name, pin in config.get("pins", {}).items():
        path = Path(pin["path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256_file(path) != pin["sha256"]:
            raise VSRContractError(f"preregistered pin mismatch: {name}")
    return config


def validate_resume(path: Path, samples: Sequence[Mapping[str, Any]], arm: str, manifest_sha: str) -> set[str]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()
    allowed = {str(row["sample_id"]) for row in samples}
    done: set[str] = set()
    kind, budget = arm_kind_budget(arm)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        sample_id = str(row.get("sample_id"))
        if sample_id not in allowed or sample_id in done:
            raise VSRContractError(f"invalid resume sample at line {line_number}")
        if row.get("arm") != arm or row.get("manifest_sha256") != manifest_sha:
            raise VSRContractError(f"resume identity mismatch at line {line_number}")
        if row.get("record_sha256") != canonical_record_sha256(row):
            raise VSRContractError(f"resume record hash mismatch at line {line_number}")
        if kind == "dense" and row.get("requested_budget") is not None:
            raise VSRContractError("dense resume record has a budget")
        if kind != "dense" and int(row.get("requested_budget", -1)) != budget:
            raise VSRContractError("resume budget mismatch")
        done.add(sample_id)
    return done


if __name__ == "__main__":
    print(json.dumps(write_eval_manifest(), indent=2, ensure_ascii=False))
