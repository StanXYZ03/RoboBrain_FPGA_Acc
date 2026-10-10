#!/usr/bin/env python3
"""Post-hoc, CPU-only, read-only heterogeneity diagnostic (full2638 Prompt gain).

Frozen formal result being explained (never changed here):
``PROMPT_GAIN_NOT_CONFIRMED_FULL2638`` from
``reports/cv_bench/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json``
(``prompt_minus_integer_l2`` official delta ``+0.010510006`` with 95% CI
``[-0.005430244, +0.025699187]``; quantisation gate passes).

This module does **not** run a model, GPU, network, training or dependency
install and does not modify any authoritative input.  It re-pairs the two frozen
arm JSONLs exactly by ``sample_id``, re-verifies every SHA-256 pin / record hash /
config field, and then describes - purely exploratorily - where the aggregate
Prompt effect comes from:

* Prompt-minus-integer wrong-to-right / right-to-wrong / unchanged transitions;
* the same transitions and paired accuracy differences broken down by ``task``,
  ``source`` / ``source_dataset``, ``identity_bypass`` and pre-selector real-group
  bins ``<=96, 97-256, 257-512, 513-1024, >1024``;
* paired bootstrap 95% CIs for task / source strata (seed 20261004, 5000 repeats);
* non-bypass selection-index overlap (exact-match rate, Jaccard mean/min);
* whether the aggregate gain is a small residual of large opposing flows and
  whether any task/source regresses.

Hard rules
----------
1. The **only** primary metric stays the official
   ``score(...)["cv_bench_accuracy"]``; nothing here is a new primary metric and
   no subgroup number is used as a gate.
2. The official ``cv_bench_accuracy`` needs all three sources (ADE20K, COCO,
   Omni3D) in a subset and is therefore undefined inside single-source task /
   source / source-dataset strata.  Those stratum CIs therefore use the
   unweighted per-sample accuracy (``score(...)["accuracy"]`` semantics), are
   labelled exploratory and are never a formal decision input.
3. Any missing file, SHA pin mismatch or failed contract check raises
   :class:`InputContractError` before anything is written (fail closed).

Scoring / parsing is imported from :mod:`evaluation.build_cv_bench_pruning_report`
(``parse_choice``, ``answer_letter``, ``score``, ``paired_bootstrap_delta``); the
record / array digest helpers are replicated bit-for-bit from the frozen GPU
runner so this CPU-only script never imports ``torch``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT  # noqa: E402
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

FORMAL_REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json"
)
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
ARM_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
)
ARM_INTEGER_JSONL = ARM_OUTPUT_DIR / "corrected_l2_integer_k96_full2638.jsonl"
ARM_PROMPT_JSONL = (
    ARM_OUTPUT_DIR / "prompt_plus_corrected_l2_integer_k96_full2638.jsonl"
)

REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-full2638-prompt-gain-heterogeneity.json"
)
REPORT_CSV = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-full2638-prompt-gain-heterogeneity.csv"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-full2638-prompt-gain-heterogeneity.md"
)

ARM_INTEGER = "corrected_l2_integer"
ARM_PROMPT = "prompt_plus_corrected_l2_integer"
ARMS = (ARM_INTEGER, ARM_PROMPT)

FORMAL_OUTCOME = "PROMPT_GAIN_NOT_CONFIRMED_FULL2638"
PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = 'score(...)["accuracy"] (unweighted sample accuracy)'
EXPLORATORY_NOTE = (
    "All subgroup / bootstrap / overlap numbers in this report are post-hoc "
    "exploratory descriptions of the frozen PROMPT_GAIN_NOT_CONFIRMED_FULL2638 "
    "result. They are not a new primary metric, not a formal gate and must not "
    "be quoted as evidence for a new claim."
)

BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
FULL_SAMPLE_COUNT = 2638
SCORE_MODE = "integer"
PRUNE_AFTER_LAYER = 0
CUTPOINT = "B1_after_vision_block_0"
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
CI_LEVEL = 0.95

CHECKPOINT_PIN_KEYS = ("checkpoint_sha256", "wv_sha256", "wv_scale_sha256")
CONFIG_PIN_NAMES = {
    "checkpoint_projection_npz": "checkpoint_sha256",
    "checkpoint_wv_int8": "wv_sha256",
    "checkpoint_wv_per_tensor_scale": "wv_scale_sha256",
}

#: ``real_group_count`` on disk is the POST-selection count of REAL source slots
#: (87 for every non-bypass sample; ``G`` for bypass samples).  The requested
#: *pre-selector* real group count is the number of real candidate groups in the
#: full pre-selection visual sequence, which equals ``full_llm_visual_tokens``
#: (one group per 2x2 merge unit) in this B1 setting.
PRE_SELECTOR_REAL_GROUP_COUNT_DEFINITION = (
    "Number of real (non-context) candidate groups before selection = "
    "full_llm_visual_tokens (one group per 2x2 merge unit). The raw JSONL "
    "field 'real_group_count' is the POST-selection count of REAL source slots "
    "(87 for every non-bypass sample; G for bypass samples) and is reported "
    "separately for audit."
)

#: Requested pre-selector ``real_group_count`` bins.
PRE_SELECTOR_BINS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("<=96", lambda value: value <= 96),
    ("97-256", lambda value: 97 <= value <= 256),
    ("257-512", lambda value: 257 <= value <= 512),
    ("513-1024", lambda value: 513 <= value <= 1024),
    (">1024", lambda value: value > 1024),
)
PRE_SELECTOR_BIN_ORDER = tuple(label for label, _ in PRE_SELECTOR_BINS)

#: Pinned SHA-256 of every authoritative input, re-verified fail-closed.
DEFAULT_INPUT_PINS: dict[str, str] = {
    "formal_report": "53ec2ee10fde7b31b08a840263bd26409e35781ed25b89b20166ea90e84af6ea",
    "full_manifest": "22399a767f4c4e9644073d3b2f6bfb6f88377795099012aecb2bfb336ece3d91",
    "corrected_l2_integer_arm": (
        "d0e302f6b91279374ddd507db4d300ca0c29e16f522db57f47c8de059f103a5f"
    ),
    "prompt_plus_corrected_l2_integer_arm": (
        "6d3b97e8ce768066a99c9b9f517ddf602480fa75d57ce2e36b54e26baabc90fa"
    ),
}

TRANSITIONS = (
    "wrong_to_right",
    "right_to_wrong",
    "unchanged_right",
    "unchanged_wrong",
)


class InputContractError(RuntimeError):
    """Raised when an input is missing or violates the frozen contract."""


@dataclass(frozen=True)
class InputPaths:
    formal_report: Path = FORMAL_REPORT_JSON
    full_manifest: Path = FULL_MANIFEST
    arm_integer_jsonl: Path = ARM_INTEGER_JSONL
    arm_prompt_jsonl: Path = ARM_PROMPT_JSONL

    def logical_paths(self) -> dict[str, Path]:
        return {
            "formal_report": self.formal_report,
            "full_manifest": self.full_manifest,
            "corrected_l2_integer_arm": self.arm_integer_jsonl,
            "prompt_plus_corrected_l2_integer_arm": self.arm_prompt_jsonl,
        }


DEFAULT_PATHS = InputPaths()


@dataclass(frozen=True)
class OutputPaths:
    json: Path = REPORT_JSON
    csv: Path = REPORT_CSV
    md: Path = REPORT_MD


DEFAULT_OUTPUTS = OutputPaths()


# ---------------------------------------------------------------------------
# Deterministic digest helpers (bit-identical to the frozen GPU runner)
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    """Canonical SHA256 of an array's dtype, shape and C-order bytes."""

    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    header = f"{contiguous.dtype.str}:{contiguous.shape}".encode("ascii")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself."""

    payload = {
        key: value for key, value in record.items() if key != "record_sha256"
    }
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise InputContractError(
            f"missing required input '{label}': {path} (fail closed)"
        )


def load_rows(path: Path, label: str) -> list[dict[str, Any]]:
    require_file(path, label)
    rows = load_jsonl(path)
    if not rows:
        raise InputContractError(f"input '{label}' is empty: {path} (fail closed)")
    return rows


def index_unique(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    missing: list[int] = []
    for position, row in enumerate(rows):
        if "sample_id" not in row:
            missing.append(position)
            continue
        sample_id = str(row["sample_id"])
        if sample_id in index:
            duplicates.append(sample_id)
        index[sample_id] = dict(row)
    if missing:
        raise InputContractError(
            f"input '{label}' has rows without sample_id at {missing[:10]}"
        )
    if duplicates:
        raise InputContractError(
            f"input '{label}' has duplicate sample_id values: "
            f"{sorted(set(duplicates))[:10]}"
        )
    return index


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputContractError(f"{label} must be an int, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------
def verify_pins(
    paths: InputPaths, expected_pins: Mapping[str, str] | None
) -> dict[str, dict[str, Any]]:
    pinned: dict[str, dict[str, Any]] = {}
    for name, path in paths.logical_paths().items():
        require_file(path, name)
        digest = sha256_file(path)
        pinned[name] = {"path": str(path), "sha256": digest}
        if expected_pins is None:
            continue
        expected = expected_pins.get(name)
        if expected is not None and expected != digest:
            raise InputContractError(
                f"input '{name}' sha256 mismatch: expected {expected}, got "
                f"{digest} ({path}) (fail closed)"
            )
    return pinned


def extract_checkpoint_hashes(report: Mapping[str, Any]) -> dict[str, str]:
    """Read the frozen checkpoint pins out of the formal report."""

    pins: dict[str, str] = {}
    checks = report.get("config_pin_verification", {}).get("checks", [])
    if not isinstance(checks, list):
        raise InputContractError(
            "formal report config_pin_verification.checks must be a list"
        )
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        name = str(check.get("name"))
        if name in CONFIG_PIN_NAMES:
            pins[CONFIG_PIN_NAMES[name]] = str(check.get("expected_sha256"))
    missing = [key for key in CHECKPOINT_PIN_KEYS if key not in pins]
    if missing:
        raise InputContractError(
            f"formal report is missing checkpoint pins {missing} (fail closed)"
        )
    return pins


def verify_formal_report(
    report: Mapping[str, Any],
    *,
    manifest_sha256: str,
    integer_sha256: str,
    prompt_sha256: str,
    expected_sample_count: int = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Confirm the frozen formal result is unchanged and self-consistent."""

    decision = report.get("decision", {})
    outcome = str(decision.get("outcome"))
    if outcome != FORMAL_OUTCOME:
        raise InputContractError(
            f"formal report outcome {outcome!r} != {FORMAL_OUTCOME!r} (fail closed)"
        )
    fixed = report.get("fixed_configuration", {})
    for field, expected in (
        ("budget_K", BUDGET),
        ("region_grid", REGION_GRID),
        ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
        ("integration_score_mode", SCORE_MODE),
        ("bootstrap_seed", BOOTSTRAP_SEED),
        ("bootstrap_repeats", BOOTSTRAP_REPEATS),
        ("sample_count", expected_sample_count),
    ):
        if field not in fixed or fixed[field] != expected:
            raise InputContractError(
                f"formal report fixed_configuration[{field!r}]="
                f"{fixed.get(field)!r} != {expected!r} (fail closed)"
            )
    inputs = report.get("inputs_sha256", {})
    for name, expected in (
        ("full_manifest", manifest_sha256),
        ("corrected_l2_integer_jsonl", integer_sha256),
        ("prompt_plus_corrected_l2_integer_jsonl", prompt_sha256),
    ):
        actual = inputs.get(name, {}).get("sha256")
        if str(actual) != expected:
            raise InputContractError(
                f"formal report inputs_sha256[{name!r}]={actual!r} != {expected!r} "
                "(the frozen report does not describe these exact inputs; fail "
                "closed)"
            )
    comparison = report.get("comparisons", {}).get("prompt_minus_integer_l2", {})
    if comparison.get("metric") != PRIMARY_METRIC:
        raise InputContractError(
            "formal report prompt_minus_integer_l2 metric is not the official "
            f"{PRIMARY_METRIC!r} (fail closed)"
        )
    return {
        "formal_outcome": outcome,
        "metric": PRIMARY_METRIC,
        "frozen_delta_point_estimate": float(
            comparison.get("delta_point_estimate")
        ),
        "frozen_delta_ci95": [
            float(comparison.get("delta_ci95", [0.0, 0.0])[0]),
            float(comparison.get("delta_ci95", [0.0, 0.0])[1]),
        ],
        "frozen_budget_K": int(fixed["budget_K"]),
        "frozen_region_grid": int(fixed["region_grid"]),
        "frozen_local_candidate_cap": int(fixed["local_candidate_cap"]),
    }


def _check_arm_record(
    row: Mapping[str, Any],
    *,
    arm: str,
    manifest_row: Mapping[str, Any],
    manifest_sha256: str,
    checkpoint_hashes: Mapping[str, str] | None,
) -> None:
    sid = str(row.get("sample_id"))
    label = f"{arm}/{sid}"

    if str(row.get("arm")) != arm:
        raise InputContractError(
            f"{label} arm field {row.get('arm')!r} != {arm!r} (fail closed)"
        )
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise InputContractError(
            f"{label} task {row.get('task')!r} != manifest "
            f"{manifest_row.get('task')!r} (fail closed)"
        )
    for field, expected in (
        ("requested_budget", BUDGET),
        ("region_grid", REGION_GRID),
        ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
        ("prune_after_layer", PRUNE_AFTER_LAYER),
    ):
        if field not in row or int(row[field]) != expected:
            raise InputContractError(
                f"{label} {field}={row.get(field)!r} != {expected} (fail closed)"
            )
    for field, expected in (
        ("score_mode", SCORE_MODE),
        ("integration_score_mode", SCORE_MODE),
        ("cutpoint", CUTPOINT),
    ):
        if str(row.get(field)) != expected:
            raise InputContractError(
                f"{label} {field}={row.get(field)!r} != {expected!r} (fail closed)"
            )
    if str(row.get("manifest_sha256")) != manifest_sha256:
        raise InputContractError(
            f"{label} manifest_sha256 != pinned manifest digest (fail closed)"
        )
    if str(row.get("record_sha256")) != record_sha256(row):
        raise InputContractError(
            f"{label} record_sha256 mismatch (stale or tampered record; fail closed)"
        )
    if checkpoint_hashes is not None:
        if arm == ARM_PROMPT:
            for key in CHECKPOINT_PIN_KEYS:
                if str(row.get(key)) != str(checkpoint_hashes[key]):
                    raise InputContractError(
                        f"{label} {key} does not match the config pin (fail closed)"
                    )
        elif row.get("checkpoint_sha256") is not None:
            raise InputContractError(
                f"{label} no-prompt arm must not carry checkpoint_sha256 "
                "(fail closed)"
            )

    grid = [_as_int(value, f"{label}.image_grid_thw") for value in row["image_grid_thw"]]
    full_patch = _as_int(row.get("full_patch_tokens"), f"{label}.full_patch_tokens")
    full_llm = _as_int(
        row.get("full_llm_visual_tokens"), f"{label}.full_llm_visual_tokens"
    )
    if len(grid) != 3 or grid[0] != 1 or grid[1] * grid[2] != full_patch:
        raise InputContractError(f"{label} invalid single-image grid (fail closed)")
    if full_llm * MERGE_UNIT != full_patch:
        raise InputContractError(
            f"{label} full_llm_visual_tokens*{MERGE_UNIT} != full_patch_tokens "
            "(fail closed)"
        )
    output_count = _as_int(row.get("output_group_count"), f"{label}.output_group_count")
    bypass = bool(row.get("identity_bypass"))
    if bypass:
        if output_count != full_llm or output_count <= 0 or output_count > BUDGET:
            raise InputContractError(
                f"{label} identity-bypass must emit G == full_llm_visual_tokens <= "
                f"{BUDGET} (fail closed)"
            )
    elif output_count != BUDGET:
        raise InputContractError(
            f"{label} non-bypass must emit K={BUDGET} (fail closed)"
        )
    selected = [_as_int(value, f"{label}.selected_group_indices") for value in row["selected_group_indices"]]
    if len(selected) != output_count:
        raise InputContractError(
            f"{label} len(selected_group_indices) != output_group_count (fail closed)"
        )
    if len(set(selected)) != len(selected):
        raise InputContractError(
            f"{label} selected_group_indices has duplicates (fail closed)"
        )
    if selected != sorted(selected):
        raise InputContractError(
            f"{label} selected_group_indices is not strictly ascending (fail closed)"
        )
    if any(value < 0 for value in selected):
        raise InputContractError(
            f"{label} selected_group_indices must be non-negative (fail closed)"
        )
    stored = row.get("selected_group_indices_sha256")
    actual = array_sha256(np.asarray(selected, dtype=np.int64))
    if str(stored) != actual:
        raise InputContractError(
            f"{label} selected_group_indices_sha256 mismatch (fail closed)"
        )


def validate_contract(
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    manifest_sha256: str,
    checkpoint_hashes: Mapping[str, str] | None,
    expected_sample_count: int,
) -> dict[str, Any]:
    """Fail-closed id / hash / config / selection contract across both arms."""

    manifest_ids = list(manifest_index)
    if len(manifest_ids) != expected_sample_count:
        raise InputContractError(
            f"manifest must hold {expected_sample_count} unique samples, got "
            f"{len(manifest_ids)} (fail closed)"
        )
    manifest_set = set(manifest_ids)
    for sid, row in manifest_index.items():
        for field in ("task", "source", "source_dataset"):
            if not str(row.get(field, "")).strip():
                raise InputContractError(
                    f"manifest sample {sid!r} is missing metadata field {field!r}"
                )
    for arm in ARMS:
        index = arm_indices[arm]
        if len(index) != expected_sample_count:
            raise InputContractError(
                f"arm {arm!r} must hold {expected_sample_count} unique samples, got "
                f"{len(index)} (fail closed)"
            )
        missing = sorted(manifest_set - set(index))
        extra = sorted(set(index) - manifest_set)
        if missing or extra:
            raise InputContractError(
                f"arm {arm!r} sample_id set != manifest: missing={missing[:10]} "
                f"extra={extra[:10]} (fail closed)"
            )

    for sid in manifest_ids:
        for arm in ARMS:
            _check_arm_record(
                arm_indices[arm][sid],
                arm=arm,
                manifest_row=manifest_index[sid],
                manifest_sha256=manifest_sha256,
                checkpoint_hashes=checkpoint_hashes,
            )

    cross_fields = (
        "full_llm_visual_tokens",
        "full_patch_tokens",
        "identity_bypass",
        "output_group_count",
        "real_group_count",
        "context_group_count",
        "image_grid_thw",
    )
    for sid in manifest_ids:
        for field in cross_fields:
            values = [
                json.dumps(arm_indices[arm][sid].get(field), sort_keys=True)
                for arm in ARMS
            ]
            if len(set(values)) != 1:
                raise InputContractError(
                    f"sample {sid!r} disagrees across arms on {field!r}: "
                    f"{values} (fail closed)"
                )

    return {
        "expected_sample_count": expected_sample_count,
        "sample_ids_identical": True,
        "unique_sample_ids_per_arm": True,
        "task_matches_manifest": True,
        "budget_96_region_grid_3_local_cap_16": True,
        "score_mode_integer": True,
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "cutpoint": CUTPOINT,
        "record_sha256_verified": True,
        "manifest_sha256_verified": True,
        "selected_indices_strictly_ascending_unique_and_hashed": True,
        "identity_bypass_matches_G": True,
        "non_bypass_emits_K": BUDGET,
        "checkpoint_hashes_verified": checkpoint_hashes is not None,
        "cross_arm_config_fields_consistent": True,
        "inputs_unchanged_during_analysis": True,
        "checks": [
            "exactly N identical unique sample_ids in the manifest and both arm files",
            "no duplicate sample_id in any input",
            "every arm record carries arm / budget=96 / region_grid=3 / local_cap=16 / integer score mode",
            "every arm record manifest_sha256 equals the pinned manifest digest",
            "every arm record record_sha256 re-verifies",
            "task matches the manifest; source/source_dataset come from the manifest",
            "non-bypass emits K=96; bypass emits G == full_llm_visual_tokens",
            "selected_group_indices strictly ascending, unique, length==output_group_count, sha256 re-verified",
            "prompt arm carries the config-pinned checkpoint hashes; integer arm carries none",
            "G / grid / bypass / counts identical across the two arms",
        ],
    }


# ---------------------------------------------------------------------------
# Joining and strata
# ---------------------------------------------------------------------------
def build_joined(
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    formal_joined_index: Mapping[str, Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for sample_id, meta in manifest_index.items():
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "task": str(meta["task"]),
            "type": str(meta["type"]),
            "source": str(meta["source"]),
            "source_dataset": str(meta["source_dataset"]),
            "ground_truth": truth,
        }
        for arm in ARMS:
            record = arm_indices[arm][sample_id]
            parsed = parse_choice(str(record["output_text"]), choices)
            row[f"{arm}_answer"] = parsed
            row[f"{arm}_correct"] = parsed == truth
            row[f"{arm}_indices"] = [
                int(value) for value in record["selected_group_indices"]
            ]
        int_row = arm_indices[ARM_INTEGER][sample_id]
        row["G"] = int(int_row["full_llm_visual_tokens"])
        row["pre_selector_real_group_count"] = row["G"]
        row["recorded_real_group_count"] = int(int_row["real_group_count"])
        row["context_group_count"] = int(int_row["context_group_count"])
        row["identity_bypass"] = bool(int_row["identity_bypass"])
        row["pre_selector_real_group_count_bin"] = pre_selector_bin(row["G"])
        if formal_joined_index is not None:
            frozen = formal_joined_index.get(sample_id)
            if frozen is None:
                raise InputContractError(
                    f"formal report joined is missing sample_id {sample_id!r} "
                    "(fail closed)"
                )
            for arm in ARMS:
                key = f"{arm}_correct"
                if key in frozen and bool(frozen[key]) != bool(row[key]):
                    raise InputContractError(
                        f"sample {sample_id!r} {key} disagrees with the frozen "
                        "formal report (fail closed)"
                    )
            row["dense_answer"] = frozen.get("dense_answer")
            row["dense_correct"] = frozen.get("dense_correct")
        else:
            row["dense_answer"] = None
            row["dense_correct"] = None

        int_set = set(row[f"{ARM_INTEGER}_indices"])
        prompt_set = set(row[f"{ARM_PROMPT}_indices"])
        union = int_set | prompt_set
        row["index_exact_match"] = int_set == prompt_set
        row["index_intersection"] = len(int_set & prompt_set)
        row["index_union"] = len(union)
        row["index_jaccard"] = (
            len(int_set & prompt_set) / len(union) if union else 1.0
        )
        row["transition"] = transition_label(row)
        joined.append(row)
    return joined


def pre_selector_bin(value: int) -> str:
    for label, predicate in PRE_SELECTOR_BINS:
        if predicate(value):
            return label
    return "other"


def transition_label(row: Mapping[str, Any]) -> str:
    integer_right = bool(row[f"{ARM_INTEGER}_correct"])
    prompt_right = bool(row[f"{ARM_PROMPT}_correct"])
    if not integer_right and prompt_right:
        return "wrong_to_right"
    if integer_right and not prompt_right:
        return "right_to_wrong"
    if integer_right and prompt_right:
        return "unchanged_right"
    return "unchanged_wrong"


def _arm_score_rows(
    records: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    correct_key = f"{arm}_correct"
    answer_key = f"{arm}_answer"
    return [
        {
            "task": row["task"],
            "source": row["source"],
            "source_dataset": row["source_dataset"],
            "correct": bool(row[correct_key]),
            "answer": row[answer_key],
        }
        for row in records
    ]


def _pair_score_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "task": row["task"],
            "source": row["source"],
            "source_dataset": row["source_dataset"],
            "left_correct": bool(row[f"{ARM_PROMPT}_correct"]),
            "left_answer": row[f"{ARM_PROMPT}_answer"],
            "right_correct": bool(row[f"{ARM_INTEGER}_correct"]),
            "right_answer": row[f"{ARM_INTEGER}_answer"],
        }
        for row in records
    ]


def official_accuracy(
    records: Sequence[Mapping[str, Any]], arm: str
) -> float | None:
    """Official ``cv_bench_accuracy``; ``None`` if a source is absent."""

    if {str(row["source"]) for row in records} < {"ADE20K", "COCO", "Omni3D"}:
        return None
    return float(score(_arm_score_rows(records, arm), "correct")["cv_bench_accuracy"])


def unweighted_accuracy(
    records: Sequence[Mapping[str, Any]], arm: str
) -> float:
    return statistics.fmean(bool(row[f"{arm}_correct"]) for row in records)


def bootstrap_unweighted_paired_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> list[float]:
    """Stratified paired bootstrap of the unweighted accuracy difference.

    Mirrors :func:`paired_bootstrap_delta`'s ``task::source_dataset`` resampling
    but scores the plain per-sample mean because the official
    ``cv_bench_accuracy`` is undefined inside single-source strata.
    """

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[f"{row['task']}::{row['source_dataset']}"].append(row)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        left = 0
        right = 0
        total = 0
        for group in groups.values():
            for _ in range(len(group)):
                sample = rng.choice(group)
                left += bool(sample[f"{ARM_PROMPT}_correct"])
                right += bool(sample[f"{ARM_INTEGER}_correct"])
                total += 1
        deltas.append((left - right) / total)
    deltas.sort()
    return [
        float(deltas[int(0.025 * (repeats - 1))]),
        float(deltas[int(0.975 * (repeats - 1))]),
    ]


def stratum_row(
    label: str,
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap: bool,
) -> dict[str, Any]:
    n = len(records)
    counts = {name: 0 for name in TRANSITIONS}
    for row in records:
        counts[row["transition"]] += 1
    integer_acc = unweighted_accuracy(records, ARM_INTEGER)
    prompt_acc = unweighted_accuracy(records, ARM_PROMPT)
    integer_official = official_accuracy(records, ARM_INTEGER)
    prompt_official = official_accuracy(records, ARM_PROMPT)
    result: dict[str, Any] = {
        "stratum": label,
        "n": n,
        "wrong_to_right": counts["wrong_to_right"],
        "right_to_wrong": counts["right_to_wrong"],
        "unchanged_right": counts["unchanged_right"],
        "unchanged_wrong": counts["unchanged_wrong"],
        "wrong_to_right_rate": counts["wrong_to_right"] / n,
        "right_to_wrong_rate": counts["right_to_wrong"] / n,
        "unchanged_right_rate": counts["unchanged_right"] / n,
        "unchanged_wrong_rate": counts["unchanged_wrong"] / n,
        "net_corrections": counts["wrong_to_right"] - counts["right_to_wrong"],
        "integer_unweighted_accuracy": integer_acc,
        "prompt_unweighted_accuracy": prompt_acc,
        "unweighted_paired_delta": prompt_acc - integer_acc,
        "sources_present": sorted({str(row["source"]) for row in records}),
    }
    official_defined = integer_official is not None and prompt_official is not None
    result["official_metric_defined"] = official_defined
    result["integer_official_cv_bench_accuracy"] = integer_official
    result["prompt_official_cv_bench_accuracy"] = prompt_official
    result["official_cv_bench_accuracy_delta"] = (
        prompt_official - integer_official if official_defined else None
    )
    if bootstrap:
        result["bootstrap_ci95_unweighted_paired_delta"] = (
            bootstrap_unweighted_paired_delta(records)
        )
        result["bootstrap_metric"] = (
            "unweighted sample accuracy (exploratory; the official "
            "cv_bench_accuracy is undefined within this stratum)"
        )
        result["bootstrap_seed"] = BOOTSTRAP_SEED
        result["bootstrap_repeats"] = BOOTSTRAP_REPEATS
    return result


def _group_rows(
    joined: Sequence[Mapping[str, Any]],
    key: str,
    order: Sequence[str] | None = None,
    *,
    bootstrap: bool,
) -> list[dict[str, Any]]:
    labels = sorted({str(row[key]) for row in joined})
    if order is not None:
        labels = [label for label in order if label in set(labels)]
    rows: list[dict[str, Any]] = []
    for label in labels:
        subset = [row for row in joined if str(row[key]) == label]
        rows.append(stratum_row(label, subset, bootstrap=bootstrap))
    return rows


def selection_overlap(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    non_bypass = [row for row in joined if not row["identity_bypass"]]
    n = len(non_bypass)
    if n == 0:
        return {"n": 0, "note": "no non-bypass samples"}
    exact = sum(bool(row["index_exact_match"]) for row in non_bypass)
    jaccards = [float(row["index_jaccard"]) for row in non_bypass]
    by_task: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in non_bypass}):
        subset = [row for row in non_bypass if row["task"] == task]
        subset_exact = sum(bool(row["index_exact_match"]) for row in subset)
        subset_jaccard = [float(row["index_jaccard"]) for row in subset]
        by_task.append(
            {
                "task": task,
                "n": len(subset),
                "exact_match_count": subset_exact,
                "exact_match_rate": subset_exact / len(subset),
                "jaccard_mean": statistics.fmean(subset_jaccard),
                "jaccard_min": min(subset_jaccard),
            }
        )
    return {
        "scope": "non-bypass samples only (both arms emit K=96)",
        "n": n,
        "exact_match_count": exact,
        "exact_match_rate": exact / n,
        "jaccard_mean": statistics.fmean(jaccards),
        "jaccard_min": min(jaccards),
        "jaccard_max": max(jaccards),
        "by_task": by_task,
    }


def concentration(
    joined: Sequence[Mapping[str, Any]],
    by_task: Sequence[Mapping[str, Any]],
    by_source_dataset: Sequence[Mapping[str, Any]],
    by_bin: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    wrong_to_right = sum(row["transition"] == "wrong_to_right" for row in joined)
    right_to_wrong = sum(row["transition"] == "right_to_wrong" for row in joined)
    net = wrong_to_right - right_to_wrong

    def _best(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        best = max(rows, key=lambda item: item["net_corrections"])
        return {
            "stratum": best["stratum"],
            "net_corrections": best["net_corrections"],
        }

    def _worst(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        if not rows:
            return None
        worst = min(rows, key=lambda item: item["net_corrections"])
        return {
            "stratum": worst["stratum"],
            "net_corrections": worst["net_corrections"],
        }

    best_source = _best(by_source_dataset)
    share = (
        best_source["net_corrections"] / net
        if best_source is not None and net > 0
        else None
    )
    return {
        "gross_wrong_to_right": wrong_to_right,
        "gross_right_to_wrong": right_to_wrong,
        "net_corrections": net,
        "gross_churn": wrong_to_right + right_to_wrong,
        "net_is_small_residual_of_opposing_flows": (
            (wrong_to_right + right_to_wrong) > abs(net)
        ),
        "positive_net_task_count": sum(
            row["net_corrections"] > 0 for row in by_task
        ),
        "negative_net_task_count": sum(
            row["net_corrections"] < 0 for row in by_task
        ),
        "positive_net_source_dataset_count": sum(
            row["net_corrections"] > 0 for row in by_source_dataset
        ),
        "negative_net_source_dataset_count": sum(
            row["net_corrections"] < 0 for row in by_source_dataset
        ),
        "largest_positive_task": _best(by_task),
        "largest_negative_task": _worst(by_task),
        "largest_positive_source_dataset": best_source,
        "largest_negative_source_dataset": _worst(by_source_dataset),
        "largest_positive_source_dataset_share_of_net": share,
        "by_task_net": {
            row["stratum"]: row["net_corrections"] for row in by_task
        },
        "by_source_dataset_net": {
            row["stratum"]: row["net_corrections"] for row in by_source_dataset
        },
        "by_pre_selector_bin_net": {
            row["stratum"]: row["net_corrections"] for row in by_bin
        },
    }


def regression_check(
    by_task: Sequence[Mapping[str, Any]],
    by_source_dataset: Sequence[Mapping[str, Any]],
    by_source: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    strata = list(by_task) + list(by_source_dataset) + list(by_source)
    negative = [
        {
            "stratum": row["stratum"],
            "grouping_scope": scope,
            "n": row["n"],
            "unweighted_paired_delta": row["unweighted_paired_delta"],
            "bootstrap_ci95_unweighted_paired_delta": row.get(
                "bootstrap_ci95_unweighted_paired_delta"
            ),
        }
        for scope, rows in (
            ("task", by_task),
            ("source_dataset", by_source_dataset),
            ("source", by_source),
        )
        for row in rows
        if row["unweighted_paired_delta"] < 0
    ]
    entirely_below_zero = [
        item
        for item in negative
        if item["bootstrap_ci95_unweighted_paired_delta"] is not None
        and item["bootstrap_ci95_unweighted_paired_delta"][1] < 0
    ]
    return {
        "negative_point_delta_strata": negative,
        "strata_with_bootstrap_ci_entirely_below_zero": entirely_below_zero,
        "count_negative_point_delta": len(negative),
        "count_ci_entirely_below_zero": len(entirely_below_zero),
    }


def dense_descriptive(
    formal_joined_index: Mapping[str, Mapping[str, Any]] | None,
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Dense accuracy by task/source as descriptive context only."""

    if formal_joined_index is None:
        return None
    rows = []
    for sample_id, frozen in formal_joined_index.items():
        if "dense_correct" not in frozen:
            return None
        rows.append(
            {
                "task": frozen.get("task"),
                "source": frozen.get("source"),
                "source_dataset": frozen.get("source_dataset"),
                "correct": bool(frozen["dense_correct"]),
                "answer": frozen.get("dense_answer"),
            }
        )
    if {str(row["source"]) for row in rows} < {"ADE20K", "COCO", "Omni3D"}:
        return None
    scored = score(rows, "correct")
    frozen_metrics = report.get("metrics", {}).get("dense", {})
    return {
        "role": "descriptive context only; never a gate",
        "source": "frozen formal report joined.dense_correct via the existing scorer",
        "cv_bench_accuracy": float(
            frozen_metrics.get(
                "cv_bench_accuracy", scored["cv_bench_accuracy"]
            )
        ),
        "unweighted_task_accuracy": {
            task: float(value) for task, value in scored["task_accuracy"].items()
        },
        "unweighted_source_accuracy": {
            source: float(value)
            for source, value in scored["source_accuracy"].items()
        },
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_pins: Mapping[str, str] | None = DEFAULT_INPUT_PINS,
    expected_sample_count: int = FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    formal_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the frozen inputs and compute the exploratory diagnostic."""

    pinned = verify_pins(paths, expected_pins)

    manifest_rows = load_rows(paths.full_manifest, "full_manifest")
    manifest_index = index_unique(manifest_rows, "full_manifest")
    arm_rows = {
        ARM_INTEGER: load_rows(paths.arm_integer_jsonl, "corrected_l2_integer"),
        ARM_PROMPT: load_rows(
            paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"
        ),
    }
    arm_indices = {arm: index_unique(rows, arm) for arm, rows in arm_rows.items()}

    integer_sha = pinned["corrected_l2_integer_arm"]["sha256"]
    prompt_sha = pinned["prompt_plus_corrected_l2_integer_arm"]["sha256"]
    actual_manifest_sha = (
        manifest_sha256
        if manifest_sha256 is not None
        else pinned["full_manifest"]["sha256"]
    )

    if formal_report is None:
        if not paths.formal_report.is_file():
            raise InputContractError(
                f"missing required input 'formal_report': {paths.formal_report}"
            )
        formal_report = json.loads(paths.formal_report.read_text(encoding="utf-8"))
    formal_checks = verify_formal_report(
        formal_report,
        manifest_sha256=actual_manifest_sha,
        integer_sha256=integer_sha,
        prompt_sha256=prompt_sha,
        expected_sample_count=expected_sample_count,
    )
    checkpoint_hashes = extract_checkpoint_hashes(formal_report)

    contract = validate_contract(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        manifest_sha256=actual_manifest_sha,
        checkpoint_hashes=checkpoint_hashes,
        expected_sample_count=expected_sample_count,
    )

    formal_joined = formal_report.get("joined")
    if formal_joined is not None:
        formal_joined_index = index_unique(formal_joined, "formal_report.joined")
    else:
        formal_joined_index = None

    joined = build_joined(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        formal_joined_index=formal_joined_index,
    )

    by_task = _group_rows(joined, "task", bootstrap=True)
    by_source_dataset = _group_rows(joined, "source_dataset", bootstrap=True)
    by_source = _group_rows(joined, "source", bootstrap=True)
    by_bypass = _group_rows(joined, "identity_bypass", bootstrap=True)
    by_pre_selector_bin = _group_rows(
        joined,
        "pre_selector_real_group_count_bin",
        order=PRE_SELECTOR_BIN_ORDER,
        bootstrap=True,
    )
    # Collapse the recorded field into the requested bins as an audit cross-check.
    recorded = []
    for label, predicate in PRE_SELECTOR_BINS[:-1]:
        subset = [
            row for row in joined if predicate(row["recorded_real_group_count"])
        ]
        if subset:
            recorded.append(stratum_row(label, subset, bootstrap=False))
    large = [
        row for row in joined if row["recorded_real_group_count"] > 1024
    ]
    if large:
        recorded.append(stratum_row(">1024", large, bootstrap=False))

    overall_integer_official = official_accuracy(joined, ARM_INTEGER)
    overall_prompt_official = official_accuracy(joined, ARM_PROMPT)
    official_delta = (
        overall_prompt_official - overall_integer_official
        if overall_integer_official is not None
        and overall_prompt_official is not None
        else None
    )
    official_ci = paired_bootstrap_delta(
        _pair_score_rows(joined),
        "left_correct",
        "right_correct",
        BOOTSTRAP_SEED,
        repeats=BOOTSTRAP_REPEATS,
    )

    frozen_matches = (
        official_delta is not None
        and abs(
            official_delta - formal_checks["frozen_delta_point_estimate"]
        )
        <= 1e-12
        and abs(official_ci[0] - formal_checks["frozen_delta_ci95"][0]) <= 1e-12
        and abs(official_ci[1] - formal_checks["frozen_delta_ci95"][1]) <= 1e-12
    )
    if not frozen_matches:
        raise InputContractError(
            "recomputed official prompt-minus-integer delta/CI does not reproduce "
            "the frozen formal report (fail closed)"
        )

    overall = stratum_row("overall", joined, bootstrap=True)
    overall.update(
        {
            "integer_official_cv_bench_accuracy": overall_integer_official,
            "prompt_official_cv_bench_accuracy": overall_prompt_official,
            "official_cv_bench_accuracy_delta": official_delta,
            "official_delta_ci95": [float(official_ci[0]), float(official_ci[1])],
            "official_metric_defined": official_delta is not None,
        }
    )

    dense = dense_descriptive(formal_joined_index, formal_report)

    # Re-verify read-only inputs after the analysis (nothing may have changed).
    for name, spec in pinned.items():
        current = sha256_file(Path(spec["path"]))
        if current != spec["sha256"]:
            raise InputContractError(
                f"input '{name}' changed during analysis: {spec['sha256']} -> "
                f"{current} (fail closed)"
            )

    per_sample = [
        {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "source": row["source"],
            "source_dataset": row["source_dataset"],
            "ground_truth": row["ground_truth"],
            "G": row["G"],
            "pre_selector_real_group_count": row["pre_selector_real_group_count"],
            "pre_selector_real_group_count_bin": row[
                "pre_selector_real_group_count_bin"
            ],
            "recorded_real_group_count": row["recorded_real_group_count"],
            "context_group_count": row["context_group_count"],
            "identity_bypass": row["identity_bypass"],
            "integer_answer": row[f"{ARM_INTEGER}_answer"],
            "prompt_answer": row[f"{ARM_PROMPT}_answer"],
            "integer_correct": row[f"{ARM_INTEGER}_correct"],
            "prompt_correct": row[f"{ARM_PROMPT}_correct"],
            "dense_correct": row["dense_correct"],
            "transition": row["transition"],
            "index_exact_match": row["index_exact_match"],
            "index_intersection": row["index_intersection"],
            "index_union": row["index_union"],
            "index_jaccard": row["index_jaccard"],
        }
        for row in joined
    ]

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "analysis": "full2638_prompt_gain_heterogeneity",
        "scope": (
            "post-hoc exploratory diagnostic of the frozen formal result "
            "PROMPT_GAIN_NOT_CONFIRMED_FULL2638; CPU-only, read-only; NOT a new "
            "primary metric and NOT a formal gate"
        ),
        "diagnostic_only": True,
        "formal_result_unchanged": True,
        "formal_outcome": FORMAL_OUTCOME,
        "primary_metric": PRIMARY_METRIC,
        "unweighted_metric": UNWEIGHTED_METRIC,
        "exploratory_note": EXPLORATORY_NOTE,
        "pre_selector_real_group_count_definition": (
            PRE_SELECTOR_REAL_GROUP_COUNT_DEFINITION
        ),
        "bootstrapping": {
            "seed": BOOTSTRAP_SEED,
            "repeats": BOOTSTRAP_REPEATS,
            "ci_level": CI_LEVEL,
            "gate": False,
        },
        "inputs_sha256": pinned,
        "contract": contract,
        "formal_report_checks": formal_checks,
        "formal_report_reproduction": {
            "recomputed_official_delta": official_delta,
            "recomputed_official_delta_ci95": [float(official_ci[0]), float(official_ci[1])],
            "matches_frozen_report": frozen_matches,
        },
        "overall": overall,
        "by_task": by_task,
        "by_source_dataset": by_source_dataset,
        "by_source": by_source,
        "by_identity_bypass": by_bypass,
        "by_pre_selector_real_group_count_bin": by_pre_selector_bin,
        "by_recorded_real_group_count_bin": recorded,
        "selection_index_overlap_non_bypass": selection_overlap(joined),
        "concentration": concentration(
            joined, by_task, by_source_dataset, by_pre_selector_bin
        ),
        "regression_check": regression_check(by_task, by_source_dataset, by_source),
        "dense_descriptive": dense,
        "limitations": [
            "The formal conclusion PROMPT_GAIN_NOT_CONFIRMED_FULL2638 is unchanged; every subgroup number here is post-hoc and exploratory.",
            "The official cv_bench_accuracy requires all three sources and is undefined inside single-source task/source strata; those stratum CIs use unweighted sample accuracy and are not a gate.",
            "No GPU, model, board, network or training is used; this is a pure CPU read-only re-analysis of frozen outputs.",
            "The raw JSONL real_group_count is the post-selection real-slot count; the requested pre-selector bins use full_llm_visual_tokens.",
        ],
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
CSV_COLUMNS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "ground_truth",
    "G",
    "pre_selector_real_group_count",
    "pre_selector_real_group_count_bin",
    "recorded_real_group_count",
    "context_group_count",
    "identity_bypass",
    "integer_answer",
    "prompt_answer",
    "integer_correct",
    "prompt_correct",
    "dense_correct",
    "transition",
    "index_exact_match",
    "index_intersection",
    "index_union",
    "index_jaccard",
)


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in report["per_sample"]:
            writer.writerow({name: row.get(name) for name in CSV_COLUMNS})


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_signed(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.6f}"
    return str(value)


def _fmt_ci(ci: Sequence[float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{float(ci[0]):+.6f}, {float(ci[1]):+.6f}]"


def _stratum_table(
    lines: list[str],
    title: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| 层 | n | W→R | R→W | 不变对 | 不变错 | 净修正 | int(unweighted) | "
        "prompt(unweighted) | 配对差 | 95% CI（unweighted，探索性） | "
        "official Δ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in rows:
        ci = row.get("bootstrap_ci95_unweighted_paired_delta")
        official_delta = row.get("official_cv_bench_accuracy_delta")
        lines.append(
            f"| {row['stratum']} | {row['n']} | {row['wrong_to_right']} | "
            f"{row['right_to_wrong']} | {row['unchanged_right']} | "
            f"{row['unchanged_wrong']} | {row['net_corrections']:+d} | "
            f"{_fmt(row['integer_unweighted_accuracy'])} | "
            f"{_fmt(row['prompt_unweighted_accuracy'])} | "
            f"{_fmt_signed(row['unweighted_paired_delta'])} | {_fmt_ci(ci)} | "
            f"{_fmt_signed(official_delta)} |"
        )
    lines.append("")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# full2638 Prompt 增益异质性诊断（只读、事后探索）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：{report['scope']}")
    lines.append(
        f"- 冻结正式结论（不变）：**{report['formal_outcome']}**；"
        f"唯一主指标 `{report['primary_metric']}`"
    )
    lines.append(f"> {report['exploratory_note']}")
    lines.append("")
    lines.append(
        "- pre-selector real group count 定义：" + report["pre_selector_real_group_count_definition"]
    )
    lines.append(
        f"- bootstrap：seed={report['bootstrapping']['seed']} / repeats="
        f"{report['bootstrapping']['repeats']} / level={report['bootstrapping']['ci_level']}；"
        "**不作为正式门。**"
    )
    lines.append("")

    overall = report["overall"]
    lines.append("## 1. 总体迁移与配对差")
    lines.append("")
    lines.append(
        f"- n={overall['n']}；wrong→right={overall['wrong_to_right']}"
        f"（率 {_fmt(overall['wrong_to_right_rate'])}）、"
        f"right→wrong={overall['right_to_wrong']}"
        f"（率 {_fmt(overall['right_to_wrong_rate'])}）、"
        f"不变对={overall['unchanged_right']}、不变错={overall['unchanged_wrong']}。"
    )
    lines.append(
        f"- 净修正={overall['net_corrections']:+d}；unweighted 配对差="
        f"{_fmt_signed(overall['unweighted_paired_delta'])}；"
        f"official `cv_bench_accuracy` Δ={_fmt_signed(overall['official_cv_bench_accuracy_delta'])}"
        f"，95% CI {_fmt_ci(overall['official_delta_ci95'])}（与冻结报告逐位一致："
        f"{report['formal_report_reproduction']['matches_frozen_report']}）。"
    )
    lines.append("")

    _stratum_table(lines, "2. 按 task", report["by_task"])
    _stratum_table(lines, "3. 按 source_dataset", report["by_source_dataset"])
    _stratum_table(lines, "4. 按 source", report["by_source"])
    _stratum_table(lines, "5. 按 identity bypass", report["by_identity_bypass"])
    _stratum_table(
        lines,
        "6. 按 pre-selector real group count 桶（= full_llm_visual_tokens）",
        report["by_pre_selector_real_group_count_bin"],
    )
    _stratum_table(
        lines,
        "7. 按原始字段 real_group_count 桶（审计用；该字段为选择后计数）",
        report["by_recorded_real_group_count_bin"],
    )

    overlap = report["selection_index_overlap_non_bypass"]
    lines.append("## 8. 非 bypass 样本的选择索引重叠")
    lines.append("")
    lines.append(
        f"- 范围：{overlap['scope']}；n={overlap['n']}；精确匹配="
        f"{overlap['exact_match_count']}（率 {_fmt(overlap['exact_match_rate'])}）；"
        f"Jaccard mean={_fmt(overlap['jaccard_mean'])}、min={_fmt(overlap['jaccard_min'])}。"
    )
    lines.append("")
    lines.append("| task | n | 精确匹配 | 精确匹配率 | Jaccard mean | Jaccard min |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in overlap.get("by_task", []):
        lines.append(
            f"| {row['task']} | {row['n']} | {row['exact_match_count']} | "
            f"{_fmt(row['exact_match_rate'])} | {_fmt(row['jaccard_mean'])} | "
            f"{_fmt(row['jaccard_min'])} |"
        )
    lines.append("")

    conc = report["concentration"]
    lines.append("## 9. 增益集中度与回归")
    lines.append("")
    lines.append(
        f"- 毛增益 W→R={conc['gross_wrong_to_right']}、毛损失 R→W="
        f"{conc['gross_right_to_wrong']}、净修正={conc['net_corrections']:+d}、"
        f"总翻转={conc['gross_churn']}；净值为大额对向流的残差："
        f"{conc['net_is_small_residual_of_opposing_flows']}。"
    )
    lines.append(
        f"- task 净值：{conc['by_task_net']}；source_dataset 净值："
        f"{conc['by_source_dataset_net']}；pre-selector 桶净值："
        f"{conc['by_pre_selector_bin_net']}。"
    )
    lines.append(
        f"- 最大正向 task={conc['largest_positive_task']}；最大负向 task="
        f"{conc['largest_negative_task']}；最大正向 source_dataset="
        f"{conc['largest_positive_source_dataset']}；最大负向 source_dataset="
        f"{conc['largest_negative_source_dataset']}。"
    )
    reg = report["regression_check"]
    lines.append(
        f"- 负点估计层数={reg['count_negative_point_delta']}；"
        f"bootstrap 95% CI 整体 <0 的层数={reg['count_ci_entirely_below_zero']}。"
    )
    if reg["strata_with_bootstrap_ci_entirely_below_zero"]:
        for item in reg["strata_with_bootstrap_ci_entirely_below_zero"]:
            lines.append(
                f"  - `{item['grouping_scope']}::{item['stratum']}`：n={item['n']}，"
                f"unweighted 配对差={_fmt_signed(item['unweighted_paired_delta'])}，"
                f"CI {_fmt_ci(item['bootstrap_ci95_unweighted_paired_delta'])}。"
            )
    lines.append("")

    dense = report.get("dense_descriptive")
    if dense:
        lines.append("## 10. dense 描述性上下文（非门）")
        lines.append("")
        lines.append(
            f"- dense official `cv_bench_accuracy`={_fmt(dense['cv_bench_accuracy'])}"
            f"（{dense['role']}）。"
        )
        lines.append(f"- dense unweighted by task：{ {k: round(v, 6) for k, v in dense['unweighted_task_accuracy'].items()} }")
        lines.append(f"- dense unweighted by source：{ {k: round(v, 6) for k, v in dense['unweighted_source_accuracy'].items()} }")
        lines.append("")

    lines.append("## 11. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")

    lines.append("## 12. 限制")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "> CPU-only 只读诊断：未改任何权威输入、既有配置/报告或 "
        "`configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。"
    )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any], outputs: OutputPaths = DEFAULT_OUTPUTS
) -> None:
    outputs.json.parent.mkdir(parents=True, exist_ok=True)
    outputs.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(report, outputs.csv)
    outputs.md.write_text(render_markdown(report), encoding="utf-8")


def run(
    paths: InputPaths = DEFAULT_PATHS,
    outputs: OutputPaths = DEFAULT_OUTPUTS,
    *,
    expected_pins: Mapping[str, str] | None = DEFAULT_INPUT_PINS,
    expected_sample_count: int = FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
) -> tuple[int, dict[str, Any] | None]:
    try:
        report = analyze(
            paths,
            expected_pins=expected_pins,
            expected_sample_count=expected_sample_count,
            manifest_sha256=manifest_sha256,
        )
    except (InputContractError, ValueError, KeyError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2, None
    write_outputs(report, outputs)
    return 0, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-report", type=Path, default=FORMAL_REPORT_JSON)
    parser.add_argument("--manifest", type=Path, default=FULL_MANIFEST)
    parser.add_argument("--integer-arm", type=Path, default=ARM_INTEGER_JSONL)
    parser.add_argument("--prompt-arm", type=Path, default=ARM_PROMPT_JSONL)
    parser.add_argument("--json-out", type=Path, default=REPORT_JSON)
    parser.add_argument("--csv-out", type=Path, default=REPORT_CSV)
    parser.add_argument("--md-out", type=Path, default=REPORT_MD)
    parser.add_argument(
        "--skip-pins",
        action="store_true",
        help="test-only escape hatch: do not verify the pinned input SHA-256 values",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = InputPaths(
        formal_report=args.formal_report,
        full_manifest=args.manifest,
        arm_integer_jsonl=args.integer_arm,
        arm_prompt_jsonl=args.prompt_arm,
    )
    outputs = OutputPaths(json=args.json_out, csv=args.csv_out, md=args.md_out)
    code, report = run(
        paths,
        outputs,
        expected_pins=None if args.skip_pins else DEFAULT_INPUT_PINS,
    )
    if code != 0 or report is None:
        return code
    overall = report["overall"]
    print(
        "overall: n={n} W2R={w} R2W={r} net={net:+d} official_delta={d} "
        "ci={ci}".format(
            n=overall["n"],
            w=overall["wrong_to_right"],
            r=overall["right_to_wrong"],
            net=overall["net_corrections"],
            d=overall["official_cv_bench_accuracy_delta"],
            ci=overall["official_delta_ci95"],
        )
    )
    print(f"formal_outcome: {report['formal_outcome']} (unchanged)")
    print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
