#!/usr/bin/env python3
"""CPU-only forced-index vs control (same budget) screening100 summariser.

Summariser for the preregistered screening100 causal diagnostic
``configs/2026-10-04-layer5-forced-index-ac-screening100.preregistered.json``.

Experiment
----------
Two arms are produced by the **same** unified layer-5 forward in
``evaluation/hardware_regular_layer5_forced_index_ablation.py`` via the runner
``evaluation/run_hf_layer5_forced_index_ablation.py``:

* ``control`` - keeps ``K_C`` groups with the control rule (one anchor per
  non-empty 3x3 region + global L2 ranking);
* ``forced``  - keeps exactly the frozen historical ``shallow_r0250`` indices
  for that sample.

For every sample ``K_C = len(reference selected_group_indices)`` and both arms
use that same ``K_C``; the single experimental variable is
``selected_group_indices``.  The run metadata is BF16 / SDPA / greedy, the cut
point is layer 5 and the region grid is 3.  The relaxed ``K_C`` alignment is a
diagnostic setting and is explicitly **not** an FPGA-deployable candidate.

Decision
--------
The **only** decision is the forced-minus-control point estimate and its 95%
paired stratified bootstrap CI on the official primary metric
``score(...)["cv_bench_accuracy"]`` (stratified by ``task::source_dataset``,
seed 20261004, 5000 repeats).  CI lower bound ``> 0`` yields
``HISTORICAL_INDEX_SET_BETTER_CONFIRMED``; otherwise ``NOT_CONFIRMED``.

Fail-closed contract (any violation raises :class:`InputContractError` and no
report is written):

* every non-null ``inputs_sha256`` pin in the preregistration is re-verified;
* exactly the same 100 unique ``sample_id`` values in manifest / reference /
  control / forced, with no duplicates;
* each arm record carries the right ``arm`` / diagnostic flags / input hashes /
  ``K_C`` / ``G`` / reference indices and their SHA-256;
* the forced selection equals the historical reference indices exactly, and the
  per-sample ``K_C`` is identical across control and forced;
* the per-record and run metadata decoding is BF16 / SDPA / greedy;
* both arms are all-real (``source_type`` all 0), context-free, exact-``K_C``.

Multi-choice answers are parsed by re-using the official helpers
``answer_letter`` / ``parse_choice`` from
``evaluation.build_cv_bench_pruning_report`` (the module that also owns the
official ``score``) and scored with ``score`` only.  No second scorer is
implemented.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
)
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-04-layer5-forced-index-ac-screening100.preregistered.json"
)
SCREENING_MANIFEST = ROOT / "datasets" / "cv_bench_screening100.jsonl"
REFERENCE_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_budget64_pilot100_20260927"
    / "shallow_r0250.jsonl"
)
CONTROL_ARM = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-forced-index-ac"
    / "control_same_budget_screening100.jsonl"
)
FORCED_ARM = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-forced-index-ac"
    / "forced_r25_indices_screening100.jsonl"
)
REPORT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-layer5-forced-index-ac-screening100.json"
)
REPORT_CSV = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-layer5-forced-index-ac-screening100.csv"
)
REPORT_MD = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-layer5-forced-index-ac-screening100.md"
)

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
SECONDARY_METRIC = 'score(...)["accuracy"]'
UNWEIGHTED_NOTE = (
    "unweighted sample accuracy is SECONDARY diagnostic only and is never the "
    "official CV-Bench metric"
)

SCREENING_SAMPLE_COUNT = 100
PRUNE_AFTER_LAYER = 5
REGION_GRID = 3
SCORE_MODE = "l2"
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
STRATIFIED_BY = "task::source_dataset"

ARM_CONTROL = "control"
ARM_FORCED = "forced"
ARMS = (ARM_CONTROL, ARM_FORCED)
MODE_CONTROL = "control"
MODE_FORCED = "forced"
SELECTION_RULE_CONTROL = "control_global_l2_plus_3x3_anchor"
SELECTION_RULE_FORCED = "forced_exact_historical_group_indices"
ONLY_VARIABLE = "selected_group_indices"
REFERENCE_SELECTOR = "shallow_r0250"
ARM_LABEL = "layer5_forced_index_ablation_block5"

HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC = True
CAUSAL_DIAGNOSTIC_ONLY = True
FPGA_DEPLOYABLE_CANDIDATE = False

# The runner records the CLI spelling while mapping it to torch.bfloat16.
REQUIRED_DTYPE = "bf16"
REQUIRED_ATTENTION = "sdpa"
REQUIRED_DECODING = "greedy_argmax"
SOURCE_REAL = 0

DECISION_CONFIRMED = "HISTORICAL_INDEX_SET_BETTER_CONFIRMED"
DECISION_NOT_CONFIRMED = "NOT_CONFIRMED"
DECISION_THRESHOLD = "delta_ci95_lower > 0"
DECISION_SCOPE_NOTE = (
    "screening100 causal diagnostic only; this is NOT a full-dataset "
    "(2638-sample) conclusion"
)

DIAGNOSTIC_SCOPE = (
    "fixed 100-sample CV-Bench screening causal diagnostic (forced vs control, "
    "same per-sample K_C); NOT a formal full2638 conclusion"
)

FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

ARMS_DESCRIPTION = {
    ARM_CONTROL: (
        "layer-5 control arm: one anchor per non-empty 3x3 region + global L2 "
        "ranking, exactly K_C real groups"
    ),
    ARM_FORCED: (
        "layer-5 forced arm: exactly the frozen shallow_r0250 historical indices, "
        "same K_C real groups"
    ),
}

G_BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("G<=96", lambda value: value <= 96),
    ("97-256", lambda value: 97 <= value <= 256),
    ("257-768", lambda value: 257 <= value <= 768),
    ("G>768", lambda value: value > 768),
)
K_BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("K<=48", lambda value: value <= 48),
    ("49-96", lambda value: 49 <= value <= 96),
    ("97-192", lambda value: 97 <= value <= 192),
    ("K>192", lambda value: value > 192),
)

#: Logical input names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = (
    "screening_manifest",
    "reference_jsonl",
    "control_arm",
    "forced_arm",
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the frozen contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    reference_jsonl: Path = REFERENCE_JSONL
    control_arm: Path = CONTROL_ARM
    forced_arm: Path = FORCED_ARM


DEFAULT_PATHS = InputPaths()


@dataclass(frozen=True)
class OutputPaths:
    json: Path = REPORT_JSON
    csv: Path = REPORT_CSV
    md: Path = REPORT_MD


DEFAULT_OUTPUTS = OutputPaths()


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_sha256(indices: Sequence[int]) -> str:
    """Match the frozen runner's ``indices_sha256`` exactly."""
    payload = ",".join(str(int(index)) for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


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
            f"input '{label}' has rows without sample_id at positions {missing[:10]}"
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
    return int(value)


def _is_true(value: Any) -> bool:
    return value is True


def _is_false(value: Any) -> bool:
    return value is False


def validate_selected_indices(row: Mapping[str, Any], label: str) -> list[int]:
    """Non-empty unique integer indices with a reproducible normalized SHA-256.

    The normalized (ascending) digest must match ``selected_group_indices_sha256``
    exactly as the frozen runner records it.
    """
    raw = row.get("selected_group_indices")
    if not isinstance(raw, list) or not raw:
        raise InputContractError(
            f"{label} must carry a non-empty selected_group_indices list"
        )
    values = [_as_int(value, f"{label}.selected_group_indices") for value in raw]
    if any(value < 0 for value in values):
        raise InputContractError(f"{label} selected_group_indices must be non-negative")
    if len(set(values)) != len(values):
        raise InputContractError(f"{label} selected_group_indices contains duplicates")
    normalized = sorted(values)
    recorded = row.get("selected_group_indices_sha256")
    recomputed = selection_sha256(normalized)
    if not isinstance(recorded, str) or recorded != recomputed:
        raise InputContractError(
            f"{label} selected_group_indices_sha256 mismatch: recorded={recorded!r} "
            f"recomputed={recomputed}"
        )
    return normalized


def grid_signature(row: Mapping[str, Any]) -> tuple[tuple[int, ...], int, int]:
    for field in ("image_grid_thw", "full_patch_tokens", "full_llm_visual_tokens"):
        if field not in row:
            raise InputContractError(f"record is missing {field}")
    return (
        tuple(int(value) for value in row["image_grid_thw"]),
        int(row["full_patch_tokens"]),
        int(row["full_llm_visual_tokens"]),
    )


# ---------------------------------------------------------------------------
# Preregistration pin verification
# ---------------------------------------------------------------------------


def _logical_pin_path(name: str, paths: InputPaths) -> Path | None:
    return {
        "screening_manifest": paths.screening_manifest,
        "reference_jsonl": paths.reference_jsonl,
        "control_arm": paths.control_arm,
        "forced_arm": paths.forced_arm,
    }.get(name)


def verify_config_pins(
    config: Mapping[str, Any],
    paths: InputPaths = DEFAULT_PATHS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Re-verify every non-null ``inputs_sha256`` pin; fail closed on mismatch.

    ``null`` pins (the two arms before they are generated) are only required to
    exist when they are logical inputs; their digest is not checked, exactly as
    preregistered (``sha256 = null`` / ``TO_BE_GENERATED``).
    """
    checks: list[dict[str, Any]] = []
    skipped: list[str] = []
    pins = config.get("inputs_sha256", {})
    if not isinstance(pins, Mapping):
        raise InputContractError("config inputs_sha256 must be an object")
    for name, spec in pins.items():
        if isinstance(spec, Mapping):
            expected = spec.get("sha256")
            relative = spec.get("path")
        else:
            expected = spec
            relative = None
        logical = _logical_pin_path(str(name), paths)
        if logical is not None:
            path = logical
            label = str(name)
        elif relative is not None:
            path = Path(relative)
            if not path.is_absolute():
                path = root / path
            label = str(relative)
        else:
            raise InputContractError(
                f"pin '{name}' has neither a logical input nor a relative path"
            )
        if expected is None:
            # Preregistered null pin (arm not generated yet): only existence.
            require_file(path, label)
            skipped.append(str(name))
            continue
        expected = str(expected).lower()
        require_file(path, label)
        actual = sha256_file(path)
        if actual != expected:
            raise InputContractError(
                f"input '{label}' sha256 mismatch: expected {expected}, got {actual} "
                "(fail closed)"
            )
        checks.append(
            {
                "name": str(name),
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": True,
            }
        )
    return {"pins_verified": True, "checks": checks, "null_pins_skipped": skipped}


# ---------------------------------------------------------------------------
# Frozen historical reference contract
# ---------------------------------------------------------------------------


def validate_reference_record(
    row: Mapping[str, Any], manifest_row: Mapping[str, Any], sample_id: str
) -> dict[str, Any]:
    label = f"reference/{sample_id}"
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise InputContractError(
            f"{label} task {row.get('task')!r} != manifest "
            f"{manifest_row.get('task')!r}"
        )
    if _as_int(row.get("prune_after_layer"), f"{label}.prune_after_layer") != (
        PRUNE_AFTER_LAYER
    ):
        raise InputContractError(
            f"{label} prune_after_layer must be {PRUNE_AFTER_LAYER}"
        )
    group_count = _as_int(
        row.get("baseline_llm_visual_tokens"),
        f"{label}.baseline_llm_visual_tokens",
    )
    if group_count <= 0:
        raise InputContractError(
            f"{label}.baseline_llm_visual_tokens must be positive"
        )
    indices = validate_selected_indices(row, label)
    k_c = len(indices)
    recorded_k = _as_int(row.get("llm_visual_tokens"), f"{label}.llm_visual_tokens")
    if recorded_k != k_c:
        raise InputContractError(
            f"{label}.llm_visual_tokens={recorded_k} != len(indices)={k_c}"
        )
    if k_c >= group_count and k_c != group_count:
        raise InputContractError(
            f"{label} K_C={k_c} must not exceed G={group_count}"
        )
    return {
        "sample_id": sample_id,
        "task": str(manifest_row.get("task")),
        "G": group_count,
        "K_C": k_c,
        "group_indices": indices,
        "group_indices_sha256": selection_sha256(indices),
    }


# ---------------------------------------------------------------------------
# Arm record contract
# ---------------------------------------------------------------------------


def _validate_run_metadata(
    run: Any,
    *,
    arm: str,
    manifest_sha256: str,
    reference_sha256: str,
    label: str,
) -> None:
    if not isinstance(run, Mapping):
        raise InputContractError(f"{label}.run must be an object")
    if str(run.get("arm")) != arm:
        raise InputContractError(
            f"{label}.run.arm {run.get('arm')!r} != {arm!r}"
        )
    if str(run.get("mode")) != arm:
        raise InputContractError(f"{label}.run.mode {run.get('mode')!r} != {arm!r}")
    rule = SELECTION_RULE_CONTROL if arm == ARM_CONTROL else SELECTION_RULE_FORCED
    if str(run.get("selection_rule")) != rule:
        raise InputContractError(f"{label}.run.selection_rule must be {rule!r}")
    if _as_int(run.get("prune_after_layer"), f"{label}.run.prune_after_layer") != (
        PRUNE_AFTER_LAYER
    ):
        raise InputContractError(
            f"{label}.run.prune_after_layer must be {PRUNE_AFTER_LAYER}"
        )
    if _as_int(run.get("region_grid"), f"{label}.run.region_grid") != REGION_GRID:
        raise InputContractError(f"{label}.run.region_grid must be {REGION_GRID}")
    if str(run.get("manifest_sha256")) != manifest_sha256:
        raise InputContractError(f"{label}.run.manifest_sha256 mismatch (fail closed)")
    if str(run.get("reference_jsonl_sha256")) != reference_sha256:
        raise InputContractError(
            f"{label}.run.reference_jsonl_sha256 mismatch (fail closed)"
        )
    if str(run.get("reference_selector")) != REFERENCE_SELECTOR:
        raise InputContractError(
            f"{label}.run.reference_selector must be {REFERENCE_SELECTOR!r}"
        )
    if str(run.get("dtype")) != REQUIRED_DTYPE:
        raise InputContractError(
            f"{label}.run.dtype must be {REQUIRED_DTYPE!r}, got {run.get('dtype')!r}"
        )
    if str(run.get("attn_implementation")) != REQUIRED_ATTENTION:
        raise InputContractError(
            f"{label}.run.attn_implementation must be {REQUIRED_ATTENTION!r}"
        )
    if not _is_false(run.get("do_sample")):
        raise InputContractError(f"{label}.run.do_sample must be false (greedy)")
    if not _is_true(run.get("custom_greedy_decode")):
        raise InputContractError(f"{label}.run.custom_greedy_decode must be true")
    if not _is_false(run.get("has_fixed_budget_knob")):
        raise InputContractError(f"{label}.run.has_fixed_budget_knob must be false")
    if run.get("local_candidate_cap") is not None:
        raise InputContractError(
            f"{label}.run.local_candidate_cap must be null, got "
            f"{run.get('local_candidate_cap')!r}"
        )
    for flag in (
        "uses_context_pooling",
        "uses_context_slot_reservation",
        "uses_local_candidate_cap",
        "uses_b1_trained_wv",
    ):
        if not _is_false(run.get(flag)):
            raise InputContractError(f"{label}.run.{flag} must be false")
    if run.get("projection_npz") is not None:
        raise InputContractError(f"{label}.run.projection_npz must be null")
    if not _is_false(run.get("training")):
        raise InputContractError(f"{label}.run.training must be false")
    if not _is_true(run.get("K_C_is_per_sample_historical_length")):
        raise InputContractError(
            f"{label}.run.K_C_is_per_sample_historical_length must be true"
        )


def validate_arm_record(
    row: Mapping[str, Any],
    *,
    arm: str,
    manifest_row: Mapping[str, Any],
    reference: Mapping[str, Any],
    manifest_sha256: str,
    reference_sha256: str,
) -> dict[str, Any]:
    """Validate one control/forced record against the frozen contract."""
    if arm not in ARMS:  # pragma: no cover - guarded by caller
        raise InputContractError(f"unknown arm {arm!r}")
    sample_id = str(row.get("sample_id"))
    label = f"{arm}/{sample_id}"
    if str(row.get("task")) != str(manifest_row.get("task")):
        raise InputContractError(
            f"{label} task {row.get('task')!r} != manifest "
            f"{manifest_row.get('task')!r}"
        )
    if str(row.get("arm")) != arm:
        raise InputContractError(f"{label} arm field {row.get('arm')!r} != {arm!r}")
    rule = SELECTION_RULE_CONTROL if arm == ARM_CONTROL else SELECTION_RULE_FORCED
    if str(row.get("selection_rule")) != rule:
        raise InputContractError(f"{label} selection_rule must be {rule!r}")
    if _as_int(row.get("prune_after_layer"), f"{label}.prune_after_layer") != (
        PRUNE_AFTER_LAYER
    ):
        raise InputContractError(
            f"{label} prune_after_layer must be {PRUNE_AFTER_LAYER}"
        )
    if _as_int(row.get("region_grid"), f"{label}.region_grid") != REGION_GRID:
        raise InputContractError(f"{label} region_grid must be {REGION_GRID}")
    # Diagnostic flags: relaxed alignment, diagnostic-only, NOT an FPGA candidate.
    if not _is_true(row.get("hardware_alignment_relaxed_for_diagnostic")):
        raise InputContractError(
            f"{label} hardware_alignment_relaxed_for_diagnostic must be true"
        )
    if not _is_true(row.get("causal_diagnostic_only")):
        raise InputContractError(f"{label} causal_diagnostic_only must be true")
    if not _is_false(row.get("fpga_deployable_candidate")):
        raise InputContractError(
            f"{label} fpga_deployable_candidate must be false (diagnostic only)"
        )
    if str(row.get("only_variable")) != ONLY_VARIABLE:
        raise InputContractError(f"{label} only_variable must be {ONLY_VARIABLE!r}")
    # Input hashes carried by the record must match the actual files.
    if str(row.get("manifest_sha256")) != manifest_sha256:
        raise InputContractError(f"{label} manifest_sha256 mismatch (fail closed)")
    if str(row.get("reference_jsonl_sha256")) != reference_sha256:
        raise InputContractError(
            f"{label} reference_jsonl_sha256 mismatch (fail closed)"
        )
    if str(row.get("reference_selector")) != REFERENCE_SELECTOR:
        raise InputContractError(
            f"{label} reference_selector must be {REFERENCE_SELECTOR!r}"
        )
    # All-real / no-context declaration.
    if not _is_false(row.get("uses_context_pooling")):
        raise InputContractError(f"{label} uses_context_pooling must be false")
    if not _is_false(row.get("uses_context_slot_reservation")):
        raise InputContractError(
            f"{label} uses_context_slot_reservation must be false"
        )
    if not _is_false(row.get("uses_local_candidate_cap")):
        raise InputContractError(f"{label} uses_local_candidate_cap must be false")
    if row.get("local_candidate_cap") is not None:
        raise InputContractError(
            f"{label} local_candidate_cap must be null, got "
            f"{row.get('local_candidate_cap')!r}"
        )
    if not _is_false(row.get("uses_pooled_context_group_values")):
        raise InputContractError(
            f"{label} uses_pooled_context_group_values must be false"
        )
    if not _is_false(row.get("uses_b1_trained_wv")):
        raise InputContractError(f"{label} uses_b1_trained_wv must be false")
    if not _is_true(row.get("source_types_all_real")):
        raise InputContractError(f"{label} source_types_all_real must be true")
    if _as_int(row.get("context_group_count"), f"{label}.context_group_count") != 0:
        raise InputContractError(f"{label} context_group_count must be 0")
    if not _is_false(row.get("do_sample")):
        raise InputContractError(f"{label} do_sample must be false")
    if str(row.get("decoding")) != REQUIRED_DECODING:
        raise InputContractError(
            f"{label} decoding must be {REQUIRED_DECODING!r}"
        )
    # G / K_C / reference indices.
    group_count = _as_int(
        row.get("full_llm_visual_tokens"), f"{label}.full_llm_visual_tokens"
    )
    if group_count <= 0:
        raise InputContractError(f"{label} full_llm_visual_tokens must be positive")
    if _as_int(
        row.get("baseline_llm_visual_tokens"), f"{label}.baseline_llm_visual_tokens"
    ) != group_count:
        raise InputContractError(
            f"{label} baseline_llm_visual_tokens != full_llm_visual_tokens"
        )
    if group_count != int(reference["G"]):
        raise InputContractError(
            f"{label} G={group_count} != reference G={reference['G']}"
        )
    if _as_int(row.get("reference_G"), f"{label}.reference_G") != int(reference["G"]):
        raise InputContractError(f"{label} reference_G mismatch")
    if _as_int(
        row.get("reference_baseline_llm_visual_tokens"),
        f"{label}.reference_baseline_llm_visual_tokens",
    ) != int(reference["G"]):
        raise InputContractError(
            f"{label} reference_baseline_llm_visual_tokens mismatch"
        )
    k_c = _as_int(row.get("K_C"), f"{label}.K_C")
    if k_c <= 0:
        raise InputContractError(f"{label} K_C must be positive")
    if k_c != int(reference["K_C"]):
        raise InputContractError(
            f"{label} K_C={k_c} != reference K_C={reference['K_C']}"
        )
    if _as_int(row.get("reference_K_C"), f"{label}.reference_K_C") != int(
        reference["K_C"]
    ):
        raise InputContractError(f"{label} reference_K_C mismatch")
    if _as_int(
        row.get("reference_llm_visual_tokens"), f"{label}.reference_llm_visual_tokens"
    ) != int(reference["K_C"]):
        raise InputContractError(f"{label} reference_llm_visual_tokens mismatch")
    if _as_int(row.get("requested_K_C"), f"{label}.requested_K_C") != k_c:
        raise InputContractError(f"{label} requested_K_C != K_C")
    if _as_int(row.get("llm_visual_tokens"), f"{label}.llm_visual_tokens") != k_c:
        raise InputContractError(f"{label} llm_visual_tokens != K_C")
    # Reference indices embedded in the record must equal the reference file.
    embedded = validate_selected_indices(
        {
            "selected_group_indices": row.get("reference_selected_group_indices"),
            "selected_group_indices_sha256": row.get(
                "reference_selected_group_indices_sha256"
            ),
        },
        f"{label}.reference_selected_group_indices",
    )
    if embedded != list(reference["group_indices"]):
        raise InputContractError(
            f"{label} embedded reference indices do not equal the reference file"
        )
    # Selected indices of this arm.
    selected = validate_selected_indices(row, label)
    identity = row.get("identity_bypass")
    if not isinstance(identity, bool):
        raise InputContractError(f"{label} identity_bypass must be a bool")
    if identity:
        if selected != list(range(group_count)):
            raise InputContractError(
                f"{label} identity bypass must emit all {group_count} groups"
            )
    else:
        if selected != sorted(selected) or len(selected) != k_c:
            raise InputContractError(f"{label} must emit exactly K_C={k_c} groups")
    if _as_int(row.get("output_group_count"), f"{label}.output_group_count") != len(
        selected
    ):
        raise InputContractError(f"{label} output_group_count != len(selected)")
    if _as_int(row.get("real_group_count"), f"{label}.real_group_count") != len(
        selected
    ):
        raise InputContractError(f"{label} real_group_count != len(selected)")
    if len(selected) < k_c:
        raise InputContractError(f"{label} emitted fewer than K_C groups")
    source_type = row.get("source_type")
    if not isinstance(source_type, list) or len(source_type) != len(selected):
        raise InputContractError(f"{label} source_type length mismatch")
    if any(int(value) != SOURCE_REAL for value in source_type):
        raise InputContractError(f"{label} source_type must be all real (0)")
    for field in ("region_id", "valid"):
        if not isinstance(row.get(field), list) or len(row[field]) != len(selected):
            raise InputContractError(f"{label} {field} length mismatch")
    if not all(bool(value) for value in row["valid"]):
        raise InputContractError(f"{label} contains an invalid group")
    if selected != sorted(selected) or len(set(selected)) != len(selected):
        raise InputContractError(f"{label} selected indices must be strictly ascending")
    _validate_run_metadata(
        row.get("run"),
        arm=arm,
        manifest_sha256=manifest_sha256,
        reference_sha256=reference_sha256,
        label=label,
    )
    return {
        "sample_id": sample_id,
        "task": str(row.get("task")),
        "G": group_count,
        "K_C": k_c,
        "selected_group_indices": selected,
        "selected_group_indices_sha256": selection_sha256(selected),
        "reference_group_indices": embedded,
        "reference_group_indices_sha256": selection_sha256(embedded),
        "identity_bypass": identity,
        "grid_signature": grid_signature(row),
    }


# ---------------------------------------------------------------------------
# Validation of the four inputs together
# ---------------------------------------------------------------------------


def validate_inputs(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    reference_index: Mapping[str, Mapping[str, Any]],
    control_index: Mapping[str, Mapping[str, Any]],
    forced_index: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    reference_sha256: str,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
) -> dict[str, Any]:
    if len(screening_index) != expected_screening_count:
        raise InputContractError(
            f"screening manifest must hold {expected_screening_count} unique samples, "
            f"got {len(screening_index)}"
        )
    for label, index in (
        ("reference", reference_index),
        ("control", control_index),
        ("forced", forced_index),
    ):
        if len(index) != expected_screening_count:
            raise InputContractError(
                f"{label} must hold {expected_screening_count} unique samples, "
                f"got {len(index)}"
            )
        missing = sorted(set(screening_index) - set(index))
        extra = sorted(set(index) - set(screening_index))
        if missing or extra:
            raise InputContractError(
                f"{label} sample_id set does not equal the screening manifest: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

    reference_records: dict[str, dict[str, Any]] = {}
    for sample_id, row in reference_index.items():
        reference_records[sample_id] = validate_reference_record(
            row, screening_index[sample_id], sample_id
        )

    control_records: dict[str, dict[str, Any]] = {}
    forced_records: dict[str, dict[str, Any]] = {}
    for sample_id in screening_index:
        reference = reference_records[sample_id]
        manifest_row = screening_index[sample_id]
        control_records[sample_id] = validate_arm_record(
            control_index[sample_id],
            arm=ARM_CONTROL,
            manifest_row=manifest_row,
            reference=reference,
            manifest_sha256=manifest_sha256,
            reference_sha256=reference_sha256,
        )
        forced_records[sample_id] = validate_arm_record(
            forced_index[sample_id],
            arm=ARM_FORCED,
            manifest_row=manifest_row,
            reference=reference,
            manifest_sha256=manifest_sha256,
            reference_sha256=reference_sha256,
        )

    for sample_id in screening_index:
        control = control_records[sample_id]
        forced = forced_records[sample_id]
        reference = reference_records[sample_id]
        # The frozen contract: control and forced share the same per-sample K_C.
        if control["K_C"] != forced["K_C"] or control["K_C"] != reference["K_C"]:
            raise InputContractError(
                f"{sample_id} K_C differs across control/forced/reference: "
                f"{control['K_C']}/{forced['K_C']}/{reference['K_C']}"
            )
        if control["G"] != forced["G"] or control["G"] != reference["G"]:
            raise InputContractError(
                f"{sample_id} G differs across control/forced/reference"
            )
        # The single experimental variable: the forced arm must keep exactly the
        # historical reference indices.
        if forced["selected_group_indices"] != reference["group_indices"]:
            raise InputContractError(
                f"{sample_id} forced selection does not equal the historical "
                f"reference indices (fail closed)"
            )
        if forced["reference_group_indices"] != reference["group_indices"]:
            raise InputContractError(
                f"{sample_id} forced embedded reference indices mismatch"
            )
        if control["reference_group_indices"] != reference["group_indices"]:
            raise InputContractError(
                f"{sample_id} control embedded reference indices mismatch"
            )
        if control["grid_signature"] != forced["grid_signature"]:
            raise InputContractError(
                f"{sample_id} control/forced input grid signatures differ"
            )

    return {
        "reference": reference_records,
        "control": control_records,
        "forced": forced_records,
    }


# ---------------------------------------------------------------------------
# Metrics / decision
# ---------------------------------------------------------------------------


def _record_for_scorer(
    sample_id: str,
    meta: Mapping[str, Any],
    control_row: Mapping[str, Any],
    forced_row: Mapping[str, Any],
) -> dict[str, Any]:
    choices = list(meta["choices"])
    truth = answer_letter(str(meta["answer"]))
    control_parsed = parse_choice(str(control_row.get("output_text", "")), choices)
    forced_parsed = parse_choice(str(forced_row.get("output_text", "")), choices)
    return {
        "sample_id": sample_id,
        "task": meta["task"],
        "type": meta["type"],
        "source": meta["source"],
        "source_dataset": meta["source_dataset"],
        "ground_truth": truth,
        "control_answer": f"({control_parsed})" if control_parsed else None,
        "control_correct": control_parsed == truth,
        "forced_answer": f"({forced_parsed})" if forced_parsed else None,
        "forced_correct": forced_parsed == truth,
        "control_parsed": control_parsed,
        "forced_parsed": forced_parsed,
        "control_output_text": str(control_row.get("output_text", "")),
        "forced_output_text": str(forced_row.get("output_text", "")),
    }


def build_joined(
    screening_index: Mapping[str, Mapping[str, Any]],
    control_index: Mapping[str, Mapping[str, Any]],
    forced_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        _record_for_scorer(
            sample_id, meta, control_index[sample_id], forced_index[sample_id]
        )
        for sample_id, meta in screening_index.items()
    ]


def arm_metrics(
    joined: Sequence[Mapping[str, Any]], arm: str, label: str
) -> dict[str, Any]:
    result = score(list(joined), f"{arm}_correct")
    return {
        "label": label,
        "sample_count": int(result["sample_count"]),
        "primary_metric": PRIMARY_METRIC,
        "cv_bench_accuracy": float(result["cv_bench_accuracy"]),
        "accuracy_2d": float(result["accuracy_2d"]),
        "accuracy_3d": float(result["accuracy_3d"]),
        "unweighted_sample_accuracy": float(result["accuracy"]),
        "unweighted_metric": SECONDARY_METRIC,
        "parse_success_rate": float(result["parse_success_rate"]),
        "unweighted_task_accuracy": dict(result["task_accuracy"]),
        "unweighted_source_accuracy": dict(result["source_accuracy"]),
    }


def paired_comparison(
    joined: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    """Paired stratified bootstrap on the official CV-Bench primary only."""
    point = float(
        score(list(joined), f"{left}_correct")["cv_bench_accuracy"]
        - score(list(joined), f"{right}_correct")["cv_bench_accuracy"]
    )
    ci = paired_bootstrap_delta(
        list(joined),
        f"{left}_correct",
        f"{right}_correct",
        BOOTSTRAP_SEED,
        repeats=BOOTSTRAP_REPEATS,
    )
    return {
        "left": left,
        "right": right,
        "comparison": f"{left}_minus_{right}",
        "metric": PRIMARY_METRIC,
        "delta_point_estimate": point,
        "delta_ci95": [float(ci[0]), float(ci[1])],
        "stratified_by": STRATIFIED_BY,
        "seed": BOOTSTRAP_SEED,
        "repeats": BOOTSTRAP_REPEATS,
        "ci_level": 0.95,
    }


def decide(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """The only decision: forced-minus-control CI lower bound > 0."""
    lower = float(comparison["delta_ci95"][0])
    upper = float(comparison["delta_ci95"][1])
    confirmed = lower > 0.0
    return {
        "type": "forced_minus_control_ci",
        "metric": PRIMARY_METRIC,
        "comparison": "forced_minus_control",
        "point_estimate": float(comparison["delta_point_estimate"]),
        "delta_ci95": [lower, upper],
        "threshold": DECISION_THRESHOLD,
        "seed": BOOTSTRAP_SEED,
        "repeats": BOOTSTRAP_REPEATS,
        "stratified_by": STRATIFIED_BY,
        "historical_index_set_better": bool(confirmed),
        "outcome": DECISION_CONFIRMED if confirmed else DECISION_NOT_CONFIRMED,
        "uses_only_primary_metric": True,
        "scope_note": DECISION_SCOPE_NOTE,
    }


# ---------------------------------------------------------------------------
# Descriptive summaries
# ---------------------------------------------------------------------------


def _set_stats(left: Sequence[int], right: Sequence[int]) -> dict[str, Any]:
    left_set, right_set = set(left), set(right)
    intersection = left_set & right_set
    union = left_set | right_set
    return {
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": len(intersection) / len(union) if union else 1.0,
        "left_superset_right": left_set >= right_set,
        "right_superset_left": right_set >= left_set,
        "equal": left_set == right_set,
    }


def transition_counts(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "both_correct": 0,
        "control_only_correct": 0,
        "forced_only_correct": 0,
        "both_incorrect": 0,
    }
    for row in joined:
        control = bool(row["control_correct"])
        forced = bool(row["forced_correct"])
        if control and forced:
            counts["both_correct"] += 1
        elif control and not forced:
            counts["control_only_correct"] += 1
        elif not control and forced:
            counts["forced_only_correct"] += 1
        else:
            counts["both_incorrect"] += 1
    counts["output_text_changed"] = sum(
        1
        for row in joined
        if row["control_output_text"] != row["forced_output_text"]
    )
    counts["answer_changed"] = sum(
        1 for row in joined if row["control_parsed"] != row["forced_parsed"]
    )
    counts["sample_count"] = len(joined)
    counts["output_text_changed_rate"] = counts["output_text_changed"] / len(joined)
    counts["answer_changed_rate"] = counts["answer_changed"] / len(joined)
    return counts


def _bucket_label(value: int, labels: Sequence[tuple[str, Callable[[int], bool]]]) -> str:
    for label, predicate in labels:
        if predicate(value):
            return label
    return "other"


def budget_overview(
    reference: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    g_values = [int(row["G"]) for row in reference.values()]
    k_values = [int(row["K_C"]) for row in reference.values()]
    retention = [k / g for k, g in zip(k_values, g_values)]
    from collections import Counter

    g_buckets = Counter(_bucket_label(value, G_BUCKETS) for value in g_values)
    k_buckets = Counter(_bucket_label(value, K_BUCKETS) for value in k_values)
    return {
        "sample_count": len(reference),
        "G_min": min(g_values),
        "G_max": max(g_values),
        "G_mean": statistics.fmean(g_values),
        "K_C_min": min(k_values),
        "K_C_max": max(k_values),
        "K_C_mean": statistics.fmean(k_values),
        "K_C_total": sum(k_values),
        "retention_mean": statistics.fmean(retention),
        "retention_min": min(retention),
        "retention_max": max(retention),
        "G_buckets": {name: g_buckets.get(name, 0) for name, _ in G_BUCKETS},
        "K_C_buckets": {name: k_buckets.get(name, 0) for name, _ in K_BUCKETS},
        "K_C_distinct_count": len(set(k_values)),
        "K_C_histogram": {str(k): v for k, v in sorted(Counter(k_values).items())},
        "identity_bypass_count": sum(
            1 for row in reference.values() if int(row["K_C"]) >= int(row["G"])
        ),
    }


def selection_overview(
    reference: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
    forced: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    jaccards: list[float] = []
    control_superset = 0
    forced_superset = 0
    equal = 0
    forced_matches_reference = 0
    for sample_id in reference:
        stats = _set_stats(
            control[sample_id]["selected_group_indices"],
            forced[sample_id]["selected_group_indices"],
        )
        jaccards.append(stats["jaccard"])
        control_superset += int(stats["left_superset_right"])
        forced_superset += int(stats["right_superset_left"])
        equal += int(stats["equal"])
        if forced[sample_id]["selected_group_indices"] == list(
            reference[sample_id]["group_indices"]
        ):
            forced_matches_reference += 1
    return {
        "comparison": "control_vs_forced",
        "jaccard_mean": statistics.fmean(jaccards),
        "jaccard_min": min(jaccards),
        "jaccard_max": max(jaccards),
        "control_superset_forced_count": control_superset,
        "forced_superset_control_count": forced_superset,
        "equal_count": equal,
        "forced_equals_reference_count": forced_matches_reference,
        "sample_count": len(jaccards),
    }


_TIMING_FIELDS = (
    "model_inference_seconds",
    "application_end_to_end_seconds",
    "vision_total_seconds",
    "llm_prefill_seconds",
    "decoder_seconds",
    "selector_cpu_roundtrip_seconds",
)


def _field_stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def timing_overview(
    control: Mapping[str, Mapping[str, Any]],
    forced: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Descriptive timing only; never used for the decision."""
    overview: dict[str, Any] = {"used_for_decision": False}
    for arm, index in (("control", control), ("forced", forced)):
        arm_stats: dict[str, Any] = {}
        for field in _TIMING_FIELDS:
            values = [
                float(row[field]) for row in index.values() if field in row
            ]
            if values:
                arm_stats[field] = _field_stats(values)
        overview[arm] = arm_stats
    return overview


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def analyze(
    *,
    screening_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    forced_rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    reference_sha256: str,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate the four inputs and compute the diagnostic (no file I/O)."""
    screening_index = index_unique(screening_rows, "screening_manifest")
    reference_index = index_unique(reference_rows, "reference_jsonl")
    control_index = index_unique(control_rows, "control_arm")
    forced_index = index_unique(forced_rows, "forced_arm")

    validated = validate_inputs(
        screening_index=screening_index,
        reference_index=reference_index,
        control_index=control_index,
        forced_index=forced_index,
        manifest_sha256=manifest_sha256,
        reference_sha256=reference_sha256,
        expected_screening_count=expected_screening_count,
    )

    joined = build_joined(screening_index, control_index, forced_index)
    metrics = {
        ARM_CONTROL: arm_metrics(joined, ARM_CONTROL, ARM_CONTROL),
        ARM_FORCED: arm_metrics(joined, ARM_FORCED, ARM_FORCED),
    }
    comparison = paired_comparison(joined, ARM_FORCED, ARM_CONTROL)

    per_sample: list[dict[str, Any]] = []
    for row in joined:
        sample_id = row["sample_id"]
        control = validated["control"][sample_id]
        forced = validated["forced"][sample_id]
        reference = validated["reference"][sample_id]
        stats = _set_stats(
            control["selected_group_indices"], forced["selected_group_indices"]
        )
        per_sample.append(
            {
                "sample_id": sample_id,
                "task": row["task"],
                "type": row["type"],
                "source": row["source"],
                "source_dataset": row["source_dataset"],
                "ground_truth": row["ground_truth"],
                "control_answer": row["control_parsed"],
                "forced_answer": row["forced_parsed"],
                "control_correct": bool(row["control_correct"]),
                "forced_correct": bool(row["forced_correct"]),
                "G": reference["G"],
                "K_C": reference["K_C"],
                "control_K": control["K_C"],
                "forced_K": forced["K_C"],
                "reference_K_C": reference["K_C"],
                "control_indices_sha256": control["selected_group_indices_sha256"],
                "forced_indices_sha256": forced["selected_group_indices_sha256"],
                "reference_indices_sha256": reference["group_indices_sha256"],
                "control_equals_forced": bool(stats["equal"]),
                "jaccard": stats["jaccard"],
                "control_superset_forced": bool(stats["left_superset_right"]),
                "forced_superset_control": bool(stats["right_superset_left"]),
                "output_text_changed": row["control_output_text"]
                != row["forced_output_text"],
                "answer_changed": row["control_parsed"] != row["forced_parsed"],
                "forced_equals_reference": forced["selected_group_indices"]
                == list(reference["group_indices"]),
            }
        )

    return {
        "joined": joined,
        "validated": validated,
        "metrics": metrics,
        "comparison": comparison,
        "decision": decide(comparison),
        "transitions": transition_counts(joined),
        "budget_overview": budget_overview(validated["reference"]),
        "selection_overview": selection_overview(
            validated["reference"], validated["control"], validated["forced"]
        ),
        "timing_overview": timing_overview(
            validated["control"], validated["forced"]
        ),
        "per_sample": per_sample,
    }


def _inputs_sha256(paths: InputPaths) -> dict[str, Any]:
    return {
        "preregistered_config": {
            "path": str(paths.preregistered),
            "sha256": sha256_file(paths.preregistered),
        },
        "screening_manifest": {
            "path": str(paths.screening_manifest),
            "sha256": sha256_file(paths.screening_manifest),
        },
        "reference_jsonl": {
            "path": str(paths.reference_jsonl),
            "sha256": sha256_file(paths.reference_jsonl),
        },
        "control_arm": {
            "path": str(paths.control_arm),
            "sha256": sha256_file(paths.control_arm),
        },
        "forced_arm": {
            "path": str(paths.forced_arm),
            "sha256": sha256_file(paths.forced_arm),
        },
    }


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    verify_pins: bool = True,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load every required input, validate, score and assemble the report.

    Any missing file, SHA pin mismatch or failed contract check raises
    :class:`InputContractError` before a report is produced (fail closed).
    """
    require_file(paths.preregistered, "preregistered_config")
    config = json.loads(paths.preregistered.read_text(encoding="utf-8"))
    pins = verify_config_pins(config, paths, root=root) if verify_pins else None

    manifest_sha256 = sha256_file(paths.screening_manifest)
    reference_sha256 = sha256_file(paths.reference_jsonl)

    analysis = analyze(
        screening_rows=load_rows(paths.screening_manifest, "screening_manifest"),
        reference_rows=load_rows(paths.reference_jsonl, "reference_jsonl"),
        control_rows=load_rows(paths.control_arm, "control_arm"),
        forced_rows=load_rows(paths.forced_arm, "forced_arm"),
        manifest_sha256=manifest_sha256,
        reference_sha256=reference_sha256,
        expected_screening_count=expected_screening_count,
    )

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "analysis": "layer5_forced_index_ac_screening100",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": True,
        "formal_full2638_conclusion": False,
        "explicit_non_formal_statement": (
            "Fixed 100-sample screening causal diagnostic: the forced vs control "
            "signal is diagnostic only and is NOT a formal full2638 CV-Bench "
            "conclusion. It must not be quoted as the official CV-Bench accuracy "
            "of any approach."
        ),
        "single_variable_contract": {
            "only_variable": ONLY_VARIABLE,
            "arm_label": ARM_LABEL,
            "same_implementation": (
                "evaluation/hardware_regular_layer5_forced_index_ablation.py "
                "(one unified forward, layer-5 cut)"
            ),
            "same_per_sample_budget": True,
            "control": "control rule, exactly K_C real groups",
            "forced": "exact frozen shallow_r0250 historical indices, same K_C",
            "alignment_is_diagnostic_only": True,
            "fpga_deployable_candidate": FPGA_DEPLOYABLE_CANDIDATE,
            "hardware_alignment_relaxed_for_diagnostic": (
                HARDWARE_ALIGNMENT_RELAXED_FOR_DIAGNOSTIC
            ),
            "causal_diagnostic_only": CAUSAL_DIAGNOSTIC_ONLY,
        },
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
            "reference": str(paths.reference_jsonl),
            "reference_selector": REFERENCE_SELECTOR,
            "sample_count": SCREENING_SAMPLE_COUNT,
            "cut_point": PRUNE_AFTER_LAYER,
            "region_grid": REGION_GRID,
            "score_mode": SCORE_MODE,
            "local_candidate_cap": None,
            "dtype": REQUIRED_DTYPE,
            "attention_implementation": REQUIRED_ATTENTION,
            "decoding": REQUIRED_DECODING,
            "do_sample": False,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "stratified_by": STRATIFIED_BY,
            "b1_trained_wv_used": False,
            "uses_context_pooling": False,
            "uses_context_slot_reservation": False,
            "uses_local_candidate_cap": False,
        },
        "primary_metric": {
            "identifier": PRIMARY_METRIC,
            "formula": "((accuracy_ADE20K + accuracy_COCO)/2 + accuracy_Omni3D)/2",
            "rule": "the only primary metric for both arms and the only decision metric",
        },
        "unweighted_metric": {
            "identifier": "unweighted_sample_accuracy",
            "underlying_field": SECONDARY_METRIC,
            "rule": UNWEIGHTED_NOTE,
        },
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "arms": {
            ARM_CONTROL: {
                "path": str(paths.control_arm),
                "description": ARMS_DESCRIPTION[ARM_CONTROL],
                "selection_rule": SELECTION_RULE_CONTROL,
            },
            ARM_FORCED: {
                "path": str(paths.forced_arm),
                "description": ARMS_DESCRIPTION[ARM_FORCED],
                "selection_rule": SELECTION_RULE_FORCED,
            },
        },
        "contract": {
            "sample_count": SCREENING_SAMPLE_COUNT,
            "unique_sample_ids_per_input": True,
            "sample_id_sets_equal": True,
            "reference_indices_reproduced": True,
            "forced_equals_reference": True,
            "per_sample_K_identical_control_forced": True,
            "G_identical_control_forced_reference": True,
            "score_mode": SCORE_MODE,
            "prune_after_layer": PRUNE_AFTER_LAYER,
            "region_grid": REGION_GRID,
            "dtype": REQUIRED_DTYPE,
            "attention_implementation": REQUIRED_ATTENTION,
            "decoding": REQUIRED_DECODING,
            "all_real": True,
            "no_context": True,
        },
        "config_pin_verification": pins,
        "inputs_sha256": _inputs_sha256(paths),
        "metrics": analysis["metrics"],
        "comparison": analysis["comparison"],
        "decision": analysis["decision"],
        "correctness_transitions": analysis["transitions"],
        "budget_overview": analysis["budget_overview"],
        "selection_overview": analysis["selection_overview"],
        "timing_overview_descriptive_only": analysis["timing_overview"],
        "limitations": [
            "Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.",
            "The relaxed K_C alignment is a diagnostic setting; it is NOT an FPGA-deployable candidate.",
            "Both arms run the same unified forward; the only variable is selected_group_indices.",
            "Only score(...)[\"cv_bench_accuracy\"] drives the decision.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.",
            "Timings are descriptive only and never enter the decision.",
            "Neither arm uses the B1-trained Wv projection.",
        ],
        "per_sample": analysis["per_sample"],
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


CSV_COLUMNS = (
    "sample_id",
    "task",
    "type",
    "source",
    "source_dataset",
    "ground_truth",
    "control_answer",
    "forced_answer",
    "control_correct",
    "forced_correct",
    "G",
    "K_C",
    "control_K",
    "forced_K",
    "reference_K_C",
    "control_indices_sha256",
    "forced_indices_sha256",
    "reference_indices_sha256",
    "control_equals_forced",
    "jaccard",
    "control_superset_forced",
    "forced_superset_control",
    "output_text_changed",
    "answer_changed",
    "forced_equals_reference",
)


def render_csv(report: Mapping[str, Any]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
    writer.writeheader()
    for row in report["per_sample"]:
        writer.writerow({name: row[name] for name in CSV_COLUMNS})
    return stream.getvalue()


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _check(value: Any) -> str:
    return "✅" if value else "❌"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append(
        "# Layer-5 forced-index（forced vs control，同预算 K_C）screening100 因果诊断"
    )
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append("> ⚠️ " + report["explicit_non_formal_statement"])
    lines.append("")
    lines.append(
        "- 唯一主指标（唯一决策依据）：`" + report["primary_metric"]["identifier"] + "` = "
        "`((ADE20K + COCO)/2 + Omni3D)/2`；"
        "`unweighted_sample_accuracy`（`" + report["unweighted_metric"]["underlying_field"]
        + "`）仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。"
    )
    lines.append("- 禁用主指标：" + ", ".join(report["forbidden_primary_metrics"]))
    lines.append("")
    lines.append("## 1. 单变量合同与固定配置")
    lines.append("")
    sv = report["single_variable_contract"]
    cfg = report["fixed_configuration"]
    lines.append(
        f"- 唯一变量：`{sv['only_variable']}`；实现：`{sv['same_implementation']}`；"
        f"两臂逐样本同 K_C：{_check(sv['same_per_sample_budget'])}"
    )
    lines.append(
        f"- control：{sv['control']}；forced：{sv['forced']}"
    )
    lines.append(
        f"- alignment 仅为诊断放宽（非 FPGA 候选）：relaxed="
        f"{sv['hardware_alignment_relaxed_for_diagnostic']}；"
        f"fpga_deployable_candidate={sv['fpga_deployable_candidate']}；"
        f"causal_diagnostic_only={sv['causal_diagnostic_only']}"
    )
    lines.append(
        f"- manifest=`{cfg['manifest']}`；reference=`{cfg['reference']}`"
        f"（selector={cfg['reference_selector']}）；样本数={cfg['sample_count']}"
    )
    lines.append(
        f"- cut_point={cfg['cut_point']}；region_grid={cfg['region_grid']}；"
        f"score_mode={cfg['score_mode']}；local_candidate_cap={cfg['local_candidate_cap']}"
    )
    lines.append(
        f"- dtype={cfg['dtype']}；attention={cfg['attention_implementation']}；"
        f"decoding={cfg['decoding']}；do_sample={cfg['do_sample']}；"
        f"bootstrap seed={cfg['bootstrap_seed']} / repeats={cfg['bootstrap_repeats']} / "
        f"stratified_by={cfg['stratified_by']}"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验（fail-closed）")
    lines.append("")
    c = report["contract"]
    lines.append(f"- 100 个 sample_id 唯一且四输入一致：{_check(c['sample_id_sets_equal'])}")
    lines.append(f"- reference 索引与 hash 复核：{_check(c['reference_indices_reproduced'])}")
    lines.append(
        f"- forced 选择精确等于 reference：{_check(c['forced_equals_reference'])}"
    )
    lines.append(
        "- control/forced 每样本 K_C 相同："
        f"{_check(c['per_sample_K_identical_control_forced'])}"
    )
    lines.append(
        "- control/forced/reference 每样本 G 相同："
        f"{_check(c['G_identical_control_forced_reference'])}"
    )
    lines.append(
        f"- 两臂 BF16/SDPA/greedy：dtype={c['dtype']}、"
        f"attn={c['attention_implementation']}、decoding={c['decoding']}"
    )
    lines.append(
        f"- 两臂 all-real / 无 context：{_check(c['all_real'] and c['no_context'])}"
    )
    pins = report.get("config_pin_verification")
    if pins is not None:
        lines.append(
            f"- 预注册 SHA pin 复核：{_check(pins['pins_verified'])}"
            f"（已校验 {len(pins['checks'])} 项；null 待生成 pin "
            f"{len(pins['null_pins_skipped'])} 项：{pins['null_pins_skipped']}）"
        )
    lines.append("")
    lines.append("## 3. 两臂官方主分")
    lines.append("")
    lines.append(
        "| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | "
        "unweighted_sample_accuracy（次要） | 2D | 3D | parse |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        m = report["metrics"][arm]
        lines.append(
            f"| {arm} | {m['sample_count']} | **{_fmt(m['cv_bench_accuracy'])}** | "
            f"{_fmt(m['unweighted_sample_accuracy'])} | {_fmt(m['accuracy_2d'])} | "
            f"{_fmt(m['accuracy_3d'])} | {_fmt(m['parse_success_rate'])} |"
        )
    comp = report["comparison"]
    lines.append("")
    lines.append(
        f"- forced − control 点估计：**{comp['delta_point_estimate']:+.6f}**；"
        f"95% CI：[{comp['delta_ci95'][0]:+.6f}, {comp['delta_ci95'][1]:+.6f}]"
    )
    lines.append("")
    lines.append("## 4. 唯一决策（forced − control，主指标）")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 阈值：`{d['threshold']}`；结果：**{d['outcome']}**")
    lines.append(
        f"- 点估计={_fmt(d['point_estimate'])}；CI 下界={_fmt(d['delta_ci95'][0])}；"
        f"历史索引集合更好={d['historical_index_set_better']}"
    )
    lines.append(f"- 边界：{d['scope_note']}")
    lines.append("")
    lines.append("## 5. Correctness transitions")
    lines.append("")
    t = report["correctness_transitions"]
    lines.append(
        f"- 都对：{t['both_correct']}；仅 control 对：{t['control_only_correct']}；"
        f"仅 forced 对：{t['forced_only_correct']}；都错：{t['both_incorrect']}"
    )
    lines.append(
        f"- 输出文本变化：{t['output_text_changed']}（{_fmt(t['output_text_changed_rate'])}）；"
        f"答案变化：{t['answer_changed']}（{_fmt(t['answer_changed_rate'])}）"
    )
    lines.append("")
    lines.append("## 6. K / G 分布（reference = 逐样本 K_C）")
    lines.append("")
    b = report["budget_overview"]
    lines.append(
        f"- G：{b['G_min']}–{b['G_max']}（mean={_fmt(b['G_mean'])}）；"
        f"K_C：{b['K_C_min']}–{b['K_C_max']}（mean={_fmt(b['K_C_mean'])}、"
        f"total={b['K_C_total']}）"
    )
    lines.append(
        f"- retention：mean={_fmt(b['retention_mean'])}"
        f"（{_fmt(b['retention_min'])}–{_fmt(b['retention_max'])}）；"
        f"identity bypass={b['identity_bypass_count']}"
    )
    lines.append(
        "- G 桶：" + "、".join(f"{k}={v}" for k, v in b["G_buckets"].items())
        + "；K_C 桶：" + "、".join(f"{k}={v}" for k, v in b["K_C_buckets"].items())
    )
    lines.append(
        f"- K_C 不同取值数：{b['K_C_distinct_count']}；直方图：{b['K_C_histogram']}"
    )
    lines.append("")
    lines.append("## 7. 索引集合 Jaccard / 嵌套（control vs forced）")
    lines.append("")
    s = report["selection_overview"]
    lines.append(
        f"- Jaccard：mean={_fmt(s['jaccard_mean'])}"
        f"（{_fmt(s['jaccard_min'])}–{_fmt(s['jaccard_max'])}）"
    )
    lines.append(
        f"- control ⊇ forced：{s['control_superset_forced_count']}；"
        f"forced ⊇ control：{s['forced_superset_control_count']}；"
        f"集合相等：{s['equal_count']}；forced == reference：{s['forced_equals_reference_count']}"
    )
    lines.append("")
    lines.append("## 8. 时间（仅描述，**不用于决策**）")
    lines.append("")
    timing = report["timing_overview_descriptive_only"]
    lines.append(f"- used_for_decision={timing['used_for_decision']}")
    for arm in ARMS:
        arm_stats = timing.get(arm, {})
        if "model_inference_seconds" in arm_stats:
            item = arm_stats["model_inference_seconds"]
            lines.append(
                f"- {arm} model_inference_seconds：mean={_fmt(item['mean'])}、"
                f"median={_fmt(item['median'])}、min={_fmt(item['min'])}、"
                f"max={_fmt(item['max'])}"
            )
        if "application_end_to_end_seconds" in arm_stats:
            item = arm_stats["application_end_to_end_seconds"]
            lines.append(
                f"- {arm} application_end_to_end_seconds：mean={_fmt(item['mean'])}、"
                f"median={_fmt(item['median'])}、min={_fmt(item['min'])}、"
                f"max={_fmt(item['max'])}"
            )
    lines.append("")
    lines.append("## 9. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 10. 限制")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "> CPU-only 只读汇总：未改原始 JSONL、正式 quality report 或 "
        "`configs/evaluation_protocol.json`；无 FPGA/时延/加速声明。"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistered", type=Path, default=PREREGISTERED_CONFIG)
    parser.add_argument("--manifest", type=Path, default=SCREENING_MANIFEST)
    parser.add_argument("--reference-jsonl", type=Path, default=REFERENCE_JSONL)
    parser.add_argument("--control-jsonl", type=Path, default=CONTROL_ARM)
    parser.add_argument("--forced-jsonl", type=Path, default=FORCED_ARM)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = InputPaths(
        preregistered=args.preregistered,
        screening_manifest=args.manifest,
        reference_jsonl=args.reference_jsonl,
        control_arm=args.control_jsonl,
        forced_arm=args.forced_jsonl,
    )
    outputs = OutputPaths(json=args.output_json, csv=args.output_csv, md=args.output_md)
    try:
        report = build_report(paths)
    except (InputContractError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2

    # Render everything before touching disk so a render failure cannot leave a
    # partial report; a contract failure above already wrote nothing.
    markdown = render_markdown(report)
    csv_text = render_csv(report)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"

    outputs.json.parent.mkdir(parents=True, exist_ok=True)
    outputs.json.write_text(payload, encoding="utf-8")
    outputs.csv.write_text(csv_text, encoding="utf-8")
    outputs.md.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(outputs.json),
                "output_csv": str(outputs.csv),
                "output_md": str(outputs.md),
                "scope": report["report_scope"],
                "decision": report["decision"]["outcome"],
                "forced_minus_control": report["comparison"]["delta_point_estimate"],
                "forced_minus_control_ci95": report["comparison"]["delta_ci95"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
