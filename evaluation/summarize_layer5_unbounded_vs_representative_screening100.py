#!/usr/bin/env python3
"""Diagnostic screening100 unbounded real-only vs raw-representative summariser.

Summariser for the preregistered experiment
``configs/2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json``.

Both Selector arms are *raw real groups* at the same layer-5 cut point, K=96,
``region_grid=3``, BF16 + SDPA + greedy decoding.  They differ **only** in the
selection constraint / indexing:

* ``representative`` (arm A) - the frozen layer-5 context-pooling ablation arm
  (``f09f5a53...``) where each ``SOURCE_CONTEXT`` slot uses the raw
  representative 2x2 group; its selection metadata is the frozen hardware-regular
  selector output (context pooling + context slot reservation + local cap 16);
* ``unbounded`` (arm B) - the independent global-L2 + 3x3 anchor real-only arm
  produced later by ``evaluation/run_hf_hardware_regular_layer5_unbounded_l2.py``
  with **no** context pooling, **no** context slot reservation and **no**
  ``local_candidate_cap``;
* ``dense`` - the frozen custom dense full2638 JSONL restricted to the 100 ids;
  reported for **reference only** and never used by the decision.

Hard rules enforced here:

1. The **only** primary metric is the official
   ``score(...)["cv_bench_accuracy"]`` returned by the frozen
   :mod:`evaluation.build_cv_bench_pruning_report` scorer.  The plain per-sample
   mean is exposed strictly as ``unweighted_sample_accuracy`` and never drives
   the decision.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Inputs are validated before any score: exactly the same 100 ``sample_id``
   values with no duplicates; every record must have ``score_mode == "l2"``,
   ``requested_budget == 96`` and ``prune_after_layer == 5``; every non-bypass
   output must be K=96; and the ``(image_grid_thw, full_patch_tokens,
   full_llm_visual_tokens)`` tuple must match between the two Selector arms.
4. The representative arm must declare ``context_group_values_source ==
   "representative_raw_group"`` and ``uses_pooled_context_group_values ==
   false``; the unbounded arm must prove it is all-real
   (``source_type`` all ``0``), context-free (``context_group_count == 0``),
   never local-cap bounded (``uses_local_candidate_cap == false``,
   ``local_candidate_cap`` null) and either an exact-K emission or the ``G <= K``
   identity bypass.
5. Config-pinned input SHA256 values are re-verified and a missing required file
   or any failed check raises :class:`InputContractError`; no report is written
   (fail closed).

Decision: the 95% paired stratified bootstrap CI (``task::source_dataset``,
seed 20261004, 5000 repeats) of ``unbounded_minus_representative`` on
``score(...)["cv_bench_accuracy"]``.  If the CI lower bound is strictly greater
than ``0.02`` the outcome is
``HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED``, otherwise
``HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED``.  Dense never influences the
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    / "2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json"
)
SCREENING_MANIFEST = ROOT / "datasets" / "cv_bench_screening100.jsonl"
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DENSE_FULL_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-context-pooling-ablation"
    / "layer5_context_ablation_k96_screening100.jsonl"
)
UNBOUNDED_LAYER5_L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-unbounded-real-only"
    / "layer5_unbounded_real_only_k96_screening100.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-layer5-unbounded-vs-representative-screening100.json"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-layer5-unbounded-vs-representative-screening100.md"
)

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = "unweighted_sample_accuracy"
SECONDARY_METRIC = 'score(...)["accuracy"]'
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

BUDGET = 96
SCREENING_SAMPLE_COUNT = 100
FULL_SAMPLE_COUNT = 2638
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
SELECTION_CONSTRAINT_CI_LOWER = 0.02
SCORE_MODE = "l2"
PRUNE_AFTER_LAYER = 5
REGION_GRID = 3

CONTEXT_GROUP_VALUES_SOURCE = "representative_raw_group"
SELECTION_RULE = "unbounded_real_only_global_l2_3x3_anchor"
ARM_LABEL = "layer5_unbounded_real_only_block5"
SOURCE_REAL = 0

HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED = (
    "HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED"
)
HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED = (
    "HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED"
)
DECISION_ORDER = (
    HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED,
    HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED,
)
DECISION_COMPARISON = "unbounded_minus_representative"

DIAGNOSTIC_SCOPE = (
    "diagnostic screening100; NOT a formal full2638 CV-Bench conclusion"
)
GRID_SIGNATURE_DEFINITION = (
    "(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)"
)
DENSE_ROLE = "reference_only"

#: Scalar metadata that must match within an arm.
SELECTION_COUNT_FIELDS = ("context_group_count", "real_group_count")

#: Logical input names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = (
    "screening_manifest",
    "full_manifest",
    "dense_full_jsonl",
    "representative_layer5_l2_screening_jsonl",
    "unbounded_layer5_l2_screening_jsonl",
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the preregistered contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    representative_layer5_l2_screening_jsonl: Path = (
        REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL
    )
    unbounded_layer5_l2_screening_jsonl: Path = UNBOUNDED_LAYER5_L2_SCREENING_JSONL


DEFAULT_PATHS = InputPaths()


# ---------------------------------------------------------------------------
# Small deterministic helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Index rows by ``sample_id``; a duplicate or missing id is a hard error."""

    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    missing_ids: list[int] = []
    for position, row in enumerate(rows):
        if "sample_id" not in row:
            missing_ids.append(position)
            continue
        sample_id = str(row["sample_id"])
        if sample_id in index:
            duplicates.append(sample_id)
        index[sample_id] = dict(row)
    if missing_ids:
        raise InputContractError(
            f"input '{label}' has rows without sample_id at positions {missing_ids[:10]}"
        )
    if duplicates:
        raise InputContractError(
            f"input '{label}' has duplicate sample_id values: "
            f"{sorted(set(duplicates))[:10]}"
        )
    return index


def grid_signature(row: Mapping[str, Any]) -> tuple[tuple[int, ...], int, int]:
    """The per-sample input-grid signature used to prove both arms match."""

    for field in ("image_grid_thw", "full_patch_tokens", "full_llm_visual_tokens"):
        if field not in row:
            raise InputContractError(f"record is missing {field}")
    return (
        tuple(int(value) for value in row["image_grid_thw"]),
        int(row["full_patch_tokens"]),
        int(row["full_llm_visual_tokens"]),
    )


def _check_common_arm_fields(
    index: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    """Every Selector arm record must be K=96, L2 and structurally complete."""

    bad_score_mode = sorted(
        sid for sid, row in index.items() if str(row.get("score_mode")) != SCORE_MODE
    )
    if bad_score_mode:
        raise InputContractError(
            f"{label} records must all have score_mode == '{SCORE_MODE}'; "
            f"offenders: {bad_score_mode[:10]}"
        )
    bad_budget: list[str] = []
    for sid, row in index.items():
        if "requested_budget" not in row:
            bad_budget.append(sid)
            continue
        try:
            budget = int(row["requested_budget"])
        except (TypeError, ValueError):
            bad_budget.append(sid)
            continue
        if budget != BUDGET:
            bad_budget.append(sid)
    if bad_budget:
        raise InputContractError(
            f"{label} records must all have requested_budget == {BUDGET}; "
            f"offenders: {sorted(bad_budget)[:10]}"
        )
    bad_k: list[str] = []
    for sid, row in index.items():
        if "output_group_count" not in row:
            continue
        if bool(row.get("identity_bypass")):
            continue
        if int(row["output_group_count"]) != BUDGET:
            bad_k.append(sid)
    if bad_k:
        raise InputContractError(
            f"{label} non-bypass records must emit exactly K={BUDGET} groups; "
            f"offenders: {sorted(bad_k)[:10]}"
        )
    missing_layer: list[str] = []
    bad_layer: list[str] = []
    for sid, row in index.items():
        if "prune_after_layer" not in row:
            missing_layer.append(sid)
            continue
        try:
            layer = int(row["prune_after_layer"])
        except (TypeError, ValueError):
            bad_layer.append(sid)
            continue
        if layer != PRUNE_AFTER_LAYER:
            bad_layer.append(sid)
    if missing_layer:
        raise InputContractError(
            f"{label} records must all carry prune_after_layer == "
            f"{PRUNE_AFTER_LAYER}; missing on: {sorted(missing_layer)[:10]}"
        )
    if bad_layer:
        raise InputContractError(
            f"{label} records must all have prune_after_layer == "
            f"{PRUNE_AFTER_LAYER}; offenders: {sorted(bad_layer)[:10]}"
        )


def _check_representative_declaration(
    index: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    """Arm A must still declare the raw-representative context ablation."""

    for sid, row in index.items():
        source = str(row.get("context_group_values_source"))
        if source != CONTEXT_GROUP_VALUES_SOURCE:
            raise InputContractError(
                f"{label} record {sid} must declare context_group_values_source == "
                f"'{CONTEXT_GROUP_VALUES_SOURCE}', got {source!r}"
            )
        if bool(row.get("uses_pooled_context_group_values", True)):
            raise InputContractError(
                f"{label} record {sid} must set uses_pooled_context_group_values=false"
            )


def _check_unbounded_declaration(
    index: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    """Arm B must prove all-real, context-free, exact-K-or-identity selection."""

    for sid, row in index.items():
        if bool(row.get("uses_context_pooling", True)):
            raise InputContractError(
                f"{label} record {sid} must set uses_context_pooling=false"
            )
        if bool(row.get("uses_context_slot_reservation", True)):
            raise InputContractError(
                f"{label} record {sid} must set uses_context_slot_reservation=false"
            )
        if bool(row.get("uses_local_candidate_cap", True)):
            raise InputContractError(
                f"{label} record {sid} must set uses_local_candidate_cap=false"
            )
        if "local_candidate_cap" in row and row["local_candidate_cap"] not in (None, 0):
            raise InputContractError(
                f"{label} record {sid} must not carry a positive local_candidate_cap, "
                f"got {row['local_candidate_cap']!r}"
            )
        if str(row.get("selection_rule")) != SELECTION_RULE:
            raise InputContractError(
                f"{label} record {sid} must declare selection_rule == "
                f"'{SELECTION_RULE}', got {row.get('selection_rule')!r}"
            )
        if "source_type" not in row:
            raise InputContractError(
                f"{label} record {sid} is missing source_type"
            )
        if any(int(value) != SOURCE_REAL for value in row["source_type"]):
            raise InputContractError(
                f"{label} record {sid} must be all-real (source_type all 0)"
            )
        if "context_group_count" not in row or int(row["context_group_count"]) != 0:
            raise InputContractError(
                f"{label} record {sid} must have context_group_count == 0"
            )
        if "real_group_count" not in row or "output_group_count" not in row:
            raise InputContractError(
                f"{label} record {sid} is missing real_group_count/output_group_count"
            )
        if int(row["real_group_count"]) != int(row["output_group_count"]):
            raise InputContractError(
                f"{label} record {sid} real_group_count "
                f"{row['real_group_count']} != output_group_count "
                f"{row['output_group_count']}"
            )
        if bool(row.get("identity_bypass")):
            continue
        if int(row["output_group_count"]) != BUDGET:
            raise InputContractError(
                f"{label} record {sid} non-bypass output_group_count "
                f"{row['output_group_count']} != K={BUDGET}"
            )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_arms(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    representative_index: Mapping[str, Mapping[str, Any]],
    unbounded_index: Mapping[str, Mapping[str, Any]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate the preregistered input contract; raise ``InputContractError``."""

    screening_ids = list(screening_index)
    if len(screening_ids) != expected_screening_count:
        raise InputContractError(
            f"screening manifest must hold {expected_screening_count} unique samples, "
            f"got {len(screening_ids)}"
        )
    if expected_full_count is not None and len(dense_index) != expected_full_count:
        raise InputContractError(
            f"dense full JSONL must hold {expected_full_count} unique samples, "
            f"got {len(dense_index)}"
        )

    screening_set = set(screening_ids)
    missing_dense = sorted(screening_set - set(dense_index))
    if missing_dense:
        raise InputContractError(
            f"dense full JSONL is missing screening sample_ids: {missing_dense[:10]}"
        )

    for label, index in (
        ("representative", representative_index),
        ("unbounded", unbounded_index),
    ):
        index_set = set(index)
        missing = sorted(screening_set - index_set)
        extra = sorted(index_set - screening_set)
        if missing or extra:
            raise InputContractError(
                f"{label} screening100 sample_id set does not equal the screening "
                f"manifest: missing={missing[:10]} extra={extra[:10]}"
            )

    _check_common_arm_fields(representative_index, "representative")
    _check_common_arm_fields(unbounded_index, "unbounded")
    _check_representative_declaration(representative_index, "representative")
    _check_unbounded_declaration(unbounded_index, "unbounded")

    grid_mismatches: list[dict[str, Any]] = []
    for sid in screening_ids:
        representative_sig = grid_signature(representative_index[sid])
        unbounded_sig = grid_signature(unbounded_index[sid])
        if unbounded_sig != representative_sig:
            grid_mismatches.append(
                {
                    "sample_id": sid,
                    "unbounded": list(unbounded_sig[0])
                    + [unbounded_sig[1], unbounded_sig[2]],
                    "representative": list(representative_sig[0])
                    + [representative_sig[1], representative_sig[2]],
                }
            )
    if grid_mismatches:
        raise InputContractError(
            "unbounded input grid is not consistent with the representative arm for "
            f"{len(grid_mismatches)} samples; first: {grid_mismatches[0]}"
        )

    return {
        "screening_sample_count": len(screening_ids),
        "dense_full_sample_count": len(dense_index),
        "representative_sample_count": len(representative_index),
        "unbounded_sample_count": len(unbounded_index),
        "sample_ids_identical": True,
        "duplicate_sample_ids": {
            "screening": [],
            "dense": [],
            "representative": [],
            "unbounded": [],
        },
        "all_score_mode_l2": True,
        "all_requested_budget_96": True,
        "all_prune_after_layer_5": True,
        "input_grid_consistent_representative_vs_unbounded": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "representative_declaration_ok": True,
        "context_group_values_source": CONTEXT_GROUP_VALUES_SOURCE,
        "unbounded_declaration_ok": True,
        "unbounded_selection_rule": SELECTION_RULE,
        "unbounded_all_real": True,
        "unbounded_no_context": True,
        "unbounded_no_context_slot_reservation": True,
        "unbounded_no_local_candidate_cap": True,
        "unbounded_exact_k_or_identity": True,
        "selection_metadata_identity_required": False,
        "checks": [
            "100 identical unique sample_ids across manifest/dense/representative/unbounded",
            "no duplicate sample_id in any input",
            'every representative and unbounded record score_mode == "l2"',
            f"every representative and unbounded record requested_budget == {BUDGET}",
            f"every representative and unbounded record prune_after_layer == {PRUNE_AFTER_LAYER}",
            "representative input grid == unbounded input grid for every sample",
            "representative arm declares the raw-representative context ablation",
            "unbounded arm proves all-real (source_type all 0) and context-free "
            "(context_group_count == 0, no pooling / slot reservation / local cap)",
            "unbounded arm emits exactly K=96 or the G<=K identity bypass",
        ],
    }


# ---------------------------------------------------------------------------
# Config pin verification
# ---------------------------------------------------------------------------


def _logical_pin_path(name: str, paths: InputPaths) -> Path | None:
    return {
        "screening_manifest": paths.screening_manifest,
        "full_manifest": paths.full_manifest,
        "dense_full_jsonl": paths.dense_full_jsonl,
        "representative_layer5_l2_screening_jsonl": (
            paths.representative_layer5_l2_screening_jsonl
        ),
        "unbounded_layer5_l2_screening_jsonl": (
            paths.unbounded_layer5_l2_screening_jsonl
        ),
    }.get(name)


def verify_config_pins(
    config: Mapping[str, Any],
    paths: InputPaths = DEFAULT_PATHS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Re-verify every non-null ``inputs_sha256`` pin; fail closed on mismatch."""

    checks: list[dict[str, Any]] = []
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
        if expected is None:
            logical = _logical_pin_path(str(name), paths)
            if logical is not None:
                require_file(logical, str(name))
            continue
        expected = str(expected).lower()
        logical = _logical_pin_path(str(name), paths)
        if logical is not None:
            path = logical
            label = str(name)
        elif relative is not None:
            path = root / relative
            label = str(relative)
        else:
            raise InputContractError(
                f"pin '{name}' has neither a logical input nor a relative path"
            )
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
    return {"pins_verified": True, "checks": checks}


# ---------------------------------------------------------------------------
# Official scoring and paired bootstrap
# ---------------------------------------------------------------------------


def build_joined(
    screening_ids: Sequence[str],
    screening_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build official joined records; correctness is recomputed with the scorer."""

    joined: list[dict[str, Any]] = []
    for sample_id in screening_ids:
        meta = screening_index[sample_id]
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "task": meta["task"],
            "type": meta["type"],
            "source": meta["source"],
            "source_dataset": meta["source_dataset"],
            "ground_truth": truth,
        }
        for arm, index in arm_indices.items():
            parsed = parse_choice(str(index[sample_id]["output_text"]), choices)
            record[f"{arm}_answer"] = parsed
            record[f"{arm}_correct"] = parsed == truth
        joined.append(record)
    return joined


def arm_metrics(
    joined: Sequence[Mapping[str, Any]], arm: str, label: str
) -> dict[str, Any]:
    """Official metrics for one arm; primary is ``cv_bench_accuracy`` only."""

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
        "metric": PRIMARY_METRIC,
        "delta_point_estimate": point,
        "delta_ci95": [float(ci[0]), float(ci[1])],
        "stratified_by": "task::source_dataset",
        "seed": BOOTSTRAP_SEED,
        "repeats": BOOTSTRAP_REPEATS,
    }


def decide(unbounded_minus_representative_ci: Sequence[float]) -> dict[str, Any]:
    """Apply the preregistered hardware-selection-constraint decision."""

    lower, upper = (
        float(unbounded_minus_representative_ci[0]),
        float(unbounded_minus_representative_ci[1]),
    )
    if lower > SELECTION_CONSTRAINT_CI_LOWER:
        outcome = HARDWARE_SELECTION_CONSTRAINT_HARM_CONFIRMED
        reason = (
            f"unbounded_minus_representative CI lower {lower:.6f} > "
            f"{SELECTION_CONSTRAINT_CI_LOWER}"
        )
    else:
        outcome = HARDWARE_SELECTION_CONSTRAINT_HARM_NOT_CONFIRMED
        reason = (
            f"unbounded_minus_representative CI lower {lower:.6f} <= "
            f"{SELECTION_CONSTRAINT_CI_LOWER}"
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "metric": PRIMARY_METRIC,
        "unbounded_minus_representative_ci95": [lower, upper],
        "threshold": SELECTION_CONSTRAINT_CI_LOWER,
        "ordered_rules": list(DECISION_ORDER),
        "comparison": DECISION_COMPARISON,
        "uses_only_primary_metric": True,
        "dense_role": DENSE_ROLE,
        "dense_drives_decision": False,
    }


def analyze(
    *,
    screening_rows: Sequence[Mapping[str, Any]],
    dense_full_rows: Sequence[Mapping[str, Any]],
    representative_rows: Sequence[Mapping[str, Any]],
    unbounded_rows: Sequence[Mapping[str, Any]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate inputs and compute the diagnostic result (no file I/O)."""

    screening_index = index_unique(screening_rows, "screening_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    representative_index = index_unique(
        representative_rows, "representative_layer5_l2_screening100"
    )
    unbounded_index = index_unique(
        unbounded_rows, "unbounded_layer5_l2_screening100"
    )

    validation = validate_arms(
        screening_index=screening_index,
        dense_index=dense_index,
        representative_index=representative_index,
        unbounded_index=unbounded_index,
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    joined = build_joined(
        list(screening_index),
        screening_index,
        {
            "dense": dense_index,
            "representative": representative_index,
            "unbounded": unbounded_index,
        },
    )

    metrics = {
        "dense": arm_metrics(joined, "dense", "dense"),
        "representative": arm_metrics(
            joined, "representative", "layer5_representative_block5"
        ),
        "unbounded": arm_metrics(
            joined, "unbounded", "layer5_unbounded_real_only_block5"
        ),
    }

    dense_primary = metrics["dense"]["cv_bench_accuracy"]
    if dense_primary == 0:
        raise InputContractError(
            "dense cv_bench_accuracy is 0; relative retention undefined"
        )
    relative = {
        "representative": metrics["representative"]["cv_bench_accuracy"] / dense_primary,
        "unbounded": metrics["unbounded"]["cv_bench_accuracy"] / dense_primary,
        "definition": "arm cv_bench_accuracy / dense cv_bench_accuracy (reference only)",
        "dense_role": DENSE_ROLE,
    }

    comparisons = {
        "unbounded_minus_representative": paired_comparison(
            joined, "unbounded", "representative"
        ),
        "unbounded_minus_dense": paired_comparison(joined, "unbounded", "dense"),
        "representative_minus_dense": paired_comparison(
            joined, "representative", "dense"
        ),
    }

    decision = decide(comparisons[DECISION_COMPARISON]["delta_ci95"])

    return {
        "validation": validation,
        "joined": joined,
        "metrics": metrics,
        "relative_retention_vs_dense": relative,
        "comparisons": comparisons,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


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
        "full_manifest": {
            "path": str(paths.full_manifest),
            "sha256": sha256_file(paths.full_manifest),
        },
        "dense_full_jsonl": {
            "path": str(paths.dense_full_jsonl),
            "sha256": sha256_file(paths.dense_full_jsonl),
        },
        "representative_layer5_l2_screening_jsonl": {
            "path": str(paths.representative_layer5_l2_screening_jsonl),
            "sha256": sha256_file(paths.representative_layer5_l2_screening_jsonl),
        },
        "unbounded_layer5_l2_screening_jsonl": {
            "path": str(paths.unbounded_layer5_l2_screening_jsonl),
            "sha256": sha256_file(paths.unbounded_layer5_l2_screening_jsonl),
        },
    }


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
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

    screening_rows = load_rows(paths.screening_manifest, "screening_manifest")
    full_index = index_unique(
        load_rows(paths.full_manifest, "full_manifest"), "full_manifest"
    )
    screening_ids = {str(row["sample_id"]) for row in screening_rows}
    missing_from_full = sorted(screening_ids - set(full_index))
    if missing_from_full:
        raise InputContractError(
            "screening manifest sample_ids are not a subset of the full2638 "
            f"manifest: {missing_from_full[:10]}"
        )

    analysis = analyze(
        screening_rows=screening_rows,
        dense_full_rows=load_rows(paths.dense_full_jsonl, "dense_full2638"),
        representative_rows=load_rows(
            paths.representative_layer5_l2_screening_jsonl,
            "representative_layer5_l2_screening100",
        ),
        unbounded_rows=load_rows(
            paths.unbounded_layer5_l2_screening_jsonl,
            "unbounded_layer5_l2_screening100",
        ),
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": True,
        "formal_full2638_conclusion": False,
        "explicit_non_formal_statement": (
            "Fixed stratified 100-sample screening: a single-variable "
            "selection-constraint attribution signal only, NOT a formal full2638 "
            "CV-Bench conclusion. It must not be quoted as the official CV-Bench "
            "accuracy of any approach."
        ),
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
            "budget_K": BUDGET,
            "cut_point": PRUNE_AFTER_LAYER,
            "region_grid": REGION_GRID,
            "local_candidate_cap": None,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "decoding": "greedy_argmax",
            "do_sample": False,
            "score_mode": SCORE_MODE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "b1_trained_wv_used": False,
            "representative_changed_variable": (
                "raw representative 2x2 context groups under the frozen hardware-regular "
                "selector (context pooling + slot reservation + local_candidate_cap=16)"
            ),
            "unbounded_selection_rule": SELECTION_RULE,
            "unbounded_uses_context_pooling": False,
            "unbounded_uses_context_slot_reservation": False,
            "unbounded_uses_local_candidate_cap": False,
        },
        "primary_metric": {
            "identifier": PRIMARY_METRIC,
            "formula": "((accuracy_ADE20K + accuracy_COCO)/2 + accuracy_Omni3D)/2",
            "rule": "the only primary metric for every arm and every comparison",
        },
        "unweighted_metric": {
            "identifier": UNWEIGHTED_METRIC,
            "underlying_field": SECONDARY_METRIC,
            "rule": "secondary diagnostic only; never the CV-Bench primary and never decision-driving",
        },
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "arms": {
            "dense": {"path": str(paths.dense_full_jsonl), "role": DENSE_ROLE},
            "representative_layer5_l2": {
                "path": str(paths.representative_layer5_l2_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "context_group_values_source": CONTEXT_GROUP_VALUES_SOURCE,
                "uses_pooled_context_group_values": False,
            },
            "unbounded_layer5_l2": {
                "path": str(paths.unbounded_layer5_l2_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "selection_rule": SELECTION_RULE,
                "uses_context_pooling": False,
                "uses_context_slot_reservation": False,
                "uses_local_candidate_cap": False,
                "local_candidate_cap": None,
            },
        },
        "validation": analysis["validation"],
        "config_pin_verification": pins,
        "metrics": analysis["metrics"],
        "relative_retention_vs_dense": analysis["relative_retention_vs_dense"],
        "comparisons": analysis["comparisons"],
        "decision": analysis["decision"],
        "inputs_sha256": _inputs_sha256(paths),
        "limitations": [
            "Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.",
            "Both Selector arms are software L2 only; no FPGA, latency or acceleration claim.",
            "Dense is the frozen custom dense path and is reference only; it never drives the decision.",
            "The decision uses score(...)[\"cv_bench_accuracy\"] exclusively.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.",
            "Both arms are raw real groups; they differ only in the selection constraint "
            "and indexing (bounded context-pooling selector vs unbounded global-L2 + 3x3 anchor).",
            "Neither arm uses the B1-trained Wv projection.",
        ],
    }
    return report


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Layer-5 unbounded real-only vs raw-representative（screening100 诊断）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append("> ⚠️ " + report["explicit_non_formal_statement"])
    lines.append("")
    lines.append(
        "- 正式主指标（唯一）：`" + report["primary_metric"]["identifier"] + "` "
        "= `((ADE20K + COCO)/2 + Omni3D)/2`；"
        "`" + report["unweighted_metric"]["identifier"] + "` 仅为次要诊断，"
        "**不得写作 CV-Bench 主分**，也不参与决策。"
    )
    lines.append("- 禁用主指标：" + ", ".join(report["forbidden_primary_metrics"]))
    lines.append("- dense 仅为参考（reference only），不参与决策。")
    lines.append("")
    lines.append("## 1. 固定配置")
    lines.append("")
    cfg = report["fixed_configuration"]
    lines.append(
        f"- 样本：`{cfg['manifest']}`；K={cfg['budget_K']}；切点：{cfg['cut_point']}；"
        f"region_grid={cfg['region_grid']}；local_cap={cfg['local_candidate_cap']}"
    )
    lines.append(
        f"- dtype={cfg['dtype']}；attention={cfg['attention_implementation']}；"
        f"decoding={cfg['decoding']}；score_mode={cfg['score_mode']}；"
        f"bootstrap seed={cfg['bootstrap_seed']} / repeats={cfg['bootstrap_repeats']}"
    )
    lines.append(
        f"- 变量：representative={cfg['representative_changed_variable']}；"
        f"unbounded selection_rule={cfg['unbounded_selection_rule']}；"
        "unbounded 无 context pooling / 无 context slot reservation / 无 local_candidate_cap"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- screening 样本数：{v['screening_sample_count']}")
    lines.append(f"- representative layer5 样本数：{v['representative_sample_count']}")
    lines.append(f"- unbounded layer5 样本数：{v['unbounded_sample_count']}")
    lines.append(f"- 100 个 sample_id 完全一致：{'✅' if v['sample_ids_identical'] else '❌'}")
    lines.append(
        f"- score_mode 全为 `{SCORE_MODE}`：{'✅' if v['all_score_mode_l2'] else '❌'}"
    )
    lines.append(
        f"- requested_budget 全为 {BUDGET}（K={BUDGET}）："
        f"{'✅' if v['all_requested_budget_96'] else '❌'}"
    )
    lines.append(
        f"- prune_after_layer 全为 {PRUNE_AFTER_LAYER}："
        f"{'✅' if v['all_prune_after_layer_5'] else '❌'}"
    )
    lines.append(
        f"- 输入 grid 两臂一致（{v['grid_signature_definition']}）："
        f"{'✅' if v['input_grid_consistent_representative_vs_unbounded'] else '❌'}"
    )
    lines.append(
        f"- representative 臂声明 context_group_values_source="
        f"`{v['context_group_values_source']}`："
        f"{'✅' if v['representative_declaration_ok'] else '❌'}"
    )
    lines.append(
        "- unbounded 臂 all-real / 无 context（pooling、slot reservation、"
        "local_candidate_cap）/ 精确 K 或 identity："
        f"{'✅' if v['unbounded_declaration_ok'] else '❌'}"
    )
    pins = report.get("config_pin_verification")
    if pins is not None:
        lines.append(
            f"- 输入 SHA 复核：{'✅' if pins['pins_verified'] else '❌'}"
            f"（{len(pins['checks'])} 项）"
        )
    lines.append("")
    lines.append("## 3. 三臂正式主分")
    lines.append("")
    lines.append(
        "| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | "
        "unweighted_sample_accuracy（次要） | 相对 dense 保留（参考） |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    retention = report["relative_retention_vs_dense"]
    for arm in ("dense", "representative", "unbounded"):
        m = report["metrics"][arm]
        rel = "1.000000" if arm == "dense" else _pct(retention[arm])
        lines.append(
            f"| {m['label']} | {m['sample_count']} | **{_pct(m['cv_bench_accuracy'])}** | "
            f"{_pct(m['unweighted_sample_accuracy'])} | {rel} |"
        )
    lines.append("")
    lines.append("## 4. 配对分层 bootstrap（95% CI，主指标）")
    lines.append("")
    lines.append("| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |")
    lines.append("|---|---:|---:|---:|")
    for name in (
        "unbounded_minus_representative",
        "unbounded_minus_dense",
        "representative_minus_dense",
    ):
        c = report["comparisons"][name]
        lines.append(
            f"| {c['left']} − {c['right']} | {c['delta_point_estimate']:+.6f} | "
            f"{c['delta_ci95'][0]:+.6f} | {c['delta_ci95'][1]:+.6f} |"
        )
    lines.append("")
    lines.append("## 5. 预注册决策（仅用 unbounded − representative 主指标）")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 结论：**{d['outcome']}**")
    lines.append(f"- 依据：{d['reason']}")
    lines.append(
        f"- unbounded−representative 95% CI：["
        f"{d['unbounded_minus_representative_ci95'][0]:+.6f}, "
        f"{d['unbounded_minus_representative_ci95'][1]:+.6f}]；阈值 > {d['threshold']}"
    )
    lines.append(f"- dense 作用：{d['dense_role']}（不参与决策）")
    lines.append("")
    lines.append("## 6. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 7. 限制")
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
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_report(DEFAULT_PATHS)
    except (InputContractError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "scope": report["report_scope"],
                "decision": report["decision"]["outcome"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
