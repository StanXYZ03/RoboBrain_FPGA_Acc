#!/usr/bin/env python3
"""Diagnostic screening100 layer-5 context-pooling ablation summariser.

Summariser for the preregistered experiment
``configs/2026-10-04-layer5-context-pooling-ablation-screening100.preregistered.json``.

Both Selector arms use the *same* frozen layer-5 L2 selector, K=96,
``region_grid=3``, ``local_candidate_cap=16`` and the *same* selection metadata
(``selected_group_indices`` / ``source_type`` / ``region_id`` / ``valid``); they
differ only in the ``SOURCE_CONTEXT`` group values:

* ``pooled``         - the frozen layer-5 pooled artifact (``5395d5e2...``);
* ``representative`` - the ablation arm where each context slot uses the raw
  representative 2x2 group (produced later by
  ``evaluation/run_hf_hardware_regular_layer5_context_ablation.py``);
* ``dense``          - the frozen custom dense full2638 JSONL restricted to the
  100 ids; reported for **reference only** and never used by the decision.

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
   output must be K=96; the ``(image_grid_thw, full_patch_tokens,
   full_llm_visual_tokens)`` tuple must match between arms; and the selection
   metadata must be element-wise identical between the pooled baseline and the
   representative ablation arm.
4. The representative arm must declare ``context_group_values_source ==
   "representative_raw_group"`` and ``uses_pooled_context_group_values == false``
   with ``context_slots_replaced == context_group_count``.
5. Config-pinned input SHA256 values are re-verified and a missing required file
   or any failed check raises :class:`InputContractError`; no report is written
   (fail closed).

Decision: the 95% paired stratified bootstrap CI (``task::source_dataset``,
seed 20261004, 5000 repeats) of ``representative_minus_pooled`` on
``score(...)["cv_bench_accuracy"]``.  If the CI lower bound is strictly greater
than ``0.02`` the outcome is ``CONTEXT_POOLING_HARM_CONFIRMED``, otherwise
``CONTEXT_POOLING_HARM_NOT_CONFIRMED``.  Dense never influences the decision.
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
    / "2026-10-04-layer5-context-pooling-ablation-screening100.preregistered.json"
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
POOLED_LAYER5_L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-l2-diagnostic"
    / "layer5_l2_k96_screening100.jsonl"
)
REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-context-pooling-ablation"
    / "layer5_context_ablation_k96_screening100.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-layer5-context-pooling-ablation-screening100.json"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-layer5-context-pooling-ablation-screening100.md"
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
CONTEXT_POOLING_CI_LOWER = 0.02
SCORE_MODE = "l2"
PRUNE_AFTER_LAYER = 5

CONTEXT_GROUP_VALUES_SOURCE = "representative_raw_group"
CONTEXT_POOLING_HARM_CONFIRMED = "CONTEXT_POOLING_HARM_CONFIRMED"
CONTEXT_POOLING_HARM_NOT_CONFIRMED = "CONTEXT_POOLING_HARM_NOT_CONFIRMED"
DECISION_ORDER = (
    CONTEXT_POOLING_HARM_CONFIRMED,
    CONTEXT_POOLING_HARM_NOT_CONFIRMED,
)
DECISION_COMPARISON = "representative_minus_pooled"

DIAGNOSTIC_SCOPE = (
    "diagnostic screening100; NOT a formal full2638 CV-Bench conclusion"
)
GRID_SIGNATURE_DEFINITION = (
    "(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)"
)
DENSE_ROLE = "reference_only"

#: Metadata arrays that must be element-wise identical across the two Selector arms.
SELECTION_METADATA_FIELDS = (
    "selected_group_indices",
    "source_type",
    "region_id",
    "valid",
)
#: Scalar metadata that must also match.
SELECTION_COUNT_FIELDS = ("context_group_count", "real_group_count")

#: Logical input names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = (
    "screening_manifest",
    "full_manifest",
    "dense_full_jsonl",
    "pooled_layer5_l2_screening_jsonl",
    "representative_layer5_l2_screening_jsonl",
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the preregistered contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    pooled_layer5_l2_screening_jsonl: Path = POOLED_LAYER5_L2_SCREENING_JSONL
    representative_layer5_l2_screening_jsonl: Path = (
        REPRESENTATIVE_LAYER5_L2_SCREENING_JSONL
    )


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


def metadata_signature(row: Mapping[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Selection metadata that must be identical between the pooled and ablation arms."""

    signature: dict[str, tuple[Any, ...]] = {}
    for field in SELECTION_METADATA_FIELDS:
        if field not in row:
            raise InputContractError(f"record is missing selection metadata '{field}'")
        signature[field] = tuple(row[field])
    for field in SELECTION_COUNT_FIELDS:
        if field not in row:
            raise InputContractError(f"record is missing selection metadata '{field}'")
        signature[field] = (int(row[field]),)
    return signature


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


def _check_ablation_declaration(
    index: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    """The representative arm must declare the single-variable context ablation."""

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
        if "context_slots_replaced" in row and "context_group_count" in row:
            if int(row["context_slots_replaced"]) != int(row["context_group_count"]):
                raise InputContractError(
                    f"{label} record {sid} context_slots_replaced "
                    f"{row['context_slots_replaced']} != context_group_count "
                    f"{row['context_group_count']}"
                )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_arms(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    pooled_index: Mapping[str, Mapping[str, Any]],
    representative_index: Mapping[str, Mapping[str, Any]],
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

    for label, index in (("pooled", pooled_index), ("representative", representative_index)):
        index_set = set(index)
        missing = sorted(screening_set - index_set)
        extra = sorted(index_set - screening_set)
        if missing or extra:
            raise InputContractError(
                f"{label} screening100 sample_id set does not equal the screening "
                f"manifest: missing={missing[:10]} extra={extra[:10]}"
            )

    _check_common_arm_fields(pooled_index, "pooled")
    _check_common_arm_fields(representative_index, "representative")
    _check_ablation_declaration(representative_index, "representative")

    grid_mismatches: list[dict[str, Any]] = []
    metadata_mismatches: list[dict[str, Any]] = []
    for sid in screening_ids:
        pooled_row = pooled_index[sid]
        representative_row = representative_index[sid]
        pooled_sig = grid_signature(pooled_row)
        representative_sig = grid_signature(representative_row)
        if representative_sig != pooled_sig:
            grid_mismatches.append(
                {
                    "sample_id": sid,
                    "representative": list(representative_sig[0])
                    + [representative_sig[1], representative_sig[2]],
                    "pooled": list(pooled_sig[0]) + [pooled_sig[1], pooled_sig[2]],
                }
            )
        pooled_meta = metadata_signature(pooled_row)
        representative_meta = metadata_signature(representative_row)
        if pooled_meta != representative_meta:
            differing = sorted(
                key
                for key in pooled_meta
                if pooled_meta[key] != representative_meta[key]
            )
            metadata_mismatches.append(
                {"sample_id": sid, "fields": differing}
            )
    if grid_mismatches:
        raise InputContractError(
            "representative input grid is not consistent with the pooled baseline for "
            f"{len(grid_mismatches)} samples; first: {grid_mismatches[0]}"
        )
    if metadata_mismatches:
        raise InputContractError(
            "representative selection metadata/indices/positions differ from the "
            f"frozen pooled baseline for {len(metadata_mismatches)} samples; "
            f"first: {metadata_mismatches[0]}"
        )

    return {
        "screening_sample_count": len(screening_ids),
        "dense_full_sample_count": len(dense_index),
        "pooled_sample_count": len(pooled_index),
        "representative_sample_count": len(representative_index),
        "sample_ids_identical": True,
        "duplicate_sample_ids": {
            "screening": [],
            "dense": [],
            "pooled": [],
            "representative": [],
        },
        "all_score_mode_l2": True,
        "all_requested_budget_96": True,
        "all_prune_after_layer_5": True,
        "selection_metadata_fields": list(SELECTION_METADATA_FIELDS),
        "selection_metadata_identical": True,
        "patch_positions_equivalence": (
            "representative patch positions are group_index*4+arange(4), so "
            "identical selected_group_indices imply identical patch positions"
        ),
        "input_grid_consistent_pooled_vs_representative": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "ablation_declaration_ok": True,
        "context_group_values_source": CONTEXT_GROUP_VALUES_SOURCE,
        "checks": [
            "100 identical unique sample_ids across manifest/dense/pooled/representative",
            "no duplicate sample_id in any input",
            'every pooled and representative record score_mode == "l2"',
            f"every pooled and representative record requested_budget == {BUDGET}",
            f"every pooled and representative record prune_after_layer == {PRUNE_AFTER_LAYER}",
            "representative input grid == pooled input grid for every sample",
            "representative selection metadata/indices/positions == pooled for every sample",
            "representative arm declares the raw-representative context ablation",
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
        "pooled_layer5_l2_screening_jsonl": paths.pooled_layer5_l2_screening_jsonl,
        "representative_layer5_l2_screening_jsonl": (
            paths.representative_layer5_l2_screening_jsonl
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


def decide(representative_minus_pooled_ci: Sequence[float]) -> dict[str, Any]:
    """Apply the preregistered context-pooling decision (primary metric only)."""

    lower, upper = (
        float(representative_minus_pooled_ci[0]),
        float(representative_minus_pooled_ci[1]),
    )
    if lower > CONTEXT_POOLING_CI_LOWER:
        outcome = CONTEXT_POOLING_HARM_CONFIRMED
        reason = (
            f"representative_minus_pooled CI lower {lower:.6f} > "
            f"{CONTEXT_POOLING_CI_LOWER}"
        )
    else:
        outcome = CONTEXT_POOLING_HARM_NOT_CONFIRMED
        reason = (
            f"representative_minus_pooled CI lower {lower:.6f} <= "
            f"{CONTEXT_POOLING_CI_LOWER}"
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "metric": PRIMARY_METRIC,
        "representative_minus_pooled_ci95": [lower, upper],
        "threshold": CONTEXT_POOLING_CI_LOWER,
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
    pooled_rows: Sequence[Mapping[str, Any]],
    representative_rows: Sequence[Mapping[str, Any]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate inputs and compute the diagnostic result (no file I/O)."""

    screening_index = index_unique(screening_rows, "screening_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    pooled_index = index_unique(pooled_rows, "pooled_layer5_l2_screening100")
    representative_index = index_unique(
        representative_rows, "representative_layer5_l2_screening100"
    )

    validation = validate_arms(
        screening_index=screening_index,
        dense_index=dense_index,
        pooled_index=pooled_index,
        representative_index=representative_index,
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    joined = build_joined(
        list(screening_index),
        screening_index,
        {
            "dense": dense_index,
            "pooled": pooled_index,
            "representative": representative_index,
        },
    )

    metrics = {
        "dense": arm_metrics(joined, "dense", "dense"),
        "pooled": arm_metrics(joined, "pooled", "layer5_pooled_block5"),
        "representative": arm_metrics(
            joined, "representative", "layer5_representative_block5"
        ),
    }

    dense_primary = metrics["dense"]["cv_bench_accuracy"]
    if dense_primary == 0:
        raise InputContractError(
            "dense cv_bench_accuracy is 0; relative retention undefined"
        )
    relative = {
        "pooled": metrics["pooled"]["cv_bench_accuracy"] / dense_primary,
        "representative": metrics["representative"]["cv_bench_accuracy"] / dense_primary,
        "definition": "arm cv_bench_accuracy / dense cv_bench_accuracy (reference only)",
        "dense_role": DENSE_ROLE,
    }

    comparisons = {
        "representative_minus_pooled": paired_comparison(
            joined, "representative", "pooled"
        ),
        "representative_minus_dense": paired_comparison(
            joined, "representative", "dense"
        ),
        "pooled_minus_dense": paired_comparison(joined, "pooled", "dense"),
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
        "pooled_layer5_l2_screening_jsonl": {
            "path": str(paths.pooled_layer5_l2_screening_jsonl),
            "sha256": sha256_file(paths.pooled_layer5_l2_screening_jsonl),
        },
        "representative_layer5_l2_screening_jsonl": {
            "path": str(paths.representative_layer5_l2_screening_jsonl),
            "sha256": sha256_file(paths.representative_layer5_l2_screening_jsonl),
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
        pooled_rows=load_rows(
            paths.pooled_layer5_l2_screening_jsonl,
            "pooled_layer5_l2_screening100",
        ),
        representative_rows=load_rows(
            paths.representative_layer5_l2_screening_jsonl,
            "representative_layer5_l2_screening100",
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
            "Fixed stratified 100-sample screening: a single-variable context-pooling "
            "attribution signal only, NOT a formal full2638 CV-Bench conclusion. It must "
            "not be quoted as the official CV-Bench accuracy of any approach."
        ),
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
            "budget_K": BUDGET,
            "cut_point": PRUNE_AFTER_LAYER,
            "region_grid": 3,
            "local_candidate_cap": 16,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "decoding": "greedy_argmax",
            "do_sample": False,
            "score_mode": SCORE_MODE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "b1_trained_wv_used": False,
            "ablation": "SOURCE_CONTEXT pooled group -> raw representative 2x2 group",
            "context_group_values_source": CONTEXT_GROUP_VALUES_SOURCE,
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
            "pooled_layer5_l2": {
                "path": str(paths.pooled_layer5_l2_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "uses_pooled_context_group_values": True,
            },
            "representative_layer5_l2": {
                "path": str(paths.representative_layer5_l2_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "context_group_values_source": CONTEXT_GROUP_VALUES_SOURCE,
                "uses_pooled_context_group_values": False,
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
            "The single changed variable is the SOURCE_CONTEXT group value; all selection metadata, indices and positions are identical.",
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
    lines.append("# Layer-5 context-pooling ablation（screening100 诊断）")
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
    lines.append(f"- 消融：{cfg['ablation']}")
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- screening 样本数：{v['screening_sample_count']}")
    lines.append(f"- pooled layer5 样本数：{v['pooled_sample_count']}")
    lines.append(f"- representative layer5 样本数：{v['representative_sample_count']}")
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
        f"{'✅' if v['input_grid_consistent_pooled_vs_representative'] else '❌'}"
    )
    lines.append(
        "- selection metadata/indices/positions 两臂逐元素一致（"
        + ", ".join(v["selection_metadata_fields"])
        + f"）：{'✅' if v['selection_metadata_identical'] else '❌'}"
    )
    lines.append(
        f"- representative 臂声明 context_group_values_source="
        f"`{v['context_group_values_source']}`："
        f"{'✅' if v['ablation_declaration_ok'] else '❌'}"
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
    for arm in ("dense", "pooled", "representative"):
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
        "representative_minus_pooled",
        "representative_minus_dense",
        "pooled_minus_dense",
    ):
        c = report["comparisons"][name]
        lines.append(
            f"| {c['left']} − {c['right']} | {c['delta_point_estimate']:+.6f} | "
            f"{c['delta_ci95'][0]:+.6f} | {c['delta_ci95'][1]:+.6f} |"
        )
    lines.append("")
    lines.append("## 5. 预注册决策（仅用 representative − pooled 主指标）")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 结论：**{d['outcome']}**")
    lines.append(f"- 依据：{d['reason']}")
    lines.append(
        f"- representative−pooled 95% CI：["
        f"{d['representative_minus_pooled_ci95'][0]:+.6f}, "
        f"{d['representative_minus_pooled_ci95'][1]:+.6f}]；阈值 > {d['threshold']}"
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
