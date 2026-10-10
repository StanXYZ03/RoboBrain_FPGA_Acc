#!/usr/bin/env python3
"""CPU summariser: layer-5 adaptive-budget vs fixed-K96 unbounded vs dense.

Summariser for the preregistered screening100 diagnostic
``configs/2026-10-04-layer5-adaptive-budget-screening100.preregistered.json``.

Arms
----
* ``adaptive`` - the layer-5 **adaptive hardware-rule budget** real-only arm
  produced by ``evaluation/run_hf_hardware_regular_layer5_adaptive_budget.py``:
  ``K_raw = max(96, ceil(0.25 * G))``, ``K_eff = min(G, 16 * ceil(K_raw / 16))``
  and the ``G <= K_eff`` all-real identity bypass over the frozen unbounded
  real-only global-L2 + 3x3 anchor algorithm.
* ``fixed_k96`` - the existing fixed-K=96 unbounded real-only artifact
  (``layer5_unbounded_real_only_k96_screening100.jsonl``).
* ``dense`` - the frozen custom dense full2638 JSONL restricted to the 100 ids;
  **reference only**, never decision-driving.

Hard rules enforced here (fail closed):
1. The **only** primary metric is ``score(...)["cv_bench_accuracy"]`` from
   :mod:`evaluation.build_cv_bench_pruning_report`.  The plain per-sample mean
   is exposed strictly as ``unweighted_sample_accuracy``.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Exactly the same 100 unique ``sample_id`` values across the manifest, dense
   subset, fixed-K96 arm and adaptive arm, with no duplicates.
4. Every record is ``score_mode == "l2"`` and ``prune_after_layer == 5``; both
   Selector arms are all-real (``source_type`` all ``0``),
   ``context_group_count == 0``, with no context pooling / slot reservation /
   local candidate cap, and the frozen
   ``selection_rule == "unbounded_real_only_global_l2_3x3_anchor"``.
5. The adaptive arm's per-record ``raw_budget`` / ``effective_budget`` /
   ``requested_budget`` / ``adaptive_retention`` / ``identity_bypass`` are
   re-derived from the real ``full_llm_visual_tokens`` and must match exactly;
   the fixed-K96 arm keeps ``requested_budget == 96`` and emits exactly 96
   groups (or the ``G <= 96`` identity bypass).
6. The adaptive arm's per-sample input grid tuple equals the fixed-K96 arm's;
   config-pinned input SHA256 values are re-verified.  A missing file or any
   failed check raises :class:`InputContractError` and writes no report.

The paired stratified bootstrap CI (``task::source_dataset``, seed 20261004,
5000 repeats) is reported on the official primary metric only.  The
``adaptive/dense`` ratio is emitted under ``project_interpretation`` and is
explicitly **not** an official CV-Bench metric.
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
from evaluation.layer5_adaptive_budget import (  # noqa: E402
    BUDGET_ALIGNMENT,
    BUDGET_FORMULA,
    MIN_BUDGET,
    verify_adaptive_budget_fields,
)
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-04-layer5-adaptive-budget-screening100.preregistered.json"
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
FIXED_K96_UNBOUNDED_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-unbounded-real-only"
    / "layer5_unbounded_real_only_k96_screening100.jsonl"
)
ADAPTIVE_BUDGET_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-adaptive-budget"
    / "layer5_adaptive_budget_screening100.jsonl"
)
REPORT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-layer5-adaptive-budget-screening100.json"
)
REPORT_MD = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-layer5-adaptive-budget-screening100.md"
)

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = "unweighted_sample_accuracy"
SECONDARY_METRIC = 'score(...)["accuracy"]'
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

BUDGET_K96 = 96
SCREENING_SAMPLE_COUNT = 100
FULL_SAMPLE_COUNT = 2638
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
SCORE_MODE = "l2"
PRUNE_AFTER_LAYER = 5
REGION_GRID = 3

SELECTION_RULE = "unbounded_real_only_global_l2_3x3_anchor"
FIXED_K96_ARM_LABEL = "layer5_unbounded_real_only_block5"
ADAPTIVE_ARM_LABEL = "layer5_adaptive_budget_unbounded_real_only_block5"
BUDGET_MODE = "adaptive_layer5_unbounded_real_only"
SOURCE_REAL = 0

DIAGNOSTIC_SCOPE = (
    "diagnostic screening100; NOT a formal full2638 CV-Bench conclusion"
)
GRID_SIGNATURE_DEFINITION = (
    "(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)"
)
DENSE_ROLE = "reference_only"
PROJECT_RATIO_ROLE = "project_interpretation_not_official_metric"

DECISION_TYPE = "descriptive_only"
DECISION_OUTCOME = "DIAGNOSTIC_NO_PROMOTION_THRESHOLD"

#: Logical input names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = (
    "screening_manifest",
    "full_manifest",
    "dense_full_jsonl",
    "fixed_k96_unbounded_screening_jsonl",
    "adaptive_budget_screening_jsonl",
)

COMPARISONS = (
    "adaptive_minus_fixed_k96",
    "adaptive_minus_dense",
    "fixed_k96_minus_dense",
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    fixed_k96_unbounded_screening_jsonl: Path = FIXED_K96_UNBOUNDED_SCREENING_JSONL
    adaptive_budget_screening_jsonl: Path = ADAPTIVE_BUDGET_SCREENING_JSONL


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
    for field in ("image_grid_thw", "full_patch_tokens", "full_llm_visual_tokens"):
        if field not in row:
            raise InputContractError(f"record is missing {field}")
    return (
        tuple(int(value) for value in row["image_grid_thw"]),
        int(row["full_patch_tokens"]),
        int(row["full_llm_visual_tokens"]),
    )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def _check_common_fields(index: Mapping[str, Mapping[str, Any]], label: str) -> None:
    """Every Selector arm record must be L2, layer-5 and structurally complete."""

    bad_score_mode = sorted(
        sid for sid, row in index.items() if str(row.get("score_mode")) != SCORE_MODE
    )
    if bad_score_mode:
        raise InputContractError(
            f"{label} records must all have score_mode == '{SCORE_MODE}'; "
            f"offenders: {bad_score_mode[:10]}"
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


def _check_all_real_no_context(
    index: Mapping[str, Mapping[str, Any]], label: str
) -> None:
    """Both Selector arms must prove all-real / context-free selection."""

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
            raise InputContractError(f"{label} record {sid} is missing source_type")
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


def _check_fixed_k96(index: Mapping[str, Mapping[str, Any]], label: str) -> None:
    """Fixed-K96 arm: requested_budget == 96 and exactly K=96 or identity."""

    for sid, row in index.items():
        if "requested_budget" not in row or int(row["requested_budget"]) != BUDGET_K96:
            raise InputContractError(
                f"{label} record {sid} must have requested_budget == {BUDGET_K96}, "
                f"got {row.get('requested_budget')!r}"
            )
        identity = bool(row.get("identity_bypass"))
        if identity:
            if "full_llm_visual_tokens" not in row:
                raise InputContractError(
                    f"{label} record {sid} identity bypass is missing "
                    "full_llm_visual_tokens"
                )
            if int(row["full_llm_visual_tokens"]) > BUDGET_K96:
                raise InputContractError(
                    f"{label} record {sid} claims identity bypass with G="
                    f"{row['full_llm_visual_tokens']} > {BUDGET_K96}"
                )
            expected = int(row["full_llm_visual_tokens"])
        else:
            expected = BUDGET_K96
        if int(row["output_group_count"]) != expected:
            raise InputContractError(
                f"{label} record {sid} output_group_count "
                f"{row['output_group_count']} != expected {expected}"
            )


def _check_adaptive(index: Mapping[str, Mapping[str, Any]], label: str) -> None:
    """Adaptive arm: re-derive the frozen budget formula from the real G."""

    for sid, row in index.items():
        if str(row.get("budget_mode")) != BUDGET_MODE:
            raise InputContractError(
                f"{label} record {sid} must declare budget_mode == "
                f"'{BUDGET_MODE}', got {row.get('budget_mode')!r}"
            )
        if str(row.get("budget_formula")) != BUDGET_FORMULA:
            raise InputContractError(
                f"{label} record {sid} budget_formula does not match the frozen formula"
            )
        if int(row.get("budget_min", -1)) != MIN_BUDGET:
            raise InputContractError(
                f"{label} record {sid} budget_min != {MIN_BUDGET}"
            )
        if int(row.get("budget_alignment", -1)) != BUDGET_ALIGNMENT:
            raise InputContractError(
                f"{label} record {sid} budget_alignment != {BUDGET_ALIGNMENT}"
            )
        try:
            resolved = verify_adaptive_budget_fields(row)
        except (TypeError, ValueError) as exc:
            raise InputContractError(
                f"{label} record {sid} violates the frozen budget formula: {exc}"
            ) from exc
        if "full_llm_visual_tokens" not in row:
            raise InputContractError(
                f"{label} record {sid} is missing full_llm_visual_tokens"
            )
        if int(row["full_llm_visual_tokens"]) != resolved.full_llm_visual_tokens:
            raise InputContractError(
                f"{label} record {sid} full_llm_visual_tokens mismatch"
            )
        expected = (
            resolved.full_llm_visual_tokens
            if resolved.identity_bypass
            else resolved.effective_budget
        )
        if int(row["output_group_count"]) != expected:
            raise InputContractError(
                f"{label} record {sid} output_group_count "
                f"{row['output_group_count']} != formula-expected {expected}"
            )
        if bool(row.get("identity_bypass")) != resolved.identity_bypass:
            raise InputContractError(
                f"{label} record {sid} identity_bypass disagrees with the formula"
            )


def validate_arms(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    fixed_k96_index: Mapping[str, Mapping[str, Any]],
    adaptive_index: Mapping[str, Mapping[str, Any]],
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
        ("fixed_k96", fixed_k96_index),
        ("adaptive", adaptive_index),
    ):
        index_set = set(index)
        missing = sorted(screening_set - index_set)
        extra = sorted(index_set - screening_set)
        if missing or extra:
            raise InputContractError(
                f"{label} screening100 sample_id set does not equal the screening "
                f"manifest: missing={missing[:10]} extra={extra[:10]}"
            )

    _check_common_fields(fixed_k96_index, "fixed_k96")
    _check_common_fields(adaptive_index, "adaptive")
    _check_all_real_no_context(fixed_k96_index, "fixed_k96")
    _check_all_real_no_context(adaptive_index, "adaptive")
    _check_fixed_k96(fixed_k96_index, "fixed_k96")
    _check_adaptive(adaptive_index, "adaptive")

    grid_mismatches: list[dict[str, Any]] = []
    for sid in screening_ids:
        fixed_sig = grid_signature(fixed_k96_index[sid])
        adaptive_sig = grid_signature(adaptive_index[sid])
        if adaptive_sig != fixed_sig:
            grid_mismatches.append(
                {
                    "sample_id": sid,
                    "adaptive": list(adaptive_sig[0])
                    + [adaptive_sig[1], adaptive_sig[2]],
                    "fixed_k96": list(fixed_sig[0]) + [fixed_sig[1], fixed_sig[2]],
                }
            )
    if grid_mismatches:
        raise InputContractError(
            "adaptive input grid is not consistent with the fixed-K96 arm for "
            f"{len(grid_mismatches)} samples; first: {grid_mismatches[0]}"
        )

    return {
        "screening_sample_count": len(screening_ids),
        "dense_full_sample_count": len(dense_index),
        "fixed_k96_sample_count": len(fixed_k96_index),
        "adaptive_sample_count": len(adaptive_index),
        "sample_ids_identical": True,
        "all_score_mode_l2": True,
        "all_prune_after_layer_5": True,
        "input_grid_consistent_fixed_k96_vs_adaptive": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "fixed_k96_declaration_ok": True,
        "adaptive_declaration_ok": True,
        "adaptive_budget_formula_ok": True,
        "adaptive_identity_or_exact_budget": True,
        "fixed_k96_identity_or_exact_96": True,
        "selection_rule": SELECTION_RULE,
        "all_real": True,
        "no_context": True,
        "no_context_slot_reservation": True,
        "no_local_candidate_cap": True,
        "selection_metadata_identity_required": False,
        "checks": [
            "100 identical unique sample_ids across manifest/dense/fixed_k96/adaptive",
            "no duplicate sample_id in any input",
            'every fixed_k96 and adaptive record score_mode == "l2"',
            f"every fixed_k96 and adaptive record prune_after_layer == {PRUNE_AFTER_LAYER}",
            f"every fixed_k96 record requested_budget == {BUDGET_K96} and emits "
            "exactly 96 groups or the G<=96 identity bypass",
            "every adaptive record re-derives K_raw/K_eff/retention/identity from "
            "the real full_llm_visual_tokens and emits exactly K_eff or the identity groups",
            "adaptive input grid == fixed_k96 input grid for every sample",
            "both arms prove all-real (source_type all 0) and context-free "
            "(context_group_count == 0, no pooling / slot reservation / local cap)",
            "the frozen selection_rule unbounded_real_only_global_l2_3x3_anchor is preserved",
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
        "fixed_k96_unbounded_screening_jsonl": (
            paths.fixed_k96_unbounded_screening_jsonl
        ),
        "adaptive_budget_screening_jsonl": paths.adaptive_budget_screening_jsonl,
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
# Official scoring, paired bootstrap and project ratio
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
            # Persist the frozen CV-Bench answer label form ``(X)`` (or ``None``)
            # exactly as the official scorer expects to read it back.
            record[f"{arm}_answer"] = f"({parsed})" if parsed is not None else None
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


def project_interpretation(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Adaptive/dense and adaptive/fixed-K96 ratios as project interpretation only."""

    dense = float(metrics["dense"]["cv_bench_accuracy"])
    if dense == 0:
        raise InputContractError(
            "dense cv_bench_accuracy is 0; project ratio undefined"
        )
    adaptive = float(metrics["adaptive"]["cv_bench_accuracy"])
    fixed_k96 = float(metrics["fixed_k96"]["cv_bench_accuracy"])
    return {
        "role": PROJECT_RATIO_ROLE,
        "is_official_cv_bench_metric": False,
        "definition": (
            "arm cv_bench_accuracy / dense cv_bench_accuracy; a project "
            "interpretation ratio, NOT an official CV-Bench metric"
        ),
        "adaptive_over_dense_ratio": adaptive / dense,
        "fixed_k96_over_dense_ratio": fixed_k96 / dense,
        "adaptive_over_fixed_k96_ratio": (
            adaptive / fixed_k96 if fixed_k96 else None
        ),
        "dense_role": DENSE_ROLE,
        "primary_metric": PRIMARY_METRIC,
    }


def describe_decision() -> dict[str, Any]:
    """No promotion threshold is preregistered for this screening100 diagnostic."""

    return {
        "type": DECISION_TYPE,
        "outcome": DECISION_OUTCOME,
        "metric": PRIMARY_METRIC,
        "uses_only_primary_metric": True,
        "dense_drives_decision": False,
        "dense_role": DENSE_ROLE,
        "note": (
            "screening100 diagnostic only: the paired stratified bootstrap CI on "
            "score(...)[\"cv_bench_accuracy\"] is reported descriptively; no "
            "CONFIRMED/NOT_CONFIRMED promotion claim is made."
        ),
    }


def analyze(
    *,
    screening_rows: Sequence[Mapping[str, Any]],
    dense_full_rows: Sequence[Mapping[str, Any]],
    fixed_k96_rows: Sequence[Mapping[str, Any]],
    adaptive_rows: Sequence[Mapping[str, Any]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate inputs and compute the diagnostic result (no file I/O)."""

    screening_index = index_unique(screening_rows, "screening_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    fixed_k96_index = index_unique(
        fixed_k96_rows, "fixed_k96_unbounded_screening100"
    )
    adaptive_index = index_unique(
        adaptive_rows, "adaptive_budget_screening100"
    )

    validation = validate_arms(
        screening_index=screening_index,
        dense_index=dense_index,
        fixed_k96_index=fixed_k96_index,
        adaptive_index=adaptive_index,
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    joined = build_joined(
        list(screening_index),
        screening_index,
        {
            "dense": dense_index,
            "fixed_k96": fixed_k96_index,
            "adaptive": adaptive_index,
        },
    )

    metrics = {
        "dense": arm_metrics(joined, "dense", "dense"),
        "fixed_k96": arm_metrics(joined, "fixed_k96", FIXED_K96_ARM_LABEL),
        "adaptive": arm_metrics(joined, "adaptive", ADAPTIVE_ARM_LABEL),
    }

    comparisons = {
        name: paired_comparison(joined, *name.split("_minus_"))
        for name in COMPARISONS
    }

    # Sanity: the adaptive budget never exceeds the fixed K=96 budget.
    over_budget = sorted(
        sid
        for sid, row in adaptive_index.items()
        if int(row["effective_budget"]) > BUDGET_K96
    )

    return {
        "validation": validation,
        "joined": joined,
        "metrics": metrics,
        "project_interpretation": project_interpretation(metrics),
        "comparisons": comparisons,
        "decision": describe_decision(),
        "adaptive_budget_overview": _budget_overview(adaptive_index),
        "adaptive_records_exceeding_fixed_k96": over_budget,
    }


def _budget_overview(
    index: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    effective = [int(row["effective_budget"]) for row in index.values()]
    raw = [int(row["raw_budget"]) for row in index.values()]
    retention = [float(row["adaptive_retention"]) for row in index.values()]
    identity = sorted(
        sid for sid, row in index.items() if bool(row["identity_bypass"])
    )
    return {
        "sample_count": len(index),
        "raw_budget_min": min(raw),
        "raw_budget_max": max(raw),
        "effective_budget_min": min(effective),
        "effective_budget_max": max(effective),
        "effective_budget_mean": sum(effective) / len(effective),
        "retention_mean": sum(retention) / len(retention),
        "identity_bypass_sample_ids": identity,
        "identity_bypass_count": len(identity),
        "fixed_k96_budget": BUDGET_K96,
        "adaptive_never_exceeds_fixed_k96": all(
            value <= BUDGET_K96 for value in effective
        ),
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
        "fixed_k96_unbounded_screening_jsonl": {
            "path": str(paths.fixed_k96_unbounded_screening_jsonl),
            "sha256": sha256_file(paths.fixed_k96_unbounded_screening_jsonl),
        },
        "adaptive_budget_screening_jsonl": {
            "path": str(paths.adaptive_budget_screening_jsonl),
            "sha256": sha256_file(paths.adaptive_budget_screening_jsonl),
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
        fixed_k96_rows=load_rows(
            paths.fixed_k96_unbounded_screening_jsonl,
            "fixed_k96_unbounded_screening100",
        ),
        adaptive_rows=load_rows(
            paths.adaptive_budget_screening_jsonl,
            "adaptive_budget_screening100",
        ),
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": True,
        "formal_full2638_conclusion": False,
        "explicit_non_formal_statement": (
            "Fixed stratified 100-sample screening: a diagnostic adaptive-budget "
            "signal only, NOT a formal full2638 CV-Bench conclusion. It must not "
            "be quoted as the official CV-Bench accuracy of any approach."
        ),
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
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
            "budget_formula": BUDGET_FORMULA,
            "budget_min": MIN_BUDGET,
            "budget_retention_ratio": 0.25,
            "budget_alignment": BUDGET_ALIGNMENT,
            "fixed_k96_budget": BUDGET_K96,
            "selection_rule": SELECTION_RULE,
            "uses_context_pooling": False,
            "uses_context_slot_reservation": False,
            "uses_local_candidate_cap": False,
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
            "fixed_k96": {
                "path": str(paths.fixed_k96_unbounded_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "requested_budget": BUDGET_K96,
                "selection_rule": SELECTION_RULE,
            },
            "adaptive": {
                "path": str(paths.adaptive_budget_screening_jsonl),
                "prune_after_layer": PRUNE_AFTER_LAYER,
                "score_mode": SCORE_MODE,
                "budget_mode": BUDGET_MODE,
                "budget_formula": BUDGET_FORMULA,
                "selection_rule": SELECTION_RULE,
            },
        },
        "validation": analysis["validation"],
        "config_pin_verification": pins,
        "adaptive_budget_overview": analysis["adaptive_budget_overview"],
        "metrics": analysis["metrics"],
        "project_interpretation": analysis["project_interpretation"],
        "comparisons": analysis["comparisons"],
        "decision": analysis["decision"],
        "inputs_sha256": _inputs_sha256(paths),
        "limitations": [
            "Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.",
            "The adaptive budget is derived from the measured real G; it is a "
            "software selection diagnostic and makes no FPGA/latency/acceleration claim.",
            "Both Selector arms use the same frozen unbounded real-only global-L2 + "
            "3x3 anchor algorithm and raw real groups; only the per-sample budget differs.",
            "Dense is the frozen custom dense path and is reference only; it never "
            "drives the decision.",
            "Only score(...)[\"cv_bench_accuracy\"] is reported as the primary metric; "
            "the adaptive/dense ratio is project interpretation, not an official "
            "CV-Bench metric.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are "
            "never reported.",
            "Neither arm uses the B1-trained Wv projection.",
        ],
    }


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Layer-5 adaptive-budget unbounded real-only（screening100 诊断）")
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
    lines.append(
        "- dense 仅为参考（reference only）；adaptive/dense 比例为项目解释量，"
        "**不是 CV-Bench 官方指标**。"
    )
    lines.append("")
    lines.append("## 1. 固定配置与预算公式")
    lines.append("")
    cfg = report["fixed_configuration"]
    lines.append(
        f"- 样本：`{cfg['manifest']}`；切点：{cfg['cut_point']}；"
        f"region_grid={cfg['region_grid']}；local_cap={cfg['local_candidate_cap']}"
    )
    lines.append(
        f"- dtype={cfg['dtype']}；attention={cfg['attention_implementation']}；"
        f"decoding={cfg['decoding']}；score_mode={cfg['score_mode']}；"
        f"bootstrap seed={cfg['bootstrap_seed']} / repeats={cfg['bootstrap_repeats']}"
    )
    lines.append(f"- 预算公式：`{cfg['budget_formula']}`")
    lines.append(
        f"- budget_min={cfg['budget_min']}；retention={cfg['budget_retention_ratio']}；"
        f"alignment={cfg['budget_alignment']}；fixed_k96={cfg['fixed_k96_budget']}"
    )
    lines.append(
        f"- selection_rule=`{cfg['selection_rule']}`；无 context pooling / "
        "无 context slot reservation / 无 local_candidate_cap"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- screening 样本数：{v['screening_sample_count']}")
    lines.append(f"- dense 全量样本数：{v['dense_full_sample_count']}")
    lines.append(f"- fixed-K96 unbounded 样本数：{v['fixed_k96_sample_count']}")
    lines.append(f"- adaptive 样本数：{v['adaptive_sample_count']}")
    lines.append(f"- 100 个 sample_id 完全一致：{'✅' if v['sample_ids_identical'] else '❌'}")
    lines.append(
        f"- score_mode 全为 `{SCORE_MODE}`：{'✅' if v['all_score_mode_l2'] else '❌'}"
    )
    lines.append(
        f"- prune_after_layer 全为 {PRUNE_AFTER_LAYER}："
        f"{'✅' if v['all_prune_after_layer_5'] else '❌'}"
    )
    lines.append(
        "- adaptive 预算公式（K_raw/K_eff/retention/identity）逐样本复核："
        f"{'✅' if v['adaptive_budget_formula_ok'] else '❌'}"
    )
    lines.append(
        "- fixed-K96 精确 K=96 或 G<=96 identity："
        f"{'✅' if v['fixed_k96_identity_or_exact_96'] else '❌'}"
    )
    lines.append(
        f"- adaptive 输入 grid 与 fixed-K96 一致（{v['grid_signature_definition']}）："
        f"{'✅' if v['input_grid_consistent_fixed_k96_vs_adaptive'] else '❌'}"
    )
    lines.append(
        "- 两臂 all-real / 无 context（pooling、slot reservation、local_candidate_cap）："
        f"{'✅' if (v['all_real'] and v['no_context'] and v['no_local_candidate_cap']) else '❌'}"
    )
    pins = report.get("config_pin_verification")
    if pins is not None:
        lines.append(
            f"- 输入 SHA 复核：{'✅' if pins['pins_verified'] else '❌'}"
            f"（{len(pins['checks'])} 项）"
        )
    lines.append("")
    lines.append("## 3. 自适应预算概览")
    lines.append("")
    b = report["adaptive_budget_overview"]
    lines.append(
        f"- K_raw 范围：{b['raw_budget_min']}–{b['raw_budget_max']}；"
        f"K_eff 范围：{b['effective_budget_min']}–{b['effective_budget_max']}；"
        f"K_eff 均值：{b['effective_budget_mean']:.3f}"
    )
    lines.append(
        f"- retention 均值：{b['retention_mean']:.6f}；identity bypass："
        f"{b['identity_bypass_count']} 个样本 {b['identity_bypass_sample_ids']}"
    )
    lines.append(
        f"- adaptive K_eff 不超过 fixed-K96={b['fixed_k96_budget']}："
        f"{'✅' if b['adaptive_never_exceeds_fixed_k96'] else '❌'}"
    )
    lines.append("")
    lines.append("## 4. 三臂正式主分")
    lines.append("")
    lines.append(
        "| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | "
        "unweighted_sample_accuracy（次要） |"
    )
    lines.append("|---|---:|---:|---:|")
    for arm in ("dense", "fixed_k96", "adaptive"):
        m = report["metrics"][arm]
        lines.append(
            f"| {m['label']} | {m['sample_count']} | **{_fmt(m['cv_bench_accuracy'])}** | "
            f"{_fmt(m['unweighted_sample_accuracy'])} |"
        )
    pi = report["project_interpretation"]
    lines.append("")
    lines.append(
        f"- 项目解释量（**非官方指标**）：adaptive/dense="
        f"{_fmt(pi['adaptive_over_dense_ratio'])}；fixed-K96/dense="
        f"{_fmt(pi['fixed_k96_over_dense_ratio'])}；adaptive/fixed-K96="
        f"{_fmt(pi['adaptive_over_fixed_k96_ratio'])}"
    )
    lines.append("")
    lines.append("## 5. 配对分层 bootstrap（95% CI，主指标）")
    lines.append("")
    lines.append("| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |")
    lines.append("|---|---:|---:|---:|")
    for name in COMPARISONS:
        c = report["comparisons"][name]
        lines.append(
            f"| {c['left']} − {c['right']} | {c['delta_point_estimate']:+.6f} | "
            f"{c['delta_ci95'][0]:+.6f} | {c['delta_ci95'][1]:+.6f} |"
        )
    lines.append("")
    lines.append("## 6. 诊断结论（仅主指标，dense 不参与）")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 类型：{d['type']}；结果：**{d['outcome']}**")
    lines.append(f"- 说明：{d['note']}")
    lines.append("")
    lines.append("## 7. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 8. 限制")
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
