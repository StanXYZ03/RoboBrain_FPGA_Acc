#!/usr/bin/env python3
"""Strict CPU-only, read-only diagnostic: adaptive layer-5 vs historical r25.

This module compares three arms over the **same** fixed 100-sample CV-Bench
screening set, using only the official scorer
``evaluation.build_cv_bench_pruning_report.score(...)["cv_bench_accuracy"]`` as
the primary metric:

* ``A`` adaptive  = ``artifacts/token_selector/2026-10-04-layer5-adaptive-budget/
  layer5_adaptive_budget_screening100.jsonl`` (layer-5 unbounded real-only
  global-L2 + 3x3 anchor selection, per-sample ``K_raw = max(96, ceil(0.25*G))``,
  ``K_eff = min(G, 16*ceil(K_raw/16))``);
* ``B`` fixed K96 = ``artifacts/token_selector/2026-10-04-layer5-unbounded-real-only/
  layer5_unbounded_real_only_k96_screening100.jsonl`` (same algorithm, K=96);
* ``C`` historical r25 = ``evaluation/results/cv_bench_l2_budget64_pilot100_20260927/
  shallow_r0250.jsonl`` (frozen historical shallow-block 25% retention path).

The diagnostic is deliberately fail-closed and read-only: it re-verifies the
sample-id contract, the task / ``prune_after_layer`` contract, the strict
ascending / unique / SHA-256-reproducible ``selected_group_indices`` contract and
the pinned input SHA-256 values before writing anything, and it re-checks that no
input changed while it ran.

It does **not** run any model, GPU, network, training or dependency install.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    parse_choice,
    score,
)
from evaluation.layer5_adaptive_budget import compute_adaptive_budget  # noqa: E402
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

SCREENING_MANIFEST = ROOT / "datasets" / "cv_bench_screening100.jsonl"
ADAPTIVE_ARM_PATH = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-adaptive-budget"
    / "layer5_adaptive_budget_screening100.jsonl"
)
FIXED_K96_ARM_PATH = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-unbounded-real-only"
    / "layer5_unbounded_real_only_k96_screening100.jsonl"
)
R25_ARM_PATH = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_budget64_pilot100_20260927"
    / "shallow_r0250.jsonl"
)

ADAPTIVE_PREREG_PATH = (
    ROOT / "configs" / "2026-10-04-layer5-adaptive-budget-screening100.preregistered.json"
)
UNBOUNDED_PREREG_PATH = (
    ROOT
    / "configs"
    / "2026-10-04-layer5-unbounded-vs-representative-screening100.preregistered.json"
)

REPORT_JSON = ROOT / "reports" / "cv_bench" / "2026-10-04-adaptive-vs-r25-screening100.json"
REPORT_CSV = ROOT / "reports" / "cv_bench" / "2026-10-04-adaptive-vs-r25-screening100.csv"
REPORT_MD = ROOT / "reports" / "cv_bench" / "2026-10-04-adaptive-vs-r25-screening100.md"

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
SECONDARY_METRIC = 'score(...)["accuracy"]'
UNWEIGHTED_NOTE = (
    "unweighted sample accuracy is SECONDARY diagnostic only and is never the "
    "official CV-Bench metric"
)

SAMPLE_COUNT = 100
PRUNE_AFTER_LAYER = 5
FIXED_K96 = 96
EXPECTED_TASKS = ("Count", "Depth", "Distance", "Relation")

ARM_A = "A_adaptive"
ARM_B = "B_fixed_k96"
ARM_C = "C_r25"
ARMS = (ARM_A, ARM_B, ARM_C)
ARM_DESCRIPTIONS = {
    ARM_A: "adaptive layer-5 budget, unbounded real-only global-L2 + 3x3 anchor",
    ARM_B: "fixed K=96, same unbounded real-only global-L2 + 3x3 anchor",
    ARM_C: "historical shallow-block r=0.25 (budget64 pilot) path",
}

#: Pinned SHA-256 of every raw input, re-verified fail-closed at run time.
DEFAULT_INPUT_PINS: dict[str, str] = {
    "screening_manifest": "31bf313286c81fa50911ca3822118184ac03ca74df2cf2c260250cab2768b50c",
    "adaptive_arm": "8f9442dd0f533e8a475fd51784682b729a64decb31f3166a7f32d72b4405d3e3",
    "fixed_k96_arm": "fd22c5a4b1941a8ad0a2c47bf15f0abcbeaa56553a9f9a790deb6e14054df94c",
    "r25_arm": "ce019edd1ee35d11c83a3ff7c11bd9db46539cf56745aadcf0780f15abdbf7d0",
    "adaptive_prereg": "55a39d9f1b2f0df264391e6e8c99198d8c4086221116bbbd2509d3e00e52025f",
    "unbounded_prereg": "33d46aa0fa5779bf586fa9262b656bf72caf48eb45f958028cd1ca70e6cff020",
}

G_BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("G<=96", lambda value: value <= 96),
    ("97-256", lambda value: 97 <= value <= 256),
    ("257-768", lambda value: 257 <= value <= 768),
    ("G>768", lambda value: value > 768),
)
KDIFF_BUCKETS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("0", lambda value: value == 0),
    ("1-15", lambda value: 1 <= value <= 15),
    ("16-31", lambda value: 16 <= value <= 31),
    ("32+", lambda value: value >= 32),
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the frozen contract."""


@dataclass(frozen=True)
class InputPaths:
    screening_manifest: Path = SCREENING_MANIFEST
    adaptive_arm: Path = ADAPTIVE_ARM_PATH
    fixed_k96_arm: Path = FIXED_K96_ARM_PATH
    r25_arm: Path = R25_ARM_PATH
    adaptive_prereg: Path = ADAPTIVE_PREREG_PATH
    unbounded_prereg: Path = UNBOUNDED_PREREG_PATH


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
            f"input '{label}' has duplicate sample_id values: {sorted(set(duplicates))[:10]}"
        )
    return index


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputContractError(f"{label} must be an int, got {value!r}")
    return value


def validate_selected_indices(row: Mapping[str, Any], label: str) -> list[int]:
    """Strictly ascending, unique indices plus a reproducible SHA-256 digest."""

    raw = row.get("selected_group_indices")
    if not isinstance(raw, list) or not raw:
        raise InputContractError(f"{label} must carry a non-empty selected_group_indices")
    indices = [_as_int(value, f"{label}.selected_group_indices") for value in raw]
    if any(value < 0 for value in indices):
        raise InputContractError(f"{label} selected_group_indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise InputContractError(f"{label} selected_group_indices has duplicates")
    if indices != sorted(indices):
        raise InputContractError(f"{label} selected_group_indices is not strictly ascending")
    recorded = row.get("selected_group_indices_sha256")
    recomputed = selection_sha256(indices)
    if not isinstance(recorded, str) or recorded != recomputed:
        raise InputContractError(
            f"{label} selected_group_indices_sha256 mismatch: recorded={recorded!r} "
            f"recomputed={recomputed}"
        )
    return indices


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


def verify_pins(
    paths: InputPaths,
    expected_pins: Mapping[str, str] | None,
) -> dict[str, dict[str, Any]]:
    resolved = {
        "screening_manifest": paths.screening_manifest,
        "adaptive_arm": paths.adaptive_arm,
        "fixed_k96_arm": paths.fixed_k96_arm,
        "r25_arm": paths.r25_arm,
        "adaptive_prereg": paths.adaptive_prereg,
        "unbounded_prereg": paths.unbounded_prereg,
    }
    pinned: dict[str, dict[str, Any]] = {}
    for name, path in resolved.items():
        require_file(path, name)
        digest = sha256_file(path)
        pinned[name] = {"path": str(path), "sha256": digest}
        if expected_pins is None:
            continue
        expected = expected_pins.get(name)
        if expected is not None and expected != digest:
            raise InputContractError(
                f"input '{name}' sha256 mismatch: expected {expected}, got {digest} "
                f"({path}) (fail closed)"
            )
    return pinned


def verify_existing_config_pins(paths: InputPaths) -> dict[str, Any]:
    """Re-verify every non-null ``inputs_sha256`` pin in the existing configs."""

    results: dict[str, Any] = {}
    for label, config_path in (
        ("adaptive_prereg", paths.adaptive_prereg),
        ("unbounded_prereg", paths.unbounded_prereg),
    ):
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        pins = payload.get("inputs_sha256", {})
        if not isinstance(pins, Mapping):
            raise InputContractError(f"{label} inputs_sha256 must be an object")
        checked = 0
        for name, spec in pins.items():
            if not isinstance(spec, Mapping):
                continue
            expected = spec.get("sha256")
            raw_path = spec.get("path")
            if expected is None or not raw_path:
                continue
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            if not candidate.is_file():
                raise InputContractError(
                    f"{label} pin '{name}' points at missing file {candidate} (fail closed)"
                )
            actual = sha256_file(candidate)
            if actual != expected:
                raise InputContractError(
                    f"{label} pin '{name}' sha256 mismatch: expected {expected}, got {actual}"
                )
            checked += 1
        results[label] = {"pins_checked": checked}
    return results


def derive_budget(row: Mapping[str, Any], arm: str, label: str) -> dict[str, Any]:
    indices = validate_selected_indices(row, label)
    k = len(indices)
    if arm == ARM_A:
        if row.get("budget_mode") != "adaptive_layer5_unbounded_real_only":
            raise InputContractError(
                f"{label} must set budget_mode='adaptive_layer5_unbounded_real_only'"
            )
        g = _as_int(row.get("full_llm_visual_tokens"), f"{label}.full_llm_visual_tokens")
        budget = compute_adaptive_budget(g)
        if k != budget.effective_budget:
            raise InputContractError(
                f"{label} K={k} != adaptive effective_budget={budget.effective_budget} (G={g})"
            )
        if int(row.get("effective_budget", -1)) != budget.effective_budget:
            raise InputContractError(f"{label} effective_budget field mismatch")
    elif arm == ARM_B:
        g = _as_int(row.get("full_llm_visual_tokens"), f"{label}.full_llm_visual_tokens")
        if int(row.get("requested_budget", -1)) != FIXED_K96:
            raise InputContractError(f"{label} requested_budget must be {FIXED_K96}")
        expected = min(g, FIXED_K96)
        if k != expected:
            raise InputContractError(f"{label} K={k} != min(G={g}, {FIXED_K96})={expected}")
    elif arm == ARM_C:
        g = _as_int(row.get("baseline_llm_visual_tokens"), f"{label}.baseline_llm_visual_tokens")
        k_field = _as_int(row.get("llm_visual_tokens"), f"{label}.llm_visual_tokens")
        if k_field != k:
            raise InputContractError(f"{label} llm_visual_tokens={k_field} != K={k}")
        retention = row.get("actual_token_retention")
        if retention is None or abs(float(retention) - k / g) > 1e-9:
            raise InputContractError(f"{label} actual_token_retention inconsistent with K/G")
    else:  # pragma: no cover - guarded by ARMS
        raise InputContractError(f"unknown arm {arm!r}")
    if g <= 0:
        raise InputContractError(f"{label} G must be positive, got {g}")
    return {
        "G": g,
        "K": k,
        "retention": k / g,
        "identity_bypass": bool(row.get("identity_bypass", False)),
        "indices": indices,
    }


def validate_inputs(
    paths: InputPaths = DEFAULT_PATHS,
    expected_pins: Mapping[str, str] | None = DEFAULT_INPUT_PINS,
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    """Fail-closed validation of the whole three-arm input contract."""

    pinned = verify_pins(paths, expected_pins)
    config_pins = verify_existing_config_pins(paths)

    manifest_rows = load_rows(paths.screening_manifest, "screening_manifest")
    manifest = index_unique(manifest_rows, "screening_manifest")
    if len(manifest_rows) != sample_count or len(manifest) != sample_count:
        raise InputContractError(
            f"screening_manifest must have exactly {sample_count} unique samples, "
            f"got rows={len(manifest_rows)} unique={len(manifest)}"
        )

    arm_paths = {
        ARM_A: paths.adaptive_arm,
        ARM_B: paths.fixed_k96_arm,
        ARM_C: paths.r25_arm,
    }
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    budgets: dict[str, dict[str, dict[str, Any]]] = {}
    for arm, path in arm_paths.items():
        rows = load_rows(path, arm)
        if len(rows) != sample_count:
            raise InputContractError(
                f"{arm} must have exactly {sample_count} rows, got {len(rows)}"
            )
        index = index_unique(rows, arm)
        if len(index) != sample_count:
            raise InputContractError(f"{arm} must have {sample_count} unique sample_ids")
        indexes[arm] = index

    reference_ids = set(manifest)
    for arm in ARMS:
        if set(indexes[arm]) != reference_ids:
            missing = sorted(reference_ids - set(indexes[arm]))[:10]
            extra = sorted(set(indexes[arm]) - reference_ids)[:10]
            raise InputContractError(
                f"{arm} sample_id set differs from screening_manifest; "
                f"missing={missing} extra={extra}"
            )

    for arm in ARMS:
        arm_budgets: dict[str, dict[str, Any]] = {}
        for sample_id, row in indexes[arm].items():
            manifest_row = manifest[sample_id]
            if str(row.get("task")) != str(manifest_row.get("task")):
                raise InputContractError(
                    f"{arm}/{sample_id} task {row.get('task')!r} != manifest "
                    f"{manifest_row.get('task')!r}"
                )
            if int(row.get("prune_after_layer", -1)) != PRUNE_AFTER_LAYER:
                raise InputContractError(
                    f"{arm}/{sample_id} prune_after_layer must be {PRUNE_AFTER_LAYER}"
                )
            arm_budgets[sample_id] = derive_budget(row, arm, f"{arm}/{sample_id}")
        budgets[arm] = arm_budgets

    for sample_id in reference_ids:
        g_values = {arm: budgets[arm][sample_id]["G"] for arm in ARMS}
        if len(set(g_values.values())) != 1:
            raise InputContractError(
                f"{sample_id} G differs across arms: {g_values} (fail closed)"
            )

    tasks = {str(manifest[sid].get("task")) for sid in manifest}
    if not tasks.issubset(set(EXPECTED_TASKS)):
        raise InputContractError(f"unexpected task labels in manifest: {sorted(tasks)}")

    return {
        "manifest": manifest,
        "indexes": indexes,
        "budgets": budgets,
        "pins": pinned,
        "config_pins": config_pins,
    }


# ---------------------------------------------------------------------------
# Analysis
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


def build_joined(
    manifest: Mapping[str, Mapping[str, Any]],
    indexes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    budgets: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for sample_id, meta in manifest.items():
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "task": str(meta["task"]),
            "source": str(meta["source"]),
            "source_dataset": str(meta["source_dataset"]),
            "ground_truth": truth,
            "G": budgets[ARM_A][sample_id]["G"],
        }
        for arm in ARMS:
            row = indexes[arm][sample_id]
            answer = parse_choice(str(row.get("output_text", "")), choices)
            record[f"{arm}_answer"] = answer
            record[f"{arm}_correct"] = bool(answer is not None and answer == truth)
            record[f"{arm}_output_text"] = str(row.get("output_text", ""))
            record[f"{arm}_K"] = budgets[arm][sample_id]["K"]
            record[f"{arm}_retention"] = budgets[arm][sample_id]["retention"]
            record[f"{arm}_identity"] = budgets[arm][sample_id]["identity_bypass"]
        record["A_minus_C_K"] = record[f"{ARM_A}_K"] - record[f"{ARM_C}_K"]
        record["A_minus_B_K"] = record[f"{ARM_A}_K"] - record[f"{ARM_B}_K"]
        stats_ac = _set_stats(
            budgets[ARM_A][sample_id]["indices"], budgets[ARM_C][sample_id]["indices"]
        )
        stats_ab = _set_stats(
            budgets[ARM_A][sample_id]["indices"], budgets[ARM_B][sample_id]["indices"]
        )
        for name, value in stats_ac.items():
            record[f"A_C_{name}"] = value
        for name, value in stats_ab.items():
            record[f"A_B_{name}"] = value
        record["A_C_output_equal"] = record[f"{ARM_A}_output_text"] == record[f"{ARM_C}_output_text"]
        record["A_B_output_equal"] = record[f"{ARM_A}_output_text"] == record[f"{ARM_B}_output_text"]
        record["A_C_answer_equal"] = record[f"{ARM_A}_answer"] == record[f"{ARM_C}_answer"]
        record["A_B_answer_equal"] = record[f"{ARM_A}_answer"] == record[f"{ARM_B}_answer"]
        joined.append(record)
    return joined


def arm_scores(joined: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Official per-arm scores via the frozen scorer only."""

    return {
        arm: score(list(joined), f"{arm}_correct") for arm in ARMS
    }


def transition_counts(
    joined: Sequence[Mapping[str, Any]], reference: str, treatment: str
) -> dict[str, Any]:
    right = f"{reference}_correct"
    treat = f"{treatment}_correct"
    counts = {
        "reference_correct_treatment_correct": 0,
        "reference_correct_treatment_incorrect": 0,
        "reference_incorrect_treatment_correct": 0,
        "reference_incorrect_treatment_incorrect": 0,
    }
    for row in joined:
        if row[right] and row[treat]:
            counts["reference_correct_treatment_correct"] += 1
        elif row[right] and not row[treat]:
            counts["reference_correct_treatment_incorrect"] += 1
        elif not row[right] and row[treat]:
            counts["reference_incorrect_treatment_correct"] += 1
        else:
            counts["reference_incorrect_treatment_incorrect"] += 1
    text_key = f"{treatment}_output_text"
    ref_text_key = f"{reference}_output_text"
    counts["output_text_changed"] = sum(
        1 for row in joined if row[ref_text_key] != row[text_key]
    )
    counts["answer_changed"] = sum(
        1 for row in joined if row[f"{reference}_answer"] != row[f"{treatment}_answer"]
    )
    counts["output_text_changed_rate"] = counts["output_text_changed"] / len(joined)
    counts["answer_changed_rate"] = counts["answer_changed"] / len(joined)
    return counts


def aggregate_set_stats(
    joined: Sequence[Mapping[str, Any]], prefix: str
) -> dict[str, Any]:
    jaccards = [row[f"{prefix}_jaccard"] for row in joined]
    return {
        "mean_jaccard": statistics.fmean(jaccards),
        "min_jaccard": min(jaccards),
        "max_jaccard": max(jaccards),
        "mean_intersection_count": statistics.fmean(
            row[f"{prefix}_intersection_count"] for row in joined
        ),
        "mean_union_count": statistics.fmean(
            row[f"{prefix}_union_count"] for row in joined
        ),
        "left_superset_right_count": sum(
            bool(row[f"{prefix}_left_superset_right"]) for row in joined
        ),
        "right_superset_left_count": sum(
            bool(row[f"{prefix}_right_superset_left"]) for row in joined
        ),
        "equal_count": sum(bool(row[f"{prefix}_equal"]) for row in joined),
    }


def budget_summary(
    joined: Sequence[Mapping[str, Any]], arm: str
) -> dict[str, Any]:
    g_values = [row["G"] for row in joined]
    k_values = [row[f"{arm}_K"] for row in joined]
    retentions = [row[f"{arm}_retention"] for row in joined]
    return {
        "G_min": min(g_values),
        "G_max": max(g_values),
        "G_mean": statistics.fmean(g_values),
        "K_min": min(k_values),
        "K_max": max(k_values),
        "K_mean": statistics.fmean(k_values),
        "K_total": sum(k_values),
        "retention_mean": statistics.fmean(retentions),
        "retention_min": min(retentions),
        "retention_max": max(retentions),
        "identity_bypass_count": sum(bool(row[f"{arm}_identity"]) for row in joined),
    }


def _unweighted_accuracy(
    joined: Sequence[Mapping[str, Any]], arm: str
) -> float:
    """Secondary diagnostic only; never the official CV-Bench metric."""

    return statistics.fmean(bool(row[f"{arm}_correct"]) for row in joined)


def _bucket_rows(
    joined: Sequence[Mapping[str, Any]],
    labels: Sequence[tuple[str, Callable[[int], bool]]],
    value: Callable[[Mapping[str, Any]], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, predicate in labels:
        subset = [row for row in joined if predicate(value(row))]
        if not subset:
            continue
        a_acc = _unweighted_accuracy(subset, ARM_A)
        b_acc = _unweighted_accuracy(subset, ARM_B)
        c_acc = _unweighted_accuracy(subset, ARM_C)
        rows.append(
            {
                "bucket": label,
                "n": len(subset),
                "A_unweighted_accuracy_secondary": a_acc,
                "B_unweighted_accuracy_secondary": b_acc,
                "C_unweighted_accuracy_secondary": c_acc,
                "A_minus_C_unweighted_delta_secondary": a_acc - c_acc,
                "A_minus_B_unweighted_delta_secondary": a_acc - b_acc,
                "mean_G": statistics.fmean(row["G"] for row in subset),
                "mean_A_K": statistics.fmean(row[f"{ARM_A}_K"] for row in subset),
                "mean_C_K": statistics.fmean(row[f"{ARM_C}_K"] for row in subset),
                "mean_A_minus_C_K": statistics.fmean(row["A_minus_C_K"] for row in subset),
                "mean_A_C_jaccard": statistics.fmean(row["A_C_jaccard"] for row in subset),
                "A_superset_C_count": sum(row["A_C_left_superset_right"] for row in subset),
                "C_superset_A_count": sum(row["A_C_right_superset_left"] for row in subset),
                "A_C_both_correct": sum(
                    row[f"{ARM_A}_correct"] and row[f"{ARM_C}_correct"] for row in subset
                ),
                "A_C_A_only_correct": sum(
                    row[f"{ARM_A}_correct"] and not row[f"{ARM_C}_correct"] for row in subset
                ),
                "A_C_C_only_correct": sum(
                    not row[f"{ARM_A}_correct"] and row[f"{ARM_C}_correct"] for row in subset
                ),
            }
        )
    return rows


def task_rows(joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in joined}):
        subset = [row for row in joined if row["task"] == task]
        a_acc = _unweighted_accuracy(subset, ARM_A)
        b_acc = _unweighted_accuracy(subset, ARM_B)
        c_acc = _unweighted_accuracy(subset, ARM_C)
        rows.append(
            {
                "task": task,
                "n": len(subset),
                "A_unweighted_accuracy_secondary": a_acc,
                "B_unweighted_accuracy_secondary": b_acc,
                "C_unweighted_accuracy_secondary": c_acc,
                "A_minus_C_unweighted_delta_secondary": a_acc - c_acc,
                "mean_G": statistics.fmean(row["G"] for row in subset),
                "mean_A_minus_C_K": statistics.fmean(row["A_minus_C_K"] for row in subset),
                "mean_A_C_jaccard": statistics.fmean(row["A_C_jaccard"] for row in subset),
                "A_superset_C_count": sum(row["A_C_left_superset_right"] for row in subset),
                "C_superset_A_count": sum(row["A_C_right_superset_left"] for row in subset),
            }
        )
    return rows


def attribution(
    scores: Mapping[str, Mapping[str, Any]],
    budget: Mapping[str, Mapping[str, Any]],
    sets_ac: Mapping[str, Any],
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    """Strict, non-causal attribution of the A<C gap to the budget."""

    a = scores[ARM_A]["cv_bench_accuracy"]
    b = scores[ARM_B]["cv_bench_accuracy"]
    c = scores[ARM_C]["cv_bench_accuracy"]
    a_minus_c = a - c
    a_minus_b = a - b
    k_nonnegative = (
        budget["A_minus_C_K_min"] >= 0
    )
    nested = (
        sets_ac["left_superset_right_count"] == sample_count
        or sets_ac["right_superset_left_count"] == sample_count
    )
    attribution_supported = (
        a_minus_c < 0 and k_nonnegative and not nested
    )
    if attribution_supported:
        verdict = "NOT_DIRECT_EVIDENCE_OF_TOKEN_INSUFFICIENCY"
        statement = (
            "不是 token 数量不足的直接证据；A 的 K 在每个样本上都 >= C，但官方分更低，"
            "且 A/C 选择集合大多非嵌套（A 超集 C 仅少量样本、C 超集 A 为 0），"
            "因此主要可观测差异是选择集合/实现路径，而不是预算大小。"
        )
    else:
        verdict = "BUDGET_ATTRIBUTION_NOT_ESTABLISHED"
        statement = (
            "现有观测不足以把差距归因于预算；A 的 K>=C 不成立或集合关系为嵌套，"
            "需要最小因果实验才能区分预算与选择集合。"
        )
    return {
        "verdict": verdict,
        "statement_zh": statement,
        "A_official_cv_bench_accuracy": a,
        "B_official_cv_bench_accuracy": b,
        "C_official_cv_bench_accuracy": c,
        "A_minus_C_official_delta": a_minus_c,
        "A_minus_B_official_delta": a_minus_b,
        "A_K_ge_C_K_all_samples": k_nonnegative,
        "A_min_K_minus_C_K": budget["A_minus_C_K_min"],
        "max_K_minus_C_K": budget["A_minus_C_K_max"],
        "selection_sets_nested": nested,
        "A_superset_C_count": sets_ac["left_superset_right_count"],
        "C_superset_A_count": sets_ac["right_superset_left_count"],
        "causality_disclaimer": (
            "仅凭本筛选集上的观测不能声称因果；unweighted/子组数字均为 secondary 诊断。"
        ),
    }


def minimal_causal_experiments() -> list[dict[str, Any]]:
    """Same implementation, same budget, only the index set is replaced."""

    common = (
        "同一实现（evaluation/hardware_regular_layer5_unbounded_l2.py 的 layer-5 "
        "cut=5、blocks 0..5 → compact hidden → DeepStack index 5 → blocks 6..23 路径）、"
        "同一预算、仅替换选中的 selected_group_indices；不改变 score_mode/region_grid/"
        "BF16/SDPA/greedy。"
    )
    return [
        {
            "id": "E1_forced_index_set_A_vs_C",
            "design": (
                common
                + " 对每个样本固定预算 K=K_C(K_C<=K_A)，臂 X 强制使用 C 的冻结索引集合，"
                "臂 Y 用同一选择算法以预算 K_C 重新选出索引集合；两臂仅索引集合不同。"
            ),
            "implementation": "evaluation/hardware_regular_layer5_unbounded_l2.py",
            "budget": "per-sample K_C (fixed, identical in both arms)",
            "only_variable": "selected_group_indices",
            "requires_runner_change": (
                "需要给 runner 增加只读 --forced-selected-indices JSONL 入口（不改既有默认路径）。"
            ),
            "executed": False,
            "requires_model_forward": True,
        },
    ]


def analyze(
    paths: InputPaths = DEFAULT_PATHS,
    expected_pins: Mapping[str, str] | None = DEFAULT_INPUT_PINS,
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    validated = validate_inputs(paths, expected_pins, sample_count)
    indexes = validated["indexes"]
    budgets = validated["budgets"]
    manifest = validated["manifest"]

    joined = build_joined(manifest, indexes, budgets)
    scores = arm_scores(joined)

    k_ac = [row["A_minus_C_K"] for row in joined]
    budget = {
        ARM_A: budget_summary(joined, ARM_A),
        ARM_B: budget_summary(joined, ARM_B),
        ARM_C: budget_summary(joined, ARM_C),
        "A_minus_C_K_min": min(k_ac),
        "A_minus_C_K_max": max(k_ac),
        "A_minus_C_K_mean": statistics.fmean(k_ac),
        "A_K_ge_C_K_all_samples": all(value >= 0 for value in k_ac),
        "A_minus_B_K_min": min(row["A_minus_B_K"] for row in joined),
        "A_minus_B_K_max": max(row["A_minus_B_K"] for row in joined),
    }

    sets_ac = aggregate_set_stats(joined, "A_C")
    sets_ab = aggregate_set_stats(joined, "A_B")

    transitions = {
        "A_vs_C": transition_counts(joined, ARM_C, ARM_A),
        "A_vs_B": transition_counts(joined, ARM_B, ARM_A),
        "B_vs_C": transition_counts(joined, ARM_C, ARM_B),
    }

    tasks = task_rows(joined)
    g_buckets = _bucket_rows(joined, G_BUCKETS, lambda row: row["G"])
    kdiff_buckets = _bucket_rows(joined, KDIFF_BUCKETS, lambda row: row["A_minus_C_K"])

    official = {
        arm: {
            "cv_bench_accuracy": scores[arm]["cv_bench_accuracy"],
            "unweighted_sample_accuracy_secondary": scores[arm]["accuracy"],
            "accuracy_2d": scores[arm]["accuracy_2d"],
            "accuracy_3d": scores[arm]["accuracy_3d"],
            "source_accuracy": scores[arm]["source_accuracy"],
            "task_accuracy": scores[arm]["task_accuracy"],
            "sample_count": scores[arm]["sample_count"],
        }
        for arm in ARMS
    }

    # Re-verify read-only inputs (nothing may have changed during the analysis).
    for name, spec in validated["pins"].items():
        current = sha256_file(Path(spec["path"]))
        if current != spec["sha256"]:
            raise InputContractError(
                f"input '{name}' changed during analysis: {spec['sha256']} -> {current} "
                "(fail closed)"
            )

    per_sample = [
        {
            "sample_id": row["sample_id"],
            "task": row["task"],
            "source": row["source"],
            "source_dataset": row["source_dataset"],
            "ground_truth": row["ground_truth"],
            "G": row["G"],
            "A_K": row[f"{ARM_A}_K"],
            "B_K": row[f"{ARM_B}_K"],
            "C_K": row[f"{ARM_C}_K"],
            "A_retention": row[f"{ARM_A}_retention"],
            "B_retention": row[f"{ARM_B}_retention"],
            "C_retention": row[f"{ARM_C}_retention"],
            "A_minus_C_K": row["A_minus_C_K"],
            "A_minus_B_K": row["A_minus_B_K"],
            "A_answer": row[f"{ARM_A}_answer"],
            "B_answer": row[f"{ARM_B}_answer"],
            "C_answer": row[f"{ARM_C}_answer"],
            "A_correct": row[f"{ARM_A}_correct"],
            "B_correct": row[f"{ARM_B}_correct"],
            "C_correct": row[f"{ARM_C}_correct"],
            "A_C_transition": _transition_label(row, ARM_C, ARM_A),
            "A_B_transition": _transition_label(row, ARM_B, ARM_A),
            "A_C_output_equal": row["A_C_output_equal"],
            "A_B_output_equal": row["A_B_output_equal"],
            "A_C_answer_equal": row["A_C_answer_equal"],
            "A_B_answer_equal": row["A_B_answer_equal"],
            "A_C_intersection": row["A_C_intersection_count"],
            "A_C_union": row["A_C_union_count"],
            "A_C_jaccard": row["A_C_jaccard"],
            "A_superset_C": row["A_C_left_superset_right"],
            "C_superset_A": row["A_C_right_superset_left"],
            "A_B_intersection": row["A_B_intersection_count"],
            "A_B_union": row["A_B_union_count"],
            "A_B_jaccard": row["A_B_jaccard"],
            "A_superset_B": row["A_B_left_superset_right"],
            "B_superset_A": row["A_B_right_superset_left"],
            "G_bucket": _bucket_label(row["G"], G_BUCKETS),
            "Kdiff_bucket": _bucket_label(row["A_minus_C_K"], KDIFF_BUCKETS),
        }
        for row in joined
    ]

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "analysis": "adaptive_vs_r25_screening100",
        "scope": (
            "fixed 100-sample CV-Bench screening; CPU-only read-only diagnostic; "
            "NOT a formal full2638 conclusion"
        ),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metric": SECONDARY_METRIC,
        "unweighted_note": UNWEIGHTED_NOTE,
        "arms": ARM_DESCRIPTIONS,
        "inputs_sha256": validated["pins"],
        "config_pins_checked": validated["config_pins"],
        "contract": {
            "sample_count": sample_count,
            "unique_sample_ids_per_arm": True,
            "sample_id_sets_equal": True,
            "task_consistent_with_manifest": True,
            "prune_after_layer": PRUNE_AFTER_LAYER,
            "selected_indices_strictly_ascending_unique": True,
            "selected_indices_sha256_reproduced": True,
            "G_consistent_across_arms": True,
            "inputs_unchanged_during_analysis": True,
        },
        "official_scores": official,
        "official_deltas": {
            "A_minus_C": official[ARM_A]["cv_bench_accuracy"]
            - official[ARM_C]["cv_bench_accuracy"],
            "A_minus_B": official[ARM_A]["cv_bench_accuracy"]
            - official[ARM_B]["cv_bench_accuracy"],
            "B_minus_C": official[ARM_B]["cv_bench_accuracy"]
            - official[ARM_C]["cv_bench_accuracy"],
        },
        "budget": budget,
        "selection_sets": {"A_vs_C": sets_ac, "A_vs_B": sets_ab},
        "transitions": transitions,
        "by_task": tasks,
        "by_G_bucket": g_buckets,
        "by_Kdiff_bucket": kdiff_buckets,
        "attribution": attribution(scores, budget, sets_ac, sample_count),
        "minimal_causal_experiments": minimal_causal_experiments(),
        "limitations": [
            "screening100 only; not the formal 2638-sample benchmark",
            "unweighted and subgroup accuracies are secondary diagnostics",
            "no GPU/model/board execution in this analysis",
            "A/C also differ in implementation path, not only in selected indices",
        ],
        "per_sample": per_sample,
    }


def _transition_label(
    row: Mapping[str, Any], reference: str, treatment: str
) -> str:
    right = bool(row[f"{reference}_correct"])
    treat = bool(row[f"{treatment}_correct"])
    if right and treat:
        return "both_correct"
    if right and not treat:
        return "reference_only_correct"
    if not right and treat:
        return "treatment_only_correct"
    return "both_incorrect"


def _bucket_label(
    value: int, labels: Sequence[tuple[str, Callable[[int], bool]]]
) -> str:
    for label, predicate in labels:
        if predicate(value):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


CSV_COLUMNS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "G",
    "A_K",
    "B_K",
    "C_K",
    "A_retention",
    "B_retention",
    "C_retention",
    "A_minus_C_K",
    "A_minus_B_K",
    "A_answer",
    "B_answer",
    "C_answer",
    "A_correct",
    "B_correct",
    "C_correct",
    "A_C_transition",
    "A_B_transition",
    "A_C_output_equal",
    "A_B_output_equal",
    "A_C_answer_equal",
    "A_B_answer_equal",
    "A_C_intersection",
    "A_C_union",
    "A_C_jaccard",
    "A_superset_C",
    "C_superset_A",
    "A_B_intersection",
    "A_B_union",
    "A_B_jaccard",
    "A_superset_B",
    "B_superset_A",
    "G_bucket",
    "Kdiff_bucket",
)


def write_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in report["per_sample"]:
            writer.writerow({name: row[name] for name in CSV_COLUMNS})


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Adaptive (A) vs fixed-K96 (B) vs historical r25 (C) — screening100 严格诊断")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 范围：{report['scope']}")
    lines.append(f"- 唯一主指标：`{report['primary_metric']}`；unweighted = `{report['secondary_metric']}`（**仅次要**）")
    lines.append(
        "- 三臂：A=adaptive layer-5 budget；B=fixed K96；C=historical shallow r0250；"
        "同一 100 样本、同一 task、`prune_after_layer=5`。"
    )
    lines.append("")

    lines.append("## 1. 官方分数（`score(...)[\"cv_bench_accuracy\"]`）")
    lines.append("")
    lines.append("| 臂 | cv_bench_accuracy | A−C | A−B | unweighted(次要) | 2D | 3D |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    official = report["official_scores"]
    deltas = report["official_deltas"]
    for arm in ARMS:
        item = official[arm]
        lines.append(
            f"| {arm} | {_fmt(item['cv_bench_accuracy'])} | "
            f"{_fmt(item['cv_bench_accuracy'] - official[ARM_C]['cv_bench_accuracy'])} | "
            f"{_fmt(item['cv_bench_accuracy'] - official[ARM_B]['cv_bench_accuracy'])} | "
            f"{_fmt(item['unweighted_sample_accuracy_secondary'])} | "
            f"{_fmt(item['accuracy_2d'])} | {_fmt(item['accuracy_3d'])} |"
        )
    lines.append("")
    lines.append(
        f"- A 相对 C 的官方分差 **{_fmt(deltas['A_minus_C'])}**；"
        f"A 相对 B 的官方分差 **{_fmt(deltas['A_minus_B'])}**。"
    )
    lines.append("")

    budget = report["budget"]
    lines.append("## 2. 每样本 G / K / retention（汇总）")
    lines.append("")
    lines.append("| 臂 | G 范围 | K 范围 | K 均值 | retention 均值 | identity 数 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for arm in ARMS:
        item = budget[arm]
        lines.append(
            f"| {arm} | {item['G_min']}–{item['G_max']} | {item['K_min']}–{item['K_max']} | "
            f"{_fmt(item['K_mean'])} | {_fmt(item['retention_mean'])} | {item['identity_bypass_count']} |"
        )
    lines.append("")
    lines.append(
        f"- A−C 的 K 差：min={budget['A_minus_C_K_min']}、max={budget['A_minus_C_K_max']}、"
        f"mean={_fmt(budget['A_minus_C_K_mean'])}；"
        f"A 的 K 在所有样本上 >= C：**{budget['A_K_ge_C_K_all_samples']}**。"
    )
    lines.append("")

    lines.append("## 3. 正确性迁移与文本变化")
    lines.append("")
    lines.append("| 对比 | 都对 | 仅参考对 | 仅处理对 | 都错 | 文本变化 | 答案变化 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, item in report["transitions"].items():
        lines.append(
            f"| {name} | {item['reference_correct_treatment_correct']} | "
            f"{item['reference_correct_treatment_incorrect']} | "
            f"{item['reference_incorrect_treatment_correct']} | "
            f"{item['reference_incorrect_treatment_incorrect']} | "
            f"{item['output_text_changed']} | {item['answer_changed']} |"
        )
    lines.append("")

    lines.append("## 4. 选择集合关系")
    lines.append("")
    lines.append("| 对比 | 平均 Jaccard | 平均交集 | 平均并集 | A⊇other | other⊇A | 相等 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, item in report["selection_sets"].items():
        lines.append(
            f"| {name} | {_fmt(item['mean_jaccard'])} | "
            f"{_fmt(item['mean_intersection_count'])} | {_fmt(item['mean_union_count'])} | "
            f"{item['left_superset_right_count']} | {item['right_superset_left_count']} | "
            f"{item['equal_count']} |"
        )
    lines.append("")
    lines.append("> A_vs_C 中 left=A、right=C；A_vs_B 中 left=A、right=B。")
    lines.append("")

    lines.append("## 5. 分 task 统计")
    lines.append("")
    lines.append("| task | n | A(unweighted) | B(unweighted) | C(unweighted) | A−C | 平均 A−C K | 平均 Jaccard | A⊇C | C⊇A |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["by_task"]:
        lines.append(
            f"| {item['task']} | {item['n']} | "
            f"{_fmt(item['A_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['B_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['C_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['A_minus_C_unweighted_delta_secondary'])} | "
            f"{_fmt(item['mean_A_minus_C_K'])} | {_fmt(item['mean_A_C_jaccard'])} | "
            f"{item['A_superset_C_count']} | {item['C_superset_A_count']} |"
        )
    lines.append("")

    lines.append("## 6. 按 G 桶统计（unweighted 次要）")
    lines.append("")
    lines.append("| G 桶 | n | A | B | C | A−C | 平均 A−C K | A⊇C | C⊇A |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["by_G_bucket"]:
        lines.append(
            f"| {item['bucket']} | {item['n']} | "
            f"{_fmt(item['A_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['B_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['C_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['A_minus_C_unweighted_delta_secondary'])} | "
            f"{_fmt(item['mean_A_minus_C_K'])} | {item['A_superset_C_count']} | "
            f"{item['C_superset_A_count']} |"
        )
    lines.append("")

    lines.append("## 7. 按 K 差（A−C）桶统计（unweighted 次要）")
    lines.append("")
    lines.append("| K 差桶 | n | A | B | C | A−C | 平均 Jaccard | A⊇C | C⊇A |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in report["by_Kdiff_bucket"]:
        lines.append(
            f"| {item['bucket']} | {item['n']} | "
            f"{_fmt(item['A_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['B_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['C_unweighted_accuracy_secondary'])} | "
            f"{_fmt(item['A_minus_C_unweighted_delta_secondary'])} | "
            f"{_fmt(item['mean_A_C_jaccard'])} | {item['A_superset_C_count']} | "
            f"{item['C_superset_A_count']} |"
        )
    lines.append("")

    attr = report["attribution"]
    lines.append("## 8. 严格归因判断")
    lines.append("")
    lines.append(f"- A=**{_fmt(attr['A_official_cv_bench_accuracy'])}**、"
                 f"C=**{_fmt(attr['C_official_cv_bench_accuracy'])}**、"
                 f"B=**{_fmt(attr['B_official_cv_bench_accuracy'])}**。")
    lines.append(f"- A−C 官方差：**{_fmt(attr['A_minus_C_official_delta'])}**；"
                 f"A−B 官方差：**{_fmt(attr['A_minus_B_official_delta'])}**。")
    lines.append(f"- A 的 K>=C 在所有样本上成立：**{attr['A_K_ge_C_K_all_samples']}**"
                 f"（min A−C K={attr['A_min_K_minus_C_K']}，max={attr['max_K_minus_C_K']}）。")
    lines.append(f"- 选择集合是否为嵌套：**{attr['selection_sets_nested']}**"
                 f"（A⊇C={attr['A_superset_C_count']}/100，C⊇A={attr['C_superset_A_count']}/100）。")
    lines.append(f"- **结论：{attr['verdict']}** — {attr['statement_zh']}")
    lines.append(f"- 免责：{attr['causality_disclaimer']}")
    lines.append("")

    lines.append("## 9. 最小因果实验（同一实现、同一预算、仅替换索引集合；本次未执行、无 GPU）")
    lines.append("")
    for item in report["minimal_causal_experiments"]:
        lines.append(f"- **{item['id']}**：{item['design']}")
        lines.append(
            f"  - 实现={item['implementation']}；预算={item['budget']}；"
            f"唯一变量={item['only_variable']}；已执行={item['executed']}；"
            f"需要模型前向={item['requires_model_forward']}。"
        )
        lines.append(f"  - runner 变更：{item['requires_runner_change']}")
    lines.append("")

    lines.append("## 10. 输入 SHA256（只读校验）")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("| --- | --- |")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")

    lines.append("## 11. 限制")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(report: Mapping[str, Any], outputs: OutputPaths = DEFAULT_OUTPUTS) -> None:
    outputs.json.parent.mkdir(parents=True, exist_ok=True)
    outputs.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(report, outputs.csv)
    outputs.md.write_text(render_markdown(report), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening-manifest", type=Path, default=SCREENING_MANIFEST)
    parser.add_argument("--adaptive-arm", type=Path, default=ADAPTIVE_ARM_PATH)
    parser.add_argument("--fixed-k96-arm", type=Path, default=FIXED_K96_ARM_PATH)
    parser.add_argument("--r25-arm", type=Path, default=R25_ARM_PATH)
    parser.add_argument("--adaptive-prereg", type=Path, default=ADAPTIVE_PREREG_PATH)
    parser.add_argument("--unbounded-prereg", type=Path, default=UNBOUNDED_PREREG_PATH)
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
        screening_manifest=args.screening_manifest,
        adaptive_arm=args.adaptive_arm,
        fixed_k96_arm=args.fixed_k96_arm,
        r25_arm=args.r25_arm,
        adaptive_prereg=args.adaptive_prereg,
        unbounded_prereg=args.unbounded_prereg,
    )
    report = analyze(paths, None if args.skip_pins else DEFAULT_INPUT_PINS)
    write_outputs(
        report,
        OutputPaths(json=args.json_out, csv=args.csv_out, md=args.md_out),
    )
    official = report["official_scores"]
    print(
        "official cv_bench_accuracy: "
        + ", ".join(
            f"{arm}={official[arm]['cv_bench_accuracy']:.6f}" for arm in ARMS
        )
    )
    print(f"attribution: {report['attribution']['verdict']}")
    print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
