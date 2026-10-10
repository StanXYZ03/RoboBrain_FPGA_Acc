#!/usr/bin/env python3
"""Fail-closed development summariser for the L2-guarded integer reranker v2.

Preregistered experiment:
``configs/2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json``.

This is a **development** decision only.  It distinguishes
``DEVELOPMENT_GO_EXTERNAL_VALIDATION`` (permission to acquire and run one fresh
untouched external spatial-reasoning benchmark) from ``DEVELOPMENT_STOP``.  It
is never a paper confirmation, deployability claim, FPGA latency claim or
official accuracy claim.

The single frozen v2 arm is compared, sample-for-sample, against the frozen
corrected integer-L2 K96 arm.  The summariser verifies every input identity, the
exact K96 real/context layout, the L2-guarded swap audit
(``legal_gain == semantic_gain - l2_loss == utility_added - utility_removed > 0``)
and the leakage flags before computing any score.  Any missing file, SHA pin
mismatch or contract violation raises and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    paired_bootstrap_delta,
    parse_choice,
    score,
)
from evaluation.conservative_l2_guarded_reranker import (  # noqa: E402
    BUDGET,
    CONTEXT_POOLING_BASIS,
    LEGAL_GAIN_FORMULA,
    LOCAL_CANDIDATE_CAP,
    MAX_SWAPS,
    MECHANISM,
    REGION_GRID,
    UTILITY_FORMULA,
)
from evaluation.hardware_regular_b1_corrected_l2 import array_sha256  # noqa: E402
from evaluation.hardware_regular_token_selector import (  # noqa: E402
    MERGE_UNIT,
    build_regions,
    region_id_by_group,
)
from evaluation.summarize_b1_corrected_l2_prompt_ablation_screening100 import (  # noqa: E402
    InputContractError,
    index_unique,
    load_rows,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.preregistered.json"
)
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DENSE_FULL_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
CORRECTED_L2_INTEGER_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
NEW_ARM_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638"
    / "conservative_l2_guarded_rerank_v2_k96_full2638.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.json"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-06-conservative-l2-guarded-reranker-v2-full2638-development.md"
)

FULL_SAMPLE_COUNT = 2638
#: Exact pinned dense BF16 official score from the frozen report.
DENSE_SCORE = 0.8727298774444869
#: Computed, never hand-rounded: 90% of the exact dense score.
GO_SCORE_FLOOR = 0.9 * DENSE_SCORE
#: Descriptive strongest prior same-budget development comparator (not a gate).
PRIOR_PROMPT_FUSION_SCORE = 0.7773617325561999
CORRECTED_L2_SCORE = 0.7668517265
BOOTSTRAP_SEED = 20261006
BOOTSTRAP_REPEATS = 5000
CI_LEVEL = 0.95
PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'

OUTCOME_GO = "DEVELOPMENT_GO_EXTERNAL_VALIDATION"
OUTCOME_STOP = "DEVELOPMENT_STOP"

LEAKAGE_FLAGS = (
    "uses_labels",
    "uses_answer_tokens",
    "uses_gradients",
    "uses_floating_lambda",
    "uses_learned_new_weights",
    "uses_task_source_ids",
)

EXPECTED_REAL = 87
EXPECTED_CONTEXT = 9


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    corrected_l2_integer_jsonl: Path = CORRECTED_L2_INTEGER_JSONL
    new_arm_jsonl: Path = NEW_ARM_JSONL
    report_json: Path = REPORT_JSON
    report_md: Path = REPORT_MD


DEFAULT_PATHS = InputPaths()


def _merge_grid_sizes(grid: Sequence[int]) -> tuple[int, int]:
    if len(grid) != 3 or int(grid[0]) != 1:
        raise InputContractError(f"image_grid_thw {list(grid)!r} is not a single image")
    grid_h = int(grid[1]) // 2
    grid_w = int(grid[2]) // 2
    if grid_h <= 0 or grid_w <= 0:
        raise InputContractError(f"image_grid_thw {list(grid)!r} is degenerate")
    return grid_h, grid_w


def _expected_context_count(grid: Sequence[int]) -> int:
    grid_h, grid_w = _merge_grid_sizes(grid)
    regions = build_regions(grid_h, grid_w, REGION_GRID)
    return len(
        [region for region in regions if region.members and len(region.members) >= 2]
    )


def _check_swap_audit(
    row: Mapping[str, Any],
    sample_id: str,
    *,
    max_swaps: int,
) -> None:
    """Verify the L2-guarded swap audit with widened signed arithmetic."""

    swap_count = int(row.get("swap_count", -1))
    if swap_count < 0 or swap_count > max_swaps:
        raise InputContractError(
            f"new arm {sample_id!r} swap_count {swap_count} out of range"
        )
    removed = [int(v) for v in row["removed_group_indices"]]
    added = [int(v) for v in row["added_group_indices"]]
    regions = [int(v) for v in row["swap_region_ids"]]
    sem_gains = [int(v) for v in row["swap_semantic_gain_q15"]]
    l2_losses = [int(v) for v in row["swap_l2_loss_q15"]]
    legal_gains = [int(v) for v in row["swap_legal_gain_q16"]]
    util_removed = [int(v) for v in row["swap_utility_removed_q16"]]
    util_added = [int(v) for v in row["swap_utility_added_q16"]]
    lengths = {
        len(removed),
        len(added),
        len(regions),
        len(sem_gains),
        len(l2_losses),
        len(legal_gains),
        len(util_removed),
        len(util_added),
    }
    if lengths != {swap_count}:
        raise InputContractError(
            f"new arm {sample_id!r} swap audit lengths disagree with swap_count"
        )
    if len(set(removed)) != len(removed) or len(set(added)) != len(added):
        raise InputContractError(f"new arm {sample_id!r} reuses a swapped group")
    if set(removed) & set(added):
        raise InputContractError(
            f"new arm {sample_id!r} both removes and adds a group"
        )
    for index, legal in enumerate(legal_gains):
        if legal <= 0:
            raise InputContractError(
                f"new arm {sample_id!r} swap {index} has non-positive legal gain"
            )
        if legal != sem_gains[index] - l2_losses[index]:
            raise InputContractError(
                f"new arm {sample_id!r} swap {index} legal != semantic_gain - l2_loss"
            )
        if legal != util_added[index] - util_removed[index]:
            raise InputContractError(
                f"new arm {sample_id!r} swap {index} legal != utility delta"
            )
    before = [int(v) for v in row["per_region_selected_counts_before"]]
    after = [int(v) for v in row["per_region_selected_counts_after"]]
    if before != after:
        raise InputContractError(
            f"new arm {sample_id!r} changed a region's selected count"
        )
    baseline_anchors = set(int(v) for v in row["baseline_anchors"])
    baseline_real = set(int(v) for v in row["baseline_selected_real"])
    reranked_real = set(int(v) for v in row["reranked_selected_real"])
    if not baseline_anchors.issubset(baseline_real):
        raise InputContractError(
            f"new arm {sample_id!r} baseline anchors not in baseline real set"
        )
    if not baseline_anchors.issubset(reranked_real):
        raise InputContractError(f"new arm {sample_id!r} dropped a frozen anchor")
    if len(reranked_real) != len(baseline_real):
        raise InputContractError(
            f"new arm {sample_id!r} changed real count via swaps"
        )
    if set(added) - reranked_real or (baseline_real - set(removed)) | set(
        added
    ) != reranked_real:
        raise InputContractError(
            f"new arm {sample_id!r} swap audit does not reconstruct reranked set"
        )
    if not set(removed).issubset(baseline_real):
        raise InputContractError(
            f"new arm {sample_id!r} removed group was not baseline-selected"
        )
    if set(added) & baseline_real:
        raise InputContractError(
            f"new arm {sample_id!r} added group was already selected"
        )
    grid_h, grid_w = _merge_grid_sizes([int(v) for v in row["image_grid_thw"]])
    region_lookup = region_id_by_group(grid_h, grid_w, REGION_GRID)
    for index, removed_group in enumerate(removed):
        added_group = added[index]
        if int(region_lookup[removed_group]) != int(regions[index]):
            raise InputContractError(
                f"new arm {sample_id!r} swap {index} removed group region mismatch"
            )
        if int(region_lookup[added_group]) != int(regions[index]):
            raise InputContractError(
                f"new arm {sample_id!r} swap {index} added group region mismatch"
            )


def _check_new_arm_records(
    index: Mapping[str, Mapping[str, Any]],
    manifest_index: Mapping[str, Mapping[str, Any]],
    *,
    manifest_sha256: str,
    max_swaps: int,
) -> dict[str, Any]:
    """Validate every v2 record against the frozen layout/leakage contract."""

    if set(index) != set(manifest_index):
        missing = sorted(set(manifest_index) - set(index))
        extra = sorted(set(index) - set(manifest_index))
        raise InputContractError(
            f"new arm sample ids do not match the full manifest; missing={missing[:10]} "
            f"extra={extra[:10]}"
        )
    bypass_ids: list[str] = []
    compressed_ids: list[str] = []
    real_count_ids: list[str] = []
    context_count_ids: list[str] = []
    swap_ids: list[str] = []
    for sample_id, row in index.items():
        meta = manifest_index[sample_id]
        if str(row.get("arm")) != MECHANISM:
            raise InputContractError(
                f"new arm {sample_id!r} arm={row.get('arm')!r} != {MECHANISM!r}"
            )
        if str(row.get("mechanism")) != MECHANISM:
            raise InputContractError(
                f"new arm {sample_id!r} mechanism={row.get('mechanism')!r}"
            )
        for field, expected in (
            ("requested_budget", BUDGET),
            ("region_grid", REGION_GRID),
            ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
            ("max_swaps", max_swaps),
        ):
            if int(row.get(field, -1)) != expected:
                raise InputContractError(
                    f"new arm {sample_id!r} {field}={row.get(field)!r} != {expected}"
                )
        if str(row.get("utility_formula")) != UTILITY_FORMULA:
            raise InputContractError(
                f"new arm {sample_id!r} utility_formula={row.get('utility_formula')!r}"
            )
        if str(row.get("legal_gain_formula")) != LEGAL_GAIN_FORMULA:
            raise InputContractError(
                f"new arm {sample_id!r} legal_gain_formula="
                f"{row.get('legal_gain_formula')!r}"
            )
        if str(row.get("context_pooling_basis")) != CONTEXT_POOLING_BASIS:
            raise InputContractError(
                f"new arm {sample_id!r} context_pooling_basis="
                f"{row.get('context_pooling_basis')!r}"
            )
        if str(row.get("manifest_sha256")) != manifest_sha256:
            raise InputContractError(
                f"new arm {sample_id!r} manifest_sha256 mismatch"
            )
        if str(row.get("task")) != str(meta.get("task")):
            raise InputContractError(f"new arm {sample_id!r} task mismatch")
        if str(row.get("record_sha256")) != _record_sha256(row):
            raise InputContractError(
                f"new arm {sample_id!r} record_sha256 mismatch (stale/corrupted)"
            )
        for flag in LEAKAGE_FLAGS:
            if bool(row.get(flag)):
                raise InputContractError(
                    f"new arm {sample_id!r} declares forbidden {flag}=True"
                )
        run = row.get("run") or {}
        for flag in LEAKAGE_FLAGS:
            if bool(run.get(flag)):
                raise InputContractError(
                    f"new arm {sample_id!r} run metadata declares forbidden "
                    f"{flag}=True"
                )

        grid = [int(value) for value in row["image_grid_thw"]]
        full_patch = int(row["full_patch_tokens"])
        full_llm = int(row["full_llm_visual_tokens"])
        output_count = int(row["output_group_count"])
        real_count = int(row["real_group_count"])
        context_count = int(row["context_group_count"])
        if grid[0] != 1 or grid[1] * grid[2] != full_patch:
            raise InputContractError(f"new arm {sample_id!r} grid mismatch")
        if full_llm * MERGE_UNIT != full_patch:
            raise InputContractError(f"new arm {sample_id!r} merge-unit mismatch")
        selected = [int(value) for value in row["selected_group_indices"]]
        if len(selected) != output_count:
            raise InputContractError(f"new arm {sample_id!r} selected length mismatch")
        if selected and not all(
            later > earlier for earlier, later in zip(selected, selected[1:])
        ):
            raise InputContractError(
                f"new arm {sample_id!r} selected_group_indices not ascending"
            )
        if str(row.get("selected_group_indices_sha256")) != array_sha256(
            np.asarray(selected, dtype=np.int64)
        ):
            raise InputContractError(
                f"new arm {sample_id!r} selected index hash mismatch"
            )

        if bool(row.get("identity_bypass")):
            if output_count != full_llm or output_count > BUDGET:
                raise InputContractError(
                    f"new arm {sample_id!r} identity bypass must emit G<=96"
                )
            if real_count != full_llm or context_count != 0:
                raise InputContractError(
                    f"new arm {sample_id!r} identity bypass real/context mismatch"
                )
            if int(row.get("swap_count", -1)) != 0:
                raise InputContractError(
                    f"new arm {sample_id!r} identity bypass must have swap_count 0"
                )
            bypass_ids.append(sample_id)
            continue

        compressed_ids.append(sample_id)
        if output_count != BUDGET:
            raise InputContractError(
                f"new arm {sample_id!r} compressed output_group_count != {BUDGET}"
            )
        expected_context = _expected_context_count(grid)
        if context_count != expected_context:
            raise InputContractError(
                f"new arm {sample_id!r} context_group_count {context_count} != "
                f"expected {expected_context}"
            )
        if real_count != BUDGET - context_count:
            raise InputContractError(f"new arm {sample_id!r} real+context != K")
        if real_count != EXPECTED_REAL or context_count != EXPECTED_CONTEXT:
            real_count_ids.append(sample_id)
        if context_count == EXPECTED_CONTEXT:
            context_count_ids.append(sample_id)

        _check_swap_audit(row, sample_id, max_swaps=max_swaps)
        swap_ids.append(sample_id)

    return {
        "bypass_count": len(bypass_ids),
        "bypass_sample_ids": bypass_ids[:20],
        "compressed_count": len(compressed_ids),
        "compressed_sample_ids_prefix": compressed_ids[:20],
        "full_size_87_9_count": len(context_count_ids),
        "non_87_9_compressed": real_count_ids[:20],
        "samples_with_swaps": len(
            [sid for sid in swap_ids if int(index[sid]["swap_count"]) > 0]
        ),
    }


def _record_sha256(record: Mapping[str, Any]) -> str:
    import hashlib

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def verify_config_pins(
    config: Mapping[str, Any], paths: InputPaths
) -> dict[str, Any]:
    """Verify every non-null config SHA pin against the on-disk file."""

    verified: dict[str, Any] = {}
    pins = config.get("inputs_sha256") or {}
    logical = {
        "full_manifest": paths.full_manifest,
        "dense_full_jsonl": paths.dense_full_jsonl,
        "corrected_l2_integer_jsonl": paths.corrected_l2_integer_jsonl,
        "new_arm_jsonl": paths.new_arm_jsonl,
        "preregistered_config": paths.preregistered,
    }
    for name, spec in pins.items():
        if not isinstance(spec, Mapping):
            raise InputContractError(f"config pin {name!r} is malformed")
        expected = spec.get("sha256")
        if expected is None:
            verified[name] = {"sha256": None, "status": "not_yet_available"}
            continue
        path = logical.get(name)
        if path is None:
            path = ROOT / str(spec.get("path", ""))
        if not path.is_file():
            raise InputContractError(
                f"pinned input {name!r} is missing: {path} (fail closed)"
            )
        actual = sha256_file(path)
        if actual != str(expected):
            raise InputContractError(
                f"config pin {name!r} sha256 mismatch: {actual} != {expected}"
            )
        verified[name] = {"sha256": actual, "status": "verified"}
    return verified


def verify_output_paths(config: Mapping[str, Any], paths: InputPaths) -> None:
    arm = (config.get("arms") or {}).get(MECHANISM) or {}
    configured = arm.get("output_path")
    if configured is None:
        raise InputContractError(f"config does not define arm {MECHANISM!r}")
    if Path(str(configured)).name != paths.new_arm_jsonl.name:
        raise InputContractError(
            f"config arm output {configured!r} does not match {paths.new_arm_jsonl}"
        )


def build_joined(
    sample_ids: Sequence[str],
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        meta = manifest_index[sample_id]
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


def _arm_score_rows(
    joined: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    return [
        {
            "task": record["task"],
            "type": record["type"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "correct": record[f"{arm}_correct"],
            "answer": record[f"{arm}_answer"],
        }
        for record in joined
    ]


def _pair_score_rows(
    joined: Sequence[Mapping[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    return [
        {
            "task": record["task"],
            "type": record["type"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "left_correct": record[f"{left}_correct"],
            "left_answer": record[f"{left}_answer"],
            "right_correct": record[f"{right}_correct"],
            "right_answer": record[f"{right}_answer"],
        }
        for record in joined
    ]


def arm_metrics(joined: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    result = score(_arm_score_rows(joined, arm), "correct")
    return {
        "cv_bench_accuracy": float(result["cv_bench_accuracy"]),
        "accuracy_2d": float(result["accuracy_2d"]),
        "accuracy_3d": float(result["accuracy_3d"]),
        "unweighted_sample_accuracy": float(result["accuracy"]),
        "parse_success_rate": float(result["parse_success_rate"]),
        "sample_count": int(result["sample_count"]),
    }


def paired_comparison(
    joined: Sequence[Mapping[str, Any]], left: str, right: str
) -> dict[str, Any]:
    rows = _pair_score_rows(joined, left, right)
    point = float(
        score(rows, "left_correct")["cv_bench_accuracy"]
        - score(rows, "right_correct")["cv_bench_accuracy"]
    )
    ci = paired_bootstrap_delta(
        rows,
        "left_correct",
        "right_correct",
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
        "ci_level": CI_LEVEL,
    }


def decide(
    *,
    new_score: float,
    parse_rate: float,
    layout_ok: bool,
    ci_lower: float,
    leakage_ok: bool,
) -> dict[str, Any]:
    """Corrected v2 gates (the prior-fusion point gate is descriptive only)."""

    gates = {
        "all_2638_complete_parseable": bool(parse_rate >= 1.0),
        "exact_k96_layout_contract": bool(layout_ok),
        "score_at_least_90pct_dense": bool(new_score >= GO_SCORE_FLOOR),
        "paired_ci_lower_nonnegative": bool(ci_lower >= 0.0),
        "no_forbidden_leakage": bool(leakage_ok),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        return {
            "outcome": OUTCOME_STOP,
            "reason": "failed gates: " + ", ".join(failed),
            "gates": gates,
            "uses_only_primary_metric": True,
            "prior_prompt_fusion_is_only_descriptive": True,
        }
    return {
        "outcome": OUTCOME_GO,
        "reason": (
            "all development gates passed; this is only permission to acquire a "
            "fresh untouched external spatial-reasoning benchmark, never a paper "
            "confirmation"
        ),
        "gates": gates,
        "uses_only_primary_metric": True,
        "prior_prompt_fusion_is_only_descriptive": True,
    }


def analyze(
    *,
    config: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    corrected_l2_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    paths: InputPaths,
    expected_full_count: int = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate all inputs and compute the development decision (no file I/O)."""

    manifest_index = index_unique(manifest_rows, "full_manifest")
    if len(manifest_index) != expected_full_count:
        raise InputContractError(
            f"full manifest must hold {expected_full_count} unique samples, got "
            f"{len(manifest_index)}"
        )
    dense_index = index_unique(dense_rows, "dense_full2638")
    corrected_index = index_unique(corrected_l2_rows, "corrected_l2_integer")
    new_index = index_unique(new_rows, "conservative_l2_guarded_rerank_v2")
    for label, index in (
        ("dense_full2638", dense_index),
        ("corrected_l2_integer", corrected_index),
        ("conservative_l2_guarded_rerank_v2", new_index),
    ):
        if len(index) != expected_full_count or set(index) != set(manifest_index):
            raise InputContractError(
                f"{label} must hold exactly the {expected_full_count} manifest sample "
                f"ids, got {len(index)}"
            )

    manifest_sha = sha256_file(paths.full_manifest)
    layout = _check_new_arm_records(
        new_index,
        manifest_index,
        manifest_sha256=manifest_sha,
        max_swaps=int(
            config.get("fixed_configuration", {}).get("max_swaps", MAX_SWAPS)
        ),
    )

    sample_ids = list(manifest_index)
    joined = build_joined(
        sample_ids,
        manifest_index,
        {
            "new": new_index,
            "corrected_l2_integer": corrected_index,
            "dense": dense_index,
        },
    )
    metrics = {
        "new": arm_metrics(joined, "new"),
        "corrected_l2_integer": arm_metrics(joined, "corrected_l2_integer"),
        "dense": arm_metrics(joined, "dense"),
    }
    comparison = paired_comparison(joined, "new", "corrected_l2_integer")

    leakage_ok = all(
        not bool(row.get(flag))
        for row in new_index.values()
        for flag in LEAKAGE_FLAGS
    )
    decision = decide(
        new_score=float(metrics["new"]["cv_bench_accuracy"]),
        parse_rate=float(metrics["new"]["parse_success_rate"]),
        layout_ok=(
            layout["compressed_count"] + layout["bypass_count"]
            == expected_full_count
            and layout["full_size_87_9_count"] == layout["compressed_count"]
        ),
        ci_lower=float(comparison["delta_ci95"][0]),
        leakage_ok=leakage_ok,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "status": "DEVELOPMENT_DECISION",
        "mechanism": MECHANISM,
        "development_only": True,
        "paper_confirmation": False,
        "primary_metric": PRIMARY_METRIC,
        "reference_scores": {
            "dense_bf16": DENSE_SCORE,
            "corrected_l2_integer": CORRECTED_L2_SCORE,
            "prior_prompt_plus_corrected_l2_descriptive": PRIOR_PROMPT_FUSION_SCORE,
            "go_score_floor_90pct_dense": GO_SCORE_FLOOR,
        },
        "fixed_configuration": dict(config.get("fixed_configuration", {})),
        "layout": layout,
        "metrics": metrics,
        "comparison_new_minus_corrected_l2_integer": comparison,
        "leakage_ok": leakage_ok,
        "decision": decision,
        "inputs_sha256": {
            "full_manifest": manifest_sha,
            "new_arm_jsonl": sha256_file(paths.new_arm_jsonl),
            "corrected_l2_integer_jsonl": sha256_file(
                paths.corrected_l2_integer_jsonl
            ),
            "dense_full_jsonl": sha256_file(paths.dense_full_jsonl),
        },
        "limitations": [
            "Development evidence only; the full CV-Bench 2638 was observed "
            "repeatedly and is never an untouched confirmation set.",
            "Passing is permission to acquire/run a fresh untouched external "
            "spatial-reasoning benchmark, not an official confirmation, "
            "deployability claim, FPGA latency claim or paper accuracy claim.",
            "The Selector path is a software bridge; no FPGA/board/latency claim.",
            "Do not tune v2 after seeing full2638.",
            "The prior prompt-fusion score is a descriptive same-budget comparator "
            "only and is not a redundant gate.",
        ],
    }


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    metrics = report["metrics"]
    comparison = report["comparison_new_minus_corrected_l2_integer"]
    layout = report["layout"]
    lines = [
        "# L2 守卫整数重排 v2（full2638 发展性判定）",
        "",
        f"- 机制：`{report['mechanism']}`",
        f"- 判定：`{decision['outcome']}`",
        f"- 理由：{decision['reason']}",
        "- 口径：**仅发展性证据**，不是论文确认、不是可部署性、不是 FPGA 时延/加速主张。",
        "",
        "## 1. 唯一主指标",
        "",
        f"`{report['primary_metric']}`（官方 CV-Bench 组合分）。",
        "",
        "## 2. 分数",
        "",
        "| 臂 | cv_bench_accuracy | 未加权 | parse |",
        "| --- | --- | --- | --- |",
    ]
    for label in ("new", "corrected_l2_integer", "dense"):
        row = metrics[label]
        lines.append(
            f"| {label} | {_fmt(row['cv_bench_accuracy'])} | "
            f"{_fmt(row['unweighted_sample_accuracy'])} | "
            f"{_fmt(row['parse_success_rate'])} |"
        )
    lines += [
        "",
        "## 3. 配对比较（new − corrected integer-L2）",
        "",
        f"- 点估计：{_fmt(comparison['delta_point_estimate'])}",
        f"- 95% CI：[{_fmt(comparison['delta_ci95'][0])}, "
        f"{_fmt(comparison['delta_ci95'][1])}]",
        f"- 分层：{comparison['stratified_by']}，seed {comparison['seed']}，"
        f"repeats {comparison['repeats']}",
        "",
        "## 4. 布局审计",
        "",
        f"- 压缩样本：{layout['compressed_count']}（其中 87 real + 9 context："
        f"{layout['full_size_87_9_count']}）",
        f"- identity bypass：{layout['bypass_count']}",
        f"- 发生交换的样本数：{layout['samples_with_swaps']}",
        "",
        "## 5. 门",
        "",
        "| 门 | 通过 |",
        "| --- | --- |",
    ]
    for name, passed in decision["gates"].items():
        lines.append(f"| {name} | {'是' if passed else '否'} |")
    lines += [
        "",
        "## 6. 输入 SHA256",
        "",
        "| 输入 | SHA256 |",
        "| --- | --- |",
    ]
    for name, digest in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{digest}` |")
    lines += ["", "## 7. 限制", ""]
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_reports(report: Mapping[str, Any], paths: InputPaths) -> None:
    paths.report_json.parent.mkdir(parents=True, exist_ok=True)
    paths.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths.report_md.write_text(render_markdown(report), encoding="utf-8")


def run(paths: InputPaths = DEFAULT_PATHS) -> dict[str, Any]:
    config = json.loads(paths.preregistered.read_text(encoding="utf-8"))
    verify_config_pins(config, paths)
    verify_output_paths(config, paths)
    manifest_rows = load_rows(paths.full_manifest, "full_manifest")
    dense_rows = load_rows(paths.dense_full_jsonl, "dense_full_jsonl")
    corrected_rows = load_rows(
        paths.corrected_l2_integer_jsonl, "corrected_l2_integer_jsonl"
    )
    new_rows = load_rows(paths.new_arm_jsonl, "new_arm_jsonl")
    report = analyze(
        config=config,
        manifest_rows=manifest_rows,
        dense_rows=dense_rows,
        corrected_l2_rows=corrected_rows,
        new_rows=new_rows,
        paths=paths,
    )
    write_reports(report, paths)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistered", type=Path, default=PREREGISTERED_CONFIG)
    parser.add_argument("--full-manifest", type=Path, default=FULL_MANIFEST)
    parser.add_argument("--dense-full-jsonl", type=Path, default=DENSE_FULL_JSONL)
    parser.add_argument(
        "--corrected-l2-integer-jsonl", type=Path, default=CORRECTED_L2_INTEGER_JSONL
    )
    parser.add_argument("--new-arm-jsonl", type=Path, default=NEW_ARM_JSONL)
    parser.add_argument("--report-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = InputPaths(
        preregistered=args.preregistered,
        full_manifest=args.full_manifest,
        dense_full_jsonl=args.dense_full_jsonl,
        corrected_l2_integer_jsonl=args.corrected_l2_integer_jsonl,
        new_arm_jsonl=args.new_arm_jsonl,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    try:
        report = run(paths)
    except InputContractError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
