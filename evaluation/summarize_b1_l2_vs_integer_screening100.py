#!/usr/bin/env python3
"""Diagnostic screening100 comparison: L2 vs current integer Selector vs dense.

This is the summariser for the preregistered experiment
``configs/2026-10-04-b1-l2-vs-integer-screening100.preregistered.json``.

Arms (all on the fixed 100-sample manifest
``datasets/cv_bench_screening100.jsonl``):

* ``dense``   - the frozen custom dense full2638 JSONL restricted to the 100 ids;
* ``current`` - the current integer Selector K=96 full2638 JSONL restricted to
  the 100 ids;
* ``l2``      - ``b1_l2_k96_screening100.jsonl`` produced by
  ``evaluation/run_hf_hardware_regular_b1.py --score-mode l2``.

Hard rules enforced here:

1. The **only** primary metric is the official
   ``score(...)["cv_bench_accuracy"]`` returned by the frozen
   :mod:`evaluation.build_cv_bench_pruning_report` scorer.  The plain per-sample
   mean is exposed strictly as ``unweighted_sample_accuracy`` and never drives
   the decision.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Inputs are validated before any score: the four inputs must select exactly
   the same 100 ``sample_id`` values with no duplicates; every L2 record must be
   ``score_mode == "l2"`` with ``requested_budget == 96``; and for every sample
   the L2 ``(image_grid_thw, full_patch_tokens, full_llm_visual_tokens)`` tuple
   must equal the current integer record's tuple.
4. **Fail closed**: a missing required input file, or any failed contract check,
   raises :class:`InputContractError` and no report is written.

Any result is a diagnostic attribution signal on a fixed 100-sample screening
set; it is explicitly **not** a formal full2638 CV-Bench conclusion.
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
    ROOT / "configs" / "2026-10-04-b1-l2-vs-integer-screening100.preregistered.json"
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
CURRENT_FULL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-current-selector-cvbench"
    / "current_selector_k96.jsonl"
)
L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-l2-diagnostic"
    / "b1_l2_k96_screening100.jsonl"
)
REPORT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-b1-l2-vs-integer-screening100.json"
)
REPORT_MD = ROOT / "reports" / "cv_bench" / "2026-10-04-b1-l2-vs-integer-screening100.md"

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = "unweighted_sample_accuracy"
SECONDARY_METRIC = 'score(...)["accuracy"]'
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

BUDGET = 96
SCREENING_SAMPLE_COUNT = 100
FULL_SAMPLE_COUNT = 2638
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
RANKER_DEFECT_CI_LOWER = 0.02
B1_PREMATURE_CI_UPPER = 0.02
RELATIVE_RETENTION_THRESHOLD = 0.92
L2_SCORE_MODE = "l2"

RANKER_DEFECT_PRIMARY = "RANKER_DEFECT_PRIMARY"
B1_PREMATURE_PRUNING_PRIMARY = "B1_PREMATURE_PRUNING_PRIMARY"
MIXED_INCONCLUSIVE = "MIXED_INCONCLUSIVE"

DECISION_ORDER = (
    RANKER_DEFECT_PRIMARY,
    B1_PREMATURE_PRUNING_PRIMARY,
    MIXED_INCONCLUSIVE,
)

DIAGNOSTIC_SCOPE = (
    "diagnostic screening100; NOT a formal full2638 CV-Bench conclusion"
)
GRID_SIGNATURE_DEFINITION = (
    "(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)"
)

FORBIDDEN_SOURCE_KEYS = ("pointing", "grounding", "ndcg", "retained_mass")


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the preregistered contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    current_full_jsonl: Path = CURRENT_FULL_JSONL
    l2_screening_jsonl: Path = L2_SCREENING_JSONL


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


def index_unique(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, dict[str, Any]]:
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
    """The per-sample input-grid signature used to prove L2 and current match."""

    if "image_grid_thw" not in row:
        raise InputContractError("record is missing image_grid_thw")
    if "full_patch_tokens" not in row:
        raise InputContractError("record is missing full_patch_tokens")
    if "full_llm_visual_tokens" not in row:
        raise InputContractError("record is missing full_llm_visual_tokens")
    return (
        tuple(int(value) for value in row["image_grid_thw"]),
        int(row["full_patch_tokens"]),
        int(row["full_llm_visual_tokens"]),
    )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def validate_arms(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    current_index: Mapping[str, Mapping[str, Any]],
    l2_index: Mapping[str, Mapping[str, Any]],
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
    if expected_full_count is not None:
        if len(dense_index) != expected_full_count:
            raise InputContractError(
                f"dense full JSONL must hold {expected_full_count} unique samples, "
                f"got {len(dense_index)}"
            )
        if len(current_index) != expected_full_count:
            raise InputContractError(
                f"current full JSONL must hold {expected_full_count} unique samples, "
                f"got {len(current_index)}"
            )

    screening_set = set(screening_ids)
    missing_dense = sorted(screening_set - set(dense_index))
    missing_current = sorted(screening_set - set(current_index))
    if missing_dense:
        raise InputContractError(
            f"dense full JSONL is missing screening sample_ids: {missing_dense[:10]}"
        )
    if missing_current:
        raise InputContractError(
            f"current full JSONL is missing screening sample_ids: {missing_current[:10]}"
        )

    l2_set = set(l2_index)
    missing_l2 = sorted(screening_set - l2_set)
    extra_l2 = sorted(l2_set - screening_set)
    if missing_l2 or extra_l2:
        raise InputContractError(
            "L2 screening100 sample_id set does not equal the screening manifest: "
            f"missing={missing_l2[:10]} extra={extra_l2[:10]}"
        )

    bad_score_mode = sorted(
        sid
        for sid, row in l2_index.items()
        if str(row.get("score_mode")) != L2_SCORE_MODE
    )
    if bad_score_mode:
        raise InputContractError(
            "L2 records must all have score_mode == "
            f"'{L2_SCORE_MODE}'; offenders: {bad_score_mode[:10]}"
        )

    bad_budget: list[str] = []
    for sid, row in l2_index.items():
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
            f"L2 records must all have requested_budget == {BUDGET}; "
            f"offenders: {sorted(bad_budget)[:10]}"
        )

    grid_mismatches: list[dict[str, Any]] = []
    for sid in screening_ids:
        l2_sig = grid_signature(l2_index[sid])
        current_sig = grid_signature(current_index[sid])
        if l2_sig != current_sig:
            grid_mismatches.append(
                {
                    "sample_id": sid,
                    "l2": list(l2_sig[0]) + [l2_sig[1], l2_sig[2]],
                    "current": list(current_sig[0]) + [current_sig[1], current_sig[2]],
                }
            )
    if grid_mismatches:
        raise InputContractError(
            "L2 input grid is not consistent with the current integer input for "
            f"{len(grid_mismatches)} samples; first: {grid_mismatches[0]}"
        )

    return {
        "screening_sample_count": len(screening_ids),
        "dense_full_sample_count": len(dense_index),
        "current_full_sample_count": len(current_index),
        "l2_sample_count": len(l2_index),
        "sample_ids_identical": True,
        "duplicate_sample_ids": {"screening": [], "dense": [], "current": [], "l2": []},
        "l2_score_mode_all_l2": True,
        "l2_requested_budget_all_96": True,
        "input_grid_consistent_with_current": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "checks": [
            "100 identical unique sample_ids across manifest/dense/current/l2",
            "no duplicate sample_id in any input",
            'every L2 record score_mode == "l2"',
            f"every L2 record requested_budget == {BUDGET}",
            "L2 input grid == current input grid for every sample",
        ],
    }


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


def decide(
    l2_minus_current_ci: Sequence[float],
    l2_relative_retention: float,
    current_relative_retention: float,
) -> dict[str, Any]:
    """Apply the preregistered ordered decision rule."""

    lower, upper = float(l2_minus_current_ci[0]), float(l2_minus_current_ci[1])
    if lower > RANKER_DEFECT_CI_LOWER:
        outcome = RANKER_DEFECT_PRIMARY
        reason = f"l2_minus_current CI lower {lower:.6f} > {RANKER_DEFECT_CI_LOWER}"
    elif (
        upper <= B1_PREMATURE_CI_UPPER
        and l2_relative_retention < RELATIVE_RETENTION_THRESHOLD
        and current_relative_retention < RELATIVE_RETENTION_THRESHOLD
    ):
        outcome = B1_PREMATURE_PRUNING_PRIMARY
        reason = (
            f"l2_minus_current CI upper {upper:.6f} <= {B1_PREMATURE_CI_UPPER} and "
            f"l2 retention {l2_relative_retention:.6f} < {RELATIVE_RETENTION_THRESHOLD} "
            f"and current retention {current_relative_retention:.6f} < "
            f"{RELATIVE_RETENTION_THRESHOLD}"
        )
    else:
        outcome = MIXED_INCONCLUSIVE
        reason = (
            f"no rule matched: ci=[{lower:.6f}, {upper:.6f}], "
            f"l2 retention={l2_relative_retention:.6f}, "
            f"current retention={current_relative_retention:.6f}"
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "l2_minus_current_ci95": [lower, upper],
        "l2_relative_retention_vs_dense": float(l2_relative_retention),
        "current_relative_retention_vs_dense": float(current_relative_retention),
        "thresholds": {
            "ranker_defect_ci_lower": RANKER_DEFECT_CI_LOWER,
            "b1_premature_ci_upper": B1_PREMATURE_CI_UPPER,
            "min_relative_retention": RELATIVE_RETENTION_THRESHOLD,
        },
        "uses_only_primary_metric": True,
    }


def analyze(
    *,
    screening_rows: Sequence[Mapping[str, Any]],
    dense_full_rows: Sequence[Mapping[str, Any]],
    current_full_rows: Sequence[Mapping[str, Any]],
    l2_rows: Sequence[Mapping[str, Any]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate inputs and compute the diagnostic result (no file I/O)."""

    screening_index = index_unique(screening_rows, "screening_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    current_index = index_unique(current_full_rows, "current_selector_k96_full2638")
    l2_index = index_unique(l2_rows, "l2_screening100")

    validation = validate_arms(
        screening_index=screening_index,
        dense_index=dense_index,
        current_index=current_index,
        l2_index=l2_index,
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    joined = build_joined(
        list(screening_index),
        screening_index,
        {"dense": dense_index, "current": current_index, "l2": l2_index},
    )

    metrics = {
        "dense": arm_metrics(joined, "dense", "dense"),
        "current": arm_metrics(joined, "current", "current_integer"),
        "l2": arm_metrics(joined, "l2", "l2"),
    }

    dense_primary = metrics["dense"]["cv_bench_accuracy"]
    if dense_primary == 0:
        raise InputContractError("dense cv_bench_accuracy is 0; relative retention undefined")
    l2_relative = metrics["l2"]["cv_bench_accuracy"] / dense_primary
    current_relative = metrics["current"]["cv_bench_accuracy"] / dense_primary

    comparisons = {
        "l2_minus_current": paired_comparison(joined, "l2", "current"),
        "l2_minus_dense": paired_comparison(joined, "l2", "dense"),
        "current_minus_dense": paired_comparison(joined, "current", "dense"),
    }

    decision = decide(
        comparisons["l2_minus_current"]["delta_ci95"],
        l2_relative,
        current_relative,
    )

    return {
        "validation": validation,
        "joined": joined,
        "metrics": metrics,
        "relative_retention_vs_dense": {
            "l2": l2_relative,
            "current": current_relative,
            "definition": "arm cv_bench_accuracy / dense cv_bench_accuracy",
        },
        "comparisons": comparisons,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
) -> dict[str, Any]:
    """Load every required input, validate, score and assemble the report.

    Any missing file or failed contract check raises :class:`InputContractError`
    before a report is produced (fail closed).
    """

    require_file(paths.preregistered, "preregistered_config")
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
        current_full_rows=load_rows(paths.current_full_jsonl, "current_selector_k96"),
        l2_rows=load_rows(paths.l2_screening_jsonl, "l2_screening100"),
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
    )

    inputs_sha256 = {
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
        "current_full_jsonl": {
            "path": str(paths.current_full_jsonl),
            "sha256": sha256_file(paths.current_full_jsonl),
        },
        "l2_screening_jsonl": {
            "path": str(paths.l2_screening_jsonl),
            "sha256": sha256_file(paths.l2_screening_jsonl),
        },
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": True,
        "formal_full2638_conclusion": False,
        "explicit_non_formal_statement": (
            "Fixed stratified 100-sample screening: an attribution signal only, "
            "NOT a formal full2638 CV-Bench conclusion. It must not be quoted as "
            "the official CV-Bench accuracy of any approach."
        ),
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
            "budget_K": BUDGET,
            "boundary": "B1 = after vision block 0",
            "region_grid": 3,
            "local_candidate_cap": 16,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "decoding": "greedy_argmax",
            "do_sample": False,
            "score_mode_l2": L2_SCORE_MODE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
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
            "dense": str(paths.dense_full_jsonl),
            "current_integer": str(paths.current_full_jsonl),
            "l2": str(paths.l2_screening_jsonl),
        },
        "validation": analysis["validation"],
        "metrics": analysis["metrics"],
        "relative_retention_vs_dense": analysis["relative_retention_vs_dense"],
        "comparisons": analysis["comparisons"],
        "decision": analysis["decision"],
        "inputs_sha256": inputs_sha256,
        "limitations": [
            "Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.",
            "Software Selector path only; no FPGA, latency or acceleration claim.",
            "Dense is the frozen custom dense path and is only a reference.",
            "The decision uses score(...)[\"cv_bench_accuracy\"] exclusively.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.",
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
    lines.append("# B1 L2 vs current integer Selector（screening100 诊断）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append(
        "> ⚠️ " + report["explicit_non_formal_statement"]
    )
    lines.append("")
    lines.append(
        "- 正式主指标（唯一）：`" + report["primary_metric"]["identifier"] + "` "
        "= `((ADE20K + COCO)/2 + Omni3D)/2`；"
        "`" + report["unweighted_metric"]["identifier"] + "` 仅为次要诊断，"
        "**不得写作 CV-Bench 主分**，也不参与决策。"
    )
    lines.append(
        "- 禁用主指标：" + ", ".join(report["forbidden_primary_metrics"])
    )
    lines.append("")
    lines.append("## 1. 固定配置")
    lines.append("")
    cfg = report["fixed_configuration"]
    lines.append(
        f"- 样本：`{cfg['manifest']}`；K={cfg['budget_K']}；切点：{cfg['boundary']}；"
        f"region_grid={cfg['region_grid']}；local_cap={cfg['local_candidate_cap']}"
    )
    lines.append(
        f"- dtype={cfg['dtype']}；attention={cfg['attention_implementation']}；"
        f"decoding={cfg['decoding']}；bootstrap seed={cfg['bootstrap_seed']} / "
        f"repeats={cfg['bootstrap_repeats']}"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- screening 样本数：{v['screening_sample_count']}")
    lines.append(f"- L2 样本数：{v['l2_sample_count']}")
    lines.append(f"- 100 个 sample_id 完全一致：{'✅' if v['sample_ids_identical'] else '❌'}")
    lines.append(f"- 无重复 sample_id：{'✅' if not any(v['duplicate_sample_ids'].values()) else '❌'}")
    lines.append(
        f"- L2 score_mode 全为 `{L2_SCORE_MODE}`：{'✅' if v['l2_score_mode_all_l2'] else '❌'}"
    )
    lines.append(
        f"- L2 requested_budget 全为 {BUDGET}：{'✅' if v['l2_requested_budget_all_96'] else '❌'}"
    )
    lines.append(
        f"- L2 输入 grid 与 current 一致（{v['grid_signature_definition']}）："
        f"{'✅' if v['input_grid_consistent_with_current'] else '❌'}"
    )
    lines.append("")
    lines.append("## 3. 三臂正式主分")
    lines.append("")
    lines.append(
        "| 臂 | 样本 | **CV-Bench 主分**（`cv_bench_accuracy`） | "
        "unweighted_sample_accuracy（次要） | 相对 dense 保留 |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    for arm in ("dense", "current", "l2"):
        m = report["metrics"][arm]
        retention = (
            "1.000000"
            if arm == "dense"
            else _pct(report["relative_retention_vs_dense"][arm])
        )
        lines.append(
            f"| {m['label']} | {m['sample_count']} | **{_pct(m['cv_bench_accuracy'])}** | "
            f"{_pct(m['unweighted_sample_accuracy'])} | {retention} |"
        )
    lines.append("")
    lines.append("## 4. 配对分层 bootstrap（95% CI，主指标）")
    lines.append("")
    lines.append("| 对比 | 点估计 | 95% CI 下界 | 95% CI 上界 |")
    lines.append("|---|---:|---:|---:|")
    for name in ("l2_minus_current", "l2_minus_dense", "current_minus_dense"):
        c = report["comparisons"][name]
        lines.append(
            f"| {c['left']} − {c['right']} | {c['delta_point_estimate']:+.6f} | "
            f"{c['delta_ci95'][0]:+.6f} | {c['delta_ci95'][1]:+.6f} |"
        )
    lines.append("")
    lines.append("## 5. 预注册决策")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 结论：**{d['outcome']}**")
    lines.append(f"- 依据：{d['reason']}")
    lines.append(
        f"- L2−current 95% CI：[{d['l2_minus_current_ci95'][0]:+.6f}, "
        f"{d['l2_minus_current_ci95'][1]:+.6f}]；"
        f"L2 相对 dense 保留={_pct(d['l2_relative_retention_vs_dense'])}；"
        f"current 相对 dense 保留={_pct(d['current_relative_retention_vs_dense'])}"
    )
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
