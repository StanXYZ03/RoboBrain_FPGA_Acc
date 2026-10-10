#!/usr/bin/env python3
"""Out-of-fold, deployment-observable G-gated Prompt diagnostic (full2638).

Exploratory model selection, **not** a new formal claim.

Objective
---------
Test whether a hardware-trivial gate that only observes the pre-selector group
count ``G = full_llm_visual_tokens`` can choose between the two already-frozen
full2638 arms *without* any task/source/dataset identity:

* integer-L2 arm: ``corrected_l2_integer`` (a.k.a. ``never_prompt``);
* Prompt+integer-L2 arm: ``prompt_plus_corrected_l2_integer`` (``always_prompt``).

Candidate policies are fixed before scoring and depend only on ``G``:

``never_prompt``, ``always_prompt``,
``prompt_if_G_gt_96``, ``prompt_if_G_gt_256``, ``prompt_if_G_gt_512``,
``prompt_if_G_gt_1024``.

Design (frozen in this module before any scoring)
-------------------------------------------------
1. Deterministic 5-fold split stratified by ``task::source_dataset`` with seed
   ``20261004``; every sample appears in exactly one held-out fold.
2. In each fold the single best policy is selected using the *other four folds*
   only, maximising the official ``score(...)["cv_bench_accuracy"]``; exact
   accuracy ties prefer the lower Prompt-use rate, then the fixed policy order.
3. The selected policy is applied unchanged to the held-out fold; pooled
   held-out predictions form the out-of-fold (OOF) policy.
4. OOF is compared with ``never_prompt``, ``always_prompt`` and the dense
   descriptive reference; the paired stratified bootstrap (seed 20261004,
   5000 repeats) yields the OOF-minus-never and OOF-minus-always CIs.

Hard rules
----------
1. The **only** metric is the official
   :func:`evaluation.build_cv_bench_pruning_report.score`
   ``cv_bench_accuracy``; dense is descriptive and never a gate.
2. This is threshold discovery on the same full2638 corpus under
   cross-validation.  It therefore still needs an external frozen dataset
   before it may be promoted to a paper/formal claim.
3. Any missing file, SHA pin mismatch, sample-id / config / record-integrity
   mismatch raises :class:`InputContractError` before anything is written
   (fail closed).  No prior report/config/raw result is modified.

All parsing/scoring/statistics are imported from the frozen helpers; this module
never imports ``torch`` and never runs a GPU, model, network or training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import analyze_full2638_prompt_gain_heterogeneity as het  # noqa: E402
from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    paired_bootstrap_delta,
    score,
)

ROOT = Path(__file__).resolve().parents[1]

FORMAL_REPORT_JSON = het.FORMAL_REPORT_JSON
FULL_MANIFEST = het.FULL_MANIFEST
ARM_INTEGER_JSONL = het.ARM_INTEGER_JSONL
ARM_PROMPT_JSONL = het.ARM_PROMPT_JSONL

REPORT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-full2638-oof-gated-prompt.json"
)
REPORT_CSV = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-full2638-oof-gated-prompt.csv"
)
REPORT_MD = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-full2638-oof-gated-prompt.md"
)

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = 'score(...)["accuracy"] (unweighted sample accuracy)'
FORMAL_OUTCOME = het.FORMAL_OUTCOME
REQUIRED_SOURCES = frozenset({"ADE20K", "COCO", "Omni3D"})

FOLDS = 5
FOLD_SEED = 20261004
BOOTSTRAP_SEED = het.BOOTSTRAP_SEED
BOOTSTRAP_REPEATS = het.BOOTSTRAP_REPEATS
CI_LEVEL = het.CI_LEVEL
TIE_TOLERANCE = 1e-12

ARM_NEVER = "never_prompt"
ARM_ALWAYS = "always_prompt"
GATED_POLICIES = (
    "prompt_if_G_gt_96",
    "prompt_if_G_gt_256",
    "prompt_if_G_gt_512",
    "prompt_if_G_gt_1024",
)
POLICY_ORDER = (ARM_NEVER, ARM_ALWAYS) + GATED_POLICIES
GATED_THRESHOLDS = {
    "prompt_if_G_gt_96": 96,
    "prompt_if_G_gt_256": 256,
    "prompt_if_G_gt_512": 512,
    "prompt_if_G_gt_1024": 1024,
}
POLICY_RULE = (
    "never_prompt uses the integer-L2 arm on every sample; always_prompt uses "
    "the Prompt+integer-L2 arm on every sample; prompt_if_G_gt_T uses the "
    "Prompt arm iff the pre-selector G = full_llm_visual_tokens is strictly "
    "greater than T, else the integer-L2 arm. No task/source/dataset identity "
    "enters any policy."
)
SELECTION_RULE = (
    "In each fold keep the other four folds, score all six candidate policies "
    "with the official cv_bench_accuracy, pick the maximum; exact ties prefer "
    "the lower Prompt-use rate, then the fixed order "
    "(never_prompt, always_prompt, prompt_if_G_gt_96, prompt_if_G_gt_256, "
    "prompt_if_G_gt_512, prompt_if_G_gt_1024). Apply the winner unchanged to "
    "the held-out fold."
)
EXTERNAL_VALIDATION_CAVEAT = (
    "Threshold discovery here uses the same full2638 corpus under 5-fold "
    "cross-validation, so the OOF score is an internally cross-validated "
    "estimate, not an untouched external evaluation. A paper/formal claim "
    "still requires replaying the frozen policy on an external frozen dataset."
)
EXTERNAL_VALIDATION_CAVEAT_ZH = (
    "阈值发现与评估共用同一 full2638 语料（5-fold 交叉验证），因此 OOF 分数只是"
    "内部交叉验证估计，不是未触碰的外部评测；在提出论文/正式主张之前，必须把冻结"
    "策略在外部冻结数据集上重放。"
)

CSV_COLUMNS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "G",
    "G_bin",
    "identity_bypass",
    "fold",
    "selected_policy",
    "used_prompt",
    "oof_answer",
    "oof_correct",
    "never_answer",
    "never_correct",
    "always_answer",
    "always_correct",
    "dense_answer",
    "dense_correct",
    "transition_vs_never",
    "transition_vs_always",
)

TRANSITIONS = (
    "wrong_to_right",
    "right_to_wrong",
    "unchanged_right",
    "unchanged_wrong",
)


class InputContractError(het.InputContractError):
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
DEFAULT_INPUT_PINS = het.DEFAULT_INPUT_PINS


@dataclass(frozen=True)
class OutputPaths:
    json: Path = REPORT_JSON
    csv: Path = REPORT_CSV
    md: Path = REPORT_MD


DEFAULT_OUTPUTS = OutputPaths()


# ---------------------------------------------------------------------------
# Candidate policies (fixed before scoring)
# ---------------------------------------------------------------------------
def policy_uses_prompt(policy: str, group_count: int) -> bool:
    """Return whether ``policy`` selects the Prompt arm for this ``G``."""

    if policy == ARM_NEVER:
        return False
    if policy == ARM_ALWAYS:
        return True
    if policy in GATED_THRESHOLDS:
        return int(group_count) > GATED_THRESHOLDS[policy]
    raise ValueError(f"unknown policy {policy!r}")


# ---------------------------------------------------------------------------
# Deterministic stratified folds
# ---------------------------------------------------------------------------
def stratum_key(record: Mapping[str, Any]) -> str:
    return f"{record['task']}::{record['source_dataset']}"


def _sample_sort_key(sample_id: str) -> tuple[int, Any]:
    text = str(sample_id)
    if text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


def assign_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    folds: int = FOLDS,
    seed: int = FOLD_SEED,
) -> dict[str, int]:
    """Deterministic stratified fold assignment.

    Strata ``task::source_dataset`` are processed in sorted order; inside each
    stratum the sample ids are sorted by their numeric value (falling back to
    lexicographic) and shuffled with one shared seeded RNG, then dealt
    round-robin across the folds.  Every sample therefore lands in exactly one
    held-out fold and each stratum is spread as evenly as possible.
    """

    if folds < 2:
        raise ValueError("folds must be >= 2")
    strata: dict[str, list[str]] = defaultdict(list)
    for record in records:
        strata[stratum_key(record)].append(str(record["sample_id"]))
    rng = random.Random(seed)
    fold_of: dict[str, int] = {}
    for key in sorted(strata):
        ids = sorted(strata[key], key=_sample_sort_key)
        rng.shuffle(ids)
        for position, sample_id in enumerate(ids):
            fold_of[sample_id] = position % folds
    if len(fold_of) != len(records):
        raise InputContractError(
            "fold assignment lost or duplicated sample ids (fail closed)"
        )
    return fold_of


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _score_rows(
    records: Sequence[Mapping[str, Any]],
    correctness_key: str,
    answer_key: str,
) -> list[dict[str, Any]]:
    return [
        {
            "task": record["task"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "correct": bool(record[correctness_key]),
            "answer": record[answer_key],
        }
        for record in records
    ]


def official_accuracy(
    rows: Sequence[Mapping[str, Any]], correctness_key: str = "correct"
) -> float | None:
    """Official ``cv_bench_accuracy``; ``None`` if a required source is absent."""

    if {str(row["source"]) for row in rows} < REQUIRED_SOURCES:
        return None
    return float(score(list(rows), correctness_key)["cv_bench_accuracy"])


def policy_rows(
    records: Sequence[Mapping[str, Any]], policy: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        use_prompt = policy_uses_prompt(policy, record["G"])
        rows.append(
            {
                "task": record["task"],
                "source": record["source"],
                "source_dataset": record["source_dataset"],
                "correct": bool(
                    record["always_correct"] if use_prompt else record["never_correct"]
                ),
                "answer": (
                    record["always_answer"] if use_prompt else record["never_answer"]
                ),
            }
        )
    return rows


def policy_cv_bench_accuracy(
    records: Sequence[Mapping[str, Any]], policy: str
) -> float | None:
    return official_accuracy(policy_rows(records, policy), "correct")


def prompt_use_rate(records: Sequence[Mapping[str, Any]], policy: str) -> float:
    return statistics.fmean(
        policy_uses_prompt(policy, record["G"]) for record in records
    )


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_policy(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Pick the best candidate under the fixed tie-break rule.

    ``candidates`` must be ordered by the fixed policy order and each carry
    ``policy``, ``train_cv_bench_accuracy`` and ``train_prompt_use_rate``.
    """

    best: Mapping[str, Any] | None = None
    for candidate in candidates:
        if best is None:
            best = candidate
            continue
        accuracy = float(candidate["train_cv_bench_accuracy"])
        best_accuracy = float(best["train_cv_bench_accuracy"])
        rate = float(candidate["train_prompt_use_rate"])
        best_rate = float(best["train_prompt_use_rate"])
        if accuracy > best_accuracy + TIE_TOLERANCE:
            best = candidate
        elif abs(accuracy - best_accuracy) <= TIE_TOLERANCE:
            if rate < best_rate - TIE_TOLERANCE:
                best = candidate
            elif abs(rate - best_rate) <= TIE_TOLERANCE:
                continue  # keep the earlier policy in the fixed order
    if best is None:
        raise InputContractError("no candidate policy supplied (fail closed)")
    return str(best["policy"])


# ---------------------------------------------------------------------------
# Transition / strata diagnostics
# ---------------------------------------------------------------------------
def transition_label(
    left_correct: bool, right_correct: bool
) -> str:
    """``right`` is the reference; ``left`` is the compared policy."""

    if not right_correct and left_correct:
        return "wrong_to_right"
    if right_correct and not left_correct:
        return "right_to_wrong"
    if right_correct and left_correct:
        return "unchanged_right"
    return "unchanged_wrong"


def transition_counts(
    records: Sequence[Mapping[str, Any]], reference_key: str
) -> dict[str, int]:
    counts = {name: 0 for name in TRANSITIONS}
    for record in records:
        label = transition_label(
            bool(record["oof_correct"]), bool(record[reference_key])
        )
        counts[label] += 1
    counts["net_corrections"] = counts["wrong_to_right"] - counts["right_to_wrong"]
    return counts


def stratum_summary(
    label: str, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    n = len(records)
    oof_rows = _score_rows(records, "oof_correct", "oof_answer")
    never_rows = _score_rows(records, "never_correct", "never_answer")
    always_rows = _score_rows(records, "always_correct", "always_answer")
    dense_rows = _score_rows(records, "dense_correct", "dense_answer")
    oof_official = official_accuracy(oof_rows)
    oof_unweighted = statistics.fmean(bool(record["oof_correct"]) for record in records)
    never_unweighted = statistics.fmean(
        bool(record["never_correct"]) for record in records
    )
    always_unweighted = statistics.fmean(
        bool(record["always_correct"]) for record in records
    )
    dense_unweighted = statistics.fmean(
        bool(record["dense_correct"]) for record in records
    )
    return {
        "stratum": label,
        "n": n,
        "prompt_use_count": sum(bool(record["used_prompt"]) for record in records),
        "prompt_use_rate": statistics.fmean(
            bool(record["used_prompt"]) for record in records
        ),
        "oof_official_cv_bench_accuracy": oof_official,
        "oof_unweighted_accuracy": oof_unweighted,
        "never_unweighted_accuracy": never_unweighted,
        "always_unweighted_accuracy": always_unweighted,
        "dense_unweighted_accuracy": dense_unweighted,
        "oof_minus_never_unweighted": oof_unweighted - never_unweighted,
        "oof_minus_always_unweighted": oof_unweighted - always_unweighted,
        "transitions_vs_never": transition_counts(records, "never_correct"),
        "transitions_vs_always": transition_counts(records, "always_correct"),
        "selected_policies": sorted({str(record["selected_policy"]) for record in records}),
    }


def _group_summaries(
    records: Sequence[Mapping[str, Any]],
    key: str,
    order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    labels = sorted({str(record[key]) for record in records})
    if order is not None:
        labels = [label for label in order if label in set(labels)]
    return [
        stratum_summary(label, [r for r in records if str(r[key]) == label])
        for label in labels
    ]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def _flag_rows(
    records: Sequence[Mapping[str, Any]], correctness_key: str, answer_key: str
) -> list[dict[str, Any]]:
    return _score_rows(records, correctness_key, answer_key)


def _cross_check_reproduction(
    report: Mapping[str, Any],
    *,
    never_official: float | None,
    always_official: float | None,
    dense_official: float | None,
) -> dict[str, Any]:
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise InputContractError(
            "formal report is missing metrics; cannot cross-check reproduced "
            "baselines (fail closed)"
        )

    def _metric(key: str) -> float:
        spec = metrics.get(key)
        if not isinstance(spec, Mapping) or spec.get("cv_bench_accuracy") is None:
            raise InputContractError(
                f"formal report metrics[{key!r}].cv_bench_accuracy missing "
                "(fail closed)"
            )
        return float(spec["cv_bench_accuracy"])

    checks = {
        "never_matches_formal_integer": (
            never_official is not None
            and abs(never_official - _metric("corrected_l2_integer")) <= 1e-12
        ),
        "always_matches_formal_prompt": (
            always_official is not None
            and abs(always_official - _metric("prompt_plus_corrected_l2_integer"))
            <= 1e-12
        ),
        "dense_matches_formal_dense": (
            dense_official is not None
            and abs(dense_official - _metric("dense")) <= 1e-12
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise InputContractError(
            "recomputed baselines do not reproduce the frozen formal report: "
            f"{failed} (fail closed)"
        )
    return checks


def analyze(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_pins: Mapping[str, str] | None = DEFAULT_INPUT_PINS,
    expected_sample_count: int = het.FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    folds: int = FOLDS,
    fold_seed: int = FOLD_SEED,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> dict[str, Any]:
    """Validate the frozen inputs and compute the OOF G-gate diagnostic."""

    pinned = het.verify_pins(paths, expected_pins)

    manifest_rows = het.load_rows(paths.full_manifest, "full_manifest")
    manifest_index = het.index_unique(manifest_rows, "full_manifest")
    arm_indices = {
        het.ARM_INTEGER: het.index_unique(
            het.load_rows(paths.arm_integer_jsonl, "corrected_l2_integer"),
            "corrected_l2_integer",
        ),
        het.ARM_PROMPT: het.index_unique(
            het.load_rows(
                paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"
            ),
            "prompt_plus_corrected_l2_integer",
        ),
    }

    integer_sha = pinned["corrected_l2_integer_arm"]["sha256"]
    prompt_sha = pinned["prompt_plus_corrected_l2_integer_arm"]["sha256"]
    actual_manifest_sha = (
        manifest_sha256
        if manifest_sha256 is not None
        else pinned["full_manifest"]["sha256"]
    )

    if not paths.formal_report.is_file():
        raise InputContractError(
            f"missing required input 'formal_report': {paths.formal_report}"
        )
    formal_report = json.loads(paths.formal_report.read_text(encoding="utf-8"))

    formal_checks = het.verify_formal_report(
        formal_report,
        manifest_sha256=actual_manifest_sha,
        integer_sha256=integer_sha,
        prompt_sha256=prompt_sha,
        expected_sample_count=expected_sample_count,
    )
    checkpoint_hashes = het.extract_checkpoint_hashes(formal_report)

    contract = het.validate_contract(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        manifest_sha256=actual_manifest_sha,
        checkpoint_hashes=checkpoint_hashes,
        expected_sample_count=expected_sample_count,
    )

    formal_joined = formal_report.get("joined")
    if formal_joined is not None:
        formal_joined_index = het.index_unique(
            formal_joined, "formal_report.joined"
        )
    else:
        formal_joined_index = None

    joined = het.build_joined(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        formal_joined_index=formal_joined_index,
    )

    records: list[dict[str, Any]] = []
    for row in joined:
        if row.get("dense_correct") is None:
            raise InputContractError(
                f"sample {row['sample_id']!r} lacks a frozen dense prediction; "
                "the dense descriptive baseline cannot be reproduced (fail closed)"
            )
        records.append(
            {
                "sample_id": str(row["sample_id"]),
                "task": str(row["task"]),
                "source": str(row["source"]),
                "source_dataset": str(row["source_dataset"]),
                "G": int(row["G"]),
                "identity_bypass": bool(row["identity_bypass"]),
                "never_correct": bool(row[f"{het.ARM_INTEGER}_correct"]),
                "never_answer": row[f"{het.ARM_INTEGER}_answer"],
                "always_correct": bool(row[f"{het.ARM_PROMPT}_correct"]),
                "always_answer": row[f"{het.ARM_PROMPT}_answer"],
                "dense_correct": bool(row["dense_correct"]),
                "dense_answer": row["dense_answer"],
            }
        )

    fold_of = assign_folds(records, folds=folds, seed=fold_seed)

    fold_selections: list[dict[str, Any]] = []
    selected_by_fold: dict[int, str] = {}
    for fold in range(folds):
        train = [r for r in records if fold_of[r["sample_id"]] != fold]
        held_out = [r for r in records if fold_of[r["sample_id"]] == fold]
        candidates: list[dict[str, Any]] = []
        for policy in POLICY_ORDER:
            accuracy = policy_cv_bench_accuracy(train, policy)
            if accuracy is None:
                raise InputContractError(
                    f"fold {fold}: official cv_bench_accuracy undefined for "
                    f"policy {policy!r}; a required source is missing from the "
                    "training folds (fail closed)"
                )
            candidates.append(
                {
                    "policy": policy,
                    "train_cv_bench_accuracy": float(accuracy),
                    "train_prompt_use_rate": prompt_use_rate(train, policy),
                }
            )
        selected = select_policy(candidates)
        selected_by_fold[fold] = selected
        held_accuracy = policy_cv_bench_accuracy(held_out, selected)
        fold_selections.append(
            {
                "fold": fold,
                "train_sample_count": len(train),
                "held_out_sample_count": len(held_out),
                "selected_policy": selected,
                "selected_policy_train_cv_bench_accuracy": float(
                    next(
                        c["train_cv_bench_accuracy"]
                        for c in candidates
                        if c["policy"] == selected
                    )
                ),
                "selected_policy_held_out_cv_bench_accuracy": (
                    float(held_accuracy) if held_accuracy is not None else None
                ),
                "held_out_prompt_use_rate": prompt_use_rate(held_out, selected),
                "candidates": candidates,
            }
        )

    oof_records: list[dict[str, Any]] = []
    for record in records:
        policy = selected_by_fold[fold_of[record["sample_id"]]]
        use_prompt = policy_uses_prompt(policy, record["G"])
        oof_records.append(
            {
                **record,
                "fold": fold_of[record["sample_id"]],
                "selected_policy": policy,
                "used_prompt": use_prompt,
                "oof_correct": (
                    record["always_correct"] if use_prompt else record["never_correct"]
                ),
                "oof_answer": (
                    record["always_answer"] if use_prompt else record["never_answer"]
                ),
                "transition_vs_never": transition_label(
                    bool(record["always_correct"] if use_prompt else record["never_correct"]),
                    bool(record["never_correct"]),
                ),
                "transition_vs_always": transition_label(
                    bool(record["always_correct"] if use_prompt else record["never_correct"]),
                    bool(record["always_correct"]),
                ),
            }
        )

    never_official = official_accuracy(
        _flag_rows(oof_records, "never_correct", "never_answer")
    )
    always_official = official_accuracy(
        _flag_rows(oof_records, "always_correct", "always_answer")
    )
    dense_official = official_accuracy(
        _flag_rows(oof_records, "dense_correct", "dense_answer")
    )
    oof_official = official_accuracy(
        _flag_rows(oof_records, "oof_correct", "oof_answer")
    )
    if None in (never_official, always_official, dense_official, oof_official):
        raise InputContractError(
            "official cv_bench_accuracy undefined on the pooled corpus "
            "(a required source is missing; fail closed)"
        )

    reproduction = _cross_check_reproduction(
        formal_report,
        never_official=never_official,
        always_official=always_official,
        dense_official=dense_official,
    )

    bootstrap_rows = [
        {
            "task": record["task"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "oof_correct": record["oof_correct"],
            "oof_answer": record["oof_answer"],
            "never_correct": record["never_correct"],
            "never_answer": record["never_answer"],
            "always_correct": record["always_correct"],
            "always_answer": record["always_answer"],
        }
        for record in oof_records
    ]
    ci_never = paired_bootstrap_delta(
        bootstrap_rows, "oof_correct", "never_correct", bootstrap_seed,
        repeats=bootstrap_repeats,
    )
    ci_always = paired_bootstrap_delta(
        bootstrap_rows, "oof_correct", "always_correct", bootstrap_seed,
        repeats=bootstrap_repeats,
    )

    transitions_never = transition_counts(oof_records, "never_correct")
    transitions_always = transition_counts(oof_records, "always_correct")

    selection_counts = Counter(selected_by_fold.values())
    selection_stable = len(selection_counts) == 1

    overall = stratum_summary("overall", oof_records)
    overall.update(
        {
            "oof_official_cv_bench_accuracy": oof_official,
            "never_official_cv_bench_accuracy": never_official,
            "always_official_cv_bench_accuracy": always_official,
            "dense_official_cv_bench_accuracy": dense_official,
        }
    )

    by_task = _group_summaries(oof_records, "task")
    by_source = _group_summaries(oof_records, "source")
    by_source_dataset = _group_summaries(oof_records, "source_dataset")
    by_g_bin = _group_summaries(
        [dict(record, G_bin=het.pre_selector_bin(record["G"])) for record in oof_records],
        "G_bin",
        order=het.PRE_SELECTOR_BIN_ORDER,
    )

    # Re-verify read-only inputs after the analysis (nothing may have changed).
    for name, spec in pinned.items():
        current = het.sha256_file(Path(spec["path"]))
        if current != spec["sha256"]:
            raise InputContractError(
                f"input '{name}' changed during analysis: {spec['sha256']} -> "
                f"{current} (fail closed)"
            )

    per_sample = [
        {
            "sample_id": record["sample_id"],
            "task": record["task"],
            "source": record["source"],
            "source_dataset": record["source_dataset"],
            "G": record["G"],
            "G_bin": het.pre_selector_bin(record["G"]),
            "identity_bypass": record["identity_bypass"],
            "fold": record["fold"],
            "selected_policy": record["selected_policy"],
            "used_prompt": record["used_prompt"],
            "oof_answer": record["oof_answer"],
            "oof_correct": record["oof_correct"],
            "never_answer": record["never_answer"],
            "never_correct": record["never_correct"],
            "always_answer": record["always_answer"],
            "always_correct": record["always_correct"],
            "dense_answer": record["dense_answer"],
            "dense_correct": record["dense_correct"],
            "transition_vs_never": record["transition_vs_never"],
            "transition_vs_always": record["transition_vs_always"],
        }
        for record in oof_records
    ]

    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "analysis": "full2638_oof_gated_prompt",
        "scope": (
            "Exploratory out-of-fold G-gated Prompt model selection on the "
            "frozen full2638 corrected-L2 arms; CPU-only, read-only; NOT a new "
            "formal claim and NOT a gate."
        ),
        "exploratory_only": True,
        "not_a_formal_claim": True,
        "formal_result_unchanged": True,
        "formal_outcome": FORMAL_OUTCOME,
        "primary_metric": PRIMARY_METRIC,
        "unweighted_metric": UNWEIGHTED_METRIC,
        "external_validation_caveat": EXTERNAL_VALIDATION_CAVEAT,
        "external_validation_caveat_zh": EXTERNAL_VALIDATION_CAVEAT_ZH,
        "design": {
            "folds": folds,
            "fold_seed": fold_seed,
            "stratified_by": "task::source_dataset",
            "fold_assignment": (
                "strata processed in sorted order; ids sorted (numeric-aware) "
                "and shuffled with one random.Random(fold_seed); dealt "
                "round-robin across the folds"
            ),
            "candidate_policies": list(POLICY_ORDER),
            "policy_rule": POLICY_RULE,
            "selection_rule": SELECTION_RULE,
            "tie_tolerance": TIE_TOLERANCE,
            "bootstrapping": {
                "seed": bootstrap_seed,
                "repeats": bootstrap_repeats,
                "ci_level": CI_LEVEL,
                "stratified_by": "task::source_dataset",
                "gate": False,
            },
            "dense_role": "descriptive_reference_only",
        },
        "inputs_sha256": pinned,
        "contract": contract,
        "formal_report_checks": formal_checks,
        "reproduction": reproduction,
        "fold_assignments": {sid: fold_of[sid] for sid in sorted(fold_of, key=_sample_sort_key)},
        "fold_selections": fold_selections,
        "selection_counts": dict(selection_counts),
        "selection_stable_across_folds": selection_stable,
        "oof": overall,
        "baselines": {
            ARM_NEVER: {
                "cv_bench_accuracy": never_official,
                "unweighted_sample_accuracy": statistics.fmean(
                    bool(record["never_correct"]) for record in oof_records
                ),
                "role": "frozen integer-L2 arm",
            },
            ARM_ALWAYS: {
                "cv_bench_accuracy": always_official,
                "unweighted_sample_accuracy": statistics.fmean(
                    bool(record["always_correct"]) for record in oof_records
                ),
                "role": "frozen Prompt+integer-L2 arm",
            },
            "dense": {
                "cv_bench_accuracy": dense_official,
                "unweighted_sample_accuracy": statistics.fmean(
                    bool(record["dense_correct"]) for record in oof_records
                ),
                "role": "descriptive_reference_only; never a gate",
            },
        },
        "comparisons": {
            "oof_minus_never": {
                "left": "oof_selected_policy",
                "right": ARM_NEVER,
                "metric": PRIMARY_METRIC,
                "delta_point_estimate": float(oof_official - never_official),
                "delta_ci95": [float(ci_never[0]), float(ci_never[1])],
                "stratified_by": "task::source_dataset",
                "seed": bootstrap_seed,
                "repeats": bootstrap_repeats,
            },
            "oof_minus_always": {
                "left": "oof_selected_policy",
                "right": ARM_ALWAYS,
                "metric": PRIMARY_METRIC,
                "delta_point_estimate": float(oof_official - always_official),
                "delta_ci95": [float(ci_always[0]), float(ci_always[1])],
                "stratified_by": "task::source_dataset",
                "seed": bootstrap_seed,
                "repeats": bootstrap_repeats,
            },
            "oof_minus_dense_descriptive": {
                "left": "oof_selected_policy",
                "right": "dense",
                "metric": PRIMARY_METRIC,
                "delta_point_estimate": float(oof_official - dense_official),
                "delta_ci95": None,
                "role": "descriptive context only; no bootstrap is run for dense",
            },
        },
        "transitions": {
            "oof_vs_never": transitions_never,
            "oof_vs_always": transitions_always,
        },
        "by_task": by_task,
        "by_source": by_source,
        "by_source_dataset": by_source_dataset,
        "by_G_bin": by_g_bin,
        "limitations": [
            EXTERNAL_VALIDATION_CAVEAT,
            "This is exploratory model selection over frozen outputs; it changes "
            "no raw output, config or prior report and creates no formal claim.",
            "The official cv_bench_accuracy requires all three sources "
            "(ADE20K, COCO, Omni3D); inside single-source task/source strata only "
            "the unweighted sample accuracy is reported and it is not a gate.",
            "Threshold discovery and evaluation share the same full2638 corpus "
            "under cross-validation; selections being stable across folds does "
            "not make the score an external evaluation.",
            "No GPU, model, board, network or training is used; the whole "
            "diagnostic is a CPU-only read-only re-analysis of frozen JSONLs.",
        ],
        "per_sample": per_sample,
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------
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


def _summary_table(lines: list[str], title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        "| 层 | n | Prompt 使用率 | OOF(unweighted) | never | always | OOF−never | "
        "OOF−always | 对 never 净修正 | OOF official |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['stratum']} | {row['n']} | {_fmt(row['prompt_use_rate'])} | "
            f"{_fmt(row['oof_unweighted_accuracy'])} | "
            f"{_fmt(row['never_unweighted_accuracy'])} | "
            f"{_fmt(row['always_unweighted_accuracy'])} | "
            f"{_fmt_signed(row['oof_minus_never_unweighted'])} | "
            f"{_fmt_signed(row['oof_minus_always_unweighted'])} | "
            f"{row['transitions_vs_never']['net_corrections']:+d} | "
            f"{_fmt(row['oof_official_cv_bench_accuracy'])} |"
        )
    lines.append("")


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# full2638 out-of-fold G-gated Prompt 诊断（探索性、只读）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：{report['scope']}")
    lines.append(
        "- 探索性模型选择，**不是新正式主张、不构成门**；冻结正式结论不变："
        f"**{report['formal_outcome']}**"
    )
    lines.append(f"> {report['external_validation_caveat']}")
    lines.append("")
    lines.append(f"> ⚠️ {report['external_validation_caveat_zh']}")
    lines.append("")
    lines.append(
        f"- 唯一主指标：`{report['primary_metric']}`；unweighted 仅次要诊断。"
    )
    lines.append(f"- dense 角色：{report['design']['dense_role']}。")
    lines.append("")

    design = report["design"]
    lines.append("## 1. 固定设计")
    lines.append("")
    lines.append(
        f"- {design['folds']}-fold，seed={design['fold_seed']}，分层="
        f"`{design['stratified_by']}`；分配：{design['fold_assignment']}。"
    )
    lines.append(f"- 候选策略（固定顺序）：{design['candidate_policies']}。")
    lines.append(f"- 策略规则：{design['policy_rule']}")
    lines.append(f"- 选择规则：{design['selection_rule']}")
    lines.append(
        f"- bootstrap：seed={design['bootstrapping']['seed']} / repeats="
        f"{design['bootstrapping']['repeats']} / level={design['bootstrapping']['ci_level']}"
        f" / 分层={design['bootstrapping']['stratified_by']}；**不作门**。"
    )
    lines.append("")

    lines.append("## 2. 输入合同与复现")
    lines.append("")
    lines.append(f"- 输入 SHA pin 全部通过：{report['contract']['inputs_unchanged_during_analysis']}")
    lines.append(
        "- 复现冻结基线：never=integer 臂 "
        f"{report['reproduction']['never_matches_formal_integer']}；always=Prompt 臂 "
        f"{report['reproduction']['always_matches_formal_prompt']}；dense "
        f"{report['reproduction']['dense_matches_formal_dense']}。"
    )
    lines.append("")

    lines.append("## 3. Fold 选择")
    lines.append("")
    lines.append(
        "| fold | train n | held-out n | 选中策略 | train official | held-out official | "
        "held-out Prompt 使用率 |"
    )
    lines.append("|---:|---:|---:|---|---:|---:|---:|")
    for fold in report["fold_selections"]:
        lines.append(
            f"| {fold['fold']} | {fold['train_sample_count']} | "
            f"{fold['held_out_sample_count']} | {fold['selected_policy']} | "
            f"{_fmt(fold['selected_policy_train_cv_bench_accuracy'])} | "
            f"{_fmt(fold['selected_policy_held_out_cv_bench_accuracy'])} | "
            f"{_fmt(fold['held_out_prompt_use_rate'])} |"
        )
    lines.append("")
    lines.append(
        f"- 选择跨 fold 稳定：**{report['selection_stable_across_folds']}**；"
        f"各策略被选次数：{report['selection_counts']}。"
    )
    lines.append("")

    lines.append("## 4. OOF 结果与对比")
    lines.append("")
    oof = report["oof"]
    lines.append(
        f"- OOF official `cv_bench_accuracy`=**{_fmt(oof['oof_official_cv_bench_accuracy'])}**"
        f"；unweighted={_fmt(oof['oof_unweighted_accuracy'])}；Prompt 使用率="
        f"**{_fmt(oof['prompt_use_rate'])}**（{oof['prompt_use_count']}/{oof['n']}）。"
    )
    lines.append(
        f"- never_prompt official={_fmt(report['baselines'][ARM_NEVER]['cv_bench_accuracy'])}"
        f"；always_prompt official={_fmt(report['baselines'][ARM_ALWAYS]['cv_bench_accuracy'])}"
        f"；dense（描述性）={_fmt(report['baselines']['dense']['cv_bench_accuracy'])}。"
    )
    n = report["comparisons"]["oof_minus_never"]
    a = report["comparisons"]["oof_minus_always"]
    d = report["comparisons"]["oof_minus_dense_descriptive"]
    lines.append(
        f"- OOF−never = **{_fmt_signed(n['delta_point_estimate'])}**，95% CI "
        f"{_fmt_ci(n['delta_ci95'])}。"
    )
    lines.append(
        f"- OOF−always = **{_fmt_signed(a['delta_point_estimate'])}**，95% CI "
        f"{_fmt_ci(a['delta_ci95'])}。"
    )
    lines.append(
        f"- OOF−dense（仅描述，无 bootstrap）= {_fmt_signed(d['delta_point_estimate'])}。"
    )
    lines.append("")

    lines.append("## 5. 迁移（相对 never / always）")
    lines.append("")
    for name, key in (("相对 never_prompt（integer 臂）", "oof_vs_never"),
                      ("相对 always_prompt（Prompt 臂）", "oof_vs_always")):
        t = report["transitions"][key]
        lines.append(
            f"- {name}：W→R={t['wrong_to_right']}、R→W={t['right_to_wrong']}、"
            f"不变对={t['unchanged_right']}、不变错={t['unchanged_wrong']}、"
            f"净修正={t['net_corrections']:+d}。"
        )
    lines.append("")

    _summary_table(lines, "6. 按 task", report["by_task"])
    _summary_table(lines, "7. 按 source", report["by_source"])
    _summary_table(lines, "8. 按 source_dataset", report["by_source_dataset"])
    _summary_table(
        lines,
        "9. 按 pre-selector G 桶（G = full_llm_visual_tokens）",
        report["by_G_bin"],
    )

    lines.append("## 10. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")

    lines.append("## 11. 限制")
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


def write_outputs(report: Mapping[str, Any], outputs: OutputPaths = DEFAULT_OUTPUTS) -> None:
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
    expected_sample_count: int = het.FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    folds: int = FOLDS,
    fold_seed: int = FOLD_SEED,
) -> tuple[int, dict[str, Any] | None]:
    try:
        report = analyze(
            paths,
            expected_pins=expected_pins,
            expected_sample_count=expected_sample_count,
            manifest_sha256=manifest_sha256,
            folds=folds,
            fold_seed=fold_seed,
        )
    except (InputContractError, ValueError, KeyError, statistics.StatisticsError) as exc:
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
        "--folds", type=int, default=FOLDS, help="number of stratified folds"
    )
    parser.add_argument("--fold-seed", type=int, default=FOLD_SEED)
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
        folds=args.folds,
        fold_seed=args.fold_seed,
    )
    if code != 0 or report is None:
        return code
    selections = {
        fold["fold"]: fold["selected_policy"] for fold in report["fold_selections"]
    }
    print(
        "oof_cv_bench_accuracy={oof:.6f} never={never:.6f} always={always:.6f} "
        "prompt_use_rate={pu:.6f}".format(
            oof=report["oof"]["oof_official_cv_bench_accuracy"],
            never=report["baselines"][ARM_NEVER]["cv_bench_accuracy"],
            always=report["baselines"][ARM_ALWAYS]["cv_bench_accuracy"],
            pu=report["oof"]["prompt_use_rate"],
        )
    )
    print(f"fold_selections: {selections}")
    print(f"selection_stable_across_folds: {report['selection_stable_across_folds']}")
    print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
