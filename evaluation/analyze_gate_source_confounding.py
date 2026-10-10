#!/usr/bin/env python3
"""Quantify source-dataset confounding of the frozen ``G > 512`` Prompt gate.

This is a CPU-only, read-only diagnostic over the frozen full2638 corrected-L2
arms.  It never imports ``torch`` and never runs a GPU, model, network or
training.  It reuses the frozen parsing/scoring/statistics helpers from
:mod:`evaluation.analyze_full2638_prompt_gain_heterogeneity` and
:mod:`evaluation.build_cv_bench_pruning_report`.

What it does
------------
1. Builds the ``source_dataset x G-bin`` contingency table (five descriptive
   bins plus the binary gate bins ``G<=512`` / ``G>512``).
2. Diagnoses how strongly the gate indicator ``G>512`` depends on
   ``source_dataset`` with Cramer's V and mutual information.  Both are marked
   exploratory only; they are never a gate.
3. Reports, inside every ``source_dataset``, the Prompt-minus-integer
   wrong-to-right / right-to-wrong / net correction counts, the unweighted
   paired delta and a stratified paired bootstrap CI.
4. For ADE20K only -- the one source dataset that straddles the G=512 gate --
   computes the same-source effect difference (interaction)
   ``(Prompt-Integer | G>512) - (Prompt-Integer | G<=512)`` together with a
   stratified paired bootstrap interaction CI.

The formal CV-Bench conclusion is not re-opened.  ``formal_result_unchanged``
is always true and ``formal_outcome`` is echoed from the frozen report.  If the
ADE20K interaction CI is not entirely ``> 0`` (lower bound ``<= 0``) the
conclusion is forced to ``SOURCE_CONFOUNDING_NOT_RESOLVED``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import analyze_full2638_prompt_gain_heterogeneity as het  # noqa: E402
from evaluation.build_cv_bench_pruning_report import score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

REPORT_JSON = ROOT / "reports" / "cv_bench" / "2026-10-04-gate-source-confounding.json"
REPORT_CSV = ROOT / "reports" / "cv_bench" / "2026-10-04-gate-source-confounding.csv"
REPORT_MD = ROOT / "reports" / "cv_bench" / "2026-10-04-gate-source-confounding.md"


@dataclass(frozen=True)
class OutputPaths:
    json: Path = REPORT_JSON
    csv: Path = REPORT_CSV
    md: Path = REPORT_MD


DEFAULT_OUTPUTS = OutputPaths()

ANALYSIS = "full2638_gate_source_confounding"
SCOPE = (
    "Exploratory source-dataset confounding quantification of the frozen "
    "prompt_if_G_gt_512 gate on the frozen full2638 corrected-L2 arms; "
    "CPU-only, read-only; NOT a new formal claim and NOT a gate."
)

FORMAL_OUTCOME = het.FORMAL_OUTCOME
PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = 'score(...)["accuracy"] (unweighted sample accuracy)'

GATE_THRESHOLD = 512
GATE_RULE = (
    "prompt_if_G_gt_512 uses the Prompt+integer-L2 arm iff the pre-selector "
    "G = full_llm_visual_tokens is strictly greater than 512, else the "
    "integer-L2 arm. No task/source/dataset identity enters the gate."
)

BOOTSTRAP_SEED = het.BOOTSTRAP_SEED
BOOTSTRAP_REPEATS = het.BOOTSTRAP_REPEATS
CI_LEVEL = het.CI_LEVEL

ARM_INTEGER = het.ARM_INTEGER
ARM_PROMPT = het.ARM_PROMPT

CONCLUSION_NOT_RESOLVED = "SOURCE_CONFOUNDING_NOT_RESOLVED"
CONCLUSION_RESOLVED = "SOURCE_CONFOUNDING_RESOLVED"
CONCLUSION_UNDEFINED = "SOURCE_CONFOUNDING_UNDEFINED"

EXPLORATORY_NOTE = (
    "Every subgroup / bootstrap / dependency number in this report is a "
    "post-hoc exploratory description of the frozen "
    "PROMPT_GAIN_NOT_CONFIRMED_FULL2638 result. Cramer's V and mutual "
    "information are descriptive association measures only, not causal "
    "evidence and not a gate. Nothing here changes the formal CV-Bench "
    "conclusion."
)
EXPLORATORY_NOTE_ZH = (
    "本报告所有子组 / bootstrap / 依赖诊断数字均为冻结结论 "
    "PROMPT_GAIN_NOT_CONFIRMED_FULL2638 的事后探索性描述；Cramer's V 与互信息"
    "仅为描述性关联度量，不是因果证据、不作为门。正式 CV-Bench 结论不变。"
)

#: Descriptive five-bin split used for the ``source_dataset x G-bin`` table.
G_BINS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("<=96", lambda value: value <= 96),
    ("97-256", lambda value: 97 <= value <= 256),
    ("257-512", lambda value: 257 <= value <= 512),
    ("513-1024", lambda value: 513 <= value <= 1024),
    (">1024", lambda value: value > 1024),
)
G_BIN_ORDER = tuple(label for label, _ in G_BINS)

#: Binary gate bins.
GATE_BINS = ("G<=512", "G>512")

TRANSITIONS = het.TRANSITIONS

CSV_COLUMNS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "G",
    "G_bin",
    "gate_G_gt_512",
    "identity_bypass",
    "integer_answer",
    "prompt_answer",
    "integer_correct",
    "prompt_correct",
    "transition",
)

CONTRACT_CHECK_NAMES = (
    "exactly 2638 identical unique sample_ids in manifest and both arms",
    "task/source/source_dataset come from the manifest",
    "G taken from the integer arm full_llm_visual_tokens",
    "gate is exactly G > 512 with no task/source identity",
    "recomputed overall prompt-minus-integer delta reproduces the frozen report",
    "inputs unchanged (re-hashed) before and after the analysis",
)


class InputContractError(het.InputContractError):
    """Raised when an input is missing or violates the frozen contract."""


def gate_label(group_count: int) -> str:
    """Return the binary gate bin label for ``group_count``."""

    return "G>512" if int(group_count) > GATE_THRESHOLD else "G<=512"


def gate_selected(group_count: int) -> bool:
    return int(group_count) > GATE_THRESHOLD


def g_bin(group_count: int) -> str:
    for label, predicate in G_BINS:
        if predicate(int(group_count)):
            return label
    return "other"


def _sample_sort_key(sample_id: str) -> tuple[int, Any]:
    text = str(sample_id)
    if text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


# ---------------------------------------------------------------------------
# Association diagnostics (exploratory only)
# ---------------------------------------------------------------------------
def cramers_v(table: Sequence[Sequence[int]]) -> float:
    """Cramer's V for a non-degenerate contingency table (exploratory)."""

    rows = [[int(value) for value in row] for row in table]
    if not rows or not rows[0]:
        raise ValueError("empty contingency table")
    n = sum(sum(row) for row in rows)
    if n <= 0:
        raise ValueError("empty contingency table")
    row_totals = [sum(row) for row in rows]
    col_totals = [sum(rows[r][c] for r in range(len(rows))) for c in range(len(rows[0]))]
    chi2 = 0.0
    for r, row in enumerate(rows):
        for c, observed in enumerate(row):
            expected = row_totals[r] * col_totals[c] / n
            if expected <= 0:
                continue
            chi2 += (observed - expected) ** 2 / expected
    denom = n * min(len(rows) - 1, len(rows[0]) - 1)
    if denom <= 0:
        return 0.0
    return math.sqrt(chi2 / denom)


def mutual_information_bits(table: Sequence[Sequence[int]]) -> float:
    """Mutual information in bits for a contingency table (exploratory)."""

    rows = [[int(value) for value in row] for row in table]
    if not rows or not rows[0]:
        raise ValueError("empty contingency table")
    n = sum(sum(row) for row in rows)
    if n <= 0:
        raise ValueError("empty contingency table")
    row_totals = [sum(row) for row in rows]
    col_totals = [sum(rows[r][c] for r in range(len(rows))) for c in range(len(rows[0]))]
    mi = 0.0
    for r, row in enumerate(rows):
        for c, observed in enumerate(row):
            if observed <= 0:
                continue
            p_xy = observed / n
            p_x = row_totals[r] / n
            p_y = col_totals[c] / n
            mi += p_xy * math.log(p_xy / (p_x * p_y))
    return mi / math.log(2.0)


def contingency(
    records: Sequence[Mapping[str, Any]],
    row_key: Callable[[Mapping[str, Any]], str],
    row_values: Sequence[str],
    col_key: Callable[[Mapping[str, Any]], str],
    col_values: Sequence[str],
) -> list[list[int]]:
    index = {value: position for position, value in enumerate(row_values)}
    col_index = {value: position for position, value in enumerate(col_values)}
    table = [[0] * len(col_values) for _ in row_values]
    for record in records:
        r = index[str(row_key(record))]
        c = col_index[str(col_key(record))]
        table[r][c] += 1
    return table


# ---------------------------------------------------------------------------
# Paired stratified bootstrap of the ADE20K interaction
# ---------------------------------------------------------------------------
def bootstrap_interaction_delta(
    records: Sequence[Mapping[str, Any]],
    *,
    bin_key: Callable[[Mapping[str, Any]], str],
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
) -> list[float]:
    """Stratified paired bootstrap CI of ``effect(G>512) - effect(G<=512)``.

    Resampling is within ``task :: bin`` strata so both bin sample sizes are
    preserved exactly in every replicate.  The per-bin effect is the unweighted
    paired accuracy difference between the Prompt and integer arms (the
    official ``cv_bench_accuracy`` is undefined inside a single source).
    """

    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(str(row["task"]), str(bin_key(row)))].append(row)
    counts: dict[str, int] = defaultdict(int)
    for (task, label), group in groups.items():
        counts[label] += len(group)
    if counts.get("G>512", 0) == 0 or counts.get("G<=512", 0) == 0:
        raise InputContractError(
            "interaction requires both G>512 and G<=512 records (fail closed)"
        )
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        high_gain = 0
        low_gain = 0
        for (task, label), group in groups.items():
            prompt_sum = 0
            integer_sum = 0
            for _ in range(len(group)):
                sample = rng.choice(group)
                prompt_sum += bool(sample[f"{ARM_PROMPT}_correct"])
                integer_sum += bool(sample[f"{ARM_INTEGER}_correct"])
            if label == "G>512":
                high_gain += prompt_sum - integer_sum
            else:
                low_gain += prompt_sum - integer_sum
        deltas.append(high_gain / counts["G>512"] - low_gain / counts["G<=512"])
    deltas.sort()
    return [
        float(deltas[int(0.025 * (repeats - 1))]),
        float(deltas[int(0.975 * (repeats - 1))]),
    ]


def paired_unweighted_delta(records: Sequence[Mapping[str, Any]]) -> float:
    if not records:
        raise InputContractError("cannot compute delta of an empty stratum")
    return float(
        sum(bool(row[f"{ARM_PROMPT}_correct"]) for row in records)
        - sum(bool(row[f"{ARM_INTEGER}_correct"]) for row in records)
    ) / len(records)


# ---------------------------------------------------------------------------
# Tables and effects
# ---------------------------------------------------------------------------
def source_g_bin_table(
    joined: Sequence[Mapping[str, Any]],
    source_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    source_datasets = sorted({str(row["source_dataset"]) for row in joined})
    if source_order is not None:
        source_datasets = [s for s in source_order if s in set(source_datasets)]
    rows: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        subset = [row for row in joined if str(row["source_dataset"]) == source_dataset]
        counts = {label: 0 for label in G_BIN_ORDER}
        for row in subset:
            counts[str(row["G_bin"])] += 1
        gate_on = sum(1 for row in subset if gate_selected(row["G"]))
        row = {
            "source_dataset": source_dataset,
            "n": len(subset),
            "counts": counts,
            "gate_G_gt_512_count": gate_on,
            "gate_G_le_512_count": len(subset) - gate_on,
            "gate_selected_fraction": gate_on / len(subset) if subset else 0.0,
            "fully_inside_gate": gate_on == len(subset) and len(subset) > 0,
            "fully_outside_gate": gate_on == 0 and len(subset) > 0,
            "tasks": sorted({str(row["task"]) for row in subset}),
        }
        rows.append(row)
    return rows


def source_dataset_effects(
    joined: Sequence[Mapping[str, Any]],
    source_order: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    source_datasets = sorted({str(row["source_dataset"]) for row in joined})
    if source_order is not None:
        source_datasets = [s for s in source_order if s in set(source_datasets)]
    effects: list[dict[str, Any]] = []
    for source_dataset in source_datasets:
        subset = [row for row in joined if str(row["source_dataset"]) == source_dataset]
        row = het.stratum_row(source_dataset, subset, bootstrap=True)
        effects.append(row)
    return effects


def ade_interaction(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Same-source ADE20K effect difference ``G>512`` minus ``G<=512``."""

    ade = [row for row in joined if str(row["source_dataset"]).startswith("ADE20K")]
    if not ade:
        return {
            "defined": False,
            "reason": "no ADE20K records",
            "interaction": None,
            "bootstrap_ci95_interaction": None,
        }
    low = [row for row in ade if not gate_selected(row["G"])]
    high = [row for row in ade if gate_selected(row["G"])]
    low_row = het.stratum_row("ADE20K::G<=512", low, bootstrap=True)
    high_row = het.stratum_row("ADE20K::G>512", high, bootstrap=True)
    low_delta = low_row["unweighted_paired_delta"]
    high_delta = high_row["unweighted_paired_delta"]
    interaction = high_delta - low_delta if low and high else None
    ci: list[float] | None = None
    if low and high:
        ci = bootstrap_interaction_delta(
            ade, bin_key=lambda row: gate_label(int(row["G"]))
        )
    official_defined = {str(row["source"]) for row in ade} >= {
        "ADE20K",
        "COCO",
        "Omni3D",
    }
    return {
        "defined": bool(low and high),
        "source_dataset": "ADE20K Validation Set",
        "n": len(ade),
        "n_G_le_512": len(low),
        "n_G_gt_512": len(high),
        "low_G_le_512": low_row,
        "high_G_gt_512": high_row,
        "effect_G_le_512": low_delta,
        "effect_G_gt_512": high_delta,
        "interaction_G_gt_512_minus_G_le_512": interaction,
        "bootstrap_ci95_interaction": ci,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_strata": "task::gate_bin",
        "bootstrap_metric": (
            "unweighted sample accuracy (exploratory; the official "
            "cv_bench_accuracy is undefined inside a single source)"
        ),
        "official_metric_defined_within_source": official_defined,
    }


def conclusion_for(ade: Mapping[str, Any]) -> str:
    """Map the ADE20K interaction to the preregistered conclusion string."""

    if not ade.get("defined"):
        return CONCLUSION_UNDEFINED
    ci = ade.get("bootstrap_ci95_interaction")
    if not ci:
        return CONCLUSION_UNDEFINED
    return CONCLUSION_RESOLVED if ci[0] > 0.0 else CONCLUSION_NOT_RESOLVED


def gate_composition(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate_on = [row for row in joined if gate_selected(row["G"])]
    gate_off = [row for row in joined if not gate_selected(row["G"])]
    by_source_on = {
        source: sum(1 for row in gate_on if str(row["source"]) == source)
        for source in sorted({str(row["source"]) for row in joined})
    }
    by_source_dataset_on = {
        source_dataset: sum(
            1 for row in gate_on if str(row["source_dataset"]) == source_dataset
        )
        for source_dataset in sorted({str(row["source_dataset"]) for row in joined})
    }
    return {
        "gate_on_count": len(gate_on),
        "gate_off_count": len(gate_off),
        "gate_on_fraction": len(gate_on) / len(joined),
        "gate_on_by_source": by_source_on,
        "gate_on_by_source_dataset": by_source_dataset_on,
    }


def gate_policy_effect(joined: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """In-sample descriptive effect of the gate (not the OOF estimate)."""

    def record(row: Mapping[str, Any]) -> dict[str, Any]:
        use_prompt = gate_selected(row["G"])
        return {
            "task": row["task"],
            "source": row["source"],
            "source_dataset": row["source_dataset"],
            "gated_answer": row[f"{ARM_PROMPT}_answer"] if use_prompt else row[f"{ARM_INTEGER}_answer"],
            "gated_correct": bool(row[f"{ARM_PROMPT}_correct"])
            if use_prompt
            else bool(row[f"{ARM_INTEGER}_correct"]),
            "integer_answer": row[f"{ARM_INTEGER}_answer"],
            "integer_correct": bool(row[f"{ARM_INTEGER}_correct"]),
            "prompt_answer": row[f"{ARM_PROMPT}_answer"],
            "prompt_correct": bool(row[f"{ARM_PROMPT}_correct"]),
        }

    rows = [record(row) for row in joined]
    sources_present = {str(row["source"]) for row in joined}
    official_defined = sources_present >= {"ADE20K", "COCO", "Omni3D"}
    gated_official = (
        float(score(rows, "gated_correct")["cv_bench_accuracy"])
        if official_defined
        else None
    )
    integer_official = (
        float(score(rows, "integer_correct")["cv_bench_accuracy"])
        if official_defined
        else None
    )
    prompt_official = (
        float(score(rows, "prompt_correct")["cv_bench_accuracy"])
        if official_defined
        else None
    )
    n = len(rows)
    gated_unweighted = sum(bool(row["gated_correct"]) for row in rows) / n
    integer_unweighted = sum(bool(row["integer_correct"]) for row in rows) / n
    prompt_unweighted = sum(bool(row["prompt_correct"]) for row in rows) / n
    return {
        "note": (
            "in-sample descriptive gate effect; the frozen out-of-fold estimate "
            "lives in the OOF report and is not recomputed as a new claim"
        ),
        "gated_official_cv_bench_accuracy": gated_official,
        "integer_official_cv_bench_accuracy": integer_official,
        "prompt_official_cv_bench_accuracy": prompt_official,
        "gated_minus_integer_official": (
            gated_official - integer_official
            if gated_official is not None and integer_official is not None
            else None
        ),
        "gated_unweighted_accuracy": gated_unweighted,
        "integer_unweighted_accuracy": integer_unweighted,
        "prompt_unweighted_accuracy": prompt_unweighted,
        "gated_minus_integer_unweighted": gated_unweighted - integer_unweighted,
        "gated_minus_prompt_unweighted": gated_unweighted - prompt_unweighted,
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze(
    paths: het.InputPaths = het.DEFAULT_PATHS,
    *,
    expected_pins: Mapping[str, str] | None = het.DEFAULT_INPUT_PINS,
    expected_sample_count: int = het.FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    formal_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the frozen inputs and compute the confounding diagnostic."""

    pinned = het.verify_pins(paths, expected_pins)

    manifest_rows = het.load_rows(paths.full_manifest, "full_manifest")
    manifest_index = het.index_unique(manifest_rows, "full_manifest")
    arm_rows = {
        ARM_INTEGER: het.load_rows(paths.arm_integer_jsonl, "corrected_l2_integer"),
        ARM_PROMPT: het.load_rows(
            paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"
        ),
    }
    arm_indices = {arm: het.index_unique(rows, arm) for arm, rows in arm_rows.items()}

    integer_sha = pinned["corrected_l2_integer_arm"]["sha256"]
    prompt_sha = pinned["prompt_plus_corrected_l2_integer_arm"]["sha256"]
    actual_manifest_sha = (
        manifest_sha256
        if manifest_sha256 is not None
        else pinned["full_manifest"]["sha256"]
    )

    if formal_report is None:
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
    formal_joined_index = (
        het.index_unique(formal_joined, "formal_report.joined")
        if formal_joined is not None
        else None
    )
    joined = het.build_joined(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        formal_joined_index=formal_joined_index,
    )
    for row in joined:
        row["G_bin"] = g_bin(int(row["G"]))
        row["gate_label"] = gate_label(int(row["G"]))

    # Recompute the frozen official delta/CI to prove the arms are bit-identical
    # to the formal report before any confounding number is produced.
    overall_integer_official = het.official_accuracy(joined, ARM_INTEGER)
    overall_prompt_official = het.official_accuracy(joined, ARM_PROMPT)
    official_delta = (
        overall_prompt_official - overall_integer_official
        if overall_integer_official is not None and overall_prompt_official is not None
        else None
    )
    frozen_matches = (
        official_delta is not None
        and abs(official_delta - formal_checks["frozen_delta_point_estimate"]) <= 1e-12
    )
    if not frozen_matches:
        raise InputContractError(
            "recomputed overall prompt-minus-integer delta does not reproduce the "
            "frozen formal report (fail closed)"
        )

    table = source_g_bin_table(joined)
    source_datasets = [row["source_dataset"] for row in table]
    effects = source_dataset_effects(joined, source_order=source_datasets)

    # 2-D contingency of interest: rows = gate on / gate off, columns =
    # source_dataset; plus the five-bin variant for the descriptive table.
    gate_by_source = [[0] * len(source_datasets) for _ in range(2)]
    bin_by_source = [[0] * len(source_datasets) for _ in range(len(G_BIN_ORDER))]
    source_index = {name: position for position, name in enumerate(source_datasets)}
    for row in joined:
        gate_by_source[0 if gate_selected(row["G"]) else 1][source_index[row["source_dataset"]]] += 1
        bin_by_source[G_BIN_ORDER.index(row["G_bin"])][source_index[row["source_dataset"]]] += 1

    dependency = {
        "exploratory_only": True,
        "not_a_gate": True,
        "gate_indicator": f"G>{GATE_THRESHOLD}",
        "row_values": ["G>512", "G<=512"],
        "column_values": source_datasets,
        "gate_by_source_counts": gate_by_source,
        "cramers_v_gate_vs_source_dataset": cramers_v(gate_by_source),
        "mutual_information_bits_gate_vs_source_dataset": mutual_information_bits(
            gate_by_source
        ),
        "five_bin_row_values": list(G_BIN_ORDER),
        "bin_by_source_counts": bin_by_source,
        "cramers_v_gbin_vs_source_dataset": cramers_v(bin_by_source),
        "mutual_information_bits_gbin_vs_source_dataset": mutual_information_bits(
            bin_by_source
        ),
    }

    ade = ade_interaction(joined)
    composition = gate_composition(joined)
    gate_effect = gate_policy_effect(joined)

    if not ade["defined"]:
        conclusion = CONCLUSION_UNDEFINED
    else:
        conclusion = conclusion_for(ade)

    contract_checks = [
        {
            "name": CONTRACT_CHECK_NAMES[0],
            "ok": contract.get("expected_sample_count")
            == expected_sample_count
            and contract.get("sample_ids_identical") is True
            and contract.get("unique_sample_ids_per_arm") is True,
        },
        {
            "name": CONTRACT_CHECK_NAMES[1],
            "ok": contract.get("task_matches_manifest") is True,
        },
        {
            "name": CONTRACT_CHECK_NAMES[2],
            "ok": all(
                int(row["G"]) == int(arm_indices[ARM_INTEGER][row["sample_id"]]["full_llm_visual_tokens"])
                for row in joined
            ),
        },
        {
            "name": CONTRACT_CHECK_NAMES[3],
            "ok": all(
                (row["gate_label"] == "G>512") == (int(row["G"]) > GATE_THRESHOLD)
                for row in joined
            ),
        },
        {"name": CONTRACT_CHECK_NAMES[4], "ok": bool(frozen_matches)},
        {"name": CONTRACT_CHECK_NAMES[5], "ok": True},
    ]
    contract["gate_contract_checks"] = contract_checks

    # Re-verify read-only inputs after the analysis.
    for name, spec in pinned.items():
        current = het.sha256_file(Path(spec["path"]))
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
            "G": int(row["G"]),
            "G_bin": row["G_bin"],
            "gate_G_gt_512": gate_selected(row["G"]),
            "identity_bypass": bool(row["identity_bypass"]),
            "integer_answer": row[f"{ARM_INTEGER}_answer"],
            "prompt_answer": row[f"{ARM_PROMPT}_answer"],
            "integer_correct": bool(row[f"{ARM_INTEGER}_correct"]),
            "prompt_correct": bool(row[f"{ARM_PROMPT}_correct"]),
            "transition": row["transition"],
        }
        for row in sorted(joined, key=lambda item: _sample_sort_key(str(item["sample_id"])))
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "analysis": ANALYSIS,
        "scope": SCOPE,
        "exploratory_only": True,
        "not_a_formal_claim": True,
        "formal_result_unchanged": True,
        "formal_outcome": FORMAL_OUTCOME,
        "conclusion": conclusion,
        "primary_metric": PRIMARY_METRIC,
        "unweighted_metric": UNWEIGHTED_METRIC,
        "exploratory_note": EXPLORATORY_NOTE,
        "exploratory_note_zh": EXPLORATORY_NOTE_ZH,
        "gate": {
            "policy": "prompt_if_G_gt_512",
            "threshold": GATE_THRESHOLD,
            "rule": GATE_RULE,
            "definition_of_G": (
                "G = full_llm_visual_tokens, the pre-selector real visual group "
                "count (one group per 2x2 merge unit)"
            ),
        },
        "inputs_sha256": pinned,
        "contract": contract,
        "source_g_bin_table": table,
        "source_dataset_effects": effects,
        "dependency": dependency,
        "ade_interaction": ade,
        "gate_composition": composition,
        "gate_policy_effect": gate_effect,
        "frozen_reproduction": {
            "metric": PRIMARY_METRIC,
            "frozen_delta_point_estimate": formal_checks["frozen_delta_point_estimate"],
            "recomputed_delta_point_estimate": official_delta,
            "frozen_delta_ci95": formal_checks["frozen_delta_ci95"],
            "reproduced": bool(frozen_matches),
        },
        "per_sample": per_sample,
    }
    return report


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_csv(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in report["per_sample"]:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_ci(ci: Sequence[float] | None) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# G>512 门控的数据源混杂诊断（full2638，探索性）")
    lines.append("")
    lines.append(f"- 生成时间：{report['generated_at']}")
    lines.append(f"- 分析：`{report['analysis']}`")
    lines.append(f"- **结论：`{report['conclusion']}`**")
    lines.append(f"- 正式结论（不变）：`{report['formal_outcome']}`，`formal_result_unchanged=true`")
    lines.append(f"- 范围：{report['scope']}")
    lines.append("")
    lines.append(f"> {report['exploratory_note_zh']}")
    lines.append("")
    lines.append("## 1. 门控定义")
    lines.append("")
    lines.append(f"- 门：`{report['gate']['policy']}`，阈值 `{report['gate']['threshold']}`")
    lines.append(f"- 规则：{report['gate']['rule']}")
    lines.append(f"- G 定义：{report['gate']['definition_of_G']}")
    comp = report["gate_composition"]
    lines.append(
        f"- 门内样本 `{comp['gate_on_count']}`、门外 `{comp['gate_off_count']}`"
        f"（门内占比 {comp['gate_on_fraction']:.6f}）"
    )
    lines.append("")
    lines.append("## 2. source_dataset × G-bin 分布")
    lines.append("")
    header = "| source_dataset | " + " | ".join(G_BIN_ORDER) + " | n | G>512 | G<=512 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(G_BIN_ORDER) + 4))
    for row in report["source_g_bin_table"]:
        cells = " | ".join(str(row["counts"][label]) for label in G_BIN_ORDER)
        lines.append(
            f"| {row['source_dataset']} | {cells} | {row['n']} | "
            f"{row['gate_G_gt_512_count']} | {row['gate_G_le_512_count']} |"
        )
    lines.append("")
    lines.append("## 3. 门控 / 数据源依赖诊断（探索性，不作门）")
    lines.append("")
    dep = report["dependency"]
    lines.append(
        f"- Cramer's V（`G>512` × source_dataset）= "
        f"{dep['cramers_v_gate_vs_source_dataset']:.6f}"
    )
    lines.append(
        f"- 互信息（`G>512` × source_dataset）= "
        f"{dep['mutual_information_bits_gate_vs_source_dataset']:.6f} bits"
    )
    lines.append(
        f"- Cramer's V（五档 G-bin × source_dataset）= "
        f"{dep['cramers_v_gbin_vs_source_dataset']:.6f}"
    )
    lines.append(
        f"- 互信息（五档 G-bin × source_dataset）= "
        f"{dep['mutual_information_bits_gbin_vs_source_dataset']:.6f} bits"
    )
    lines.append("- 上述两指标均为描述性关联度量，仅探索性，不是因果证据、不作为门。")
    lines.append("")
    lines.append("## 4. 每个 source_dataset 内 Prompt−integer 效应")
    lines.append("")
    lines.append(
        "| source_dataset | n | W→R | R→W | net | unweighted Δ | 95% CI |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for row in report["source_dataset_effects"]:
        lines.append(
            f"| {row['stratum']} | {row['n']} | {row['wrong_to_right']} | "
            f"{row['right_to_wrong']} | {row['net_corrections']:+d} | "
            f"{row['unweighted_paired_delta']:+.6f} | "
            f"{_fmt_ci(row.get('bootstrap_ci95_unweighted_paired_delta'))} |"
        )
    lines.append("")
    lines.append("（CI 为分层配对 bootstrap，重采样层为 `task::source_dataset`；")
    lines.append("单数据源内官方 `cv_bench_accuracy` 未定义，故用无权重样本准确率，仅次要诊断。）")
    lines.append("")
    lines.append("## 5. 焦点：ADE20K 同源效应差（interaction）")
    lines.append("")
    ade = report["ade_interaction"]
    if not ade["defined"]:
        lines.append(f"- 未定义：{ade.get('reason')}")
    else:
        lines.append(
            f"- ADE20K `G<=512`：n={ade['n_G_le_512']}，W→R={ade['low_G_le_512']['wrong_to_right']}，"
            f"R→W={ade['low_G_le_512']['right_to_wrong']}，net={ade['low_G_le_512']['net_corrections']:+d}，"
            f"Δ={ade['effect_G_le_512']:+.6f}"
        )
        lines.append(
            f"- ADE20K `G>512`：n={ade['n_G_gt_512']}，W→R={ade['high_G_gt_512']['wrong_to_right']}，"
            f"R→W={ade['high_G_gt_512']['right_to_wrong']}，net={ade['high_G_gt_512']['net_corrections']:+d}，"
            f"Δ={ade['effect_G_gt_512']:+.6f}"
        )
        lines.append(
            f"- **interaction = Δ(G>512) − Δ(G<=512) = "
            f"{ade['interaction_G_gt_512_minus_G_le_512']:+.6f}**"
        )
        lines.append(
            f"- **95% 分层配对 bootstrap CI = "
            f"{_fmt_ci(ade['bootstrap_ci95_interaction'])}**"
        )
        lines.append(
            f"- 分层方式：`{ade['bootstrap_strata']}`，seed `{ade['bootstrap_seed']}`，"
            f"repeats `{ade['bootstrap_repeats']}`"
        )
        lines.append(
            "- 判定规则：仅当 interaction CI 整体 `> 0`（下界 > 0）才可视为"
            "同源 G 效应得到区分；否则必须为 `SOURCE_CONFOUNDING_NOT_RESOLVED`。"
        )
    lines.append("")
    lines.append("## 6. 门控策略的样本内描述效应")
    lines.append("")
    ge = report["gate_policy_effect"]
    lines.append(f"- 说明：{ge['note']}")
    lines.append(
        f"- 门控官方 `cv_bench_accuracy` = {_fmt(ge['gated_official_cv_bench_accuracy'])}；"
        f"integer = {_fmt(ge['integer_official_cv_bench_accuracy'])}；"
        f"prompt = {_fmt(ge['prompt_official_cv_bench_accuracy'])}"
    )
    lines.append(
        f"- 门控 − integer（无权重） = {_fmt(ge['gated_minus_integer_unweighted'])}；"
        f"门控 − prompt（无权重） = {_fmt(ge['gated_minus_prompt_unweighted'])}"
    )
    lines.append("")
    lines.append("## 7. 冻结复现与契约")
    lines.append("")
    fr = report["frozen_reproduction"]
    lines.append(
        f"- 冻结官方 Prompt−integer Δ = {fr['frozen_delta_point_estimate']:.6f}；"
        f"重算 = {_fmt(fr['recomputed_delta_point_estimate'])}；"
        f"复现 = {_fmt(fr['reproduced'])}"
    )
    for check in report["contract"]["gate_contract_checks"]:
        lines.append(f"- [{'x' if check['ok'] else ' '}] {check['name']}")
    lines.append("")
    lines.append("## 8. 复现命令")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python evaluation/analyze_gate_source_confounding.py "
        "--json reports/cv_bench/2026-10-04-gate-source-confounding.json "
        "--csv reports/cv_bench/2026-10-04-gate-source-confounding.csv "
        "--md reports/cv_bench/2026-10-04-gate-source-confounding.md"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    report: Mapping[str, Any],
    outputs: OutputPaths = DEFAULT_OUTPUTS,
) -> None:
    outputs.json.parent.mkdir(parents=True, exist_ok=True)
    outputs.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(report, outputs.csv)
    outputs.md.write_text(render_markdown(report), encoding="utf-8")


def run(
    paths: het.InputPaths = het.DEFAULT_PATHS,
    outputs: OutputPaths = DEFAULT_OUTPUTS,
    *,
    expected_pins: Mapping[str, str] | None = het.DEFAULT_INPUT_PINS,
    expected_sample_count: int = het.FULL_SAMPLE_COUNT,
    write: bool = True,
) -> dict[str, Any]:
    report = analyze(
        paths,
        expected_pins=expected_pins,
        expected_sample_count=expected_sample_count,
    )
    if write:
        write_outputs(report, outputs)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=REPORT_JSON)
    parser.add_argument("--csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--md", type=Path, default=REPORT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze()
    write_outputs(report, OutputPaths(json=args.json, csv=args.csv, md=args.md))
    print(
        json.dumps(
            {
                "conclusion": report["conclusion"],
                "formal_outcome": report["formal_outcome"],
                "ade_interaction": report["ade_interaction"][
                    "interaction_G_gt_512_minus_G_le_512"
                ],
                "ade_interaction_ci95": report["ade_interaction"][
                    "bootstrap_ci95_interaction"
                ],
                "json": str(args.json),
                "csv": str(args.csv),
                "md": str(args.md),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
