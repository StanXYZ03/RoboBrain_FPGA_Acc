#!/usr/bin/env python3
"""Freeze the 100 ADE20K samples for the same-image resolution counterfactual.

CPU-only and read-only.  The selection is deterministic and stratified by the
original (full2638) gate bin ``G<=512`` / ``G>512``, the CV-Bench task
(``Count`` / ``Relation``) and the original integer-arm correctness.  Exactly
one hundred ADE20K samples are selected with a proportional largest-remainder
allocation and a per-stratum floor; the remaining eligible samples are frozen
as a deterministic reserve list used by the pre-registered G-separation
drop/replace rule.

This script is motivated by the frozen gate/source confounding result
(``SOURCE_CONFOUNDING_NOT_RESOLVED``): inside ADE20K the same-source
Prompt-minus-integer effect difference between the original ``G>512`` and
``G<=512`` bins has a 95% CI that is not entirely positive, so the original G
contrast cannot separate resolution from source identity.  The counterfactual
fixes the image and changes only the processor pixel budget.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import analyze_full2638_prompt_gain_heterogeneity as het  # noqa: E402
from evaluation.analyze_gate_source_confounding import (  # noqa: E402
    CONCLUSION_NOT_RESOLVED,
    gate_label,
)
from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    parse_choice,
)

ROOT = Path(__file__).resolve().parents[1]

DESIGN = "2026-10-04-same-image-resolution-counterfactual"
SCOPE = (
    "Frozen 100-sample ADE20K manifest for a same-image dual-processor-pixel-"
    "budget counterfactual; CPU-only, read-only; no model was run."
)
GATE_REPORT_JSON = (
    ROOT / "reports" / "cv_bench" / "2026-10-04-gate-source-confounding.json"
)
SELECTION_REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual-selection.json"
)
MANIFEST_JSONL = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl"
)
RESERVE_JSONL = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k-reserve.jsonl"
)

ADE_SOURCE_DATASET = "ADE20K Validation Set"
GATE_THRESHOLD = 512
TARGET_TOTAL = 100
MINIMUM_PER_STRATUM = 4
SELECTION_SEED = 20261004

TASKS = ("Count", "Relation")
CORRECTNESS = (True, False)
GATE_ORDER = ("G<=512", "G>512")


class SelectionError(RuntimeError):
    """Raised when an input or selection invariant is violated."""


def _sample_sort_key(sample_id: str) -> tuple[int, Any]:
    text = str(sample_id)
    if text.lstrip("-").isdigit():
        return (0, int(text))
    return (1, text)


def stratum_key(row: Mapping[str, Any]) -> str:
    return (
        f"{gate_label(int(row['G']))}::{row['task']}::"
        f"{'integer_correct' if row['integer_correct'] else 'integer_wrong'}"
    )


def stratum_order() -> list[str]:
    order: list[str] = []
    for gate in GATE_ORDER:
        for task in TASKS:
            for correct in CORRECTNESS:
                label = "integer_correct" if correct else "integer_wrong"
                order.append(f"{gate}::{task}::{label}")
    return order


def largest_remainder_allocation(
    counts: Mapping[str, int],
    *,
    total: int,
    minimum: int,
    order: Sequence[str],
) -> dict[str, int]:
    """Deterministic proportional allocation with a per-stratum floor."""

    labels = list(order)
    available = {label: int(counts.get(label, 0)) for label in labels}
    allocation = {label: min(int(minimum), available[label]) for label in labels}
    remaining = int(total) - sum(allocation.values())
    if remaining < 0:
        raise SelectionError("minimum per stratum exceeds the target total")
    # Proportional pass using largest remainders, capped by availability.
    while remaining > 0:
        open_labels = [label for label in labels if allocation[label] < available[label]]
        if not open_labels:
            break
        pool = sum(available[label] for label in open_labels)
        remainders: list[tuple[float, str]] = []
        for label in open_labels:
            quota = remaining * available[label] / pool
            remainders.append((quota - int(quota), label))
        remainders.sort(key=lambda item: (-item[0], labels.index(item[1])))
        for _remainder, label in remainders:
            if remaining <= 0:
                break
            if allocation[label] < available[label]:
                allocation[label] += 1
                remaining -= 1
    return allocation


def build_candidates(
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sample_id, meta in manifest_index.items():
        if str(meta["source_dataset"]) != ADE_SOURCE_DATASET:
            continue
        integer = arm_indices[het.ARM_INTEGER][sample_id]
        choices = list(meta["choices"])
        integer_answer = parse_choice(str(integer["output_text"]), choices)
        truth = answer_letter(str(meta["answer"]))
        g = int(integer["full_llm_visual_tokens"])
        candidates.append(
            {
                "sample_id": str(sample_id),
                "task": str(meta["task"]),
                "source": str(meta["source"]),
                "source_dataset": str(meta["source_dataset"]),
                "G": g,
                "integer_correct": integer_answer == truth,
            }
        )
    return candidates


def select(
    candidates: Sequence[Mapping[str, Any]],
    *,
    total: int = TARGET_TOTAL,
    minimum: int = MINIMUM_PER_STRATUM,
    seed: int = SELECTION_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    order = stratum_order()
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[stratum_key(candidate)].append(candidate)
    counts = {label: len(grouped.get(label, [])) for label in order}
    allocation = largest_remainder_allocation(
        counts, total=total, minimum=minimum, order=order
    )
    selected: list[dict[str, Any]] = []
    reserve: list[dict[str, Any]] = []
    for label in order:
        group = sorted(grouped.get(label, []), key=lambda row: _sample_sort_key(row["sample_id"]))
        rng = random.Random(f"{seed}|{label}")
        rng.shuffle(group)
        take = allocation[label]
        for rank, candidate in enumerate(group):
            record = dict(candidate)
            record["stratum"] = label
            record["selection_rank"] = rank
            record["selected"] = rank < take
            if rank < take:
                selected.append(record)
            else:
                reserve.append(record)
    allocation_rows = [
        {
            "stratum": label,
            "n_available": counts[label],
            "target": allocation[label],
            "selected": allocation[label],
            "reserve": counts[label] - allocation[label],
        }
        for label in order
    ]
    selected.sort(key=lambda row: (order.index(row["stratum"]), row["selection_rank"]))
    reserve.sort(key=lambda row: (order.index(row["stratum"]), row["selection_rank"]))
    allocation_meta = {
        "method": (
            "proportional largest-remainder allocation over the eight "
            "(gate_bin, task, original_integer_correct) strata with a per-stratum "
            "floor of min(minimum, n_available)"
        ),
        "target_total": total,
        "minimum_per_stratum": minimum,
        "selection_seed": seed,
        "strata_order": order,
        "per_stratum": allocation_rows,
    }
    return selected, reserve, allocation_meta


def _manifest_record(
    meta: Mapping[str, Any], candidate: Mapping[str, Any], *, reserve: bool
) -> dict[str, Any]:
    record = dict(meta)
    record["resolution_counterfactual"] = {
        "design": DESIGN,
        "split": "reserve" if reserve else "selected",
        "stratum": candidate["stratum"],
        "selection_rank": int(candidate["selection_rank"]),
        "original_G": int(candidate["G"]),
        "original_G_bin": gate_label(int(candidate["G"])),
        "original_integer_correct": bool(candidate["integer_correct"]),
        "selection_seed": SELECTION_SEED,
    }
    return record


def sha256_file(path: Path) -> str:
    return het.sha256_file(path)


def analyze(
    *,
    expected_pins: Mapping[str, str] | None = het.DEFAULT_INPUT_PINS,
    expected_gate_report_pin: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    paths = het.DEFAULT_PATHS
    pinned = het.verify_pins(paths, expected_pins)

    if not GATE_REPORT_JSON.is_file():
        raise SelectionError(
            f"missing gate/source confounding report {GATE_REPORT_JSON} "
            "(run evaluation/analyze_gate_source_confounding.py first; fail closed)"
        )
    gate_sha = sha256_file(GATE_REPORT_JSON)
    if expected_gate_report_pin is not None and gate_sha != expected_gate_report_pin:
        raise SelectionError(
            f"gate report sha256 mismatch: expected {expected_gate_report_pin}, "
            f"got {gate_sha} (fail closed)"
        )
    gate_report = json.loads(GATE_REPORT_JSON.read_text(encoding="utf-8"))
    if gate_report.get("conclusion") != CONCLUSION_NOT_RESOLVED:
        raise SelectionError(
            "the counterfactual is preregistered only for the "
            f"SOURCE_CONFOUNDING_NOT_RESOLVED result, got "
            f"{gate_report.get('conclusion')!r} (fail closed)"
        )
    if gate_report.get("formal_outcome") != het.FORMAL_OUTCOME:
        raise SelectionError(
            "gate report does not preserve the frozen formal outcome (fail closed)"
        )

    manifest_rows = het.load_rows(paths.full_manifest, "full_manifest")
    manifest_index = het.index_unique(manifest_rows, "full_manifest")
    arm_indices = {
        het.ARM_INTEGER: het.index_unique(
            het.load_rows(paths.arm_integer_jsonl, "corrected_l2_integer"),
            "corrected_l2_integer",
        ),
        het.ARM_PROMPT: het.index_unique(
            het.load_rows(paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"),
            "prompt_plus_corrected_l2_integer",
        ),
    }
    candidates = build_candidates(manifest_index, arm_indices)
    selected, reserve, allocation = select(candidates)

    selected_records = [
        _manifest_record(manifest_index[row["sample_id"]], row, reserve=False)
        for row in selected
    ]
    reserve_records = [
        _manifest_record(manifest_index[row["sample_id"]], row, reserve=True)
        for row in reserve
    ]

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "schema_version": "1.0",
        "design": DESIGN,
        "scope": SCOPE,
        "executed_on_gpu": False,
        "model_was_run": False,
        "gate_report": {
            "path": str(GATE_REPORT_JSON),
            "sha256": gate_sha,
            "conclusion": gate_report.get("conclusion"),
            "formal_outcome": gate_report.get("formal_outcome"),
        },
        "inputs_sha256": pinned,
        "allocation": allocation,
        "selected_count": len(selected_records),
        "reserve_count": len(reserve_records),
    }
    report["contract_checks"] = [
        {"name": "every selected record is ADE20K", "ok": all(r["source_dataset"] == ADE_SOURCE_DATASET for r in selected_records)},
        {"name": "exactly 100 selected", "ok": len(selected_records) == TARGET_TOTAL},
        {"name": "selected sample_ids are unique", "ok": len({r["sample_id"] for r in selected_records}) == len(selected_records)},
        {"name": "selected and reserve are disjoint", "ok": not ({r["sample_id"] for r in selected_records} & {r["sample_id"] for r in reserve_records})},
        {"name": "both original gate bins present", "ok": {r["resolution_counterfactual"]["original_G_bin"] for r in selected_records} >= {"G<=512", "G>512"}},
        {"name": "both tasks present", "ok": {r["task"] for r in selected_records} >= {"Count", "Relation"}},
        {"name": "both original correctness classes present", "ok": {r["resolution_counterfactual"]["original_integer_correct"] for r in selected_records} >= {True, False}},
    ]
    if write:
        MANIFEST_JSONL.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_JSONL.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in selected_records),
            encoding="utf-8",
        )
        RESERVE_JSONL.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in reserve_records),
            encoding="utf-8",
        )
        report["manifest"] = {
            "path": str(MANIFEST_JSONL),
            "sha256": sha256_file(MANIFEST_JSONL),
            "count": len(selected_records),
        }
        report["reserve"] = {
            "path": str(RESERVE_JSONL),
            "sha256": sha256_file(RESERVE_JSONL),
            "count": len(reserve_records),
        }
        SELECTION_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        SELECTION_REPORT_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        report["manifest"] = {"path": str(MANIFEST_JSONL), "sha256": None, "count": len(selected_records)}
        report["reserve"] = {"path": str(RESERVE_JSONL), "sha256": None, "count": len(reserve_records)}

    return report


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    report = analyze(write=True)
    print(
        json.dumps(
            {
                "design": report["design"],
                "selected_count": report["selected_count"],
                "reserve_count": report["reserve_count"],
                "manifest": report["manifest"],
                "selection_report": str(SELECTION_REPORT_JSON),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
