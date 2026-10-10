#!/usr/bin/env python3
"""Decision-only fail-closed summariser for the B1 oracle challenge/control v2.

Connects exactly four sources over the frozen 24-sample enriched manifest:

* ``dense``    - eligibility reference **only** (never a third accuracy arm);
* ``control``  - corrected integer-L2 B1/K96 raw JSONL;
* ``oracle``   - the new B1 answer-margin oracle raw JSONL (missing -> fail
  closed, no report written);
* ``manifest`` - the 24 enriched samples with cell / stratum / leakage labels.

Official parsing only: :func:`evaluation.build_cv_bench_pruning_report.parse_choice`
and :func:`~evaluation.build_cv_bench_pruning_report.answer_letter`.  This is a
**decision-only** summariser: it reports the pre-registered challenge-repair and
control-regression counts, per-cell repair/regression counts, per-sample
transitions and contract checks.  It deliberately does **not** report a
24-sample result as official CV-Bench accuracy and it uses **no confidence
interval and no significance language**.

Pre-registered gate (all must hold):

* challenge repairs >= 8 of the 12 samples corrected L2 got wrong;
* control regressions <= 1 of the 12 samples corrected L2 got right;
* oracle parse success 24/24;
* exact K96 layout contract 24/24 (output=96, real=87, context=9);
* every hash / config / arm / sample-id / challenge-control label verifies.

-> ``PILOT_GO`` otherwise ``PILOT_STOP``.  Any contract failure raises
:class:`InputContractError` and the formal report is not written.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.b1_answer_margin_oracle import (  # noqa: E402
    ARM_ORACLE,
    BUDGET,
    EXPECTED_CONTEXT,
    EXPECTED_OUTPUT,
    EXPECTED_REAL,
    LOCAL_CANDIDATE_CAP,
    REGION_GRID_FIXED,
    array_sha256,
    file_sha256,
    record_sha256,
)
from evaluation.b1_answer_margin_oracle_challenge_control import (  # noqa: E402
    CELLS,
    CHALLENGE,
    CONTROL,
    GATE_CHALLENGE_REPAIR_MIN,
    GATE_CONTROL_REGRESSION_MAX,
    GATE_ORACLE_PARSE_MIN,
    OMNI3D_MIN_PER_SUBDATASET,
    OMNI3D_SOURCE,
    OMNI3D_SUBDATASETS,
    PER_CELL,
    PER_STRATUM,
    PILOT_SAMPLE_COUNT,
    STATUS_GO,
    STATUS_STOP,
    STRATA,
    challenge_control_label,
    omni3d_coverage,
)
from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STEM = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot24"

CONFIG = ROOT / "configs" / f"{STEM}.preregistered.json"
MANIFEST = ROOT / "datasets" / f"{STEM}.jsonl"
DENSE_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
CONTROL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
    / "corrected_l2_integer_k96_full2638.jsonl"
)
ORACLE_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / STEM
    / "oracle_answer_margin_b1_saliency_k96.jsonl"
)
REPORT_DIR = ROOT / "reports" / "token_selector"
REPORT_STEM = "2026-10-05-b1-answer-margin-oracle-challenge-control-pilot-summary"
REPORT_JSON = REPORT_DIR / f"{REPORT_STEM}.json"
REPORT_MD = REPORT_DIR / f"{REPORT_STEM}.md"
REPORT_CSV = REPORT_DIR / f"{REPORT_STEM}.csv"

__all__ = [
    "CONTROL_JSONL",
    "DENSE_JSONL",
    "MANIFEST",
    "ORACLE_JSONL",
    "InputContractError",
    "InputPaths",
    "analyze",
    "build_parser",
    "build_report",
    "main",
    "render_markdown",
]


class InputContractError(RuntimeError):
    """Raised when any summariser input violates the pre-registered contract."""


@dataclass(frozen=True)
class InputPaths:
    config: Path = CONFIG
    manifest: Path = MANIFEST
    dense: Path = DENSE_JSONL
    control: Path = CONTROL_JSONL
    oracle: Path = ORACLE_JSONL


DEFAULT_PATHS = InputPaths()


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InputContractError(f"missing required input {label!r}: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputContractError(
                f"{label} line {line_number} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise InputContractError(f"{label} line {line_number} is not an object")
        rows.append(row)
    return rows


def _index_unique(
    rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    index: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if "sample_id" not in row:
            raise InputContractError(f"{label} record is missing sample_id")
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise InputContractError(f"{label} has duplicate sample_id {sample_id!r}")
        index[sample_id] = row
    return index


def _verify_pins(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for section in ("inputs_sha256", "summarizer_inputs"):
        specs = config.get(section, {})
        if not isinstance(specs, Mapping):
            raise InputContractError(f"config {section} must be an object")
        for name, spec in specs.items():
            if not isinstance(spec, Mapping):
                raise InputContractError(f"config pin {section}.{name} must be an object")
            expected = spec.get("sha256")
            relative = spec.get("path")
            if expected is None:
                continue
            if relative is None:
                raise InputContractError(f"config pin {section}.{name} has no path")
            path = Path(str(relative))
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise InputContractError(
                    f"pinned file {path} is missing (fail closed)"
                )
            actual = file_sha256(path)
            if actual != str(expected):
                raise InputContractError(
                    f"pin {section}.{name} sha256 mismatch: expected {expected}, "
                    f"got {actual} (fail closed)"
                )
            checks.append(
                {
                    "name": f"{section}.{name}",
                    "path": str(path),
                    "sha256": actual,
                    "matches": True,
                }
            )
    return {"pins_verified": True, "checks": checks}


def _validate_oracle_record(
    sample_id: str,
    record: Mapping[str, Any],
    *,
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    if str(record.get("arm")) != ARM_ORACLE:
        raise InputContractError(
            f"oracle sample {sample_id!r} arm={record.get('arm')!r} != {ARM_ORACLE!r}"
        )
    for field, expected in (
        ("manifest_sha256", manifest_sha256),
        ("config_sha256", config_sha256),
    ):
        if str(record.get(field)) != str(expected):
            raise InputContractError(
                f"oracle sample {sample_id!r} {field} mismatch (fail closed)"
            )
    if str(record.get("record_sha256")) != record_sha256(record):
        raise InputContractError(
            f"oracle sample {sample_id!r} record_sha256 mismatch (fail closed)"
        )
    for field, expected in (
        ("requested_budget", BUDGET),
        ("region_grid", REGION_GRID_FIXED),
        ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
        ("output_group_count", EXPECTED_OUTPUT),
        ("real_group_count", EXPECTED_REAL),
        ("context_group_count", EXPECTED_CONTEXT),
    ):
        try:
            actual = int(record.get(field))
        except (TypeError, ValueError) as exc:
            raise InputContractError(
                f"oracle sample {sample_id!r} {field}={record.get(field)!r} is not "
                f"an integer (fail closed)"
            ) from exc
        if actual != int(expected):
            raise InputContractError(
                f"oracle sample {sample_id!r} {field}={actual} != {expected} "
                f"(fail closed)"
            )
    if bool(record.get("identity_bypass")):
        raise InputContractError(
            f"oracle sample {sample_id!r} is an identity bypass (fail closed)"
        )
    grid = [int(value) for value in record.get("image_grid_thw", [])]
    if len(grid) != 3 or grid[0] != 1 or grid[1] * grid[2] != int(
        record.get("full_patch_tokens", -1)
    ):
        raise InputContractError(
            f"oracle sample {sample_id!r} image_grid_thw {grid} is inconsistent"
        )
    full_groups = int(record.get("full_llm_visual_tokens", -1))
    if full_groups <= BUDGET:
        raise InputContractError(
            f"oracle sample {sample_id!r} G={full_groups} <= {BUDGET} (fail closed)"
        )
    selected = [int(value) for value in record.get("selected_group_indices", [])]
    if len(selected) != BUDGET:
        raise InputContractError(
            f"oracle sample {sample_id!r} selected {len(selected)} groups != {BUDGET}"
        )
    if any(later <= earlier for earlier, later in zip(selected, selected[1:])):
        raise InputContractError(
            f"oracle sample {sample_id!r} selected_group_indices not strictly ascending"
        )
    if sorted(selected) != selected or len(set(selected)) != len(selected):
        raise InputContractError(
            f"oracle sample {sample_id!r} selected_group_indices not unique"
        )
    label = record.get("label_tokenization") or {}
    if not bool(label.get("contract_ok")):
        raise InputContractError(
            f"oracle sample {sample_id!r} label branch contract not ok (fail closed)"
        )
    margin = record.get("margin") or {}
    margin_value = margin.get("value")
    if margin_value is None or not np.isfinite(float(margin_value)):
        raise InputContractError(
            f"oracle sample {sample_id!r} margin is not finite (fail closed)"
        )
    if not bool(record.get("model_parameter_grads_all_none")):
        raise InputContractError(
            f"oracle sample {sample_id!r} has non-None parameter grads (fail closed)"
        )
    values = np.asarray(
        record.get("saliency_group_values", []), dtype=np.float64
    ).reshape(-1)
    if values.size != full_groups:
        raise InputContractError(
            f"oracle sample {sample_id!r} saliency size {values.size} != G {full_groups}"
        )
    if str(record.get("saliency_sha256")) != array_sha256(values):
        raise InputContractError(
            f"oracle sample {sample_id!r} saliency_sha256 mismatch (fail closed)"
        )
    layout = record.get("layout") or {}
    if not bool(layout.get("passed")):
        raise InputContractError(
            f"oracle sample {sample_id!r} layout contract did not pass (fail closed)"
        )
    for field, expected in (
        ("output_group_count", EXPECTED_OUTPUT),
        ("real_group_count", EXPECTED_REAL),
        ("context_group_count", EXPECTED_CONTEXT),
    ):
        if int(layout.get(field, -1)) != int(expected):
            raise InputContractError(
                f"oracle sample {sample_id!r} layout.{field}={layout.get(field)!r} "
                f"!= {expected} (fail closed)"
            )
    if bool(layout.get("identity_bypass")):
        raise InputContractError(
            f"oracle sample {sample_id!r} layout is an identity bypass (fail closed)"
        )
    return {
        "layout_ok": True,
        "full_llm_visual_tokens": full_groups,
        "selected_group_indices": selected,
        "margin_value": float(margin_value),
    }


def _manifest_structure(
    manifest_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if len(manifest_index) != PILOT_SAMPLE_COUNT:
        raise InputContractError(
            f"pilot manifest must hold {PILOT_SAMPLE_COUNT} unique samples, got "
            f"{len(manifest_index)}"
        )
    cell_counts: dict[str, int] = {}
    stratum_counts = {CHALLENGE: 0, CONTROL: 0}
    cell_strata: dict[str, dict[str, int]] = {}
    omni_records: list[dict[str, Any]] = []
    for sample_id, record in manifest_index.items():
        meta = record.get("oracle_pilot")
        if not isinstance(meta, Mapping):
            raise InputContractError(
                f"manifest sample {sample_id!r} has no oracle_pilot metadata"
            )
        cell = str(meta.get("cell", ""))
        stratum = str(meta.get("stratum", ""))
        if cell not in {item.label() for item in CELLS}:
            raise InputContractError(
                f"manifest sample {sample_id!r} has unknown cell {cell!r}"
            )
        if stratum not in STRATA:
            raise InputContractError(
                f"manifest sample {sample_id!r} has unknown stratum {stratum!r}"
            )
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        stratum_counts[stratum] += 1
        per_cell = cell_strata.setdefault(cell, {CHALLENGE: 0, CONTROL: 0})
        per_cell[stratum] += 1
        if str(record.get("source")) == OMNI3D_SOURCE:
            omni_records.append(dict(record))
    expected_cells = {item.label() for item in CELLS}
    if set(cell_counts) != expected_cells:
        raise InputContractError(
            f"manifest cells {sorted(cell_counts)} != {sorted(expected_cells)}"
        )
    for cell, counts in cell_strata.items():
        if counts[CHALLENGE] != PER_STRATUM or counts[CONTROL] != PER_STRATUM:
            raise InputContractError(
                f"manifest cell {cell!r} has {counts[CHALLENGE]} challenge / "
                f"{counts[CONTROL]} control (expected {PER_STRATUM} each)"
            )
    if stratum_counts[CHALLENGE] != 12 or stratum_counts[CONTROL] != 12:
        raise InputContractError(
            f"manifest stratum totals are {stratum_counts} (expected 12/12)"
        )
    if len(omni_records) != 8:
        raise InputContractError(
            f"manifest holds {len(omni_records)} Omni3D samples (expected 8)"
        )
    coverage = omni3d_coverage(omni_records)
    if not all(
        coverage.get(name, 0) >= OMNI3D_MIN_PER_SUBDATASET
        for name in OMNI3D_SUBDATASETS
    ):
        raise InputContractError(
            f"Omni3D subdataset coverage {coverage} does not meet "
            f"{OMNI3D_MIN_PER_SUBDATASET} each (fail closed)"
        )
    return {
        "cell_counts": cell_counts,
        "stratum_counts": stratum_counts,
        "omni3d_subdataset_counts": coverage,
    }


def analyze(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    dense_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Validate the four sources and compute the pre-registered decision."""

    manifest_index = _index_unique(manifest_rows, "pilot_manifest")
    dense_index = _index_unique(dense_rows, "dense")
    control_index = _index_unique(control_rows, "control")
    oracle_index = _index_unique(oracle_rows, "oracle")
    structure = _manifest_structure(manifest_index)
    expected_ids = set(manifest_index)
    # The frozen full2638 dense/control outputs are pinned supersets; the oracle
    # output must match the 24-sample manifest exactly.
    if not expected_ids <= set(dense_index):
        missing = sorted(expected_ids - set(dense_index))
        raise InputContractError(
            f"dense is missing manifest sample_ids {missing[:5]} (fail closed)"
        )
    if not expected_ids <= set(control_index):
        missing = sorted(expected_ids - set(control_index))
        raise InputContractError(
            f"control is missing manifest sample_ids {missing[:5]} (fail closed)"
        )
    if set(oracle_index) != expected_ids:
        missing = sorted(expected_ids - set(oracle_index))
        extra = sorted(set(oracle_index) - expected_ids)
        raise InputContractError(
            f"oracle sample_id set != pilot manifest: missing={missing[:5]} "
            f"extra={extra[:5]} (fail closed)"
        )

    oracle_checks: dict[str, Any] = {}
    for sample_id in expected_ids:
        oracle_checks[sample_id] = _validate_oracle_record(
            sample_id,
            oracle_index[sample_id],
            manifest_sha256=manifest_sha256,
            config_sha256=config_sha256,
        )

    per_sample: list[dict[str, Any]] = []
    for sample_id, meta in manifest_index.items():
        choices = [str(choice) for choice in meta.get("choices", [])]
        truth = answer_letter(str(meta["answer"]))
        dense_correct, dense_answer = _arm_correct(meta, dense_index[sample_id], choices, truth)
        control_correct, control_answer = _arm_correct(
            meta, control_index[sample_id], choices, truth
        )
        oracle_correct, oracle_answer = _arm_correct(
            meta, oracle_index[sample_id], choices, truth
        )
        pilot = meta["oracle_pilot"]
        expected_stratum = challenge_control_label(dense_correct, control_correct)
        if expected_stratum is None:
            raise InputContractError(
                f"manifest sample {sample_id!r} is not dense-correct in the frozen "
                f"dense output (eligibility label mismatch, fail closed)"
            )
        if str(pilot.get("stratum")) != expected_stratum:
            raise InputContractError(
                f"manifest sample {sample_id!r} stratum={pilot.get('stratum')!r} != "
                f"recomputed {expected_stratum!r} (challenge/control label mismatch)"
            )
        if bool(pilot.get("dense_correct")) != bool(dense_correct):
            raise InputContractError(
                f"manifest sample {sample_id!r} dense_correct label mismatch"
            )
        if bool(pilot.get("control_correct")) != bool(control_correct):
            raise InputContractError(
                f"manifest sample {sample_id!r} control_correct label mismatch"
            )
        if str(pilot.get("selection_sha256_rank")) not in {None, ""}:
            from evaluation.b1_answer_margin_oracle import stable_sha256_rank

            if str(pilot["selection_sha256_rank"]) != stable_sha256_rank(sample_id):
                raise InputContractError(
                    f"manifest sample {sample_id!r} selection_sha256_rank mismatch"
                )
        per_sample.append(
            {
                "sample_id": sample_id,
                "task": str(meta["task"]),
                "source": str(meta["source"]),
                "source_dataset": str(meta["source_dataset"]),
                "cell": str(pilot.get("cell")),
                "stratum": str(pilot.get("stratum")),
                "ground_truth": truth,
                "dense_answer": dense_answer,
                "dense_correct": bool(dense_correct),
                "control_answer": control_answer,
                "control_correct": bool(control_correct),
                "oracle_answer": oracle_answer,
                "oracle_correct": bool(oracle_correct),
                "oracle_parse_ok": oracle_answer is not None,
                "oracle_margin": oracle_checks[sample_id]["margin_value"],
                "transition": _transition(control_correct, oracle_correct),
            }
        )

    per_sample.sort(
        key=lambda row: (
            [cell.label() for cell in CELLS].index(row["cell"]),
            STRATA.index(row["stratum"]),
            row["sample_id"],
        )
    )

    challenge_rows = [row for row in per_sample if row["stratum"] == CHALLENGE]
    control_rows_by_stratum = [row for row in per_sample if row["stratum"] == CONTROL]
    repairs = sum(
        1
        for row in challenge_rows
        if (not row["control_correct"]) and row["oracle_correct"]
    )
    regressions = sum(
        1
        for row in control_rows_by_stratum
        if row["control_correct"] and (not row["oracle_correct"])
    )
    parse_count = sum(1 for row in per_sample if row["oracle_parse_ok"])
    layout_pass = sum(
        1 for sample_id in expected_ids if oracle_checks[sample_id]["layout_ok"]
    )

    per_cell: dict[str, Any] = {}
    for cell in CELLS:
        cell_rows = [row for row in per_sample if row["cell"] == cell.label()]
        per_cell[cell.label()] = {
            "sample_count": len(cell_rows),
            "challenge_count": sum(
                1 for row in cell_rows if row["stratum"] == CHALLENGE
            ),
            "control_count": sum(1 for row in cell_rows if row["stratum"] == CONTROL),
            "challenge_repairs": sum(
                1
                for row in cell_rows
                if row["stratum"] == CHALLENGE
                and (not row["control_correct"])
                and row["oracle_correct"]
            ),
            "control_regressions": sum(
                1
                for row in cell_rows
                if row["stratum"] == CONTROL
                and row["control_correct"]
                and (not row["oracle_correct"])
            ),
            "oracle_parse_ok": sum(1 for row in cell_rows if row["oracle_parse_ok"]),
            "transitions": {
                transition: sum(1 for row in cell_rows if row["transition"] == transition)
                for transition in ("W->R", "R->W", "unchanged_correct", "unchanged_wrong")
            },
        }

    checks = {
        "challenge_repairs_ge_min": repairs >= GATE_CHALLENGE_REPAIR_MIN,
        "control_regressions_le_max": regressions <= GATE_CONTROL_REGRESSION_MAX,
        "oracle_parse_is_24": parse_count == GATE_ORACLE_PARSE_MIN,
        "exact_k96_layout_is_24": layout_pass == PILOT_SAMPLE_COUNT,
        "manifest_structure_ok": True,
        "challenge_control_labels_ok": True,
        "dense_is_eligibility_reference": all(row["dense_correct"] for row in per_sample),
        "hashes_and_ids_ok": True,
    }
    status = STATUS_GO if all(checks.values()) else STATUS_STOP

    return {
        "sample_count": PILOT_SAMPLE_COUNT,
        "per_sample": per_sample,
        "per_cell": per_cell,
        "manifest_structure": structure,
        "decision": {
            "challenge_repairs": repairs,
            "challenge_repair_min": GATE_CHALLENGE_REPAIR_MIN,
            "challenge_denominator": len(challenge_rows),
            "control_regressions": regressions,
            "control_regression_max": GATE_CONTROL_REGRESSION_MAX,
            "control_denominator": len(control_rows_by_stratum),
            "oracle_parse_count": parse_count,
            "oracle_parse_denominator": PILOT_SAMPLE_COUNT,
            "layout_contract_pass_count": layout_pass,
            "layout_contract_denominator": PILOT_SAMPLE_COUNT,
            "dense_correct_reference_count": sum(
                1 for row in per_sample if row["dense_correct"]
            ),
            "checks": checks,
            "status": status,
            "uses_confidence_interval": False,
            "is_official_cv_bench_accuracy": False,
            "dense_is_only_eligibility_reference": True,
        },
        "transitions": {
            transition: sum(1 for row in per_sample if row["transition"] == transition)
            for transition in ("W->R", "R->W", "unchanged_correct", "unchanged_wrong")
        },
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
    }


def _arm_correct(
    manifest_row: Mapping[str, Any],
    output_row: Mapping[str, Any],
    choices: Sequence[str],
    truth: str,
) -> tuple[bool, str | None]:
    parsed = parse_choice(str(output_row.get("output_text", "")), list(choices))
    return bool(parsed is not None and parsed == truth), parsed


def _transition(control_correct: bool, oracle_correct: bool) -> str:
    if not control_correct and oracle_correct:
        return "W->R"
    if control_correct and not oracle_correct:
        return "R->W"
    return "unchanged_correct" if control_correct else "unchanged_wrong"


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    config_path = paths.config if paths.config.is_absolute() else root / paths.config
    if not config_path.is_file():
        raise InputContractError(f"missing pre-registered config {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_sha256 = file_sha256(config_path)
    pins = _verify_pins(config, root)
    for path in (paths.manifest, paths.dense, paths.control, paths.oracle):
        if not path.is_file():
            raise InputContractError(f"missing required input {path} (fail closed)")

    manifest_rows = _load_jsonl(paths.manifest, "pilot_manifest")
    dense_rows = _load_jsonl(paths.dense, "dense")
    control_rows = _load_jsonl(paths.control, "control")
    oracle_rows = _load_jsonl(paths.oracle, "oracle")
    manifest_sha256 = file_sha256(paths.manifest)
    pinned_manifest = config.get("manifest", {}).get("sha256")
    if pinned_manifest is not None and str(pinned_manifest) != manifest_sha256:
        raise InputContractError(
            f"pilot manifest sha256 {manifest_sha256} != config pin {pinned_manifest}"
        )

    result = analyze(
        manifest_rows=manifest_rows,
        dense_rows=dense_rows,
        control_rows=control_rows,
        oracle_rows=oracle_rows,
        manifest_sha256=manifest_sha256,
        config_sha256=config_sha256,
    )
    result["status"] = result["decision"]["status"]
    result["date"] = "2026-10-05"
    result["design"] = config.get("design")
    result["outcome_enriched_diagnostic"] = True
    result["outcome_label_leakage"] = True
    result["non_deployable"] = True
    result["non_representative"] = True
    result["forbidden_for_official_absolute_accuracy"] = True
    result["is_label_aware_oracle"] = True
    result["ground_truth_answer_leakage"] = True
    result["deployable_selector"] = False
    result["is_paper_accuracy_claim"] = False
    result["accuracy_improvement_confirmed"] = False
    result["uses_confidence_interval"] = False
    result["pins"] = pins
    result["inputs_sha256"] = {
        "config": {"path": str(config_path), "sha256": config_sha256},
        "manifest": {"path": str(paths.manifest), "sha256": manifest_sha256},
        "dense": {"path": str(paths.dense), "sha256": file_sha256(paths.dense)},
        "control": {"path": str(paths.control), "sha256": file_sha256(paths.control)},
        "oracle": {"path": str(paths.oracle), "sha256": file_sha256(paths.oracle)},
    }
    result["required_statements"] = {
        "outcome_label_leakage": (
            "样本按 dense/control 正确性分层，属于 outcome/label leakage；"
            "非代表性、不可部署，禁止用于官方绝对精度主张。"
        ),
        "label_aware_oracle": (
            "这是 label-aware oracle 上界（ground-truth answer leakage），不是可部署 selector。"
        ),
        "not_paper_accuracy": (
            "24 样本只是泄漏 oracle 筛选，不是官方 CV-Bench 精度，不使用置信区间或"
            "『精度提升已确认』措辞。"
        ),
        "decision_only": (
            "本汇总器只输出 challenge 修复/control 回归与契约检查的 GO/STOP 决策，"
            "不报告 24 样本官方精度。"
        ),
    }
    result["forbidden_claims"] = list(config.get("forbidden_claims", []))
    result["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat()
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# B1 answer-margin oracle challenge/control pilot24 v2 汇总（decision-only）",
        "",
        f"- 日期：{report['date']}",
        f"- 状态：**{report['status']}**",
        f"- 设计：`{report['design']}`",
        "",
        "## 1. 口径",
        "",
        "- **outcome/label leakage**：样本按 dense/control 正确性分层，非代表性、不可部署；",
        "  禁止把 24 样本结果当作官方 CV-Bench 精度。",
        "- **ground-truth answer leakage**：oracle 用官方正确答案分叉 token，只是可证伪上界。",
        "- 本报告**不计算置信区间**、不使用显著性/『精度提升已确认』措辞，dense 仅资格参照。",
        "",
        "## 2. 预注册 GO/STOP 决策",
        "",
        "| 条件 | 要求 | 实际 | 通过 |",
        "| --- | --- | --- | --- |",
        f"| challenge 修复 | ≥ {decision['challenge_repair_min']}/"
        f"{decision['challenge_denominator']} | {decision['challenge_repairs']} | "
        f"{'✅' if decision['checks']['challenge_repairs_ge_min'] else '❌'} |",
        f"| control 回归 | ≤ {decision['control_regression_max']}/"
        f"{decision['control_denominator']} | {decision['control_regressions']} | "
        f"{'✅' if decision['checks']['control_regressions_le_max'] else '❌'} |",
        f"| oracle parse | 24/24 | {decision['oracle_parse_count']} | "
        f"{'✅' if decision['checks']['oracle_parse_is_24'] else '❌'} |",
        f"| 精确 K96 布局 | 24/24 | {decision['layout_contract_pass_count']} | "
        f"{'✅' if decision['checks']['exact_k96_layout_is_24'] else '❌'} |",
        f"| 哈希/ID/分层标签 | 全部通过 | ✅ | ✅ |",
        "",
        f"- 结论：**{decision['status']}**（满足全部门才 GO）。",
        "- 该门只是 outcome-enriched 诊断的筛选决策，**不是精度、部署或加速结论**。",
        "",
        "## 3. 各单元修复/回归",
        "",
        "| 单元 | 样本 | challenge | control | 修复 | 回归 | W→R | R→W |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in CELLS:
        item = report["per_cell"][cell.label()]
        lines.append(
            f"| {cell.label()} | {item['sample_count']} | {item['challenge_count']} | "
            f"{item['control_count']} | {item['challenge_repairs']} | "
            f"{item['control_regressions']} | {item['transitions']['W->R']} | "
            f"{item['transitions']['R->W']} |"
        )
    lines += [
        "",
        "## 4. 逐样本迁移",
        "",
        "| sample_id | cell | stratum | truth | control | oracle | 迁移 | margin |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["per_sample"]:
        lines.append(
            f"| {row['sample_id']} | {row['cell']} | {row['stratum']} | "
            f"{row['ground_truth']} | {row['control_answer']} | {row['oracle_answer']} | "
            f"{row['transition']} | {row['oracle_margin']:.4f} |"
        )
    lines += [
        "",
        "## 5. 契约与输入 SHA256",
        "",
        f"- manifest：`{report['inputs_sha256']['manifest']['path']}` → "
        f"`{report['inputs_sha256']['manifest']['sha256']}`",
        f"- control：`{report['inputs_sha256']['control']['path']}` → "
        f"`{report['inputs_sha256']['control']['sha256']}`",
        f"- oracle：`{report['inputs_sha256']['oracle']['path']}` → "
        f"`{report['inputs_sha256']['oracle']['sha256']}`",
        f"- dense（仅资格参照，不作第三臂）：`"
        f"{report['inputs_sha256']['dense']['path']}`",
        "",
        "禁止据此报告声称：" + "、".join(report["forbidden_claims"]) + "。",
        "",
    ]
    return "\n".join(lines)


def _csv_rows(per_sample: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "sample_id",
        "task",
        "source",
        "source_dataset",
        "cell",
        "stratum",
        "ground_truth",
        "dense_answer",
        "dense_correct",
        "control_answer",
        "control_correct",
        "oracle_answer",
        "oracle_correct",
        "oracle_parse_ok",
        "oracle_margin",
        "transition",
    )
    return [{field: row.get(field) for field in fields} for row in per_sample]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dense", type=Path, default=DENSE_JSONL)
    parser.add_argument("--control", type=Path, default=CONTROL_JSONL)
    parser.add_argument("--oracle", type=Path, default=ORACLE_JSONL)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = InputPaths(
        config=args.config,
        manifest=args.manifest,
        dense=args.dense,
        control=args.control,
        oracle=args.oracle,
    )
    try:
        report = build_report(paths)
    except (InputContractError, OSError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2

    if not args.no_write:
        report_dir = args.report_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / REPORT_JSON.name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (report_dir / REPORT_MD.name).write_text(
            render_markdown(report), encoding="utf-8"
        )
        rows = _csv_rows(report["per_sample"])
        with (report_dir / REPORT_CSV.name).open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(
        json.dumps(
            {
                "status": report["status"],
                "challenge_repairs": report["decision"]["challenge_repairs"],
                "control_regressions": report["decision"]["control_regressions"],
                "oracle_parse_count": report["decision"]["oracle_parse_count"],
                "layout_contract_pass_count": report["decision"][
                    "layout_contract_pass_count"
                ],
                "official_cv_bench_accuracy": None,
                "uses_confidence_interval": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
