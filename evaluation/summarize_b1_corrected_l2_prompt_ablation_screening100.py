#!/usr/bin/env python3
"""Fail-closed summariser: B1 corrected-L2 / Prompt ablation (screening100).

Preregistered experiment:
``configs/2026-10-04-b1-corrected-l2-prompt-ablation-screening100.preregistered.json``.

Arms (fixed 100-sample ``datasets/cv_bench_screening100.jsonl``, B1 after vision
block 0, K=96, region_grid=3, local_candidate_cap=16):

* ``corrected_l2_float``                - ``mean(norm(each of 4 patches))`` float;
* ``corrected_l2_integer``              - bit-exact integer corrected patch L2;
* ``prompt_plus_corrected_l2_integer``  - accepted integer semantic chain with
  the corrected patch L2 (frozen seed-20261003 Wt/Wv_q).

Descriptive references (never decision-driving): dense, the old defective-L2
screening100 JSONL and the old current-integer full2638 JSONL.

Hard rules
----------
1. The **only** primary metric is the official
   ``score(...)["cv_bench_accuracy"]`` from
   :mod:`evaluation.build_cv_bench_pruning_report`.  The plain per-sample mean is
   exposed strictly as ``unweighted_sample_accuracy`` and never drives the
   decision.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Inputs are validated and config-pinned SHA256 values re-verified before any
   score; any missing file or failed check raises :class:`InputContractError`
   and **no report is written** (fail closed).

Decision
--------
* ``integer_minus_float`` = arm2 - arm1; quantisation gate passes iff the paired
  95% CI lower bound >= -0.02.
* ``prompt_minus_integer_l2`` = arm3 - arm2; Prompt gain is confirmed iff the
  paired 95% CI lower bound > 0.

Outcome (ordered):

* quantisation fails -> ``CORRECTED_L2_INTEGERIZATION_FAIL``;
* else Prompt gain confirmed -> ``CORRECTED_PROMPT_SELECTOR_SCREENING_PASS``;
* else -> ``PROMPT_GAIN_NOT_CONFIRMED``.

Paired bootstrap reuses
:func:`evaluation.build_cv_bench_pruning_report.paired_bootstrap_delta`,
stratified by ``task::source_dataset``, seed 20261004, 5000 repeats.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ALLOWED_ARMS,
    array_sha256,
)
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-screening100.preregistered.json"
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
OLD_DEFECTIVE_L2_SCREENING_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-l2-diagnostic"
    / "b1_l2_k96_screening100.jsonl"
)
OLD_CURRENT_INTEGER_FULL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-current-selector-cvbench"
    / "current_selector_k96.jsonl"
)
ARM_OUTPUT_DIR = (
    ROOT / "artifacts" / "token_selector" / "2026-10-04-b1-corrected-l2-prompt-ablation"
)
ARM_FLOAT_JSONL = ARM_OUTPUT_DIR / "corrected_l2_float_k96_screening100.jsonl"
ARM_INTEGER_JSONL = ARM_OUTPUT_DIR / "corrected_l2_integer_k96_screening100.jsonl"
ARM_PROMPT_JSONL = (
    ARM_OUTPUT_DIR / "prompt_plus_corrected_l2_integer_k96_screening100.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-screening100.json"
)
REPORT_CSV = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-screening100.csv"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-screening100.md"
)

PRIMARY_METRIC = 'score(...)["cv_bench_accuracy"]'
UNWEIGHTED_METRIC = "unweighted_sample_accuracy"
SECONDARY_METRIC = 'score(...)["accuracy"]'
FORBIDDEN_PRIMARY_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

ARM_FLOAT = "corrected_l2_float"
ARM_INTEGER = "corrected_l2_integer"
ARM_PROMPT = "prompt_plus_corrected_l2_integer"
ARM_ORDER = (ARM_FLOAT, ARM_INTEGER, ARM_PROMPT)
if tuple(ALLOWED_ARMS) != ARM_ORDER:  # pragma: no cover - contract guard
    raise RuntimeError(
        f"summariser arms {ARM_ORDER} drifted from module arms {ALLOWED_ARMS}"
    )
DESCRIPTIVE_ORDER = ("dense", "old_defective_l2", "old_current_integer")

BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
SCREENING_SAMPLE_COUNT = 100
FULL_SAMPLE_COUNT = 2638
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
CI_LEVEL = 0.95
QUANTIZATION_CI_FLOOR = -0.02
PROMPT_CI_FLOOR = 0.0
SCORE_MODE = "integer"

OUTCOME_PASS = "CORRECTED_PROMPT_SELECTOR_SCREENING_PASS"
OUTCOME_PROMPT_NOT_CONFIRMED = "PROMPT_GAIN_NOT_CONFIRMED"
OUTCOME_QUANTIZATION_FAIL = "CORRECTED_L2_INTEGERIZATION_FAIL"
DECISION_ORDER = (
    OUTCOME_PASS,
    OUTCOME_PROMPT_NOT_CONFIRMED,
    OUTCOME_QUANTIZATION_FAIL,
)

DIAGNOSTIC_SCOPE = (
    "diagnostic screening100; NOT a formal full2638 CV-Bench conclusion"
)
GRID_SIGNATURE_DEFINITION = (
    "(tuple(image_grid_thw), full_patch_tokens, full_llm_visual_tokens)"
)
DESCRIPTIVE_ROLE = "descriptive_reference_only"

#: Config pin names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = {
    "screening_manifest": "screening_manifest",
    "full_manifest": "full_manifest",
    "dense_full_jsonl": "dense_full_jsonl",
    "old_defective_l2_screening_jsonl": "old_defective_l2_screening_jsonl",
    "old_current_integer_full_jsonl": "old_current_integer_full_jsonl",
    "corrected_l2_float_jsonl": "arm_float_jsonl",
    "corrected_l2_integer_jsonl": "arm_integer_jsonl",
    "prompt_plus_corrected_l2_integer_jsonl": "arm_prompt_jsonl",
}

#: Raw-record fields that must equal the config's pinned checkpoint hashes.
CHECKPOINT_PIN_KEYS = ("checkpoint_sha256", "wv_sha256", "wv_scale_sha256")

CSV_FIELDS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "ground_truth",
) + tuple(
    field
    for arm in (*ARM_ORDER, *DESCRIPTIVE_ORDER)
    for field in (f"{arm}_answer", f"{arm}_correct")
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the preregistered contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    screening_manifest: Path = SCREENING_MANIFEST
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    old_defective_l2_screening_jsonl: Path = OLD_DEFECTIVE_L2_SCREENING_JSONL
    old_current_integer_full_jsonl: Path = OLD_CURRENT_INTEGER_FULL_JSONL
    arm_float_jsonl: Path = ARM_FLOAT_JSONL
    arm_integer_jsonl: Path = ARM_INTEGER_JSONL
    arm_prompt_jsonl: Path = ARM_PROMPT_JSONL


DEFAULT_PATHS = InputPaths()


# ---------------------------------------------------------------------------
# Deterministic helpers
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
    """Index rows by ``sample_id``; duplicate or missing ids are hard errors."""

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
            f"input '{label}' has rows without sample_id at {missing_ids[:10]}"
        )
    if duplicates:
        raise InputContractError(
            f"input '{label}' has duplicate sample_id values: "
            f"{sorted(set(duplicates))[:10]}"
        )
    return index


def grid_signature(row: Mapping[str, Any]) -> tuple[tuple[int, ...], int, int]:
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


def _validate_arm_records(
    index: Mapping[str, Mapping[str, Any]], arm: str
) -> None:
    bad_arm = sorted(
        sid for sid, row in index.items() if str(row.get("arm")) != arm
    )
    if bad_arm:
        raise InputContractError(
            f"arm {arm!r}: records must carry arm == {arm!r}; "
            f"offenders: {bad_arm[:10]}"
        )
    for field, expected in (
        ("requested_budget", BUDGET),
        ("region_grid", REGION_GRID),
        ("local_candidate_cap", LOCAL_CANDIDATE_CAP),
    ):
        bad: list[str] = []
        for sid, row in index.items():
            if field not in row:
                bad.append(sid)
                continue
            try:
                value = int(row[field])
            except (TypeError, ValueError):
                bad.append(sid)
                continue
            if value != expected:
                bad.append(sid)
        if bad:
            raise InputContractError(
                f"arm {arm!r}: records must carry {field} == {expected}; "
                f"offenders: {sorted(bad)[:10]}"
            )
    bad_k: list[str] = []
    bad_indices: list[str] = []
    bad_index_hash: list[str] = []
    for sid, row in index.items():
        if "output_group_count" not in row:
            bad_k.append(sid)
            continue
        output_count = int(row["output_group_count"])
        selected = [int(value) for value in row.get("selected_group_indices", [])]
        if bool(row.get("identity_bypass")):
            if output_count <= 0:
                bad_k.append(sid)
        elif output_count != BUDGET:
            bad_k.append(sid)
        if len(selected) != output_count:
            bad_indices.append(sid)
            continue
        if selected and not all(
            later > earlier for earlier, later in zip(selected, selected[1:])
        ):
            bad_indices.append(sid)
        stored_hash = row.get("selected_group_indices_sha256")
        if stored_hash is not None:
            actual = array_sha256(np.asarray(selected, dtype=np.int64))
            if str(stored_hash) != actual:
                bad_index_hash.append(sid)
    if bad_k:
        raise InputContractError(
            f"arm {arm!r}: non-bypass records must emit K={BUDGET}; "
            f"offenders: {sorted(bad_k)[:10]}"
        )
    if bad_indices:
        raise InputContractError(
            f"arm {arm!r}: selected_group_indices must be strictly ascending and "
            f"match output_group_count; offenders: {sorted(bad_indices)[:10]}"
        )
    if bad_index_hash:
        raise InputContractError(
            f"arm {arm!r}: selected_group_indices_sha256 mismatch; "
            f"offenders: {sorted(bad_index_hash)[:10]}"
        )


def expected_checkpoint_hashes(config: Mapping[str, Any]) -> dict[str, str] | None:
    """Return the config-pinned checkpoint hashes, or ``None`` if ungated."""

    pins = config.get("checkpoint_pins")
    if not isinstance(pins, Mapping):
        return None
    try:
        return {
            "checkpoint_sha256": str(pins["projection_npz"]["sha256"]),
            "wv_sha256": str(pins["wv_int8"]["sha256"]),
            "wv_scale_sha256": str(pins["wv_per_tensor_scale"]["sha256"]),
        }
    except (KeyError, TypeError) as exc:
        raise InputContractError(
            f"config checkpoint_pins is malformed: {exc} (fail closed)"
        ) from exc


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
def validate_arms(
    *,
    screening_index: Mapping[str, Mapping[str, Any]],
    full_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    old_defective_index: Mapping[str, Mapping[str, Any]],
    old_current_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the preregistered input contract; raise ``InputContractError``."""

    screening_ids = list(screening_index)
    if len(screening_ids) != expected_screening_count:
        raise InputContractError(
            f"screening manifest must hold {expected_screening_count} unique "
            f"samples, got {len(screening_ids)}"
        )
    if expected_full_count is not None:
        if len(full_index) != expected_full_count:
            raise InputContractError(
                f"full manifest must hold {expected_full_count} unique samples, "
                f"got {len(full_index)}"
            )
        if len(dense_index) != expected_full_count:
            raise InputContractError(
                f"dense full JSONL must hold {expected_full_count} unique samples, "
                f"got {len(dense_index)}"
            )
        if len(old_current_index) != expected_full_count:
            raise InputContractError(
                f"old current-integer full JSONL must hold {expected_full_count} "
                f"unique samples, got {len(old_current_index)}"
            )
    screening_set = set(screening_ids)
    missing_from_full = sorted(screening_set - set(full_index))
    if missing_from_full:
        raise InputContractError(
            f"screening manifest sample_ids are not a subset of the full2638 "
            f"manifest: {missing_from_full[:10]}"
        )
    missing_dense = sorted(screening_set - set(dense_index))
    if missing_dense:
        raise InputContractError(
            f"dense full JSONL is missing screening sample_ids: {missing_dense[:10]}"
        )
    missing_old_current = sorted(screening_set - set(old_current_index))
    if missing_old_current:
        raise InputContractError(
            "old current-integer full JSONL is missing screening sample_ids: "
            f"{missing_old_current[:10]}"
        )

    for label, index in (
        ("old_defective_l2", old_defective_index),
        ("corrected_l2_float", arm_indices[ARM_FLOAT]),
        ("corrected_l2_integer", arm_indices[ARM_INTEGER]),
        ("prompt_plus_corrected_l2_integer", arm_indices[ARM_PROMPT]),
    ):
        index_set = set(index)
        missing = sorted(screening_set - index_set)
        extra = sorted(index_set - screening_set)
        if missing or extra:
            raise InputContractError(
                f"{label} screening100 sample_id set does not equal the screening "
                f"manifest: missing={missing[:10]} extra={extra[:10]}"
            )

    for arm in ARM_ORDER:
        _validate_arm_records(arm_indices[arm], arm)

    if checkpoint_hashes is not None:
        bad_prompt = sorted(
            sid
            for sid, row in arm_indices[ARM_PROMPT].items()
            if any(
                str(row.get(key)) != str(checkpoint_hashes[key])
                for key in CHECKPOINT_PIN_KEYS
            )
        )
        if bad_prompt:
            raise InputContractError(
                "prompt arm records must carry the config-pinned checkpoint "
                f"hashes; offenders: {bad_prompt[:10]}"
            )
        bad_no_prompt = sorted(
            sid
            for arm in (ARM_FLOAT, ARM_INTEGER)
            for sid, row in arm_indices[arm].items()
            if row.get("checkpoint_sha256") is not None
        )
        if bad_no_prompt:
            raise InputContractError(
                "no-prompt arms must not carry a checkpoint hash; offenders: "
                f"{bad_no_prompt[:10]}"
            )

    # Grid consistency across the three new arms (same unified B1 forward).
    grid_mismatches: list[dict[str, Any]] = []
    for sid in screening_ids:
        signatures = {
            arm: grid_signature(arm_indices[arm][sid]) for arm in ARM_ORDER
        }
        if len(set(signatures.values())) != 1:
            grid_mismatches.append(
                {
                    "sample_id": sid,
                    **{
                        arm: list(sig[0]) + [sig[1], sig[2]]
                        for arm, sig in signatures.items()
                    },
                }
            )
    if grid_mismatches:
        raise InputContractError(
            "the three corrected-L2 arms disagree on the input grid for "
            f"{len(grid_mismatches)} samples; first: {grid_mismatches[0]}"
        )

    return {
        "screening_sample_count": len(screening_ids),
        "full_manifest_sample_count": len(full_index),
        "dense_full_sample_count": len(dense_index),
        "old_current_integer_full_sample_count": len(old_current_index),
        "old_defective_l2_sample_count": len(old_defective_index),
        "arm_sample_counts": {arm: len(arm_indices[arm]) for arm in ARM_ORDER},
        "sample_ids_identical": True,
        "duplicate_sample_ids": {},
        "all_arms_present_and_identical_ids": True,
        "all_arms_budget_96_region_grid_3_local_cap_16": True,
        "all_non_bypass_arms_emit_K_96": True,
        "selected_indices_strictly_ascending_and_hashed": True,
        "checkpoint_hashes_verified": checkpoint_hashes is not None,
        "input_grid_consistent_across_arms": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "checks": [
            "100 identical unique sample_ids across manifest and all six arms",
            "no duplicate sample_id in any input",
            "every new-arm record carries arm / budget=96 / region_grid=3 / cap=16",
            "every non-bypass new-arm record emits K=96",
            "selected_group_indices strictly ascending, length == K, hash re-verified",
            "input grid identical across the three corrected-L2 arms",
        ],
    }


# ---------------------------------------------------------------------------
# Config pin verification
# ---------------------------------------------------------------------------
def _logical_pin_path(name: str, paths: InputPaths) -> Path | None:
    attribute = LOGICAL_PIN_NAMES.get(name)
    return getattr(paths, attribute) if attribute is not None else None


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
        logical = _logical_pin_path(str(name), paths)
        if expected is None:
            if logical is not None:
                require_file(logical, str(name))
            continue
        expected = str(expected).lower()
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
                f"input '{label}' sha256 mismatch: expected {expected}, got "
                f"{actual} (fail closed)"
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


def verify_output_paths(
    config: Mapping[str, Any],
    paths: InputPaths = DEFAULT_PATHS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Confirm the config's pinned arm output paths match the summariser paths."""

    output_paths = config.get("output_paths")
    if not isinstance(output_paths, Mapping):
        raise InputContractError("config must declare output_paths")
    expected = {
        ARM_FLOAT: paths.arm_float_jsonl,
        ARM_INTEGER: paths.arm_integer_jsonl,
        ARM_PROMPT: paths.arm_prompt_jsonl,
    }
    checks: list[dict[str, Any]] = []

    def _normalise(value: Any) -> Path:
        candidate = Path(str(value))
        if not candidate.is_absolute():
            candidate = root / candidate
        return candidate.resolve()

    for arm, path in expected.items():
        declared = output_paths.get(arm)
        declared_path = (
            declared.get("path") if isinstance(declared, Mapping) else declared
        )
        if declared_path is None or _normalise(declared_path) != path.resolve():
            raise InputContractError(
                f"config output_paths[{arm!r}]={declared_path!r} != summariser "
                f"path {path} (fail closed)"
            )
        checks.append({"arm": arm, "path": str(path), "matches": True})
    return {"output_paths_verified": True, "checks": checks}


# ---------------------------------------------------------------------------
# Official scoring and paired bootstrap
# ---------------------------------------------------------------------------
def build_joined(
    screening_ids: Sequence[str],
    screening_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build official joined records; correctness is recomputed by the scorer."""

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


def _arm_score_rows(
    joined: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    """Rows for the frozen ``score`` with collision-free generic keys.

    The frozen scorer derives ``<correct_key>.replace("_correct", "_answer")``;
    the corrected-L2 arm names contain the substring ``corrected`` (hence
    ``_correct``), so the per-arm correctness columns are remapped to the generic
    ``correct`` / ``answer`` pair to keep the official scorer bit-exact.
    """

    rows: list[dict[str, Any]] = []
    for record in joined:
        rows.append(
            {
                "task": record["task"],
                "type": record["type"],
                "source": record["source"],
                "source_dataset": record["source_dataset"],
                "correct": record[f"{arm}_correct"],
                "answer": record[f"{arm}_answer"],
            }
        )
    return rows


def _pair_score_rows(
    joined: Sequence[Mapping[str, Any]], left: str, right: str
) -> list[dict[str, Any]]:
    """Rows for the paired bootstrap with collision-free generic keys."""

    rows: list[dict[str, Any]] = []
    for record in joined:
        rows.append(
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
        )
    return rows


def arm_metrics(joined: Sequence[Mapping[str, Any]], arm: str, label: str) -> dict[str, Any]:
    """Official metrics for one arm; primary is ``cv_bench_accuracy`` only."""

    result = score(_arm_score_rows(joined, arm), "correct")
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
    joined: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
    *,
    decision_relevant: bool,
) -> dict[str, Any]:
    """Paired stratified bootstrap on the official CV-Bench primary only."""

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
        "decision_relevant": bool(decision_relevant),
    }


def decide(
    integer_minus_float_ci: Sequence[float],
    prompt_minus_integer_ci: Sequence[float],
) -> dict[str, Any]:
    """Apply the preregistered ordered decision rule (primary metric only)."""

    quant_lower, quant_upper = (
        float(integer_minus_float_ci[0]),
        float(integer_minus_float_ci[1]),
    )
    prompt_lower, prompt_upper = (
        float(prompt_minus_integer_ci[0]),
        float(prompt_minus_integer_ci[1]),
    )
    quantization_gate_passed = quant_lower >= QUANTIZATION_CI_FLOOR
    prompt_gain_confirmed = prompt_lower > PROMPT_CI_FLOOR
    if not quantization_gate_passed:
        outcome = OUTCOME_QUANTIZATION_FAIL
        reason = (
            f"integer_minus_float CI lower {quant_lower:.6f} < "
            f"{QUANTIZATION_CI_FLOOR}"
        )
    elif prompt_gain_confirmed:
        outcome = OUTCOME_PASS
        reason = (
            f"quantization gate passes (CI lower {quant_lower:.6f} >= "
            f"{QUANTIZATION_CI_FLOOR}) and prompt_minus_integer_l2 CI lower "
            f"{prompt_lower:.6f} > {PROMPT_CI_FLOOR}"
        )
    else:
        outcome = OUTCOME_PROMPT_NOT_CONFIRMED
        reason = (
            f"quantization gate passes (CI lower {quant_lower:.6f} >= "
            f"{QUANTIZATION_CI_FLOOR}) but prompt_minus_integer_l2 CI lower "
            f"{prompt_lower:.6f} <= {PROMPT_CI_FLOOR}"
        )
    return {
        "outcome": outcome,
        "reason": reason,
        "metric": PRIMARY_METRIC,
        "ordered_rules": list(DECISION_ORDER),
        "uses_only_primary_metric": True,
        "integer_minus_float": {
            "ci95": [quant_lower, quant_upper],
            "ci_lower": quant_lower,
            "quantization_gate_passed": quantization_gate_passed,
            "threshold_ci_lower": QUANTIZATION_CI_FLOOR,
        },
        "prompt_minus_integer_l2": {
            "ci95": [prompt_lower, prompt_upper],
            "ci_lower": prompt_lower,
            "prompt_gain_confirmed": prompt_gain_confirmed,
            "threshold_ci_lower": PROMPT_CI_FLOOR,
        },
        "descriptive_references_cannot_override": True,
    }


def analyze(
    *,
    screening_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    dense_full_rows: Sequence[Mapping[str, Any]],
    old_defective_rows: Sequence[Mapping[str, Any]],
    old_current_rows: Sequence[Mapping[str, Any]],
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate inputs and compute the diagnostic result (no file I/O)."""

    screening_index = index_unique(screening_rows, "screening_manifest")
    full_index = index_unique(full_rows, "full_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    old_defective_index = index_unique(
        old_defective_rows, "old_defective_l2_screening100"
    )
    old_current_index = index_unique(
        old_current_rows, "old_current_integer_k96_full2638"
    )
    arm_indices = {
        arm: index_unique(arm_rows[arm], arm) for arm in ARM_ORDER
    }

    validation = validate_arms(
        screening_index=screening_index,
        full_index=full_index,
        dense_index=dense_index,
        old_defective_index=old_defective_index,
        old_current_index=old_current_index,
        arm_indices=arm_indices,
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
        checkpoint_hashes=checkpoint_hashes,
    )

    joined = build_joined(
        list(screening_index),
        screening_index,
        {
            ARM_FLOAT: arm_indices[ARM_FLOAT],
            ARM_INTEGER: arm_indices[ARM_INTEGER],
            ARM_PROMPT: arm_indices[ARM_PROMPT],
            "dense": dense_index,
            "old_defective_l2": old_defective_index,
            "old_current_integer": old_current_index,
        },
    )

    metrics = {
        ARM_FLOAT: arm_metrics(joined, ARM_FLOAT, "corrected_l2_float"),
        ARM_INTEGER: arm_metrics(joined, ARM_INTEGER, "corrected_l2_integer"),
        ARM_PROMPT: arm_metrics(
            joined, ARM_PROMPT, "prompt_plus_corrected_l2_integer"
        ),
        "dense": arm_metrics(joined, "dense", "dense"),
        "old_defective_l2": arm_metrics(
            joined, "old_defective_l2", "old_defective_l2"
        ),
        "old_current_integer": arm_metrics(
            joined, "old_current_integer", "old_current_integer"
        ),
    }

    comparisons = {
        "integer_minus_float": paired_comparison(
            joined, ARM_INTEGER, ARM_FLOAT, decision_relevant=True
        ),
        "prompt_minus_integer_l2": paired_comparison(
            joined, ARM_PROMPT, ARM_INTEGER, decision_relevant=True
        ),
        "dense_minus_corrected_float": paired_comparison(
            joined, "dense", ARM_FLOAT, decision_relevant=False
        ),
        "old_defective_l2_minus_corrected_float": paired_comparison(
            joined, "old_defective_l2", ARM_FLOAT, decision_relevant=False
        ),
        "old_current_integer_minus_corrected_float": paired_comparison(
            joined, "old_current_integer", ARM_FLOAT, decision_relevant=False
        ),
    }

    decision = decide(
        comparisons["integer_minus_float"]["delta_ci95"],
        comparisons["prompt_minus_integer_l2"]["delta_ci95"],
    )

    return {
        "validation": validation,
        "joined": joined,
        "metrics": metrics,
        "comparisons": comparisons,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _inputs_sha256(paths: InputPaths) -> dict[str, Any]:
    logical = {
        name: getattr(paths, attribute)
        for name, attribute in LOGICAL_PIN_NAMES.items()
    }
    result = {
        str(name): {"path": str(path), "sha256": sha256_file(path)}
        for name, path in logical.items()
    }
    result["preregistered_config"] = {
        "path": str(paths.preregistered),
        "sha256": sha256_file(paths.preregistered),
    }
    return result


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
    verify_pins: bool = True,
    verify_outputs: bool = True,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load every required input, validate, score and assemble the report.

    Any missing file, SHA pin mismatch or failed contract check raises
    :class:`InputContractError` before a report is produced (fail closed).
    """

    require_file(paths.preregistered, "preregistered_config")
    config = json.loads(paths.preregistered.read_text(encoding="utf-8"))
    pins = verify_config_pins(config, paths, root=root) if verify_pins else None
    output_pins = (
        verify_output_paths(config, paths, root=root) if verify_outputs else None
    )

    screening_rows = load_rows(paths.screening_manifest, "screening_manifest")
    full_rows = load_rows(paths.full_manifest, "full_manifest")
    analysis = analyze(
        screening_rows=screening_rows,
        full_rows=full_rows,
        dense_full_rows=load_rows(paths.dense_full_jsonl, "dense_full2638"),
        old_defective_rows=load_rows(
            paths.old_defective_l2_screening_jsonl, "old_defective_l2_screening100"
        ),
        old_current_rows=load_rows(
            paths.old_current_integer_full_jsonl,
            "old_current_integer_k96_full2638",
        ),
        arm_rows={
            ARM_FLOAT: load_rows(paths.arm_float_jsonl, "corrected_l2_float"),
            ARM_INTEGER: load_rows(paths.arm_integer_jsonl, "corrected_l2_integer"),
            ARM_PROMPT: load_rows(
                paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"
            ),
        },
        expected_screening_count=expected_screening_count,
        expected_full_count=expected_full_count,
        checkpoint_hashes=expected_checkpoint_hashes(config),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": True,
        "formal_full2638_conclusion": False,
        "explicit_non_formal_statement": (
            "Fixed stratified 100-sample screening: a corrected-L2 / Prompt "
            "ablation signal only, NOT a formal full2638 CV-Bench conclusion. It "
            "must not be quoted as the official CV-Bench accuracy of any approach."
        ),
        "fixed_configuration": {
            "manifest": str(paths.screening_manifest),
            "budget_K": BUDGET,
            "boundary": "B1 = after vision block 0",
            "region_grid": REGION_GRID,
            "local_candidate_cap": LOCAL_CANDIDATE_CAP,
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "decoding": "greedy_argmax",
            "do_sample": False,
            "max_new_tokens": 16,
            "integration_score_mode": SCORE_MODE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
            "bootstrap_ci_level": CI_LEVEL,
            "quantization_ci_floor": QUANTIZATION_CI_FLOOR,
            "prompt_ci_floor": PROMPT_CI_FLOOR,
        },
        "primary_metric": {
            "identifier": PRIMARY_METRIC,
            "formula": "((accuracy_ADE20K + accuracy_COCO)/2 + accuracy_Omni3D)/2",
            "rule": "the only primary metric for every arm and every comparison",
        },
        "unweighted_metric": {
            "identifier": UNWEIGHTED_METRIC,
            "underlying_field": SECONDARY_METRIC,
            "rule": (
                "secondary diagnostic only; never the CV-Bench primary and never "
                "decision-driving"
            ),
        },
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "arms": {
            ARM_FLOAT: {
                "path": str(paths.arm_float_jsonl),
                "role": "primary_arm",
                "formula": "mean(norm(each of 4 patches)), float min-max",
            },
            ARM_INTEGER: {
                "path": str(paths.arm_integer_jsonl),
                "role": "primary_arm",
                "formula": "bit-exact integer corrected patch L2",
            },
            ARM_PROMPT: {
                "path": str(paths.arm_prompt_jsonl),
                "role": "primary_arm",
                "formula": (
                    "accepted integer semantic Q0.15 + corrected patch L2 Q0.15, "
                    "lambda=0.5"
                ),
            },
            "dense": {"path": str(paths.dense_full_jsonl), "role": DESCRIPTIVE_ROLE},
            "old_defective_l2": {
                "path": str(paths.old_defective_l2_screening_jsonl),
                "role": DESCRIPTIVE_ROLE,
            },
            "old_current_integer": {
                "path": str(paths.old_current_integer_full_jsonl),
                "role": DESCRIPTIVE_ROLE,
            },
        },
        "validation": analysis["validation"],
        "config_pin_verification": pins,
        "output_path_verification": output_pins,
        "metrics": analysis["metrics"],
        "comparisons": analysis["comparisons"],
        "decision": analysis["decision"],
        "joined": analysis["joined"],
        "inputs_sha256": _inputs_sha256(paths),
        "limitations": [
            "Screening100 is a fixed diagnostic subset, not the formal 2638-sample benchmark.",
            "Software Selector path only; no FPGA, latency or acceleration claim.",
            "Dense and the two old selectors are descriptive references only and never drive the decision.",
            "The decision uses score(...)[\"cv_bench_accuracy\"] exclusively.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.",
        ],
    }


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_ci(ci: Sequence[float]) -> str:
    return f"[{float(ci[0]):+.6f}, {float(ci[1]):+.6f}]"


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# B1 corrected-L2 / Prompt 消融（screening100 诊断）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append("> ⚠️ " + report["explicit_non_formal_statement"])
    lines.append("")
    lines.append(
        "- 正式主指标（唯一）：`" + report["primary_metric"]["identifier"] + "` "
        "= `((ADE20K + COCO)/2 + Omni3D)/2`；`"
        + report["unweighted_metric"]["identifier"]
        + "` 仅为次要诊断，**不得写作 CV-Bench 主分**，也不参与决策。"
    )
    lines.append("- 禁用主指标：" + ", ".join(report["forbidden_primary_metrics"]))
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
        f"repeats={cfg['bootstrap_repeats']}；quantization CI 下界阈值="
        f"{cfg['quantization_ci_floor']}；prompt CI 下界阈值={cfg['prompt_ci_floor']}"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- screening 样本数：{v['screening_sample_count']}")
    lines.append(f"- 三臂样本数：{v['arm_sample_counts']}")
    lines.append(f"- 100 个 sample_id 完全一致：{'✅' if v['sample_ids_identical'] else '❌'}")
    lines.append(
        f"- 三臂 budget=96 / region_grid=3 / local_cap=16："
        f"{'✅' if v['all_arms_budget_96_region_grid_3_local_cap_16'] else '❌'}"
    )
    lines.append(
        f"- 非 bypass 臂输出 K=96：{'✅' if v['all_non_bypass_arms_emit_K_96'] else '❌'}"
    )
    lines.append(
        f"- 三臂输入 grid 一致（{v['grid_signature_definition']}）："
        f"{'✅' if v['input_grid_consistent_across_arms'] else '❌'}"
    )
    lines.append("")
    lines.append("## 3. 六臂正式主分")
    lines.append("")
    lines.append(
        "| 臂 | 角色 | 样本 | **CV-Bench 主分** | unweighted（次要） |"
    )
    lines.append("|---|---|---:|---:|---:|")
    for arm in (*ARM_ORDER, *DESCRIPTIVE_ORDER):
        m = report["metrics"][arm]
        arm_info = report["arms"][arm]
        role = "primary" if arm_info["role"] == "primary_arm" else "descriptive"
        lines.append(
            f"| {m['label']} | {role} | {m['sample_count']} | "
            f"**{_pct(m['cv_bench_accuracy'])}** | "
            f"{_pct(m['unweighted_sample_accuracy'])} |"
        )
    lines.append("")
    lines.append("## 4. 配对分层 bootstrap（95% CI，主指标）")
    lines.append("")
    lines.append("| 对比 | 决策相关 | 点估计 | 95% CI |")
    lines.append("|---|---:|---:|---:|")
    for name, c in report["comparisons"].items():
        relevant = "是" if c["decision_relevant"] else "否（descriptive）"
        lines.append(
            f"| {c['left']} − {c['right']} | {relevant} | "
            f"{c['delta_point_estimate']:+.6f} | {_fmt_ci(c['delta_ci95'])} |"
        )
    lines.append("")
    lines.append("## 5. 预注册决策")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 结论：**{d['outcome']}**")
    lines.append(f"- 依据：{d['reason']}")
    q = d["integer_minus_float"]
    p = d["prompt_minus_integer_l2"]
    lines.append(
        f"- integer_minus_float：{_fmt_ci(q['ci95'])}，下界 {q['ci_lower']:+.6f}；"
        f"量化门 {'✅ 通过' if q['quantization_gate_passed'] else '❌ 未通过'}"
        f"（阈值 >= {q['threshold_ci_lower']}）"
    )
    lines.append(
        f"- prompt_minus_integer_l2：{_fmt_ci(p['ci95'])}，下界 {p['ci_lower']:+.6f}；"
        f"Prompt 增益 {'✅ 确认' if p['prompt_gain_confirmed'] else '❌ 未确认'}"
        f"（阈值 > {p['threshold_ci_lower']}）"
    )
    lines.append(
        "- dense 与两个旧选择器为 descriptive reference，**不得覆盖**上述决策。"
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
# Output / CLI
# ---------------------------------------------------------------------------
def _csv_rows(joined: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in joined:
        row = {field: record.get(field) for field in CSV_FIELDS}
        rows.append(row)
    return rows


def write_reports(
    report: Mapping[str, Any],
    *,
    output_json: Path,
    output_csv: Path,
    output_md: Path,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in _csv_rows(report["joined"]):
            writer.writerow(row)
    output_md.write_text(render_markdown(report), encoding="utf-8")


def run(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    output_json: Path = REPORT_JSON,
    output_csv: Path = REPORT_CSV,
    output_md: Path = REPORT_MD,
    verify_pins: bool = True,
    verify_outputs: bool = True,
    expected_screening_count: int = SCREENING_SAMPLE_COUNT,
    expected_full_count: int | None = FULL_SAMPLE_COUNT,
    root: Path = ROOT,
) -> tuple[int, dict[str, Any] | None]:
    """Build the report and write it only on success (fail closed)."""

    try:
        report = build_report(
            paths,
            expected_screening_count=expected_screening_count,
            expected_full_count=expected_full_count,
            verify_pins=verify_pins,
            verify_outputs=verify_outputs,
            root=root,
        )
    except (InputContractError, ValueError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2, None
    write_reports(
        report, output_json=output_json, output_csv=output_csv, output_md=output_md
    )
    return 0, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=REPORT_JSON)
    parser.add_argument("--output-csv", type=Path, default=REPORT_CSV)
    parser.add_argument("--output-md", type=Path, default=REPORT_MD)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, report = run(
        DEFAULT_PATHS,
        output_json=args.output_json,
        output_csv=args.output_csv,
        output_md=args.output_md,
    )
    if code == 0 and report is not None:
        print(
            json.dumps(
                {
                    "output_json": str(args.output_json),
                    "output_csv": str(args.output_csv),
                    "output_md": str(args.output_md),
                    "scope": report["report_scope"],
                    "decision": report["decision"]["outcome"],
                },
                ensure_ascii=False,
            )
        )
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
