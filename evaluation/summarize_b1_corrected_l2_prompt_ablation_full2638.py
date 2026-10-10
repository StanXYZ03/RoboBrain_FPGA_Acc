#!/usr/bin/env python3
"""Fail-closed summariser: B1 corrected-L2 / Prompt ablation (full2638).

Preregistered experiment:
``configs/2026-10-04-b1-corrected-l2-prompt-ablation-full2638.preregistered.json``.

This is the formal (non-diagnostic) promotion of the frozen screening100
experiment that returned ``CORRECTED_PROMPT_SELECTOR_SCREENING_PASS``.  Nothing
is tuned from that result: the manifest, arms, runner, boundary, K and scoring
are unchanged; only the sample set changes from the fixed 100-sample screening
manifest to the full 2638-sample CV-Bench manifest.

Arms (``datasets/cv_bench_full2638.jsonl``, B1 after vision block 0, K=96,
``region_grid=3``, ``local_candidate_cap=16``):

* ``corrected_l2_float``                - ``mean(norm(each of 4 patches))`` float;
* ``corrected_l2_integer``              - bit-exact integer corrected patch L2;
* ``prompt_plus_corrected_l2_integer``  - accepted integer semantic chain with
  the corrected patch L2 (frozen seed-20261003 Wt/Wv_q).

Descriptive references (never decision-driving): dense full2638 and the old
current-integer K96 full2638 JSONL.

Hard rules
----------
1. The **only** primary metric is the official
   ``score(...)["cv_bench_accuracy"]`` from
   :mod:`evaluation.build_cv_bench_pruning_report`; the paired stratified
   bootstrap lower bounds are the only decision inputs.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Every required input, every config SHA pin, every arm record ``arm`` /
   ``manifest_sha256`` / ``record_sha256`` / budget / grid / K / checkpoint
   contract and the exact 2638-unique-id identity are validated before any
   score; any failure raises :class:`InputContractError` and **no report is
   written** (fail closed).

Decision (unchanged from screening)
-----------------------------------
* ``integer_minus_float`` = arm2 - arm1; quantisation gate passes iff the paired
  95% CI lower bound >= -0.02.
* ``prompt_minus_integer_l2`` = arm3 - arm2; Prompt gain is confirmed iff the
  paired 95% CI lower bound > 0.

Outcome (ordered):

* quantisation fails -> ``CORRECTED_L2_INTEGERIZATION_FAIL``;
* else Prompt gain confirmed -> ``CORRECTED_PROMPT_SELECTOR_FULL2638_PASS``;
* else -> ``PROMPT_GAIN_NOT_CONFIRMED_FULL2638``.

Validated helpers (hashing, id indexing, grid signatures, official scoring,
per-arm metrics, paired bootstrap, report rendering primitives and the ordered
decision gate) are imported from
:mod:`evaluation.summarize_b1_corrected_l2_prompt_ablation_screening100` rather
than duplicated; this module only adds the full2638 input contract, paths,
outcome names and report assembly.
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
    paired_bootstrap_delta,
    score,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ALLOWED_ARMS,
    array_sha256,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT  # noqa: E402
from evaluation.summarize_b1_corrected_l2_prompt_ablation_screening100 import (  # noqa: E402
    ARM_FLOAT,
    ARM_INTEGER,
    ARM_PROMPT,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    BUDGET,
    CHECKPOINT_PIN_KEYS,
    CI_LEVEL,
    FORBIDDEN_PRIMARY_METRICS,
    GRID_SIGNATURE_DEFINITION,
    LOCAL_CANDIDATE_CAP,
    PRIMARY_METRIC,
    PROMPT_CI_FLOOR,
    QUANTIZATION_CI_FLOOR,
    REGION_GRID,
    SECONDARY_METRIC,
    UNWEIGHTED_METRIC,
    InputContractError,
    _arm_score_rows,
    _fmt_ci,
    _pair_score_rows,
    _pct,
    arm_metrics,
    build_joined,
    decide as _screening_decide,
    expected_checkpoint_hashes,
    grid_signature,
    index_unique,
    load_rows,
    require_file,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]

PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638.preregistered.json"
)
FULL_MANIFEST = ROOT / "datasets" / "cv_bench_full2638.jsonl"
DENSE_FULL_JSONL = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_0875_full2638_20260916"
    / "shallow_r1000.jsonl"
)
OLD_CURRENT_INTEGER_FULL_JSONL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-current-selector-cvbench"
    / "current_selector_k96.jsonl"
)
ARM_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638"
)
ARM_FLOAT_JSONL = ARM_OUTPUT_DIR / "corrected_l2_float_k96_full2638.jsonl"
ARM_INTEGER_JSONL = ARM_OUTPUT_DIR / "corrected_l2_integer_k96_full2638.jsonl"
ARM_PROMPT_JSONL = (
    ARM_OUTPUT_DIR / "prompt_plus_corrected_l2_integer_k96_full2638.jsonl"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638.json"
)
REPORT_CSV = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638.csv"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-b1-corrected-l2-prompt-ablation-full2638.md"
)

ARM_ORDER = (ARM_FLOAT, ARM_INTEGER, ARM_PROMPT)
if tuple(ALLOWED_ARMS) != ARM_ORDER:  # pragma: no cover - contract guard
    raise RuntimeError(
        f"summariser arms {ARM_ORDER} drifted from module arms {ALLOWED_ARMS}"
    )
DESCRIPTIVE_ORDER = ("dense", "old_current_integer")

FULL_SAMPLE_COUNT = 2638
SCORE_MODE = "integer"

OUTCOME_PASS = "CORRECTED_PROMPT_SELECTOR_FULL2638_PASS"
OUTCOME_PROMPT_NOT_CONFIRMED = "PROMPT_GAIN_NOT_CONFIRMED_FULL2638"
OUTCOME_QUANTIZATION_FAIL = "CORRECTED_L2_INTEGERIZATION_FAIL"
DECISION_ORDER = (
    OUTCOME_QUANTIZATION_FAIL,
    OUTCOME_PASS,
    OUTCOME_PROMPT_NOT_CONFIRMED,
)
# Explicit mapping (kept literal so a screening outcome rename is caught).
_OUTCOME_MAP = {
    "CORRECTED_PROMPT_SELECTOR_SCREENING_PASS": OUTCOME_PASS,
    "PROMPT_GAIN_NOT_CONFIRMED": OUTCOME_PROMPT_NOT_CONFIRMED,
    "CORRECTED_L2_INTEGERIZATION_FAIL": OUTCOME_QUANTIZATION_FAIL,
}

DIAGNOSTIC_SCOPE = (
    "formal full2638 CV-Bench conclusion for the frozen corrected-L2 / Prompt "
    "protocol"
)
FORMAL_STATEMENT = (
    "Formal full2638 CV-Bench evaluation of the frozen corrected-L2 / Prompt "
    "protocol. The Selector path is a software bridge only; no FPGA, latency or "
    "acceleration claim is made."
)
DESCRIPTIVE_ROLE = "descriptive_reference_only"

#: Config pin names that map onto caller-supplied :class:`InputPaths`.
LOGICAL_PIN_NAMES = {
    "full_manifest": "full_manifest",
    "dense_full_jsonl": "dense_full_jsonl",
    "old_current_integer_full_jsonl": "old_current_integer_full_jsonl",
    "corrected_l2_float_jsonl": "arm_float_jsonl",
    "corrected_l2_integer_jsonl": "arm_integer_jsonl",
    "prompt_plus_corrected_l2_integer_jsonl": "arm_prompt_jsonl",
}

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


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    full_manifest: Path = FULL_MANIFEST
    dense_full_jsonl: Path = DENSE_FULL_JSONL
    old_current_integer_full_jsonl: Path = OLD_CURRENT_INTEGER_FULL_JSONL
    arm_float_jsonl: Path = ARM_FLOAT_JSONL
    arm_integer_jsonl: Path = ARM_INTEGER_JSONL
    arm_prompt_jsonl: Path = ARM_PROMPT_JSONL


DEFAULT_PATHS = InputPaths()


# ---------------------------------------------------------------------------
# Raw-record digest (bit-identical to the frozen GPU runner)
# ---------------------------------------------------------------------------
def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself.

    Reproduces :func:`evaluation.run_hf_hardware_regular_b1_corrected_l2.record_sha256`
    exactly (sorted keys, compact separators, UTF-8) without importing the
    GPU runner (which pulls ``torch``).
    """

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _config_pinned_manifest_sha(config: Mapping[str, Any]) -> str | None:
    dataset = config.get("dataset_pins")
    if isinstance(dataset, Mapping):
        spec = dataset.get("full_manifest")
        if isinstance(spec, Mapping) and spec.get("sha256"):
            return str(spec["sha256"])
    pins = config.get("inputs_sha256")
    if isinstance(pins, Mapping):
        spec = pins.get("full_manifest")
        if isinstance(spec, Mapping) and spec.get("sha256"):
            return str(spec["sha256"])
    return None


# ---------------------------------------------------------------------------
# Arm-record contract (full2638)
# ---------------------------------------------------------------------------
def _validate_arm_records(
    index: Mapping[str, Mapping[str, Any]],
    arm: str,
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    manifest_sha256: str,
    checkpoint_hashes: Mapping[str, str] | None,
) -> None:
    """Validate one primary arm's records against the full2638 contract."""

    missing = sorted(sid for sid in index if sid not in manifest_index)
    if missing:
        raise InputContractError(
            f"arm {arm!r}: records absent from the full manifest: {missing[:10]}"
        )

    bad_arm = sorted(sid for sid, row in index.items() if str(row.get("arm")) != arm)
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
        bad = sorted(
            sid
            for sid, row in index.items()
            if field not in row or int(row[field]) != expected
        )
        if bad:
            raise InputContractError(
                f"arm {arm!r}: records must carry {field} == {expected}; "
                f"offenders: {bad[:10]}"
            )

    bad_manifest = sorted(
        sid
        for sid, row in index.items()
        if str(row.get("manifest_sha256")) != manifest_sha256
    )
    if bad_manifest:
        raise InputContractError(
            f"arm {arm!r}: records must carry manifest_sha256 == "
            f"{manifest_sha256}; offenders: {bad_manifest[:10]}"
        )

    bad_task = sorted(
        sid
        for sid, row in index.items()
        if str(row.get("task")) != str(manifest_index[sid].get("task"))
    )
    if bad_task:
        raise InputContractError(
            f"arm {arm!r}: record task does not match the full manifest; "
            f"offenders: {bad_task[:10]}"
        )

    bad_source = sorted(
        sid
        for sid, row in index.items()
        if "source" in row
        and str(row.get("source")) != str(manifest_index[sid].get("source"))
    )
    if bad_source:
        raise InputContractError(
            f"arm {arm!r}: record source does not match the full manifest; "
            f"offenders: {bad_source[:10]}"
        )

    bad_record_hash = sorted(
        sid
        for sid, row in index.items()
        if str(row.get("record_sha256")) != record_sha256(row)
    )
    if bad_record_hash:
        raise InputContractError(
            f"arm {arm!r}: record_sha256 mismatch (stale or corrupted record); "
            f"offenders: {bad_record_hash[:10]}"
        )

    bad_grid: list[str] = []
    bad_bypass: list[str] = []
    bad_k: list[str] = []
    bad_indices: list[str] = []
    bad_index_hash: list[str] = []
    for sid, row in index.items():
        try:
            grid = [int(value) for value in row["image_grid_thw"]]
            full_patch = int(row["full_patch_tokens"])
            full_llm = int(row["full_llm_visual_tokens"])
            output_count = int(row["output_group_count"])
        except (KeyError, TypeError, ValueError):
            bad_grid.append(sid)
            continue
        if len(grid) != 3 or grid[0] != 1 or grid[1] * grid[2] != full_patch:
            bad_grid.append(sid)
        if full_llm * MERGE_UNIT != full_patch:
            bad_grid.append(sid)
        if bool(row.get("identity_bypass")):
            if output_count != full_llm or output_count <= 0 or output_count > BUDGET:
                bad_bypass.append(sid)
        elif output_count != BUDGET:
            bad_k.append(sid)
        selected = [int(value) for value in row.get("selected_group_indices", [])]
        if len(selected) != output_count:
            bad_indices.append(sid)
            continue
        if selected and not all(
            later > earlier for earlier, later in zip(selected, selected[1:])
        ):
            bad_indices.append(sid)
        stored_hash = row.get("selected_group_indices_sha256")
        if stored_hash is None:
            bad_index_hash.append(sid)
        else:
            actual = array_sha256(np.asarray(selected, dtype=np.int64))
            if str(stored_hash) != actual:
                bad_index_hash.append(sid)
    if bad_grid:
        raise InputContractError(
            f"arm {arm!r}: records must carry a single-image grid with "
            f"grid[1]*grid[2] == full_patch_tokens == {MERGE_UNIT}*"
            f"full_llm_visual_tokens; offenders: {sorted(bad_grid)[:10]}"
        )
    if bad_bypass:
        raise InputContractError(
            f"arm {arm!r}: identity-bypass records must emit G == "
            f"full_llm_visual_tokens (<= K={BUDGET}); "
            f"offenders: {sorted(bad_bypass)[:10]}"
        )
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
            f"arm {arm!r}: selected_group_indices_sha256 missing or mismatched; "
            f"offenders: {sorted(bad_index_hash)[:10]}"
        )

    if checkpoint_hashes is not None:
        if arm == ARM_PROMPT:
            bad_prompt = sorted(
                sid
                for sid, row in index.items()
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
        else:
            bad_no_prompt = sorted(
                sid
                for sid, row in index.items()
                if row.get("checkpoint_sha256") is not None
            )
            if bad_no_prompt:
                raise InputContractError(
                    "no-prompt arms must not carry a checkpoint hash; offenders: "
                    f"{bad_no_prompt[:10]}"
                )


def validate(
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    dense_index: Mapping[str, Mapping[str, Any]],
    old_current_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    expected_full_count: int = FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate the full2638 preregistered input contract; fail closed."""

    manifest_ids = list(manifest_index)
    if len(manifest_ids) != expected_full_count:
        raise InputContractError(
            f"full manifest must hold {expected_full_count} unique samples, got "
            f"{len(manifest_ids)}"
        )
    manifest_set = set(manifest_ids)
    for sid, row in manifest_index.items():
        for field in ("task", "source", "source_dataset"):
            if not str(row.get(field, "")).strip():
                raise InputContractError(
                    f"full manifest sample {sid!r} is missing required metadata "
                    f"field {field!r} (fail closed)"
                )

    for label, index in (
        ("dense_full2638", dense_index),
        ("old_current_integer_k96_full2638", old_current_index),
    ):
        if len(index) != expected_full_count:
            raise InputContractError(
                f"{label} must hold {expected_full_count} unique samples, got "
                f"{len(index)}"
            )
        missing = sorted(manifest_set - set(index))
        extra = sorted(set(index) - manifest_set)
        if missing or extra:
            raise InputContractError(
                f"{label} sample_id set does not equal the full manifest: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

    for arm in ARM_ORDER:
        index = arm_indices[arm]
        if len(index) != expected_full_count:
            raise InputContractError(
                f"arm {arm!r} must hold exactly {expected_full_count} unique "
                f"samples, got {len(index)}"
            )
        missing = sorted(manifest_set - set(index))
        extra = sorted(set(index) - manifest_set)
        if missing or extra:
            raise InputContractError(
                f"arm {arm!r} sample_id set does not equal the full manifest: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

    if manifest_sha256 is None:
        raise InputContractError(
            "a full-manifest SHA256 is required to validate arm records "
            "(fail closed)"
        )

    for arm in ARM_ORDER:
        _validate_arm_records(
            arm_indices[arm],
            arm,
            manifest_index=manifest_index,
            manifest_sha256=manifest_sha256,
            checkpoint_hashes=checkpoint_hashes,
        )

    # Grid consistency across the three new arms (same unified B1 forward).
    grid_mismatches: list[dict[str, Any]] = []
    for sid in manifest_ids:
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
        "full_manifest_sample_count": len(manifest_ids),
        "dense_full_sample_count": len(dense_index),
        "old_current_integer_full_sample_count": len(old_current_index),
        "arm_sample_counts": {arm: len(arm_indices[arm]) for arm in ARM_ORDER},
        "sample_ids_identical": True,
        "duplicate_sample_ids": {},
        "exactly_2638_unique_ids_per_arm": True,
        "all_arms_present_and_identical_ids": True,
        "all_arms_budget_96_region_grid_3_local_cap_16": True,
        "all_non_bypass_arms_emit_K_96": True,
        "identity_bypass_matches_G": True,
        "selected_indices_strictly_ascending_and_hashed": True,
        "record_hashes_verified": True,
        "manifest_sha256_verified": True,
        "checkpoint_hashes_verified": checkpoint_hashes is not None,
        "input_grid_consistent_across_arms": True,
        "grid_signature_definition": GRID_SIGNATURE_DEFINITION,
        "checks": [
            "exactly 2638 identical unique sample_ids in the manifest and all five inputs",
            "no duplicate sample_id in any input",
            "every arm record carries arm / budget=96 / region_grid=3 / cap=16",
            "every arm record manifest_sha256 equals the pinned full-manifest SHA256",
            "every arm record task/source matches the full manifest",
            "every arm record record_sha256 re-verifies",
            "every non-bypass record emits K=96; bypass emits G == full_llm_visual_tokens",
            "selected_group_indices strictly ascending, length == K, hash re-verified",
            "prompt arm checkpoint hashes match the config pins; no-prompt arms carry none",
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
# Decision (unchanged gate, full2638 outcome names)
# ---------------------------------------------------------------------------
def decide(
    integer_minus_float_ci: Sequence[float],
    prompt_minus_integer_ci: Sequence[float],
) -> dict[str, Any]:
    """Apply the preregistered ordered decision rule (primary metric only)."""

    result = _screening_decide(integer_minus_float_ci, prompt_minus_integer_ci)
    try:
        result["outcome"] = _OUTCOME_MAP[result["outcome"]]
    except KeyError as exc:  # pragma: no cover - contract guard
        raise RuntimeError(
            f"unmapped screening outcome {result['outcome']!r}"
        ) from exc
    result["ordered_rules"] = list(DECISION_ORDER)
    result["formal_full2638_conclusion"] = True
    return result


# ---------------------------------------------------------------------------
# Official scoring and paired bootstrap
# ---------------------------------------------------------------------------
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


def analyze(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    dense_full_rows: Sequence[Mapping[str, Any]],
    old_current_rows: Sequence[Mapping[str, Any]],
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_full_count: int = FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    checkpoint_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate inputs and compute the formal result (no file I/O)."""

    manifest_index = index_unique(manifest_rows, "full_manifest")
    dense_index = index_unique(dense_full_rows, "dense_full2638")
    old_current_index = index_unique(
        old_current_rows, "old_current_integer_k96_full2638"
    )
    arm_indices = {arm: index_unique(arm_rows[arm], arm) for arm in ARM_ORDER}

    validation = validate(
        manifest_index=manifest_index,
        dense_index=dense_index,
        old_current_index=old_current_index,
        arm_indices=arm_indices,
        expected_full_count=expected_full_count,
        manifest_sha256=manifest_sha256,
        checkpoint_hashes=checkpoint_hashes,
    )

    joined = build_joined(
        list(manifest_index),
        manifest_index,
        {
            ARM_FLOAT: arm_indices[ARM_FLOAT],
            ARM_INTEGER: arm_indices[ARM_INTEGER],
            ARM_PROMPT: arm_indices[ARM_PROMPT],
            "dense": dense_index,
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
    expected_full_count: int = FULL_SAMPLE_COUNT,
    verify_pins: bool = True,
    verify_outputs: bool = True,
    manifest_sha256: str | None = None,
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

    if manifest_sha256 is None:
        manifest_sha256 = _config_pinned_manifest_sha(config)
    if manifest_sha256 is None:
        require_file(paths.full_manifest, "full_manifest")
        manifest_sha256 = sha256_file(paths.full_manifest)

    analysis = analyze(
        manifest_rows=load_rows(paths.full_manifest, "full_manifest"),
        dense_full_rows=load_rows(paths.dense_full_jsonl, "dense_full2638"),
        old_current_rows=load_rows(
            paths.old_current_integer_full_jsonl,
            "old_current_integer_k96_full2638",
        ),
        arm_rows={
            ARM_FLOAT: load_rows(paths.arm_float_jsonl, "corrected_l2_float"),
            ARM_INTEGER: load_rows(
                paths.arm_integer_jsonl, "corrected_l2_integer"
            ),
            ARM_PROMPT: load_rows(
                paths.arm_prompt_jsonl, "prompt_plus_corrected_l2_integer"
            ),
        },
        expected_full_count=expected_full_count,
        manifest_sha256=manifest_sha256,
        checkpoint_hashes=expected_checkpoint_hashes(config),
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "report_scope": DIAGNOSTIC_SCOPE,
        "diagnostic_only": False,
        "formal_full2638_conclusion": True,
        "explicit_non_formal_statement": "",
        "formal_statement": FORMAL_STATEMENT,
        "promotion_of": (
            "2026-10-04-b1-corrected-l2-prompt-ablation-screening100 "
            "(CORRECTED_PROMPT_SELECTOR_SCREENING_PASS; not tuned)"
        ),
        "fixed_configuration": {
            "manifest": str(paths.full_manifest),
            "sample_count": expected_full_count,
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
            "Software Selector bridge only; no FPGA, latency or acceleration claim.",
            "Dense and the old current-integer selector are descriptive references only and never drive the decision.",
            "The decision uses score(...)[\"cv_bench_accuracy\"] exclusively, via paired stratified bootstrap CI lower bounds.",
            "Forbidden primary metrics (pointing/grounding/ndcg/retained_mass) are never reported.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# B1 corrected-L2 / Prompt 消融（full2638 正式确认）")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append(f"- 升格自：{report['promotion_of']}")
    lines.append("> " + report["formal_statement"])
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
        f"- 样本：`{cfg['manifest']}`（{cfg['sample_count']}）；K={cfg['budget_K']}；"
        f"切点：{cfg['boundary']}；region_grid={cfg['region_grid']}；"
        f"local_cap={cfg['local_candidate_cap']}"
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
    lines.append(f"- full manifest 样本数：{v['full_manifest_sample_count']}")
    lines.append(f"- 三臂样本数：{v['arm_sample_counts']}")
    lines.append(
        f"- 每臂恰好 2638 个唯一 ID 且与 manifest 一致："
        f"{'✅' if v['exactly_2638_unique_ids_per_arm'] else '❌'}"
    )
    lines.append(
        f"- 三臂 budget=96 / region_grid=3 / local_cap=16："
        f"{'✅' if v['all_arms_budget_96_region_grid_3_local_cap_16'] else '❌'}"
    )
    lines.append(
        f"- 非 bypass 臂输出 K=96；bypass 输出 G："
        f"{'✅' if v['all_non_bypass_arms_emit_K_96'] and v['identity_bypass_matches_G'] else '❌'}"
    )
    lines.append(
        f"- record_sha256 / manifest_sha256 / checkpoint 校验："
        f"{'✅' if v['record_hashes_verified'] and v['manifest_sha256_verified'] else '❌'}"
    )
    lines.append(
        f"- 三臂输入 grid 一致（{v['grid_signature_definition']}）："
        f"{'✅' if v['input_grid_consistent_across_arms'] else '❌'}"
    )
    lines.append("")
    lines.append("## 3. 五臂正式主分")
    lines.append("")
    lines.append("| 臂 | 角色 | 样本 | **CV-Bench 主分** | unweighted（次要） |")
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
    for c in report["comparisons"].values():
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
        "- dense 与旧 current-integer 为 descriptive reference，**不得覆盖**上述决策。"
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
    expected_full_count: int = FULL_SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    root: Path = ROOT,
) -> tuple[int, dict[str, Any] | None]:
    """Build the report and write it only on success (fail closed)."""

    try:
        report = build_report(
            paths,
            expected_full_count=expected_full_count,
            verify_pins=verify_pins,
            verify_outputs=verify_outputs,
            manifest_sha256=manifest_sha256,
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
