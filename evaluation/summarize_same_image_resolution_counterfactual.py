#!/usr/bin/env python3
"""Fail-closed summariser: same-image dual-resolution counterfactual.

Preregistered experiment:
``configs/2026-10-04-same-image-resolution-counterfactual.preregistered.json``.

Four GPU arm files are required (two arms x two processor pixel budgets):

* ``corrected_l2_integer``               x ``low`` / ``high``
* ``prompt_plus_corrected_l2_integer``   x ``low`` / ``high``

Hard rules
----------
1. The **only** primary estimand is the preregistered unweighted per-sample
   paired difference

       mean_i [ (Prompt_correct - Integer_correct)_high
                - (Prompt_correct - Integer_correct)_low ]

   The official ``score(...)["cv_bench_accuracy"]`` is *undefined* inside the
   single ADE20K source (no COCO / Omni3D rows) and is therefore never computed
   here.
2. Pointing / Grounding / NDCG / retained-mass are never reported.
3. Every required input, every frozen SHA pin (manifest / config / gate report /
   seed-20261003 checkpoint / Wv / Wv scale), every arm / budget / grid / K /
   cutpoint / decoding metadata field and the exact 100-unique-id four-file
   identity are validated before any score; any missing file or failed check
   raises :class:`InputContractError` and **no report is written** (fail closed).
4. The four arm files must separate on the gate: every ``low`` record has
   ``G <= 512`` and every ``high`` record has ``G > 512``, and the per-sample
   ``G`` values must equal the authoritative processor dry-run measurements.
   Same-image pairing is enforced per sample and per pixel budget: the integer
   and prompt arms must share the identical ``image_grid_thw`` (and the
   identical ``full_llm_visual_tokens``), so only the selector arm may differ,
   and the low and high budgets must carry different grids.

Decision (preregistered)
------------------------
Resample samples with replacement *within each original stratum* (paired by
``sample_id``), seed 20261004, 5000 repeats; the 95% CI lower bound of the
primary estimand must be strictly greater than 0:

* lower bound > 0 -> ``RESOLUTION_DEPENDENT_PROMPT_GAIN_CONFIRMED``;
* otherwise      -> ``RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED``.

Either way the formal full2638 conclusion ``PROMPT_GAIN_NOT_CONFIRMED_FULL2638``
is unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from evaluation.build_cv_bench_pruning_report import (  # noqa: E402
    answer_letter,
    parse_choice,
)
from evaluation.hardware_regular_b1_corrected_l2 import (  # noqa: E402
    ARM_INTEGER,
    ARM_PROMPT_INTEGER,
    array_sha256,
)
from evaluation.hardware_regular_token_selector import MERGE_UNIT  # noqa: E402
from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

DESIGN = "2026-10-04-same-image-resolution-counterfactual"
PREREGISTERED_CONFIG = (
    ROOT
    / "configs"
    / "2026-10-04-same-image-resolution-counterfactual.preregistered.json"
)
SEPARATED_MANIFEST = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100-separated.jsonl"
)
PREREGISTERED_MANIFEST = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k100.jsonl"
)
RESERVE_MANIFEST = (
    ROOT
    / "datasets"
    / "2026-10-04-same-image-resolution-counterfactual-ade20k-reserve.jsonl"
)
GATE_SEPARATION_REPORT = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual-gate-separation.json"
)
ARM_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-same-image-resolution-counterfactual"
)
ARM_INTEGER_LOW_JSONL = ARM_OUTPUT_DIR / "corrected_l2_integer_low.jsonl"
ARM_INTEGER_HIGH_JSONL = ARM_OUTPUT_DIR / "corrected_l2_integer_high.jsonl"
ARM_PROMPT_LOW_JSONL = (
    ARM_OUTPUT_DIR / "prompt_plus_corrected_l2_integer_low.jsonl"
)
ARM_PROMPT_HIGH_JSONL = (
    ARM_OUTPUT_DIR / "prompt_plus_corrected_l2_integer_high.jsonl"
)
CHECKPOINT_PATH = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "spatial_prompt_contrastive_trainval"
    / "B_SPATIAL_KL_CONTRASTIVE"
    / "20261003"
    / "best.npz"
)
WV_PATH = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "selector_wv_scale_regularization"
    / "20261003"
    / "Wv_int8_rowmajor.bin"
)
WV_SCALE_PATH = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "selector_wv_scale_regularization"
    / "20261003"
    / "Wv_per_tensor_scale_f32le.bin"
)
REPORT_JSON = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual.json"
)
REPORT_CSV = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual.csv"
)
REPORT_MD = (
    ROOT
    / "reports"
    / "cv_bench"
    / "2026-10-04-same-image-resolution-counterfactual.md"
)

#: Frozen metadata (must match the preregistered config).
GATE_THRESHOLD = 512
BUDGET = 96
REGION_GRID = 3
LOCAL_CANDIDATE_CAP = 16
PRUNE_AFTER_LAYER = 0
CUTPOINT = "B1_after_vision_block_0"
SCORE_MODE = "integer"
DTYPE = "bfloat16"
RUN_DTYPE_LABEL = "bf16"
ATTENTION_IMPLEMENTATION = "sdpa"
DECODING = "greedy_argmax"
MAX_NEW_TOKENS = 16
SAMPLE_COUNT = 100

PIXEL_BUDGETS: dict[str, tuple[int, int]] = {
    "low": (65536, 524288),
    "high": (525312, 16777216),
}
BUDGET_ORDER = ("low", "high")

#: Checkpoint pins (replicated from the torch-free deployment constants).
CHECKPOINT_SHA256 = (
    "1f1d379e3229f310d11fee4ff620480e13fea9005abefc762c66f0d331ca72ff"
)
WV_SHA256 = "126bd8421dd128a318115c301bf9d1c7d336912150d19d2ec35203ed55d89d3b"
WV_SCALE_SHA256 = (
    "93585a3bd2d584a612b40d827f8d15f43c87acdc764d7e217d99df473428b13f"
)
DEPLOYMENT_SEED = 20261003
CHECKPOINT_PIN_KEYS = ("checkpoint_sha256", "wv_sha256", "wv_scale_sha256")

ARMS = (ARM_INTEGER, ARM_PROMPT_INTEGER)
ARM_LABELS = {
    ARM_INTEGER: "corrected_l2_integer",
    ARM_PROMPT_INTEGER: "prompt_plus_corrected_l2_integer",
}
ARM_SCORE_MODE = {ARM_INTEGER: "integer", ARM_PROMPT_INTEGER: "integer"}
ARM_USES_B1_TRAINED_WV = {ARM_INTEGER: False, ARM_PROMPT_INTEGER: True}

STRATA = (
    "G<=512::Count::integer_correct",
    "G<=512::Count::integer_wrong",
    "G<=512::Relation::integer_correct",
    "G<=512::Relation::integer_wrong",
    "G>512::Count::integer_correct",
    "G>512::Count::integer_wrong",
    "G>512::Relation::integer_correct",
    "G>512::Relation::integer_wrong",
)
GATE_BINS = ("G<=512", "G>512")

PRIMARY_ESTIMAND = "mean_i [ (Prompt-Integer)_high - (Prompt-Integer)_low ]"
PRIMARY_ESTIMAND_EXPANDED = (
    "mean_i [ (Prompt_correct - Integer_correct)_high "
    "- (Prompt_correct - Integer_correct)_low ]"
)
PRIMARY_METRIC = (
    "unweighted per-sample paired difference (the official cv_bench_accuracy is "
    "undefined inside the single ADE20K source)"
)
STRATIFIED_BY = "original_stratum"
BOOTSTRAP_SEED = 20261004
BOOTSTRAP_REPEATS = 5000
CI_LEVEL = 0.95
DECISION_THRESHOLD = 0.0

OUTCOME_CONFIRMED = "RESOLUTION_DEPENDENT_PROMPT_GAIN_CONFIRMED"
OUTCOME_NOT_CONFIRMED = "RESOLUTION_DEPENDENT_PROMPT_GAIN_NOT_CONFIRMED"
FORMAL_CONCLUSION_UNCHANGED = "PROMPT_GAIN_NOT_CONFIRMED_FULL2638"
FORBIDDEN_METRICS = ("pointing", "grounding", "ndcg", "retained_mass")

#: Fields every raw arm record must carry (config ``required_record_fields`` plus
#: the metadata the frozen B1 runner always writes).
REQUIRED_RECORD_FIELDS = (
    "sample_id",
    "task",
    "arm",
    "score_mode",
    "integration_score_mode",
    "prune_after_layer",
    "cutpoint",
    "pixel_budget_label",
    "min_pixels",
    "max_pixels",
    "image_grid_thw",
    "full_patch_tokens",
    "full_llm_visual_tokens",
    "output_group_count",
    "identity_bypass",
    "selected_group_indices",
    "selected_group_indices_sha256",
    "output_text",
    "requested_budget",
    "region_grid",
    "local_candidate_cap",
    "do_sample",
    "decoding",
    "max_new_tokens",
    "manifest_sha256",
    "checkpoint_sha256",
    "wv_sha256",
    "wv_scale_sha256",
    "deployment_seed",
    "run",
    "record_sha256",
)

CSV_FIELDS = (
    "sample_id",
    "task",
    "source",
    "source_dataset",
    "stratum",
    "original_G_bin",
    "original_integer_correct",
    "low_integer_answer",
    "low_integer_correct",
    "low_prompt_answer",
    "low_prompt_correct",
    "high_integer_answer",
    "high_integer_correct",
    "high_prompt_answer",
    "high_prompt_correct",
    "effect",
)


class InputContractError(RuntimeError):
    """Raised when a required input is missing or violates the frozen contract."""


@dataclass(frozen=True)
class InputPaths:
    preregistered: Path = PREREGISTERED_CONFIG
    separated_manifest: Path = SEPARATED_MANIFEST
    gate_separation_report: Path = GATE_SEPARATION_REPORT
    arm_integer_low_jsonl: Path = ARM_INTEGER_LOW_JSONL
    arm_integer_high_jsonl: Path = ARM_INTEGER_HIGH_JSONL
    arm_prompt_low_jsonl: Path = ARM_PROMPT_LOW_JSONL
    arm_prompt_high_jsonl: Path = ARM_PROMPT_HIGH_JSONL
    checkpoint: Path = CHECKPOINT_PATH
    wv: Path = WV_PATH
    wv_scale: Path = WV_SCALE_PATH


DEFAULT_PATHS = InputPaths()


def arm_paths(paths: InputPaths) -> dict[str, Path]:
    """Map ``"<arm>::<label>"`` to the caller-supplied arm file path."""

    return {
        f"{ARM_INTEGER}::low": paths.arm_integer_low_jsonl,
        f"{ARM_INTEGER}::high": paths.arm_integer_high_jsonl,
        f"{ARM_PROMPT_INTEGER}::low": paths.arm_prompt_low_jsonl,
        f"{ARM_PROMPT_INTEGER}::high": paths.arm_prompt_high_jsonl,
    }


if tuple(ARMS) != (ARM_INTEGER, ARM_PROMPT_INTEGER):  # pragma: no cover
    raise RuntimeError("summariser arms drifted from the frozen B1 arms")


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


def load_json_object(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputContractError(
            f"input '{label}' is not valid JSON: {exc} (fail closed)"
        ) from exc
    if not isinstance(payload, dict):
        raise InputContractError(f"input '{label}' must be a JSON object")
    return payload


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
            f"{sorted(set(duplicates))[:10]} (resume/duplicate pollution, fail closed)"
        )
    return index


def record_sha256(record: Mapping[str, Any]) -> str:
    """Canonical digest of a raw record, excluding the digest field itself.

    Reproduces :func:`evaluation.run_hf_hardware_regular_b1_corrected_l2.record_sha256`
    (sorted keys, compact separators, UTF-8) without importing the GPU runner.
    """

    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    blob = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Frozen config / gate-report pin verification
# ---------------------------------------------------------------------------
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InputContractError(message + " (fail closed)")


def verify_config(
    config: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_sample_count: int = SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Re-verify the preregistered config and every physical frozen SHA pin."""

    _require(config.get("design") == DESIGN, "config design mismatch")
    _require(
        int(config.get("frozen_sample_count", -1)) == expected_sample_count,
        "config frozen_sample_count mismatch",
    )
    _require(
        config.get("gate_threshold") == GATE_THRESHOLD,
        "config gate_threshold mismatch",
    )

    manifest_pin = config.get("manifest") or {}
    manifest_path = root / str(manifest_pin.get("path"))
    require_file(manifest_path, "config.manifest.path")
    _require(
        sha256_file(manifest_path) == str(manifest_pin.get("sha256")),
        "config manifest sha256 pin mismatch",
    )
    _require(
        int(manifest_pin.get("count", -1)) == expected_sample_count,
        "config manifest count mismatch",
    )
    if manifest_sha256 is not None:
        _require(
            str(manifest_pin.get("sha256")) == manifest_sha256,
            "config manifest pin does not match the summariser --manifest",
        )

    reserve_pin = config.get("reserve") or {}
    reserve_path = root / str(reserve_pin.get("path"))
    require_file(reserve_path, "config.reserve.path")
    _require(
        sha256_file(reserve_path) == str(reserve_pin.get("sha256")),
        "config reserve sha256 pin mismatch",
    )

    budgets = config.get("processor_pixel_budgets") or {}
    for label, (min_pixels, max_pixels) in PIXEL_BUDGETS.items():
        spec = budgets.get(label) or {}
        _require(
            int(spec.get("min_pixels", -1)) == min_pixels
            and int(spec.get("max_pixels", -1)) == max_pixels,
            f"config processor_pixel_budgets[{label!r}] mismatch",
        )

    frozen = config.get("frozen_forward") or {}
    expected_frozen = {
        "cutpoint": CUTPOINT,
        "prune_after_layer": PRUNE_AFTER_LAYER,
        "budget": BUDGET,
        "region_grid": REGION_GRID,
        "local_candidate_cap": LOCAL_CANDIDATE_CAP,
        "dtype": DTYPE,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "decoding": DECODING,
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy_generation": True,
        "deployment_seed": DEPLOYMENT_SEED,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "wv_sha256": WV_SHA256,
        "wv_scale_sha256": WV_SCALE_SHA256,
    }
    for key, value in expected_frozen.items():
        _require(
            frozen.get(key) == value,
            f"config frozen_forward[{key!r}] mismatch",
        )

    physical_pins = (
        (paths.checkpoint, CHECKPOINT_SHA256, "checkpoint"),
        (paths.wv, WV_SHA256, "Wv"),
        (paths.wv_scale, WV_SCALE_SHA256, "Wv scale"),
    )
    for path, expected, label in physical_pins:
        require_file(path, label)
        actual = sha256_file(path)
        _require(actual == expected, f"{label} frozen sha256 mismatch")

    declared_arms = {
        str(entry.get("arm")): entry for entry in config.get("arms", [])
    }
    _require(
        set(declared_arms) == set(ARMS), "config arms do not match the frozen arms"
    )
    for arm, entry in declared_arms.items():
        _require(
            entry.get("score_mode") == ARM_SCORE_MODE[arm],
            f"config arm {arm!r} score_mode mismatch",
        )
        _require(
            bool(entry.get("uses_b1_trained_wv")) == ARM_USES_B1_TRAINED_WV[arm],
            f"config arm {arm!r} uses_b1_trained_wv mismatch",
        )

    comparison = config.get("comparison") or {}
    _require(
        comparison.get("primary_estimand") == PRIMARY_ESTIMAND,
        "config primary_estimand mismatch",
    )
    _require(
        comparison.get("primary_metric") == PRIMARY_METRIC,
        "config primary_metric mismatch",
    )
    bootstrap = comparison.get("bootstrap") or {}
    _require(
        int(bootstrap.get("seed", -1)) == BOOTSTRAP_SEED
        and int(bootstrap.get("repeats", -1)) == BOOTSTRAP_REPEATS
        and float(bootstrap.get("ci_level", -1)) == CI_LEVEL
        and bootstrap.get("stratified_by") == STRATIFIED_BY,
        "config bootstrap contract mismatch",
    )
    decision_rule = comparison.get("decision_rule") or {}
    _require(
        set(decision_rule)
        == {OUTCOME_CONFIRMED, OUTCOME_NOT_CONFIRMED},
        "config decision_rule outcomes mismatch",
    )

    _require(
        str((gate_report.get("config") or {}).get("sha256"))
        == sha256_file(paths.preregistered),
        "gate report config sha256 does not match the preregistered config",
    )

    return {
        "pins_verified": True,
        "manifest_sha256": sha256_file(manifest_path),
        "reserve_sha256": sha256_file(reserve_path),
        "config_sha256": sha256_file(paths.preregistered),
        "checkpoint_sha256": sha256_file(paths.checkpoint),
        "wv_sha256": sha256_file(paths.wv),
        "wv_scale_sha256": sha256_file(paths.wv_scale),
        "checks": [
            "config design / sample count / gate threshold match",
            "config manifest and reserve SHA256 pins re-verified on disk",
            "config low / high processor pixel budgets match",
            "config frozen_forward (B1 cutpoint, K=96, grid=3, cap=16, BF16, "
            "SDPA, greedy, max_new_tokens=16, seed 20261003) matches",
            "seed-20261003 checkpoint / Wv / Wv scale SHA256 re-verified on disk",
            "config arms and bootstrap / decision contract match",
            "gate report config SHA256 matches the preregistered config",
        ],
    }


# ---------------------------------------------------------------------------
# Gate-report and arm-record contract
# ---------------------------------------------------------------------------
def verify_gate_report(
    gate_report: Mapping[str, Any],
    *,
    manifest_sha256: str,
    expected_sample_count: int = SAMPLE_COUNT,
) -> dict[str, dict[str, Any]]:
    """Validate the authoritative processor dry run and index its measurements."""

    _require(gate_report.get("design") == DESIGN, "gate report design mismatch")
    _require(
        gate_report.get("mode") == "processor",
        "gate report is not the authoritative processor dry run",
    )
    _require(
        gate_report.get("authoritative") is True,
        "gate report is not marked authoritative",
    )
    _require(
        gate_report.get("model_was_run") is False,
        "gate report claims a model forward pass was run",
    )
    _require(
        gate_report.get("gate_threshold") == GATE_THRESHOLD,
        "gate report gate_threshold mismatch",
    )
    _require(
        str((gate_report.get("manifest") or {}).get("sha256")) == manifest_sha256,
        "gate report preregistered manifest sha256 mismatch",
    )
    final = gate_report.get("final_manifest") or {}
    _require(
        str(final.get("sha256")) == manifest_sha256,
        "gate report final separated manifest sha256 mismatch",
    )
    plan = gate_report.get("plan") or {}
    _require(plan.get("resolved") is True, "gate report plan unresolved")
    _require(
        plan.get("rejected") == []
        and plan.get("replaced") == []
        and plan.get("unresolved") == [],
        "gate report contains rejected / replaced / unresolved samples",
    )
    _require(
        int(plan.get("accepted_count", -1)) == expected_sample_count,
        "gate report accepted_count mismatch",
    )
    accepted = [str(value) for value in plan.get("accepted", [])]
    _require(
        len(set(accepted)) == expected_sample_count,
        "gate report accepted sample_id list is not unique",
    )
    for check in gate_report.get("contract_checks", []):
        _require(
            bool(check.get("ok")) is True,
            f"gate report contract check failed: {check.get('name')!r}",
        )

    measurements: dict[str, dict[str, Any]] = {}
    for entry in gate_report.get("measurements", []):
        sample_id = str(entry.get("sample_id"))
        if sample_id in measurements:
            raise InputContractError(
                f"gate report has duplicate measurement sample_id {sample_id!r} "
                "(fail closed)"
            )
        measurements[sample_id] = entry
    _require(
        set(measurements) == set(accepted),
        "gate report measurements do not match the accepted sample_ids",
    )
    return measurements


def _validate_run_metadata(
    run: Mapping[str, Any],
    *,
    arm: str,
    label: str,
    manifest_sha256: str,
    sample_id: str,
) -> None:
    min_pixels, max_pixels = PIXEL_BUDGETS[label]
    pixel_budget = run.get("pixel_budget") or {}
    _require(
        run.get("arm") == arm,
        f"record {sample_id!r} run.arm mismatch",
    )
    _require(
        run.get("budget") == BUDGET
        and run.get("region_grid") == REGION_GRID
        and run.get("local_candidate_cap") == LOCAL_CANDIDATE_CAP
        and run.get("prune_after_layer") == PRUNE_AFTER_LAYER,
        f"record {sample_id!r} run budget / grid / cap / cutpoint mismatch",
    )
    _require(
        str(run.get("manifest_sha256")) == manifest_sha256,
        f"record {sample_id!r} run.manifest_sha256 mismatch",
    )
    _require(
        run.get("do_sample") is False and run.get("dtype") == RUN_DTYPE_LABEL,
        f"record {sample_id!r} run decoding / dtype mismatch",
    )
    _require(
        run.get("same_image_resolution_counterfactual") is True,
        f"record {sample_id!r} run is not tagged as the counterfactual design",
    )
    _require(
        pixel_budget.get("design") == DESIGN
        and pixel_budget.get("label") == label
        and int(pixel_budget.get("min_pixels", -1)) == min_pixels
        and int(pixel_budget.get("max_pixels", -1)) == max_pixels,
        f"record {sample_id!r} run.pixel_budget mismatch",
    )


def _validate_arm_record(
    row: Mapping[str, Any],
    *,
    arm: str,
    label: str,
    manifest_row: Mapping[str, Any],
    manifest_sha256: str,
    gate_measurement: Mapping[str, Any],
) -> None:
    """Fail closed on any id / hash / metadata / G-threshold contract violation."""

    sample_id = str(row.get("sample_id"))
    missing = [field for field in REQUIRED_RECORD_FIELDS if field not in row]
    if missing:
        raise InputContractError(
            f"record {sample_id!r} is missing required fields {missing} "
            "(partial record, fail closed)"
        )

    _require(row.get("arm") == arm, f"record {sample_id!r} arm mismatch")
    _require(
        row.get("score_mode") == SCORE_MODE
        and row.get("integration_score_mode") == SCORE_MODE,
        f"record {sample_id!r} score_mode mismatch",
    )
    _require(
        row.get("prune_after_layer") == PRUNE_AFTER_LAYER
        and row.get("cutpoint") == CUTPOINT,
        f"record {sample_id!r} B1 cutpoint mismatch",
    )
    _require(
        row.get("pixel_budget_label") == label,
        f"record {sample_id!r} pixel_budget_label mismatch",
    )
    min_pixels, max_pixels = PIXEL_BUDGETS[label]
    _require(
        int(row.get("min_pixels", -1)) == min_pixels
        and int(row.get("max_pixels", -1)) == max_pixels,
        f"record {sample_id!r} min_pixels / max_pixels mismatch",
    )
    _require(
        int(row.get("requested_budget", -1)) == BUDGET
        and int(row.get("region_grid", -1)) == REGION_GRID
        and int(row.get("local_candidate_cap", -1)) == LOCAL_CANDIDATE_CAP,
        f"record {sample_id!r} K / region_grid / local_candidate_cap mismatch",
    )
    _require(
        row.get("do_sample") is False
        and row.get("decoding") == DECODING
        and int(row.get("max_new_tokens", -1)) == MAX_NEW_TOKENS,
        f"record {sample_id!r} greedy decoding metadata mismatch",
    )
    _require(
        str(row.get("manifest_sha256")) == manifest_sha256,
        f"record {sample_id!r} manifest_sha256 mismatch",
    )
    _require(
        str(row.get("task")) == str(manifest_row.get("task")),
        f"record {sample_id!r} task does not match the manifest",
    )
    # The frozen B1 runner writes task but does not duplicate dataset provenance
    # into each raw record. The SHA-pinned manifest is authoritative for source
    # metadata; if a future runner does emit these optional fields, still reject
    # any disagreement instead of silently trusting it.
    for field in ("source", "source_dataset"):
        if field in row:
            _require(
                str(row.get(field)) == str(manifest_row.get(field)),
                f"record {sample_id!r} {field} does not match the manifest",
            )

    if arm == ARM_PROMPT_INTEGER:
        _require(
            str(row.get("checkpoint_sha256")) == CHECKPOINT_SHA256
            and str(row.get("wv_sha256")) == WV_SHA256
            and str(row.get("wv_scale_sha256")) == WV_SCALE_SHA256
            and int(row.get("deployment_seed", -1)) == DEPLOYMENT_SEED,
            f"record {sample_id!r} prompt checkpoint / Wv hashes mismatch",
        )
    else:
        _require(
            all(row.get(key) is None for key in CHECKPOINT_PIN_KEYS)
            and row.get("deployment_seed") is None,
            f"record {sample_id!r} no-prompt arm must not carry checkpoint hashes",
        )

    _validate_run_metadata(
        row["run"],
        arm=arm,
        label=label,
        manifest_sha256=manifest_sha256,
        sample_id=sample_id,
    )

    try:
        grid = [int(value) for value in row["image_grid_thw"]]
        full_patch = int(row["full_patch_tokens"])
        full_llm = int(row["full_llm_visual_tokens"])
        output_count = int(row["output_group_count"])
    except (TypeError, ValueError) as exc:
        raise InputContractError(
            f"record {sample_id!r} has non-integer grid / token counts "
            "(fail closed)"
        ) from exc
    _require(
        len(grid) == 3 and grid[0] == 1 and grid[1] * grid[2] == full_patch,
        f"record {sample_id!r} image_grid_thw is not a single aligned image",
    )
    _require(
        full_patch == MERGE_UNIT * full_llm,
        f"record {sample_id!r} full_patch_tokens != {MERGE_UNIT} * full_llm_visual_tokens",
    )

    # The dual-pixel-budget gate: low G <= 512, high G > 512.
    if label == "low":
        _require(
            full_llm <= GATE_THRESHOLD,
            f"record {sample_id!r} low budget G={full_llm} > {GATE_THRESHOLD} "
            "(gate contamination)",
        )
    else:
        _require(
            full_llm > GATE_THRESHOLD,
            f"record {sample_id!r} high budget G={full_llm} <= {GATE_THRESHOLD} "
            "(gate contamination)",
        )

    # Cross-check against the authoritative processor dry-run measurement.
    measured = int(gate_measurement.get(f"G_{label}", -1))
    _require(
        measured == full_llm,
        f"record {sample_id!r} {label} G={full_llm} != authoritative dry-run "
        f"G_{label}={measured}",
    )

    if bool(row.get("identity_bypass")):
        _require(
            output_count == full_llm and 0 < output_count <= BUDGET,
            f"record {sample_id!r} identity bypass must emit G groups",
        )
    else:
        _require(
            output_count == BUDGET,
            f"record {sample_id!r} non-bypass record must emit K={BUDGET}",
        )

    selected = [int(value) for value in row["selected_group_indices"]]
    _require(
        len(selected) == output_count,
        f"record {sample_id!r} selected_group_indices length mismatch",
    )
    _require(
        all(later > earlier for earlier, later in zip(selected, selected[1:])),
        f"record {sample_id!r} selected_group_indices not strictly ascending",
    )
    actual_hash = array_sha256(np.asarray(selected, dtype=np.int64))
    _require(
        str(row.get("selected_group_indices_sha256")) == actual_hash,
        f"record {sample_id!r} selected_group_indices_sha256 mismatch",
    )
    _require(
        str(row.get("record_sha256")) == record_sha256(row),
        f"record {sample_id!r} record_sha256 mismatch (stale or corrupted)",
    )


def verify_same_image_pairing(
    manifest_ids: Sequence[str],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Enforce the same-image / same-budget pairing contract.

    Within one sample and one pixel budget the integer and prompt arms must
    tokenise the identical image to the identical ``image_grid_thw`` and the
    identical ``full_llm_visual_tokens`` -- only the selector arm may differ.
    Within one arm the low and high pixel budgets must produce different grids.
    Any violation raises :class:`InputContractError` (fail closed).
    """

    cross_arm_grid_mismatches: list[str] = []
    cross_arm_token_mismatches: list[str] = []
    unchanged_budget_grids: list[str] = []
    for sid in manifest_ids:
        integer_low = arm_indices[f"{ARM_INTEGER}::low"][sid]
        integer_high = arm_indices[f"{ARM_INTEGER}::high"][sid]
        prompt_low = arm_indices[f"{ARM_PROMPT_INTEGER}::low"][sid]
        prompt_high = arm_indices[f"{ARM_PROMPT_INTEGER}::high"][sid]

        integer_low_grid = tuple(int(value) for value in integer_low["image_grid_thw"])
        integer_high_grid = tuple(
            int(value) for value in integer_high["image_grid_thw"]
        )
        prompt_low_grid = tuple(int(value) for value in prompt_low["image_grid_thw"])
        prompt_high_grid = tuple(int(value) for value in prompt_high["image_grid_thw"])

        if (
            integer_low_grid != prompt_low_grid
            or integer_high_grid != prompt_high_grid
        ):
            cross_arm_grid_mismatches.append(sid)
        if (
            integer_low_grid == integer_high_grid
            or prompt_low_grid == prompt_high_grid
        ):
            unchanged_budget_grids.append(sid)
        if (
            int(integer_low["full_llm_visual_tokens"])
            != int(prompt_low["full_llm_visual_tokens"])
            or int(integer_high["full_llm_visual_tokens"])
            != int(prompt_high["full_llm_visual_tokens"])
        ):
            cross_arm_token_mismatches.append(sid)

    if cross_arm_grid_mismatches:
        raise InputContractError(
            "integer and prompt arms do not share the identical image_grid_thw "
            f"at the same sample / pixel budget for "
            f"{len(cross_arm_grid_mismatches)} samples (same-image contract "
            "broken: only the selector arm may differ); first grid mismatch: "
            f"{cross_arm_grid_mismatches[0]}"
        )
    if cross_arm_token_mismatches:
        raise InputContractError(
            "integer and prompt arms do not share the identical "
            "full_llm_visual_tokens at the same sample / pixel budget for "
            f"{len(cross_arm_token_mismatches)} samples (same-image contract "
            f"broken); first token mismatch: {cross_arm_token_mismatches[0]}"
        )
    if unchanged_budget_grids:
        raise InputContractError(
            "low and high recordings share an identical image_grid_thw for "
            f"{len(unchanged_budget_grids)} samples (same-image pairing broken: "
            "the pixel-budget counterfactual did not change the grid); first "
            f"unchanged grid: {unchanged_budget_grids[0]}"
        )

    return {
        "cross_arm_grid_identical": True,
        "cross_arm_token_counts_identical": True,
        "low_high_grids_distinct": True,
    }


def validate(
    *,
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    gate_report: Mapping[str, Any],
    expected_sample_count: int = SAMPLE_COUNT,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the four-file frozen input contract; raise on any violation."""

    manifest_ids = list(manifest_index)
    _require(
        len(manifest_ids) == expected_sample_count,
        f"separated manifest must hold {expected_sample_count} unique samples, "
        f"got {len(manifest_ids)}",
    )
    _require(
        manifest_sha256 is not None,
        "a separated-manifest SHA256 is required to validate arm records",
    )
    gate_measurements = verify_gate_report(
        gate_report,
        manifest_sha256=str(manifest_sha256),
        expected_sample_count=expected_sample_count,
    )
    _require(
        set(gate_measurements) == set(manifest_ids),
        "gate report measurements do not equal the separated manifest ids",
    )

    for sid, row in manifest_index.items():
        for field in ("task", "source", "source_dataset", "choices", "answer"):
            if field not in row or str(row.get(field, "")).strip() == "":
                raise InputContractError(
                    f"manifest sample {sid!r} is missing required field {field!r}"
                )
        _require(
            int(row.get("max_new_tokens", -1)) == MAX_NEW_TOKENS,
            f"manifest sample {sid!r} max_new_tokens != {MAX_NEW_TOKENS}",
        )
        meta = row.get("resolution_counterfactual")
        if not isinstance(meta, Mapping):
            raise InputContractError(
                f"manifest sample {sid!r} is missing resolution_counterfactual "
                "metadata"
            )
        _require(
            meta.get("design") == DESIGN
            and int(meta.get("selection_seed", -1)) == BOOTSTRAP_SEED,
            f"manifest sample {sid!r} counterfactual design/seed mismatch",
        )
        stratum = str(meta.get("stratum"))
        _require(
            stratum in STRATA,
            f"manifest sample {sid!r} has unknown stratum {stratum!r}",
        )
        gate_bin = str(meta.get("original_G_bin"))
        _require(
            gate_bin in GATE_BINS and stratum.split("::", 1)[0] == gate_bin,
            f"manifest sample {sid!r} original_G_bin / stratum mismatch",
        )
        _require(
            isinstance(meta.get("original_integer_correct"), bool),
            f"manifest sample {sid!r} original_integer_correct must be a bool",
        )

    manifest_set = set(manifest_ids)
    counts: dict[str, int] = {}
    for key, index in arm_indices.items():
        if len(index) != expected_sample_count:
            raise InputContractError(
                f"arm file {key!r} must hold exactly {expected_sample_count} "
                f"unique samples, got {len(index)}"
            )
        missing = sorted(manifest_set - set(index))
        extra = sorted(set(index) - manifest_set)
        if missing or extra:
            raise InputContractError(
                f"arm file {key!r} sample_id set does not equal the separated "
                f"manifest: missing={missing[:10]} extra={extra[:10]}"
            )
        counts[key] = len(index)

    for arm in ARMS:
        for label in BUDGET_ORDER:
            key = f"{arm}::{label}"
            for sid, row in arm_indices[key].items():
                _validate_arm_record(
                    row,
                    arm=arm,
                    label=label,
                    manifest_row=manifest_index[sid],
                    manifest_sha256=str(manifest_sha256),
                    gate_measurement=gate_measurements[sid],
                )

    # Same-image pairing: every sample id is present in all four files, the
    # integer and prompt arms share the identical grid (and G) at each pixel
    # budget, the low and high grids differ, and G strictly increases with the
    # pixel budget.
    pairing = verify_same_image_pairing(manifest_ids, arm_indices)

    return {
        "manifest_sample_count": len(manifest_ids),
        "arm_file_sample_counts": counts,
        "strata": sorted({str(manifest_index[sid]["resolution_counterfactual"]["stratum"]) for sid in manifest_ids}),
        "exactly_100_unique_ids_per_file": True,
        "four_file_id_sets_identical": True,
        "gate_report_authoritative_processor": True,
        "gate_report_resolved_100": True,
        "low_G_le_512": True,
        "high_G_gt_512": True,
        "G_matches_authoritative_measurements": True,
        "same_image_paired": True,
        "same_image_cross_arm_grid_identical": pairing["cross_arm_grid_identical"],
        "same_image_cross_arm_token_counts_identical": pairing[
            "cross_arm_token_counts_identical"
        ],
        "low_high_grids_distinct": pairing["low_high_grids_distinct"],
        "no_mixed_arm": True,
        "no_duplicate_or_extra_records": True,
        "record_hashes_verified": True,
        "manifest_sha256_verified": True,
        "checkpoint_and_wv_hashes_verified": True,
        "checks": [
            "exactly 100 unique sample_ids in the separated manifest and each of "
            "the four arm files; all five id sets identical",
            "authoritative processor dry run: resolved, 0 rejected / replaced / "
            "unresolved, 100 accepted",
            "every low record G <= 512 and every high record G > 512",
            "every record G equals the authoritative per-sample dry-run "
            "measurement",
            "every record arm / pixel_budget_label / min_pixels / max_pixels "
            "matches its file",
            "every record K=96 / region_grid=3 / local_candidate_cap=16 / "
            "B1 cutpoint / greedy / max_new_tokens=16",
            "prompt records carry the frozen checkpoint / Wv / Wv scale hashes; "
            "no-prompt records carry none",
            "every record record_sha256 re-verifies; no duplicate ids",
            "same-image low/high pairing with distinct input grids",
            "integer and prompt arms share the identical image_grid_thw and "
            "full_llm_visual_tokens at every sample / pixel budget",
        ],
    }


# ---------------------------------------------------------------------------
# Scoring (primary estimand only)
# ---------------------------------------------------------------------------
def build_joined(
    manifest_ids: Sequence[str],
    manifest_index: Mapping[str, Mapping[str, Any]],
    arm_indices: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Join the four arm files by ``sample_id`` and recompute correctness."""

    joined: list[dict[str, Any]] = []
    for sample_id in manifest_ids:
        meta = manifest_index[sample_id]
        choices = list(meta["choices"])
        truth = answer_letter(str(meta["answer"]))
        counterfactual = meta["resolution_counterfactual"]
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "task": meta["task"],
            "source": meta["source"],
            "source_dataset": meta["source_dataset"],
            "stratum": str(counterfactual["stratum"]),
            "original_G_bin": str(counterfactual["original_G_bin"]),
            "original_integer_correct": bool(
                counterfactual["original_integer_correct"]
            ),
            "image": str(meta.get("image", "")),
        }
        for arm in ARMS:
            for label in BUDGET_ORDER:
                row = arm_indices[f"{arm}::{label}"][sample_id]
                parsed = parse_choice(str(row["output_text"]), choices)
                record[f"{label}_{arm}_answer"] = parsed
                record[f"{label}_{arm}_correct"] = int(parsed == truth)
                record[f"{label}_G"] = int(row["full_llm_visual_tokens"])
        integer_low = record[f"low_{ARM_INTEGER}_correct"]
        integer_high = record[f"high_{ARM_INTEGER}_correct"]
        prompt_low = record[f"low_{ARM_PROMPT_INTEGER}_correct"]
        prompt_high = record[f"high_{ARM_PROMPT_INTEGER}_correct"]
        record["effect"] = int(
            (prompt_high - integer_high) - (prompt_low - integer_low)
        )
        joined.append(record)
    return joined


def budget_metrics(
    joined: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    """Per-budget correctness, parse rate and Prompt-vs-Integer transitions."""

    def _answers(arm: str) -> list[Any]:
        return [row[f"{label}_{arm}_answer"] for row in joined]

    def _correct(arm: str) -> list[bool]:
        return [bool(row[f"{label}_{arm}_correct"]) for row in joined]

    integer_correct = _correct(ARM_INTEGER)
    prompt_correct = _correct(ARM_PROMPT_INTEGER)
    n = len(joined)
    wrong_to_right = sum(
        1
        for integer, prompt in zip(integer_correct, prompt_correct)
        if not integer and prompt
    )
    right_to_wrong = sum(
        1
        for integer, prompt in zip(integer_correct, prompt_correct)
        if integer and not prompt
    )
    g_values = [int(row[f"{label}_G"]) for row in joined]
    return {
        "label": label,
        "sample_count": n,
        "min_pixels": PIXEL_BUDGETS[label][0],
        "max_pixels": PIXEL_BUDGETS[label][1],
        "G_min": min(g_values),
        "G_max": max(g_values),
        "G_all_within_gate": (
            all(g <= GATE_THRESHOLD for g in g_values)
            if label == "low"
            else all(g > GATE_THRESHOLD for g in g_values)
        ),
        "integer_correct_count": sum(integer_correct),
        "prompt_correct_count": sum(prompt_correct),
        "integer_unweighted_accuracy": sum(integer_correct) / n,
        "prompt_unweighted_accuracy": sum(prompt_correct) / n,
        "integer_parse_rate": sum(
            answer is not None for answer in _answers(ARM_INTEGER)
        )
        / n,
        "prompt_parse_rate": sum(
            answer is not None for answer in _answers(ARM_PROMPT_INTEGER)
        )
        / n,
        "wrong_to_right": wrong_to_right,
        "right_to_wrong": right_to_wrong,
        "net_correction": wrong_to_right - right_to_wrong,
        "both_correct": sum(
            1
            for integer, prompt in zip(integer_correct, prompt_correct)
            if integer and prompt
        ),
        "both_wrong": sum(
            1
            for integer, prompt in zip(integer_correct, prompt_correct)
            if not integer and not prompt
        ),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    strata: Sequence[str],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
    ci_level: float = CI_LEVEL,
) -> tuple[float, float]:
    """Deterministic stratified paired bootstrap of the per-sample mean.

    Samples are resampled with replacement *within each original stratum*; the
    returned interval uses the frozen percentile convention
    ``deltas[int(p * (repeats - 1))]``.
    """

    if len(values) != len(strata):
        raise InputContractError("values and strata must have equal length")
    if repeats <= 0:
        raise InputContractError("bootstrap repeats must be positive")
    groups: dict[str, list[float]] = defaultdict(list)
    for value, stratum in zip(values, strata):
        groups[str(stratum)].append(float(value))
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(repeats):
        total = 0.0
        count = 0
        for key in sorted(groups):
            group = groups[key]
            size = len(group)
            for _ in range(size):
                total += group[rng.randrange(size)]
            count += size
        deltas.append(total / count)
    deltas.sort()
    lower_p = (1.0 - ci_level) / 2.0
    upper_p = 1.0 - lower_p
    lower_index = int(lower_p * (repeats - 1))
    upper_index = int(upper_p * (repeats - 1))
    return deltas[lower_index], deltas[upper_index]


def primary_comparison(
    joined: Sequence[Mapping[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    repeats: int = BOOTSTRAP_REPEATS,
    ci_level: float = CI_LEVEL,
) -> dict[str, Any]:
    """The preregistered primary estimand with its stratified bootstrap CI."""

    effects = [float(row["effect"]) for row in joined]
    strata = [str(row["stratum"]) for row in joined]
    point = sum(effects) / len(effects)
    lower, upper = bootstrap_mean_ci(
        effects, strata, seed=seed, repeats=repeats, ci_level=ci_level
    )
    per_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted(set(strata)):
        values = [
            effect
            for effect, current in zip(effects, strata)
            if current == stratum
        ]
        per_stratum[stratum] = {
            "sample_count": len(values),
            "mean_effect": sum(values) / len(values),
        }
    return {
        "estimand": PRIMARY_ESTIMAND_EXPANDED,
        "preregistered_estimand_literal": PRIMARY_ESTIMAND,
        "metric": PRIMARY_METRIC,
        "point_estimate": point,
        "ci95": [lower, upper],
        "ci_lower": lower,
        "stratified_by": STRATIFIED_BY,
        "seed": seed,
        "repeats": repeats,
        "ci_level": ci_level,
        "per_stratum": per_stratum,
        "note": (
            "the official score(...)[\"cv_bench_accuracy\"] is undefined inside "
            "the single ADE20K source and is never computed"
        ),
    }


def decide(ci_lower: float) -> dict[str, Any]:
    """Apply the preregistered decision rule to the CI lower bound only."""

    confirmed = float(ci_lower) > DECISION_THRESHOLD
    outcome = OUTCOME_CONFIRMED if confirmed else OUTCOME_NOT_CONFIRMED
    reason = (
        f"95% stratified paired bootstrap CI lower bound {float(ci_lower):+.6f} "
        f"{'>' if confirmed else '<='} {DECISION_THRESHOLD} "
        f"({PRIMARY_ESTIMAND_EXPANDED})"
    )
    return {
        "outcome": outcome,
        "reason": reason,
        "confirmed": confirmed,
        "ci_lower": float(ci_lower),
        "threshold": DECISION_THRESHOLD,
        "estimand": PRIMARY_ESTIMAND_EXPANDED,
        "preregistered_estimand_literal": PRIMARY_ESTIMAND,
        "uses_only_primary_estimand": True,
        "forbidden_metrics_never_used": list(FORBIDDEN_METRICS),
        "formal_full2638_conclusion": FORMAL_CONCLUSION_UNCHANGED,
        "formal_full2638_conclusion_unchanged": True,
    }


def analyze(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    gate_report: Mapping[str, Any],
    expected_sample_count: int = SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_repeats: int | None = None,
    ci_level: float | None = None,
) -> dict[str, Any]:
    """Validate inputs and compute the counterfactual result (no file I/O)."""

    manifest_index = index_unique(manifest_rows, "separated_manifest")
    arm_indices = {
        key: index_unique(rows, key) for key, rows in arm_rows.items()
    }
    missing_keys = set(arm_paths(DEFAULT_PATHS)) - set(arm_indices)
    if missing_keys:
        raise InputContractError(
            f"missing arm files for {sorted(missing_keys)} (fail closed)"
        )

    validation = validate(
        manifest_index=manifest_index,
        arm_indices=arm_indices,
        gate_report=gate_report,
        expected_sample_count=expected_sample_count,
        manifest_sha256=manifest_sha256,
    )

    joined = build_joined(list(manifest_index), manifest_index, arm_indices)

    metrics = {label: budget_metrics(joined, label) for label in BUDGET_ORDER}
    comparison = primary_comparison(
        joined,
        seed=BOOTSTRAP_SEED if bootstrap_seed is None else bootstrap_seed,
        repeats=(
            BOOTSTRAP_REPEATS if bootstrap_repeats is None else bootstrap_repeats
        ),
        ci_level=CI_LEVEL if ci_level is None else ci_level,
    )
    decision = decide(comparison["ci_lower"])

    return {
        "validation": validation,
        "joined": joined,
        "budget_metrics": metrics,
        "primary_comparison": comparison,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _inputs_sha256(paths: InputPaths) -> dict[str, Any]:
    logical = {
        "preregistered_config": paths.preregistered,
        "separated_manifest": paths.separated_manifest,
        "gate_separation_report": paths.gate_separation_report,
        "checkpoint": paths.checkpoint,
        "wv": paths.wv,
        "wv_scale": paths.wv_scale,
        **arm_paths(paths),
    }
    return {
        str(name): {"path": str(path), "sha256": sha256_file(path)}
        for name, path in logical.items()
    }


def build_report(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    expected_sample_count: int = SAMPLE_COUNT,
    verify_pins: bool = True,
    manifest_sha256: str | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_repeats: int | None = None,
    ci_level: float | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load every required input, validate, score and assemble the report.

    Any missing file, SHA pin mismatch or failed contract check raises
    :class:`InputContractError` before a report is produced (fail closed).
    """

    require_file(paths.preregistered, "preregistered_config")
    config = load_json_object(paths.preregistered, "preregistered_config")
    gate_report = load_json_object(
        paths.gate_separation_report, "gate_separation_report"
    )
    if manifest_sha256 is None:
        require_file(paths.separated_manifest, "separated_manifest")
        manifest_sha256 = sha256_file(paths.separated_manifest)

    config_pins = (
        verify_config(
            config,
            gate_report,
            paths,
            expected_sample_count=expected_sample_count,
            manifest_sha256=manifest_sha256,
            root=root,
        )
        if verify_pins
        else None
    )

    manifest_rows = load_rows(paths.separated_manifest, "separated_manifest")
    arm_rows = {
        key: load_rows(path, key) for key, path in arm_paths(paths).items()
    }

    analysis = analyze(
        manifest_rows=manifest_rows,
        arm_rows=arm_rows,
        gate_report=gate_report,
        expected_sample_count=expected_sample_count,
        manifest_sha256=manifest_sha256,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=bootstrap_repeats,
        ci_level=ci_level,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "design": DESIGN,
        "report_scope": (
            "preregistered same-image dual-pixel-budget counterfactual on frozen "
            "ADE20K samples"
        ),
        "formal_counterfactual_conclusion": True,
        "formal_full2638_conclusion": FORMAL_CONCLUSION_UNCHANGED,
        "formal_full2638_conclusion_unchanged": True,
        "fixed_configuration": {
            "manifest": str(paths.separated_manifest),
            "sample_count": expected_sample_count,
            "source_dataset": "ADE20K Validation Set",
            "gate_threshold": GATE_THRESHOLD,
            "budget_K": BUDGET,
            "boundary": CUTPOINT,
            "prune_after_layer": PRUNE_AFTER_LAYER,
            "region_grid": REGION_GRID,
            "local_candidate_cap": LOCAL_CANDIDATE_CAP,
            "dtype": DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
            "decoding": DECODING,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "score_mode": SCORE_MODE,
            "low_pixel_budget": {
                "min_pixels": PIXEL_BUDGETS["low"][0],
                "max_pixels": PIXEL_BUDGETS["low"][1],
            },
            "high_pixel_budget": {
                "min_pixels": PIXEL_BUDGETS["high"][0],
                "max_pixels": PIXEL_BUDGETS["high"][1],
            },
            "bootstrap_seed": BOOTSTRAP_SEED if bootstrap_seed is None else bootstrap_seed,
            "bootstrap_repeats": (
                BOOTSTRAP_REPEATS if bootstrap_repeats is None else bootstrap_repeats
            ),
            "bootstrap_ci_level": CI_LEVEL if ci_level is None else ci_level,
        },
        "primary_metric": {
            "identifier": PRIMARY_ESTIMAND_EXPANDED,
            "preregistered_estimand_literal": PRIMARY_ESTIMAND,
            "metric": PRIMARY_METRIC,
            "rule": (
                "the only decision input; the official cv_bench_accuracy is "
                "undefined inside the single ADE20K source"
            ),
        },
        "forbidden_primary_metrics": list(FORBIDDEN_METRICS),
        "arms": {
            ARM_INTEGER: {
                "score_mode": ARM_SCORE_MODE[ARM_INTEGER],
                "uses_b1_trained_wv": ARM_USES_B1_TRAINED_WV[ARM_INTEGER],
            },
            ARM_PROMPT_INTEGER: {
                "score_mode": ARM_SCORE_MODE[ARM_PROMPT_INTEGER],
                "uses_b1_trained_wv": ARM_USES_B1_TRAINED_WV[ARM_PROMPT_INTEGER],
            },
        },
        "arm_files": {
            key: str(path) for key, path in arm_paths(paths).items()
        },
        "validation": analysis["validation"],
        "config_pin_verification": config_pins,
        "budget_metrics": analysis["budget_metrics"],
        "primary_comparison": analysis["primary_comparison"],
        "decision": analysis["decision"],
        "per_sample": analysis["joined"],
        "inputs_sha256": _inputs_sha256(paths),
        "limitations": [
            "Software Selector bridge only; no FPGA, latency or acceleration "
            "claim is made.",
            "The official score(...)[\"cv_bench_accuracy\"] is undefined inside "
            "the single ADE20K source; only the preregistered unweighted paired "
            "difference is computed.",
            "Pointing / grounding / nDCG / retained-mass are never reported.",
            "Either outcome leaves PROMPT_GAIN_NOT_CONFIRMED_FULL2638 unchanged.",
            "The four GPU arm files were produced with a single A6000 BF16/SDPA "
            "forward; the counterfactual only varies the processor pixel budget "
            "of the same frozen image.",
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
    lines.append("# 同图双分辨率反事实（ADE20K100）汇总")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- 范围：**{report['report_scope']}**")
    lines.append(
        "- 正式结论（full2638，不因本实验改变）：**"
        + report["formal_full2638_conclusion"]
        + "**"
    )
    lines.append("")
    lines.append(
        "- 唯一主估计量：`" + report["primary_metric"]["identifier"] + "`；"
        "官方 `score(...)[\"cv_bench_accuracy\"]` 在单一 ADE20K 数据源内未定义，"
        "本报告**不计算**。"
    )
    lines.append("- 禁用主指标：" + ", ".join(report["forbidden_primary_metrics"]))
    lines.append("")
    lines.append("## 1. 固定配置")
    lines.append("")
    cfg = report["fixed_configuration"]
    lines.append(
        f"- 样本：`{cfg['manifest']}`（{cfg['sample_count']}）；"
        f"门控阈值 G={cfg['gate_threshold']}；K={cfg['budget_K']}；"
        f"切点：{cfg['boundary']}；region_grid={cfg['region_grid']}；"
        f"local_cap={cfg['local_candidate_cap']}"
    )
    lines.append(
        f"- low pixel budget：{cfg['low_pixel_budget']}；high pixel budget："
        f"{cfg['high_pixel_budget']}"
    )
    lines.append(
        f"- dtype={cfg['dtype']}；attention={cfg['attention_implementation']}；"
        f"decoding={cfg['decoding']}；max_new_tokens={cfg['max_new_tokens']}；"
        f"bootstrap seed={cfg['bootstrap_seed']} / repeats={cfg['bootstrap_repeats']}"
    )
    lines.append("")
    lines.append("## 2. 输入合同校验")
    lines.append("")
    v = report["validation"]
    lines.append(f"- 四文件样本数：{v['arm_file_sample_counts']}")
    lines.append(f"- 层（stratum）：{', '.join(v['strata'])}")
    checks = [
        (
            f"每文件恰好 {v['manifest_sample_count']} 个唯一 sample_id",
            v["exactly_100_unique_ids_per_file"],
        ),
        ("四文件 ID 集合一致且等于 separated manifest", v["four_file_id_sets_identical"]),
        ("权威 processor dry run 且 100/100 resolved", v["gate_report_resolved_100"]),
        ("low 每样本 G <= 512", v["low_G_le_512"]),
        ("high 每样本 G > 512", v["high_G_gt_512"]),
        ("G 与权威 dry-run 测量一致", v["G_matches_authoritative_measurements"]),
        ("同图 low/high 成对", v["same_image_paired"]),
        (
            "同预算 integer/prompt 两臂 grid 完全一致",
            v["same_image_cross_arm_grid_identical"],
        ),
        (
            "同预算 integer/prompt 两臂 G 完全一致",
            v["same_image_cross_arm_token_counts_identical"],
        ),
        ("low/high grid 不同", v["low_high_grids_distinct"]),
        ("无混臂 / 无重复或多余记录（断点续跑污染）", v["no_mixed_arm"]),
        ("record_sha256 复核通过", v["record_hashes_verified"]),
        ("checkpoint / Wv 哈希一致", v["checkpoint_and_wv_hashes_verified"]),
    ]
    for name, ok in checks:
        lines.append(f"- {name}：{'✅' if ok else '❌'}")
    lines.append("")
    lines.append("## 3. 各档指标（W→R / R→W / net、parse rate）")
    lines.append("")
    lines.append(
        "| 档 | G 范围 | Integer 准确率 | Prompt 准确率 | Integer parse | "
        "Prompt parse | W→R | R→W | net |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for label in BUDGET_ORDER:
        m = report["budget_metrics"][label]
        lines.append(
            f"| {label} | {m['G_min']}..{m['G_max']} | "
            f"{_pct(m['integer_unweighted_accuracy'])} | "
            f"{_pct(m['prompt_unweighted_accuracy'])} | "
            f"{_pct(m['integer_parse_rate'])} | {_pct(m['prompt_parse_rate'])} | "
            f"{m['wrong_to_right']} | {m['right_to_wrong']} | "
            f"{m['net_correction']:+d} |"
        )
    lines.append("")
    lines.append("## 4. 主估计量配对分层 bootstrap")
    lines.append("")
    c = report["primary_comparison"]
    lines.append(f"- 估计量：`{c['estimand']}`（{c['metric']}）")
    lines.append(
        f"- 点估计：**{c['point_estimate']:+.6f}**；95% CI："
        f"**{_fmt_ci(c['ci95'])}**；分层：{c['stratified_by']}；"
        f"seed={c['seed']}；repeats={c['repeats']}"
    )
    lines.append("")
    lines.append("| 层 | 样本 | 层内均值 |")
    lines.append("|---|---:|---:|")
    for stratum, item in c["per_stratum"].items():
        lines.append(
            f"| {stratum} | {item['sample_count']} | {item['mean_effect']:+.6f} |"
        )
    lines.append("")
    lines.append("## 5. 预注册决策")
    lines.append("")
    d = report["decision"]
    lines.append(f"- 结论：**{d['outcome']}**")
    lines.append(f"- 依据：{d['reason']}")
    lines.append(
        f"- CI 下界 {d['ci_lower']:+.6f} {'>' if d['confirmed'] else '<='} "
        f"{d['threshold']} → {'CONFIRMED' if d['confirmed'] else 'NOT_CONFIRMED'}"
    )
    lines.append(
        "- 无论确认与否，正式 full2638 结论 "
        f"`{d['formal_full2638_conclusion']}` **不变**。"
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
        "> CPU-only 只读汇总：未改原始 JSONL、预注册 config 或既有 gate 报告；"
        "**不作时延或 FPGA 加速声明**。"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output / CLI
# ---------------------------------------------------------------------------
def _csv_rows(per_sample: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in per_sample:
        rows.append(
            {
                "sample_id": record["sample_id"],
                "task": record["task"],
                "source": record["source"],
                "source_dataset": record["source_dataset"],
                "stratum": record["stratum"],
                "original_G_bin": record["original_G_bin"],
                "original_integer_correct": record["original_integer_correct"],
                "low_integer_answer": record[f"low_{ARM_INTEGER}_answer"],
                "low_integer_correct": record[f"low_{ARM_INTEGER}_correct"],
                "low_prompt_answer": record[f"low_{ARM_PROMPT_INTEGER}_answer"],
                "low_prompt_correct": record[f"low_{ARM_PROMPT_INTEGER}_correct"],
                "high_integer_answer": record[f"high_{ARM_INTEGER}_answer"],
                "high_integer_correct": record[f"high_{ARM_INTEGER}_correct"],
                "high_prompt_answer": record[
                    f"high_{ARM_PROMPT_INTEGER}_answer"
                ],
                "high_prompt_correct": record[
                    f"high_{ARM_PROMPT_INTEGER}_correct"
                ],
                "effect": record["effect"],
            }
        )
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
        for row in _csv_rows(report["per_sample"]):
            writer.writerow(row)
    output_md.write_text(render_markdown(report), encoding="utf-8")


def run(
    paths: InputPaths = DEFAULT_PATHS,
    *,
    output_json: Path = REPORT_JSON,
    output_csv: Path = REPORT_CSV,
    output_md: Path = REPORT_MD,
    verify_pins: bool = True,
    expected_sample_count: int = SAMPLE_COUNT,
    manifest_sha256: str | None = None,
    bootstrap_seed: int | None = None,
    bootstrap_repeats: int | None = None,
    ci_level: float | None = None,
    root: Path = ROOT,
) -> tuple[int, dict[str, Any] | None]:
    """Build the report and write it only on success (fail closed)."""

    try:
        report = build_report(
            paths,
            expected_sample_count=expected_sample_count,
            verify_pins=verify_pins,
            manifest_sha256=manifest_sha256,
            bootstrap_seed=bootstrap_seed,
            bootstrap_repeats=bootstrap_repeats,
            ci_level=ci_level,
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
