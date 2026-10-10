#!/usr/bin/env python3
"""Operational prompt-utility diagnostic for the prompt-contrastive selector.

This is a **zero-training, preregistered, validation-only, CPU-only, read-only
exploratory diagnostic**.  It answers one question the prior contrastive
train/val experiment could not: even though the contrastive student failed the
Spearman prompt-discrimination gate, does its prompt-conditioned output carry a
usable *deployment* utility at the real fixed-budget selector output?

It never touches the internal test split, never trains, never loads the 4B model,
never reads a raw image and never re-extracts a feature.  It reuses the frozen
``A_SPATIAL_KL_CONTROL`` / ``B_SPATIAL_KL_CONTRASTIVE`` three-seed best
checkpoints and the same frozen 76 val records / 38 images.

For every record, arm, seed and budget ``K in {48, 64, 96}`` it builds:

* the **correct-prompt** selector -- the record's own prompt;
* the **wrong-prompt** selector -- the same image's other-task prompt.

Both go through the identical deployment chain: the frozen ``prompt_mean_vector``
scorer, ``combine_scores`` (min-max, ``lambda_semantic=0.5``), the existing 3x3
anchor ``select_real_groups`` rule and the corrected selected-set NDCG.  It
reports retained spatial-target mass, selected-set NDCG and correct-vs-wrong
selected Jaccard/overlap.

Aggregation is frozen: first average the grounding/pointing record values of an
image (target definition), then take the per-image median across the three
seeds, then resample over the 38 independent images (2000 bootstrap, 10000
sign-flip).  The primary Holm family has exactly two K48 retained-mass tests:

* ``P1`` -- B correct selector minus B wrong selector;
* ``P2`` -- (B correct-minus-wrong gap) minus (A correct-minus-wrong gap).

Both must satisfy mean>0, CI lower>0, ``|dz|>=0.2``, Holm p<0.05 and >=2/3
positive seed means.  Auxiliary gates S1-S4 are outside the Holm family.

Reuse only (no second implementation of any split / loader / target / scorer /
selector / metric / statistic):

* split / loader:   :func:`evaluation.prompt_projection_distillation.split_sample_ids`
                    and :func:`evaluation.train_pairwise_prompt_selector.load_split_samples`
* spatial targets:  :func:`evaluation.train_spatial_prompt_selector.attach_spatial_targets`
* scorer:           :func:`evaluation.frozen_scorer_reduction_ablation.score_candidate`
                    (``prompt_mean_vector``, exactly what ``val_statistics`` uses)
* combine / anchor: :func:`evaluation.hardware_regular_token_selector.combine_scores`
                    and :func:`evaluation.train_spatial_prompt_selector._select_groups`
* metrics:          :func:`evaluation.prompt_projection_distillation.{saliency_mass_retained,ndcg_for_selected_indices,jaccard}`
* statistics:       :func:`evaluation.prompt_identifiability_diagnostic.paired_effect`
                    and :func:`evaluation.frozen_scorer_reduction_ablation.holm_bonferroni`
* checkpoint:       :func:`evaluation.run_hf_hardware_regular_b1.load_projection`

The prior contrastive report JSON is parsed **only** for the S4 gate booleans
(G1/G4/G5/G7) and the pinned input SHAs; its validation-statistics metrics and its
val-samples CSV are never parsed and are never used as a metric source here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from evaluation.frozen_scorer_reduction_ablation import (
    holm_bonferroni,
    score_candidate,
)
from evaluation.hardware_regular_token_selector import combine_scores
from evaluation.prompt_identifiability_diagnostic import (
    paired_effect,
    same_image_partner_map,
)
from evaluation.prompt_projection_distillation import (
    DistillationContractError,
    jaccard,
    l2_baseline_scores,
    ndcg_for_selected_indices,
    saliency_mass_retained,
    split_sample_ids,
)
from evaluation.run_hf_hardware_regular_b1 import load_projection
from evaluation.spatial_teacher_feasibility_audit import load_clean_manifest
from evaluation.teacher_pair_signal_audit import load_index_metadata
from evaluation.train_pairwise_prompt_selector import load_split_samples, sha256_file
from evaluation.train_spatial_prompt_selector import (
    _select_groups as select_anchor_groups,
    attach_spatial_targets,
)

REPORT_STEM = "2026-10-03-operational-prompt-utility-diagnostic"
DATE = "2026-10-03"
SCHEMA_VERSION = "1.0"

ROOT = Path(__file__).resolve().parents[1]

PREREG_CONFIG_PATH = Path("configs/operational_prompt_utility_diagnostic.preregistered.json")
PREREG_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-operational-prompt-utility-diagnostic-preregistration.json"
)
CONTRASTIVE_CONFIG_PATH = Path(
    "configs/spatial_prompt_contrastive_trainval.preregistered.json"
)
INDEX_PATH = Path("artifacts/token_selector/prompt_projection_features/index.jsonl")
MANIFEST_PATH = Path("datasets/prompt_projection_clean_inductive_v1.jsonl")
ARTIFACTS_DIR = Path("artifacts/token_selector/spatial_prompt_contrastive_trainval")
PRIOR_REPORT_PATH = Path(
    "reports/token_selector/2026-10-03-spatial-prompt-contrastive-trainval.json"
)
REPORTS_DIR = Path("reports/token_selector")
LOGS_DIR = REPORTS_DIR / "logs"

CONTROL_ARM = "A_SPATIAL_KL_CONTROL"
PRIMARY_ARM = "B_SPATIAL_KL_CONTRASTIVE"
ARMS: tuple[str, ...] = (CONTROL_ARM, PRIMARY_ARM)
SEEDS: tuple[int, ...] = (20261003, 20261004, 20261005)
BUDGETS: tuple[int, ...] = (48, 64, 96)
TASKS: tuple[str, ...] = ("grounding", "pointing")
LAMBDA_SEMANTIC = 0.5
STUDENT_SCORER = "prompt_mean_vector"

BOOTSTRAP_RESAMPLES = 2000
PERMUTATION_RESAMPLES = 10000
STATISTICS_SEED = 20261003
ALPHA = 0.05
EFFECT_SIZE_FLOOR = 0.2
POSITIVE_SEED_MIN = 2
JACCARD_GATE_THRESHOLD = 0.95
EQUALITY_TOLERANCE = 1e-12

P1_KEY = "P1_B_correct_minus_wrong_selector_mass_k48"
P2_KEY = "P2_B_minus_A_correct_minus_wrong_gap_mass_k48"

PRIOR_GATE_KEYS: dict[str, str] = {
    "G1": "G1_correct_alignment_non_inferiority",
    "G4": "G4_deployed_K48_utility",
    "G5": "G5_collapse_removed",
    "G7": "G7_hardware_contract",
}

METRICS: tuple[str, ...] = (
    "correct_mass",
    "wrong_mass",
    "gap_mass",
    "correct_ndcg",
    "wrong_ndcg",
    "gap_ndcg",
    "jaccard",
    "overlap_count",
)

CSV_FIELDS: tuple[str, ...] = (
    "arm",
    "seed",
    "sample_id",
    "image_sha256",
    "task",
    "group_count",
    "grid_h",
    "grid_w",
    "budget",
    "lambda_semantic",
    "correct_retained_spatial_target_mass",
    "wrong_retained_spatial_target_mass",
    "gap_retained_spatial_target_mass",
    "correct_selected_set_ndcg",
    "wrong_selected_set_ndcg",
    "gap_selected_set_ndcg",
    "correct_wrong_selected_jaccard",
    "correct_wrong_selected_overlap_count",
    "correct_selected_count",
    "wrong_selected_count",
)


# ---------------------------------------------------------------------------
# Preregistration identity and pins
# ---------------------------------------------------------------------------
def load_prereg(path: Path = PREREG_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_prereg_report(path: Path = PREREG_REPORT_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_prereg_identity(
    prereg: Mapping[str, Any],
    prereg_report: Mapping[str, Any],
    *,
    root: Path = ROOT,
    config_path: Path = PREREG_CONFIG_PATH,
    report_path: Path = PREREG_REPORT_PATH,
) -> dict[str, Any]:
    """Check the config/report pair and the authority prompt SHA256."""
    cfg_path = root / config_path
    rep_path = root / report_path
    cfg_actual = sha256_file(cfg_path)
    rep_actual = sha256_file(rep_path)
    expected_cfg = str(prereg_report["config"]["sha256"])
    authority_rel = str(prereg_report["authority_prompt"]["path"])
    authority_expected = str(prereg_report["authority_prompt"]["sha256"])
    config_authority = str(prereg["authority_prompt"]["sha256"])
    authority_actual = sha256_file(root / authority_rel)
    consistent = bool(
        cfg_actual == expected_cfg
        and authority_actual == authority_expected
        and config_authority == authority_expected
    )
    return {
        "config_path": str(config_path),
        "config_sha256": cfg_actual,
        "config_sha256_matches_report": cfg_actual == expected_cfg,
        "report_path": str(report_path),
        "report_sha256": rep_actual,
        "authority_prompt_path": authority_rel,
        "authority_prompt_sha256": authority_actual,
        "authority_prompt_sha256_matches_report": authority_actual == authority_expected,
        "authority_prompt_sha256_matches_config": config_authority == authority_expected,
        "consistent": consistent,
    }


def verify_prereg_pins(
    prereg_report: Mapping[str, Any],
    *,
    root: Path = ROOT,
    artifacts_dir: Path = ARTIFACTS_DIR,
    strict: bool = True,
) -> dict[str, Any]:
    """Recompute every pinned input and best-checkpoint SHA256."""
    results: dict[str, Any] = {}
    mismatches: list[str] = []
    for rel, expected in prereg_report.get("frozen_inputs", {}).items():
        path = root / str(rel)
        actual = sha256_file(path) if path.is_file() else None
        ok = actual == str(expected)
        results[str(rel)] = {"expected": str(expected), "actual": actual, "unchanged": bool(ok)}
        if not ok:
            mismatches.append(str(rel))
    for key, expected in prereg_report.get("checkpoints", {}).items():
        arm, _, seed = str(key).partition("/")
        rel = Path(artifacts_dir) / arm / str(seed) / "best.npz"
        path = root / rel
        actual = sha256_file(path) if path.is_file() else None
        ok = actual == str(expected)
        results[str(rel)] = {"expected": str(expected), "actual": actual, "unchanged": bool(ok)}
        if not ok:
            mismatches.append(str(rel))
    report = {
        "checked": len(results),
        "mismatches": mismatches,
        "pins_unchanged": not mismatches,
        "per_input": results,
    }
    if mismatches and strict:
        raise DistillationContractError(f"pinned inputs changed: {mismatches}")
    return report


def prior_gate_reference(
    *,
    root: Path = ROOT,
    prior_report_path: Path = PRIOR_REPORT_PATH,
) -> dict[str, Any]:
    """Read the prior contrastive report's G1/G4/G5/G7 gate booleans only."""
    path = root / prior_report_path
    if not path.is_file():
        return {
            "available": False,
            "prior_report": str(prior_report_path),
            "gates": {},
            "all_pass": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    gate = payload.get("primary_gate", {})
    gates: dict[str, Any] = {}
    for short, key in PRIOR_GATE_KEYS.items():
        entry = gate.get(key)
        gates[short] = {
            "key": key,
            "present": entry is not None,
            "pass": bool(entry.get("pass", False)) if isinstance(entry, Mapping) else False,
        }
    return {
        "available": True,
        "prior_report": str(prior_report_path),
        "prior_report_sha256": sha256_file(path),
        "prior_decision": (payload.get("decision") or {}).get("decision"),
        "referenced_gate_keys": dict(PRIOR_GATE_KEYS),
        "gates": gates,
        "all_pass": all(entry["pass"] for entry in gates.values()),
        "note": (
            "The prior contrastive report JSON is parsed only for these gate "
            "booleans (S4); its validation-statistics metrics and val-samples CSV "
            "are never parsed and never used as a metric source."
        ),
    }


# ---------------------------------------------------------------------------
# Per-record deployment metrics
# ---------------------------------------------------------------------------
def record_utility_metrics(
    *,
    correct_scores: Any,
    wrong_scores: Any,
    l2_scores: Any,
    target: Any,
    grid_h: int,
    grid_w: int,
    budgets: Sequence[int] = BUDGETS,
    lambda_semantic: float = LAMBDA_SEMANTIC,
) -> dict[int, dict[str, float | int]]:
    """Correct/wrong deployed selector metrics for one record at each budget."""
    relevance = np.asarray(target, dtype=np.float64).reshape(-1)
    group_count = int(relevance.size)
    correct = np.asarray(correct_scores, dtype=np.float64)
    wrong = np.asarray(wrong_scores, dtype=np.float64)
    l2 = np.asarray(l2_scores, dtype=np.float64)
    correct_combined = combine_scores(correct, l2, float(lambda_semantic))
    wrong_combined = combine_scores(wrong, l2, float(lambda_semantic))
    result: dict[int, dict[str, float | int]] = {}
    for budget in budgets:
        effective = min(int(budget), group_count)
        correct_selected = np.asarray(
            select_anchor_groups(correct_combined, grid_h, grid_w, effective),
            dtype=np.int64,
        )
        wrong_selected = np.asarray(
            select_anchor_groups(wrong_combined, grid_h, grid_w, effective),
            dtype=np.int64,
        )
        correct_mass = float(saliency_mass_retained(correct_selected, relevance))
        wrong_mass = float(saliency_mass_retained(wrong_selected, relevance))
        correct_ndcg = float(
            ndcg_for_selected_indices(
                correct_selected, correct_combined, relevance, effective
            )
        )
        wrong_ndcg = float(
            ndcg_for_selected_indices(wrong_selected, wrong_combined, relevance, effective)
        )
        correct_set = {int(value) for value in correct_selected}
        wrong_set = {int(value) for value in wrong_selected}
        result[int(budget)] = {
            "correct_mass": correct_mass,
            "wrong_mass": wrong_mass,
            "gap_mass": correct_mass - wrong_mass,
            "correct_ndcg": correct_ndcg,
            "wrong_ndcg": wrong_ndcg,
            "gap_ndcg": correct_ndcg - wrong_ndcg,
            "jaccard": float(jaccard(sorted(correct_set), sorted(wrong_set))),
            "overlap_count": int(len(correct_set & wrong_set)),
            "correct_selected_count": int(len(correct_set)),
            "wrong_selected_count": int(len(wrong_set)),
        }
    return result


# ---------------------------------------------------------------------------
# Image-level aggregation and paired statistics
# ---------------------------------------------------------------------------
def per_image_median(per_seed_values: Mapping[int, Sequence[float]]) -> list[float]:
    """Collapse the three seeds to one median per position (image)."""
    seeds = sorted(per_seed_values)
    if not seeds:
        raise DistillationContractError("no seeds to aggregate")
    lengths = {len(list(per_seed_values[s])) for s in seeds}
    if len(lengths) != 1:
        raise DistillationContractError("per-seed series must share a length")
    matrix = np.asarray([list(per_seed_values[s]) for s in seeds], dtype=np.float64)
    return [float(value) for value in np.median(matrix, axis=0)]


def summarize(
    values: Mapping[str, float],
    *,
    seed: int = STATISTICS_SEED,
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    n_permutation: int = PERMUTATION_RESAMPLES,
    equality_tolerance: float = EQUALITY_TOLERANCE,
) -> dict[str, Any]:
    """Per-image mean/median/95% CI/dz/p plus better-equal-worse against zero."""
    keys = sorted(values)
    if not keys:
        return {
            "n_images": 0,
            "mean": None,
            "median": None,
            "ci_low": None,
            "ci_high": None,
            "cohens_dz": None,
            "p_value_two_sided": None,
            "better": None,
            "equal": None,
            "worse": None,
        }
    array = np.asarray([float(values[key]) for key in keys], dtype=np.float64)
    effect = paired_effect(
        array.tolist(),
        seed=int(seed),
        n_bootstrap=int(n_bootstrap),
        n_permutation=int(n_permutation),
        sample_count=len(keys),
        independent_image_units=len(keys),
    )
    return {
        "n_images": len(keys),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci_low": effect["ci_low"],
        "ci_high": effect["ci_high"],
        "cohens_dz": effect["cohens_dz"],
        "p_value_two_sided": effect["p_value_two_sided"],
        "better": int(np.sum(array > equality_tolerance)),
        "equal": int(np.sum(np.abs(array) <= equality_tolerance)),
        "worse": int(np.sum(array < -equality_tolerance)),
    }


def _primary_entry(
    series: Mapping[str, float],
    *,
    seed_means: Mapping[int, float],
    definition: str,
    kind: str,
) -> dict[str, Any]:
    summary = summarize(series)
    positive = sum(1 for value in seed_means.values() if float(value) > 0.0)
    entry = dict(summary)
    entry.update(
        {
            "definition": definition,
            "kind": kind,
            "aggregation": (
                "per-image grounding/pointing mean then median across the three "
                "seeds, then paired effect over independent images"
            ),
            "independent_unit": "image",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "permutation_resamples": PERMUTATION_RESAMPLES,
            "seed_means": {str(int(k)): float(v) for k, v in sorted(seed_means.items())},
            "positive_seed_count": int(positive),
            "positive_seed_min": POSITIVE_SEED_MIN,
            "raw_p_value": summary["p_value_two_sided"],
            "mean_positive": bool(summary["mean"] is not None and summary["mean"] > 0.0),
            "ci_low_positive": bool(summary["ci_low"] is not None and summary["ci_low"] > 0.0),
            "effect_size_ok": bool(
                summary["cohens_dz"] is not None
                and abs(float(summary["cohens_dz"])) >= EFFECT_SIZE_FLOOR
            ),
            "seed_sign_ok": bool(positive >= POSITIVE_SEED_MIN),
            "effect_size_floor": EFFECT_SIZE_FLOOR,
            "holm_alpha": ALPHA,
        }
    )
    return entry


def evaluate_primary_tests(
    p1_series: Mapping[str, float],
    p2_series: Mapping[str, float],
    *,
    p1_seed_means: Mapping[int, float],
    p2_seed_means: Mapping[int, float],
    alpha: float = ALPHA,
) -> dict[str, Any]:
    """The frozen two-test K48 retained-mass Holm family (P1, P2)."""
    p1 = _primary_entry(
        p1_series,
        seed_means=p1_seed_means,
        definition=(
            "arm B deployed correct-prompt selector minus the same-image other-task "
            "wrong-prompt selector, retained spatial-target mass at K48"
        ),
        kind="correct_minus_wrong",
    )
    p2 = _primary_entry(
        p2_series,
        seed_means=p2_seed_means,
        definition=(
            "arm B minus arm A of the (correct-prompt minus wrong-prompt) deployed "
            "selector retained spatial-target mass gap at K48"
        ),
        kind="gap_arm_delta",
    )
    raw = {P1_KEY: float(p1["raw_p_value"]), P2_KEY: float(p2["raw_p_value"])}
    adjusted = holm_bonferroni(raw, alpha=float(alpha))
    tests = {P1_KEY: p1, P2_KEY: p2}
    for key, entry in tests.items():
        entry["holm_adjusted_p"] = float(adjusted[key])
        entry["pass"] = bool(
            entry["mean_positive"]
            and entry["ci_low_positive"]
            and entry["effect_size_ok"]
            and entry["seed_sign_ok"]
            and float(adjusted[key]) < float(alpha)
        )
    return {
        "family_size": 2,
        "family": [P1_KEY, P2_KEY],
        "holm_alpha": float(alpha),
        "multiple_comparison": "Holm-Bonferroni step-down, family size exactly 2",
        "tests": tests,
        "all_pass": all(entry["pass"] for entry in tests.values()),
    }


def evaluate_auxiliary_gates(
    *,
    primary_series: Mapping[str, Mapping[int, Mapping[str, Mapping[str, float]]]],
    primary_task_series: Mapping[
        str, Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]]
    ],
    images: Sequence[str],
    task_images: Mapping[str, Sequence[str]],
    prior: Mapping[str, Any],
    pins_unchanged: bool | None,
    identity_consistent: bool,
) -> dict[str, Any]:
    """Auxiliary gates S1-S4 (outside the Holm family)."""

    def _mean(mapping: Mapping[str, float]) -> float | None:
        return float(np.mean(list(mapping.values()))) if mapping else None

    gap_ndcg = primary_series[48]["gap_ndcg"]
    jaccard_series = primary_series[48]["jaccard"]
    grounding = {
        image: float(primary_task_series[48]["gap_mass"]["grounding"][image])
        for image in task_images["grounding"]
        if image in primary_task_series[48]["gap_mass"]["grounding"]
    }
    pointing = {
        image: float(primary_task_series[48]["gap_mass"]["pointing"][image])
        for image in task_images["pointing"]
        if image in primary_task_series[48]["gap_mass"]["pointing"]
    }
    grounding_mean = _mean(grounding)
    pointing_mean = _mean(pointing)
    gap_ndcg_mean = _mean({image: float(gap_ndcg[image]) for image in images})
    jaccard_mean = _mean({image: float(jaccard_series[image]) for image in images})
    pins_ok = True if pins_unchanged is None else bool(pins_unchanged)
    s4_pass = bool(
        prior.get("all_pass", False) and pins_ok and bool(identity_consistent)
    )
    return {
        "S1": {
            "definition": (
                "arm B grounding and pointing K48 correct-minus-wrong retained-mass "
                "means are both > 0"
            ),
            "grounding_mean": grounding_mean,
            "pointing_mean": pointing_mean,
            "pass": bool(
                grounding_mean is not None
                and pointing_mean is not None
                and grounding_mean > 0.0
                and pointing_mean > 0.0
            ),
        },
        "S2": {
            "definition": "arm B K48 correct-minus-wrong selected-set NDCG mean > 0",
            "mean": gap_ndcg_mean,
            "pass": bool(gap_ndcg_mean is not None and gap_ndcg_mean > 0.0),
        },
        "S3": {
            "definition": (
                "arm B K48 correct-vs-wrong selected Jaccard mean < 0.95"
            ),
            "mean": jaccard_mean,
            "threshold": JACCARD_GATE_THRESHOLD,
            "pass": bool(
                jaccard_mean is not None and jaccard_mean < JACCARD_GATE_THRESHOLD
            ),
        },
        "S4": {
            "definition": (
                "prior contrastive report G1/G4/G5/G7 all pass and every pinned "
                "input SHA matches"
            ),
            "prior_gates": dict(prior.get("gates", {})),
            "prior_gates_all_pass": bool(prior.get("all_pass", False)),
            "pins_unchanged": pins_unchanged,
            "identity_consistent": bool(identity_consistent),
            "pass": s4_pass,
        },
        "all_pass": bool(
            (grounding_mean is not None and pointing_mean is not None and grounding_mean > 0.0 and pointing_mean > 0.0)
            and gap_ndcg_mean is not None
            and gap_ndcg_mean > 0.0
            and jaccard_mean is not None
            and jaccard_mean < JACCARD_GATE_THRESHOLD
            and s4_pass
        ),
    }


def decide(
    primary: Mapping[str, Any], auxiliary: Mapping[str, Any]
) -> dict[str, Any]:
    supported = bool(primary.get("all_pass", False) and auxiliary.get("all_pass", False))
    return {
        "decision": (
            "OPERATIONAL_PROMPT_UTILITY_SUPPORTED"
            if supported
            else "OPERATIONAL_PROMPT_UTILITY_NOT_ESTABLISHED"
        ),
        "primary_holm_family_all_pass": bool(primary.get("all_pass", False)),
        "auxiliary_gates_all_pass": bool(auxiliary.get("all_pass", False)),
        "supported": supported,
        "rule": "supported iff P1, P2 and S1-S4 all pass",
    }


# ---------------------------------------------------------------------------
# Split rule
# ---------------------------------------------------------------------------
def load_split_rule(
    root: Path = ROOT,
    *,
    split_rule: Mapping[str, Any] | None = None,
    contrastive_config_path: Path = CONTRASTIVE_CONFIG_PATH,
) -> dict[str, Any]:
    """The frozen split rule, taken from the pinned contrastive prereg config."""
    if split_rule is not None:
        return {
            "seed": int(split_rule["seed"]),
            "ratios": dict(split_rule["ratios"]),
            "salt": str(split_rule["salt"]),
        }
    config = json.loads((root / contrastive_config_path).read_text(encoding="utf-8"))
    rule = config["split_rule"]
    return {
        "seed": int(rule["seed"]),
        "ratios": dict(rule["ratios"]),
        "salt": str(rule["salt"]),
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def run_diagnostic(
    *,
    root: Path = ROOT,
    prereg_config_path: Path = PREREG_CONFIG_PATH,
    prereg_report_path: Path = PREREG_REPORT_PATH,
    prior_report_path: Path = PRIOR_REPORT_PATH,
    index_path: Path = INDEX_PATH,
    manifest_path: Path = MANIFEST_PATH,
    artifacts_dir: Path = ARTIFACTS_DIR,
    features_dir: Path | None = None,
    arms: Sequence[str] = ARMS,
    seeds: Sequence[int] = SEEDS,
    budgets: Sequence[int] = BUDGETS,
    split_rule: Mapping[str, Any] | None = None,
    cache_loader: Callable[..., dict[str, Any]] | None = None,
    verify_pins: bool = True,
    strict_pins: bool = True,
) -> dict[str, Any]:
    """Run the full read-only diagnostic and return the report bundle."""
    started = time.time()
    cfg_path = root / prereg_config_path
    rep_path = root / prereg_report_path
    prereg = json.loads(cfg_path.read_text(encoding="utf-8"))
    prereg_report = json.loads(rep_path.read_text(encoding="utf-8"))

    identity = verify_prereg_identity(
        prereg,
        prereg_report,
        root=root,
        config_path=prereg_config_path,
        report_path=prereg_report_path,
    )
    if not identity["consistent"]:
        raise DistillationContractError(
            "preregistration identity mismatch: config/report/authority prompt SHA"
        )

    pin_report = (
        verify_prereg_pins(
            prereg_report, root=root, artifacts_dir=artifacts_dir, strict=strict_pins
        )
        if verify_pins
        else {"skipped": True, "pins_unchanged": None, "checked": 0, "mismatches": ["skipped"]}
    )

    rule = load_split_rule(root, split_rule=split_rule)
    index_records = load_index_metadata(root / index_path)
    splits = split_sample_ids(
        index_records,
        seed=int(rule["seed"]),
        ratios=rule["ratios"],
        salt=str(rule["salt"]),
    )
    val_ids = [str(value) for value in splits["val"]]
    test_ids = {str(value) for value in splits["test"]}
    train_ids = {str(value) for value in splits["train"]}
    directory = Path(features_dir) if features_dir else (root / index_path).parent

    # Internal test paths are only compared as strings for the isolation log; no
    # test path is ever resolved, hashed, opened, parsed or scored.
    test_path_strings = {
        str(directory / str(record["npz"]))
        for record in index_records
        if str(record["sample_id"]) in test_ids
    }
    train_path_strings = {
        str(directory / str(record["npz"]))
        for record in index_records
        if str(record["sample_id"]) in train_ids
    }

    opened: set[str] = set()
    val_samples_raw, val_aggregate, val_paths = load_split_samples(
        root / index_path,
        directory,
        val_ids,
        opened=opened,
        cache_loader=cache_loader,
    )
    opened_list = sorted(opened)
    test_opened = sorted(path for path in opened_list if path in test_path_strings)
    train_opened = sorted(path for path in opened_list if path in train_path_strings)
    if test_opened:
        raise DistillationContractError(f"internal test NPZ were opened: {test_opened}")
    if train_opened:
        raise DistillationContractError(
            f"train NPZ were opened during a val-only diagnostic: {train_opened}"
        )

    manifest_by_id, _ = load_clean_manifest(root / manifest_path)
    val_samples = attach_spatial_targets(val_samples_raw, manifest_by_id)

    by_id = {str(sample["sample_id"]): sample for sample in val_samples}
    partner = same_image_partner_map(val_samples)
    image_tasks: dict[str, list[str]] = defaultdict(list)
    image_group_count: dict[str, int] = {}
    image_grid: dict[str, tuple[int, int]] = {}
    task_images: dict[str, set[str]] = {task: set() for task in TASKS}
    for sample in val_samples:
        image = str(sample["source_image_sha256"])
        task = str(sample["task"])
        image_tasks[image].append(task)
        image_group_count[image] = int(sample["group_count"])
        image_grid[image] = (int(sample["grid_h"]), int(sample["grid_w"]))
        task_images[task].add(image)
    images = sorted(image_group_count)
    task_images_sorted = {task: sorted(values) for task, values in task_images.items()}

    # task_seed[arm][seed][(budget, metric, task)][image] = value
    task_seed: dict[str, dict[int, dict[tuple[int, str, str], dict[str, float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    csv_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, Any] = {}
    for arm in arms:
        for seed in seeds:
            seed = int(seed)
            ckpt_path = root / artifacts_dir / str(arm) / str(seed) / "best.npz"
            projection, file_sha = load_projection(ckpt_path)
            canonical = None
            with np.load(ckpt_path, allow_pickle=False) as archive:
                if "weights_sha256" in archive:
                    canonical = str(np.asarray(archive["weights_sha256"]).reshape(-1)[0])
            checkpoint_hashes[f"{arm}/{seed}"] = {
                "path": str(ckpt_path.relative_to(root)),
                "file_sha256": file_sha,
                "canonical_weights_sha256": canonical,
            }
            Wt, Wv = projection.Wt, projection.Wv
            for sample in val_samples:
                sample_id = str(sample["sample_id"])
                other = by_id[partner[sample_id]]
                target = np.asarray(sample["spatial_target"], dtype=np.float64)
                correct_scores = np.asarray(
                    score_candidate(
                        STUDENT_SCORER,
                        sample["group_mean"],
                        sample["prompt_embeddings"],
                        Wt,
                        Wv,
                    ),
                    dtype=np.float64,
                )
                wrong_scores = np.asarray(
                    score_candidate(
                        STUDENT_SCORER,
                        sample["group_mean"],
                        other["prompt_embeddings"],
                        Wt,
                        Wv,
                    ),
                    dtype=np.float64,
                )
                l2_scores = np.asarray(
                    l2_baseline_scores(sample["group_mean"]), dtype=np.float64
                )
                metrics = record_utility_metrics(
                    correct_scores=correct_scores,
                    wrong_scores=wrong_scores,
                    l2_scores=l2_scores,
                    target=target,
                    grid_h=int(sample["grid_h"]),
                    grid_w=int(sample["grid_w"]),
                    budgets=budgets,
                )
                image = str(sample["source_image_sha256"])
                task = str(sample["task"])
                for budget in budgets:
                    budget = int(budget)
                    entry = metrics[budget]
                    csv_rows.append(
                        {
                            "arm": str(arm),
                            "seed": seed,
                            "sample_id": sample_id,
                            "image_sha256": image,
                            "task": task,
                            "group_count": int(sample["group_count"]),
                            "grid_h": int(sample["grid_h"]),
                            "grid_w": int(sample["grid_w"]),
                            "budget": budget,
                            "lambda_semantic": float(LAMBDA_SEMANTIC),
                            "correct_retained_spatial_target_mass": entry["correct_mass"],
                            "wrong_retained_spatial_target_mass": entry["wrong_mass"],
                            "gap_retained_spatial_target_mass": entry["gap_mass"],
                            "correct_selected_set_ndcg": entry["correct_ndcg"],
                            "wrong_selected_set_ndcg": entry["wrong_ndcg"],
                            "gap_selected_set_ndcg": entry["gap_ndcg"],
                            "correct_wrong_selected_jaccard": entry["jaccard"],
                            "correct_wrong_selected_overlap_count": entry["overlap_count"],
                            "correct_selected_count": entry["correct_selected_count"],
                            "wrong_selected_count": entry["wrong_selected_count"],
                        }
                    )
                    for metric in METRICS:
                        task_seed[str(arm)][seed][(budget, metric, task)][image] = float(
                            entry[metric]
                        )

    # Per-seed image means for overall (grounding+pointing averaged), and the
    # per-image median across seeds for both overall and per-task series.
    overall_seed: dict[
        str, dict[int, dict[tuple[int, str], dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(dict))
    series: dict[str, dict[int, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    series_task: dict[
        str, dict[int, dict[str, dict[str, dict[str, float]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for arm in arms:
        arm = str(arm)
        for seed in seeds:
            seed = int(seed)
            accum: dict[tuple[int, str], dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for (budget, metric, _task), per_image in task_seed[arm][seed].items():
                for image, value in per_image.items():
                    accum[(budget, metric)][image].append(float(value))
            for (budget, metric), per_image in accum.items():
                overall_seed[arm][seed][(budget, metric)] = {
                    image: float(np.mean(values)) for image, values in per_image.items()
                }
        metrics_keys = {
            (budget, metric, task)
            for seed in seeds
            for (budget, metric, task) in task_seed[arm][int(seed)].keys()
        }
        for budget, metric, task in metrics_keys:
            per_image_seed: dict[str, dict[int, float]] = defaultdict(dict)
            for seed in seeds:
                for image, value in task_seed[arm][int(seed)][(budget, metric, task)].items():
                    per_image_seed[image][int(seed)] = float(value)
            for image, seed_values in per_image_seed.items():
                series_task[arm][budget][metric][task][image] = per_image_median(
                    {seed: [value] for seed, value in seed_values.items()}
                )[0]
        overall_keys = {
            (budget, metric)
            for seed in seeds
            for (budget, metric) in overall_seed[arm][int(seed)].keys()
        }
        for budget, metric in overall_keys:
            per_image_seed = defaultdict(dict)
            for seed in seeds:
                for image, value in overall_seed[arm][int(seed)][(budget, metric)].items():
                    per_image_seed[image][int(seed)] = float(value)
            for image, seed_values in per_image_seed.items():
                series[arm][budget][metric][image] = per_image_median(
                    {seed: [value] for seed, value in seed_values.items()}
                )[0]

    primary = series[PRIMARY_ARM]
    control = series[CONTROL_ARM]
    p1_series = {image: float(primary[48]["gap_mass"][image]) for image in images}
    p2_series = {
        image: float(primary[48]["gap_mass"][image]) - float(control[48]["gap_mass"][image])
        for image in images
    }
    p1_seed_means = {
        int(seed): float(
            np.mean(
                [
                    overall_seed[PRIMARY_ARM][int(seed)][(48, "gap_mass")][image]
                    for image in images
                ]
            )
        )
        for seed in seeds
    }
    p2_seed_means = {
        int(seed): float(
            np.mean(
                [
                    overall_seed[PRIMARY_ARM][int(seed)][(48, "gap_mass")][image]
                    - overall_seed[CONTROL_ARM][int(seed)][(48, "gap_mass")][image]
                    for image in images
                ]
            )
        )
        for seed in seeds
    }
    primary_tests = evaluate_primary_tests(
        p1_series, p2_series, p1_seed_means=p1_seed_means, p2_seed_means=p2_seed_means
    )
    prior = prior_gate_reference(root=root, prior_report_path=prior_report_path)
    auxiliary = evaluate_auxiliary_gates(
        primary_series=primary,
        primary_task_series=series_task[PRIMARY_ARM],
        images=images,
        task_images=task_images_sorted,
        prior=prior,
        pins_unchanged=pin_report.get("pins_unchanged"),
        identity_consistent=identity["consistent"],
    )
    decision = decide(primary_tests, auxiliary)

    # Derived strata summaries (overall / task / grid / group_count) per arm.
    by_group_count: dict[str, list[str]] = defaultdict(list)
    by_grid: dict[str, list[str]] = defaultdict(list)
    for image in images:
        by_group_count[str(image_group_count[image])].append(image)
        grid = image_grid[image]
        by_grid[f"{grid[0]}x{grid[1]}"].append(image)

    def _summary_block(arm: str) -> dict[str, Any]:
        strata_specs: list[tuple[str, Sequence[str], str | None]] = [
            ("overall", images, None),
            ("grounding", task_images_sorted["grounding"], "grounding"),
            ("pointing", task_images_sorted["pointing"], "pointing"),
        ]
        for key in sorted(by_grid):
            strata_specs.append((f"grid={key}", by_grid[key], None))
        for key in sorted(by_group_count):
            strata_specs.append((f"group_count={key}", by_group_count[key], None))
        block: dict[str, Any] = {}
        for lookup, stratum_images, task in strata_specs:
            per_budget: dict[str, Any] = {}
            for budget in budgets:
                budget = int(budget)
                metrics_out: dict[str, Any] = {}
                for metric in METRICS:
                    if task is None:
                        source = series[arm][budget][metric]
                    else:
                        source = series_task[arm][budget][metric].get(task, {})
                    values = {
                        image: float(source[image])
                        for image in stratum_images
                        if image in source
                    }
                    metrics_out[metric] = summarize(values)
                per_budget[f"k{budget}"] = metrics_out
            block[lookup] = per_budget
        return block

    arm_summaries = {str(arm): _summary_block(str(arm)) for arm in arms}

    prereg_sha_before = identity["config_sha256"]
    prereg_report_sha_before = identity["report_sha256"]
    prereg_sha_after = sha256_file(cfg_path)
    prereg_report_sha_after = sha256_file(rep_path)

    def _k_gap(arm: str, budget: int, metric: str = "gap_mass") -> dict[str, Any]:
        return arm_summaries[arm]["overall"][f"k{int(budget)}"][metric]

    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "date": DATE,
        "status": decision["decision"],
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exploratory": {
            "validation_already_used": True,
            "statement": (
                "This is an exploratory operational prompt-utility diagnostic on "
                "already-used validation data; it may only inform the next "
                "preregistration and cannot support a final accuracy or promotion "
                "conclusion."
            ),
            "no_internal_test": True,
            "no_gpu": True,
            "no_training": True,
            "not_final_accuracy": True,
        },
        "provenance": {
            "prereg_config": {
                "path": str(prereg_config_path),
                "sha256_before": prereg_sha_before,
                "sha256_after": prereg_sha_after,
                "unchanged": prereg_sha_before == prereg_sha_after,
            },
            "prereg_report": {
                "path": str(prereg_report_path),
                "sha256_before": prereg_report_sha_before,
                "sha256_after": prereg_report_sha_after,
                "unchanged": prereg_report_sha_before == prereg_report_sha_after,
            },
            "identity": identity,
            "pins": pin_report,
            "checkpoints": checkpoint_hashes,
            "val_npz_aggregate_sha256": val_aggregate,
            "val_npz_aggregate_matches_prereg": (
                val_aggregate
                == prereg.get("data", {}).get("val_npz_aggregate_sha256")
            ),
            "val_npz_count": len(val_paths),
        },
        "isolation": {
            "opened_npz_count": len(opened_list),
            "opened_only_val_npz": not test_opened and not train_opened,
            "test_npz_opened": test_opened,
            "train_npz_opened": train_opened,
            "test_npz_resolved_hashed_or_opened": False,
            "internal_test_report_files_parsed": False,
            "prior_report_json_parsed_for_gate_reference_only": bool(prior.get("available")),
            "restricted_val_report_parsed_note": (
                "the prior contrastive report JSON is byte-hashed by the pins and "
                "additionally read only for the S4 G1/G4/G5/G7 gate booleans; its "
                "validation statistics are never used as a metric source"
            ),
            "restricted_val_samples_csv_or_json_parsed_as_metric_source": False,
        },
        "split_counts": {
            "train_records": len(splits["train"]),
            "val_records": len(splits["val"]),
            "test_records": len(splits["test"]),
            "val_images": len(images),
            "val_grounding_images": len(task_images_sorted["grounding"]),
            "val_pointing_images": len(task_images_sorted["pointing"]),
            "expected_val_records": int(prereg.get("data", {}).get("expected_records", 76)),
            "expected_val_images": int(prereg.get("data", {}).get("expected_images", 38)),
        },
        "config": {
            "arms": [str(arm) for arm in arms],
            "primary_arm": PRIMARY_ARM,
            "control_arm": CONTROL_ARM,
            "seeds": [int(seed) for seed in seeds],
            "budgets": [int(budget) for budget in budgets],
            "lambda_semantic": LAMBDA_SEMANTIC,
            "student_scorer": STUDENT_SCORER,
            "region_grid": 3,
            "correct_prompt": "record prompt",
            "wrong_prompt": "same-image other-task prompt",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "sign_flip_resamples": PERMUTATION_RESAMPLES,
            "statistics_seed": STATISTICS_SEED,
            "alpha": ALPHA,
            "effect_size_floor": EFFECT_SIZE_FLOOR,
            "positive_seed_min": POSITIVE_SEED_MIN,
            "split_rule": rule,
        },
        "arms": arm_summaries,
        "primary_holm_family": primary_tests,
        "auxiliary_gates": auxiliary,
        "decision": decision,
        "prior_report_reference": prior,
        "k48_gap_mass": {
            "B_correct_minus_wrong": _k_gap(PRIMARY_ARM, 48),
            "A_correct_minus_wrong": _k_gap(CONTROL_ARM, 48),
            "B_minus_A": summarize(p2_series),
        },
        "budget_summary": {
            f"k{int(budget)}": {
                "B_correct_minus_wrong_gap_mass": _k_gap(PRIMARY_ARM, budget),
                "A_correct_minus_wrong_gap_mass": _k_gap(CONTROL_ARM, budget),
            }
            for budget in budgets
        },
        "task_k48": {
            task: {
                "B_correct_minus_wrong_gap_mass_mean": arm_summaries[PRIMARY_ARM][task][
                    "k48"
                ]["gap_mass"]["mean"],
                "B_correct_minus_wrong_gap_ndcg_mean": arm_summaries[PRIMARY_ARM][task][
                    "k48"
                ]["gap_ndcg"]["mean"],
                "B_correct_wrong_jaccard_mean": arm_summaries[PRIMARY_ARM][task]["k48"][
                    "jaccard"
                ]["mean"],
            }
            for task in TASKS
        },
        "k48_jaccard": arm_summaries[PRIMARY_ARM]["overall"]["k48"]["jaccard"],
        "limitations": [
            "Exploratory diagnostic on already-used validation; not a final accuracy conclusion.",
            "No internal test NPZ was resolved, hashed, opened, parsed or scored.",
            "The val split was already used during development; this is not final or CV-Bench accuracy.",
            "No hardware synthesis, latency or speedup claim.",
            "The prior contrastive decision PROMPT_CONTRASTIVE_STUDENT_NOT_ESTABLISHED is unchanged; this diagnostic only adds exploratory evidence.",
            "The prior report JSON is parsed only for the S4 gate booleans; its statistics and val-samples CSV are never metric sources.",
            "Per-image statistics treat the 38 images as independent units.",
            "The selector chain is descriptive; passing P1/P2 does not by itself establish task accuracy or promotion.",
        ],
        "duration_seconds": round(time.time() - started, 3),
    }
    return {"bundle": bundle, "csv_rows": csv_rows, "prereg": prereg}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def render_markdown(bundle: Mapping[str, Any]) -> str:
    decision = bundle["decision"]
    primary = bundle["primary_holm_family"]["tests"]
    aux = bundle["auxiliary_gates"]
    iso = bundle["isolation"]
    prov = bundle["provenance"]
    lines = [
        "# 部署态 Prompt 效用诊断（探索性）",
        "",
        f"- 日期：{bundle['date']}；状态：`{bundle['status']}`",
        "- 这是**已使用 validation 的零训练探索性诊断**，不是最终精度，不改上一 contrastive 结论，未触碰 internal test。",
        "",
        "## 主 Holm 家族（恰好 2 项，仅 K48 retained mass）",
        "",
        "| 检验 | mean | CI low | CI high | dz | raw p | Holm p | +seeds | pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, entry in primary.items():
        lines.append(
            f"| {key} | {entry['mean']:+.6f} | {entry['ci_low']:+.6f} | "
            f"{entry['ci_high']:+.6f} | {entry['cohens_dz']:.4f} | "
            f"{entry['raw_p_value']:.5f} | {entry['holm_adjusted_p']:.5f} | "
            f"{entry['positive_seed_count']}/3 | {entry['pass']} |"
        )
    lines += [
        "",
        "## 辅助门槛 S1–S4（不进入 Holm）",
        "",
        f"- S1 grounding mean = {aux['S1']['grounding_mean']}, pointing mean = "
        f"{aux['S1']['pointing_mean']}；pass = {aux['S1']['pass']}",
        f"- S2 K48 gap NDCG mean = {aux['S2']['mean']}；pass = {aux['S2']['pass']}",
        f"- S3 K48 correct-vs-wrong Jaccard mean = {aux['S3']['mean']} "
        f"(threshold {aux['S3']['threshold']})；pass = {aux['S3']['pass']}",
        f"- S4 prior gates all pass = {aux['S4']['prior_gates_all_pass']}, "
        f"pins unchanged = {aux['S4']['pins_unchanged']}, "
        f"identity = {aux['S4']['identity_consistent']}；pass = {aux['S4']['pass']}",
        "",
        "## 各任务 K48（arm B，correct − wrong）",
        "",
        "| 任务 | gap mass mean | gap NDCG mean | correct-vs-wrong Jaccard |",
        "| --- | --- | --- | --- |",
    ]
    for task in TASKS:
        entry = bundle["task_k48"][task]
        lines.append(
            f"| {task} | {entry['B_correct_minus_wrong_gap_mass_mean']:+.6f} | "
            f"{entry['B_correct_minus_wrong_gap_ndcg_mean']:+.6f} | "
            f"{entry['B_correct_wrong_jaccard_mean']:.4f} |"
        )
    lines += [
        "",
        "## K48/64/96（arm B 与 A 的 correct − wrong gap mass，overall）",
        "",
        "| budget | B gap mass mean | A gap mass mean |",
        "| --- | --- | --- |",
    ]
    for budget in bundle["budget_summary"]:
        entry = bundle["budget_summary"][budget]
        lines.append(
            f"| {budget} | {entry['B_correct_minus_wrong_gap_mass']['mean']:+.6f} | "
            f"{entry['A_correct_minus_wrong_gap_mass']['mean']:+.6f} |"
        )
    lines += [
        "",
        "## 决策",
        "",
        f"- **{decision['decision']}**（P1/P2 = {decision['primary_holm_family_all_pass']}，"
        f"S1–S4 = {decision['auxiliary_gates_all_pass']}）",
        "",
        "## 隔离与溯源",
        "",
        f"- 打开 NPZ 数：{iso['opened_npz_count']}（仅 val：{iso['opened_only_val_npz']}）",
        f"- test NPZ 打开：{iso['test_npz_opened']}；train NPZ 打开：{iso['train_npz_opened']}",
        f"- test 路径解析/哈希/打开：{iso['test_npz_resolved_hashed_or_opened']}",
        f"- 预注册 config 运行前后 SHA 不变：{prov['prereg_config']['unchanged']}",
        f"- 预注册 report 运行前后 SHA 不变：{prov['prereg_report']['unchanged']}",
        f"- 所有 pin 输入不变：{prov['pins'].get('pins_unchanged')}",
        f"- val NPZ aggregate 与预注册一致：{prov['val_npz_aggregate_matches_prereg']}",
        "",
        "## 限制",
        "",
    ]
    for item in bundle["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    bundle: Mapping[str, Any],
    csv_rows: Sequence[Mapping[str, Any]],
    *,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, str]:
    json_path = reports_dir / f"{REPORT_STEM}.json"
    md_path = reports_dir / f"{REPORT_STEM}.md"
    csv_path = reports_dir / f"{REPORT_STEM}-records.csv"
    _write_json(json_path, bundle)
    md_path.write_text(render_markdown(bundle), encoding="utf-8")
    _write_csv(csv_path, csv_rows)
    return {
        "json": str(json_path),
        "md": str(md_path),
        "csv": str(csv_path),
        "json_sha256": sha256_file(json_path),
        "md_sha256": sha256_file(md_path),
        "csv_sha256": sha256_file(csv_path),
    }


def harness_payload(bundle: Mapping[str, Any], files: Mapping[str, str]) -> dict[str, Any]:
    primary = bundle["primary_holm_family"]["tests"]
    aux = bundle["auxiliary_gates"]
    decision = bundle["decision"]

    def _test(key: str) -> dict[str, Any]:
        entry = primary[key]
        return {
            "mean": entry["mean"],
            "ci_low": entry["ci_low"],
            "ci_high": entry["ci_high"],
            "cohens_dz": entry["cohens_dz"],
            "raw_p_value": entry["raw_p_value"],
            "holm_adjusted_p": entry["holm_adjusted_p"],
            "positive_seed_count": entry["positive_seed_count"],
            "seed_means": entry["seed_means"],
            "pass": entry["pass"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "report_stem": REPORT_STEM,
        "status": bundle["status"],
        "decision": decision,
        "P1": _test(P1_KEY),
        "P2": _test(P2_KEY),
        "S1": {"pass": aux["S1"]["pass"], "grounding_mean": aux["S1"]["grounding_mean"], "pointing_mean": aux["S1"]["pointing_mean"]},
        "S2": {"pass": aux["S2"]["pass"], "mean": aux["S2"]["mean"]},
        "S3": {"pass": aux["S3"]["pass"], "mean": aux["S3"]["mean"], "threshold": aux["S3"]["threshold"]},
        "S4": {
            "pass": aux["S4"]["pass"],
            "prior_gates_all_pass": aux["S4"]["prior_gates_all_pass"],
            "pins_unchanged": aux["S4"]["pins_unchanged"],
            "identity_consistent": aux["S4"]["identity_consistent"],
        },
        "task_k48": bundle["task_k48"],
        "budget_summary": bundle["budget_summary"],
        "k48_jaccard": bundle["k48_jaccard"],
        "isolation": bundle["isolation"],
        "provenance": {
            "prereg_config_unchanged": bundle["provenance"]["prereg_config"]["unchanged"],
            "prereg_report_unchanged": bundle["provenance"]["prereg_report"]["unchanged"],
            "pins_unchanged": bundle["provenance"]["pins"].get("pins_unchanged"),
            "val_npz_aggregate_matches_prereg": bundle["provenance"][
                "val_npz_aggregate_matches_prereg"
            ],
        },
        "limitations": list(bundle.get("limitations", [])),
        "next_decision": (
            "operational prompt utility supported at the fixed-budget selector "
            "output on already-used validation; a fresh preregistered held-out "
            "evaluation is required before any promotion or task-accuracy claim"
            if decision["supported"]
            else "operational prompt utility not established; do not promote; a "
            "fresh preregistration may test a different prompt-supervision signal"
        ),
        "files": dict(files),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--no-pin-verify", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    root = args.root.resolve()
    result = run_diagnostic(root=root, verify_pins=not args.no_pin_verify)
    bundle = result["bundle"]
    files: dict[str, str] = {}
    if not args.no_write:
        files = write_outputs(bundle, result["csv_rows"], reports_dir=root / REPORTS_DIR)
        harness_path = root / LOGS_DIR / f"{REPORT_STEM}-harness-result.json"
        _write_json(harness_path, harness_payload(bundle, files))
        files["harness_result"] = str(harness_path)
    summary = {
        "status": bundle["status"],
        "decision": bundle["decision"],
        "P1": bundle["primary_holm_family"]["tests"][P1_KEY],
        "P2": bundle["primary_holm_family"]["tests"][P2_KEY],
        "auxiliary_gates": {
            key: bundle["auxiliary_gates"][key].get("pass")
            for key in ("S1", "S2", "S3", "S4")
        },
        "task_k48": bundle["task_k48"],
        "files": files,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
