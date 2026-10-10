#!/usr/bin/env python3
"""Train a safety-anchored selector from measured joint-deletion outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.joint_deletion import (
    candidate_set_utilities,
    pairwise_set_ranking_loss,
)
from evaluation.learned_selector import L2ResidualSelectorHead, head_cost
from evaluation.metrics import load_jsonl
from evaluation.shallow_block_pruning import structured_group_indices
from evaluation.train_coordinate_selector import (
    deterministic_split,
    file_sha256,
    image_id,
    pearson,
    ranks,
)


@dataclass(frozen=True)
class JointRecord:
    sample_id: str
    image_id: str
    image: str
    task: str
    grid_h: int
    grid_w: int
    retention: float
    features: torch.Tensor
    normalized_l2: torch.Tensor
    candidate_ids: tuple[str, ...]
    candidate_swap_counts: tuple[int, ...]
    candidate_indices: tuple[tuple[int, ...], ...]
    coordinate_nll: torch.Tensor


def indexed_rows(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id in {path}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def candidate_config(teachers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    configs = []
    for row in teachers.values():
        run = row.get("run", {})
        configs.append(
            {
                "base_seed": int(run["seed"]),
                "swap_counts": [int(value) for value in run["swap_counts"]],
                "candidates_per_swap": int(run["candidates_per_swap"]),
                "region_grid": int(run["region_grid"]),
            }
        )
    first = configs[0]
    if any(config != first for config in configs[1:]):
        raise ValueError("teacher records use inconsistent candidate policies")
    return first


def load_joint_records(
    feature_path: Path,
    teacher_path: Path,
) -> tuple[list[JointRecord], dict[str, Any]]:
    features = indexed_rows(feature_path)
    teachers = indexed_rows(teacher_path)
    if features.keys() != teachers.keys():
        missing_features = sorted(teachers.keys() - features.keys())
        missing_teachers = sorted(features.keys() - teachers.keys())
        raise ValueError(
            f"feature/teacher IDs differ: missing_features={missing_features[:3]} "
            f"missing_teachers={missing_teachers[:3]}"
        )

    records: list[JointRecord] = []
    paired_masks: dict[str, tuple[tuple[int, ...], ...]] = {}
    for sample_id in sorted(features):
        feature = features[sample_id]
        teacher = teachers[sample_id]
        if feature["task"] != teacher["task"] or feature["grid_hw"] != teacher["grid_hw"]:
            raise ValueError(f"metadata mismatch for {sample_id}")
        grid_h, grid_w = map(int, feature["grid_hw"])
        group_count = grid_h * grid_w
        selector_features = torch.tensor(
            feature["selector_features"], dtype=torch.float32
        )
        normalized_l2 = torch.tensor(feature["normalized_l2"], dtype=torch.float32)
        if selector_features.shape != (group_count, 5):
            raise ValueError(f"bad feature shape for {sample_id}")
        if normalized_l2.shape != (group_count,):
            raise ValueError(f"bad L2 shape for {sample_id}")
        if not bool(
            torch.isfinite(selector_features).all()
            and torch.isfinite(normalized_l2).all()
        ):
            raise ValueError(f"non-finite feature record {sample_id}")

        retention = float(teacher["retention"])
        expected_keep = math.ceil(group_count * retention)
        candidates = teacher["candidates"]
        if len(candidates) < 2 or candidates[0]["candidate_id"] != "baseline":
            raise ValueError(f"missing baseline candidate for {sample_id}")
        candidate_ids = tuple(str(row["candidate_id"]) for row in candidates)
        candidate_swaps = tuple(int(row["swap_count"]) for row in candidates)
        indices: list[tuple[int, ...]] = []
        nll: list[float] = []
        for candidate in candidates:
            one = tuple(int(index) for index in candidate["selected_group_indices"])
            if len(one) != expected_keep or len(set(one)) != expected_keep:
                raise ValueError(f"bad candidate budget for {sample_id}")
            if tuple(sorted(one)) != one or one[0] < 0 or one[-1] >= group_count:
                raise ValueError(f"bad candidate indices for {sample_id}")
            value = float(candidate["coordinate_nll"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite coordinate NLL for {sample_id}")
            indices.append(one)
            nll.append(value)

        baseline = structured_group_indices(
            normalized_l2, grid_h, grid_w, retention, anchor_grid=3
        )
        if tuple(int(index) for index in baseline.tolist()) != indices[0]:
            raise ValueError(f"feature L2 baseline differs from teacher for {sample_id}")

        one_image_id = image_id(sample_id)
        signature = tuple(indices)
        previous = paired_masks.setdefault(one_image_id, signature)
        if previous != signature:
            raise ValueError(
                f"paired tasks do not share candidate masks for {one_image_id}"
            )
        records.append(
            JointRecord(
                sample_id=sample_id,
                image_id=one_image_id,
                image=str(teacher["image"]),
                task=str(feature["task"]),
                grid_h=grid_h,
                grid_w=grid_w,
                retention=retention,
                features=selector_features,
                normalized_l2=normalized_l2,
                candidate_ids=candidate_ids,
                candidate_swap_counts=candidate_swaps,
                candidate_indices=tuple(indices),
                coordinate_nll=torch.tensor(nll, dtype=torch.float32),
            )
        )

    retentions = {record.retention for record in records}
    if len(retentions) != 1:
        raise ValueError("joint teacher records use multiple retention values")
    return records, candidate_config(teachers)


def masked_features(record: JointRecord, feature_mask: tuple[int, ...]) -> torch.Tensor:
    result = record.features.clone()
    disabled = sorted(set(range(5)) - set(feature_mask))
    if disabled:
        result[:, disabled] = 0
    return result


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def one_record_metrics(
    logits: torch.Tensor,
    correction: torch.Tensor,
    record: JointRecord,
    *,
    tie_epsilon: float,
) -> dict[str, float]:
    utilities = candidate_set_utilities(logits, record.candidate_indices)
    predicted = int(torch.argmax(utilities).item())
    oracle = int(torch.argmin(record.coordinate_nll).item())
    baseline_nll = float(record.coordinate_nll[0])
    selected_nll = float(record.coordinate_nll[predicted])
    oracle_nll = float(record.coordinate_nll[oracle])

    correct = 0.0
    pair_count = 0
    for left in range(len(record.candidate_indices)):
        for right in range(left + 1, len(record.candidate_indices)):
            nll_delta = float(
                record.coordinate_nll[right] - record.coordinate_nll[left]
            )
            if abs(nll_delta) <= tie_epsilon:
                continue
            utility_delta = float(utilities[left] - utilities[right])
            pair_count += 1
            if abs(utility_delta) <= 1e-12:
                correct += 0.5
            elif nll_delta * utility_delta > 0:
                correct += 1.0
    selected = set(record.candidate_indices[predicted])
    baseline = set(record.candidate_indices[0])
    return {
        "pairwise_accuracy": correct / pair_count if pair_count else 0.5,
        "candidate_spearman": pearson(
            ranks(utilities), ranks(-record.coordinate_nll)
        ),
        "selected_delta_nll_vs_l2": selected_nll - baseline_nll,
        "oracle_delta_nll_vs_l2": oracle_nll - baseline_nll,
        "top1_regret": selected_nll - oracle_nll,
        "oracle_hit": float(predicted == oracle),
        "beats_l2": float(selected_nll < baseline_nll - tie_epsilon),
        "no_worse_than_l2": float(selected_nll <= baseline_nll + tie_epsilon),
        "l2_selection_overlap": len(selected & baseline) / len(baseline),
        "selected_swap_count": float(record.candidate_swap_counts[predicted]),
        "selected_candidate_index": float(predicted),
        "correction_rms": float(correction.square().mean().sqrt()),
    }


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    keys = [key for key in items[0] if key != "selected_candidate_index"]
    result = {
        key: sum(item[key] for item in items) / len(items)
        for key in keys
    }
    regrets = [item["top1_regret"] for item in items]
    deltas = [item["selected_delta_nll_vs_l2"] for item in items]
    result.update(
        {
            "median_top1_regret": statistics.median(regrets),
            "p95_top1_regret": percentile(regrets, 0.95),
            "max_top1_regret": max(regrets),
            "median_selected_delta_nll_vs_l2": statistics.median(deltas),
            "p95_selected_delta_nll_vs_l2": percentile(deltas, 0.95),
            "max_selected_delta_nll_vs_l2": max(deltas),
        }
    )
    return result


def evaluate(
    head: L2ResidualSelectorHead | None,
    records: list[JointRecord],
    feature_mask: tuple[int, ...],
    *,
    tie_epsilon: float,
) -> dict[str, Any]:
    metrics: list[dict[str, float]] = []
    losses: list[float] = []
    by_task: dict[str, list[dict[str, float]]] = {}
    choices: Counter[str] = Counter()
    if head is not None:
        head.eval()
    with torch.inference_mode():
        for record in records:
            if head is None:
                correction = torch.zeros_like(record.normalized_l2)
                logits = record.normalized_l2
            else:
                features = masked_features(record, feature_mask)
                correction = head.correction(features)
                logits = record.normalized_l2 + correction
            try:
                loss = pairwise_set_ranking_loss(
                    logits,
                    record.candidate_indices,
                    record.coordinate_nll,
                    tie_epsilon=tie_epsilon,
                    utility_scale=len(record.candidate_indices[0]),
                )
                losses.append(float(loss))
            except ValueError as error:
                if "tied" not in str(error):
                    raise
            one = one_record_metrics(
                logits, correction, record, tie_epsilon=tie_epsilon
            )
            metrics.append(one)
            by_task.setdefault(record.task, []).append(one)
            choice = record.candidate_ids[int(one["selected_candidate_index"])]
            choices[choice] += 1
    return {
        "ranking_loss": sum(losses) / len(losses) if losses else None,
        **aggregate(metrics),
        "selected_candidate_counts": dict(sorted(choices.items())),
        "by_task": {
            task: aggregate(task_items)
            for task, task_items in sorted(by_task.items())
        },
    }


def variant_label(feature_name: str, penalty: float) -> str:
    encoded = f"{penalty:g}".replace(".", "p")
    return f"{feature_name}_lambda_{encoded}"


def train_one(
    train: list[JointRecord],
    calibration: list[JointRecord],
    *,
    feature_mask: tuple[int, ...],
    penalty: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    tie_epsilon: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    head = L2ResidualSelectorHead()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(train)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        ranking_sum = torch.zeros(())
        correction_sum = torch.zeros(())
        ranked_records = 0
        for index in order:
            record = train[index]
            features = masked_features(record, feature_mask)
            correction = head.correction(features)
            logits = record.normalized_l2 + correction
            try:
                ranking_sum = ranking_sum + pairwise_set_ranking_loss(
                    logits,
                    record.candidate_indices,
                    record.coordinate_nll,
                    tie_epsilon=tie_epsilon,
                    utility_scale=len(record.candidate_indices[0]),
                )
                ranked_records += 1
            except ValueError as error:
                if "tied" not in str(error):
                    raise
            correction_sum = correction_sum + correction.square().mean()
        if ranked_records == 0:
            raise RuntimeError("all training candidate sets are tied")
        ranking_loss = ranking_sum / ranked_records
        correction_penalty = correction_sum / len(train)
        objective = ranking_loss + penalty * correction_penalty
        objective.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            calibration_metrics = evaluate(
                head,
                calibration,
                feature_mask,
                tie_epsilon=tie_epsilon,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_ranking_loss": float(ranking_loss.detach()),
                    "train_correction_penalty": float(correction_penalty.detach()),
                    "calibration_top1_regret": calibration_metrics["top1_regret"],
                    "calibration_pairwise_accuracy": calibration_metrics[
                        "pairwise_accuracy"
                    ],
                }
            )
            key = (
                calibration_metrics["top1_regret"],
                -calibration_metrics["pairwise_accuracy"],
                -calibration_metrics["l2_selection_overlap"],
            )
            if best_metrics is None:
                improved = True
            else:
                best_key = (
                    best_metrics["top1_regret"],
                    -best_metrics["pairwise_accuracy"],
                    -best_metrics["l2_selection_overlap"],
                )
                improved = key < best_key
            if improved:
                best_epoch = epoch
                best_metrics = calibration_metrics
                best_state = {
                    key: value.detach().clone()
                    for key, value in head.state_dict().items()
                }
    if best_state is None or best_metrics is None:
        raise RuntimeError("no joint-deletion selector checkpoint")
    head.load_state_dict(best_state)
    return best_state, {
        "penalty": penalty,
        "best_epoch": best_epoch,
        "train_at_best": evaluate(
            head, train, feature_mask, tie_epsilon=tie_epsilon
        ),
        "calibration_at_best": best_metrics,
        "history": history,
    }


def development_gate(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    criteria = {
        "pairwise_accuracy_at_least_0p52": metrics["pairwise_accuracy"] >= 0.52,
        "mean_coordinate_nll_better_than_l2": (
            metrics["selected_delta_nll_vs_l2"] < 0.0
        ),
        "top1_regret_reduced_by_at_least_5_percent": (
            metrics["top1_regret"] <= 0.95 * baseline["top1_regret"]
        ),
        "mean_l2_selection_overlap_at_least_0p95": (
            metrics["l2_selection_overlap"] >= 0.95
        ),
    }
    return {
        "passed": all(criteria.values()),
        "criteria": criteria,
        "scope": (
            "development coordinate-NLL gate only; not a task-accuracy or "
            "held-out-test claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--penalties",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    )
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20270825)
    parser.add_argument("--tie-epsilon", type=float, default=1e-4)
    args = parser.parse_args()
    if any(value < 0 for value in args.penalties):
        parser.error("penalties must be nonnegative")
    if args.tie_epsilon < 0:
        parser.error("tie epsilon must be nonnegative")

    records, policy = load_joint_records(args.features, args.teacher)
    train, calibration, train_ids, calibration_ids = deterministic_split(
        records, args.train_images, args.seed
    )
    feature_masks = {
        "full": (0, 1, 2, 3, 4),
        "no_text": (0, 1, 3, 4),
    }
    baseline = {
        "train": evaluate(
            None, train, feature_masks["full"], tie_epsilon=args.tie_epsilon
        ),
        "calibration": evaluate(
            None,
            calibration,
            feature_masks["full"],
            tie_epsilon=args.tie_epsilon,
        ),
    }

    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    variant_masks: dict[str, list[int]] = {}
    results: dict[str, Any] = {}
    for feature_name, feature_mask in feature_masks.items():
        for penalty in args.penalties:
            name = variant_label(feature_name, penalty)
            state, result = train_one(
                train,
                calibration,
                feature_mask=feature_mask,
                penalty=penalty,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                tie_epsilon=args.tie_epsilon,
            )
            checkpoints[name] = state
            variant_masks[name] = list(feature_mask)
            results[name] = result
            metrics = result["calibration_at_best"]
            print(
                f"RESULT {name} epoch={result['best_epoch']} "
                f"pair={metrics['pairwise_accuracy']:.4f} "
                f"regret={metrics['top1_regret']:.6f} "
                f"delta={metrics['selected_delta_nll_vs_l2']:+.6f} "
                f"overlap={metrics['l2_selection_overlap']:.4f}",
                flush=True,
            )

    gates = {
        name: development_gate(
            result["calibration_at_best"], baseline["calibration"]
        )
        for name, result in results.items()
    }
    passing = [name for name, gate in gates.items() if gate["passed"]]
    selection_pool = passing or list(results)
    selected = min(
        selection_pool,
        key=lambda name: (
            results[name]["calibration_at_best"]["top1_regret"],
            results[name]["calibration_at_best"]["selected_delta_nll_vs_l2"],
            -results[name]["calibration_at_best"]["pairwise_accuracy"],
            -results[name]["calibration_at_best"]["l2_selection_overlap"],
            results[name]["penalty"],
            name,
        ),
    )
    split_payload = json.dumps(
        {"train": train_ids, "calibration": calibration_ids}, sort_keys=True
    ).encode()
    split_sha256 = hashlib.sha256(split_payload).hexdigest()
    retention = records[0].retention
    artifact = {
        "format": "joint-deletion-selector-v1",
        "selected_variant": selected,
        "selected_state_dict": checkpoints[selected],
        "variant_state_dicts": checkpoints,
        "variant_feature_masks": variant_masks,
        "split_sha256": split_sha256,
        "seed": args.seed,
        "retention": retention,
        "candidate_config": policy,
        "development_gate_passed": gates[selected]["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        "format": "joint-deletion-selector-training-summary-v1",
        "status": (
            "DEVELOPMENT_GATE_PASS"
            if gates[selected]["passed"]
            else "DEVELOPMENT_GATE_STOP"
        ),
        "features": str(args.features.resolve()),
        "features_sha256": file_sha256(args.features),
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": file_sha256(args.teacher),
        "records": len(records),
        "task_counts": dict(sorted(Counter(record.task for record in records).items())),
        "train_images": len(train_ids),
        "calibration_images": len(calibration_ids),
        "train_image_ids": train_ids,
        "calibration_image_ids": calibration_ids,
        "split_sha256": split_sha256,
        "retention": retention,
        "candidate_config": policy,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "tie_epsilon": args.tie_epsilon,
            "penalties": args.penalties,
            "utility_scale": "retained_group_count",
        },
        "head_cost": head_cost(),
        "feature_masks": {
            name: list(mask) for name, mask in feature_masks.items()
        },
        "l2_baseline": baseline,
        "variants": results,
        "development_gates": gates,
        "selected_variant": selected,
        "selected_development_gate": gates[selected],
        "selection_rule": (
            "minimum calibration top-1 coordinate-NLL regret among variants "
            "passing the pre-registered development gate; otherwise diagnostic "
            "best only and stop before real-output inference"
        ),
        "real_output_gate_preregistered": {
            "reference": "L2@87.5 on the same 32-image development calibration split",
            "per_primary_metric_max_drop": 0.02,
            "requires_at_least_one_primary_metric_improvement": True,
            "scope": (
                "calibration gate only; a new untouched test set is required "
                "for any final accuracy claim"
            ),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        f"SELECTED {selected} gate={gates[selected]['passed']} "
        f"artifact={args.output} summary={args.summary}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
