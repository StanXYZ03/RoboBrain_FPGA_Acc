#!/usr/bin/env python3
"""Train a candidate-level interaction head for safe joint deletion."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from evaluation.joint_deletion import (
    candidate_interaction_features,
    candidate_set_utilities,
)
from evaluation.learned_selector import CandidateInteractionHead, candidate_head_cost
from evaluation.train_coordinate_selector import deterministic_split, pearson, ranks
from evaluation.train_joint_deletion_selector import (
    JointRecord,
    aggregate,
    load_joint_records,
)


def descriptor_batch(
    records: list[JointRecord],
    feature_mask: tuple[int, ...],
    *,
    task_conditioned: bool,
) -> torch.Tensor:
    descriptors = []
    disabled = sorted(set(range(5)) - set(feature_mask))
    for record in records:
        features = record.features.clone()
        if disabled:
            features[:, disabled] = 0
        if task_conditioned:
            task_value = {"grounding": -1.0, "pointing": 1.0}[record.task]
        else:
            task_value = 0.0
        descriptors.append(
            candidate_interaction_features(
                features,
                record.normalized_l2,
                record.candidate_indices[0],
                record.candidate_indices,
                task_value=task_value,
            )
        )
    return torch.stack(descriptors)


def pairwise_candidate_loss(
    scores: torch.Tensor,
    coordinate_nll: torch.Tensor,
    *,
    tie_epsilon: float,
) -> torch.Tensor:
    if scores.shape != coordinate_nll.shape or scores.ndim != 2:
        raise ValueError("scores and coordinate NLL must be matching matrices")
    margins = []
    for left in range(scores.shape[1]):
        for right in range(left + 1, scores.shape[1]):
            nll_delta = coordinate_nll[:, right] - coordinate_nll[:, left]
            valid = nll_delta.abs() > tie_epsilon
            if bool(valid.any()):
                direction = torch.sign(nll_delta[valid])
                margin = (scores[valid, left] - scores[valid, right]) * direction
                margins.append(margin)
    if not margins:
        raise ValueError("all candidate coordinate NLL values are tied")
    return F.softplus(-torch.cat(margins)).mean()


def one_record_metrics(
    scores: torch.Tensor,
    record: JointRecord,
    *,
    tie_epsilon: float,
) -> dict[str, float]:
    predicted = int(torch.argmax(scores).item())
    oracle = int(torch.argmin(record.coordinate_nll).item())
    baseline_nll = float(record.coordinate_nll[0])
    selected_nll = float(record.coordinate_nll[predicted])
    oracle_nll = float(record.coordinate_nll[oracle])
    correct = 0.0
    pair_count = 0
    for left in range(scores.numel()):
        for right in range(left + 1, scores.numel()):
            nll_delta = float(
                record.coordinate_nll[right] - record.coordinate_nll[left]
            )
            if abs(nll_delta) <= tie_epsilon:
                continue
            score_delta = float(scores[left] - scores[right])
            pair_count += 1
            if abs(score_delta) <= 1e-12:
                correct += 0.5
            elif nll_delta * score_delta > 0:
                correct += 1.0
    selected = set(record.candidate_indices[predicted])
    baseline = set(record.candidate_indices[0])
    centered_scores = scores - scores[0]
    return {
        "pairwise_accuracy": correct / pair_count if pair_count else 0.5,
        "candidate_spearman": pearson(ranks(scores), ranks(-record.coordinate_nll)),
        "selected_delta_nll_vs_l2": selected_nll - baseline_nll,
        "oracle_delta_nll_vs_l2": oracle_nll - baseline_nll,
        "top1_regret": selected_nll - oracle_nll,
        "oracle_hit": float(predicted == oracle),
        "beats_l2": float(selected_nll < baseline_nll - tie_epsilon),
        "no_worse_than_l2": float(selected_nll <= baseline_nll + tie_epsilon),
        "l2_selection_overlap": len(selected & baseline) / len(baseline),
        "selected_swap_count": float(record.candidate_swap_counts[predicted]),
        "selected_candidate_index": float(predicted),
        "correction_rms": float(centered_scores.square().mean().sqrt()),
    }


def evaluate_scores(
    scores: torch.Tensor,
    records: list[JointRecord],
    *,
    tie_epsilon: float,
) -> dict[str, Any]:
    metrics = []
    by_task: dict[str, list[dict[str, float]]] = {}
    choices: Counter[str] = Counter()
    nll = torch.stack([record.coordinate_nll for record in records])
    loss = float(
        pairwise_candidate_loss(scores, nll, tie_epsilon=tie_epsilon)
    )
    for index, record in enumerate(records):
        one = one_record_metrics(
            scores[index], record, tie_epsilon=tie_epsilon
        )
        metrics.append(one)
        by_task.setdefault(record.task, []).append(one)
        choices[record.candidate_ids[int(one["selected_candidate_index"])]] += 1
    return {
        "ranking_loss": loss,
        **aggregate(metrics),
        "selected_candidate_counts": dict(sorted(choices.items())),
        "by_task": {
            task: aggregate(items) for task, items in sorted(by_task.items())
        },
    }


def evaluate_head(
    head: CandidateInteractionHead,
    descriptors: torch.Tensor,
    records: list[JointRecord],
    *,
    tie_epsilon: float,
) -> dict[str, Any]:
    head.eval()
    with torch.inference_mode():
        scores = head(descriptors)
        scores = scores - scores[:, :1]
    return evaluate_scores(scores, records, tie_epsilon=tie_epsilon)


def l2_baseline_scores(records: list[JointRecord]) -> torch.Tensor:
    return torch.stack(
        [
            candidate_set_utilities(
                record.normalized_l2,
                record.candidate_indices,
                scale=len(record.candidate_indices[0]),
            )
            for record in records
        ]
    )


def train_one(
    train_records: list[JointRecord],
    calibration_records: list[JointRecord],
    train_descriptors: torch.Tensor,
    calibration_descriptors: torch.Tensor,
    *,
    penalty: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    tie_epsilon: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    head = CandidateInteractionHead()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_nll = torch.stack(
        [record.coordinate_nll for record in train_records]
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        scores = head(train_descriptors)
        scores = scores - scores[:, :1]
        ranking_loss = pairwise_candidate_loss(
            scores, train_nll, tie_epsilon=tie_epsilon
        )
        correction_penalty = scores[:, 1:].square().mean()
        objective = ranking_loss + penalty * correction_penalty
        objective.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            metrics = evaluate_head(
                head,
                calibration_descriptors,
                calibration_records,
                tie_epsilon=tie_epsilon,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_ranking_loss": float(ranking_loss.detach()),
                    "train_correction_penalty": float(
                        correction_penalty.detach()
                    ),
                    "calibration_top1_regret": metrics["top1_regret"],
                    "calibration_pairwise_accuracy": metrics[
                        "pairwise_accuracy"
                    ],
                }
            )
            key = (
                metrics["top1_regret"],
                -metrics["pairwise_accuracy"],
                -metrics["l2_selection_overlap"],
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
                best_metrics = metrics
                best_state = {
                    name: value.detach().clone()
                    for name, value in head.state_dict().items()
                }
    if best_state is None or best_metrics is None:
        raise RuntimeError("no candidate interaction checkpoint")
    head.load_state_dict(best_state)
    return best_state, {
        "penalty": penalty,
        "best_epoch": best_epoch,
        "train_at_best": evaluate_head(
            head, train_descriptors, train_records, tie_epsilon=tie_epsilon
        ),
        "calibration_at_best": best_metrics,
        "history": history,
    }


def gate(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    task_pairwise = all(
        values["pairwise_accuracy"] >= 0.50
        for values in metrics["by_task"].values()
    )
    task_delta = all(
        values["selected_delta_nll_vs_l2"] <= 0.0
        for values in metrics["by_task"].values()
    )
    criteria = {
        "overall_pairwise_accuracy_at_least_0p52": (
            metrics["pairwise_accuracy"] >= 0.52
        ),
        "each_task_pairwise_accuracy_at_least_0p50": task_pairwise,
        "overall_mean_coordinate_nll_better_than_l2": (
            metrics["selected_delta_nll_vs_l2"] < 0.0
        ),
        "each_task_mean_coordinate_nll_no_worse_than_l2": task_delta,
        "top1_regret_reduced_by_at_least_5_percent": (
            metrics["top1_regret"] <= 0.95 * baseline["top1_regret"]
        ),
        "no_worse_than_l2_rate_at_least_0p60": (
            metrics["no_worse_than_l2"] >= 0.60
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


def variant_label(feature_name: str, penalty: float) -> str:
    return f"{feature_name}_lambda_{f'{penalty:g}'.replace('.', 'p')}"


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
        default=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
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

    records, policy = load_joint_records(args.features, args.teacher)
    train_records, calibration_records, train_ids, calibration_ids = (
        deterministic_split(records, args.train_images, args.seed)
    )
    variants = {
        "full_task": ((0, 1, 2, 3, 4), True),
        "no_text_task": ((0, 1, 3, 4), True),
        "full_no_task": ((0, 1, 2, 3, 4), False),
        "no_text_no_task": ((0, 1, 3, 4), False),
    }
    baseline = {
        "train": evaluate_scores(
            l2_baseline_scores(train_records),
            train_records,
            tie_epsilon=args.tie_epsilon,
        ),
        "calibration": evaluate_scores(
            l2_baseline_scores(calibration_records),
            calibration_records,
            tie_epsilon=args.tie_epsilon,
        ),
    }
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    results: dict[str, Any] = {}
    artifact_variants: dict[str, dict[str, Any]] = {}
    for feature_name, (feature_mask, task_conditioned) in variants.items():
        train_descriptors = descriptor_batch(
            train_records,
            feature_mask,
            task_conditioned=task_conditioned,
        )
        calibration_descriptors = descriptor_batch(
            calibration_records,
            feature_mask,
            task_conditioned=task_conditioned,
        )
        for penalty in args.penalties:
            name = variant_label(feature_name, penalty)
            state, result = train_one(
                train_records,
                calibration_records,
                train_descriptors,
                calibration_descriptors,
                penalty=penalty,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=args.seed,
                tie_epsilon=args.tie_epsilon,
            )
            checkpoints[name] = state
            artifact_variants[name] = {
                "feature_mask": list(feature_mask),
                "task_conditioned": task_conditioned,
            }
            results[name] = result
            metrics = result["calibration_at_best"]
            print(
                f"RESULT {name} epoch={result['best_epoch']} "
                f"pair={metrics['pairwise_accuracy']:.4f} "
                f"regret={metrics['top1_regret']:.6f} "
                f"delta={metrics['selected_delta_nll_vs_l2']:+.6f} "
                f"safe={metrics['no_worse_than_l2']:.4f}",
                flush=True,
            )

    gates = {
        name: gate(result["calibration_at_best"], baseline["calibration"])
        for name, result in results.items()
    }
    passing = [name for name, one_gate in gates.items() if one_gate["passed"]]
    pool = passing or list(results)
    selected = min(
        pool,
        key=lambda name: (
            results[name]["calibration_at_best"]["top1_regret"],
            results[name]["calibration_at_best"]["selected_delta_nll_vs_l2"],
            -results[name]["calibration_at_best"]["pairwise_accuracy"],
            results[name]["penalty"],
            name,
        ),
    )
    split_payload = json.dumps(
        {"train": train_ids, "calibration": calibration_ids}, sort_keys=True
    ).encode()
    split_sha256 = hashlib.sha256(split_payload).hexdigest()
    artifact = {
        "format": "candidate-interaction-selector-v1",
        "selected_variant": selected,
        "selected_state_dict": checkpoints[selected],
        "variant_state_dicts": checkpoints,
        "variants": artifact_variants,
        "input_dim": 32,
        "hidden": 16,
        "split_sha256": split_sha256,
        "seed": args.seed,
        "retention": records[0].retention,
        "candidate_config": policy,
        "development_gate_passed": gates[selected]["passed"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        "format": "candidate-interaction-selector-training-summary-v1",
        "status": (
            "DEVELOPMENT_GATE_PASS"
            if gates[selected]["passed"]
            else "DEVELOPMENT_GATE_STOP"
        ),
        "features": str(args.features.resolve()),
        "teacher": str(args.teacher.resolve()),
        "records": len(records),
        "task_counts": dict(
            sorted(Counter(record.task for record in records).items())
        ),
        "train_images": len(train_ids),
        "calibration_images": len(calibration_ids),
        "train_image_ids": train_ids,
        "calibration_image_ids": calibration_ids,
        "split_sha256": split_sha256,
        "retention": records[0].retention,
        "candidate_config": policy,
        "descriptor": {
            "dimension": 32,
            "channels": (
                "delta(6), absolute_delta(6), added_times_removed(6), "
                "global_mean(6), global_std(6), swap_fraction, task_value"
            ),
        },
        "head_cost": candidate_head_cost(),
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "tie_epsilon": args.tie_epsilon,
            "penalties": args.penalties,
        },
        "variant_protocols": artifact_variants,
        "l2_baseline": baseline,
        "results": results,
        "development_gates": gates,
        "selected_variant": selected,
        "selected_development_gate": gates[selected],
        "selection_rule": (
            "minimum calibration top-1 coordinate-NLL regret among variants "
            "passing every pre-registered gate; otherwise diagnostic best only"
        ),
        "real_output_gate_preregistered": {
            "reference": "L2@87.5 on the same 32-image development calibration split",
            "per_primary_metric_max_drop": 0.02,
            "requires_at_least_one_primary_metric_improvement": True,
            "requires_both_tasks_noninferior": True,
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
