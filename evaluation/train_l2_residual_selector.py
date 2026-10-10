#!/usr/bin/env python3
"""Train a zero-initialized correction on top of the robust L2 selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.learned_selector import L2ResidualSelectorHead, head_cost, selector_distillation_loss
from evaluation.shallow_block_pruning import regional_quota_indices
from evaluation.train_coordinate_selector import (
    Record,
    deterministic_split,
    file_sha256,
    load_records,
    masked_features,
    record_metrics,
)


def require_l2(records: list[Record]) -> None:
    for record in records:
        if record.normalized_l2 is None or record.normalized_l2.shape != (
            record.grid_h * record.grid_w,
        ):
            raise ValueError(f"missing or malformed normalized L2 for {record.sample_id}")


def evaluate(
    head: L2ResidualSelectorHead | None,
    records: list[Record],
    retention: float,
) -> dict[str, Any]:
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    correction_rms: list[float] = []
    l2_selection_overlap: list[float] = []
    by_task: dict[str, list[dict[str, float]]] = {}
    if head is not None:
        head.eval()
    with torch.inference_mode():
        for record in records:
            assert record.normalized_l2 is not None
            features = masked_features(record, (0, 1, 2, 3, 4))
            l2 = record.normalized_l2
            if head is None:
                logits = l2
                correction = torch.zeros_like(l2)
            else:
                correction = head.correction(features)
                logits = l2 + correction
            losses.append(
                float(
                    selector_distillation_loss(
                        logits, record.saliency, record.grid_h, record.grid_w
                    )
                )
            )
            one = record_metrics(logits, record, retention)
            metrics.append(one)
            by_task.setdefault(record.task, []).append(one)
            correction_rms.append(float(correction.square().mean().sqrt()))
            selected = set(
                regional_quota_indices(
                    logits, record.grid_h, record.grid_w, retention, region_grid=3
                ).tolist()
            )
            selected_l2 = set(
                regional_quota_indices(
                    l2, record.grid_h, record.grid_w, retention, region_grid=3
                ).tolist()
            )
            l2_selection_overlap.append(len(selected & selected_l2) / len(selected_l2))

    def average(items: list[dict[str, float]]) -> dict[str, float]:
        return {key: sum(item[key] for item in items) / len(items) for key in items[0]}

    return {
        "loss": sum(losses) / len(losses),
        **average(metrics),
        "correction_rms": sum(correction_rms) / len(correction_rms),
        "l2_selection_overlap": sum(l2_selection_overlap) / len(l2_selection_overlap),
        "by_task": {task: average(items) for task, items in sorted(by_task.items())},
    }


def label(penalty: float) -> str:
    return "lambda_" + f"{penalty:g}".replace(".", "p")


def train_one(
    train: list[Record],
    calibration: list[Record],
    *,
    penalty: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    retention: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    head = L2ResidualSelectorHead()
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(train)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        distillation_sum = torch.zeros(())
        correction_penalty_sum = torch.zeros(())
        for index in order:
            record = train[index]
            assert record.normalized_l2 is not None
            features = masked_features(record, (0, 1, 2, 3, 4))
            correction = head.correction(features)
            logits = record.normalized_l2 + correction
            distillation_sum = distillation_sum + selector_distillation_loss(
                logits, record.saliency, record.grid_h, record.grid_w
            )
            correction_penalty_sum = correction_penalty_sum + correction.square().mean()
        distillation = distillation_sum / len(train)
        correction_penalty = correction_penalty_sum / len(train)
        objective = distillation + penalty * correction_penalty
        objective.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            metrics = evaluate(head, calibration, retention)
            history.append(
                {
                    "epoch": epoch,
                    "train_distillation": float(distillation.detach()),
                    "train_correction_penalty": float(correction_penalty.detach()),
                    "calibration_loss": metrics["loss"],
                }
            )
            if best_metrics is None or metrics["loss"] < best_metrics["loss"]:
                best_epoch = epoch
                best_metrics = metrics
                best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
    if best_state is None or best_metrics is None:
        raise RuntimeError("no residual selector checkpoint")
    head.load_state_dict(best_state)
    return best_state, {
        "penalty": penalty,
        "best_epoch": best_epoch,
        "train_at_best": evaluate(head, train, retention),
        "calibration_at_best": best_metrics,
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--saliency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--penalties", type=float, nargs="+", default=(0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0))
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20270825)
    parser.add_argument("--retention", type=float, default=0.75)
    args = parser.parse_args()
    if any(value < 0 for value in args.penalties):
        parser.error("penalties must be nonnegative")
    records = load_records(args.features, args.saliency)
    require_l2(records)
    train, calibration, train_ids, calibration_ids = deterministic_split(
        records, args.train_images, args.seed
    )
    baseline = {
        "train": evaluate(None, train, args.retention),
        "calibration": evaluate(None, calibration, args.retention),
    }
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    results: dict[str, Any] = {}
    for penalty in args.penalties:
        name = label(penalty)
        state, result = train_one(
            train,
            calibration,
            penalty=penalty,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            retention=args.retention,
        )
        checkpoints[name] = state
        results[name] = result
        metric = result["calibration_at_best"]
        print(
            f"RESULT {name} epoch={result['best_epoch']} loss={metric['loss']:.6f} "
            f"rho={metric['spearman']:.4f} overlap={metric['topk_overlap']:.4f} "
            f"mass={metric['saliency_mass_recall']:.4f} l2_overlap={metric['l2_selection_overlap']:.4f}",
            flush=True,
        )
    safe_variants = [
        name for name, result in results.items()
        if result["calibration_at_best"]["l2_selection_overlap"] >= 0.95
    ]
    if safe_variants:
        selected = max(
            safe_variants,
            key=lambda name: (
                results[name]["calibration_at_best"]["saliency_mass_recall"],
                results[name]["penalty"],
            ),
        )
    else:
        selected = max(
            results,
            key=lambda name: results[name]["calibration_at_best"]["l2_selection_overlap"],
        )
    split_payload = json.dumps(
        {"train": train_ids, "calibration": calibration_ids}, sort_keys=True
    ).encode()
    artifact = {
        "format": "l2-residual-selector-v1",
        "selected_variant": selected,
        "selected_state_dict": checkpoints[selected],
        "variant_state_dicts": checkpoints,
        "split_sha256": hashlib.sha256(split_payload).hexdigest(),
        "seed": args.seed,
        "retention": args.retention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        "format": "l2-residual-selector-training-summary-v1",
        "features": str(args.features.resolve()),
        "features_sha256": file_sha256(args.features),
        "saliency": str(args.saliency.resolve()),
        "saliency_sha256": file_sha256(args.saliency),
        "train_images": len(train_ids),
        "calibration_images": len(calibration_ids),
        "train_image_ids": train_ids,
        "calibration_image_ids": calibration_ids,
        "split_sha256": artifact["split_sha256"],
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "retention": args.retention,
            "penalties": args.penalties,
        },
        "head_cost": head_cost(),
        "l2_baseline": baseline,
        "variants": results,
        "selected_variant": selected,
        "selection_rule": (
            "maximum calibration saliency-mass recall among variants with at least "
            "95% regional-selection overlap versus L2"
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"SELECTED {selected} artifact={args.output} summary={args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
