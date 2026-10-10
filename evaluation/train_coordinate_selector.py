#!/usr/bin/env python3
"""Train and calibrate the tiny coordinate-aware token selector head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evaluation.learned_selector import CoordinateSelectorHead, head_cost, selector_distillation_loss
from evaluation.metrics import load_jsonl
from evaluation.shallow_block_pruning import regional_quota_indices


@dataclass(frozen=True)
class Record:
    sample_id: str
    image_id: str
    task: str
    grid_h: int
    grid_w: int
    features: torch.Tensor
    saliency: torch.Tensor
    normalized_l2: torch.Tensor | None = None


def image_id(sample_id: str) -> str:
    return sample_id.rsplit("-", 1)[0]


def load_records(feature_path: Path, saliency_path: Path) -> list[Record]:
    features = {row["sample_id"]: row for row in load_jsonl(feature_path)}
    teachers = {row["sample_id"]: row for row in load_jsonl(saliency_path)}
    if features.keys() != teachers.keys():
        missing_features = sorted(teachers.keys() - features.keys())
        missing_teachers = sorted(features.keys() - teachers.keys())
        raise ValueError(
            f"feature/teacher IDs differ: missing_features={missing_features[:3]} "
            f"missing_teachers={missing_teachers[:3]}"
        )
    records: list[Record] = []
    for sample_id in sorted(features):
        feature = features[sample_id]
        teacher = teachers[sample_id]
        if feature["task"] != teacher["task"] or feature["grid_hw"] != teacher["grid_hw"]:
            raise ValueError(f"metadata mismatch for {sample_id}")
        feature_tensor = torch.tensor(feature["selector_features"], dtype=torch.float32)
        saliency_tensor = torch.tensor(teacher["group_saliency"], dtype=torch.float32)
        grid_h, grid_w = map(int, feature["grid_hw"])
        if feature_tensor.shape != (grid_h * grid_w, 5):
            raise ValueError(f"bad feature shape for {sample_id}")
        if saliency_tensor.shape != (grid_h * grid_w,):
            raise ValueError(f"bad saliency shape for {sample_id}")
        if not bool(torch.isfinite(feature_tensor).all() and torch.isfinite(saliency_tensor).all()):
            raise ValueError(f"non-finite record {sample_id}")
        records.append(
            Record(
                sample_id=sample_id,
                image_id=image_id(sample_id),
                task=str(feature["task"]),
                grid_h=grid_h,
                grid_w=grid_w,
                features=feature_tensor,
                saliency=saliency_tensor,
                normalized_l2=(
                    torch.tensor(feature["normalized_l2"], dtype=torch.float32)
                    if "normalized_l2" in feature
                    else None
                ),
            )
        )
    return records


def deterministic_split(
    records: list[Record], train_images: int, seed: int
) -> tuple[list[Record], list[Record], list[str], list[str]]:
    images = sorted({record.image_id for record in records})
    if not (0 < train_images < len(images)):
        raise ValueError("train_images must leave nonempty train and calibration sets")
    ranked = sorted(
        images,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    train_ids = sorted(ranked[:train_images])
    calibration_ids = sorted(ranked[train_images:])
    train_set = set(train_ids)
    train = [record for record in records if record.image_id in train_set]
    calibration = [record for record in records if record.image_id not in train_set]
    return train, calibration, train_ids, calibration_ids


def masked_features(record: Record, feature_mask: tuple[int, ...]) -> torch.Tensor:
    result = record.features.clone()
    disabled = set(range(result.shape[1])) - set(feature_mask)
    if disabled:
        result[:, sorted(disabled)] = 0
    return result


def ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    result = torch.empty_like(values, dtype=torch.float32)
    result[order] = torch.arange(values.numel(), dtype=torch.float32)
    return result


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum() / denominator.clamp_min(1e-12))


def record_metrics(logits: torch.Tensor, record: Record, retention: float) -> dict[str, float]:
    keep = math.ceil(logits.numel() * retention)
    predicted_top = set(torch.argsort(logits, descending=True, stable=True)[:keep].tolist())
    teacher_top = set(
        torch.argsort(record.saliency, descending=True, stable=True)[:keep].tolist()
    )
    selected = regional_quota_indices(
        logits, record.grid_h, record.grid_w, retention, region_grid=3
    )
    mass_recall = float(record.saliency.index_select(0, selected).sum() / record.saliency.sum())
    return {
        "spearman": pearson(ranks(logits), ranks(record.saliency)),
        "topk_overlap": len(predicted_top & teacher_top) / keep,
        "saliency_mass_recall": mass_recall,
    }


def evaluate(
    head: CoordinateSelectorHead,
    records: list[Record],
    feature_mask: tuple[int, ...],
    region_weight: float,
    retention: float,
) -> dict[str, Any]:
    losses: list[float] = []
    metrics: list[dict[str, float]] = []
    task_metrics: dict[str, list[dict[str, float]]] = {}
    head.eval()
    with torch.inference_mode():
        for record in records:
            logits = head(masked_features(record, feature_mask))
            losses.append(
                float(
                    selector_distillation_loss(
                        logits,
                        record.saliency,
                        record.grid_h,
                        record.grid_w,
                        region_weight=region_weight,
                    )
                )
            )
            one = record_metrics(logits, record, retention)
            metrics.append(one)
            task_metrics.setdefault(record.task, []).append(one)

    def average(items: list[dict[str, float]]) -> dict[str, float]:
        return {key: sum(item[key] for item in items) / len(items) for key in items[0]}

    return {
        "loss": sum(losses) / len(losses),
        **average(metrics),
        "by_task": {task: average(items) for task, items in sorted(task_metrics.items())},
    }


def train_variant(
    name: str,
    train: list[Record],
    calibration: list[Record],
    *,
    feature_mask: tuple[int, ...],
    region_weight: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    retention: float,
    evaluation_region_weight: float = 0.25,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(seed)
    random.seed(seed)
    head = CoordinateSelectorHead()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_calibration: dict[str, Any] | None = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = list(range(len(train)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros(())
        for index in order:
            record = train[index]
            logits = head(masked_features(record, feature_mask))
            loss_sum = loss_sum + selector_distillation_loss(
                logits,
                record.saliency,
                record.grid_h,
                record.grid_w,
                region_weight=region_weight,
            )
        train_loss = loss_sum / len(train)
        train_loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            calibration_metrics = evaluate(
                head, calibration, feature_mask, evaluation_region_weight, retention
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss.detach()),
                    "calibration_loss": calibration_metrics["loss"],
                }
            )
            if best_calibration is None or calibration_metrics["loss"] < best_calibration["loss"]:
                best_epoch = epoch
                best_calibration = calibration_metrics
                best_state = {
                    key: value.detach().clone() for key, value in head.state_dict().items()
                }
    if best_state is None or best_calibration is None:
        raise RuntimeError("training produced no checkpoint")
    head.load_state_dict(best_state)
    return best_state, {
        "name": name,
        "feature_mask": list(feature_mask),
        "training_region_weight": region_weight,
        "evaluation_region_weight": evaluation_region_weight,
        "best_epoch": best_epoch,
        "train_at_best": evaluate(
            head, train, feature_mask, evaluation_region_weight, retention
        ),
        "calibration_at_best": best_calibration,
        "history": history,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--saliency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--train-images", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20270825)
    parser.add_argument("--retention", type=float, default=0.75)
    args = parser.parse_args()
    records = load_records(args.features, args.saliency)
    train, calibration, train_ids, calibration_ids = deterministic_split(
        records, args.train_images, args.seed
    )
    variants = {
        "full": ((0, 1, 2, 3, 4), 0.25),
        "no_text": ((0, 1, 3, 4), 0.25),
        "no_coordinates": ((0, 1, 2), 0.25),
        "no_region_loss": ((0, 1, 2, 3, 4), 0.0),
    }
    checkpoints: dict[str, dict[str, torch.Tensor]] = {}
    results: dict[str, Any] = {}
    for offset, (name, (feature_mask, region_weight)) in enumerate(variants.items()):
        state, result = train_variant(
            name,
            train,
            calibration,
            feature_mask=feature_mask,
            region_weight=region_weight,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed + offset,
            retention=args.retention,
        )
        checkpoints[name] = state
        results[name] = result
        metric = result["calibration_at_best"]
        print(
            f"RESULT variant={name} epoch={result['best_epoch']} loss={metric['loss']:.6f} "
            f"rho={metric['spearman']:.4f} overlap={metric['topk_overlap']:.4f} "
            f"mass={metric['saliency_mass_recall']:.4f}",
            flush=True,
        )
    selected = min(results, key=lambda name: results[name]["calibration_at_best"]["loss"])
    split_payload = json.dumps(
        {"train": train_ids, "calibration": calibration_ids}, sort_keys=True
    ).encode()
    artifact = {
        "format": "coordinate-selector-v1",
        "selected_variant": selected,
        "selected_state_dict": checkpoints[selected],
        "variant_state_dicts": checkpoints,
        "variant_feature_masks": {
            name: list(feature_mask) for name, (feature_mask, _) in variants.items()
        },
        "feature_channels": ["residual", "energy", "text_relevance", "x", "y"],
        "split_sha256": hashlib.sha256(split_payload).hexdigest(),
        "seed": args.seed,
        "retention": args.retention,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    summary = {
        "format": "coordinate-selector-training-summary-v1",
        "features": str(args.features.resolve()),
        "features_sha256": file_sha256(args.features),
        "saliency": str(args.saliency.resolve()),
        "saliency_sha256": file_sha256(args.saliency),
        "train_images": len(train_ids),
        "calibration_images": len(calibration_ids),
        "train_records": len(train),
        "calibration_records": len(calibration),
        "train_image_ids": train_ids,
        "calibration_image_ids": calibration_ids,
        "split_sha256": artifact["split_sha256"],
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "retention": args.retention,
        },
        "head_cost": head_cost(),
        "variants": results,
        "selected_variant": selected,
        "selection_rule": "minimum common calibration KL-plus-region distillation loss",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"SELECTED {selected} artifact={args.output} summary={args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
