#!/usr/bin/env python3
"""Deterministic trainer for the B1 prompt-projection distillation.

Trains explicit ``Wt[128,2560]`` / ``Wv[128,1024]`` against the cached dense
teacher target and exports a float32 npz that the frozen B1 runner can load
with ``evaluation.run_hf_hardware_regular_b1.load_projection``.

Discipline
----------
* Deterministic: the sample split is a SHA256 of
  ``(salt, seed, source_image_sha256)``; parameter init, minibatch order and
  torch/numpy seeds all derive from the configured seed.
* The split is grouped by the source image SHA256, never by ``sample_id``
  independently: every prompt of one image lands in exactly one split and a
  missing image hash is a hard error.
* The main loss is the listwise ``KL(teacher_soft || softmax(student/T))``
  (optionally scaled by ``T**2``); the score MSE term is added only when
  ``loss.score_mse_weight > 0`` in the config.
* Model selection uses the **validation** split only.  The test split is
  evaluated exactly once, after the best epoch has been frozen; it never
  influences epochs, learning rate or any other choice.
* Every split is reported with the student on the correct prompts, an
  ``l2_baseline`` and a deterministic cyclic ``student_shuffled_prompt``
  control, plus the correct-minus-shuffled deltas.  A split with fewer than two
  samples marks the shuffled control unavailable instead of fabricating it.
* The formal method must beat L2 on val and the correct prompts must beat the
  shuffled control; no arbitrary numeric margin is fixed a priori.
* Export refuses to run with an untrained/empty state: ``trained=True`` and a
  completed best epoch are recorded in the npz metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from evaluation.metrics import load_jsonl
from evaluation.prompt_projection_distillation import (
    DEFAULT_CONFIG_FILENAME,
    DistillationContractError,
    FEATURE_DIM,
    PRIMARY_PROMOTION_METRICS,
    PROMPT_DIM,
    SCORE_DIM,
    SHUFFLED_UNAVAILABLE_REASON,
    config_sha256,
    cyclic_shuffled_prompt_map,
    distillation_loss,
    hardware_real_group_metrics,
    l2_baseline_scores,
    load_distillation_config,
    load_feature_cache,
    metric_bundle,
    promotion_gate,
    rank_metric_differences,
    save_projection_npz,
    split_group_summary,
    split_sample_ids,
    student_cosine_scores,
    student_cosine_scores_torch,
    validate_source_image_sha256,
)

DEFAULT_CONFIG_PATH = Path("configs") / DEFAULT_CONFIG_FILENAME


def index_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_index_samples(
    index_path: Path, features_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Load every cached sample referenced by the JSONL index."""
    directory = features_dir or index_path.parent
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in load_jsonl(index_path):
        sample_id = str(record["sample_id"])
        if sample_id in seen:
            raise DistillationContractError(f"duplicate index sample_id {sample_id}")
        seen.add(sample_id)
        npz_path = Path(record["npz"])
        if not npz_path.is_absolute():
            npz_path = directory / npz_path
        cache = load_feature_cache(npz_path)
        image_hash = record.get(
            "source_image_sha256",
            cache["metadata"].get("source_image_sha256"),
        )
        source_image_sha256 = validate_source_image_sha256(image_hash)
        samples.append(
            {
                "sample_id": sample_id,
                "task": str(record.get("task", cache["metadata"].get("task", ""))),
                "source_image_sha256": source_image_sha256,
                "grid_h": int(record["grid_h"]),
                "grid_w": int(record["grid_w"]),
                "group_mean": cache["group_mean"].astype(np.float32),
                "prompt_embeddings": cache["prompt_embeddings"].astype(np.float32),
                "teacher_raw": cache["teacher_scores"].astype(np.float32),
                "teacher_normalized": cache["teacher_scores_normalized"].astype(
                    np.float32
                ),
                "teacher_soft": cache["teacher_soft_target"].astype(np.float32),
                "npz": str(npz_path),
            }
        )
    if not samples:
        raise DistillationContractError(f"index {index_path} contains no samples")
    return samples


def init_projection(seed: int, init_std: float) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    """Deterministic small-random initialisation of the two projection matrices."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    Wt = torch.nn.Parameter(
        torch.randn(SCORE_DIM, PROMPT_DIM, generator=generator) * float(init_std)
    )
    Wv = torch.nn.Parameter(
        torch.randn(SCORE_DIM, FEATURE_DIM, generator=generator) * float(init_std)
    )
    return Wt, Wv


def _mean_metric_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric leaves of a list of identically-shaped metric dicts."""
    if not dicts:
        return {}
    result: dict[str, Any] = {}
    for key in dicts[0]:
        values = [item[key] for item in dicts]
        if isinstance(values[0], dict):
            result[key] = _mean_metric_dicts(values)
        elif isinstance(values[0], (int, float)) and not isinstance(values[0], bool):
            result[key] = float(np.mean([float(value) for value in values]))
        else:
            result[key] = values[0]
    return result


def _tensorize(sample: dict[str, Any], device: str) -> dict[str, torch.Tensor]:
    return {
        "group_mean": torch.as_tensor(sample["group_mean"], device=device),
        "prompt_embeddings": torch.as_tensor(
            sample["prompt_embeddings"], device=device
        ),
        "teacher_soft": torch.as_tensor(sample["teacher_soft"], device=device),
        "teacher_normalized": torch.as_tensor(
            sample["teacher_normalized"], device=device
        ),
    }


def _sample_loss(
    tensors: dict[str, torch.Tensor],
    Wt: torch.Tensor,
    Wv: torch.Tensor,
    *,
    temperature: float,
    listwise_kl_weight: float,
    score_mse_weight: float,
    temperature_squared: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores = student_cosine_scores_torch(
        tensors["group_mean"], tensors["prompt_embeddings"], Wt, Wv
    )
    return distillation_loss(
        scores,
        tensors["teacher_soft"],
        tensors["teacher_normalized"],
        temperature=temperature,
        listwise_kl_weight=listwise_kl_weight,
        score_mse_weight=score_mse_weight,
        temperature_squared=temperature_squared,
    )


@torch.no_grad()
def evaluate_loss(
    samples: list[dict[str, Any]],
    tensors: dict[str, dict[str, torch.Tensor]],
    Wt: torch.Tensor,
    Wv: torch.Tensor,
    loss_config: dict[str, Any],
) -> float:
    if not samples:
        return 0.0
    totals: list[float] = []
    for sample in samples:
        _, parts = _sample_loss(
            tensors[sample["sample_id"]], Wt, Wv, **loss_config
        )
        totals.append(parts["listwise_kl"])
    return float(np.mean(totals))


def evaluate_split_metrics(
    samples: list[dict[str, Any]],
    Wt_np: np.ndarray,
    Wv_np: np.ndarray,
    *,
    lambda_semantic: float,
    budgets: list[int],
) -> dict[str, Any]:
    """Rank/hardware metrics for the student, the L2 baseline and the shuffled control.

    ``student_correct`` uses each sample's own prompt.  ``l2_baseline`` ranks by
    the per-group L2 norm (no training).  ``student_shuffled_prompt`` cyclically
    mismatches prompt embeddings within this split, sorted by ``sample_id``,
    using the same ``Wt``/``Wv``.  A split with fewer than two samples reports
    the shuffled control as unavailable and never fabricates a value.
    """
    correct_rank: list[dict[str, Any]] = []
    correct_hardware: list[dict[str, Any]] = []
    l2_rank: list[dict[str, Any]] = []
    for sample in samples:
        scores = student_cosine_scores(
            sample["group_mean"],
            sample["prompt_embeddings"],
            Wt_np,
            Wv_np,
        )
        correct_rank.append(
            metric_bundle(
                scores,
                sample["teacher_raw"],
                sample["teacher_normalized"],
                sample["teacher_soft"],
                budgets=budgets,
            )
        )
        correct_hardware.append(
            hardware_real_group_metrics(
                sample["group_mean"],
                sample["grid_h"],
                sample["grid_w"],
                text_embeddings=sample["prompt_embeddings"],
                Wt=Wt_np,
                Wv=Wv_np,
                lambda_semantic=lambda_semantic,
                teacher_raw=sample["teacher_raw"],
                teacher_soft=sample["teacher_soft"],
                budgets=budgets,
            )
        )
        l2_scores = l2_baseline_scores(sample["group_mean"])
        l2_rank.append(
            metric_bundle(
                l2_scores,
                sample["teacher_raw"],
                sample["teacher_normalized"],
                sample["teacher_soft"],
                budgets=budgets,
            )
        )

    by_id = {sample["sample_id"]: sample for sample in samples}
    shuffled_map = cyclic_shuffled_prompt_map(list(by_id))
    result: dict[str, Any] = {
        "count": len(samples),
        "student_correct": {
            "count": len(samples),
            "rank": _mean_metric_dicts(correct_rank),
            "hardware_constrained": _mean_metric_dicts(correct_hardware),
        },
        "l2_baseline": {
            "count": len(samples),
            "rank": _mean_metric_dicts(l2_rank),
            "definition": (
                "per-group L2 norm of group_mean; equals "
                "hardware_regular_token_selector.group_l2_saliency on the "
                "[G,4,1024] broadcast"
            ),
        },
    }
    if shuffled_map is None:
        result["student_shuffled_prompt"] = {
            "available": False,
            "count": len(samples),
            "reason": SHUFFLED_UNAVAILABLE_REASON,
        }
        result["correct_minus_shuffled"] = {"available": False}
        return result

    shuffled_rank: list[dict[str, Any]] = []
    for sample in samples:
        donor = by_id[shuffled_map[sample["sample_id"]]]
        scores = student_cosine_scores(
            sample["group_mean"],
            donor["prompt_embeddings"],
            Wt_np,
            Wv_np,
        )
        shuffled_rank.append(
            metric_bundle(
                scores,
                sample["teacher_raw"],
                sample["teacher_normalized"],
                sample["teacher_soft"],
                budgets=budgets,
            )
        )
    shuffled_mean = _mean_metric_dicts(shuffled_rank)
    result["student_shuffled_prompt"] = {
        "available": True,
        "count": len(samples),
        "rank": shuffled_mean,
        "definition": (
            "within-split cyclic prompt mismatch sorted by sample_id; sample i "
            "receives the prompt embeddings of sample i+1 modulo n; same Wt/Wv"
        ),
    }
    result["correct_minus_shuffled"] = {
        "available": True,
        **rank_metric_differences(result["student_correct"]["rank"], shuffled_mean),
    }
    return result


def train_from_cache(
    *,
    config: dict[str, Any],
    index_path: Path,
    output_npz: Path,
    features_dir: Path | None = None,
    report_path: Path | None = None,
    device: str = "cpu",
    seed: int | None = None,
    max_epochs: int | None = None,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full deterministic training and export one npz + JSON report."""
    seed_value = int(config["seed"] if seed is None else seed)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    rng = np.random.default_rng(seed_value)

    samples = load_index_samples(index_path, features_dir)
    by_id = {sample["sample_id"]: sample for sample in samples}
    split = split_sample_ids(
        samples,
        seed=int(config["split"]["seed"]),
        ratios=config["split"]["ratios"],
        salt=str(config["split"]["salt"]),
    )
    image_hashes = {
        sample["sample_id"]: sample["source_image_sha256"] for sample in samples
    }
    group_summary = split_group_summary(split, image_hashes)
    budgets = [int(value) for value in config["budgets"]]
    loss_cfg = config["loss"]
    optim_cfg = config["optim"]
    temperature = float(config["temperature"])
    lambda_semantic = float(config["lambda_semantic"])
    loss_kwargs = {
        "temperature": temperature,
        "listwise_kl_weight": float(loss_cfg["listwise_kl_weight"]),
        "score_mse_weight": float(loss_cfg["score_mse_weight"]),
        "temperature_squared": bool(loss_cfg.get("temperature_squared_scaling", True)),
    }

    train_samples = [by_id[sample_id] for sample_id in split["train"]]
    val_samples = [by_id[sample_id] for sample_id in split["val"]]
    test_samples = [by_id[sample_id] for sample_id in split["test"]]
    if not train_samples:
        raise DistillationContractError("the train split is empty")
    if not val_samples:
        raise DistillationContractError("the val split is empty")

    tensors = {sample["sample_id"]: _tensorize(sample, device) for sample in samples}
    Wt, Wv = init_projection(seed_value, float(optim_cfg.get("init_std", 0.02)))
    Wt = torch.nn.Parameter(Wt.to(device))
    Wv = torch.nn.Parameter(Wv.to(device))
    optimizer_name = str(optim_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise DistillationContractError(f"unsupported optimizer {optimizer_name!r}")
    optimizer = torch.optim.AdamW(
        [Wt, Wv],
        lr=float(optim_cfg["lr"]),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
    grad_clip = float(optim_cfg.get("grad_clip_norm", 0.0))
    batch_size = int(optim_cfg.get("batch_size", 0)) or len(train_samples)
    epochs = int(optim_cfg["epochs"] if max_epochs is None else max_epochs)
    if epochs < 1:
        raise DistillationContractError("epochs must be >= 1")

    epoch_train_losses: list[float] = []
    epoch_val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        order = rng.permutation(len(train_samples))
        batch_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch_ids = [train_samples[int(i)]["sample_id"] for i in order[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            for sample_id in batch_ids:
                loss, _ = _sample_loss(
                    tensors[sample_id], Wt, Wv, **loss_kwargs
                )
                total = total + loss
            total = total / max(len(batch_ids), 1)
            total.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_([Wt, Wv], grad_clip)
            optimizer.step()
            batch_losses.append(float(total.detach()))
        train_loss = float(np.mean(batch_losses))
        val_loss = evaluate_loss(val_samples, tensors, Wt, Wv, loss_kwargs)
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                "Wt": Wt.detach().clone(),
                "Wv": Wv.detach().clone(),
            }

    if best_state is None:
        raise DistillationContractError("training produced no best state; refusing export")
    with torch.no_grad():
        Wt.copy_(best_state["Wt"])
        Wv.copy_(best_state["Wv"])

    Wt_np = Wt.detach().cpu().numpy().astype(np.float32)
    Wv_np = Wv.detach().cpu().numpy().astype(np.float32)
    split_metrics = {
        "train": evaluate_split_metrics(
            train_samples, Wt_np, Wv_np, lambda_semantic=lambda_semantic, budgets=budgets
        ),
        "val": evaluate_split_metrics(
            val_samples, Wt_np, Wv_np, lambda_semantic=lambda_semantic, budgets=budgets
        ),
        "test": evaluate_split_metrics(
            test_samples, Wt_np, Wv_np, lambda_semantic=lambda_semantic, budgets=budgets
        ),
    }
    gate = promotion_gate(split_metrics, val_split="val")
    if gate["status"] == "EVALUATED":
        val_gate = {
            "student_beats_l2": bool(gate["student_beats_l2"]),
            "correct_beats_shuffled": bool(gate["correct_beats_shuffled"]),
        }
    else:
        val_gate = {
            "student_beats_l2": bool(gate.get("student_beats_l2", False)),
            "correct_beats_shuffled": None,
        }
    metadata = {
        "trained": True,
        "training_complete": True,
        "epochs_configured": epochs,
        "best_epoch": best_epoch,
        "best_val_listwise_kl": best_val,
        "temperature": temperature,
        "lambda_semantic": lambda_semantic,
        "seed": seed_value,
        "config_sha256": config_sha256(config_path) if config_path else None,
        "index_sha256": index_file_sha256(index_path),
        "split_seed": int(config["split"]["seed"]),
        "split_ratios": config["split"]["ratios"],
        "split_salt": str(config["split"]["salt"]),
        "split_group_by": "source_image_sha256",
        "split_counts": {name: len(value) for name, value in split.items()},
        "split_group_summary": group_summary,
        "loss": loss_cfg,
        "optim": optim_cfg,
    }
    export = save_projection_npz(output_npz, Wt_np, Wv_np, metadata)

    report = {
        "status": "TRAINED",
        "index": str(index_path),
        "output_npz": str(output_npz),
        "weights_sha256": export["weights_sha256"],
        "sample_count": len(samples),
        "split": split,
        "split_counts": metadata["split_counts"],
        "split_group_by": "source_image_sha256",
        "split_group_summary": group_summary,
        "epochs_run": epochs,
        "best_epoch": best_epoch,
        "best_val_listwise_kl": best_val,
        "epoch_train_losses": epoch_train_losses,
        "epoch_val_losses": epoch_val_losses,
        "loss_decreased": bool(
            len(epoch_train_losses) >= 2 and epoch_train_losses[-1] < epoch_train_losses[0]
        ),
        "temperature": temperature,
        "budgets": budgets,
        "metrics": split_metrics,
        "promotion_gate": gate,
        "val_gate": val_gate,
        "primary_promotion_metrics": list(PRIMARY_PROMOTION_METRICS),
        "test_used_for_selection": False,
        "test_evaluated_once_after_freeze": True,
        "metadata": metadata,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/token_selector/prompt_projection_features/index.jsonl"),
    )
    parser.add_argument("--features-dir", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/token_selector/prompt_projection_WtWv.npz"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/token_selector/2026-09-27-prompt-projection-training.json"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_distillation_config(args.config)
    report = train_from_cache(
        config=config,
        index_path=args.index,
        output_npz=args.out,
        features_dir=args.features_dir,
        report_path=args.report,
        device=args.device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        config_path=args.config,
    )
    print(
        f"TRAINED samples={report['sample_count']} best_epoch={report['best_epoch']} "
        f"kl={report['best_val_listwise_kl']:.6f} hash={report['weights_sha256'][:16]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
