#!/usr/bin/env python3
"""Train a prompt-mean VSR relation student from the B1 margin teacher cache."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from evaluation.prompt_projection_distillation import (
    distillation_loss,
    l2_baseline_scores,
    metric_bundle,
    save_projection_npz,
    split_sample_ids,
)
from evaluation.train_prompt_projection import init_projection, load_index_samples

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/2026-10-10-vsr-prompt-mean-relation-student.preregistered.json"
DEFAULT_INDEX = ROOT / "artifacts/token_selector/2026-10-10-vsr-b1-margin-teacher/student_index.jsonl"
DEFAULT_FEATURES = DEFAULT_INDEX.parent
DEFAULT_OUTPUT = ROOT / "artifacts/token_selector/2026-10-10-vsr-prompt-mean-relation-student"
DEFAULT_REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-prompt-mean-relation-student.json"


class RelationStudentError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pins(config: Mapping[str, Any]) -> None:
    for name, pin in config.get("pins", {}).items():
        path = ROOT / str(pin["path"])
        if not path.is_file() or sha256_file(path) != str(pin["sha256"]):
            raise RelationStudentError(f"pinned input mismatch: {name}")


def prompt_mean_scores_torch(group_mean: torch.Tensor, prompt: torch.Tensor, Wt: torch.Tensor, Wv: torch.Tensor) -> torch.Tensor:
    visual = F.normalize(group_mean @ Wv.T, dim=-1)
    prompt_rows = F.normalize(prompt @ Wt.T, dim=-1)
    query = F.normalize(prompt_rows.mean(dim=0), dim=-1)
    return visual @ query


def prompt_mean_scores_numpy(group_mean: Any, prompt: Any, Wt: Any, Wv: Any) -> np.ndarray:
    visual = np.asarray(group_mean, dtype=np.float64) @ np.asarray(Wv, dtype=np.float64).T
    prompt_rows = np.asarray(prompt, dtype=np.float64) @ np.asarray(Wt, dtype=np.float64).T
    visual /= np.maximum(np.linalg.norm(visual, axis=1, keepdims=True), 1e-12)
    prompt_rows /= np.maximum(np.linalg.norm(prompt_rows, axis=1, keepdims=True), 1e-12)
    query = prompt_rows.mean(axis=0)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    return (visual @ query).astype(np.float32)


def donor_map(samples: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    ordered = sorted(samples, key=lambda row: str(row["sample_id"]))
    result: dict[str, str] = {}
    for index, sample in enumerate(ordered):
        source_hash = str(sample["source_image_sha256"])
        for offset in range(1, len(ordered)):
            donor = ordered[(index + offset) % len(ordered)]
            if str(donor["source_image_sha256"]) != source_hash:
                result[str(sample["sample_id"])] = str(donor["sample_id"])
                break
        if str(sample["sample_id"]) not in result:
            raise RelationStudentError("cannot construct a cross-image prompt donor")
    return result


def bootstrap_ci(values_by_image: Mapping[str, Sequence[float]], seed: int, reps: int) -> dict[str, float]:
    values = np.asarray([float(np.mean(items)) for _, items in sorted(values_by_image.items())], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for index in range(reps):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    return {"mean": float(values.mean()), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)), "image_count": int(len(values))}


def tensorize(samples: Sequence[Mapping[str, Any]], device: str) -> dict[str, dict[str, torch.Tensor]]:
    return {str(sample["sample_id"]): {
        "group_mean": torch.as_tensor(sample["group_mean"], device=device),
        "prompt": torch.as_tensor(sample["prompt_embeddings"], device=device),
        "teacher_soft": torch.as_tensor(sample["teacher_soft"], device=device),
        "teacher_normalized": torch.as_tensor(sample["teacher_normalized"], device=device),
    } for sample in samples}


def sample_loss(sample_id: str, donor_id: str, tensors: Mapping[str, Mapping[str, torch.Tensor]], Wt: torch.Tensor, Wv: torch.Tensor, cfg: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    item = tensors[sample_id]
    correct = prompt_mean_scores_torch(item["group_mean"], item["prompt"], Wt, Wv)
    wrong = prompt_mean_scores_torch(item["group_mean"], tensors[donor_id]["prompt"], Wt, Wv)
    base, parts = distillation_loss(
        correct, item["teacher_soft"], item["teacher_normalized"],
        temperature=float(cfg["temperature"]), listwise_kl_weight=1.0,
        score_mse_weight=0.0, temperature_squared=True,
    )
    teacher = item["teacher_soft"]
    temp = float(cfg["temperature"])
    ce_correct = -(teacher * F.log_softmax(correct / temp, dim=-1)).sum()
    ce_wrong = -(teacher * F.log_softmax(wrong / temp, dim=-1)).sum()
    scale = float(cfg["contrastive_softplus_scale"])
    contrast = F.softplus((float(cfg["contrastive_margin_nats"]) - (ce_wrong - ce_correct)) / scale) * scale
    total = base + float(cfg["contrastive_weight"]) * contrast
    return total, {"base": float(parts["total"]), "contrast": float(contrast.detach()),
                   "ce_advantage": float((ce_wrong - ce_correct).detach())}


@torch.no_grad()
def validation_objective(samples: Sequence[Mapping[str, Any]], donors: Mapping[str, str], tensors: Mapping[str, Mapping[str, torch.Tensor]], Wt: torch.Tensor, Wv: torch.Tensor, cfg: Mapping[str, Any]) -> float:
    return float(np.mean([float(sample_loss(str(row["sample_id"]), donors[str(row["sample_id"])], tensors, Wt, Wv, cfg)[0]) for row in samples]))


def evaluate(samples: Sequence[Mapping[str, Any]], Wt: np.ndarray, Wv: np.ndarray, *, seed: int, reps: int) -> dict[str, Any]:
    donors = donor_map(samples)
    by_id = {str(row["sample_id"]): row for row in samples}
    correct_metrics, wrong_metrics, l2_metrics = [], [], []
    deltas: dict[str, dict[str, list[float]]] = {
        "spearman": defaultdict(list), "ndcg_full": defaultdict(list), "mass_k96": defaultdict(list)
    }
    for sample in samples:
        sid = str(sample["sample_id"])
        correct = metric_bundle(prompt_mean_scores_numpy(sample["group_mean"], sample["prompt_embeddings"], Wt, Wv),
                                sample["teacher_raw"], sample["teacher_normalized"], sample["teacher_soft"], budgets=[96])
        donor = by_id[donors[sid]]
        wrong = metric_bundle(prompt_mean_scores_numpy(sample["group_mean"], donor["prompt_embeddings"], Wt, Wv),
                              sample["teacher_raw"], sample["teacher_normalized"], sample["teacher_soft"], budgets=[96])
        l2 = metric_bundle(l2_baseline_scores(sample["group_mean"]), sample["teacher_raw"],
                           sample["teacher_normalized"], sample["teacher_soft"], budgets=[96])
        correct_metrics.append(correct); wrong_metrics.append(wrong); l2_metrics.append(l2)
        image = str(sample["source_image_sha256"])
        deltas["spearman"][image].append(float(correct["spearman"] - wrong["spearman"]))
        deltas["ndcg_full"][image].append(float(correct["ndcg_full"] - wrong["ndcg_full"]))
        deltas["mass_k96"][image].append(float(correct["per_budget"]["96"]["saliency_mass_retained"] - wrong["per_budget"]["96"]["saliency_mass_retained"]))
    def mean_metric(items: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {"spearman": float(np.mean([x["spearman"] for x in items])),
                "ndcg_full": float(np.mean([x["ndcg_full"] for x in items])),
                "mass_k96": float(np.mean([x["per_budget"]["96"]["saliency_mass_retained"] for x in items]))}
    correct_mean, wrong_mean, l2_mean = mean_metric(correct_metrics), mean_metric(wrong_metrics), mean_metric(l2_metrics)
    cis = {name: bootstrap_ci(groups, seed + offset, reps) for offset, (name, groups) in enumerate(deltas.items())}
    return {"sample_count": len(samples), "image_count": len({x["source_image_sha256"] for x in samples}),
            "correct": correct_mean, "cross_image_wrong_prompt": wrong_mean, "l2": l2_mean,
            "correct_minus_wrong": {key: correct_mean[key] - wrong_mean[key] for key in correct_mean},
            "correct_minus_l2": {key: correct_mean[key] - l2_mean[key] for key in correct_mean},
            "correct_minus_wrong_image_bootstrap": cis}


def train_seed(seed: int, train: list[dict[str, Any]], val: list[dict[str, Any]], tensors: Mapping[str, Mapping[str, torch.Tensor]], cfg: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    Wt, Wv = init_projection(seed, float(cfg["init_std"]))
    Wt = torch.nn.Parameter(Wt.to(cfg["device"])); Wv = torch.nn.Parameter(Wv.to(cfg["device"]))
    optimizer = torch.optim.AdamW([Wt, Wv], lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    train_donors, val_donors = donor_map(train), donor_map(val)
    best = None; history = []
    for epoch in range(int(cfg["epochs"])):
        order = rng.permutation(len(train)); epoch_values = []
        for start in range(0, len(order), int(cfg["batch_size"])):
            optimizer.zero_grad(set_to_none=True); loss = torch.zeros((), device=cfg["device"])
            batch = order[start:start + int(cfg["batch_size"])]
            for position in batch:
                sid = str(train[int(position)]["sample_id"])
                loss = loss + sample_loss(sid, train_donors[sid], tensors, Wt, Wv, cfg)[0]
            loss = loss / len(batch); loss.backward()
            torch.nn.utils.clip_grad_norm_([Wt, Wv], float(cfg["grad_clip_norm"])); optimizer.step()
            epoch_values.append(float(loss.detach()))
        val_obj = validation_objective(val, val_donors, tensors, Wt, Wv, cfg)
        history.append({"epoch": epoch, "train_objective": float(np.mean(epoch_values)), "val_objective": val_obj})
        if best is None or val_obj < best["val_objective"]:
            best = {"epoch": epoch, "val_objective": val_obj, "Wt": Wt.detach().cpu().numpy().copy(), "Wv": Wv.detach().cpu().numpy().copy()}
    assert best is not None
    metrics = evaluate(val, best["Wt"], best["Wv"], seed=seed, reps=int(cfg["bootstrap_repetitions"]))
    ckpt = out_dir / f"seed-{seed}.npz"
    export = save_projection_npz(ckpt, best["Wt"], best["Wv"], {"trained": True, "architecture": "prompt_mean_vector", "seed": seed, "best_epoch": best["epoch"]})
    return {"seed": seed, "best_epoch": best["epoch"], "best_val_objective": best["val_objective"],
            "checkpoint": str(ckpt), "weights_sha256": export["weights_sha256"], "history": history,
            "val": metrics, "_Wt": best["Wt"], "_Wv": best["Wv"]}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX); parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text(encoding="utf-8")); verify_pins(cfg)
    cfg["device"] = str(cfg.get("device", "cuda:0")); args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_index_samples(args.index, args.features)
    split = split_sample_ids(samples, seed=int(cfg["split_seed"]), ratios=cfg["split_ratios"], salt=str(cfg["split_salt"]))
    by_id = {str(x["sample_id"]): x for x in samples}; train = [by_id[x] for x in split["train"]]; val = [by_id[x] for x in split["val"]]; test = [by_id[x] for x in split["test"]]
    tensors = tensorize(samples, cfg["device"])
    runs = [train_seed(int(seed), train, val, tensors, cfg, args.out_dir) for seed in cfg["seeds"]]
    def key(run: Mapping[str, Any]) -> tuple[float, float, float]:
        ci = run["val"]["correct_minus_wrong_image_bootstrap"]
        return (min(ci["spearman"]["ci_low"], ci["ndcg_full"]["ci_low"], ci["mass_k96"]["ci_low"]),
                run["val"]["correct_minus_l2"]["ndcg_full"], -float(run["best_val_objective"]))
    winner = max(runs, key=key)
    test_metrics = evaluate(test, winner["_Wt"], winner["_Wv"], seed=int(cfg["bootstrap_seed"]), reps=int(cfg["bootstrap_repetitions"]))
    val_ci = winner["val"]["correct_minus_wrong_image_bootstrap"]
    gates = {
        "prompt_sensitivity_ci_lower_positive_all_three": all(val_ci[name]["ci_low"] > 0 for name in ("spearman", "ndcg_full", "mass_k96")),
        "teacher_alignment_beats_l2_spearman_and_ndcg": all(winner["val"]["correct_minus_l2"][name] > 0 for name in ("spearman", "ndcg_full")),
    }
    gates["pass"] = all(gates.values())
    for run in runs: run.pop("_Wt"); run.pop("_Wv")
    report = {"status": "PROMPT_MEAN_RELATION_STUDENT_PASS" if gates["pass"] else "PROMPT_MEAN_RELATION_STUDENT_NOT_ESTABLISHED",
              "scope": "VSR zero-shot train internal grouped train/val/test only; no VSR dev/test use",
              "config": str(args.config), "index": str(args.index), "split_counts": {k: len(v) for k,v in split.items()},
              "runs": runs, "winner_seed": winner["seed"], "winner_checkpoint": winner["checkpoint"],
              "winner_val": winner["val"], "winner_internal_test_once": test_metrics, "gates": gates,
              "official_vsr_dev_used": False, "official_vsr_test_opened": False,
              "hardware_contract": {"prompt_query_shape": [1,128], "prompt_query_int8_bytes": 128, "fpga_dot_mac_per_group": 128, "fixed_future_budget": 96}}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']} winner={winner['seed']} gates={gates}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
