#!/usr/bin/env python3
"""Train the prompt-mean relation student on the variable VSR caption span."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.prompt_projection_distillation import split_sample_ids
from evaluation.train_prompt_projection import load_index_samples
from evaluation.train_vsr_prompt_mean_relation_student import (
    DEFAULT_FEATURES, DEFAULT_INDEX, evaluate, tensorize, train_seed, verify_pins,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/2026-10-10-vsr-caption-relation-student.preregistered.json"
DEFAULT_SPANS = DEFAULT_FEATURES / "caption_token_spans.json"
DEFAULT_OUTPUT = ROOT / "artifacts/token_selector/2026-10-10-vsr-caption-relation-student"
DEFAULT_REPORT = ROOT / "reports/token_selector/2026-10-10-vsr-caption-relation-student.json"


def apply_caption_spans(samples: list[dict[str, Any]], span_path: Path) -> dict[str, float]:
    payload = json.loads(span_path.read_text(encoding="utf-8")); spans = payload["spans"]
    lengths = []
    for sample in samples:
        sample_id = str(sample["sample_id"]); span = spans[sample_id]
        prompt = sample["prompt_embeddings"]
        if int(span["full_token_count"]) != int(prompt.shape[0]):
            raise ValueError(f"prompt token count mismatch: {sample_id}")
        start, end = int(span["start"]), int(span["end_exclusive"])
        sample["prompt_embeddings"] = prompt[start:end].copy(); lengths.append(end - start)
    return {"min": min(lengths), "max": max(lengths), "mean": sum(lengths) / len(lengths)}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX); parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--spans", type=Path, default=DEFAULT_SPANS); parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT); args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); verify_pins(cfg); cfg["device"] = str(cfg.get("device", "cuda:0"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_index_samples(args.index, args.features); span_stats = apply_caption_spans(samples, args.spans)
    split = split_sample_ids(samples, seed=int(cfg["split_seed"]), ratios=cfg["split_ratios"], salt=str(cfg["split_salt"]))
    by_id = {str(x["sample_id"]): x for x in samples}; train = [by_id[x] for x in split["train"]]; val = [by_id[x] for x in split["val"]]; test = [by_id[x] for x in split["test"]]
    tensors = tensorize(samples, cfg["device"])
    runs = [train_seed(int(seed), train, val, tensors, cfg, args.out_dir) for seed in cfg["seeds"]]
    def key(run: Mapping[str, Any]) -> tuple[float, float, float]:
        ci = run["val"]["correct_minus_wrong_image_bootstrap"]
        return (min(ci[name]["ci_low"] for name in ("spearman", "ndcg_full", "mass_k96")),
                run["val"]["correct_minus_l2"]["ndcg_full"], -float(run["best_val_objective"]))
    winner = max(runs, key=key)
    test_metrics = evaluate(test, winner["_Wt"], winner["_Wv"], seed=int(cfg["bootstrap_seed"]), reps=int(cfg["bootstrap_repetitions"]))
    ci = winner["val"]["correct_minus_wrong_image_bootstrap"]
    gates = {"prompt_sensitivity_ci_lower_positive_all_three": all(ci[name]["ci_low"] > 0 for name in ("spearman", "ndcg_full", "mass_k96")),
             "teacher_alignment_beats_l2_spearman_and_ndcg": all(winner["val"]["correct_minus_l2"][name] > 0 for name in ("spearman", "ndcg_full"))}
    gates["pass"] = all(gates.values())
    for run in runs: run.pop("_Wt"); run.pop("_Wv")
    report = {"status": "CAPTION_RELATION_STUDENT_PASS" if gates["pass"] else "CAPTION_RELATION_STUDENT_NOT_ESTABLISHED",
              "scope": "VSR zero-shot train internal grouped split; caption span derived from raw Prompt character offsets; no relation/label input",
              "caption_token_count": span_stats, "split_counts": {k: len(v) for k,v in split.items()}, "runs": runs,
              "winner_seed": winner["seed"], "winner_checkpoint": winner["checkpoint"], "winner_val": winner["val"],
              "winner_internal_test_once": test_metrics, "gates": gates, "official_vsr_dev_used": False, "official_vsr_test_opened": False,
              "hardware_contract": {"query_shape": [1,128], "query_int8_bytes": 128, "fpga_dot_mac_per_group": 128, "fixed_budget": 96}}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']} winner={winner['seed']} gates={gates}"); return 0


if __name__ == "__main__": raise SystemExit(main())
