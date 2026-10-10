#!/usr/bin/env python3
"""Coarse-to-fine 64-token 100-question discrimination report.

Reuses the frozen CV-Bench parser/scorer and the frozen stratified paired
bootstrap. It only aggregates the raw records produced by the two existing
runners; it does not change the algorithm or the parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.build_cv_bench_pruning_report import answer_letter, parse_choice, score
from evaluation.build_l2_retention_curve import csv_rows, paired_accuracy_ci
from evaluation.metrics import load_jsonl

CONFIGS = ("dense", "l2_50", "l2_25", "global256", "global_local")
REFINEMENT_STRATEGY = {
    "dense": "dense",
    "global256": "global256",
    "global_local": "cosine_global_roi",
}
PRUNING_FILE = {"l2_50": "shallow_r0500.jsonl", "l2_25": "shallow_r0250.jsonl"}
DISPLAY = {
    "dense": "原图Dense",
    "l2_50": "layer5 L2+3x3 锚点 retention=0.50",
    "l2_25": "layer5 L2+3x3 锚点 retention=0.25",
    "global256": "低清全图256",
    "global_local": "低清全图256＋任务感知局部384",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_unique(rows: list[dict], expected: set[str], label: str, repeat: int | None = None):
    subset = rows if repeat is None else [row for row in rows if row.get("repeat") == repeat]
    indexed = {str(row["sample_id"]): row for row in subset}
    if len(subset) != len(indexed) or set(indexed) != expected:
        raise ValueError(f"{label}: incomplete, duplicate or foreign records")
    return indexed


def refinement_latency(rows: list[dict], strategy: str) -> dict[str, Any]:
    subset = [row for row in rows if row["strategy"] == strategy]
    core = [row["vision_ms"] + row["prefill_ms"] + (row["decode_ms"] or 0.0) for row in subset]
    e2e = [row["model_end_to_end_ms"] for row in subset]
    return {
        "runs": len(subset),
        "model_core_ms": statistics.fmean(core),
        "model_end_to_end_ms": statistics.fmean(e2e),
        "core_p50": statistics.median(core),
        "e2e_p50": statistics.median(e2e),
    }


def pruning_latency(rows: list[dict]) -> dict[str, Any]:
    core = [row["timed_stage_seconds"] * 1000.0 for row in rows]
    e2e = [(row["preprocess_seconds"] + row["elapsed_seconds"]) * 1000.0 for row in rows]
    return {
        "runs": len(rows),
        "model_core_ms": statistics.fmean(core),
        "model_end_to_end_ms": statistics.fmean(e2e),
        "core_p50": statistics.median(core),
        "e2e_p50": statistics.median(e2e),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--refinement-jsonl", type=Path, required=True)
    parser.add_argument("--pruning-dir", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--per-question-csv", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--harness-json", type=Path, required=True)
    parser.add_argument("--refinement-runner", type=Path, required=True)
    parser.add_argument("--pruning-runner", type=Path, required=True)
    parser.add_argument("--lock-config", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    if any(row["max_new_tokens"] != 64 for row in samples):
        raise ValueError("this report implements the 64-token budget only")
    ids = {str(row["sample_id"]) for row in samples}
    if len(ids) != len(samples):
        raise ValueError("duplicate manifest IDs")

    refinement = load_jsonl(args.refinement_jsonl)
    signatures = {json.dumps(row["run"], sort_keys=True) for row in refinement}
    if len(signatures) != 1:
        raise ValueError("mixed refinement signatures")
    run = refinement[0]["run"]
    if run.get("manifest_sha256") != sha256(args.manifest) or run.get("repeats") != 3:
        raise ValueError("refinement run does not match the frozen 64-token protocol")
    if set(run["strategies"]) != set(REFINEMENT_STRATEGY.values()):
        raise ValueError("refinement strategies mismatch")

    dense_ref = index_unique(
        [row for row in refinement if row["strategy"] == REFINEMENT_STRATEGY["dense"]],
        ids, "refinement dense", repeat=0,
    )
    predictions: dict[str, dict[str, dict]] = {}
    for config in CONFIGS:
        if config in REFINEMENT_STRATEGY:
            predictions[config] = index_unique(
                [row for row in refinement if row["strategy"] == REFINEMENT_STRATEGY[config]],
                ids, config, repeat=0,
            )
        else:
            path = args.pruning_dir / PRUNING_FILE[config]
            predictions[config] = index_unique(load_jsonl(path), ids, config)

    # Cross-runner identity check: pruning retention=1.0 is the same dense path.
    pruning_dense_path = args.pruning_dir / "shallow_r1000.jsonl"
    pruning_dense_identity = None
    if pruning_dense_path.is_file():
        pd = index_unique(load_jsonl(pruning_dense_path), ids, "pruning dense")
        pruning_dense_identity = statistics.fmean(
            pd[str(sample["sample_id"])]["output_text"] == dense_ref[str(sample["sample_id"])]["output_text"]
            for sample in samples
        )

    dense_tokens = {str(sample["sample_id"]): int(dense_ref[str(sample["sample_id"])]["final_visual_tokens"])
                    for sample in samples}

    per_config: dict[str, dict[str, Any]] = {}
    scored_rows: dict[str, list[dict]] = {}
    per_question: list[dict] = []
    for config in CONFIGS:
        rows = []
        for sample in samples:
            sample_id = str(sample["sample_id"])
            record = predictions[config][sample_id]
            truth = answer_letter(str(sample["answer"]))
            answer = parse_choice(str(record["output_text"]), list(sample["choices"]))
            dense_answer = parse_choice(str(dense_ref[sample_id]["output_text"]), list(sample["choices"]))
            if config in REFINEMENT_STRATEGY:
                visual_tokens = int(record["final_visual_tokens"])
                baseline_tokens = dense_tokens[sample_id]
                core_ms = record["vision_ms"] + record["prefill_ms"] + (record["decode_ms"] or 0.0)
                e2e_ms = record["model_end_to_end_ms"]
            else:
                visual_tokens = int(record["llm_visual_tokens"])
                baseline_tokens = int(record["baseline_llm_visual_tokens"])
                core_ms = record["timed_stage_seconds"] * 1000.0
                e2e_ms = (record["preprocess_seconds"] + record["elapsed_seconds"]) * 1000.0
            rows.append({
                "sample_id": sample_id, "config": config, "task": sample["task"],
                "source": sample["source"], "source_dataset": sample["source_dataset"],
                "ground_truth": truth, "pred_answer": answer, "dense_answer": dense_answer,
                "pred_correct": answer == truth, "dense_correct": dense_answer == truth,
                "output_text": record["output_text"], "output_tokens": int(record["output_tokens"]),
                "visual_tokens": visual_tokens, "baseline_visual_tokens": baseline_tokens,
                "token_retention": visual_tokens / baseline_tokens,
                "model_core_ms": core_ms, "model_end_to_end_ms": e2e_ms,
            })
        scored_rows[config] = rows
        metrics = score(rows, "pred_correct")
        if config in REFINEMENT_STRATEGY:
            latency = refinement_latency(refinement, REFINEMENT_STRATEGY[config])
        else:
            latency = pruning_latency(list(predictions[config].values()))
        per_config[config] = {
            "official_combined": metrics["cv_bench_accuracy"],
            "official_2d": metrics["accuracy_2d"],
            "official_3d": metrics["accuracy_3d"],
            "parse_success_rate": metrics["parse_success_rate"],
            "output_cap_hit_count": sum(row["output_tokens"] == 64 for row in rows),
            "actual_merged_visual_tokens_mean": statistics.fmean(row["visual_tokens"] for row in rows),
            "actual_token_retention_mean": statistics.fmean(row["token_retention"] for row in rows),
            "model_core_latency_ms": latency["model_core_ms"],
            "model_end_to_end_ms": latency["model_end_to_end_ms"],
            "model_core_p50_ms": latency["core_p50"],
            "model_end_to_end_p50_ms": latency["e2e_p50"],
            "timed_runs": latency["runs"],
            "mean_output_tokens": statistics.fmean(row["output_tokens"] for row in rows),
            "task_accuracy": metrics["task_accuracy"],
            "source_accuracy": metrics["source_accuracy"],
        }

    dense_score = per_config["dense"]["official_combined"]
    for config in CONFIGS:
        per_config[config]["relative_score_retention_vs_dense"] = (
            per_config[config]["official_combined"] / dense_score
        )

    def paired(config: str, reference: str) -> dict[str, float]:
        ref_correct = {row["sample_id"]: row["pred_correct"] for row in scored_rows[reference]}
        rows = [{**row, "dense_correct": ref_correct[row["sample_id"]]} for row in scored_rows[config]]
        lo, hi = (float(value) for value in paired_accuracy_ci(rows))
        delta = per_config[config]["official_combined"] - per_config[reference]["official_combined"]
        return {"delta": delta, "ci95_low": lo, "ci95_high": hi}

    comparisons = {
        "l2_50_vs_dense": paired("l2_50", "dense"),
        "l2_25_vs_dense": paired("l2_25", "dense"),
        "global256_vs_dense": paired("global256", "dense"),
        "global_local_vs_dense": paired("global_local", "dense"),
        "global_local_vs_global256": paired("global_local", "global256"),
        "global_local_vs_l2_50": paired("global_local", "l2_50"),
    }

    def dominates(left: str, right: str) -> bool:
        a, b = per_config[left], per_config[right]
        return (a["official_combined"] >= b["official_combined"]
                and a["model_end_to_end_ms"] <= b["model_end_to_end_ms"]
                and (a["official_combined"] > b["official_combined"]
                     or a["model_end_to_end_ms"] < b["model_end_to_end_ms"]))

    dominance = {config: sorted(other for other in CONFIGS if other != config and dominates(config, other))
                 for config in CONFIGS}
    dominated_by = {config: sorted(other for other in CONFIGS if other != config and dominates(other, config))
                    for config in CONFIGS}
    frontier = sorted(config for config in CONFIGS if not dominated_by[config])

    payload = {
        "status": "completed",
        "task": "coarse-to-fine-64token-pilot100",
        "decision": "REPORTED_SCORE_AND_LATENCY_NOT_NONINFERIORITY",
        "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest),
                     "n_samples": len(samples), "max_new_tokens": 64},
        "lock_config": {"path": str(args.lock_config), "sha256": sha256(args.lock_config)},
        "runners": {
            "refinement": {"path": str(args.refinement_runner), "sha256": sha256(args.refinement_runner)},
            "pruning": {"path": str(args.pruning_runner), "sha256": sha256(args.pruning_runner)},
        },
        "raw_inputs": {
            "refinement_jsonl": str(args.refinement_jsonl),
            "refinement_records": len(refinement),
            "pruning_dir": str(args.pruning_dir),
            "pruning_records": sum(len(predictions[c]) for c in CONFIGS if c in PRUNING_FILE),
        },
        "protocol": {
            "configs": list(CONFIGS),
            "same_manifest_order": True, "same_prompt": True, "seed": 0,
            "max_new_tokens": 64, "do_sample": False,
            "hf_hub_offline": 1, "transformers_offline": 1,
            "l2": {"prune_after_layer": 5, "anchor_grid": 3, "selector": "l2_anchor",
                   "retentions": [0.5, 0.25]},
            "global_long_edge": 256, "roi_long_edge": 384,
            "reused_16token_results": False,
        },
        "metrics": per_config,
        "relative_score_retention_vs_dense": {
            config: per_config[config]["relative_score_retention_vs_dense"] for config in CONFIGS
        },
        "paired_deltas": comparisons,
        "pareto": {"objective": "max official_combined, min model_end_to_end_ms",
                   "frontier": frontier, "dominates": dominance, "dominated_by": dominated_by},
        "cross_checks": {
            "pruning_dense_matches_refinement_dense_text_rate": pruning_dense_identity,
            "refinement_repeats": 3, "pruning_repeats_per_config": 1,
        },
        "latency_boundary": {
            "model_core_ms": "CUDA-event sum of vision+prefill+decode for both runners",
            "model_end_to_end_ms": ("refinement: image already decoded to generation end, excluding "
                                    "JPEG/text decode and model load; pruning: processor+H2D+inference, "
                                    "excluding JPEG/text decode and model load"),
        },
    }

    stem = args.output_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload["records"] = [
        {k: v for k, v in row.items() if k not in {"dense_answer", "dense_correct"}}
        for config in CONFIGS for row in scored_rows[config]
    ]
    stem.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    summary_rows = []
    for config in CONFIGS:
        item = per_config[config]
        summary_rows.append({
            "status": "completed", "config": config,
            "official_combined": round(item["official_combined"], 6),
            "official_2d": round(item["official_2d"], 6),
            "official_3d": round(item["official_3d"], 6),
            "relative_score_retention_vs_dense": round(item["relative_score_retention_vs_dense"], 6),
            "parse_success_rate": round(item["parse_success_rate"], 6),
            "output_cap_hit_count": item["output_cap_hit_count"],
            "actual_merged_visual_tokens": round(item["actual_merged_visual_tokens_mean"], 4),
            "actual_token_retention": round(item["actual_token_retention_mean"], 6),
            "model_core_latency_ms": round(item["model_core_latency_ms"], 3),
            "model_end_to_end_ms": round(item["model_end_to_end_ms"], 3),
            "n_questions": len(samples),
            "note": "64-token budget; A6000 BF16; single-run accuracy, model timing excludes model load",
        })
    csv_rows(stem.with_suffix(".csv"), summary_rows)
    csv_rows(args.per_question_csv,
             [{k: v for k, v in row.items() if k not in {"dense_answer", "dense_correct"}}
              for config in CONFIGS for row in scored_rows[config]])

    names = DISPLAY
    ci = lambda key: comparisons[key]
    lines = [
        "# 粗到细 64-Token 100 题判别实验", "",
        f"- 状态：`completed`；样本：固定分层 100 题（`{args.manifest}`，max_new_tokens=64）；模型：RoboBrain2.5-4B BF16。",
        "- 仅比较 5 个配置，复用既有运行器与冻结解析器：`dense`、`l2_50`、`l2_25`、`global256`、`global_local`。",
        "- 生成：`do_sample=False`、seed=0、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`；未复用 16-Token 结果。", "",
        "## 官方得分与对齐指标（64 Token，100 题）", "",
        "| 配置 | 官方组合分 | 2D | 3D | 相对Dense保留 | 解析成功率 | 输出上限命中 | 平均合并视觉Token | Token保留率 | 模型核心均值ms | 模型端到端均值ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        item = per_config[config]
        lines.append(
            f"| {names[config]} | {item['official_combined']:.4%} | {item['official_2d']:.4%} | "
            f"{item['official_3d']:.4%} | {item['relative_score_retention_vs_dense']:.4%} | "
            f"{item['parse_success_rate']:.4%} | {item['output_cap_hit_count']} | "
            f"{item['actual_merged_visual_tokens_mean']:.1f} | {item['actual_token_retention_mean']:.4%} | "
            f"{item['model_core_latency_ms']:.3f} | {item['model_end_to_end_ms']:.3f} |"
        )
    lines += ["", "## 成对差值与 paired bootstrap 95%CI", "",
              "分层配对 bootstrap 5000 次，沿用冻结评分权重（ADE20K/COCO 各 0.25，Omni3D 0.5）；"
              "差值为配置减参照，单位百分点。", "",
              "| 比较 | 差值（百分点） | 95% CI（百分点） |", "|---|---:|---:|"]
    for key, item in comparisons.items():
        lines.append(f"| {key} | {item['delta'] * 100:+.4f} | "
                     f"[{item['ci95_low'] * 100:+.4f}, {item['ci95_high'] * 100:+.4f}] |")
    lines += ["", "## 帕累托支配（官方分越高越好，模型端到端越低越好）", "",
              f"- 前沿集合：{', '.join(frontier) if frontier else '（空）'}",
              f"- 支配关系：{json.dumps(dominance, ensure_ascii=False)}",
              f"- 被支配关系：{json.dumps(dominated_by, ensure_ascii=False)}", "",
              "## 边界", "",
              "- 只覆盖 CV-Bench 100 题空间多选问答；不运行完整 2638，不运行 Grounding/Pointing/PCK/IoU，不搜索新算法。",
              "- 模型核心时延为 vision+prefill+decode 的 CUDA 事件和；模型端到端不含模型加载、JPEG 解码与文本解码；未测功耗或板卡。",
              "- 解析失败计为错误；输出上限命中不等于答案被截断；未设置非劣通过线，不宣称统计非劣。",
              "- 未修改原始模型、`app/inference.py`、`api.py`；未删除文件；未 kill 进程；未访问封存数据或 ZCU102。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")

    artifact = {
        "manifest_id": "2026-09-27-coarse-to-fine-64token-pilot100",
        "status": "completed",
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "base_manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest), "n_samples": len(samples)},
        "lock_config": {"path": str(args.lock_config), "sha256": sha256(args.lock_config)},
        "runners": {
            "refinement": {"path": str(args.refinement_runner), "sha256": sha256(args.refinement_runner)},
            "pruning": {"path": str(args.pruning_runner), "sha256": sha256(args.pruning_runner)},
        },
        "raw_inputs": payload["raw_inputs"],
        "comparison": payload["protocol"],
        "metrics": per_config,
        "paired_deltas": comparisons,
        "pareto": payload["pareto"],
        "outputs": {
            "json": str(stem.with_suffix(".json")), "csv": str(stem.with_suffix(".csv")),
            "md": str(stem.with_suffix(".md")), "per_question_csv": str(args.per_question_csv),
        },
        "compliance": {
            "full_2638_run": False, "grounding_pointing_pck_iou": False,
            "new_algorithm_search": False, "reused_16token_results": False,
            "original_model_or_app_modified": False, "files_deleted": False,
            "processes_killed": False, "sealed_data_access": False, "board_access": False,
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Compact handoff JSON capped at 1200 bytes; short keys keep it inside budget.
    short_key = {
        "official_combined": "s", "official_2d": "d2", "official_3d": "d3",
        "relative_score_retention_vs_dense": "r", "parse_success_rate": "p",
        "output_cap_hit_count": "cap", "actual_merged_visual_tokens_mean": "t",
        "actual_token_retention_mean": "tr", "model_core_latency_ms": "c",
        "model_end_to_end_ms": "e",
    }
    integral = {"output_cap_hit_count"}
    coarse = {"model_core_latency_ms", "model_end_to_end_ms", "actual_merged_visual_tokens_mean"}

    def compact(item: dict, fields: list[str]) -> dict:
        out = {}
        for key in fields:
            value = item[key]
            if key in integral:
                out[short_key[key]] = int(value)
            elif key in coarse:
                out[short_key[key]] = round(float(value), 1)
            else:
                out[short_key[key]] = round(float(value), 3)
        return out

    field_order = ["official_combined", "relative_score_retention_vs_dense",
                   "model_end_to_end_ms", "model_core_latency_ms", "parse_success_rate",
                   "output_cap_hit_count", "actual_token_retention_mean",
                   "actual_merged_visual_tokens_mean", "official_2d", "official_3d"]
    paired_compact = {
        "gl_g256": [round(comparisons["global_local_vs_global256"]["delta"] * 100, 2),
                    round(comparisons["global_local_vs_global256"]["ci95_low"] * 100, 2),
                    round(comparisons["global_local_vs_global256"]["ci95_high"] * 100, 2)],
        "gl_l2_50": [round(comparisons["global_local_vs_l2_50"]["delta"] * 100, 2),
                     round(comparisons["global_local_vs_l2_50"]["ci95_low"] * 100, 2),
                     round(comparisons["global_local_vs_l2_50"]["ci95_high"] * 100, 2)],
    }

    def build(fields: list[str]) -> str:
        result = {
            "status": "completed",
            "report": str(stem.with_suffix(".md").resolve()),
            "raw_json": str(stem.with_suffix(".json").resolve()),
            "raw_csv": str(stem.with_suffix(".csv").resolve()),
            "manifest": str(args.artifact.resolve()),
            "key_metrics": {
                **{config: compact(per_config[config], fields) for config in CONFIGS},
                "paired_deltas": paired_compact,
                "pareto": frontier,
            },
            "errors": [],
        }
        return json.dumps(result, ensure_ascii=False, separators=(",", ":"))

    fields = list(field_order)
    text = build(fields)
    while len(text.encode("utf-8")) > 1200 and len(fields) > 2:
        fields.pop()
        text = build(fields)
    if len(text.encode("utf-8")) > 1200:
        raise RuntimeError("cannot fit the 1200-byte handoff JSON")
    args.harness_json.parent.mkdir(parents=True, exist_ok=True)
    args.harness_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
