#!/usr/bin/env python3
"""Measure the revealed upper bound of choosing between two ROI resolutions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl, score_record


def load_deterministic_strategy(
    path: Path, strategy: str, expected_repeats: int
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_jsonl(path):
        if record.get("strategy") == strategy:
            grouped[str(record["sample_id"])].append(record)
    if not grouped:
        raise ValueError(f"strategy {strategy!r} not found in {path}")
    selected: dict[str, dict[str, Any]] = {}
    for sample_id, records in grouped.items():
        if len(records) != expected_repeats:
            raise ValueError(
                f"{strategy}:{sample_id}: expected {expected_repeats} repeats, got {len(records)}"
            )
        if len({str(record.get("output_text", "")) for record in records}) != 1:
            raise ValueError(f"{strategy}:{sample_id}: repeated outputs are not deterministic")
        selected[sample_id] = min(records, key=lambda record: int(record["repeat"]))
    return selected


def selection_utility(task: str, score: dict[str, Any]) -> float:
    if task == "grounding":
        return float(score["iou"])
    error = score.get("point_nme_bbox_diag")
    return -float(error) if error is not None and math.isfinite(float(error)) else -1e9


def summarize_policy(
    samples: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    low: dict[str, dict[str, Any]],
    high: dict[str, dict[str, Any]],
    choices: dict[str, str],
    *,
    active_escalation: bool,
) -> dict[str, Any]:
    scores: list[dict[str, Any]] = []
    final_tokens: list[int] = []
    processed_tokens: list[int] = []
    baseline_tokens: list[int] = []
    high_count = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        choice = choices[sample_id]
        chosen = high[sample_id] if choice == "high" else low[sample_id]
        scores.append(score_record(sample, chosen))
        low_tokens = int(low[sample_id]["final_context_visual_tokens"])
        high_tokens = int(high[sample_id]["final_context_visual_tokens"])
        chosen_tokens = high_tokens if choice == "high" else low_tokens
        final_tokens.append(chosen_tokens)
        processed_tokens.append(
            low_tokens + high_tokens if active_escalation and choice == "high" else chosen_tokens
        )
        baseline_tokens.append(int(baseline[sample_id]["final_context_visual_tokens"]))
        high_count += int(choice == "high")

    common = {
        "n": len(samples),
        "high_selected": high_count,
        "high_selected_rate": high_count / len(samples),
        "parse_rate": statistics.fmean(bool(score["parse_success"]) for score in scores),
        "mean_final_context_visual_tokens": statistics.fmean(final_tokens),
        "mean_cumulative_processed_visual_tokens": statistics.fmean(processed_tokens),
        "final_token_ratio_to_dense": sum(final_tokens) / sum(baseline_tokens),
        "processed_token_ratio_to_dense": sum(processed_tokens) / sum(baseline_tokens),
    }
    task = str(samples[0]["task"])
    if task == "grounding":
        common.update(
            mean_iou=statistics.fmean(float(score["iou"]) for score in scores),
            acc_0_5=statistics.fmean(bool(score["acc_0_5"]) for score in scores),
        )
    else:
        errors = [float(score["point_nme_bbox_diag"]) for score in scores if score.get("point_nme_bbox_diag") is not None]
        common.update(
            mean_point_nme_bbox_diag=statistics.fmean(errors) if errors else None,
            pck_0_1=statistics.fmean(bool(score["pck_0_1"]) for score in scores),
            pck_0_2=statistics.fmean(bool(score["pck_0_2"]) for score in scores),
        )
    return common


def analyze(
    samples: list[dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    low: dict[str, dict[str, Any]],
    high: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    analyses: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in baseline and sample_id in low and sample_id in high:
            by_task[str(sample["task"])].append(sample)

    for task, task_samples in sorted(by_task.items()):
        choices: dict[str, str] = {}
        relation = Counter()
        strata_high: dict[str, Counter[str]] = defaultdict(Counter)
        for sample in task_samples:
            sample_id = str(sample["sample_id"])
            low_score = score_record(sample, low[sample_id])
            high_score = score_record(sample, high[sample_id])
            low_utility = selection_utility(task, low_score)
            high_utility = selection_utility(task, high_score)
            gain = high_utility - low_utility
            if gain > 1e-12:
                choice = "high"
                relation["high_better"] += 1
            elif gain < -1e-12:
                choice = "low"
                relation["high_worse"] += 1
            else:
                choice = "low"
                relation["equal"] += 1
            choices[sample_id] = choice
            for dimension in (
                "target_area_bin",
                "edge_bin",
                "overlap_bin",
                "scene_complexity_bin",
            ):
                value = sample.get("stratification", {}).get(dimension)
                if value is not None:
                    strata_high[dimension][str(value)] += int(choice == "high")
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "choice": choice,
                    "revealed_utility_gain_high_minus_low": gain,
                    "low_tokens": int(low[sample_id]["final_context_visual_tokens"]),
                    "high_tokens": int(high[sample_id]["final_context_visual_tokens"]),
                    "stratification": sample.get("stratification", {}),
                }
            )

        all_low = {str(sample["sample_id"]): "low" for sample in task_samples}
        all_high = {str(sample["sample_id"]): "high" for sample in task_samples}
        analyses[task] = {
            "action_complementarity": dict(relation),
            "policies": {
                "fixed_low_384": summarize_policy(
                    task_samples, baseline, low, high, all_low, active_escalation=False
                ),
                "fixed_high_512": summarize_policy(
                    task_samples, baseline, low, high, all_high, active_escalation=False
                ),
                "revealed_one_shot_oracle": summarize_policy(
                    task_samples, baseline, low, high, choices, active_escalation=False
                ),
                "revealed_active_escalation_oracle": summarize_policy(
                    task_samples, baseline, low, high, choices, active_escalation=True
                ),
            },
            "high_selection_count_by_stratum": {
                dimension: dict(values) for dimension, values in sorted(strata_high.items())
            },
        }
    return analyses, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-results", type=Path, required=True)
    parser.add_argument("--baseline-strategy", default="dense")
    parser.add_argument("--low-results", type=Path, required=True)
    parser.add_argument("--low-strategy", default="roi_only")
    parser.add_argument("--high-results", type=Path, required=True)
    parser.add_argument("--high-strategy", required=True)
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    samples = load_jsonl(args.manifest)
    baseline = load_deterministic_strategy(
        args.baseline_results, args.baseline_strategy, args.expected_repeats
    )
    low = load_deterministic_strategy(args.low_results, args.low_strategy, args.expected_repeats)
    high = load_deterministic_strategy(args.high_results, args.high_strategy, args.expected_repeats)
    common_ids = set(baseline) & set(low) & set(high)
    selected_samples = [sample for sample in samples if str(sample["sample_id"]) in common_ids]
    analyses, rows = analyze(selected_samples, baseline, low, high)
    report = {
        "status": "ROI_RESOLUTION_REVEALED_ORACLE_COMPLETE",
        "evidence_level": "revealed 32-image development upper bound; not a deployable selector",
        "selection_rule": {
            "grounding": "choose the resolution with higher ground-truth IoU",
            "pointing": "choose the resolution with lower ground-truth bbox-normalized point error",
            "ties": "choose low resolution",
        },
        "cost_boundary": {
            "one_shot": "only the selected resolution is encoded",
            "active_escalation": "low resolution is encoded first; selected high resolution is then re-encoded and both token counts are charged",
        },
        "analyses": analyses,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ROI分辨率动作互补性上界",
        "",
        "状态：`ROI_RESOLUTION_REVEALED_ORACLE_COMPLETE`。本分析读取开发集真值逐图选择384或512，只是控制器潜力上界，不是可部署策略。",
        "",
        "| 任务 | 512更好/相同/更差 | 策略 | 质量主指标 | 次指标 | 选512比例 | 最终Token/Dense | 累计Token/Dense |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for task, analysis in analyses.items():
        relation = analysis["action_complementarity"]
        relation_text = f"{relation.get('high_better', 0)}/{relation.get('equal', 0)}/{relation.get('high_worse', 0)}"
        for name, policy in analysis["policies"].items():
            if task == "grounding":
                primary = f"IoU {policy['mean_iou']:.4f}"
                secondary = f"Acc@0.5 {policy['acc_0_5']:.4f}"
            else:
                primary = f"PCK@0.1 {policy['pck_0_1']:.4f}"
                secondary = f"PCK@0.2 {policy['pck_0_2']:.4f}"
            lines.append(
                f"| {task} | {relation_text} | {name} | {primary} | {secondary} | "
                f"{policy['high_selected_rate']:.3f} | {policy['final_token_ratio_to_dense']:.3f} | "
                f"{policy['processed_token_ratio_to_dense']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `revealed_one_shot_oracle`假设编码前已知最佳尺度，不能部署，只回答两种动作是否互补。",
            "- `revealed_active_escalation_oracle`先处理384，再对被选样本重编码512；质量相同，但累计Token计入重复视觉处理。",
            "- 若Oracle明显优于两个固定尺度，只说明存在可学习信号的必要条件；仍需使用不读真值的特征训练控制器，并在校准集和封存测试集验证。",
            "- 本报告不比较跨运行延迟，因为GPU负载条件不同。",
        ]
    )
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "samples": len(selected_samples)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
