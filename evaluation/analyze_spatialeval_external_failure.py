#!/usr/bin/env python3
"""Diagnose the failed frozen SpatialEval three-arm external confirmation."""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import spatialeval_official_compatible_metric as metric
from evaluation.spatialeval_external_contract import (
    ARM_DENSE,
    ARM_L2,
    ARM_OUTPUT_PATHS,
    ARM_V2,
    ARMS,
    sha256_file,
)
from evaluation.summarize_spatialeval_three_arm import load_arm_records


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports/token_selector/2026-10-09-spatialeval-three-arm-summary.json"
DEFAULT_JSON = ROOT / "reports/token_selector/2026-10-09-spatialeval-external-failure-analysis.json"
DEFAULT_MD = ROOT / "reports/token_selector/2026-10-09-spatialeval-external-failure-analysis.md"


def _mean(records: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records if isinstance(record.get(field), (int, float))]
    if not values:
        raise ValueError(f"no numeric values for {field}")
    return statistics.fmean(values)


def _correctness(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    scored = metric.score_records(list(records), include_record_rows=True)
    return {str(row["sample_id"]): row for row in scored["evaluated_records"]}


def _paired_contingency(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(left) != set(right):
        raise ValueError("paired arms have different sample sets")
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"left_only_correct": 0, "right_only_correct": 0, "both_correct": 0, "both_wrong": 0}
    )
    for sample_id in sorted(left):
        task = str(left[sample_id]["task"])
        lval = int(left[sample_id]["official_compatible_correct"])
        rval = int(right[sample_id]["official_compatible_correct"])
        key = (
            "both_correct" if lval and rval else
            "left_only_correct" if lval else
            "right_only_correct" if rval else
            "both_wrong"
        )
        totals[task][key] += 1
        totals["all"][key] += 1
    return dict(totals)


def analyze() -> dict[str, Any]:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    records = {arm: load_arm_records(ARM_OUTPUT_PATHS[arm], arm) for arm in ARMS}
    correctness = {arm: _correctness(records[arm]) for arm in ARMS}

    dense_tokens = _mean(records[ARM_DENSE], "full_llm_visual_tokens")
    l2_tokens = _mean(records[ARM_L2], "output_group_count")
    v2_tokens = _mean(records[ARM_V2], "output_group_count")
    dense_elapsed = _mean(records[ARM_DENSE], "elapsed_seconds")
    dense_decode = _mean(records[ARM_DENSE], "decoder_seconds")
    v2_elapsed = _mean(records[ARM_V2], "elapsed_seconds")
    v2_selector = _mean(records[ARM_V2], "selector_cpu_roundtrip_seconds")
    v2_prefill = _mean(records[ARM_V2], "llm_prefill_seconds")
    dense_prefill = _mean(records[ARM_DENSE], "llm_prefill_seconds")
    v2_no_selector_lower_bound = max(v2_elapsed - v2_selector, 0.0)
    output_lengths = {
        arm: [int(record["output_token_count"]) for record in records[arm]]
        for arm in ARMS
    }
    output_length_summary = {
        arm: {
            "mean": statistics.fmean(values),
            "minimum": min(values),
            "maximum": max(values),
            "at_max_new_tokens_64_count": sum(value == 64 for value in values),
            "at_max_new_tokens_64_rate": sum(value == 64 for value in values) / len(values),
        }
        for arm, values in output_lengths.items()
    }

    score = {
        arm: float(summary["scores"][arm]["equal_task_macro_project_summary"]["official_compatible_accuracy"])
        for arm in ARMS
    }
    return {
        "status": "CURRENT_K96_SELECTOR_REJECTED_BY_EXTERNAL_CONFIRMATION",
        "scope": summary["scope"],
        "source_summary": str(SUMMARY_PATH),
        "source_summary_sha256": sha256_file(SUMMARY_PATH),
        "raw_outputs": {
            arm: {
                "path": str(ARM_OUTPUT_PATHS[arm]),
                "sha256": sha256_file(ARM_OUTPUT_PATHS[arm]),
                "records": len(records[arm]),
            }
            for arm in ARMS
        },
        "quality": {
            "official_compatible_project_equal_task_macro": score,
            "v2_minus_dense_absolute": score[ARM_V2] - score[ARM_DENSE],
            "v2_over_dense": score[ARM_V2] / score[ARM_DENSE],
            "v2_minus_l2_absolute": score[ARM_V2] - score[ARM_L2],
            "v2_minus_l2_ci95": [
                summary["comparison"]["v2_minus_l2_stratified_paired_bootstrap"]["ci_95_lower"],
                summary["comparison"]["v2_minus_l2_stratified_paired_bootstrap"]["ci_95_upper"],
            ],
            "v2_vs_l2_paired_contingency": _paired_contingency(correctness[ARM_V2], correctness[ARM_L2]),
        },
        "token_budget": {
            "dense_mean_visual_tokens": dense_tokens,
            "l2_mean_selected_tokens": l2_tokens,
            "v2_mean_selected_tokens": v2_tokens,
            "v2_aggregate_retention": v2_tokens / dense_tokens,
            "v2_aggregate_reduction": 1.0 - v2_tokens / dense_tokens,
            "fixed_k": 96,
            "input_g_values": sorted({int(record["full_llm_visual_tokens"]) for record in records[ARM_V2]}),
        },
        "latency": {
            "dense_elapsed_mean_ms": dense_elapsed * 1000.0,
            "v2_elapsed_mean_ms": v2_elapsed * 1000.0,
            "measured_dense_over_v2_speed_ratio": dense_elapsed / v2_elapsed,
            "measured_v2_change_vs_dense": v2_elapsed / dense_elapsed - 1.0,
            "dense_decode_mean_ms": dense_decode * 1000.0,
            "dense_decode_fraction_of_elapsed": dense_decode / dense_elapsed,
            "amdahl_zero_all_non_decode_max_speedup": dense_elapsed / dense_decode,
            "dense_prefill_mean_ms": dense_prefill * 1000.0,
            "v2_prefill_mean_ms": v2_prefill * 1000.0,
            "prefill_speedup": dense_prefill / v2_prefill,
            "prefill_time_saved_ms": (dense_prefill - v2_prefill) * 1000.0,
            "v2_selector_cpu_roundtrip_mean_ms": v2_selector * 1000.0,
            "optimistic_v2_elapsed_if_cpu_selector_were_zero_ms": v2_no_selector_lower_bound * 1000.0,
            "optimistic_dense_over_v2_speedup_if_cpu_selector_were_zero": dense_elapsed / v2_no_selector_lower_bound,
            "output_token_counts": output_length_summary,
            "benchmark_prompt_boundary": (
                "Most samples hit the 64-token cap because the frozen official prompt requests reasoning. "
                "Decode dominance is valid for this benchmark configuration but must not be generalized "
                "to coordinate-only or action-code outputs without a separate matched experiment."
            ),
            "warning": (
                "The zero-selector figure is an optimistic software subtraction, not an FPGA prediction; "
                "it omits real transfer, synchronization, and PL execution costs."
            ),
        },
        "decision": {
            "external_gate_pass": False,
            "continue_current_k96_v2_as_primary_method": False,
            "spatialeval_test_may_be_used_for_retuning": False,
            "reason": (
                "The frozen v2 arm retained only 74.42% of Dense quality and did not beat the matched L2 baseline; "
                "Decode dominates latency, so visual-token selection alone cannot meet the 333 ms system target."
            ),
            "next_research_direction": (
                "Return to development data for an adaptive spatial-relation-preserving budget and pair it with "
                "structured short decoding or a non-autoregressive action head. Reserve a different untouched "
                "benchmark for the next external confirmation."
            ),
        },
    }


def _markdown(result: Mapping[str, Any]) -> str:
    q = result["quality"]
    t = result["token_budget"]
    l = result["latency"]
    c = q["v2_vs_l2_paired_contingency"]
    dense_output = l["output_token_counts"][ARM_DENSE]
    v2_output = l["output_token_counts"][ARM_V2]
    lines = [
        "# SpatialEval 外部确认失败归因",
        "",
        f"- 结论：`{result['status']}`。当前 K96 v2 不能作为论文主方案继续晋级。",
        "- 数据：三个官方可评分任务各 1500 条，共 4500 条；三臂合计 13500 次推理。",
        "- 该 test 已经解封并分析，今后不得用于调参或再次充当未见外部确认集。",
        "",
        "## 质量结果",
        "",
        f"- Dense：`{q['official_compatible_project_equal_task_macro'][ARM_DENSE]:.6f}`。",
        f"- L2@K96：`{q['official_compatible_project_equal_task_macro'][ARM_L2]:.6f}`。",
        f"- v2@K96：`{q['official_compatible_project_equal_task_macro'][ARM_V2]:.6f}`。",
        f"- v2/Dense：`{q['v2_over_dense']:.6f}`，即保留 `74.42%`，低于冻结门槛 `90%`。",
        f"- v2-L2：`{q['v2_minus_l2_absolute']:+.6f}`，95% CI "
        f"`[{q['v2_minus_l2_ci95'][0]:+.6f}, {q['v2_minus_l2_ci95'][1]:+.6f}]`；没有优于 L2 的证据。",
        f"- 配对结果（全体）：v2 独对 `{c['all']['left_only_correct']}` 条，L2 独对 "
        f"`{c['all']['right_only_correct']}` 条，两者都对 `{c['all']['both_correct']}` 条。",
        "",
        "## Token 与时延",
        "",
        f"- Dense 平均视觉 token：`{t['dense_mean_visual_tokens']:.2f}`；v2 固定输出 `96`，"
        f"总体减少 `{100*t['v2_aggregate_reduction']:.2f}%`。",
        f"- Dense 实测 mean elapsed：`{l['dense_elapsed_mean_ms']:.2f} ms`；v2："
        f"`{l['v2_elapsed_mean_ms']:.2f} ms`，当前软件路径反而慢 `{100*l['measured_v2_change_vs_dense']:.2f}%`。",
        f"- Dense Decode：`{l['dense_decode_mean_ms']:.2f} ms`，占总时延 "
        f"`{100*l['dense_decode_fraction_of_elapsed']:.2f}%`。即使所有非 Decode 阶段都变成零开销，"
        f"Amdahl 上限也只有 `{l['amdahl_zero_all_non_decode_max_speedup']:.3f}x`。",
        f"- Dense 有 `{dense_output['at_max_new_tokens_64_count']}/4500` 条、v2 有 "
        f"`{v2_output['at_max_new_tokens_64_count']}/4500` 条输出撞到 64-token 上限。Decode 主导是"
        "本次官方解释型 Prompt 的真实结果，但不能直接外推到只输出坐标或动作码的短结构化任务。",
        f"- Prefill 从 `{l['dense_prefill_mean_ms']:.2f} ms` 降至 `{l['v2_prefill_mean_ms']:.2f} ms`，"
        f"加速 `{l['prefill_speedup']:.3f}x`，但只节省 `{l['prefill_time_saved_ms']:.2f} ms`。",
        f"- v2 CPU selector 往返均值 `{l['v2_selector_cpu_roundtrip_mean_ms']:.2f} ms`。把它乐观地直接减为零，"
        f"总加速也仅约 `{l['optimistic_dense_over_v2_speedup_if_cpu_selector_were_zero']:.3f}x`；这不是 FPGA 预测。",
        "",
        "## 决策",
        "",
        "1. 停止将当前固定 K96 v2 作为主创新点继续晋级；外部质量门已明确失败。",
        "2. 不在 SpatialEval test 上调 K、权重或锚点，避免把外部集变成发展集。",
        "3. 在原发展数据上研究自适应预算和空间关系保持，并与结构化短解码或非自回归动作头联合设计。",
        "4. 下一版方法冻结后，必须换一个从未观察过的公开基准做新的外部确认。",
        "5. Jetson/ZCU102 到位后只验证真实链路与功耗，不能用上板结果推翻本次质量失败。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    result = analyze()
    DEFAULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    DEFAULT_MD.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(DEFAULT_JSON), "markdown": str(DEFAULT_MD)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
