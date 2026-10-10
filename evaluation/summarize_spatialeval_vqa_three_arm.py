#!/usr/bin/env python3
"""Summarise the SpatialEval VQA 4500 three-arm external confirmation (CPU-only).

Requires all three raw JSONL arms (``dense_edge576`` /
``corrected_l2_integer_b1_k96`` / ``v2_edge576_b1_k96``) to exist with exactly
4500 records and an identical ``sample_id`` set.  Each arm is scored with the
official-compatible SpatialEval evaluator
(:mod:`evaluation.spatialeval_official_evaluator`) and the project-side strict
diagnostic.  ``spatialreal`` (135) is excluded: the upstream evaluator has no
handler for it.

Pre-registered project gates (not an official overall score):

* ``v2_task_macro / dense_task_macro >= 0.90``;
* the ``task::question_id`` stratified paired bootstrap (seed **20261009**, 5000
  repetitions, 95%) lower bound of the ``v2 - corrected_l2`` **task-macro**
  difference is ``>= 0``.

Parsing failures and ``None`` parses count as wrong.  Per-task figures are
reported only; no per-task gate is applied.  A missing or inconsistent arm
aborts with exit code 2 and writes no report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.spatialeval_official_evaluator import (  # noqa: E402
    HANDLER_TASKS,
    NON_HANDLER_TASKS,
    official_parsers,
    score_predictions,
)
from evaluation.spatialeval_vqa_contract import (  # noqa: E402
    ARM_DENSE,
    ARM_L2,
    ARM_OUTPUT_PATHS,
    ARM_V2,
    ARMS,
    EXPECTED_SAMPLE_COUNT,
    REFERENCE_MANIFEST,
    SpatialEvalContractError,
    load_reference_manifest,
    select_handler_records,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-09-spatialeval-vqa-three-arm-summary.json"
)
REPORT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-09-spatialeval-vqa-three-arm-summary.md"
)
REPORT_QID_CSV = (
    ROOT
    / "reports"
    / "token_selector"
    / "2026-10-09-spatialeval-vqa-three-arm-qid-detail.csv"
)

GATE_V2_DENSE_MACRO_RATIO_MIN = 0.90
BOOTSTRAP_SEED = 20261009
BOOTSTRAP_REPETITIONS = 5000
BOOTSTRAP_CONFIDENCE = 0.95

METRIC_NAME = "spatialeval_official_compatible_string_inclusion"


class SpatialEvalSummaryError(RuntimeError):
    """Raised when the three-arm summary contract is violated (fail closed)."""


# ---------------------------------------------------------------------------
# Loading / consistency
# ---------------------------------------------------------------------------
def load_arm_predictions(
    path: Path, *, arm: str, expected_count: int = EXPECTED_SAMPLE_COUNT
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise SpatialEvalSummaryError(
            f"missing {arm} raw output {path} (fail closed)"
        )
    predictions: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalSummaryError(
                f"{path} line {line_number} is not valid JSON: {exc} (fail closed)"
            ) from exc
        if not isinstance(record, dict):
            raise SpatialEvalSummaryError(
                f"{path} line {line_number} is not a JSON object (fail closed)"
            )
        if "sample_id" not in record or "output_text" not in record:
            raise SpatialEvalSummaryError(
                f"{path} line {line_number} is missing sample_id/output_text "
                "(fail closed)"
            )
        if str(record.get("arm")) != arm:
            raise SpatialEvalSummaryError(
                f"{path} line {line_number} arm={record.get('arm')!r} != {arm!r} "
                "(fail closed)"
            )
        predictions.append(
            {"sample_id": str(record["sample_id"]), "output_text": record["output_text"]}
        )
    if len(predictions) != expected_count:
        raise SpatialEvalSummaryError(
            f"{arm} has {len(predictions)} records != {expected_count} "
            "(fail closed)"
        )
    ids = [item["sample_id"] for item in predictions]
    if len(ids) != len(set(ids)):
        raise SpatialEvalSummaryError(
            f"{arm} has duplicate sample_id values (fail closed)"
        )
    return predictions


def require_identical_sample_sets(
    predictions_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_sample_ids: Sequence[str],
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> list[str]:
    if set(predictions_by_arm) != set(ARMS):
        raise SpatialEvalSummaryError(
            f"expected arms {ARMS}, got {sorted(predictions_by_arm)} (fail closed)"
        )
    expected = set(str(value) for value in expected_sample_ids)
    if len(expected) != expected_count:
        raise SpatialEvalSummaryError(
            f"expected {expected_count} sample ids, got {len(expected)} "
            "(fail closed)"
        )
    reference: set[str] | None = None
    for arm in ARMS:
        ids = {str(item["sample_id"]) for item in predictions_by_arm[arm]}
        if ids != expected:
            raise SpatialEvalSummaryError(
                f"{arm} sample_id set differs from the frozen 4500 set (fail closed)"
            )
        if reference is None:
            reference = ids
        elif ids != reference:
            raise SpatialEvalSummaryError(
                f"{arm} sample_id set differs from the other arms (fail closed)"
            )
    return sorted(reference or set())


# ---------------------------------------------------------------------------
# Task-macro stratified paired bootstrap
# ---------------------------------------------------------------------------
def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise SpatialEvalSummaryError("cannot take a percentile of an empty list")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])
    index = q * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = index - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def _macro_difference(pairs_by_task: Mapping[str, Sequence[tuple[int, int]]]) -> float:
    task_differences = []
    for task, pairs in sorted(pairs_by_task.items()):
        total = len(pairs)
        if total == 0:
            continue
        treated_hits = sum(pair[0] for pair in pairs)
        control_hits = sum(pair[1] for pair in pairs)
        task_differences.append((treated_hits - control_hits) / total)
    if not task_differences:
        raise SpatialEvalSummaryError("no paired strata to bootstrap (fail closed)")
    return sum(task_differences) / len(task_differences)


def stratified_paired_bootstrap_macro(
    per_sample_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    treated_arm: str,
    control_arm: str,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
) -> dict[str, Any]:
    """Paired bootstrap of the task-macro ``treated - control`` difference.

    Resampling is stratified by ``task::question_id``; each stratum is resampled
    with replacement and the two arms stay paired sample-by-sample.  The observed
    statistic is the mean over the three tasks of the per-task accuracy
    difference.
    """

    if treated_arm not in per_sample_by_arm or control_arm not in per_sample_by_arm:
        raise SpatialEvalSummaryError(
            f"missing arm for bootstrap {treated_arm}/{control_arm} (fail closed)"
        )
    if repetitions <= 0:
        raise SpatialEvalSummaryError("repetitions must be positive (fail closed)")
    control = {item["sample_id"]: item for item in per_sample_by_arm[control_arm]}
    treated = {item["sample_id"]: item for item in per_sample_by_arm[treated_arm]}
    if set(control) != set(treated):
        raise SpatialEvalSummaryError(
            "paired bootstrap requires identical sample sets (fail closed)"
        )
    strata: dict[str, list[tuple[int, int]]] = {}
    pairs_by_task: dict[str, list[tuple[int, int]]] = {}
    for sample_id, control_item in control.items():
        treated_item = treated[sample_id]
        task = str(control_item["task"])
        key = f"{task}::{int(control_item['question_id'])}"
        pair = (int(treated_item["official_correct"]), int(control_item["official_correct"]))
        strata.setdefault(key, []).append(pair)
        pairs_by_task.setdefault(task, []).append(pair)
    observed = _macro_difference(pairs_by_task)

    rng = random.Random(int(seed))
    tasks = sorted(pairs_by_task)
    strata_by_task = {task: [] for task in tasks}
    for key, pairs in strata.items():
        task = key.split("::")[0]
        strata_by_task[task].append(pairs)
    distribution: list[float] = []
    for _ in range(int(repetitions)):
        task_differences = []
        for task in tasks:
            treated_hits = 0
            control_hits = 0
            total = 0
            for pairs in strata_by_task[task]:
                count = len(pairs)
                for _ in range(count):
                    treated_value, control_value = pairs[rng.randrange(count)]
                    treated_hits += treated_value
                    control_hits += control_value
                total += count
            task_differences.append((treated_hits - control_hits) / total)
        distribution.append(sum(task_differences) / len(task_differences))
    distribution.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    lower = _percentile(distribution, alpha)
    upper = _percentile(distribution, 1.0 - alpha)
    return {
        "treated_arm": treated_arm,
        "control_arm": control_arm,
        "statistic": "project_task_macro_difference",
        "stratification": "task::question_id",
        "paired": True,
        "seed": int(seed),
        "repetitions": int(repetitions),
        "confidence": float(confidence),
        "observed": observed,
        "ci_lower": lower,
        "ci_upper": upper,
        "ci": [lower, upper],
    }


# ---------------------------------------------------------------------------
# Summary assembly
# ---------------------------------------------------------------------------
def summarize_arms(
    predictions_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    oracle_by_id: Mapping[str, Any],
    *,
    expected_sample_ids: Sequence[str],
    reference_manifest_sha256: str,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
) -> dict[str, Any]:
    require_identical_sample_sets(
        predictions_by_arm,
        expected_sample_ids=expected_sample_ids,
        expected_count=expected_count,
    )
    parsers = official_parsers()
    arm_reports: dict[str, Any] = {}
    per_sample_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        report = score_predictions(
            predictions_by_arm[arm],
            oracle_by_id,
            parsers=parsers,
            expected_sample_ids=expected_sample_ids,
            include_per_sample=True,
        )
        arm_reports[arm] = report
        per_sample_by_arm[arm] = list(report["per_sample"])

    dense_macro = float(arm_reports[ARM_DENSE]["project_task_macro"]["accuracy"])
    l2_macro = float(arm_reports[ARM_L2]["project_task_macro"]["accuracy"])
    v2_macro = float(arm_reports[ARM_V2]["project_task_macro"]["accuracy"])
    if dense_macro <= 0.0:
        raise SpatialEvalSummaryError(
            "dense task-macro is non-positive; the ratio gate is undefined "
            "(fail closed)"
        )
    ratio = v2_macro / dense_macro
    bootstrap = stratified_paired_bootstrap_macro(
        per_sample_by_arm,
        treated_arm=ARM_V2,
        control_arm=ARM_L2,
        seed=seed,
        repetitions=repetitions,
        confidence=confidence,
    )
    ratio_pass = ratio >= GATE_V2_DENSE_MACRO_RATIO_MIN
    ci_pass = float(bootstrap["ci_lower"]) >= 0.0
    per_task_qid: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        per_task_qid[arm] = arm_reports[arm]["per_task_qid"]
    return {
        "status": (
            "SPATIALEVAL_EXTERNAL_GATES_PASS"
            if ratio_pass and ci_pass
            else "SPATIALEVAL_EXTERNAL_GATES_FAIL"
        ),
        "metric": METRIC_NAME,
        "reference_manifest_sha256": reference_manifest_sha256,
        "count_per_arm": len(expected_sample_ids),
        "sample_set_identical": True,
        "handler_tasks": list(HANDLER_TASKS),
        "excluded_tasks": {
            "tasks": list(NON_HANDLER_TASKS),
            "reason": "upstream evaluation.py has no spatialreal handler",
            "in_main_gate": False,
            "exploratory_only": True,
        },
        "arms": {
            arm: {
                "metric": arm_reports[arm]["metric"],
                "accuracy": arm_reports[arm]["accuracy"],
                "correct": arm_reports[arm]["correct"],
                "count": arm_reports[arm]["count"],
                "parse_failure_count": arm_reports[arm]["parse_failure_count"],
                "parse_success_rate": arm_reports[arm]["parse_success_rate"],
                "project_task_macro": arm_reports[arm]["project_task_macro"],
                "project_micro": arm_reports[arm]["project_micro"],
                "per_task": arm_reports[arm]["per_task"],
                "strict_equality_diagnostic": arm_reports[arm][
                    "strict_equality_diagnostic"
                ],
            }
            for arm in ARMS
        },
        "project_gate": {
            "v2_dense_task_macro": {
                "dense": dense_macro,
                "v2": v2_macro,
                "ratio": ratio,
                "minimum": GATE_V2_DENSE_MACRO_RATIO_MIN,
                "pass": ratio_pass,
                "semantics": "project_ratio_of_task_macro_not_official_total",
            },
            "v2_minus_l2_task_macro_bootstrap": {
                **bootstrap,
                "lower_bound_minimum": 0.0,
                "pass": ci_pass,
                "semantics": "project_task_macro_difference_stratified_paired_bootstrap",
            },
            "overall_pass": bool(ratio_pass and ci_pass),
        },
        "per_task_qid": per_task_qid,
        "declarations": {
            "official_cross_task_total_exists": False,
            "macro_micro_are_project_aggregations": True,
            "parse_failure_counts_wrong": True,
            "labels_observed": True,
            "not_label_blind": True,
            "not_used_for_selector_tuning": True,
            "per_task_has_no_gate": True,
            "gpu_runs": 0,
            "model_loaded": False,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# SpatialEval VQA 4500 三臂外部确认汇总（CPU-only）",
        "",
        f"- 状态：`{report['status']}`",
        f"- 主指标：`{report['metric']}`（官方兼容 `oracle_answer.lower() in parsed.lower()`；解析异常/None 计错）",
        f"- 每臂样本数：{report['count_per_arm']}；三臂 sample 集合一致：{report['sample_set_identical']}",
        f"- 官方无跨任务总分；`project_task_macro` 与 `project_micro` 均为**项目汇总**，不是官方总分。",
        f"- `spatialreal` 135 条为探索性、**不进入主门**（官方无 handler）。",
        "",
        "## 三臂主分与 strict 诊断",
        "",
        "| arm | official acc | task macro | micro | strict acc | parse 失败 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, arm_report in report["arms"].items():
        lines.append(
            f"| {arm} | {arm_report['accuracy']:.6f} | "
            f"{arm_report['project_task_macro']['accuracy']:.6f} | "
            f"{arm_report['project_micro']['accuracy']:.6f} | "
            f"{arm_report['strict_equality_diagnostic']['accuracy']:.6f} | "
            f"{arm_report['parse_failure_count']} |"
        )
    gate = report["project_gate"]
    lines += [
        "",
        "## 预注册项目主门",
        "",
        f"- v2/dense task-macro 比值：{gate['v2_dense_task_macro']['ratio']:.6f} "
        f"（门 >= {gate['v2_dense_task_macro']['minimum']}）："
        f"{'通过' if gate['v2_dense_task_macro']['pass'] else '未通过'}",
        f"- v2−l2 task-macro 分层配对 bootstrap（seed "
        f"{gate['v2_minus_l2_task_macro_bootstrap']['seed']}，reps "
        f"{gate['v2_minus_l2_task_macro_bootstrap']['repetitions']}，95%）"
        f"观测 {gate['v2_minus_l2_task_macro_bootstrap']['observed']:.6f}，"
        f"CI [{gate['v2_minus_l2_task_macro_bootstrap']['ci_lower']:.6f}, "
        f"{gate['v2_minus_l2_task_macro_bootstrap']['ci_upper']:.6f}]，"
        f"下界 >=0：{'通过' if gate['v2_minus_l2_task_macro_bootstrap']['pass'] else '未通过'}",
        f"- 主门总体：{'通过' if gate['overall_pass'] else '未通过'}",
        "",
        "## 说明",
        "",
        "- 逐任务仅报告，不设门；解析失败计为错误。",
        "- 本汇总为 CPU-only 只读统计，不含任何模型加载或 GPU 运行。",
        "- 标签已 materialize/审计（`labels_observed=true`），**不是 label-blind**，不得据此调参。",
        "",
    ]
    return "\n".join(lines)


def _write_qid_detail_csv(report: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "task", "question_id", "count", "correct", "accuracy",
                         "parse_failure_count"])
        for arm, buckets in report["per_task_qid"].items():
            for key in sorted(buckets):
                bucket = buckets[key]
                writer.writerow(
                    [
                        arm,
                        bucket["task"],
                        bucket["question_id"],
                        bucket["count"],
                        bucket["correct"],
                        f"{bucket['accuracy']:.10f}",
                        bucket["parse_failure_count"],
                    ]
                )


def write_report(report: Mapping[str, Any], *, json_path: Path = REPORT_JSON,
                 md_path: Path = REPORT_MD, csv_path: Path = REPORT_QID_CSV) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    _write_qid_detail_csv(report, csv_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm.replace('_', '-')}", type=Path,
                            default=ARM_OUTPUT_PATHS[arm])
    parser.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--json-out", type=Path, default=REPORT_JSON)
    parser.add_argument("--md-out", type=Path, default=REPORT_MD)
    parser.add_argument("--qid-csv-out", type=Path, default=REPORT_QID_CSV)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--repetitions", type=int, default=BOOTSTRAP_REPETITIONS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        reference_records = load_reference_manifest(args.reference_manifest)
        selected = select_handler_records(reference_records)
        oracle_by_id = {
            str(record["sample_id"]): record["oracle_answer"] for record in selected
        }
        expected_sample_ids = [str(record["sample_id"]) for record in selected]
        predictions_by_arm = {
            arm: load_arm_predictions(getattr(args, arm), arm=arm) for arm in ARMS
        }
        report = summarize_arms(
            predictions_by_arm,
            oracle_by_id,
            expected_sample_ids=expected_sample_ids,
            reference_manifest_sha256=sha256_file(Path(args.reference_manifest)),
            seed=args.seed,
            repetitions=args.repetitions,
        )
    except (SpatialEvalContractError, SpatialEvalSummaryError) as exc:
        print(f"SPATIALEVAL_SUMMARY_FAIL_CLOSED: {exc}", file=sys.stderr, flush=True)
        return 2
    write_report(
        report,
        json_path=args.json_out,
        md_path=args.md_out,
        csv_path=args.qid_csv_out,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
