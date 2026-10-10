#!/usr/bin/env python3
"""Summarize the frozen SpatialEval VQA external three-arm confirmation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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
    PRIMARY_SAMPLE_COUNT,
    SpatialEvalExternalContractError,
    canonical_record_sha256,
    task_and_qid,
)


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20261009
BOOTSTRAP_REPETITIONS = 5000
RETENTION_THRESHOLD = 0.90
V2_MINUS_L2_CI_LOWER_THRESHOLD = 0.0
DEFAULT_JSON = ROOT / "reports/token_selector/2026-10-09-spatialeval-three-arm-summary.json"
DEFAULT_CSV = ROOT / "reports/token_selector/2026-10-09-spatialeval-three-arm-summary.csv"
DEFAULT_MD = ROOT / "reports/token_selector/2026-10-09-spatialeval-three-arm-summary.md"


def load_arm_records(
    path: Path, arm: str, *, expected_count: int | None = PRIMARY_SAMPLE_COUNT
) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        raise SpatialEvalExternalContractError(f"missing arm output {path}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpatialEvalExternalContractError(
                f"invalid JSON at {path}:{line_number}"
            ) from exc
        sample_id = str(record.get("sample_id"))
        if sample_id in seen:
            raise SpatialEvalExternalContractError(
                f"duplicate sample {sample_id!r} in {path}"
            )
        seen.add(sample_id)
        if record.get("arm") != arm:
            raise SpatialEvalExternalContractError(
                f"record {sample_id!r} arm {record.get('arm')!r} != {arm!r}"
            )
        if record.get("uses_labels") is not False:
            raise SpatialEvalExternalContractError(
                f"record {sample_id!r} does not prove label-free selection"
            )
        if canonical_record_sha256(record) != record.get("record_sha256"):
            raise SpatialEvalExternalContractError(
                f"record hash mismatch for {sample_id!r}"
            )
        records.append(record)
    if expected_count is not None and len(records) != expected_count:
        raise SpatialEvalExternalContractError(
            f"arm {arm} count {len(records)} != {expected_count}"
        )
    return records


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise SpatialEvalExternalContractError("cannot percentile an empty sequence")
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def stratified_paired_bootstrap(
    v2_correct: Mapping[str, int],
    l2_correct: Mapping[str, int],
    *,
    seed: int = BOOTSTRAP_SEED,
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    if set(v2_correct) != set(l2_correct):
        raise SpatialEvalExternalContractError("bootstrap arms have different sample sets")
    if repetitions <= 0:
        raise SpatialEvalExternalContractError("bootstrap repetitions must be positive")
    strata: dict[str, list[str]] = defaultdict(list)
    for sample_id in sorted(v2_correct):
        task, qid = task_and_qid(sample_id)
        strata[f"{task}::{qid}"].append(sample_id)
    rng = random.Random(seed)
    differences: list[float] = []
    ordered_strata = [strata[key] for key in sorted(strata)]
    for _ in range(repetitions):
        total = 0
        count = 0
        for ids in ordered_strata:
            for _ in range(len(ids)):
                sample_id = ids[rng.randrange(len(ids))]
                total += int(v2_correct[sample_id]) - int(l2_correct[sample_id])
                count += 1
        differences.append(total / count)
    observed = statistics.fmean(
        int(v2_correct[sample_id]) - int(l2_correct[sample_id])
        for sample_id in v2_correct
    )
    return {
        "stratification": "task::question_id",
        "seed": seed,
        "repetitions": repetitions,
        "observed_difference": observed,
        "ci_95_lower": _percentile(differences, 0.025),
        "ci_95_upper": _percentile(differences, 0.975),
        "percentile_method": "linear interpolation over sorted bootstrap replicates",
    }


def _latency_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = (
        "application_end_to_end_seconds",
        "elapsed_seconds",
        "vision_encoder_seconds",
        "llm_prefill_seconds",
        "decoder_seconds",
        "selector_cpu_roundtrip_seconds",
    )
    result: dict[str, Any] = {}
    for field in fields:
        values = [
            float(record[field])
            for record in records
            if isinstance(record.get(field), (int, float))
        ]
        if values:
            result[field] = {
                "available_count": len(values),
                "mean_ms": statistics.fmean(values) * 1000.0,
                "median_ms": statistics.median(values) * 1000.0,
                "p95_ms": _percentile(values, 0.95) * 1000.0,
            }
    return result


def summarize_records(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_count: int | None = PRIMARY_SAMPLE_COUNT,
    official_module: Any = None,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
) -> dict[str, Any]:
    if set(records_by_arm) != set(ARMS):
        raise SpatialEvalExternalContractError(
            f"expected arms {ARMS}, got {tuple(records_by_arm)}"
        )
    id_sets = {
        arm: {str(record["sample_id"]) for record in records}
        for arm, records in records_by_arm.items()
    }
    if len({frozenset(ids) for ids in id_sets.values()}) != 1:
        raise SpatialEvalExternalContractError("three arms have different sample sets")
    if expected_count is not None and any(
        len(records_by_arm[arm]) != expected_count for arm in ARMS
    ):
        raise SpatialEvalExternalContractError("one or more arms have wrong count")

    module = official_module or metric.load_official_evaluator()
    scored: dict[str, dict[str, Any]] = {}
    correctness: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        arm_score = metric.score_records(
            list(records_by_arm[arm]),
            expected_count=expected_count,
            official_module=module,
        )
        evaluated = arm_score.pop("evaluated_records")
        correctness[arm] = {
            str(row["sample_id"]): int(row["official_compatible_correct"])
            for row in evaluated
        }
        scored[arm] = arm_score

    dense_macro = float(
        scored[ARM_DENSE]["equal_task_macro_project_summary"]
        ["official_compatible_accuracy"]
    )
    v2_macro = float(
        scored[ARM_V2]["equal_task_macro_project_summary"]
        ["official_compatible_accuracy"]
    )
    retention = v2_macro / dense_macro if dense_macro > 0 else None
    bootstrap = stratified_paired_bootstrap(
        correctness[ARM_V2],
        correctness[ARM_L2],
        seed=bootstrap_seed,
        repetitions=bootstrap_repetitions,
    )
    retention_pass = retention is not None and retention >= RETENTION_THRESHOLD
    ci_pass = bootstrap["ci_95_lower"] >= V2_MINUS_L2_CI_LOWER_THRESHOLD
    return {
        "status": (
            "EXTERNAL_CONFIRMATION_GATES_PASS"
            if retention_pass and ci_pass
            else "EXTERNAL_CONFIRMATION_GATES_FAIL"
        ),
        "scope": "SpatialEval VQA primary synthetic tasks only (4500 records)",
        "spatialreal": "excluded from primary gate; official repository has no valid parser",
        "official_total_score_exists": False,
        "project_macro_notice": (
            "The equal-task macro is a preregistered project aggregate, not an "
            "official SpatialEval total score."
        ),
        "scores": scored,
        "latency": {
            arm: _latency_summary(records_by_arm[arm]) for arm in ARMS
        },
        "comparison": {
            "v2_dense_equal_task_macro_retention": retention,
            "v2_minus_l2_stratified_paired_bootstrap": bootstrap,
        },
        "gates": {
            "v2_dense_macro_retention_minimum": RETENTION_THRESHOLD,
            "v2_dense_macro_retention_pass": retention_pass,
            "v2_minus_l2_ci95_lower_minimum": V2_MINUS_L2_CI_LOWER_THRESHOLD,
            "v2_minus_l2_ci95_lower_pass": ci_pass,
            "all_pass": retention_pass and ci_pass,
            "per_task_gates": None,
        },
    }


def _csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        score = summary["scores"][arm]
        for task, values in score["per_task"].items():
            rows.append({"arm": arm, "scope": task, **values})
        for key, values in score["per_task_question_id"].items():
            rows.append({"arm": arm, "scope": key, **values})
        rows.append(
            {
                "arm": arm,
                "scope": "project_equal_task_macro",
                "count": PRIMARY_SAMPLE_COUNT,
                "official_compatible_accuracy": score[
                    "equal_task_macro_project_summary"
                ]["official_compatible_accuracy"],
                "strict_equal_accuracy": score["equal_task_macro_project_summary"][
                    "strict_equal_accuracy"
                ],
            }
        )
    return rows


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# SpatialEval 三臂外部确认结果",
        "",
        f"- 状态：`{summary['status']}`",
        "- 主集：Spatial-Map、Maze-Nav、Spatial-Grid，共 4500 条。",
        "- Spatial-Real 135 条不进入主门槛，因为官方仓库没有有效解析器。",
        "- 下表宏平均是本项目预注册汇总，不是 SpatialEval 官方总分。",
        "",
        "| 臂 | 官方兼容宏平均 | 严格相等宏平均 |",
        "| --- | ---: | ---: |",
    ]
    for arm in ARMS:
        values = summary["scores"][arm]["equal_task_macro_project_summary"]
        lines.append(
            f"| {arm} | {values['official_compatible_accuracy']:.6f} | "
            f"{values['strict_equal_accuracy']:.6f} |"
        )
    comparison = summary["comparison"]
    bootstrap = comparison["v2_minus_l2_stratified_paired_bootstrap"]
    lines.extend(
        [
            "",
            f"- v2/Dense 宏平均保留率：`{comparison['v2_dense_equal_task_macro_retention']}`",
            f"- v2-L2：`{bootstrap['observed_difference']:.6f}`，95% CI "
            f"`[{bootstrap['ci_95_lower']:.6f}, {bootstrap['ci_95_upper']:.6f}]`。",
            f"- 两项主门全部通过：`{summary['gates']['all_pass']}`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    records_by_arm = {
        arm: load_arm_records(path, arm) for arm, path in ARM_OUTPUT_PATHS.items()
    }
    summary = summarize_records(records_by_arm)
    for path in (args.json, args.csv, args.markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = _csv_rows(summary)
    fieldnames = sorted({key for row in rows for key in row})
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(args.json)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
