#!/usr/bin/env python3
"""Summarise the layer-5 score-formula replay screening100 diagnostic.

The single primary decision is whether changing only the per-group score
aggregation from ``norm(mean(x_0..x_3))`` to ``mean(norm(x_0)..norm(x_3))``
reproduces the frozen historical ``shallow_r0250`` index sets:

* ``100/100`` exact candidate/reference matches =>
  ``L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET``;
* otherwise ``PARTIAL_OR_NOT_EXPLAINED``.

This summariser is fail closed: it validates the frozen manifest, reference and
control inputs, requires exactly 100 well-formed replay records with matching
provenance, and refuses to write any report if anything is missing or
inconsistent.  Timings are descriptive only and never enter the decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.layer5_score_formula_replay import (
    EXPECTED_SAMPLES,
    FORMULA_CANDIDATE,
    FORMULA_CONTROL,
    InputContractError,
    ReplayInputs,
    load_replay_inputs,
    read_jsonl,
    validate_replay_record,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    ROOT
    / "evaluation"
    / "results"
    / "cv_bench_l2_budget64_pilot100_20260927"
    / "shallow_r0250.jsonl"
)
DEFAULT_CONTROL = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-forced-index-ac"
    / "control_same_budget_screening100.jsonl"
)
DEFAULT_REPLAY = (
    ROOT
    / "artifacts"
    / "token_selector"
    / "2026-10-04-layer5-score-formula-replay"
    / "layer5_score_formula_replay_screening100.jsonl"
)
REPORT_STEM = "2026-10-04-layer5-score-formula-replay-screening100"

DECISION_FULL = "L2_AGGREGATION_DIFFERENCE_FULLY_EXPLAINS_INDEX_SET"
DECISION_PARTIAL = "PARTIAL_OR_NOT_EXPLAINED"

SCOPE_STATEMENT = (
    "固定 100 样本 screening100 因果诊断（layer-5 cut=5、region_grid=3、每样本 "
    "K_C 取自冻结参考、仅改变每组分聚合公式）。不是正式全量 2638 结论，不是 "
    "FPGA 可部署配置；K_C 对齐放宽仅为诊断设置。"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--control-jsonl", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--replay-jsonl", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "reports" / "cv_bench" / f"{REPORT_STEM}.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=ROOT / "reports" / "cv_bench" / f"{REPORT_STEM}.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "reports" / "cv_bench" / f"{REPORT_STEM}.md",
    )
    return parser


def _stat(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _distribution(values: Sequence[int]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(Counter(values).items())}


def _load_records(
    replay_path: Path,
    inputs: ReplayInputs,
    *,
    provenance: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(replay_path, "replay")
    index: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        if "sample_id" not in row:
            raise InputContractError(f"replay row {position} is missing sample_id")
        sample_id = str(row["sample_id"])
        if sample_id in index:
            raise InputContractError(f"duplicate replay sample_id {sample_id!r}")
        index[sample_id] = row
    expected_ids = [str(row["sample_id"]) for row in inputs.manifest_rows]
    missing = sorted(set(expected_ids) - set(index))
    extra = sorted(set(index) - set(expected_ids))
    if missing:
        raise InputContractError(
            f"replay is missing {len(missing)} sample(s): {missing[:10]} (fail closed)"
        )
    if extra:
        raise InputContractError(
            f"replay has {len(extra)} extra sample(s): {extra[:10]} (fail closed)"
        )
    if len(index) != EXPECTED_SAMPLES:
        raise InputContractError(
            f"replay has {len(index)} records, expected {EXPECTED_SAMPLES}"
        )
    ordered: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        record = index[sample_id]
        validate_replay_record(
            record,
            inputs.reference_map[sample_id],
            inputs.control_map[sample_id],
            provenance=provenance,
        )
        _validate_run_metadata(record, sample_id)
        ordered.append(record)
    return ordered


def _validate_run_metadata(record: Mapping[str, Any], sample_id: str) -> None:
    run = record.get("run")
    if not isinstance(run, dict):
        raise InputContractError(f"replay/{sample_id} is missing run metadata")
    if run.get("dtype") != "bf16":
        raise InputContractError(f"replay/{sample_id} run.dtype must be bf16")
    if run.get("attn_implementation") != "sdpa":
        raise InputContractError(
            f"replay/{sample_id} run.attn_implementation must be sdpa"
        )
    if run.get("llm_decode") is not False:
        raise InputContractError(f"replay/{sample_id} run.llm_decode must be False")
    if run.get("training") is not False:
        raise InputContractError(f"replay/{sample_id} run.training must be False")


def build_report(
    inputs: ReplayInputs,
    records: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    exact_count = sum(
        1 for record in records if record.get("candidate_equals_reference") is True
    )
    mismatch_ids = sorted(
        str(record["sample_id"])
        for record in records
        if record.get("candidate_equals_reference") is not True
    )
    candidate_jaccards = [float(record["candidate_vs_reference_jaccard"]) for record in records]
    control_jaccards = [float(record["control_vs_reference_jaccard"]) for record in records]
    g_values = [int(record["reference_G"]) for record in records]
    k_values = [int(record["K_C"]) for record in records]
    decision = (
        DECISION_FULL if exact_count == EXPECTED_SAMPLES else DECISION_PARTIAL
    )
    record_rows = [
        {
            "sample_id": str(record["sample_id"]),
            "task": str(record["task"]),
            "G": int(record["reference_G"]),
            "K_C": int(record["K_C"]),
            "merged_grid": [int(v) for v in record["merged_grid"]],
            "candidate_equals_reference": bool(
                record["candidate_equals_reference"]
            ),
            "candidate_vs_reference_jaccard": float(
                record["candidate_vs_reference_jaccard"]
            ),
            "control_vs_reference_jaccard": float(
                record["control_vs_reference_jaccard"]
            ),
            "candidate_selected_group_indices_sha256": record[
                "candidate_selected_group_indices_sha256"
            ],
            "control_selected_group_indices_sha256": record[
                "control_selected_group_indices_sha256"
            ],
            "reference_selected_group_indices_sha256": record[
                "reference_selected_group_indices_sha256"
            ],
            "vision_total_seconds": float(record["vision_total_seconds"]),
            "model_inference_seconds": float(record["model_inference_seconds"]),
            "peak_cuda_bytes": int(record["peak_cuda_bytes"]),
        }
        for record in records
    ]
    return {
        "schema_version": "1.0",
        "report": REPORT_STEM,
        "decision": decision,
        "decision_rule": (
            "100/100 exact candidate/reference matches => "
            f"{DECISION_FULL}; otherwise {DECISION_PARTIAL}"
        ),
        "diagnostic_only": True,
        "fpga_deployable_candidate": False,
        "scope": SCOPE_STATEMENT,
        "score_formula_control": FORMULA_CONTROL,
        "score_formula_candidate": FORMULA_CANDIDATE,
        "sample_count": len(records),
        "expected_sample_count": EXPECTED_SAMPLES,
        "exact_candidate_reference_matches": int(exact_count),
        "mismatch_ids": mismatch_ids,
        "candidate_vs_reference": {
            "exact_count": int(exact_count),
            **_stat(candidate_jaccards),
        },
        "control_vs_reference": {
            "exact_count": sum(
                1 for record in records if record.get("control_vs_reference_exact") is True
            ),
            **_stat(control_jaccards),
        },
        "g_distribution": _distribution(g_values),
        "k_distribution": _distribution(k_values),
        "timing_descriptive_only": {
            "vision_total_seconds": _stat(
                [float(record["vision_total_seconds"]) for record in records]
            ),
            "model_inference_seconds": _stat(
                [float(record["model_inference_seconds"]) for record in records]
            ),
            "peak_cuda_bytes": _stat(
                [float(record["peak_cuda_bytes"]) for record in records]
            ),
        },
        "provenance": dict(provenance),
        "inputs": {
            "manifest_sha256": inputs.manifest_sha256,
            "reference_jsonl_sha256": inputs.reference_sha256,
            "control_jsonl_sha256": inputs.control_sha256,
            "reference_selector": "shallow_r0250",
            "manifest_records": len(inputs.manifest_rows),
        },
        "records": record_rows,
    }


def write_json(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


CSV_FIELDS = (
    "sample_id",
    "task",
    "G",
    "K_C",
    "merged_grid",
    "candidate_equals_reference",
    "candidate_vs_reference_jaccard",
    "control_vs_reference_jaccard",
    "candidate_selected_group_indices_sha256",
    "control_selected_group_indices_sha256",
    "reference_selected_group_indices_sha256",
    "vision_total_seconds",
    "model_inference_seconds",
    "peak_cuda_bytes",
)


def write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in report["records"]:
            payload = dict(row)
            payload["merged_grid"] = "x".join(str(value) for value in row["merged_grid"])
            writer.writerow(payload)


def render_markdown(report: Mapping[str, Any]) -> str:
    candidate = report["candidate_vs_reference"]
    control = report["control_vs_reference"]
    lines = [
        f"# Layer-5 score-formula replay screening100（{report['report']}）",
        "",
        f"- **决定**：`{report['decision']}`",
        f"- **判据**：{report['decision_rule']}",
        f"- **样本**：{report['sample_count']}/{report['expected_sample_count']}",
        f"- **精确匹配（candidate==reference）**："
        f"{report['exact_candidate_reference_matches']}/{report['sample_count']}",
        f"- **不匹配 sample_id**：{report['mismatch_ids'] if report['mismatch_ids'] else '无'}",
        "",
        "## 分数公式（唯一变量）",
        "",
        f"- control：`{report['score_formula_control']}`",
        f"- candidate：`{report['score_formula_candidate']}`",
        "",
        "## Jaccard（candidate/control vs 冻结参考）",
        "",
        "| 对比 | exact | mean | min |",
        "| --- | --- | --- | --- |",
        f"| candidate vs reference | {candidate['exact_count']}/{report['sample_count']} | "
        f"{candidate['mean']:.6f} | {candidate['min']:.6f} |",
        f"| control vs reference | {control['exact_count']}/{report['sample_count']} | "
        f"{control['mean']:.6f} | {control['min']:.6f} |",
        "",
        "## G / K 分布",
        "",
        f"- G：{report['g_distribution']}",
        f"- K_C：{report['k_distribution']}",
        "",
        "## 计时（仅描述，不参与决定）",
        "",
        f"- vision_total_seconds：{report['timing_descriptive_only']['vision_total_seconds']}",
        f"- model_inference_seconds：{report['timing_descriptive_only']['model_inference_seconds']}",
        f"- peak_cuda_bytes：{report['timing_descriptive_only']['peak_cuda_bytes']}",
        "",
        "## 范围与限制",
        "",
        f"- {report['scope']}",
        "- 本诊断不是正式全量 2638 结论，也不是 FPGA 可部署配置。",
        "- 计时仅描述，绝不参与决定。",
        "",
        "## 输入 SHA256",
        "",
        f"- manifest：`{report['inputs']['manifest_sha256']}`",
        f"- reference (`shallow_r0250`)：`{report['inputs']['reference_jsonl_sha256']}`",
        f"- control：`{report['inputs']['control_jsonl_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    inputs = load_replay_inputs(
        args.manifest,
        args.reference_jsonl,
        args.control_jsonl,
        expected_samples=EXPECTED_SAMPLES,
    )
    provenance = {
        "manifest_sha256": inputs.manifest_sha256,
        "reference_jsonl_sha256": inputs.reference_sha256,
        "control_jsonl_sha256": inputs.control_sha256,
    }
    records = _load_records(args.replay_jsonl, inputs, provenance=provenance)
    report = build_report(inputs, records, provenance=provenance)
    write_json(args.output_json, report)
    write_csv(args.output_csv, report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"DECISION {report['decision']} "
        f"exact={report['exact_candidate_reference_matches']}/{report['sample_count']} "
        f"candidate_mean_jaccard={report['candidate_vs_reference']['mean']:.6f} "
        f"control_mean_jaccard={report['control_vs_reference']['mean']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
