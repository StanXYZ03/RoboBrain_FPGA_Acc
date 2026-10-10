#!/usr/bin/env python3
"""Summarize the 2026-10-04 current-Selector latency run and join it with CV-Bench quality.

Contract (execution-only; no research decisions):
  * Four JSONL inputs (identity / K48 / K64 / K96), each hard-validated to contain
    exactly 300 runs = 100 unique ``sample_id`` x repeats {0, 1, 2}, with
    ``warmup == false``, ``run.warmup_per_sample == 1``, ``mode == "latency"`` and no
    duplicate ``(sample_id, repeat)`` pair. Any violation fails the run.
  * For every tier compute two scopes: raw 300 runs and per-sample median 100 runs.
  * Report mean / median / min / max / std(ddof=0) / p95 for the required latency
    fields plus the derived counts (retention, group counts, visual tokens, output
    tokens, peak CUDA bytes).
  * K-tier speedup is identity / candidate over the *same* scope mean, separately for
    the model and the application end-to-end number.
  * The old interleaved native-Dense core mean (225.16090276495865 ms) is carried only
    as a cross-run reference; it must never be read as a same-run speedup.
  * Merge the CV-Bench quality scores, the relative BF16 retention and the paired
    bootstrap CI.

Everything here is software Selector measurement, not an FPGA measurement.
Primary quality metric is the CV-Bench choice accuracy; Pointing / Grounding / NDCG
are forbidden as primary metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "token_selector" / "2026-10-04-current-selector-cvbench"
QUALITY_JSON = ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-cvbench.json"
OUTPUT_JSON = ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-quality-latency.json"
OUTPUT_MD = ROOT / "reports" / "cv_bench" / "2026-10-04-current-selector-quality-latency.md"

INPUT_PATH: dict[str, Path] = {
    "identity": ARTIFACT_DIR / "latency100_identity4096_r3.jsonl",
    "K48": ARTIFACT_DIR / "latency100_k48_r3.jsonl",
    "K64": ARTIFACT_DIR / "latency100_k64_r3.jsonl",
    "K96": ARTIFACT_DIR / "latency100_k96_r3.jsonl",
}
TIER_ORDER = ["identity", "K48", "K64", "K96"]
SPEEDUP_TIERS = ["K48", "K64", "K96"]

# Frozen hard-validation contract.
EXPECTED_RECORDS = 300
EXPECTED_SAMPLES = 100
EXPECTED_REPEATS = (0, 1, 2)
REQUIRED_MODE = "latency"
REQUIRED_WARMUP_PER_SAMPLE = 1

# Required per-run latency fields (seconds).
LATENCY_FIELDS = [
    "model_inference_seconds",
    "application_end_to_end_seconds",
    "vision_total_seconds",
    "gpu_vision_seconds",
    "selector_software_seconds",
    "llm_prepare_seconds",
    "llm_prefill_seconds",
    "decoder_seconds",
    "processor_seconds",
    "host_to_device_seconds",
    "prompt_text_embedding_seconds",
    "postprocess_seconds",
]
# Derived per-run quantities (mean for these, max for the peak memory field).
DERIVED_MEAN_FIELDS = [
    "actual_token_retention",
    "output_group_count",
    "full_llm_visual_tokens",
    "output_tokens",
]
DERIVED_MAX_FIELD = "peak_cuda_bytes"
REQUIRED_NUMERIC_FIELDS = LATENCY_FIELDS + DERIVED_MEAN_FIELDS + [DERIVED_MAX_FIELD]

# Cross-run only reference; NOT a same-run speedup numerator.
OLD_INTERLEAVED_DENSE_CORE_MEAN_MS = 225.16090276495865
OLD_INTERLEAVED_REFERENCE_JSON = "reports/cv_bench/2026-09-16-l2-interleaved-latency100.json"

FORBIDDEN_PRIMARY_METRICS = ["pointing", "grounding", "ndcg"]
PRIMARY_METRIC = "cv_bench_accuracy"

SELECTOR_IMPLEMENTATION = "software"
FPGA_MEASURED = False


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], probability: float) -> float:
    """Nearest-rank percentile, matching the repo's existing frozen convention."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot take a percentile of an empty sequence")
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def describe(values: Iterable[float]) -> dict[str, float]:
    """mean/median/min/max/std(ddof=0)/p95 for one numeric field."""
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot describe an empty sequence")
    return {
        "mean": statistics.fmean(numbers),
        "median": statistics.median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "std": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
        "p95": percentile(numbers, 0.95),
    }


def validate_records(
    records: list[dict[str, Any]],
    label: str,
    expected_records: int = EXPECTED_RECORDS,
    expected_samples: int = EXPECTED_SAMPLES,
) -> dict[str, Any]:
    """Hard completeness contract. Raise ValueError on the first violated rule set."""
    errors: list[str] = []

    if len(records) != expected_records:
        errors.append(f"expected exactly {expected_records} records, found {len(records)}")

    seen_pairs: dict[tuple[str, int], int] = {}
    repeats_by_sample: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if sample_id is None:
            errors.append(f"record {index}: missing sample_id")
            continue
        sample_id = str(sample_id)

        repeat = record.get("repeat")
        if isinstance(repeat, bool) or repeat not in EXPECTED_REPEATS:
            errors.append(f"record {index} (sample {sample_id}): repeat={repeat!r} not in {list(EXPECTED_REPEATS)}")
        else:
            repeats_by_sample[sample_id].add(repeat)
            key = (sample_id, repeat)
            if key in seen_pairs:
                errors.append(f"duplicate (sample_id, repeat)={key!r} at records {seen_pairs[key]} and {index}")
            else:
                seen_pairs[key] = index

        if record.get("mode") != REQUIRED_MODE:
            errors.append(f"record {index} (sample {sample_id}): mode={record.get('mode')!r} != {REQUIRED_MODE!r}")

        if record.get("warmup") is not False:
            errors.append(f"record {index} (sample {sample_id}): warmup={record.get('warmup')!r} must be false")

        run = record.get("run")
        if not isinstance(run, dict):
            errors.append(f"record {index} (sample {sample_id}): run must be an object")
        elif run.get("warmup_per_sample") != REQUIRED_WARMUP_PER_SAMPLE:
            errors.append(
                f"record {index} (sample {sample_id}): run.warmup_per_sample="
                f"{run.get('warmup_per_sample')!r} != {REQUIRED_WARMUP_PER_SAMPLE}"
            )

        for field in REQUIRED_NUMERIC_FIELDS:
            value = record.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"record {index} (sample {sample_id}): field {field!r} is not numeric")
            elif not math.isfinite(float(value)):
                errors.append(f"record {index} (sample {sample_id}): field {field!r} is not finite")

    if len(repeats_by_sample) != expected_samples:
        errors.append(f"expected exactly {expected_samples} unique sample_id, found {len(repeats_by_sample)}")

    for sample_id in sorted(repeats_by_sample):
        repeats = repeats_by_sample[sample_id]
        if repeats != set(EXPECTED_REPEATS):
            errors.append(f"sample {sample_id}: repeat set {sorted(repeats)} != {list(EXPECTED_REPEATS)}")

    if errors:
        preview = "; ".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f" (+{len(errors) - 20} more)"
        raise ValueError(f"{label}: latency contract validation failed: {preview}{suffix}")

    return {
        "records": len(records),
        "unique_samples": len(repeats_by_sample),
        "repeats": list(EXPECTED_REPEATS),
        "mode": REQUIRED_MODE,
        "warmup": False,
        "warmup_per_sample": REQUIRED_WARMUP_PER_SAMPLE,
        "duplicate_pairs": 0,
        "passed": True,
    }


def sort_sample_ids(sample_ids: Iterable[Any]) -> list[str]:
    def key(value: Any) -> tuple[int, Any]:
        text = str(value)
        return (0, int(text)) if text.isdigit() else (1, text)

    return sorted((str(value) for value in sample_ids), key=key)


def per_sample_median_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce three repeats per sample to one median record per sample_id (100 rows)."""
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sample[str(record["sample_id"])].append(record)

    reduced: list[dict[str, Any]] = []
    for sample_id in sort_sample_ids(by_sample):
        runs = by_sample[sample_id]
        row: dict[str, Any] = {"sample_id": sample_id}
        for field in REQUIRED_NUMERIC_FIELDS:
            row[field] = statistics.median(float(run[field]) for run in runs)
        reduced.append(row)
    return reduced


def summarize_scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {field: describe(run[field] for run in records) for field in LATENCY_FIELDS}
    derived_mean = {field: statistics.fmean(float(run[field]) for run in records) for field in DERIVED_MEAN_FIELDS}
    return {
        "n": len(records),
        "fields": fields,
        "derived_mean": derived_mean,
        f"{DERIVED_MAX_FIELD}_max": max(float(run[DERIVED_MAX_FIELD]) for run in records),
    }


def summarize_tier(records: list[dict[str, Any]]) -> dict[str, Any]:
    medians = per_sample_median_records(records)
    return {
        "raw": summarize_scope(records),
        "per_sample_median": summarize_scope(medians),
    }


def speedups_vs_identity(tiers: dict[str, Any]) -> dict[str, Any]:
    identity = tiers["identity"]
    result: dict[str, Any] = {}
    for label in SPEEDUP_TIERS:
        candidate = tiers[label]
        per_scope: dict[str, Any] = {}
        for scope in ("raw", "per_sample_median"):
            identity_fields = identity[scope]["fields"]
            candidate_fields = candidate[scope]["fields"]
            entry: dict[str, Any] = {}
            for field in ("model_inference_seconds", "application_end_to_end_seconds"):
                identity_mean = identity_fields[field]["mean"]
                candidate_mean = candidate_fields[field]["mean"]
                entry[field] = {
                    "identity_mean": identity_mean,
                    "candidate_mean": candidate_mean,
                    "speedup": identity_mean / candidate_mean,
                }
            per_scope[scope] = entry
        result[label] = per_scope
    return result


def load_quality(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates = {str(candidate["label"]): candidate for candidate in data["candidates"]}

    dense_refs = [candidate["dense_bf16_reference"] for candidate in data["candidates"]]
    if any(ref != dense_refs[0] for ref in dense_refs):
        raise ValueError("CV-Bench report has inconsistent dense_bf16_reference across candidates")
    dense = dense_refs[0]
    native = data["candidates"][0]["native_bf16_reference"]

    quality: dict[str, Any] = {
        "identity": {
            "tier": "identity",
            "quality_source": "dense_bf16_reference",
            "candidate_cv_bench_accuracy": dense["cv_bench_accuracy"],
            "candidate_accuracy": dense["accuracy"],
            "candidate_parse_success_rate": dense["parse_success_rate"],
            "candidate_sample_count": dense["sample_count"],
            "dense_bf16_cv_bench_accuracy": dense["cv_bench_accuracy"],
            "native_bf16_cv_bench_accuracy": native["cv_bench_accuracy"],
            "absolute_delta_vs_dense": 0.0,
            "relative_retention_vs_bf16": 1.0,
            "paired_bootstrap_ci95_vs_dense": [0.0, 0.0],
            "paired_bootstrap_ci95_vs_native": [0.0, 0.0],
            "bootstrap_seed": data["candidates"][0]["bootstrap_seed"],
            "bootstrap_repeats": data["candidates"][0]["bootstrap_repeats"],
            "candidate_jsonl": None,
            "note": "identity is the dense/native BF16 reference, so relative retention is defined as 1.0",
        }
    }
    for label in SPEEDUP_TIERS:
        candidate = candidates[label]
        quality[label] = {
            "tier": label,
            "quality_source": "candidate",
            "candidate_cv_bench_accuracy": candidate["candidate"]["cv_bench_accuracy"],
            "candidate_accuracy": candidate["candidate"]["accuracy"],
            "candidate_parse_success_rate": candidate["candidate"]["parse_success_rate"],
            "candidate_sample_count": candidate["candidate"]["sample_count"],
            "dense_bf16_cv_bench_accuracy": candidate["dense_bf16_reference"]["cv_bench_accuracy"],
            "native_bf16_cv_bench_accuracy": candidate["native_bf16_reference"]["cv_bench_accuracy"],
            "absolute_delta_vs_dense": candidate["absolute_delta_vs_dense"],
            "relative_retention_vs_bf16": candidate["relative_retention_vs_dense"],
            "paired_bootstrap_ci95_vs_dense": candidate["paired_bootstrap_ci95_vs_dense"],
            "paired_bootstrap_ci95_vs_native": candidate["paired_bootstrap_ci95_vs_native"],
            "bootstrap_seed": candidate["bootstrap_seed"],
            "bootstrap_repeats": candidate["bootstrap_repeats"],
            "candidate_jsonl": candidate["candidate_jsonl"],
        }

    return {
        "source": str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() and ROOT in Path(path).parents else str(path),
        "source_sha256": sha256_path(Path(path)),
        "manifest": data.get("manifest"),
        "manifest_sha256": data.get("manifest_sha256"),
        "formal_sample_count": data.get("formal_sample_count"),
        "generated_at": data.get("generated_at"),
        "status": data.get("status"),
        "source_report_forbidden_primary_metrics": data.get("forbidden_primary_metrics"),
        "tiers": quality,
    }


def quality_latency_table(tiers: dict[str, Any], quality: dict[str, Any], speedups: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for label in TIER_ORDER:
        tier_quality = quality["tiers"][label]
        entry: dict[str, Any] = {
            "cv_bench_accuracy": tier_quality["candidate_cv_bench_accuracy"],
            "relative_retention_vs_bf16": tier_quality["relative_retention_vs_bf16"],
            "paired_bootstrap_ci95_vs_dense": tier_quality["paired_bootstrap_ci95_vs_dense"],
            "paired_bootstrap_ci95_vs_native": tier_quality["paired_bootstrap_ci95_vs_native"],
            "model_inference_seconds_mean_raw": tiers[label]["raw"]["fields"]["model_inference_seconds"]["mean"],
            "application_end_to_end_seconds_mean_raw": tiers[label]["raw"]["fields"]["application_end_to_end_seconds"]["mean"],
            "model_inference_seconds_mean_per_sample_median": tiers[label]["per_sample_median"]["fields"]["model_inference_seconds"]["mean"],
            "application_end_to_end_seconds_mean_per_sample_median": tiers[label]["per_sample_median"]["fields"]["application_end_to_end_seconds"]["mean"],
        }
        if label in SPEEDUP_TIERS:
            entry["model_speedup_raw"] = speedups[label]["raw"]["model_inference_seconds"]["speedup"]
            entry["application_speedup_raw"] = speedups[label]["raw"]["application_end_to_end_seconds"]["speedup"]
            entry["model_speedup_per_sample_median"] = speedups[label]["per_sample_median"]["model_inference_seconds"]["speedup"]
            entry["application_speedup_per_sample_median"] = speedups[label]["per_sample_median"]["application_end_to_end_seconds"]["speedup"]
        table[label] = entry
    return table


def build_report(
    input_paths: dict[str, Path] | None = None,
    quality_path: Path | None = None,
    expected_records: int = EXPECTED_RECORDS,
    expected_samples: int = EXPECTED_SAMPLES,
) -> dict[str, Any]:
    input_paths = dict(input_paths or INPUT_PATH)
    quality_path = Path(quality_path or QUALITY_JSON)

    raw_records: dict[str, list[dict[str, Any]]] = {}
    validation: dict[str, Any] = {}
    inputs: dict[str, Any] = {}
    for label in TIER_ORDER:
        path = Path(input_paths[label])
        records = load_jsonl(path)
        validation[label] = validate_records(
            records, label, expected_records=expected_records, expected_samples=expected_samples
        )
        raw_records[label] = records
        inputs[label] = {
            "path": str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path),
            "sha256": sha256_path(path),
            "records": len(records),
        }

    tiers = {label: summarize_tier(raw_records[label]) for label in TIER_ORDER}
    speedups = speedups_vs_identity(tiers)
    quality = load_quality(quality_path)

    selector_software_claim = False
    for label in TIER_ORDER:
        for record in raw_records[label]:
            selector_software_claim = bool(
                selector_software_claim or record.get("selector_software_is_acceleration_claim", False)
            )
        if any(record.get("selector_path") != "software" for record in raw_records[label]):
            raise ValueError(f"{label}: selector_path is not 'software' for every record")

    report = {
        "schema_version": 1,
        "report": "2026-10-04-current-selector-quality-latency",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "latency100 (identity/K48/K64/K96; 300 runs per tier = 100 samples x 3 repeats) "
            "joined with the CV-Bench full2638 quality report"
        ),
        "primary_metric": PRIMARY_METRIC,
        "forbidden_primary_metrics": list(FORBIDDEN_PRIMARY_METRICS),
        "std_ddof": 0,
        "percentile_method": "nearest-rank (ceil(p*n)-1, repo convention)",
        "selector_implementation": SELECTOR_IMPLEMENTATION,
        "fpga_measured": FPGA_MEASURED,
        "selector_software_is_acceleration_claim": selector_software_claim,
        "latency_fields": LATENCY_FIELDS,
        "derived_mean_fields": DERIVED_MEAN_FIELDS,
        "derived_max_field": DERIVED_MAX_FIELD,
        "contract": {
            "expected_records_per_tier": expected_records,
            "expected_unique_samples_per_tier": expected_samples,
            "expected_repeats": list(EXPECTED_REPEATS),
            "required_mode": REQUIRED_MODE,
            "required_warmup": False,
            "required_warmup_per_sample": REQUIRED_WARMUP_PER_SAMPLE,
        },
        "validation": {"passed": True, "tiers": validation},
        "inputs": inputs,
        "latency": tiers,
        "speedups_vs_identity": speedups,
        "quality": quality,
        "quality_latency": quality_latency_table(tiers, quality, speedups),
        "cross_run_reference": {
            "old_interleaved_native_dense_core_mean_ms": OLD_INTERLEAVED_DENSE_CORE_MEAN_MS,
            "source": OLD_INTERLEAVED_REFERENCE_JSON,
            "scope": "inference_ms.mean for retention 1.0 in the 2026-09-16 interleaved latency100 run",
            "same_run": False,
            "usable_as_same_run_speedup": False,
            "note": (
                "Cross-run reference only: this number comes from a different (2026-09-16) "
                "interleaved run and must not be divided by these means to claim a speedup."
            ),
        },
        "notes": [
            "Selector is still the software path; these numbers are software Selector latency, not an FPGA measurement.",
            "Cross-tier speedups use identity in the same scope (raw 300 or per-sample median 100) as the denominator.",
            "The 225.16090276495865 ms native-Dense core mean is a cross-run reference and not a same-run speedup baseline.",
            "CV-Bench choice accuracy is the primary quality metric; Pointing, Grounding and NDCG are forbidden as primary metrics.",
            "std is population std with ddof=0; p95 uses the nearest-rank convention ceil(0.95*n)-1.",
        ],
    }
    return report


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if value == 0:
            return "0"
        return f"{value:.{digits}f}"
    return str(value)


def stats_table(rows: list[str], field_map: dict[str, dict[str, float]]) -> list[str]:
    lines = ["| 字段 | mean | median | min | max | std(ddof=0) | p95 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for field in rows:
        stat = field_map[field]
        lines.append(
            "| `{field}` | {mean} | {median} | {min} | {max} | {std} | {p95} |".format(
                field=field,
                mean=fmt(stat["mean"]),
                median=fmt(stat["median"]),
                min=fmt(stat["min"]),
                max=fmt(stat["max"]),
                std=fmt(stat["std"]),
                p95=fmt(stat["p95"]),
            )
        )
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    tiers = report["latency"]
    quality = report["quality"]["tiers"]
    speedups = report["speedups_vs_identity"]

    lines: list[str] = []
    lines.append("# Current Selector 质量—延迟联合汇总（2026-10-04）")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：{report['generated_at']}")
    lines.append(f"- 范围：{report['scope']}")
    lines.append(f"- 主指标：CV-Bench 选择准确率（`{report['primary_metric']}`）")
    lines.append(f"- 禁止作为主指标：{'、'.join(report['forbidden_primary_metrics'])}")
    lines.append("- **Selector 仍为软件实现（`selector_path=software`），本报告不是 FPGA 实测。**")
    lines.append("- std 为总体标准差（ddof=0）；p95 采用 nearest-rank：`ceil(0.95*n)-1`。")
    lines.append("")

    lines.append("## 1. 硬校验合同")
    lines.append("")
    lines.append("每个输入必须恰好 300 条 = 100 个唯一 `sample_id` × repeat {0,1,2}，`warmup=false`，`run.warmup_per_sample=1`，`mode=latency`，且无重复 `(sample_id, repeat)`；任一不满足即失败。")
    lines.append("")
    lines.append("| 档位 | 记录数 | 唯一样本 | repeats | 校验 | SHA256 |")
    lines.append("| --- | ---: | ---: | --- | --- | --- |")
    for label in TIER_ORDER:
        info = report["inputs"][label]
        check = report["validation"]["tiers"][label]
        lines.append(
            f"| {label} | {check['records']} | {check['unique_samples']} | {check['repeats']} | "
            f"{'通过' if check['passed'] else '失败'} | `{info['sha256'][:16]}…` |"
        )
    lines.append("")

    lines.append("## 2. 延迟统计（raw 300 条）")
    lines.append("")
    for label in TIER_ORDER:
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(stats_table(LATENCY_FIELDS, tiers[label]["raw"]["fields"]))
        lines.append("")
    lines.append("派生量（raw 300 条）：")
    lines.append("")
    lines.append("| 档位 | actual_token_retention mean | output_group_count mean | full_llm_visual_tokens mean | output_tokens mean | peak_cuda_bytes max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label in TIER_ORDER:
        raw = tiers[label]["raw"]
        lines.append(
            "| {label} | {ret} | {grp} | {vis} | {out} | {peak} |".format(
                label=label,
                ret=fmt(raw["derived_mean"]["actual_token_retention"]),
                grp=fmt(raw["derived_mean"]["output_group_count"]),
                vis=fmt(raw["derived_mean"]["full_llm_visual_tokens"]),
                out=fmt(raw["derived_mean"]["output_tokens"]),
                peak=fmt(raw[f"{report['derived_max_field']}_max"]),
            )
        )
    lines.append("")

    lines.append("## 3. 延迟统计（每样本三次中位数，100 条）")
    lines.append("")
    for label in TIER_ORDER:
        lines.append(f"### {label}")
        lines.append("")
        lines.extend(stats_table(LATENCY_FIELDS, tiers[label]["per_sample_median"]["fields"]))
        lines.append("")
    lines.append("派生量（每样本三次中位数，100 条）：")
    lines.append("")
    lines.append("| 档位 | actual_token_retention mean | output_group_count mean | full_llm_visual_tokens mean | output_tokens mean | peak_cuda_bytes max |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for label in TIER_ORDER:
        med = tiers[label]["per_sample_median"]
        lines.append(
            "| {label} | {ret} | {grp} | {vis} | {out} | {peak} |".format(
                label=label,
                ret=fmt(med["derived_mean"]["actual_token_retention"]),
                grp=fmt(med["derived_mean"]["output_group_count"]),
                vis=fmt(med["derived_mean"]["full_llm_visual_tokens"]),
                out=fmt(med["derived_mean"]["output_tokens"]),
                peak=fmt(med[f"{report['derived_max_field']}_max"]),
            )
        )
    lines.append("")

    lines.append("## 4. K 档加速比（对 identity 同口径 mean 相除）")
    lines.append("")
    lines.append("| 档位 | 口径 | model identity mean (s) | model mean (s) | model speedup | application identity mean (s) | application mean (s) | application speedup |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in SPEEDUP_TIERS:
        for scope in ("raw", "per_sample_median"):
            model = speedups[label][scope]["model_inference_seconds"]
            application = speedups[label][scope]["application_end_to_end_seconds"]
            scope_cn = "raw 300" if scope == "raw" else "每样本中位数 100"
            lines.append(
                "| {label} | {scope} | {im} | {cm} | **{sp}** | {ia} | {ca} | **{sa}** |".format(
                    label=label,
                    scope=scope_cn,
                    im=fmt(model["identity_mean"]),
                    cm=fmt(model["candidate_mean"]),
                    sp=fmt(model["speedup"], 4),
                    ia=fmt(application["identity_mean"]),
                    ca=fmt(application["candidate_mean"]),
                    sa=fmt(application["speedup"], 4),
                )
            )
    lines.append("")

    reference = report["cross_run_reference"]
    lines.append("## 5. 跨运行参照（不可当同运行加速比）")
    lines.append("")
    lines.append(
        f"- 旧交错原生 Dense 核心均值：**{fmt(reference['old_interleaved_native_dense_core_mean_ms'])} ms**"
        f"（来源 `{reference['source']}`，口径 `{reference['scope']}`）。"
    )
    lines.append(f"- `same_run={fmt(reference['same_run'])}`，`usable_as_same_run_speedup={fmt(reference['usable_as_same_run_speedup'])}`。")
    lines.append(f"- {reference['note']}")
    lines.append("")

    lines.append("## 6. CV-Bench 质量合并（相对 BF16 保持率与 CI）")
    lines.append("")
    lines.append("| 档位 | CV-Bench 准确率 | 相对 BF16 保持率 | 绝对差 vs Dense | CI95 vs Dense | CI95 vs Native | 样本数 |")
    lines.append("| --- | ---: | ---: | ---: | --- | --- | ---: |")
    for label in TIER_ORDER:
        entry = quality[label]
        ci_dense = entry["paired_bootstrap_ci95_vs_dense"]
        ci_native = entry["paired_bootstrap_ci95_vs_native"]
        lines.append(
            "| {label} | {acc} | {ret} | {delta} | [{lo1}, {hi1}] | [{lo2}, {hi2}] | {n} |".format(
                label=label,
                acc=fmt(entry["candidate_cv_bench_accuracy"]),
                ret=fmt(entry["relative_retention_vs_bf16"]),
                delta=fmt(entry["absolute_delta_vs_dense"]),
                lo1=fmt(ci_dense[0]),
                hi1=fmt(ci_dense[1]),
                lo2=fmt(ci_native[0]),
                hi2=fmt(ci_native[1]),
                n=entry["candidate_sample_count"],
            )
        )
    lines.append("")

    lines.append("## 7. 质量—延迟联合表")
    lines.append("")
    lines.append("| 档位 | 质量 CV-Bench | 相对 BF16 保持率 | model mean raw (s) | app mean raw (s) | model speedup raw | app speedup raw | model speedup median | app speedup median |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label in TIER_ORDER:
        entry = report["quality_latency"][label]
        lines.append(
            "| {label} | {acc} | {ret} | {mr} | {ar} | {msr} | {asr} | {msm} | {asm} |".format(
                label=label,
                acc=fmt(entry["cv_bench_accuracy"]),
                ret=fmt(entry["relative_retention_vs_bf16"]),
                mr=fmt(entry["model_inference_seconds_mean_raw"]),
                ar=fmt(entry["application_end_to_end_seconds_mean_raw"]),
                msr=fmt(entry.get("model_speedup_raw"), 4) if "model_speedup_raw" in entry else "—",
                asr=fmt(entry.get("application_speedup_raw"), 4) if "application_speedup_raw" in entry else "—",
                msm=fmt(entry.get("model_speedup_per_sample_median"), 4) if "model_speedup_per_sample_median" in entry else "—",
                asm=fmt(entry.get("application_speedup_per_sample_median"), 4) if "application_speedup_per_sample_median" in entry else "—",
            )
        )
    lines.append("")

    lines.append("## 8. 限制与声明")
    lines.append("")
    for note in report["notes"]:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], output_json: Path = OUTPUT_JSON, output_md: Path = OUTPUT_MD) -> None:
    output_json = Path(output_json)
    output_md = Path(output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", default=str(OUTPUT_JSON))
    parser.add_argument("--output-md", default=str(OUTPUT_MD))
    args = parser.parse_args(argv)

    report = build_report()
    write_report(report, Path(args.output_json), Path(args.output_md))

    summary = {
        "validation_passed": report["validation"]["passed"],
        "outputs": [str(args.output_json), str(args.output_md)],
        "selector_implementation": report["selector_implementation"],
        "fpga_measured": report["fpga_measured"],
        "speedups_vs_identity": {
            label: {
                "model_raw": report["speedups_vs_identity"][label]["raw"]["model_inference_seconds"]["speedup"],
                "application_raw": report["speedups_vs_identity"][label]["raw"]["application_end_to_end_seconds"]["speedup"],
                "model_per_sample_median": report["speedups_vs_identity"][label]["per_sample_median"]["model_inference_seconds"]["speedup"],
                "application_per_sample_median": report["speedups_vs_identity"][label]["per_sample_median"]["application_end_to_end_seconds"]["speedup"],
            }
            for label in SPEEDUP_TIERS
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
