#!/usr/bin/env python3
"""Heterogeneous-latency projection for the 2026-10-04 current integer Selector.

**This is a pure analysis estimate, not a ZCU102 or Jetson measurement.**  No
board is involved, no link is measured, and no model is run here.

Inputs (all hashed and stored in the report):

* the formal latency-100 JSONL for K48 / K64 / K96 under
  ``artifacts/token_selector/2026-10-04-current-selector-cvbench/`` (100 unique
  samples x 3 repeats, warmup excluded);
* the accepted compact-pipeline HLS report
  ``reports/token_selector/2026-10-03-integer-selector-compact-interface-hls.json``
  and the accepted ``summarize_csynth.py`` cycle model
  (``_selector_cycles`` / ``_compact_overhead_cycles``, scorer phase constants);
* the frozen ``transfer_audit.py`` helpers and the frozen
  ``fpga_transfer_layout`` authority.

Model (Wv resident, no overlap, i.e. strictly serial):

    projected_model_E2E
        = measured_model_inference
        - measured_software_selector
        + fpga_compute
        + jetson_to_fpga_transfer
        + fpga_to_jetson_transfer

``G <= K`` uses the local identity-bypass branch of the accepted selector cycle
model (no region search); the compact arena overhead is still paid.  Transfer is
reported under the explicit 1 GbE 110 MB/s assumption and a hypothetical 10 GbE
1.1 GB/s assumption.  FPGA compute uses the accepted 200 MHz target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import load_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "token_selector" / "2026-10-04-current-selector-cvbench"
LATENCY_JSONL = {
    "K48": ARTIFACT_DIR / "latency100_k48_r3.jsonl",
    "K64": ARTIFACT_DIR / "latency100_k64_r3.jsonl",
    "K96": ARTIFACT_DIR / "latency100_k96_r3.jsonl",
}
ACCEPTED_HLS_REPORT = (
    ROOT / "reports" / "token_selector" / "2026-10-03-integer-selector-compact-interface-hls.json"
)
ACCEPTED_CSYNTH_MODEL = (
    ROOT / "fpga_accel" / "integer_selector_pipeline_compact_hls" / "scripts" / "summarize_csynth.py"
)
TRANSFER_AUDIT_MODULE = (
    ROOT / "fpga_accel" / "integer_selector_pipeline_compact_hls" / "tools" / "transfer_audit.py"
)

DEFAULT_OUTPUT_JSON = (
    ROOT / "reports" / "token_selector" / "2026-10-04-current-selector-heterogeneous-latency-projection.json"
)
DEFAULT_OUTPUT_MD = (
    ROOT / "reports" / "token_selector" / "2026-10-04-current-selector-heterogeneous-latency-projection.md"
)

CANDIDATE_ORDER = ("K48", "K64", "K96")

EXPECTED_LATENCY_RECORDS = 300
EXPECTED_UNIQUE_SAMPLES = 100
EXPECTED_REPEATS = (0, 1, 2)

GBE_CENTRAL_BYTES_PER_SECOND = 110.0e6
GBE_10G_ASSUMED_BYTES_PER_SECOND = 1100.0e6
WV_BYTES = 128 * 1024

SCOPE = "PURE ANALYSIS ESTIMATE — not a ZCU102 or Jetson measurement"
MEASUREMENT_STATUS = "not_measured"

DISCLAIMERS = (
    "Pure analytic projection: no bitstream, no board, no link measurement and no model run.",
    "1 GbE time uses an explicit 110 MB/s effective-payload-bandwidth assumption.",
    "10 GbE time assumes a hypothetical 1.1 GB/s effective payload bandwidth; it is not a measured link.",
    "Wv is assumed resident: one 131,072 B load amortised away, 0 B per image.",
    "FPGA compute uses the accepted compact-pipeline formula at the accepted 200 MHz target.",
    "No pipelining / overlap: Jetson->FPGA, FPGA compute and FPGA->Jetson are strictly serial.",
    "The measured baselines are the latency-100 software-Selector runs, not FPGA runs.",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_accepted_cycle_model(
    report_path: Path = ACCEPTED_HLS_REPORT,
    csynth_path: Path = ACCEPTED_CSYNTH_MODEL,
) -> dict[str, Any]:
    """Load the accepted HLS cycle formula constants + functions."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    phase = report["csynth"]["scorer_phase"]
    module = _load_module("accepted_compact_csynth_model", csynth_path)
    return {
        "wv_load_cycles": float(phase["wv_load_cycles"]),
        "pass1_cycles_per_group": float(phase["pass1_cycles_per_group"]),
        "pass2_cycles_per_group": float(phase["pass2_cycles_per_group"]),
        "per_group_cycles": float(phase["per_group_cycles"]),
        "target_fmax_mhz": float(module.TARGET_FMAX_MHZ),
        "selector_cycles": module._selector_cycles,
        "compact_overhead_cycles": module._compact_overhead_cycles,
    }


def load_transfer_model(path: Path = TRANSFER_AUDIT_MODULE) -> Any:
    return _load_module("accepted_transfer_audit", path)


def load_latency_samples(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != EXPECTED_LATENCY_RECORDS:
        raise ValueError(
            f"{path}: expected {EXPECTED_LATENCY_RECORDS} runs, got {len(rows)}"
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if row["mode"] != "latency" or bool(row["warmup"]):
            raise ValueError(f"{path}: non-latency or warmup record for {row['sample_id']}")
        key = (str(row["sample_id"]), int(row["repeat"]))
        if key in seen:
            raise ValueError(f"{path}: duplicate run {key}")
        seen.add(key)
        grouped.setdefault(str(row["sample_id"]), []).append(row)
    if len(grouped) != EXPECTED_UNIQUE_SAMPLES:
        raise ValueError(
            f"{path}: expected {EXPECTED_UNIQUE_SAMPLES} unique samples, got {len(grouped)}"
        )
    samples: list[dict[str, Any]] = []
    for sample_id in sorted(grouped, key=int):
        runs = sorted(grouped[sample_id], key=lambda r: int(r["repeat"]))
        repeats = tuple(int(r["repeat"]) for r in runs)
        if repeats != EXPECTED_REPEATS:
            raise ValueError(f"{path}: sample {sample_id} repeats {repeats} != {EXPECTED_REPEATS}")
        first = runs[0]
        samples.append(
            {
                "sample_id": sample_id,
                "task": first["task"],
                "source": first["source"],
                "group_count": int(first["full_llm_visual_tokens"]),
                "grid_h": int(first["image_grid_thw"][1]) // 2,
                "grid_w": int(first["image_grid_thw"][2]) // 2,
                "model_inference_seconds": statistics.median(
                    float(r["model_inference_seconds"]) for r in runs
                ),
                "selector_software_seconds": statistics.median(
                    float(r["selector_software_seconds"]) for r in runs
                ),
                "application_end_to_end_seconds": statistics.median(
                    float(r["application_end_to_end_seconds"]) for r in runs
                ),
                "repeats": len(runs),
            }
        )
    return samples


def project_sample(
    budget: int,
    sample: Mapping[str, Any],
    cycle_model: Mapping[str, Any],
    transfer: Any,
) -> dict[str, Any]:
    group_count = int(sample["group_count"])
    identity_bypass = group_count <= budget
    scorer_cycles = cycle_model["wv_load_cycles"] + group_count * cycle_model["per_group_cycles"]
    selector_cycles = cycle_model["selector_cycles"](
        group_count, budget, int(sample["grid_h"]), int(sample["grid_w"])
    )
    overhead_cycles = cycle_model["compact_overhead_cycles"](budget)
    total_cycles = scorer_cycles + selector_cycles + overhead_cycles
    compute_seconds = total_cycles / (cycle_model["target_fmax_mhz"] * 1e6)

    jetson_to_fpga_bytes = transfer.input_arena_bytes(group_count)  # Wv resident => 0 B/image
    fpga_to_jetson_bytes = int(transfer.fpga_transfer_layout(budget).total_bytes)
    transfer_bytes = jetson_to_fpga_bytes + fpga_to_jetson_bytes
    transfer_1gbe_seconds = transfer_bytes / GBE_CENTRAL_BYTES_PER_SECOND
    transfer_10gbe_seconds = transfer_bytes / GBE_10G_ASSUMED_BYTES_PER_SECOND

    baseline_inference = float(sample["model_inference_seconds"])
    software_selector = float(sample["selector_software_seconds"])
    baseline_application = float(sample["application_end_to_end_seconds"])
    replacement = baseline_inference - software_selector

    def projected(transfer_seconds: float) -> float:
        return replacement + compute_seconds + transfer_seconds

    model_1gbe = projected(transfer_1gbe_seconds)
    model_10gbe = projected(transfer_10gbe_seconds)
    app_1gbe = baseline_application - software_selector + compute_seconds + transfer_1gbe_seconds
    app_10gbe = baseline_application - software_selector + compute_seconds + transfer_10gbe_seconds

    return {
        "sample_id": sample["sample_id"],
        "task": sample["task"],
        "source": sample["source"],
        "group_count": group_count,
        "output_group_count": group_count if identity_bypass else budget,
        "identity_bypass": identity_bypass,
        "grid": [int(sample["grid_h"]), int(sample["grid_w"])],
        "scorer_cycles": scorer_cycles,
        "selector_cycles_estimate": selector_cycles,
        "compact_overhead_cycles_estimate": overhead_cycles,
        "total_cycles_estimate": total_cycles,
        "fpga_compute_seconds": compute_seconds,
        "jetson_to_fpga_bytes": jetson_to_fpga_bytes,
        "fpga_to_jetson_arena_bytes": fpga_to_jetson_bytes,
        "transfer_bytes": transfer_bytes,
        "transfer_1gbe_110MBps_seconds": transfer_1gbe_seconds,
        "transfer_10gbe_1p1GBps_assumed_seconds": transfer_10gbe_seconds,
        "measured_model_inference_seconds": baseline_inference,
        "measured_selector_software_seconds": software_selector,
        "measured_application_end_to_end_seconds": baseline_application,
        "projected_model_e2e_1gbe_110MBps_seconds": model_1gbe,
        "projected_model_e2e_10gbe_1p1GBps_assumed_seconds": model_10gbe,
        "projected_application_e2e_1gbe_110MBps_seconds": app_1gbe,
        "projected_application_e2e_10gbe_1p1GBps_assumed_seconds": app_10gbe,
    }


def _stats(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "std_ddof0": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p95": ordered[max(0, -(-len(ordered) * 95 // 100) - 1)],
    }


def _stats_ms(projected: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    return _stats([1000.0 * float(row[key]) for row in projected])


def project_budget(
    label: str,
    budget: int,
    samples: Sequence[Mapping[str, Any]],
    cycle_model: Mapping[str, Any],
    transfer: Any,
) -> dict[str, Any]:
    projected = [project_sample(budget, s, cycle_model, transfer) for s in samples]
    identity_count = sum(1 for row in projected if row["identity_bypass"])
    measured_model = [1000.0 * float(r["measured_model_inference_seconds"]) for r in projected]
    measured_software = [
        1000.0 * float(r["measured_selector_software_seconds"]) for r in projected
    ]
    measured_application = [
        1000.0 * float(r["measured_application_end_to_end_seconds"]) for r in projected
    ]
    proj_model_1gbe = [
        1000.0 * float(r["projected_model_e2e_1gbe_110MBps_seconds"]) for r in projected
    ]
    proj_model_10gbe = [
        1000.0 * float(r["projected_model_e2e_10gbe_1p1GBps_assumed_seconds"]) for r in projected
    ]
    return {
        "label": label,
        "budget": budget,
        "sample_count": len(projected),
        "identity_bypass_count": identity_count,
        "identity_bypass_fraction": identity_count / len(projected),
        "fpga_compute_ms": _stats_ms(projected, "fpga_compute_seconds"),
        "transfer_ms": {
            "1gbe_110MBps": _stats_ms(projected, "transfer_1gbe_110MBps_seconds"),
            "10gbe_1p1GBps_assumed": _stats_ms(
                projected, "transfer_10gbe_1p1GBps_assumed_seconds"
            ),
        },
        "bytes": {
            "jetson_to_fpga_mean": statistics.fmean(
                float(r["jetson_to_fpga_bytes"]) for r in projected
            ),
            "fpga_to_jetson_arena_fixed": int(
                transfer.fpga_transfer_layout(budget).total_bytes
            ),
        },
        "total_cycles_estimate": _stats(
            [float(r["total_cycles_estimate"]) for r in projected]
        ),
        "measured_baseline_ms": {
            "model_inference": _stats(measured_model),
            "selector_software": _stats(measured_software),
            "application_end_to_end": _stats(measured_application),
        },
        "projected_model_e2e_ms": {
            "1gbe_110MBps": _stats(proj_model_1gbe),
            "10gbe_1p1GBps_assumed": _stats(proj_model_10gbe),
        },
        "projected_application_e2e_ms": {
            "1gbe_110MBps": _stats_ms(projected, "projected_application_e2e_1gbe_110MBps_seconds"),
            "10gbe_1p1GBps_assumed": _stats_ms(
                projected, "projected_application_e2e_10gbe_1p1GBps_assumed_seconds"
            ),
        },
        "delta_projected_model_minus_measured_model_ms": {
            "1gbe_110MBps": _stats(
                [p - m for p, m in zip(proj_model_1gbe, measured_model)]
            ),
            "10gbe_1p1GBps_assumed": _stats(
                [p - m for p, m in zip(proj_model_10gbe, measured_model)]
            ),
        },
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    latency_paths = {
        label: root / path.relative_to(ROOT) for label, path in LATENCY_JSONL.items()
    }
    accepted_report = root / ACCEPTED_HLS_REPORT.relative_to(ROOT)
    csynth_model = root / ACCEPTED_CSYNTH_MODEL.relative_to(ROOT)
    transfer_module = root / TRANSFER_AUDIT_MODULE.relative_to(ROOT)

    cycle_model = load_accepted_cycle_model(accepted_report, csynth_model)
    transfer = load_transfer_model(transfer_module)
    samples = {label: load_latency_samples(path) for label, path in latency_paths.items()}

    per_budget = {
        label: project_budget(label, int(label[1:]), samples[label], cycle_model, transfer)
        for label in CANDIDATE_ORDER
    }

    inputs_sha256 = {
        "accepted_hls_report": {"path": str(accepted_report), "sha256": sha256_file(accepted_report)},
        "accepted_csynth_model": {"path": str(csynth_model), "sha256": sha256_file(csynth_model)},
        "transfer_audit_module": {"path": str(transfer_module), "sha256": sha256_file(transfer_module)},
    }
    for label, path in latency_paths.items():
        inputs_sha256[f"latency100_{label}"] = {"path": str(path), "sha256": sha256_file(path)}

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": SCOPE,
        "measurement_status": MEASUREMENT_STATUS,
        "not_a_measurement": True,
        "disclaimers": list(DISCLAIMERS),
        "assumptions": {
            "wv_resident": True,
            "wv_bytes_per_image": 0,
            "wv_one_time_bytes": WV_BYTES,
            "gbe_effective_bytes_per_second": GBE_CENTRAL_BYTES_PER_SECOND,
            "gbe_effective_mbytes_per_second": GBE_CENTRAL_BYTES_PER_SECOND / 1e6,
            "gbe_10g_assumed_bytes_per_second": GBE_10G_ASSUMED_BYTES_PER_SECOND,
            "gbe_10g_assumed_mbytes_per_second": GBE_10G_ASSUMED_BYTES_PER_SECOND / 1e6,
            "fpga_target_mhz": cycle_model["target_fmax_mhz"],
            "overlap": "none (strictly serial J->F, compute, F->J)",
            "wv_load_cycles": cycle_model["wv_load_cycles"],
            "per_group_cycles": cycle_model["per_group_cycles"],
        },
        "projection_formula": (
            "projected_model_E2E = measured_model_inference - measured_software_selector "
            "+ fpga_compute + jetson_to_fpga_transfer + fpga_to_jetson_transfer"
        ),
        "inputs_sha256": inputs_sha256,
        "cycle_model_source": {
            "accepted_report": str(accepted_report),
            "accepted_functions": [
                "summarize_csynth._selector_cycles",
                "summarize_csynth._compact_overhead_cycles",
            ],
            "transfer_helpers": [
                "transfer_audit.input_arena_bytes",
                "transfer_audit.fpga_transfer_layout",
            ],
        },
        "per_budget": per_budget,
        "limitations": list(DISCLAIMERS)
        + [
            "The accepted cycle model is a formula estimate built from per-loop unit costs; "
            "it is not a measured FPGA latency.",
            "Latency-100 G values differ from the accepted production grids, so the cycle "
            "model is extrapolated over the observed CV-Bench grid range.",
            "The measured baselines include GPU/CPU scheduling noise; only their medians "
            "over 3 repeats are used and then aggregated over 100 samples.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# 当前 Selector 异构时延投影（K48/K64/K96）")
    lines.append("")
    lines.append(f"> **{report['scope']}**")
    lines.append("")
    lines.append(f"- 生成时间：`{report['generated_at']}`")
    lines.append(f"- measurement_status：`{report['measurement_status']}`")
    lines.append(f"- 投影公式：`{report['projection_formula']}`")
    lines.append("- 条件：Wv resident、`G<=K` 本地 identity bypass、无重叠（严格串行）")
    lines.append("")
    lines.append("## 1. 假设")
    lines.append("")
    a = report["assumptions"]
    lines.append(f"- Wv resident：每图 0 B（一次性 {a['wv_one_time_bytes']} B 摊销）")
    lines.append(f"- 1 GbE 有效带宽：**{a['gbe_effective_mbytes_per_second']:.0f} MB/s**")
    lines.append(
        f"- 假设 10 GbE 有效带宽：**{a['gbe_10g_assumed_mbytes_per_second']:.0f} MB/s**（假设值，非实测）"
    )
    lines.append(f"- FPGA 目标频率：{a['fpga_target_mhz']:.0f} MHz（accepted）")
    lines.append(f"- 重叠：{a['overlap']}")
    lines.append("")
    lines.append("## 2. 每 K 汇总（ms，100 样本）")
    lines.append("")
    lines.append(
        "| K | identity 样本 | FPGA compute mean | 传输 1GbE mean | 传输 10GbE mean | "
        "实测 model mean | 实测 selector-soft mean | 投影 model 1GbE mean | 投影 model 10GbE mean |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in CANDIDATE_ORDER:
        b = report["per_budget"][label]
        lines.append(
            f"| {label} | {b['identity_bypass_count']}/{b['sample_count']} | "
            f"{b['fpga_compute_ms']['mean']:.3f} | "
            f"{b['transfer_ms']['1gbe_110MBps']['mean']:.3f} | "
            f"{b['transfer_ms']['10gbe_1p1GBps_assumed']['mean']:.3f} | "
            f"{b['measured_baseline_ms']['model_inference']['mean']:.3f} | "
            f"{b['measured_baseline_ms']['selector_software']['mean']:.3f} | "
            f"{b['projected_model_e2e_ms']['1gbe_110MBps']['mean']:.3f} | "
            f"{b['projected_model_e2e_ms']['10gbe_1p1GBps_assumed']['mean']:.3f} |"
        )
    lines.append("")
    lines.append("## 3. 每 K 明细（mean / median / p95，ms）")
    lines.append("")
    for label in CANDIDATE_ORDER:
        b = report["per_budget"][label]
        lines.append(f"### {label}")
        lines.append("")
        lines.append("| 量 | mean | median | p95 |")
        lines.append("|---|---:|---:|---:|")
        for name, stats in (
            ("FPGA compute", b["fpga_compute_ms"]),
            ("传输 1GbE", b["transfer_ms"]["1gbe_110MBps"]),
            ("传输 10GbE(假设)", b["transfer_ms"]["10gbe_1p1GBps_assumed"]),
            ("实测 model inference", b["measured_baseline_ms"]["model_inference"]),
            ("实测 selector software", b["measured_baseline_ms"]["selector_software"]),
            ("投影 model E2E 1GbE", b["projected_model_e2e_ms"]["1gbe_110MBps"]),
            ("投影 model E2E 10GbE(假设)", b["projected_model_e2e_ms"]["10gbe_1p1GBps_assumed"]),
            (
                "投影 − 实测 model 1GbE",
                b["delta_projected_model_minus_measured_model_ms"]["1gbe_110MBps"],
            ),
            (
                "投影 − 实测 model 10GbE(假设)",
                b["delta_projected_model_minus_measured_model_ms"]["10gbe_1p1GBps_assumed"],
            ),
        ):
            lines.append(
                f"| {name} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['p95']:.3f} |"
            )
        lines.append("")
    lines.append("## 4. 输入 SHA256")
    lines.append("")
    lines.append("| 输入 | SHA256 |")
    lines.append("|---|---|")
    for name, item in report["inputs_sha256"].items():
        lines.append(f"| {name} | `{item['sha256']}` |")
    lines.append("")
    lines.append("## 5. 限制（必须随结论引用）")
    lines.append("")
    for note in report["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        "> **本文件是纯分析估算，不是 ZCU102/Jetson 实测。** 未上板、未测链路、未跑模型；"
        "不得作为加速/时延实测或精度结论引用。"
    )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args(argv)

    report = build_report(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "measurement_status": report["measurement_status"],
                "budgets": {
                    label: {
                        "identity_bypass_count": report["per_budget"][label]["identity_bypass_count"],
                        "projected_model_e2e_1gbe_mean_ms": report["per_budget"][label][
                            "projected_model_e2e_ms"
                        ]["1gbe_110MBps"]["mean"],
                    }
                    for label in CANDIDATE_ORDER
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
