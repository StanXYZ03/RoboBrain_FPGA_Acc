#!/usr/bin/env python3
"""Platform-independent latency and energy sensitivity model for visual cut points."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def serialization_ms(payload_bytes: int, gbps: float, efficiency: float) -> float:
    if payload_bytes < 0 or gbps <= 0 or not 0 < efficiency <= 1:
        raise ValueError("invalid serialization input")
    return payload_bytes * 8.0 / (gbps * 1e9 * efficiency) * 1000.0


def estimate(
    *,
    full: dict[str, float],
    compact: dict[str, float],
    one_way_bytes: int,
    round_trip_bytes: int,
    link: dict[str, Any],
    prefix_speedup: float,
    power: dict[str, float],
) -> dict[str, float]:
    if prefix_speedup <= 0:
        raise ValueError("prefix_speedup must be positive")
    full_compute_ms = sum(full.values())
    compact_compute_ms = sum(compact.values())
    one_way_link_ms = serialization_ms(one_way_bytes, link["gbps"], link["efficiency"])
    round_trip_link_ms = serialization_ms(round_trip_bytes, link["gbps"], link["efficiency"])

    prefix_ms = compact["vision_prefix_ms"]
    fixed_one_way_ms = float(link["fixed_one_way_ms"])
    resident_host_ms = compact_compute_ms - prefix_ms
    resident_accel_ms = prefix_ms / prefix_speedup
    resident_ms = (
        resident_host_ms
        + resident_accel_ms
        + one_way_link_ms
        + fixed_one_way_ms
    )
    round_trip_ms = (
        compact_compute_ms
        + round_trip_link_ms
        + float(link["fixed_round_trip_ms"])
    )
    software_ms = compact_compute_ms

    baseline_energy_mj = full_compute_ms * power["host_compute"] / 1000.0
    software_energy_mj = software_ms * power["host_compute"] / 1000.0
    resident_energy_mj = (
        resident_host_ms * power["host_compute"]
        + resident_accel_ms * power["accelerator_compute"]
        + one_way_link_ms * power["link_active_total"]
    ) / 1000.0
    round_trip_energy_mj = (
        compact_compute_ms * power["host_compute"]
        + round_trip_link_ms * power["link_active_total"]
    ) / 1000.0
    overhead_ms = one_way_link_ms + fixed_one_way_ms
    break_even_prefix_speedup = (
        prefix_ms / (prefix_ms - overhead_ms) if prefix_ms > overhead_ms else None
    )
    compute_saving_budget_ms = prefix_ms * (1.0 - 1.0 / prefix_speedup) - fixed_one_way_ms
    required_link_gbps = (
        one_way_bytes * 8.0 / (compute_saving_budget_ms / 1000.0)
        / 1e9 / link["efficiency"]
        if compute_saving_budget_ms > 0 and one_way_bytes > 0
        else None
    )
    return {
        "baseline_compute_ms": full_compute_ms,
        "software_compact_ms": software_ms,
        "resident_ms": resident_ms,
        "round_trip_ms": round_trip_ms,
        "one_way_link_ms": one_way_link_ms,
        "round_trip_link_ms": round_trip_link_ms,
        "resident_delta_vs_full_ms": resident_ms - full_compute_ms,
        "round_trip_delta_vs_full_ms": round_trip_ms - full_compute_ms,
        "resident_delta_vs_software_ms": resident_ms - software_ms,
        "round_trip_delta_vs_software_ms": round_trip_ms - software_ms,
        "baseline_energy_mj": baseline_energy_mj,
        "software_energy_mj": software_energy_mj,
        "resident_energy_mj": resident_energy_mj,
        "round_trip_energy_mj": round_trip_energy_mj,
        "break_even_prefix_speedup_vs_software": break_even_prefix_speedup,
        "required_link_gbps_vs_software": required_link_gbps,
    }


def load_records(results_dir: Path, cut_name: str, retention: float) -> list[dict[str, Any]]:
    tag = int(round(retention * 1000))
    path = results_dir / f"oracle_{cut_name}_r{tag:04d}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage_medians(records: list[dict[str, Any]]) -> dict[str, float]:
    fields = {
        "vision_prefix_ms": "vision_prefix_block_seconds",
        "vision_pruned_ms": "vision_pruned_block_seconds",
        "vision_other_ms": "vision_other_seconds",
        "llm_prefill_ms": "llm_prefill_seconds",
    }
    return {
        output: median(float(record[source]) for record in records) * 1000.0
        for output, source in fields.items()
    }


def build_rows(config: dict[str, Any], results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    power = config["power_sensitivity_w"]
    for cut_name in config["cut_points"]:
        full = stage_medians(load_records(results_dir, cut_name, 1.0))
        for retention in config["retentions"]:
            records = load_records(results_dir, cut_name, retention)
            compact = stage_medians(records)
            one_way_bytes = round(median(r["communication"]["one_way_bytes"] for r in records))
            round_trip_bytes = round(median(r["communication"]["round_trip_bytes"] for r in records))
            for link in config["link_profiles"]:
                for speedup in config["accelerator_prefix_speedups"]:
                    rows.append({
                        "cut_point": cut_name,
                        "retention": retention,
                        "link_profile": link["name"],
                        "link_gbps": link["gbps"],
                        "prefix_speedup": speedup,
                        "one_way_bytes": one_way_bytes,
                        "round_trip_bytes": round_trip_bytes,
                        **estimate(
                            full=full,
                            compact=compact,
                            one_way_bytes=one_way_bytes,
                            round_trip_bytes=round_trip_bytes,
                            link=link,
                            prefix_speedup=speedup,
                            power=power,
                        ),
                    })
    return rows


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    def number(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    representative = [
        row for row in rows
        if row["link_profile"] == "ethernet_10g" and row["prefix_speedup"] == 4.0
    ]
    lines = [
        "# 平台无关异构切入点收益敏感性模型",
        "",
        "## 证据边界",
        "",
        "这是基于既有A6000 BF16开发Oracle阶段计时、静态张量字节和假设功耗的敏感性模型，不是Q8_0、ZCU102或未来AI核心平台的实测结果。模型不包含Decode、后处理、协议头、真实DMA/驱动、布局变换，也不假设计算与通信重叠。",
        "",
        "下表仅展示10 Gb/s、有效率0.8、加速器视觉前缀加速4倍、固定链路开销为0的乐观序列化下界。负的增量表示相对完整视觉计算更快；相对软件压缩为正则说明异构路径仍不如在主平台直接压缩。",
        "",
        "| 切入点 | 保留率 | 完整计算/ms | 软件压缩/ms | 单向驻留/ms | 外挂往返/ms | 驻留相对软件/ms | 4倍前缀所需链路/Gb/s | 当前链路所需前缀加速 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in representative:
        lines.append(
            f"| {row['cut_point']} | {row['retention']:.3f} | "
            f"{row['baseline_compute_ms']:.3f} | {row['software_compact_ms']:.3f} | "
            f"{row['resident_ms']:.3f} | {row['round_trip_ms']:.3f} | "
            f"{row['resident_delta_vs_software_ms']:+.3f} | "
            f"{number(row['required_link_gbps_vs_software'])} | "
            f"{number(row['break_even_prefix_speedup_vs_software'])} |"
        )
    resident_wins = sum(row["resident_delta_vs_software_ms"] < 0 for row in rows)
    roundtrip_wins = sum(row["round_trip_delta_vs_software_ms"] < 0 for row in rows)
    lines += [
        "",
        "## 扫描摘要",
        "",
        f"- 总敏感性组合：{len(rows)}。",
        f"- 单向驻留优于主平台软件压缩：{resident_wins}/{len(rows)}；该比较依赖假设的视觉前缀加速倍数。",
        f"- 外挂往返优于主平台软件压缩：{roundtrip_wins}/{len(rows)}。当前零固定开销下界仍只能增加链路时间，因此不会获胜。",
        "- 能量列使用配置文件中的假设功耗，只用于比较趋势，不能引用为板级能耗。",
        "",
        "## 使用方法",
        "",
        "    python3 evaluation/heterogeneous_benefit_model.py",
        "",
        "确定AI核心平台后，应把配置中的链路吞吐、固定开销和功耗替换为实测值，并补充Q8_0各阶段稳态计时；脚本与输出格式无需更换。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/platform_agnostic_benefit_model.json")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "evaluation/results/cutpoint_oracle_dev")
    parser.add_argument("--json", type=Path, default=ROOT / "reports/2026-09-10-platform-agnostic-benefit-model.json")
    parser.add_argument("--csv", type=Path, default=ROOT / "reports/2026-09-10-platform-agnostic-benefit-model.csv")
    parser.add_argument("--markdown", type=Path, default=ROOT / "reports/2026-09-10-platform-agnostic-benefit-model.md")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = build_rows(config, args.results_dir)
    payload = {
        "format": "platform-agnostic-benefit-model-v1",
        "status": "SENSITIVITY_MODEL_NOT_TARGET_PLATFORM_MEASUREMENT",
        "config": str(args.config),
        "source_results": str(args.results_dir),
        "row_count": len(rows),
        "rows": rows,
        "limitations": config["notes"],
    }
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with args.csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(args.markdown, rows)
    print(json.dumps({"status": payload["status"], "rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
