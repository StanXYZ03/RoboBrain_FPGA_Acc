#!/usr/bin/env python3
"""Jetson Xavier NX + ZCU102 selector requirement model (baseline-corrected).

This is a **model-only** analysis. It does not use a GPU, does not access the
ZCU102, does not touch the network, and makes no measurement or speedup claim.

Baseline correction
-------------------
The superseded v1 model used a single ``J_ONLY`` baseline that already contained
the Jetson software selector *and* the K-token post-B1 vision. Because ``J_F_J``
shares that same K-token vision, no post-B1 saving could offset the FPGA round
trip, and against a dense baseline score generation was wrongly treated as
cancelling. This model therefore defines four placements and two strictly
separated requirement sets:

* ``J_DENSE``       no score, no selector, dense post-B1 Jetson vision;
* ``J_PRUNED``      Jetson score + software selector + K-token Jetson vision;
* ``J_F_J``         Jetson score + FPGA selector round trip + the *same*
                    K-token Jetson vision;
* ``J_F_CONTINUE``  Jetson score + FPGA selector + a real FPGA downstream core.

Comparison A (algorithm benefit vs ``J_DENSE``)::

    J_PRUNED       S_prune > T_score + T_sel_sw
    J_F_J          S_prune > T_score + T_link_dense + T_sel_fpga + T_link_return + 2F
    J_F_CONTINUE   T_vision_dense - T_fpga_downstream > T_score + T_link_dense + T_sel_fpga + F

Comparison B (deployment choice vs ``J_PRUNED``)::

    J_F_J          T_sel_sw > T_link_dense + T_sel_fpga + T_link_return + 2F
    J_F_CONTINUE   T_sel_sw + T_vision_pruned > T_link_dense + T_sel_fpga + F + T_fpga_downstream

``T_score``, ``T_sel_sw``, ``T_vision_dense``, ``T_vision_pruned`` and
``T_fpga_downstream`` are unmeasured. Only the communication + FPGA-selector
part is numeric; every reported millisecond value is that known lower bound and
must be quoted together with ``+T_score`` or ``"T_sel_sw must exceed X"``, never
as a full break-even.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import hardware_regular_selector_fixedpoint as fp  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "heterogeneous_selector_break_even.json"
DEFAULT_REPORT_DIR = ROOT / "reports" / "token_selector"
DEFAULT_REPORT_STEM = "2026-09-28-heterogeneous-break-even"
DEFAULT_HARNESS_RESULT = Path("/tmp/robobrain_harness_result.json")

BASELINES = ("J_DENSE", "J_PRUNED")
PLACEMENTS = ("J_DENSE", "J_PRUNED", "J_F_J", "J_F_CONTINUE")
COMPARISONS = ("A_vs_J_DENSE", "B_vs_J_PRUNED")
COMPARISON_BASELINE = {"A_vs_J_DENSE": "J_DENSE", "B_vs_J_PRUNED": "J_PRUNED"}
COMPARISON_CANDIDATES = {
    "A_vs_J_DENSE": ("J_PRUNED", "J_F_J", "J_F_CONTINUE"),
    "B_vs_J_PRUNED": ("J_F_J", "J_F_CONTINUE"),
}
FIXED_OVERHEADS = {"J_DENSE": 0, "J_PRUNED": 0, "J_F_J": 2, "J_F_CONTINUE": 1}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dig(data: dict[str, Any], path: list[str]) -> Any:
    node: Any = data
    for key in path:
        node = node[key]
    return node


def cycles_to_ms(cycles: float, clock_hz: float) -> float:
    if cycles < 0 or clock_hz <= 0:
        raise ValueError("cycles must be >= 0 and clock_hz > 0")
    return float(cycles) / float(clock_hz) * 1000.0


def link_ms(nbytes: int, throughput_mbps: float, fixed_ms: float = 0.0) -> float:
    """One-way serialization time; decimal MB/s, no overlap with compute."""
    if nbytes < 0 or throughput_mbps <= 0 or fixed_ms < 0:
        raise ValueError("invalid link input")
    return nbytes / (throughput_mbps * 1000.0) + fixed_ms


# ---------------------------------------------------------------------------
# byte contract (derived from the real implementation, never hard-coded)
# ---------------------------------------------------------------------------


def dense_input_bytes(group_count: int) -> int:
    """G int8[4,1024] tokens plus G uint16 Q0.15 scores."""
    if group_count <= 0:
        raise ValueError("group_count must be positive")
    return group_count * fp.GROUP_ELEMENTS + group_count * 2


def selected_output_bytes(budget: int) -> int:
    """64 B-aligned 5-segment host layout total for K selected groups."""
    return fp.fpga_transfer_layout(budget).total_bytes


def byte_contract_rows(group_counts: list[int], budgets: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_count in group_counts:
        for budget in budgets:
            layout = fp.fpga_transfer_layout(budget)
            rows.append(
                {
                    "group_count": group_count,
                    "budget": budget,
                    "dense_input_bytes": dense_input_bytes(group_count),
                    "selected_output_bytes": selected_output_bytes(budget),
                    "round_trip_bytes": dense_input_bytes(group_count)
                    + selected_output_bytes(budget),
                    "segment_sizes": {s.name: s.size for s in layout.segments},
                }
            )
    return rows


# ---------------------------------------------------------------------------
# evidence (read from v4 artifacts)
# ---------------------------------------------------------------------------


def load_selector_evidence(config: dict[str, Any]) -> dict[str, Any]:
    evidence_cfg = config["selector_evidence"]
    csynth_path = ROOT / evidence_cfg["csynth_summary"]
    cosim_path = ROOT / evidence_cfg["cosim_summary"]
    ooc_path = ROOT / evidence_cfg["ooc_summary"]
    csynth = json.loads(csynth_path.read_text())
    cosim = json.loads(cosim_path.read_text())
    ooc = json.loads(ooc_path.read_text())

    worst_cycles = _dig(csynth, evidence_cfg["csynth_worst_cycles_json_path"])
    cosim_latency = cosim["cosim"]["latency_cycles"]
    cases = [
        {
            "name": case["name"],
            "group_count": int(case["group_count"]),
            "budget": int(case["budget"]),
        }
        for case in cosim["cases"]
    ]
    return {
        "csynth_worst_cycles": int(round(float(worst_cycles))),
        "csynth_average_cycles": int(round(float(csynth["latency"]["average_cycles"]))),
        "cosim_latency_cycles": {
            "min": int(cosim_latency["min"]),
            "avg": int(round(float(cosim_latency["avg"]))),
            "max": int(cosim_latency["max"]),
        },
        "cosim_total_execution_cycles": int(cosim["cosim"]["total_execution_cycles"]),
        "cosim_cases": cases,
        "cosim_result": cosim["cosim"]["result"],
        "csynth_result": "PASS" if csynth.get("gate_met", csynth.get("latency", {}).get("gate_met")) else "UNKNOWN",
        "ooc_clock_period_ns": float(ooc["clock"]["period_ns"]),
        "ooc_wns_ns": float(ooc["timing"]["wns_ns"]),
        "ooc_status": ooc["status"],
        "paths": {
            "csynth_summary": str(csynth_path.relative_to(ROOT)),
            "cosim_summary": str(cosim_path.relative_to(ROOT)),
            "ooc_summary": str(ooc_path.relative_to(ROOT)),
        },
        "sha256": {
            "csynth_summary": sha256_file(csynth_path),
            "cosim_summary": sha256_file(cosim_path),
            "ooc_summary": sha256_file(ooc_path),
        },
    }


def selector_scenarios(evidence: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Map config scenario ids to the cycle counts read from evidence."""
    lookup = {
        "csynth_worst_bound": evidence["csynth_worst_cycles"],
        "cosim_observed_max": evidence["cosim_latency_cycles"]["max"],
        "cosim_observed_avg": evidence["cosim_latency_cycles"]["avg"],
        "cosim_observed_min": evidence["cosim_latency_cycles"]["min"],
    }
    scenarios: list[dict[str, Any]] = []
    for entry in config["selector_cycle_scenarios"]:
        if entry["id"] not in lookup:
            raise KeyError(f"unknown selector cycle scenario: {entry['id']}")
        scenarios.append({**entry, "cycles": int(lookup[entry["id"]])})
    return scenarios


# ---------------------------------------------------------------------------
# requirement engine
# ---------------------------------------------------------------------------


def known_lower_bound_ms(
    *,
    placement: str,
    dense_bytes: int,
    selected_bytes: int,
    selector_cycles: int,
    clock_hz: float,
    throughput_mbps: float,
    fixed_overhead_ms: float,
) -> dict[str, float]:
    """Communication + FPGA-selector part known without any Jetson measurement.

    ``J_DENSE`` and ``J_PRUNED`` use no link leg and no FPGA selector, so their
    known part is exactly zero. ``J_F_J`` adds a dense send, the FPGA selector,
    a selected-token return and two fixed overheads. ``J_F_CONTINUE`` adds only
    a dense send, the FPGA selector and one fixed overhead.
    """
    if placement not in PLACEMENTS:
        raise ValueError(f"unknown placement: {placement}")
    if placement in ("J_DENSE", "J_PRUNED"):
        t_dense = t_selector = t_return = 0.0
        fixed_overheads = 0
    else:
        t_dense = link_ms(dense_bytes, throughput_mbps)
        t_selector = cycles_to_ms(selector_cycles, clock_hz)
        t_return = (
            link_ms(selected_bytes, throughput_mbps) if placement == "J_F_J" else 0.0
        )
        fixed_overheads = FIXED_OVERHEADS[placement]
    fixed_terms = fixed_overheads * fixed_overhead_ms
    total = t_dense + t_selector + t_return + fixed_terms
    return {
        "t_link_dense_ms": t_dense,
        "t_sel_fpga_ms": t_selector,
        "t_link_return_ms": t_return,
        "fixed_terms_ms": fixed_terms,
        "fixed_overheads": float(fixed_overheads),
        "known_lower_bound_ms": total,
    }


def requirement_spec(
    comparison: str, placement: str, known_ms: float
) -> dict[str, Any]:
    """Symbolic requirement for one comparison/placement at a known lower bound."""
    if comparison not in COMPARISONS:
        raise ValueError(f"unknown comparison: {comparison}")
    if placement not in COMPARISON_CANDIDATES[comparison]:
        raise ValueError(f"placement {placement} is not a candidate for {comparison}")
    if comparison == "A_vs_J_DENSE":
        if placement == "J_PRUNED":
            return {
                "requirement": "S_prune > T_score + T_sel_sw",
                "threshold_quantity": "S_prune",
                "symbolic_missing_terms": [
                    "T_score",
                    "T_sel_sw",
                    "T_vision_dense",
                    "T_vision_pruned",
                ],
                "numeric_threshold_ms": None,
                "is_exact_threshold": False,
            }
        if placement == "J_F_J":
            return {
                "requirement": f"S_prune > T_score + {known_ms:.6f} ms",
                "threshold_quantity": "S_prune",
                "symbolic_missing_terms": ["T_score", "T_vision_dense", "T_vision_pruned"],
                "numeric_threshold_ms": None,
                "is_exact_threshold": False,
            }
        return {
            "requirement": (
                "T_vision_dense - T_fpga_downstream > T_score + "
                f"{known_ms:.6f} ms"
            ),
            "threshold_quantity": "T_vision_dense - T_fpga_downstream",
            "symbolic_missing_terms": ["T_score", "T_vision_dense", "T_fpga_downstream"],
            "numeric_threshold_ms": None,
            "is_exact_threshold": False,
        }
    # comparison B: deployment choice vs J_PRUNED
    if placement == "J_F_J":
        return {
            "requirement": f"T_sel_sw > {known_ms:.6f} ms",
            "threshold_quantity": "T_sel_sw",
            "symbolic_missing_terms": ["T_sel_sw"],
            "numeric_threshold_ms": known_ms,
            "is_exact_threshold": True,
        }
    return {
        "requirement": (
            f"T_sel_sw + T_vision_pruned > {known_ms:.6f} ms + T_fpga_downstream"
        ),
        "threshold_quantity": "T_sel_sw + T_vision_pruned",
        "symbolic_missing_terms": ["T_sel_sw", "T_vision_pruned", "T_fpga_downstream"],
        "numeric_threshold_ms": None,
        "is_exact_threshold": False,
    }


def build_sweep(
    config: dict[str, Any],
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    clock_hz = float(config["selector_clock_hz"])
    throughputs = config["link_profiles"]["effective_throughput_MBps"]
    overheads = config["link_profiles"]["fixed_software_and_dma_overhead_ms"]
    scenarios = selector_scenarios(evidence, config)
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for group_count in config["grid_group_counts"]:
            for budget in config["budgets"]:
                dense_bytes = dense_input_bytes(group_count)
                selected_bytes = selected_output_bytes(budget)
                for comparison in COMPARISONS:
                    for placement in COMPARISON_CANDIDATES[comparison]:
                        for throughput in throughputs:
                            for overhead in overheads:
                                known = known_lower_bound_ms(
                                    placement=placement,
                                    dense_bytes=dense_bytes,
                                    selected_bytes=selected_bytes,
                                    selector_cycles=scenario["cycles"],
                                    clock_hz=clock_hz,
                                    throughput_mbps=float(throughput),
                                    fixed_overhead_ms=float(overhead),
                                )
                                spec = requirement_spec(
                                    comparison, placement, known["known_lower_bound_ms"]
                                )
                                rows.append(
                                    {
                                        "comparison": comparison,
                                        "baseline": COMPARISON_BASELINE[comparison],
                                        "placement": placement,
                                        "group_count": group_count,
                                        "budget": budget,
                                        "selector_cycle_scenario": scenario["id"],
                                        "selector_cycles": scenario["cycles"],
                                        "link_throughput_MBps": float(throughput),
                                        "fixed_overhead_ms": float(overhead),
                                        "dense_input_bytes": dense_bytes,
                                        "selected_output_bytes": selected_bytes,
                                        "round_trip_bytes": dense_bytes + selected_bytes,
                                        **known,
                                        "threshold_quantity": spec["threshold_quantity"],
                                        "required_expression": spec["requirement"],
                                        "symbolic_missing_terms": ";".join(
                                            spec["symbolic_missing_terms"]
                                        ),
                                        "numeric_threshold_ms": spec["numeric_threshold_ms"],
                                        "is_exact_threshold": spec["is_exact_threshold"],
                                        "lower_bound_only": True,
                                    }
                                )
    return rows


def key_thresholds(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Representative known lower bounds at the most likely operating point."""
    picks = {
        "scenario": "csynth_worst_bound",
        "throughput_MBps": 2000.0,
        "fixed_overhead_ms": 0.5,
        "points": [
            {"group_count": 256, "budget": 64},
            {"group_count": 1024, "budget": 96},
        ],
    }
    out: dict[str, Any] = {
        k: picks[k] for k in ("scenario", "throughput_MBps", "fixed_overhead_ms")
    }
    points: list[dict[str, Any]] = []
    for pick in picks["points"]:
        for comparison in COMPARISONS:
            for placement in COMPARISON_CANDIDATES[comparison]:
                match = [
                    row
                    for row in rows
                    if row["comparison"] == comparison
                    and row["placement"] == placement
                    and row["group_count"] == pick["group_count"]
                    and row["budget"] == pick["budget"]
                    and row["selector_cycle_scenario"] == picks["scenario"]
                    and row["link_throughput_MBps"] == picks["throughput_MBps"]
                    and row["fixed_overhead_ms"] == picks["fixed_overhead_ms"]
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"key threshold lookup failed: {pick} {comparison} {placement}"
                    )
                row = match[0]
                points.append(
                    {
                        "group_count": row["group_count"],
                        "budget": row["budget"],
                        "comparison": row["comparison"],
                        "baseline": row["baseline"],
                        "placement": row["placement"],
                        "selector_cycles": row["selector_cycles"],
                        "dense_input_bytes": row["dense_input_bytes"],
                        "selected_output_bytes": row["selected_output_bytes"],
                        "t_link_dense_ms": row["t_link_dense_ms"],
                        "t_sel_fpga_ms": row["t_sel_fpga_ms"],
                        "t_link_return_ms": row["t_link_return_ms"],
                        "fixed_terms_ms": row["fixed_terms_ms"],
                        "known_lower_bound_ms": row["known_lower_bound_ms"],
                        "required_expression": row["required_expression"],
                        "threshold_quantity": row["threshold_quantity"],
                        "symbolic_missing_terms": row["symbolic_missing_terms"].split(";"),
                        "numeric_threshold_ms": row["numeric_threshold_ms"],
                        "is_exact_threshold": row["is_exact_threshold"],
                    }
                )
    out["points"] = points

    fastest = config["link_profiles"]["effective_throughput_MBps"][-1]
    min_overhead = min(config["link_profiles"]["fixed_software_and_dma_overhead_ms"])

    def best_for(comparison: str, placement: str) -> dict[str, Any]:
        candidates = [
            row
            for row in rows
            if row["comparison"] == comparison
            and row["placement"] == placement
            and row["selector_cycle_scenario"] == "csynth_worst_bound"
            and row["link_throughput_MBps"] == fastest
            and row["fixed_overhead_ms"] == min_overhead
        ]
        best = min(candidates, key=lambda row: row["known_lower_bound_ms"])
        return {
            "known_lower_bound_ms": best["known_lower_bound_ms"],
            "required_expression": best["required_expression"],
            "group_count": best["group_count"],
            "budget": best["budget"],
            "link_throughput_MBps": best["link_throughput_MBps"],
            "fixed_overhead_ms": best["fixed_overhead_ms"],
        }

    out["optimistic_min_known_lower_bound"] = {
        "context": {"link_throughput_MBps": fastest, "fixed_overhead_ms": min_overhead},
        "J_F_J_vs_J_PRUNED_T_sel_sw_threshold": best_for("B_vs_J_PRUNED", "J_F_J"),
        "J_F_CONTINUE_vs_J_PRUNED_plus_T_fpga_downstream": best_for(
            "B_vs_J_PRUNED", "J_F_CONTINUE"
        ),
        "J_F_J_vs_J_DENSE_plus_T_score": best_for("A_vs_J_DENSE", "J_F_J"),
        "J_F_CONTINUE_vs_J_DENSE_plus_T_score": best_for(
            "A_vs_J_DENSE", "J_F_CONTINUE"
        ),
    }
    out["forbidden_claim"] = (
        "These are communication + FPGA-selector lower bounds only. Do not quote "
        "them as a full break-even relative to the superseded J_ONLY baseline, and "
        "always attach '+ T_score' or 'T_sel_sw must exceed X'."
    )
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_markdown(
    path: Path,
    config: dict[str, Any],
    evidence: dict[str, Any],
    contract_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    key: dict[str, Any],
) -> None:
    clock_hz = float(config["selector_clock_hz"])
    lines: list[str] = [
        "# Jetson Xavier NX + ZCU102 Selector 异构需求模型（基线修正版 v2）",
        "",
        "- 日期：2026-09-28",
        "- 决策：`HETEROGENEOUS_SELECTOR_BASELINE_CORRECTED / NO_FULL_BREAK_EVEN / NO_MEASURED_SPEEDUP_CLAIM`",
        "- 范围：**只做模型与参数扫描**。不使用 GPU、不访问 ZCU102、不联网、不做板级实测，不声称当前已加速。",
        "",
        "## 0. 为什么必须修正基线（先读）",
        "",
        "旧 v1 用单一 `J_ONLY` 作基线，而 `J_ONLY` 已包含 Jetson 软件 selector **和** K-token 后续视觉；",
        "`J_F_J` 与它的 K-token 后续视觉完全相同，因此不能用“post-B1 节省”抵消 FPGA 往返。",
        "若改对 Dense 基线（`J_DENSE`），score 生成（`T_score`）不再抵消，必须显式保留。",
        "本版据此拆成 `J_DENSE` 与 `J_PRUNED` 两个基线，并输出两套严格公式。",
        "- **禁止**再输出相对旧 `J_ONLY` 的 9–12 ms“完整 break-even”。数表只给 communication + FPGA-selector 的 lower bound，",
        "  必须以 `+T_score` 或“`T_sel_sw` 需超过 X”表达。",
        "",
        "## 1. 四个 placement 定义",
        "",
        "| placement | score | selector | K tokens | 后续视觉 | 链路腿 | 固定同步开销 | 总延迟 |",
        "|---|---|---|---|---|---|---:|---|",
        "| `J_DENSE` | 无 | 无 | 不裁剪 | dense Jetson vision | 0 | 0 | `T_vision_dense` |",
        "| `J_PRUNED` | 在 Jetson | Jetson 软件 | 留 Jetson | K-token Jetson vision | 0 | 0 | `T_score + T_sel_sw + T_vision_pruned` |",
        "| `J_F_J` | 在 Jetson | FPGA | 返回 Jetson | 同一 K-token Jetson vision | 2 | 2 | `T_score + T_link_dense + T_sel_fpga + T_link_return + 2F + T_vision_pruned` |",
        "| `J_F_CONTINUE` | 在 Jetson | FPGA | 留 FPGA | 真实 FPGA 下游核 | 1 | 1 | `T_score + T_link_dense + T_sel_fpga + F + T_fpga_downstream` |",
        "",
        "**Selector IP 不含 Wt/Wv score 生成。** `T_score` 单列，且不计入 FPGA 核 cycles。",
        "它在 `J_F_J` vs `J_PRUNED` 中抵消（连同 `T_vision_pruned`），但在对 `J_DENSE` 的算法收益中**必须保留**。",
        "",
        "## 2. 两套严格公式",
        "",
        "记 `S_prune = T_vision_dense - T_vision_pruned`，`F` 为每条链路腿的一次固定软件+DMA 开销。",
        "",
        "### A. 算法收益 vs `J_DENSE`（`J_DENSE = T_vision_dense`）",
        "",
        "```",
        "J_PRUNED       : S_prune > T_score + T_sel_sw",
        "J_F_J          : S_prune > T_score + T_link_dense + T_sel_fpga + T_link_return + 2F",
        "J_F_CONTINUE   : T_vision_dense - T_fpga_downstream > T_score + T_link_dense + T_sel_fpga + F",
        "```",
        "",
        "`J_F_J` 对 dense 基线的要求里**含 `T_score` 符号项**，不能把 score 当作抵消。",
        "",
        "### B. 部署选择 vs `J_PRUNED`（`J_PRUNED = T_score + T_sel_sw + T_vision_pruned`）",
        "",
        "```",
        "J_F_J          : T_sel_sw > T_link_dense + T_sel_fpga + T_link_return + 2F",
        "J_F_CONTINUE   : T_sel_sw + T_vision_pruned > T_link_dense + T_sel_fpga + F + T_fpga_downstream",
        "```",
        "",
        "`J_F_J` 对 `J_PRUNED` 的公式**不含任何 vision saving**：`T_vision_pruned` 与 `T_score` 严格抵消，",
        "只有 `T_sel_sw` 超过“往返 + FPGA selector + 2F”时才更快。",
        "",
        "## 3. 字节合同（从真实实现推导）",
        "",
        f"- 来源：`{config['byte_contract']['source_module']}` 的 `fpga_transfer_layout` 与 "
        f"`GROUP_ELEMENTS={fp.GROUP_ELEMENTS}`。",
        "- dense 输入 = `G*4096 B`（int8 `[G,4,1024]`）+ `G*2 B`（uint16 Q0.15 score）。",
        "- 输出 = 5 段布局：`tokens_int8 K*4096 B` + `group_index_uint16 K*2` + "
        "`source_type_uint8 K` + `region_id_uint8 K` + `valid_uint8 K`，每段 64 B 对齐。",
        "",
        "| G | K | dense 输入/B | 选中输出/B | 往返/B |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in contract_rows:
        lines.append(
            f"| {row['group_count']} | {row['budget']} | {row['dense_input_bytes']} | "
            f"{row['selected_output_bytes']} | {row['round_trip_bytes']} |"
        )

    lines += [
        "",
        "## 4. Selector 周期证据（读取，不手填）",
        "",
        "- v4 csynth：worst bound **{:,} cycles** = {:.4f} ms @ {} MHz（pre-route 静态界，对 G=256/576/1024 保守）。".format(
            evidence["csynth_worst_cycles"],
            cycles_to_ms(evidence["csynth_worst_cycles"], clock_hz),
            int(clock_hz / 1e6),
        ),
        "- v4 cosim：功能 PASS，3 个代表 case，聚合 latency min/avg/max = **{:,} / {:,} / {:,} cycles**；cosim **只验证功能，不证明 200 MHz**。".format(
            evidence["cosim_latency_cycles"]["min"],
            evidence["cosim_latency_cycles"]["avg"],
            evidence["cosim_latency_cycles"]["max"],
        ),
        "- 200 MHz 换算的时序依据是独立 OOC route：period {} ns、WNS {:+.3f} ns、status {}（OOC，非整机/上板）。".format(
            evidence["ooc_clock_period_ns"], evidence["ooc_wns_ns"], evidence["ooc_status"]
        ),
        "",
        "| cosim case | G | K |",
        "|---|---:|---:|",
    ]
    for case in evidence["cosim_cases"]:
        lines.append(f"| `{case['name']}` | {case['group_count']} | {case['budget']} |")

    lines += [
        "",
        "## 5. 数表：只有 communication + FPGA-selector lower bound",
        "",
        "以下为 `csynth worst bound`、2000 MB/s、固定开销 0.5 ms 的代表点。",
        "`known lower bound` 只含链路纯序列化 + FPGA selector + 固定项，**不含任何未实测的 Jetson 项**；",
        "因此它只是所需量（`S_prune` / `T_vision_dense - T_fpga_downstream` / `T_sel_sw` 等）的**下界**。",
        "",
        "| G | K | comparison | placement | T_link_dense/ms | T_sel_fpga/ms | T_link_return/ms | 固定项/ms | known lower bound/ms | 严格需求 |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for point in key["points"]:
        lines.append(
            f"| {point['group_count']} | {point['budget']} | `{point['comparison']}` | "
            f"`{point['placement']}` | {_fmt(point['t_link_dense_ms'])} | "
            f"{_fmt(point['t_sel_fpga_ms'])} | {_fmt(point['t_link_return_ms'])} | "
            f"{_fmt(point['fixed_terms_ms'])} | {_fmt(point['known_lower_bound_ms'])} | "
            f"`{point['required_expression']}` |"
        )
    optimistic = key["optimistic_min_known_lower_bound"]
    lines += [
        "",
        "最快扫描链路（{} MB/s，固定开销 {} ms）下最乐观的 known lower bound：".format(
            optimistic["context"]["link_throughput_MBps"],
            optimistic["context"]["fixed_overhead_ms"],
        ),
        "- `J_F_J` vs `J_PRUNED`：`{}`（即要求 `T_sel_sw > {} ms`）。".format(
            optimistic["J_F_J_vs_J_PRUNED_T_sel_sw_threshold"]["required_expression"],
            _fmt(
                optimistic["J_F_J_vs_J_PRUNED_T_sel_sw_threshold"][
                    "known_lower_bound_ms"
                ]
            ),
        ),
        "- `J_F_J` vs `J_DENSE`：`{}`。".format(
            optimistic["J_F_J_vs_J_DENSE_plus_T_score"]["required_expression"]
        ),
        "- `J_F_CONTINUE` vs `J_DENSE`：`{}`。".format(
            optimistic["J_F_CONTINUE_vs_J_DENSE_plus_T_score"]["required_expression"]
        ),
        "- `J_F_CONTINUE` vs `J_PRUNED`：`{}`。".format(
            optimistic["J_F_CONTINUE_vs_J_PRUNED_plus_T_fpga_downstream"][
                "required_expression"
            ]
        ),
        "",
        f"> {key['forbidden_claim']}",
        "",
        "## 6. 扫描摘要",
        "",
        f"- 扫描行数：{len(rows)}。",
        f"- comparison：{', '.join(COMPARISONS)}；baseline：{', '.join(BASELINES)}。",
        f"- selector 周期场景：{', '.join(s['id'] for s in selector_scenarios(evidence, config))}。",
        f"- 有效吞吐扫描（MB/s）：{config['link_profiles']['effective_throughput_MBps']}。",
        f"- 单次固定软件+DMA 开销扫描（ms）：{config['link_profiles']['fixed_software_and_dma_overhead_ms']}。",
        f"- `symbolic_missing_terms`：{', '.join(config['symbolic_missing_terms'])}。",
        "- 全部行的 `lower_bound_only=True`；没有任何一行的数值可单独判定胜负。",
        "",
        "## 7. 必须补齐的实测（Gate）",
        "",
        "1. Jetson 上实测 `T_score`（Wt/Wv score 生成）。",
        "2. Jetson 上实测 `T_sel_sw`（软件 selector）；实测后代入 B 组公式。",
        "3. Jetson 上实测 `T_vision_dense`、`T_vision_pruned`（`J_DENSE` / `J_PRUNED` 基线）。",
        "4. 实测 Jetson↔ZCU102 有效吞吐与固定软硬件+DMA 开销（当前为参数扫描）。",
        "5. 若走 `J_F_CONTINUE`：必须存在并实测真实 FPGA 下游核 `T_fpga_downstream`；当前仓库没有该核。",
        "6. 板级 200 MHz 与功能验证（当前仅 OOC route 报告 + 功能 cosim）。",
        "",
        "在上述实测完成前，**不得**声称异构 selector 已加速，也不得给出“完整 break-even 毫秒数”。",
        "",
        "## 8. 复现命令",
        "",
        "```bash",
        "python3 scripts/analyze_heterogeneous_selector_break_even.py \\",
        "  --config configs/heterogeneous_selector_break_even.json",
        "```",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_report_payload(
    config: dict[str, Any],
    evidence: dict[str, Any],
    contract_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    key: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "heterogeneous-selector-break-even-v2",
        "date": "2026-09-28",
        "status": "MODEL_ONLY_BASELINE_CORRECTED_NO_FULL_BREAK_EVEN",
        "scope": config["scope"],
        "baseline_confusion_fix": config["baseline_confusion_fix"],
        "prior_art_audit": config["prior_art_audit"],
        "baselines": {name: config["placements"][name] for name in BASELINES},
        "placements": config["placements"],
        "symbolic_missing_terms": config["symbolic_missing_terms"],
        "unmeasured_terms": config["unmeasured_terms"],
        "symbol_legend": config["symbol_legend"],
        "formulas": {
            "comparison_A_algorithm_vs_J_DENSE": config[
                "comparison_A_algorithm_vs_J_DENSE"
            ],
            "comparison_B_deployment_vs_J_PRUNED": config[
                "comparison_B_deployment_vs_J_PRUNED"
            ],
            "break_even_definition": config["break_even_definition"],
        },
        "byte_contract": config["byte_contract"],
        "byte_contract_rows": contract_rows,
        "selector_evidence": evidence,
        "selector_cycle_scenarios": selector_scenarios(evidence, config),
        "link_scan": config["link_profiles"],
        "row_count": len(rows),
        "sweep": rows,
        "key_thresholds": key,
        "warnings": config["warnings"],
        "limitations": [
            "not a Jetson Xavier NX measurement",
            "cosim verifies function only; OOC route timing is not an on-board sign-off",
            "T_score, T_sel_sw, T_vision_dense, T_vision_pruned and T_fpga_downstream are unmeasured",
            "numeric values are communication + FPGA-selector lower bounds, never a full break-even",
            "link throughput and fixed software+DMA overhead are scanned assumptions",
            "J_F_CONTINUE requires a real FPGA downstream core that does not yet exist in this repository",
        ],
    }


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def build_harness_result(
    *,
    payload: dict[str, Any],
    tests_status: str,
    files_changed: list[str],
    next_gate: str,
) -> dict[str, Any]:
    key = payload["key_thresholds"]
    return {
        "status": "HETEROGENEOUS_SELECTOR_BASELINE_CORRECTED_NO_FULL_BREAK_EVEN",
        "baselines": {
            "J_DENSE": payload["baselines"]["J_DENSE"],
            "J_PRUNED": payload["baselines"]["J_PRUNED"],
            "superseded": payload["baseline_confusion_fix"]["superseded"],
            "fix": payload["baseline_confusion_fix"]["fix"],
        },
        "contracts": {
            "dense_input": "G*4096 B int8[G,4,1024] + G*2 B uint16 Q0.15 score",
            "selected_output": "5-segment 64B-aligned layout; tokens_int8 = K*4096 B",
            "output_source": "evaluation/hardware_regular_selector_fixedpoint.fpga_transfer_layout",
            "score_generation": "T_score listed separately, never counted in FPGA selector cycles; absent from J_DENSE and NOT cancelled against dense",
            "comparison_A_vs_J_DENSE": payload["formulas"][
                "comparison_A_algorithm_vs_J_DENSE"
            ]["requirements"],
            "comparison_B_vs_J_PRUNED": payload["formulas"][
                "comparison_B_deployment_vs_J_PRUNED"
            ]["requirements"],
            "J_F_J_vs_J_PRUNED_vision_terms_cancelled": True,
            "fixed_overheads": "J_F_J=2 (send+return), J_F_CONTINUE=1 (send only), J_DENSE/J_PRUNED=0",
            "symbolic_missing_terms": payload["symbolic_missing_terms"],
            "prior_art": "reused configs/platform_agnostic_benefit_model.json link + fixed-overhead sensitivity definitions and the decimal serialization convention; did not inherit its stale 2026-09-10 EXTERNAL_SELECTOR_ROUND_TRIP_REJECT conclusion or its A6000 BF16 oracle stage timings",
        },
        "evidence": {
            "csynth_worst_cycles": payload["selector_evidence"]["csynth_worst_cycles"],
            "cosim_latency_cycles": payload["selector_evidence"]["cosim_latency_cycles"],
            "cosim_cases": payload["selector_evidence"]["cosim_cases"],
            "ooc_route": {
                "period_ns": payload["selector_evidence"]["ooc_clock_period_ns"],
                "wns_ns": payload["selector_evidence"]["ooc_wns_ns"],
                "status": payload["selector_evidence"]["ooc_status"],
            },
            "paths": payload["selector_evidence"]["paths"],
            "sha256": payload["selector_evidence"]["sha256"],
        },
        "placements": payload["placements"],
        "sweep": {
            "row_count": payload["row_count"],
            "comparisons": list(COMPARISONS),
            "baselines": list(BASELINES),
            "candidate_placements": {
                comparison: list(COMPARISON_CANDIDATES[comparison])
                for comparison in COMPARISONS
            },
            "group_counts": sorted({row["group_count"] for row in payload["sweep"]}),
            "budgets": sorted({row["budget"] for row in payload["sweep"]}),
            "throughput_MBps": payload["link_scan"]["effective_throughput_MBps"],
            "fixed_overhead_ms": payload["link_scan"][
                "fixed_software_and_dma_overhead_ms"
            ],
            "selector_cycle_scenarios": [
                s["id"] for s in payload["selector_cycle_scenarios"]
            ],
            "lower_bound_only": all(row["lower_bound_only"] for row in payload["sweep"]),
        },
        "key_thresholds": {
            "context": {
                "selector_cycle_scenario": key["scenario"],
                "throughput_MBps": key["throughput_MBps"],
                "fixed_overhead_ms": key["fixed_overhead_ms"],
            },
            "points": key["points"],
            "optimistic_min_known_lower_bound": key["optimistic_min_known_lower_bound"],
            "forbidden_claim": key["forbidden_claim"],
        },
        "tests": tests_status,
        "warnings": payload["warnings"] + payload["limitations"],
        "files_changed": files_changed,
        "next_gate": next_gate,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--report-stem", default=DEFAULT_REPORT_STEM)
    parser.add_argument("--harness-result", type=Path, default=DEFAULT_HARNESS_RESULT)
    parser.add_argument("--tests-status", default="NOT_RUN")
    parser.add_argument("--skip-harness-result", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(args.config.read_text())
    evidence = load_selector_evidence(config)
    contract_rows = byte_contract_rows(config["grid_group_counts"], config["budgets"])
    rows = build_sweep(config, evidence)
    key = key_thresholds(rows, config)
    payload = build_report_payload(config, evidence, contract_rows, rows, key)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"{args.report_stem}.json"
    csv_path = args.report_dir / f"{args.report_stem}.csv"
    md_path = args.report_dir / f"{args.report_stem}.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    )
    write_csv(csv_path, rows)
    write_markdown(md_path, config, evidence, contract_rows, rows, key)

    if not args.skip_harness_result:
        next_gate = (
            "Measure on the Jetson: T_score, T_sel_sw, T_vision_dense, T_vision_pruned, "
            "and the real Jetson<->ZCU102 effective throughput / fixed software+DMA overhead; "
            "provide and measure a real FPGA downstream core before any J_F_CONTINUE claim. "
            "No full break-even and no speedup may be quoted before these."
        )
        files_changed = [
            _repo_relative(args.config),
            "scripts/analyze_heterogeneous_selector_break_even.py",
            "evaluation/tests/test_heterogeneous_selector_break_even.py",
            _repo_relative(md_path),
            _repo_relative(csv_path),
            _repo_relative(json_path),
            "CURRENT_STATUS.md",
        ]
        harness = build_harness_result(
            payload=payload,
            tests_status=args.tests_status,
            files_changed=files_changed,
            next_gate=next_gate,
        )
        args.harness_result.write_text(
            json.dumps(harness, ensure_ascii=False, indent=2) + "\n"
        )

    best = key["optimistic_min_known_lower_bound"]["J_F_J_vs_J_PRUNED_T_sel_sw_threshold"]
    print("heterogeneous selector requirement model: MODEL_ONLY_BASELINE_CORRECTED")
    print(
        f"selector cycles: csynth worst {evidence['csynth_worst_cycles']}  "
        f"cosim min/avg/max {evidence['cosim_latency_cycles']['min']}/"
        f"{evidence['cosim_latency_cycles']['avg']}/{evidence['cosim_latency_cycles']['max']}"
    )
    print("baselines: J_DENSE, J_PRUNED  placements: J_DENSE, J_PRUNED, J_F_J, J_F_CONTINUE")
    print(f"sweep rows: {len(rows)}  report: {md_path}")
    print(
        f"J_F_J vs J_PRUNED: T_sel_sw must exceed {best['known_lower_bound_ms']:.4f} ms "
        f"(G={best['group_count']}, K={best['budget']}); no vision saving"
    )
    print("J_F_J vs J_DENSE: required S_prune > T_score + known lower bound")
    print("no full break-even and no speedup claimed; unmeasured: T_score, T_sel_sw,")
    print("T_vision_dense, T_vision_pruned, T_fpga_downstream")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
