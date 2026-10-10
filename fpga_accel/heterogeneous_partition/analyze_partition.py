#!/usr/bin/env python3
"""Compute communication and decoder-bandwidth gates for heterogenous splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_bytes(elements: int, numerator: int, denominator: int) -> int:
    """Return whole-byte packed storage, rounding up incomplete format groups."""
    if elements < 0 or numerator <= 0 or denominator <= 0:
        raise ValueError("invalid packed-byte arguments")
    return (elements * numerator + denominator - 1) // denominator


def boundary_elements(contract: dict) -> dict[str, dict]:
    lang = contract["language"]
    work = contract["workload"]
    kv_per_token = lang["layers"] * lang["kv_heads"] * (
        lang["key_length"] + lang["value_length"]
    )
    return {
        "vision_to_llm_visual_hidden": {
            "elements": work["merged_visual_tokens"] * lang["hidden"],
            "transactions": 1,
            "meaning": "host Vision Encoder/merger output; FPGA runs language Prefill and Decode",
        },
        "full_prompt_hidden": {
            "elements": work["observed_total_prompt_tokens"] * lang["hidden"],
            "transactions": 1,
            "meaning": "already assembled LLM input hidden states before language Prefill",
        },
        "prefill_to_decoder_kv": {
            "elements": work["observed_total_prompt_tokens"] * kv_per_token,
            "transactions": 1,
            "meaning": "all-layer K and V cache after host Prefill; FPGA runs Decode",
        },
        "model_to_action_head_hidden": {
            "elements": lang["hidden"],
            "transactions": 1,
            "meaning": "one final hidden state; FPGA only runs a small action/safety head",
        },
        "per_layer_roundtrip_hidden": {
            "elements": lang["layers"] * lang["hidden"],
            "transactions": lang["layers"],
            "meaning": "one hidden vector exchanged at every Decoder layer; latency-sensitive anti-pattern",
        },
    }


def analyze(contract: dict) -> dict:
    formats = contract["formats"]
    links = contract["links"]
    periods_ms = {"3Hz": 1000.0 / 3.0, "5Hz": 200.0}
    boundaries = []
    for boundary_name, boundary in boundary_elements(contract).items():
        for format_name, fmt in formats.items():
            byte_count = packed_bytes(
                boundary["elements"], fmt["bytes_numerator"], fmt["bytes_denominator"]
            )
            link_results = []
            for link_name, link in links.items():
                latency_ms = byte_count / link["payload_bytes_per_second"] * 1000.0
                link_results.append({
                    "link": link_name,
                    "payload_bytes_per_second": link["payload_bytes_per_second"],
                    "measured": link["measured"],
                    "transfer_lower_bound_ms": latency_ms,
                    "max_fresh_transfers_per_second": (
                        link["payload_bytes_per_second"] / byte_count if byte_count else math.inf
                    ),
                    "period_share": {
                        rate: latency_ms / period for rate, period in periods_ms.items()
                    },
                    "within_20pct_period_budget": {
                        rate: latency_ms <= 0.2 * period for rate, period in periods_ms.items()
                    },
                    "note": link["note"],
                })
            boundaries.append({
                "boundary": boundary_name,
                "meaning": boundary["meaning"],
                "format": format_name,
                "elements": boundary["elements"],
                "bytes": byte_count,
                "mib": byte_count / (1024.0 * 1024.0),
                "transactions": boundary["transactions"],
                "links": link_results,
            })

    model = contract["model"]
    full_weight_bytes = model["decoder_weight_bytes_per_token"]
    constrained_weight_bytes = (
        model["decoder_layer_weight_bytes"] + model["output_norm_bytes"]
    )
    decoder = []
    for name, bandwidth in contract["decoder_bandwidths"].items():
        rate = bandwidth["bytes_per_second"]
        targets = {}
        for target in (3.0, 5.0):
            allowed = rate / target
            targets[f"{target:g}_token_per_second"] = {
                "allowed_weight_bytes_per_token": allowed,
                "required_reduction_from_full": max(0.0, 1.0 - allowed / full_weight_bytes),
                "full_stream_feasible_ignoring_other_traffic": full_weight_bytes <= allowed,
            }
        decoder.append({
            "bandwidth": name,
            **bandwidth,
            "full_vocabulary_upper_bound_token_per_second": rate / full_weight_bytes,
            "constrained_head_upper_bound_token_per_second": rate / constrained_weight_bytes,
            "targets": targets,
        })

    return {
        "schema_version": 1,
        "decision": {
            "runtime_ffn_numeric": "PASS (separate evidence)",
            "ordinary_1gbe_prefill_to_decoder_kv": "REJECT_FOR_PER_FRESH_FRAME",
            "vision_hidden_over_1gbe": "CONDITIONAL_ON_Q8_AND_MEASURED_LINK",
            "full_vocabulary_5_token_s_on_zcu102": "REQUIRES_WEIGHT_TRAFFIC_REDUCTION",
            "freeze_candidates": [
                "host Vision Encoder -> FPGA complete language Prefill+Decode",
                "host Vision+Prefill -> FPGA Decode only, only with fast link or reused/compressed KV",
                "host complete VLM -> FPGA action/safety head as low-latency control baseline",
            ],
        },
        "assumptions": {
            "communication_values_are_payload_only_lower_bounds": True,
            "startup_copy_layout_conversion_energy_included": False,
            "fresh_frame_rates": [3, 5],
            "communication_budget_fraction": 0.2,
            "decoder_bounds_ignore_kv_activation_and_control_traffic": True,
            "one_action_update_equals_one_generated_token": False,
        },
        "model_contract_sha256": None,
        "boundaries": boundaries,
        "decoder_weight_stream": {
            "full_vocabulary_bytes_per_token": full_weight_bytes,
            "constrained_head_bytes_per_token": constrained_weight_bytes,
            "output_head_saving_fraction": model["output_weight_bytes"] / full_weight_bytes,
            "bandwidth_cases": decoder,
        },
        "limitations": [
            "3-5 Hz is not generally equal to 3-5 token/s; text coordinates may require several tokens per update.",
            "A special action token or fixed-size action head is required before using one-token-per-update arithmetic.",
            "No Jetson-ZCU102 physical link was measured in this milestone.",
            "Q8_0/Q4_0 KV entries are packed-payload sizes; conversion metadata/work and accuracy are unvalidated.",
            "Decoder token/s values are bandwidth-only upper bounds, not projected end-to-end performance.",
        ],
    }


def render_markdown(result: dict) -> str:
    selected = {(x["boundary"], x["format"]): x for x in result["boundaries"]}

    def transfer(boundary: str, fmt: str, link: str) -> tuple[float, float]:
        row = selected[(boundary, fmt)]
        item = next(x for x in row["links"] if x["link"] == link)
        return row["mib"], item["transfer_lower_bound_ms"]

    kv_mib, kv_1g = transfer("prefill_to_decoder_kv", "BF16_or_FP16", "1GbE_80pct_scenario")
    kv_q8_mib, kv_q8_1g = transfer("prefill_to_decoder_kv", "Q8_0_payload", "1GbE_80pct_scenario")
    hid_mib, hid_1g = transfer("vision_to_llm_visual_hidden", "BF16_or_FP16", "1GbE_80pct_scenario")
    hid_q8_mib, hid_q8_1g = transfer("vision_to_llm_visual_hidden", "Q8_0_payload", "1GbE_80pct_scenario")
    decoder = result["decoder_weight_stream"]
    cases = {x["bandwidth"]: x for x in decoder["bandwidth_cases"]}
    peak = cases["DDR4_2133_line_rate"]
    pct85 = cases["DDR4_2133_85pct_scenario"]
    reduction_5 = pct85["targets"]["5_token_per_second"]["required_reduction_from_full"]
    return f"""# Jetson–FPGA 异构切分第一轮门控

## 结论

本轮冻结三种候选边界，但不直接确定最终架构：

1. Jetson完成视觉编码，FPGA完成语言Prefill和Decode；
2. Jetson完成视觉编码与Prefill，FPGA只做Decode；
3. Jetson完成完整VLM，FPGA只做固定大小的动作或安全头。

普通1 GbE下，方案2若每张新图传一次完整KV Cache，通信量过大，应停止作为逐帧方案；
只有使用更快互连、压缩KV并跨帧复用时才保留。方案1的边界较小，但FPGA仍需承担完整
语言Prefill，现有ZCU102性能还不能支持3–5 Hz。方案3通信最小，却没有加速VLM主体，适合
作为低时延控制对照组。

## 模型合同

- 语言层数36，隐藏维2560，KV Head 8，K/V维度各128；
- 代表性图像合并后1080个视觉Token，实测完整Prompt为1142 Token；
- 每个Token的K+V元素数：`36 × 8 × (128 + 128) = 73,728`；
- Decode每Token需流过的权重有效载荷约{decoder['full_vocabulary_bytes_per_token'] / 1e9:.3f} GB，
  包含36层权重和输出头，不把整张输入嵌入表重复计入。

## 跨设备通信下限

以下1 GbE按100 MB/s有效载荷情景计算，并非Jetson–ZCU102实测；未计协议、复制、布局转换和启动开销。

| 切分边界 | 格式 | 数据量 | 单向传输下限 |
|---|---:|---:|---:|
| Vision merger → LLM | BF16/FP16 | {hid_mib:.2f} MiB | {hid_1g:.1f} ms |
| Vision merger → LLM | Q8_0 payload | {hid_q8_mib:.2f} MiB | {hid_q8_1g:.1f} ms |
| Prefill → Decoder KV | BF16/FP16 | {kv_mib:.2f} MiB | {kv_1g:.1f} ms |
| Prefill → Decoder KV | Q8_0 payload | {kv_q8_mib:.2f} MiB | {kv_q8_1g:.1f} ms |

5 Hz周期是200 ms，若通信只允许占20%，预算为40 ms。由此可见：BF16视觉隐藏状态在
1 GbE下已经超过预算，Q8_0情景可能进入预算；完整KV即使Q8_0仍接近1 s，不能逐帧迁移。

## ZCU102 Decode权重带宽上限

| 带宽口径 | 完整输出头上限 | 约束输出头上限 |
|---|---:|---:|
| DDR4-2133线速17.065 GB/s | {peak['full_vocabulary_upper_bound_token_per_second']:.2f} token/s | {peak['constrained_head_upper_bound_token_per_second']:.2f} token/s |
| 线速85%情景14.505 GB/s | {pct85['full_vocabulary_upper_bound_token_per_second']:.2f} token/s | {pct85['constrained_head_upper_bound_token_per_second']:.2f} token/s |

这些是忽略KV读取、激活、控制、空泡的上限。若按85%带宽情景追求5 token/s，完整模型
至少还需减少{reduction_5 * 100:.1f}%的每Token权重流量。去掉完整词表输出头只能减少
{decoder['output_head_saving_fraction'] * 100:.1f}%，仍不够，因此还需要结构化稀疏、层/通道裁剪、
权重缓存复用或更高带宽平台。

## 对3–5 Hz目标的解释

3–5 Hz只有在每次控制更新对应一个固定大小动作Token或动作头输出时，才近似要求3–5 token/s。
若坐标、轨迹继续以多个文本Token生成，所需token/s还要乘以每次输出Token数。后续应把“文本化动作”
和“固定动作头/动作Token”作为两条基线分别报告，不能直接把token/s写成图片Hz。

## 下一门槛

1. 在真实互连确定后实测有效带宽、P50/P95、复制次数和启动延迟；
2. 对KV的BF16、Q8和更低位宽做真实任务精度门控；
3. 建立Decode权重流微基准，目标是先接近DDR有效带宽，而不是继续增加GEMM PE；
4. 计算达到3/5 token/s所需的结构化权重削减比例，并与任务精度联合验证。

原始机读结果见同目录`summary.json`。本轮未使用CUDA、未访问板卡、未生成bitstream，也未修改SD卡。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path(__file__).with_name("model_contract.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    result = analyze(contract)
    result["model_contract_sha256"] = sha256_file(args.contract)
    with (args.output_dir / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with (args.output_dir / "report.md").open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(result))
    print(json.dumps(result["decision"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
