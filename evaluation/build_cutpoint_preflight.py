#!/usr/bin/env python3
"""Build a no-weight preflight for cut-point compute and communication costs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from transformers import AutoProcessor

from evaluation.cutpoint_oracle import CUT_POINTS, transfer_accounting
from evaluation.metrics import load_jsonl
from evaluation.run_hf_token_reduction import make_messages, processor_inputs, resolve_image
from evaluation.token_reduction import (
    qwen3_vl_compute_proxy,
    qwen3_vl_shallow_pruned_compute_proxy,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values: list[float | int]) -> float:
    return float(statistics.median(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--retentions", type=float, nargs="+", default=(1.0, 0.75, 0.5, 0.25)
    )
    args = parser.parse_args()
    samples = load_jsonl(args.manifest)
    if not samples or any(
        sample.get("dataset_role") != "cutpoint_oracle_dev" for sample in samples
    ):
        parser.error("manifest must be a non-empty cutpoint_oracle_dev set")
    processor = AutoProcessor.from_pretrained(str(args.processor), local_files_only=True)

    image_rows = []
    seen: set[str] = set()
    for sample in samples:
        image_key = str(sample["image"])
        if image_key in seen:
            continue
        seen.add(image_key)
        image_path = resolve_image(args.manifest, image_key)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        batch = processor_inputs(
            processor, make_messages(image_path, str(sample["prompt"])), image
        )
        grid = [int(value) for value in batch.image_grid_thw[0].tolist()]
        full_patch_tokens = math.prod(grid)
        image_rows.append(
            {
                "image": image_key,
                "grid_thw": grid,
                "full_patch_tokens": full_patch_tokens,
                "full_llm_visual_tokens": full_patch_tokens // 4,
            }
        )

    configs = []
    for cut_name, point in CUT_POINTS.items():
        for retention in args.retentions:
            samples_at_config = []
            for image in image_rows:
                full_patch_tokens = int(image["full_patch_tokens"])
                full_groups = full_patch_tokens // 4
                kept_groups = max(1, min(full_groups, math.ceil(full_groups * retention)))
                full_proxy = qwen3_vl_compute_proxy(full_patch_tokens)
                proxy = qwen3_vl_shallow_pruned_compute_proxy(
                    full_patch_tokens,
                    kept_groups,
                    prune_after_layer=point.prune_after_layer,
                )
                communication = transfer_accounting(
                    cut_name=cut_name,
                    full_patch_tokens=full_patch_tokens,
                    kept_groups=kept_groups,
                )
                samples_at_config.append(
                    {
                        "full_patch_tokens": full_patch_tokens,
                        "kept_groups": kept_groups,
                        "encoder_mac_ratio": proxy.encoder_macs / full_proxy.encoder_macs,
                        "llm_prefill_mac_ratio": (
                            proxy.llm_prefill_macs / full_proxy.llm_prefill_macs
                        ),
                        "one_way_bytes": communication["one_way_bytes"],
                        "round_trip_bytes": communication["round_trip_bytes"],
                        "one_way_10gbe_seconds": communication[
                            "one_way_serialization_seconds"
                        ]["10GbE"],
                        "round_trip_10gbe_seconds": communication[
                            "round_trip_serialization_seconds"
                        ]["10GbE"],
                    }
                )
            configs.append(
                {
                    "cut_name": cut_name,
                    "prune_after_layer_zero_based": point.prune_after_layer,
                    "completed_vision_blocks": point.completed_vision_blocks,
                    "transfer_space": point.transfer_space,
                    "requested_retention": retention,
                    "median_full_patch_tokens": median(
                        [row["full_patch_tokens"] for row in samples_at_config]
                    ),
                    "median_kept_groups": median(
                        [row["kept_groups"] for row in samples_at_config]
                    ),
                    "median_encoder_mac_ratio": median(
                        [row["encoder_mac_ratio"] for row in samples_at_config]
                    ),
                    "median_llm_prefill_mac_ratio": median(
                        [row["llm_prefill_mac_ratio"] for row in samples_at_config]
                    ),
                    "median_one_way_mib": median(
                        [row["one_way_bytes"] for row in samples_at_config]
                    )
                    / 2**20,
                    "median_round_trip_mib": median(
                        [row["round_trip_bytes"] for row in samples_at_config]
                    )
                    / 2**20,
                    "median_one_way_10gbe_ms": 1000.0
                    * median([row["one_way_10gbe_seconds"] for row in samples_at_config]),
                    "median_round_trip_10gbe_ms": 1000.0
                    * median(
                        [row["round_trip_10gbe_seconds"] for row in samples_at_config]
                    ),
                }
            )

    # Preserve the larger 4320-patch contract observed in the board-side
    # representative image.  The 512-pixel development set is appropriate for
    # a fail-fast accuracy gate but must not understate deployment traffic.
    canonical_full_patch_tokens = 4320
    canonical_configs = []
    canonical_full_proxy = qwen3_vl_compute_proxy(canonical_full_patch_tokens)
    for cut_name, point in CUT_POINTS.items():
        for retention in args.retentions:
            full_groups = canonical_full_patch_tokens // 4
            kept_groups = max(1, min(full_groups, math.ceil(full_groups * retention)))
            proxy = qwen3_vl_shallow_pruned_compute_proxy(
                canonical_full_patch_tokens,
                kept_groups,
                prune_after_layer=point.prune_after_layer,
            )
            communication = transfer_accounting(
                cut_name=cut_name,
                full_patch_tokens=canonical_full_patch_tokens,
                kept_groups=kept_groups,
            )
            canonical_configs.append(
                {
                    "cut_name": cut_name,
                    "completed_vision_blocks": point.completed_vision_blocks,
                    "requested_retention": retention,
                    "encoder_mac_ratio": (
                        proxy.encoder_macs / canonical_full_proxy.encoder_macs
                    ),
                    "llm_prefill_mac_ratio": (
                        proxy.llm_prefill_macs / canonical_full_proxy.llm_prefill_macs
                    ),
                    "one_way_mib": communication["one_way_bytes"] / 2**20,
                    "round_trip_mib": communication["round_trip_bytes"] / 2**20,
                    "one_way_10gbe_ms": 1000.0
                    * communication["one_way_serialization_seconds"]["10GbE"],
                    "round_trip_10gbe_ms": 1000.0
                    * communication["round_trip_serialization_seconds"]["10GbE"],
                    "one_way_1gbe_ms": 1000.0
                    * communication["one_way_serialization_seconds"]["1GbE"],
                    "round_trip_1gbe_ms": 1000.0
                    * communication["round_trip_serialization_seconds"]["1GbE"],
                }
            )

    payload = {
        "format": "cutpoint-oracle-preflight-v1",
        "status": "STATIC_ACCOUNTING_NOT_MEASURED_LATENCY",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "processor": str(args.processor.resolve()),
        "image_count": len(image_rows),
        "sample_count": len(samples),
        "image_token_rows": image_rows,
        "configs": configs,
        "canonical_4320_patch_contract": {
            "source": "existing ZCU102 representative image contract",
            "full_patch_tokens": canonical_full_patch_tokens,
            "full_llm_visual_tokens": canonical_full_patch_tokens // 4,
            "configs": canonical_configs,
        },
        "assumptions": {
            "activation_dtype_bytes": 2,
            "index_and_position_metadata_bytes_per_retained_group": 16,
            "practical_link_efficiency": 0.8,
            "latency_model_excludes": [
                "protocol headers",
                "DMA setup",
                "driver scheduling",
                "layout conversion",
                "compute time",
            ],
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# 多切入点 Oracle 实验：运行前审计",
        "",
        "状态：`STATIC_ACCOUNTING_NOT_MEASURED_LATENCY`。本表只计算真实预处理网格上的主算子 MAC 比例和链路串行化时间，不是推理实测。",
        "",
        f"开发集含 {len(image_rows)} 张新图、{len(samples)} 条 Pointing/Grounding 请求；与既有实验图像的重合检查由实验锁记录。",
        "",
        "| 切入点 | 已完成视觉块 | 保留率 | Encoder MAC | LLM prefill MAC | 单向载荷/MiB | 往返载荷/MiB | 10GbE单向/ms | 10GbE往返/ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in configs:
        lines.append(
            f"| {row['cut_name']} | {row['completed_vision_blocks']} | "
            f"{row['requested_retention']:.2f} | {row['median_encoder_mac_ratio']:.3f} | "
            f"{row['median_llm_prefill_mac_ratio']:.3f} | {row['median_one_way_mib']:.3f} | "
            f"{row['median_round_trip_mib']:.3f} | {row['median_one_way_10gbe_ms']:.3f} | "
            f"{row['median_round_trip_10gbe_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            "其中“单向”指 FPGA 驻留从输入到该切入点的连续前缀，只向 GPU/NPU 发送压缩后的状态；“往返”指 GPU/NPU 先生成稠密中间张量，再交给 FPGA 筛选并取回。后者额外搬运完整张量，尚未计入驱动和同步开销。",
            "",
            "切入点编号约定：`vision_block_1/3` 分别完成 1/3 个视觉块；`deepstack_layer_5` 对应 Transformers 的零基索引 5，即完成 6 个块，也是第一个 DeepStack 提取点；`post_merger` 用零基索引 23 的固定掩码实现，固定逐组 merger 下与 merger 后再选数值等价。",
            "",
            "## 既有板端4320 Patch合同",
            "",
            "开发集经过512像素缩放，不能代表既有板端4320 Patch大图的通信量。下表单列该合同，防止低估异构链路压力。",
            "",
            "| 切入点 | 保留率 | 单向/MiB | 往返/MiB | 1GbE单向/ms | 1GbE往返/ms | 10GbE单向/ms | 10GbE往返/ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in canonical_configs:
        lines.append(
            f"| {row['cut_name']} | {row['requested_retention']:.2f} | "
            f"{row['one_way_mib']:.3f} | {row['round_trip_mib']:.3f} | "
            f"{row['one_way_1gbe_ms']:.3f} | {row['round_trip_1gbe_ms']:.3f} | "
            f"{row['one_way_10gbe_ms']:.3f} | {row['round_trip_10gbe_ms']:.3f} |"
        )
    args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps({"json": str(args.json), "markdown": str(args.markdown), "images": len(image_rows), "configs": len(configs)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
