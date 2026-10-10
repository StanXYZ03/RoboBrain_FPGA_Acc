"""Pure helpers for spatial-token reduction experiments.

The first experiment deliberately separates two questions:

* ``uniform`` measures the ordinary resolution/token trade-off.
* ``oracle_crop`` uses the labeled target box to preserve the relevant region.
  It is an unattainable upper bound, not a deployable method.

Both paths reduce the real Qwen3-VL image grid; they do not merely mask tokens
while leaving the model's compute unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image

from evaluation.metrics import parse_box, parse_point


@dataclass(frozen=True)
class ComputeProxy:
    encoder_tokens: int
    llm_visual_tokens: int
    encoder_macs: int
    llm_prefill_macs: int


def clamp_bbox_1000(values: Iterable[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in values)
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding box must have positive area")
    return x1, y1, x2, y2


def expanded_bbox_1000(
    values: Iterable[float], margin_fraction: float
) -> tuple[float, float, float, float]:
    if margin_fraction < 0:
        raise ValueError("margin_fraction must be non-negative")
    x1, y1, x2, y2 = clamp_bbox_1000(values)
    dx = (x2 - x1) * margin_fraction
    dy = (y2 - y1) * margin_fraction
    return (
        max(0.0, x1 - dx),
        max(0.0, y1 - dy),
        min(1000.0, x2 + dx),
        min(1000.0, y2 + dy),
    )


def centered_square_1000(
    point_1000: Iterable[float], half_extent: float
) -> tuple[float, float, float, float]:
    """Return a fixed-size square around a predicted point, shifted at edges."""
    if not (0.0 < half_extent <= 500.0):
        raise ValueError("half_extent must be in (0, 500]")
    x, y = (float(value) for value in point_1000)
    if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        raise ValueError("point must be in normalized 0-1000 coordinates")
    size = 2.0 * half_extent
    x1 = max(0.0, min(1000.0 - size, x - half_extent))
    y1 = max(0.0, min(1000.0 - size, y - half_extent))
    return x1, y1, x1 + size, y1 + size


def scout_crop_box_1000(
    task: str,
    scout_output_text: str,
    *,
    grounding_margin: float,
    point_half_extent: float,
) -> tuple[float, float, float, float] | None:
    """Create a crop only from a first-pass model output; no label is consumed."""
    if task == "pointing":
        point = parse_point(scout_output_text)
        return (
            centered_square_1000(point, point_half_extent)
            if point is not None
            else None
        )
    if task == "grounding":
        box = parse_box(scout_output_text)
        return expanded_bbox_1000(box, grounding_margin) if box is not None else None
    raise ValueError(f"unsupported task: {task}")


def crop_image_1000(
    image: Image.Image, crop_box_1000: Iterable[float]
) -> Image.Image:
    x1, y1, x2, y2 = clamp_bbox_1000(crop_box_1000)
    width, height = image.size
    pixel_box = (
        max(0, int(x1 * width / 1000.0)),
        max(0, int(y1 * height / 1000.0)),
        min(width, max(1, int(round(x2 * width / 1000.0)))),
        min(height, max(1, int(round(y2 * height / 1000.0)))),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError("pixel crop is empty")
    return image.crop(pixel_box)


def remap_point_from_crop(
    point_1000: Iterable[float], crop_box_1000: Iterable[float]
) -> tuple[float, float]:
    x, y = (float(value) for value in point_1000)
    x1, y1, x2, y2 = clamp_bbox_1000(crop_box_1000)
    return x1 + x * (x2 - x1) / 1000.0, y1 + y * (y2 - y1) / 1000.0


def remap_box_from_crop(
    box_1000: Iterable[float], crop_box_1000: Iterable[float]
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box_1000)
    mapped_x1, mapped_y1 = remap_point_from_crop((x1, y1), crop_box_1000)
    mapped_x2, mapped_y2 = remap_point_from_crop((x2, y2), crop_box_1000)
    return mapped_x1, mapped_y1, mapped_x2, mapped_y2


def remap_output_text(task: str, text: str, crop_box_1000: Iterable[float]) -> str:
    if task == "pointing":
        parsed = parse_point(text)
        if parsed is None:
            return text
        x, y = remap_point_from_crop(parsed, crop_box_1000)
        return f"[({x:.6f}, {y:.6f})]"
    if task == "grounding":
        parsed = parse_box(text)
        if parsed is None:
            return text
        x1, y1, x2, y2 = remap_box_from_crop(parsed, crop_box_1000)
        return f"[{x1:.6f}, {y1:.6f}, {x2:.6f}, {y2:.6f}]"
    raise ValueError(f"unsupported task: {task}")


def qwen3_vl_compute_proxy(
    encoder_tokens: int,
    *,
    merge_size: int = 2,
    vision_depth: int = 24,
    vision_hidden: int = 1024,
    vision_intermediate: int = 4096,
    text_depth: int = 36,
    text_hidden: int = 2560,
    text_intermediate: int = 9728,
    query_width: int = 4096,
    kv_width: int = 1024,
) -> ComputeProxy:
    if encoder_tokens <= 0 or encoder_tokens % (merge_size * merge_size) != 0:
        raise ValueError("encoder token count must be positive and merge aligned")
    llm_tokens = encoder_tokens // (merge_size * merge_size)

    # Vision block: QKV+O, two FFN projections, QK^T and AV. Patch embedding,
    # merger and DeepStack projection are omitted, so this is an explicit
    # dominant-operator proxy rather than a claim of exact whole-model MACs.
    vision_linear = 4 * encoder_tokens * vision_hidden * vision_hidden
    vision_ffn = 2 * encoder_tokens * vision_hidden * vision_intermediate
    vision_attention = 2 * encoder_tokens * encoder_tokens * vision_hidden
    encoder_macs = vision_depth * (vision_linear + vision_ffn + vision_attention)

    # Qwen3-VL text block: Q/K/V/O projections, SwiGLU gate/up/down, and the
    # visual-token self-attention term. Text/special tokens are intentionally
    # excluded so retention ratios remain comparable across samples.
    text_projection = llm_tokens * (
        text_hidden * (query_width + 2 * kv_width) + query_width * text_hidden
    )
    text_ffn = 3 * llm_tokens * text_hidden * text_intermediate
    text_attention = 2 * llm_tokens * llm_tokens * query_width
    llm_prefill_macs = text_depth * (text_projection + text_ffn + text_attention)
    return ComputeProxy(
        encoder_tokens=encoder_tokens,
        llm_visual_tokens=llm_tokens,
        encoder_macs=encoder_macs,
        llm_prefill_macs=llm_prefill_macs,
    )


def qwen3_vl_shallow_pruned_compute_proxy(
    full_encoder_tokens: int,
    kept_llm_visual_tokens: int,
    *,
    prune_after_layer: int = 5,
    merge_size: int = 2,
    vision_depth: int = 24,
    vision_hidden: int = 1024,
    vision_intermediate: int = 4096,
    text_depth: int = 36,
    text_hidden: int = 2560,
    text_intermediate: int = 9728,
    query_width: int = 4096,
    kv_width: int = 1024,
) -> ComputeProxy:
    """Dominant MAC proxy when pruning after an early vision layer.

    Layers through ``prune_after_layer`` run on the complete grid; remaining
    vision layers and every text layer run on the selected 2x2 patch groups.
    """
    merge_unit = merge_size * merge_size
    if full_encoder_tokens <= 0 or full_encoder_tokens % merge_unit != 0:
        raise ValueError("full encoder token count must be positive and merge aligned")
    full_groups = full_encoder_tokens // merge_unit
    if not (0 < kept_llm_visual_tokens <= full_groups):
        raise ValueError("kept visual tokens must be in (0, full groups]")
    if not (-1 <= prune_after_layer < vision_depth):
        raise ValueError("prune_after_layer must be -1 or index a vision layer")
    kept_encoder_tokens = kept_llm_visual_tokens * merge_unit

    def vision_layer(tokens: int) -> int:
        return (
            4 * tokens * vision_hidden * vision_hidden
            + 2 * tokens * vision_hidden * vision_intermediate
            + 2 * tokens * tokens * vision_hidden
        )

    full_layers = prune_after_layer + 1
    encoder_macs = full_layers * vision_layer(full_encoder_tokens) + (
        vision_depth - full_layers
    ) * vision_layer(kept_encoder_tokens)
    text_projection = kept_llm_visual_tokens * (
        text_hidden * (query_width + 2 * kv_width) + query_width * text_hidden
    )
    text_ffn = 3 * kept_llm_visual_tokens * text_hidden * text_intermediate
    text_attention = 2 * kept_llm_visual_tokens * kept_llm_visual_tokens * query_width
    llm_prefill_macs = text_depth * (text_projection + text_ffn + text_attention)
    return ComputeProxy(
        encoder_tokens=kept_encoder_tokens,
        llm_visual_tokens=kept_llm_visual_tokens,
        encoder_macs=encoder_macs,
        llm_prefill_macs=llm_prefill_macs,
    )


def retention_label(retention: float) -> str:
    if not (0.0 < retention <= 1.0):
        raise ValueError("retention must be in (0, 1]")
    return f"r{int(round(retention * 1000)):04d}"
