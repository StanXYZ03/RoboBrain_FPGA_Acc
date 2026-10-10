"""Pure helpers for task-aware global-plus-ROI visual refinement."""

from __future__ import annotations

from typing import Any, Iterable

from PIL import Image

from evaluation.token_reduction import clamp_bbox_1000


def resize_long_edge(
    image: Image.Image, long_edge: int, *, allow_upscale: bool = False
) -> Image.Image:
    """Resize while preserving aspect ratio and avoid invented detail by default."""
    if long_edge <= 0:
        raise ValueError("long_edge must be positive")
    width, height = image.size
    scale = long_edge / max(width, height)
    if not allow_upscale:
        scale = min(scale, 1.0)
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    if resized == image.size:
        return image.copy()
    return image.resize(resized, Image.Resampling.LANCZOS)


def crop_image_exact_1000(
    image: Image.Image, crop_box_1000: Iterable[float]
) -> tuple[Image.Image, tuple[float, float, float, float], tuple[int, int, int, int]]:
    """Crop an ROI and return the exact normalized box induced by pixel rounding."""
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
    effective = (
        1000.0 * pixel_box[0] / width,
        1000.0 * pixel_box[1] / height,
        1000.0 * pixel_box[2] / width,
        1000.0 * pixel_box[3] / height,
    )
    return image.crop(pixel_box), effective, pixel_box


def make_global_roi_messages(prompt: str) -> list[dict[str, Any]]:
    """Create an unambiguous two-image prompt with ROI-relative output."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1 is a low-resolution view of the full scene.\n"},
                {"type": "image"},
                {"type": "text", "text": "\nImage 2 is a high-resolution crop from Image 1.\n"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "\nUse Image 1 for context and Image 2 for precise localization. "
                        "Report every coordinate relative to Image 2, normalized from 0 to 1000. "
                        + prompt
                    ),
                },
            ],
        }
    ]


def make_labeled_global_roi_messages(prompt: str, *, roi_first: bool) -> list[dict[str, Any]]:
    """Use Qwen's native Picture IDs and an explicit output-coordinate contract."""
    roi_number = 1 if roi_first else 2
    global_number = 2 if roi_first else 1
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"Picture {global_number} is a low-resolution view of the full scene. "
                        f"Picture {roi_number} is an enlarged crop from that scene. "
                        f"Use Picture {global_number} only for scene context and perform precise "
                        f"localization in Picture {roi_number}. Return every coordinate relative "
                        f"to Picture {roi_number}, normalized from 0 to 1000. "
                        + prompt
                    ),
                },
            ],
        }
    ]


def make_roi_only_messages(prompt: str) -> list[dict[str, Any]]:
    """Create a one-crop prompt using the same ROI-relative output contract."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "This image is a crop from a larger scene. Report every coordinate "
                        "relative to this crop, normalized from 0 to 1000. " + prompt
                    ),
                },
            ],
        }
    ]


def grid_token_counts(
    image_grid_thw: Iterable[Iterable[int]], *, merge_size: int
) -> dict[str, Any]:
    if merge_size <= 0:
        raise ValueError("merge_size must be positive")
    grids = [[int(value) for value in grid] for grid in image_grid_thw]
    if not grids or any(len(grid) != 3 for grid in grids):
        raise ValueError("image_grid_thw must contain [t, h, w] entries")
    encoder_per_image = [grid[0] * grid[1] * grid[2] for grid in grids]
    merge_area = merge_size * merge_size
    if any(tokens % merge_area for tokens in encoder_per_image):
        raise ValueError("encoder token count is not spatial-merge aligned")
    llm_per_image = [tokens // merge_area for tokens in encoder_per_image]
    return {
        "image_grid_thw": grids,
        "encoder_tokens_per_image": encoder_per_image,
        "llm_visual_tokens_per_image": llm_per_image,
        "encoder_tokens_total": sum(encoder_per_image),
        "llm_visual_tokens_total": sum(llm_per_image),
    }
