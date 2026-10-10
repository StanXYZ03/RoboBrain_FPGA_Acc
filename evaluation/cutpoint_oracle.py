"""Pure accounting helpers for the multi-cut-point visual-token oracle gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CutPoint:
    name: str
    prune_after_layer: int
    completed_vision_blocks: int
    transfer_space: str
    transfer_hidden: int


CUT_POINTS = {
    item.name: item
    for item in (
        CutPoint("patch_embedding", -1, 0, "patch_hidden", 1024),
        CutPoint("vision_block_1", 0, 1, "patch_hidden", 1024),
        CutPoint("vision_block_3", 2, 3, "patch_hidden", 1024),
        # This is Transformers' zero-based layer index 5 and the first
        # RoboBrain/Qwen3-VL DeepStack extraction point.
        CutPoint("deepstack_layer_5", 5, 6, "patch_hidden", 1024),
        # With a fixed oracle mask, selecting before the per-group merger is
        # numerically identical to merging all groups and selecting afterwards.
        CutPoint("post_merger", 23, 24, "llm_visual_hidden", 2560),
    )
}


def cut_point(name: str) -> CutPoint:
    try:
        return CUT_POINTS[name]
    except KeyError as error:
        raise ValueError(f"unknown cut point: {name}") from error


def transfer_accounting(
    *,
    cut_name: str,
    full_patch_tokens: int,
    kept_groups: int,
    merge_unit: int = 4,
    dtype_bytes: int = 2,
    metadata_bytes_per_group: int = 16,
    link_gbps: tuple[float, ...] = (1.0, 2.5, 10.0, 40.0),
    practical_efficiency: float = 0.8,
) -> dict[str, object]:
    """Return payload sizes and serialization lower bounds for two placements.

    ``one_way`` means the FPGA owns the graph prefix through this cut and emits
    only compacted state.  ``round_trip`` means another processor produces the
    dense cut tensor, sends it to FPGA for selection, and receives compacted
    state back.  Protocol/software latency is deliberately excluded.
    """
    point = cut_point(cut_name)
    if full_patch_tokens <= 0 or full_patch_tokens % merge_unit:
        raise ValueError("full_patch_tokens must be positive and merge aligned")
    full_groups = full_patch_tokens // merge_unit
    if not (0 < kept_groups <= full_groups):
        raise ValueError("kept_groups must be in (0, full_groups]")
    if dtype_bytes <= 0 or metadata_bytes_per_group < 0:
        raise ValueError("invalid transfer element size")
    if not (0.0 < practical_efficiency <= 1.0):
        raise ValueError("practical_efficiency must be in (0, 1]")

    if point.transfer_space == "patch_hidden":
        full_elements = full_patch_tokens * point.transfer_hidden
        compact_elements = kept_groups * merge_unit * point.transfer_hidden
    else:
        full_elements = full_groups * point.transfer_hidden
        compact_elements = kept_groups * point.transfer_hidden
    full_bytes = full_elements * dtype_bytes
    compact_bytes = compact_elements * dtype_bytes
    metadata_bytes = kept_groups * metadata_bytes_per_group
    one_way_bytes = compact_bytes + metadata_bytes
    round_trip_bytes = full_bytes + one_way_bytes

    def times(payload_bytes: int) -> dict[str, float]:
        return {
            f"{speed:g}GbE": payload_bytes * 8.0 / (speed * 1e9 * practical_efficiency)
            for speed in link_gbps
        }

    return {
        "cut_point": asdict(point),
        "dtype_bytes": dtype_bytes,
        "metadata_bytes_per_group": metadata_bytes_per_group,
        "practical_link_efficiency": practical_efficiency,
        "full_cut_tensor_bytes": full_bytes,
        "compact_cut_tensor_bytes": compact_bytes,
        "metadata_bytes": metadata_bytes,
        "one_way_bytes": one_way_bytes,
        "round_trip_bytes": round_trip_bytes,
        "one_way_serialization_seconds": times(one_way_bytes),
        "round_trip_serialization_seconds": times(round_trip_bytes),
        "excludes": [
            "protocol headers",
            "DMA setup",
            "driver scheduling",
            "tensor layout conversion",
            "compute time",
        ],
    }

