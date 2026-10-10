"""CPU-only bit-exact tests for the fixed-point selector golden."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from evaluation import hardware_regular_selector_fixedpoint as fp
from evaluation.hardware_regular_token_selector import (
    fpga_transfer_layout as float_fpga_transfer_layout,
)
from evaluation.hardware_regular_token_selector import (
    select_hardware_regular_groups,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _tokens(group_count: int) -> np.ndarray:
    group = np.arange(group_count)[:, None, None]
    patch = np.arange(fp.MERGE_UNIT)[None, :, None]
    feature = np.arange(fp.FEATURE_DIM)[None, None, :]
    return (((group * 7 + patch * 3 + feature * 13) % 256) - 128).astype(np.int8)


def _scores(group_count: int, mode: str = "rand", seed: int = 12345) -> np.ndarray:
    if mode == "zero":
        return np.zeros((group_count,), dtype=np.uint16)
    if mode == "same":
        return np.full((group_count,), 12345, dtype=np.uint16)
    base = (np.arange(group_count).astype(np.uint64) * 1103515245 + seed) % 32768
    return base.astype(np.uint16)


def _remaining_by_region(selection: fp.FixedpointSelection, regions: fp.FixedpointRegions):
    real_indices = {
        int(index)
        for index, source in zip(selection.group_index, selection.source_type)
        if int(source) == fp.SOURCE_REAL
    }
    result = {}
    for region_id in regions.non_empty:
        result[region_id] = [
            group for group in regions.members[region_id] if group not in real_indices
        ]
    return result


def _context_row(selection: fp.FixedpointSelection, region_id: int) -> int:
    matches = [
        row
        for row in range(selection.output_count)
        if int(selection.source_type[row]) == fp.SOURCE_CONTEXT
        and int(selection.region_id[row]) == region_id
    ]
    if len(matches) != 1:
        raise AssertionError(f"region {region_id} has {len(matches)} context rows")
    return int(matches[0])


def _legacy_bands(size: int) -> list[tuple[int, int]]:
    return [(r * size // 3, (r + 1) * size // 3) for r in range(3)]


def _legacy_select(
    tokens: np.ndarray,
    scores: np.ndarray,
    grid_h: int,
    grid_w: int,
    budget: int,
    *,
    local_cap: int = fp.LOCAL_CANDIDATE_CAP,
    scale: float = 1.0,
    candidate_mode: str = "full_sort",
) -> dict:
    """Independent literal re-implementation of the *pre-refactor* algorithm.

    It collects every region member and does full sorts (the O(G^2) shape the
    HLS kernel no longer uses) so it can cross-check the frozen golden on
    random inputs.  It is deliberately written with different data structures.

    ``candidate_mode="v3_insertion"`` instead reproduces the frozen v3 bounded
    insertion kernel (per-region stable insertion of exactly ``local_cap``
    candidates, bounded insertion sort of the <= 144 pool and of the selected
    set) so the new v4 multi-pass kernel can be diffed against it.
    """
    tokens = np.asarray(tokens, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.uint16)
    group_count = int(tokens.shape[0])

    def error(status: int) -> dict:
        return {
            "status": status,
            "output_count": 0,
            "real_count": 0,
            "context_count": 0,
            "tokens_out": np.zeros((0, fp.MERGE_UNIT, fp.FEATURE_DIM), dtype=np.int8),
            "group_index": [],
            "source_type": [],
            "region_id": [],
            "valid": [],
            "scale_out": float(scale),
        }

    if group_count < 1 or group_count > fp.MAX_GROUP_INDEX or group_count > fp.MAX_GROUPS:
        return error(fp.STATUS_ERR_GROUP_LIMIT)
    if (
        grid_h < 1
        or grid_w < 1
        or grid_h > fp.MAX_GRID_DIM
        or grid_w > fp.MAX_GRID_DIM
        or grid_h * grid_w != group_count
    ):
        return error(fp.STATUS_ERR_GRID)
    if budget <= 0 or budget % fp.REQUIRED_BUDGET_ALIGNMENT != 0 or budget > fp.MAX_BUDGET:
        return error(fp.STATUS_ERR_BUDGET)
    if local_cap <= 0 or local_cap > fp.MAX_LOCAL_CANDIDATE_CAP:
        return error(fp.STATUS_ERR_LOCAL_CAP)

    members: list[list[int]] = []
    region_of = [-1] * group_count
    row_bands = _legacy_bands(grid_h)
    col_bands = _legacy_bands(grid_w)
    for region_row in range(3):
        for region_col in range(3):
            region_id = region_row * 3 + region_col
            current: list[int] = []
            for row in range(row_bands[region_row][0], row_bands[region_row][1]):
                for col in range(col_bands[region_col][0], col_bands[region_col][1]):
                    group = row * grid_w + col
                    current.append(group)
                    region_of[group] = region_id
            members.append(current)

    if group_count <= budget:
        return {
            "status": fp.STATUS_OK,
            "output_count": group_count,
            "real_count": group_count,
            "context_count": 0,
            "tokens_out": np.ascontiguousarray(tokens.copy()),
            "group_index": list(range(group_count)),
            "source_type": [fp.SOURCE_REAL] * group_count,
            "region_id": [region_of[group] for group in range(group_count)],
            "valid": [1] * group_count,
            "scale_out": float(scale),
        }

    non_empty = [region for region in range(fp.NUM_REGIONS) if members[region]]
    context_regions = [region for region in non_empty if len(members[region]) >= 2]
    context_count = len(context_regions)
    if budget - context_count < len(non_empty):
        return error(fp.STATUS_ERR_INSUFFICIENT)
    real_budget = budget - context_count

    anchors = {
        region: min(members[region], key=lambda group: (-int(scores[group]), group))
        for region in non_empty
    }
    reserved = set(context_regions)
    anchor_values = set(anchors.values())

    def _key(group: int) -> tuple[int, int]:
        return (-int(scores[group]), int(group))

    if candidate_mode == "v3_insertion":
        # Literal v3 stable bounded insertion: each region keeps exactly the top
        # `local_cap` non-anchor candidates best-first (higher score, lower
        # index); the <= 144 pool and the selected set use bounded insertion
        # sorts.  This is the frozen kernel the v4 multi-pass must match.
        local_top: dict[int, list[int]] = {region: [] for region in non_empty}
        for group in range(group_count):
            region = region_of[group]
            if group in anchor_values:
                continue
            reserve = 1 if region in reserved else 0
            allowed = max(0, len(members[region]) - 1 - reserve)
            limit = min(local_cap, allowed)
            if limit <= 0:
                continue
            current = local_top[region]
            if len(current) < limit:
                pos = len(current)
                while pos > 0 and not (_key(current[pos - 1]) < _key(group)):
                    pos -= 1
                current.insert(pos, group)
            elif _key(group) < _key(current[limit - 1]):
                pos = limit - 1
                while pos > 0 and not (_key(current[pos - 1]) < _key(group)):
                    pos -= 1
                current.insert(pos, group)
                current.pop()
        pool = [group for region in non_empty for group in local_top[region]]
        for i in range(1, len(pool)):
            key_group = pool[i]
            j = i - 1
            while j >= 0 and not (_key(pool[j]) < _key(key_group)):
                pool[j + 1] = pool[j]
                j -= 1
            pool[j + 1] = key_group
    else:
        candidates: set[int] = set()
        for region in non_empty:
            ranked = sorted(
                (group for group in members[region] if group != anchors[region]),
                key=_key,
            )
            reserve = 1 if region in reserved else 0
            allowed = max(0, len(members[region]) - 1 - reserve)
            take = min(len(ranked), local_cap, allowed)
            candidates.update(ranked[:take])
        pool = sorted(candidates - anchor_values, key=_key)

    details_needed = real_budget - len(anchors)
    if len(pool) < details_needed:
        return error(fp.STATUS_ERR_CANDIDATE_POOL)
    details = pool[:details_needed]
    selected = list(anchors.values()) + details
    for i in range(1, len(selected)):
        key_group = selected[i]
        j = i - 1
        while j >= 0 and selected[j] > key_group:
            selected[j + 1] = selected[j]
            j -= 1
        selected[j + 1] = key_group
    selected_set = set(selected)

    entries: list[tuple[int, int, int, np.ndarray]] = []
    for group in selected:
        entries.append((int(group), fp.SOURCE_REAL, int(region_of[group]), tokens[group].copy()))
    for region in context_regions:
        remaining = [group for group in members[region] if group not in selected_set]
        if not remaining:
            return error(fp.STATUS_ERR_INTERNAL)
        weights = [int(scores[group]) for group in remaining]
        total = sum(weights)
        if total == 0:
            weights = [1] * len(remaining)
            total = len(remaining)
        accumulator = np.zeros((fp.MERGE_UNIT, fp.FEATURE_DIM), dtype=np.int64)
        for weight, group in zip(weights, remaining):
            accumulator += weight * tokens[group].astype(np.int64)
        sign = np.where(accumulator < 0, -1, 1)
        pooled = (sign * ((2 * np.abs(accumulator) + total) // (2 * total))).astype(np.int8)
        row_sum = sum(weight * (group // grid_w) for weight, group in zip(weights, remaining))
        col_sum = sum(weight * (group % grid_w) for weight, group in zip(weights, remaining))
        representative = min(
            remaining,
            key=lambda group: (
                ((group // grid_w) * total - row_sum) ** 2
                + ((group % grid_w) * total - col_sum) ** 2,
                group,
            ),
        )
        entries.append((int(representative), fp.SOURCE_CONTEXT, int(region), pooled))

    entries.sort(key=lambda entry: (entry[0], entry[1], entry[2]))
    return {
        "status": fp.STATUS_OK,
        "output_count": len(entries),
        "real_count": sum(1 for entry in entries if entry[1] == fp.SOURCE_REAL),
        "context_count": sum(1 for entry in entries if entry[1] == fp.SOURCE_CONTEXT),
        "tokens_out": np.stack([entry[3] for entry in entries], axis=0).astype(np.int8),
        "group_index": [entry[0] for entry in entries],
        "source_type": [entry[1] for entry in entries],
        "region_id": [entry[2] for entry in entries],
        "valid": [1] * len(entries),
        "scale_out": float(scale),
    }


def _load_fixture_generator():
    import importlib.util
    import sys

    path = WORKSPACE_ROOT / "fpga_accel" / "token_selector_hls" / "tools" / "generate_fixtures.py"
    spec = importlib.util.spec_from_file_location("_ts_fixture_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the module through sys.modules while executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedpointSelectorTest(unittest.TestCase):
    # -- identity -----------------------------------------------------------
    def test_identity_bypass_is_byte_exact(self) -> None:
        cases = [(4, 4, 48), (8, 8, 64), (9, 9, 96), (2, 9, 48), (5, 4, 48)]
        for grid_h, grid_w, budget in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                tokens = _tokens(group_count)
                scores = _scores(group_count)
                result = fp.select_fixedpoint(
                    tokens, scores, grid_h, grid_w, budget, scale=0.03125
                )
                self.assertEqual(result.status, fp.STATUS_OK)
                self.assertEqual(result.output_count, group_count)
                self.assertEqual(result.real_count, group_count)
                self.assertEqual(result.context_count, 0)
                self.assertTrue(
                    np.array_equal(result.tokens_out, tokens),
                    "identity bypass must copy tokens byte-for-byte",
                )
                self.assertTrue(
                    np.array_equal(result.group_index, np.arange(group_count, dtype=np.uint16))
                )
                self.assertTrue(np.all(result.source_type == fp.SOURCE_REAL))
                self.assertTrue(np.all(result.valid == 1))
                self.assertEqual(result.scale_out, 0.03125)
                regions = fp.build_fixedpoint_regions(grid_h, grid_w)
                self.assertTrue(
                    np.array_equal(result.region_id, regions.region_of_group.astype(np.uint8))
                )

    # -- budget / structure -------------------------------------------------
    def test_strict_budget_dynamic_context_and_uniqueness(self) -> None:
        cases = [
            ((9, 9), 48),
            ((9, 9), 64),
            ((10, 12), 48),
            ((10, 12), 64),
            ((10, 12), 96),
        ]
        for (grid_h, grid_w), budget in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                tokens = _tokens(group_count)
                scores = _scores(group_count)
                result = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
                self.assertEqual(result.status, fp.STATUS_OK)
                self.assertEqual(result.output_count, budget)
                self.assertEqual(result.real_count + result.context_count, budget)
                self.assertLessEqual(result.context_count, fp.NUM_REGIONS)
                self.assertEqual(result.context_count, len(result.context_region_ids))
                indices = [int(value) for value in result.group_index]
                self.assertEqual(len(indices), len(set(indices)))
                self.assertEqual(indices, sorted(indices))
                self.assertTrue(np.all(result.valid == 1))
                self.assertTrue(np.all(result.region_id < fp.NUM_REGIONS))

    def test_grid_shapes_with_empty_and_single_regions(self) -> None:
        # 2x25 > 48 with three empty region rows; 2x9 / 5x4 compress at K=16 and
        # exercise dynamic context counts (6 and 7) plus empty/single-member regions.
        cases = [((2, 25), 48, 6), ((2, 9), 16, 6), ((5, 4), 16, 7)]
        for (grid_h, grid_w), budget, expected_context in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                regions = fp.build_fixedpoint_regions(grid_h, grid_w)
                result = fp.select_fixedpoint(
                    _tokens(group_count), _scores(group_count), grid_h, grid_w, budget
                )
                self.assertEqual(result.status, fp.STATUS_OK)
                self.assertEqual(result.output_count, budget)
                self.assertEqual(result.context_count, expected_context)
                self.assertEqual(result.context_count, len(regions.context_regions))
                # one anchor per non-empty region
                anchors = set(result.anchors)
                self.assertEqual(len(anchors), len(regions.non_empty))
                for region_id in regions.non_empty:
                    region_anchors = anchors.intersection(regions.members[region_id])
                    self.assertEqual(len(region_anchors), 1)
                # every context region keeps at least one unselected group
                remaining = _remaining_by_region(result, regions)
                for region_id in regions.context_regions:
                    self.assertGreaterEqual(len(remaining[region_id]), 1)
                # single-member regions do not produce context
                for region_id in regions.non_empty:
                    if len(regions.members[region_id]) < 2:
                        self.assertNotIn(region_id, result.context_region_ids)

    def test_anchor_is_highest_score_lowest_index(self) -> None:
        grid_h, grid_w, budget = 9, 9, 48
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count)
        regions = fp.build_fixedpoint_regions(grid_h, grid_w)
        result = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
        for region_id in regions.non_empty:
            members = regions.members[region_id]
            expected = min(members, key=lambda group: (-int(scores[group]), group))
            self.assertIn(expected, set(result.anchors))

    # -- score degeneracies -------------------------------------------------
    def test_all_same_score_breaks_ties_by_low_index(self) -> None:
        grid_h, grid_w, budget = 9, 9, 48
        group_count = grid_h * grid_w
        result = fp.select_fixedpoint(
            _tokens(group_count), _scores(group_count, "same"), grid_h, grid_w, budget
        )
        regions = fp.build_fixedpoint_regions(grid_h, grid_w)
        for region_id in regions.non_empty:
            expected_anchor = min(regions.members[region_id])
            self.assertIn(expected_anchor, set(result.anchors))
        self.assertTrue(np.all(result.valid == 1))

    def test_all_zero_score_uses_uniform_weights(self) -> None:
        grid_h, grid_w, budget = 9, 9, 48
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count, "zero")
        result = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
        regions = fp.build_fixedpoint_regions(grid_h, grid_w)
        remaining = _remaining_by_region(result, regions)
        for region_id in regions.context_regions:
            rem = remaining[region_id]
            weights, total = fp.context_pooling_weights_q(scores, rem)
            self.assertTrue(all(weight == 1 for weight in weights))
            self.assertEqual(total, len(rem))
            expected = fp.pool_context_tokens_q(tokens, scores, rem)
            row = _context_row(result, region_id)
            self.assertTrue(np.array_equal(result.tokens_out[row], expected))

    # -- determinism --------------------------------------------------------
    def test_repeated_runs_are_bit_identical(self) -> None:
        grid_h, grid_w, budget = 10, 12, 64
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count)
        first = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
        for _ in range(4):
            again = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
            self.assertTrue(np.array_equal(first.tokens_out, again.tokens_out))
            self.assertTrue(np.array_equal(first.group_index, again.group_index))
            self.assertTrue(np.array_equal(first.source_type, again.source_type))
            self.assertTrue(np.array_equal(first.region_id, again.region_id))

    # -- context pooling / representative -----------------------------------
    def test_context_pooling_matches_manual_integer_math(self) -> None:
        grid_h, grid_w, budget = 10, 12, 48
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count)
        result = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
        regions = fp.build_fixedpoint_regions(grid_h, grid_w)
        remaining = _remaining_by_region(result, regions)
        for region_id in result.context_region_ids:
            rem = remaining[region_id]
            weights = [int(scores[group]) for group in rem]
            total = sum(weights)
            for patch in range(fp.MERGE_UNIT):
                for feature in range(0, fp.FEATURE_DIM, 97):
                    numerator = sum(
                        weight * int(tokens[group][patch][feature])
                        for weight, group in zip(weights, rem)
                    )
                    expected = fp.round_half_away_from_zero(numerator, total)
                    row = _context_row(result, region_id)
                    self.assertEqual(int(result.tokens_out[row][patch][feature]), expected)

    def test_representative_is_nearest_and_tie_low_index(self) -> None:
        grid_h, grid_w = 5, 4
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count)
        regions = fp.build_fixedpoint_regions(grid_h, grid_w)
        for region_id in regions.context_regions:
            members = [group for group in regions.members[region_id]]
            representative = fp.context_representative_q(scores, members, grid_w)
            weights, total = fp.context_pooling_weights_q(scores, members)
            row_sum = sum(
                weight * (group // grid_w) for weight, group in zip(weights, members)
            )
            col_sum = sum(
                weight * (group % grid_w) for weight, group in zip(weights, members)
            )
            best = None
            for group in members:
                delta_row = (group // grid_w) * total - row_sum
                delta_col = (group % grid_w) * total - col_sum
                key = (delta_row * delta_row + delta_col * delta_col, group)
                if best is None or key < best:
                    best = key
            self.assertEqual(representative, best[1])

    def test_context_region_representatives_are_globally_unique(self) -> None:
        grid_h, grid_w, budget = 10, 12, 64
        group_count = grid_h * grid_w
        result = fp.select_fixedpoint(
            _tokens(group_count), _scores(group_count), grid_h, grid_w, budget
        )
        context_indices = [
            int(index)
            for index, source in zip(result.group_index, result.source_type)
            if int(source) == fp.SOURCE_CONTEXT
        ]
        self.assertEqual(len(context_indices), len(set(context_indices)))
        self.assertEqual(result.context_count, len(context_indices))

    # -- rejection paths ----------------------------------------------------
    def test_illegal_budget_is_rejected(self) -> None:
        grid_h, grid_w = 9, 9
        group_count = grid_h * grid_w
        tokens = _tokens(group_count)
        scores = _scores(group_count)
        for budget in (0, -48, 1, 16 - 1, 47, 80 + 1, 100, 33, 112, 128):
            with self.subTest(budget=budget):
                result = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
                self.assertEqual(result.status, fp.STATUS_ERR_BUDGET)
                self.assertEqual(result.output_count, 0)
                self.assertEqual(result.status_name, "ERR_BUDGET")
        for budget in (16, 32, 48, 64, 80, 96):
            self.assertEqual(fp.budget_status(budget), fp.STATUS_OK)
        for budget in (112, 128, 4096):
            self.assertEqual(fp.budget_status(budget), fp.STATUS_ERR_BUDGET)
        # the compile-time bound is what sizes every HLS output loop.
        self.assertEqual(fp.MAX_BUDGET, 96)

    def test_grid_mismatch_is_rejected(self) -> None:
        result = fp.select_fixedpoint(_tokens(18), _scores(18), 9, 9, 48)
        self.assertEqual(result.status, fp.STATUS_ERR_GRID)
        result = fp.select_fixedpoint(_tokens(18), _scores(18), 0, 5, 48)
        self.assertEqual(result.status, fp.STATUS_ERR_GRID)

    # -- MAX_GRID_DIM boundary / extreme aspect ratios -----------------------
    def test_grid_dim_boundary_128_and_129(self) -> None:
        # 128 is accepted by the helper and by the selection kernel.
        self.assertTrue(fp.grid_dims_ok(128, 1))
        self.assertTrue(fp.grid_dims_ok(1, 128))
        self.assertTrue(fp.grid_dims_ok(128, 32))
        self.assertTrue(fp.grid_dims_ok(2, 128))
        self.assertFalse(fp.grid_dims_ok(129, 1))
        self.assertFalse(fp.grid_dims_ok(1, 129))
        self.assertFalse(fp.grid_dims_ok(0, 5))
        self.assertFalse(fp.grid_dims_ok(5, 0))
        self.assertFalse(fp.grid_dims_ok(True, 5))

        accepted = [(1, 128, 48), (128, 1, 48), (2, 128, 64), (128, 2, 48)]
        for grid_h, grid_w, budget in accepted:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                result = fp.select_fixedpoint(
                    _tokens(group_count), _scores(group_count), grid_h, grid_w, budget
                )
                self.assertEqual(result.status, fp.STATUS_OK)
                self.assertEqual(result.output_count, budget)
                indices = [int(value) for value in result.group_index]
                self.assertEqual(indices, sorted(indices))
                self.assertEqual(len(indices), len(set(indices)))

        # one dimension strictly above 128 is always ERR_GRID, even when the
        # product would still be within the group limit.
        rejected = [(1, 129, 48), (129, 1, 48), (2, 129, 48), (129, 2, 48)]
        for grid_h, grid_w, budget in rejected:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                result = fp.select_fixedpoint(
                    _tokens(group_count), _scores(group_count), grid_h, grid_w, budget
                )
                self.assertEqual(result.status, fp.STATUS_ERR_GRID)
                self.assertEqual(result.status_name, "ERR_GRID")
                self.assertEqual(result.output_count, 0)

        # identity bypass still enforces the dimension bound and the budget
        # bound (K <= MAX_BUDGET = 96).
        identity = fp.select_fixedpoint(
            _tokens(96), _scores(96), 1, 96, 96
        )
        self.assertEqual(identity.status, fp.STATUS_OK)
        self.assertEqual(identity.output_count, 96)
        self.assertEqual(identity.context_count, 0)
        over_budget = fp.select_fixedpoint(
            _tokens(128), _scores(128), 1, 128, 128
        )
        self.assertEqual(over_budget.status, fp.STATUS_ERR_BUDGET)

    def test_extreme_1xN_2xN_grids(self) -> None:
        cases = [
            (1, 1, 48),
            (1, 2, 48),
            (1, 64, 48),
            (1, 128, 48),
            (2, 1, 48),
            (2, 64, 48),
            (2, 128, 64),
        ]
        for grid_h, grid_w, budget in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget):
                group_count = grid_h * grid_w
                result = fp.select_fixedpoint(
                    _tokens(group_count), _scores(group_count), grid_h, grid_w, budget
                )
                self.assertEqual(result.status, fp.STATUS_OK)
                expected = group_count if group_count <= budget else budget
                self.assertEqual(result.output_count, expected)
                self.assertEqual(result.real_count + result.context_count, expected)
                self.assertTrue(np.all(result.valid == 1))
        # the same extremes one column/row past the bound are rejected.
        for grid_h, grid_w in ((1, 129), (2, 129), (129, 1), (129, 2)):
            with self.subTest(grid=(grid_h, grid_w)):
                group_count = grid_h * grid_w
                result = fp.select_fixedpoint(
                    _tokens(group_count), _scores(group_count), grid_h, grid_w, 48
                )
                self.assertEqual(result.status, fp.STATUS_ERR_GRID)

    def test_local_cap_is_validated(self) -> None:
        grid_h, grid_w, budget = 9, 9, 48
        group_count = grid_h * grid_w
        for cap in (0, -1, fp.MAX_LOCAL_CANDIDATE_CAP + 1):
            result = fp.select_fixedpoint(
                _tokens(group_count), _scores(group_count), grid_h, grid_w, budget,
                local_cap=cap,
            )
            self.assertEqual(result.status, fp.STATUS_ERR_LOCAL_CAP)

    def test_group_index_guard(self) -> None:
        self.assertTrue(fp.group_limit_ok(1))
        self.assertTrue(fp.group_limit_ok(fp.MAX_GROUPS))
        self.assertTrue(fp.group_limit_ok(fp.MAX_GROUP_INDEX))
        self.assertFalse(fp.group_limit_ok(0))
        self.assertFalse(fp.group_limit_ok(fp.MAX_GROUP_INDEX + 1))

    def test_int64_centroid_and_pooling_bound_is_proven(self) -> None:
        """Machine-checked overflow proof for the frozen MAX_GRID_DIM = 128."""
        int64_max = (1 << 63) - 1

        def band_sizes(size: int) -> list[int]:
            return [(r + 1) * size // fp.REGION_GRID - r * size // fp.REGION_GRID
                    for r in range(fp.REGION_GRID)]

        worst_distance = 0
        worst_member = 0
        for grid_h in range(1, fp.MAX_GRID_DIM + 1):
            for grid_w in range(1, fp.MAX_GRID_DIM + 1):
                if grid_h * grid_w > fp.MAX_GROUPS:
                    continue
                for band_h in band_sizes(grid_h):
                    for band_w in band_sizes(grid_w):
                        members = band_h * band_w
                        if members == 0:
                            continue
                        worst_member = max(worst_member, members)
                        total = members * fp.SCORE_Q_MAX
                        delta_row = total * (band_h - 1)
                        delta_col = total * (band_w - 1)
                        worst_distance = max(
                            worst_distance, delta_row * delta_row + delta_col * delta_col
                        )
        # exact enumeration max plus the closed-form headroom statement.
        self.assertEqual(worst_distance, 447756141868573384)
        self.assertLess(worst_distance, int64_max)
        # analytic bound: (gh+2)(gw+2) <= 4420 -> n <= 491 -> delta <= 42*total.
        analytic_members = 491
        analytic_total = analytic_members * fp.SCORE_Q_MAX
        analytic_delta = analytic_total * (fp.MAX_GRID_DIM // fp.REGION_GRID)
        self.assertLess(2 * analytic_delta * analytic_delta, int64_max)
        # pooling accumulator stays far inside signed 64-bit as well.
        self.assertEqual(worst_member, 484)
        self.assertLess(worst_member * fp.SCORE_Q_MAX * 128, int64_max)

    def test_scale_is_pass_through_only(self) -> None:
        grid_h, grid_w, budget = 9, 9, 48
        group_count = grid_h * grid_w
        scores = _scores(group_count)
        first = fp.select_fixedpoint(
            _tokens(group_count), scores, grid_h, grid_w, budget, scale=0.5
        )
        second = fp.select_fixedpoint(
            _tokens(group_count), scores, grid_h, grid_w, budget, scale=8.0
        )
        self.assertEqual(first.scale_out, 0.5)
        self.assertEqual(second.scale_out, 8.0)
        self.assertTrue(np.array_equal(first.tokens_out, second.tokens_out))
        self.assertTrue(np.array_equal(first.group_index, second.group_index))

    # -- layout -------------------------------------------------------------
    def test_layout_offsets_sizes_and_alignment(self) -> None:
        expected = {
            48: (196928, [0, 196608, 196736, 196800, 196864], [196608, 96, 48, 48, 48]),
            64: (262464, [0, 262144, 262272, 262336, 262400], [262144, 128, 64, 64, 64]),
            96: (393792, [0, 393216, 393408, 393536, 393664], [393216, 192, 96, 96, 96]),
        }
        for budget, (total, offsets, sizes) in expected.items():
            with self.subTest(budget=budget):
                layout = fp.fpga_transfer_layout(budget)
                self.assertEqual(layout.total_bytes, total)
                self.assertEqual(list(layout.offsets().values()), offsets)
                self.assertEqual(list(layout.sizes().values()), sizes)
                for segment in layout.segments:
                    self.assertEqual(segment.offset % fp.FPGA_LAYOUT_ALIGNMENT, 0)
                    self.assertEqual(segment.offset + segment.size, segment.end)

    def test_layout_matches_float_selector(self) -> None:
        for budget in (16, 32, 48, 64, 96):
            fixed = fp.fpga_transfer_layout(budget)
            floating = float_fpga_transfer_layout(budget)
            self.assertEqual(fixed.total_bytes, floating.total_bytes)
            self.assertEqual(fixed.offsets(), floating.offsets())
            self.assertEqual(fixed.sizes(), floating.sizes())

    def test_round_half_away_from_zero(self) -> None:
        self.assertEqual(fp.round_half_away_from_zero(5, 2), 3)
        self.assertEqual(fp.round_half_away_from_zero(-5, 2), -3)
        self.assertEqual(fp.round_half_away_from_zero(7, 2), 4)
        self.assertEqual(fp.round_half_away_from_zero(-7, 2), -4)
        self.assertEqual(fp.round_half_away_from_zero(1, 3), 0)
        self.assertEqual(fp.round_half_away_from_zero(-1, 3), 0)
        self.assertEqual(fp.round_half_away_from_zero(2, 3), 1)
        self.assertEqual(fp.round_half_away_from_zero(-2, 3), -1)
        with self.assertRaises(fp.FixedpointContractError):
            fp.round_half_away_from_zero(1, 0)

    # -- consistency with the frozen float selector -------------------------
    def test_selection_structure_matches_float_selector(self) -> None:
        cases = [
            ((9, 9), 48, "rand"),
            ((9, 9), 64, "rand"),
            ((10, 12), 48, "rand"),
            ((10, 12), 96, "rand"),
            ((2, 25), 48, "rand"),
            ((5, 4), 16, "rand"),
            ((2, 9), 16, "rand"),
            ((9, 9), 48, "same"),
            ((10, 12), 64, "zero"),
            ((5, 4), 16, "zero"),
        ]
        for (grid_h, grid_w), budget, mode in cases:
            with self.subTest(grid=(grid_h, grid_w), budget=budget, mode=mode):
                group_count = grid_h * grid_w
                tokens = _tokens(group_count)
                scores = _scores(group_count, mode)
                fixed = fp.select_fixedpoint(tokens, scores, grid_h, grid_w, budget)
                floating = select_hardware_regular_groups(
                    tokens.astype(np.float64),
                    grid_h,
                    grid_w,
                    budget=budget,
                    combined_scores=scores.astype(np.float64) / float(fp.SCORE_Q_SCALE),
                )
                self.assertEqual(
                    list(map(int, fixed.group_index)),
                    [int(value) for value in floating.group_indices],
                )
                self.assertEqual(
                    list(map(int, fixed.source_type)),
                    [int(value) for value in floating.source_type],
                )
                self.assertEqual(
                    list(map(int, fixed.region_id)),
                    [int(value) for value in floating.region_id],
                )
                self.assertEqual(tuple(fixed.anchors), tuple(int(v) for v in floating.anchors))
                self.assertEqual(tuple(fixed.details), tuple(int(v) for v in floating.details))

    # -- random differential against the pre-refactor algorithm -------------
    def test_random_differential_matches_legacy_reference(self) -> None:
        """At least 100 deterministic random cases, item-by-item vs legacy."""
        generator = _load_fixture_generator()
        cases = generator.build_random_cases(120)
        self.assertGreaterEqual(len(cases), 100)
        for case in cases:
            with self.subTest(case=case.name):
                group_count = case.group_count
                tokens = generator.make_tokens(group_count)
                scores = generator.make_scores(group_count, case.score_mode, case.seed)
                golden = fp.select_fixedpoint(
                    tokens,
                    scores,
                    case.grid_h,
                    case.grid_w,
                    case.budget,
                    local_cap=case.local_cap,
                    scale=case.scale,
                )
                legacy = _legacy_select(
                    tokens,
                    scores,
                    case.grid_h,
                    case.grid_w,
                    case.budget,
                    local_cap=case.local_cap,
                    scale=case.scale,
                )
                self.assertEqual(golden.status, legacy["status"])
                self.assertEqual(golden.output_count, legacy["output_count"])
                self.assertEqual(golden.real_count, legacy["real_count"])
                self.assertEqual(golden.context_count, legacy["context_count"])
                self.assertEqual(golden.scale_out, legacy["scale_out"])
                self.assertTrue(
                    np.array_equal(golden.tokens_out, legacy["tokens_out"]),
                    f"{case.name}: tokens_out differs",
                )
                self.assertEqual(
                    [int(value) for value in golden.group_index], legacy["group_index"]
                )
                self.assertEqual(
                    [int(value) for value in golden.source_type], legacy["source_type"]
                )
                self.assertEqual(
                    [int(value) for value in golden.region_id], legacy["region_id"]
                )
                self.assertEqual(
                    [int(value) for value in golden.valid], legacy["valid"]
                )

    # -- v4 timing redesign: >= 200 diffs against the frozen v3 insertion -----
    def test_v4_multipass_matches_v3_insertion_200plus(self) -> None:
        """>= 200 random cases: frozen v3 bounded insertion == golden == v4.

        ``_legacy_select(..., candidate_mode="v3_insertion")`` is the literal
        frozen v3 kernel; the golden is the independent collect-all reference and
        the C++/ASan testbench diffs the new multi-pass packed kernel directly
        against the retained v3 int8 kernel on the same class of inputs.  These
        cases force many score ties so the (score, lower-index) rules are hit.
        """
        generator = _load_fixture_generator()
        cases = generator.build_random_cases(220)
        self.assertGreaterEqual(len(cases), 200)
        for case in cases:
            with self.subTest(case=case.name):
                group_count = case.group_count
                tokens = generator.make_tokens(group_count)
                scores = generator.make_scores(group_count, case.score_mode, case.seed)
                golden = fp.select_fixedpoint(
                    tokens,
                    scores,
                    case.grid_h,
                    case.grid_w,
                    case.budget,
                    local_cap=case.local_cap,
                    scale=case.scale,
                )
                v3 = _legacy_select(
                    tokens,
                    scores,
                    case.grid_h,
                    case.grid_w,
                    case.budget,
                    local_cap=case.local_cap,
                    scale=case.scale,
                    candidate_mode="v3_insertion",
                )
                self.assertEqual(golden.status, v3["status"])
                self.assertEqual(golden.output_count, v3["output_count"])
                self.assertEqual(golden.real_count, v3["real_count"])
                self.assertEqual(golden.context_count, v3["context_count"])
                self.assertEqual(golden.scale_out, v3["scale_out"])
                self.assertTrue(
                    np.array_equal(golden.tokens_out, v3["tokens_out"]),
                    f"{case.name}: tokens_out differs",
                )
                self.assertEqual(
                    [int(value) for value in golden.group_index], v3["group_index"]
                )
                self.assertEqual(
                    [int(value) for value in golden.source_type], v3["source_type"]
                )
                self.assertEqual(
                    [int(value) for value in golden.region_id], v3["region_id"]
                )
                self.assertEqual([int(value) for value in golden.valid], v3["valid"])

    # -- environment --------------------------------------------------------
    def test_module_does_not_import_torch(self) -> None:
        code = (
            "import sys; import evaluation.hardware_regular_selector_fixedpoint; "
            "assert 'torch' not in sys.modules, 'torch was imported'"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
