"""Focused CPU tests for the frozen-weight scorer-reduction ablation.

These tests never train, never load a model, never use CUDA and never touch the
network.  They cover the pre-registered candidate set, the reducer math, the
token-weight concentration, the Holm-Bonferroni correction, the promotion gate
and winner selection, the analytic hardware contract, the pre-registration hash
and the val/test isolation guarantee (changing test prompts/files must not move
the val metrics, the winner or the pre-registration hash).
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Mapping
from unittest import mock

import numpy as np

from evaluation.frozen_scorer_reduction_ablation import (
    ALPHA,
    BASELINE_CANDIDATE,
    BOOTSTRAP_RESAMPLES,
    CANDIDATE_CSV_FIELDS,
    CANDIDATE_ORDER,
    CANDIDATES,
    CANDIDATE_SPEC,
    COLLAPSE_EXACT_INVARIANCE_MAX,
    DERANGEMENT_SPLIT_OFFSETS,
    EFFECT_SIZE_FLOOR,
    HOLM_FAMILY_SIZE,
    NON_BASELINE_CANDIDATES,
    PREREGISTRATION_SEED,
    PRIMARY_METRICS,
    REPORT_STEM,
    ROLE_CSV_FIELDS,
    ROLE_CORRECT,
    ROLE_CROSS,
    ROLE_SAME,
    _candidate_rows,
    _role_rows,
    apply_holm_to_val,
    build_preregistration,
    evaluate_gates,
    evaluate_split,
    hardware_contract,
    hardware_contract_for,
    holm_bonferroni,
    preregistration_sha256,
    prompt_cosine_matrix,
    reduce_cosine_matrix,
    render_markdown,
    run_ablation,
    score_candidate,
    select_winner,
    token_weight_matrix,
)
from evaluation.prompt_projection_distillation import (
    FEATURE_DIM,
    PROMPT_DIM,
    SCORE_DIM,
    load_feature_cache,
    save_feature_cache,
    save_projection_npz,
    split_sample_ids,
    teacher_target,
)
from evaluation.prompt_identifiability_diagnostic import BOOTSTRAP_SEED


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=-1, keepdims=True)


class CandidateSetFrozenTests(unittest.TestCase):
    def test_names_and_order_are_the_pre_registered_list(self) -> None:
        self.assertEqual(
            CANDIDATE_ORDER,
            (
                "hard_max",
                "mean_cos",
                "topk_mean_k2",
                "topk_mean_k4",
                "norm_lse_tau003",
                "norm_lse_tau010",
                "norm_lse_tau030",
                "softmax_mean_tau003",
                "softmax_mean_tau010",
                "softmax_mean_tau030",
                "prompt_mean_vector",
            ),
        )
        self.assertEqual(BASELINE_CANDIDATE, "hard_max")
        self.assertEqual(len(NON_BASELINE_CANDIDATES), 10)
        self.assertEqual(len(CANDIDATES), 11)

    def test_temperatures_and_topk_are_frozen(self) -> None:
        self.assertEqual(CANDIDATE_SPEC["topk_mean_k2"]["params"], {"k": 2})
        self.assertEqual(CANDIDATE_SPEC["topk_mean_k4"]["params"], {"k": 4})
        for name, tau in (
            ("norm_lse_tau003", 0.03),
            ("norm_lse_tau010", 0.10),
            ("norm_lse_tau030", 0.30),
            ("softmax_mean_tau003", 0.03),
            ("softmax_mean_tau010", 0.10),
            ("softmax_mean_tau030", 0.30),
        ):
            self.assertEqual(CANDIDATE_SPEC[name]["params"], {"tau": tau})

    def test_token_competition_flags(self) -> None:
        self.assertFalse(CANDIDATE_SPEC["mean_cos"]["token_competition"])
        self.assertFalse(CANDIDATE_SPEC["prompt_mean_vector"]["token_competition"])
        for name in CANDIDATE_ORDER:
            if name in {"mean_cos", "prompt_mean_vector"}:
                continue
            self.assertTrue(CANDIDATE_SPEC[name]["token_competition"], name)

    def test_statistics_constants_and_holm_family(self) -> None:
        self.assertEqual(PREREGISTRATION_SEED, 20261002)
        self.assertEqual(BOOTSTRAP_RESAMPLES, 2000)
        self.assertEqual(EFFECT_SIZE_FLOOR, 0.2)
        self.assertEqual(COLLAPSE_EXACT_INVARIANCE_MAX, 0.05)
        self.assertEqual(HOLM_FAMILY_SIZE, 10 * 2 * 2)
        self.assertEqual(DERANGEMENT_SPLIT_OFFSETS, {"train": 0, "val": 101, "test": 202})


class ReducerMathTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.cosines = rng.uniform(-1.0, 1.0, size=(7, 6))

    def test_hard_max_and_mean(self) -> None:
        np.testing.assert_allclose(
            reduce_cosine_matrix("hard_max", self.cosines), self.cosines.max(axis=1)
        )
        np.testing.assert_allclose(
            reduce_cosine_matrix("mean_cos", self.cosines), self.cosines.mean(axis=1)
        )

    def test_topk_mean_uses_top_k_and_falls_back_to_all(self) -> None:
        order = np.argsort(-self.cosines, axis=1)[:, :2]
        rows = np.arange(self.cosines.shape[0])[:, None]
        expected = self.cosines[rows, order].mean(axis=1)
        np.testing.assert_allclose(reduce_cosine_matrix("topk_mean_k2", self.cosines), expected)
        single_token = self.cosines[:, :1]
        np.testing.assert_allclose(
            reduce_cosine_matrix("topk_mean_k4", single_token),
            single_token.mean(axis=1),
        )
        three_tokens = self.cosines[:, :3]
        np.testing.assert_allclose(
            reduce_cosine_matrix("topk_mean_k4", three_tokens),
            three_tokens.mean(axis=1),
        )

    def test_norm_lse_matches_the_closed_form(self) -> None:
        tau = 0.1
        scaled = self.cosines / tau
        expected = tau * (
            np.log(np.exp(scaled).sum(axis=1)) - np.log(self.cosines.shape[1])
        )
        np.testing.assert_allclose(
            reduce_cosine_matrix("norm_lse_tau010", self.cosines), expected, rtol=1e-10
        )

    def test_small_tau_limits_towards_max(self) -> None:
        lse = reduce_cosine_matrix("norm_lse_tau003", self.cosines)
        hard = reduce_cosine_matrix("hard_max", self.cosines)
        self.assertTrue(np.all(hard - lse >= -1e-9))
        # exact bound: max - norm_lse <= tau * log(T)
        bound = 0.03 * np.log(self.cosines.shape[1]) + 1e-9
        self.assertTrue(np.all(hard - lse <= bound))

    def test_softmax_mean_limits_towards_max(self) -> None:
        soft = reduce_cosine_matrix("softmax_mean_tau003", self.cosines)
        hard = reduce_cosine_matrix("hard_max", self.cosines)
        self.assertLess(float(np.max(np.abs(soft - hard))), 0.05)

    def test_token_weights(self) -> None:
        hard = token_weight_matrix("hard_max", self.cosines)
        self.assertTrue(np.allclose(hard.sum(axis=1), 1.0))
        self.assertTrue(set(np.unique(hard)) <= {0.0, 1.0})
        top2 = token_weight_matrix("topk_mean_k2", self.cosines)
        self.assertTrue(np.allclose(top2.sum(axis=1), 1.0))
        self.assertTrue(set(np.unique(top2)) <= {0.0, 0.5})
        for name in ("norm_lse_tau010", "softmax_mean_tau010"):
            weights = token_weight_matrix(name, self.cosines)
            self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertIsNone(token_weight_matrix("mean_cos", self.cosines))
        self.assertIsNone(token_weight_matrix("prompt_mean_vector", self.cosines))

    def test_prompt_mean_vector_is_normalised_mean_and_permutation_invariant(self) -> None:
        rng = np.random.default_rng(9)
        group_mean = rng.normal(size=(4, FEATURE_DIM)).astype(np.float32)
        prompt = rng.normal(size=(5, PROMPT_DIM)).astype(np.float32)
        Wt = rng.normal(size=(SCORE_DIM, PROMPT_DIM)).astype(np.float32) * 0.05
        Wv = rng.normal(size=(SCORE_DIM, FEATURE_DIM)).astype(np.float32) * 0.05
        _, visual_unit, prompt_unit = prompt_cosine_matrix(group_mean, prompt, Wt, Wv)
        query = _unit(prompt_unit.mean(axis=0))
        expected = visual_unit @ query
        np.testing.assert_allclose(
            score_candidate("prompt_mean_vector", group_mean, prompt, Wt, Wv),
            expected.astype(np.float32),
            rtol=1e-5,
        )
        permuted = prompt[[2, 0, 4, 1, 3]]
        np.testing.assert_allclose(
            score_candidate("prompt_mean_vector", group_mean, permuted, Wt, Wv),
            score_candidate("prompt_mean_vector", group_mean, prompt, Wt, Wv),
            rtol=1e-5,
        )


def _collapse_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A rank-structured case where hard_max collapses across two prompts."""
    Wv = np.zeros((SCORE_DIM, FEATURE_DIM), dtype=np.float32)
    Wv[0, 0] = 1.0
    Wt = np.zeros((SCORE_DIM, PROMPT_DIM), dtype=np.float32)
    Wt[0, 0] = 1.0
    Wt[1, 1] = 1.0
    Wt[2, 2] = 1.0
    group_mean = np.zeros((5, FEATURE_DIM), dtype=np.float32)
    group_mean[:, 0] = 1.0
    prompt_a = np.zeros((2, PROMPT_DIM), dtype=np.float32)
    prompt_a[0, 0] = 1.0
    prompt_a[1, 1] = 0.5
    prompt_b = np.zeros((3, PROMPT_DIM), dtype=np.float32)
    prompt_b[0, 0] = 1.0
    prompt_b[1, 1] = 0.5
    prompt_b[2, 2] = 0.5
    return group_mean, prompt_a, prompt_b, (Wv, Wt)


class CollapseMechanismTests(unittest.TestCase):
    def test_hard_max_collapses_on_shared_dominant_token(self) -> None:
        group_mean, prompt_a, prompt_b, (Wv, Wt) = _collapse_case()
        a = score_candidate("hard_max", group_mean, prompt_a, Wt, Wv)
        b = score_candidate("hard_max", group_mean, prompt_b, Wt, Wv)
        self.assertTrue(np.array_equal(a, b))

    def test_prompt_mean_vector_does_not_collapse_here(self) -> None:
        group_mean, prompt_a, prompt_b, (Wv, Wt) = _collapse_case()
        a = score_candidate("prompt_mean_vector", group_mean, prompt_a, Wt, Wv)
        b = score_candidate("prompt_mean_vector", group_mean, prompt_b, Wt, Wv)
        self.assertFalse(np.array_equal(a, b))


class HolmBonferroniTests(unittest.TestCase):
    def test_step_down_matches_a_worked_example(self) -> None:
        adjusted = holm_bonferroni(
            {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        )
        self.assertAlmostEqual(adjusted["a"], 0.04)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["d"], 0.06)

    def test_adjusted_never_below_raw_and_is_monotone(self) -> None:
        rng = np.random.default_rng(2)
        raw = {f"k{i}": float(value) for i, value in enumerate(rng.uniform(0, 1, 20))}
        adjusted = holm_bonferroni(raw)
        for key, value in raw.items():
            self.assertGreaterEqual(adjusted[key] + 1e-12, value)
        ordered = sorted(raw.items(), key=lambda item: item[1])
        values = [adjusted[key] for key, _ in ordered]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted.values()))


class HardwareContractTests(unittest.TestCase):
    def test_prompt_mean_vector_is_one_by_128_int8_query(self) -> None:
        contract = hardware_contract_for("prompt_mean_vector", group_count=250, token_count=43)
        self.assertEqual(contract["host_prompt_query_shape"], [1, SCORE_DIM])
        self.assertEqual(contract["host_prompt_query_tokens"], 1)
        self.assertEqual(contract["prompt_query_transfer_bytes_int8_per_sample"], 128)
        self.assertEqual(contract["prompt_query_transfer_bytes_fp32_per_sample"], 512)
        self.assertEqual(contract["fpga_scorer_dot_mac_per_group"], SCORE_DIM)
        self.assertEqual(contract["fpga_scorer_compare_per_group"], 0)
        self.assertEqual(contract["fpga_scorer_exp_per_group"], 0)
        self.assertEqual(contract["fpga_scorer_reduction_add_per_group"], 0)
        self.assertEqual(contract["fpga_scorer_total_ops_per_image"], 250 * SCORE_DIM)

    def test_matrix_reducer_ships_the_full_query(self) -> None:
        contract = hardware_contract_for("hard_max", group_count=250, token_count=43)
        self.assertEqual(contract["host_prompt_query_shape"], [43, SCORE_DIM])
        self.assertEqual(contract["prompt_query_transfer_bytes_int8_per_sample"], 43 * 128)
        self.assertEqual(contract["fpga_scorer_dot_mac_per_group"], 43 * 128)
        self.assertEqual(contract["fpga_scorer_compare_per_group"], 42)
        self.assertEqual(contract["fpga_scorer_exp_per_group"], 0)

    def test_every_reducer_contract_is_internally_consistent(self) -> None:
        contract = hardware_contract(group_count=250, token_count=43)
        self.assertIn("NOT synthesised", contract["scope_note"])
        self.assertIn("ZCU102", contract["scope_note"])
        for name in CANDIDATE_ORDER:
            entry = contract["per_reducer"][name]
            self.assertEqual(
                entry["fpga_scorer_total_ops_per_group"],
                entry["fpga_scorer_dot_mac_per_group"]
                + entry["fpga_scorer_compare_per_group"]
                + entry["fpga_scorer_exp_per_group"]
                + entry["fpga_scorer_reduction_add_per_group"],
            )
            self.assertEqual(
                entry["prompt_query_transfer_bytes_int8_per_sample"],
                entry["host_prompt_query_tokens"] * SCORE_DIM,
            )
            self.assertGreaterEqual(entry["fpga_scorer_total_ops_per_group"], 0)

    def test_prompt_mean_vector_is_cheapest_under_the_cost_key(self) -> None:
        contract = hardware_contract(group_count=250, token_count=43)
        costs = {
            name: (
                contract["per_reducer"][name]["fpga_scorer_total_ops_per_image"],
                contract["per_reducer"][name]["prompt_query_transfer_bytes_int8_per_sample"],
            )
            for name in CANDIDATE_ORDER
        }
        cheapest = min(costs, key=lambda name: costs[name])
        self.assertEqual(cheapest, "prompt_mean_vector")


def _fake_val(
    *,
    min_delta: Mapping[str, float] | None = None,
    exact_invariance: Mapping[str, float] | None = None,
    correct: Mapping[str, Mapping[str, float]] | None = None,
    fail_gate: Mapping[str, set[str]] | None = None,
) -> dict:
    min_delta = dict(min_delta or {})
    exact_invariance = dict(exact_invariance or {})
    correct = dict(correct or {})
    fail_gate = dict(fail_gate or {})
    val: dict = {
        "split": "val",
        "count": 76,
        "independent_image_units": 38,
        "roles": {ROLE_CORRECT: {}},
        "deltas": {},
        "invariance": {},
    }
    for name in CANDIDATE_ORDER:
        base_correct = {"spearman": 0.05, "ndcg_full": 0.5}
        base_correct.update(correct.get(name, {}))
        val["roles"][ROLE_CORRECT][name] = base_correct
        default_exact = 1.0 if name == BASELINE_CANDIDATE else 0.0
        val["invariance"][name] = {
            "exact_prompt_invariance_fraction": float(
                exact_invariance.get(name, default_exact)
            )
        }
        failed = fail_gate.get(name, set())
        val["deltas"][name] = {}
        for role in (ROLE_SAME, ROLE_CROSS):
            val["deltas"][name][role] = {}
            for metric in PRIMARY_METRICS:
                entry = {
                    "mean": float(min_delta.get(name, 0.02)),
                    "ci_low": 0.01,
                    "cohens_dz": 0.5,
                    "p_value_two_sided": 0.001,
                    "holm_adjusted_p": 0.01,
                }
                if f"{role}:{metric}" in failed:
                    entry["mean"] = -0.02
                    entry["ci_low"] = -0.05
                val["deltas"][name][role][metric] = entry
    return val


class GateAndWinnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hardware = hardware_contract(group_count=250, token_count=43)

    def test_baseline_fails_the_collapse_gate_and_cannot_win(self) -> None:
        val = _fake_val()
        gate = evaluate_gates(val, hardware=self.hardware)
        self.assertFalse(gate["candidates"][BASELINE_CANDIDATE]["C_collapse_removed"]["pass"])
        self.assertNotIn(BASELINE_CANDIDATE, gate["passing_candidates"])
        winner = select_winner(val, gate, hardware=self.hardware)
        self.assertNotEqual(winner["winner"], BASELINE_CANDIDATE)

    def test_no_candidate_passing_is_no_promotion(self) -> None:
        val = _fake_val(exact_invariance={name: 1.0 for name in CANDIDATE_ORDER})
        gate = evaluate_gates(val, hardware=self.hardware)
        self.assertEqual(gate["passing_candidates"], [])
        winner = select_winner(val, gate, hardware=self.hardware)
        self.assertEqual(winner["winner"], "NO_PROMOTION")

    def test_gate_a_requires_all_four_conditions(self) -> None:
        val = _fake_val(fail_gate={"norm_lse_tau030": {f"{ROLE_SAME}:spearman"}})
        gate = evaluate_gates(val, hardware=self.hardware)
        entry = gate["candidates"]["norm_lse_tau030"]
        self.assertFalse(entry["A_prompt_sensitivity"]["pass"])
        self.assertTrue(entry["B_shuffled_discrimination"]["pass"])
        self.assertFalse(entry["passed_all"])

    def test_gate_b_requires_cross_image_discrimination(self) -> None:
        val = _fake_val(fail_gate={"topk_mean_k4": {f"{ROLE_CROSS}:ndcg_full"}})
        gate = evaluate_gates(val, hardware=self.hardware)
        self.assertFalse(gate["candidates"]["topk_mean_k4"]["passed_all"])

    def test_gate_d_rejects_teacher_alignment_regression(self) -> None:
        val = _fake_val(
            correct={"softmax_mean_tau030": {"spearman": 0.0, "ndcg_full": 0.45}}
        )
        gate = evaluate_gates(val, hardware=self.hardware)
        self.assertFalse(gate["candidates"]["softmax_mean_tau030"]["D_teacher_non_regression"]["pass"])

    def test_min_primary_delta_dominates_the_lexicographic_order(self) -> None:
        val = _fake_val(min_delta={"softmax_mean_tau030": 0.5})
        gate = evaluate_gates(val, hardware=self.hardware)
        winner = select_winner(val, gate, hardware=self.hardware)
        self.assertEqual(winner["winner"], "softmax_mean_tau030")

    def test_hardware_cost_breaks_a_full_tie(self) -> None:
        # Without a delta difference the cheapest contract must win.
        val = _fake_val()
        gate = evaluate_gates(val, hardware=self.hardware)
        winner = select_winner(val, gate, hardware=self.hardware)
        self.assertEqual(winner["winner"], "prompt_mean_vector")

    def test_holm_is_attached_only_to_non_baseline_candidates(self) -> None:
        val = _fake_val()
        apply_holm_to_val(val)
        for name in NON_BASELINE_CANDIDATES:
            for role in (ROLE_SAME, ROLE_CROSS):
                for metric in PRIMARY_METRICS:
                    entry = val["deltas"][name][role][metric]
                    self.assertIn("holm_adjusted_p", entry)
                    self.assertEqual(entry["holm_family_size"], HOLM_FAMILY_SIZE)


def _seed_all(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_synthetic_case(root: Path, image_count: int = 80) -> dict[str, Path]:
    """Tiny but schema-exact feature cache + projection + config."""
    rng = _seed_all(13)
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    index_records = []
    for image_index in range(image_count):
        image_hash = hashlib.sha256(f"ablation-image-{image_index}".encode("utf-8")).hexdigest()
        group_count = 4
        merged = rng.normal(size=(group_count, PROMPT_DIM)).astype(np.float32)
        group_mean = rng.normal(size=(group_count, FEATURE_DIM)).astype(np.float32)
        for task in ("grounding", "pointing"):
            prompt = rng.normal(size=(3, PROMPT_DIM)).astype(np.float32)
            target = teacher_target(merged, prompt, 0.1)
            sample_id = f"abl-{image_index:04d}-{task}"
            npz_name = f"{sample_id}.npz"
            save_feature_cache(
                features_dir / npz_name,
                group_mean=group_mean,
                prompt_embeddings=prompt,
                teacher_raw=target["raw"],
                teacher_normalized=target["normalized"],
                teacher_soft=target["soft"],
                metadata={
                    "sample_id": sample_id,
                    "task": task,
                    "grid_h": 2,
                    "grid_w": 2,
                    "temperature": 0.1,
                    "source_image_sha256": image_hash,
                },
            )
            index_records.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "npz": npz_name,
                    "grid_h": 2,
                    "grid_w": 2,
                    "source_image_sha256": image_hash,
                }
            )
    index_path = features_dir / "index.jsonl"
    index_path.write_text(
        "\n".join(json.dumps(record) for record in index_records) + "\n",
        encoding="utf-8",
    )
    projection_path = root / "projection.npz"
    save_projection_npz(
        projection_path,
        rng.normal(size=(SCORE_DIM, PROMPT_DIM)).astype(np.float32) * 0.02,
        rng.normal(size=(SCORE_DIM, FEATURE_DIM)).astype(np.float32) * 0.02,
    )
    config_path = root / "config.json"
    repo_config = Path(__file__).resolve().parents[2] / "configs" / "prompt_projection_distillation.json"
    config_path.write_text(repo_config.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "config": config_path,
        "index": index_path,
        "features_dir": features_dir,
        "projection": projection_path,
    }


class PreregistrationTests(unittest.TestCase):
    def _build(self, tmp: Path) -> dict:
        paths = build_synthetic_case(tmp, image_count=30)
        return build_preregistration(
            config_path=paths["config"],
            index_path=paths["index"],
            projection_path=paths["projection"],
            split_summary={"val": {"sample_count": 6, "image_count": 3}},
            seed=PREREGISTRATION_SEED,
        )

    def test_hash_matches_canonical_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
        self.assertEqual(payload["preregistration_sha256"], preregistration_sha256(payload))
        self.assertEqual(len(payload["preregistration_sha256"]), 64)

    def test_candidate_set_is_frozen_in_the_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
        self.assertEqual(payload["candidate_order"], list(CANDIDATE_ORDER))
        self.assertEqual(payload["seed"], 20261002)
        self.assertEqual(
            [candidate["name"] for candidate in payload["candidates"]], list(CANDIDATE_ORDER)
        )
        self.assertTrue(payload["test_discipline"]["val_decides"])
        self.assertIn("no_post_hoc_selection_on_test", payload["test_discipline"])

    def test_payload_contains_no_test_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = self._build(Path(tmp))
        blob = json.dumps(payload).lower()
        for banned in ("test_correct", "test_spearman", "test_ndcg"):
            self.assertNotIn(banned, blob)

    def test_hash_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._build(root)
            second = self._build(root)
        self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])


class EndToEndSyntheticTests(unittest.TestCase):
    def test_full_run_writes_artifacts_in_the_right_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_synthetic_case(root, image_count=80)
            out_dir = root / "reports"
            bundle = run_ablation(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                split_names=("train", "val", "test"),
                out_dir=out_dir,
                write_artifacts=True,
            )
            prereg = out_dir / f"{REPORT_STEM}-preregistration.json"
            val_gate = out_dir / f"{REPORT_STEM}-val-gate.json"
            self.assertTrue(prereg.is_file())
            self.assertTrue(val_gate.is_file())
            on_disk = json.loads(prereg.read_text(encoding="utf-8"))
            self.assertEqual(
                on_disk["preregistration_sha256"], bundle["preregistration_sha256"]
            )
            self.assertTrue(bundle["test_once"]["evaluated"])
            self.assertTrue(bundle["test_once"]["evaluated_after_winner_written"])
            self.assertFalse(bundle["test_once"]["used_for_selection"])
        self.assertEqual(set(bundle["splits"]), {"train", "val", "test"})
        self.assertEqual(bundle["splits"]["val"]["count"] + bundle["splits"]["train"]["count"] + bundle["splits"]["test"]["count"], 160)
        for name in ("train", "val", "test"):
            self.assertGreater(bundle["splits"][name]["independent_image_units"], 1)
        for candidate in CANDIDATE_ORDER:
            for role in (ROLE_CORRECT, ROLE_SAME, ROLE_CROSS):
                entry = bundle["splits"]["val"]["roles"][role][candidate]
                self.assertTrue(np.isfinite(entry["spearman"]))
                self.assertTrue(np.isfinite(entry["ndcg_full"]))

    def test_val_only_does_not_evaluate_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_synthetic_case(root, image_count=80)
            bundle = run_ablation(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                split_names=("val",),
                write_artifacts=False,
            )
        self.assertEqual(bundle["splits_evaluated"], ["val"])
        self.assertFalse(bundle["test_once"]["evaluated"])
        self.assertIsNotNone(bundle["val_gate"])
        self.assertIsNotNone(bundle["winner"])

    def test_markdown_and_csv_projections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_synthetic_case(root, image_count=80)
            bundle = run_ablation(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                split_names=("val", "test"),
                write_artifacts=False,
            )
        text = render_markdown(bundle)
        self.assertIn("prompt_mean_vector", text)
        self.assertIn("NO_PROMOTION", text)
        candidate_rows = _candidate_rows(bundle)
        self.assertTrue(candidate_rows)
        for row in candidate_rows:
            for field in CANDIDATE_CSV_FIELDS:
                self.assertIn(field, row)
        role_rows = _role_rows(bundle)
        self.assertTrue(role_rows)
        for row in role_rows:
            for field in ROLE_CSV_FIELDS:
                self.assertIn(field, row)


class ValTestIsolationTests(unittest.TestCase):
    def test_changing_test_npz_does_not_move_val_winner_or_prereg_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_synthetic_case(root, image_count=60)
            records = [
                json.loads(line)
                for line in paths["index"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            repo_config = json.loads(paths["config"].read_text(encoding="utf-8"))
            split = split_sample_ids(
                records,
                seed=int(repo_config["split"]["seed"]),
                ratios=repo_config["split"]["ratios"],
                salt=str(repo_config["split"]["salt"]),
            )
            test_ids = set(split["test"])
            self.assertTrue(test_ids)

            first = run_ablation(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                split_names=("val",),
                write_artifacts=False,
            )
            # Rewrite only the test-split caches with different prompts/targets.
            rng = _seed_all(777)
            for record in records:
                if str(record["sample_id"]) not in test_ids:
                    continue
                group_mean = rng.normal(size=(4, FEATURE_DIM)).astype(np.float32)
                prompt = rng.normal(size=(3, PROMPT_DIM)).astype(np.float32)
                target = teacher_target(rng.normal(size=(4, PROMPT_DIM)), prompt, 0.1)
                save_feature_cache(
                    paths["features_dir"] / record["npz"],
                    group_mean=group_mean,
                    prompt_embeddings=prompt,
                    teacher_raw=target["raw"],
                    teacher_normalized=target["normalized"],
                    teacher_soft=target["soft"],
                    metadata={
                        "sample_id": record["sample_id"],
                        "task": record["task"],
                        "grid_h": 2,
                        "grid_w": 2,
                        "temperature": 0.1,
                        "source_image_sha256": record["source_image_sha256"],
                    },
                )
            second = run_ablation(
                config_path=paths["config"],
                index_path=paths["index"],
                features_dir=paths["features_dir"],
                projection_path=paths["projection"],
                split_names=("val",),
                write_artifacts=False,
            )
        self.assertEqual(first["preregistration_sha256"], second["preregistration_sha256"])
        self.assertEqual(first["winner"]["winner"], second["winner"]["winner"])
        for candidate in CANDIDATE_ORDER:
            for role in (ROLE_CORRECT, ROLE_SAME, ROLE_CROSS):
                for metric in ("spearman", "ndcg_full"):
                    self.assertEqual(
                        first["splits"]["val"]["roles"][role][candidate][metric],
                        second["splits"]["val"]["roles"][role][candidate][metric],
                    )

    def test_val_phase_never_opens_a_test_npz(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = build_synthetic_case(root, image_count=60)
            records = [
                json.loads(line)
                for line in paths["index"].read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            repo_config = json.loads(paths["config"].read_text(encoding="utf-8"))
            split = split_sample_ids(
                records,
                seed=int(repo_config["split"]["seed"]),
                ratios=repo_config["split"]["ratios"],
                salt=str(repo_config["split"]["salt"]),
            )
            test_ids = set(split["test"])
            test_npz = {
                str(paths["features_dir"] / record["npz"])
                for record in records
                if str(record["sample_id"]) in test_ids
            }
            real_loader = load_feature_cache
            opened: list[str] = []

            def recording_loader(path, validate: bool = True):
                opened.append(str(path))
                return real_loader(path, validate=validate)

            import evaluation.frozen_scorer_reduction_ablation as module

            with mock.patch.object(module, "load_feature_cache", recording_loader):
                run_ablation(
                    config_path=paths["config"],
                    index_path=paths["index"],
                    features_dir=paths["features_dir"],
                    projection_path=paths["projection"],
                    split_names=("val",),
                    write_artifacts=False,
                )
            self.assertTrue(opened)
            self.assertEqual(test_npz & set(opened), set())


if __name__ == "__main__":
    unittest.main()
