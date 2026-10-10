"""CPU-only tests for the B1 answer-margin autograd preflight.

These tests use light fake tensors and a fake ``lm_head`` only.  They never load
model weights, never require CUDA, and never train anything.  They cover the
gather order, forward shapes, margin numerics, finite/non-zero gradients,
fail-closed input validation, and an explicit negative control proving that a
detached compact tensor leaves the original B1 hidden states without gradient.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from evaluation.answer_margin_b1_preflight import (
    PREFLIGHT_STATUS_PASS,
    AnswerMarginContractError,
    PreflightSettings,
    choice_margin,
    choice_margin_from_positions,
    differentiable_group_gather,
    main,
    render_markdown,
    run_preflight,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _arange_hidden(patch_tokens: int, hidden_dim: int) -> torch.Tensor:
    return torch.arange(
        patch_tokens * hidden_dim, dtype=torch.float32
    ).reshape(patch_tokens, hidden_dim)


class DifferentiableGroupGatherTest(unittest.TestCase):
    def test_gather_shape_and_row_order(self) -> None:
        hidden = _arange_hidden(12, 3)  # G = 3 groups of 4
        out = differentiable_group_gather(hidden, [2, 0], merge_unit=4)
        self.assertEqual(out.shape, (8, 3))
        expected_patches = [8, 9, 10, 11, 0, 1, 2, 3]
        self.assertTrue(torch.equal(out, hidden.index_select(0, torch.tensor(expected_patches))))
        # Row k*merge_unit + j really is group_k's j-th patch.
        for k, group in enumerate([2, 0]):
            for j in range(4):
                self.assertTrue(
                    torch.equal(out[k * 4 + j], hidden[group * 4 + j])
                )

    def test_gather_accepts_tensor_and_supports_merge_unit_one(self) -> None:
        hidden = _arange_hidden(5, 2)
        indices = torch.tensor([4, 1, 3], dtype=torch.int64)
        out = differentiable_group_gather(hidden, indices, merge_unit=1)
        self.assertTrue(torch.equal(out, hidden.index_select(0, indices)))

    def test_gather_stays_differentiable_and_scatters_grad(self) -> None:
        hidden = _arange_hidden(12, 3).requires_grad_(True)
        out = differentiable_group_gather(hidden, [2, 0], merge_unit=4)
        self.assertTrue(out.requires_grad)
        out.sum().backward()
        self.assertIsNotNone(hidden.grad)
        expected = torch.zeros_like(hidden)
        expected[0:4] = 1.0
        expected[8:12] = 1.0
        self.assertTrue(torch.equal(hidden.grad, expected))

    def test_gather_does_not_detach(self) -> None:
        hidden = _arange_hidden(8, 2).requires_grad_(True)
        out = differentiable_group_gather(hidden, [0, 1], merge_unit=4)
        self.assertIsNotNone(out.grad_fn)
        self.assertTrue(out.requires_grad)

    def test_gather_rejects_bad_inputs(self) -> None:
        hidden = _arange_hidden(12, 3)
        cases = [
            ("hidden_1d", lambda: differentiable_group_gather(hidden[0], [0])),
            (
                "hidden_int_dtype",
                lambda: differentiable_group_gather(
                    hidden.to(torch.int64), [0]
                ),
            ),
            ("merge_unit_zero", lambda: differentiable_group_gather(hidden, [0], 0)),
            (
                "merge_unit_negative",
                lambda: differentiable_group_gather(hidden, [0], -4),
            ),
            (
                "merge_unit_bool",
                lambda: differentiable_group_gather(hidden, [0], True),
            ),
            (
                "not_divisible",
                lambda: differentiable_group_gather(hidden[:10], [0]),
            ),
            ("empty_indices", lambda: differentiable_group_gather(hidden, [])),
            ("duplicate_indices", lambda: differentiable_group_gather(hidden, [1, 1])),
            ("out_of_range", lambda: differentiable_group_gather(hidden, [3])),
            ("negative_index", lambda: differentiable_group_gather(hidden, [-1])),
            (
                "float_index",
                lambda: differentiable_group_gather(hidden, [0.0, 1.0]),
            ),
            (
                "bool_index",
                lambda: differentiable_group_gather(hidden, [True, False]),
            ),
            (
                "indices_2d",
                lambda: differentiable_group_gather(
                    hidden, torch.zeros((1, 1), dtype=torch.long)
                ),
            ),
            (
                "indices_float_tensor",
                lambda: differentiable_group_gather(
                    hidden, torch.zeros(1, dtype=torch.float32)
                ),
            ),
        ]
        for name, call in cases:
            with self.subTest(case=name):
                with self.assertRaises(AnswerMarginContractError):
                    call()


class ChoiceMarginTest(unittest.TestCase):
    def test_margin_numeric_with_variable_option_counts(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 1.0, 5.0, 2.0, -3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            ]
        )
        margins = choice_margin(
            logits,
            [[0, 1, 2, 3], [5, 4, 3, 2, 1, 0]],
            [2, 0],
        )
        # Q0: 5 - max(0, 1, 2) = 3; Q1: 60 - max(50, 40, 30, 20, 10) = 10.
        torch.testing.assert_close(margins, torch.tensor([3.0, 10.0]))

    def test_margin_supports_a_to_f_style_options(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        margins = choice_margin(logits, [[0, 1, 2, 3, 4, 5]], [3])
        # Correct is index 3 (logit 4); the best other option is index 5 (6).
        torch.testing.assert_close(margins, torch.tensor([-2.0]))

    def test_margin_returns_strict_python_scalar_shape(self) -> None:
        logits = torch.zeros(3, 4)
        margins = choice_margin(
            logits, [[0, 1, 2, 3], [3, 2, 1, 0], [0, 2]], [0, 3, 1]
        )
        self.assertEqual(tuple(margins.shape), (3,))

    def test_margin_is_differentiable_with_finite_nonzero_grad(self) -> None:
        logits = torch.randn(2, 5, generator=torch.Generator().manual_seed(7))
        logits.requires_grad_(True)
        margins = choice_margin(
            logits,
            [[0, 1, 2, 3], [4, 3, 2, 1, 0]],
            [1, 4],
        )
        self.assertTrue(margins.requires_grad)
        margins.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all().item()))
        self.assertTrue(bool((logits.grad.abs() > 0).any().item()))
        # +1 on the chosen option token, -1 on the strongest alternative.
        self.assertEqual(float(logits.grad[0, 1].item()), 1.0)
        # Q1 options are [4, 3, 2, 1, 0] with correct option index 4 -> token 0.
        self.assertEqual(float(logits.grad[1, 0].item()), 1.0)
        self.assertEqual(float(logits.grad.sum().item()), 0.0)

    def test_margin_rejects_bad_inputs(self) -> None:
        logits = torch.zeros(2, 5)
        cases = [
            ("logits_1d", lambda: choice_margin(logits[0], [[0, 1]], [0])),
            (
                "logits_int",
                lambda: choice_margin(logits.to(torch.int64), [[0, 1]], [0]),
            ),
            ("question_count_mismatch", lambda: choice_margin(logits, [[0, 1]], [0])),
            (
                "correct_count_mismatch",
                lambda: choice_margin(logits, [[0, 1], [0, 1]], [0]),
            ),
            ("too_few_options", lambda: choice_margin(logits, [[0], [0, 1]], [0, 0])),
            (
                "duplicate_option_ids",
                lambda: choice_margin(logits, [[1, 1], [0, 1]], [0, 0]),
            ),
            (
                "option_id_out_of_range",
                lambda: choice_margin(logits, [[0, 9], [0, 1]], [0, 0]),
            ),
            (
                "negative_option_id",
                lambda: choice_margin(logits, [[0, -1], [0, 1]], [0, 0]),
            ),
            (
                "correct_index_out_of_range",
                lambda: choice_margin(logits, [[0, 1], [0, 1]], [2, 0]),
            ),
            (
                "float_option_id",
                lambda: choice_margin(logits, [[0.0, 1.0], [0, 1]], [0, 0]),
            ),
            (
                "bool_option_id",
                lambda: choice_margin(logits, [[True, False], [0, 1]], [0, 0]),
            ),
            (
                "empty_options",
                lambda: choice_margin(logits, [[], [0, 1]], [0, 0]),
            ),
            (
                "options_not_nested",
                lambda: choice_margin(logits, [0, 1], [0, 0]),
            ),
            (
                "correct_index_2d",
                lambda: choice_margin(
                    logits,
                    [[0, 1], [0, 1]],
                    torch.zeros((1, 1), dtype=torch.long),
                ),
            ),
        ]
        for name, call in cases:
            with self.subTest(case=name):
                with self.assertRaises(AnswerMarginContractError):
                    call()

    def test_margin_from_positions_selects_and_stays_differentiable(self) -> None:
        token_logits = torch.randn(
            9, 5, generator=torch.Generator().manual_seed(11)
        ).requires_grad_(True)
        margins = choice_margin_from_positions(
            token_logits,
            [2, 6],
            [[0, 1, 2], [4, 3, 2, 1, 0]],
            [0, 2],
        )
        self.assertEqual(tuple(margins.shape), (2,))
        margins.sum().backward()
        self.assertIsNotNone(token_logits.grad)
        self.assertTrue(bool(torch.isfinite(token_logits.grad).all().item()))
        self.assertTrue(bool((token_logits.grad.abs() > 0).any().item()))

    def test_margin_from_positions_rejects_bad_positions(self) -> None:
        token_logits = torch.zeros(4, 3)
        for positions in ([], [4], [-1], [0.5]):
            with self.subTest(positions=positions):
                with self.assertRaises(AnswerMarginContractError):
                    choice_margin_from_positions(
                        token_logits, positions, [[0, 1]], [0]
                    )


class NegativeControlTest(unittest.TestCase):
    def test_detach_blocks_gradient_to_original_b1_hidden(self) -> None:
        generator = torch.Generator().manual_seed(23)
        hidden = torch.randn(16, 8, generator=generator).requires_grad_(True)
        selected = [0, 3, 1]
        head = torch.nn.Linear(8, 6, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.randn(6, 8, generator=generator) / math.sqrt(8)
            )
        options = [[0, 1, 2], [3, 4, 5]]
        correct = [0, 2]

        compact = differentiable_group_gather(hidden, selected, 4).detach()
        logits = head(compact.mean(dim=0))
        margins = choice_margin(logits.unsqueeze(0).expand(2, 6), options, correct)

        # The graph still exists through the trainable head weights, but the
        # detach must keep it from ever reaching the original B1 hidden states.
        margins.sum().backward()
        self.assertIsNotNone(head.weight.grad)
        self.assertTrue(bool((head.weight.grad.abs() > 0).any().item()))
        self.assertIsNone(hidden.grad)

    def test_differentiable_path_does_populate_gradient(self) -> None:
        generator = torch.Generator().manual_seed(29)
        hidden = torch.randn(16, 8, generator=generator).requires_grad_(True)
        selected = [0, 3, 1]
        head = torch.nn.Linear(8, 6, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.randn(6, 8, generator=generator) / math.sqrt(8)
            )
        compact = differentiable_group_gather(hidden, selected, 4)
        logits = head(compact.mean(dim=0))
        margins = choice_margin(
            logits.unsqueeze(0).expand(2, 6), [[0, 1, 2], [3, 4, 5]], [0, 2]
        )
        margins.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(bool(torch.isfinite(hidden.grad).all().item()))
        self.assertTrue(bool((hidden.grad.abs() > 0).any().item()))


class PreflightDriverTest(unittest.TestCase):
    def test_run_preflight_passes_and_is_json_serializable(self) -> None:
        result = run_preflight()
        self.assertEqual(result["status"], PREFLIGHT_STATUS_PASS)
        self.assertTrue(result["engineering_feasibility_only"])
        self.assertFalse(result["is_selector_accuracy_evidence"])
        for item in result["checks"]:
            self.assertTrue(item["passed"], item["name"])
        self.assertTrue(result["negative_control"]["b1_control_grad_is_none"])
        self.assertFalse(result["negative_control"]["control_graph_connected"])
        self.assertTrue(result["negative_control"]["control_backward_raises"])
        self.assertEqual(result["settings"]["option_counts"], [4, 6, 3])
        self.assertEqual(
            result["gather"]["output_shape"],
            [result["settings"]["selected_group_count"] * 4, result["settings"]["hidden_dim"]],
        )
        self.assertTrue(result["gather"]["order_preserved"])
        self.assertFalse(result["gather"]["uses_detach"])
        self.assertFalse(result["gather"]["uses_numpy"])
        self.assertFalse(result["gather"]["uses_cpu_rebuild"])
        self.assertTrue(result["gradient"]["b1_hidden_grad_finite"])
        self.assertTrue(result["gradient"]["b1_hidden_grad_nonzero"])
        # Round-trips as JSON.
        json.dumps(result, ensure_ascii=False)

    def test_required_caveats_are_present_in_json_and_markdown(self) -> None:
        result = run_preflight()
        statements = result["required_statements"]
        self.assertIn("不是 selector 精度证据", statements["engineering_feasibility_not_accuracy"])
        self.assertIn("full2638", statements["label_hygiene"])
        self.assertIn("一张图", statements["gpu_smoke"])
        self.assertIn("不得训练", statements["gpu_smoke"])
        markdown = render_markdown(result)
        for needle in (
            "工程可行性",
            "不是 selector 精度证据",
            "full2638",
            "一张图",
            "不得训练",
            "负对照",
            "PREFLIGHT_PASS",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, markdown)

    def test_main_writes_report_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "out"
            exit_code = main(
                [
                    "--report-dir",
                    str(report_dir),
                    "--workspace-root",
                    str(WORKSPACE_ROOT),
                ]
            )
            self.assertEqual(exit_code, 0)
            json_path = report_dir / "2026-10-05-b1-answer-margin-autograd-preflight.json"
            md_path = report_dir / "2026-10-05-b1-answer-margin-autograd-preflight.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], PREFLIGHT_STATUS_PASS)
            self.assertIn("artifacts", payload)
            self.assertTrue(payload["artifacts"]["module"]["sha256"])
            self.assertTrue(payload["artifacts"]["test"]["sha256"])

    def test_custom_settings_are_validated(self) -> None:
        with self.assertRaises(AnswerMarginContractError):
            run_preflight(PreflightSettings(selected_group_count=0))
        with self.assertRaises(AnswerMarginContractError):
            run_preflight(PreflightSettings(selected_group_count=99))
        with self.assertRaises(AnswerMarginContractError):
            run_preflight(PreflightSettings(merge_unit=0))
        result = run_preflight(
            PreflightSettings(
                seed=7,
                hidden_dim=16,
                vocab_size=32,
                group_count=10,
                selected_group_count=3,
                merge_unit=2,
            )
        )
        self.assertEqual(result["status"], PREFLIGHT_STATUS_PASS)


if __name__ == "__main__":
    unittest.main()
