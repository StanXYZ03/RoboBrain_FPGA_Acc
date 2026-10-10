"""CPU-only tests for the B1 answer-margin GPU smoke helpers.

These tests never load model weights, never touch CUDA, and never train.  They
cover the pure contract logic that must hold *before* any GPU work: real-tokenizer
label branch parsing (common prefix/suffix, exactly one distinct branch token),
fail-closed behaviour for duplicate or multi-token branches, answer-index
parsing, and the resource gate that stops the smoke when CUDA is unavailable.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evaluation.run_b1_answer_margin_gpu_smoke import (
    STATUS_FAIL,
    STATUS_PASS,
    AnswerMarginSmokeError,
    SmokeConfig,
    answer_index,
    build_option_labels,
    parse_label_branches,
    render_markdown,
    run_smoke,
    tokenize_labels,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class FakeTokenizer:
    """Minimal tokenizer: space-separated character codes, records the kwargs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, bool(kwargs.get("add_special_tokens", False))))
        return {"input_ids": [ord(character) for character in text]}


class BuildOptionLabelsTest(unittest.TestCase):
    def test_six_options_are_a_to_f(self) -> None:
        self.assertEqual(
            build_option_labels(6),
            ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"],
        )

    def test_rejects_too_few_or_too_many(self) -> None:
        for value in (0, 1, 27, True):
            with self.subTest(value=value):
                with self.assertRaises(AnswerMarginSmokeError):
                    build_option_labels(value)


class TokenizeLabelsTest(unittest.TestCase):
    def test_uses_add_special_tokens_false_and_keeps_order(self) -> None:
        tokenizer = FakeTokenizer()
        labels = ["(A)", "(B)"]
        tokenised = tokenize_labels(tokenizer, labels)
        self.assertEqual(tokenised, [[ord("("), ord("A"), ord(")")],
                                     [ord("("), ord("B"), ord(")")]])
        self.assertEqual(tokenizer.calls, [("(A)", False), ("(B)", False)])

    def test_rejects_empty_tokenisation(self) -> None:
        tokenizer = FakeTokenizer()
        with self.assertRaises(AnswerMarginSmokeError):
            tokenize_labels(tokenizer, [""])

    def test_rejects_non_string_labels(self) -> None:
        tokenizer = FakeTokenizer()
        with self.assertRaises(AnswerMarginSmokeError):
            tokenize_labels(tokenizer, [1, 2])


class ParseLabelBranchesTest(unittest.TestCase):
    def test_common_prefix_and_common_suffix(self) -> None:
        branches = parse_label_branches(
            [[10, 20, 8], [10, 21, 8], [10, 22, 8]]
        )
        self.assertEqual(branches.common_prefix, [10])
        self.assertEqual(branches.branch_token_ids, [20, 21, 22])
        self.assertEqual(branches.common_suffix, [8])
        self.assertEqual(branches.prefix_length, 1)
        self.assertEqual(branches.suffix_length, 1)

    def test_no_prefix_only_common_suffix(self) -> None:
        # This mirrors the real Qwen tokenizer for "(A)".."(F)": [(A, )], [(B, )].
        branches = parse_label_branches(
            [[4346, 8], [5349, 8], [3025, 8], [5432, 8], [10722, 8], [7832, 8]]
        )
        self.assertEqual(branches.common_prefix, [])
        self.assertEqual(branches.common_suffix, [8])
        self.assertEqual(
            branches.branch_token_ids, [4346, 5349, 3025, 5432, 10722, 7832]
        )

    def test_no_prefix_no_suffix(self) -> None:
        branches = parse_label_branches([[1], [2], [3]])
        self.assertEqual(branches.common_prefix, [])
        self.assertEqual(branches.common_suffix, [])
        self.assertEqual(branches.branch_token_ids, [1, 2, 3])

    def test_multi_token_branch_fails_closed(self) -> None:
        # After the common prefix the labels differ, but a label keeps an extra
        # inner token, so it is not a single-branch-token contract.
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[10, 20, 30, 8], [10, 21, 9, 8], [10, 22, 31, 8]])

    def test_duplicate_branch_tokens_fail_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[10, 20, 8], [10, 21, 8], [10, 20, 8]])

    def test_label_consumed_by_common_prefix_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[1, 2], [1, 2, 3]])

    def test_fewer_than_two_labels_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[1]])

    def test_empty_sequence_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[1, 2], []])

    def test_non_int_token_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            parse_label_branches([[1, 2.0], [1, 3]])


class AnswerIndexTest(unittest.TestCase):
    def test_maps_e_to_four(self) -> None:
        labels = build_option_labels(6)
        self.assertEqual(answer_index("(E)", labels), 4)
        self.assertEqual(answer_index("(A)", labels), 0)
        self.assertEqual(answer_index("(F)", labels), 5)

    def test_missing_answer_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            answer_index("(G)", build_option_labels(6))

    def test_duplicate_labels_fail_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            answer_index("(A)", ["(A)", "(A)"])

    def test_empty_labels_fail_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            answer_index("(A)", [])

    def test_non_string_answer_fails_closed(self) -> None:
        with self.assertRaises(AnswerMarginSmokeError):
            answer_index(4, build_option_labels(6))


class ResourceGateTest(unittest.TestCase):
    def test_cuda_unavailable_fails_closed_without_loading_model(self) -> None:
        config = SmokeConfig(
            model=Path("/nonexistent/model"),
            manifest=Path("/nonexistent/manifest.jsonl"),
            workspace_root=WORKSPACE_ROOT,
        )
        with mock.patch(
            "evaluation.run_b1_answer_margin_gpu_smoke.torch.cuda.is_available",
            return_value=False,
        ):
            result = run_smoke(config)
        self.assertEqual(result["status"], STATUS_FAIL)
        self.assertNotEqual(result["status"], STATUS_PASS)
        self.assertEqual(result["fail_closed_reason"], "cuda_unavailable")
        self.assertIsNone(result["sample"])
        self.assertIsNone(result["margin"])
        self.assertIsNone(result["gradient"])
        # Serialises cleanly even on the stop path.
        json.dumps(result, ensure_ascii=False)
        markdown = render_markdown(result)
        for needle in ("FAIL_CLOSED", "不是 selector 精度证据", "不是训练", "full2638"):
            with self.subTest(needle=needle):
                self.assertIn(needle, markdown)


class MarkdownContractTest(unittest.TestCase):
    def test_caveats_present(self) -> None:
        with mock.patch(
            "evaluation.run_b1_answer_margin_gpu_smoke.torch.cuda.is_available",
            return_value=False,
        ):
            result = run_smoke(SmokeConfig(workspace_root=WORKSPACE_ROOT))
        markdown = render_markdown(result)
        for needle in (
            "不是 selector 精度证据",
            "不是训练",
            "full2638",
            "一张图",
            "不是稳态推理延时",
            "synchronize",
        ):
            with self.subTest(needle=needle):
                self.assertIn(needle, markdown)

    def test_main_writes_json_and_markdown(self) -> None:
        from evaluation.run_b1_answer_margin_gpu_smoke import main

        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory) / "out"
            with mock.patch(
                "evaluation.run_b1_answer_margin_gpu_smoke.torch.cuda.is_available",
                return_value=False,
            ):
                exit_code = main(
                    [
                        "--report-dir",
                        str(report_dir),
                        "--workspace-root",
                        str(WORKSPACE_ROOT),
                    ]
                )
            self.assertEqual(exit_code, 1)
            json_path = report_dir / "2026-10-05-b1-answer-margin-gpu-smoke.json"
            md_path = report_dir / "2026-10-05-b1-answer-margin-gpu-smoke.md"
            self.assertTrue(json_path.is_file())
            self.assertTrue(md_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], STATUS_FAIL)
            self.assertEqual(payload["fail_closed_reason"], "cuda_unavailable")


if __name__ == "__main__":
    unittest.main()
