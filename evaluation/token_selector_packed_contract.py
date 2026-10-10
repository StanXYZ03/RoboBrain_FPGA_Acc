"""128-bit packed token-word contract for the fixed-point selector.

The fixed-point selector golden (:mod:`hardware_regular_selector_fixedpoint`)
stays the frozen ``int8[G, 4, 1024]`` byte contract.  The hardware datapath is
widened to 128-bit words on the token ports only:

* one word holds 16 consecutive **little-endian** int8 lanes; lane ``L`` is the
  byte at offset ``L`` inside the word (bits ``[8L+7:8L]``),
* one 4096 B group is exactly ``256`` words,
* the word at ``(group, w)`` corresponds to the original contiguous byte at
  ``group * 4096 + w * 16``,
* the 5-segment host layout, its 64 B alignment and its total byte counts are
  unchanged; only the token segment transfer element is re-typed to 128 bit.

Nothing here runs a model, GPU or board; it is pure CPU integer arithmetic.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from evaluation import hardware_regular_selector_fixedpoint as fp

PACKED_LANES = 16
WORD_BYTES = 16
GROUP_WORDS = fp.GROUP_ELEMENTS // PACKED_LANES  # 256
TOKEN_SEGMENT_NAME = "tokens_int8"


def byte_to_word_lane(byte_index: int) -> tuple[int, int]:
    """Map a contiguous token byte index to ``(word, lane)`` (little endian)."""
    if byte_index < 0:
        raise fp.FixedpointContractError("byte_index must be non-negative")
    return int(byte_index) // PACKED_LANES, int(byte_index) % PACKED_LANES


def group_word_address(group_index: int, word_index: int) -> int:
    """Flat word index equivalent to the original ``group*4096 + word*16`` byte."""
    if group_index < 0 or word_index < 0 or word_index >= GROUP_WORDS:
        raise fp.FixedpointContractError("group/word index out of range")
    return int(group_index) * GROUP_WORDS + int(word_index)


def pack_int8_to_words(
    values: Any, start_byte: int = 0, total_bytes: int | None = None
) -> np.ndarray:
    """Pack int8 values into ``(n_words, 16)`` words, little-endian lanes.

    The first value lands at global byte ``start_byte`` so an arbitrary lane
    offset is representable; out-of-range bytes (prefix and tail) are zero.
    Row ``w`` lane ``l`` is always global byte ``w * 16 + l``.
    """
    flat = np.asarray(values, dtype=np.int8).reshape(-1)
    if start_byte < 0:
        raise fp.FixedpointContractError("start_byte must be non-negative")
    if total_bytes is None:
        total_bytes = start_byte + int(flat.size)
    if total_bytes < start_byte + int(flat.size):
        raise fp.FixedpointContractError("total_bytes too small for the payload")
    n_words = (int(total_bytes) + PACKED_LANES - 1) // PACKED_LANES
    buffer = np.zeros(n_words * PACKED_LANES, dtype=np.int8)
    buffer[start_byte : start_byte + int(flat.size)] = flat
    return buffer.reshape(-1, PACKED_LANES)


def unpack_words_to_int8(
    words: Any, start_byte: int = 0, byte_count: int | None = None
) -> np.ndarray:
    """Inverse of :func:`pack_int8_to_words` for arbitrary byte offsets."""
    flat = np.asarray(words, dtype=np.int8).reshape(-1)
    if start_byte < 0:
        raise fp.FixedpointContractError("start_byte must be non-negative")
    if byte_count is None:
        byte_count = int(flat.size) - start_byte
    end = start_byte + int(byte_count)
    if end > flat.size:
        raise fp.FixedpointContractError("byte_count exceeds the word payload")
    return np.ascontiguousarray(flat[start_byte:end])


def packed_tokens_view(tokens_q: Any) -> np.ndarray:
    """Return ``(G, 256, 16)`` little-endian word view of the frozen tokens."""
    tokens_q = fp.validate_tokens_q(tokens_q)
    return tokens_q.reshape(tokens_q.shape[0], GROUP_WORDS, PACKED_LANES)


def tokens_segment_words(budget: int) -> int:
    """Number of 128-bit words in the tokens segment (bytes/16, unchanged)."""
    return fp.int8_token_bytes(budget) // WORD_BYTES


def tokens_segment_bytes(budget: int) -> int:
    """Unchanged byte count of the tokens segment."""
    return fp.int8_token_bytes(budget)


def pool_context_tokens_q_packed(
    tokens_q: Any,
    score_q: Any,
    remaining: Sequence[int],
) -> np.ndarray:
    """Word-based context pooling that mirrors ``fp.pool_context_tokens_q``.

    Each 128-bit word contributes its 16 lanes at once; the accumulation is
    exact int64 and there is one round-half-away-from-zero division per output
    element, so the result is byte-identical to the int8 golden.
    """
    tokens_q = fp.validate_tokens_q(tokens_q)
    scores = fp.validate_score_q(score_q, tokens_q.shape[0])
    indices = [int(group) for group in remaining]
    if not indices:
        raise fp.FixedpointContractError("context pooling requires remaining groups")
    weights, total = fp.context_pooling_weights_q(scores, indices)
    group_words = packed_tokens_view(tokens_q)
    accumulator = np.zeros((GROUP_WORDS, PACKED_LANES), dtype=np.int64)
    for weight, group in zip(weights, indices):
        accumulator += int(weight) * group_words[group].astype(np.int64)
    sign = np.where(accumulator < 0, -1, 1)
    quotient = (2 * np.abs(accumulator) + total) // (2 * total)
    pooled = (sign * quotient).astype(np.int8)
    return pooled.reshape(fp.MERGE_UNIT, fp.FEATURE_DIM)


def transfer_layout_word_summary(budget: int) -> dict[str, int]:
    """The token segment in words plus the unchanged byte totals."""
    layout = fp.fpga_transfer_layout(budget)
    token_segment = next(
        segment for segment in layout.segments if segment.name == TOKEN_SEGMENT_NAME
    )
    return {
        "budget": int(budget),
        "token_bytes": int(token_segment.size),
        "token_words": int(token_segment.size) // WORD_BYTES,
        "token_offset": int(token_segment.offset),
        "total_bytes": int(layout.total_bytes),
        "total_words_if_all_packed": int(layout.total_bytes) // WORD_BYTES,
    }
