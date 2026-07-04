"""Unit tests for the True/False logit extractor.

These tests use a tiny mock tokenizer + handcrafted logits so they run without
loading any actual model weights.
"""

from __future__ import annotations

import math

import pytest
import torch

from utility_mrag.scoring.true_false_logits import (
    TrueFalseLogitExtractor,
    extract_true_false_logits,
)


class MockTokenizer:
    """Minimal tokenizer mirroring the HF interface used by the extractor."""

    def __init__(self, mapping: dict[str, list[int]]):
        self._mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False):
        if text in self._mapping:
            return list(self._mapping[text])
        if text == " " + text.strip() and text.strip() in self._mapping:
            return list(self._mapping[text.strip()])
        raise KeyError(text)


def _vocab_logits(true_id: int, false_id: int, vocab_size: int = 100):
    logits = torch.zeros(vocab_size)
    logits[true_id] = 5.0
    logits[false_id] = 2.0
    return logits


def test_single_token_true_false():
    tok = MockTokenizer({"True": [10], " True": [10], "False": [20], " False": [20]})
    logits = _vocab_logits(true_id=10, false_id=20)
    out = extract_true_false_logits(first_step_logits=logits, tokenizer=tok)
    assert out["true_logit"] == pytest.approx(5.0)
    assert out["false_logit"] == pytest.approx(2.0)
    # softmax over [5, 2] -> p_true = e^5 / (e^5 + e^2)
    expected_p = math.exp(5.0) / (math.exp(5.0) + math.exp(2.0))
    assert out["p_true"] == pytest.approx(expected_p, rel=1e-6)
    assert out["true_token_id"] == 10
    assert out["false_token_id"] == 20
    assert out["true_is_single_token"] is True
    assert out["false_is_single_token"] is True
    assert out["score"] == pytest.approx(5.0)


def test_falls_back_to_leading_space_when_raw_is_multi_token():
    tok = MockTokenizer({
        "True": [11, 12],
        " True": [13],
        "False": [21, 22],
        " False": [23],
    })
    logits = _vocab_logits(true_id=13, false_id=23)
    out = extract_true_false_logits(first_step_logits=logits, tokenizer=tok)
    assert out["true_token_id"] == 13
    assert out["false_token_id"] == 23
    assert out["true_is_single_token"] is True
    assert out["false_is_single_token"] is True


def test_multi_token_fallback_uses_first_piece():
    tok = MockTokenizer({
        "True": [30, 31, 32],
        " True": [33, 34],
        "False": [40, 41],
        " False": [42, 43],
    })
    logits = torch.zeros(100)
    logits[30] = 4.0
    logits[40] = 1.0
    out = extract_true_false_logits(first_step_logits=logits, tokenizer=tok)
    assert out["true_is_single_token"] is False
    assert out["false_is_single_token"] is False
    assert out["true_token_id"] == 30
    assert out["false_token_id"] == 40


def test_class_caches_token_ids():
    tok = MockTokenizer({"True": [10], " True": [10], "False": [20], " False": [20]})
    extractor = TrueFalseLogitExtractor(tok)
    assert extractor.positive_token_id == 10
    assert extractor.negative_token_id == 20
    logits = _vocab_logits(10, 20)
    out = extractor.compute(logits)
    assert out["score"] == pytest.approx(5.0)


def test_compute_from_scores_uses_first_step():
    tok = MockTokenizer({"True": [10], " True": [10], "False": [20], " False": [20]})
    extractor = TrueFalseLogitExtractor(tok)
    step0 = _vocab_logits(10, 20).unsqueeze(0)  # [1, vocab]
    step1 = torch.zeros(1, 100)  # ignored
    out = extractor.compute_from_scores([step0, step1])
    assert out["true_logit"] == pytest.approx(5.0)


def test_empty_scores_raises():
    tok = MockTokenizer({"True": [10], " True": [10], "False": [20], " False": [20]})
    extractor = TrueFalseLogitExtractor(tok)
    with pytest.raises(ValueError):
        extractor.compute_from_scores([])
