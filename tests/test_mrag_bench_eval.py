"""Tests for MRAG-Bench multiple-choice scoring and answer extraction."""

from __future__ import annotations

from utility_mrag.evaluation.mrag_bench import (
    _extract_leading_choice,
    parse_multi_choice_response,
    score_mrag_bench,
)


def test_parse_plain_letter():
    assert parse_multi_choice_response("B", ("A", "B", "C", "D"), {}) == "B"


def test_parse_bracketed_letter():
    assert parse_multi_choice_response("The answer is (C).", ("A", "B", "C", "D"), {}) == "C"


def test_extract_leading_choice_with_label():
    # "<letter>: label" is a very common VLM answer format.
    assert _extract_leading_choice("B: Yorkshire_terrier") == "B"
    assert _extract_leading_choice("(C) New York City") == "C"
    assert _extract_leading_choice("D. capuchin") == "D"
    assert _extract_leading_choice("A") == "A"


def test_extract_leading_choice_ignores_prose():
    # A sentence merely starting with A-D must not be mistaken for a choice.
    assert _extract_leading_choice("Based on the image, it is a dog.") is None
    assert _extract_leading_choice("Cats are visible here.") is None
    assert _extract_leading_choice("") is None


def test_score_recovers_letter_label_answers_without_random_fallback():
    choices = {"A": "poodle", "B": "Yorkshire terrier", "C": "capuchin", "D": "gibbon"}
    records = [
        {"output": "B: Yorkshire terrier", "gt_choice": "B", **choices},
        {"output": "C: capuchin", "gt_choice": "C", **choices},
    ]
    result = score_mrag_bench(records)
    assert result["unparsed"] == 0
    assert result["overall_accuracy"] == 1.0


def test_score_counts_unparsed_when_no_choice_present():
    choices = {"A": "poodle", "B": "Yorkshire terrier", "C": "capuchin", "D": "gibbon"}
    records = [
        {"output": "It is impossible to determine from the photo.", "gt_choice": "A", **choices},
    ]
    result = score_mrag_bench(records)
    assert result["unparsed"] == 1
