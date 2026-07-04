"""Tests for deterministic Top-K selection."""

from __future__ import annotations

from utility_mrag.selection.topk import select_top_k, sort_records_descending


def test_basic_top_k():
    records = [
        {"image_id": "a", "score": 1.0},
        {"image_id": "b", "score": 3.0},
        {"image_id": "c", "score": 2.0},
    ]
    top1 = select_top_k(records, 1)
    assert [r["image_id"] for r in top1] == ["b"]
    top3 = select_top_k(records, 3)
    assert [r["image_id"] for r in top3] == ["b", "c", "a"]
    assert [r["rank"] for r in top3] == [1, 2, 3]


def test_zero_or_negative_k_returns_empty():
    records = [{"image_id": "a", "score": 1.0}]
    assert select_top_k(records, 0) == []
    assert select_top_k(records, -1) == []


def test_k_greater_than_n_returns_all_sorted():
    records = [
        {"image_id": "a", "score": 1.0},
        {"image_id": "b", "score": 2.0},
    ]
    out = select_top_k(records, 10)
    assert len(out) == 2
    assert out[0]["image_id"] == "b"


def test_ties_broken_by_input_order_deterministically():
    records = [
        {"image_id": "first", "score": 1.0},
        {"image_id": "second", "score": 1.0},
        {"image_id": "third", "score": 1.0},
    ]
    sorted_recs = sort_records_descending(records)
    assert [r["image_id"] for r in sorted_recs] == ["first", "second", "third"]
    # Re-running yields the same ordering.
    again = sort_records_descending(records)
    assert sorted_recs == again


def test_top_k_preserves_full_score_payload():
    records = [
        {"image_id": "a", "score": 0.5, "image_path": "/x.jpg", "extra": 7},
        {"image_id": "b", "score": 0.9, "image_path": "/y.jpg"},
    ]
    out = select_top_k(records, 1)
    assert out[0]["image_path"] == "/y.jpg"
    assert "score" in out[0]
    assert out[0]["rank"] == 1


def test_dataclass_input_supported():
    from dataclasses import dataclass

    @dataclass
    class R:
        image_id: str
        score: float

    out = select_top_k([R("a", 0.1), R("b", 0.8)], 1)
    assert out[0]["image_id"] == "b"
