"""Tests for the candidate-pool JSONL parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from utility_mrag.data.candidate_pool import (
    CandidateImage,
    CandidatePoolExample,
    load_candidate_pool,
    parse_candidate_pool_strict,
    write_candidate_pool,
)


def _make_example() -> CandidatePoolExample:
    return CandidatePoolExample(
        qid="q1",
        query="What animal is shown?",
        candidate_images=[
            CandidateImage("img-1", "/tmp/a.jpg", "gt"),
            CandidateImage("img-2", "/tmp/b.jpg", "retrieved", extra={"rank": 2}),
        ],
        gt_image_ids=["img-1"],
        answer="A",
        choices={"A": "cat", "B": "dog", "C": "bird", "D": "fish"},
        question_image_path="/tmp/q.jpg",
        metadata={"scenario": "Angle"},
    )


def test_round_trip_jsonl(tmp_path: Path):
    ex = _make_example()
    out = tmp_path / "pool.jsonl"
    n = write_candidate_pool([ex], out)
    assert n == 1
    loaded = list(load_candidate_pool(out))
    assert len(loaded) == 1
    rec = loaded[0]
    assert rec.qid == "q1"
    assert rec.candidate_images[0].image_id == "img-1"
    assert rec.candidate_images[0].source == "gt"
    assert rec.candidate_images[1].extra["rank"] == 2
    assert rec.gt_image_ids == ["img-1"]
    assert rec.answer == "A"
    assert rec.metadata == {"scenario": "Angle"}
    assert rec.question_image_path == "/tmp/q.jpg"


def test_strict_parser_validates_required_fields():
    parsed = parse_candidate_pool_strict([_make_example().to_dict()])
    assert len(parsed) == 1

    with pytest.raises(ValueError):
        parse_candidate_pool_strict([{"qid": "x"}])
    with pytest.raises(ValueError):
        parse_candidate_pool_strict([{"query": "x"}])
    with pytest.raises(ValueError):
        parse_candidate_pool_strict([{"qid": "x", "query": "y"}])


def test_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "pool.jsonl"
    payload = json.dumps(_make_example().to_dict())
    path.write_text(f"\n{payload}\n\n{payload}\n", encoding="utf-8")
    loaded = list(load_candidate_pool(path))
    assert len(loaded) == 2


def test_to_dict_round_trip_preserves_extra_candidate_fields():
    ex = _make_example()
    d = ex.to_dict()
    assert d["candidate_images"][1]["rank"] == 2
    rebuilt = CandidatePoolExample.from_dict(d)
    assert rebuilt.candidate_images[1].extra["rank"] == 2


def test_visual_rag_style_no_choices(tmp_path: Path):
    ex = CandidatePoolExample(
        qid="vr-1",
        query="What color is its tail?",
        candidate_images=[CandidateImage("a", "/x.jpg")],
        answer="red",
    )
    path = tmp_path / "vr.jsonl"
    write_candidate_pool([ex], path)
    loaded = list(load_candidate_pool(path))
    assert loaded[0].choices is None
    assert loaded[0].question_image_path is None
