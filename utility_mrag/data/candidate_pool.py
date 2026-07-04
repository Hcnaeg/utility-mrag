"""Candidate-pool JSONL representation.

Each example in the candidate pool corresponds to a single query and bundles
(a) the question metadata, (b) any input image required by the dataset
(MRAG-Bench), (c) the list of candidate images to be reranked, and (d) the
ground-truth answer used by evaluation.

The on-disk format is JSON Lines, one example per line, matching the schema in
``OPEN_SOURCE_PREP_PROMPT.md``::

    {
      "qid": "...",
      "query": "...",
      "question_image_path": "...",          # optional; only MRAG-Bench
      "candidate_images": [
        {"image_id": "...", "image_path": "...", "source": "gt"},
        {"image_id": "...", "image_path": "...", "source": "retrieved"}
      ],
      "gt_image_ids": ["..."],
      "answer": "...",
      "choices": {"A": "...", "B": "...", "C": "...", "D": "..."},  # optional
      "metadata": {}
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class CandidateImage:
    image_id: str
    image_path: str
    source: str = "retrieved"  # "gt" | "retrieved"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {"image_id": self.image_id, "image_path": self.image_path, "source": self.source}
        d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateImage":
        known = {"image_id", "image_path", "source"}
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            image_id=str(d["image_id"]),
            image_path=str(d["image_path"]),
            source=str(d.get("source", "retrieved")),
            extra=extra,
        )


@dataclass
class CandidatePoolExample:
    qid: str
    query: str
    candidate_images: List[CandidateImage]
    gt_image_ids: List[str] = field(default_factory=list)
    answer: Optional[str] = None
    choices: Optional[Dict[str, str]] = None
    question_image_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "qid": self.qid,
            "query": self.query,
            "candidate_images": [c.to_dict() for c in self.candidate_images],
            "gt_image_ids": list(self.gt_image_ids),
            "metadata": dict(self.metadata),
        }
        if self.answer is not None:
            d["answer"] = self.answer
        if self.choices is not None:
            d["choices"] = dict(self.choices)
        if self.question_image_path is not None:
            d["question_image_path"] = self.question_image_path
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidatePoolExample":
        cands_raw = d.get("candidate_images", [])
        cands = [CandidateImage.from_dict(c) for c in cands_raw]
        return cls(
            qid=str(d["qid"]),
            query=str(d["query"]),
            candidate_images=cands,
            gt_image_ids=[str(x) for x in d.get("gt_image_ids", [])],
            answer=d.get("answer"),
            choices=d.get("choices"),
            question_image_path=d.get("question_image_path"),
            metadata=d.get("metadata", {}),
        )


def write_candidate_pool(
    examples: Iterable[CandidatePoolExample | Dict[str, Any]],
    path: str | Path,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            payload = ex.to_dict() if isinstance(ex, CandidatePoolExample) else dict(ex)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_candidate_pool(path: str | Path) -> Iterator[CandidatePoolExample]:
    """Yield :class:`CandidatePoolExample` records from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield CandidatePoolExample.from_dict(json.loads(line))


def parse_candidate_pool_strict(records: Sequence[Dict[str, Any]]) -> List[CandidatePoolExample]:
    """Validate a list of dicts and return parsed examples (used in tests)."""
    out: List[CandidatePoolExample] = []
    for i, r in enumerate(records):
        if "qid" not in r:
            raise ValueError(f"record {i} missing 'qid'")
        if "query" not in r:
            raise ValueError(f"record {i} missing 'query'")
        if "candidate_images" not in r:
            raise ValueError(f"record {i} missing 'candidate_images'")
        out.append(CandidatePoolExample.from_dict(r))
    return out
