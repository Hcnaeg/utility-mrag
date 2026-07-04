"""Light-weight loaders for MRAG-Bench and Visual-RAG.

These iterators are intentionally minimal: they walk over the official splits
on disk and yield raw question records that ``scripts/prepare_*`` then turn
into the unified candidate-pool format.

Raw datasets are *not* shipped with this repo; users must download them from
the official sources documented in ``data/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_mrag_bench(
    input_dir: str | Path,
    split: str = "test",
    qids_file: Optional[str | Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield MRAG-Bench questions.

    Tries (in order):

    1. The Hugging Face ``datasets`` package, loading
       ``uclanlp/MRAG-Bench`` from ``input_dir`` if it points to a local
       cache, otherwise from the hub.
    2. A plain JSONL file at ``<input_dir>/mrag_bench_<split>.jsonl``.
    """
    input_dir = Path(input_dir)

    jsonl_path = input_dir / f"mrag_bench_{split}.jsonl"
    if jsonl_path.exists():
        valid_qids: Optional[set[str]] = None
        if qids_file is not None:
            valid_qids = set(json.loads(Path(qids_file).read_text()))
        for rec in _read_jsonl(jsonl_path):
            if valid_qids is not None and rec.get("qid") not in valid_qids:
                continue
            yield rec
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading MRAG-Bench from HuggingFace requires `datasets`. "
            "Install with `uv add datasets` or use a local JSONL fallback."
        ) from exc

    ds = load_dataset("uclanlp/MRAG-Bench", split=split, cache_dir=str(input_dir))
    valid_qids = None
    if qids_file is not None:
        valid_qids = set(json.loads(Path(qids_file).read_text()))
    for rec in ds:
        if valid_qids is not None and str(rec.get("qid")) not in valid_qids:
            continue
        yield rec


def iter_visual_rag(
    input_dir: str | Path,
    split: str = "test",
    qids_file: Optional[str | Path] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield Visual-RAG questions from a JSONL file at
    ``<input_dir>/visual_rag_<split>.jsonl``.
    """
    input_dir = Path(input_dir)
    path = input_dir / f"visual_rag_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find Visual-RAG split at {path}. "
            "Convert the official release to JSONL first; see docs/data_preparation.md."
        )
    valid_qids: Optional[set[str]] = None
    if qids_file is not None:
        valid_qids = set(json.loads(Path(qids_file).read_text()))
    for rec in _read_jsonl(path):
        if valid_qids is not None and rec.get("qid") not in valid_qids:
            continue
        yield rec
