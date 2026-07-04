#!/usr/bin/env python
"""Build the unified candidate-pool JSONL for MRAG-Bench.

Reads the official MRAG-Bench questions and a retrieval file (mapping each
question to a ranked list of retrieved image paths), then emits one
:class:`CandidatePoolExample` per query under
``--output``.

Usage::

    uv run python scripts/prepare_mrag_bench.py \\
        --input_dir /path/to/mrag_bench \\
        --retrieval_file /path/to/retrieved_candidates.jsonl \\
        --output data/manifests/mrag_bench_candidates.jsonl

The retrieval file is expected to be JSONL with at least::

    {"qid": "...", "retrieved": [{"image_id": "...", "image_path": "..."}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from utility_mrag.data.candidate_pool import (
    CandidateImage,
    CandidatePoolExample,
    write_candidate_pool,
)
from utility_mrag.data.dataset_loaders import iter_mrag_bench


def _load_retrieval(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["qid"])] = list(rec.get("retrieved", []))
    return out


def _resolve_choices(rec: Dict[str, Any]) -> Dict[str, str]:
    return {k: rec.get(k, "") for k in ("A", "B", "C", "D")}


def build_examples(
    *,
    questions: Iterable[Dict[str, Any]],
    retrieval: Dict[str, List[Dict[str, Any]]],
    image_root: Path | None = None,
) -> Iterable[CandidatePoolExample]:
    for q in questions:
        qid = str(q.get("qid") or q.get("id"))
        retrieved = retrieval.get(qid, [])
        gt_image_paths = q.get("gt_image_paths") or q.get("gt_images") or []
        gt_image_ids: List[str] = []
        candidates: List[CandidateImage] = []
        for i, gt in enumerate(gt_image_paths):
            img_id = f"{qid}_gt_{i}"
            gt_image_ids.append(img_id)
            candidates.append(
                CandidateImage(
                    image_id=img_id,
                    image_path=str(_resolve_path(gt, image_root)),
                    source="gt",
                )
            )
        for i, ret in enumerate(retrieved):
            img_id = str(ret.get("image_id") or f"{qid}_ret_{i}")
            candidates.append(
                CandidateImage(
                    image_id=img_id,
                    image_path=str(_resolve_path(ret["image_path"], image_root)),
                    source="retrieved",
                    extra={k: v for k, v in ret.items() if k not in {"image_id", "image_path"}},
                )
            )

        question_image_path = q.get("question_image_path") or q.get("input_image_path")
        if question_image_path is not None:
            question_image_path = str(_resolve_path(question_image_path, image_root))

        yield CandidatePoolExample(
            qid=qid,
            query=str(q.get("question") or q.get("query")),
            candidate_images=candidates,
            gt_image_ids=gt_image_ids,
            answer=q.get("answer") or q.get("gt_choice"),
            choices=_resolve_choices(q) if any(k in q for k in ("A", "B", "C", "D")) else None,
            question_image_path=question_image_path,
            metadata={"scenario": q.get("scenario", "Unknown")},
        )


def _resolve_path(p: str, root: Path | None) -> str:
    if root is None or Path(p).is_absolute():
        return p
    return str(root / p)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--input_dir", required=True, help="Directory holding MRAG-Bench data.")
    parser.add_argument("--retrieval_file", required=True, help="Retrieval JSONL.")
    parser.add_argument("--output", required=True, help="Output candidate-pool JSONL path.")
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument("--image_root", default=None, help="Optional root for relative image paths.")
    parser.add_argument("--qids_file", default=None, help="Optional JSON list of qids to keep.")
    args = parser.parse_args(argv)

    retrieval = _load_retrieval(Path(args.retrieval_file))
    questions = iter_mrag_bench(args.input_dir, split=args.split, qids_file=args.qids_file)
    image_root = Path(args.image_root) if args.image_root else None
    examples = build_examples(
        questions=questions,
        retrieval=retrieval,
        image_root=image_root,
    )
    n = write_candidate_pool(examples, args.output)
    print(f"Wrote {n} examples to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
