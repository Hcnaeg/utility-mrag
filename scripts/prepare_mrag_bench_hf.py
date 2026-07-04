#!/usr/bin/env python
"""Build a MRAG-Bench candidate-pool manifest directly from the HuggingFace release.

Unlike :mod:`scripts.prepare_mrag_bench` -- which expects an image corpus on
disk plus a separate retrieval JSONL -- this script is a **zero-config**
entrypoint: the official ``uclanlp/MRAG-Bench`` dataset already ships, for every
question, the input image, the ground-truth images, and the officially
CLIP-retrieved candidate images (all inline). We extract those images to disk
and emit the unified manifest consumed by ``scripts/run_selection.py``.

Usage::

    uv run python scripts/prepare_mrag_bench_hf.py \\
        --output data/manifests/mrag_bench_candidates.jsonl \\
        --image_dir data/images/mrag_bench \\
        --split test

Use ``--limit N`` to prepare a small smoke-test subset, and
``--num_candidates K`` to cap how many retrieved images are kept per question.

Requires the ``datasets`` extra (``uv sync --extra datasets``), which provides
``pyarrow``; the dataset parquet is located via ``huggingface_hub`` (offline
friendly if already cached).
"""

from __future__ import annotations

import argparse
import glob
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from utility_mrag.data.candidate_pool import (
    CandidateImage,
    CandidatePoolExample,
    write_candidate_pool,
)


def _locate_parquet(split: str) -> List[str]:
    """Return local paths to the dataset parquet shards for ``split``.

    Uses the local HuggingFace cache when present, otherwise downloads only the
    parquet files.
    """
    from huggingface_hub import snapshot_download

    local_dir = snapshot_download(
        repo_id="uclanlp/MRAG-Bench",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
    )
    shards = sorted(glob.glob(str(Path(local_dir) / "data" / f"{split}-*.parquet")))
    if not shards:
        raise FileNotFoundError(
            f"No parquet shards for split {split!r} under {local_dir}/data. "
            f"Available: {sorted(p.name for p in (Path(local_dir) / 'data').glob('*.parquet'))}"
        )
    return shards


def _iter_rows(shards: List[str], limit: Optional[int]):
    import pyarrow.parquet as pq

    seen = 0
    for shard in shards:
        table = pq.read_table(shard)
        for row in table.to_pylist():
            yield row
            seen += 1
            if limit is not None and seen >= limit:
                return


def _save_image(img_struct: Dict[str, Any], path: Path) -> None:
    from PIL import Image

    Image.open(io.BytesIO(img_struct["bytes"])).convert("RGB").save(path)


def build_examples(
    rows,
    image_dir: Path,
    *,
    num_candidates: Optional[int],
    include_gt: bool,
):
    for rec in rows:
        qid = str(rec["id"])
        qdir = image_dir / qid
        qdir.mkdir(parents=True, exist_ok=True)

        question_image_path = qdir / "input.png"
        _save_image(rec["image"], question_image_path)

        candidates: List[CandidateImage] = []
        gt_image_ids: List[str] = []

        if include_gt:
            for j, gt in enumerate(rec.get("gt_images") or []):
                p = qdir / f"gt_{j}.png"
                _save_image(gt, p)
                iid = f"{qid}_gt_{j}"
                gt_image_ids.append(iid)
                candidates.append(
                    CandidateImage(image_id=iid, image_path=str(p), source="gt")
                )

        retrieved = rec.get("retrieved_images") or []
        if num_candidates is not None:
            retrieved = retrieved[:num_candidates]
        for j, ret in enumerate(retrieved):
            p = qdir / f"ret_{j}.png"
            _save_image(ret, p)
            candidates.append(
                CandidateImage(
                    image_id=f"{qid}_ret_{j}", image_path=str(p), source="retrieved"
                )
            )

        yield CandidatePoolExample(
            qid=qid,
            query=str(rec["question"]),
            candidate_images=candidates,
            gt_image_ids=gt_image_ids,
            answer=rec.get("answer_choice"),
            choices={k: rec.get(k, "") for k in ("A", "B", "C", "D")},
            question_image_path=str(question_image_path),
            metadata={
                "scenario": rec.get("scenario", "Unknown"),
                "aspect": rec.get("aspect"),
            },
        )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--output", required=True, help="Output candidate-pool JSONL path.")
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Directory to extract per-question images into.",
    )
    parser.add_argument("--split", default="test", help="Dataset split (default: test).")
    parser.add_argument(
        "--limit", type=int, default=None, help="Prepare only the first N questions."
    )
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=None,
        help="Keep at most K retrieved images per question (default: all).",
    )
    parser.add_argument(
        "--no_gt",
        action="store_true",
        help="Do not include ground-truth images as candidates.",
    )
    args = parser.parse_args(argv)

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    shards = _locate_parquet(args.split)
    rows = _iter_rows(shards, args.limit)
    examples = build_examples(
        rows,
        image_dir,
        num_candidates=args.num_candidates,
        include_gt=not args.no_gt,
    )
    n = write_candidate_pool(examples, args.output)
    print(f"Wrote {n} examples to {args.output} (images under {image_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
