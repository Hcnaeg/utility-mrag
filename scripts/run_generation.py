#!/usr/bin/env python
"""Generate final answers with the main model on selected Top-K evidence.

Usage::

    uv run python scripts/run_generation.py \\
        --selection_file outputs/selection/mrag_bench/qwen3_vl_2b/top3.jsonl \\
        --main_model_config configs/models/qwen3_vl_8b.yaml \\
        --output outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm

from utility_mrag.config import model_config_from_yaml
from utility_mrag.models.base import build_model
from utility_mrag.scoring.prompt_templates import format_generation_prompt

logger = logging.getLogger("run_generation")


def _detect_dataset(record: dict) -> str:
    return "mrag_bench" if record.get("choices") else "visual_rag"


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--selection_file", required=True)
    parser.add_argument("--main_model_config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=["mrag_bench", "visual_rag"], default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--question_image_lookup", default=None,
                        help="Optional candidate-pool JSONL used to recover question_image_path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    selection_path = Path(args.selection_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    qid_to_question_image = {}
    if args.question_image_lookup:
        from utility_mrag.data.candidate_pool import load_candidate_pool

        for ex in load_candidate_pool(args.question_image_lookup):
            if ex.question_image_path:
                qid_to_question_image[ex.qid] = ex.question_image_path

    config = model_config_from_yaml(args.main_model_config)
    config.role = "main"
    model = build_model(config)
    model.ensure_loaded()

    n = 0
    with output_path.open("w", encoding="utf-8") as out_f:
        for rec in tqdm(_load_jsonl(selection_path), desc="generating"):
            if args.limit is not None and n >= args.limit:
                break
            dataset = args.dataset or _detect_dataset(rec)
            choices = rec.get("choices")
            prompt = format_generation_prompt(
                dataset=dataset,
                question=rec["query"],
                choices=choices,
            )

            image_paths: list[str] = []
            qimg = qid_to_question_image.get(rec["qid"])
            if dataset == "mrag_bench" and qimg:
                image_paths.append(qimg)
            for cand in rec["selected_images"]:
                if cand.get("image_path"):
                    image_paths.append(cand["image_path"])

            response = model.generate_answer(
                query=prompt,
                image_paths=image_paths,
                max_new_tokens=args.max_new_tokens,
            )
            out_f.write(
                json.dumps(
                    {
                        "qid": rec["qid"],
                        "query": rec["query"],
                        "top_k": rec.get("top_k"),
                        "output": response,
                        "selected_image_ids": [c["image_id"] for c in rec["selected_images"]],
                        "answer": rec.get("answer"),
                        "gt_choice": rec.get("answer"),
                        "choices": choices,
                        "scenario": rec.get("metadata", {}).get("scenario"),
                        **{k: rec.get("choices", {}).get(k) for k in ("A", "B", "C", "D") if rec.get("choices")},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1

    print(f"Wrote {n} generations to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
