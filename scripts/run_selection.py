#!/usr/bin/env python
"""Score a candidate pool with the surrogate model and emit Top-K selections.

Usage::

    uv run python scripts/run_selection.py \\
        --manifest data/manifests/mrag_bench_candidates.jsonl \\
        --surrogate_config configs/models/qwen3_vl_2b_surrogate.yaml \\
        --top_k 1 3 5 \\
        --output_dir outputs/selection/mrag_bench/qwen3_vl_2b
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
from utility_mrag.data.candidate_pool import load_candidate_pool
from utility_mrag.models.base import build_model
from utility_mrag.scoring.helpfulness_score import HelpfulnessScorer
from utility_mrag.selection.surrogate_selector import SurrogateSelector

logger = logging.getLogger("run_selection")


def _detect_dataset(manifest_path: Path) -> str:
    """Heuristic: the prepare-* scripts label every example with a question_image
    when the dataset is MRAG-Bench. We sniff the first record."""
    for ex in load_candidate_pool(manifest_path):
        return "mrag_bench" if ex.question_image_path else "visual_rag"
    raise ValueError(f"Manifest {manifest_path} is empty.")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--surrogate_config", required=True)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", choices=["mrag_bench", "visual_rag"], default=None)
    parser.add_argument("--template", choices=["helpfulness", "relevance"], default="helpfulness")
    parser.add_argument("--score_key", choices=["true_logit", "p_true"], default="true_logit")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N examples.")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset or _detect_dataset(manifest_path)
    logger.info("Dataset: %s", dataset)

    config = model_config_from_yaml(args.surrogate_config)
    config.role = "surrogate"
    model = build_model(config)
    model.ensure_loaded()

    scorer = HelpfulnessScorer(
        model=model,
        dataset=dataset,
        template=args.template,
        score_key=args.score_key,
    )
    selector = SurrogateSelector(scorer)

    files = {k: (output_dir / f"top{k}.jsonl").open("w", encoding="utf-8") for k in args.top_k}
    all_scores_file = (output_dir / "all_scores.jsonl").open("w", encoding="utf-8")

    n = 0
    try:
        examples = load_candidate_pool(manifest_path)
        for example in tqdm(examples, desc="scoring"):
            if args.limit is not None and n >= args.limit:
                break
            result = selector.select(example.to_dict(), top_ks=args.top_k)
            for k in args.top_k:
                payload = {
                    "qid": result["qid"],
                    "query": result["query"],
                    "top_k": int(k),
                    "selected_images": result["selections"][str(int(k))],
                    "answer": result.get("answer"),
                    "choices": result.get("choices"),
                    "metadata": result.get("metadata", {}),
                    "gt_image_ids": result.get("gt_image_ids", []),
                }
                files[k].write(json.dumps(payload, ensure_ascii=False) + "\n")
            all_scores_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1
    finally:
        for f in files.values():
            f.close()
        all_scores_file.close()

    print(f"Wrote selections for {n} examples to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
