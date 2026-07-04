"""Quickstart: rank a tiny toy candidate pool by mocked True/False logits.

This example runs end-to-end *without* loading any real model weights -- it
mocks the surrogate model's ``score_true_false_logits`` so you can verify the
plumbing (candidate-pool parsing, scoring, Top-K selection) on a CPU-only
machine.

Usage::

    uv run python examples/quickstart_selection.py
"""

from __future__ import annotations

import json
from pathlib import Path

from utility_mrag.data.candidate_pool import load_candidate_pool
from utility_mrag.scoring.helpfulness_score import HelpfulnessScorer
from utility_mrag.selection.surrogate_selector import SurrogateSelector


class MockSurrogate:
    """Returns helpfulness scores derived from the candidate's filename.

    Useful for smoke tests; the real surrogate is a multimodal LLM and is
    constructed via :func:`utility_mrag.models.base.build_model`.
    """

    family = "mock"
    tokenizer = None  # bypasses the optional sanity-check in the scorer

    def score_true_false_logits(self, *, query: str, image_paths):
        # Score = number of "good" tokens in the candidate path; tie-break by length.
        path = str(image_paths[-1])
        true_logit = float(path.count("good") * 5.0 + path.count("evidence"))
        false_logit = 1.0
        import math

        denom = math.exp(true_logit) + math.exp(false_logit)
        return {
            "true_logit": true_logit,
            "false_logit": false_logit,
            "p_true": math.exp(true_logit) / denom,
            "p_false": math.exp(false_logit) / denom,
        }


def main() -> int:
    here = Path(__file__).parent
    manifest = here / "toy_candidate_pool.jsonl"

    examples = list(load_candidate_pool(manifest))
    print(f"Loaded {len(examples)} toy examples from {manifest}")

    scorer = HelpfulnessScorer(model=MockSurrogate(), dataset="visual_rag")
    selector = SurrogateSelector(scorer)

    for ex in examples:
        result = selector.select(ex.to_dict(), top_ks=(1, 2, 3))
        print(json.dumps(
            {
                "qid": result["qid"],
                "query": result["query"],
                "top1": [c["image_id"] for c in result["selections"]["1"]],
                "top3": [c["image_id"] for c in result["selections"]["3"]],
                "scores": [(c["image_id"], round(c["score"], 3)) for c in result["all_scores"]],
            },
            indent=2,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
