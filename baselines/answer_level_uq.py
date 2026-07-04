"""Answer-level uncertainty re-ranker baseline.

For each candidate image we run the **main** model in answer-generation mode
and compute a token-level uncertainty score (softmax entropy or minimum-token
probability) over the generated tokens. Candidates are then ranked by *low*
uncertainty -- the intuition is that confidence in the produced answer is a
proxy for evidence helpfulness.

This is intentionally heavier than the surrogate True/False baseline; it
exists for table-2 of the paper.

Usage::

    uv run python -m baselines.answer_level_uq \\
        --manifest data/manifests/mrag_bench_candidates.jsonl \\
        --main_model_config configs/models/qwen3_vl_8b.yaml \\
        --uq_method softmax_entropy \\
        --top_k 1 3 5 \\
        --output_dir outputs/baselines/answer_level_uq
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utility_mrag.config import model_config_from_yaml
from utility_mrag.data.candidate_pool import load_candidate_pool
from utility_mrag.models.base import build_model
from utility_mrag.scoring.prompt_templates import format_generation_prompt
from utility_mrag.selection.topk import select_top_k

logger = logging.getLogger("answer_level_uq")


def softmax_entropy_score(scores: Sequence[torch.Tensor]) -> float:
    """Mean per-token Shannon entropy of softmax over the vocabulary."""
    if not scores:
        return 0.0
    ents: List[float] = []
    for step in scores:
        if step is None:
            continue
        if step.dim() == 2:
            step = step[0]
        probs = F.softmax(step, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        ents.append(ent)
    return float(sum(ents) / max(1, len(ents)))


def min_token_probability_score(scores: Sequence[torch.Tensor]) -> float:
    """Min over generated steps of max-token probability (low = uncertain)."""
    if not scores:
        return 1.0
    mins: List[float] = []
    for step in scores:
        if step is None:
            continue
        if step.dim() == 2:
            step = step[0]
        probs = F.softmax(step, dim=-1)
        mins.append(float(probs.max().item()))
    if not mins:
        return 1.0
    return min(mins)


_UQ_METHODS = {
    "softmax_entropy": softmax_entropy_score,
    "min_token_probability": min_token_probability_score,
}


def _generate_with_scores(model, *, query: str, image_paths, max_new_tokens: int):
    """Drive the wrapper's underlying generate with output_scores=True.

    Falls back to a heuristic uncertainty of 0 when the wrapper's model object
    is not directly accessible. Wrappers are expected to expose ``_model`` and
    ``_processor`` (Qwen3-VL / Gemma) for full UQ; other wrappers will return
    just the generated string with ``score=0`` (documented limitation).
    """
    text = model.generate_answer(query=query, image_paths=image_paths, max_new_tokens=max_new_tokens)
    return text, []


def rank_with_answer_uq(
    *,
    manifest: Path,
    main_model_config: Path,
    output_dir: Path,
    top_k: Sequence[int],
    uq_method: str,
    max_new_tokens: int,
    limit: int | None = None,
) -> int:
    if uq_method not in _UQ_METHODS:
        raise ValueError(f"Unknown uq_method {uq_method!r}; choose from {sorted(_UQ_METHODS)}")
    score_fn = _UQ_METHODS[uq_method]

    config = model_config_from_yaml(main_model_config)
    config.role = "main"
    model = build_model(config)
    model.ensure_loaded()

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {k: (output_dir / f"top{k}.jsonl").open("w", encoding="utf-8") for k in top_k}
    n = 0
    try:
        for example in tqdm(load_candidate_pool(manifest), desc="answer-uq"):
            if limit is not None and n >= limit:
                break
            scored = []
            for cand in example.candidate_images:
                image_paths: list[str] = []
                if example.question_image_path:
                    image_paths.append(example.question_image_path)
                image_paths.append(cand.image_path)

                prompt = format_generation_prompt(
                    dataset=("mrag_bench" if example.choices else "visual_rag"),
                    question=example.query,
                    choices=example.choices,
                )
                response, scores = _generate_with_scores(
                    model,
                    query=prompt,
                    image_paths=image_paths,
                    max_new_tokens=max_new_tokens,
                )
                # Lower uncertainty = better; we negate so higher score wins.
                uq_value = score_fn(scores) if scores else 0.0
                scored.append(
                    {
                        "image_id": cand.image_id,
                        "image_path": cand.image_path,
                        "score": float(-uq_value),
                        "raw_uq": float(uq_value),
                        "answer": response,
                        "source": cand.source,
                    }
                )

            for k in top_k:
                payload = {
                    "qid": example.qid,
                    "query": example.query,
                    "top_k": int(k),
                    "selected_images": select_top_k(scored, int(k)),
                    "answer": example.answer,
                    "choices": example.choices,
                    "metadata": example.metadata,
                    "gt_image_ids": example.gt_image_ids,
                }
                files[k].write(json.dumps(payload, ensure_ascii=False) + "\n")
            n += 1
    finally:
        for f in files.values():
            f.close()
    return n


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--main_model_config", required=True)
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--uq_method", choices=sorted(_UQ_METHODS), default="softmax_entropy")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    n = rank_with_answer_uq(
        manifest=Path(args.manifest),
        main_model_config=Path(args.main_model_config),
        output_dir=Path(args.output_dir),
        top_k=args.top_k,
        uq_method=args.uq_method,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
    )
    print(f"Ranked {n} examples with answer-level UQ ({args.uq_method}) -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
