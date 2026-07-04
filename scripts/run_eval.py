#!/usr/bin/env python
"""Evaluate generation outputs on MRAG-Bench or Visual-RAG.

Usage::

    uv run python scripts/run_eval.py \\
        --dataset mrag_bench \\
        --pred outputs/generation/mrag_bench/qwen3_vl_8b_top3.jsonl \\
        --output outputs/eval/mrag_bench/qwen3_vl_8b_top3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from utility_mrag.evaluation.mrag_bench import score_mrag_bench
from utility_mrag.evaluation.visual_rag import score_visual_rag


def _read_jsonl_or_json(path: Path):
    text = path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", choices=["mrag_bench", "visual_rag"], required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Visual-RAG only. Uses OPENAI_API_KEY from environment.")
    parser.add_argument("--judge_model", default="gpt-4o-mini")
    args = parser.parse_args(argv)

    records = _read_jsonl_or_json(Path(args.pred))
    if args.dataset == "mrag_bench":
        result = score_mrag_bench(records)
    else:
        result = score_visual_rag(
            records,
            use_llm_judge=args.use_llm_judge,
            judge_model=args.judge_model,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in {"per_scenario"}}
    if "per_scenario" in result:
        summary["per_scenario"] = result["per_scenario"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
