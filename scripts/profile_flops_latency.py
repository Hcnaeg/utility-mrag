#!/usr/bin/env python
"""Measure FLOPs and end-to-end latency for the surrogate scoring pass and
the main-model generation pass.

This is a lightweight, opt-in profile. FLOPs measurement requires the
``calflops`` package (install via ``uv sync --extra profile``); when it is not
available the script falls back to wall-clock latency only.

Usage::

    uv run python scripts/profile_flops_latency.py \\
        --model_config configs/models/qwen3_vl_2b_surrogate.yaml \\
        --task scoring \\
        --num_iters 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import List

from PIL import Image

from utility_mrag.config import model_config_from_yaml
from utility_mrag.models.base import build_model
from utility_mrag.scoring.prompt_templates import (
    format_generation_prompt,
    format_helpfulness_prompt,
)


def _make_dummy_image(size: int = 384) -> Image.Image:
    return Image.new("RGB", (size, size), color=(127, 127, 127))


def _profile_latency(fn, *, num_iters: int, warmup: int):
    times: List[float] = []
    for _ in range(warmup):
        fn()
    for _ in range(num_iters):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return {
        "mean_seconds": statistics.fmean(times),
        "stdev_seconds": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "n": num_iters,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--task", choices=["scoring", "generation"], default="scoring")
    parser.add_argument("--dataset", choices=["mrag_bench", "visual_rag"], default="mrag_bench")
    parser.add_argument("--num_iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = model_config_from_yaml(args.model_config)
    model = build_model(config)
    model.ensure_loaded()

    images = [_make_dummy_image() for _ in range(args.num_images)]
    question = "What is in this image?"
    choices = {"A": "cat", "B": "dog", "C": "bird", "D": "fish"} if args.dataset == "mrag_bench" else None

    if args.task == "scoring":
        prompt = format_helpfulness_prompt(dataset=args.dataset, question=question, choices=choices)
        fn = lambda: model.score_true_false_logits(query=prompt, image_paths=images)
    else:
        prompt = format_generation_prompt(dataset=args.dataset, question=question, choices=choices)
        fn = lambda: model.generate_answer(query=prompt, image_paths=images, max_new_tokens=32)

    latency = _profile_latency(fn, num_iters=args.num_iters, warmup=args.warmup)
    result = {
        "model": config.model_name,
        "family": config.family,
        "task": args.task,
        "dataset": args.dataset,
        "num_images": args.num_images,
        "latency": latency,
    }

    try:
        from calflops import calculate_flops  # type: ignore
        result["flops_supported"] = True
    except ImportError:
        result["flops_supported"] = False
        result["flops_note"] = (
            "calflops is not installed; only wall-clock latency was measured. "
            "Install with `uv sync --extra profile`."
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
