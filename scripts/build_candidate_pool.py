#!/usr/bin/env python
"""Convenience wrapper that dispatches to the per-dataset builder script.

Usage::

    uv run python scripts/build_candidate_pool.py \\
        --dataset mrag_bench \\
        --input_dir ... --retrieval_file ... --output ...
"""

from __future__ import annotations

import argparse
import sys
from typing import List

from . import prepare_mrag_bench, prepare_visual_rag


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--dataset", choices=["mrag_bench", "visual_rag"], required=True)
    args, rest = parser.parse_known_args(argv)
    if args.dataset == "mrag_bench":
        return prepare_mrag_bench.main(rest)
    return prepare_visual_rag.main(rest)


if __name__ == "__main__":
    sys.exit(main())
