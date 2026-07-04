"""Candidate-pool construction and dataset loaders."""

from .candidate_pool import (
    CandidateImage,
    CandidatePoolExample,
    load_candidate_pool,
    write_candidate_pool,
)
from .dataset_loaders import iter_mrag_bench, iter_visual_rag

__all__ = [
    "CandidateImage",
    "CandidatePoolExample",
    "load_candidate_pool",
    "write_candidate_pool",
    "iter_mrag_bench",
    "iter_visual_rag",
]
