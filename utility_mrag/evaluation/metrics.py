"""Generic metric helpers shared by MRAG-Bench and Visual-RAG evaluators."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence


def accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Exact-match accuracy after stripping whitespace."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(targets)} targets"
        )
    if not predictions:
        return 0.0
    correct = sum(1 for p, t in zip(predictions, targets) if str(p).strip() == str(t).strip())
    return correct / len(predictions)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, target: str) -> int:
    """Squad-style normalised exact match."""
    return int(_normalize(prediction) == _normalize(target))


def f1_score(prediction: str, target: str) -> float:
    """Token-level F1 between two strings (Squad-style)."""
    pred_tokens = _normalize(prediction).split()
    target_tokens = _normalize(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(target_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def aggregate_by(
    predictions: Iterable[str],
    targets: Iterable[str],
    groups: Iterable[str],
) -> dict:
    """Group exact-match accuracy by ``groups`` (e.g. MRAG-Bench scenarios)."""
    buckets: dict[str, List[int]] = {}
    for pred, tgt, grp in zip(predictions, targets, groups):
        buckets.setdefault(grp, []).append(int(str(pred).strip() == str(tgt).strip()))
    return {grp: (sum(vals) / len(vals) if vals else 0.0) for grp, vals in buckets.items()}
