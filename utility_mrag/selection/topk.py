"""Deterministic Top-K selection over scored candidates.

Records are sorted by ``score`` descending. Ties are broken deterministically
using the candidate's stable ``tie_breaker`` field (default: original index)
so that re-running selection on the same scores yields identical output.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _to_dict(record: Any) -> Dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"Cannot coerce record of type {type(record).__name__} to dict")


def sort_records_descending(
    records: Iterable[Any],
    *,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Return records sorted by ``score_key`` descending.

    Ties are broken by the original input index, so the sort is stable and
    deterministic. Each output is a plain dict with an added ``rank`` (1-based)
    and ``tie_breaker`` field (the original index).
    """
    indexed: List[tuple[int, Dict[str, Any]]] = []
    for idx, rec in enumerate(records):
        d = _to_dict(rec)
        if score_key not in d:
            raise KeyError(f"record is missing score key {score_key!r}: {d}")
        indexed.append((idx, d))

    indexed.sort(key=lambda item: (-float(item[1][score_key]), item[0]))

    out: List[Dict[str, Any]] = []
    for rank, (orig_idx, d) in enumerate(indexed, start=1):
        d = dict(d)
        d["rank"] = rank
        d["tie_breaker"] = orig_idx
        out.append(d)
    return out


def select_top_k(
    records: Sequence[Any],
    k: int,
    *,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Return the deterministic Top-K records.

    Args:
        records: Iterable of dicts or dataclasses each containing ``score_key``.
        k: Number of records to return. ``k <= 0`` returns ``[]``;
            ``k >= len(records)`` returns all records, sorted.
        score_key: Field name to rank by.
    """
    if k <= 0:
        return []
    sorted_records = sort_records_descending(records, score_key=score_key)
    return sorted_records[:k]
