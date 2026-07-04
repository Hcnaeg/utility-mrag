"""MRAG-Bench scoring (multiple-choice).

Ports the MMMU-style answer-extraction logic from the original MRAG-Bench
``eval/score.py``, then computes overall and per-scenario accuracy.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

# Deterministic seed mirrors the original MRAG-Bench implementation; we keep it
# scoped to a local Random instance to avoid leaking into the global state.
_RNG = random.Random(42)


def parse_multi_choice_response(
    response: str,
    all_choices: Sequence[str] = ("A", "B", "C", "D"),
    index2ans: Optional[Dict[str, str]] = None,
) -> str:
    """Extract a multiple-choice answer letter from a free-form response.

    Adapted from MMMU/MRAG-Bench's ``parse_multi_choice_response``.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates: List[str] = []
    ans_with_brack = False
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    index_ans = True
    if not candidates and index2ans and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        return response.strip()
    if len(candidates) == 1:
        return candidates[0]

    starts: List[int] = []
    if index_ans:
        target = (lambda c: response.rfind(f"({c})")) if ans_with_brack else (
            lambda c: response.rfind(f" {c} ")
        )
        for c in candidates:
            starts.append(target(c))
    else:
        for c in candidates:
            starts.append(response.lower().rfind(index2ans[c].lower()))

    best_idx = starts.index(max(starts))
    return candidates[best_idx]


def score_mrag_bench(
    records: Iterable[Dict[str, Any]],
    *,
    pred_key: str = "output",
    gt_key: str = "gt_choice",
    scenario_key: str = "scenario",
) -> Dict[str, Any]:
    """Compute overall + per-scenario accuracy for MRAG-Bench predictions.

    Args:
        records: Iterable of dicts. Each dict must have:
          - ``output``: the raw model response (string)
          - ``gt_choice``: the ground-truth letter (``"A"``..``"D"``)
        pred_key, gt_key, scenario_key: override field names if needed.

    Returns:
        Dict with ``overall_accuracy``, ``per_scenario`` (dict),
        ``num_examples``, and ``unparsed`` count.
    """
    records = list(records)
    all_choices = ("A", "B", "C", "D")
    preds: List[str] = []
    gts: List[str] = []
    scenarios: List[str] = []
    unparsed = 0

    for item in records:
        gt = item.get(gt_key)
        out = str(item.get(pred_key, "")).strip()
        index2ans = {
            "A": item.get("A", ""),
            "B": item.get("B", ""),
            "C": item.get("C", ""),
            "D": item.get("D", ""),
        }
        parsed = parse_multi_choice_response(out, all_choices, index2ans)
        if parsed not in all_choices:
            unparsed += 1
            parsed = _RNG.choice(all_choices)
        preds.append(parsed)
        gts.append(gt)
        scenarios.append(str(item.get(scenario_key, "Unknown")))

    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    overall = correct / len(preds) if preds else 0.0

    by_scenario: Dict[str, float] = {}
    seen = OrderedDict()
    for s in scenarios:
        seen[s] = None
    for scen in seen:
        sub = [(p, g) for p, g, s in zip(preds, gts, scenarios) if s == scen]
        if not sub:
            by_scenario[scen] = 0.0
        else:
            by_scenario[scen] = sum(1 for p, g in sub if p == g) / len(sub)

    return {
        "overall_accuracy": overall,
        "per_scenario": by_scenario,
        "num_examples": len(preds),
        "unparsed": unparsed,
    }
