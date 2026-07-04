"""Visual-RAG evaluation.

Visual-RAG questions are open-ended factual queries about an organism's visual
attributes. Following the original release, evaluation can be done either with
exact-match / F1 style string metrics (offline, deterministic) or with an
LLM-as-judge (online, requires an OpenAI key). The judge is optional and
always opt-in via ``--use_llm_judge``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .llm_judge import LLMJudge
from .metrics import exact_match, f1_score


def score_visual_rag(
    records: Iterable[Dict[str, Any]],
    *,
    pred_key: str = "output",
    gt_key: str = "answer",
    use_llm_judge: bool = False,
    judge_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Score Visual-RAG predictions.

    Args:
        records: Iterable of dicts with at least ``pred_key`` and ``gt_key``.
        use_llm_judge: When ``True``, additionally calls :class:`LLMJudge`.
        judge_model: OpenAI model id for the judge.

    Returns:
        Dict with ``exact_match``, ``f1``, ``num_examples`` (and ``llm_judge``
        if ``use_llm_judge=True``).
    """
    records = list(records)
    em_scores: List[int] = []
    f1_scores: List[float] = []
    judge_records: List[Dict[str, Any]] = []

    judge: LLMJudge | None = None
    if use_llm_judge:
        judge = LLMJudge(model=judge_model)

    for r in records:
        pred = str(r.get(pred_key, ""))
        gt = str(r.get(gt_key, ""))
        em_scores.append(exact_match(pred, gt))
        f1_scores.append(f1_score(pred, gt))
        if judge is not None:
            verdict = judge.judge(question=str(r.get("query", "")), prediction=pred, target=gt)
            judge_records.append(verdict)

    out: Dict[str, Any] = {
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "num_examples": len(records),
    }
    if judge is not None:
        out["llm_judge"] = {
            "accuracy": sum(1 for r in judge_records if r.get("correct")) / max(1, len(judge_records)),
            "raw": judge_records,
        }
    return out
