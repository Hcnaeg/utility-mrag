"""Surrogate-driven Top-K candidate selection.

Wraps a :class:`HelpfulnessScorer` and a deterministic Top-K cut into a single
end-to-end ``select`` call that consumes a candidate-pool example dict (the
JSONL schema produced by :mod:`utility_mrag.data.candidate_pool`) and returns
both the per-candidate scores and the Top-K selection for one or more ``k``
values.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence

from utility_mrag.scoring.helpfulness_score import HelpfulnessRecord, HelpfulnessScorer

from .topk import select_top_k, sort_records_descending


class SurrogateSelector:
    """Score candidates with a surrogate model and pick the Top-K.

    Image loading is delegated to the underlying model wrapper, so candidate
    paths are passed through unchanged. This keeps the selector cheap and
    side-effect free for unit tests / mock models.
    """

    def __init__(self, scorer: HelpfulnessScorer) -> None:
        self.scorer = scorer

    @property
    def dataset(self) -> str:
        return self.scorer.dataset

    def score_example(self, example: Dict[str, Any]) -> List[HelpfulnessRecord]:
        question_image_path: Any = None
        if self.dataset == "mrag_bench":
            question_image_path = example.get("question_image_path") or example.get("question_image")
            if question_image_path is None:
                raise ValueError(
                    f"MRAG-Bench example {example.get('qid')!r} missing question_image_path"
                )

        candidates = example.get("candidate_images", [])
        records: List[HelpfulnessRecord] = []
        for cand in candidates:
            img_path = cand.get("image_path")
            if img_path is None:
                raise ValueError(f"Candidate missing image_path: {cand}")
            extra = {k: v for k, v in cand.items() if k not in {"image_id", "image_path"}}
            rec = self.scorer.score_one(
                query=example["query"],
                candidate_image=img_path,
                question_image=question_image_path,
                choices=example.get("choices"),
                image_id=cand.get("image_id", img_path),
                extra_metadata=extra,
            )
            rec.metadata.setdefault("image_path", img_path)
            records.append(rec)
        return records

    @staticmethod
    def _record_to_payload(rec: HelpfulnessRecord) -> Dict[str, Any]:
        d = asdict(rec)
        # Pull image_path back to the top-level for convenience.
        d["image_path"] = d.get("image_path") or d["metadata"].get("image_path")
        return d

    def select(
        self,
        example: Dict[str, Any],
        *,
        top_ks: Sequence[int] = (1, 3, 5),
    ) -> Dict[str, Any]:
        """Score candidates and return per-k selections.

        Returns a dict shaped::

            {
                "qid": ...,
                "query": ...,
                "all_scores": [ ... ],          # full ranking
                "selections": {
                    "1": [...top1...],
                    "3": [...top3...],
                    "5": [...top5...],
                },
            }
        """
        records = self.score_example(example)
        payloads = [self._record_to_payload(r) for r in records]
        sorted_records = sort_records_descending(payloads, score_key="score")

        selections: Dict[str, List[Dict[str, Any]]] = {}
        for k in top_ks:
            selections[str(int(k))] = select_top_k(payloads, int(k), score_key="score")

        return {
            "qid": example.get("qid"),
            "query": example.get("query"),
            "all_scores": sorted_records,
            "selections": selections,
            "metadata": example.get("metadata", {}),
            "answer": example.get("answer"),
            "choices": example.get("choices"),
            "gt_image_ids": example.get("gt_image_ids", []),
        }
