"""High-level helpfulness scorer that drives a surrogate multimodal model.

For each ``(query, candidate_image)`` pair the scorer:

1. Builds a helpfulness prompt using the configured template.
2. Runs a single-step generation through the surrogate model to obtain the
   first-step logits.
3. Extracts ``True``/``False`` token logits and returns the helpfulness score
   plus diagnostics.

The surrogate model is expected to follow the
:class:`utility_mrag.models.base.BaseMultimodalModel` interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from .prompt_templates import format_helpfulness_prompt
from .true_false_logits import TrueFalseLogitExtractor

logger = logging.getLogger(__name__)


@dataclass
class HelpfulnessRecord:
    image_id: str
    image_path: Optional[str]
    score: float
    true_logit: float
    false_logit: float
    p_true: float
    p_false: float
    metadata: Dict[str, Any]


class HelpfulnessScorer:
    """Score candidate images by surrogate-model True-token logit.

    Args:
        model: A loaded :class:`BaseMultimodalModel` instance providing
            :meth:`score_true_false_logits`.
        dataset: Either ``"mrag_bench"`` or ``"visual_rag"`` -- selects the
            prompt template used for the helpfulness query.
        template: ``"helpfulness"`` (default) or ``"relevance"``.
        score_key: Which key from the per-candidate logit dict to use as the
            ranking score. Defaults to ``"true_logit"`` (the paper's
            definition); ``"p_true"`` is also valid if you prefer the
            softmax-normalised score.
    """

    def __init__(
        self,
        model,
        dataset: str,
        template: str = "helpfulness",
        score_key: str = "true_logit",
    ) -> None:
        if score_key not in {"true_logit", "p_true"}:
            raise ValueError(f"score_key must be 'true_logit' or 'p_true', got {score_key!r}")
        self.model = model
        self.dataset = dataset
        self.template = template
        self.score_key = score_key

        # The extractor is constructed lazily when the model exposes a
        # tokenizer attribute. Concrete model wrappers return logits
        # already tied to True/False ids, so this is mainly a sanity-check.
        self._extractor: Optional[TrueFalseLogitExtractor] = None
        if hasattr(model, "tokenizer") and getattr(model, "tokenizer") is not None:
            try:
                self._extractor = TrueFalseLogitExtractor(model.tokenizer)
                for w in self._extractor.warn_if_multi_token():
                    logger.warning("[helpfulness] %s", w)
            except Exception as exc:  # noqa: BLE001 - we only want to log
                logger.warning("Could not build TrueFalseLogitExtractor: %s", exc)

    # ------------------------------------------------------------------
    def build_prompt(
        self,
        question: str,
        choices: Optional[dict | list] = None,
    ) -> str:
        return format_helpfulness_prompt(
            dataset=self.dataset,
            question=question,
            choices=choices,
            template=self.template,
        )

    # ------------------------------------------------------------------
    def score_one(
        self,
        *,
        query: str,
        candidate_image: Image.Image | str,
        question_image: Optional[Image.Image | str] = None,
        choices: Optional[dict | list] = None,
        image_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> HelpfulnessRecord:
        prompt = self.build_prompt(query, choices)

        # MRAG-Bench uses the input image + retrieved candidate; Visual-RAG
        # only uses the candidate image.
        if self.dataset == "mrag_bench":
            if question_image is None:
                raise ValueError("MRAG-Bench requires a question_image alongside the candidate.")
            images = [question_image, candidate_image]
        else:
            images = [candidate_image]

        result = self.model.score_true_false_logits(
            query=prompt,
            image_paths=images,
        )

        score = float(result.get(self.score_key, result.get("true_logit", 0.0)))
        return HelpfulnessRecord(
            image_id=image_id or str(getattr(candidate_image, "filename", "candidate")),
            image_path=str(candidate_image) if isinstance(candidate_image, str) else None,
            score=score,
            true_logit=float(result.get("true_logit", 0.0)),
            false_logit=float(result.get("false_logit", 0.0)),
            p_true=float(result.get("p_true", 0.0)),
            p_false=float(result.get("p_false", 0.0)),
            metadata=extra_metadata or {},
        )

    # ------------------------------------------------------------------
    def score_candidates(
        self,
        *,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        question_image: Optional[Image.Image | str] = None,
        choices: Optional[dict | list] = None,
    ) -> List[HelpfulnessRecord]:
        """Score a list of candidate dicts.

        Each candidate dict must have ``image_id`` and ``image_path``. Any
        additional fields are preserved in the record's ``metadata``.
        """
        records: List[HelpfulnessRecord] = []
        for cand in candidates:
            extra = {k: v for k, v in cand.items() if k not in {"image_id", "image_path"}}
            records.append(
                self.score_one(
                    query=query,
                    candidate_image=cand["image_path"],
                    question_image=question_image,
                    choices=choices,
                    image_id=cand["image_id"],
                    extra_metadata=extra,
                )
            )
        return records
