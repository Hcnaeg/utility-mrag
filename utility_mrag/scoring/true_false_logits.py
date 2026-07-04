"""Extract first-step True/False token logits from a multimodal model.

The helpfulness score is the final-layer logit of the ``"True"`` token at the
first generated step. Tokenizers vary in how they encode ``"True"``/``"False"``
(with or without a leading space, and into one or several pieces). This module
resolves robust token ids for both choices, supports a fallback for
multi-token cases, and returns both raw logits plus the softmax-normalised
binary probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class _TokenSpec:
    """Resolved token-id information for a single positive/negative choice."""

    token_str: str
    token_ids: tuple[int, ...]
    is_single_token: bool

    @property
    def primary_id(self) -> int:
        return self.token_ids[0]


def _to_id_list(value: Any) -> List[int]:
    """Coerce tokenizer encoding output to a Python list of ints."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif not isinstance(value, list):
        value = list(value)
    return [int(v) for v in value]


def _resolve_token_ids(tokenizer, token_str: str) -> _TokenSpec:
    """Encode ``token_str`` robustly, trying with and without a leading space.

    Returns a ``_TokenSpec`` whose ``token_ids`` is the encoding that the model
    is most likely to actually produce as the first generated piece. Preference
    order:

    1. The raw encoding if it is a single token.
    2. The leading-space encoding if it is a single token.
    3. Otherwise the raw encoding (multi-piece; we will fall back to its first
       id) -- documented behaviour, see :func:`extract_true_false_logits`.
    """
    raw = _to_id_list(tokenizer.encode(token_str, add_special_tokens=False))
    leading = _to_id_list(tokenizer.encode(" " + token_str, add_special_tokens=False))

    if len(raw) == 1:
        return _TokenSpec(token_str, tuple(raw), True)
    if len(leading) == 1:
        return _TokenSpec(token_str, tuple(leading), True)
    if raw:
        return _TokenSpec(token_str, tuple(raw), False)
    if leading:
        return _TokenSpec(token_str, tuple(leading), False)
    raise ValueError(f"Tokenizer cannot encode token {token_str!r}")


def extract_true_false_logits(
    *,
    first_step_logits: torch.Tensor,
    tokenizer,
    positive_token: str = "True",
    negative_token: str = "False",
) -> Dict[str, Any]:
    """Compute True/False logit-based helpfulness metrics.

    Args:
        first_step_logits: 1-D tensor of shape ``[vocab_size]`` (or 2-D
            ``[batch, vocab]`` -- in which case row 0 is used) holding the
            unnormalised logits for the **first** generated token.
        tokenizer: Hugging Face tokenizer used by the multimodal model.
        positive_token: Token whose logit is interpreted as helpful.
        negative_token: Token whose logit is interpreted as not-helpful.

    Returns:
        A dict with keys::

            true_logit, false_logit          -- raw logits (Python floats)
            p_true, p_false                  -- softmax over [pos, neg] only
            true_token_id, false_token_id    -- primary token ids used
            true_is_single_token,
            false_is_single_token            -- whether each token was a clean
                                                single-piece encoding (False
                                                indicates a multi-token
                                                fallback was used)
            score                            -- the canonical helpfulness
                                                score = true_logit (the paper's
                                                default)

    The helpfulness score returned in ``score`` matches the paper's definition.
    Downstream code may instead read ``p_true`` if it prefers the
    softmax-normalised binary probability.
    """
    if first_step_logits is None:
        raise ValueError("first_step_logits is None")
    logits = first_step_logits
    if logits.dim() == 2:
        logits = logits[0]
    if logits.dim() != 1:
        raise ValueError(
            f"first_step_logits must be 1-D or 2-D, got shape {tuple(logits.shape)}"
        )

    pos_spec = _resolve_token_ids(tokenizer, positive_token)
    neg_spec = _resolve_token_ids(tokenizer, negative_token)

    true_logit = float(logits[pos_spec.primary_id].item())
    false_logit = float(logits[neg_spec.primary_id].item())

    # Softmax restricted to the positive/negative pair.
    pair = torch.tensor([true_logit, false_logit], dtype=torch.float32)
    pair_probs = F.softmax(pair, dim=-1)

    return {
        "true_logit": true_logit,
        "false_logit": false_logit,
        "p_true": float(pair_probs[0].item()),
        "p_false": float(pair_probs[1].item()),
        "true_token_id": int(pos_spec.primary_id),
        "false_token_id": int(neg_spec.primary_id),
        "true_is_single_token": bool(pos_spec.is_single_token),
        "false_is_single_token": bool(neg_spec.is_single_token),
        "score": true_logit,
    }


class TrueFalseLogitExtractor:
    """Stateful helper bound to a (tokenizer, token-pair) pair.

    Caches the resolved token ids so repeated calls (one per candidate image)
    skip re-encoding. Use :meth:`compute_from_scores` with the ``scores``
    tuple returned by ``model.generate(..., output_scores=True,
    return_dict_in_generate=True)``.
    """

    def __init__(
        self,
        tokenizer,
        positive_token: str = "True",
        negative_token: str = "False",
    ) -> None:
        self.tokenizer = tokenizer
        self.positive_token = positive_token
        self.negative_token = negative_token
        self._pos_spec = _resolve_token_ids(tokenizer, positive_token)
        self._neg_spec = _resolve_token_ids(tokenizer, negative_token)

    @property
    def positive_token_id(self) -> int:
        return self._pos_spec.primary_id

    @property
    def negative_token_id(self) -> int:
        return self._neg_spec.primary_id

    def warn_if_multi_token(self) -> List[str]:
        """Return human-readable warnings about multi-token fallbacks."""
        msgs: List[str] = []
        if not self._pos_spec.is_single_token:
            msgs.append(
                f"positive token {self.positive_token!r} is multi-piece "
                f"({self._pos_spec.token_ids}); using first-piece logit fallback"
            )
        if not self._neg_spec.is_single_token:
            msgs.append(
                f"negative token {self.negative_token!r} is multi-piece "
                f"({self._neg_spec.token_ids}); using first-piece logit fallback"
            )
        return msgs

    def compute(self, first_step_logits: torch.Tensor) -> Dict[str, Any]:
        return extract_true_false_logits(
            first_step_logits=first_step_logits,
            tokenizer=self.tokenizer,
            positive_token=self.positive_token,
            negative_token=self.negative_token,
        )

    def compute_from_scores(
        self,
        scores: Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        """Convenience: extract from generation ``scores[0]``."""
        if not scores:
            raise ValueError("scores is empty")
        return self.compute(scores[0])

    def compute_batch(self, batched_first_step_logits: Iterable[torch.Tensor]) -> List[Dict[str, Any]]:
        return [self.compute(x) for x in batched_first_step_logits]
