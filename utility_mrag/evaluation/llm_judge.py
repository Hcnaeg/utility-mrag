"""Optional LLM-as-judge for Visual-RAG free-form answers.

The judge issues a single OpenAI chat-completions request per (question,
prediction, target) triple and asks the judge model whether the prediction is
correct. The OpenAI API key **must** come from the ``OPENAI_API_KEY``
environment variable; it is never read from a file or hard-coded.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict


_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for an open-ended visual question answering benchmark.

You will be shown:
- a question,
- the ground-truth answer,
- a model's prediction.

Decide whether the model's prediction is semantically correct with respect to
the ground-truth. Minor wording differences are acceptable; the prediction is
correct if it conveys the same factual answer.

Respond with strict JSON of the form
{{"correct": true_or_false, "reason": "<one short sentence>"}}.

Question: {question}
Ground truth: {target}
Prediction: {prediction}
"""


@dataclass
class LLMJudge:
    """Wraps the OpenAI client for the Visual-RAG correctness check."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 128

    def __post_init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLMJudge requires the OPENAI_API_KEY environment variable. "
                "Set it before invoking the judge; never hard-code keys."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "LLMJudge requires the `openai` package. Install with "
                "`uv sync --extra eval`."
            ) from exc
        self._client = OpenAI(api_key=api_key)

    def judge(self, *, question: str, prediction: str, target: str) -> Dict[str, Any]:
        prompt = _JUDGE_PROMPT_TEMPLATE.format(
            question=question, target=target, prediction=prediction
        )
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = completion.choices[0].message.content or ""
        return _parse_judge_response(text)


def _parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse a strict JSON judge response, falling back to regex on noise."""
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if not match:
            return {"correct": False, "reason": "unparsable judge response", "raw": text}
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"correct": False, "reason": "unparsable judge response", "raw": text}
    return {
        "correct": bool(obj.get("correct", False)),
        "reason": str(obj.get("reason", "")),
        "raw": text,
    }
