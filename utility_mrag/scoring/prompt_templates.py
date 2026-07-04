"""Prompt templates for helpfulness scoring and final answer generation.

The helpfulness templates ask the surrogate model to judge whether a candidate
image is useful for answering the query, with a binary True/False output space.
The True-token logit (or its softmax normalisation) is used as the helpfulness
score. The relevance variants ask a softer relevance question, used for
ablations.
"""

from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Helpfulness templates (default; used in the main paper experiments)
# ---------------------------------------------------------------------------

HELPFULNESS_PROMPT_MRAG_BENCH = """You will be given two images and a multiple-choice question.

- The first image is the input image that the question is about.
- The second image is a retrieved image intended to provide additional visual evidence.

The retrieved image does not need to answer the question by itself.
It is only meant to help answer the question together with the input image.

Question:
{question}

Choices:
{choices}

Based on the images provided, does the retrieved image provide helpful visual or factual information that could assist in answering the question correctly?

Answer with True or False."""


HELPFULNESS_PROMPT_VISUAL_RAG = """You will be given one image and a question about a visual attribute of an organism.

The image is retrieved as potential visual evidence.
Not all retrieved images contain the information needed to answer the question.

Question:
{question}

Based on the image provided, does this image contain the key visual information needed to answer the question?

Answer with True or False."""


# ---------------------------------------------------------------------------
# Relevance templates (paraphrased helpfulness variant used in ablations)
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT_MRAG_BENCH = """You will be given two images and a multiple-choice question.

- The first image is the input image that the question is about.
- The second image is a retrieved image.

Question:
{question}

Choices:
{choices}

Based on the images and the question, is the retrieved image relevant to the question?
Relevance means that the image is related to the subject, entities, or attributes mentioned
in the question, even if it does not directly help determine the correct answer.

Answer with True or False."""


RELEVANCE_PROMPT_VISUAL_RAG = """You will be given one image and a question about a visual attribute of an organism.

Question:
{question}

Based on the image and the question, is the image relevant to the question?
Relevance means that the image is related to the organism, entity, or visual attribute
mentioned in the question, even if it does not contain sufficient information
to fully answer the question.

Answer with True or False."""


# ---------------------------------------------------------------------------
# Generation templates (used by the main model on selected Top-K evidence)
# ---------------------------------------------------------------------------

GENERATION_PROMPT_MRAG_BENCH = (
    "You will be given one question concerning several images. "
    "The first image is the input image, others are retrieved examples to help you. "
    "Answer with the option's letter from the given choices directly."
)


GENERATION_PROMPT_VISUAL_RAG = (
    "Use the provided images as visual evidence to answer the following question. "
    "Provide a concise factual answer."
)


PROMPT_REGISTRY = {
    "mrag_bench": {
        "helpfulness": HELPFULNESS_PROMPT_MRAG_BENCH,
        "relevance": RELEVANCE_PROMPT_MRAG_BENCH,
        "generation": GENERATION_PROMPT_MRAG_BENCH,
    },
    "visual_rag": {
        "helpfulness": HELPFULNESS_PROMPT_VISUAL_RAG,
        "relevance": RELEVANCE_PROMPT_VISUAL_RAG,
        "generation": GENERATION_PROMPT_VISUAL_RAG,
    },
}


def format_choices(choices: dict | list | None) -> str:
    """Render a choices block. Accepts either a dict (e.g. {'A': '...', ...})
    or a list of strings (rendered as A: ..., B: ...)."""
    if choices is None:
        return ""
    if isinstance(choices, dict):
        return "\n".join(f"{k}: {v}" for k, v in choices.items())
    letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
    return "\n".join(f"{letters[i]}: {c}" for i, c in enumerate(choices))


def format_helpfulness_prompt(
    *,
    dataset: str,
    question: str,
    choices: Optional[dict | list] = None,
    template: str = "helpfulness",
) -> str:
    """Format the helpfulness prompt for a given dataset.

    Args:
        dataset: One of ``"mrag_bench"`` or ``"visual_rag"``.
        question: The textual query.
        choices: For multiple-choice datasets (MRAG-Bench), a dict or list of
            choice strings. Ignored for Visual-RAG.
        template: ``"helpfulness"`` (default) or ``"relevance"``.
    """
    if dataset not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if template not in {"helpfulness", "relevance"}:
        raise ValueError(f"Unknown template: {template!r}")
    raw = PROMPT_REGISTRY[dataset][template]
    if dataset == "mrag_bench":
        return raw.format(question=question, choices=format_choices(choices))
    return raw.format(question=question)


def format_generation_prompt(
    *,
    dataset: str,
    question: str,
    choices: Optional[dict | list] = None,
) -> str:
    """Format the final-answer generation prompt for the main model."""
    if dataset not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    base = PROMPT_REGISTRY[dataset]["generation"]
    if dataset == "mrag_bench":
        return f"{base}\n\nQuestion: {question}\nChoices:\n{format_choices(choices)}"
    return f"{base}\n\nQuestion: {question}"
