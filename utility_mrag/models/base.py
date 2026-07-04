"""Base interface for multimodal models used in utility-mrag.

A wrapper provides three things to the rest of the pipeline:

1. ``score_true_false_logits(query, image_paths)`` -- return a dict with
   ``true_logit`` / ``false_logit`` / ``p_true`` / ``p_false`` for the first
   generated token.
2. ``generate_answer(query, image_paths, max_new_tokens)`` -- free-form text
   generation used by the main model on Top-K selected evidence.
3. ``format_helpfulness_input(query, image_path)`` (optional) -- a hook that
   subclasses may override if their internal prompt format diverges from the
   plain ``query + images`` convention.

Concrete wrappers live in :mod:`utility_mrag.models.qwen_vl`,
:mod:`.minicpm`, :mod:`.gemma`, :mod:`.ovis`, and :mod:`.internvl`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

ImageLike = Any  # PIL.Image.Image or filesystem path; resolved per-wrapper.


@dataclass
class ModelConfig:
    """Lightweight, YAML-friendly model description."""

    family: str  # "qwen3_vl" | "minicpm" | "gemma" | "ovis" | "internvl"
    model_name: str  # Hugging Face id or local path
    role: str = "surrogate"  # "surrogate" | "main"
    device: Optional[str] = None
    dtype: Optional[str] = None  # "bfloat16" | "float16" | "float32" | "auto"
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseMultimodalModel(ABC):
    """Abstract interface every model wrapper must implement."""

    family: str = "base"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def load(self) -> None:
        """Load weights, processor, and tokenizer onto the configured device."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    @property
    @abstractmethod
    def tokenizer(self) -> Any:
        """Return the tokenizer (used for True/False token-id resolution)."""

    # ------------------------------------------------------------------
    # Prompt formatting hook (optional override)
    # ------------------------------------------------------------------
    def format_helpfulness_input(
        self,
        query: str,
        image_path: ImageLike | Sequence[ImageLike],
    ) -> Dict[str, Any]:
        """Default hook: just package the query + images.

        Wrappers that need a model-specific chat template can override this.
        """
        if isinstance(image_path, (list, tuple)):
            images = list(image_path)
        else:
            images = [image_path]
        return {"query": query, "images": images}

    # ------------------------------------------------------------------
    # Required scoring + generation
    # ------------------------------------------------------------------
    @abstractmethod
    def score_true_false_logits(
        self,
        *,
        query: str,
        image_paths: Sequence[ImageLike] | ImageLike,
    ) -> Dict[str, float]:
        """Run a single-step forward pass and return True/False logits.

        Must return a dict with at minimum::

            {"true_logit": float, "false_logit": float,
             "p_true": float,    "p_false": float}
        """

    @abstractmethod
    def generate_answer(
        self,
        *,
        query: str,
        image_paths: Sequence[ImageLike] | ImageLike,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate a free-form answer using the supplied images."""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, type] = {}


def register_model(family: str):
    """Decorator that registers a wrapper under ``family``."""

    def _wrap(cls: type) -> type:
        _REGISTRY[family] = cls
        cls.family = family
        return cls

    return _wrap


def build_model(config: Union[ModelConfig, Dict[str, Any]]) -> BaseMultimodalModel:
    """Instantiate a wrapper from a :class:`ModelConfig` or plain dict."""
    if isinstance(config, dict):
        config = ModelConfig(**config)
    if config.family not in _REGISTRY:
        raise KeyError(
            f"Unknown model family {config.family!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[config.family](config)
