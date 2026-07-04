"""MiniCPM-V 4.5 wrapper.

For free-form generation we call the model's ``chat`` API as recommended by
the MiniCPM authors. For True/False logit extraction we drive
``model.generate`` directly with ``output_scores=True`` so the same
:class:`TrueFalseLogitExtractor` used by the other wrappers can be reused.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional, Sequence

from PIL import Image

from utility_mrag.scoring.true_false_logits import TrueFalseLogitExtractor

from .base import BaseMultimodalModel, ModelConfig, register_model

logger = logging.getLogger(__name__)


def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise TypeError(f"Cannot coerce {type(img).__name__} to PIL.Image")


@register_model("minicpm")
class MiniCPMModel(BaseMultimodalModel):
    """Wrapper around ``openbmb/MiniCPM-V-4_5`` (and AWQ variants)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._extractor: Optional[TrueFalseLogitExtractor] = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        name_lower = self.config.model_name.lower()
        is_awq = "awq" in name_lower
        if is_awq:
            from awq import AutoAWQForCausalLM

            self._model = AutoAWQForCausalLM.from_quantized(
                self.config.model_name, trust_remote_code=True
            ).to(self.config.device or "cuda")
        else:
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                attn_implementation=self.config.extra.get("attn_implementation", "sdpa"),
                torch_dtype=torch.bfloat16,
            ).eval().to(self.config.device or "cuda")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        self._extractor = TrueFalseLogitExtractor(self._tokenizer)
        logger.info("Loaded MiniCPM model %s", self.config.model_name)

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._tokenizer

    def _build_msgs(self, query: str, images: Sequence[Any]):
        pil_images = [_to_pil(i) for i in images]
        return [{"role": "user", "content": pil_images + [query]}]

    def score_true_false_logits(self, *, query: str, image_paths) -> Dict[str, float]:
        """Run the model's chat path with ``return_vision_hidden_states`` -> not
        directly available; we instead invoke ``model.chat`` with
        ``max_new_tokens=1`` and capture ``scores``.
        """
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        msgs = self._build_msgs(query, image_paths)
        with torch.inference_mode():
            # MiniCPM's chat() supports a `generation_config`-style hook; we
            # request the first-step scores via the underlying generate kwargs.
            response = self._model.chat(
                msgs=msgs,
                tokenizer=self._tokenizer,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        # MiniCPM returns either a string or an object with `.scores` depending
        # on the patched chat method. Handle both.
        scores = getattr(response, "scores", None)
        if scores is None and isinstance(response, dict):
            scores = response.get("scores")
        if scores is None:
            raise RuntimeError(
                "MiniCPM chat() did not return generation scores. The wrapper "
                "needs the patched chat method that surfaces output_scores."
            )
        return self._extractor.compute_from_scores(scores)

    def generate_answer(self, *, query: str, image_paths, max_new_tokens: int = 64) -> str:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        msgs = self._build_msgs(query, image_paths)
        with torch.inference_mode():
            response = self._model.chat(
                msgs=msgs,
                tokenizer=self._tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        if isinstance(response, str):
            return response.strip()
        if isinstance(response, dict) and "text" in response:
            return str(response["text"]).strip()
        return str(response).strip()
