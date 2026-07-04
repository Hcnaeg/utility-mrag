"""Gemma 3 wrapper using the `Gemma3ForConditionalGeneration` API."""

from __future__ import annotations

import io
import logging
import os
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


@register_model("gemma")
class GemmaModel(BaseMultimodalModel):
    """Wrapper around `Gemma3ForConditionalGeneration` (e.g. ``google/gemma-3-12b-it``)."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None
        self._extractor: Optional[TrueFalseLogitExtractor] = None

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        hf_token = self.config.extra.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        kwargs: Dict[str, Any] = {
            "device_map": self.config.extra.get("device_map", "auto"),
            "torch_dtype": torch.bfloat16,
        }
        if hf_token:
            kwargs["token"] = hf_token
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.config.model_name, **kwargs
        ).eval()
        proc_kwargs: Dict[str, Any] = {}
        if hf_token:
            proc_kwargs["token"] = hf_token
        self._processor = AutoProcessor.from_pretrained(self.config.model_name, **proc_kwargs)
        self._extractor = TrueFalseLogitExtractor(self._processor.tokenizer)
        logger.info("Loaded Gemma 3 model %s", self.config.model_name)

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._processor.tokenizer

    def _prepare_inputs(self, query: str, images: Sequence[Any]):
        import torch

        pil_images = [_to_pil(i) for i in images]
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)
        return inputs

    def score_true_false_logits(self, *, query: str, image_paths) -> Dict[str, float]:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        inputs = self._prepare_inputs(query, image_paths)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return self._extractor.compute_from_scores(generated.scores)

    def generate_answer(self, *, query: str, image_paths, max_new_tokens: int = 64) -> str:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        inputs = self._prepare_inputs(query, image_paths)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
