"""Ovis 2.5 wrapper."""

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


@register_model("ovis")
class OvisModel(BaseMultimodalModel):
    """Wrapper around ``AIDC-AI/Ovis2.5-{2B,9B}``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._extractor: Optional[TrueFalseLogitExtractor] = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        # Compatibility shim: the Ovis2.5 remote modeling code was written for
        # transformers 4.x and reads ``is_parallelizable`` off its inner LLM,
        # an attribute that transformers 5.x removed from ``PreTrainedModel``.
        # Qwen3-VL requires transformers 5.x, so instead of downgrading we
        # restore the attribute as a harmless class default.
        from transformers.modeling_utils import PreTrainedModel as _PTM

        if not hasattr(_PTM, "is_parallelizable"):
            _PTM.is_parallelizable = False

        hf_token = self.config.extra.get("hf_token") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if hf_token:
            kwargs["token"] = hf_token
        self._model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **kwargs).eval()

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.cuda() if device == "cuda" and torch.cuda.is_available() else self._model.cpu()

        self._tokenizer = self._model.text_tokenizer
        self._extractor = TrueFalseLogitExtractor(self._tokenizer)
        logger.info("Loaded Ovis model %s", self.config.model_name)

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._tokenizer

    def _prepare_inputs(self, query: str, images: Sequence[Any], max_pixels: int):
        pil_images = [_to_pil(i) for i in images]
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": query})
        messages = [{"role": "user", "content": content}]
        input_ids, pixel_values, grid_thws = self._model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=True,
            max_pixels=max_pixels,
        )
        device = self._model.device
        input_ids = input_ids.to(device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(self._model.dtype)
        if grid_thws is not None:
            grid_thws = grid_thws.to(device)
        return input_ids, pixel_values, grid_thws

    def score_true_false_logits(self, *, query: str, image_paths) -> Dict[str, float]:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        max_pixels = int(self.config.extra.get("max_pixels", 896 * 896))
        input_ids, pixel_values, grid_thws = self._prepare_inputs(
            query, image_paths, max_pixels=max_pixels
        )
        with torch.inference_mode():
            generated = self._model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=1,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        return self._extractor.compute_from_scores(generated.scores)

    def generate_answer(self, *, query: str, image_paths, max_new_tokens: int = 64) -> str:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        max_pixels = int(self.config.extra.get("max_pixels", 896 * 896))
        input_ids, pixel_values, grid_thws = self._prepare_inputs(
            query, image_paths, max_pixels=max_pixels
        )
        with torch.inference_mode():
            outputs = self._model.generate(
                inputs=input_ids,
                pixel_values=pixel_values,
                grid_thws=grid_thws,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
