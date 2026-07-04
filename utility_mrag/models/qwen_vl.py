"""Qwen3-VL wrapper: surrogate scoring + main-model answer generation."""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Sequence

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


def _resolve_dtype(dtype: Optional[str]):
    import torch

    if dtype is None or dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


@register_model("qwen3_vl")
class Qwen3VLModel(BaseMultimodalModel):
    """Wrapper around `Qwen3VLForConditionalGeneration`.

    Suitable for both surrogate scoring (use the 2B/4B model) and final answer
    generation (use the 8B / FP8 variant).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._processor = None
        self._extractor: Optional[TrueFalseLogitExtractor] = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        torch_dtype = _resolve_dtype(self.config.dtype)
        kwargs: Dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": self.config.extra.get("device_map", "auto"),
        }
        if self.config.extra.get("attn_implementation"):
            kwargs["attn_implementation"] = self.config.extra["attn_implementation"]
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.config.model_name, **kwargs
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.extra.get("processor_name", self.config.model_name)
        )
        self._extractor = TrueFalseLogitExtractor(self._processor.tokenizer)
        logger.info("Loaded Qwen3-VL model %s", self.config.model_name)

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._processor.tokenizer

    # ------------------------------------------------------------------
    def _build_messages(
        self,
        query: str,
        images: Sequence[Image.Image],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": query})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, query: str, images: Sequence[Any]):
        pil_images = [_to_pil(i) for i in images]
        messages = self._build_messages(query, pil_images)
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self._model.device)

    # ------------------------------------------------------------------
    def score_true_false_logits(
        self,
        *,
        query: str,
        image_paths,
    ) -> Dict[str, float]:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        inputs = self._prepare_inputs(query, image_paths)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )
        return self._extractor.compute_from_scores(generated.scores)

    # ------------------------------------------------------------------
    def generate_answer(
        self,
        *,
        query: str,
        image_paths,
        max_new_tokens: int = 64,
    ) -> str:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        inputs = self._prepare_inputs(query, image_paths)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._processor.tokenizer.eos_token_id,
            )
        trimmed = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
        ]
        text = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return text.strip()
