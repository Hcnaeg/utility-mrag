"""InternVL 3.5 wrapper.

Uses the patched chat path from the original repo when available; otherwise
falls back to a vanilla ``chat`` call. For True/False logit extraction we drive
``model.generate`` directly so the same :class:`TrueFalseLogitExtractor`
applies.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from utility_mrag.scoring.true_false_logits import TrueFalseLogitExtractor

from .base import BaseMultimodalModel, ModelConfig, register_model

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_pil(img: Any) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise TypeError(f"Cannot coerce {type(img).__name__} to PIL.Image")


def _build_transform(image_size: int = 448):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _preprocess_images(
    images: Sequence[Image.Image],
    *,
    image_size: int,
    device,
    dtype,
) -> Tuple[Any, List[int]]:
    import torch

    if not images:
        return None, []
    transform = _build_transform(image_size)
    tensors, num_patches = [], []
    for img in images:
        x = transform(img.convert("RGB")).unsqueeze(0)
        tensors.append(x)
        num_patches.append(1)
    pixel_values = torch.cat(tensors, dim=0).to(device=device, dtype=dtype)
    return pixel_values, num_patches


@register_model("internvl")
class InternVLModel(BaseMultimodalModel):
    """Wrapper around ``OpenGVLab/InternVL3_5-*``."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._model = None
        self._tokenizer = None
        self._extractor: Optional[TrueFalseLogitExtractor] = None

    def load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self._model = AutoModel.from_pretrained(
            self.config.model_name,
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.config.extra.get("device_map", "auto"),
        ).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, use_fast=False
        )
        self._extractor = TrueFalseLogitExtractor(self._tokenizer)
        logger.info("Loaded InternVL model %s", self.config.model_name)

    @property
    def tokenizer(self):
        self.ensure_loaded()
        return self._tokenizer

    def _build_question(self, query: str, n_images: int) -> str:
        if "<image>" in query:
            return query
        return "".join("<image>\n" for _ in range(n_images)) + query

    def _prepare(self, query: str, images: Sequence[Any]):
        import torch

        pil_images = [_to_pil(i) for i in images]
        device = self._model.device
        dtype = next(self._model.parameters()).dtype
        pixel_values, num_patches_list = _preprocess_images(
            pil_images,
            image_size=int(self.config.extra.get("image_size", 448)),
            device=device,
            dtype=dtype,
        )
        question = self._build_question(query, len(pil_images))
        return pixel_values, num_patches_list, question

    def score_true_false_logits(self, *, query: str, image_paths) -> Dict[str, float]:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        pixel_values, num_patches_list, question = self._prepare(query, image_paths)
        with torch.inference_mode():
            response = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={
                    "max_new_tokens": 1,
                    "do_sample": False,
                    "output_scores": True,
                    "return_dict_in_generate": True,
                },
                num_patches_list=num_patches_list,
                return_full_output=True,
            )
        scores = getattr(response, "scores", None)
        if scores is None and isinstance(response, dict):
            scores = response.get("scores")
        if scores is None:
            raise RuntimeError(
                "InternVL chat() did not return generation scores. The wrapper "
                "needs the patched chat method that exposes output_scores."
            )
        return self._extractor.compute_from_scores(scores)

    def generate_answer(self, *, query: str, image_paths, max_new_tokens: int = 64) -> str:
        import torch

        self.ensure_loaded()
        if not isinstance(image_paths, (list, tuple)):
            image_paths = [image_paths]
        pixel_values, num_patches_list, question = self._prepare(query, image_paths)
        with torch.inference_mode():
            response = self._model.chat(
                tokenizer=self._tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
                num_patches_list=num_patches_list,
                return_full_output=False,
            )
        if isinstance(response, str):
            return response.strip()
        return str(response).strip()
