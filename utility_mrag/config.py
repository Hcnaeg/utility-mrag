"""Tiny YAML-config helper used by the CLI scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .models.base import ModelConfig


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_config_from_yaml(path: str | Path) -> ModelConfig:
    """Build a :class:`ModelConfig` from a YAML file.

    The YAML keys mirror the dataclass fields::

        family: qwen3_vl
        model_name: Qwen/Qwen3-VL-2B-Instruct
        role: surrogate
        device: cuda
        dtype: bfloat16
        extra:
          attn_implementation: flash_attention_2
    """
    payload = load_yaml(path)
    return ModelConfig(
        family=payload["family"],
        model_name=payload["model_name"],
        role=payload.get("role", "surrogate"),
        device=payload.get("device"),
        dtype=payload.get("dtype"),
        extra=payload.get("extra", {}) or {},
    )
