"""Model wrappers for the surrogate scorer and the main answer-generation model.

All wrappers conform to :class:`BaseMultimodalModel`, so the rest of the
codebase can swap models by changing a YAML config.
"""

from .base import BaseMultimodalModel, ModelConfig, build_model
from .gemma import GemmaModel
from .internvl import InternVLModel
from .minicpm import MiniCPMModel
from .ovis import OvisModel
from .qwen_vl import Qwen3VLModel

__all__ = [
    "BaseMultimodalModel",
    "ModelConfig",
    "build_model",
    "Qwen3VLModel",
    "MiniCPMModel",
    "GemmaModel",
    "OvisModel",
    "InternVLModel",
]
