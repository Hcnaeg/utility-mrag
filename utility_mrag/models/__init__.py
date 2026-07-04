"""Model wrappers for the surrogate scorer and the main answer-generation model.

All wrappers conform to :class:`BaseMultimodalModel`, so the rest of the
codebase can swap models by changing a YAML config.

Each concrete wrapper is imported defensively: a wrapper that needs an optional
third-party dependency (e.g. ``torchvision`` for InternVL) is skipped with a
warning if that dependency is missing, rather than breaking ``import
utility_mrag.models`` for everyone else. The wrapper only becomes available via
:func:`build_model` once its dependencies are installed.
"""

from __future__ import annotations

import logging

from .base import BaseMultimodalModel, ModelConfig, build_model, register_model

logger = logging.getLogger(__name__)

__all__ = [
    "BaseMultimodalModel",
    "ModelConfig",
    "build_model",
    "register_model",
]

# (attribute name, module, class) for every built-in wrapper. Importing the
# module runs its @register_model decorator and makes the family available to
# build_model().
_WRAPPERS = [
    ("Qwen3VLModel", ".qwen_vl", "Qwen3VLModel"),
    ("MiniCPMModel", ".minicpm", "MiniCPMModel"),
    ("GemmaModel", ".gemma", "GemmaModel"),
    ("OvisModel", ".ovis", "OvisModel"),
    ("InternVLModel", ".internvl", "InternVLModel"),
]

for _attr, _module, _cls in _WRAPPERS:
    try:
        _mod = __import__(f"{__name__}{_module}", fromlist=[_cls])
        globals()[_attr] = getattr(_mod, _cls)
        __all__.append(_attr)
    except Exception as exc:  # noqa: BLE001 - optional deps may be absent
        logger.warning(
            "Model wrapper %s is unavailable (%s: %s). Install its extra "
            "dependencies to enable it.",
            _attr,
            type(exc).__name__,
            exc,
        )
