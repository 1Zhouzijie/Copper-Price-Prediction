"""Copper interval price prediction package."""

from typing import TYPE_CHECKING

from .config import (
    CNN_WINDOW,
    CORR_WINDOW,
    NODE_FEATURE_DIM,
    GraphSpec,
    NodeSpec,
    default_graph_specs,
)

if TYPE_CHECKING:
    from .model import CopperIntervalPredictor, CopperIntervalPredictorConfig

__all__ = [
    "CNN_WINDOW",
    "CORR_WINDOW",
    "NODE_FEATURE_DIM",
    "GraphSpec",
    "NodeSpec",
    "default_graph_specs",
    "CopperIntervalPredictor",
    "CopperIntervalPredictorConfig",
]


def __getattr__(name: str):
    if name in {"CopperIntervalPredictor", "CopperIntervalPredictorConfig"}:
        from .model import CopperIntervalPredictor, CopperIntervalPredictorConfig

        return {
            "CopperIntervalPredictor": CopperIntervalPredictor,
            "CopperIntervalPredictorConfig": CopperIntervalPredictorConfig,
        }[name]
    raise AttributeError(name)
