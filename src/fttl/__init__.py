"""Fault-tolerant transformer training lab."""

from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer

__version__ = "0.2.0"

__all__ = ["ExperimentConfig", "ModelConfig", "TinyTransformer", "__version__"]
