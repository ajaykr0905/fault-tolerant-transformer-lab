from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _require_integer(name: str, value: int, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")


def _require_finite_number(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON field: {name}")
        value[name] = item
    return value


def _validate_fields(value: Any, config_type: type, name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} configuration must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} field names must be strings")
    unknown = sorted(value.keys() - {item.name for item in fields(config_type)})
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 64
    block_size: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_integer("vocab_size", self.vocab_size, 2)
        _require_integer("block_size", self.block_size, 2)
        _require_integer("d_model", self.d_model, 4)
        _require_integer("n_heads", self.n_heads, 1)
        _require_integer("n_layers", self.n_layers, 1)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be at least 4 and divisible by n_heads")
        _require_finite_number("dropout", self.dropout)
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 2712
    steps: int = 8
    batch_size: int = 4
    learning_rate: float = 3e-4
    checkpoint_every: int = 4

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig):
            raise ValueError("model must be a ModelConfig")
        _require_integer("seed", self.seed, 0)
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32 for NumPy seeding")
        _require_integer("steps", self.steps, 1)
        _require_integer("batch_size", self.batch_size, 1)
        _require_integer("checkpoint_every", self.checkpoint_every, 1)
        _require_finite_number("learning_rate", self.learning_rate)
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        _validate_fields(value, cls, "experiment")
        model_value = value.get("model", {})
        _validate_fields(model_value, ModelConfig, "model")
        model = ModelConfig(**model_value)
        experiment_fields = {key: item for key, item in value.items() if key != "model"}
        return cls(model=model, **experiment_fields)

    @classmethod
    def from_json(cls, source: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(source, object_pairs_hook=_unique_json_object))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
