from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 64
    block_size: int = 16
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2")
        if self.d_model < 4 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be at least 4 and divisible by n_heads")
        if self.n_layers < 1:
            raise ValueError("n_layers must be at least 1")
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
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        model = ModelConfig(**value.get("model", {}))
        fields = {key: item for key, item in value.items() if key != "model"}
        return cls(model=model, **fields)

    @classmethod
    def from_json(cls, source: str) -> "ExperimentConfig":
        return cls.from_dict(json.loads(source))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
