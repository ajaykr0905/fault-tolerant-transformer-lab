from __future__ import annotations

import torch

from fttl.config import ExperimentConfig


def synthetic_token_stream(length: int, vocab_size: int) -> torch.Tensor:
    """Return a deterministic, non-random token stream for smoke verification."""
    if length < 2:
        raise ValueError("length must be at least 2")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    positions = torch.arange(length, dtype=torch.long)
    return (positions * 17 + positions.square() * 3 + 11) % vocab_size


def batch_for_step(tokens: torch.Tensor, config: ExperimentConfig, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a stable batch keyed by training step instead of mutable RNG state."""
    width = config.model.block_size + 1
    if tokens.numel() < width:
        raise ValueError("token stream is shorter than one training sequence")
    max_start = tokens.numel() - width + 1
    starts = [((step * config.batch_size + index) * width) % max_start for index in range(config.batch_size)]
    windows = torch.stack([tokens[start : start + width] for start in starts])
    return windows[:, :-1], windows[:, 1:]
