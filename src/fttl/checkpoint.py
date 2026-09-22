from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from fttl.config import ExperimentConfig


class CheckpointMismatchError(ValueError):
    pass


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    step: int,
    tokens_seen: int,
    losses: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "losses": losses,
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_config: ExperimentConfig,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise CheckpointMismatchError("unsupported checkpoint schema")
    if payload.get("config_fingerprint") != expected_config.fingerprint():
        raise CheckpointMismatchError("checkpoint configuration does not match experiment")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    return payload
