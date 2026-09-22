from __future__ import annotations

import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from fttl.checkpoint import load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig
from fttl.data import batch_for_step, synthetic_token_stream
from fttl.model import TinyTransformer


@dataclass(frozen=True)
class TrainingResult:
    steps: int
    tokens_seen: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float
    checkpoint: str
    config_fingerprint: str
    device: str
    torch_version: str
    python_version: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def run_training(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
    stop_after_step: int | None = None,
) -> tuple[TinyTransformer, TrainingResult]:
    seed_everything(config.seed)
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    tokens_seen = 0
    losses: list[float] = []

    if resume_from is not None:
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            expected_config=config,
        )
        start_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        losses = [float(value) for value in payload["losses"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    token_count = max(2_048, config.model.block_size * config.batch_size * 32)
    tokens = synthetic_token_stream(token_count, config.model.vocab_size)
    final_step = min(config.steps, stop_after_step or config.steps)
    started = time.perf_counter()

    model.train()
    for step in range(start_step, final_step):
        inputs, targets = batch_for_step(tokens, config, step)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        tokens_seen += inputs.numel()
        completed_step = step + 1
        if completed_step % config.checkpoint_every == 0 or completed_step == final_step:
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=completed_step,
                tokens_seen=tokens_seen,
                losses=losses,
            )

    if not losses:
        raise ValueError("checkpoint already completed the requested training range")

    result = TrainingResult(
        steps=final_step,
        tokens_seen=tokens_seen,
        initial_loss=losses[0],
        final_loss=losses[-1],
        elapsed_seconds=round(time.perf_counter() - started, 6),
        checkpoint=str(checkpoint_path),
        config_fingerprint=config.fingerprint(),
        device="cpu",
        torch_version=torch.__version__,
        python_version=platform.python_version(),
    )
    (output_dir / "result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return model, result
