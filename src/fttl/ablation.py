from __future__ import annotations

import copy
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from fttl.config import ExperimentConfig
from fttl.data import batch_for_step, synthetic_token_stream
from fttl.lora import inject_lora
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, restore_rng_state
from fttl.train import seed_everything


@dataclass(frozen=True)
class TuningRun:
    mode: str
    trainable_parameters: int
    total_parameters: int
    initial_loss: float
    final_loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TuningComparison:
    schema_version: int
    config_fingerprint: str
    device: str
    python_version: str
    torch_version: str
    full: TuningRun
    lora: TuningRun
    limitations: tuple[str, ...]


def _train(model: TinyTransformer, config: ExperimentConfig, mode: str) -> TuningRun:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    token_count = max(2_048, config.model.block_size * config.batch_size * 32)
    tokens = synthetic_token_stream(token_count, config.model.vocab_size)
    losses: list[float] = []
    started = time.perf_counter()
    model.train()
    for step in range(config.steps):
        inputs, targets = batch_for_step(tokens, config, step)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return TuningRun(
        mode=mode,
        trainable_parameters=sum(parameter.numel() for parameter in parameters),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        initial_loss=losses[0],
        final_loss=losses[-1],
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )


def compare_tuning(config: ExperimentConfig, output: Path, *, rank: int = 4) -> TuningComparison:
    seed_everything(config.seed)
    baseline = TinyTransformer(config.model)
    base_state = copy.deepcopy(baseline.state_dict())

    full = TinyTransformer(config.model)
    full.load_state_dict(base_state)

    lora = TinyTransformer(config.model)
    lora.load_state_dict(base_state)
    inject_lora(lora, rank=rank, alpha=float(rank * 2))

    training_rng = capture_rng_state()
    full_run = _train(full, config, "full")
    restore_rng_state(training_rng)
    lora_run = _train(lora, config, "lora")

    comparison = TuningComparison(
        schema_version=1,
        config_fingerprint=config.fingerprint(),
        device="cpu",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        full=full_run,
        lora=lora_run,
        limitations=(
            "Synthetic tokens measure implementation behavior, not language quality.",
            "Elapsed time is a local CPU observation and not a hardware-independent benchmark.",
            "The comparison does not claim GPU, multi-GPU, memory, or production serving results.",
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(comparison), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return comparison
