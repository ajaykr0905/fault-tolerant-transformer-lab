from __future__ import annotations

import copy
import hashlib
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
from fttl.state import capture_rng_state, code_fingerprint, restore_rng_state
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
    """Schema 2 adds a comparison contract without changing config identity."""

    schema_version: int
    config_fingerprint: str
    comparison_fingerprint: str
    comparison_contract: dict[str, object]
    device: str
    python_version: str
    torch_version: str
    full: TuningRun
    lora: TuningRun
    limitations: tuple[str, ...]


def _workload_token_count(config: ExperimentConfig) -> int:
    return max(2_048, config.model.block_size * config.batch_size * 32)


def _train(model: TinyTransformer, config: ExperimentConfig, mode: str) -> TuningRun:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    token_count = _workload_token_count(config)
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
    alpha = float(rank * 2)
    target_names = ("qkv", "output")
    summary = inject_lora(lora, rank=rank, alpha=alpha, target_names=target_names)
    comparison_contract = {
        "schema_version": 1,
        "experiment_config": config.to_dict(),
        "adapter": {
            "rank": rank,
            "alpha": alpha,
            "target_names": list(target_names),
            "replaced_modules": list(summary.replaced_modules),
        },
        "workload": {
            "kind": "synthetic-token-stream-v1",
            "token_count": _workload_token_count(config),
        },
        "execution": {
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in full.parameters()}),
            "parameter_devices": sorted({str(parameter.device) for parameter in full.parameters()}),
            "code_fingerprint": code_fingerprint(),
            "training_rng_policy": "shared-cpu-python-numpy-state-after-model-initialization-v1",
        },
    }
    encoded_contract = json.dumps(
        comparison_contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    training_rng = capture_rng_state()
    full_run = _train(full, config, "full")
    restore_rng_state(training_rng)
    lora_run = _train(lora, config, "lora")

    comparison = TuningComparison(
        schema_version=2,
        config_fingerprint=config.fingerprint(),
        comparison_fingerprint=hashlib.sha256(encoded_contract.encode("utf-8")).hexdigest(),
        comparison_contract=comparison_contract,
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
