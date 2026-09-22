from pathlib import Path

import pytest
import torch

from fttl.checkpoint import CheckpointMismatchError, load_checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.train import run_training


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(vocab_size=24, block_size=6, d_model=12, n_heads=3, n_layers=1),
        seed=19,
        steps=4,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=2,
    )


def test_checkpoint_restart_matches_uninterrupted_training(tmp_path: Path):
    config = tiny_config()
    uninterrupted, direct_result = run_training(config, tmp_path / "direct")
    _, interrupted_result = run_training(config, tmp_path / "restart", stop_after_step=2)
    resumed, resumed_result = run_training(
        config,
        tmp_path / "restart",
        resume_from=Path(interrupted_result.checkpoint),
    )

    assert direct_result.steps == resumed_result.steps == 4
    assert direct_result.tokens_seen == resumed_result.tokens_seen
    for name, value in uninterrupted.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)


def test_checkpoint_rejects_configuration_drift(tmp_path: Path):
    config = tiny_config()
    _, result = run_training(config, tmp_path / "run", stop_after_step=2)
    changed = ExperimentConfig(
        model=config.model,
        seed=config.seed,
        steps=config.steps,
        batch_size=config.batch_size,
        learning_rate=5e-4,
        checkpoint_every=config.checkpoint_every,
    )
    model = TinyTransformer(changed.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=changed.learning_rate)
    with pytest.raises(CheckpointMismatchError, match="does not match"):
        load_checkpoint(
            Path(result.checkpoint),
            model=model,
            optimizer=optimizer,
            expected_config=changed,
        )
