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
        resume_from=tmp_path / "restart" / interrupted_result.checkpoint,
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
            tmp_path / "run" / result.checkpoint,
            model=model,
            optimizer=optimizer,
            expected_config=changed,
        )


def test_resume_rejects_a_boundary_at_or_behind_the_checkpoint(tmp_path: Path):
    config = tiny_config()
    _, result = run_training(config, tmp_path / "run", stop_after_step=2)

    with pytest.raises(ValueError, match="greater than the checkpoint"):
        run_training(
            config,
            tmp_path / "run",
            resume_from=tmp_path / "run" / result.checkpoint,
            stop_after_step=1,
        )

    with pytest.raises(ValueError, match="greater than the checkpoint"):
        run_training(
            config,
            tmp_path / "run",
            resume_from=tmp_path / "run" / result.checkpoint,
            stop_after_step=2,
        )


def test_training_rejects_non_positive_stop_boundary(tmp_path: Path):
    with pytest.raises(ValueError, match="at least 1"):
        run_training(tiny_config(), tmp_path / "run", stop_after_step=0)


def test_trusted_legacy_synthetic_checkpoint_has_an_explicit_v2_migration_path(
    tmp_path: Path,
):
    config = tiny_config()
    control, control_result = run_training(config, tmp_path / "control")
    _, partial = run_training(config, tmp_path / "partial", stop_after_step=2)
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    payload = load_checkpoint(
        tmp_path / "partial" / partial.checkpoint,
        model=model,
        optimizer=optimizer,
        expected_config=config,
    )
    legacy = tmp_path / "trusted-legacy.pt"
    torch.save(
        {
            "schema_version": 1,
            "config": config.to_dict(),
            "config_fingerprint": config.fingerprint(),
            "model": payload["model"],
            "optimizer": payload["optimizer"],
            "step": payload["step"],
            "tokens_seen": payload["tokens_seen"],
            "losses": payload["losses"],
            "torch_rng_state": payload["rng_state"]["torch_cpu"],
        },
        legacy,
    )

    resumed, resumed_result = run_training(
        config,
        tmp_path / "legacy-resume",
        resume_from=legacy,
    )

    assert resumed_result.batch_ids == control_result.batch_ids
    assert resumed_result.sample_ids == control_result.sample_ids
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
