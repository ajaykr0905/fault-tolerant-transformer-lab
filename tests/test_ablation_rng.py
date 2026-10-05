import random

import numpy as np
import pytest
import torch

from fttl import ablation
from fttl.config import ExperimentConfig, ModelConfig
from fttl.state import capture_rng_state, restore_rng_state, state_digest


def _config(dropout=0.35):
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1, dropout=dropout
        ),
        seed=73,
        steps=2,
        batch_size=2,
    )


def _caller_state():
    return (
        state_digest(capture_rng_state()),
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


@pytest.fixture(autouse=True)
def isolated_caller_state():
    previous = capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(101)
    np.random.seed(102)
    torch.random.default_generator.manual_seed(103)
    random.gauss(0, 1)
    np.random.normal()
    try:
        yield
    finally:
        restore_rng_state(previous)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.mark.parametrize("deterministic, warn_only", [(False, False), (False, True), (True, True)])
@pytest.mark.parametrize("dropout", [0.0, 0.35])
def test_successful_comparison_preserves_caller_rng_and_deterministic_flags(
    tmp_path, dropout, deterministic, warn_only
):
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    before = _caller_state()
    result = ablation.compare_tuning(_config(dropout), tmp_path / "result.json", rank=2)
    assert _caller_state() == before
    assert result.full.initial_loss == result.lora.initial_loss
    assert (tmp_path / "result.json").is_file()


@pytest.mark.parametrize("rank", [0, -1, True, False, 1.5, "2", None, 9])
def test_invalid_rank_fails_before_model_construction_or_caller_mutation(
    tmp_path, monkeypatch, rank
):
    def unexpected_model(*args, **kwargs):
        pytest.fail("invalid rank reached model construction")

    monkeypatch.setattr(ablation, "TinyTransformer", unexpected_model)
    torch.use_deterministic_algorithms(False, warn_only=True)
    before = _caller_state()
    output = tmp_path / "new-run" / "result.json"
    with pytest.raises(
        ValueError, match="rank must be an integer fitting the base linear dimensions"
    ):
        ablation.compare_tuning(_config(), output, rank=rank)
    assert _caller_state() == before
    assert not output.parent.exists()


@pytest.mark.parametrize("config", [None, {}, ModelConfig()])
def test_invalid_config_fails_without_mutating_caller(tmp_path, config):
    before = _caller_state()
    output = tmp_path / "new-run" / "result.json"
    with pytest.raises(ValueError, match="config must be an ExperimentConfig"):
        ablation.compare_tuning(config, output, rank=2)
    assert _caller_state() == before
    assert not output.parent.exists()


@pytest.mark.parametrize("failing_mode", ["full", "lora"])
@pytest.mark.parametrize("deterministic, warn_only", [(False, False), (True, True)])
def test_training_exceptions_restore_caller_rng_and_deterministic_flags(
    tmp_path, monkeypatch, failing_mode, deterministic, warn_only
):
    original_train = ablation._train

    def failing_train(model, config, mode):
        if mode == failing_mode:
            random.random()
            np.random.random()
            torch.rand(3)
            torch.use_deterministic_algorithms(False, warn_only=True)
            raise RuntimeError("training failed")
        return original_train(model, config, mode)

    monkeypatch.setattr(ablation, "_train", failing_train)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    before = _caller_state()
    output = tmp_path / "new-run" / "result.json"
    with pytest.raises(RuntimeError, match="training failed"):
        ablation.compare_tuning(_config(), output, rank=2)
    assert _caller_state() == before
    assert not output.parent.exists()


@pytest.mark.parametrize("deterministic, warn_only", [(False, False), (True, True)])
def test_late_report_publication_failure_restores_caller_state(
    tmp_path, monkeypatch, deterministic, warn_only
):
    def failed_publication(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(3)
        torch.use_deterministic_algorithms(False, warn_only=True)
        raise OSError("publication failed")

    monkeypatch.setattr(ablation, "publish_json", failed_publication)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    before = _caller_state()
    output = tmp_path / "new-run" / "result.json"
    with pytest.raises(OSError, match="publication failed"):
        ablation.compare_tuning(_config(), output, rank=2)
    assert _caller_state() == before
    assert not output.parent.exists()


def test_late_concurrent_report_preserves_winner_and_restores_caller_state(tmp_path, monkeypatch):
    original_publish = ablation.publish_json
    output = tmp_path / "result.json"

    def competing_publication(path, value):
        path.write_bytes(b"winning experiment evidence\n")
        original_publish(path, value)

    monkeypatch.setattr(ablation, "publish_json", competing_publication)
    torch.use_deterministic_algorithms(False, warn_only=True)
    before = _caller_state()
    with pytest.raises(FileExistsError):
        ablation.compare_tuning(_config(), output, rank=2)
    assert _caller_state() == before
    assert output.read_bytes() == b"winning experiment evidence\n"
