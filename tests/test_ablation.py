import json
import random
from dataclasses import asdict

import numpy as np
import pytest
import torch

from fttl import ablation
from fttl.config import ExperimentConfig, ModelConfig
from fttl.state import capture_rng_state, state_digest


def _config(dropout):
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1, dropout=dropout
        ),
        seed=73,
        steps=3,
        batch_size=2,
    )


@pytest.mark.parametrize("dropout", [0.0, 0.35])
def test_tuning_arms_start_with_identical_stochastic_loss(dropout, tmp_path):
    comparison = ablation.compare_tuning(_config(dropout), tmp_path / "result.json", rank=2)
    assert comparison.full.initial_loss == comparison.lora.initial_loss


def test_tuning_arms_replay_the_same_cpu_rng_and_dropout_sequence(monkeypatch, tmp_path):
    original_train = ablation._train
    entry_states = {}
    dropout_states = {}

    def observed_train(model, config, mode):
        entry_states[mode] = state_digest(capture_rng_state())
        dropout_states[mode] = []

        def record_dropout_rng(module, inputs):
            dropout_states[mode].append(torch.get_rng_state().clone())

        hooks = [
            module.register_forward_pre_hook(record_dropout_rng)
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        try:
            return original_train(model, config, mode)
        finally:
            for hook in hooks:
                hook.remove()

    monkeypatch.setattr(ablation, "_train", observed_train)
    ablation.compare_tuning(_config(0.35), tmp_path / "result.json", rank=2)
    assert entry_states["full"] == entry_states["lora"]
    assert len(dropout_states["full"]) == len(dropout_states["lora"]) == 6
    for full_state, lora_state in zip(dropout_states["full"], dropout_states["lora"], strict=True):
        assert torch.equal(full_state, lora_state)


def test_tuning_comparison_repeats_exactly_except_observation_time(tmp_path):
    config = _config(0.35)
    first = ablation.compare_tuning(config, tmp_path / "first.json", rank=2)
    random.random()
    np.random.random()
    torch.rand(17)
    second = ablation.compare_tuning(config, tmp_path / "second.json", rank=2)

    for comparison, path in ((first, "first.json"), (second, "second.json")):
        serialized = json.loads((tmp_path / path).read_text(encoding="utf-8"))
        for mode in ("full", "lora"):
            assert serialized[mode] == asdict(getattr(comparison, mode))
            assert serialized[mode]["elapsed_seconds"] >= 0
    first_values, second_values = asdict(first), asdict(second)
    for mode in ("full", "lora"):
        first_values[mode].pop("elapsed_seconds")
        second_values[mode].pop("elapsed_seconds")
    assert first_values == second_values
