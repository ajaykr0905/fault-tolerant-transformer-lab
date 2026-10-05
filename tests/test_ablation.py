import hashlib
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
            if module.training:
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


def test_adapter_rank_changes_comparison_identity_without_changing_config_identity(tmp_path):
    config = _config(0.35)
    rank_one = ablation.compare_tuning(config, tmp_path / "rank-one.json", rank=1)
    rank_two = ablation.compare_tuning(config, tmp_path / "rank-two.json", rank=2)
    assert rank_one.config_fingerprint == rank_two.config_fingerprint == config.fingerprint()
    assert rank_one.comparison_fingerprint != rank_two.comparison_fingerprint
    assert rank_one.comparison_contract["adapter"]["rank"] == 1
    assert rank_two.comparison_contract["adapter"]["rank"] == 2
    assert rank_one.comparison_contract["adapter"]["alpha"] == 2.0
    assert rank_two.comparison_contract["adapter"]["alpha"] == 4.0


def test_comparison_contract_records_actual_injection_and_workload(monkeypatch, tmp_path):
    observed = {}
    original_inject = ablation.inject_lora

    def observed_inject(model, **kwargs):
        observed.update(kwargs)
        observed["target_names"] = list(kwargs["target_names"])
        summary = original_inject(model, **kwargs)
        observed["replaced_modules"] = list(summary.replaced_modules)
        return summary

    monkeypatch.setattr(ablation, "inject_lora", observed_inject)
    config = _config(0.35)
    comparison = ablation.compare_tuning(config, tmp_path / "result.json", rank=2)
    contract = comparison.comparison_contract
    assert contract["experiment_config"] == config.to_dict()
    assert contract["adapter"] == observed
    assert contract["workload"] == {
        "kind": "synthetic-token-stream-v1",
        "token_count": 2048,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert comparison.comparison_fingerprint == hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_serialized_comparison_contract_reproduces_training_results(tmp_path):
    original = ablation.compare_tuning(_config(0.35), tmp_path / "original.json", rank=2)
    serialized = json.loads((tmp_path / "original.json").read_text(encoding="utf-8"))
    assert serialized["schema_version"] == 2
    contract = serialized["comparison_contract"]
    assert contract["schema_version"] == 1
    assert contract["adapter"]["target_names"] == ["qkv", "output"]
    assert contract["adapter"]["replaced_modules"] == [
        "blocks.0.attention.qkv",
        "blocks.0.attention.output",
    ]
    config = ExperimentConfig.from_dict(contract["experiment_config"])
    reproduced = ablation.compare_tuning(
        config, tmp_path / "reproduced.json", rank=contract["adapter"]["rank"]
    )
    assert reproduced.comparison_fingerprint == original.comparison_fingerprint
    assert reproduced.comparison_contract == original.comparison_contract
    for mode in ("full", "lora"):
        expected, actual = asdict(getattr(original, mode)), asdict(getattr(reproduced, mode))
        expected.pop("elapsed_seconds")
        actual.pop("elapsed_seconds")
        assert expected == actual


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_comparison_contract_records_initialized_parameter_dtype_and_device(
    dtype, monkeypatch, tmp_path
):
    original_train = ablation._train
    observed = []

    def observed_train(model, config, mode):
        observed.append(
            (
                sorted({str(parameter.dtype) for parameter in model.parameters()}),
                sorted({str(parameter.device) for parameter in model.parameters()}),
            )
        )
        return original_train(model, config, mode)

    monkeypatch.setattr(ablation, "_train", observed_train)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        result = ablation.compare_tuning(_config(0.35), tmp_path / "result.json", rank=2)
    finally:
        torch.set_default_dtype(previous_dtype)

    execution = result.comparison_contract["execution"]
    assert execution["parameter_dtypes"] == [str(dtype)]
    assert execution["parameter_devices"] == ["cpu"]
    assert observed == [([str(dtype)], ["cpu"])] * 2
    assert execution["training_rng_policy"] == (
        "shared-cpu-python-numpy-state-after-model-initialization-v1"
    )
    assert execution["code_fingerprint"] == ablation.code_fingerprint()


def test_code_identity_changes_comparison_fingerprint(monkeypatch, tmp_path):
    config = _config(0.35)
    monkeypatch.setattr(ablation, "code_fingerprint", lambda: "0" * 64)
    first = ablation.compare_tuning(config, tmp_path / "first.json", rank=2)
    monkeypatch.setattr(ablation, "code_fingerprint", lambda: "1" * 64)
    second = ablation.compare_tuning(config, tmp_path / "second.json", rank=2)
    assert first.config_fingerprint == second.config_fingerprint
    assert first.comparison_contract["execution"]["code_fingerprint"] == "0" * 64
    assert second.comparison_contract["execution"]["code_fingerprint"] == "1" * 64
    assert first.comparison_fingerprint != second.comparison_fingerprint


def test_tuning_comparison_preserves_existing_evidence_and_rng(tmp_path):
    output = tmp_path / "result.json"
    output.write_bytes(b"existing experiment evidence\n")
    before = state_digest(capture_rng_state())
    with pytest.raises(ValueError, match="fresh"):
        ablation.compare_tuning(_config(0.35), output, rank=2)
    assert output.read_bytes() == b"existing experiment evidence\n"
    assert state_digest(capture_rng_state()) == before


def test_tuning_comparison_cannot_overwrite_a_report_created_during_training(monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    original = ablation._train

    def racing_train(model, config, mode):
        if mode == "lora":
            output.write_bytes(b"concurrent evidence\n")
        return original(model, config, mode)

    monkeypatch.setattr(ablation, "_train", racing_train)
    with pytest.raises(FileExistsError):
        ablation.compare_tuning(_config(0.35), output, rank=2)
    assert output.read_bytes() == b"concurrent evidence\n"
