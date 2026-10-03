import json

import pytest

from fttl.config import ExperimentConfig, ModelConfig


def test_fingerprint_is_independent_of_json_key_order():
    first = ExperimentConfig.from_json(json.dumps({"steps": 4, "seed": 7, "model": {"d_model": 16, "n_heads": 2}}))
    second = ExperimentConfig.from_json(json.dumps({"model": {"n_heads": 2, "d_model": 16}, "seed": 7, "steps": 4}))
    assert first.fingerprint() == second.fingerprint()


def test_model_dimension_must_split_across_heads():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(d_model=15, n_heads=4)


@pytest.mark.parametrize("field", ["vocab_size", "block_size", "d_model", "n_heads", "n_layers"])
@pytest.mark.parametrize("value", [True, False, 4.0, "4", None])
def test_model_dimensions_require_integers(field, value):
    with pytest.raises(ValueError, match=field):
        ModelConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1])
def test_attention_heads_must_be_positive(value):
    with pytest.raises(ValueError, match="n_heads"):
        ModelConfig(n_heads=value)


@pytest.mark.parametrize("field", ["seed", "steps", "batch_size", "checkpoint_every"])
@pytest.mark.parametrize("value", [True, False, 4.0, "4", None])
def test_experiment_counts_require_integers(field, value):
    with pytest.raises(ValueError, match=field):
        ExperimentConfig(**{field: value})


@pytest.mark.parametrize("value", [-1, 2**32])
def test_seed_must_fit_numpy_seed_range(value):
    with pytest.raises(ValueError, match="seed"):
        ExperimentConfig(seed=value)


@pytest.mark.parametrize("value", [0, 2**32 - 1])
def test_seed_accepts_numpy_seed_boundaries(value):
    config = ExperimentConfig(seed=value)
    assert ExperimentConfig.from_json(config.canonical_json()) == config


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, "0.1", None])
@pytest.mark.parametrize("field", ["dropout", "learning_rate"])
def test_training_rates_require_finite_numbers(field, value):
    config_type = ModelConfig if field == "dropout" else ExperimentConfig
    with pytest.raises(ValueError, match=field):
        config_type(**{field: value})


def test_config_accepts_valid_integer_rates():
    config = ExperimentConfig(model=ModelConfig(dropout=0), learning_rate=1)
    assert ExperimentConfig.from_json(config.canonical_json()) == config
