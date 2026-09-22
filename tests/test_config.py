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
