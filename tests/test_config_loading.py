import copy
import hashlib
import json
import sys

import pytest
import torch

from fttl import evaluation_cli
from fttl.config import ExperimentConfig, ModelConfig


@pytest.mark.parametrize(
    "source, field",
    [
        ('{"steps": 2, "steps": 4}', "steps"),
        ('{"steps": 2, "steps": 2}', "steps"),
        ('{"model": {}, "model": {"n_layers": 1}}', "model"),
        ('{"model": {"n_layers": 1, "n_layers": 2}}', "n_layers"),
        ('{"model": {"n_\\u006cayers": 1, "n_layers": 2}}', "n_layers"),
    ],
)
def test_json_rejects_duplicate_fields_instead_of_silently_selecting_one(source, field):
    with pytest.raises(ValueError, match=f"^duplicate JSON field: {field}$"):
        ExperimentConfig.from_json(source)


@pytest.mark.parametrize("value", [None, [], [1], "config", 42, True])
def test_loading_requires_an_object_root(value):
    with pytest.raises(ValueError, match="^experiment configuration must be an object$"):
        ExperimentConfig.from_dict(value)
    with pytest.raises(ValueError, match="^experiment configuration must be an object$"):
        ExperimentConfig.from_json(json.dumps(value))


@pytest.mark.parametrize("value", [None, [], [1], "model", 42, True])
def test_loading_requires_an_object_model(value):
    with pytest.raises(ValueError, match="^model configuration must be an object$"):
        ExperimentConfig.from_dict({"model": value})
    with pytest.raises(ValueError, match="^model configuration must be an object$"):
        ExperimentConfig.from_json(json.dumps({"model": value}))


@pytest.mark.parametrize(
    "value, message",
    [
        ({"zebra": 1, "beta": 2}, "unknown experiment fields: beta, zebra"),
        ({"model": {"zebra": 1, "beta": 2}}, "unknown model fields: beta, zebra"),
        ({1: 2}, "experiment field names must be strings"),
        ({"model": {1: 2}}, "model field names must be strings"),
    ],
)
def test_unknown_fields_fail_with_stable_messages_without_mutating_input(value, message):
    original = copy.deepcopy(value)
    with pytest.raises(ValueError) as error:
        ExperimentConfig.from_dict(value)
    assert str(error.value) == message
    assert value == original


@pytest.mark.parametrize("model", [None, {}, [], "model", 42, True])
def test_direct_experiment_construction_requires_a_validated_model(model):
    with pytest.raises(ValueError, match="^model must be a ModelConfig$"):
        ExperimentConfig(model=model)


@pytest.mark.parametrize("value", [{}, {"model": {}}, {"steps": 3, "model": {"n_layers": 1}}])
def test_valid_partial_configs_round_trip_without_mutating_input(value):
    original = copy.deepcopy(value)
    config = ExperimentConfig.from_dict(value)
    assert value == original
    assert ExperimentConfig.from_json(config.canonical_json()) == config
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert isinstance(config.model, ModelConfig)


def test_default_canonical_config_and_fingerprint_remain_unchanged():
    canonical = (
        '{"batch_size":4,"checkpoint_every":4,"learning_rate":0.0003,'
        '"model":{"block_size":16,"d_model":32,"dropout":0.0,"n_heads":4,'
        '"n_layers":2,"vocab_size":64},"seed":2712,"steps":8}'
    )
    config = ExperimentConfig.from_json("{}")
    assert config.canonical_json() == canonical
    assert config.fingerprint() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(
    "source, message",
    [
        ("[]", "experiment configuration must be an object"),
        ('{"model": null}', "model configuration must be an object"),
        ('{"steps": 2, "steps": 4}', "duplicate JSON field: steps"),
        ('{"unknown": 1}', "unknown experiment fields: unknown"),
    ],
)
def test_evaluation_cli_rejects_hostile_config_before_work_or_artifacts(
    tmp_path, monkeypatch, capsys, source, message
):
    config_path = tmp_path / "input.json"
    config_path.write_text(source, encoding="utf-8")
    output = tmp_path / "new-run" / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fttl-evaluate-checkpoint",
            "--config",
            str(config_path),
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--dataset-manifest",
            str(tmp_path / "missing-manifest.json"),
            "--output",
            str(output),
        ],
    )

    def unexpected_evaluation(*args, **kwargs):
        pytest.fail("invalid configuration reached checkpoint evaluation")

    monkeypatch.setattr(evaluation_cli, "evaluate_checkpoint", unexpected_evaluation)
    rng = torch.get_rng_state().clone()
    with pytest.raises(SystemExit) as error:
        evaluation_cli.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert message in stderr
    assert "Traceback" not in stderr
    assert not output.parent.exists()
    assert torch.equal(torch.get_rng_state(), rng)
