import random

import numpy as np
import pytest
import torch
from test_evaluation import byte_model
from test_real_data_recovery import prepared_fixture

from fttl import evaluation
from fttl.dataset import load_dataset_snapshot
from fttl.state import capture_rng_state, state_trees_equal


@pytest.mark.parametrize(
    "kind",
    ["unused-nan", "buffer-nan", "buffer-inf", "buffer-meta", "buffer-sparse", "integer-parameter"],
)
def test_invalid_complete_model_state_rejects_before_dataset_access(tmp_path, monkeypatch, kind):
    model = byte_model().train()
    if kind == "unused-nan":
        with torch.no_grad():
            model.position_embedding.weight[-1].fill_(float("nan"))
    elif kind == "integer-parameter":
        model.position_embedding.weight = torch.nn.Parameter(
            torch.ones_like(model.position_embedding.weight, dtype=torch.long), requires_grad=False
        )
    else:
        value = {
            "buffer-nan": torch.tensor([float("nan")]),
            "buffer-inf": torch.tensor([float("inf")]),
            "buffer-meta": torch.empty(1, device="meta"),
            "buffer-sparse": torch.ones(2).to_sparse(),
        }[kind]
        model.register_buffer("private", value, persistent=False)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    rng = capture_rng_state()
    monkeypatch.setattr(
        evaluation, "load_dataset_snapshot", lambda _: pytest.fail("dataset accessed")
    )
    with pytest.raises(ValueError, match="model state"):
        evaluation.evaluate_held_out(model, tmp_path / "missing", max_tokens=1)
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_valid_float_models_and_integer_private_buffers_are_supported(tmp_path, dtype):
    snapshot = load_dataset_snapshot(prepared_fixture(tmp_path))
    model = byte_model().to(dtype=dtype)
    model.register_buffer(
        "integer_private", torch.tensor([1, 2], dtype=torch.long), persistent=False
    )
    result = evaluation.evaluate_held_out(model, snapshot, max_tokens=13)
    assert result.evaluated_target_tokens == 13
    assert result.mean_negative_log_likelihood > 0


def test_direct_evaluation_uses_cpu_windows_under_meta_default(tmp_path):
    snapshot = load_dataset_snapshot(prepared_fixture(tmp_path))
    model = byte_model()
    expected = evaluation.evaluate_held_out(model, snapshot, max_tokens=13)
    rng = capture_rng_state()
    with torch.device("meta"):
        actual = evaluation.evaluate_held_out(model, snapshot, max_tokens=13)
        assert torch.empty(1).device.type == "meta"
    assert actual == expected
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("failure", [False, True])
def test_model_hooks_cannot_leak_cpu_rng_modes_or_gradient_flags(tmp_path, failure):
    snapshot = load_dataset_snapshot(prepared_fixture(tmp_path))
    model = byte_model().train()
    model.blocks[0].attention.dropout.eval()
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in parameters]
    rng = capture_rng_state()

    def hook(_module, _inputs):
        random.random()
        np.random.random(5)
        torch.rand(3)
        parameters[0].requires_grad = False
        if failure:
            raise ValueError("injected evaluation failure")

    handle = model.register_forward_pre_hook(hook)
    try:
        if failure:
            with pytest.raises(ValueError, match="injected"):
                evaluation.evaluate_held_out(model, snapshot, max_tokens=13)
        else:
            evaluation.evaluate_held_out(model, snapshot, max_tokens=13)
    finally:
        handle.remove()
    assert state_trees_equal(rng, capture_rng_state())
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in parameters]
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in parameters)


@pytest.mark.parametrize("kind", ["parameter", "private-buffer"])
def test_mutated_complete_model_identity_cannot_be_returned_as_valid_evidence(tmp_path, kind):
    snapshot = load_dataset_snapshot(prepared_fixture(tmp_path))
    model = byte_model()
    model.register_buffer("private", torch.ones(1), persistent=False)

    def mutate(module, _inputs):
        with torch.no_grad():
            if kind == "parameter":
                module.position_embedding.weight[-1].add_(1)
            else:
                module.private.add_(1)

    handle = model.register_forward_pre_hook(mutate)
    try:
        with pytest.raises(ValueError, match="model state.*changed"):
            evaluation.evaluate_held_out(model, snapshot, max_tokens=1)
    finally:
        handle.remove()
