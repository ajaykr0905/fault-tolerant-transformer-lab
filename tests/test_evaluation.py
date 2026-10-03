import math

import pytest
import torch
from test_real_data_recovery import prepared_fixture

from fttl.config import ModelConfig
from fttl.dataset import load_prepared_documents
from fttl.evaluation import evaluate_held_out
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def byte_model():
    return TinyTransformer(
        ModelConfig(vocab_size=257, block_size=8, d_model=8, n_heads=2, n_layers=1, dropout=0.5)
    )


def test_uniform_model_counts_every_held_out_target_once(tmp_path):
    manifest = prepared_fixture(tmp_path)
    model = byte_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    documents = load_prepared_documents(manifest, split="validation")
    expected = sum(len(document.text.encode()) for document in documents)
    result = evaluate_held_out(model, manifest, max_tokens=100000)
    assert result.evaluated_target_tokens == result.available_target_tokens == expected
    assert result.evaluated_documents == len(documents)
    assert result.windows == sum(
        math.ceil(len(document.text.encode()) / 8) for document in documents
    )
    assert result.complete_split
    assert result.mean_negative_log_likelihood == pytest.approx(math.log(257), rel=1e-6)
    assert result.perplexity == pytest.approx(257, rel=1e-6)


def test_evaluation_is_bounded_and_preserves_modes_gradients_rng_and_state(tmp_path):
    manifest = prepared_fixture(tmp_path)
    model = byte_model().train()
    model.blocks[0].attention.dropout.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = capture_rng_state()
    first = evaluate_held_out(model, manifest, max_tokens=13)
    second = evaluate_held_out(model, manifest, max_tokens=13)
    assert first == second
    assert first.evaluated_target_tokens == 13
    assert first.windows == 2
    assert not first.complete_split
    assert [module.training for module in model.modules()] == modes
    assert state_trees_equal(before, model.state_dict())
    assert state_trees_equal(rng, capture_rng_state())
    assert all(
        torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters()
    )


def test_failed_evaluation_restores_module_modes(tmp_path, monkeypatch):
    manifest = prepared_fixture(tmp_path)
    model = byte_model().train()
    model.blocks[0].eval()
    modes = [module.training for module in model.modules()]
    monkeypatch.setattr(
        model, "forward", lambda tokens: (torch.full((1, tokens.shape[1], 257), float("nan")), None)
    )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_held_out(model, manifest)
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_evaluation_rejects_invalid_budget_before_manifest_access(tmp_path, budget):
    with pytest.raises(ValueError, match="max_tokens"):
        evaluate_held_out(byte_model(), tmp_path / "missing", max_tokens=budget)


def test_training_split_cannot_be_reported_as_held_out(tmp_path):
    with pytest.raises(ValueError, match="held-out"):
        evaluate_held_out(byte_model(), tmp_path / "missing", split="train")
