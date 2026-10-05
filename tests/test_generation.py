import random

import numpy as np
import pytest
import torch

from fttl.config import ModelConfig
from fttl.generation import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, generate_tokens
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_digest, state_trees_equal


def _model(vocab_size=8):
    return TinyTransformer(
        ModelConfig(
            vocab_size=vocab_size, block_size=4, d_model=8, n_heads=2, n_layers=1, dropout=0.5
        )
    )


def _fixed_logits(model, values, monkeypatch, calls=None):
    def forward(tokens):
        if calls is not None:
            calls.append(tokens.clone())
        return torch.tensor(values).view(1, 1, -1).expand(1, tokens.shape[1], -1), None

    monkeypatch.setattr(model, "forward", forward)


def test_greedy_generation_has_complete_identity_and_bounded_context(monkeypatch):
    model = _model()
    calls = []
    _fixed_logits(model, [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0], monkeypatch, calls)
    result = generate_tokens(model, [0, 1, 2, 4, 5, 6], max_new_tokens=3)
    assert result.prompt_token_ids == (0, 1, 2, 4, 5, 6)
    assert result.generated_token_ids == (3, 3, 3)
    assert result.token_ids == result.prompt_token_ids + result.generated_token_ids
    assert result.generated_count == 3
    assert result.stopping_reason == "max_new_tokens"
    assert [tokens.tolist()[0] for tokens in calls] == [[2, 4, 5, 6], [4, 5, 6, 3], [5, 6, 3, 3]]
    assert result.model_state_digest == state_digest(model.state_dict())
    assert len(result.model_config_fingerprint) == 64
    assert result.to_dict()["generated_count"] == 3


def test_positions_reset_after_context_crop():
    model = _model()
    positions = []
    hook = model.position_embedding.register_forward_pre_hook(
        lambda module, args: positions.append(args[0].clone())
    )
    try:
        generate_tokens(model, [1, 2, 3, 4, 5], max_new_tokens=3)
    finally:
        hook.remove()
    assert [value.tolist() for value in positions] == [[0, 1, 2, 3]] * 3


def test_seeded_sampling_replays_without_using_global_rng(monkeypatch):
    model = _model()
    _fixed_logits(model, [0.0] * 8, monkeypatch)
    first = generate_tokens(model, [1], max_new_tokens=32, method="sample", seed=17, top_k=4)
    random.random()
    np.random.random()
    torch.rand(23)
    second = generate_tokens(model, [1], max_new_tokens=32, method="sample", seed=17, top_k=4)
    other = generate_tokens(model, [1], max_new_tokens=32, method="sample", seed=18, top_k=4)
    assert first == second
    assert first.generated_token_ids != other.generated_token_ids
    assert set(first.generated_token_ids) <= {0, 1, 2, 3}


@pytest.mark.parametrize("method", ["greedy", "sample"])
def test_ties_prefer_low_token_id_and_top_one_matches_greedy(monkeypatch, method):
    model = _model()
    _fixed_logits(model, [1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], monkeypatch)
    assert generate_tokens(
        model, [0], max_new_tokens=3, method=method, top_k=1
    ).generated_token_ids == (1, 1, 1)


def test_stop_token_is_included_only_when_newly_generated(monkeypatch):
    model = _model()
    _fixed_logits(model, [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0], monkeypatch)
    result = generate_tokens(model, [2, 1], max_new_tokens=9, stop_token_id=2)
    assert result.generated_token_ids == (2,)
    assert result.generated_count == 1
    assert result.stopping_reason == "stop_token"


def test_byte_prompt_and_zero_length_generation_do_not_call_model(monkeypatch):
    model = _model(257)
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    result = generate_tokens(model, b"hi", max_new_tokens=0)
    assert result.token_ids == (104, 105)
    assert result.generated_token_ids == ()
    assert result.generated_count == 0 and result.stopping_reason == "max_new_tokens"


@pytest.mark.parametrize("method", ["greedy", "sample"])
def test_generation_preserves_state_gradients_modes_flags_and_cpu_rng(method):
    model = _model().train()
    model.blocks[0].attention.dropout.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    rng = capture_rng_state()
    deterministic = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    generate_tokens(model, [0, 1, 2], max_new_tokens=4, method=method, seed=7)
    assert state_trees_equal(rng, capture_rng_state())
    assert state_trees_equal(state, model.state_dict())
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert all(
        torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters()
    )
    assert deterministic == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


def test_failure_restores_heterogeneous_modes_and_cpu_rng(monkeypatch):
    model = _model().train()
    model.blocks[0].eval()
    modes = [module.training for module in model.modules()]

    def fail(_):
        random.random()
        np.random.random()
        torch.rand(3)
        raise RuntimeError("backend failure")

    monkeypatch.setattr(model, "forward", fail)
    rng = capture_rng_state()
    with pytest.raises(RuntimeError, match="backend failure"):
        generate_tokens(model, [1], max_new_tokens=1)
    assert state_trees_equal(rng, capture_rng_state())
    assert [module.training for module in model.modules()] == modes


@pytest.mark.parametrize(
    "options,message",
    [
        ({"max_new_tokens": -1}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"max_new_tokens": 1.5}, "max_new_tokens"),
        ({"max_new_tokens": MAX_NEW_TOKENS + 1}, "max_new_tokens"),
        ({"method": "beam"}, "method"),
        ({"method": []}, "method"),
        ({"temperature": 0}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": float("inf")}, "temperature"),
        ({"temperature": 10**400}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": True}, "top_k"),
        ({"top_k": 9}, "top_k"),
        ({"top_k": 1.5}, "top_k"),
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"stop_token_id": 8}, "stop_token_id"),
        ({"stop_token_id": True}, "stop_token_id"),
    ],
)
def test_invalid_controls_fail_before_forward(monkeypatch, options, message):
    model = _model()
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    rng = capture_rng_state()
    with pytest.raises(ValueError, match=message):
        generate_tokens(model, [0], **options)
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize(
    "prompt", [[], "text", iter([1]), [True], [1.5], [-1], [8], [0] * (MAX_PROMPT_TOKENS + 1)]
)
def test_invalid_prompt_fails_before_forward(monkeypatch, prompt):
    model = _model()
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    with pytest.raises(ValueError, match="prompt"):
        generate_tokens(model, prompt)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_logits_reject_and_restore_modes(monkeypatch, value):
    model = _model().train()
    _fixed_logits(model, [value] * 8, monkeypatch)
    with pytest.raises(FloatingPointError, match="generation logits"):
        generate_tokens(model, [1], max_new_tokens=1)
    assert model.training


def test_sampling_handles_subnormal_temperature_and_large_finite_logits(monkeypatch):
    model = _model()
    _fixed_logits(model, [0.0, 1e30, -1e30, 0.0, 0.0, 0.0, 0.0, 0.0], monkeypatch)
    result = generate_tokens(model, [1], max_new_tokens=3, method="sample", temperature=5e-324)
    assert result.generated_token_ids == (1, 1, 1)


def test_non_cpu_model_rejects_before_forward(monkeypatch):
    model = _model().to("meta")
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    with pytest.raises(ValueError, match="CPU"):
        generate_tokens(model, [1])


def test_cpu_generation_overrides_inherited_default_device():
    model = _model()
    with torch.device("meta"):
        result = generate_tokens(model, [1], max_new_tokens=1, method="sample")
    assert result.generated_count == 1


def test_invalid_logit_shape_is_rejected(monkeypatch):
    model = _model()
    monkeypatch.setattr(model, "forward", lambda _: (torch.zeros(1, 1, 9), None))
    with pytest.raises(ValueError, match="logits violate"):
        generate_tokens(model, [1], max_new_tokens=1)


@pytest.mark.parametrize("persistent", [False, True])
def test_all_named_buffers_must_be_finite_before_forward(monkeypatch, persistent):
    model = _model()
    model.register_buffer("extra", torch.tensor(float("nan")), persistent=persistent)
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    with pytest.raises(FloatingPointError, match="model state.buffers.extra"):
        generate_tokens(model, [1], max_new_tokens=1)


def test_nonpersistent_non_cpu_buffer_is_rejected_before_forward(monkeypatch):
    model = _model()
    model.register_buffer("extra", torch.zeros(1, device="meta"), persistent=False)
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    with pytest.raises(ValueError, match="CPU"):
        generate_tokens(model, [1], max_new_tokens=1)


def test_sparse_cpu_parameter_rejects_before_forward(monkeypatch):
    model = _model()
    model.final_norm.weight = torch.nn.Parameter(model.final_norm.weight.detach().to_sparse())
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="materialized dense"):
        generate_tokens(model, [1], max_new_tokens=1)
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("persistent", [False, True])
def test_sparse_cpu_buffer_rejects_before_forward(monkeypatch, persistent):
    model = _model()
    model.register_buffer("extra", torch.ones(3).to_sparse(), persistent=persistent)
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("called model"))
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="materialized dense"):
        generate_tokens(model, [1], max_new_tokens=1)
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("method", ["greedy", "sample"])
def test_sparse_cpu_logits_reject_cleanly_and_restore_caller_state(monkeypatch, method):
    model = _model().train()
    model.blocks[0].attention.eval()
    modes = [module.training for module in model.modules()]
    monkeypatch.setattr(model, "forward", lambda _: (torch.ones(1, 1, 8).to_sparse(), None))
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="logits violate"):
        generate_tokens(model, [1], max_new_tokens=1, method=method)
    assert state_trees_equal(rng, capture_rng_state())
    assert [module.training for module in model.modules()] == modes
