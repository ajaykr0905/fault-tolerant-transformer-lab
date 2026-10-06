import pytest
import torch

from fttl.config import ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def _model():
    model = TinyTransformer(
        ModelConfig(vocab_size=24, block_size=4, d_model=8, n_heads=2, n_layers=1, dropout=0.5)
    ).train()
    model.blocks[0].attention.dropout.eval()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    return model


def _unchanged_after_rejection(model, tokens, targets, message):
    model.blocks[0].attention.output.bias.requires_grad = False
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    rng = capture_rng_state()
    embedding_calls = []
    hook = model.token_embedding.register_forward_pre_hook(
        lambda module, inputs: embedding_calls.append(True)
    )
    try:
        with pytest.raises(ValueError, match=message):
            model(tokens, targets)
    finally:
        hook.remove()
    assert not embedding_calls, "invalid inputs must fail before embedding or dropout"
    assert state_trees_equal(state, model.state_dict())
    assert state_trees_equal(rng, capture_rng_state())
    assert modes == [module.training for module in model.modules()]
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert all(
        torch.equal(parameter.grad, gradients[name]) for name, parameter in model.named_parameters()
    )


@pytest.mark.parametrize("kind", ["float", "int32", "outside", "negative", "all-ignored"])
def test_invalid_training_targets_reject_before_rng_or_model_changes(kind):
    tokens = torch.arange(8).reshape(2, 4)
    targets = {
        "float": tokens.float(),
        "int32": tokens.int(),
        "outside": torch.full_like(tokens, 24),
        "negative": torch.full_like(tokens, -1),
        "all-ignored": torch.full_like(tokens, -100),
    }[kind]
    _unchanged_after_rejection(_model(), tokens, targets, "targets")


@pytest.mark.parametrize("shape", [(), (8,), (2, 2, 2), (0, 4), (2, 0), (2, 5)])
def test_invalid_token_dimensions_preserve_caller_state(shape):
    tokens = torch.zeros(shape, dtype=torch.long)
    _unchanged_after_rejection(_model(), tokens, None, "tokens|sequence")


@pytest.mark.parametrize(
    "kind", ["float", "bool", "int16", "outside", "negative", "list", "sparse", "meta"]
)
def test_invalid_token_tensors_reject_before_embedding(kind):
    tokens = torch.arange(8).reshape(2, 4)
    inputs = {
        "float": tokens.float(),
        "bool": tokens.bool(),
        "int16": tokens.short(),
        "outside": torch.full_like(tokens, 24),
        "negative": torch.full_like(tokens, -1),
        "list": tokens.tolist(),
        "sparse": tokens.to_sparse(),
        "meta": torch.empty((2, 4), dtype=torch.long, device="meta"),
    }[kind]
    _unchanged_after_rejection(_model(), inputs, None, "tokens")


@pytest.mark.parametrize("kind", ["list", "sparse", "meta", "wrong-shape"])
def test_invalid_target_tensor_contract_rejects_before_embedding(kind):
    tokens = torch.arange(8).reshape(2, 4)
    targets = {
        "list": tokens.tolist(),
        "sparse": tokens.to_sparse(),
        "meta": torch.empty((2, 4), dtype=torch.long, device="meta"),
        "wrong-shape": tokens.reshape(1, 8),
    }[kind]
    _unchanged_after_rejection(_model(), tokens, targets, "targets")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_valid_integer_token_types_keep_exact_logits_and_autograd(dtype):
    model = _model().eval()
    tokens = torch.arange(8).reshape(2, 4)
    expected, _ = model(tokens)
    logits, loss = model(tokens.to(dtype), tokens)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert loss is not None and torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA hardware")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_matching_cuda_inputs_remain_supported_and_wrong_device_rejects(dtype):
    model = _model().cuda()
    tokens = torch.arange(8, device="cuda").reshape(2, 4)
    logits, loss = model(tokens.to(dtype), tokens)
    assert logits.device.type == "cuda" and loss is not None and torch.isfinite(loss)
    loss.backward()
    rng = torch.cuda.get_rng_state().clone()
    with pytest.raises(ValueError, match="targets.*device"):
        model(tokens, tokens.cpu())
    assert torch.equal(torch.cuda.get_rng_state(), rng)
    with pytest.raises(ValueError, match="tokens.*device"):
        model(tokens.cpu())
    assert torch.equal(torch.cuda.get_rng_state(), rng)
