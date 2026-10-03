import pytest
import torch

from fttl.config import ModelConfig
from fttl.lora import LoRALinear, inject_lora
from fttl.model import TinyTransformer


def test_lora_starts_as_an_exact_no_op_and_freezes_base_weights():
    torch.manual_seed(31)
    config = ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config).eval()
    tokens = torch.arange(8).reshape(1, 8) % config.vocab_size
    with torch.no_grad():
        before, _ = model(tokens)

    summary = inject_lora(model, rank=2, alpha=4)
    with torch.no_grad():
        after, _ = model(tokens)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert summary.replaced_modules == ("blocks.0.attention.qkv", "blocks.0.attention.output")
    assert 0 < summary.trainable_parameters < summary.total_parameters
    assert all(
        parameter.requires_grad
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for name, parameter in module.named_parameters()
        if name.startswith("lora_")
    )
    assert all(
        not parameter.requires_grad
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in module.base.parameters()
    )


def test_lora_update_changes_logits_without_unfreezing_base():
    torch.manual_seed(41)
    config = ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config)
    inject_lora(model, rank=2, alpha=4)
    tokens = torch.arange(8).reshape(1, 8) % config.vocab_size
    targets = (tokens + 1) % config.vocab_size
    before, _ = model(tokens)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    _, loss = model(tokens, targets)
    assert loss is not None
    loss.backward()
    optimizer.step()
    after, _ = model(tokens)
    assert not torch.equal(before, after)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_lora_preserves_cpu_base_dtype_and_supports_optimizer_updates(dtype, bias):
    torch.manual_seed(51)
    base = torch.nn.Linear(8, 6, bias=bias, dtype=dtype, device="cpu")
    inputs = torch.randn(4, 8, dtype=dtype, requires_grad=True)
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items()}
    before = base(inputs).detach()
    layer = LoRALinear(base, rank=2, alpha=4)

    assert layer.lora_a.dtype == layer.lora_b.dtype == base.weight.dtype
    assert layer.lora_a.device == layer.lora_b.device == base.weight.device
    torch.testing.assert_close(layer(inputs), before, rtol=0, atol=0)

    optimizer = torch.optim.AdamW([layer.lora_a, layer.lora_b], lr=1e-2)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        layer(inputs).square().mean().backward()
        for parameter in (layer.lora_a, layer.lora_b):
            assert parameter.grad is not None
            assert parameter.grad.dtype == dtype
            assert torch.isfinite(parameter.grad).all()
        assert inputs.grad is not None
        assert torch.isfinite(inputs.grad).all()
        assert all(parameter.grad is None for parameter in base.parameters())
        optimizer.step()

    assert not torch.equal(layer(inputs), before)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, base_state[name], rtol=0, atol=0)


def test_lora_injection_preserves_float64_transformer_outputs():
    torch.manual_seed(61)
    config = ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config).double().eval()
    tokens = torch.arange(8).reshape(1, 8) % config.vocab_size
    with torch.no_grad():
        before, _ = model(tokens)

    inject_lora(model, rank=2, alpha=4)
    with torch.no_grad():
        after, _ = model(tokens)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": True},
        {"rank": 1.5},
        {"rank": 4},
        {"alpha": 0},
        {"alpha": float("nan")},
        {"alpha": float("inf")},
        {"alpha": True},
        {"alpha": "4"},
        {"target_names": ("missing",)},
        {"target_names": ()},
        {"target_names": "qkv"},
        {"target_names": ("qkv", None)},
    ],
)
def test_rejected_lora_injection_preserves_model_state(kwargs):
    model = torch.nn.Module()
    model.qkv = torch.nn.Linear(8, 8)
    model.output = torch.nn.Linear(2, 2)
    model.output.bias.requires_grad = False
    modules_before = dict(model.named_modules())
    parameters_before = dict(model.named_parameters())
    flags_before = {name: parameter.requires_grad for name, parameter in parameters_before.items()}
    state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    options = {"rank": 2, "alpha": 4, **kwargs}
    with pytest.raises(ValueError):
        inject_lora(model, **options)

    assert dict(model.named_modules()) == modules_before
    assert all(parameter is parameters_before[name] for name, parameter in model.named_parameters())
    assert {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    } == flags_before
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("rank", [True, 1.5, "2", None])
def test_lora_linear_rejects_noninteger_rank_without_freezing_base(rank):
    base = torch.nn.Linear(8, 6)
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(base, rank=rank, alpha=4)
    assert all(parameter.requires_grad for parameter in base.parameters())


@pytest.mark.parametrize("alpha", [True, float("nan"), float("inf"), "4", None])
def test_lora_linear_rejects_invalid_alpha_without_freezing_base(alpha):
    base = torch.nn.Linear(8, 6)
    with pytest.raises(ValueError, match="alpha"):
        LoRALinear(base, rank=2, alpha=alpha)
    assert all(parameter.requires_grad for parameter in base.parameters())
