import pytest
import torch

from fttl.adapter_merge import merge_lora_for_inference
from fttl.config import ModelConfig
from fttl.lora import LoRALinear, inject_lora
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_digest


def adapted_model(dtype=torch.float32):
    model = TinyTransformer(
        ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)
    ).to(dtype=dtype)
    inject_lora(model, rank=2, alpha=4)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.normal_(std=0.02)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_merged_model_matches_adapted_logits_and_is_standalone(dtype):
    source = adapted_model(dtype).eval()
    inputs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected, _ = source(inputs)
    merged = merge_lora_for_inference(source)
    actual, _ = merged(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
    assert not any(isinstance(module, LoRALinear) for module in merged.modules())
    assert not any(parameter.requires_grad for parameter in merged.parameters())
    assert not any(module.training for module in merged.modules())
    assert merged.lm_head.weight is merged.token_embedding.weight
    assert merged.parameter_count() < source.parameter_count()
    # The folded model can be reconstructed with the ordinary architecture.
    reconstructed = TinyTransformer(merged.config).to(dtype=dtype).eval()
    reconstructed.load_state_dict(merged.state_dict())
    torch.testing.assert_close(reconstructed(inputs)[0], actual, rtol=0, atol=0)


def test_merge_preserves_source_modes_flags_gradients_state_and_rng():
    source = adapted_model().train()
    source.blocks[0].attention.dropout.eval()
    for parameter in source.parameters():
        parameter.grad = torch.ones_like(parameter)
    modes = [module.training for module in source.modules()]
    flags = [parameter.requires_grad for parameter in source.parameters()]
    before = state_digest(source.state_dict())
    rng = state_digest(capture_rng_state())
    merged = merge_lora_for_inference(source)
    assert [module.training for module in source.modules()] == modes
    assert [parameter.requires_grad for parameter in source.parameters()] == flags
    assert state_digest(source.state_dict()) == before
    assert state_digest(capture_rng_state()) == rng
    assert all(
        torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in source.parameters()
    )
    assert all(parameter.grad is None for parameter in merged.parameters())
    with torch.no_grad():
        merged.token_embedding.weight.add_(1)
    assert state_digest(source.state_dict()) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_merge_rejects_nonfinite_adapter_without_mutating_source(value):
    source = adapted_model()
    with torch.no_grad():
        source.blocks[0].attention.qkv.lora_b.fill_(value)
    before = source.blocks[0].attention.qkv.lora_b.clone()
    with pytest.raises(FloatingPointError, match="finite"):
        merge_lora_for_inference(source)
    torch.testing.assert_close(source.blocks[0].attention.qkv.lora_b, before, equal_nan=True)


def test_finite_adapters_whose_product_overflows_are_rejected():
    source = adapted_model()
    with torch.no_grad():
        source.blocks[0].attention.qkv.lora_a.fill_(1e30)
        source.blocks[0].attention.qkv.lora_b.fill_(1e30)
    before = state_digest(source.state_dict())
    with pytest.raises(FloatingPointError, match="merged adapter weight"):
        merge_lora_for_inference(source)
    assert state_digest(source.state_dict()) == before


def test_merge_rejects_adapter_on_tied_language_head():
    source = TinyTransformer(
        ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)
    )
    inject_lora(source, rank=2, alpha=4, target_names=("lm_head",))
    before = state_digest(source.state_dict())
    with pytest.raises(ValueError, match="shared base"):
        merge_lora_for_inference(source)
    assert state_digest(source.state_dict()) == before


def test_merge_requires_adapter_model():
    source = TinyTransformer(ModelConfig())
    with pytest.raises(ValueError, match="LoRA"):
        merge_lora_for_inference(source)
    with pytest.raises(ValueError, match="TinyTransformer"):
        merge_lora_for_inference(torch.nn.Linear(4, 4))
