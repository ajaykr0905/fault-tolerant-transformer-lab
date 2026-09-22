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
