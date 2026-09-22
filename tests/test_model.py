import torch

from fttl.config import ModelConfig
from fttl.model import TinyTransformer, causal_attention


def test_model_shapes_and_tied_embeddings():
    config = ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config)
    tokens = torch.arange(16).reshape(2, 8) % config.vocab_size
    logits, loss = model(tokens, tokens)
    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


def test_causal_attention_passes_autograd_gradient_check():
    torch.manual_seed(4)
    query = torch.randn(1, 1, 3, 2, dtype=torch.double, requires_grad=True)
    key = torch.randn(1, 1, 3, 2, dtype=torch.double, requires_grad=True)
    value = torch.randn(1, 1, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(causal_attention, (query, key, value), eps=1e-6, atol=1e-4)


def test_future_tokens_do_not_change_past_logits():
    torch.manual_seed(9)
    config = ModelConfig(vocab_size=32, block_size=6, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 9, 10, 11]])
    with torch.no_grad():
        first_logits, _ = model(first)
        second_logits, _ = model(second)
    torch.testing.assert_close(first_logits[:, :3], second_logits[:, :3], rtol=0, atol=0)
