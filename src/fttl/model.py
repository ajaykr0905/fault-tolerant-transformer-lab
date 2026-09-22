from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fttl.config import ModelConfig


def causal_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    scale = query.size(-1) ** -0.5
    scores = query @ key.transpose(-2, -1) * scale
    sequence = query.size(-2)
    causal_mask = torch.ones(sequence, sequence, dtype=torch.bool, device=query.device).tril()
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    return scores.softmax(dim=-1) @ value


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_size = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, sequence, self.n_heads, self.head_size).transpose(1, 2)

        attended = causal_attention(split_heads(query), split_heads(key), split_heads(value))
        merged = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.dropout(self.output(merged))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        _, sequence = tokens.shape
        if sequence > self.config.block_size:
            raise ValueError("sequence exceeds configured block_size")
        positions = torch.arange(sequence, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            hidden = block(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
