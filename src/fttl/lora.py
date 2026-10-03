from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _validate_lora(base: nn.Linear, rank: int, alpha: float) -> None:
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or rank < 1
        or rank > min(base.in_features, base.out_features)
    ):
        raise ValueError("rank must be an integer fitting the base linear dimensions")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool):
        raise ValueError("alpha must be finite and positive")
    try:
        valid_alpha = math.isfinite(alpha) and alpha > 0
    except OverflowError:
        valid_alpha = False
    if not valid_alpha:
        raise ValueError("alpha must be finite and positive")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank update."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        _validate_lora(base, rank, alpha)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = (inputs @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return self.base(inputs) + update * self.scale


@dataclass(frozen=True)
class LoRASummary:
    replaced_modules: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    target_names: tuple[str, ...] = ("qkv", "output"),
) -> LoRASummary:
    if (
        not isinstance(target_names, tuple)
        or not target_names
        or any(not isinstance(name, str) or not name for name in target_names)
    ):
        raise ValueError("target_names must be a nonempty tuple of nonempty module names")

    targets = [
        (module_path, module)
        for module_path, module in model.named_modules()
        if isinstance(module, nn.Linear) and module_path.split(".")[-1] in target_names
    ]
    if not targets:
        raise ValueError("no target linear modules were found")
    for _, module in targets:
        _validate_lora(module, rank, alpha)

    for parameter in model.parameters():
        parameter.requires_grad = False

    replaced: list[str] = []
    for module_path, module in targets:
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha))
        replaced.append(module_path)

    return LoRASummary(
        replaced_modules=tuple(replaced),
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    )
