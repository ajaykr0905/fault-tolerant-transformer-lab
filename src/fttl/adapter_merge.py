"""Convert attention adapters into a frozen, standalone local inference model."""

from __future__ import annotations

import copy
from collections import Counter

import torch

from fttl.lora import LoRALinear
from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state


def merge_lora_for_inference(model: TinyTransformer) -> TinyTransformer:
    """Return an independent eval-only clone with adapter updates folded into weights.

    Floating-point operation order changes, so logits match within numerical
    tolerance, not bitwise equality. Shared adapter-base parameters are rejected:
    updating one tied use cannot safely update every other use of that weight.
    The source, its gradients, modes, parameter flags, and caller RNG are untouched.
    This is a local model conversion, not a serving or performance benchmark.
    """
    if not isinstance(model, TinyTransformer):
        raise ValueError("adapter merge requires TinyTransformer")
    adapters = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    if not adapters:
        raise ValueError("adapter merge requires at least one LoRA module")
    require_finite_state(model.state_dict(), "source model state")
    references = Counter(
        id(parameter) for _, parameter in model.named_parameters(remove_duplicate=False)
    )
    merged_weights = {}
    with torch.no_grad():
        for name, module in adapters:
            if any(references[id(parameter)] != 1 for parameter in module.base.parameters()):
                raise ValueError("adapter merge does not support shared base parameters")
            merged = module.base.weight + module.scale * (module.lora_b @ module.lora_a)
            require_finite_state(merged, "merged adapter weight")
            merged_weights[name] = merged
        result = copy.deepcopy(model)
        for name, module in adapters:
            replacement = copy.deepcopy(module.base)
            replacement.weight.copy_(merged_weights[name])
            parent_name, _, child_name = name.rpartition(".")
            parent = result.get_submodule(parent_name) if parent_name else result
            setattr(parent, child_name, replacement)
    result.eval()
    result.requires_grad_(False)
    for parameter in result.parameters():
        parameter.grad = None
    return result
