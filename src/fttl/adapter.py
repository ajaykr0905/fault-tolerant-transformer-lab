"""Portable LoRA state bound to an exact frozen local transformer base.

Digests detect accidental corruption and mismatched training inputs, not artifact
authenticity. Only load artifacts from a trusted local store. Disk loading uses
``torch.load(..., weights_only=True)`` and never silently injects LoRA modules.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from fttl.lora import LoRALinear
from fttl.model import TinyTransformer
from fttl.state import state_digest, state_trees_equal

ADAPTER_SCHEMA_VERSION = 1
_MODEL_TYPE = "fttl.model.TinyTransformer"
_PAYLOAD_KEYS = {
    "schema_version",
    "model_type",
    "model_config",
    "base_state_digest",
    "modules",
    "adapter_state_digest",
}
_MODULE_KEYS = {"rank", "scale", "in_features", "out_features", "dtype", "lora_a", "lora_b"}


def _require_keys(value: Any, expected: set[str], label: str) -> None:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a plain dictionary with string keys")
    if set(value) != expected:
        raise ValueError(f"{label} has missing or unexpected fields")


def _tensor_snapshot(value: torch.Tensor, label: str) -> torch.Tensor:
    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError(f"{label} must be a materialized dense tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{label} contains non-finite values")
    return value.detach().to(device="cpu").contiguous().clone()


def _inspect_model(model: TinyTransformer) -> tuple[dict[str, LoRALinear], str]:
    if type(model) is not TinyTransformer:
        raise ValueError("adapter state requires a TinyTransformer")
    modules = {
        name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)
    }
    if not modules:
        raise ValueError("model must already have injected LoRA modules")
    adapter_names = {f"{name}.{field}" for name in modules for field in ("lora_a", "lora_b")}
    adapter_storages: set[tuple[torch.device, int]] = set()
    for name, module in modules.items():
        if type(module) is not LoRALinear or type(module.base) is not torch.nn.Linear:
            raise ValueError(f"{name} must use the supported LoRALinear implementation")
        if (
            tuple(module.base.weight.shape) != (module.base.out_features, module.base.in_features)
            or not module.base.weight.is_floating_point()
            or (
                module.base.bias is not None
                and (
                    tuple(module.base.bias.shape) != (module.base.out_features,)
                    or module.base.bias.dtype != module.base.weight.dtype
                    or module.base.bias.device != module.base.weight.device
                )
            )
        ):
            raise ValueError(f"{name} has an invalid base linear contract")
        if (
            type(module.rank) is not int
            or not 1 <= module.rank <= min(module.base.in_features, module.base.out_features)
            or type(module.scale) is not float
            or not math.isfinite(module.scale)
            or module.scale <= 0
        ):
            raise ValueError(f"{name} has an invalid rank or scale")
        for field, shape in (
            ("lora_a", (module.rank, module.base.in_features)),
            ("lora_b", (module.base.out_features, module.rank)),
        ):
            parameter = getattr(module, field)
            if (
                not isinstance(parameter, torch.nn.Parameter)
                or tuple(parameter.shape) != shape
                or parameter.dtype != module.base.weight.dtype
                or parameter.device != module.base.weight.device
                or not parameter.is_floating_point()
                or not parameter.is_contiguous()
            ):
                raise ValueError(f"{name}.{field} has an invalid shape, dtype, or device")
            _tensor_snapshot(parameter, f"{name}.{field}")
            storage = (parameter.device, parameter.untyped_storage().data_ptr())
            if storage in adapter_storages:
                raise ValueError("adapter parameters must not share storage")
            adapter_storages.add(storage)

    base_state = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name not in adapter_names:
            if parameter.requires_grad:
                raise ValueError(f"base parameter {name} must be frozen")
            snapshot = _tensor_snapshot(parameter, name)
            storage = (parameter.device, parameter.untyped_storage().data_ptr())
            if storage in adapter_storages:
                raise ValueError("base parameters must not share adapter storage")
            base_state[f"parameter:{name}"] = snapshot
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if buffer.requires_grad:
            raise ValueError(f"base buffer {name} must be frozen")
        snapshot = _tensor_snapshot(buffer, name)
        storage = (buffer.device, buffer.untyped_storage().data_ptr())
        if storage in adapter_storages:
            raise ValueError("base buffers must not share adapter storage")
        base_state[f"buffer:{name}"] = snapshot
    return modules, state_digest(base_state)


def _module_state(module: LoRALinear) -> dict[str, Any]:
    return {
        "rank": module.rank,
        "scale": module.scale,
        "in_features": module.base.in_features,
        "out_features": module.base.out_features,
        "dtype": str(module.lora_a.dtype),
        "lora_a": _tensor_snapshot(module.lora_a, "lora_a"),
        "lora_b": _tensor_snapshot(module.lora_b, "lora_b"),
    }


def export_adapter(model: TinyTransformer) -> dict[str, Any]:
    """Return isolated CPU adapter tensors plus weights-only-safe binding metadata.

    No frozen base weights are exported; a loader must already possess exactly
    that base and matching injected module configuration.
    """
    modules, base_digest = _inspect_model(model)
    states = {name: _module_state(module) for name, module in modules.items()}
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "model_type": _MODEL_TYPE,
        "model_config": asdict(model.config),
        "base_state_digest": base_digest,
        "modules": states,
        "adapter_state_digest": state_digest(states),
    }


def load_adapter(model: TinyTransformer, payload: dict[str, Any]) -> None:
    """Validate and stage the entire artifact before copying any adapter values.

    Validation failures leave all model state and ``requires_grad`` flags intact.
    Loading does not restore optimizer state or RNGs, nor change training mode.
    """
    _require_keys(payload, _PAYLOAD_KEYS, "adapter payload")
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != ADAPTER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported adapter schema_version")
    if type(payload["model_type"]) is not str or payload["model_type"] != _MODEL_TYPE:
        raise ValueError("adapter model_type does not match")
    for field in ("base_state_digest", "adapter_state_digest"):
        if type(payload[field]) is not str or re.fullmatch(r"[0-9a-f]{64}", payload[field]) is None:
            raise ValueError(f"invalid {field}")
    modules, base_digest = _inspect_model(model)
    expected_config = asdict(model.config)
    _require_keys(payload["model_config"], set(expected_config), "model_config")
    if not state_trees_equal(payload["model_config"], expected_config):
        raise ValueError("adapter model_config does not match")
    if payload["base_state_digest"] != base_digest:
        raise ValueError("adapter frozen base state does not match")
    _require_keys(payload["modules"], set(modules), "adapter modules")

    staged: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    canonical_states = {}
    for name, module in modules.items():
        state = payload["modules"][name]
        _require_keys(state, _MODULE_KEYS, f"adapter module {name}")
        expected = _module_state(module)
        for field in ("rank", "scale", "in_features", "out_features", "dtype"):
            if not state_trees_equal(state[field], expected[field]):
                raise ValueError(f"adapter module {name} {field} does not match")
        canonical = {
            field: expected[field] for field in expected if field not in ("lora_a", "lora_b")
        }
        for field in ("lora_a", "lora_b"):
            tensor = state[field]
            parameter = getattr(module, field)
            if (
                type(tensor) is not torch.Tensor
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or tensor.device.type != "cpu"
                or tensor.requires_grad
            ):
                raise ValueError(f"adapter module {name}.{field} has an invalid tensor contract")
            snapshot = _tensor_snapshot(tensor, f"adapter module {name}.{field}")
            canonical[field] = snapshot
            staged.append((parameter, snapshot.to(device=parameter.device)))
        canonical_states[name] = canonical
    if payload["adapter_state_digest"] != state_digest(canonical_states):
        raise ValueError("adapter state digest does not match")
    with torch.no_grad():
        for parameter, tensor in staged:
            parameter.copy_(tensor)


def save_adapter(model: TinyTransformer, output: Path) -> None:
    """Publish a complete trusted local artifact once, without replacing evidence.

    Requires same-filesystem hard links. The file is flushed before its final
    name becomes visible. This is not a power-loss proof; a process death may
    leave a private temporary. Digests are not signatures or authorization.
    """
    payload = export_adapter(model)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"adapter output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.tmp-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_adapter_file(model: TinyTransformer, source: Path) -> None:
    """Load a trusted local adapter using PyTorch's restricted weights-only loader."""
    payload = torch.load(Path(source), map_location="cpu", weights_only=True)
    load_adapter(model, payload)
