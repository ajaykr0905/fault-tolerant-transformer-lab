"""Selected-device CUDA controls for the bounded eager FP32 recovery experiment."""

from __future__ import annotations

import os
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from fttl.state import capture_rng_state, restore_rng_state

_prepared_workspace: str | None = None


def _selected_device(device: torch.device) -> torch.device:
    selected = torch.device(device)
    if selected != torch.device("cuda:0"):
        raise ValueError("this experiment requires the explicit selected device cuda:0")
    return selected


def prepare_cuda_execution() -> tuple[torch.device, dict[str, Any]]:
    """Require a real CUDA device and configure deterministic, eager FP32 execution.

    The notebook must set CUBLAS_WORKSPACE_CONFIG before initializing CUDA.
    No CPU fallback or cross-device reproducibility claim is made.
    """
    global _prepared_workspace
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError("set CUBLAS_WORKSPACE_CONFIG=:4096:8 before initializing CUDA")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("CUDA experiment requires the default dtype torch.float32")
    if _prepared_workspace is None and torch.cuda.is_initialized():
        raise RuntimeError("CUDA was already initialized; run this module in a fresh subprocess")
    if _prepared_workspace is not None and (
        workspace != _prepared_workspace
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.get_float32_matmul_precision() != "highest"
    ):
        raise RuntimeError("prepared CUDA execution policy changed; use a fresh subprocess")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable; this experiment does not fall back to CPU")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(device)
    metadata = {
        "device": str(device),
        "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cublas_workspace_config": workspace,
        "precision": "float32",
        "initialization_dtype": str(torch.get_default_dtype()),
        "execution": "eager",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "optimizer_policy": "AdamW foreach=False fused=False",
    }
    _prepared_workspace = workspace
    return device, metadata


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(item) for item in value) if isinstance(value, list) else value


def validate_cpu_rng_state(state: Mapping[str, Any]) -> None:
    """Validate checkpoint CPU RNGs with isolated candidates, without restoring them."""
    if not isinstance(state, Mapping):
        raise ValueError("RNG state must be a mapping")
    try:
        random.Random().setstate(_tuple_tree(state["python"]))
        numpy_state = state["numpy"]
        np.random.RandomState().set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        tensor = state["torch_cpu"]
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            raise ValueError("CPU Torch RNG state must be a CPU tensor")
        torch.Generator(device="cpu").set_state(tensor)
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"invalid CPU RNG state: {error}") from error


def validate_cuda_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    """Validate CPU and selected CUDA RNG states using independent generators."""
    device = _selected_device(device)
    validate_cpu_rng_state(state)
    try:
        cuda_state = state["torch_cuda"]
        if not isinstance(cuda_state, Mapping) or set(cuda_state) != {
            "schema_version",
            "device_index",
            "state",
        }:
            raise ValueError("CUDA RNG declaration must contain the exact versioned fields")
        if type(cuda_state["schema_version"]) is not int or cuda_state["schema_version"] != 1:
            raise ValueError("unsupported CUDA RNG schema")
        if type(cuda_state["device_index"]) is not int or cuda_state["device_index"] != 0:
            raise ValueError("CUDA RNG state is not for selected device 0")
        tensor = cuda_state["state"]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
        ):
            raise ValueError("CUDA RNG state must be a one-dimensional CPU ByteTensor")
        torch.Generator(device=device).set_state(tensor)
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"invalid selected CUDA RNG state: {error}") from error


def capture_cuda_rng_state(device: torch.device) -> dict[str, Any]:
    device = _selected_device(device)
    state = capture_rng_state()
    state["torch_cuda"] = {
        "schema_version": 1,
        "device_index": 0,
        "state": torch.cuda.get_rng_state(device).cpu().clone(),
    }
    return state


def restore_cuda_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    """Restore all selected RNGs together, rolling back if a setter fails."""
    device = _selected_device(device)
    validate_cuda_rng_state(state, device)
    original = capture_cuda_rng_state(device)
    try:
        restore_rng_state(state)
        torch.cuda.set_rng_state(state["torch_cuda"]["state"], device)
    except Exception:
        restore_rng_state(original)
        torch.cuda.set_rng_state(original["torch_cuda"]["state"], device)
        raise


def synchronize_cuda(device: torch.device) -> None:
    torch.cuda.synchronize(_selected_device(device))
