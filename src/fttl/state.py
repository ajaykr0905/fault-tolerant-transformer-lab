from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch


@lru_cache(maxsize=1)
def code_fingerprint() -> str:
    """Hash the installed Python implementation used by the run."""

    package_dir = Path(__file__).resolve().parent
    hasher = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py"), key=lambda item: item.name):
        hasher.update(path.name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


@lru_cache(maxsize=1)
def git_revision() -> str:
    """Return a public-safe source revision, marking tracked local edits."""

    repository = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{revision}-dirty" if dirty else revision


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_tuple_tree(item) for item in value)
    return value


def capture_rng_state() -> dict[str, Any]:
    """Capture CPU RNGs using only weights-only-safe primitive and tensor values."""
    numpy_state = np.random.get_state()
    return {
        "python": list(random.getstate()),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }


def _cpu_rng_candidates(state: Mapping[str, Any]) -> tuple[Any, Any, torch.Tensor]:
    try:
        required = {"python", "numpy", "torch_cpu"}
        if (
            not isinstance(state, Mapping)
            or not required.issubset(state)
            or set(state).difference(required | {"torch_cuda"})
        ):
            raise ValueError("state must declare the CPU fields and optional selected CUDA state")
        python_state = _tuple_tree(state["python"])
        if (
            not isinstance(python_state, tuple)
            or len(python_state) != 3
            or type(python_state[0]) is not int
            or python_state[0] != 3
            or not isinstance(python_state[1], tuple)
            or len(python_state[1]) != 625
            or any(type(word) is not int or not 0 <= word < 2**32 for word in python_state[1][:-1])
            or type(python_state[1][-1]) is not int
            or not 0 <= python_state[1][-1] <= 624
            or (
                python_state[2] is not None
                and (type(python_state[2]) is not float or not math.isfinite(python_state[2]))
            )
        ):
            raise ValueError("Python state must contain version 3, MT19937 words, and finite cache")
        numpy_state = state["numpy"]
        if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
            "bit_generator",
            "keys",
            "position",
            "has_gauss",
            "cached_gaussian",
        }:
            raise ValueError("NumPy state fields do not match MT19937")
        keys = numpy_state["keys"]
        if (
            type(numpy_state["bit_generator"]) is not str
            or numpy_state["bit_generator"] != "MT19937"
            or not isinstance(keys, (list, tuple))
            or len(keys) != 624
            or any(type(word) is not int or not 0 <= word < 2**32 for word in keys)
            or type(numpy_state["position"]) is not int
            or not 0 <= numpy_state["position"] <= 624
            or type(numpy_state["has_gauss"]) is not int
            or numpy_state["has_gauss"] not in (0, 1)
            or type(numpy_state["cached_gaussian"]) is not float
            or not math.isfinite(numpy_state["cached_gaussian"])
        ):
            raise ValueError(
                "NumPy state must contain unsigned words, integer counters, finite cache"
            )
        numpy_candidate = (
            numpy_state["bit_generator"],
            np.asarray(keys, dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
        tensor = state["torch_cpu"]
        if (
            type(tensor) is not torch.Tensor
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.dtype != torch.uint8
            or tensor.ndim != 1
            or not tensor.is_contiguous()
        ):
            raise ValueError("Torch state must be a contiguous one-dimensional CPU ByteTensor")
        tensor = tensor.clone()
        random.Random().setstate(python_state)
        np.random.RandomState().set_state(numpy_candidate)
        torch.Generator(device="cpu").set_state(tensor)
        return python_state, numpy_candidate, tensor
    except (KeyError, TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError(f"invalid CPU RNG state: {error}") from error


def validate_cpu_rng_state(state: Mapping[str, Any]) -> None:
    """Validate primitive CPU checkpoint states without changing global generators."""
    _cpu_rng_candidates(state)


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Validate every CPU RNG state before changing any global generator."""
    python_state, numpy_candidate, tensor = _cpu_rng_candidates(state)
    random.setstate(python_state)
    np.random.set_state(numpy_candidate)
    torch.set_rng_state(tensor)


def _mapping_key_order(key: Any) -> tuple[str, Any]:
    key_type = type(key)
    if key_type is str:
        return "str", key
    if key is None:
        return "null", 0
    if key_type is bool:
        return "bool", key
    if key_type is int:
        return "int", key
    if key_type is float:
        if not math.isfinite(key):
            raise ValueError("state mapping keys must be finite")
        return "float", key
    raise TypeError("state mapping keys must be primitive strings, numbers, booleans, or null")


def _update_digest(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"tensor\0")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        hasher.update(b"\0")
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping{")
        for key in sorted(value, key=_mapping_key_order):
            if type(key) is not str:
                # String keys retain their historical encoding; other primitives
                # receive an unambiguous type tag before their JSON scalar bytes.
                hasher.update(f"{_mapping_key_order(key)[0]}-key\0".encode("ascii"))
            _update_digest(hasher, key)
            _update_digest(hasher, value[key])
        hasher.update(b"}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        hasher.update(b"sequence[")
        for item in value:
            _update_digest(hasher, item)
        hasher.update(b"]")
        return
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    hasher.update(encoded.encode("utf-8"))
    hasher.update(b"\0")


def state_digest(value: Any) -> str:
    hasher = hashlib.sha256()
    _update_digest(hasher, value)
    return hasher.hexdigest()


def state_trees_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            state_trees_equal(left[key], right[key]) for key in left
        )
    if (
        isinstance(left, Sequence)
        and isinstance(right, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
        and not isinstance(right, (str, bytes, bytearray))
    ):
        return len(left) == len(right) and all(
            state_trees_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return type(left) is type(right) and left == right
