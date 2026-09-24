from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
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


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(_tuple_tree(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])


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
        for key in sorted(value, key=lambda item: str(item)):
            _update_digest(hasher, str(key))
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
