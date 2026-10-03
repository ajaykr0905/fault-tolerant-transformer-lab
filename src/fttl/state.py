from __future__ import annotations

import hashlib
import json
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
    """Validate every CPU RNG state before changing any global generator."""
    python_state = _tuple_tree(state["python"])
    numpy_state = state["numpy"]
    numpy_candidate = (
        str(numpy_state["bit_generator"]),
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    )
    random.Random().setstate(python_state)
    np.random.RandomState().set_state(numpy_candidate)
    torch.Generator(device="cpu").set_state(state["torch_cpu"])
    random.setstate(python_state)
    np.random.set_state(numpy_candidate)
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
