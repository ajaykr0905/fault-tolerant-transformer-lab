from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def require_finite_state(value: Any, label: str) -> None:
    """Reject non-finite numerical values before publishing training state."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} contains non-finite tensor values")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            require_finite_state(item, f"{label}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            require_finite_state(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError(f"{label} contains non-finite scalar values")
