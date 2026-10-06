import hashlib
import json
from collections.abc import Mapping, Sequence

import pytest
import torch

from fttl.state import state_digest


@pytest.mark.parametrize("key, string", [(1, "1"), (True, "True"), (1.5, "1.5"), (None, "None")])
def test_primitive_mapping_keys_do_not_collide_with_their_string_forms(key, string):
    assert state_digest({key: "value"}) != state_digest({string: "value"})


@pytest.mark.parametrize("left, right", [(True, 1), (False, 0), (1, 1.0), (False, 0.0)])
def test_numeric_mapping_key_types_have_distinct_identities(left, right):
    assert state_digest({left: "value"}) != state_digest({right: "value"})


def test_mixed_primitive_keys_are_independent_of_insertion_order():
    pairs = [(1, "integer"), ("1", "string"), (None, "null"), (False, "boolean"), (2.5, "float")]
    assert state_digest(dict(pairs)) == state_digest(dict(reversed(pairs)))


def test_nested_optimizer_keys_are_typed_and_order_independent():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    first = {"state": {7: {"moment": tensor}, "7": {"moment": tensor + 1}}}
    reordered = {"state": {"7": {"moment": tensor + 1}, 7: {"moment": tensor}}}
    swapped = {"state": {"7": {"moment": tensor}, 7: {"moment": tensor + 1}}}
    assert state_digest(first) == state_digest(reordered)
    assert state_digest(first) != state_digest(swapped)


@pytest.mark.parametrize("key", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_mapping_keys_are_rejected(key):
    with pytest.raises(ValueError, match="mapping keys.*finite"):
        state_digest({key: "value"})


@pytest.mark.parametrize("key", [(1, 2), frozenset({1}), object(), torch.tensor(1)])
def test_unsupported_mapping_keys_are_rejected_without_string_coercion(key):
    with pytest.raises(TypeError, match="mapping keys"):
        state_digest({key: "value"})


def test_custom_primitive_subclass_is_not_coerced():
    class CustomInteger(int):
        def __str__(self):
            raise AssertionError("custom key string conversion must not run")

    with pytest.raises(TypeError, match="mapping keys"):
        state_digest({CustomInteger(2): "value"})


def _legacy_string_key_digest(value):
    """Reference the pre-typed-key encoding used by published string-key evidence."""
    hasher = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            hasher.update(b"tensor\0")
            hasher.update(str(tensor.dtype).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            hasher.update(b"mapping{")
            assert all(type(key) is str for key in item)
            for key in sorted(item):
                update(key)
                update(item[key])
            hasher.update(b"}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            hasher.update(b"sequence[")
            for child in item:
                update(child)
            hasher.update(b"]")
        else:
            hasher.update(
                json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            )
            hasher.update(b"\0")

    update(value)
    return hasher.hexdigest()


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"z": False, "a": [1, 2.5, None]},
        {"é": "unicode", "z": "ASCII", 'quote"': {"line\n": "newline"}},
        {"model": {"bias": torch.tensor(2.0), "weight": torch.arange(6).reshape(2, 3).T}},
    ],
)
def test_existing_all_string_mapping_digests_remain_byte_compatible(value):
    assert state_digest(value) == _legacy_string_key_digest(value)


def test_tensor_values_remain_sensitive_under_integer_keys():
    assert state_digest({1: torch.tensor([1.0])}) != state_digest({1: torch.tensor([2.0])})
