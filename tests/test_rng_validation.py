import copy
import io
import random

import numpy as np
import pytest
import torch

from fttl.cuda_runtime import validate_cpu_rng_state
from fttl.state import capture_rng_state, restore_rng_state, state_trees_equal


@pytest.fixture(autouse=True)
def preserve_caller_rngs():
    previous = capture_rng_state()
    try:
        yield
    finally:
        restore_rng_state(previous)


@pytest.mark.parametrize("operation", [restore_rng_state, validate_cpu_rng_state])
@pytest.mark.parametrize(
    "path, value",
    [
        (("extra",), 1),
        (("python",), [True, (), None]),
        (("python",), [3, [False] * 624 + [624], None]),
        (("python",), [3, [2**32] * 624 + [624], None]),
        (("python",), [3, [-1] * 624 + [624], None]),
        (("python",), [3, [1] * 624 + [625], None]),
        (("python",), [3, [1] * 624 + [True], None]),
        (("python",), [3, [1] * 624 + [624], float("nan")]),
        (("python",), [3, [1] * 624 + [624], float("inf")]),
        (("python",), [3, [1] * 624 + [624], True]),
        (("numpy", "extra"), 1),
        (("numpy", "position"), 1.75),
        (("numpy", "position"), "1"),
        (("numpy", "position"), True),
        (("numpy", "position"), -1),
        (("numpy", "position"), 625),
        (("numpy", "has_gauss"), 1.5),
        (("numpy", "has_gauss"), "1"),
        (("numpy", "has_gauss"), True),
        (("numpy", "has_gauss"), 2),
        (("numpy", "cached_gaussian"), "0.0"),
        (("numpy", "cached_gaussian"), True),
        (("numpy", "cached_gaussian"), float("nan")),
        (("numpy", "cached_gaussian"), float("inf")),
        (("numpy", "keys"), [1.5] * 624),
        (("numpy", "keys"), [True] * 624),
        (("numpy", "keys"), ["1"] * 624),
        (("numpy", "keys"), [-1] * 624),
        (("numpy", "keys"), [2**32] * 624),
        (("numpy", "keys"), [1] * 623),
        (("torch_cpu",), torch.zeros(1, dtype=torch.uint8)),
        (("torch_cpu",), torch.zeros(5056, dtype=torch.float32)),
        (("torch_cpu",), torch.zeros(5056, dtype=torch.uint8, device="meta")),
    ],
)
def test_malformed_cpu_state_rejects_before_any_global_rng_change(operation, path, value):
    invalid = copy.deepcopy(capture_rng_state())
    if path == ("python",):
        invalid["python"] = value
    elif len(path) == 1:
        invalid[path[0]] = value
    else:
        invalid[path[0]][path[1]] = value
    before = capture_rng_state()
    with pytest.raises(ValueError, match="CPU RNG"):
        operation(invalid)
    assert state_trees_equal(before, capture_rng_state())


@pytest.mark.parametrize("operation", [restore_rng_state, validate_cpu_rng_state])
@pytest.mark.parametrize("value", [None, [], {}, {"python": []}])
def test_incomplete_cpu_state_is_rejected_consistently(operation, value):
    before = capture_rng_state()
    with pytest.raises(ValueError, match="CPU RNG"):
        operation(value)
    assert state_trees_equal(before, capture_rng_state())


@pytest.mark.parametrize("python_form", ["captured", "tuple", "nested-list"])
def test_weights_only_snapshot_forms_replay_python_numpy_and_torch_exactly(python_form):
    random.seed(701)
    np.random.seed(702)
    torch.random.default_generator.manual_seed(703)
    random.gauss(0, 1)
    np.random.normal()
    state = capture_rng_state()
    if python_form == "tuple":
        state["python"] = tuple(state["python"])
    elif python_form == "nested-list":
        state["python"][1] = list(state["python"][1])
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    decoded = torch.load(buffer, weights_only=True)
    expected = (random.gauss(0, 1), np.random.normal(), torch.rand(3))
    before = capture_rng_state()
    validate_cpu_rng_state(decoded)
    assert state_trees_equal(before, capture_rng_state())
    restore_rng_state(decoded)
    actual = (random.gauss(0, 1), np.random.normal(), torch.rand(3))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_cpu_validator_allows_selected_cuda_extension_without_querying_hardware(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CPU validator queried CUDA")
    )
    state = capture_rng_state()
    state["torch_cuda"] = {"validated_separately": True}
    before = capture_rng_state()
    validate_cpu_rng_state(state)
    restore_rng_state(state)
    assert state_trees_equal(before, capture_rng_state())
