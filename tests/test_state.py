import random

import numpy as np
import torch

from fttl.state import capture_rng_state, restore_rng_state, state_digest, state_trees_equal


def test_rng_state_round_trip_replays_python_numpy_and_torch():
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    state = capture_rng_state()

    expected = (random.random(), float(np.random.random()), torch.rand(3))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_state_digest_is_stable_and_sensitive_to_tensor_bytes():
    first = {"b": [1, 2], "a": torch.tensor([[1.0, 2.0]])}
    reordered = {"a": torch.tensor([[1.0, 2.0]]), "b": [1, 2]}
    changed = {"a": torch.tensor([[1.0, 3.0]]), "b": [1, 2]}

    assert state_digest(first) == state_digest(reordered)
    assert state_digest(first) != state_digest(changed)
    assert state_trees_equal(first, reordered)
    assert not state_trees_equal(first, changed)


def test_state_digest_supports_scalar_optimizer_tensors():
    assert state_digest(torch.tensor(2.0)) == state_digest(torch.tensor(2.0))
    assert state_digest(torch.tensor(2.0)) != state_digest(torch.tensor(3.0))
