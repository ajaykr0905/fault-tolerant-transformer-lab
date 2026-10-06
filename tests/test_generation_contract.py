import random
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch
from test_generation import _model

from fttl.generation import MAX_PROMPT_TOKENS, generate_tokens
from fttl.state import capture_rng_state, state_trees_equal


class ChangingPrompt(Sequence):
    def __init__(self, count, declared=1):
        self.count = count
        self.declared = declared
        self.reads = 0

    def __len__(self):
        return self.declared

    def __getitem__(self, index):
        if index >= self.count:
            raise IndexError
        self.reads += 1
        return 1


@pytest.mark.parametrize("count", [0, 2, MAX_PROMPT_TOKENS + 1])
def test_actual_captured_prompt_rejects_empty_oversized_or_inconsistent_sequence(
    count, monkeypatch
):
    model = _model()
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("forward on invalid prompt"))
    with pytest.raises(ValueError, match="prompt"):
        generate_tokens(model, ChangingPrompt(count), max_new_tokens=0)


def test_infinite_sequence_iteration_is_bounded_before_rejection(monkeypatch):
    model = _model()
    prompt = ChangingPrompt(float("inf"))
    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("forward on invalid prompt"))
    with pytest.raises(ValueError, match="prompt"):
        generate_tokens(model, prompt, max_new_tokens=0)
    assert prompt.reads == MAX_PROMPT_TOKENS + 1


@pytest.mark.parametrize("fail", [False, True])
def test_custom_prompt_capture_preserves_all_caller_rngs(fail):
    class RandomPrompt(ChangingPrompt):
        def __getitem__(self, index):
            random.random()
            np.random.random()
            torch.rand(1)
            return super().__getitem__(index)

    model = _model()
    before = capture_rng_state()
    prompt = RandomPrompt(2 if fail else 1)
    if fail:
        with pytest.raises(ValueError, match="prompt"):
            generate_tokens(model, prompt, max_new_tokens=0)
    else:
        generate_tokens(model, prompt, max_new_tokens=0)
    assert state_trees_equal(before, capture_rng_state())


@pytest.mark.parametrize("factory", [bytes, list, tuple])
@pytest.mark.parametrize("length", [1, MAX_PROMPT_TOKENS])
def test_ordinary_prompt_boundary_capture_remains_compatible(factory, length):
    result = generate_tokens(_model(), factory([1] * length), max_new_tokens=0)
    assert result.prompt_token_ids == (1,) * length
    assert result.generated_count == 0


@pytest.mark.parametrize("step", ["eval", "first", "last"])
def test_configuration_drift_never_returns_success_and_restores_rng_and_modes(step, monkeypatch):
    model = _model().train()
    model.blocks[0].attention.eval()
    modes = [module.training for module in model.modules()]
    count = []

    def drift():
        random.random()
        np.random.random()
        torch.rand(1)
        model.config = replace(model.config, block_size=2)

    if step == "eval":
        original_eval = model.eval

        def changed_eval():
            original_eval()
            drift()
            return model

        monkeypatch.setattr(model, "eval", changed_eval)
        new_tokens = 0
    else:
        new_tokens = 2

        def hook(module, args, output):
            count.append(args[0].shape[1])
            if len(count) == (1 if step == "first" else 2):
                drift()

        model.register_forward_hook(hook)
    before = capture_rng_state()
    with pytest.raises(ValueError, match="config.*changed"):
        generate_tokens(model, [1, 2, 3], max_new_tokens=new_tokens)
    assert state_trees_equal(before, capture_rng_state())
    assert [module.training for module in model.modules()] == modes
    assert model.config.block_size == 2  # Detection is not arbitrary callback rollback.
    if step == "first":
        assert count == [3]  # Never execute a second token under a changed contract.


def test_prompt_callback_cannot_rebind_the_initial_configuration(monkeypatch):
    model = _model().train()
    model.blocks[0].attention.eval()
    modes = [module.training for module in model.modules()]

    class ConfigPrompt(ChangingPrompt):
        def __getitem__(self, index):
            random.random()
            np.random.random()
            torch.rand(1)
            model.config = replace(model.config, block_size=2)
            return super().__getitem__(index)

    monkeypatch.setattr(model, "forward", lambda _: pytest.fail("forward after prompt drift"))
    before = capture_rng_state()
    with pytest.raises(ValueError, match="config.*changed"):
        generate_tokens(model, ConfigPrompt(1), max_new_tokens=0)
    assert state_trees_equal(before, capture_rng_state())
    assert [module.training for module in model.modules()] == modes
    assert model.config.block_size == 2


def test_final_state_dict_hook_cannot_change_configuration_before_success():
    model = _model().train()
    model.blocks[0].attention.eval()
    modes = [module.training for module in model.modules()]
    calls = []

    def change_final_config(module, state, prefix, metadata):
        calls.append(True)
        if len(calls) == 2:
            random.random()
            np.random.random()
            torch.rand(1)
            module.config = replace(module.config, block_size=2)

    model.register_state_dict_post_hook(change_final_config)
    before = capture_rng_state()
    with pytest.raises(ValueError, match="config.*changed"):
        generate_tokens(model, [1], max_new_tokens=1)
    assert calls == [True, True]
    assert state_trees_equal(before, capture_rng_state())
    assert [module.training for module in model.modules()] == modes
    assert model.config.block_size == 2
