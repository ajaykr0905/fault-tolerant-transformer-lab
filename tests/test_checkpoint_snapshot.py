import io
import os

import pytest
import torch

import fttl.checkpoint as checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def _setup(marker):
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)
    )
    model = TinyTransformer(config.model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(marker)
    return config, model, torch.optim.AdamW(model.parameters())


def _save(store, marker):
    config, model, optimizer = _setup(marker)
    return checkpoint.save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=1,
        tokens_seen=8,
        losses=[1.0],
    )


def _replacement(state):
    payload = torch.load(state, weights_only=True)
    for tensor in payload["model"].values():
        tensor.fill_(9.0)
    stream = io.BytesIO()
    torch.save(payload, stream)
    replacement = stream.getvalue()
    assert len(replacement) == state.stat().st_size
    return replacement


@pytest.mark.parametrize("mutation", ["replace", "in-place", "unlink"])
def test_path_mutation_after_capture_never_installs_unverified_weights(
    tmp_path, monkeypatch, mutation
):
    store = tmp_path / "store"
    _save(store, 1.0)
    state = store / "generation-00000001" / "state.pt"
    replacement = _replacement(state)
    config, target, optimizer = _setup(3.0)
    before = {name: tensor.clone() for name, tensor in target.state_dict().items()}
    rng = capture_rng_state()
    original_load = torch.load

    def mutate_then_decode(source, **kwargs):
        assert kwargs["weights_only"] is True
        if mutation == "replace":
            other = state.with_name("replacement.pt")
            other.write_bytes(replacement)
            os.replace(other, state)
        elif mutation == "in-place":
            state.write_bytes(replacement)
        else:
            state.unlink()
        return original_load(source, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", mutate_then_decode)
    try:
        payload = checkpoint.load_checkpoint(
            store, model=target, optimizer=optimizer, expected_config=config, restore_rng=False
        )
    except checkpoint.CheckpointIntegrityError:
        assert state_trees_equal(before, target.state_dict())
    else:
        assert float(target.token_embedding.weight[0, 0].detach()) == 1.0
        assert float(payload["model"]["token_embedding.weight"][0, 0]) == 1.0
    assert state_trees_equal(rng, capture_rng_state())


def test_decoder_receives_exact_verified_byte_snapshot(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _save(store, 1.0)
    committed = (store / "generation-00000001" / "state.pt").read_bytes()
    original_load = torch.load
    observed = []

    def inspect_decode(source, **kwargs):
        assert isinstance(source, io.BytesIO)
        assert source.getvalue() == committed
        observed.append(True)
        return original_load(source, **kwargs)

    monkeypatch.setattr(checkpoint.torch, "load", inspect_decode)
    config, model, optimizer = _setup(3.0)
    checkpoint.load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert observed == [True]


def test_replacement_after_initial_hash_is_revalidated_and_falls_back(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _save(store, 1.0)
    _save(store, 2.0)
    state = store / "generation-00000002" / "state.pt"
    replacement = _replacement(state)
    original_hash = checkpoint._sha256

    def swap_after_hash(path):
        digest = original_hash(path)
        if path == state:
            path.write_bytes(replacement)
        return digest

    monkeypatch.setattr(checkpoint, "_sha256", swap_after_hash)
    config, model, optimizer = _setup(3.0)
    payload = checkpoint.load_checkpoint(
        store, model=model, optimizer=optimizer, expected_config=config, restore_rng=False
    )
    assert payload["selected_generation"] == 1
    assert float(model.token_embedding.weight[0, 0].detach()) == 1.0


def test_initial_corruption_rejects_before_decode_and_receiver_changes(tmp_path, monkeypatch):
    store = tmp_path / "store"
    _save(store, 1.0)
    state = store / "generation-00000001" / "state.pt"
    damaged = bytearray(state.read_bytes())
    damaged[-1] ^= 1
    state.write_bytes(damaged)
    config, model, optimizer = _setup(3.0)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    rng = capture_rng_state()
    monkeypatch.setattr(
        checkpoint.torch, "load", lambda *args, **kwargs: pytest.fail("decoded corruption")
    )
    with pytest.raises(checkpoint.CheckpointIntegrityError, match="SHA-256"):
        checkpoint.load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert state_trees_equal(before, model.state_dict())
    assert state_trees_equal(rng, capture_rng_state())
