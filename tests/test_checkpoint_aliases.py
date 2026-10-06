import copy
import hashlib
import json

import pytest
import torch

from fttl.checkpoint import CheckpointIntegrityError, load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def _setup(tied=True, buffers=False):
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)
    )
    model = TinyTransformer(config.model)
    if not tied:
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    if buffers:
        shared = torch.ones(2)
        model.register_buffer("first_control", shared)
        model.register_buffer("second_control", shared)
    return config, model, torch.optim.AdamW(model.parameters())


def _save(store, config, model, optimizer):
    return save_checkpoint(
        store, model=model, optimizer=optimizer, config=config, step=1, tokens_seen=8, losses=[1.0]
    )


def _rewrite(store, names, values):
    generation = store / (store / "LATEST").read_text().strip()
    path = generation / "state.pt"
    payload = torch.load(path, weights_only=True)
    for name, value in zip(names, values, strict=True):
        payload["model"][name] = torch.full_like(payload["model"][name], value)
    torch.save(payload, path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), state_bytes=path.stat().st_size
    )
    manifest_path.write_text(json.dumps(manifest))
    return payload


@pytest.mark.parametrize("buffers", [False, True])
def test_conflicting_aliases_fall_back_before_receiver_mutation(tmp_path, buffers):
    config, source, optimizer = _setup(buffers=buffers)
    store = tmp_path / "store"
    _save(store, config, source, optimizer)
    valid = copy.deepcopy(source.state_dict())
    _save(store, config, source, optimizer)
    names = (
        ("first_control", "second_control")
        if buffers
        else ("token_embedding.weight", "lm_head.weight")
    )
    _rewrite(store, names, [3, 7])
    _, receiver, receiver_optimizer = _setup(buffers=buffers)
    rng = capture_rng_state()
    result = load_checkpoint(
        store,
        model=receiver,
        optimizer=receiver_optimizer,
        expected_config=config,
        restore_rng=False,
    )
    assert result["selected_generation"] == 1
    assert state_trees_equal(receiver.state_dict(), valid)
    assert receiver.token_embedding.weight is receiver.lm_head.weight
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("legacy", [False, True])
def test_only_bad_alias_payload_rejects_without_model_optimizer_rng_changes(tmp_path, legacy):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, model, optimizer)
    payload = _rewrite(store, ("token_embedding.weight", "lm_head.weight"), [3, 7])
    if legacy:
        payload["schema_version"] = 1
        store = tmp_path / "legacy.pt"
        torch.save(payload, store)
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    rng = capture_rng_state()
    with pytest.raises(CheckpointIntegrityError, match="alias"):
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert state_trees_equal(before, (model.state_dict(), optimizer.state_dict()))
    assert model.token_embedding.weight is model.lm_head.weight
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("tied,values", [(True, [3, 3]), (False, [3, 7])])
def test_equal_distinct_storage_aliases_and_untied_models_remain_supported(tmp_path, tied, values):
    config, model, optimizer = _setup(tied=tied)
    store = tmp_path / "store"
    _save(store, config, model, optimizer)
    payload = _rewrite(store, ("token_embedding.weight", "lm_head.weight"), values)
    result = load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert result["selected_generation"] == 1
    assert state_trees_equal(payload["model"], model.state_dict())
    assert (model.token_embedding.weight is model.lm_head.weight) is tied


def test_save_rejects_custom_export_with_conflicting_declared_aliases_before_writes(
    tmp_path, monkeypatch
):
    config, model, optimizer = _setup()
    export = model.state_dict()
    export["token_embedding.weight"] = torch.full_like(export["token_embedding.weight"], 3)
    export["lm_head.weight"] = torch.full_like(export["lm_head.weight"], 7)
    monkeypatch.setattr(model, "state_dict", lambda: export)
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="alias"):
        _save(store, config, model, optimizer)
    assert not store.exists()
