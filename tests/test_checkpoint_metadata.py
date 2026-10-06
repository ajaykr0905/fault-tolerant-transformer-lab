import copy
import hashlib
import json
from dataclasses import replace

import pytest
import torch

from fttl.checkpoint import (
    CheckpointIntegrityError,
    CheckpointManifestV2,
    CheckpointMismatchError,
    load_checkpoint,
    save_checkpoint,
)
from fttl.config import ExperimentConfig
from fttl.state import capture_rng_state, state_trees_equal


def _setup():
    config = ExperimentConfig()
    model = torch.nn.Linear(2, 2)
    return config, model, torch.optim.AdamW(model.parameters())


def _save(store, config, model, optimizer, **kwargs):
    return save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=kwargs.pop("step", 1),
        tokens_seen=kwargs.pop("tokens_seen", 8),
        losses=[1.0],
        **kwargs,
    )


def _rewrite(store, field, value, *, manifest_only=False):
    generation = store / (store / "LATEST").read_text().strip()
    path = generation / "state.pt"
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest_only:
        manifest[field] = value
    else:
        payload = torch.load(path, weights_only=True)
        payload[field] = value
        torch.save(payload, path)
        manifest["state_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["state_bytes"] = path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))


_CURSOR = {
    "schema_version": 1,
    "epoch": 0,
    "document_id": "public",
    "token_window_offset": 0,
    "batch_id": "batch-1",
    "next_sample_ids": ["sample-1"],
    "batch_index": 1,
}
_BAD_PAYLOAD = [("step", True), ("tokens_seen", 8.0), ("schema_version", 2.0)]


@pytest.mark.parametrize("field,value", _BAD_PAYLOAD)
def test_scalar_aliases_fall_back_to_valid_predecessor(tmp_path, field, value):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, model, optimizer)
    valid = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight.fill_(9)
    _save(store, config, model, optimizer)
    _rewrite(store, field, value)
    rng = capture_rng_state()
    result = load_checkpoint(
        store, model=model, optimizer=optimizer, expected_config=config, restore_rng=False
    )
    assert result["selected_generation"] == 1
    assert state_trees_equal(valid, model.state_dict())
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("field,value", _BAD_PAYLOAD)
def test_all_bad_scalar_metadata_preserves_receiver_and_rng(tmp_path, field, value):
    config, source, source_optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, source, source_optimizer)
    _rewrite(store, field, value)
    _, model, optimizer = _setup()
    before = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    rng = capture_rng_state()
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert state_trees_equal(before, (model.state_dict(), optimizer.state_dict()))
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize(
    "field", ["schema_version", "generation", "state_bytes", "completed_step", "tokens_seen"]
)
@pytest.mark.parametrize("value", [True, 2.0, -1])
def test_manifest_constructor_and_decoding_validate_integer_contract(tmp_path, field, value):
    config, model, optimizer = _setup()
    manifest = _save(tmp_path / "store", config, model, optimizer)
    expected = (
        CheckpointMismatchError
        if field == "schema_version" and value == -1
        else CheckpointIntegrityError
    )
    with pytest.raises(expected):
        replace(manifest, **{field: value})
    raw = manifest.to_dict()
    raw[field] = value
    with pytest.raises(expected):
        CheckpointManifestV2.from_dict(raw)


@pytest.mark.parametrize(
    "field,value", [("schema_version", 2.0), ("completed_step", True), ("tokens_seen", 8.0)]
)
def test_malformed_manifest_falls_back(tmp_path, field, value):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, model, optimizer)
    _save(store, config, model, optimizer)
    _rewrite(store, field, value, manifest_only=True)
    assert (
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)[
            "selected_generation"
        ]
        == 1
    )


@pytest.mark.parametrize("manifest_only", [False, True])
def test_future_integer_schema_does_not_silently_fall_back(tmp_path, manifest_only):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, model, optimizer)
    _save(store, config, model, optimizer)
    _rewrite(store, "schema_version", 3, manifest_only=manifest_only)
    with pytest.raises(CheckpointMismatchError, match="schema"):
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)


@pytest.mark.parametrize(
    "field,value", [("epoch", False), ("batch_index", 1.0), ("token_window_offset", -1)]
)
def test_typed_cursor_rejected_before_save_and_bad_stored_cursor_falls_back(tmp_path, field, value):
    config, model, optimizer = _setup()
    bad = {**_CURSOR, field: value}
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="cursor"):
        _save(store, config, model, optimizer, cursor=bad)
    assert not store.exists()
    _save(store, config, model, optimizer, cursor=_CURSOR)
    _save(store, config, model, optimizer, cursor=_CURSOR)
    _rewrite(store, "cursor", bad)
    _rewrite(store, "cursor", bad, manifest_only=True)
    assert (
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)[
            "selected_generation"
        ]
        == 1
    )


@pytest.mark.parametrize("cursor", [None, {"batch_id": 0}, _CURSOR])
@pytest.mark.parametrize("step,tokens", [(0, 0), (1, 8)])
def test_valid_integer_metadata_and_generic_history_cursor_roundtrip(
    tmp_path, cursor, step, tokens
):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    manifest = _save(store, config, model, optimizer, cursor=cursor, step=step, tokens_seen=tokens)
    assert CheckpointManifestV2.from_dict(manifest.to_dict()) == manifest
    result = load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert result["step"] == step and type(result["step"]) is int
    assert result["tokens_seen"] == tokens and type(result["tokens_seen"]) is int
    assert result["cursor"] == cursor
