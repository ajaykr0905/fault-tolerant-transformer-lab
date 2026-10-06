import copy
import hashlib
import json

import pytest
import torch

from fttl.checkpoint import CheckpointIntegrityError, load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def _setup():
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1),
        steps=2,
        batch_size=2,
    )
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    return config, model, optimizer


def _save(store, config, model, optimizer, step=1):
    return save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        tokens_seen=step * 8,
        losses=[1.0] * step,
    )


def _files(store):
    return {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }


def _corrupt_runtime(model, optimizer, component, value):
    with torch.no_grad():
        if component == "model":
            model.position_embedding.weight[-1, 0] = value
        elif component == "optimizer_tensor":
            next(iter(optimizer.state.values()))["exp_avg"][0, 0] = value
        elif component == "optimizer_scalar":
            optimizer.param_groups[0]["lr"] = value
        elif component == "nonpersistent_buffer":
            model.register_buffer("hidden_control", torch.tensor(value), persistent=False)
        else:
            raise AssertionError(component)


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize(
    "component", ["model", "optimizer_tensor", "optimizer_scalar", "nonpersistent_buffer"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_runtime_is_rejected_before_store_writes_or_cleanup(
    tmp_path, existing, component, value
):
    config, model, optimizer = _setup()
    store = tmp_path / "store"
    if existing:
        _save(store, config, model, optimizer)
        (store / ".LATEST.tmp-preserved").write_bytes(b"existing temporary evidence")
    _corrupt_runtime(model, optimizer, component, value)
    files = _files(store)
    before_model = copy.deepcopy(model.state_dict())
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    flags = [parameter.requires_grad for parameter in model.parameters()]
    rng = capture_rng_state()
    with pytest.raises(FloatingPointError, match="non-finite"):
        _save(store, config, model, optimizer, step=2)
    assert store.exists() is existing
    assert _files(store) == files
    torch.testing.assert_close(model.state_dict(), before_model, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        optimizer.state_dict(), before_optimizer, rtol=0, atol=0, equal_nan=True
    )
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert model.training
    assert state_trees_equal(rng, capture_rng_state())
    if component == "nonpersistent_buffer":
        torch.testing.assert_close(model.hidden_control, torch.tensor(value), equal_nan=True)


def _rewrite_payload(store, component, value):
    generation = store / (store / "LATEST").read_text().strip()
    path = generation / "state.pt"
    payload = torch.load(path, weights_only=True)
    if component == "model":
        payload["model"]["position_embedding.weight"][-1, 0] = value
    elif component == "optimizer_tensor":
        next(iter(payload["optimizer"]["state"].values()))["exp_avg"][0, 0] = value
    elif component == "optimizer_scalar":
        payload["optimizer"]["param_groups"][0]["lr"] = value
    elif component == "losses":
        payload["losses"][-1] = value
    else:
        raise AssertionError(component)
    torch.save(payload, path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state_bytes"] = path.stat().st_size
    manifest["state_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize("component", ["model", "optimizer_tensor", "optimizer_scalar", "losses"])
@pytest.mark.parametrize("restore_rng", [False, True])
def test_integrity_matching_nonfinite_generation_falls_back_before_receiver_mutation(
    tmp_path, component, restore_rng
):
    config, source, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, source, optimizer)
    valid_model = copy.deepcopy(source.state_dict())
    valid_optimizer = copy.deepcopy(optimizer.state_dict())
    _save(store, config, source, optimizer, step=2)
    _rewrite_payload(store, component, float("nan"))
    _, target, target_optimizer = _setup()
    payload = load_checkpoint(
        store,
        model=target,
        optimizer=target_optimizer,
        expected_config=config,
        restore_rng=restore_rng,
    )
    assert payload["selected_generation"] == 1
    assert state_trees_equal(target.state_dict(), valid_model)
    assert state_trees_equal(target_optimizer.state_dict(), valid_optimizer)


@pytest.mark.parametrize("component", ["model", "optimizer_tensor", "optimizer_scalar", "losses"])
def test_only_nonfinite_generation_fails_closed_preserving_receiver(tmp_path, component):
    config, source, optimizer = _setup()
    store = tmp_path / "store"
    _save(store, config, source, optimizer)
    _rewrite_payload(store, component, float("inf"))
    _, target, target_optimizer = _setup()
    before_model = copy.deepcopy(target.state_dict())
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    flags = [parameter.requires_grad for parameter in target.parameters()]
    rng = capture_rng_state()
    with pytest.raises(CheckpointIntegrityError, match="non-finite"):
        load_checkpoint(
            store,
            model=target,
            optimizer=target_optimizer,
            expected_config=config,
            restore_rng=False,
        )
    assert state_trees_equal(target.state_dict(), before_model)
    assert state_trees_equal(target_optimizer.state_dict(), before_optimizer)
    assert flags == [parameter.requires_grad for parameter in target.parameters()]
    assert target.training
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("kind", ["parameter", "buffer"])
@pytest.mark.parametrize("layout", ["sparse", "meta"])
def test_unmaterialized_or_nondense_source_state_is_rejected_before_store_creation(
    tmp_path, kind, layout
):
    config, source, optimizer = _setup()
    tensor = torch.eye(2).to_sparse() if layout == "sparse" else torch.empty(2, 2, device="meta")
    if kind == "parameter":
        source.register_parameter("unsupported", torch.nn.Parameter(tensor, requires_grad=False))
    else:
        source.register_buffer("unsupported", tensor, persistent=False)
    rng = capture_rng_state()
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="materialized dense tensor"):
        _save(store, config, source, optimizer)
    assert not store.exists()
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legitimate_nested_optimizer_state_and_finite_runtime_buffers_roundtrip(tmp_path, dtype):
    config, source, optimizer = _setup()
    source.to(dtype=dtype)
    source.register_buffer("finite_control", torch.tensor([1.0, 2.0]), persistent=False)
    store = tmp_path / "store"
    _save(store, config, source, optimizer)
    _, target, target_optimizer = _setup()
    target.to(dtype=dtype)
    rng = capture_rng_state()
    payload = load_checkpoint(
        store, model=target, optimizer=target_optimizer, expected_config=config, restore_rng=False
    )
    assert payload["selected_generation"] == 1
    assert state_trees_equal(source.state_dict(), target.state_dict())
    assert state_trees_equal(rng, capture_rng_state())
    torch.testing.assert_close(source.finite_control, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
