import copy
import hashlib
import json

import pytest
import torch

from fttl.checkpoint import CheckpointIntegrityError, load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig
from fttl.state import capture_rng_state, state_trees_equal


def _setup(kind=torch.optim.AdamW, dtype=torch.float32, amsgrad=False, initialized=True):
    model = torch.nn.Linear(2, 2, dtype=dtype)
    optimizer = kind(model.parameters(), amsgrad=amsgrad)
    if initialized:
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    return ExperimentConfig(), model, optimizer


def _save(store, setup):
    config, model, optimizer = setup
    return save_checkpoint(
        store, model=model, optimizer=optimizer, config=config, step=1, tokens_seen=8, losses=[1.0]
    )


def _damage(state, defect):
    first = next(iter(state["state"].values()))
    if defect == "missing":
        del first["exp_avg_sq"]
    elif defect == "amsgrad":
        del first["max_exp_avg_sq"]
    elif defect == "counter":
        first["step"] = torch.ones(2)
    elif defect == "binding":
        state["param_groups"][0]["params"].append(99)
    else:
        first[defect] = torch.ones(3)


def _rewrite(store, defect=None, step=None):
    generation = store / (store / "LATEST").read_text().strip()
    path = generation / "state.pt"
    payload = torch.load(path, weights_only=True)
    if defect:
        _damage(payload["optimizer"], defect)
    if step is not None:
        for state in payload["optimizer"]["state"].values():
            state["step"] = step
    torch.save(payload, path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        state_bytes=path.stat().st_size, state_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest))
    return payload


@pytest.mark.parametrize(
    "defect", ["exp_avg", "exp_avg_sq", "missing", "amsgrad", "counter", "binding"]
)
def test_bad_initialized_adam_state_falls_back_before_selection(tmp_path, defect):
    setup = _setup(amsgrad=True)
    store = tmp_path / "store"
    _save(store, setup)
    valid = copy.deepcopy(setup[2].state_dict())
    _save(store, setup)
    _rewrite(store, defect)
    result = load_checkpoint(
        store, model=setup[1], optimizer=setup[2], expected_config=setup[0], restore_rng=False
    )
    assert result["selected_generation"] == 1
    assert state_trees_equal(valid, setup[2].state_dict())
    setup[2].step()


@pytest.mark.parametrize("legacy", [False, True])
def test_sole_bad_state_preserves_receiver_and_rng(tmp_path, legacy):
    setup = _setup()
    store = tmp_path / "store"
    _save(store, setup)
    payload = _rewrite(store, "exp_avg")
    if legacy:
        payload["schema_version"] = 1
        store = tmp_path / "legacy.pt"
        torch.save(payload, store)
    receiver = _setup(dtype=torch.float64)
    before = copy.deepcopy((receiver[1].state_dict(), receiver[2].state_dict()))
    rng = capture_rng_state()
    with pytest.raises(CheckpointIntegrityError, match="Adam"):
        load_checkpoint(
            store, model=receiver[1], optimizer=receiver[2], expected_config=receiver[0]
        )
    assert state_trees_equal(before, (receiver[1].state_dict(), receiver[2].state_dict()))
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("kind", [torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "step", [1, 1.0, torch.tensor(1.0), torch.tensor([1.0], dtype=torch.float64)]
)
def test_valid_moments_cast_to_receiver_and_resume_step(tmp_path, kind, dtype, step):
    setup = _setup(kind, amsgrad=True)
    store = tmp_path / "store"
    _save(store, setup)
    _rewrite(store, step=step)
    receiver = _setup(kind, dtype=dtype, amsgrad=True, initialized=False)
    load_checkpoint(store, model=receiver[1], optimizer=receiver[2], expected_config=receiver[0])
    for parameter in receiver[1].parameters():
        parameter.grad = torch.ones_like(parameter)
        assert receiver[2].state[parameter]["exp_avg"].dtype == dtype
    receiver[2].step()


def test_empty_adam_state_is_valid_and_can_initialize_after_resume(tmp_path):
    setup = _setup(initialized=False)
    store = tmp_path / "store"
    _save(store, setup)
    load_checkpoint(store, model=setup[1], optimizer=setup[2], expected_config=setup[0])
    for parameter in setup[1].parameters():
        parameter.grad = torch.ones_like(parameter)
    setup[2].step()


def test_save_rejects_malformed_moments_before_writes(tmp_path):
    setup = _setup()
    first = next(iter(setup[2].state.values()))
    first["exp_avg"] = torch.ones(3)
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="Adam"):
        _save(store, setup)
    assert not store.exists()


def test_custom_adam_subclass_state_layout_is_not_subject_to_builtin_validator(tmp_path):
    class CustomAdam(torch.optim.Adam):
        pass

    setup = _setup(CustomAdam)
    next(iter(setup[2].state.values()))["custom_scalar"] = torch.ones(3)
    store = tmp_path / "store"
    _save(store, setup)
    result = load_checkpoint(store, model=setup[1], optimizer=setup[2], expected_config=setup[0])
    assert next(iter(result["optimizer"]["state"].values()))["custom_scalar"].shape == (3,)
