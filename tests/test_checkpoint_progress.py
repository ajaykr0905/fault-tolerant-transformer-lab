import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import pytest
import torch

import fttl.checkpoint as checkpoint
from fttl.config import ExperimentConfig
from fttl.state import capture_rng_state, state_trees_equal


def _setup():
    config = ExperimentConfig()
    model = torch.nn.Linear(2, 2)
    return config, model, torch.optim.AdamW(model.parameters())


def _save(store, setup, step, tokens):
    config, model, optimizer = setup
    return checkpoint.save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        tokens_seen=tokens,
        losses=[1.0],
    )


def _files(store):
    return {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }


@pytest.mark.parametrize("step,tokens", [(1, 16), (2, 15), (1, 8), (3, 15)])
def test_serialized_stale_publication_preserves_all_committed_progress(tmp_path, step, tokens):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup, 2, 16)
    before_files = _files(store)
    before_state = copy.deepcopy((setup[1].state_dict(), setup[2].state_dict()))
    rng = capture_rng_state()
    with pytest.raises(checkpoint.CheckpointMismatchError, match="progress"):
        _save(store, setup, step, tokens)
    assert _files(store) == before_files
    assert state_trees_equal(before_state, (setup[1].state_dict(), setup[2].state_dict()))
    assert state_trees_equal(rng, capture_rng_state())


def test_payload_staged_before_lock_cannot_overwrite_newer_committed_progress(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup, 0, 0)
    staged, release = threading.Event(), threading.Event()
    original = checkpoint._checkpoint_writer_lock

    @contextmanager
    def delayed_lock(root, timeout):
        if threading.current_thread().name.startswith("stale"):
            staged.set()
            assert release.wait(5)
        with original(root, timeout):
            yield

    monkeypatch.setattr(checkpoint, "_checkpoint_writer_lock", delayed_lock)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stale") as pool:
        stale = pool.submit(_save, store, setup, 1, 8)
        try:
            assert staged.wait(5)
            _save(store, setup, 2, 16)
            before = _files(store)
        finally:
            release.set()
        with pytest.raises(checkpoint.CheckpointMismatchError, match="progress"):
            stale.result(timeout=5)
    assert _files(store) == before


@pytest.mark.parametrize("step,tokens", [(2, 16), (2, 17), (3, 16), (4, 99)])
def test_nondecreasing_progress_and_equal_progress_repeats_are_allowed(tmp_path, step, tokens):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup, 2, 16)
    assert _save(store, setup, step, tokens).generation == 2
    result = checkpoint.load_checkpoint(
        store, model=setup[1], optimizer=setup[2], expected_config=setup[0], restore_rng=False
    )
    assert result["step"] == step and result["tokens_seen"] == tokens


def _corrupt_numerics(store):
    generation = store / (store / "LATEST").read_text().strip()
    path = generation / "state.pt"
    payload = torch.load(path, weights_only=True)
    payload["model"]["weight"][0, 0] = float("nan")
    torch.save(payload, path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        state_bytes=path.stat().st_size, state_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    manifest_path.write_text(json.dumps(manifest))


def test_corrupt_newest_uses_valid_predecessor_progress_and_allows_forward_repair(tmp_path):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup, 1, 8)
    _save(store, setup, 9, 72)
    _corrupt_numerics(store)
    assert _save(store, setup, 2, 16).generation == 3
    result = checkpoint.load_checkpoint(
        store, model=setup[1], optimizer=setup[2], expected_config=setup[0], restore_rng=False
    )
    assert result["step"] == 2


def test_no_valid_payload_preserves_existing_repair_save_behavior(tmp_path):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup, 9, 72)
    _corrupt_numerics(store)
    assert _save(store, setup, 1, 8).generation == 2
