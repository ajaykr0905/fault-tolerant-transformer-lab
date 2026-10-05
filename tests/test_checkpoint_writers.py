import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import fttl.checkpoint as checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer


def _setup():
    config = ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1)
    )
    model = TinyTransformer(config.model)
    return config, model, torch.optim.AdamW(model.parameters())


def _save(store, setup, *, step=1, injector=None, **kwargs):
    config, model, optimizer = setup
    return checkpoint.save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        tokens_seen=step * 8,
        losses=[1.0] * step,
        failure_injector=injector,
        **kwargs,
    )


def _files(store):
    return {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }


def test_overlapping_writers_do_not_clean_active_generations_and_readers_keep_committed_state(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup)
    paused = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    second_cleanup = threading.Event()
    original_cleanup = checkpoint._cleanup_stale_temporaries

    def observed_cleanup(root):
        if second_started.is_set():
            second_cleanup.set()
        original_cleanup(root)

    def pause(stage):
        if stage == "after-state-serialize":
            paused.set()
            assert release.wait(5)

    def second_writer():
        second_started.set()
        return _save(store, setup, step=3)

    monkeypatch.setattr(checkpoint, "_cleanup_stale_temporaries", observed_cleanup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(_save, store, setup, step=2, injector=pause)
        try:
            assert paused.wait(5)
            active = list(store.glob(".generation-*.tmp-*"))
            assert len(active) == 1
            second = pool.submit(second_writer)
            assert second_started.wait(5)
            assert not second_cleanup.wait(0.2)
            assert active[0].is_dir()
            config, model, optimizer = _setup()
            loaded = checkpoint.load_checkpoint(
                store, model=model, optimizer=optimizer, expected_config=config, restore_rng=False
            )
            assert loaded["step"] == 1
        finally:
            release.set()
        assert first.result(timeout=5).generation == 2
        assert second.result(timeout=5).generation == 3
    lock = store / ".writer.lock"
    inode = lock.stat().st_ino
    _save(store, setup, step=4)
    assert lock.stat().st_ino == inode
    assert not list(store.glob(".generation-*.tmp-*"))


def test_writer_timeout_preserves_committed_and_active_files(tmp_path):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup)
    paused = threading.Event()
    release = threading.Event()

    def pause(stage):
        if stage == "after-state-serialize":
            paused.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(_save, store, setup, step=2, injector=pause)
        try:
            assert paused.wait(5)
            before = _files(store)
            with pytest.raises(TimeoutError, match="writer lock"):
                _save(store, setup, step=3, writer_lock_timeout=0.05)
            assert _files(store) == before
        finally:
            release.set()
        assert first.result(timeout=5).generation == 2


def test_exception_releases_lock_and_other_store_never_waits(tmp_path):
    first_store, other_store = tmp_path / "first", tmp_path / "other"
    setup = _setup()

    def fail(stage):
        if stage == "after-state-serialize":
            assert _save(other_store, setup, writer_lock_timeout=0.05).generation == 1
            raise RuntimeError("injected writer failure")

    with pytest.raises(RuntimeError, match="injected"):
        _save(first_store, setup, injector=fail)
    inode = (first_store / ".writer.lock").stat().st_ino
    assert _save(first_store, setup, writer_lock_timeout=0.05).generation == 1
    assert (first_store / ".writer.lock").stat().st_ino == inode


@pytest.mark.parametrize("value", [True, None, "1", 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_does_not_create_store(tmp_path, value):
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="writer lock timeout"):
        _save(store, _setup(), writer_lock_timeout=value)
    assert not store.exists()


def test_process_death_releases_advisory_lock_without_deleting_inode(tmp_path):
    store = tmp_path / "store"
    setup = _setup()
    _save(store, setup)
    lock = store / ".writer.lock"
    inode = lock.stat().st_ino
    script = "import fcntl,sys,time; f=open(sys.argv[1], 'a+b'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(30)"
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(lock)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "locked"
        before = _files(store)
        with pytest.raises(TimeoutError, match="writer lock"):
            _save(store, setup, step=2, writer_lock_timeout=0.05)
        assert _files(store) == before
        os.kill(child.pid, signal.SIGKILL)
        assert child.wait(timeout=5) == -signal.SIGKILL
        assert _save(store, setup, step=2, writer_lock_timeout=0.5).generation == 2
        assert lock.stat().st_ino == inode
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_unsupported_lock_platform_fails_before_store_creation(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint, "fcntl", None, raising=False)
    store = tmp_path / "store"
    with pytest.raises(RuntimeError, match="POSIX"):
        _save(store, _setup())
    assert not store.exists()
