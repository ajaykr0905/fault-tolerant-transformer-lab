"""CPU rejection/harness contracts; GPU evidence requires the real CUDA integration test."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
from dataclasses import replace

import pytest
import torch
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl import cuda_process_recovery as recovery
from fttl import cuda_recovery
from fttl.checkpoint import load_checkpoint
from fttl.data import PreparedDatasetBatchSource
from fttl.process_recovery import (
    ProcessRecoveryTimeout,
    ProcessRecoveryWorkerError,
    _run_worker,
    _WorkerOutcome,
)
from fttl.recovery import RecoveryVerificationError
from fttl.state import capture_rng_state


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), "1", None])
def test_invalid_deadline_fails_before_manifest_or_worker_access(tmp_path, monkeypatch, timeout):
    monkeypatch.setattr(recovery, "_run_worker", lambda *a, **k: pytest.fail("worker spawned"))
    with pytest.raises(ValueError, match="timeout_seconds"):
        recovery.verify_cuda_process_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=tmp_path / "missing",
            timeout_seconds=timeout,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["steps", "checkpoint", "dropout"])
def test_invalid_training_contract_fails_before_workers(tmp_path, monkeypatch, change):
    config = real_data_config()
    if change == "steps":
        config = replace(config, steps=2)
    elif change == "checkpoint":
        config = replace(config, checkpoint_every=2)
    else:
        config = replace(config, model=replace(config.model, dropout=0.0))
    monkeypatch.setattr(recovery, "_run_worker", lambda *a, **k: pytest.fail("worker spawned"))
    with pytest.raises(ValueError):
        recovery.verify_cuda_process_recovery(
            config, tmp_path / "run", dataset_manifest=tmp_path / "missing"
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["empty", "nonempty", "file"])
def test_existing_output_is_preserved(tmp_path, kind):
    output = tmp_path / "run"
    if kind == "file":
        output.write_text("keep", encoding="utf-8")
    else:
        output.mkdir()
        if kind == "nonempty":
            (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output"):
        recovery.verify_cuda_process_recovery(
            real_data_config(), output, dataset_manifest=tmp_path / "missing"
        )
    if kind == "file":
        assert output.read_text() == "keep"
    elif kind == "nonempty":
        assert (output / "keep.txt").read_text() == "keep"
    else:
        assert not list(output.iterdir())


def test_invalid_manifest_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "_run_worker", lambda *a, **k: pytest.fail("worker spawned"))
    with pytest.raises((ValueError, FileNotFoundError)):
        recovery.verify_cuda_process_recovery(
            real_data_config(), tmp_path / "run", dataset_manifest=tmp_path / "missing"
        )
    assert not (tmp_path / "run").exists()


def test_initialized_parent_is_rejected_before_manifest_or_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="fresh process"):
        recovery.verify_cuda_process_recovery(
            real_data_config(), tmp_path / "run", dataset_manifest=tmp_path / "missing"
        )
    assert not (tmp_path / "run").exists()


def test_cpu_cannot_be_substituted_for_a_cuda_worker(tmp_path, monkeypatch):
    class Sender:
        messages = []
        closed = False

        def send(self, value):
            self.messages.append(value)

        def close(self):
            self.closed = True

    sender = Sender()
    monkeypatch.setattr(recovery, "prepare_cuda_execution", lambda: (torch.device("cpu"), {}))
    with pytest.raises(SystemExit) as error:
        recovery._cuda_worker(
            real_data_config().to_dict(),
            str(tmp_path / "run"),
            str(tmp_path / "missing"),
            None,
            None,
            None,
            sender,
        )
    assert error.value.code == 1 and sender.closed
    assert sender.messages[0] == {"event": "started", "pid": os.getpid()}
    assert sender.messages[-1]["event"] == "error"
    assert "actual CUDA execution" in sender.messages[-1]["message"]
    assert not (tmp_path / "run").exists()


def test_worker_failure_cannot_publish_a_success_report(tmp_path, monkeypatch):
    def fail_worker(*args, **kwargs):
        raise ProcessRecoveryWorkerError("control", "RuntimeError", "CUDA unavailable")

    monkeypatch.setattr(recovery, "_run_worker", fail_worker)
    with pytest.raises(ProcessRecoveryWorkerError, match="CUDA unavailable"):
        recovery.verify_cuda_process_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=prepared_fixture(tmp_path / "dataset"),
        )
    assert not (tmp_path / "run" / "cuda-process-recovery-report.json").exists()


@pytest.mark.parametrize(
    "field",
    [
        "pid",
        "config_fingerprint",
        "data_fingerprint",
        "tokenizer_fingerprint",
        "code_fingerprint",
        "code_revision",
        "contract",
        "execution",
    ],
)
def test_control_metadata_mismatch_cannot_publish_a_gpu_report(tmp_path, monkeypatch, field):
    config = real_data_config()
    manifest = prepared_fixture(tmp_path / "dataset")
    source = PreparedDatasetBatchSource.from_manifest(config, manifest)
    execution = {"device": "cuda:0"}
    metadata = {
        "pid": 123456,
        "execution": execution,
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": source.data_fingerprint,
        "tokenizer_fingerprint": source.tokenizer_fingerprint,
        "code_fingerprint": recovery.code_fingerprint(),
        "code_revision": recovery.git_revision(),
        "contract": recovery._contract(config, source, execution),
    }
    metadata[field] = (
        {"device": "cpu"} if field == "execution" else 654321 if field == "pid" else "mismatch"
    )
    monkeypatch.setattr(
        recovery,
        "_run_worker",
        lambda *a, **k: _WorkerOutcome(123456, 0, 0.1, {}),
    )
    monkeypatch.setattr(recovery, "_read_worker", lambda *a: (metadata, {}))
    with pytest.raises(RecoveryVerificationError, match="parent execution contract"):
        recovery.verify_cuda_process_recovery(config, tmp_path / "run", dataset_manifest=manifest)
    assert not (tmp_path / "run" / "cuda-process-recovery-report.json").exists()


def _never_complete(config, directory, manifest, resume, stop, failure, sender):
    sender.send({"event": "started", "pid": os.getpid()})
    threading.Event().wait()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process termination")
def test_reused_spawn_harness_reaps_a_timed_out_worker(tmp_path):
    before = {child.pid for child in multiprocessing.active_children()}
    with pytest.raises(ProcessRecoveryTimeout):
        _run_worker(
            real_data_config(),
            tmp_path / "run",
            role="cuda-timeout-contract",
            dataset_manifest=None,
            timeout_seconds=2,
            worker_target=_never_complete,
        )
    assert {child.pid for child in multiprocessing.active_children()} == before
    assert not (tmp_path / "run").exists()


def test_advance_observer_pauses_before_second_checkpoint_on_cpu_unit_only(tmp_path, monkeypatch):
    config = real_data_config()
    source = PreparedDatasetBatchSource.from_manifest(
        config, prepared_fixture(tmp_path / "dataset")
    )
    model, optimizer = cuda_recovery._new_training_objects(config, torch.device("cpu"))
    events = []
    monkeypatch.setattr(cuda_recovery, "capture_cuda_rng_state", lambda device: capture_rng_state())

    def observe(event, values):
        events.append((event, values))
        if event == "after-optimizer" and values["attempted_step"] == 2:
            raise RuntimeError("unit boundary reached")

    with pytest.raises(RuntimeError, match="unit boundary reached"):
        cuda_recovery._advance(
            model,
            optimizer,
            config,
            source,
            torch.device("cpu"),
            tmp_path / "checkpoints",
            "observer-unit-only",
            config.steps,
            observer=observe,
        )
    assert [event for event, _ in events] == [
        "after-optimizer",
        "after-checkpoint",
        "after-optimizer",
    ]
    assert events[-1][1]["durable_step"] == 1
    assert events[-1][1]["batch_id"] == source.batch(source.cursor_at(1)).batch_id
    payload = load_checkpoint(
        tmp_path / "checkpoints",
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_run_contract_fingerprint="observer-unit-only",
        restore_rng=False,
    )
    assert payload["step"] == payload["selected_generation"] == 1
    assert not (tmp_path / "checkpoints" / "generation-00000002").exists()
    assert not list(tmp_path.glob("*report.json"))


def test_module_help_does_not_initialize_cuda():
    completed = subprocess.run(
        [sys.executable, "-m", "fttl.cuda_process_recovery", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert "--dataset-manifest" in completed.stdout and "--timeout-seconds" in completed.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGKILL")
def test_real_cuda_sigkill_replays_failed_batch_and_preserves_full_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    report = recovery.verify_cuda_process_recovery(
        real_data_config(),
        tmp_path / "run",
        dataset_manifest=prepared_fixture(tmp_path / "dataset"),
    )
    assert report["exact_equality"] and all(report["equality"].values())
    assert report["interrupted_exitcode"] == -signal.SIGKILL
    assert len(set(report["worker_pids"].values())) == 4
    assert report["selected_step"] == report["selected_generation"] == 1
    assert report["durable_step_before_kill"] == 1 and report["attempted_step"] == 2
    assert report["failed_batch_id"] == report["replayed_batch_id"]
    assert report["failed_sample_ids"] == report["replayed_sample_ids"]
    assert report["steps"] == 4 and report["tokens_seen"] == 64
    assert report["execution"]["device"] == "cuda:0"
    assert report["timings_seconds"]["resume_spawn_to_replayed_commit_receipt"] > 0
    assert not torch.cuda.is_initialized()
    raw = (tmp_path / "run" / "cuda-process-recovery-report.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    assert json.loads(raw) == report
