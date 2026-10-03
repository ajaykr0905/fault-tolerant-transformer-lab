import gzip
import hashlib
import json
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from fttl.config import ExperimentConfig, ModelConfig
from fttl.dataset import DatasetSource, prepare_peps
from fttl.process_recovery import (
    ProcessRecoveryTimeout,
    ProcessRecoveryWorkerError,
    _run_worker,
    verify_process_recovery,
)
from fttl.recovery import RecoveryVerificationError
from fttl.train import FAILURE_POINTS, run_training

pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires POSIX SIGKILL")
FIXTURE = Path(__file__).parent / "fixtures" / "public_domain_peps_fixture.jsonl"


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257, block_size=8, d_model=8, n_heads=2, n_layers=1, dropout=0.25
        ),
        seed=23,
        steps=3,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )


def _prepared_fixture(tmp_path: Path, revision: str = "fixture-v1") -> Path:
    archive = gzip.compress(FIXTURE.read_bytes(), mtime=0)
    source = DatasetSource(
        repository="fttl://tests/public-domain-pep-fixture",
        revision=revision,
        path="public_domain_peps_fixture.jsonl.gz",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=("Independently written CI fixture.",),
    )

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(archive)

    output = tmp_path / "prepared"
    prepare_peps(tmp_path / "cache", output, downloader=downloader, source=source)
    return output / "manifest.json"


@pytest.mark.parametrize("failure_point", FAILURE_POINTS)
def test_sigkill_at_each_boundary_matches_independent_control(tmp_path: Path, failure_point: str):
    output = tmp_path / "evidence"
    report = verify_process_recovery(
        _config(),
        output,
        dataset_manifest=_prepared_fixture(tmp_path / "dataset"),
        failure_point=failure_point,
    )

    assert report.exact_equality
    assert all(report.equality.values())
    assert report.interrupted_exitcode == -signal.SIGKILL
    assert report.process_start_method == "spawn"
    assert len(set(report.worker_pids.values())) == 4
    assert os.getpid() not in report.worker_pids.values()
    assert report.selected_step == report.durable_step_before_kill == 1
    assert report.attempted_step == 2
    assert report.selected_generation == 1
    assert report.selected_tokens_seen == 16
    assert report.failed_batch_id == report.replayed_batch_id
    assert report.failed_sample_ids == report.replayed_sample_ids
    assert report.replayed_tokens == 16
    assert report.discarded_compute_tokens == (0 if failure_point == "before-forward" else 16)
    assert report.durable_committed_steps_lost == report.durable_committed_tokens_lost == 0
    assert report.completed_steps == 3
    assert report.tokens_seen == 48
    assert report.staging_cleanup_complete
    assert bool(report.abandoned_staging_files) == (failure_point == "during-checkpoint-write")
    assert all(
        path.startswith("recovered/checkpoints/.") for path in report.abandoned_staging_files
    )
    assert all(seconds >= 0 for seconds in report.timings_seconds.values())
    assert report.timings_seconds["resume_spawn_to_replayed_commit_receipt"] > 0
    assert report.data_kind == "prepared-dataset"
    public_json = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.json"))
    assert str(tmp_path) not in public_json
    assert "process-recovery-report.json" in {path.name for path in output.iterdir()}
    assert not set(report.worker_pids.values()).intersection(
        process.pid for process in multiprocessing.active_children()
    )


def test_synthetic_workload_is_explicitly_labeled(tmp_path: Path):
    config = replace(_config(), model=replace(_config().model, vocab_size=64))
    report = verify_process_recovery(config, tmp_path / "evidence", failure_point="after-optimizer")
    assert report.exact_equality
    assert report.data_kind == "synthetic-smoke"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeout_fails_before_output_creation(tmp_path: Path, timeout: float):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="finite and greater than zero"):
        verify_process_recovery(
            _config(), output, failure_point="after-optimizer", timeout_seconds=timeout
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("config", "failure_point", "expected"),
    [
        (replace(_config(), checkpoint_every=2), "after-optimizer", "checkpoint_every=1"),
        (replace(_config(), steps=2), "after-optimizer", "at least 3 steps"),
        (_config(), "invalid", "unknown failure point"),
    ],
)
def test_invalid_recovery_contract_fails_before_output_creation(
    tmp_path: Path, config: ExperimentConfig, failure_point: str, expected: str
):
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match=expected):
        verify_process_recovery(config, output, failure_point=failure_point)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path: Path):
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="fresh output"):
        verify_process_recovery(_config(), output, failure_point="after-optimizer")
    assert sentinel.read_text(encoding="utf-8") == "user evidence"


def _wait_forever_worker(*args):
    sender = args[-1]
    sender.send({"event": "started", "pid": os.getpid()})
    threading.Event().wait()


def _complete_without_failure_worker(*args):
    sender = args[-1]
    sender.send({"event": "started", "pid": os.getpid()})
    sender.send({"event": "completed"})
    sender.close()


@pytest.mark.parametrize("timeout", [0.01, 5.0])
def test_startup_and_completion_deadlines_reap_workers(tmp_path: Path, timeout: float):
    before = {process.pid for process in multiprocessing.active_children()}
    started = time.perf_counter()
    with pytest.raises(ProcessRecoveryTimeout, match="deadline"):
        _run_worker(
            _config(),
            tmp_path / "worker",
            role="timeout-test",
            dataset_manifest=None,
            timeout_seconds=timeout,
            worker_target=_wait_forever_worker,
        )
    assert time.perf_counter() - started < timeout + 6
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_unreached_failure_boundary_reaps_worker(tmp_path: Path):
    before = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(RecoveryVerificationError, match="boundary was not reached"):
        _run_worker(
            _config(),
            tmp_path / "worker",
            role="missing-boundary",
            dataset_manifest=None,
            timeout_seconds=30,
            failure_point="after-optimizer",
            worker_target=_complete_without_failure_worker,
        )
    assert {process.pid for process in multiprocessing.active_children()} == before


def _exit_without_acknowledgement_worker(*args):
    os._exit(3)


def test_unexpected_worker_exit_reaps_worker(tmp_path: Path):
    before = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(ProcessRecoveryWorkerError, match="IPC closed before completion"):
        _run_worker(
            _config(),
            tmp_path / "worker",
            role="unexpected-exit",
            dataset_manifest=None,
            timeout_seconds=30,
            worker_target=_exit_without_acknowledgement_worker,
        )
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_spawned_resume_rejects_a_different_dataset(tmp_path: Path):
    config = _config()
    manifest = _prepared_fixture(tmp_path / "original")
    changed_manifest = _prepared_fixture(tmp_path / "changed", revision="fixture-v2")
    run_dir = tmp_path / "worker"
    run_training(config, run_dir, stop_after_step=1, dataset_manifest=manifest)
    before = {process.pid for process in multiprocessing.active_children()}
    with pytest.raises(
        ProcessRecoveryWorkerError, match="data fingerprint does not match"
    ) as failure:
        _run_worker(
            config,
            run_dir,
            role="mismatched-dataset",
            dataset_manifest=changed_manifest,
            resume_from=run_dir / "checkpoints",
            timeout_seconds=30,
        )
    assert failure.value.error_type == "CheckpointMismatchError"
    assert json.loads((run_dir / "result.json").read_text(encoding="utf-8"))["steps"] == 1
    assert {process.pid for process in multiprocessing.active_children()} == before


def test_installed_cli_spawns_workers_and_emits_public_safe_report(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(_config().canonical_json(), encoding="utf-8")
    output = tmp_path / "cli-evidence"
    command = Path(sys.executable).with_name("fttl-verify-process-recovery")
    completed = subprocess.run(
        [
            str(command),
            "--config",
            str(config_path),
            "--dataset-manifest",
            str(_prepared_fixture(tmp_path / "dataset")),
            "--failure-point",
            "during-checkpoint-write",
            "--output",
            str(output),
            "--timeout-seconds",
            "30",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=130,
    )
    report = json.loads(completed.stdout)
    assert report == json.loads((output / "process-recovery-report.json").read_text())
    assert report["exact_equality"]
    assert report["data_kind"] == "prepared-dataset"
    assert report["interrupted_exitcode"] == -signal.SIGKILL
    assert report["abandoned_staging_files"]
    assert report["staging_cleanup_complete"]
    assert str(tmp_path) not in completed.stdout
