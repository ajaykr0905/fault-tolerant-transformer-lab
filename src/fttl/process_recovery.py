"""Parent-controlled SIGKILL recovery proof for independently spawned CPU workers."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import platform
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable

import torch

from fttl.checkpoint import load_checkpoint
from fttl.config import ExperimentConfig
from fttl.data import PreparedDatasetBatchSource, SyntheticBatchSource
from fttl.model import TinyTransformer
from fttl.recovery import RecoveryVerificationError
from fttl.state import state_digest, state_trees_equal
from fttl.train import FAILURE_POINTS, FailurePoint, InjectedTrainingFailure, run_training


@dataclass(frozen=True)
class ProcessRecoveryReportV1:
    schema: str
    verified_at_utc: str
    data_kind: str
    failure_point: str
    process_start_method: str
    worker_pids: dict[str, int]
    interrupted_exitcode: int
    kill_signal: str
    attempted_step: int
    durable_step_before_kill: int
    selected_step: int
    selected_generation: int
    selected_tokens_seen: int
    failed_batch_id: str
    replayed_batch_id: str
    failed_sample_ids: tuple[str, ...]
    replayed_sample_ids: tuple[str, ...]
    replayed_tokens: int
    discarded_compute_tokens: int
    durable_committed_steps_lost: int
    durable_committed_tokens_lost: int
    abandoned_staging_files: tuple[str, ...]
    staging_cleanup_complete: bool
    timings_seconds: dict[str, float]
    timing_boundaries: dict[str, str]
    completed_steps: int
    tokens_seen: int
    config_fingerprint: str
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    run_contract_fingerprint: str
    code_fingerprint: str
    code_revision: str
    equality: dict[str, bool]
    exact_equality: bool
    final_state_digest: str
    control_run_fingerprint: str
    recovered_run_fingerprint: str
    artifacts: dict[str, str]
    environment: dict[str, object]
    limitations: tuple[str, ...]


class ProcessRecoveryTimeout(TimeoutError):
    """A spawned worker did not acknowledge or finish within its deadline."""


class ProcessRecoveryWorkerError(RuntimeError):
    def __init__(self, role: str, error_type: str, message: str) -> None:
        super().__init__(f"{role} worker failed: {error_type}: {message}")
        self.error_type = error_type


@dataclass(frozen=True)
class _WorkerOutcome:
    pid: int
    exitcode: int
    elapsed_seconds: float
    commit_receipt_seconds: dict[int, float]
    failure: dict[str, Any] | None = None
    kill_to_exit_seconds: float | None = None


def _training_worker(
    config_value: dict[str, Any],
    run_dir: str,
    dataset_manifest: str | None,
    resume_from: str | None,
    stop_after_step: int | None,
    failure_point: FailurePoint | None,
    sender: Connection,
) -> None:
    """Only primitive configuration and paths cross the spawn boundary."""

    try:
        torch.set_num_threads(1)
        sender.send({"event": "started", "pid": os.getpid()})

        def observe_failure(failure: InjectedTrainingFailure) -> None:
            sender.send(
                {
                    "event": "failure-boundary",
                    "point": failure.point,
                    "attempted_step": failure.attempted_step,
                    "durable_step": failure.durable_step,
                    "batch_id": failure.batch_id,
                    "sample_ids": failure.sample_ids,
                    "attempted_tokens": failure.attempted_tokens,
                }
            )
            # The parent owns termination. This handshake holds the exact boundary
            # without a sleep race or exception unwinding the checkpoint writer.
            threading.Event().wait()

        def observe_commit(step: int, generation: int) -> None:
            sender.send({"event": "commit", "step": step, "generation": generation})

        run_training(
            ExperimentConfig.from_dict(config_value),
            Path(run_dir),
            dataset_manifest=Path(dataset_manifest) if dataset_manifest else None,
            resume_from=Path(resume_from) if resume_from else None,
            stop_after_step=stop_after_step,
            failure_point=failure_point,
            failure_step=2 if failure_point else None,
            failure_observer=observe_failure if failure_point else None,
            commit_observer=observe_commit,
        )
        sender.send({"event": "completed"})
    except Exception as error:
        sender.send({"event": "error", "error_type": type(error).__name__, "message": str(error)})
        sys.exit(1)
    finally:
        sender.close()


def _validate_timeout(timeout_seconds: float) -> None:
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be finite and greater than zero")


def _remaining(deadline: float, role: str) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise ProcessRecoveryTimeout(f"{role} worker exceeded its deadline")
    return remaining


def _run_worker(
    config: ExperimentConfig,
    run_dir: Path,
    *,
    role: str,
    dataset_manifest: Path | None,
    timeout_seconds: float,
    resume_from: Path | None = None,
    stop_after_step: int | None = None,
    failure_point: FailurePoint | None = None,
    worker_target: Callable[..., None] = _training_worker,
) -> _WorkerOutcome:
    _validate_timeout(timeout_seconds)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        name=f"fttl-{role}",
        target=worker_target,
        args=(
            config.to_dict(),
            str(run_dir.resolve()),
            str(dataset_manifest.resolve()) if dataset_manifest else None,
            str(resume_from.resolve()) if resume_from else None,
            stop_after_step,
            failure_point,
            sender,
        ),
    )
    started = time.perf_counter()
    deadline = started + timeout_seconds
    acknowledged = False
    commits: dict[int, float] = {}
    try:
        process.start()
        sender.close()
        while True:
            if not receiver.poll(_remaining(deadline, role)):
                raise ProcessRecoveryTimeout(f"{role} worker exceeded its deadline")
            try:
                message = receiver.recv()
            except EOFError as error:
                raise ProcessRecoveryWorkerError(
                    role, "UnexpectedExit", "IPC closed before completion"
                ) from error
            event = message["event"]
            if event == "started":
                if acknowledged or message["pid"] != process.pid:
                    raise RecoveryVerificationError("worker startup acknowledgement is invalid")
                acknowledged = True
            elif not acknowledged:
                raise RecoveryVerificationError("worker did not acknowledge startup")
            elif event == "commit":
                commits[message["step"]] = time.perf_counter() - started
            elif event == "error":
                raise ProcessRecoveryWorkerError(role, message["error_type"], message["message"])
            elif event == "failure-boundary":
                if failure_point is None or message["point"] != failure_point:
                    raise RecoveryVerificationError("unexpected worker failure boundary")
                kill_started = time.perf_counter()
                assert process.pid is not None
                os.kill(process.pid, signal.SIGKILL)
                process.join(_remaining(deadline, role))
                if process.is_alive():
                    raise ProcessRecoveryTimeout(f"{role} worker did not exit after SIGKILL")
                if process.exitcode != -signal.SIGKILL:
                    raise RecoveryVerificationError("interrupted worker did not exit from SIGKILL")
                return _WorkerOutcome(
                    pid=process.pid,
                    exitcode=process.exitcode,
                    elapsed_seconds=time.perf_counter() - started,
                    commit_receipt_seconds=commits,
                    failure=message,
                    kill_to_exit_seconds=time.perf_counter() - kill_started,
                )
            elif event == "completed":
                if failure_point is not None:
                    raise RecoveryVerificationError("configured failure boundary was not reached")
                process.join(_remaining(deadline, role))
                if process.is_alive():
                    raise ProcessRecoveryTimeout(f"{role} worker did not exit after completion")
                if process.exitcode != 0:
                    raise ProcessRecoveryWorkerError(role, "UnexpectedExit", str(process.exitcode))
                assert process.pid is not None
                return _WorkerOutcome(
                    pid=process.pid,
                    exitcode=process.exitcode,
                    elapsed_seconds=time.perf_counter() - started,
                    commit_receipt_seconds=commits,
                )
            else:
                raise RecoveryVerificationError(f"unknown worker event {event!r}")
    finally:
        receiver.close()
        sender.close()
        if process.pid is not None:
            if process.is_alive():
                process.kill()
                # Cleanup gets its own bounded budget even when startup timed out.
                process.join(5)
            if process.is_alive():
                raise ProcessRecoveryTimeout(f"{role} worker could not be reaped after SIGKILL")
            process.join(0)
            process.close()


def _load_state(
    config: ExperimentConfig, run_dir: Path, result: dict[str, Any] | None = None
) -> tuple[TinyTransformer, dict[str, Any]]:
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    payload = load_checkpoint(
        run_dir / "checkpoints",
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_data_fingerprint=result["data_fingerprint"] if result else None,
        expected_tokenizer_fingerprint=result["tokenizer_fingerprint"] if result else None,
        expected_run_contract_fingerprint=result["run_contract_fingerprint"] if result else None,
        restore_rng=False,
    )
    return model, payload


def _staging_files(run_dir: Path, output_dir: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            path.relative_to(output_dir).as_posix()
            for temporary in (run_dir / "checkpoints").glob(".generation-*.tmp-*")
            for path in temporary.rglob("*")
            if path.is_file()
        )
    )


def verify_process_recovery(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    failure_point: FailurePoint,
    dataset_manifest: Path | None = None,
    timeout_seconds: float = 60.0,
) -> ProcessRecoveryReportV1:
    """Kill step 2, recover durable step 1, and compare independent processes."""

    _validate_timeout(timeout_seconds)
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        raise ValueError("process-kill verification requires POSIX SIGKILL")
    if config.checkpoint_every != 1:
        raise ValueError("process-kill verification requires checkpoint_every=1")
    if config.steps < 3:
        raise ValueError("process-kill verification requires at least 3 steps")
    if failure_point not in FAILURE_POINTS:
        raise ValueError(f"unknown failure point {failure_point!r}")
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("process recovery output already exists; choose a fresh output directory")
    source = (
        SyntheticBatchSource(config)
        if dataset_manifest is None
        else PreparedDatasetBatchSource.from_manifest(config, dataset_manifest)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = output_dir / "control"
    recovered_dir = output_dir / "recovered"
    control_worker = _run_worker(
        config,
        control_dir,
        role="control",
        dataset_manifest=dataset_manifest,
        timeout_seconds=timeout_seconds,
    )
    interrupted = _run_worker(
        config,
        recovered_dir,
        role="interrupted",
        dataset_manifest=dataset_manifest,
        failure_point=failure_point,
        timeout_seconds=timeout_seconds,
    )
    assert interrupted.failure is not None
    failure = interrupted.failure
    _, selected = _load_state(config, recovered_dir)
    if failure["attempted_step"] != 2 or failure["durable_step"] != 1 or selected["step"] != 1:
        raise RecoveryVerificationError("kill did not leave durable step 1 before attempted step 2")
    if selected["data_fingerprint"] != source.data_fingerprint:
        raise RecoveryVerificationError("selected checkpoint does not match the dataset")
    abandoned = _staging_files(recovered_dir, output_dir)
    if failure_point == "during-checkpoint-write" and not abandoned:
        raise RecoveryVerificationError(
            "checkpoint-write kill did not leave an abandoned state file"
        )
    replay_worker = _run_worker(
        config,
        recovered_dir,
        role="replay",
        dataset_manifest=dataset_manifest,
        resume_from=recovered_dir / "checkpoints",
        stop_after_step=2,
        timeout_seconds=timeout_seconds,
    )
    replay = json.loads((recovered_dir / "result.json").read_text(encoding="utf-8"))
    if replay["recovered_from_generation"] != selected["selected_generation"]:
        raise RecoveryVerificationError("replay did not load the selected durable generation")
    replayed_samples = tuple(replay["sample_ids"][1])
    if (
        replayed_samples != tuple(failure["sample_ids"])
        or replay["batch_ids"][1] != failure["batch_id"]
    ):
        raise RecoveryVerificationError("replay did not consume the failed batch and sample IDs")
    if 2 not in replay_worker.commit_receipt_seconds:
        raise RecoveryVerificationError("replay did not acknowledge its durable commit")
    completion_worker = _run_worker(
        config,
        recovered_dir,
        role="completion",
        dataset_manifest=dataset_manifest,
        resume_from=recovered_dir / "checkpoints",
        timeout_seconds=timeout_seconds,
    )
    control = json.loads((control_dir / "result.json").read_text(encoding="utf-8"))
    recovered = json.loads((recovered_dir / "result.json").read_text(encoding="utf-8"))
    control_model, control_state = _load_state(config, control_dir, control)
    recovered_model, recovered_state = _load_state(config, recovered_dir, recovered)
    control_model.eval()
    recovered_model.eval()
    verification = source.batch(source.initial_cursor())
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            control_logits, _ = control_model(verification.inputs)
            recovered_logits, _ = recovered_model(verification.inputs)
    finally:
        torch.set_num_threads(original_threads)
    equality = {
        "batch_id_sequence": control["batch_ids"] == recovered["batch_ids"],
        "sample_id_sequence": control["sample_ids"] == recovered["sample_ids"],
        "loss_sequence": control["losses"] == recovered["losses"],
        "model_tensors": state_trees_equal(control_state["model"], recovered_state["model"]),
        "optimizer_tensors": state_trees_equal(
            control_state["optimizer"], recovered_state["optimizer"]
        ),
        "rng_state": state_trees_equal(control_state["rng_state"], recovered_state["rng_state"]),
        "cursor": control["final_cursor"] == recovered["final_cursor"],
        "completed_steps": control["steps"] == recovered["steps"] == config.steps,
        "token_count": control["tokens_seen"] == recovered["tokens_seen"],
        "final_logits_tensors": torch.equal(control_logits, recovered_logits),
        "final_logits_digest": (
            control["final_logits_digest"]
            == recovered["final_logits_digest"]
            == state_digest(control_logits)
            == state_digest(recovered_logits)
        ),
        "final_checkpoint_state": state_trees_equal(control_state, recovered_state),
        "final_state_digest": control["final_state_digest"] == recovered["final_state_digest"],
        "run_fingerprint": control["run_fingerprint"] == recovered["run_fingerprint"],
        "run_contract": control["run_contract_fingerprint"]
        == recovered["run_contract_fingerprint"],
    }
    cleanup_complete = not any((recovered_dir / "checkpoints").glob(".generation-*.tmp-*"))
    equality["staging_cleanup"] = cleanup_complete
    replay_commit_seconds = replay_worker.commit_receipt_seconds[2]
    report = ProcessRecoveryReportV1(
        schema="ProcessRecoveryReportV1",
        verified_at_utc=datetime.now(timezone.utc).isoformat(),
        data_kind="synthetic-smoke" if dataset_manifest is None else "prepared-dataset",
        failure_point=failure_point,
        process_start_method="spawn",
        worker_pids={
            "control": control_worker.pid,
            "interrupted": interrupted.pid,
            "replay": replay_worker.pid,
            "completion": completion_worker.pid,
        },
        interrupted_exitcode=interrupted.exitcode,
        kill_signal="SIGKILL",
        attempted_step=failure["attempted_step"],
        durable_step_before_kill=failure["durable_step"],
        selected_step=selected["step"],
        selected_generation=selected["selected_generation"],
        selected_tokens_seen=selected["tokens_seen"],
        failed_batch_id=failure["batch_id"],
        replayed_batch_id=replay["batch_ids"][1],
        failed_sample_ids=tuple(failure["sample_ids"]),
        replayed_sample_ids=replayed_samples,
        replayed_tokens=failure["attempted_tokens"],
        discarded_compute_tokens=0
        if failure_point == "before-forward"
        else failure["attempted_tokens"],
        durable_committed_steps_lost=failure["durable_step"] - selected["step"],
        durable_committed_tokens_lost=(
            failure["durable_step"] * failure["attempted_tokens"] - selected["tokens_seen"]
        ),
        abandoned_staging_files=abandoned,
        staging_cleanup_complete=cleanup_complete,
        timings_seconds={
            "control_spawn_to_exit": round(control_worker.elapsed_seconds, 6),
            "kill_request_to_exit": round(interrupted.kill_to_exit_seconds or 0.0, 6),
            "checkpoint_selection_load": replay["checkpoint_load_seconds"],
            "resume_spawn_to_replayed_commit_receipt": round(replay_commit_seconds, 6),
            "replayed_commit_receipt_to_exit": round(
                replay_worker.elapsed_seconds - replay_commit_seconds, 6
            ),
            "completion_spawn_to_exit": round(completion_worker.elapsed_seconds, 6),
        },
        timing_boundaries={
            "control_spawn_to_exit": "Parent Process.start call to joined control worker exit.",
            "kill_request_to_exit": "Parent SIGKILL request to joined interrupted worker exit.",
            "checkpoint_selection_load": "Replay worker checkpoint load call, including integrity selection and state restoration.",
            "resume_spawn_to_replayed_commit_receipt": "Parent Process.start call to receipt of step-2 commit IPC; includes spawn/import, restore, replay, durable save, and IPC delivery.",
            "replayed_commit_receipt_to_exit": "Parent receipt of step-2 commit IPC to joined replay worker exit, including result writing.",
            "completion_spawn_to_exit": "Parent Process.start call to joined final completion worker exit.",
        },
        completed_steps=recovered["steps"],
        tokens_seen=recovered["tokens_seen"],
        config_fingerprint=control["config_fingerprint"],
        dataset_fingerprint=control["data_fingerprint"],
        tokenizer_fingerprint=control["tokenizer_fingerprint"],
        run_contract_fingerprint=control["run_contract_fingerprint"],
        code_fingerprint=control["code_fingerprint"],
        code_revision=control["code_revision"],
        equality=equality,
        exact_equality=all(equality.values()),
        final_state_digest=recovered["final_state_digest"],
        control_run_fingerprint=control["run_fingerprint"],
        recovered_run_fingerprint=recovered["run_fingerprint"],
        artifacts={
            "control_result": "control/result.json",
            "recovered_result": "recovered/result.json",
            "recovered_checkpoints": "recovered/checkpoints",
        },
        environment={
            "device": "cpu",
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "worker_torch_threads": 1,
        },
        limitations=(
            "This report proves recovery after one parent-controlled POSIX SIGKILL in this pinned CPU run.",
            "The boundary handshake pauses one worker; it does not sample arbitrary asynchronous kill times.",
            "The operating system and filesystem remain running; this is not a power-loss durability proof.",
            "It does not measure model quality, GPU behavior, distributed scale, or production readiness.",
            "Timing includes local process startup and IPC; it is not a benchmark or service-level objective.",
            "Checkpoint hashes detect corruption but do not authenticate attacker-controlled artifacts.",
        ),
    )
    report_path = output_dir / "process-recovery-report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
    if not report.exact_equality:
        failed = ", ".join(name for name, passed in equality.items() if not passed)
        raise RecoveryVerificationError(f"process recovery diverged from control: {failed}")
    return report
