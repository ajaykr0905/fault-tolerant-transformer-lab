"""One parent-controlled SIGKILL boundary with four fresh CUDA workers."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fttl.checkpoint import load_checkpoint
from fttl.config import ExperimentConfig
from fttl.cuda_recovery import _advance, _cpu_snapshot, _new_training_objects
from fttl.cuda_runtime import (
    prepare_cuda_execution,
    restore_cuda_rng_state,
    synchronize_cuda,
    validate_cuda_rng_state,
)
from fttl.data import PreparedDatasetBatchSource
from fttl.model import TinyTransformer
from fttl.process_recovery import _run_worker, _validate_timeout
from fttl.recovery import RecoveryVerificationError
from fttl.state import code_fingerprint, git_revision, state_digest, state_trees_equal
from fttl.train import seed_everything


def _contract(config, source, execution):
    return state_digest(
        {
            "schema": "CudaProcessRecoveryContractV1",
            "config": config.to_dict(),
            "dataset": source.data_fingerprint,
            "tokenizer": source.tokenizer_fingerprint,
            "code": code_fingerprint(),
            "execution": execution,
            "dtype": "float32",
            "optimizer": "AdamW:foreach=False:fused=False",
        }
    )


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cuda_worker(
    config_value, run_dir, manifest, resume_from, stop_after_step, failure_point, sender
):
    """The spawn boundary receives configuration, paths and IPC, never live CUDA objects."""
    try:
        sender.send({"event": "started", "pid": os.getpid()})
        torch.set_num_threads(1)
        config = ExperimentConfig.from_dict(config_value)
        device, execution = prepare_cuda_execution()
        if device != torch.device("cuda:0"):
            raise RuntimeError("process recovery requires actual CUDA execution on cuda:0")
        source = PreparedDatasetBatchSource.from_manifest(config, Path(manifest))
        contract = _contract(config, source, execution)
        directory = Path(run_dir)
        if resume_from is None and directory.exists():
            raise ValueError("fresh CUDA worker requires a new output path")
        seed_everything(config.seed)
        model, optimizer = _new_training_objects(config, device)
        payload = None
        load_seconds = 0.0
        if resume_from is not None:
            synchronize_cuda(device)
            started = time.perf_counter()
            payload = load_checkpoint(
                Path(resume_from),
                model=model,
                optimizer=optimizer,
                expected_config=config,
                expected_data_fingerprint=source.data_fingerprint,
                expected_tokenizer_fingerprint=source.tokenizer_fingerprint,
                expected_run_contract_fingerprint=contract,
                restore_rng=False,
            )
            validate_cuda_rng_state(payload["rng_state"], device)
            restore_cuda_rng_state(payload["rng_state"], device)
            synchronize_cuda(device)
            load_seconds = time.perf_counter() - started
        directory.mkdir(parents=True, exist_ok=resume_from is not None)

        def observe(event, values):
            if event == "after-checkpoint":
                synchronize_cuda(device)
                sender.send({"event": "commit", **values})
            elif failure_point is not None and values["attempted_step"] == 2:
                synchronize_cuda(device)
                sender.send({"event": "failure-boundary", "point": "after-optimizer", **values})
                threading.Event().wait()

        state = _advance(
            model,
            optimizer,
            config,
            source,
            device,
            directory / "checkpoints",
            contract,
            config.steps if stop_after_step is None else stop_after_step,
            payload,
            observer=observe,
        )
        synchronize_cuda(device)
        state = _cpu_snapshot(state)
        torch.save(state, directory / "snapshot.pt")
        _write_json(
            directory / "worker-result.json",
            {
                "pid": os.getpid(),
                "execution": execution,
                "contract": contract,
                "config_fingerprint": config.fingerprint(),
                "data_fingerprint": source.data_fingerprint,
                "tokenizer_fingerprint": source.tokenizer_fingerprint,
                "code_fingerprint": code_fingerprint(),
                "code_revision": git_revision(),
                "snapshot_digest": state_digest(state),
                "checkpoint_load_seconds": load_seconds,
                "selected_step": None if payload is None else payload["step"],
                "selected_generation": None if payload is None else payload["selected_generation"],
            },
        )
        sender.send({"event": "completed"})
    except Exception as error:
        sender.send({"event": "error", "error_type": type(error).__name__, "message": str(error)})
        sys.exit(1)
    finally:
        sender.close()


def _read_worker(directory):
    metadata = json.loads((directory / "worker-result.json").read_text(encoding="utf-8"))
    state = torch.load(directory / "snapshot.pt", map_location="cpu", weights_only=True)
    if state_digest(state) != metadata["snapshot_digest"]:
        raise RecoveryVerificationError("worker snapshot differs from its recorded digest")
    return metadata, state


def _load_cpu_checkpoint(config, store, metadata):
    with torch.device("cpu"):
        model = TinyTransformer(config.model).float()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, foreach=False, fused=False
    )
    return load_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_data_fingerprint=metadata["data_fingerprint"],
        expected_tokenizer_fingerprint=metadata["tokenizer_fingerprint"],
        expected_run_contract_fingerprint=metadata["contract"],
        restore_rng=False,
    )


def verify_cuda_process_recovery(config, output_dir, *, dataset_manifest, timeout_seconds=120.0):
    """Parent CPU inspection only; every training/restoration GPU context is independently spawned."""
    if not isinstance(timeout_seconds, (int, float)):
        raise ValueError("timeout_seconds must be a finite positive number")
    _validate_timeout(timeout_seconds)
    if os.name != "posix" or not hasattr(signal, "SIGKILL"):
        raise ValueError("CUDA process recovery requires POSIX SIGKILL")
    if config.steps < 3 or config.checkpoint_every != 1:
        raise ValueError("requires at least 3 steps and checkpoint_every=1")
    if config.model.dropout <= 0:
        raise ValueError("CUDA process recovery requires dropout greater than zero")
    output = Path(output_dir)
    if output.exists():
        raise ValueError("CUDA process recovery requires a fresh output path")
    if torch.cuda.is_initialized():
        raise RuntimeError("run the parent in a fresh process without initializing CUDA")
    source = PreparedDatasetBatchSource.from_manifest(config, Path(dataset_manifest))
    parent_code = code_fingerprint()
    parent_revision = git_revision()
    output.mkdir(parents=True, exist_ok=False)
    control_dir, recovered_dir = output / "control", output / "recovered"
    arguments = {
        "dataset_manifest": Path(dataset_manifest),
        "timeout_seconds": timeout_seconds,
        "worker_target": _cuda_worker,
    }
    control_worker = _run_worker(config, control_dir, role="control", **arguments)
    control_metadata, control = _read_worker(control_dir)
    if (
        control_metadata["pid"] != control_worker.pid
        or control_metadata["execution"].get("device") != "cuda:0"
        or control_metadata["config_fingerprint"] != config.fingerprint()
        or control_metadata["data_fingerprint"] != source.data_fingerprint
        or control_metadata["tokenizer_fingerprint"] != source.tokenizer_fingerprint
        or control_metadata["code_fingerprint"] != parent_code
        or control_metadata["code_revision"] != parent_revision
        or control_metadata["contract"] != _contract(config, source, control_metadata["execution"])
    ):
        raise RecoveryVerificationError(
            "control worker does not match the parent execution contract"
        )
    interrupted = _run_worker(
        config, recovered_dir, role="interrupted", failure_point="after-optimizer", **arguments
    )
    failure = interrupted.failure
    selected = _load_cpu_checkpoint(config, recovered_dir / "checkpoints", control_metadata)
    if (
        failure is None
        or interrupted.exitcode != -signal.SIGKILL
        or failure["attempted_step"] != 2
        or failure["durable_step"] != 1
        or selected["step"] != 1
        or selected["selected_generation"] != 1
        or selected["tokens_seen"] != config.batch_size * config.model.block_size
    ):
        raise RecoveryVerificationError("SIGKILL did not preserve durable step 1 before step 2")
    replay_worker = _run_worker(
        config,
        recovered_dir,
        role="replay",
        resume_from=recovered_dir / "checkpoints",
        stop_after_step=2,
        **arguments,
    )
    replay_metadata, replay = _read_worker(recovered_dir)
    if (
        replay_metadata["pid"] != replay_worker.pid
        or replay_metadata["selected_step"] != 1
        or replay_metadata["selected_generation"] != 1
        or replay["steps"] != 2
        or replay["batch_ids"][1] != failure["batch_id"]
        or replay["sample_ids"][1] != tuple(failure["sample_ids"])
        or 2 not in replay_worker.commit_receipt_seconds
    ):
        raise RecoveryVerificationError("CUDA replay did not durably recommit the failed batch")
    completion_worker = _run_worker(
        config,
        recovered_dir,
        role="completion",
        resume_from=recovered_dir / "checkpoints",
        **arguments,
    )
    recovered_metadata, recovered = _read_worker(recovered_dir)
    equality = {
        key: state_trees_equal(control[key], recovered[key])
        for key in (
            "model",
            "optimizer",
            "rng",
            "logits",
            "steps",
            "tokens_seen",
            "losses",
            "batch_ids",
            "sample_ids",
            "cursor",
            "generation",
        )
    }
    for name, metadata in (("replay", replay_metadata), ("completion", recovered_metadata)):
        equality[f"{name}_execution_contract"] = all(
            metadata[key] == control_metadata[key]
            for key in (
                "execution",
                "contract",
                "config_fingerprint",
                "data_fingerprint",
                "tokenizer_fingerprint",
                "code_fingerprint",
                "code_revision",
            )
        )
    final_checkpoint = _load_cpu_checkpoint(config, recovered_dir / "checkpoints", control_metadata)
    equality["final_checkpoint_model"] = state_trees_equal(
        final_checkpoint["model"], recovered["model"]
    )
    equality["final_checkpoint_optimizer"] = state_trees_equal(
        final_checkpoint["optimizer"], recovered["optimizer"]
    )
    equality["final_checkpoint_rng"] = state_trees_equal(
        final_checkpoint["rng_state"], recovered["rng"]
    )
    equality["parent_cuda_uninitialized"] = not torch.cuda.is_initialized()
    equality["dataset_contract"] = control_metadata["data_fingerprint"] == source.data_fingerprint
    equality["completion_selected_step"] = recovered_metadata["selected_step"] == 2
    equality["completion_worker_pid"] = recovered_metadata["pid"] == completion_worker.pid
    pids = [control_worker.pid, interrupted.pid, replay_worker.pid, completion_worker.pid]
    equality["independent_worker_pids"] = len(set(pids)) == 4 and os.getpid() not in pids
    equality["expected_completed_state"] = (
        recovered["steps"] == config.steps
        and recovered["tokens_seen"] == config.steps * config.batch_size * config.model.block_size
        and recovered["cursor"] == source.cursor_at(config.steps).to_dict()
    )
    if not all(equality.values()):
        raise RecoveryVerificationError(f"CUDA process recovery diverged: {equality}")
    report = {
        "schema": "CudaProcessRecoveryReportV1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "execution": control_metadata["execution"],
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": source.data_fingerprint,
        "tokenizer_fingerprint": source.tokenizer_fingerprint,
        "code_fingerprint": control_metadata["code_fingerprint"],
        "code_revision": control_metadata["code_revision"],
        "run_contract_fingerprint": control_metadata["contract"],
        "failure_point": "after-optimizer",
        "process_start_method": "spawn",
        "kill_signal": "SIGKILL",
        "worker_pids": {
            "control": control_worker.pid,
            "interrupted": interrupted.pid,
            "replay": replay_worker.pid,
            "completion": completion_worker.pid,
        },
        "interrupted_exitcode": interrupted.exitcode,
        "attempted_step": 2,
        "durable_step_before_kill": 1,
        "selected_step": selected["step"],
        "selected_generation": selected["selected_generation"],
        "failed_batch_id": failure["batch_id"],
        "replayed_batch_id": replay["batch_ids"][1],
        "failed_sample_ids": list(failure["sample_ids"]),
        "replayed_sample_ids": list(replay["sample_ids"][1]),
        "discarded_compute_tokens": failure["attempted_tokens"],
        "durable_committed_steps_lost": 0,
        "durable_committed_tokens_lost": 0,
        "steps": recovered["steps"],
        "tokens_seen": recovered["tokens_seen"],
        "equality": equality,
        "exact_equality": True,
        "final_state_digest": state_digest(recovered),
        "timings_seconds": {
            "control_spawn_to_exit": control_worker.elapsed_seconds,
            "kill_request_to_exit": interrupted.kill_to_exit_seconds,
            "checkpoint_selection_load": replay_metadata["checkpoint_load_seconds"],
            "resume_spawn_to_replayed_commit_receipt": replay_worker.commit_receipt_seconds[2],
            "completion_spawn_to_exit": completion_worker.elapsed_seconds,
        },
        "timing_boundaries": {
            "checkpoint_selection_load": "Replay worker load, CPU/CUDA RNG restoration and CUDA synchronization.",
            "resume_spawn_to_replayed_commit_receipt": "Parent spawn to receipt after replayed durable save and CUDA synchronization; includes imports, restore and IPC.",
            "kill_request_to_exit": "Parent SIGKILL request to joined worker exit.",
            "control_spawn_to_exit": "Parent spawn to joined control exit, including synchronized GPU work and CPU snapshot writing.",
            "completion_spawn_to_exit": "Parent spawn to joined completion exit, including synchronized GPU work and CPU snapshot writing.",
        },
        "limitations": [
            "One parent-paused after-optimizer boundary on one CUDA device; not arbitrary asynchronous kill times.",
            "The OS, GPU driver and filesystem remain running; not power loss or device failure.",
            "Exact equality applies only to this FP32 eager hardware/software execution contract.",
            "No cross-device, mixed-precision, distributed-scale, quality, serving or production evidence.",
            "Local spawn/IPC measurements are observations, not benchmarks or service-level objectives.",
            "CPU snapshots and checkpoints are trusted local artifacts; hashes are not authentication.",
        ],
    }
    _write_json(output / "cuda-process-recovery-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    args = parser.parse_args()
    result = verify_cuda_process_recovery(
        ExperimentConfig.from_json(args.config.read_text(encoding="utf-8")),
        args.output,
        dataset_manifest=args.dataset_manifest,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
