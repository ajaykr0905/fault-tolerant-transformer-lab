from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from fttl.checkpoint import load_checkpoint
from fttl.config import ExperimentConfig
from fttl.model import TinyTransformer
from fttl.state import state_trees_equal
from fttl.train import FailurePoint, InjectedTrainingFailure, TrainingResult, run_training


@dataclass(frozen=True)
class FailureAttemptV1:
    failure_point: str
    attempted_step: int
    committed_step: int
    batch_id: str
    sample_ids: tuple[str, ...]
    replayed_sample_ids: tuple[str, ...]
    attempted_tokens: int
    replayed_tokens: int
    discarded_compute_tokens: int
    selected_generation: int | None
    checkpoint_load_seconds: float
    restart_to_replayed_commit_seconds: float


@dataclass(frozen=True)
class RecoveryReportV1:
    schema: str
    failure_point: str
    requested_restarts: int
    completed_restarts: int
    attempts: tuple[FailureAttemptV1, ...]
    selected_generations: tuple[int, ...]
    max_recovery_point_lag_steps: int
    durable_committed_steps_lost: int
    durable_committed_tokens_lost: int
    replayed_steps: int
    replayed_tokens: int
    discarded_compute_tokens: int
    checkpoint_selection_load_seconds: float
    recovery_duration_seconds: float
    ordinary_completion_seconds: float
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
    hardware: dict[str, object]
    limitations: tuple[str, ...]


class RecoveryVerificationError(AssertionError):
    pass


def _write_report(path: Path, report: RecoveryReportV1) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(report), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _checkpoint_state(
    config: ExperimentConfig,
    result: TrainingResult,
    run_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    load_checkpoint(
        run_dir / result.checkpoint,
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_data_fingerprint=result.data_fingerprint,
        expected_tokenizer_fingerprint=result.tokenizer_fingerprint,
        expected_run_contract_fingerprint=result.run_contract_fingerprint,
        restore_rng=False,
    )
    return model.state_dict(), optimizer.state_dict()


def verify_recovery(
    config: ExperimentConfig,
    dataset_manifest: Path,
    output_dir: Path,
    *,
    failure_point: FailurePoint,
    restarts: int,
) -> RecoveryReportV1:
    if config.checkpoint_every != 1:
        raise ValueError("recovery verification requires checkpoint_every=1")
    if restarts < 1:
        raise ValueError("restarts must be at least 1")
    if config.steps < restarts + 2:
        raise ValueError("config.steps must leave one final step after all restarts")
    control_dir = output_dir / "control"
    recovered_dir = output_dir / "recovered"
    if control_dir.exists() or recovered_dir.exists():
        raise ValueError("recovery output already exists; choose a fresh output directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    control_model, control = run_training(
        config,
        control_dir,
        dataset_manifest=dataset_manifest,
    )
    attempts: list[FailureAttemptV1] = []
    selected_generations: list[int] = []
    recovery_seconds = 0.0
    resume_from: Path | None = None
    recovered: TrainingResult | None = None

    for restart_index in range(restarts):
        target_step = restart_index + 2
        try:
            run_training(
                config,
                recovered_dir,
                resume_from=resume_from,
                dataset_manifest=dataset_manifest,
                failure_point=failure_point,
                failure_step=target_step,
            )
        except InjectedTrainingFailure as failure:
            started = time.perf_counter()
            _, recovered = run_training(
                config,
                recovered_dir,
                resume_from=Path(recovered_dir / "checkpoints"),
                stop_after_step=target_step,
                dataset_manifest=dataset_manifest,
            )
            restart_to_commit_seconds = time.perf_counter() - started
            recovery_seconds += restart_to_commit_seconds
            replayed = recovered.sample_ids[target_step - 1]
            if replayed != failure.sample_ids:
                raise RecoveryVerificationError(
                    "failed step did not replay the same sample identifiers"
                )
            selected = recovered.recovered_from_generation
            if selected is not None:
                selected_generations.append(selected)
            attempts.append(
                FailureAttemptV1(
                    failure_point=failure.point,
                    attempted_step=failure.attempted_step,
                    committed_step=failure.durable_step,
                    batch_id=failure.batch_id,
                    sample_ids=failure.sample_ids,
                    replayed_sample_ids=replayed,
                    attempted_tokens=failure.attempted_tokens,
                    replayed_tokens=failure.attempted_tokens,
                    discarded_compute_tokens=(
                        0
                        if failure.point == "before-forward"
                        else failure.attempted_tokens
                    ),
                    selected_generation=selected,
                    checkpoint_load_seconds=recovered.checkpoint_load_seconds,
                    restart_to_replayed_commit_seconds=round(
                        restart_to_commit_seconds, 6
                    ),
                )
            )
            resume_from = recovered_dir / recovered.checkpoint
        else:
            raise RecoveryVerificationError("configured failure point was not reached")

    assert recovered is not None
    completion_started = time.perf_counter()
    recovered_model, recovered = run_training(
        config,
        recovered_dir,
        resume_from=recovered_dir / recovered.checkpoint,
        dataset_manifest=dataset_manifest,
    )
    ordinary_completion_seconds = time.perf_counter() - completion_started
    if recovered.recovered_from_generation is not None:
        selected_generations.append(recovered.recovered_from_generation)

    control_model_state, control_optimizer_state = _checkpoint_state(
        config, control, control_dir
    )
    recovered_model_state, recovered_optimizer_state = _checkpoint_state(
        config, recovered, recovered_dir
    )
    equality = {
        "batch_id_sequence": control.batch_ids == recovered.batch_ids,
        "sample_id_sequence": control.sample_ids == recovered.sample_ids,
        "loss_sequence": control.losses == recovered.losses,
        "model_tensors": state_trees_equal(control_model_state, recovered_model_state),
        "optimizer_tensors": state_trees_equal(
            control_optimizer_state, recovered_optimizer_state
        ),
        "rng_state": control.rng_digest == recovered.rng_digest,
        "cursor": control.final_cursor == recovered.final_cursor,
        "completed_steps": control.steps == recovered.steps,
        "token_count": control.tokens_seen == recovered.tokens_seen,
        "final_logits": control.final_logits_digest == recovered.final_logits_digest,
        "final_state_digest": control.final_state_digest == recovered.final_state_digest,
        "run_fingerprint": control.run_fingerprint == recovered.run_fingerprint,
    }
    # Also compare the returned final model, independently of checkpoint reload.
    equality["returned_model_tensors"] = state_trees_equal(
        control_model.state_dict(), recovered_model.state_dict()
    )
    report = RecoveryReportV1(
        schema="RecoveryReportV1",
        failure_point=failure_point,
        requested_restarts=restarts,
        completed_restarts=len(attempts),
        attempts=tuple(attempts),
        selected_generations=tuple(selected_generations),
        max_recovery_point_lag_steps=max(
            attempt.attempted_step - attempt.committed_step for attempt in attempts
        ),
        durable_committed_steps_lost=0,
        durable_committed_tokens_lost=0,
        replayed_steps=len(attempts),
        replayed_tokens=sum(attempt.replayed_tokens for attempt in attempts),
        discarded_compute_tokens=sum(
            attempt.discarded_compute_tokens for attempt in attempts
        ),
        checkpoint_selection_load_seconds=round(
            sum(attempt.checkpoint_load_seconds for attempt in attempts), 6
        ),
        recovery_duration_seconds=round(recovery_seconds, 6),
        ordinary_completion_seconds=round(ordinary_completion_seconds, 6),
        config_fingerprint=config.fingerprint(),
        dataset_fingerprint=control.data_fingerprint,
        tokenizer_fingerprint=control.tokenizer_fingerprint,
        run_contract_fingerprint=control.run_contract_fingerprint,
        code_fingerprint=control.code_fingerprint,
        code_revision=control.code_revision,
        equality=equality,
        exact_equality=all(equality.values()),
        final_state_digest=recovered.final_state_digest,
        control_run_fingerprint=control.run_fingerprint,
        recovered_run_fingerprint=recovered.run_fingerprint,
        hardware={
            "device": "cpu",
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        limitations=(
            "This report proves deterministic CPU recovery for this pinned run only.",
            "It does not measure model quality, GPU behavior, distributed scale, or production readiness.",
            "Recovery duration is local wall-clock evidence and is not a service-level objective.",
            "Checkpoint hashes detect corruption but do not authenticate attacker-controlled artifacts.",
        ),
    )
    _write_report(output_dir / "recovery-report.json", report)
    if not report.exact_equality:
        failed = ", ".join(name for name, passed in equality.items() if not passed)
        raise RecoveryVerificationError(f"recovery diverged from control: {failed}")
    return report
