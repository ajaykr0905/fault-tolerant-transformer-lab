from __future__ import annotations

import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from fttl import __version__
from fttl.checkpoint import load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig
from fttl.data import (
    BatchSource,
    PreparedDatasetBatchSource,
    SyntheticBatchSource,
    TrainingCursorV1,
)
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, code_fingerprint, git_revision, state_digest


FailurePoint = Literal[
    "before-forward",
    "after-backward",
    "after-optimizer",
    "during-checkpoint-write",
]
FAILURE_POINTS: tuple[FailurePoint, ...] = (
    "before-forward",
    "after-backward",
    "after-optimizer",
    "during-checkpoint-write",
)


class InjectedTrainingFailure(RuntimeError):
    """A deterministic process-crash surrogate used by the recovery verifier."""

    def __init__(
        self,
        *,
        point: FailurePoint,
        attempted_step: int,
        durable_step: int,
        batch_id: str,
        sample_ids: tuple[str, ...],
        attempted_tokens: int,
    ) -> None:
        super().__init__(
            f"injected failure at {point} for step {attempted_step} "
            f"(durable step {durable_step})"
        )
        self.point = point
        self.attempted_step = attempted_step
        self.durable_step = durable_step
        self.batch_id = batch_id
        self.sample_ids = sample_ids
        self.attempted_tokens = attempted_tokens


@dataclass(frozen=True)
class TrainingResult:
    schema_version: int
    steps: int
    tokens_seen: int
    initial_loss: float
    final_loss: float
    losses: tuple[float, ...]
    batch_ids: tuple[str, ...]
    sample_ids: tuple[tuple[str, ...], ...]
    final_cursor: dict[str, object]
    elapsed_seconds: float
    checkpoint: str
    checkpoint_generation: int
    recovered_from_generation: int | None
    checkpoint_load_seconds: float
    config_fingerprint: str
    data_fingerprint: str
    tokenizer_fingerprint: str
    run_contract_fingerprint: str
    code_fingerprint: str
    code_revision: str
    model_digest: str
    optimizer_digest: str
    rng_digest: str
    final_logits_digest: str
    final_state_digest: str
    run_fingerprint: str
    device: str
    torch_version: str
    python_version: str
    limitations: tuple[str, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _batch_source(
    config: ExperimentConfig, dataset_manifest: Path | None
) -> BatchSource:
    if dataset_manifest is None:
        return SyntheticBatchSource(config)
    return PreparedDatasetBatchSource.from_manifest(config, dataset_manifest)


def _raise_injected(
    point: FailurePoint,
    *,
    attempted_step: int,
    durable_step: int,
    batch_id: str,
    sample_ids: tuple[str, ...],
    attempted_tokens: int,
) -> None:
    raise InjectedTrainingFailure(
        point=point,
        attempted_step=attempted_step,
        durable_step=durable_step,
        batch_id=batch_id,
        sample_ids=sample_ids,
        attempted_tokens=attempted_tokens,
    )


def run_training(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
    stop_after_step: int | None = None,
    dataset_manifest: Path | None = None,
    failure_point: FailurePoint | None = None,
    failure_step: int | None = None,
) -> tuple[TinyTransformer, TrainingResult]:
    if stop_after_step is not None and stop_after_step < 1:
        raise ValueError("stop_after_step must be at least 1")
    if failure_point is not None and failure_point not in FAILURE_POINTS:
        raise ValueError(f"unknown failure point {failure_point!r}")
    if failure_step is not None and failure_point is None:
        raise ValueError("failure_step requires failure_point")

    source = _batch_source(config, dataset_manifest)
    implementation_fingerprint = code_fingerprint()
    run_contract_fingerprint = state_digest(
        {
            "schema": "RunContractV1",
            "package_version": __version__,
            "command_contract": "fttl-train-smoke-v2",
            "model_architecture": "TinyTransformer-v1",
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": source.data_fingerprint,
            "tokenizer_fingerprint": source.tokenizer_fingerprint,
            "code_fingerprint": implementation_fingerprint,
        }
    )
    seed_everything(config.seed)
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    start_step = 0
    durable_step = 0
    tokens_seen = 0
    losses: list[float] = []
    batch_ids: list[str] = []
    sample_ids: list[tuple[str, ...]] = []
    cursor = source.initial_cursor()
    recovered_from_generation: int | None = None
    checkpoint_load_seconds = 0.0

    if resume_from is not None:
        legacy_synthetic = Path(resume_from).is_file() and dataset_manifest is None
        load_started = time.perf_counter()
        payload = load_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            expected_config=config,
            expected_data_fingerprint=(
                None if legacy_synthetic else source.data_fingerprint
            ),
            expected_tokenizer_fingerprint=(
                None if legacy_synthetic else source.tokenizer_fingerprint
            ),
            expected_run_contract_fingerprint=(
                None if legacy_synthetic else run_contract_fingerprint
            ),
        )
        checkpoint_load_seconds = round(time.perf_counter() - load_started, 6)
        start_step = durable_step = int(payload["step"])
        tokens_seen = int(payload["tokens_seen"])
        losses = [float(value) for value in payload["losses"]]
        batch_ids = [str(value) for value in payload.get("batch_ids", [])]
        sample_ids = [
            tuple(str(sample_id) for sample_id in batch)
            for batch in payload.get("sample_ids", [])
        ]
        if legacy_synthetic and not batch_ids:
            for batch_index in range(start_step):
                historical = source.batch(source.cursor_at(batch_index))
                batch_ids.append(historical.batch_id)
                sample_ids.append(historical.sample_ids)
        raw_cursor = payload.get("cursor")
        cursor = (
            TrainingCursorV1.from_dict(raw_cursor)
            if isinstance(raw_cursor, dict)
            else source.cursor_at(start_step)
        )
        if cursor.batch_index != start_step:
            raise ValueError("checkpoint cursor does not follow its completed step")
        if len(losses) != start_step or len(batch_ids) != start_step:
            raise ValueError("checkpoint histories do not match its completed step")
        if sample_ids and len(sample_ids) != start_step:
            raise ValueError("checkpoint sample history does not match its completed step")
        recovered_from_generation = int(payload.get("selected_generation", 0))

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoints"
    final_step = min(
        config.steps,
        stop_after_step if stop_after_step is not None else config.steps,
    )
    if final_step <= start_step:
        raise ValueError(
            "requested training boundary must be greater than the checkpoint's completed step"
        )
    if failure_point is not None:
        failure_step = failure_step if failure_step is not None else start_step + 1
        if not start_step < failure_step <= final_step:
            raise ValueError("failure_step must be inside the requested training range")
        if (
            failure_point == "during-checkpoint-write"
            and failure_step % config.checkpoint_every != 0
            and failure_step != final_step
        ):
            raise ValueError("checkpoint-write failure must target a checkpoint boundary")

    started = time.perf_counter()
    checkpoint_generation = 0
    model.train()
    for step_index in range(start_step, final_step):
        attempted_step = step_index + 1
        prepared = source.batch(cursor)
        attempted_tokens = prepared.inputs.numel()
        if failure_point == "before-forward" and attempted_step == failure_step:
            _raise_injected(
                "before-forward",
                attempted_step=attempted_step,
                durable_step=durable_step,
                batch_id=prepared.batch_id,
                sample_ids=prepared.sample_ids,
                attempted_tokens=attempted_tokens,
            )

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(prepared.inputs, prepared.targets)
        assert loss is not None
        loss.backward()
        if failure_point == "after-backward" and attempted_step == failure_step:
            _raise_injected(
                "after-backward",
                attempted_step=attempted_step,
                durable_step=durable_step,
                batch_id=prepared.batch_id,
                sample_ids=prepared.sample_ids,
                attempted_tokens=attempted_tokens,
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if failure_point == "after-optimizer" and attempted_step == failure_step:
            _raise_injected(
                "after-optimizer",
                attempted_step=attempted_step,
                durable_step=durable_step,
                batch_id=prepared.batch_id,
                sample_ids=prepared.sample_ids,
                attempted_tokens=attempted_tokens,
            )

        losses.append(float(loss.detach()))
        batch_ids.append(prepared.batch_id)
        sample_ids.append(prepared.sample_ids)
        tokens_seen += attempted_tokens
        cursor = prepared.next_cursor
        should_checkpoint = (
            attempted_step % config.checkpoint_every == 0
            or attempted_step == final_step
        )
        if should_checkpoint:
            failure_injector = None
            if failure_point == "during-checkpoint-write" and attempted_step == failure_step:

                def failure_injector(stage: str) -> None:
                    if stage == "after-state-serialize":
                        _raise_injected(
                            "during-checkpoint-write",
                            attempted_step=attempted_step,
                            durable_step=durable_step,
                            batch_id=prepared.batch_id,
                            sample_ids=prepared.sample_ids,
                            attempted_tokens=attempted_tokens,
                        )

            manifest = save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                step=attempted_step,
                tokens_seen=tokens_seen,
                losses=losses,
                cursor=cursor,
                batch_ids=batch_ids,
                sample_ids=sample_ids,
                data_fingerprint=source.data_fingerprint,
                tokenizer_fingerprint=source.tokenizer_fingerprint,
                run_contract_fingerprint=run_contract_fingerprint,
                failure_injector=failure_injector,
            )
            checkpoint_generation = manifest.generation
            durable_step = attempted_step

    if not losses:
        raise ValueError("checkpoint already completed the requested training range")

    rng_state = capture_rng_state()
    model_digest = state_digest(model.state_dict())
    optimizer_digest = state_digest(optimizer.state_dict())
    rng_digest = state_digest(rng_state)
    model.eval()
    verification_batch = source.batch(source.initial_cursor())
    with torch.no_grad():
        final_logits, _ = model(verification_batch.inputs)
    final_logits_digest = state_digest(final_logits)
    final_cursor = cursor.to_dict()
    final_state_digest = state_digest(
        {
            "model_digest": model_digest,
            "optimizer_digest": optimizer_digest,
            "rng_digest": rng_digest,
            "cursor": final_cursor,
            "losses": losses,
            "batch_ids": batch_ids,
            "sample_ids": sample_ids,
            "steps": final_step,
            "tokens_seen": tokens_seen,
            "final_logits_digest": final_logits_digest,
        }
    )
    run_fingerprint = state_digest(
        {
            "schema": "RunFingerprintV1",
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": source.data_fingerprint,
            "tokenizer_fingerprint": source.tokenizer_fingerprint,
            "run_contract_fingerprint": run_contract_fingerprint,
            "final_state_digest": final_state_digest,
        }
    )
    result = TrainingResult(
        schema_version=2,
        steps=final_step,
        tokens_seen=tokens_seen,
        initial_loss=losses[0],
        final_loss=losses[-1],
        losses=tuple(losses),
        batch_ids=tuple(batch_ids),
        sample_ids=tuple(sample_ids),
        final_cursor=final_cursor,
        elapsed_seconds=round(time.perf_counter() - started, 6),
        checkpoint="checkpoints",
        checkpoint_generation=checkpoint_generation,
        recovered_from_generation=recovered_from_generation,
        checkpoint_load_seconds=checkpoint_load_seconds,
        config_fingerprint=config.fingerprint(),
        data_fingerprint=source.data_fingerprint,
        tokenizer_fingerprint=source.tokenizer_fingerprint,
        run_contract_fingerprint=run_contract_fingerprint,
        code_fingerprint=implementation_fingerprint,
        code_revision=git_revision(),
        model_digest=model_digest,
        optimizer_digest=optimizer_digest,
        rng_digest=rng_digest,
        final_logits_digest=final_logits_digest,
        final_state_digest=final_state_digest,
        run_fingerprint=run_fingerprint,
        device="cpu",
        torch_version=torch.__version__,
        python_version=platform.python_version(),
        limitations=(
            "This run verifies deterministic CPU recovery behavior, not model quality.",
            "It does not demonstrate GPU scale, distributed training, or production readiness.",
            "Full resume checkpoints are trusted local artifacts; SHA-256 is not authentication.",
        ),
    )
    _write_json(output_dir / "result.json", asdict(result))
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": "RunManifestV2",
            "config": config.to_dict(),
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": source.data_fingerprint,
            "dataset_manifest": {
                "provided": dataset_manifest is not None,
                "fingerprint": source.data_fingerprint,
            },
            "tokenizer_fingerprint": source.tokenizer_fingerprint,
            "run_contract_fingerprint": run_contract_fingerprint,
            "code_fingerprint": implementation_fingerprint,
            "code_revision": git_revision(),
            "package_version": __version__,
        },
    )
    return model, result
