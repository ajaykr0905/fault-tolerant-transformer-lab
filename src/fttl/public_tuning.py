"""Paired, validation-only full versus low-rank tuning on verified byte documents."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from fttl.config import ExperimentConfig
from fttl.data import PreparedDatasetBatchSource
from fttl.dataset import PreparedDatasetSnapshot, load_dataset_snapshot
from fttl.evaluation import EvaluationResultV1, evaluate_held_out
from fttl.lora import inject_lora
from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state
from fttl.reports import publish_json
from fttl.state import (
    capture_rng_state,
    code_fingerprint,
    git_revision,
    restore_rng_state,
    state_digest,
    state_trees_equal,
)


@dataclass(frozen=True)
class PublicTuningWindow:
    sample_id: str
    document_id: str
    token_window_offset: int


@dataclass(frozen=True)
class PublicTuningStep:
    step: int
    loss: float
    target_tokens: int
    cumulative_target_tokens: int
    batch_id: str
    windows: tuple[PublicTuningWindow, ...]
    batch_tokens_digest: str
    cpu_rng_before_forward_digest: str


@dataclass(frozen=True)
class PublicTuningRun:
    mode: str
    trainable_parameters: int
    total_parameters: int
    trainable_parameter_names: tuple[str, ...]
    completed_steps: int
    target_tokens: int
    initial_loss: float
    final_loss: float
    trace: tuple[PublicTuningStep, ...]
    validation_before: EvaluationResultV1
    validation_after: EvaluationResultV1
    initial_logits_digest: str
    model_state_digest_before: str
    model_state_digest_after: str
    base_state_digest_before: str
    base_state_digest_after: str
    base_state_unchanged: bool
    base_parameters_frozen: bool
    final_training_rng_digest: str
    elapsed_seconds: float


@dataclass(frozen=True)
class PublicTuningComparison:
    schema_version: int
    config_fingerprint: str
    data_fingerprint: str
    tokenizer_fingerprint: str
    code_fingerprint: str
    code_revision: str
    base_state_digest: str
    comparison_fingerprint: str
    comparison_contract: dict[str, object]
    device: str
    python_version: str
    torch_version: str
    matched_initial_predictions: bool
    paired_batches: bool
    paired_cpu_rng_sequence: bool
    full: PublicTuningRun
    lora: PublicTuningRun
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _base_state(
    model: TinyTransformer, base_names: tuple[str, ...], replaced_modules: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Read the original named tensors, excluding adapters and undoing wrapper names."""
    state = model.state_dict()
    result = {}
    for name in base_names:
        current_name = name
        for module_path in replaced_modules:
            prefix = f"{module_path}."
            if name.startswith(prefix):
                current_name = f"{module_path}.base.{name[len(prefix) :]}"
                break
        result[name] = state[current_name]
    return result


def _initial_logits_digest(model: TinyTransformer, source: PreparedDatasetBatchSource) -> str:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            logits, _ = model(source.batch(source.initial_cursor()).inputs)
        require_finite_state(logits, "initial verification logits")
        return state_digest(logits)
    finally:
        for module, training in modes:
            module.training = training


def _train_arm(
    model: TinyTransformer,
    config: ExperimentConfig,
    source: PreparedDatasetBatchSource,
    manifest_path: Path | PreparedDatasetSnapshot,
    *,
    mode: str,
    validation_before: EvaluationResultV1,
    initial_logits_digest: str,
    base_names: tuple[str, ...],
    replaced_modules: tuple[str, ...],
    max_eval_tokens: int,
) -> PublicTuningRun:
    trainable = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    parameters = [value for _, value in trainable]
    if mode == "lora" and any(
        name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"} for name, _ in trainable
    ):
        raise ValueError("LoRA arm has trainable base parameters")
    if not parameters:
        raise ValueError("tuning arm has no trainable parameters")
    frozen = all(
        not value.requires_grad
        for name, value in model.named_parameters()
        if name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"}
    )
    base_before = copy.deepcopy(_base_state(model, base_names, replaced_modules))
    model_before = state_digest(model.state_dict())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    require_finite_state(model.state_dict(), "model state")
    require_finite_state(optimizer.state_dict(), "optimizer state")
    cursor = source.initial_cursor()
    trace: list[PublicTuningStep] = []
    tokens_seen = 0
    started = time.perf_counter()
    model.train()
    for step in range(1, config.steps + 1):
        batch = source.batch(cursor)
        optimizer.zero_grad(set_to_none=True)
        rng_digest = state_digest(torch.get_rng_state())
        _, loss = model(batch.inputs, batch.targets)
        assert loss is not None
        require_finite_state(loss, "training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        require_finite_state(model.state_dict(), "model state")
        require_finite_state(optimizer.state_dict(), "optimizer state")
        tokens_seen += batch.targets.numel()
        windows = tuple(
            PublicTuningWindow(
                sample_id, sample_id.rsplit(":", 1)[0], int(sample_id.rsplit(":", 1)[1])
            )
            for sample_id in batch.sample_ids
        )
        trace.append(
            PublicTuningStep(
                step=step,
                loss=float(loss.detach()),
                target_tokens=batch.targets.numel(),
                cumulative_target_tokens=tokens_seen,
                batch_id=batch.batch_id,
                windows=windows,
                batch_tokens_digest=state_digest(
                    {"inputs": batch.inputs, "targets": batch.targets}
                ),
                cpu_rng_before_forward_digest=rng_digest,
            )
        )
        cursor = batch.next_cursor
    final_rng_digest = state_digest(capture_rng_state())
    base_after = _base_state(model, base_names, replaced_modules)
    base_unchanged = state_trees_equal(base_before, base_after)
    if mode == "lora" and (not frozen or not base_unchanged):
        raise ValueError("LoRA arm changed or unfroze base weights")
    validation_after = evaluate_held_out(
        model, manifest_path, split="validation", max_tokens=max_eval_tokens
    )
    if (
        validation_after.data_fingerprint != source.data_fingerprint
        or validation_after.tokenizer_fingerprint != source.tokenizer_fingerprint
    ):
        raise ValueError("held-out dataset identity changed during the comparison")
    return PublicTuningRun(
        mode=mode,
        trainable_parameters=sum(value.numel() for value in parameters),
        total_parameters=model.parameter_count(),
        trainable_parameter_names=tuple(name for name, _ in trainable),
        completed_steps=len(trace),
        target_tokens=tokens_seen,
        initial_loss=trace[0].loss,
        final_loss=trace[-1].loss,
        trace=tuple(trace),
        validation_before=validation_before,
        validation_after=validation_after,
        initial_logits_digest=initial_logits_digest,
        model_state_digest_before=model_before,
        model_state_digest_after=state_digest(model.state_dict()),
        base_state_digest_before=state_digest(base_before),
        base_state_digest_after=state_digest(base_after),
        base_state_unchanged=base_unchanged,
        base_parameters_frozen=frozen,
        final_training_rng_digest=final_rng_digest,
        elapsed_seconds=round(time.perf_counter() - started, 6),
    )


def compare_public_tuning(
    config: ExperimentConfig,
    manifest_path: Path,
    output: Path,
    *,
    rank: int = 4,
    max_eval_tokens: int = 4096,
) -> PublicTuningComparison:
    """Publish one paired CPU experiment without training on validation or test data.

    The base is randomly initialized, not a pretrained language model. Caller CPU
    RNGs and deterministic-algorithm settings are restored even when a run fails.
    """
    output, manifest_path = Path(output), Path(manifest_path)
    if output.exists() or output.is_symlink():
        raise ValueError("public tuning comparison requires a fresh output file")
    if config.model.vocab_size != 257:
        raise ValueError("public byte tuning requires model vocab_size=257")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= config.model.d_model:
        raise ValueError("rank must be an integer between 1 and model d_model")
    if (
        isinstance(max_eval_tokens, bool)
        or not isinstance(max_eval_tokens, int)
        or max_eval_tokens < 1
    ):
        raise ValueError("max_eval_tokens must be a positive integer")

    caller_rng = capture_rng_state()
    caller_deterministic = torch.are_deterministic_algorithms_enabled()
    caller_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with torch.device("cpu"):
            snapshot = load_dataset_snapshot(manifest_path)
            manifest = snapshot.manifest
            source = PreparedDatasetBatchSource.from_snapshot(config, snapshot, split="train")
            random.seed(config.seed)
            np.random.seed(config.seed)
            # This experiment must not seed or initialize any CUDA generators.
            torch.random.default_generator.manual_seed(config.seed)
            torch.use_deterministic_algorithms(True)
            full = TinyTransformer(config.model)
            base_names = tuple(full.state_dict())
            base_digest = state_digest(full.state_dict())
            lora = copy.deepcopy(full)
            alpha = float(rank * 2)
            targets = ("qkv", "output")
            summary = inject_lora(lora, rank=rank, alpha=alpha, target_names=targets)
            if state_digest(_base_state(lora, base_names, summary.replaced_modules)) != base_digest:
                raise ValueError("LoRA injection changed base weights")
            full_before = evaluate_held_out(
                full, snapshot, split="validation", max_tokens=max_eval_tokens
            )
            lora_before = evaluate_held_out(
                lora, snapshot, split="validation", max_tokens=max_eval_tokens
            )
            for evaluation in (full_before, lora_before):
                if (
                    evaluation.data_fingerprint != source.data_fingerprint
                    or evaluation.tokenizer_fingerprint != source.tokenizer_fingerprint
                ):
                    raise ValueError("held-out dataset identity changed before training")
            full_logits = _initial_logits_digest(full, source)
            lora_logits = _initial_logits_digest(lora, source)
            matched_predictions = (
                full_logits == lora_logits
                and full_before.total_negative_log_likelihood
                == lora_before.total_negative_log_likelihood
            )
            if not matched_predictions:
                raise ValueError("tuning arms do not start with identical predictions")
            implementation = code_fingerprint()
            revision = git_revision()
            contract = {
                "schema_version": 1,
                "kind": "public-byte-full-versus-lora-v1",
                "experiment_config": config.to_dict(),
                "dataset": {
                    "data_fingerprint": source.data_fingerprint,
                    "tokenizer_fingerprint": source.tokenizer_fingerprint,
                    "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
                    "source": dict(manifest.source),
                    "license": dict(manifest.license),
                    "training_split": "train",
                },
                "adapter": {
                    "rank": rank,
                    "alpha": alpha,
                    "target_names": list(targets),
                    "replaced_modules": list(summary.replaced_modules),
                },
                "base": {"initialization": "random-seeded", "state_digest": base_digest},
                "training": {
                    "batch_policy": "prepared-document-local-interleaved-windows-v1",
                    "rng_policy": "shared-cpu-state-after-adapter-initialization-v1",
                    "optimizer": "AdamW",
                    "weight_decay": 0.01,
                    "gradient_clip_norm": 1.0,
                },
                "evaluation": {
                    "split": "validation",
                    "max_target_tokens": max_eval_tokens,
                    "sampling_policy": "document-id-ordered-prefix-with-reset-context-v1",
                },
                "execution": {
                    "parameter_dtypes": sorted({str(value.dtype) for value in full.parameters()}),
                    "parameter_devices": sorted({str(value.device) for value in full.parameters()}),
                    "deterministic_algorithms": True,
                    "code_fingerprint": implementation,
                },
            }
            encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
            training_rng = capture_rng_state()
            full_run = _train_arm(
                full,
                config,
                source,
                snapshot,
                mode="full",
                validation_before=full_before,
                initial_logits_digest=full_logits,
                base_names=base_names,
                replaced_modules=(),
                max_eval_tokens=max_eval_tokens,
            )
            restore_rng_state(training_rng)
            lora_run = _train_arm(
                lora,
                config,
                source,
                snapshot,
                mode="lora",
                validation_before=lora_before,
                initial_logits_digest=lora_logits,
                base_names=base_names,
                replaced_modules=summary.replaced_modules,
                max_eval_tokens=max_eval_tokens,
            )
            paired_batches = all(
                (left.batch_id, left.windows, left.batch_tokens_digest, left.target_tokens)
                == (right.batch_id, right.windows, right.batch_tokens_digest, right.target_tokens)
                for left, right in zip(full_run.trace, lora_run.trace, strict=True)
            )
            paired_rng = (
                all(
                    left.cpu_rng_before_forward_digest == right.cpu_rng_before_forward_digest
                    for left, right in zip(full_run.trace, lora_run.trace, strict=True)
                )
                and full_run.final_training_rng_digest == lora_run.final_training_rng_digest
            )
            if (
                not paired_batches
                or not paired_rng
                or full_run.initial_loss != lora_run.initial_loss
            ):
                raise ValueError("tuning arms did not replay paired batches and CPU dropout state")
            comparison = PublicTuningComparison(
                schema_version=1,
                config_fingerprint=config.fingerprint(),
                data_fingerprint=source.data_fingerprint,
                tokenizer_fingerprint=source.tokenizer_fingerprint,
                code_fingerprint=implementation,
                code_revision=revision,
                base_state_digest=base_digest,
                comparison_fingerprint=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                comparison_contract=contract,
                device="cpu",
                python_version=platform.python_version(),
                torch_version=torch.__version__,
                matched_initial_predictions=matched_predictions,
                paired_batches=paired_batches,
                paired_cpu_rng_sequence=paired_rng,
                full=full_run,
                lora=lora_run,
                limitations=(
                    "Both arms start from a randomly initialized tiny model, not pretrained weights.",
                    "Training uses train documents only; validation is an ordered bounded prefix.",
                    "Validation is not a test-set result or evidence of downstream model quality.",
                    "Training windows can repeat or overlap; target tokens count compute, not unique data.",
                    "Elapsed time includes final validation and is a local CPU observation, not a benchmark.",
                    "No GPU, distributed-scale, production, or comparative memory claim is made.",
                ),
            )
            publish_json(output, comparison.to_dict())
            return comparison
    finally:
        try:
            restore_rng_state(caller_rng)
        finally:
            torch.use_deterministic_algorithms(caller_deterministic, warn_only=caller_warn_only)
