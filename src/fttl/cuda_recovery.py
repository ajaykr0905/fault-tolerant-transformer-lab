"""Single-device checkpoint reconstruction with observed CUDA execution evidence."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from fttl.checkpoint import load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig
from fttl.cuda_runtime import (
    capture_cuda_rng_state,
    prepare_cuda_execution,
    restore_cuda_rng_state,
    synchronize_cuda,
    validate_cuda_rng_state,
)
from fttl.data import PreparedDatasetBatchSource, TrainingCursorV1
from fttl.evaluation import evaluate_held_out
from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state
from fttl.state import code_fingerprint, git_revision, state_digest, state_trees_equal
from fttl.train import seed_everything


def _cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return value


def _new_training_objects(config: ExperimentConfig, device: torch.device):
    model = TinyTransformer(config.model).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, foreach=False, fused=False
    )
    return model, optimizer


def _advance(model, optimizer, config, source, device, store, contract, end_step, payload=None):
    step = 0 if payload is None else payload["step"]
    cursor = (
        source.initial_cursor()
        if payload is None
        else TrainingCursorV1.from_dict(payload["cursor"])
    )
    losses = [] if payload is None else list(payload["losses"])
    batch_ids = [] if payload is None else list(payload["batch_ids"])
    sample_ids = [] if payload is None else [tuple(batch) for batch in payload["sample_ids"]]
    tokens_seen = 0 if payload is None else payload["tokens_seen"]
    if (
        cursor != source.cursor_at(step)
        or not len(losses) == len(batch_ids) == len(sample_ids) == step
    ):
        raise ValueError("checkpoint cursor and histories do not follow its completed step")
    expected_tokens = step * config.batch_size * config.model.block_size
    if tokens_seen != expected_tokens:
        raise ValueError("checkpoint token count does not match completed steps")
    require_finite_state(model.state_dict(), "model state")
    require_finite_state(optimizer.state_dict(), "optimizer state")
    model.train()
    generation = 0
    logits = None
    for step_index in range(step, end_step):
        batch = source.batch(cursor)
        inputs, targets = batch.inputs.to(device), batch.targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        require_finite_state(loss, "training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
        require_finite_state(model.state_dict(), "model state")
        require_finite_state(optimizer.state_dict(), "optimizer state")
        step = step_index + 1
        if step == end_step:
            model.eval()
            with torch.no_grad():
                logits, _ = model(source.batch(source.initial_cursor()).inputs.to(device))
            require_finite_state(logits, "verification logits")
            model.train()
        losses.append(float(loss.detach()))
        batch_ids.append(batch.batch_id)
        sample_ids.append(batch.sample_ids)
        tokens_seen += inputs.numel()
        cursor = batch.next_cursor
        if step % config.checkpoint_every == 0 or step == end_step:
            saved = save_checkpoint(
                store,
                model=model,
                optimizer=optimizer,
                config=config,
                step=step,
                tokens_seen=tokens_seen,
                losses=losses,
                cursor=cursor,
                batch_ids=batch_ids,
                sample_ids=sample_ids,
                data_fingerprint=source.data_fingerprint,
                tokenizer_fingerprint=source.tokenizer_fingerprint,
                run_contract_fingerprint=contract,
                rng_state=capture_cuda_rng_state(device),
            )
            generation = saved.generation
    assert logits is not None
    return {
        "model": _cpu_snapshot(model.state_dict()),
        "optimizer": _cpu_snapshot(optimizer.state_dict()),
        "rng": capture_cuda_rng_state(device),
        "logits": _cpu_snapshot(logits),
        "steps": step,
        "tokens_seen": tokens_seen,
        "losses": losses,
        "batch_ids": batch_ids,
        "sample_ids": sample_ids,
        "cursor": cursor.to_dict(),
        "generation": generation,
    }


def run_cuda_recovery(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    dataset_manifest: Path,
    interruption_step: int = 2,
    max_eval_tokens: int = 4096,
) -> dict[str, object]:
    """Verify same-runtime, same-GPU reconstruction, not process-death recovery."""
    if (
        isinstance(interruption_step, bool)
        or not isinstance(interruption_step, int)
        or not 1 <= interruption_step < config.steps
    ):
        raise ValueError("interruption_step must be an integer inside the training range")
    if (
        isinstance(max_eval_tokens, bool)
        or not isinstance(max_eval_tokens, int)
        or max_eval_tokens < 1
    ):
        raise ValueError("max_eval_tokens must be a positive integer")
    if config.model.dropout <= 0:
        raise ValueError("CUDA recovery verification requires dropout greater than zero")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("CUDA recovery requires a fresh output path")
    device, execution = prepare_cuda_execution()
    if device != torch.device("cuda:0"):
        raise RuntimeError("CUDA recovery requires actual execution on cuda:0")
    source = PreparedDatasetBatchSource.from_manifest(config, dataset_manifest)
    implementation = code_fingerprint()
    contract = state_digest(
        {
            "schema": "CudaRecoveryContractV1",
            "config_fingerprint": config.fingerprint(),
            "data_fingerprint": source.data_fingerprint,
            "tokenizer_fingerprint": source.tokenizer_fingerprint,
            "code_fingerprint": implementation,
            "execution": execution,
            "optimizer": "AdamW:foreach=False:fused=False",
            "dtype": "float32",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(device)
    synchronize_cuda(device)
    started = time.perf_counter()
    seed_everything(config.seed)
    model, optimizer = _new_training_objects(config, device)
    control = _advance(
        model, optimizer, config, source, device, output_dir / "control", contract, config.steps
    )
    synchronize_cuda(device)
    control_seconds = time.perf_counter() - started
    del model, optimizer
    synchronize_cuda(device)
    started = time.perf_counter()
    seed_everything(config.seed)
    model, optimizer = _new_training_objects(config, device)
    prefix = _advance(
        model,
        optimizer,
        config,
        source,
        device,
        output_dir / "recovered",
        contract,
        interruption_step,
    )
    del model, optimizer
    synchronize_cuda(device)
    load_started = time.perf_counter()
    model, optimizer = _new_training_objects(config, device)
    payload = load_checkpoint(
        output_dir / "recovered",
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
    load_seconds = time.perf_counter() - load_started
    selected_generation = payload["selected_generation"]
    if payload["step"] != interruption_step or selected_generation != prefix["generation"]:
        raise AssertionError("checkpoint selection did not return the interrupted boundary")
    recovered = _advance(
        model,
        optimizer,
        config,
        source,
        device,
        output_dir / "recovered",
        contract,
        config.steps,
        payload,
    )
    synchronize_cuda(device)
    recovered_seconds = time.perf_counter() - started
    equality = {
        key: state_trees_equal(control[key], recovered[key])
        for key in (
            "model",
            "optimizer",
            "logits",
            "steps",
            "tokens_seen",
            "losses",
            "batch_ids",
            "sample_ids",
            "cursor",
        )
    }
    equality["cpu_rng"] = state_trees_equal(
        {key: value for key, value in control["rng"].items() if key != "torch_cuda"},
        {key: value for key, value in recovered["rng"].items() if key != "torch_cuda"},
    )
    equality["cuda_rng"] = state_trees_equal(
        control["rng"]["torch_cuda"], recovered["rng"]["torch_cuda"]
    )
    if not all(equality.values()):
        raise AssertionError(f"CUDA checkpoint reconstruction differed: {equality}")
    memory = {
        "experiment_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "experiment_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    del model, optimizer
    cpu_model = TinyTransformer(config.model).to(dtype=torch.float32)
    cpu_model.load_state_dict(recovered["model"])
    evaluation = evaluate_held_out(cpu_model, dataset_manifest, max_tokens=max_eval_tokens)
    report: dict[str, object] = {
        "schema": "CudaRecoveryReportV1",
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "data_fingerprint": source.data_fingerprint,
        "tokenizer_fingerprint": source.tokenizer_fingerprint,
        "run_contract_fingerprint": contract,
        "code_fingerprint": implementation,
        "code_revision": git_revision(),
        "execution": execution,
        "memory": memory,
        "interruption_step": interruption_step,
        "selected_generation": selected_generation,
        "final_generation": recovered["generation"],
        "equality": equality,
        "exact_equality": True,
        "steps": recovered["steps"],
        "tokens_seen": recovered["tokens_seen"],
        "losses": recovered["losses"],
        "batch_ids": recovered["batch_ids"],
        "sample_ids": [list(batch) for batch in recovered["sample_ids"]],
        "final_cursor": recovered["cursor"],
        "digests": {
            key: state_digest(recovered[key]) for key in ("model", "optimizer", "rng", "logits")
        },
        "control_seconds": control_seconds,
        "recovered_seconds": recovered_seconds,
        "checkpoint_load_seconds": load_seconds,
        "evaluation_device": "cpu",
        "evaluation": evaluation.to_dict(),
        "limitations": [
            "Graceful same-runtime reconstruction on one CUDA device, not process death or SIGKILL.",
            "Exact equality is confined to this GPU and execution contract, not cross-device parity.",
            "FP32 eager execution; no mixed precision, distributed training or serving evidence.",
            "Experiment-wide GPU peaks include checkpoint rollback and serialization staging.",
            "Held-out byte metrics use a bounded deterministic prefix on CPU, not model-quality evidence.",
            "Checkpoints are trusted local artifacts; SHA-256 is not authentication.",
        ],
    }
    temporary = output_dir / ".cuda-recovery-report.json.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_dir / "cuda-recovery-report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interruption-step", type=int, default=2)
    parser.add_argument("--max-eval-tokens", type=int, default=4096)
    args = parser.parse_args()
    report = run_cuda_recovery(
        ExperimentConfig.from_json(args.config.read_text(encoding="utf-8")),
        args.output,
        dataset_manifest=args.dataset_manifest,
        interruption_step=args.interruption_step,
        max_eval_tokens=args.max_eval_tokens,
    )
    print(json.dumps({"exact_equality": report["exact_equality"], "steps": report["steps"]}))


if __name__ == "__main__":
    main()
