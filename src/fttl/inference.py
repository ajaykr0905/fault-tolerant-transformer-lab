"""Reconstruct a frozen CPU inference model from a trusted dataset-bound checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path

import torch

from fttl.checkpoint import CheckpointIntegrityError, load_checkpoint
from fttl.config import ExperimentConfig
from fttl.dataset import load_dataset_snapshot
from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state
from fttl.state import capture_rng_state, restore_rng_state, state_digest


@dataclass(frozen=True)
class InferenceReceiptV1:
    schema_version: int
    config_fingerprint: str
    data_fingerprint: str
    tokenizer_fingerprint: str
    training_run_contract_fingerprint: str
    selected_checkpoint_generation: int
    completed_training_steps: int
    model_state_digest: str
    device: str
    dtype: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_inference_checkpoint(
    config: ExperimentConfig, checkpoint: Path, dataset_manifest: Path
) -> tuple[TinyTransformer, InferenceReceiptV1]:
    """Load verified byte-model weights without restoring training RNG or writing artifacts.

    Only FP32 CPU reconstruction is supported. The restricted checkpoint loader
    still validates optimizer structure, even though no optimizer is returned.
    Integrity digests are not authentication; use a trusted local store.
    """
    if not isinstance(config, ExperimentConfig):
        raise ValueError("config must be an ExperimentConfig")
    if config.model.vocab_size != 257:
        raise ValueError("byte inference requires model vocab_size=257")
    snapshot = load_dataset_snapshot(dataset_manifest)
    manifest = snapshot.manifest
    original_rng = capture_rng_state()
    try:
        with torch.device("cpu"):
            model = TinyTransformer(config.model).to(dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            payload = load_checkpoint(
                checkpoint,
                model=model,
                optimizer=optimizer,
                expected_config=config,
                expected_data_fingerprint=manifest.fingerprint(),
                expected_tokenizer_fingerprint=manifest.tokenizer_fingerprint,
                restore_rng=False,
            )
            # Reject an implicit dtype conversion instead of reporting the source
            # checkpoint's identity for different reconstructed numerical weights.
            for name, tensor in model.state_dict().items():
                if payload["model"][name].dtype != tensor.dtype:
                    raise CheckpointIntegrityError("inference requires an FP32 checkpoint")
            for name, tensor in chain(model.named_parameters(), model.named_buffers()):
                if tensor.layout != torch.strided or tensor.device.type != "cpu":
                    raise ValueError(f"inference model state.{name} must be dense CPU state")
                require_finite_state(tensor, f"inference model state.{name}")
            for field in ("step", "selected_generation"):
                if type(payload[field]) is not int or payload[field] < 0:
                    raise CheckpointIntegrityError(
                        f"checkpoint {field} must be a non-negative integer"
                    )
            model.eval()
            model.requires_grad_(False)
            receipt = InferenceReceiptV1(
                schema_version=1,
                config_fingerprint=config.fingerprint(),
                data_fingerprint=manifest.fingerprint(),
                tokenizer_fingerprint=manifest.tokenizer_fingerprint,
                training_run_contract_fingerprint=payload["run_contract_fingerprint"],
                selected_checkpoint_generation=payload["selected_generation"],
                completed_training_steps=payload["step"],
                model_state_digest=state_digest(model.state_dict()),
                device="cpu",
                dtype="torch.float32",
                limitations=(
                    "Trusted local checkpoint integrity is not source authentication.",
                    "CPU FP32 reconstruction does not establish language quality or GPU performance.",
                    "Training optimizer and RNG state are not returned or restored for inference.",
                ),
            )
            return model, receipt
    finally:
        restore_rng_state(original_rng)
