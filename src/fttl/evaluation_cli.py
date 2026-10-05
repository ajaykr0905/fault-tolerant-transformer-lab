from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fttl.checkpoint import CheckpointIntegrityError, load_checkpoint
from fttl.config import ExperimentConfig
from fttl.dataset import load_dataset_snapshot
from fttl.evaluation import evaluate_held_out
from fttl.model import TinyTransformer
from fttl.reports import publish_json
from fttl.state import capture_rng_state, code_fingerprint, git_revision, restore_rng_state


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    split: str = "validation",
    max_tokens: int = 4096,
) -> dict[str, object]:
    """Evaluate one captured dataset on CPU, preserving caller CPU RNGs.

    Checkpoint binding and held-out evaluation use the same verified snapshot,
    independent of subsequent changes to the dataset path. Only FP32 checkpoint
    reconstruction is supported; other source dtypes are never silently converted.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("evaluation requires a fresh output file")
    caller_rng = capture_rng_state()
    try:
        with torch.device("cpu"):
            snapshot = load_dataset_snapshot(dataset_manifest)
            manifest = snapshot.manifest
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
            for name, tensor in model.state_dict().items():
                if payload["model"][name].dtype != tensor.dtype:
                    raise CheckpointIntegrityError("evaluation requires an FP32 checkpoint")
            evaluation = evaluate_held_out(model, snapshot, split=split, max_tokens=max_tokens)
            result = {
                "schema_version": 1,
                "evaluation": evaluation.to_dict(),
                "config": config.to_dict(),
                "config_fingerprint": config.fingerprint(),
                "completed_training_steps": payload["step"],
                "selected_checkpoint_generation": payload["selected_generation"],
                "training_run_contract_fingerprint": payload["run_contract_fingerprint"],
                "evaluation_code_fingerprint": code_fingerprint(),
                "evaluation_git_revision": git_revision(),
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
                "device": "cpu",
                "dtype": "torch.float32",
                "max_tokens": max_tokens,
                "limitations": [
                    "Context resets at each document-local window.",
                    "A token budget evaluates a document-ID-ordered prefix, not a random sample.",
                    "Loss is next-byte NLL including EOD, not word-level perplexity.",
                    "A smoke result does not establish pretrained-model quality or GPU performance.",
                ],
            }
            publish_json(output, result)
            return result
    finally:
        with torch.device("cpu"):
            restore_rng_state(caller_rng)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trusted CPU checkpoint on held-out bytes"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    try:
        config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
        result = evaluate_checkpoint(
            config,
            args.checkpoint,
            args.dataset_manifest,
            args.output,
            split=args.split,
            max_tokens=args.max_tokens,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
