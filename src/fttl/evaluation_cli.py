from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

from fttl.checkpoint import load_checkpoint
from fttl.config import ExperimentConfig
from fttl.dataset import load_dataset_manifest
from fttl.evaluation import evaluate_held_out
from fttl.model import TinyTransformer
from fttl.state import code_fingerprint, git_revision


def evaluate_checkpoint(
    config: ExperimentConfig,
    checkpoint: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    split: str = "validation",
    max_tokens: int = 4096,
) -> dict[str, object]:
    """Write one fresh report from a verified, dataset-bound local checkpoint."""
    if output.exists():
        raise ValueError("evaluation requires a fresh output file")
    manifest = load_dataset_manifest(dataset_manifest)
    with torch.random.fork_rng(devices=[]):
        model = TinyTransformer(config.model)
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
        evaluation = evaluate_held_out(model, dataset_manifest, split=split, max_tokens=max_tokens)
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
        "max_tokens": max_tokens,
        "limitations": [
            "Context resets at each document-local window.",
            "A token budget evaluates a document-ID-ordered prefix, not a random sample.",
            "Loss is next-byte NLL including EOD, not word-level perplexity.",
            "A smoke result does not establish pretrained-model quality or GPU performance.",
        ],
    }
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return result


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
