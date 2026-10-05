"""Verify bounded generation parity between control and recovered checkpoints."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.generation import generate_tokens
from fttl.inference import load_inference_checkpoint
from fttl.reports import publish_json


def verify_inference_parity(
    config: ExperimentConfig,
    control_checkpoint: Path,
    recovered_checkpoint: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    prompts: Sequence[str] = ("Hello",),
    max_new_tokens: int = 8,
    seed: int = 23,
) -> dict[str, object]:
    """Publish actual success or failure, never overwrite an earlier measurement.

    Exact equality is scoped to these prompt bytes, controls and CPU environment.
    Equal emitted tokens alone are insufficient: model and training step identities
    must also match. This is not exhaustive input coverage or a GPU tolerance test.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("parity verification requires a fresh output file")
    if (
        isinstance(prompts, (str, bytes))
        or not isinstance(prompts, Sequence)
        or not 1 <= len(prompts) <= 8
    ):
        raise ValueError("prompts must be a sequence containing 1..8 texts")
    encoded = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError("prompts must be UTF-8 text")
        raw = prompt.encode("utf-8", errors="strict")
        if not 1 <= len(raw) <= 4096:
            raise ValueError("prompts must encode to 1..4096 bytes")
        encoded.append(raw)
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 64:
        raise ValueError("max_new_tokens must be an integer in 1..64")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("seed must be an integer in 0..2**63-1")
    control, control_receipt = load_inference_checkpoint(
        config, control_checkpoint, dataset_manifest
    )
    recovered, recovered_receipt = load_inference_checkpoint(
        config, recovered_checkpoint, dataset_manifest
    )
    cases = []
    for prompt in encoded:
        for method in ("greedy", "sample"):
            arguments = dict(max_new_tokens=max_new_tokens, method=method, seed=seed, top_k=8)
            left = generate_tokens(control, prompt, **arguments)
            right = generate_tokens(recovered, prompt, **arguments)
            cases.append(
                {
                    "prompt_token_ids": list(prompt),
                    "method": method,
                    "control_token_ids": list(left.generated_token_ids),
                    "recovered_token_ids": list(right.generated_token_ids),
                    "exact_tokens": left.generated_token_ids == right.generated_token_ids,
                }
            )
    equality = {
        "model_state": control_receipt.model_state_digest == recovered_receipt.model_state_digest,
        "training_steps": control_receipt.completed_training_steps
        == recovered_receipt.completed_training_steps,
        "training_contract": control_receipt.training_run_contract_fingerprint
        == recovered_receipt.training_run_contract_fingerprint,
        "generated_tokens": all(case["exact_tokens"] for case in cases),
    }
    report = {
        "schema_version": 1,
        "exact_equality": all(equality.values()),
        "equality": equality,
        "control": control_receipt.to_dict(),
        "recovered": recovered_receipt.to_dict(),
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "sampling_top_k": 8,
        "cases": cases,
        "limitations": [
            "Only declared prompts and bounded greedy/top-k sampled sequences were tested.",
            "A parity result does not establish language quality, cross-platform tolerance or GPU serving.",
            "Generation timings and latency guarantees are not measured here.",
        ],
    }
    # Normalize tuple-valued receipt fields so API, disk and CLI have one JSON form.
    report = json.loads(json.dumps(report, allow_nan=False))
    publish_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--recovered-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
        report = verify_inference_parity(
            config,
            args.control_checkpoint,
            args.recovered_checkpoint,
            args.dataset_manifest,
            args.output,
            prompts=args.prompt or ("Hello",),
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["exact_equality"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
