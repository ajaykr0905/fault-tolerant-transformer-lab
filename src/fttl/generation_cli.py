"""Generate bounded byte-token evidence from a trusted recovered checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.generation import MAX_PROMPT_TOKENS, generate_tokens
from fttl.inference import load_inference_checkpoint
from fttl.reports import publish_json


def generate_checkpoint(
    config: ExperimentConfig,
    checkpoint: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    prompt: str,
    max_new_tokens: int = 32,
    method: str = "greedy",
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int = 0,
    stop_token_id: int | None = None,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise ValueError("generation requires a fresh output file")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be UTF-8 text")
    encoded = prompt.encode("utf-8", errors="strict")
    if not 1 <= len(encoded) <= MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt must encode to 1..{MAX_PROMPT_TOKENS} bytes")
    model, receipt = load_inference_checkpoint(config, checkpoint, dataset_manifest)
    generated = generate_tokens(
        model,
        encoded,
        max_new_tokens=max_new_tokens,
        method=method,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        stop_token_id=stop_token_id,
    )
    # EOD=256 is not a byte. Never turn arbitrary model output into invented text.
    ids = generated.generated_token_ids
    raw = bytes(token for token in ids if token != 256)
    try:
        text = raw.decode("utf-8", errors="strict") if 256 not in ids else None
    except UnicodeDecodeError:
        text = None
    result = {
        "schema_version": 1,
        "inference": receipt.to_dict(),
        "generation": generated.to_dict(),
        "generated_utf8": text,
        "generated_byte_hex_without_eod": raw.hex(),
        "limitations": [
            "Byte IDs are authoritative; UTF-8 is null for invalid bytes or any EOD token.",
            "The tiny randomly initialized model is a recovery prototype, not a language assistant.",
        ],
    }
    publish_json(output, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--method", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-token-id", type=int)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
        result = generate_checkpoint(
            config,
            args.checkpoint,
            args.dataset_manifest,
            args.output,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            method=args.method,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            stop_token_id=args.stop_token_id,
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
