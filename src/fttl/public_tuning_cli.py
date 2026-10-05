"""Run one bounded paired public-document CPU tuning comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.public_tuning import compare_public_tuning


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Byte-model experiment JSON")
    parser.add_argument(
        "--dataset-manifest", type=Path, required=True, help="Verified prepared dataset"
    )
    parser.add_argument("--output", type=Path, required=True, help="Fresh comparison JSON path")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank; alpha is twice this value")
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=4096,
        help="Validation next-byte target budget per arm",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(arguments.config.read_text(encoding="utf-8"))
        result = compare_public_tuning(
            config,
            arguments.dataset_manifest,
            arguments.output,
            rank=arguments.rank,
            max_eval_tokens=arguments.max_eval_tokens,
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result.to_dict(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
