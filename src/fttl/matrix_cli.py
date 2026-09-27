from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.matrix import verify_recovery_matrix


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete deterministic CPU real-data recovery matrix."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
    matrix = verify_recovery_matrix(
        config,
        args.dataset_manifest,
        args.output,
        restarts=args.restarts,
    )
    print(json.dumps(asdict(matrix), sort_keys=True))


if __name__ == "__main__":
    main()
