from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.recovery import verify_recovery
from fttl.train import FAILURE_POINTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prove exact CPU equivalence between uninterrupted and failure-recovered training."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--failure-point", choices=FAILURE_POINTS, required=True)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
    report = verify_recovery(
        config,
        args.dataset_manifest,
        args.output,
        failure_point=args.failure_point,
        restarts=args.restarts,
    )
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
