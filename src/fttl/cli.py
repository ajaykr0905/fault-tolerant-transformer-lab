from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.train import FAILURE_POINTS, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic CPU transformer smoke experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Validated DatasetManifestV1. Omit to retain the synthetic smoke path.",
    )
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help="Stop after this committed step to create a deterministic resume boundary.",
    )
    parser.add_argument("--failure-point", choices=FAILURE_POINTS)
    parser.add_argument("--failure-step", type=int)
    args = parser.parse_args()

    config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
    _, result = run_training(
        config,
        args.output,
        resume_from=args.resume,
        stop_after_step=args.stop_after_step,
        dataset_manifest=args.dataset_manifest,
        failure_point=args.failure_point,
        failure_step=args.failure_step,
    )
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
