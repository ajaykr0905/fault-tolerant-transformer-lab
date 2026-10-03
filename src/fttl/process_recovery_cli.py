from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.process_recovery import verify_process_recovery
from fttl.train import FAILURE_POINTS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify exact CPU recovery after parent-controlled SIGKILL of a spawned worker."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest", type=Path, help="Omit to use the labeled synthetic smoke workload."
    )
    parser.add_argument("--failure-point", choices=FAILURE_POINTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="Deadline for each spawned worker, including startup, IPC and completion.",
    )
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
    report = verify_process_recovery(
        config,
        args.output,
        failure_point=args.failure_point,
        dataset_manifest=args.dataset_manifest,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
