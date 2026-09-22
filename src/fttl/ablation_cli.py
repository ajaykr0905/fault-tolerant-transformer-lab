from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from fttl.ablation import compare_tuning
from fttl.config import ExperimentConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full tuning and LoRA on one controlled CPU workload")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tuning-comparison/result.json"))
    parser.add_argument("--rank", type=int, default=4)
    args = parser.parse_args()
    config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
    result = compare_tuning(config, args.output, rank=args.rank)
    print(json.dumps(asdict(result), sort_keys=True))


if __name__ == "__main__":
    main()
