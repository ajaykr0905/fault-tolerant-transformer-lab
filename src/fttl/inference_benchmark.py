"""Measure bounded end-to-end CPU generation; not a GPU or production SLA."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
from pathlib import Path
from time import perf_counter

import torch

from fttl.config import ExperimentConfig
from fttl.generation import generate_tokens
from fttl.inference import load_inference_checkpoint
from fttl.reports import publish_json


def benchmark_inference(
    config: ExperimentConfig,
    checkpoint: Path,
    dataset_manifest: Path,
    output: Path,
    *,
    prompt: str = "Hello",
    iterations: int = 5,
    warmups: int = 1,
    max_new_tokens: int = 8,
    seed: int = 23,
) -> dict[str, object]:
    if output.exists() or output.is_symlink():
        raise ValueError("benchmark requires a fresh output file")
    for name, value, low, high in (
        ("iterations", iterations, 1, 20),
        ("warmups", warmups, 0, 5),
        ("max_new_tokens", max_new_tokens, 1, 32),
        ("seed", seed, 0, 2**63 - 1),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode("utf-8")) <= 4096:
        raise ValueError("prompt must encode to 1..4096 UTF-8 bytes")
    model, receipt = load_inference_checkpoint(config, checkpoint, dataset_manifest)
    controls = dict(max_new_tokens=max_new_tokens, method="sample", seed=seed, top_k=8)
    encoded = prompt.encode("utf-8")
    for _ in range(warmups):
        generate_tokens(model, encoded, **controls)
    samples = []
    for _ in range(iterations):
        started = perf_counter()
        result = generate_tokens(model, encoded, **controls)
        elapsed = perf_counter() - started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("benchmark clock produced invalid elapsed time")
        samples.append(
            {
                "elapsed_seconds": elapsed,
                "generated_tokens": result.generated_count,
                "generated_token_ids": list(result.generated_token_ids),
            }
        )
    latencies = sorted(sample["elapsed_seconds"] for sample in samples)
    total_seconds = math.fsum(latencies)
    total_tokens = sum(sample["generated_tokens"] for sample in samples)
    report = {
        "schema_version": 1,
        "inference": receipt.to_dict(),
        "prompt_token_ids": list(encoded),
        "iterations": iterations,
        "warmups_excluded": warmups,
        "controls": controls,
        "samples": samples,
        "total_generated_tokens": total_tokens,
        "generation_seconds": total_seconds,
        "median_seconds": statistics.median(latencies),
        "p95_nearest_rank_seconds": latencies[math.ceil(iterations * 0.95) - 1],
        "tokens_per_second": total_tokens / total_seconds if total_seconds else None,
        "replayed_tokens_equal": all(
            sample["generated_token_ids"] == samples[0]["generated_token_ids"] for sample in samples
        ),
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "limitations": [
            "Timing includes complete generate_tokens call, state digests and RNG preservation, not checkpoint loading.",
            "Warmups are excluded; tiny sample p95 uses nearest rank, not population-tail confidence.",
            "CPU wall-clock measurements are environment-dependent, not deterministic replay evidence.",
            "No network, concurrent traffic, KV cache, GPU or production latency claim is made.",
        ],
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    publish_json(output, report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
        result = benchmark_inference(
            config,
            args.checkpoint,
            args.dataset_manifest,
            args.output,
            prompt=args.prompt,
            iterations=args.iterations,
            warmups=args.warmups,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
