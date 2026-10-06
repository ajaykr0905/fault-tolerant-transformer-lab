"""Probe the actual CPU and local-filesystem primitives required by the lab."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path

import torch

from fttl.checkpoint import load_checkpoint, save_checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.reports import publish_json
from fttl.state import capture_rng_state, restore_rng_state, state_trees_equal


def preflight(output: Path) -> dict[str, object]:
    """Publish measured checks, including failures, without naming private paths.

    Probes use a private temporary below the report's parent, then remove it.
    This uses CPU SGD with momentum, not the training pipeline's AdamW optimizer.
    It proves current local operations, not power-loss or distributed durability.
    """
    if output.exists() or output.is_symlink():
        raise ValueError("preflight requires a fresh output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    checks: dict[str, bool] = {"posix_process_signals": os.name == "posix"}
    failures = {}
    caller_rng = capture_rng_state()
    try:
        with tempfile.TemporaryDirectory(prefix=".fttl-preflight-", dir=output.parent) as directory:
            scratch = Path(directory)
            try:
                pending = scratch / "pending"
                with pending.open("wb") as handle:
                    handle.write(b"public-safe-preflight")
                    handle.flush()
                    os.fsync(handle.fileno())
                committed = scratch / "committed"
                os.replace(pending, committed)
                checks["file_flush_and_atomic_rename"] = (
                    committed.read_bytes() == b"public-safe-preflight"
                )
                publish_json(scratch / "json.json", {"probe": "complete"})
                checks["atomic_json_hardlink_publication"] = json.loads(
                    (scratch / "json.json").read_text()
                ) == {"probe": "complete"}
            except (OSError, ValueError) as error:
                checks["local_filesystem_primitives"] = False
                failures["local_filesystem_primitives"] = type(error).__name__
            try:
                with torch.device("cpu"):
                    config = ExperimentConfig(
                        model=ModelConfig(
                            vocab_size=24,
                            block_size=4,
                            d_model=8,
                            n_heads=2,
                            n_layers=1,
                            dropout=0.0,
                        )
                    )
                    model = TinyTransformer(config.model).to(dtype=torch.float32)
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=config.learning_rate,
                        momentum=0.9,
                        foreach=False,
                        fused=False,
                    )
                    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cpu")
                    logits, loss = model(tokens, tokens)
                    loss.backward()
                    optimizer.step()
                    checks["cpu_fp32_forward_backward_optimizer"] = bool(
                        torch.isfinite(loss)
                    ) and bool(torch.isfinite(logits).all())
                    store = scratch / "checkpoint"
                    save_checkpoint(
                        store,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=1,
                        tokens_seen=4,
                        losses=[float(loss.detach())],
                    )
                    restored = TinyTransformer(config.model).to(dtype=torch.float32)
                    restored_optimizer = torch.optim.SGD(
                        restored.parameters(),
                        lr=config.learning_rate,
                        momentum=0.9,
                        foreach=False,
                        fused=False,
                    )
                    payload = load_checkpoint(
                        store,
                        model=restored,
                        optimizer=restored_optimizer,
                        expected_config=config,
                        restore_rng=False,
                    )
                    momentum_states = (
                        optimizer.state_dict()["state"],
                        restored_optimizer.state_dict()["state"],
                    )
                    parameter_count = len(list(model.parameters()))
                    checks["nonempty_sgd_momentum_checkpoint_state"] = all(
                        bool(states)
                        and len(states) == parameter_count
                        and all(
                            set(value) == {"momentum_buffer"}
                            and isinstance(value["momentum_buffer"], torch.Tensor)
                            and value["momentum_buffer"].device.type == "cpu"
                            and value["momentum_buffer"].dtype == torch.float32
                            and bool(torch.isfinite(value["momentum_buffer"]).all())
                            for value in states.values()
                        )
                        for states in momentum_states
                    )
                    checks["checkpoint_roundtrip_exact"] = (
                        checks["nonempty_sgd_momentum_checkpoint_state"]
                        and payload["step"] == 1
                        and state_trees_equal(model.state_dict(), restored.state_dict())
                        and state_trees_equal(
                            optimizer.state_dict(), restored_optimizer.state_dict()
                        )
                    )
            except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
                checks["cpu_training_checkpoint_primitives"] = False
                failures["cpu_training_checkpoint_primitives"] = type(error).__name__
    finally:
        with torch.device("cpu"):
            restore_rng_state(caller_rng)
    report = {
        "schema_version": 1,
        "ready": all(checks.values()),
        "checks": checks,
        "failures": failures,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "platform": platform.system(),
            "device": "cpu",
            "dtype": "torch.float32",
            "optimizer": "torch.optim.SGD",
        },
        "optimizer_policy": {"momentum": 0.9, "foreach": False, "fused": False},
        "limitations": [
            "Local scratch operations do not prove power-loss durability, network filesystem semantics or scale.",
            "No GPU discovery, GPU performance, network binding or production readiness is tested.",
            "The CPU SGD momentum probe does not measure AdamW training or recovery readiness.",
            "POSIX support is an environment check; this preflight does not send a real kill signal.",
        ],
    }
    publish_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = preflight(args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
