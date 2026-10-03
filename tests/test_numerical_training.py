from dataclasses import replace

import pytest
import torch

from fttl.ablation import compare_tuning
from fttl.checkpoint import load_checkpoint
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal
from fttl.train import run_training


def _config(steps=2):
    return ExperimentConfig(
        model=ModelConfig(vocab_size=16, block_size=4, d_model=8, n_heads=2, n_layers=1),
        seed=101,
        steps=steps,
        batch_size=2,
        checkpoint_every=1,
    )


@pytest.mark.parametrize("corruption", ["model", "optimizer"])
def test_nonfinite_update_keeps_last_valid_generation_recoverable(
    corruption, monkeypatch, tmp_path
):
    config = _config()
    control, control_result = run_training(config, tmp_path / "control")
    output = tmp_path / "interrupted"
    original_step = torch.optim.AdamW.step
    calls = 0

    def corrupt_second_step(optimizer, *args, **kwargs):
        nonlocal calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        if calls == 2:
            parameter = optimizer.param_groups[0]["params"][0]
            with torch.no_grad():
                if corruption == "model":
                    parameter.flatten()[0] = float("nan")
                else:
                    optimizer.state[parameter]["exp_avg"].flatten()[0] = float("inf")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", corrupt_second_step)
        with pytest.raises(FloatingPointError, match="non-finite"):
            run_training(config, output)

    checkpoint = output / "checkpoints"
    assert (checkpoint / "LATEST").read_text(encoding="utf-8").strip() == "generation-00000001"
    assert not (checkpoint / "generation-00000002").exists()
    assert not (output / "result.json").exists()
    assert not (output / "manifest.json").exists()
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    payload = load_checkpoint(checkpoint, model=model, optimizer=optimizer, expected_config=config)
    assert payload["step"] == 1
    resumed, resumed_result = run_training(config, output, resume_from=checkpoint)
    assert state_trees_equal(control.state_dict(), resumed.state_dict())
    assert control_result.final_state_digest == resumed_result.final_state_digest


def test_nonfinite_loss_is_rejected_before_any_checkpoint(monkeypatch, tmp_path):
    original_forward = TinyTransformer.forward

    def nonfinite_loss(model, tokens, targets=None):
        logits, loss = original_forward(model, tokens, targets)
        return logits, loss * float("nan") if loss is not None else None

    monkeypatch.setattr(TinyTransformer, "forward", nonfinite_loss)
    with pytest.raises(FloatingPointError, match="loss.*non-finite"):
        run_training(_config(1), tmp_path / "run")
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_nonfinite_gradient_is_rejected_before_optimizer_step(monkeypatch, tmp_path):
    original_forward = TinyTransformer.forward

    def nonfinite_gradient(model, tokens, targets=None):
        logits, loss = original_forward(model, tokens, targets)
        if loss is not None:
            loss.register_hook(lambda gradient: gradient * float("inf"))
        return logits, loss

    monkeypatch.setattr(TinyTransformer, "forward", nonfinite_gradient)
    with pytest.raises(RuntimeError, match="non-finite"):
        run_training(_config(1), tmp_path / "run")
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_nonfinite_final_logits_are_rejected_before_final_checkpoint(monkeypatch, tmp_path):
    original_forward = TinyTransformer.forward

    def nonfinite_verification(model, tokens, targets=None):
        logits, loss = original_forward(model, tokens, targets)
        if not model.training:
            logits = logits * float("nan")
        return logits, loss

    monkeypatch.setattr(TinyTransformer, "forward", nonfinite_verification)
    with pytest.raises(FloatingPointError, match="logits.*non-finite"):
        run_training(_config(1), tmp_path / "run")
    assert not (tmp_path / "run" / "checkpoints").exists()
    assert not (tmp_path / "run" / "result.json").exists()


@pytest.mark.parametrize("arm", ["full", "lora"])
def test_nonfinite_tuning_update_is_rejected_without_a_comparison_report(
    arm, monkeypatch, tmp_path
):
    original_step = torch.optim.AdamW.step
    calls = 0
    corruption_step = 1 if arm == "full" else 3

    def corrupt_arm(optimizer, *args, **kwargs):
        nonlocal calls
        result = original_step(optimizer, *args, **kwargs)
        calls += 1
        if calls == corruption_step:
            parameter = optimizer.param_groups[0]["params"][0]
            optimizer.state[parameter]["exp_avg"].flatten()[0] = float("nan")
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_arm)
    output = tmp_path / "comparison.json"
    with pytest.raises(FloatingPointError, match="non-finite"):
        compare_tuning(_config(), output, rank=2)
    assert not output.exists()


@pytest.mark.parametrize("arm", ["full", "lora"])
def test_tuning_final_verification_rejects_nonfinite_logits(arm, monkeypatch, tmp_path):
    original_forward = TinyTransformer.forward
    verifications = 0
    corruption_verification = 1 if arm == "full" else 2

    def nonfinite_verification(model, tokens, targets=None):
        nonlocal verifications
        logits, loss = original_forward(model, tokens, targets)
        if not model.training:
            verifications += 1
            if verifications == corruption_verification:
                logits = logits * float("inf")
        return logits, loss

    monkeypatch.setattr(TinyTransformer, "forward", nonfinite_verification)
    output = tmp_path / "comparison.json"
    with pytest.raises(FloatingPointError, match="logits.*non-finite"):
        compare_tuning(_config(), output, rank=2)
    assert not output.exists()


@pytest.mark.parametrize("consumer", ["training", "comparison"])
def test_finite_extreme_learning_rate_cannot_publish_nonfinite_final_logits(consumer, tmp_path):
    config = replace(_config(1), learning_rate=1e30)
    output = tmp_path / "run"
    with pytest.raises(FloatingPointError, match="logits.*non-finite"):
        if consumer == "training":
            run_training(config, output)
        else:
            compare_tuning(config, output / "comparison.json", rank=2)
    assert not (output / "checkpoints").exists()
    assert not (output / "result.json").exists()
    assert not (output / "comparison.json").exists()


def test_completed_checkpoint_is_rejected_without_execution_or_artifact_changes(
    monkeypatch, tmp_path
):
    config = _config(1)
    output = tmp_path / "run"
    run_training(config, output)
    artifacts = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    rng_before = capture_rng_state()

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("completed checkpoint must not execute forward or optimizer step")

    monkeypatch.setattr(TinyTransformer, "forward", unexpected_execution)
    monkeypatch.setattr(torch.optim.AdamW, "step", unexpected_execution)
    with pytest.raises(ValueError, match="greater than the checkpoint"):
        run_training(config, output, resume_from=output / "checkpoints")
    assert state_trees_equal(rng_before, capture_rng_state())
    assert artifacts == {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
