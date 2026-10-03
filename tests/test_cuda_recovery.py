"""CUDA-only evidence tests; CPU contract tests never manufacture GPU reports."""

import json
from dataclasses import replace

import pytest
import torch
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl import cuda_recovery


@pytest.mark.parametrize("cut", [0, -1, 4, 5, True, 1.5])
def test_invalid_interruption_boundary_fails_before_gpu_or_manifest_access(
    tmp_path, monkeypatch, cut
):
    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", lambda: pytest.fail("GPU queried"))
    with pytest.raises(ValueError, match="interruption_step"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=tmp_path / "missing",
            interruption_step=cut,
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("budget", [0, -1, True, 0.5])
def test_invalid_evaluation_budget_fails_before_gpu_access(tmp_path, monkeypatch, budget):
    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", lambda: pytest.fail("GPU queried"))
    with pytest.raises(ValueError, match="max_eval_tokens"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=tmp_path / "missing",
            max_eval_tokens=budget,
        )
    assert not (tmp_path / "run").exists()


def test_dropout_is_required_to_exercise_selected_cuda_rng(tmp_path, monkeypatch):
    config = real_data_config()
    config = replace(config, model=replace(config.model, dropout=0.0))
    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", lambda: pytest.fail("GPU queried"))
    with pytest.raises(ValueError, match="dropout"):
        cuda_recovery.run_cuda_recovery(
            config, tmp_path / "run", dataset_manifest=tmp_path / "missing"
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("nonempty", [False, True])
def test_existing_output_is_never_reused_or_overwritten(tmp_path, monkeypatch, nonempty):
    output = tmp_path / "run"
    output.mkdir()
    if nonempty:
        (output / "keep.txt").write_text("user content", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", lambda: pytest.fail("GPU queried"))
    with pytest.raises(ValueError, match="fresh output"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(), output, dataset_manifest=tmp_path / "missing"
        )
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_unavailable_cuda_cannot_create_an_output_or_report(tmp_path, monkeypatch):
    def unavailable():
        raise RuntimeError("CUDA unavailable for this runtime")

    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", unavailable)
    with pytest.raises(RuntimeError, match="unavailable"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=tmp_path / "missing",
        )
    assert not (tmp_path / "run").exists()


def test_cpu_device_cannot_be_substituted_for_cuda_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(cuda_recovery, "prepare_cuda_execution", lambda: (torch.device("cpu"), {}))
    with pytest.raises(RuntimeError, match="actual execution"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=tmp_path / "missing",
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
def test_real_cuda_reconstruction_preserves_dropout_and_full_training_state(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = prepared_fixture(tmp_path / "dataset")
    report = cuda_recovery.run_cuda_recovery(
        real_data_config(),
        tmp_path / "run",
        dataset_manifest=manifest,
        max_eval_tokens=13,
    )
    assert report["exact_equality"] is True
    assert set(report["equality"]) == {
        "model",
        "optimizer",
        "cpu_rng",
        "cuda_rng",
        "logits",
        "steps",
        "tokens_seen",
        "losses",
        "batch_ids",
        "sample_ids",
        "cursor",
    }
    assert all(report["equality"].values())
    assert report["steps"] == 4 and report["tokens_seen"] == 64
    assert report["selected_generation"] == 2
    assert report["final_generation"] == 4
    assert report["memory"]["experiment_peak_allocated_bytes"] > 0
    assert report["evaluation_device"] == "cpu"
    assert report["evaluation"]["evaluated_target_tokens"] == 13
    serialized = (tmp_path / "run" / "cuda-recovery-report.json").read_text(encoding="utf-8")
    assert json.loads(serialized) == report
    assert str(tmp_path) not in serialized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
def test_real_cuda_wrong_but_valid_rng_fails_without_success_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = prepared_fixture(tmp_path / "dataset")
    original_restore = cuda_recovery.restore_cuda_rng_state

    def substitute_gpu_rng(state, device):
        original_restore(state, device)
        torch.rand(1024, device=device)

    monkeypatch.setattr(cuda_recovery, "restore_cuda_rng_state", substitute_gpu_rng)
    with pytest.raises(AssertionError, match="reconstruction differed"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=manifest,
        )
    assert not (tmp_path / "run" / "cuda-recovery-report.json").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires real CUDA hardware")
@pytest.mark.parametrize("corruption", ["loss", "gradient", "model", "optimizer", "logits"])
def test_real_cuda_numerical_failure_cannot_publish_success(tmp_path, monkeypatch, corruption):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest = prepared_fixture(tmp_path / "dataset")
    original_forward = cuda_recovery.TinyTransformer.forward
    original_step = torch.optim.AdamW.step

    def corrupt_forward(model, tokens, targets=None):
        logits, loss = original_forward(model, tokens, targets)
        if corruption == "loss" and loss is not None:
            loss = loss * float("nan")
        if corruption == "gradient" and loss is not None:
            loss.register_hook(lambda gradient: gradient * float("inf"))
        if corruption == "logits" and not model.training:
            logits = logits * float("nan")
        return logits, loss

    def corrupt_update(optimizer, *args, **kwargs):
        result = original_step(optimizer, *args, **kwargs)
        parameter = optimizer.param_groups[0]["params"][0]
        with torch.no_grad():
            if corruption == "model":
                parameter.flatten()[0] = float("nan")
            elif corruption == "optimizer":
                optimizer.state[parameter]["exp_avg"].flatten()[0] = float("inf")
        return result

    monkeypatch.setattr(cuda_recovery.TinyTransformer, "forward", corrupt_forward)
    monkeypatch.setattr(torch.optim.AdamW, "step", corrupt_update)
    with pytest.raises((FloatingPointError, RuntimeError), match="non-finite"):
        cuda_recovery.run_cuda_recovery(
            real_data_config(),
            tmp_path / "run",
            dataset_manifest=manifest,
        )
    assert not (tmp_path / "run" / "cuda-recovery-report.json").exists()
