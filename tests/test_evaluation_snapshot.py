import random

import numpy as np
import pytest
import torch
from test_dataset_snapshot import alternate_fixture, switch_directory
from test_evaluation_cli import checkpoint_run

from fttl import evaluation_cli
from fttl.checkpoint import CheckpointIntegrityError, save_checkpoint
from fttl.dataset import DatasetValidationError, PreparedDatasetSnapshot
from fttl.evaluation import evaluate_held_out
from fttl.state import capture_rng_state, state_trees_equal


def test_checkpoint_and_evaluation_share_snapshot_despite_dataset_directory_swap(
    tmp_path, monkeypatch
):
    manifest, config, model = checkpoint_run(tmp_path)
    expected = evaluate_held_out(model, manifest, max_tokens=13)
    second = alternate_fixture(tmp_path / "other")
    live = tmp_path / "live"
    live.symlink_to(manifest.parent, target_is_directory=True)
    original_load = evaluation_cli.load_checkpoint

    def swapped_after_checkpoint(*args, **kwargs):
        payload = original_load(*args, **kwargs)
        switch_directory(live, second.parent)
        return payload

    monkeypatch.setattr(evaluation_cli, "load_checkpoint", swapped_after_checkpoint)
    rng = capture_rng_state()
    result = evaluation_cli.evaluate_checkpoint(
        config,
        tmp_path / "training/checkpoints",
        live / "manifest.json",
        tmp_path / "report.json",
        max_tokens=13,
    )
    assert result["evaluation"] == expected.to_dict()
    assert state_trees_equal(rng, capture_rng_state())


def test_checkpoint_evaluation_is_cpu_scoped_under_meta_default(tmp_path):
    manifest, config, model = checkpoint_run(tmp_path)
    expected = evaluate_held_out(model, manifest, max_tokens=13)
    rng = capture_rng_state()
    with torch.device("meta"):
        result = evaluation_cli.evaluate_checkpoint(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            tmp_path / "report.json",
            max_tokens=13,
        )
        assert torch.empty(1).device.type == "meta"
    assert result["device"] == "cpu"
    assert result["evaluation"] == expected.to_dict()
    assert state_trees_equal(rng, capture_rng_state())


def test_checkpoint_evaluation_ignores_float64_default_without_changing_it(tmp_path):
    manifest, config, model = checkpoint_run(tmp_path)
    expected = evaluate_held_out(model, manifest, max_tokens=13)
    original_dtype = torch.get_default_dtype()
    rng = capture_rng_state()
    try:
        torch.set_default_dtype(torch.float64)
        result = evaluation_cli.evaluate_checkpoint(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            tmp_path / "report.json",
            max_tokens=13,
        )
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original_dtype)
    assert result["evaluation"] == expected.to_dict()
    assert result["dtype"] == "torch.float32"
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16, torch.bfloat16])
def test_non_fp32_checkpoint_is_rejected_before_evaluation_or_report(tmp_path, monkeypatch, dtype):
    from fttl.dataset import load_dataset_snapshot
    from fttl.model import TinyTransformer

    manifest_path, config, _ = checkpoint_run(tmp_path)
    manifest = load_dataset_snapshot(manifest_path).manifest
    model = TinyTransformer(config.model).to(dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    store = tmp_path / "other-checkpoint"
    save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=0,
        tokens_seen=0,
        losses=[],
        data_fingerprint=manifest.fingerprint(),
        tokenizer_fingerprint=manifest.tokenizer_fingerprint,
    )
    monkeypatch.setattr(
        evaluation_cli, "evaluate_held_out", lambda *args, **kwargs: pytest.fail("dtype converted")
    )
    rng = capture_rng_state()
    output = tmp_path / "new" / "report.json"
    with pytest.raises(CheckpointIntegrityError, match="FP32"):
        evaluation_cli.evaluate_checkpoint(config, store, manifest_path, output, max_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_exact_captured_snapshot_is_used_once_for_checkpoint_binding_and_evaluation(
    tmp_path, monkeypatch
):
    manifest, config, _ = checkpoint_run(tmp_path)
    original_capture = evaluation_cli.load_dataset_snapshot
    original_evaluate = evaluation_cli.evaluate_held_out
    original_checkpoint = evaluation_cli.load_checkpoint
    captured = []

    def capture(path):
        snapshot = original_capture(path)
        captured.append(snapshot)
        return snapshot

    def checkpoint(*args, **kwargs):
        assert kwargs["expected_data_fingerprint"] == captured[0].manifest.fingerprint()
        assert (
            kwargs["expected_tokenizer_fingerprint"] == captured[0].manifest.tokenizer_fingerprint
        )
        return original_checkpoint(*args, **kwargs)

    def evaluate(model, snapshot, **kwargs):
        assert isinstance(snapshot, PreparedDatasetSnapshot)
        assert snapshot is captured[0]
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        return original_evaluate(model, snapshot, **kwargs)

    monkeypatch.setattr(evaluation_cli, "load_dataset_snapshot", capture)
    monkeypatch.setattr(evaluation_cli, "load_checkpoint", checkpoint)
    monkeypatch.setattr(evaluation_cli, "evaluate_held_out", evaluate)
    evaluation_cli.evaluate_checkpoint(
        config,
        tmp_path / "training/checkpoints",
        manifest,
        tmp_path / "report.json",
        max_tokens=13,
    )
    assert len(captured) == 1


@pytest.mark.parametrize("stage", ["load_checkpoint", "evaluate_held_out", "publish_json"])
def test_failures_restore_all_caller_cpu_rngs_and_never_publish(tmp_path, monkeypatch, stage):
    manifest, config, _ = checkpoint_run(tmp_path)

    def fail(*args, **kwargs):
        random.random()
        np.random.random(9)
        torch.rand(7)
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(evaluation_cli, stage, fail)
    rng = capture_rng_state()
    output = tmp_path / "new" / "report.json"
    with torch.device("meta"), pytest.raises(RuntimeError, match="injected evaluation failure"):
        evaluation_cli.evaluate_checkpoint(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            output,
            max_tokens=13,
        )
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.exists()


def test_success_restores_python_numpy_and_torch_rngs_together(tmp_path, monkeypatch):
    manifest, config, _ = checkpoint_run(tmp_path)
    original = evaluation_cli.evaluate_held_out

    def consume_rng(model, snapshot, **kwargs):
        random.random()
        np.random.random(9)
        torch.rand(7)
        return original(model, snapshot, **kwargs)

    monkeypatch.setattr(evaluation_cli, "evaluate_held_out", consume_rng)
    rng = capture_rng_state()
    evaluation_cli.evaluate_checkpoint(
        config,
        tmp_path / "training/checkpoints",
        manifest,
        tmp_path / "report.json",
        max_tokens=13,
    )
    assert state_trees_equal(rng, capture_rng_state())


def test_corrupt_snapshot_rejects_before_model_creation_or_publication(tmp_path, monkeypatch):
    manifest, config, _ = checkpoint_run(tmp_path)
    manifest.write_bytes(manifest.read_bytes() + b"broken JSON")
    monkeypatch.setattr(
        evaluation_cli, "TinyTransformer", lambda _: pytest.fail("model initialized")
    )
    rng = capture_rng_state()
    output = tmp_path / "new" / "report.json"
    with pytest.raises(DatasetValidationError):
        evaluation_cli.evaluate_checkpoint(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            output,
            max_tokens=13,
        )
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()
