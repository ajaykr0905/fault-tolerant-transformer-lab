from dataclasses import replace

import pytest
import torch
from test_evaluation_cli import checkpoint_run
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl.checkpoint import CheckpointIntegrityError, CheckpointMismatchError, save_checkpoint
from fttl.inference import load_inference_checkpoint
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_digest, state_trees_equal


def test_frozen_checkpoint_reconstruction_matches_live_model_and_preserves_caller(tmp_path):
    manifest, config, trained = checkpoint_run(tmp_path)
    trained.eval()
    tokens = torch.arange(8).reshape(1, 8)
    expected = trained(tokens)[0]
    rng = capture_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    model, receipt = load_inference_checkpoint(config, tmp_path / "training/checkpoints", manifest)
    torch.testing.assert_close(model(tokens)[0], expected, rtol=0, atol=0)
    assert state_trees_equal(rng, capture_rng_state())
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert not any(module.training for module in model.modules())
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.lm_head.weight is model.token_embedding.weight
    assert receipt.model_state_digest == state_digest(trained.state_dict())
    assert receipt.completed_training_steps == 4
    assert receipt.selected_checkpoint_generation == 4
    assert receipt.config_fingerprint == config.fingerprint()
    assert receipt.device == "cpu" and receipt.dtype == "torch.float32"
    with torch.no_grad():
        model.token_embedding.weight.add_(1)
    assert receipt.model_state_digest == state_digest(trained.state_dict())


def test_corrupt_latest_uses_an_older_committed_generation(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    store = tmp_path / "training/checkpoints"
    (store / "generation-00000004/state.pt").write_bytes(b"corrupt")
    model, receipt = load_inference_checkpoint(config, store, manifest)
    assert receipt.selected_checkpoint_generation == 3
    assert receipt.completed_training_steps == 3
    assert receipt.model_state_digest == state_digest(model.state_dict())


def test_configuration_rejection_does_not_change_caller_rng(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    rng = capture_rng_state()
    with pytest.raises(CheckpointMismatchError, match="configuration"):
        load_inference_checkpoint(
            replace(config, seed=42), tmp_path / "training/checkpoints", manifest
        )
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize(
    "config",
    [None, {}, replace(real_data_config(), model=replace(real_data_config().model, vocab_size=64))],
)
def test_bad_config_rejects_before_artifact_access(tmp_path, config):
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="config|vocab_size"):
        load_inference_checkpoint(config, tmp_path / "missing", tmp_path / "missing")
    assert state_trees_equal(rng, capture_rng_state())


def test_other_manifest_cannot_relabel_checkpoint(tmp_path):
    import json

    from fttl.dataset import _sha256_json

    manifest, config, _ = checkpoint_run(tmp_path)
    value = json.loads(manifest.read_text())
    value["source"]["revision"] = "another-public-revision"
    value.pop("dataset_fingerprint")
    value["dataset_fingerprint"] = _sha256_json(value)
    manifest.write_text(json.dumps(value))
    with pytest.raises(CheckpointMismatchError, match="data fingerprint"):
        load_inference_checkpoint(config, tmp_path / "training/checkpoints", manifest)


def test_reconstruction_is_cpu_even_under_meta_default_device(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    with torch.device("meta"):
        model, receipt = load_inference_checkpoint(
            config, tmp_path / "training/checkpoints", manifest
        )
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert receipt.device == "cpu"


def test_fp64_checkpoint_is_not_silently_relabelled_fp32(tmp_path):
    from fttl.dataset import load_dataset_manifest

    manifest_path = prepared_fixture(tmp_path / "dataset")
    manifest = load_dataset_manifest(manifest_path)
    config = real_data_config()
    model = TinyTransformer(config.model).double()
    optimizer = torch.optim.AdamW(model.parameters())
    store = tmp_path / "checkpoint"
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
    rng = capture_rng_state()
    with pytest.raises(CheckpointIntegrityError, match="FP32"):
        load_inference_checkpoint(config, store, manifest_path)
    assert state_trees_equal(rng, capture_rng_state())
