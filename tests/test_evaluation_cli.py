import hashlib
import json
import subprocess
import sys
from dataclasses import replace

import pytest
import torch
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl.evaluation import evaluate_held_out
from fttl.evaluation_cli import evaluate_checkpoint
from fttl.state import capture_rng_state, state_trees_equal
from fttl.train import run_training


def checkpoint_run(tmp_path):
    manifest = prepared_fixture(tmp_path / "dataset")
    config = real_data_config()
    model, _ = run_training(config, tmp_path / "training", dataset_manifest=manifest)
    return manifest, config, model


def test_verified_checkpoint_evaluation_matches_live_model_and_preserves_rng(tmp_path):
    manifest, config, model = checkpoint_run(tmp_path)
    expected = evaluate_held_out(model, manifest, max_tokens=17)
    before = capture_rng_state()
    output = tmp_path / "report.json"
    report = evaluate_checkpoint(
        config, tmp_path / "training/checkpoints", manifest, output, max_tokens=17
    )
    assert report["evaluation"] == expected.to_dict()
    assert report["completed_training_steps"] == 4
    assert report["selected_checkpoint_generation"] == 4
    assert report["config_fingerprint"] == config.fingerprint()
    assert report == json.loads(output.read_text())
    assert state_trees_equal(before, capture_rng_state())


def test_evaluation_never_overwrites_existing_output(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        evaluate_checkpoint(real_data_config(), tmp_path / "missing", tmp_path / "missing", output)
    assert output.read_text() == "preserve"


def test_evaluation_rejects_configuration_mismatch_without_report(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="configuration"):
        evaluate_checkpoint(
            replace(config, seed=42), tmp_path / "training/checkpoints", manifest, output
        )
    assert not output.exists()


def test_evaluation_rejects_other_dataset_identity_without_report(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    altered = json.loads(manifest.read_text())
    altered["source"]["revision"] = "different-fixture-revision"
    identity = dict(altered)
    identity.pop("dataset_fingerprint")
    altered["dataset_fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(altered))
    output = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="data fingerprint"):
        evaluate_checkpoint(config, tmp_path / "training/checkpoints", manifest, output)
    assert not output.exists()


def test_evaluation_cli_runs_offline_with_exact_token_budget(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    output = tmp_path / "cli.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fttl.evaluation_cli",
            "--config",
            str(config_path),
            "--checkpoint",
            str(tmp_path / "training/checkpoints"),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--split",
            "test",
            "--max-tokens",
            "19",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result == json.loads(output.read_text())
    assert result["evaluation"]["evaluated_target_tokens"] == 19
    assert result["evaluation"]["split"] == "test"
    assert result["device"] == "cpu"
    assert torch.isfinite(torch.tensor(result["evaluation"]["mean_negative_log_likelihood"]))
