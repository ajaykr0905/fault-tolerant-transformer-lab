import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl import inference_parity
from fttl.recovery import verify_recovery
from fttl.state import capture_rng_state, state_trees_equal
from fttl.train import run_training


def test_replayed_training_preserves_greedy_and_sampled_serving_tokens(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    config = real_data_config()
    recovery = verify_recovery(
        config, manifest, tmp_path / "run", failure_point="after-optimizer", restarts=1
    )
    assert recovery.exact_equality
    before = capture_rng_state()
    output = tmp_path / "parity.json"
    report = inference_parity.verify_inference_parity(
        config,
        tmp_path / "run/control/checkpoints",
        tmp_path / "run/recovered/checkpoints",
        manifest,
        output,
        prompts=("Hello", "నమస్కారం"),
        max_new_tokens=4,
    )
    assert report == json.loads(output.read_text())
    assert report["exact_equality"] and all(report["equality"].values())
    assert len(report["cases"]) == 4
    assert state_trees_equal(before, capture_rng_state())


def test_distinct_completed_step_is_a_measured_parity_failure(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    config = real_data_config()
    run_training(config, tmp_path / "left", dataset_manifest=manifest)
    run_training(config, tmp_path / "right", dataset_manifest=manifest, stop_after_step=2)
    report = inference_parity.verify_inference_parity(
        config,
        tmp_path / "left/checkpoints",
        tmp_path / "right/checkpoints",
        manifest,
        tmp_path / "fail.json",
        max_new_tokens=1,
    )
    assert not report["exact_equality"]
    assert not report["equality"]["training_steps"] and not report["equality"]["model_state"]


@pytest.mark.parametrize(
    "controls",
    [
        {"prompts": "x"},
        {"prompts": []},
        {"prompts": [""]},
        {"prompts": [None]},
        {"prompts": ["x"] * 9},
        {"max_new_tokens": 0},
        {"max_new_tokens": True},
        {"seed": -1},
    ],
)
def test_invalid_plan_rejects_before_loading_or_publishing(tmp_path, monkeypatch, controls):
    monkeypatch.setattr(
        inference_parity, "load_inference_checkpoint", lambda *args: pytest.fail("I/O")
    )
    with pytest.raises(ValueError):
        inference_parity.verify_inference_parity(
            real_data_config(),
            tmp_path,
            tmp_path,
            tmp_path,
            tmp_path / "out.json",
            **controls,
        )
    assert not (tmp_path / "out.json").exists()


def test_existing_parity_evidence_is_not_overwritten(tmp_path):
    output = tmp_path / "report.json"
    output.write_bytes(b"old")
    with pytest.raises(ValueError, match="fresh"):
        inference_parity.verify_inference_parity(
            real_data_config(), tmp_path, tmp_path, tmp_path, output
        )
    assert output.read_bytes() == b"old"


@pytest.mark.parametrize("right_steps,expected_exit", [(4, 0), (2, 1)])
def test_installed_cli_preserves_pass_and_failure_reports(tmp_path, right_steps, expected_exit):
    manifest = prepared_fixture(tmp_path / "data")
    config = real_data_config()
    run_training(config, tmp_path / "left", dataset_manifest=manifest)
    run_training(config, tmp_path / "right", dataset_manifest=manifest, stop_after_step=right_steps)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            str(Path(sys.executable).with_name("fttl-verify-inference-parity")),
            "--config",
            str(config_path),
            "--control-checkpoint",
            str(tmp_path / "left/checkpoints"),
            "--recovered-checkpoint",
            str(tmp_path / "right/checkpoints"),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--max-new-tokens",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == expected_exit and not completed.stderr
    assert json.loads(completed.stdout) == json.loads(output.read_text())
    assert json.loads(output.read_text())["exact_equality"] == (expected_exit == 0)
