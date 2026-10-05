import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_evaluation_cli import checkpoint_run

from fttl import generation_cli
from fttl.state import capture_rng_state, state_trees_equal


@pytest.mark.parametrize("installed", [False, True])
def test_cli_generates_exact_published_report_offline(tmp_path, installed):
    manifest, config, _ = checkpoint_run(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    output = tmp_path / "generation.json"
    command = (
        [str(Path(sys.executable).with_name("fttl-generate"))]
        if installed
        else [sys.executable, "-m", "fttl.generation_cli"]
    )
    completed = subprocess.run(
        [
            *command,
            "--config",
            str(config_path),
            "--checkpoint",
            str(tmp_path / "training/checkpoints"),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--prompt",
            "నమస్కారం",
            "--method",
            "sample",
            "--top-k",
            "3",
            "--seed",
            "19",
            "--max-new-tokens",
            "4",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert not completed.stderr
    assert result == json.loads(output.read_text())
    assert result["generation"]["prompt_token_ids"] == list("నమస్కారం".encode())
    assert result["generation"]["generated_count"] == 4
    assert result["inference"]["completed_training_steps"] == 4
    assert result["generation"]["model_state_digest"] == result["inference"]["model_state_digest"]


def test_api_replays_sampling_without_mutating_caller_rng(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    before = capture_rng_state()
    results = [
        generation_cli.generate_checkpoint(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            tmp_path / f"report-{i}.json",
            prompt="Hello",
            method="sample",
            seed=9,
            max_new_tokens=5,
        )
        for i in range(2)
    ]
    assert results[0] == results[1]
    assert state_trees_equal(before, capture_rng_state())


@pytest.mark.parametrize("prompt", ["", "a" * 4097, "\ud800", None])
def test_bad_prompt_rejects_before_checkpoint_io(tmp_path, monkeypatch, prompt):
    monkeypatch.setattr(
        generation_cli, "load_inference_checkpoint", lambda *args: pytest.fail("I/O")
    )
    from test_real_data_recovery import real_data_config

    with pytest.raises(ValueError):
        generation_cli.generate_checkpoint(
            real_data_config(),
            tmp_path / "missing",
            tmp_path / "missing",
            tmp_path / "out.json",
            prompt=prompt,
        )
    assert not (tmp_path / "out.json").exists()


def test_existing_report_is_preserved_before_checkpoint_io(tmp_path):
    output = tmp_path / "report.json"
    output.write_bytes(b"prior evidence")
    from test_real_data_recovery import real_data_config

    with pytest.raises(ValueError, match="fresh"):
        generation_cli.generate_checkpoint(
            real_data_config(), tmp_path, tmp_path, output, prompt="x"
        )
    assert output.read_bytes() == b"prior evidence"


@pytest.mark.parametrize(
    "option,value",
    [("--max-new-tokens", "257"), ("--temperature", "nan"), ("--top-k", "0"), ("--seed", "-1")],
)
def test_invalid_cli_controls_publish_nothing(tmp_path, capsys, option, value):
    manifest, config, _ = checkpoint_run(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    with pytest.raises(SystemExit) as error:
        generation_cli.main(
            [
                "--config",
                str(config_path),
                "--checkpoint",
                str(tmp_path / "training/checkpoints"),
                "--dataset-manifest",
                str(manifest),
                "--output",
                str(tmp_path / "out.json"),
                "--prompt",
                "x",
                option,
                value,
            ]
        )
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "Traceback" not in captured.err
    assert not (tmp_path / "out.json").exists()
