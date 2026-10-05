import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_real_data_recovery import prepared_fixture, real_data_config

from fttl import public_tuning_cli


def _arguments(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(real_data_config().canonical_json(), encoding="utf-8")
    manifest = prepared_fixture(tmp_path / "data")
    return [
        "--config",
        str(config),
        "--dataset-manifest",
        str(manifest),
        "--output",
        str(tmp_path / "comparison.json"),
        "--rank",
        "2",
        "--max-eval-tokens",
        "17",
    ]


@pytest.mark.parametrize("installed", [False, True])
def test_cli_executes_offline_public_comparison_and_prints_published_json(tmp_path, installed):
    arguments = _arguments(tmp_path)
    command = (
        [str(Path(sys.executable).with_name("fttl-compare-public-tuning"))]
        if installed
        else [sys.executable, "-m", "fttl.public_tuning_cli"]
    )
    completed = subprocess.run(
        [*command, *arguments], capture_output=True, text=True, timeout=30, check=True
    )
    assert not completed.stderr
    report = json.loads(completed.stdout)
    assert report == json.loads((tmp_path / "comparison.json").read_text())
    assert report["paired_batches"] and report["paired_cpu_rng_sequence"]
    assert report["matched_initial_predictions"]
    assert report["lora"]["base_state_unchanged"]
    for mode in ("full", "lora"):
        assert report[mode]["completed_steps"] == 4
        assert report[mode]["validation_after"]["evaluated_target_tokens"] == 17
        assert report[mode]["validation_after"]["split"] == "validation"


@pytest.mark.parametrize(
    "option,value,message",
    [
        ("--rank", "0", "rank must"),
        ("--rank", "99", "rank must"),
        ("--rank", "invalid", "invalid int"),
        ("--max-eval-tokens", "0", "max_eval_tokens"),
        ("--max-eval-tokens", "-1", "max_eval_tokens"),
    ],
)
def test_cli_invalid_controls_fail_cleanly_without_output(tmp_path, capsys, option, value, message):
    arguments = _arguments(tmp_path)
    arguments[arguments.index(option) + 1] = value
    with pytest.raises(SystemExit) as error:
        public_tuning_cli.main(arguments)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert not captured.out and "Traceback" not in captured.err
    assert not (tmp_path / "comparison.json").exists()


@pytest.mark.parametrize(
    "config_text,message",
    [
        ('{"seed":1,"seed":2}', "duplicate JSON field"),
        ('{"model":null}', "model configuration must be an object"),
        ('{"unrecognized":1}', "unknown experiment fields"),
        ("{", "Expecting property name"),
    ],
)
def test_cli_rejects_bad_config_before_running_comparison(
    tmp_path, monkeypatch, capsys, config_text, message
):
    arguments = _arguments(tmp_path)
    (tmp_path / "config.json").write_text(config_text)
    monkeypatch.setattr(
        public_tuning_cli,
        "compare_public_tuning",
        lambda *args, **kwargs: pytest.fail("ran comparison"),
    )
    with pytest.raises(SystemExit) as error:
        public_tuning_cli.main(arguments)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "comparison.json").exists()


def test_cli_preserves_existing_evidence(tmp_path, capsys):
    arguments = _arguments(tmp_path)
    output = tmp_path / "comparison.json"
    output.write_bytes(b"earlier evidence")
    with pytest.raises(SystemExit) as error:
        public_tuning_cli.main(arguments)
    assert error.value.code == 2
    assert "fresh output" in capsys.readouterr().err
    assert output.read_bytes() == b"earlier evidence"


@pytest.mark.parametrize(
    "failure", [OSError("unavailable artifact"), FloatingPointError("nonfinite")]
)
def test_cli_reports_expected_operational_errors_without_traceback(
    tmp_path, monkeypatch, capsys, failure
):
    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(public_tuning_cli, "compare_public_tuning", fail)
    with pytest.raises(SystemExit) as error:
        public_tuning_cli.main(_arguments(tmp_path))
    assert error.value.code == 2
    assert str(failure) in capsys.readouterr().err
    assert not (tmp_path / "comparison.json").exists()


def test_cli_does_not_disguise_unexpected_backend_failures(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("backend defect")

    monkeypatch.setattr(public_tuning_cli, "compare_public_tuning", fail)
    with pytest.raises(RuntimeError, match="backend defect"):
        public_tuning_cli.main(_arguments(tmp_path))
