import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fttl import preflight
from fttl.state import capture_rng_state, state_trees_equal


@pytest.mark.skipif(os.name != "posix", reason="checkpoint writer probes require POSIX")
def test_preflight_measures_current_environment_and_preserves_rng(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("GPU discovery"))
    before = capture_rng_state()
    output = tmp_path / "probe.json"
    with torch.device("meta"):
        report = preflight.preflight(output)
    assert report["ready"] and all(report["checks"].values()) and not report["failures"]
    assert report == json.loads(output.read_text())
    assert state_trees_equal(before, capture_rng_state())
    assert list(tmp_path.iterdir()) == [output]
    assert str(tmp_path) not in output.read_text()


@pytest.mark.parametrize("stage", ["save_checkpoint", "load_checkpoint"])
def test_failed_primitives_are_reported_without_private_paths(tmp_path, monkeypatch, stage):
    monkeypatch.setattr(
        preflight, stage, lambda *a, **kw: (_ for _ in ()).throw(OSError(str(tmp_path)))
    )
    before = capture_rng_state()
    output = tmp_path / "failed.json"
    report = preflight.preflight(output)
    assert not report["ready"]
    assert report["failures"]["cpu_training_checkpoint_primitives"] == "OSError"
    assert str(tmp_path) not in output.read_text()
    assert state_trees_equal(before, capture_rng_state())
    assert list(tmp_path.iterdir()) == [output]


def test_existing_evidence_is_preserved(tmp_path):
    output = tmp_path / "prior.json"
    output.write_bytes(b"prior")
    with pytest.raises(ValueError, match="fresh"):
        preflight.preflight(output)
    assert output.read_bytes() == b"prior"


@pytest.mark.skipif(os.name != "posix", reason="checkpoint writer probes require POSIX")
def test_installed_preflight_runs_offline(tmp_path):
    output = tmp_path / "probe.json"
    result = subprocess.run(
        [str(Path(sys.executable).with_name("fttl-preflight")), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert not result.stderr
    assert json.loads(result.stdout) == json.loads(output.read_text())
