import copy
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


@pytest.mark.skipif(os.name != "posix", reason="checkpoint writer probes require POSIX")
def test_cpu_sgd_probe_avoids_accelerator_discovery_and_restores_nonempty_momentum(
    tmp_path, monkeypatch
):
    import torch

    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda **_kw: pytest.fail("accelerator discovery")
    )
    original_save = preflight.save_checkpoint
    original_load = preflight.load_checkpoint
    observed = {}

    def observe_save(*args, **kwargs):
        optimizer = kwargs["optimizer"]
        assert type(optimizer) is torch.optim.SGD
        observed["source"] = copy.deepcopy(optimizer.state_dict())
        observed["parameter_count"] = len(list(kwargs["model"].parameters()))
        return original_save(*args, **kwargs)

    def observe_load(*args, **kwargs):
        payload = original_load(*args, **kwargs)
        optimizer = kwargs["optimizer"]
        assert type(optimizer) is torch.optim.SGD
        observed["restored"] = copy.deepcopy(optimizer.state_dict())
        return payload

    monkeypatch.setattr(preflight, "save_checkpoint", observe_save)
    monkeypatch.setattr(preflight, "load_checkpoint", observe_load)
    before = capture_rng_state()
    report = preflight.preflight(tmp_path / "sgd.json")
    assert report["ready"] and report["checks"]["nonempty_sgd_momentum_checkpoint_state"]
    assert report["checks"]["checkpoint_roundtrip_exact"]
    assert report["environment"]["optimizer"] == "torch.optim.SGD"
    assert report["optimizer_policy"] == {"momentum": 0.9, "foreach": False, "fused": False}
    assert any("AdamW" in limitation for limitation in report["limitations"])
    assert state_trees_equal(observed["source"], observed["restored"])
    assert state_trees_equal(before, capture_rng_state())
    for kind in ("source", "restored"):
        state = observed[kind]
        assert state["state"] and len(state["state"]) == observed["parameter_count"]
        for group in state["param_groups"]:
            assert group["momentum"] == 0.9 and group["foreach"] is group["fused"] is False
        for value in state["state"].values():
            assert set(value) == {"momentum_buffer"}
            momentum = value["momentum_buffer"]
            assert momentum.device.type == "cpu" and momentum.dtype is torch.float32
            assert bool(torch.isfinite(momentum).all())


def test_missing_optimizer_execution_cannot_pass_empty_state_roundtrip(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.optim.SGD, "step", lambda *_a, **_kw: None)
    report = preflight.preflight(tmp_path / "unexecuted.json")
    assert not report["ready"]
    assert not report["checks"]["nonempty_sgd_momentum_checkpoint_state"]
    assert not report["checks"]["checkpoint_roundtrip_exact"]


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
