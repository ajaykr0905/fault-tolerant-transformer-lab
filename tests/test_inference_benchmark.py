import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_evaluation_cli import checkpoint_run
from test_real_data_recovery import real_data_config

from fttl import inference_benchmark
from fttl.state import capture_rng_state, state_trees_equal


def test_report_counts_real_tokens_and_defines_timing_boundaries(tmp_path, monkeypatch):
    manifest, config, _ = checkpoint_run(tmp_path)
    clock = iter([0.0, 0.2, 1.0, 1.4, 2.0, 2.6])
    monkeypatch.setattr(inference_benchmark, "perf_counter", lambda: next(clock))
    before = capture_rng_state()
    output = tmp_path / "report.json"
    report = inference_benchmark.benchmark_inference(
        config,
        tmp_path / "training/checkpoints",
        manifest,
        output,
        iterations=3,
        warmups=1,
        max_new_tokens=2,
    )
    assert report == json.loads(output.read_text())
    assert report["total_generated_tokens"] == 6
    assert report["generation_seconds"] == pytest.approx(1.2)
    assert report["median_seconds"] == pytest.approx(0.4)
    assert report["p95_nearest_rank_seconds"] == pytest.approx(0.6)
    assert report["tokens_per_second"] == pytest.approx(5)
    assert report["replayed_tokens_equal"] and report["warmups_excluded"] == 1
    assert state_trees_equal(before, capture_rng_state())


def test_zero_clock_resolution_reports_no_infinite_throughput(tmp_path, monkeypatch):
    manifest, config, _ = checkpoint_run(tmp_path)
    monkeypatch.setattr(inference_benchmark, "perf_counter", lambda: 1.0)
    result = inference_benchmark.benchmark_inference(
        config,
        tmp_path / "training/checkpoints",
        manifest,
        tmp_path / "zero.json",
        iterations=1,
        warmups=0,
        max_new_tokens=1,
    )
    assert result["tokens_per_second"] is None


@pytest.mark.parametrize("finish", [-1.0, float("nan"), float("inf")])
def test_invalid_clock_never_publishes_measurement(tmp_path, monkeypatch, finish):
    manifest, config, _ = checkpoint_run(tmp_path)
    clock = iter([0.0, finish])
    monkeypatch.setattr(inference_benchmark, "perf_counter", lambda: next(clock))
    with pytest.raises(ValueError, match="clock"):
        inference_benchmark.benchmark_inference(
            config,
            tmp_path / "training/checkpoints",
            manifest,
            tmp_path / "bad.json",
            iterations=1,
            warmups=0,
            max_new_tokens=1,
        )
    assert not (tmp_path / "bad.json").exists()


def test_existing_benchmark_evidence_is_preserved(tmp_path):
    output = tmp_path / "prior.json"
    output.write_bytes(b"prior")
    with pytest.raises(ValueError, match="fresh"):
        inference_benchmark.benchmark_inference(real_data_config(), tmp_path, tmp_path, output)
    assert output.read_bytes() == b"prior"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": 21},
        {"warmups": 6},
        {"warmups": True},
        {"max_new_tokens": 33},
        {"seed": -1},
        {"prompt": ""},
        {"prompt": "x" * 4097},
    ],
)
def test_invalid_benchmark_controls_do_not_load_or_publish(tmp_path, monkeypatch, kwargs):
    monkeypatch.setattr(
        inference_benchmark, "load_inference_checkpoint", lambda *args: pytest.fail("I/O")
    )
    with pytest.raises(ValueError):
        inference_benchmark.benchmark_inference(
            real_data_config(), tmp_path, tmp_path, tmp_path / "out.json", **kwargs
        )
    assert not (tmp_path / "out.json").exists()


def test_installed_benchmark_executes_offline(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    output = tmp_path / "benchmark.json"
    run = subprocess.run(
        [
            str(Path(sys.executable).with_name("fttl-benchmark-inference")),
            "--config",
            str(config_path),
            "--checkpoint",
            str(tmp_path / "training/checkpoints"),
            "--dataset-manifest",
            str(manifest),
            "--output",
            str(output),
            "--iterations",
            "1",
            "--max-new-tokens",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    report = json.loads(run.stdout)
    assert report == json.loads(output.read_text()) and report["total_generated_tokens"] == 1
    assert not run.stderr
