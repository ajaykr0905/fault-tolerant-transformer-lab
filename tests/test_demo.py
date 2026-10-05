import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fttl.config import ExperimentConfig
from fttl.demo import FIXTURE_SHA256, export_preview, run_demo

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/public_domain_peps_fixture.jsonl"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    if os.name != "posix":
        pytest.skip("the working demo requires POSIX SIGKILL")
    output = tmp_path_factory.mktemp("demo") / "run-001"
    completed = subprocess.run(
        [sys.executable, "-m", "fttl.demo", "--output", str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    return output, completed


def test_one_command_runs_real_kill_and_exports_measured_preview(demo):
    output, completed = demo
    preview = output / "preview"
    report = json.loads((preview / "process-recovery-report.json").read_text())
    assert report["interrupted_exitcode"] == -9
    assert len(set(report["worker_pids"].values())) == 4
    assert report["selected_step"] == 1
    assert report["completed_steps"] == 6
    assert report["tokens_seen"] == 96
    assert report["exact_equality"] and all(report["equality"].values())
    assert report["durable_committed_steps_lost"] == 0
    assert "PASS: 16/16" in completed.stdout
    page = (preview / "index.html").read_text()
    assert "does not start training in your browser" in page
    assert page.count("<strong>PASS</strong>") == len(report["equality"])
    assert report["final_state_digest"] in page
    assert "<script" not in page and "https://" not in page
    root = ET.parse(preview / "preview.svg").getroot()
    lines = root.findall(".//{http://www.w3.org/2000/svg}polyline")
    assert lines[0].attrib["points"] == lines[1].attrib["points"]
    assert len(lines[0].attrib["points"].split()) == report["completed_steps"]
    for relative in ("control/result.json", "recovered/result.json"):
        assert (preview / relative).read_bytes() == (output / "run" / relative).read_bytes()
    assert not list(preview.rglob("*.pt"))
    assert str(output) not in "".join(path.read_text() for path in preview.rglob("*.json"))


@pytest.mark.parametrize("change", ["failed-check", "false-summary", "changed-loss", "nan-loss"])
def test_preview_cannot_turn_failed_or_changed_evidence_green(demo, tmp_path, change):
    output, _ = demo
    run = tmp_path / "altered"
    shutil.copytree(output / "preview", run)
    report_path = run / "process-recovery-report.json"
    report = json.loads(report_path.read_text())
    if change == "failed-check":
        report["equality"]["model_tensors"] = False
    elif change == "false-summary":
        report["exact_equality"] = False
    else:
        for name in ("control", "recovered"):
            path = run / name / "result.json"
            value = json.loads(path.read_text())
            value["losses"][0] = float("nan") if change == "nan-loss" else 99.0
            if change == "changed-loss" and name == "control":
                continue
            path.write_text(json.dumps(value))
    report_path.write_text(json.dumps(report))
    destination = tmp_path / "preview"
    with pytest.raises(ValueError, match="diverged"):
        export_preview(run, destination)
    assert not destination.exists()


def test_report_strings_are_escaped_in_the_offline_viewer(demo, tmp_path):
    output, _ = demo
    run = tmp_path / "input"
    shutil.copytree(output / "preview", run)
    path = run / "process-recovery-report.json"
    report = json.loads(path.read_text())
    report["code_revision"] = "<script>alert('unsafe')</script>"
    report["limitations"] = ["<img src=x onerror=alert(1)>"]
    path.write_text(json.dumps(report))
    destination = tmp_path / "safe"
    export_preview(run, destination)
    page = (destination / "index.html").read_text()
    assert "<script>" not in page and "<img src=x" not in page
    assert "&lt;script&gt;" in page


@pytest.mark.parametrize(
    "file,field,value",
    [
        ("process-recovery-report", "interrupted_exitcode", 0),
        ("process-recovery-report", "kill_signal", "SIGTERM"),
        (
            "process-recovery-report",
            "worker_pids",
            {"control": 1, "interrupted": 1, "replay": 2, "completion": 3},
        ),
        ("process-recovery-report", "equality", {"model_tensors": True}),
        ("process-recovery-report", "durable_committed_steps_lost", 1),
        ("process-recovery-report", "durable_committed_tokens_lost", 1),
        ("process-recovery-report", "selected_step", 2),
        ("process-recovery-report", "attempted_step", "<script>alert(1)</script>"),
        ("process-recovery-report", "tokens_seen", True),
        ("process-recovery-report", "replayed_sample_ids", ["different sample"]),
        ("process-recovery-report", "failed_batch_id", "different batch"),
        ("control/result", "optimizer_digest", "0" * 64),
        ("recovered/result", "steps", 5),
        ("recovered/result", "tokens_seen", 95),
    ],
)
def test_preview_rejects_contradictory_kill_replay_and_trace_claims(
    demo, tmp_path, file, field, value
):
    output, _ = demo
    run = tmp_path / "altered"
    shutil.copytree(output / "preview", run)
    path = run / f"{file}.json"
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    destination = tmp_path / "preview"
    with pytest.raises(ValueError, match="diverged"):
        export_preview(run, destination)
    assert not destination.exists()


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf"), True])
def test_preview_rejects_invalid_measured_replay_timing(demo, tmp_path, seconds):
    output, _ = demo
    run = tmp_path / "altered"
    shutil.copytree(output / "preview", run)
    path = run / "process-recovery-report.json"
    record = json.loads(path.read_text())
    record["timings_seconds"]["resume_spawn_to_replayed_commit_receipt"] = seconds
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="diverged"):
        export_preview(run, tmp_path / "preview")
    assert not (tmp_path / "preview").exists()


def test_modified_fixture_is_rejected_before_any_run(tmp_path):
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_bytes(FIXTURE.read_bytes() + b"\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_demo(output, fixture=fixture)
    assert not output.exists()


def test_existing_output_and_preview_are_preserved(demo, tmp_path):
    output, _ = demo
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("user evidence")
    for target, preview in ((existing, None), (tmp_path / "unused", existing)):
        with pytest.raises(ValueError, match="fresh"):
            run_demo(target, fixture=FIXTURE, preview_dir=preview)
    with pytest.raises(FileExistsError):
        export_preview(output / "run", existing)
    assert sentinel.read_text() == "user evidence"
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("relative", ["run", "run/nested", "."])
def test_overlapping_preview_directory_is_rejected(tmp_path, relative):
    output = tmp_path / "run"
    preview = tmp_path / relative
    with pytest.raises(ValueError):
        run_demo(output, fixture=FIXTURE, preview_dir=preview)
    assert not output.exists()


def test_readme_preview_is_bound_to_the_published_run_and_public_fixture():
    preview = ROOT / "docs/demo"
    report = json.loads((preview / "process-recovery-report.json").read_text())
    control = json.loads((preview / "control/result.json").read_text())
    recovered = json.loads((preview / "recovered/result.json").read_text())
    config = ExperimentConfig.from_json((preview / "config.json").read_text())
    manifest = json.loads((preview / "dataset-manifest.json").read_text())
    assert report["code_revision"] == "9d5a901797825a9e1a3cf0a426e894874f3bec9a"
    assert report["exact_equality"] and len(report["equality"]) == 16
    assert all(value is True for value in report["equality"].values())
    assert report["config_fingerprint"] == config.fingerprint()
    assert manifest["source"]["compressed_sha256"] == FIXTURE_SHA256
    assert manifest["counts"]["documents"] == 6
    assert report["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    for name in ("losses", "batch_ids", "sample_ids", "final_state_digest"):
        assert control[name] == recovered[name]
    assert report["final_state_digest"] == recovered["final_state_digest"]
    assert report["interrupted_exitcode"] == -9
    assert report["selected_step"] == 1 and report["completed_steps"] == 6
    assert report["durable_committed_steps_lost"] == 0
    assert report["failed_sample_ids"] == report["replayed_sample_ids"]
    assert not list(preview.rglob("*.pt"))
    ET.parse(preview / "preview.svg")
