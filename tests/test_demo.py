import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fttl.demo import export_preview, run_demo

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
