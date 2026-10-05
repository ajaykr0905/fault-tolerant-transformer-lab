import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from fttl import reports


def test_published_report_is_complete_json_without_leftover_temporaries(tmp_path):
    output = tmp_path / "new/report.json"
    value = {"run": "fixture", "losses": [1.0, 0.5]}
    reports.publish_json(output, value)
    assert json.loads(output.read_text()) == value
    assert output.read_bytes().endswith(b"\n")
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_path_is_never_replaced(tmp_path, kind):
    output = tmp_path / "report.json"
    if kind == "file":
        output.write_bytes(b"original evidence")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        reports.publish_json(output, {"new": True})
    if kind == "file":
        assert output.read_bytes() == b"original evidence"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink() and not (tmp_path / "absent").exists()
    assert not list(tmp_path.glob(".report.json.tmp-*"))


@pytest.mark.parametrize("stage", ["fsync", "link"])
def test_prepublication_failure_leaves_no_partial_report(monkeypatch, tmp_path, stage):
    output = tmp_path / "report.json"

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    monkeypatch.setattr(reports.os, stage, fail)
    with pytest.raises(OSError, match="injected"):
        reports.publish_json(output, {"verified": True})
    assert not output.exists()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [{"loss": float("nan")}, {"unsupported": object()}])
def test_invalid_json_is_rejected_before_creating_parent_directories(tmp_path, value):
    output = tmp_path / "new/report.json"
    with pytest.raises((ValueError, TypeError)):
        reports.publish_json(output, value)
    assert not output.parent.exists()


def test_competing_publishers_produce_one_whole_report(monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    original_link = reports.os.link
    barrier = Barrier(2)

    def race_link(source, destination):
        barrier.wait(timeout=5)
        original_link(source, destination)

    def publish(value):
        try:
            reports.publish_json(output, value)
        except FileExistsError:
            return False
        return True

    monkeypatch.setattr(reports.os, "link", race_link)
    values = [{"run": 1, "losses": [1.0] * 100}, {"run": 2, "losses": [2.0] * 100}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = list(pool.map(publish, values))
    assert sum(winners) == 1
    assert json.loads(output.read_text()) == values[winners.index(True)]
    assert list(tmp_path.iterdir()) == [output]
