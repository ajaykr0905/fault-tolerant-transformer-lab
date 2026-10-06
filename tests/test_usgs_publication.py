import json
import os
from pathlib import Path

import pytest

from fttl import usgs
from fttl.usgs import USGSCaptureLedger, USGSValidationError, load_snapshot


def _payload(updated=42):
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "independent-publication-event",
                    "properties": {"updated": updated},
                    "geometry": {"type": "Point", "coordinates": [1, 2, 3]},
                }
            ],
        }
    ).encode()


def _capture(ledger, updated=42):
    ledger.capture(_payload(updated), feed="all-day", captured_at="2026-10-06T00:00:00Z")


@pytest.fixture
def reference(tmp_path):
    with USGSCaptureLedger(tmp_path / "reference.sqlite3") as ledger:
        _capture(ledger)
        return ledger.seal_snapshot(tmp_path / "reference", feed="all-day")


@pytest.mark.parametrize("artifact", ["data", "manifest"])
@pytest.mark.parametrize("kind", ["symlink", "dangling-symlink", "directory", "fifo"])
def test_sealing_rejects_nonregular_targets_without_receipt_or_overwriting(
    tmp_path, reference, artifact, kind
):
    data, manifest, _ = reference
    output = tmp_path / "blocked"
    output.mkdir()
    target = output / (data.name if artifact == "data" else "snapshot-manifest.json")
    outside = tmp_path / "outside.json"
    original = (data if artifact == "data" else manifest).read_bytes()
    if kind == "symlink":
        outside.write_bytes(original)
        target.symlink_to(outside)
    elif kind == "dangling-symlink":
        target.symlink_to(outside)
    elif kind == "directory":
        target.mkdir()
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO layout requires POSIX filesystem support")
        os.mkfifo(target)
    target_inode = target.lstat().st_ino
    before = set(output.iterdir())
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        with pytest.raises(USGSValidationError, match="regular"):
            ledger.seal_snapshot(output, feed="all-day")
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 0
    assert target.lstat().st_ino == target_inode
    assert set(output.iterdir()) == before
    if kind == "symlink":
        assert target.is_symlink() and outside.read_bytes() == original
    elif kind == "dangling-symlink":
        assert target.is_symlink() and not outside.exists()


def test_identical_regular_data_is_idempotent_and_different_data_never_overwrites(
    tmp_path, reference
):
    source_data, _, _ = reference
    output = tmp_path / "snapshot"
    output.mkdir()
    target = output / source_data.name
    target.write_bytes(source_data.read_bytes())
    original_inode = target.stat().st_ino
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        _, manifest, _ = ledger.seal_snapshot(output, feed="all-day")
        assert target.stat().st_ino == original_inode
        assert len(load_snapshot(manifest)[1]) == 1
    other_output = tmp_path / "different"
    other_output.mkdir()
    other = other_output / source_data.name
    other.write_bytes(b"different preexisting bytes")
    with USGSCaptureLedger(tmp_path / "other.sqlite3") as ledger:
        _capture(ledger)
        with pytest.raises(USGSValidationError, match="different bytes"):
            ledger.seal_snapshot(other_output, feed="all-day")
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 0
    assert other.read_bytes() == b"different preexisting bytes"
    assert not (other_output / "snapshot-manifest.json").exists()


def test_regular_latest_manifest_still_updates_for_new_events(tmp_path):
    output = tmp_path / "snapshot"
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        first_data, manifest, first = ledger.seal_snapshot(output, feed="all-day")
        first_bytes = first_data.read_bytes()
        _capture(ledger, updated=43)
        _, second_manifest, second = ledger.seal_snapshot(output, feed="all-day")
        assert second_manifest == manifest
        assert first.content_sha256 != second.content_sha256
        assert first_data.read_bytes() == first_bytes
        assert len(load_snapshot(manifest)[1]) == 2
        _, _, repeated = ledger.seal_snapshot(output, feed="all-day")
        assert repeated == second


@pytest.mark.parametrize("winner", ["identical", "different", "symlink"])
@pytest.mark.parametrize("artifact", ["data", "manifest"])
def test_competing_artifact_creator_cannot_be_overwritten(
    tmp_path, reference, monkeypatch, winner, artifact
):
    source_data, source_manifest, _ = reference
    source_artifact = source_data if artifact == "data" else source_manifest
    output = tmp_path / "raced"
    output.mkdir()
    target = output / source_artifact.name
    outside = tmp_path / "race-outside.json"
    outside.write_bytes(source_artifact.read_bytes())
    original_link = usgs.os.link
    created = []

    def raced_link(source, destination):
        if Path(destination) == target and not created:
            if winner == "symlink":
                target.symlink_to(outside)
            else:
                target.write_bytes(
                    source_artifact.read_bytes() if winner == "identical" else b"winner"
                )
            created.append(target.lstat().st_ino)
        return original_link(source, destination)

    monkeypatch.setattr(usgs.os, "link", raced_link)
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        if winner == "identical":
            _, manifest, _ = ledger.seal_snapshot(output, feed="all-day")
            assert len(load_snapshot(manifest)[1]) == 1
        else:
            with pytest.raises(USGSValidationError):
                ledger.seal_snapshot(output, feed="all-day")
            assert (
                ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0]
                == 0
            )
    assert created and target.lstat().st_ino == created[0]
    assert outside.read_bytes() == source_artifact.read_bytes()
    assert not tuple(output.glob(".*.tmp-*"))
    if winner == "different":
        assert target.read_bytes() == b"winner"
    elif winner == "symlink":
        assert target.is_symlink()


def test_raced_fifo_is_rejected_without_opening_a_blocking_reader(tmp_path, reference, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO layout requires POSIX filesystem support")
    source_data, _, _ = reference
    output = tmp_path / "raced-fifo"
    output.mkdir()
    target = output / source_data.name
    target.write_bytes(source_data.read_bytes())
    original_open = usgs.os.open
    raced = []

    def raced_open(path, flags, *args, **kwargs):
        if Path(path) == target and not raced:
            assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
            target.unlink()
            os.mkfifo(target)
            raced.append(target.lstat().st_ino)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(usgs.os, "open", raced_open)
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        with pytest.raises(USGSValidationError, match="regular"):
            ledger.seal_snapshot(output, feed="all-day")
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 0
    assert raced and target.lstat().st_ino == raced[0]
    assert not (output / "snapshot-manifest.json").exists()


def test_rolling_manifest_replaced_by_symlink_is_not_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "rolling"
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        _capture(ledger)
        data, manifest, _ = ledger.seal_snapshot(output, feed="all-day")
        original_data = data.read_bytes()
        outside = tmp_path / "outside-manifest.json"
        outside.write_bytes(manifest.read_bytes())
        original_manifest = outside.read_bytes()
        original_temporary = usgs.tempfile.NamedTemporaryFile
        raced = []

        def raced_temporary(*args, **kwargs):
            handle = original_temporary(*args, **kwargs)
            if kwargs.get("prefix") == ".snapshot-manifest.json.tmp-":
                manifest.unlink()
                manifest.symlink_to(outside)
                raced.append(manifest.lstat().st_ino)
            return handle

        monkeypatch.setattr(usgs.tempfile, "NamedTemporaryFile", raced_temporary)
        _capture(ledger, updated=43)
        with pytest.raises(USGSValidationError, match="regular"):
            ledger.seal_snapshot(output, feed="all-day")
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 1
    assert raced and manifest.is_symlink() and manifest.lstat().st_ino == raced[0]
    assert outside.read_bytes() == original_manifest and data.read_bytes() == original_data
    assert not tuple(output.glob(".*.tmp-*"))
