import hashlib
import json

import pytest

from fttl.usgs import (
    USGSCaptureLedger,
    USGSValidationError,
    load_snapshot,
    parse_feature_collection,
)

SOURCE = "usgs-earthquakes-all-day"


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _feature(updated=42):
    return {
        "type": "Feature",
        "id": "independent-json-event",
        "properties": {"updated": updated},
        "geometry": {"type": "Point", "coordinates": [1.0, 2.0, 3.0]},
    }


def _duplicate_feature(feature, field):
    value = _encode(feature)
    replacements = {
        "updated": (
            f'"updated":{feature["properties"]["updated"]}'.encode(),
            f'"updated":41,"updated":{feature["properties"]["updated"]}'.encode(),
        ),
        "id": (
            b'"id":"independent-json-event"',
            b'"id":"discarded-event","id":"independent-json-event"',
        ),
        "geometry": (
            b'"coordinates":[1.0,2.0,3.0]',
            b'"coordinates":[0,0,0],"coordinates":[1.0,2.0,3.0]',
        ),
    }
    old, new = replacements[field]
    assert old in value
    return value.replace(old, new, 1)


def _collection(feature):
    return b'{"type":"FeatureCollection","features":[' + feature + b"]}"


@pytest.mark.parametrize("field", ["updated", "id", "geometry"])
def test_parser_rejects_duplicate_identity_or_geometry_before_value_selection(field):
    with pytest.raises(USGSValidationError, match="duplicate"):
        parse_feature_collection(_collection(_duplicate_feature(_feature(), field)), source=SOURCE)


def test_parser_rejects_duplicate_top_level_feature_lists():
    payload = (
        b'{"type":"FeatureCollection","features":[],"features":[' + _encode(_feature()) + b"]}"
    )
    with pytest.raises(USGSValidationError, match="duplicate"):
        parse_feature_collection(payload, source=SOURCE)


def test_valid_json_retains_exact_identity_geometry_and_nested_values():
    feature = _feature()
    feature["properties"]["diagnostic"] = {"valid": [0, False, None, "text", 1.5]}
    versions = parse_feature_collection(_collection(_encode(feature)), source=SOURCE)
    assert len(versions) == 1
    assert versions[0].event_id == feature["id"] and versions[0].updated_at_ms == 42
    assert versions[0].payload == feature


@pytest.mark.parametrize("field", ["updated", "id", "geometry"])
def test_rejected_capture_preserves_all_poll_and_event_history(tmp_path, field):
    with USGSCaptureLedger(tmp_path / "ledger.sqlite3") as ledger:
        for updated in (42, 43):
            ledger.capture(
                _collection(_encode(_feature(updated))),
                feed="all-day",
                captured_at="2026-10-06T00:00:00Z",
            )
        ledger.seal_snapshot(tmp_path / "existing", feed="all-day")
        before = {
            table: ledger.connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("polls", "event_versions", "sealed_snapshots")
        }
        with pytest.raises(USGSValidationError, match="duplicate"):
            ledger.capture(
                _collection(_duplicate_feature(_feature(44), field)),
                feed="all-day",
                captured_at="2026-10-06T00:01:00Z",
            )
        after = {
            table: ledger.connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in before
        }
        assert before == after
        assert not ledger.connection.in_transaction


@pytest.mark.parametrize("field", ["updated", "id", "geometry"])
def test_sealing_rejects_rehashed_ambiguous_ledger_payload_before_publication(tmp_path, field):
    with USGSCaptureLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.capture(
            _collection(_encode(_feature())), feed="all-day", captured_at="2026-10-06T00:00:00Z"
        )
        ambiguous = _duplicate_feature(_feature(), field)
        ledger.connection.execute(
            "UPDATE event_versions SET payload_json=?, payload_sha256=?",
            (ambiguous.decode(), hashlib.sha256(ambiguous).hexdigest()),
        )
        ledger.connection.commit()
        with pytest.raises(USGSValidationError, match="duplicate"):
            ledger.seal_snapshot(tmp_path / "new-output", feed="all-day")
        assert not (tmp_path / "new-output").exists()
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 0


def _snapshot(tmp_path):
    with USGSCaptureLedger(tmp_path / "ledger.sqlite3") as ledger:
        ledger.capture(
            _collection(_encode(_feature())), feed="all-day", captured_at="2026-10-06T00:00:00Z"
        )
        return ledger.seal_snapshot(tmp_path / "snapshot", feed="all-day")


def _rebind_snapshot(manifest_path, content):
    manifest = json.loads(manifest_path.read_text())
    manifest["content_sha256"] = hashlib.sha256(content).hexdigest()
    manifest["byte_length"] = len(content)
    manifest["data_path"] = f"usgs-{manifest['content_sha256']}.jsonl"
    (manifest_path.parent / manifest["data_path"]).write_bytes(content)
    manifest.pop("snapshot_fingerprint")
    manifest["snapshot_fingerprint"] = hashlib.sha256(_encode(manifest)).hexdigest()
    manifest_path.write_bytes(_encode(manifest))


def test_snapshot_manifest_rejects_duplicate_schema_field_even_if_last_value_matches(tmp_path):
    _, manifest, _ = _snapshot(tmp_path)
    value = manifest.read_text().replace(
        '"schema_version": 1', '"schema_version": 0, "schema_version": 1', 1
    )
    manifest.write_text(value)
    with pytest.raises(USGSValidationError, match="duplicate"):
        load_snapshot(manifest)


@pytest.mark.parametrize("field", ["updated", "id", "geometry"])
def test_snapshot_embedded_feature_rejects_duplicate_keys_with_valid_outer_digests(tmp_path, field):
    data, manifest, _ = _snapshot(tmp_path)
    record = json.loads(data.read_text())
    record["text"] = _duplicate_feature(_feature(), field).decode()
    _rebind_snapshot(manifest, _encode(record) + b"\n")
    with pytest.raises(USGSValidationError, match="duplicate"):
        load_snapshot(manifest)


def test_snapshot_record_duplicate_identity_is_rejected_explicitly(tmp_path):
    data, manifest, _ = _snapshot(tmp_path)
    content = data.read_bytes().replace(
        b'"event_id":"independent-json-event"',
        b'"event_id":"discarded-event","event_id":"independent-json-event"',
        1,
    )
    _rebind_snapshot(manifest, content)
    with pytest.raises(USGSValidationError, match="duplicate"):
        load_snapshot(manifest)
