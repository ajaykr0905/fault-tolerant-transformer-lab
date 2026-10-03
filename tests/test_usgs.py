import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from fttl.config import ExperimentConfig, ModelConfig
from fttl.dataset import load_dataset_manifest
from fttl.recovery import verify_recovery
from fttl.usgs import (
    USGS_FEEDS,
    USGSCaptureLedger,
    USGSValidationError,
    fetch_feed,
    load_snapshot,
    parse_feature_collection,
    prepare_snapshot_dataset,
    snapshot_training_directory,
)

FIXTURE = Path(__file__).parent / "fixtures" / "usgs_feed.json"
SOURCE = "usgs-earthquakes-all-day"
FEED_URL = USGS_FEEDS["all-day"]


def fixture_bytes() -> bytes:
    return FIXTURE.read_bytes()


def test_parser_validates_stable_event_version_identity():
    versions = parse_feature_collection(fixture_bytes(), source=SOURCE)

    assert [(item.event_id, item.updated_at_ms) for item in versions] == [
        ("ci-event-alpha", 1700000000000),
        ("ci-event-beta", 1700000001000),
    ]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(type="Collection"), "FeatureCollection"),
        (lambda value: value["features"][0].pop("id"), "stable event id"),
        (
            lambda value: value["features"][0]["properties"].update(updated="now"),
            "updated timestamp",
        ),
        (
            lambda value: value["features"][0]["geometry"].update(coordinates=["x", 1]),
            "coordinates",
        ),
    ],
)
def test_parser_rejects_malformed_features(mutation, message: str):
    value = json.loads(fixture_bytes())
    mutation(value)

    with pytest.raises(USGSValidationError, match=message):
        parse_feature_collection(json.dumps(value).encode(), source=SOURCE)


def test_ledger_is_idempotent_and_keeps_new_event_versions(tmp_path: Path):
    database = tmp_path / "capture.sqlite3"
    with USGSCaptureLedger(database) as ledger:
        first = ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        duplicate = ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:01:00Z",
        )
        changed = json.loads(fixture_bytes())
        changed["features"][0]["properties"]["updated"] += 60_000
        versioned = ledger.capture(
            json.dumps(changed).encode(),
            feed="all-day",
            captured_at="2026-09-24T00:02:00Z",
        )

    assert first.inserted_versions == 2
    assert duplicate.inserted_versions == 0
    assert versioned.inserted_versions == 1


def test_same_event_identity_with_different_content_rolls_back_the_poll(tmp_path: Path):
    database = tmp_path / "capture.sqlite3"
    with USGSCaptureLedger(database) as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        conflicting = json.loads(fixture_bytes())
        conflicting["features"][0]["properties"]["mag"] = 9.9

        with pytest.raises(USGSValidationError, match="conflicting content"):
            ledger.capture(
                json.dumps(conflicting).encode(),
                feed="all-day",
                captured_at="2026-09-24T00:01:00Z",
            )

        assert ledger.connection.execute("SELECT COUNT(*) FROM polls").fetchone()[0] == 1
        assert ledger.connection.execute("SELECT COUNT(*) FROM event_versions").fetchone()[0] == 2


def test_feed_key_derives_source_and_url_instead_of_accepting_labels(tmp_path: Path):
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        with pytest.raises(ValueError, match="unknown USGS feed"):
            ledger.capture(
                fixture_bytes(),
                feed="pretend-feed",
                captured_at="2026-09-24T00:00:00Z",
            )


def test_sealed_snapshot_is_content_addressed_and_repeatable(tmp_path: Path):
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        first_path, first_manifest_path, first = ledger.seal_snapshot(
            tmp_path / "snapshot-a", feed="all-day"
        )
        second_path, _, second = ledger.seal_snapshot(
            tmp_path / "snapshot-b", feed="all-day"
        )

    loaded, records = load_snapshot(first_manifest_path)
    assert first.content_sha256 == second.content_sha256 == loaded.content_sha256
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first.event_versions == len(records) == 2
    assert records[0]["document_id"].startswith(f"{SOURCE}:ci-event-alpha:")


@pytest.mark.parametrize("corruption", ["content", "checksum", "event_id", "updated_at_ms"])
def test_sealing_rejects_corrupt_ledger_rows_without_side_effects(tmp_path: Path, corruption: str):
    output_dir = tmp_path / "snapshot"
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(fixture_bytes(), feed="all-day", captured_at="2026-09-24T00:00:00Z")
        payload_json, checksum = ledger.connection.execute(
            "SELECT payload_json, payload_sha256 FROM event_versions WHERE event_id = ?",
            ("ci-event-alpha",),
        ).fetchone()
        feature = json.loads(payload_json)
        if corruption == "content":
            feature["properties"]["mag"] = 9.9
        elif corruption == "checksum":
            checksum = "0" * 64
        elif corruption == "event_id":
            feature["id"] = "different-event"
        else:
            feature["properties"]["updated"] += 1
        payload_json = json.dumps(
            feature, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        if corruption in {"event_id", "updated_at_ms"}:
            checksum = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        ledger.connection.execute(
            "UPDATE event_versions SET payload_json = ?, payload_sha256 = ? WHERE event_id = ?",
            (payload_json, checksum, "ci-event-alpha"),
        )
        ledger.connection.commit()

        message = "checksum" if corruption in {"content", "checksum"} else "identity"
        with pytest.raises(USGSValidationError, match=message):
            ledger.seal_snapshot(output_dir, feed="all-day")

        assert not output_dir.exists()
        assert ledger.connection.execute("SELECT COUNT(*) FROM sealed_snapshots").fetchone()[0] == 0
        assert ledger.connection.execute("SELECT COUNT(*) FROM polls").fetchone()[0] == 1


def test_snapshot_bytes_are_independent_of_feed_record_order(tmp_path: Path):
    reversed_payload = json.loads(fixture_bytes())
    reversed_payload["features"].reverse()
    paths = []
    manifests = []
    for name, payload in (
        ("normal", fixture_bytes()),
        ("reversed", json.dumps(reversed_payload).encode()),
    ):
        with USGSCaptureLedger(tmp_path / f"{name}.sqlite3") as ledger:
            ledger.capture(
                payload,
                feed="all-day",
                captured_at="2026-09-24T00:00:00Z",
            )
            data_path, _, manifest = ledger.seal_snapshot(
                tmp_path / name, feed="all-day"
            )
            paths.append(data_path)
            manifests.append(manifest)
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert manifests[0].snapshot_fingerprint == manifests[1].snapshot_fingerprint


def test_snapshot_rejects_one_changed_byte(tmp_path: Path):
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        data_path, manifest_path, _ = ledger.seal_snapshot(
            tmp_path / "snapshot", feed="all-day"
        )
    content = data_path.read_bytes()
    data_path.write_bytes(content[:-2] + b"x\n")

    with pytest.raises(USGSValidationError, match="checksum"):
        load_snapshot(manifest_path)


def test_snapshot_loader_rejects_path_substitution_before_reading(tmp_path: Path):
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        _, manifest_path, _ = ledger.seal_snapshot(tmp_path / "snapshot", feed="all-day")
    raw = json.loads(manifest_path.read_text())
    raw["data_path"] = "../outside.jsonl"
    identity = dict(raw)
    identity.pop("snapshot_fingerprint")
    raw["snapshot_fingerprint"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(USGSValidationError, match="content-addressed"):
        load_snapshot(manifest_path)


def test_parser_rejects_non_finite_coordinates_and_large_timestamps():
    non_finite = fixture_bytes().decode().replace("-121.5", "NaN", 1)
    with pytest.raises(USGSValidationError, match="non-finite"):
        parse_feature_collection(non_finite.encode(), source=SOURCE)
    too_large = json.loads(fixture_bytes())
    too_large["features"][0]["properties"]["updated"] = 2**63
    with pytest.raises(USGSValidationError, match="updated timestamp"):
        parse_feature_collection(json.dumps(too_large).encode(), source=SOURCE)


def test_fetch_is_bounded_retries_and_never_switches_source():
    attempts = []
    sleeps = []

    def opener(_request, _timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary test failure")
        return io.BytesIO(fixture_bytes())

    payload = fetch_feed(
        FEED_URL,
        opener=opener,
        sleeper=sleeps.append,
        retries=3,
    )

    assert payload == fixture_bytes()
    assert len(attempts) == 3
    assert sleeps == [1.0, 2.0]
    with pytest.raises(ValueError, match="allowlisted"):
        fetch_feed("https://example.invalid/feed", opener=opener)


def test_fetch_rejects_an_oversized_response():
    def opener(_request, _timeout):
        return io.BytesIO(b"12345")

    with pytest.raises(USGSValidationError, match="byte limit"):
        fetch_feed(FEED_URL, opener=opener, max_response_bytes=4)


def test_sealed_feed_enters_the_same_training_and_recovery_pipeline(tmp_path: Path):
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        _, snapshot_manifest, _ = ledger.seal_snapshot(
            tmp_path / "snapshot", feed="all-day"
        )
    training_dir = tmp_path / "training"
    prepared = prepare_snapshot_dataset(snapshot_manifest, training_dir)
    loaded = load_dataset_manifest(training_dir)
    assert loaded.dataset_fingerprint == prepared.dataset_fingerprint
    assert loaded.source["artifact_kind"] == "sealed-jsonl"

    config = ExperimentConfig(
        model=ModelConfig(
            vocab_size=257,
            block_size=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dropout=0.2,
        ),
        seed=31,
        steps=3,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )
    report = verify_recovery(
        config,
        training_dir / "manifest.json",
        tmp_path / "evidence",
        failure_point="after-optimizer",
        restarts=1,
    )
    assert report.exact_equality


def test_prepared_snapshot_is_idempotent_and_new_versions_get_new_paths(
    tmp_path: Path,
):
    snapshot_root = tmp_path / "snapshot"
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(
            fixture_bytes(),
            feed="all-day",
            captured_at="2026-09-24T00:00:00Z",
        )
        _, first_manifest_path, first_snapshot = ledger.seal_snapshot(
            snapshot_root,
            feed="all-day",
        )
        first_training = snapshot_training_directory(snapshot_root, first_snapshot)
        first = prepare_snapshot_dataset(first_manifest_path, first_training)
        repeated = prepare_snapshot_dataset(first_manifest_path, first_training)

        changed = json.loads(fixture_bytes())
        changed["features"][0]["properties"]["updated"] += 60_000
        ledger.capture(
            json.dumps(changed).encode(),
            feed="all-day",
            captured_at="2026-09-24T00:01:00Z",
        )
        _, second_manifest_path, second_snapshot = ledger.seal_snapshot(
            snapshot_root,
            feed="all-day",
        )
        second_training = snapshot_training_directory(snapshot_root, second_snapshot)
        second = prepare_snapshot_dataset(second_manifest_path, second_training)

    assert repeated.dataset_fingerprint == first.dataset_fingerprint
    assert second_training != first_training
    assert second.dataset_fingerprint != first.dataset_fingerprint
    assert load_dataset_manifest(first_training).dataset_fingerprint == first.dataset_fingerprint
    assert load_dataset_manifest(second_training).dataset_fingerprint == second.dataset_fingerprint
