from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from fttl.dataset import (
    DatasetManifestV1,
    DatasetSource,
    load_dataset_manifest,
    prepare_document_records,
)

USGS_ATTRIBUTION = "U.S. Geological Survey (USGS)"
USGS_LICENSE_NOTE = (
    "USGS-authored information is generally public domain; attribution is requested. "
    "Confirm third-party material independently."
)
USGS_FEEDS = {
    "all-hour": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson",
    "all-day": "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson",
}
DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
SQLITE_MAX_INTEGER = 2**63 - 1


class USGSValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EventVersion:
    source: str
    event_id: str
    updated_at_ms: int
    payload: dict[str, Any]

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class CaptureResult:
    poll_id: int
    response_sha256: str
    received_events: int
    inserted_versions: int


@dataclass(frozen=True)
class SealedSnapshotV1:
    schema_version: int
    feed: str
    source: str
    feed_url: str
    attribution: str
    license_note: str
    content_sha256: str
    byte_length: int
    event_versions: int
    data_path: str
    snapshot_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OpenUrl = Callable[[urllib.request.Request, float], BinaryIO]
Sleep = Callable[[float], None]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _reject_json_constant(value: str) -> None:
    raise USGSValidationError(f"non-finite JSON number {value!r} is not allowed")


def _parse_finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise USGSValidationError(f"non-finite JSON number {value!r} is not allowed")
    return number


def _feed_identity(feed: str) -> tuple[str, str]:
    try:
        feed_url = USGS_FEEDS[feed]
    except KeyError as error:
        raise ValueError(f"unknown USGS feed {feed!r}") from error
    return f"usgs-earthquakes-{feed}", feed_url


def _snapshot_fingerprint(value: Mapping[str, Any]) -> str:
    identity = dict(value)
    identity.pop("snapshot_fingerprint", None)
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def parse_feature_collection(payload: bytes, *, source: str) -> tuple[EventVersion, ...]:
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise USGSValidationError("USGS response is not valid UTF-8 JSON") from error
    if not isinstance(decoded, dict) or decoded.get("type") != "FeatureCollection":
        raise USGSValidationError("USGS response must be a GeoJSON FeatureCollection")
    features = decoded.get("features")
    if not isinstance(features, list):
        raise USGSValidationError("USGS FeatureCollection.features must be a list")

    versions: list[EventVersion] = []
    seen: set[tuple[str, int]] = set()
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise USGSValidationError(f"feature {index} is not a GeoJSON Feature")
        event_id = feature.get("id")
        properties = feature.get("properties")
        geometry = feature.get("geometry")
        if not isinstance(event_id, str) or not event_id.strip():
            raise USGSValidationError(f"feature {index} has no stable event id")
        if not isinstance(properties, dict):
            raise USGSValidationError(f"feature {event_id} has invalid properties")
        updated = properties.get("updated")
        if (
            isinstance(updated, bool)
            or not isinstance(updated, int)
            or not 0 <= updated <= SQLITE_MAX_INTEGER
        ):
            raise USGSValidationError(f"feature {event_id} has invalid updated timestamp")
        if not isinstance(geometry, dict) or geometry.get("type") != "Point":
            raise USGSValidationError(f"feature {event_id} must have Point geometry")
        coordinates = geometry.get("coordinates")
        if (
            not isinstance(coordinates, list)
            or len(coordinates) < 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in coordinates
            )
        ):
            raise USGSValidationError(f"feature {event_id} has invalid coordinates")
        identity = (event_id, updated)
        if identity in seen:
            raise USGSValidationError(
                f"response repeats event version ({event_id}, {updated})"
            )
        seen.add(identity)
        versions.append(
            EventVersion(
                source=source,
                event_id=event_id,
                updated_at_ms=updated,
                payload=feature,
            )
        )
    return tuple(sorted(versions, key=lambda item: (item.event_id, item.updated_at_ms)))


def _default_open(request: urllib.request.Request, timeout: float) -> BinaryIO:
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - fixed allowlisted URLs


def fetch_feed(
    feed_url: str,
    *,
    opener: OpenUrl = _default_open,
    sleeper: Sleep = time.sleep,
    timeout_seconds: float = 15.0,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
    retries: int = 3,
) -> bytes:
    if feed_url not in USGS_FEEDS.values():
        raise ValueError("feed_url must be one of the allowlisted USGS feeds")
    if retries < 1 or retries > 5:
        raise ValueError("retries must be between 1 and 5")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        raise ValueError("timeout_seconds must be in (0, 60]")
    if max_response_bytes < 1:
        raise ValueError("max_response_bytes must be positive")

    request = urllib.request.Request(
        feed_url,
        headers={
            "Accept": "application/geo+json, application/json",
            "User-Agent": "fault-tolerant-transformer-lab/0.2 (+public-research-demo)",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with opener(request, timeout_seconds) as response:
                payload = response.read(max_response_bytes + 1)
            if len(payload) > max_response_bytes:
                raise USGSValidationError("USGS response exceeds the configured byte limit")
            return payload
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            if attempt + 1 < retries:
                sleeper(float(2**attempt))
    assert last_error is not None
    raise RuntimeError(f"USGS fetch failed after {retries} attempts") from last_error


class USGSCaptureLedger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version not in (0, 1):
            raise RuntimeError(f"unsupported USGS ledger schema version {user_version}")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS polls (
                id INTEGER PRIMARY KEY,
                source TEXT NOT NULL,
                feed_url TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                response_sha256 TEXT NOT NULL,
                received_events INTEGER NOT NULL,
                inserted_versions INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS event_versions (
                feed TEXT NOT NULL,
                source TEXT NOT NULL,
                event_id TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                first_poll_id INTEGER NOT NULL REFERENCES polls(id),
                PRIMARY KEY (source, event_id, updated_at_ms)
            );
            CREATE TABLE IF NOT EXISTS sealed_snapshots (
                content_sha256 TEXT PRIMARY KEY,
                snapshot_fingerprint TEXT NOT NULL,
                source TEXT NOT NULL,
                feed_url TEXT NOT NULL,
                byte_length INTEGER NOT NULL,
                event_versions INTEGER NOT NULL,
                data_path TEXT NOT NULL
            );
            """
        )
        self.connection.execute("PRAGMA user_version=1")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "USGSCaptureLedger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def capture(
        self,
        payload: bytes,
        *,
        feed: str,
        captured_at: str,
    ) -> CaptureResult:
        source, feed_url = _feed_identity(feed)
        versions = parse_feature_collection(payload, source=source)
        response_sha256 = hashlib.sha256(payload).hexdigest()
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO polls (
                    source, feed_url, captured_at, response_sha256,
                    received_events, inserted_versions
                ) VALUES (?, ?, ?, ?, ?, 0)
                """,
                (source, feed_url, captured_at, response_sha256, len(versions)),
            )
            poll_id = int(cursor.lastrowid)
            inserted = 0
            for version in versions:
                payload_json = _canonical_json(version.payload)
                payload_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                existing = self.connection.execute(
                    """
                    SELECT payload_sha256 FROM event_versions
                    WHERE source = ? AND event_id = ? AND updated_at_ms = ?
                    """,
                    (version.source, version.event_id, version.updated_at_ms),
                ).fetchone()
                if existing is not None:
                    if existing[0] != payload_sha256:
                        raise USGSValidationError(
                            "the same USGS event version has conflicting content"
                        )
                    continue
                self.connection.execute(
                    """
                    INSERT INTO event_versions (
                        feed, source, event_id, updated_at_ms, payload_json,
                        payload_sha256, first_poll_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feed,
                        version.source,
                        version.event_id,
                        version.updated_at_ms,
                        payload_json,
                        payload_sha256,
                        poll_id,
                    ),
                )
                inserted += 1
            self.connection.execute(
                "UPDATE polls SET inserted_versions = ? WHERE id = ?",
                (inserted, poll_id),
            )
        return CaptureResult(
            poll_id=poll_id,
            response_sha256=response_sha256,
            received_events=len(versions),
            inserted_versions=inserted,
        )

    def seal_snapshot(
        self,
        output_dir: Path,
        *,
        feed: str,
    ) -> tuple[Path, Path, SealedSnapshotV1]:
        source, feed_url = _feed_identity(feed)
        rows = self.connection.execute(
            """
            SELECT event_id, updated_at_ms, payload_json, payload_sha256
            FROM event_versions
            WHERE source = ? AND feed = ?
            ORDER BY event_id, updated_at_ms
            """,
            (source, feed),
        ).fetchall()
        if not rows:
            raise ValueError("cannot seal an empty USGS capture ledger")
        record_values = []
        for event_id, updated_at_ms, payload_json, payload_sha256 in rows:
            if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != payload_sha256:
                raise USGSValidationError(
                    "USGS ledger payload checksum does not match stored bytes"
                )
            try:
                feature = json.loads(
                    payload_json,
                    parse_constant=_reject_json_constant,
                    parse_float=_parse_finite_json_float,
                )
            except json.JSONDecodeError as error:
                raise USGSValidationError("USGS ledger payload is not valid JSON") from error
            version = parse_feature_collection(
                _canonical_json({"type": "FeatureCollection", "features": [feature]}).encode(),
                source=source,
            )[0]
            if version.event_id != event_id or version.updated_at_ms != updated_at_ms:
                raise USGSValidationError("USGS ledger payload identity does not match stored key")
            record_values.append(
                {
                    "document_id": f"{source}:{event_id}:{updated_at_ms}",
                    "event_id": event_id,
                    "source": source,
                    "text": payload_json,
                    "updated_at_ms": updated_at_ms,
                }
            )
        content = (
            "\n".join(_canonical_json(record) for record in record_values) + "\n"
        ).encode("utf-8")
        if len(content) > MAX_SNAPSHOT_BYTES:
            raise USGSValidationError("USGS snapshot exceeds the configured byte limit")
        content_sha256 = hashlib.sha256(content).hexdigest()
        output_dir.mkdir(parents=True, exist_ok=True)
        data_name = f"usgs-{content_sha256}.jsonl"
        data_path = output_dir / data_name
        _write_content_addressed(data_path, content)

        manifest_value: dict[str, Any] = {
            "schema_version": 1,
            "feed": feed,
            "source": source,
            "feed_url": feed_url,
            "attribution": USGS_ATTRIBUTION,
            "license_note": USGS_LICENSE_NOTE,
            "content_sha256": content_sha256,
            "byte_length": len(content),
            "event_versions": len(record_values),
            "data_path": data_name,
        }
        manifest_value["snapshot_fingerprint"] = _snapshot_fingerprint(manifest_value)
        manifest = SealedSnapshotV1(**manifest_value)
        manifest_path = output_dir / "snapshot-manifest.json"
        encoded_manifest = (
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _write_atomic(manifest_path, encoded_manifest)

        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO sealed_snapshots (
                    content_sha256, snapshot_fingerprint, source, feed_url,
                    byte_length, event_versions, data_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_sha256,
                    manifest.snapshot_fingerprint,
                    source,
                    feed_url,
                    len(content),
                    len(record_values),
                    str(data_path),
                ),
            )
        return data_path, manifest_path, manifest


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_content_addressed(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise USGSValidationError("content-addressed snapshot path has different bytes")
        return
    _write_atomic(path, content)


def _validated_snapshot_manifest(manifest_path: Path) -> SealedSnapshotV1:
    required = {
        "schema_version",
        "feed",
        "source",
        "feed_url",
        "attribution",
        "license_note",
        "content_sha256",
        "byte_length",
        "event_versions",
        "data_path",
        "snapshot_fingerprint",
    }
    try:
        raw = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise USGSValidationError("USGS snapshot manifest is not valid UTF-8 JSON") from error
    if not isinstance(raw, dict) or set(raw) != required:
        raise USGSValidationError("USGS snapshot manifest fields do not match schema")
    if isinstance(raw["schema_version"], bool) or raw["schema_version"] != 1:
        raise USGSValidationError("unsupported USGS snapshot schema")
    string_fields = required - {
        "schema_version",
        "byte_length",
        "event_versions",
    }
    if any(not isinstance(raw[field], str) or not raw[field] for field in string_fields):
        raise USGSValidationError("USGS snapshot manifest strings are invalid")
    for field in ("byte_length", "event_versions"):
        if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 1:
            raise USGSValidationError(f"USGS snapshot {field} must be a positive integer")
    if raw["byte_length"] > MAX_SNAPSHOT_BYTES:
        raise USGSValidationError("USGS snapshot exceeds the configured byte limit")
    source, feed_url = _feed_identity(raw["feed"])
    if raw["source"] != source or raw["feed_url"] != feed_url:
        raise USGSValidationError("USGS snapshot feed identity is inconsistent")
    if raw["attribution"] != USGS_ATTRIBUTION or raw["license_note"] != USGS_LICENSE_NOTE:
        raise USGSValidationError("USGS snapshot attribution metadata is inconsistent")
    if len(raw["content_sha256"]) != 64 or any(
        character not in "0123456789abcdef" for character in raw["content_sha256"]
    ):
        raise USGSValidationError("USGS snapshot content SHA-256 is invalid")
    expected_name = f"usgs-{raw['content_sha256']}.jsonl"
    if raw["data_path"] != expected_name or Path(raw["data_path"]).name != raw["data_path"]:
        raise USGSValidationError("USGS snapshot data path is not content-addressed")
    if raw["snapshot_fingerprint"] != _snapshot_fingerprint(raw):
        raise USGSValidationError("USGS snapshot fingerprint does not match metadata")
    return SealedSnapshotV1(**raw)


def load_snapshot(
    manifest_path: Path,
) -> tuple[SealedSnapshotV1, tuple[Mapping[str, Any], ...]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = _validated_snapshot_manifest(manifest_path)
    data_path = (manifest_path.parent / manifest.data_path).resolve()
    if data_path.parent != manifest_path.parent:
        raise USGSValidationError("USGS snapshot data path escapes its directory")
    try:
        actual_size = data_path.stat().st_size
    except OSError as error:
        raise USGSValidationError("USGS snapshot data file is unavailable") from error
    if actual_size != manifest.byte_length:
        raise USGSValidationError("USGS snapshot byte length does not match manifest")
    content = data_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest.content_sha256:
        raise USGSValidationError("USGS snapshot checksum does not match manifest")
    records: list[Mapping[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    try:
        lines = content.decode("utf-8").splitlines()
        for line in lines:
            record = json.loads(
                line,
                parse_constant=_reject_json_constant,
                parse_float=_parse_finite_json_float,
            )
            if not isinstance(record, dict) or set(record) != {
                "document_id",
                "event_id",
                "source",
                "text",
                "updated_at_ms",
            }:
                raise USGSValidationError("USGS snapshot record fields do not match schema")
            if not all(
                isinstance(record[field], str) and record[field]
                for field in ("document_id", "event_id", "source", "text")
            ):
                raise USGSValidationError("USGS snapshot record strings are invalid")
            updated = record["updated_at_ms"]
            if isinstance(updated, bool) or not isinstance(updated, int):
                raise USGSValidationError("USGS snapshot update time is invalid")
            identity = (record["event_id"], updated)
            expected_id = f"{manifest.source}:{record['event_id']}:{updated}"
            if record["source"] != manifest.source or record["document_id"] != expected_id:
                raise USGSValidationError("USGS snapshot record identity is inconsistent")
            if identity in identities:
                raise USGSValidationError("USGS snapshot repeats an event version")
            identities.add(identity)
            feature = json.loads(
                record["text"],
                parse_constant=_reject_json_constant,
                parse_float=_parse_finite_json_float,
            )
            validated = parse_feature_collection(
                _canonical_json({"type": "FeatureCollection", "features": [feature]}).encode(),
                source=manifest.source,
            )[0]
            if validated.event_id != identity[0] or validated.updated_at_ms != identity[1]:
                raise USGSValidationError("USGS snapshot payload identity is inconsistent")
            if line != _canonical_json(record):
                raise USGSValidationError("USGS snapshot record is not canonical JSON")
            records.append(record)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise USGSValidationError("USGS snapshot records are not valid UTF-8 JSON") from error
    if len(records) != manifest.event_versions:
        raise USGSValidationError("USGS snapshot record count does not match manifest")
    if records != sorted(
        records, key=lambda item: (item["event_id"], item["updated_at_ms"])
    ):
        raise USGSValidationError("USGS snapshot records are not deterministically ordered")
    return manifest, tuple(records)


def prepare_snapshot_dataset(
    snapshot_manifest_path: Path,
    output_dir: Path,
) -> DatasetManifestV1:
    """Convert a sealed live-feed capture into immutable offline training input."""

    snapshot, records = load_snapshot(snapshot_manifest_path)
    snapshot_manifest_path = Path(snapshot_manifest_path).resolve()
    data_path = snapshot_manifest_path.parent / snapshot.data_path
    source = DatasetSource(
        repository=f"usgs/earthquakes/{snapshot.feed}",
        revision=snapshot.snapshot_fingerprint,
        path=snapshot.data_path,
        compressed_sha256=snapshot.content_sha256,
        expected_document_count=snapshot.event_versions,
        license="Generally public domain when authored by USGS; attribution requested",
        license_limitations=(
            "Third-party material can have separate rights and must be checked independently.",
            "This snapshot is an ingestion/replay fixture, not earthquake prediction data.",
        ),
        artifact_kind="sealed-jsonl",
        url_override=snapshot.feed_url,
    )
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.is_file():
        prepared = load_dataset_manifest(existing_manifest)
        expected_source = {
            "repository": source.repository,
            "revision": source.revision,
            "path": source.path,
            "url": source.url,
            "artifact_kind": source.artifact_kind,
            "artifact_sha256": source.compressed_sha256,
            "compressed_sha256": source.compressed_sha256,
        }
        if any(
            prepared.source.get(field) != value
            for field, value in expected_source.items()
        ) or prepared.counts.get("documents") != source.expected_document_count:
            raise USGSValidationError(
                "existing prepared snapshot does not match the sealed source"
            )
        return prepared
    return prepare_document_records(
        records,
        output_dir,
        source_artifact=data_path,
        source=source,
    )


def snapshot_training_directory(
    snapshot_root: Path,
    snapshot: SealedSnapshotV1,
) -> Path:
    """Return the immutable prepared-data directory for one sealed snapshot."""

    return Path(snapshot_root) / "training" / snapshot.snapshot_fingerprint
