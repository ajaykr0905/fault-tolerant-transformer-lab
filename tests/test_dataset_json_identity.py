import gzip
import hashlib
import json

import pytest
from test_dataset import _fixture_source
from test_real_data_recovery import FIXTURE, prepared_fixture

from fttl import dataset


def _duplicate_row(content, kind, field):
    value = json.loads(content)[field]
    key = field
    if kind == "escaped":
        key = field[:-1] + "\\u%04x" % ord(field[-1])
    if kind == "nested":
        prefix = '"extra":{"identity":"first","identity":"last"},'
    else:
        first = value if kind == "identical" else "conflicting-first-value"
        prefix = '"' + key + '":' + json.dumps(first) + ","
    return ("{" + prefix).encode() + content[1:]


@pytest.mark.parametrize("kind", ["conflicting", "identical", "escaped", "nested"])
def test_manifest_rejects_duplicate_json_identity_before_verification(tmp_path, kind):
    path = prepared_fixture(tmp_path)
    clean = path.read_bytes()
    malformed = _duplicate_row(clean, kind, "dataset_fingerprint")
    with pytest.raises(dataset.DatasetValidationError, match="manifest.*duplicate.*key"):
        dataset.PreparedDatasetSnapshot(malformed, (path.parent / "documents.jsonl").read_bytes())


def test_manifest_rejects_duplicate_nested_source_revision(tmp_path):
    path = prepared_fixture(tmp_path)
    clean = path.read_bytes()
    payload = json.loads(clean)
    original = json.dumps(payload["source"]["revision"]).encode()
    field = b'"revision": ' + original
    changed = clean.replace(field, b'"revision":"first-value",' + field)
    assert changed != clean
    with pytest.raises(dataset.DatasetValidationError, match="manifest.*duplicate.*revision"):
        dataset.PreparedDatasetSnapshot(changed, (path.parent / "documents.jsonl").read_bytes())


@pytest.mark.parametrize("kind", ["conflicting", "identical", "escaped", "nested"])
@pytest.mark.parametrize("field", ["text", "source_id"])
def test_prepared_jsonl_rejects_duplicates_even_with_exact_byte_hashes(tmp_path, kind, field):
    path = prepared_fixture(tmp_path)
    payload = json.loads(path.read_bytes())
    lines = (path.parent / "documents.jsonl").read_bytes().splitlines(keepends=True)
    lines[0] = _duplicate_row(lines[0], kind, field)
    changed = b"".join(lines)
    payload["documents"]["byte_length"] = len(changed)
    payload["documents"]["sha256"] = hashlib.sha256(changed).hexdigest()
    payload.pop("dataset_fingerprint")
    payload["dataset_fingerprint"] = dataset._sha256_json(payload)
    with pytest.raises(dataset.DatasetValidationError, match="prepared document 1.*duplicate.*key"):
        dataset.PreparedDatasetSnapshot(json.dumps(payload).encode(), changed)


@pytest.mark.parametrize("kind", ["conflicting", "identical", "escaped", "nested"])
@pytest.mark.parametrize("field", ["text", "id"])
def test_verified_gzip_source_rejects_duplicates_before_prepared_publication(tmp_path, kind, field):
    lines = FIXTURE.read_bytes().splitlines(keepends=True)
    lines[0] = _duplicate_row(lines[0], kind, field)
    archive = gzip.compress(b"".join(lines), mtime=0)

    def fetch(_url, destination):
        destination.write_bytes(archive)

    with pytest.raises(dataset.DatasetValidationError, match="record 1.*duplicate.*key"):
        dataset.prepare_peps(
            tmp_path / "cache",
            tmp_path / "prepared",
            downloader=fetch,
            source=_fixture_source(archive),
        )
    assert not (tmp_path / "prepared").exists()


def test_duplicate_free_key_order_keeps_documents_and_canonical_fingerprints(tmp_path):
    path = prepared_fixture(tmp_path)
    original_bytes = (path.parent / "documents.jsonl").read_bytes()
    payload = json.loads(path.read_bytes())
    baseline = dataset.PreparedDatasetSnapshot(path.read_bytes(), original_bytes)
    reordered_manifest = dict(reversed(list(payload.items())))
    reordered_manifest["source"] = dict(reversed(list(payload["source"].items())))
    reordered = dataset.PreparedDatasetSnapshot(
        json.dumps(reordered_manifest).encode(), original_bytes
    )
    assert reordered.documents == baseline.documents
    assert reordered.manifest.to_dict() == baseline.manifest.to_dict()
    assert reordered.manifest.fingerprint() == baseline.manifest.fingerprint()
    assert (
        dataset.load_dataset_snapshot(path).manifest.fingerprint()
        == baseline.manifest.fingerprint()
    )


def test_duplicate_source_fields_in_nested_metadata_are_rejected(tmp_path):
    raw = b'{"text":"Public fixture text","metadata":{"id":"first","id":"last"}}\n'
    archive = gzip.compress(raw, mtime=0)
    path = tmp_path / "public-fixture.jsonl.gz"
    path.write_bytes(archive)
    with pytest.raises(dataset.DatasetValidationError, match="record 1.*duplicate.*id"):
        tuple(dataset._read_gzip_records(path))
