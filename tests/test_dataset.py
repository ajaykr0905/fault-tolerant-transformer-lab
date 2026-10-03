from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path

import pytest

import fttl.dataset as dataset_module
from fttl.dataset import (
    PEP_COMPRESSED_SHA256,
    PEP_DOCUMENT_COUNT,
    PEP_PATH,
    PEP_REPOSITORY,
    PEP_REVISION,
    TOKENIZER_EOD_ID,
    TOKENIZER_FINGERPRINT,
    TOKENIZER_VOCAB_SIZE,
    DatasetSource,
    DatasetValidationError,
    Utf8ByteTokenizer,
    load_dataset_manifest,
    load_prepared_documents,
    prepare_document_records,
    prepare_peps,
)

FIXTURE = Path(__file__).parent / "fixtures" / "public_domain_peps_fixture.jsonl"


def _fixture_archive() -> bytes:
    return gzip.compress(FIXTURE.read_bytes(), mtime=0)


def _fixture_source(archive: bytes) -> DatasetSource:
    return DatasetSource(
        repository="fttl://tests/public-domain-pep-fixture",
        revision="fixture-v1",
        path="public_domain_peps_fixture.jsonl.gz",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=("Synthetic PEP-style text; not part of the Common Pile.",),
    )


def _prepare_fixture(tmp_path: Path, *, source: DatasetSource | None = None):
    archive = _fixture_archive()
    actual_source = source or _fixture_source(archive)

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(archive)

    output = tmp_path / "prepared"
    manifest = prepare_peps(tmp_path / "cache", output, downloader=downloader, source=actual_source)
    return output, manifest


def test_official_source_is_exactly_pinned():
    assert PEP_REPOSITORY == "common-pile/python_enhancement_proposals"
    assert PEP_REVISION == "f932757e3eba16475c893e1418918c77f14a790d"
    assert PEP_PATH == "raw/documents/00000_peps.jsonl.gz"
    assert PEP_COMPRESSED_SHA256 == "659be543c5e79ba776467fb694fbc9efaa350ed022b2e26139b966bae1735b0c"
    assert PEP_DOCUMENT_COUNT == 656


def test_utf8_byte_tokenizer_has_a_fixed_complete_byte_vocabulary():
    tokenizer = Utf8ByteTokenizer()
    text = "Aé☄"
    assert tokenizer.encode(text) == (*text.encode("utf-8"), TOKENIZER_EOD_ID)
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocab_size == TOKENIZER_VOCAB_SIZE == 257
    assert tokenizer.fingerprint() == TOKENIZER_FINGERPRINT


def test_preparation_is_deterministic_and_splits_never_overlap(tmp_path: Path):
    output, first = _prepare_fixture(tmp_path / "first")
    _, second = _prepare_fixture(tmp_path / "second")
    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.documents["sha256"] == second.documents["sha256"]
    split_ids = [set(first.splits[name]["document_ids"]) for name in ("train", "validation", "test")]
    assert [len(ids) for ids in split_ids] == [2, 2, 2]
    assert not split_ids[0].intersection(split_ids[1] | split_ids[2])
    assert not split_ids[1].intersection(split_ids[2])
    assert len(load_prepared_documents(output, split=None)) == 6
    assert first.fingerprint() == first.dataset_fingerprint


def test_record_order_does_not_change_ids_or_prepared_bytes(tmp_path: Path):
    records = FIXTURE.read_text(encoding="utf-8").splitlines()
    reversed_archive = gzip.compress(("\n".join(reversed(records)) + "\n").encode(), mtime=0)
    source = _fixture_source(reversed_archive)

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(reversed_archive)

    reversed_manifest = prepare_peps(
        tmp_path / "reverse-cache",
        tmp_path / "reverse",
        downloader=downloader,
        source=source,
    )
    _, normal_manifest = _prepare_fixture(tmp_path / "normal")
    assert reversed_manifest.documents["sha256"] == normal_manifest.documents["sha256"]
    # Source bytes are intentionally part of lineage, so the overall fingerprints differ.
    assert reversed_manifest.dataset_fingerprint != normal_manifest.dataset_fingerprint


def test_compressed_checksum_is_verified_before_gzip_parsing(tmp_path: Path):
    archive = b"not a gzip stream"
    source = _fixture_source(_fixture_archive())

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(archive)

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        prepare_peps(tmp_path / "cache", tmp_path / "prepared", downloader=downloader, source=source)


@pytest.mark.parametrize(
    "record",
    [
        "not-json\n",
        json.dumps({"id": "broken", "metadata": {"license": "CC0-1.0"}}) + "\n",
        json.dumps({"id": "broken", "text": "   "}) + "\n",
    ],
)
def test_malformed_source_records_are_rejected(tmp_path: Path, record: str):
    archive = gzip.compress(record.encode(), mtime=0)
    source = DatasetSource(
        repository="fttl://tests/malformed",
        revision="fixture-v1",
        path="malformed.jsonl.gz",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=1,
        license="CC0-1.0",
        license_limitations=(),
    )

    def downloader(_url: str, destination: Path) -> None:
        destination.write_bytes(archive)

    with pytest.raises(DatasetValidationError, match="record|text"):
        prepare_peps(tmp_path / "cache", tmp_path / "prepared", downloader=downloader, source=source)


def test_changed_prepared_byte_is_rejected(tmp_path: Path):
    output, _ = _prepare_fixture(tmp_path)
    documents = output / "documents.jsonl"
    documents.write_bytes(documents.read_bytes() + b" ")
    with pytest.raises(DatasetValidationError, match="byte length mismatch"):
        load_dataset_manifest(output)


def test_tokenizer_drift_and_split_overlap_are_rejected(tmp_path: Path):
    output, _ = _prepare_fixture(tmp_path)
    manifest_path = output / "manifest.json"
    original = json.loads(manifest_path.read_text())

    tokenizer_drift = json.loads(json.dumps(original))
    tokenizer_drift["tokenizer"]["fingerprint"] = "0" * 64
    tokenizer_drift["dataset_fingerprint"] = _fingerprint_without_self(tokenizer_drift)
    manifest_path.write_text(json.dumps(tokenizer_drift), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="tokenizer fingerprint drift"):
        load_dataset_manifest(manifest_path)

    overlap = json.loads(json.dumps(original))
    duplicate = overlap["splits"]["train"]["document_ids"][0]
    overlap["splits"]["validation"]["document_ids"].append(duplicate)
    overlap["splits"]["validation"]["document_ids"].sort()
    overlap["splits"]["validation"]["document_count"] += 1
    overlap["splits"]["validation"]["document_ids_sha256"] = _sha_json(
        overlap["splits"]["validation"]["document_ids"]
    )
    overlap["counts"]["documents"] += 1
    overlap["dataset_fingerprint"] = _fingerprint_without_self(overlap)
    manifest_path.write_text(json.dumps(overlap), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="overlap"):
        load_dataset_manifest(manifest_path)


@pytest.mark.parametrize(
    "component, field, value",
    [
        ("tokenizer", "byte_ids", [1, 255]),
        ("tokenizer", "byte_ids", [False, 255]),
        ("tokenizer", "vocab_size", 257.0),
        ("tokenizer", "text_encoding", "UTF-16"),
        ("tokenizer", "text_encoding", None),
        ("tokenizer", "unknown_option", "enabled"),
        ("preprocessing", "unicode_normalization", "NFD"),
        ("preprocessing", "newlines", "CRLF"),
        ("preprocessing", "document_order", "source order"),
        ("preprocessing", "split", "random"),
        ("preprocessing", "unicode_normalization", None),
        ("preprocessing", "unknown_option", "enabled"),
    ],
)
def test_manifest_rejects_noncanonical_component_declarations_before_data_access(
    tmp_path: Path, component: str, field: str, value: object
):
    output, _ = _prepare_fixture(tmp_path)
    manifest_path = output / "manifest.json"
    altered = json.loads(manifest_path.read_text())
    if value is None:
        altered[component].pop(field)
    else:
        altered[component][field] = value
    altered["dataset_fingerprint"] = _fingerprint_without_self(altered)
    manifest_path.write_text(json.dumps(altered), encoding="utf-8")
    (output / "documents.jsonl").unlink()

    with pytest.raises(DatasetValidationError, match=f"{component} specification"):
        load_dataset_manifest(manifest_path)


def test_loaded_documents_have_stable_ids_and_default_to_train(tmp_path: Path):
    output, _ = _prepare_fixture(tmp_path)
    train_documents = load_prepared_documents(output)
    assert len(train_documents) == 2
    assert all(document.id == document.document_id for document in train_documents)
    assert all(document.split == "train" for document in train_documents)


def test_existing_output_must_match_the_requested_immutable_source(tmp_path: Path):
    output, _ = _prepare_fixture(tmp_path)
    different = DatasetSource(
        repository="fttl://tests/different-source",
        revision="fixture-v2",
        path="different.jsonl.gz",
        compressed_sha256="0" * 64,
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=(),
    )

    with pytest.raises(DatasetValidationError, match="immutable source"):
        prepare_peps(tmp_path / "cache", output, source=different)


@pytest.mark.parametrize("pipeline", ["gzip", "captured"])
@pytest.mark.parametrize("limit", ["document", "total"])
def test_ingestion_bounds_apply_after_unicode_normalization(
    tmp_path: Path, monkeypatch, pipeline: str, limit: str
):
    # NFC expands U+0344 from two UTF-8 bytes into two combining marks (four bytes).
    records = [{"id": "one", "text": "\u0344" * 64}]
    if limit == "total":
        records = [{"id": str(index), "text": "\u0344" * 32} for index in range(2)]
    raw = b"".join(json.dumps(row, ensure_ascii=False).encode() + b"\n" for row in records)
    archive = gzip.compress(raw, mtime=0) if pipeline == "gzip" else raw
    source = DatasetSource(
        repository="fttl://tests/normalization-bounds",
        revision="fixture-v1",
        path="records.jsonl.gz" if pipeline == "gzip" else "records.jsonl",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=len(records),
        license="CC0-1.0",
        license_limitations=(),
        artifact_kind="gzip-jsonl" if pipeline == "gzip" else "sealed-jsonl",
    )
    monkeypatch.setattr(dataset_module, "MAX_DOCUMENT_BYTES", 200)
    monkeypatch.setattr(dataset_module, "MAX_UNCOMPRESSED_BYTES", 240 if limit == "total" else 1000)
    output = tmp_path / "prepared"
    with pytest.raises(DatasetValidationError, match=f"normalized.*{limit}.*safety limit"):
        if pipeline == "gzip":
            prepare_peps(
                tmp_path / "cache",
                output,
                source=source,
                downloader=lambda _url, destination: destination.write_bytes(archive),
            )
        else:
            artifact = tmp_path / "records.jsonl"
            artifact.write_bytes(archive)
            prepare_document_records(records, output, source_artifact=artifact, source=source)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared.*"))


def test_captured_text_limit_includes_required_terminal_newline(tmp_path: Path, monkeypatch):
    records = [{"id": "one", "text": "1234"}]
    artifact = tmp_path / "records.jsonl"
    artifact.write_bytes(json.dumps(records).encode())
    source = DatasetSource(
        repository="fttl://tests/newline-boundary",
        revision="fixture-v1",
        path=artifact.name,
        compressed_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        expected_document_count=1,
        license="CC0-1.0",
        license_limitations=(),
        artifact_kind="sealed-jsonl",
    )
    monkeypatch.setattr(dataset_module, "MAX_DOCUMENT_BYTES", 4)
    with pytest.raises(DatasetValidationError, match="normalized.*document.*safety limit"):
        prepare_document_records(
            records, tmp_path / "rejected", source_artifact=artifact, source=source
        )
    assert not (tmp_path / "rejected").exists()
    monkeypatch.setattr(dataset_module, "MAX_DOCUMENT_BYTES", 5)
    manifest = prepare_document_records(
        records, tmp_path / "accepted", source_artifact=artifact, source=source
    )
    assert manifest.counts["utf8_bytes"] == 5


def test_gzip_reader_requests_bounded_lines(tmp_path: Path, monkeypatch):
    requests = []

    class BoundedStream(io.BytesIO):
        def __iter__(self):
            raise AssertionError("unbounded gzip line iteration")

        def readline(self, size=-1):
            requests.append(size)
            assert size == 33
            return super().readline(size)

    monkeypatch.setattr(dataset_module, "MAX_DOCUMENT_BYTES", 32)
    monkeypatch.setattr(gzip, "open", lambda *_args: BoundedStream(b"x" * 100))
    with pytest.raises(DatasetValidationError, match="per-document safety limit"):
        tuple(dataset_module._read_gzip_records(tmp_path / "source.gz"))
    assert requests == [33]


def _fingerprint_without_self(payload: dict[str, object]) -> str:
    identity = json.loads(json.dumps(payload))
    identity.pop("dataset_fingerprint")
    return _sha_json(identity)


def _sha_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
