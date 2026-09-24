from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence


DATASET_MANIFEST_SCHEMA = "DatasetManifestV1"
TOKENIZER_NAME = "utf8-byte-v1"
TOKENIZER_VOCAB_SIZE = 257
TOKENIZER_EOD_ID = 256
PREPROCESSING_NAME = "pep-jsonl-normalize-v1"
MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024

PEP_REPOSITORY = "common-pile/python_enhancement_proposals"
PEP_REVISION = "f932757e3eba16475c893e1418918c77f14a790d"
PEP_PATH = "raw/documents/00000_peps.jsonl.gz"
PEP_COMPRESSED_SHA256 = "659be543c5e79ba776467fb694fbc9efaa350ed022b2e26139b966bae1735b0c"
PEP_DOCUMENT_COUNT = 656


class DatasetValidationError(ValueError):
    """Raised when source data or a prepared dataset violates its contract."""


@dataclass(frozen=True)
class DatasetSource:
    repository: str
    revision: str
    path: str
    compressed_sha256: str
    expected_document_count: int
    license: str
    license_limitations: tuple[str, ...]
    artifact_kind: str = "gzip-jsonl"
    url_override: str | None = None

    @property
    def url(self) -> str:
        if self.url_override is not None:
            return self.url_override
        return f"https://huggingface.co/datasets/{self.repository}/resolve/{self.revision}/{self.path}"

    def validate(self) -> None:
        if not self.repository or not self.revision or not self.path:
            raise DatasetValidationError("dataset source identity fields must not be empty")
        _require_sha256(self.compressed_sha256, "source compressed SHA-256")
        if self.expected_document_count <= 0:
            raise DatasetValidationError("expected document count must be positive")
        if not self.license:
            raise DatasetValidationError("dataset license must not be empty")
        if not self.artifact_kind:
            raise DatasetValidationError("dataset artifact kind must not be empty")


PINNED_PEP_SOURCE = DatasetSource(
    repository=PEP_REPOSITORY,
    revision=PEP_REVISION,
    path=PEP_PATH,
    compressed_sha256=PEP_COMPRESSED_SHA256,
    expected_document_count=PEP_DOCUMENT_COUNT,
    license="Public domain (per the source PEP corpus metadata)",
    license_limitations=(
        "The source corpus excludes five PEPs whose licensing differs from the included documents.",
        "The Common Pile dataset card warns that automated licensing metadata can contain errors.",
    ),
)


TOKENIZER_SPEC = {
    "name": TOKENIZER_NAME,
    "vocab_size": TOKENIZER_VOCAB_SIZE,
    "byte_ids": [0, 255],
    "end_of_document_id": TOKENIZER_EOD_ID,
    "text_encoding": "UTF-8",
}
TOKENIZER_FINGERPRINT = hashlib.sha256(
    json.dumps(TOKENIZER_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()

PREPROCESSING_SPEC = {
    "name": PREPROCESSING_NAME,
    "unicode_normalization": "NFC",
    "newlines": "LF with exactly one terminal LF",
    "document_order": "document_id ascending",
    "split": "first 64 bits of SHA-256(document_id), modulo 1000; train <800, validation <900, test otherwise",
}
PREPROCESSING_FINGERPRINT = hashlib.sha256(
    json.dumps(PREPROCESSING_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


class Utf8ByteTokenizer:
    """A dependency-free tokenizer whose identity is stable across platforms."""

    name = TOKENIZER_NAME
    vocab_size = TOKENIZER_VOCAB_SIZE
    end_of_document_id = TOKENIZER_EOD_ID

    @staticmethod
    def fingerprint() -> str:
        return TOKENIZER_FINGERPRINT

    def encode(self, text: str, *, add_end_of_document: bool = True) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        token_ids = tuple(text.encode("utf-8"))
        if add_end_of_document:
            return (*token_ids, self.end_of_document_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, allow_end_of_document: bool = True) -> str:
        byte_values: list[int] = []
        for token_id in token_ids:
            if token_id == self.end_of_document_id and allow_end_of_document:
                continue
            if not isinstance(token_id, int) or not 0 <= token_id <= 255:
                raise DatasetValidationError(f"token id {token_id!r} is outside the byte vocabulary")
            byte_values.append(token_id)
        return bytes(byte_values).decode("utf-8")


@dataclass(frozen=True)
class PreparedDocument:
    document_id: str
    source_id: str
    split: str
    text: str
    text_sha256: str

    @property
    def id(self) -> str:
        return self.document_id


@dataclass(frozen=True)
class DatasetManifestV1:
    source: Mapping[str, object]
    license: Mapping[str, object]
    documents: Mapping[str, object]
    counts: Mapping[str, object]
    splits: Mapping[str, Mapping[str, object]]
    tokenizer: Mapping[str, object]
    preprocessing: Mapping[str, object]
    dataset_fingerprint: str
    schema: str = DATASET_MANIFEST_SCHEMA

    def fingerprint(self) -> str:
        return self.dataset_fingerprint

    @property
    def tokenizer_fingerprint(self) -> str:
        return str(self.tokenizer["fingerprint"])

    @property
    def preprocessing_fingerprint(self) -> str:
        return str(self.preprocessing["fingerprint"])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source": dict(self.source),
            "license": dict(self.license),
            "documents": dict(self.documents),
            "counts": dict(self.counts),
            "splits": {name: dict(details) for name, details in self.splits.items()},
            "tokenizer": dict(self.tokenizer),
            "preprocessing": dict(self.preprocessing),
            "dataset_fingerprint": self.dataset_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "DatasetManifestV1":
        if payload.get("schema") != DATASET_MANIFEST_SCHEMA:
            raise DatasetValidationError(f"expected manifest schema {DATASET_MANIFEST_SCHEMA}")
        try:
            source = _require_mapping(payload["source"], "source")
            license_details = _require_mapping(payload["license"], "license")
            documents = _require_mapping(payload["documents"], "documents")
            counts = _require_mapping(payload["counts"], "counts")
            tokenizer = _require_mapping(payload["tokenizer"], "tokenizer")
            preprocessing = _require_mapping(payload["preprocessing"], "preprocessing")
            raw_splits = _require_mapping(payload["splits"], "splits")
            splits = {str(name): _require_mapping(value, f"splits.{name}") for name, value in raw_splits.items()}
            fingerprint = str(payload["dataset_fingerprint"])
        except KeyError as error:
            raise DatasetValidationError(f"manifest is missing required field {error.args[0]!r}") from error
        manifest = cls(
            source=source,
            license=license_details,
            documents=documents,
            counts=counts,
            splits=splits,
            tokenizer=tokenizer,
            preprocessing=preprocessing,
            dataset_fingerprint=fingerprint,
        )
        _validate_manifest_contract(manifest)
        return manifest


DownloadFunction = Callable[[str, Path], None]


def prepare_peps(
    cache_dir: Path,
    output_dir: Path,
    *,
    downloader: DownloadFunction | None = None,
    source: DatasetSource = PINNED_PEP_SOURCE,
) -> DatasetManifestV1:
    """Download, verify, and prepare the pinned PEP corpus.

    ``source`` exists so tests can exercise the real preparation path with a
    tiny independently licensed archive. The command-line interface never
    exposes it and always uses :data:`PINNED_PEP_SOURCE`.
    """

    source.validate()
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.exists():
        manifest = load_dataset_manifest(existing_manifest)
        expected_source = {
            "repository": source.repository,
            "revision": source.revision,
            "path": source.path,
            "compressed_sha256": source.compressed_sha256,
        }
        if any(manifest.source.get(key) != value for key, value in expected_source.items()):
            raise DatasetValidationError(
                "existing prepared dataset does not match the requested immutable source"
            )
        if manifest.counts.get("documents") != source.expected_document_count:
            raise DatasetValidationError(
                "existing prepared dataset does not match the requested document count"
            )
        return manifest
    if output_dir.exists():
        raise DatasetValidationError(f"output directory exists without a valid manifest: {output_dir}")

    archive_path = cache_dir / "downloads" / source.compressed_sha256 / Path(source.path).name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        _verify_file(archive_path, source.compressed_sha256, "cached source archive")
    else:
        fetch = downloader or _download_url
        temporary_archive = archive_path.with_name(f".{archive_path.name}.partial")
        try:
            fetch(source.url, temporary_archive)
            _verify_file(temporary_archive, source.compressed_sha256, "downloaded source archive")
            os.replace(temporary_archive, archive_path)
        finally:
            temporary_archive.unlink(missing_ok=True)

    # This intentionally happens only after compressed-byte verification.
    records = tuple(_read_gzip_records(archive_path))
    if len(records) != source.expected_document_count:
        raise DatasetValidationError(
            f"source contains {len(records)} documents; expected {source.expected_document_count}"
        )
    return _write_prepared_dataset(records, output_dir, archive_path=archive_path, source=source)


def prepare_document_records(
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    source_artifact: Path,
    source: DatasetSource,
) -> DatasetManifestV1:
    """Prepare already captured records through the same deterministic contract."""

    source.validate()
    if len(records) != source.expected_document_count:
        raise DatasetValidationError(
            f"source contains {len(records)} documents; expected {source.expected_document_count}"
        )
    _verify_file(source_artifact, source.compressed_sha256, "source artifact")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise DatasetValidationError(f"output directory already exists: {output_dir}")
    return _write_prepared_dataset(
        records,
        output_dir,
        archive_path=source_artifact,
        source=source,
    )


def load_dataset_manifest(path: Path) -> DatasetManifestV1:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"cannot read dataset manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise DatasetValidationError("dataset manifest root must be a JSON object")
    manifest = DatasetManifestV1.from_dict(payload)
    _verify_prepared_documents(manifest_path.parent, manifest)
    return manifest


def load_prepared_documents(path: Path, *, split: str | None = "train") -> tuple[PreparedDocument, ...]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = load_dataset_manifest(manifest_path)
    requested_split = _normalize_split(split) if split is not None else None
    documents_path = manifest_path.parent / str(manifest.documents["path"])
    documents = tuple(_iter_prepared_documents(documents_path))
    if requested_split is None:
        return documents
    return tuple(document for document in documents if document.split == requested_split)


def stable_document_id(source_repository: str, source_id: str) -> str:
    if not source_repository or not source_id:
        raise DatasetValidationError("source repository and source id must not be empty")
    return hashlib.sha256(f"{source_repository}\0{source_id}".encode("utf-8")).hexdigest()


def split_for_document(document_id: str) -> str:
    _require_sha256(document_id, "document id")
    bucket = int.from_bytes(hashlib.sha256(document_id.encode("ascii")).digest()[:8], "big") % 1000
    if bucket < 800:
        return "train"
    if bucket < 900:
        return "validation"
    return "test"


def _download_url(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "fttl-prepare-peps/0.2"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_COMPRESSED_BYTES:
                raise DatasetValidationError("compressed dataset exceeds the safety limit")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())


def _read_gzip_records(path: Path) -> Iterator[Mapping[str, object]]:
    total_bytes = 0
    try:
        with gzip.open(path, "rb") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                total_bytes += len(raw_line)
                if total_bytes > MAX_UNCOMPRESSED_BYTES:
                    raise DatasetValidationError("uncompressed dataset exceeds the safety limit")
                if len(raw_line) > MAX_DOCUMENT_BYTES:
                    raise DatasetValidationError(f"record {line_number} exceeds the per-document safety limit")
                try:
                    decoded = raw_line.decode("utf-8")
                    record = json.loads(decoded)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise DatasetValidationError(f"record {line_number} is not valid UTF-8 JSON") from error
                if not isinstance(record, dict):
                    raise DatasetValidationError(f"record {line_number} must be a JSON object")
                yield record
    except (gzip.BadGzipFile, EOFError, OSError) as error:
        raise DatasetValidationError(f"verified archive cannot be decompressed: {error}") from error


def _write_prepared_dataset(
    records: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    archive_path: Path,
    source: DatasetSource,
) -> DatasetManifestV1:
    documents = _canonicalize_records(records, source.repository)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    try:
        documents_path = temporary_dir / "documents.jsonl"
        with documents_path.open("wb") as output:
            for document in documents:
                row = {
                    "document_id": document.document_id,
                    "source_id": document.source_id,
                    "split": document.split,
                    "text": document.text,
                    "text_sha256": document.text_sha256,
                }
                output.write(_canonical_json_bytes(row) + b"\n")
            output.flush()
            os.fsync(output.fileno())

        manifest_payload = _build_manifest_payload(
            documents,
            documents_path=documents_path,
            archive_path=archive_path,
            source=source,
        )
        manifest_path = temporary_dir / "manifest.json"
        with manifest_path.open("wb") as output:
            output.write(json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_dir, output_dir)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return load_dataset_manifest(output_dir / "manifest.json")


def _canonicalize_records(
    records: Sequence[Mapping[str, object]], source_repository: str
) -> tuple[PreparedDocument, ...]:
    documents: list[PreparedDocument] = []
    seen_document_ids: set[str] = set()
    for ordinal, record in enumerate(records, start=1):
        raw_text = record.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise DatasetValidationError(f"record {ordinal} has no non-empty text field")
        text = _normalize_text(raw_text)
        source_id = _extract_source_id(record, text)
        document_id = stable_document_id(source_repository, source_id)
        if document_id in seen_document_ids:
            raise DatasetValidationError(f"duplicate stable document id for source id {source_id!r}")
        seen_document_ids.add(document_id)
        documents.append(
            PreparedDocument(
                document_id=document_id,
                source_id=source_id,
                split=split_for_document(document_id),
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return tuple(sorted(documents, key=lambda document: document.document_id))


def _extract_source_id(record: Mapping[str, object], normalized_text: str) -> str:
    candidate_keys = ("id", "document_id", "url", "filename", "file_name", "path")
    containers: tuple[Mapping[str, object], ...]
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        containers = (record, metadata)
    else:
        containers = (record,)
    for container in containers:
        for key in candidate_keys:
            candidate = container.get(key)
            if isinstance(candidate, (str, int)) and str(candidate).strip():
                return unicodedata.normalize("NFC", str(candidate).strip())
    # Content identity is stable across source reordering when metadata lacks an id.
    return f"content-sha256:{hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()}"


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return normalized.rstrip("\n") + "\n"


def _build_manifest_payload(
    documents: Sequence[PreparedDocument],
    *,
    documents_path: Path,
    archive_path: Path,
    source: DatasetSource,
) -> dict[str, object]:
    split_payload: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        ids = sorted(document.document_id for document in documents if document.split == split)
        split_payload[split] = {
            "document_count": len(ids),
            "document_ids": ids,
            "document_ids_sha256": _sha256_json(ids),
        }
    payload: dict[str, object] = {
        "schema": DATASET_MANIFEST_SCHEMA,
        "source": {
            "repository": source.repository,
            "revision": source.revision,
            "path": source.path,
            "url": source.url,
            "artifact_kind": source.artifact_kind,
            "artifact_sha256": source.compressed_sha256,
            "artifact_byte_length": archive_path.stat().st_size,
            "compressed_sha256": source.compressed_sha256,
            "compressed_byte_length": archive_path.stat().st_size,
        },
        "license": {
            "description": source.license,
            "limitations": list(source.license_limitations),
        },
        "documents": {
            "path": documents_path.name,
            "sha256": _sha256_file(documents_path),
            "byte_length": documents_path.stat().st_size,
        },
        "counts": {
            "documents": len(documents),
            "utf8_bytes": sum(len(document.text.encode("utf-8")) for document in documents),
            "tokens_including_eod": sum(len(document.text.encode("utf-8")) + 1 for document in documents),
        },
        "splits": split_payload,
        "tokenizer": {**TOKENIZER_SPEC, "fingerprint": TOKENIZER_FINGERPRINT},
        "preprocessing": {**PREPROCESSING_SPEC, "fingerprint": PREPROCESSING_FINGERPRINT},
    }
    payload["dataset_fingerprint"] = _sha256_json(payload)
    return payload


def _validate_manifest_contract(manifest: DatasetManifestV1) -> None:
    _require_sha256(manifest.dataset_fingerprint, "dataset fingerprint")
    identity = manifest.to_dict()
    identity.pop("dataset_fingerprint")
    if _sha256_json(identity) != manifest.dataset_fingerprint:
        raise DatasetValidationError("dataset manifest fingerprint does not match its contents")

    for field in ("repository", "revision", "path", "url", "artifact_kind"):
        value = manifest.source.get(field)
        if not isinstance(value, str) or not value:
            raise DatasetValidationError(f"dataset source field {field!r} must be a non-empty string")
    compressed_sha256 = manifest.source.get("compressed_sha256")
    if not isinstance(compressed_sha256, str):
        raise DatasetValidationError("dataset source compressed SHA-256 is missing")
    _require_sha256(compressed_sha256, "dataset source compressed SHA-256")
    artifact_sha256 = manifest.source.get("artifact_sha256")
    if not isinstance(artifact_sha256, str):
        raise DatasetValidationError("dataset source artifact SHA-256 is missing")
    _require_sha256(artifact_sha256, "dataset source artifact SHA-256")
    if artifact_sha256 != compressed_sha256:
        raise DatasetValidationError("dataset source artifact hashes are inconsistent")
    compressed_length = manifest.source.get("compressed_byte_length")
    if not isinstance(compressed_length, int) or isinstance(compressed_length, bool) or compressed_length <= 0:
        raise DatasetValidationError("dataset source compressed byte length must be positive")
    if manifest.source.get("artifact_byte_length") != compressed_length:
        raise DatasetValidationError("dataset source artifact byte lengths are inconsistent")
    description = manifest.license.get("description")
    limitations = manifest.license.get("limitations")
    if not isinstance(description, str) or not description:
        raise DatasetValidationError("dataset license description must not be empty")
    if not isinstance(limitations, list) or not all(isinstance(value, str) for value in limitations):
        raise DatasetValidationError("dataset license limitations must be a string list")

    if manifest.tokenizer.get("name") != TOKENIZER_NAME:
        raise DatasetValidationError("unsupported tokenizer identity")
    if manifest.tokenizer.get("fingerprint") != TOKENIZER_FINGERPRINT:
        raise DatasetValidationError("tokenizer fingerprint drift detected")
    if manifest.tokenizer.get("vocab_size") != TOKENIZER_VOCAB_SIZE:
        raise DatasetValidationError("tokenizer vocabulary size drift detected")
    if manifest.tokenizer.get("end_of_document_id") != TOKENIZER_EOD_ID:
        raise DatasetValidationError("tokenizer end-of-document id drift detected")
    if manifest.preprocessing.get("name") != PREPROCESSING_NAME:
        raise DatasetValidationError("unsupported preprocessing identity")
    if manifest.preprocessing.get("fingerprint") != PREPROCESSING_FINGERPRINT:
        raise DatasetValidationError("preprocessing fingerprint drift detected")

    if set(manifest.splits) != {"train", "validation", "test"}:
        raise DatasetValidationError("manifest must define train, validation, and test splits")
    assigned_ids: set[str] = set()
    total_documents = 0
    for split_name, split_details in manifest.splits.items():
        document_ids = split_details.get("document_ids")
        if not isinstance(document_ids, list) or not all(isinstance(value, str) for value in document_ids):
            raise DatasetValidationError(f"split {split_name!r} document_ids must be a string list")
        if document_ids != sorted(document_ids) or len(document_ids) != len(set(document_ids)):
            raise DatasetValidationError(f"split {split_name!r} document ids must be sorted and unique")
        overlap = assigned_ids.intersection(document_ids)
        if overlap:
            raise DatasetValidationError(f"document ids overlap across splits: {sorted(overlap)!r}")
        assigned_ids.update(document_ids)
        if split_details.get("document_count") != len(document_ids):
            raise DatasetValidationError(f"split {split_name!r} document count is inconsistent")
        if split_details.get("document_ids_sha256") != _sha256_json(document_ids):
            raise DatasetValidationError(f"split {split_name!r} id digest is inconsistent")
        total_documents += len(document_ids)
    if manifest.counts.get("documents") != total_documents:
        raise DatasetValidationError("manifest total document count is inconsistent with its splits")
    for field in ("documents", "utf8_bytes", "tokens_including_eod"):
        value = manifest.counts.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise DatasetValidationError(f"manifest count {field!r} must be a non-negative integer")


def _verify_prepared_documents(base_dir: Path, manifest: DatasetManifestV1) -> None:
    relative_path = manifest.documents.get("path")
    if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
        raise DatasetValidationError("prepared documents path must be relative")
    base_dir = base_dir.resolve()
    documents_path = (base_dir / relative_path).resolve()
    if documents_path.parent != base_dir:
        raise DatasetValidationError("prepared documents path must stay inside the dataset directory")
    expected_length = manifest.documents.get("byte_length")
    if not isinstance(expected_length, int) or expected_length < 0:
        raise DatasetValidationError("prepared documents byte length must be non-negative")
    try:
        actual_length = documents_path.stat().st_size
    except OSError as error:
        raise DatasetValidationError(f"prepared documents file is unavailable: {error}") from error
    if actual_length != expected_length:
        raise DatasetValidationError(
            f"prepared documents byte length mismatch: expected {expected_length}, got {actual_length}"
        )
    expected_hash = manifest.documents.get("sha256")
    if not isinstance(expected_hash, str):
        raise DatasetValidationError("prepared documents SHA-256 is missing")
    _verify_file(documents_path, expected_hash, "prepared documents")

    documents = tuple(_iter_prepared_documents(documents_path))
    if len(documents) != manifest.counts.get("documents"):
        raise DatasetValidationError("prepared document count does not match manifest")
    expected_by_split = {
        split: list(details["document_ids"]) for split, details in manifest.splits.items()
    }
    actual_by_split = {
        split: sorted(document.document_id for document in documents if document.split == split)
        for split in expected_by_split
    }
    if actual_by_split != expected_by_split:
        raise DatasetValidationError("prepared document assignments do not match manifest splits")
    if [document.document_id for document in documents] != sorted(document.document_id for document in documents):
        raise DatasetValidationError("prepared documents are not in canonical order")
    utf8_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
    if utf8_bytes != manifest.counts.get("utf8_bytes"):
        raise DatasetValidationError("prepared UTF-8 byte count does not match manifest")
    if utf8_bytes + len(documents) != manifest.counts.get("tokens_including_eod"):
        raise DatasetValidationError("prepared token count does not match manifest")


def _iter_prepared_documents(path: Path) -> Iterator[PreparedDocument]:
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetValidationError(f"prepared document {line_number} is invalid JSON") from error
                if not isinstance(row, dict):
                    raise DatasetValidationError(f"prepared document {line_number} must be an object")
                required = ("document_id", "source_id", "split", "text", "text_sha256")
                if any(not isinstance(row.get(field), str) for field in required):
                    raise DatasetValidationError(f"prepared document {line_number} has invalid fields")
                document = PreparedDocument(**{field: row[field] for field in required})
                if not document.source_id:
                    raise DatasetValidationError(f"prepared document {line_number} has an empty source id")
                _require_sha256(document.document_id, f"prepared document {line_number} id")
                _require_sha256(document.text_sha256, f"prepared document {line_number} text SHA-256")
                if document.text_sha256 != hashlib.sha256(document.text.encode("utf-8")).hexdigest():
                    raise DatasetValidationError(f"prepared document {line_number} text digest mismatch")
                if _normalize_text(document.text) != document.text:
                    raise DatasetValidationError(f"prepared document {line_number} text is not canonical")
                if document.split != split_for_document(document.document_id):
                    raise DatasetValidationError(f"prepared document {line_number} split assignment drifted")
                yield document
    except (OSError, UnicodeDecodeError) as error:
        raise DatasetValidationError(f"cannot read prepared documents: {error}") from error


def _normalize_split(split: str) -> str:
    if split not in {"train", "validation", "test"}:
        raise DatasetValidationError(f"unknown dataset split {split!r}")
    return split


def _verify_file(path: Path, expected_sha256: str, label: str) -> None:
    _require_sha256(expected_sha256, f"{label} SHA-256")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise DatasetValidationError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DatasetValidationError(f"cannot hash {path}: {error}") from error
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DatasetValidationError(f"{label} must be a lowercase hexadecimal digest")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DatasetValidationError(f"manifest field {label!r} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
