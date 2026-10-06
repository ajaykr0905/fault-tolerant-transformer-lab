from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch

from fttl.config import ExperimentConfig
from fttl.dataset import PreparedDatasetSnapshot, load_dataset_snapshot
from fttl.state import state_digest

BATCH_IDENTITY_ALGORITHM = "source-bound-batch-v2"


def _bound_batch_id(contract: dict[str, object], index: int, sample_ids: tuple[str, ...]) -> str:
    identity = {"contract": contract, "batch_index": index, "sample_ids": list(sample_ids)}
    canonical = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _require_nonnegative_integer(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"cursor {field} must be a non-negative integer")


@dataclass(frozen=True)
class TrainingCursorV1:
    """The next uncommitted batch in a deterministic training stream."""

    schema_version: int
    epoch: int
    document_id: str
    token_window_offset: int
    batch_id: str
    next_sample_ids: tuple[str, ...]
    batch_index: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("TrainingCursorV1 requires schema_version=1")
        for field in ("epoch", "token_window_offset", "batch_index"):
            _require_nonnegative_integer(getattr(self, field), field)
        if any(type(value) is not str or not value for value in (self.document_id, self.batch_id)):
            raise ValueError("cursor identity fields must be nonempty strings")
        if (
            type(self.next_sample_ids) is not tuple
            or not self.next_sample_ids
            or any(type(value) is not str or not value for value in self.next_sample_ids)
        ):
            raise ValueError("cursor sample IDs must be a nonempty tuple of strings")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["next_sample_ids"] = list(self.next_sample_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingCursorV1":
        required = {
            "schema_version",
            "epoch",
            "document_id",
            "token_window_offset",
            "batch_id",
            "next_sample_ids",
            "batch_index",
        }
        if set(value) != required:
            raise ValueError("training cursor fields do not match TrainingCursorV1")
        integer_fields = ("schema_version", "epoch", "token_window_offset", "batch_index")
        if any(
            isinstance(value[field], bool) or not isinstance(value[field], int)
            for field in integer_fields
        ):
            raise ValueError("training cursor counters must be integers")
        if any(
            not isinstance(value[field], str) or not value[field]
            for field in ("document_id", "batch_id")
        ):
            raise ValueError("training cursor identity fields must be strings")
        raw_sample_ids = value["next_sample_ids"]
        if (
            not isinstance(raw_sample_ids, list)
            or not raw_sample_ids
            or not all(isinstance(item, str) and item for item in raw_sample_ids)
        ):
            raise ValueError("training cursor sample IDs must be a non-empty string list")
        return cls(
            schema_version=value["schema_version"],
            epoch=value["epoch"],
            document_id=value["document_id"],
            token_window_offset=value["token_window_offset"],
            batch_id=value["batch_id"],
            next_sample_ids=tuple(raw_sample_ids),
            batch_index=value["batch_index"],
        )


@dataclass(frozen=True)
class PreparedBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    batch_id: str
    sample_ids: tuple[str, ...]
    cursor: TrainingCursorV1
    next_cursor: TrainingCursorV1


class BatchSource(Protocol):
    data_fingerprint: str
    tokenizer_fingerprint: str

    def initial_cursor(self) -> TrainingCursorV1: ...

    def cursor_at(self, batch_index: int) -> TrainingCursorV1: ...

    def batch(self, cursor: TrainingCursorV1) -> PreparedBatch: ...


class SyntheticBatchSource:
    """Compatibility source for the original deterministic smoke workload."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        token_count = max(2_048, config.model.block_size * config.batch_size * 32)
        self.tokens = synthetic_token_stream(token_count, config.model.vocab_size)
        identity = {
            "kind": "synthetic-token-stream-v1",
            "length": token_count,
            "vocab_size": config.model.vocab_size,
        }
        canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        self.data_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.tokenizer_fingerprint = hashlib.sha256(
            f"synthetic-integer-v1:{config.model.vocab_size}".encode("utf-8")
        ).hexdigest()

    def _batch_contract(self, tokens: torch.Tensor) -> dict[str, object]:
        return {
            "algorithm": BATCH_IDENTITY_ALGORITHM,
            "source_kind": "synthetic-token-stream-v1",
            "data_fingerprint": self.data_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "content_fingerprint": state_digest(tokens),
            "block_size": self.config.model.block_size,
            "batch_size": self.config.batch_size,
        }

    @property
    def batch_identity_contract(self) -> dict[str, object]:
        return self._batch_contract(self.tokens.detach().clone())

    def _cursor(
        self, batch_index: int, *, tokens: torch.Tensor | None = None
    ) -> TrainingCursorV1:
        captured = self.tokens.detach().clone() if tokens is None else tokens
        sample_ids = tuple(
            f"synthetic:{batch_index:08d}:{index:04d}"
            for index in range(self.config.batch_size)
        )
        batch_id = _bound_batch_id(self._batch_contract(captured), batch_index, sample_ids)
        return TrainingCursorV1(
            schema_version=1,
            epoch=0,
            document_id="synthetic-token-stream-v1",
            token_window_offset=(
                batch_index
                * self.config.batch_size
                * (self.config.model.block_size + 1)
            ),
            batch_id=batch_id,
            next_sample_ids=sample_ids,
            batch_index=batch_index,
        )

    def initial_cursor(self) -> TrainingCursorV1:
        return self._cursor(0)

    def cursor_at(self, batch_index: int) -> TrainingCursorV1:
        _require_nonnegative_integer(batch_index, "batch_index")
        return self._cursor(batch_index)

    def batch(self, cursor: TrainingCursorV1) -> PreparedBatch:
        captured = self.tokens.detach().clone()
        expected = self._cursor(cursor.batch_index, tokens=captured)
        if cursor != expected:
            raise ValueError("synthetic training cursor does not match the requested batch")
        inputs, targets = batch_for_step(captured, self.config, cursor.batch_index)
        return PreparedBatch(
            inputs=inputs,
            targets=targets,
            batch_id=cursor.batch_id,
            sample_ids=cursor.next_sample_ids,
            cursor=cursor,
            next_cursor=self._cursor(cursor.batch_index + 1, tokens=captured),
        )


class PreparedDocument(Protocol):
    document_id: str
    text: str


@dataclass(frozen=True)
class _TokenWindow:
    document_id: str
    offset: int
    tokens: tuple[int, ...]

    @property
    def sample_id(self) -> str:
        return f"{self.document_id}:{self.offset:012d}"


class PreparedDatasetBatchSource:
    """Deterministic document-local byte windows loaded from a verified manifest."""

    def __init__(
        self,
        config: ExperimentConfig,
        documents: Sequence[PreparedDocument],
        *,
        data_fingerprint: str,
        tokenizer_fingerprint: str,
    ) -> None:
        if config.model.vocab_size != 257:
            raise ValueError("utf8-byte-v1 datasets require model vocab_size=257")
        self.config = config
        self.data_fingerprint = data_fingerprint
        self.tokenizer_fingerprint = tokenizer_fingerprint
        width = config.model.block_size + 1
        windows_by_document: list[tuple[_TokenWindow, ...]] = []
        content_digest = hashlib.sha256()
        captured_documents = sorted(
            ((document.document_id, document.text) for document in documents), key=lambda item: item[0]
        )
        for document_id, text in captured_documents:
            encoded_bytes = text.encode("utf-8")
            identity = [document_id, hashlib.sha256(encoded_bytes).hexdigest()]
            content_digest.update(json.dumps(identity, ensure_ascii=False).encode("utf-8") + b"\n")
            encoded = tuple(encoded_bytes) + (256,)
            if len(encoded) < width:
                continue
            offsets = list(range(0, len(encoded) - width + 1, config.model.block_size))
            final_offset = len(encoded) - width
            if offsets[-1] != final_offset:
                offsets.append(final_offset)
            windows_by_document.append(
                tuple(
                    _TokenWindow(
                        document_id=document_id,
                        offset=offset,
                        tokens=encoded[offset : offset + width],
                    )
                    for offset in offsets
                )
            )
        windows = [
            document_windows[window_index]
            for window_index in range(
                max((len(value) for value in windows_by_document), default=0)
            )
            for document_windows in windows_by_document
            if window_index < len(document_windows)
        ]
        if len(windows) < config.batch_size:
            raise ValueError("prepared dataset has fewer full windows than one batch")
        self._windows = tuple(windows)
        self._content_fingerprint = content_digest.hexdigest()

    @property
    def batch_identity_contract(self) -> dict[str, object]:
        return {
            "algorithm": BATCH_IDENTITY_ALGORITHM,
            "source_kind": "prepared-document-local-interleaved-windows-v1",
            "data_fingerprint": self.data_fingerprint,
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "content_fingerprint": self._content_fingerprint,
            "block_size": self.config.model.block_size,
            "batch_size": self.config.batch_size,
        }

    @classmethod
    def from_manifest(
        cls,
        config: ExperimentConfig,
        manifest_path: Path,
        *,
        split: str = "train",
    ) -> "PreparedDatasetBatchSource":
        return cls.from_snapshot(config, load_dataset_snapshot(manifest_path), split=split)

    @classmethod
    def from_snapshot(
        cls,
        config: ExperimentConfig,
        snapshot: PreparedDatasetSnapshot,
        *,
        split: str = "train",
    ) -> "PreparedDatasetBatchSource":
        manifest = snapshot.manifest
        return cls(
            config,
            snapshot.documents_for(split),
            data_fingerprint=manifest.fingerprint(),
            tokenizer_fingerprint=manifest.tokenizer_fingerprint,
        )

    def _window_indices(self, batch_index: int) -> tuple[int, ...]:
        start = batch_index * self.config.batch_size
        return tuple(
            (start + index) % len(self._windows)
            for index in range(self.config.batch_size)
        )

    def _cursor(self, batch_index: int) -> TrainingCursorV1:
        indices = self._window_indices(batch_index)
        selected = tuple(self._windows[index] for index in indices)
        sample_ids = tuple(window.sample_id for window in selected)
        batch_id = _bound_batch_id(self.batch_identity_contract, batch_index, sample_ids)
        absolute_start = batch_index * self.config.batch_size
        first = selected[0]
        return TrainingCursorV1(
            schema_version=1,
            epoch=absolute_start // len(self._windows),
            document_id=first.document_id,
            token_window_offset=first.offset,
            batch_id=batch_id,
            next_sample_ids=sample_ids,
            batch_index=batch_index,
        )

    def initial_cursor(self) -> TrainingCursorV1:
        return self._cursor(0)

    def cursor_at(self, batch_index: int) -> TrainingCursorV1:
        _require_nonnegative_integer(batch_index, "batch_index")
        return self._cursor(batch_index)

    def batch(self, cursor: TrainingCursorV1) -> PreparedBatch:
        expected = self._cursor(cursor.batch_index)
        if cursor != expected:
            raise ValueError("training cursor does not match the prepared dataset")
        windows = tuple(self._windows[index] for index in self._window_indices(cursor.batch_index))
        tensor = torch.tensor([window.tokens for window in windows], dtype=torch.long)
        return PreparedBatch(
            inputs=tensor[:, :-1],
            targets=tensor[:, 1:],
            batch_id=cursor.batch_id,
            sample_ids=cursor.next_sample_ids,
            cursor=cursor,
            next_cursor=self._cursor(cursor.batch_index + 1),
        )


def synthetic_token_stream(length: int, vocab_size: int) -> torch.Tensor:
    """Return a deterministic, non-random token stream for smoke verification."""
    if length < 2:
        raise ValueError("length must be at least 2")
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    positions = torch.arange(length, dtype=torch.long)
    return (positions * 17 + positions.square() * 3 + 11) % vocab_size


def batch_for_step(tokens: torch.Tensor, config: ExperimentConfig, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a stable batch keyed by training step instead of mutable RNG state."""
    width = config.model.block_size + 1
    if tokens.numel() < width:
        raise ValueError("token stream is shorter than one training sequence")
    max_start = tokens.numel() - width + 1
    starts = [((step * config.batch_size + index) * width) % max_start for index in range(config.batch_size)]
    windows = torch.stack([tokens[start : start + width] for start in starts])
    return windows[:, :-1], windows[:, 1:]
