from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch

from fttl.config import ExperimentConfig


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
        if self.schema_version != 1:
            raise ValueError("TrainingCursorV1 requires schema_version=1")
        if self.epoch < 0 or self.token_window_offset < 0 or self.batch_index < 0:
            raise ValueError("cursor counters must be non-negative")
        if not self.document_id or not self.batch_id or not self.next_sample_ids:
            raise ValueError("cursor identity fields must not be empty")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["next_sample_ids"] = list(self.next_sample_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "TrainingCursorV1":
        return cls(
            schema_version=int(value["schema_version"]),
            epoch=int(value["epoch"]),
            document_id=str(value["document_id"]),
            token_window_offset=int(value["token_window_offset"]),
            batch_id=str(value["batch_id"]),
            next_sample_ids=tuple(str(item) for item in value["next_sample_ids"]),  # type: ignore[arg-type]
            batch_index=int(value["batch_index"]),
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

    def _cursor(self, batch_index: int) -> TrainingCursorV1:
        sample_ids = tuple(
            f"synthetic:{batch_index:08d}:{index:04d}"
            for index in range(self.config.batch_size)
        )
        batch_id = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
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
        return self._cursor(batch_index)

    def batch(self, cursor: TrainingCursorV1) -> PreparedBatch:
        expected = self._cursor(cursor.batch_index)
        if cursor != expected:
            raise ValueError("synthetic training cursor does not match the requested batch")
        inputs, targets = batch_for_step(self.tokens, self.config, cursor.batch_index)
        return PreparedBatch(
            inputs=inputs,
            targets=targets,
            batch_id=cursor.batch_id,
            sample_ids=cursor.next_sample_ids,
            cursor=cursor,
            next_cursor=self._cursor(cursor.batch_index + 1),
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
        windows: list[_TokenWindow] = []
        for document in sorted(documents, key=lambda item: item.document_id):
            encoded = tuple(document.text.encode("utf-8")) + (256,)
            for offset in range(0, len(encoded) - width + 1, config.model.block_size):
                windows.append(
                    _TokenWindow(
                        document_id=document.document_id,
                        offset=offset,
                        tokens=encoded[offset : offset + width],
                    )
                )
        if len(windows) < config.batch_size:
            raise ValueError("prepared dataset has fewer full windows than one batch")
        self._windows = tuple(windows)

    @classmethod
    def from_manifest(
        cls,
        config: ExperimentConfig,
        manifest_path: Path,
        *,
        split: str = "train",
    ) -> "PreparedDatasetBatchSource":
        from fttl.dataset import load_dataset_manifest, load_prepared_documents

        manifest = load_dataset_manifest(manifest_path)
        documents = load_prepared_documents(manifest_path, split=split)
        return cls(
            config,
            documents,
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
        batch_id = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
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
