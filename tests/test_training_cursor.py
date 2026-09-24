from dataclasses import replace

import torch
import pytest

from fttl.config import ExperimentConfig, ModelConfig
from fttl.data import PreparedDatasetBatchSource, SyntheticBatchSource, TrainingCursorV1


class Document:
    def __init__(self, document_id: str, text: str) -> None:
        self.document_id = document_id
        self.text = text


def config(*, vocab_size: int = 257) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=vocab_size,
            block_size=4,
            d_model=8,
            n_heads=2,
            n_layers=1,
            dropout=0.2,
        ),
        seed=7,
        steps=3,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )


def test_training_cursor_round_trip_preserves_next_batch_identity():
    source = SyntheticBatchSource(config(vocab_size=24))
    cursor = source.initial_cursor()

    restored = TrainingCursorV1.from_dict(cursor.to_dict())

    assert restored == cursor
    assert source.batch(restored).sample_ids == cursor.next_sample_ids


@pytest.mark.parametrize(
    "change",
    [
        {"batch_index": 1.0},
        {"epoch": False},
        {"next_sample_ids": "sample"},
        {"unexpected": "field"},
    ],
)
def test_training_cursor_rejects_coercion_and_unknown_fields(change):
    value = SyntheticBatchSource(config(vocab_size=24)).initial_cursor().to_dict()
    value.update(change)
    with pytest.raises(ValueError, match="cursor"):
        TrainingCursorV1.from_dict(value)


def test_synthetic_cursor_rejects_silent_advancement():
    source = SyntheticBatchSource(config(vocab_size=24))
    cursor = source.initial_cursor()

    with pytest.raises(ValueError, match="cursor"):
        source.batch(replace(cursor, batch_index=1))


def test_document_windows_are_stable_and_do_not_cross_documents():
    source = PreparedDatasetBatchSource(
        config(),
        [
            Document("pep-alpha", "abcdefghijklmno"),
            Document("pep-beta", "qrstuvwxyz"),
        ],
        data_fingerprint="d" * 64,
        tokenizer_fingerprint="t" * 64,
    )

    first = source.batch(source.initial_cursor())
    repeated = source.batch(source.initial_cursor())

    assert first.batch_id == repeated.batch_id
    assert first.sample_ids == ("pep-alpha:000000000000", "pep-alpha:000000000004")
    assert torch.equal(first.inputs, repeated.inputs)
    assert first.next_cursor.batch_index == 1
    assert first.next_cursor.next_sample_ids == (
        "pep-alpha:000000000008",
        "pep-alpha:000000000011",
    )
    second = source.batch(first.next_cursor)
    assert second.targets[1, -1].item() == 256


def test_utf8_dataset_requires_byte_token_vocabulary():
    with pytest.raises(ValueError, match="vocab_size=257"):
        PreparedDatasetBatchSource(
            config(vocab_size=256),
            [Document("doc", "enough text for one full batch")],
            data_fingerprint="d" * 64,
            tokenizer_fingerprint="t" * 64,
        )


def test_batch_identity_changes_when_the_same_windows_repeat_in_a_later_epoch():
    source = PreparedDatasetBatchSource(
        config(),
        [Document("only", "abcdefgh")],
        data_fingerprint="d" * 64,
        tokenizer_fingerprint="t" * 64,
    )
    first = source.batch(source.cursor_at(0))
    repeated = source.batch(source.cursor_at(1))
    assert first.sample_ids == repeated.sample_ids
    assert first.batch_id != repeated.batch_id
