import hashlib

import pytest
from test_training_cursor import Document, config

from fttl.data import PreparedDatasetBatchSource, SyntheticBatchSource, TrainingCursorV1
from fttl.state import capture_rng_state, state_trees_equal


def _fields():
    return {
        "schema_version": 1,
        "epoch": 0,
        "document_id": "document",
        "token_window_offset": 0,
        "batch_id": "batch",
        "next_sample_ids": ("sample",),
        "batch_index": 0,
    }


@pytest.mark.parametrize("schema", [True, False, 1.0, "1", None, 2])
def test_direct_cursor_schema_is_exact_integer_version(schema):
    with pytest.raises(ValueError, match="schema_version"):
        TrainingCursorV1(**(_fields() | {"schema_version": schema}))


@pytest.mark.parametrize("field", ["epoch", "token_window_offset", "batch_index"])
@pytest.mark.parametrize("value", [True, False, 0.0, 0.5, float("nan"), "0", None, -1])
def test_direct_cursor_counters_cannot_coerce_or_hide_fractional_progress(field, value):
    with pytest.raises(ValueError, match="cursor"):
        TrainingCursorV1(**(_fields() | {field: value}))


@pytest.mark.parametrize("field", ["document_id", "batch_id"])
@pytest.mark.parametrize("value", ["", 17, True, None, ["identity"]])
def test_direct_cursor_identity_fields_are_nonempty_actual_strings(field, value):
    with pytest.raises(ValueError, match="cursor"):
        TrainingCursorV1(**(_fields() | {field: value}))


@pytest.mark.parametrize("value", [(), [], ["sample"], "sample", None, ("",), ("ok", None)])
def test_direct_cursor_sample_ids_are_a_nonempty_immutable_tuple_of_strings(value):
    with pytest.raises(ValueError, match="cursor"):
        TrainingCursorV1(**(_fields() | {"next_sample_ids": value}))


def _source(kind):
    if kind == "synthetic":
        return SyntheticBatchSource(config(vocab_size=24))
    return PreparedDatasetBatchSource(
        config(),
        [Document("only", "abcdefgh")],
        data_fingerprint="d" * 64,
        tokenizer_fingerprint="t" * 64,
    )


@pytest.mark.parametrize("kind", ["synthetic", "prepared"])
@pytest.mark.parametrize("value", [True, False, 1.0, 0.5, float("nan"), "1", None, -1])
def test_cursor_at_validates_before_indexing_or_formatting(kind, value, monkeypatch):
    source = _source(kind)
    rng = capture_rng_state()
    monkeypatch.setattr(source, "_cursor", lambda _: pytest.fail("invalid index used"))
    with pytest.raises(ValueError, match="batch_index"):
        source.cursor_at(value)
    assert state_trees_equal(rng, capture_rng_state())


@pytest.mark.parametrize("kind", ["synthetic", "prepared"])
@pytest.mark.parametrize("index", [0, 1, 7, 10**9])
def test_valid_indexed_cursors_remain_roundtrippable_and_keep_existing_batch_ids(kind, index):
    source = _source(kind)
    cursor = source.cursor_at(index)
    assert TrainingCursorV1.from_dict(cursor.to_dict()) == cursor
    expected = hashlib.sha256(
        f"{index}\n".encode() + "\n".join(cursor.next_sample_ids).encode()
    ).hexdigest()
    assert cursor.batch_id == expected
    assert source.batch(cursor).batch_id == expected


def test_valid_direct_constructor_roundtrips_without_changing_values():
    cursor = TrainingCursorV1(**_fields())
    assert cursor == TrainingCursorV1.from_dict(cursor.to_dict())
    assert cursor.to_dict() == (_fields() | {"next_sample_ids": ["sample"]})
