import json
from dataclasses import replace

import pytest
import torch
from test_real_data_recovery import prepared_fixture
from test_training_cursor import Document, config

from fttl import data
from fttl.dataset import _sha256_json, load_dataset_snapshot


def _config(*, vocab_size=257, block_size=4, batch_size=1):
    original = config(vocab_size=vocab_size)
    return replace(
        original, model=replace(original.model, block_size=block_size), batch_size=batch_size
    )


def _prepared(
    *,
    text="abcdefghijklmnopqrstuvwx",
    fingerprint="d" * 64,
    tokenizer="t" * 64,
    block_size=4,
    batch_size=1,
):
    return data.PreparedDatasetBatchSource(
        _config(block_size=block_size, batch_size=batch_size),
        [Document("same-document", text)],
        data_fingerprint=fingerprint,
        tokenizer_fingerprint=tokenizer,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"fingerprint": "e" * 64},
        {"tokenizer": "u" * 64},
        {"text": "qrstuvwxyzabcdefghijklmn"},
        {"text": "abcdefghijklmnopqrstuvwz"},
        {"block_size": 8},
        {"batch_size": 2},
    ],
)
def test_prepared_initial_cursors_bind_the_complete_source_and_batching_contract(change):
    first = _prepared()
    second = _prepared(**change)
    cursor = first.initial_cursor()
    assert cursor.batch_id != second.initial_cursor().batch_id
    with pytest.raises(ValueError, match="cursor"):
        second.batch(cursor)


def test_foreign_prepared_cursor_rejects_before_constructing_different_token_tensors(monkeypatch):
    first = _prepared()
    second = _prepared(text="qrstuvwxyzabcdefghijklmn")
    cursor = first.initial_cursor()
    monkeypatch.setattr(data.torch, "tensor", lambda *a, **k: pytest.fail("foreign tokens emitted"))
    with pytest.raises(ValueError, match="cursor"):
        second.batch(cursor)


@pytest.mark.parametrize("change", [{"vocab_size": 32}, {"block_size": 8}, {"batch_size": 2}])
def test_synthetic_initial_cursor_cannot_cross_data_or_batching_contracts(change):
    first = data.SyntheticBatchSource(_config(vocab_size=24))
    second = data.SyntheticBatchSource(_config(**({"vocab_size": 24} | change)))
    cursor = first.initial_cursor()
    assert cursor.batch_id != second.initial_cursor().batch_id
    with pytest.raises(ValueError, match="cursor"):
        second.batch(cursor)


def test_synthetic_public_token_mutation_invalidates_previously_issued_cursor(monkeypatch):
    source = data.SyntheticBatchSource(_config(vocab_size=24))
    cursor = source.initial_cursor()
    source.tokens[-1] = (source.tokens[-1] + 1) % 24
    monkeypatch.setattr(data, "batch_for_step", lambda *a: pytest.fail("changed tokens emitted"))
    with pytest.raises(ValueError, match="cursor"):
        source.batch(cursor)


def test_direct_constructor_captures_documents_and_does_not_follow_later_caller_mutation():
    document = Document("same-document", "abcdefghijklmnopqrstuvwx")
    source = data.PreparedDatasetBatchSource(
        _config(), [document], data_fingerprint="d" * 64, tokenizer_fingerprint="t" * 64
    )
    cursor = source.initial_cursor()
    before = source.batch(cursor)
    document.text = "caller changed this object"
    after = source.batch(cursor)
    assert cursor == source.initial_cursor()
    assert torch.equal(before.inputs, after.inputs)


@pytest.mark.parametrize("kind", ["synthetic", "prepared"])
def test_matching_sources_keep_stable_ids_and_isolate_contract_metadata(kind):
    if kind == "synthetic":
        first = data.SyntheticBatchSource(_config(vocab_size=24))
        second = data.SyntheticBatchSource(_config(vocab_size=24))
    else:
        first, second = _prepared(), _prepared()
    assert first.batch_identity_contract == second.batch_identity_contract
    contract = first.batch_identity_contract
    contract["data_fingerprint"] = "changed"
    assert first.batch_identity_contract["data_fingerprint"] != "changed"
    for index in (0, 1, 100):
        assert first.cursor_at(index) == second.cursor_at(index)
        assert (
            first.batch(first.cursor_at(index)).batch_id
            == second.batch(second.cursor_at(index)).batch_id
        )


def test_prepared_protocol_fields_are_captured_once_for_identity_and_windows():
    class ReadOnceDocument:
        identity_reads = 0
        text_reads = 0

        @property
        def document_id(self):
            self.identity_reads += 1
            return "same-document" if self.identity_reads == 1 else "changed-on-read"

        @property
        def text(self):
            self.text_reads += 1
            return "abcdefghijklmnopqrstuvwx" if self.text_reads == 1 else "changed-on-read"

    document = ReadOnceDocument()
    source = data.PreparedDatasetBatchSource(
        _config(), [document], data_fingerprint="d" * 64, tokenizer_fingerprint="t" * 64
    )
    expected = _prepared()
    assert document.identity_reads == document.text_reads == 1
    assert source.batch_identity_contract == expected.batch_identity_contract
    assert source.initial_cursor() == expected.initial_cursor()
    assert torch.equal(
        source.batch(source.initial_cursor()).inputs,
        expected.batch(expected.initial_cursor()).inputs,
    )
    assert document.identity_reads == document.text_reads == 1


def test_synthetic_validation_and_emission_share_the_same_captured_content(monkeypatch):
    source = data.SyntheticBatchSource(_config(vocab_size=24))
    reference = data.SyntheticBatchSource(_config(vocab_size=24))
    cursor = source.initial_cursor()
    expected = reference.batch(reference.initial_cursor())
    original_emit = data.batch_for_step
    calls = []

    def mutate_after_validation(captured, config, index):
        calls.append(True)
        assert captured.data_ptr() != source.tokens.data_ptr()
        source.tokens[0] = (source.tokens[0] + 1) % 24
        return original_emit(captured, config, index)

    monkeypatch.setattr(data, "batch_for_step", mutate_after_validation)
    batch = source.batch(cursor)
    assert torch.equal(batch.inputs, expected.inputs)
    assert torch.equal(batch.targets, expected.targets)
    assert batch.next_cursor == expected.next_cursor
    with pytest.raises(ValueError, match="cursor"):
        source.batch(batch.next_cursor)
    assert calls == [True]


def test_manifest_revision_alone_changes_batch_binding_with_identical_documents(tmp_path):
    path = prepared_fixture(tmp_path)
    original = load_dataset_snapshot(path)
    source = data.PreparedDatasetBatchSource.from_snapshot(_config(), original)
    payload = json.loads(path.read_text())
    payload["source"]["revision"] = "fixture-v2-metadata-only"
    payload.pop("dataset_fingerprint")
    payload["dataset_fingerprint"] = _sha256_json(payload)
    path.write_text(json.dumps(payload))
    changed = load_dataset_snapshot(path)
    other = data.PreparedDatasetBatchSource.from_snapshot(_config(), changed)
    assert original.documents == changed.documents
    assert source.initial_cursor().next_sample_ids == other.initial_cursor().next_sample_ids
    assert (
        source.batch_identity_contract["content_fingerprint"]
        == other.batch_identity_contract["content_fingerprint"]
    )
    assert source.initial_cursor().batch_id != other.initial_cursor().batch_id
    with pytest.raises(ValueError, match="cursor"):
        other.batch(source.initial_cursor())
