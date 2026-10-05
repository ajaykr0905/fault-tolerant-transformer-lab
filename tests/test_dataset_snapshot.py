import gzip
import hashlib
import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from test_real_data_recovery import FIXTURE, prepared_fixture

from fttl import dataset
from fttl.config import ExperimentConfig, ModelConfig
from fttl.data import PreparedDatasetBatchSource
from fttl.evaluation import evaluate_held_out
from fttl.model import TinyTransformer
from fttl.state import state_digest


def alternate_fixture(root):
    archive = gzip.compress(FIXTURE.read_bytes().replace(b"Proposal:", b"Altered:"), mtime=0)
    source = dataset.DatasetSource(
        repository="fttl://tests/public-domain-pep-fixture",
        revision="fixture-v2",
        path="public_domain_peps_fixture.jsonl.gz",
        compressed_sha256=hashlib.sha256(archive).hexdigest(),
        expected_document_count=6,
        license="CC0-1.0",
        license_limitations=("Independently written CI fixture.",),
    )

    def fetch(_url, destination):
        destination.write_bytes(archive)

    dataset.prepare_peps(root / "cache", root / "prepared", downloader=fetch, source=source)
    return root / "prepared" / "manifest.json"


def snapshot_config():
    return ExperimentConfig(
        model=ModelConfig(vocab_size=257, block_size=4, d_model=8, n_heads=2, n_layers=1),
        seed=73,
        steps=2,
        batch_size=2,
    )


def switch_directory(live, destination):
    replacement = live.parent / "next-live"
    replacement.symlink_to(destination, target_is_directory=True)
    os.replace(replacement, live)


def swap_after_document_open(monkeypatch, live, first, second):
    original_open = Path.open
    swapped = []

    def raced_open(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        if path == first.parent / "documents.jsonl" and not swapped:
            switch_directory(live, second.parent)
            swapped.append(True)
        return handle

    monkeypatch.setattr(Path, "open", raced_open)
    return swapped


@pytest.mark.parametrize("consumer", ["documents", "training", "evaluation"])
def test_consumers_use_the_bytes_they_verified_not_a_reopened_dataset(
    tmp_path, monkeypatch, consumer
):
    first = prepared_fixture(tmp_path / "first")
    second = alternate_fixture(tmp_path / "second")
    config = snapshot_config()
    reference_documents = dataset.load_prepared_documents(first, split="train")
    reference_source = PreparedDatasetBatchSource.from_manifest(config, first)
    model = TinyTransformer(config.model)
    reference_evaluation = evaluate_held_out(model, first, max_tokens=13)
    live = tmp_path / "live"
    live.symlink_to(first.parent, target_is_directory=True)
    swapped = swap_after_document_open(monkeypatch, live, first, second)
    path = live / "manifest.json"
    if consumer == "documents":
        assert dataset.load_prepared_documents(path, split="train") == reference_documents
    elif consumer == "training":
        source = PreparedDatasetBatchSource.from_manifest(config, path)
        assert source.data_fingerprint == reference_source.data_fingerprint
        assert state_digest(source.batch(source.initial_cursor()).inputs) == state_digest(
            reference_source.batch(reference_source.initial_cursor()).inputs
        )
    else:
        assert evaluate_held_out(model, path, max_tokens=13) == reference_evaluation
    assert swapped == [True]


def test_snapshot_metadata_and_documents_cannot_mutate_its_verified_identity(tmp_path):
    path = prepared_fixture(tmp_path)
    snapshot = dataset.load_dataset_snapshot(path)
    original = snapshot.manifest.to_dict()
    copy = snapshot.manifest
    copy.source["revision"] = "changed"
    copy.license["limitations"].append("changed")
    copy.splits["train"]["document_ids"].clear()
    assert snapshot.manifest.to_dict() == original
    with pytest.raises(FrozenInstanceError):
        snapshot.documents = ()
    with pytest.raises(FrozenInstanceError):
        snapshot.documents[0].text = "changed"
    assert snapshot.documents_for(None) == snapshot.documents
    with pytest.raises(dataset.DatasetValidationError, match="unknown dataset split"):
        snapshot.documents_for("unknown")


def test_constructor_cannot_bind_verified_manifest_to_other_document_bytes(tmp_path):
    first = prepared_fixture(tmp_path / "first")
    second = alternate_fixture(tmp_path / "second")
    with pytest.raises(dataset.DatasetValidationError, match="SHA-256|byte length"):
        dataset.PreparedDatasetSnapshot(
            first.read_bytes(), (second.parent / "documents.jsonl").read_bytes()
        )


def test_snapshot_constructor_rejects_mutable_input_bytes(tmp_path):
    path = prepared_fixture(tmp_path)
    with pytest.raises(dataset.DatasetValidationError, match="immutable bytes"):
        dataset.PreparedDatasetSnapshot(
            path.read_bytes(), bytearray((path.parent / "documents.jsonl").read_bytes())
        )


def test_capture_parses_only_the_document_bytes_it_hashed(tmp_path, monkeypatch):
    first = prepared_fixture(tmp_path / "first")
    expected = dataset.load_dataset_snapshot(first)
    second = alternate_fixture(tmp_path / "second")
    replacement = (second.parent / "documents.jsonl").read_bytes()
    original_read = Path.read_bytes

    def replaced_after_read(path):
        captured = original_read(path)
        if path == first.parent / "documents.jsonl":
            path.write_bytes(replacement)
        return captured

    monkeypatch.setattr(Path, "read_bytes", replaced_after_read)
    snapshot = dataset.load_dataset_snapshot(first)
    assert snapshot.documents == expected.documents
    assert snapshot.manifest.to_dict() == expected.manifest.to_dict()


def test_snapshot_handles_unicode_line_separators_inside_json_text(tmp_path):
    import json

    path = prepared_fixture(tmp_path)
    snapshot = dataset.load_dataset_snapshot(path)
    payload = snapshot.manifest.to_dict()
    rows = [json.loads(line) for line in (path.parent / "documents.jsonl").read_text().splitlines()]
    for row in rows:
        row["text"] = row["text"].replace("Proposal:", "Proposal:\u2028")
        row["text_sha256"] = hashlib.sha256(row["text"].encode()).hexdigest()
    content = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode() + b"\n" for row in rows
    )
    payload["documents"]["byte_length"] = len(content)
    payload["documents"]["sha256"] = hashlib.sha256(content).hexdigest()
    payload["counts"]["utf8_bytes"] += 3 * len(rows)
    payload["counts"]["tokens_including_eod"] += 3 * len(rows)
    payload.pop("dataset_fingerprint")
    payload["dataset_fingerprint"] = dataset._sha256_json(payload)
    changed = dataset.PreparedDatasetSnapshot(json.dumps(payload).encode(), content)
    assert all("\u2028" in document.text for document in changed.documents)
