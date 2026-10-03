from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import fttl.checkpoint as checkpoint_module
from fttl.checkpoint import (
    CheckpointIntegrityError,
    CheckpointMismatchError,
    load_checkpoint,
    save_checkpoint,
)
from fttl.config import ExperimentConfig, ModelConfig
from fttl.model import TinyTransformer
from fttl.state import capture_rng_state, state_trees_equal


def checkpoint_config() -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=24,
            block_size=6,
            d_model=12,
            n_heads=3,
            n_layers=1,
            dropout=0.1,
        ),
        seed=7,
        steps=4,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )


def model_and_optimizer(
    config: ExperimentConfig,
) -> tuple[TinyTransformer, torch.optim.Optimizer]:
    model = TinyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer


def publish(
    store: Path,
    config: ExperimentConfig,
    *,
    step: int,
    marker: float,
    failure_injector=None,
):
    model, optimizer = model_and_optimizer(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(marker)
    return save_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        config=config,
        step=step,
        tokens_seen=step * 12,
        losses=[2.0 / value for value in range(1, step + 1)],
        cursor={"batch_id": step, "next_sample_ids": [f"sample-{step}"]},
        batch_ids=[f"batch-{value}" for value in range(step)],
        data_fingerprint="dataset-a",
        tokenizer_fingerprint="utf8-byte-v1",
        failure_injector=failure_injector,
    )


def load(
    store: Path,
    config: ExperimentConfig,
    *,
    data_fingerprint: str = "dataset-a",
    tokenizer_fingerprint: str = "utf8-byte-v1",
    run_contract_fingerprint: str | None = None,
):
    model, optimizer = model_and_optimizer(config)
    payload = load_checkpoint(
        store,
        model=model,
        optimizer=optimizer,
        expected_config=config,
        expected_data_fingerprint=data_fingerprint,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_run_contract_fingerprint=run_contract_fingerprint,
    )
    return model, payload


def test_manifest_binds_generation_state_and_inputs(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    manifest = publish(store, config, step=1, marker=0.25)

    generation = store / "generation-00000001"
    on_disk = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert (store / "LATEST").read_text(encoding="utf-8").strip() == generation.name
    assert manifest.schema_version == on_disk["schema_version"] == 2
    assert manifest.state_bytes == (generation / "state.pt").stat().st_size
    assert len(manifest.state_sha256) == 64
    assert manifest.completed_step == 1
    assert manifest.cursor == {"batch_id": 1, "next_sample_ids": ["sample-1"]}
    assert manifest.data_fingerprint == "dataset-a"
    assert manifest.tokenizer_fingerprint == "utf8-byte-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("step", 1.5),
        ("step", True),
        ("step", -1),
        ("tokens_seen", 12.5),
        ("tokens_seen", False),
        ("retention", 2.5),
        ("retention", True),
        ("retention", 1),
        ("cursor", ["unexpected"]),
        ("cursor", {"offset": float("nan")}),
        ("data_fingerprint", 123),
        ("tokenizer_fingerprint", True),
        ("run_contract_fingerprint", ""),
    ],
)
def test_invalid_save_metadata_preserves_store(tmp_path: Path, field, value):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    before = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    model, optimizer = model_and_optimizer(config)
    arguments = dict(
        step=2,
        tokens_seen=24,
        losses=[2.0, 1.0],
        cursor={"batch_id": 2},
        data_fingerprint="dataset-a",
        tokenizer_fingerprint="utf8-byte-v1",
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        save_checkpoint(store, model=model, optimizer=optimizer, config=config, **arguments)
    after = {
        path.relative_to(store): path.read_bytes() for path in store.rglob("*") if path.is_file()
    }
    assert after == before
    assert load(store, config)[1]["step"] == 1


def test_invalid_cursor_does_not_create_store(tmp_path: Path):
    config = checkpoint_config()
    model, optimizer = model_and_optimizer(config)
    store = tmp_path / "missing"
    with pytest.raises(ValueError, match="cursor"):
        save_checkpoint(
            store,
            model=model,
            optimizer=optimizer,
            config=config,
            step=0,
            tokens_seen=0,
            losses=[],
            cursor=["unexpected"],
        )
    assert not store.exists()


def test_corrupt_newest_generation_falls_back_to_previous(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.125)
    publish(store, config, step=2, marker=0.875)
    newest_state = store / "generation-00000002" / "state.pt"
    newest_state.write_bytes(newest_state.read_bytes()[:64])

    model, payload = load(store, config)

    assert payload["step"] == 1
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.125),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("error_type", [PermissionError, OSError])
@pytest.mark.parametrize("all_unreadable", [False, True])
def test_state_read_errors_use_integrity_fallback(
    tmp_path: Path, monkeypatch, error_type, all_unreadable: bool
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.125)
    publish(store, config, step=2, marker=0.875)
    original = checkpoint_module._sha256

    def fail_read(path):
        if all_unreadable or path.parent.name == "generation-00000002":
            raise error_type("simulated state read failure")
        return original(path)

    monkeypatch.setattr(checkpoint_module, "_sha256", fail_read)
    if all_unreadable:
        with pytest.raises(CheckpointIntegrityError, match="no checkpoint generation"):
            load(store, config)
    else:
        model, payload = load(store, config)
        assert payload["step"] == payload["selected_generation"] == 1
        for parameter in model.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.125), rtol=0, atol=0)
    assert (store / "LATEST").read_text().strip() == "generation-00000002"


@pytest.mark.parametrize("invalid_component", ["model", "optimizer", "rng_state"])
def test_rejected_payload_preserves_caller_training_objects(tmp_path: Path, invalid_component):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.125)
    generation = store / "generation-00000001"
    state_path = generation / "state.pt"
    payload = torch.load(state_path, weights_only=True)
    if invalid_component == "model":
        payload["model"]["blocks.0.mlp.0.weight"] = torch.zeros(1)
    elif invalid_component == "optimizer":
        payload["optimizer"]["param_groups"] = []
    else:
        payload["rng_state"]["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    torch.save(payload, state_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest["state_bytes"] = state_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest))
    model, optimizer = model_and_optimizer(config)
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_optimizer = optimizer.state_dict()
    before_rng = capture_rng_state()
    with pytest.raises((CheckpointMismatchError, CheckpointIntegrityError)):
        load_checkpoint(store, model=model, optimizer=optimizer, expected_config=config)
    assert state_trees_equal(before_model, model.state_dict())
    assert state_trees_equal(before_optimizer, optimizer.state_dict())
    assert state_trees_equal(before_rng, capture_rng_state())


def test_manifest_corruption_falls_back_and_allows_the_next_commit(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.125)
    publish(store, config, step=2, marker=0.875)
    newest_manifest = store / "generation-00000002" / "manifest.json"
    newest_manifest.write_text("{\"truncated\":", encoding="utf-8")

    model, payload = load(store, config)
    assert payload["step"] == 1
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.125),
            rtol=0,
            atol=0,
        )

    publish(store, config, step=2, marker=0.5)

    assert (store / "LATEST").read_text(encoding="utf-8").strip() == (
        "generation-00000003"
    )
    assert sorted(path.name for path in store.glob("generation-*")) == [
        "generation-00000001",
        "generation-00000003",
    ]
    _, resumed = load(store, config)
    assert resumed["step"] == 2


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("config", "configuration does not match"),
        ("data", "data fingerprint does not match"),
        ("tokenizer", "tokenizer fingerprint does not match"),
        ("run", "run contract fingerprint does not match"),
    ],
)
def test_valid_checkpoint_rejects_contract_drift(
    tmp_path: Path, change: str, message: str
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    expected_config = config
    data = "dataset-a"
    tokenizer = "utf8-byte-v1"
    run_contract = None
    if change == "config":
        expected_config = ExperimentConfig(
            model=config.model,
            seed=config.seed,
            steps=config.steps,
            batch_size=config.batch_size,
            learning_rate=5e-4,
            checkpoint_every=config.checkpoint_every,
        )
    elif change == "data":
        data = "dataset-b"
    elif change == "tokenizer":
        tokenizer = "utf8-byte-v2"
    else:
        run_contract = "different-run-contract"

    with pytest.raises(CheckpointMismatchError, match=message):
        load(
            store,
            expected_config,
            data_fingerprint=data,
            tokenizer_fingerprint=tokenizer,
            run_contract_fingerprint=run_contract,
        )


def test_store_retains_two_newest_valid_generations(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    for step in range(1, 5):
        publish(store, config, step=step, marker=step / 10)

    generations = sorted(path.name for path in store.glob("generation-*"))
    assert generations == ["generation-00000003", "generation-00000004"]
    assert (store / "LATEST").read_text(encoding="utf-8").strip() == generations[-1]


def test_crash_before_latest_update_does_not_advance_committed_pointer(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)

    def crash(stage: str) -> None:
        if stage == "before-latest-publish":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        publish(store, config, step=2, marker=0.75, failure_injector=crash)

    assert (store / "LATEST").read_text(encoding="utf-8").strip() == (
        "generation-00000001"
    )
    model, payload = load(store, config)
    assert payload["step"] == 1
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.25),
            rtol=0,
            atol=0,
        )


def test_uncommitted_generation_never_becomes_a_later_fallback(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)

    def crash(stage: str) -> None:
        if stage == "before-latest-publish":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        publish(store, config, step=2, marker=0.50, failure_injector=crash)

    publish(store, config, step=3, marker=0.75)
    generations = sorted(path.name for path in store.glob("generation-*"))
    assert generations == ["generation-00000001", "generation-00000002"]
    newest = store / "generation-00000002" / "state.pt"
    newest.write_bytes(newest.read_bytes()[:64])

    model, payload = load(store, config)
    assert payload["step"] == 1
    for parameter in model.parameters():
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, 0.25),
            rtol=0,
            atol=0,
        )


def test_store_without_latest_never_loads_a_published_but_uncommitted_generation(
    tmp_path: Path,
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"

    def crash(stage: str) -> None:
        if stage == "before-latest-publish":
            raise RuntimeError("injected crash")

    with pytest.raises(RuntimeError, match="injected crash"):
        publish(store, config, step=1, marker=0.5, failure_injector=crash)

    with pytest.raises(CheckpointIntegrityError, match="no committed generations"):
        load(store, config)


def test_save_fails_closed_when_latest_pointer_is_malformed(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    (store / "LATEST").write_text("not-a-generation\n", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="LATEST pointer is malformed"):
        publish(store, config, step=2, marker=0.75)

    assert (store / "generation-00000001").is_dir()


def test_save_rejects_mixing_training_contracts_in_one_store(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    model, optimizer = model_and_optimizer(config)

    with pytest.raises(CheckpointMismatchError, match="data fingerprint"):
        save_checkpoint(
            store,
            model=model,
            optimizer=optimizer,
            config=config,
            step=2,
            tokens_seen=24,
            losses=[2.0, 1.0],
            data_fingerprint="dataset-b",
            tokenizer_fingerprint="utf8-byte-v1",
        )

    assert sorted(path.name for path in store.glob("generation-*")) == [
        "generation-00000001"
    ]


def test_temp_write_crash_never_replaces_valid_generation(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)

    def crash(stage: str) -> None:
        if stage == "after-state-serialize":
            raise RuntimeError("injected temp-write crash")

    with pytest.raises(RuntimeError, match="temp-write crash"):
        publish(store, config, step=2, marker=0.75, failure_injector=crash)

    _, payload = load(store, config)
    assert payload["step"] == 1
    assert list(store.glob(".generation-*.tmp-*"))


def test_length_validation_happens_before_torch_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    state_path = store / "generation-00000001" / "state.pt"
    state_path.write_bytes(state_path.read_bytes()[:-1])
    called = False

    def forbidden_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(CheckpointIntegrityError, match="no checkpoint generation"):
        load(store, config)
    assert called is False


def test_checkpoint_deserialization_is_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    original_load = torch.load
    observed: list[bool | None] = []

    def recording_load(*args, **kwargs):
        observed.append(kwargs.get("weights_only"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    _, payload = load(store, config)
    assert payload["step"] == 1
    assert observed == [True]


def test_serialized_config_is_checked_even_when_fingerprints_are_unchanged(
    tmp_path: Path,
):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    generation = store / "generation-00000001"
    state_path = generation / "state.pt"
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    payload["config"]["seed"] += 1
    torch.save(payload, state_path)
    manifest_path = generation / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["state_bytes"] = state_path.stat().st_size
    manifest["state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointMismatchError, match="serialized configuration"):
        load(store, config)


def test_next_save_cleans_stale_latest_temporary_file(tmp_path: Path):
    config = checkpoint_config()
    store = tmp_path / "checkpoints"
    publish(store, config, step=1, marker=0.25)
    stale = store / ".LATEST.tmp-abandoned"
    stale.write_text("generation-99999999\n", encoding="utf-8")

    publish(store, config, step=2, marker=0.5)

    assert not stale.exists()
