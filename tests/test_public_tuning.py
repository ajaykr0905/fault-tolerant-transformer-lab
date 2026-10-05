import hashlib
import json
import random
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from test_real_data_recovery import prepared_fixture

from fttl import public_tuning
from fttl.config import ExperimentConfig, ModelConfig
from fttl.dataset import DatasetValidationError, load_prepared_documents
from fttl.state import capture_rng_state, state_trees_equal


def _config(dropout=0.35):
    return ExperimentConfig(
        model=ModelConfig(
            vocab_size=257, block_size=4, d_model=8, n_heads=2, n_layers=1, dropout=dropout
        ),
        seed=73,
        steps=3,
        batch_size=2,
        learning_rate=1e-3,
        checkpoint_every=1,
    )


def _without_observation_time(comparison):
    values = comparison.to_dict()
    for mode in ("full", "lora"):
        values[mode].pop("elapsed_seconds")
    return values


@pytest.mark.parametrize("dropout", [0.0, 0.35])
def test_public_comparison_binds_paired_training_and_validation_evidence(tmp_path, dropout):
    manifest = prepared_fixture(tmp_path / "data")
    output = tmp_path / "reports" / "comparison.json"
    config = _config(dropout)
    result = public_tuning.compare_public_tuning(
        config, manifest, output, rank=2, max_eval_tokens=13
    )
    assert result.schema_version == 1
    assert result.config_fingerprint == config.fingerprint()
    assert result.device == "cpu"
    assert (
        result.matched_initial_predictions
        and result.paired_batches
        and result.paired_cpu_rng_sequence
    )
    assert result.full.initial_loss == result.lora.initial_loss
    assert result.full.initial_logits_digest == result.lora.initial_logits_digest
    assert (
        result.full.base_state_digest_before
        == result.lora.base_state_digest_before
        == result.base_state_digest
    )
    assert result.lora.base_state_unchanged and result.lora.base_parameters_frozen
    assert result.lora.base_state_digest_after == result.base_state_digest
    assert not result.full.base_state_unchanged and not result.full.base_parameters_frozen
    assert result.full.base_state_digest_after != result.base_state_digest
    assert 0 < result.lora.trainable_parameters < result.full.trainable_parameters
    train_ids = {
        document.document_id for document in load_prepared_documents(manifest, split="train")
    }
    held_out_ids = {
        document.document_id
        for split in ("validation", "test")
        for document in load_prepared_documents(manifest, split=split)
    }
    for run in (result.full, result.lora):
        assert run.completed_steps == config.steps
        assert run.target_tokens == config.steps * config.batch_size * config.model.block_size
        assert run.elapsed_seconds >= 0
        assert run.model_state_digest_before != run.model_state_digest_after
        for step_number, step in enumerate(run.trace, 1):
            assert step.step == step_number
            assert step.target_tokens == config.batch_size * config.model.block_size
            assert step.cumulative_target_tokens == step_number * step.target_tokens
            assert len(step.windows) == config.batch_size
            assert all(window.document_id in train_ids - held_out_ids for window in step.windows)
            assert all(
                window.sample_id == f"{window.document_id}:{window.token_window_offset:012d}"
                for window in step.windows
            )
        for evaluation in (run.validation_before, run.validation_after):
            assert evaluation.split == "validation"
            assert evaluation.evaluated_target_tokens == 13
            assert evaluation.data_fingerprint == result.data_fingerprint
            assert evaluation.tokenizer_fingerprint == result.tokenizer_fingerprint
        assert run.validation_after.model_state_digest == run.model_state_digest_after
    contract = result.comparison_contract
    assert contract["dataset"]["training_split"] == "train"
    assert contract["dataset"]["source"]["repository"] == "fttl://tests/public-domain-pep-fixture"
    assert contract["evaluation"]["split"] == "validation"
    assert contract["evaluation"]["max_target_tokens"] == 13
    assert contract["base"] == {
        "initialization": "random-seeded",
        "state_digest": result.base_state_digest,
    }
    assert contract["adapter"]["rank"] == 2
    assert contract["adapter"]["replaced_modules"] == [
        "blocks.0.attention.qkv",
        "blocks.0.attention.output",
    ]
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert result.comparison_fingerprint == hashlib.sha256(encoded.encode()).hexdigest()
    assert json.loads(output.read_text()) == json.loads(json.dumps(asdict(result)))
    assert any("not pretrained" in value for value in result.limitations)
    assert any("not a test-set" in value for value in result.limitations)


def test_public_comparison_repeats_independently_of_caller_rng(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    first = public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "one.json", rank=2, max_eval_tokens=13
    )
    random.random()
    np.random.random(19)
    torch.rand(17)
    second = public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "two.json", rank=2, max_eval_tokens=13
    )
    assert _without_observation_time(first) == _without_observation_time(second)


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, True)])
def test_success_preserves_cpu_rng_and_deterministic_settings(tmp_path, enabled, warn_only):
    manifest = prepared_fixture(tmp_path / "data")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        rng = capture_rng_state()
        public_tuning.compare_public_tuning(
            _config(), manifest, tmp_path / "comparison.json", rank=2, max_eval_tokens=13
        )
        assert state_trees_equal(rng, capture_rng_state())
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_each_training_dropout_replays_the_same_cpu_generator_state(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning._train_arm
    observed = {}

    def observed_train(model, *args, mode, **kwargs):
        observed[mode] = []

        def record_rng(module, inputs):
            if module.training:
                observed[mode].append(torch.get_rng_state().clone())

        hooks = [
            module.register_forward_pre_hook(record_rng)
            for module in model.modules()
            if isinstance(module, torch.nn.Dropout)
        ]
        try:
            return original(model, *args, mode=mode, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

    monkeypatch.setattr(public_tuning, "_train_arm", observed_train)
    public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "comparison.json", rank=2, max_eval_tokens=13
    )
    assert len(observed["full"]) == len(observed["lora"]) == 6
    assert all(
        torch.equal(full, lora)
        for full, lora in zip(observed["full"], observed["lora"], strict=True)
    )


def test_lora_base_parameters_have_no_gradients_and_never_change(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning._train_arm
    checked = []

    def observed_train(model, *args, mode, **kwargs):
        base = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"}
        }
        result = original(model, *args, mode=mode, **kwargs)
        if mode == "lora":
            for name, parameter in model.named_parameters():
                if name in base:
                    assert not parameter.requires_grad
                    assert parameter.grad is None
                    assert torch.equal(parameter, base[name])
            checked.append(True)
        return result

    monkeypatch.setattr(public_tuning, "_train_arm", observed_train)
    public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "comparison.json", rank=2, max_eval_tokens=13
    )
    assert checked == [True]


@pytest.mark.parametrize(
    "options,message",
    [
        ({"rank": 0}, "rank"),
        ({"rank": -1}, "rank"),
        ({"rank": True}, "rank"),
        ({"rank": 1.5}, "rank"),
        ({"rank": 9}, "rank"),
        ({"rank": "2"}, "rank"),
        ({"max_eval_tokens": 0}, "max_eval_tokens"),
        ({"max_eval_tokens": -1}, "max_eval_tokens"),
        ({"max_eval_tokens": True}, "max_eval_tokens"),
        ({"max_eval_tokens": 1.5}, "max_eval_tokens"),
        ({"max_eval_tokens": "13"}, "max_eval_tokens"),
    ],
)
def test_invalid_controls_reject_before_loading_or_initializing(
    monkeypatch, tmp_path, options, message
):
    monkeypatch.setattr(
        public_tuning, "TinyTransformer", lambda _: pytest.fail("model initialized")
    )
    rng = capture_rng_state()
    output = tmp_path / "new" / "comparison.json"
    with pytest.raises(ValueError, match=message):
        public_tuning.compare_public_tuning(_config(), tmp_path / "missing", output, **options)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_byte_vocabulary_required_before_any_dataset_access(tmp_path):
    config = replace(_config(), model=replace(_config().model, vocab_size=256))
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="vocab_size=257"):
        public_tuning.compare_public_tuning(config, tmp_path / "missing", tmp_path / "new" / "out")
    assert state_trees_equal(rng, capture_rng_state())
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize("kind", ["file", "directory", "dangling-symlink"])
def test_existing_output_is_preserved_before_model_or_dataset_access(monkeypatch, tmp_path, kind):
    output = tmp_path / "comparison.json"
    if kind == "file":
        output.write_bytes(b"previous experiment")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "missing-target")
    monkeypatch.setattr(
        public_tuning, "TinyTransformer", lambda _: pytest.fail("model initialized")
    )
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="fresh output"):
        public_tuning.compare_public_tuning(_config(), tmp_path / "missing", output)
    assert state_trees_equal(rng, capture_rng_state())
    if kind == "file":
        assert output.read_bytes() == b"previous experiment"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()


@pytest.mark.parametrize("failed_mode", ["full", "lora"])
def test_training_failure_restores_rng_settings_and_leaves_no_report(
    monkeypatch, tmp_path, failed_mode
):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning._train_arm

    def fail_training(*args, mode, **kwargs):
        if mode == failed_mode:
            random.random()
            np.random.random()
            torch.rand(11)
            raise RuntimeError("injected training failure")
        return original(*args, mode=mode, **kwargs)

    monkeypatch.setattr(public_tuning, "_train_arm", fail_training)
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(False)
        rng = capture_rng_state()
        output = tmp_path / "new" / "comparison.json"
        with pytest.raises(RuntimeError, match="injected training failure"):
            public_tuning.compare_public_tuning(
                _config(), manifest, output, rank=2, max_eval_tokens=13
            )
        assert state_trees_equal(rng, capture_rng_state())
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not output.parent.exists()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_non_finite_training_loss_is_never_published(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning.TinyTransformer.forward

    def invalid_loss(model, tokens, targets=None):
        logits, loss = original(model, tokens, targets)
        return logits, None if loss is None else loss * float("nan")

    monkeypatch.setattr(public_tuning.TinyTransformer, "forward", invalid_loss)
    rng = capture_rng_state()
    output = tmp_path / "new" / "comparison.json"
    with pytest.raises(FloatingPointError, match="training loss"):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2, max_eval_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_non_finite_optimizer_update_is_never_published(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = torch.optim.AdamW.step

    def invalid_update(optimizer, *args, **kwargs):
        result = original(optimizer, *args, **kwargs)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("inf"))
        return result

    monkeypatch.setattr(torch.optim.AdamW, "step", invalid_update)
    rng = capture_rng_state()
    output = tmp_path / "new" / "comparison.json"
    with pytest.raises(FloatingPointError, match="model state"):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2, max_eval_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_corrupt_dataset_rejects_before_model_initialization(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    documents = manifest.parent / "documents.jsonl"
    documents.write_bytes(documents.read_bytes() + b"corruption")
    monkeypatch.setattr(
        public_tuning, "TinyTransformer", lambda _: pytest.fail("model initialized")
    )
    rng = capture_rng_state()
    output = tmp_path / "new" / "comparison.json"
    with pytest.raises(DatasetValidationError):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_late_report_race_never_overwrites_another_run(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning.publish_json
    output = tmp_path / "comparison.json"

    def raced_publish(path, value):
        path.write_bytes(b"winning report")
        original(path, value)

    monkeypatch.setattr(public_tuning, "publish_json", raced_publish)
    rng = capture_rng_state()
    with pytest.raises(FileExistsError):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2, max_eval_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert output.read_bytes() == b"winning report"
    assert not tuple(tmp_path.glob(".comparison.json.tmp-*"))


def test_rank_and_validation_budget_are_part_of_comparison_identity(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    results = [
        public_tuning.compare_public_tuning(
            _config(),
            manifest,
            tmp_path / f"{rank}-{budget}.json",
            rank=rank,
            max_eval_tokens=budget,
        )
        for rank, budget in ((1, 13), (2, 13), (2, 14))
    ]
    assert len({result.config_fingerprint for result in results}) == 1
    assert len({result.base_state_digest for result in results}) == 1
    assert len({result.comparison_fingerprint for result in results}) == 3


def test_paired_batch_mismatch_is_rejected_not_claimed(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    original = public_tuning._train_arm

    def mismatched_arm(*args, mode, **kwargs):
        result = original(*args, mode=mode, **kwargs)
        if mode == "lora":
            trace = (replace(result.trace[0], batch_tokens_digest="different"), *result.trace[1:])
            return replace(result, trace=trace)
        return result

    monkeypatch.setattr(public_tuning, "_train_arm", mismatched_arm)
    output = tmp_path / "new" / "comparison.json"
    rng = capture_rng_state()
    with pytest.raises(ValueError, match="paired batches"):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2, max_eval_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_all_training_and_evaluation_remain_on_cpu_under_another_default_device(tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    # A meta default is sufficient to catch accidental inherited placement without a GPU.
    with torch.device("meta"):
        result = public_tuning.compare_public_tuning(
            _config(), manifest, tmp_path / "comparison.json", rank=2, max_eval_tokens=13
        )
    assert result.comparison_contract["execution"]["parameter_devices"] == ["cpu"]


def test_validation_failure_preserves_rng_and_does_not_publish(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")

    def fail_evaluation(*args, **kwargs):
        torch.rand(7)
        raise ValueError("injected validation failure")

    monkeypatch.setattr(public_tuning, "evaluate_held_out", fail_evaluation)
    rng = capture_rng_state()
    output = tmp_path / "new" / "comparison.json"
    with pytest.raises(ValueError, match="injected validation failure"):
        public_tuning.compare_public_tuning(_config(), manifest, output, rank=2, max_eval_tokens=13)
    assert state_trees_equal(rng, capture_rng_state())
    assert not output.parent.exists()


def test_code_identity_is_recorded_in_the_comparison_contract(monkeypatch, tmp_path):
    manifest = prepared_fixture(tmp_path / "data")
    monkeypatch.setattr(public_tuning, "code_fingerprint", lambda: "a" * 64)
    first = public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "first.json", rank=2, max_eval_tokens=13
    )
    monkeypatch.setattr(public_tuning, "code_fingerprint", lambda: "b" * 64)
    second = public_tuning.compare_public_tuning(
        _config(), manifest, tmp_path / "second.json", rank=2, max_eval_tokens=13
    )
    assert first.code_fingerprint == "a" * 64
    assert first.comparison_contract["execution"]["code_fingerprint"] == "a" * 64
    assert second.code_fingerprint == "b" * 64
    assert first.comparison_fingerprint != second.comparison_fingerprint
    assert first.full.trace == second.full.trace
