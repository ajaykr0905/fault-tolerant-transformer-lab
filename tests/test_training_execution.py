import json

import pytest
import torch
from test_real_data_recovery import prepared_fixture
from test_training_cursor import config

from fttl import train
from fttl.checkpoint import CheckpointMismatchError
from fttl.model import TinyTransformer


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("default_device", ["cpu", "meta"])
def test_cpu_training_is_independent_of_caller_factory_defaults(tmp_path, prepared, default_device):
    configuration = config(vocab_size=257 if prepared else 24)
    manifest = prepared_fixture(tmp_path) if prepared else None
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"):
            expected_model, expected = train.run_training(
                configuration, tmp_path / "baseline", dataset_manifest=manifest
            )
        torch.set_default_dtype(torch.float64)
        with torch.device(default_device):
            model, actual = train.run_training(
                configuration, tmp_path / "changed-defaults", dataset_manifest=manifest
            )
            assert torch.get_default_dtype() == torch.float64
            assert torch.empty(0).device.type == default_device
        assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
        for name, value in expected_model.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        assert expected.losses == actual.losses
        assert expected.batch_ids == actual.batch_ids
        assert expected.sample_ids == actual.sample_ids
        assert expected.final_cursor == actual.final_cursor
        assert expected.model_digest == actual.model_digest
        assert expected.final_logits_digest == actual.final_logits_digest
        assert expected.run_contract_fingerprint != actual.run_contract_fingerprint
        assert expected.optimizer_digest != actual.optimizer_digest
        assert expected.final_state_digest != actual.final_state_digest
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64, torch.float16])
def test_cpu_execution_policy_is_recorded_and_bound_to_the_run_contract(
    tmp_path, monkeypatch, default_dtype
):
    contracts = []
    optimizers = []
    original_digest = train.state_digest
    original_adamw = torch.optim.AdamW

    def capture(value):
        if isinstance(value, dict) and value.get("schema") == "RunContractV1":
            contracts.append(value)
        return original_digest(value)

    monkeypatch.setattr(train, "state_digest", capture)

    def capture_optimizer(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", capture_optimizer)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        _, result = train.run_training(config(vocab_size=24), tmp_path / "run")
    finally:
        torch.set_default_dtype(original_dtype)
    step_dtype = torch.float64 if default_dtype == torch.float64 else torch.float32
    policy = {
        "device": "cpu",
        "model_dtype": "torch.float32",
        "optimizer_step_dtype": str(step_dtype),
    }
    assert contracts[0]["execution"] == policy
    assert result.run_contract_fingerprint == original_digest(contracts[0])
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["execution"] == policy
    assert manifest["caller_default_dtype"] == str(default_dtype)
    assert result.model_dtype == "torch.float32"
    assert result.optimizer_step_dtype == str(step_dtype)
    assert result.caller_default_dtype == str(default_dtype)
    assert {item["step"].dtype for item in optimizers[0].state.values()} == {step_dtype}


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float64])
def test_resume_rejects_a_different_scalar_step_policy_without_publishing(tmp_path, source_dtype):
    original_dtype = torch.get_default_dtype()
    output = tmp_path / "run"
    try:
        torch.set_default_dtype(source_dtype)
        _, partial = train.run_training(config(vocab_size=24), output, stop_after_step=1)
        before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        other_dtype = torch.float64 if source_dtype == torch.float32 else torch.float32
        torch.set_default_dtype(other_dtype)
        with torch.device("meta"):
            with pytest.raises(CheckpointMismatchError, match="run contract"):
                train.run_training(
                    config(vocab_size=24), output, resume_from=output / partial.checkpoint
                )
            assert torch.empty(0).device.type == "meta"
            assert torch.get_default_dtype() == other_dtype
        assert before == {
            p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()
        }
    finally:
        torch.set_default_dtype(original_dtype)


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_same_policy_resume_under_meta_default_matches_uninterrupted_training(
    tmp_path, default_dtype
):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        baseline_model, baseline = train.run_training(config(vocab_size=24), tmp_path / "baseline")
        output = tmp_path / "resumed"
        _, partial = train.run_training(config(vocab_size=24), output, stop_after_step=1)
        with torch.device("meta"):
            resumed_model, resumed = train.run_training(
                config(vocab_size=24), output, resume_from=output / partial.checkpoint
            )
            assert torch.empty(0).device.type == "meta"
            assert torch.get_default_dtype() == default_dtype
        assert baseline.losses == resumed.losses
        assert baseline.final_cursor == resumed.final_cursor
        assert baseline.final_state_digest == resumed.final_state_digest
        assert baseline.run_fingerprint == resumed.run_fingerprint
        for name, value in baseline_model.state_dict().items():
            torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.set_default_dtype(original_dtype)


def test_direct_model_defaults_and_explicit_factories_remain_independent():
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("cpu"):
            default = TinyTransformer(config(vocab_size=24).model)
            tokens = torch.zeros((1, 4), dtype=torch.long)
            logits, loss = default(tokens, tokens)
        assert all(p.dtype == torch.float64 for p in default.parameters())
        assert logits.dtype == loss.dtype == torch.float64
        torch.manual_seed(7)
        with torch.device("meta"):
            explicit = TinyTransformer(
                config(vocab_size=24).model, device="cpu", dtype=torch.float32
            )
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(7)
        with torch.device("cpu"):
            reference = TinyTransformer(config(vocab_size=24).model)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.set_default_dtype(original_dtype)
