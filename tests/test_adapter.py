import copy
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
import torch

from fttl.adapter import export_adapter, load_adapter, load_adapter_file, save_adapter
from fttl.config import ModelConfig
from fttl.lora import LoRALinear, inject_lora
from fttl.model import TinyTransformer
from fttl.state import state_digest, state_trees_equal


def _model(*, seed=41, rank=2, alpha=4, dtype=torch.float32, config=None, targets=None):
    torch.manual_seed(seed)
    config = config or ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=2, n_layers=1)
    model = TinyTransformer(config).to(dtype=dtype)
    options = {} if targets is None else {"target_names": targets}
    inject_lora(model, rank=rank, alpha=alpha, **options)
    return model


def _train(model):
    tokens = torch.arange(16).reshape(2, 8) % model.config.vocab_size
    targets = (tokens + 1) % model.config.vocab_size
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
    )
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets)
        loss.backward()
        optimizer.step()
    return tokens


def _snapshot(model):
    return (
        {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        {name: parameter.requires_grad for name, parameter in model.named_parameters()},
        model.training,
        torch.get_rng_state().clone(),
    )


def _assert_unchanged(model, before):
    state, flags, training, rng = before
    assert state_trees_equal(state, model.state_dict())
    assert flags == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert model.training is training
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_trained_adapter_roundtrip_reproduces_logits_without_exporting_base(dtype):
    source = _model(dtype=dtype)
    tokens = _train(source)
    payload = export_adapter(source)
    target = _model(dtype=dtype)
    target.eval()
    flags_before = {name: p.requires_grad for name, p in target.named_parameters()}
    frozen_before = {
        name: p.detach().clone() for name, p in target.named_parameters() if not p.requires_grad
    }
    rng_before = torch.get_rng_state().clone()
    load_adapter(target, payload)

    source.eval()
    torch.testing.assert_close(source(tokens)[0], target(tokens)[0], rtol=0, atol=0)
    assert not target.training
    assert flags_before == {name: p.requires_grad for name, p in target.named_parameters()}
    assert torch.equal(torch.get_rng_state(), rng_before)
    for name, parameter in target.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)
    assert set(payload["modules"]) == {
        "blocks.0.attention.qkv",
        "blocks.0.attention.output",
    }
    for module in payload["modules"].values():
        assert set(module) == {
            "rank",
            "scale",
            "in_features",
            "out_features",
            "dtype",
            "lora_a",
            "lora_b",
        }
        assert module["lora_a"].device.type == module["lora_b"].device.type == "cpu"
        assert not module["lora_a"].requires_grad and not module["lora_b"].requires_grad


def test_export_and_load_do_not_alias_model_or_payload_storage():
    source = _model()
    _train(source)
    payload = export_adapter(source)
    original = copy.deepcopy(payload)
    target = _model()
    load_adapter(target, payload)
    before = _snapshot(target)
    with torch.no_grad():
        for module in source.modules():
            if isinstance(module, LoRALinear):
                module.lora_a.add_(10)
                module.lora_b.add_(10)
    assert state_trees_equal(payload, original)
    for module in payload["modules"].values():
        module["lora_a"].add_(20)
        module["lora_b"].add_(20)
    _assert_unchanged(target, before)


def _corrupt(payload, case):
    record = payload["modules"]["blocks.0.attention.output"]
    if case == "schema_bool":
        payload["schema_version"] = True
    elif case == "schema_future":
        payload["schema_version"] = 2
    elif case == "model_type":
        payload["model_type"] = "another.Model"
    elif case == "config":
        payload["model_config"]["dropout"] = 0.5
    elif case == "config_bool":
        payload["model_config"]["n_layers"] = True
    elif case == "config_extra":
        payload["model_config"]["new"] = 1
    elif case == "base_digest":
        payload["base_state_digest"] = "0" * 64
    elif case == "digest_format":
        payload["adapter_state_digest"] = "not a digest"
    elif case == "adapter_digest":
        record["lora_b"][0, 0] += 1
    elif case == "rank":
        record["rank"] = 3
    elif case == "rank_bool":
        record["rank"] = True
    elif case == "scale":
        record["scale"] = 3.0
    elif case == "scale_bool":
        record["scale"] = True
    elif case == "dimensions":
        record["in_features"] = 15
    elif case == "dtype_metadata":
        record["dtype"] = "torch.float64"
    elif case == "tensor_dtype":
        record["lora_b"] = record["lora_b"].double()
    elif case == "tensor_shape":
        record["lora_b"] = record["lora_b"].T.contiguous()
    elif case == "tensor_nan":
        record["lora_b"][0, 0] = float("nan")
        payload["adapter_state_digest"] = state_digest(payload["modules"])
    elif case == "tensor_inf":
        record["lora_b"][0, 0] = float("inf")
        payload["adapter_state_digest"] = state_digest(payload["modules"])
    elif case == "tensor_grad":
        record["lora_b"].requires_grad = True
    elif case == "tensor_meta":
        record["lora_b"] = torch.empty_like(record["lora_b"], device="meta")
    elif case == "tensor_sparse":
        record["lora_b"] = record["lora_b"].to_sparse()
    elif case == "tensor_not_tensor":
        record["lora_b"] = record["lora_b"].tolist()
    elif case == "module_extra_field":
        record["unknown"] = 1
    elif case == "module_missing_field":
        del record["scale"]
    elif case == "module_missing":
        del payload["modules"]["blocks.0.attention.output"]
    elif case == "module_extra":
        payload["modules"]["missing"] = copy.deepcopy(record)
    elif case == "payload_extra":
        payload["unknown"] = 1
    elif case == "payload_missing":
        del payload["model_config"]
    elif case == "payload_not_dict":
        return list(payload.items())
    else:
        raise AssertionError(case)
    return payload


@pytest.mark.parametrize(
    "case",
    [
        "schema_bool",
        "schema_future",
        "model_type",
        "config",
        "config_bool",
        "config_extra",
        "base_digest",
        "digest_format",
        "adapter_digest",
        "rank",
        "rank_bool",
        "scale",
        "scale_bool",
        "dimensions",
        "dtype_metadata",
        "tensor_dtype",
        "tensor_shape",
        "tensor_nan",
        "tensor_inf",
        "tensor_grad",
        "tensor_meta",
        "tensor_sparse",
        "tensor_not_tensor",
        "module_extra_field",
        "module_missing_field",
        "module_missing",
        "module_extra",
        "payload_extra",
        "payload_missing",
        "payload_not_dict",
    ],
)
def test_invalid_artifact_is_rejected_before_any_adapter_changes(case):
    source = _model()
    _train(source)
    payload = _corrupt(export_adapter(source), case)
    target = _model()
    # Preserve a caller's deliberately frozen adapter flag, not just default flags.
    target.blocks[0].attention.qkv.lora_a.requires_grad = False
    before = _snapshot(target)
    with pytest.raises(ValueError):
        load_adapter(target, payload)
    _assert_unchanged(target, before)


@pytest.mark.parametrize(
    "options",
    [{"seed": 42}, {"rank": 1}, {"alpha": 8}, {"dtype": torch.float64}, {"targets": ("qkv",)}],
)
def test_target_base_or_injection_contract_must_match(options):
    payload = export_adapter(_model())
    target = _model(**options)
    before = _snapshot(target)
    with pytest.raises(ValueError):
        load_adapter(target, payload)
    _assert_unchanged(target, before)


def test_same_weights_with_different_forward_config_are_not_accepted():
    payload = export_adapter(_model())
    config = ModelConfig(vocab_size=32, block_size=8, d_model=16, n_heads=4, n_layers=1)
    target = _model(config=config)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="model_config"):
        load_adapter(target, payload)
    _assert_unchanged(target, before)


@pytest.mark.parametrize("operation", [export_adapter, lambda model: load_adapter(model, {})])
def test_non_lora_transformer_cannot_be_exported_or_implicitly_injected(operation):
    torch.manual_seed(41)
    target = TinyTransformer(ModelConfig())
    before = _snapshot(target)
    with pytest.raises(ValueError):
        operation(target)
    _assert_unchanged(target, before)
    assert not any(isinstance(module, LoRALinear) for module in target.modules())


def test_trainable_base_is_rejected_without_changing_flags():
    source = _model()
    payload = export_adapter(source)
    source.token_embedding.weight.requires_grad = True
    before = _snapshot(source)
    with pytest.raises(ValueError, match="must be frozen"):
        export_adapter(source)
    with pytest.raises(ValueError, match="must be frozen"):
        load_adapter(source, payload)
    _assert_unchanged(source, before)


def test_base_digest_binds_buffers_even_when_nonpersistent():
    source = _model()
    source.register_buffer("control", torch.tensor([1, 2]), persistent=False)
    payload = export_adapter(source)
    target = _model()
    target.register_buffer("control", torch.tensor([1, 3]), persistent=False)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="base state"):
        load_adapter(target, payload)
    _assert_unchanged(target, before)
    target.control[1] = 2
    load_adapter(target, payload)


def test_tied_embedding_base_is_stably_bound_across_fresh_models():
    source = _model(targets=("lm_head",))
    tokens = _train(source)
    target = _model(targets=("lm_head",))
    load_adapter(target, export_adapter(source))
    assert target.lm_head.base.weight is target.token_embedding.weight
    torch.testing.assert_close(source(tokens)[0], target(tokens)[0], rtol=0, atol=0)


def test_shared_adapter_storage_is_rejected():
    model = _model()
    model.blocks[0].attention.output.lora_a = model.blocks[0].attention.qkv.lora_a
    with pytest.raises(ValueError, match="share storage"):
        export_adapter(model)


def test_adapter_storage_cannot_alias_a_base_buffer():
    model = _model()
    model.register_buffer("aliased", model.blocks[0].attention.qkv.lora_a.detach())
    with pytest.raises(ValueError, match="buffers must not share adapter storage"):
        export_adapter(model)


def test_trainable_buffer_is_not_part_of_a_frozen_base():
    model = _model()
    model.register_buffer("trainable", torch.ones(2, requires_grad=True))
    with pytest.raises(ValueError, match="base buffer trainable must be frozen"):
        export_adapter(model)


@pytest.mark.parametrize("field", ["weight", "bias"])
def test_mutated_base_dimensions_cannot_be_exported(field):
    model = _model()
    base = model.blocks[0].attention.output.base
    replacement = torch.ones(1) if field == "bias" else torch.ones(1, 16)
    setattr(base, field, torch.nn.Parameter(replacement, requires_grad=False))
    with pytest.raises(ValueError, match="base linear contract"):
        export_adapter(model)


def test_nonfinite_base_cannot_be_exported():
    model = _model()
    with torch.no_grad():
        model.token_embedding.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        export_adapter(model)


def test_disk_roundtrip_uses_weights_only_safe_artifact(tmp_path):
    source = _model()
    tokens = _train(source)
    output = tmp_path / "nested" / "adapter.pt"
    save_adapter(source, output)
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert state_trees_equal(payload, export_adapter(source))
    target = _model()
    load_adapter_file(target, output)
    torch.testing.assert_close(source(tokens)[0], target(tokens)[0], rtol=0, atol=0)
    assert sorted(output.parent.iterdir()) == [output]


@pytest.mark.parametrize("kind", ["file", "directory", "symlink"])
def test_save_never_replaces_existing_paths(tmp_path, kind):
    output = tmp_path / "adapter.pt"
    if kind == "file":
        output.write_bytes(b"existing evidence")
    elif kind == "directory":
        output.mkdir()
    else:
        output.symlink_to(tmp_path / "absent.pt")
    with pytest.raises(FileExistsError):
        save_adapter(_model(), output)
    assert sorted(tmp_path.iterdir()) == [output]
    if kind == "file":
        assert output.read_bytes() == b"existing evidence"
    elif kind == "directory":
        assert output.is_dir()
    else:
        assert output.is_symlink()


@pytest.mark.parametrize("stage", ["save", "fsync", "link"])
def test_failed_disk_publication_cleans_only_its_temporary(tmp_path, monkeypatch, stage):
    import fttl.adapter as adapter

    output = tmp_path / "adapter.pt"
    preserved = tmp_path / ".other-run.tmp"
    preserved.write_bytes(b"another writer")

    def fail(*args, **kwargs):
        raise OSError("injected publication failure")

    if stage == "save":
        monkeypatch.setattr(adapter.torch, "save", fail)
    else:
        monkeypatch.setattr(adapter.os, stage, fail)
    with pytest.raises(OSError, match="injected"):
        save_adapter(_model(), output)
    assert sorted(tmp_path.iterdir()) == [preserved]
    assert preserved.read_bytes() == b"another writer"


def test_invalid_model_does_not_create_output_parents(tmp_path):
    model = _model()
    model.token_embedding.weight.requires_grad = True
    with pytest.raises(ValueError):
        save_adapter(model, tmp_path / "new" / "adapter.pt")
    assert not (tmp_path / "new").exists()


def test_two_concurrent_publishers_expose_one_complete_artifact(tmp_path, monkeypatch):
    import fttl.adapter as adapter

    first = _model()
    second = copy.deepcopy(first)
    _train(second)
    payloads = [export_adapter(first), export_adapter(second)]
    barrier = Barrier(2)
    original_link = os.link

    def link(source, target):
        barrier.wait(timeout=10)
        original_link(source, target)

    monkeypatch.setattr(adapter.os, "link", link)
    output = tmp_path / "adapter.pt"

    def publish(model):
        try:
            save_adapter(model, output)
            return "published"
        except FileExistsError:
            return "preserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, [first, second]))
    assert sorted(results) == ["preserved", "published"]
    loaded = torch.load(output, weights_only=True)
    assert any(state_trees_equal(loaded, expected) for expected in payloads)
    assert sorted(tmp_path.iterdir()) == [output]


def test_weights_only_loader_rejects_non_safe_globals_without_model_mutation(tmp_path):
    source = tmp_path / "unsafe.pt"
    torch.save({"unexpected": Path("local-only")}, source)
    model = _model()
    before = _snapshot(model)
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load_adapter_file(model, source)
    _assert_unchanged(model, before)


def test_corrupt_disk_tensor_cannot_partially_load_a_target(tmp_path):
    source = _model()
    _train(source)
    payload = _corrupt(export_adapter(source), "tensor_nan")
    artifact = tmp_path / "corrupt.pt"
    torch.save(payload, artifact)
    target = _model()
    before = _snapshot(target)
    with pytest.raises(ValueError, match="non-finite"):
        load_adapter_file(target, artifact)
    _assert_unchanged(target, before)
