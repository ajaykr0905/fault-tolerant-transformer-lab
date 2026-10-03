"""CPU adapter tests are contract checks, not proof of CUDA hardware execution."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import fttl.cuda_runtime as cuda_runtime_module
from fttl.cuda_runtime import (
    capture_cuda_rng_state,
    prepare_cuda_execution,
    restore_cuda_rng_state,
    synchronize_cuda,
    validate_cuda_rng_state,
)
from fttl.state import capture_rng_state, state_trees_equal

DEVICE = torch.device("cuda:0")


@pytest.fixture(autouse=True)
def isolated_preparation(monkeypatch):
    monkeypatch.setattr(cuda_runtime_module, "_prepared_workspace", None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)


@pytest.fixture
def cuda_adapter(monkeypatch):
    generator_type = torch.Generator
    selected = generator_type(device="cpu").manual_seed(53)
    calls = []

    def candidate(*, device):
        calls.append(str(device))
        return generator_type(device="cpu")

    monkeypatch.setattr(torch, "Generator", candidate)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: selected.get_state())
    monkeypatch.setattr(
        torch.cuda, "set_rng_state", lambda state, device: selected.set_state(state)
    )
    return selected, calls


def test_missing_workspace_refuses_initialization(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried too early"))
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        prepare_cuda_execution()


def test_no_gpu_has_no_cpu_fallback(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="does not fall back"):
        prepare_cuda_execution()


def test_unknown_initialized_cuda_requires_a_fresh_process(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="fresh subprocess"):
        prepare_cuda_execution()


def test_non_float32_initialization_is_rejected_before_cuda(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch, "get_default_dtype", lambda: torch.float64)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried too early"))
    with pytest.raises(RuntimeError, match="default dtype torch.float32"):
        prepare_cuda_execution()


def test_prepare_records_observed_metadata_and_strict_policy(monkeypatch):
    calls = []
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(str(device)))
    monkeypatch.setattr(
        torch, "use_deterministic_algorithms", lambda enabled: calls.append(enabled)
    )
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda value: calls.append(value))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", True)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", False)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90100)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="TEST ADAPTER", major=8, minor=0, total_memory=1024),
    )
    device, metadata = prepare_cuda_execution()
    assert device == DEVICE
    assert calls == ["cuda:0", True, "highest"]
    assert metadata["gpu_name"] == "TEST ADAPTER"
    assert metadata["compute_capability"] == [8, 0]
    assert metadata["total_memory_bytes"] == 1024
    assert metadata["optimizer_policy"] == "AdamW foreach=False fused=False"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(torch, "get_float32_matmul_precision", lambda: "highest")
    assert prepare_cuda_execution() == (device, metadata)
    torch.backends.cudnn.benchmark = True
    with pytest.raises(RuntimeError, match="policy changed"):
        prepare_cuda_execution()


def test_validation_uses_independent_candidates_without_global_changes(cuda_adapter):
    selected, calls = cuda_adapter
    state = capture_cuda_rng_state(DEVICE)
    selected.manual_seed(77)
    before = capture_cuda_rng_state(DEVICE)
    validate_cuda_rng_state(state, DEVICE)
    assert state_trees_equal(before, capture_cuda_rng_state(DEVICE))
    assert calls[-1] == "cuda:0"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("device_index", True),
        ("device_index", 1),
        ("state", torch.zeros(1, dtype=torch.float32)),
        ("state", torch.zeros(1, dtype=torch.uint8)),
    ],
)
def test_invalid_cuda_declaration_leaves_all_global_rngs_unchanged(cuda_adapter, field, value):
    state = capture_cuda_rng_state(DEVICE)
    state["torch_cuda"][field] = value
    before = capture_cuda_rng_state(DEVICE)
    with pytest.raises(ValueError, match="CUDA RNG"):
        restore_cuda_rng_state(state, DEVICE)
    assert state_trees_equal(before, capture_cuda_rng_state(DEVICE))


def test_invalid_cpu_candidate_leaves_cuda_and_cpu_rngs_unchanged(cuda_adapter):
    state = capture_cuda_rng_state(DEVICE)
    state["numpy"]["bit_generator"] = "invalid"
    before = capture_cuda_rng_state(DEVICE)
    with pytest.raises(ValueError, match="CPU RNG"):
        restore_cuda_rng_state(state, DEVICE)
    assert state_trees_equal(before, capture_cuda_rng_state(DEVICE))


def test_failed_cuda_setter_rolls_back_cpu_and_selected_rngs(cuda_adapter, monkeypatch):
    selected, _ = cuda_adapter
    state = capture_cuda_rng_state(DEVICE)
    torch.manual_seed(91)
    selected.manual_seed(92)
    before = capture_cuda_rng_state(DEVICE)
    calls = 0

    def fail_once(value, device):
        nonlocal calls
        calls += 1
        selected.set_state(value)
        if calls == 1:
            raise RuntimeError("injected selected-device setter failure")

    monkeypatch.setattr(torch.cuda, "set_rng_state", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        restore_cuda_rng_state(state, DEVICE)
    assert calls == 2
    assert state_trees_equal(before, capture_cuda_rng_state(DEVICE))


def test_successful_restore_recovers_both_rngs(cuda_adapter):
    selected, _ = cuda_adapter
    state = capture_cuda_rng_state(DEVICE)
    torch.manual_seed(91)
    selected.manual_seed(92)
    restore_cuda_rng_state(copy.deepcopy(state), DEVICE)
    assert state_trees_equal(state, capture_cuda_rng_state(DEVICE))


def test_synchronize_requires_explicit_selected_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    synchronize_cuda(DEVICE)
    assert calls == ["cuda:0"]
    for device in [torch.device("cpu"), torch.device("cuda"), torch.device("cuda:1")]:
        with pytest.raises(ValueError, match="cuda:0"):
            synchronize_cuda(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA hardware unavailable")
def test_real_cuda_rng_round_trip():
    # A child cannot inherit the parent fixture's fake initialization guard.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from fttl.cuda_runtime import (
    prepare_cuda_execution, capture_cuda_rng_state,
    restore_cuda_rng_state, synchronize_cuda,
)
device, metadata = prepare_cuda_execution()
assert device == torch.device('cuda:0')
saved = capture_cuda_rng_state(device)
expected = torch.rand(8, device=device)
expected_cpu = torch.rand(8)
restore_cuda_rng_state(saved, device)
assert torch.equal(expected, torch.rand(8, device=device))
assert torch.equal(expected_cpu, torch.rand(8))
synchronize_cuda(device)
""",
        ],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_capture_cpu_mapping_is_preserved(cuda_adapter):
    cpu = capture_rng_state()
    combined = capture_cuda_rng_state(DEVICE)
    assert state_trees_equal(
        cpu, {key: value for key, value in combined.items() if key != "torch_cuda"}
    )
