import http.client
import json
import socket
import threading
from dataclasses import replace

import pytest
import torch
from test_evaluation_cli import checkpoint_run

from fttl import serving
from fttl.inference import load_inference_checkpoint
from fttl.state import capture_rng_state, state_digest, state_trees_equal


@pytest.fixture
def inference(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    return load_inference_checkpoint(config, tmp_path / "training/checkpoints", manifest)


@pytest.fixture
def running(inference):
    model, receipt = inference
    server = serving.create_inference_server(model, receipt, port=0, request_timeout_seconds=0.15)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        assert not thread.is_alive()


def _request(server, method, path, body=None, headers=None):
    client = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        raw = response.read()
        assert response.getheader("Connection") == "close"
        assert int(response.getheader("Content-Length")) == len(raw)
        return response.status, json.loads(raw)
    finally:
        client.close()


def _post(server, payload):
    return _request(
        server, "POST", "/generate", json.dumps(payload), {"Content-Type": "application/json"}
    )


def _raw(server, request):
    with socket.create_connection(server.server_address, timeout=3) as client:
        client.sendall(request)
        received = b""
        while data := client.recv(65536):
            received += data
        return received


def test_readiness_reports_verified_checkpoint_and_limits(running):
    status, value = _request(running, "GET", "/health")
    assert status == 200 and value["ready"]
    assert value["receipt"] == json.loads(json.dumps(running.receipt.to_dict()))
    assert value["limits"]["max_new_tokens"] == 256
    assert value["limits"]["max_request_bytes"] == 16384
    assert running.server_address[0] == "127.0.0.1"


def test_sampled_requests_replay_and_preserve_caller_state(running):
    model = running.model
    rng = capture_rng_state()
    identity = state_digest(model.state_dict())
    modes = [module.training for module in model.modules()]
    flags = [parameter.requires_grad for parameter in model.parameters()]
    payload = {"prompt": "hello", "method": "sample", "max_new_tokens": 5, "seed": 23, "top_k": 5}
    status, first = _post(running, payload)
    second_status, second = _post(running, payload)
    assert status == second_status == 200
    assert first == second
    assert first["receipt"] == json.loads(json.dumps(running.receipt.to_dict()))
    assert first["generation"]["prompt_token_ids"] == list(b"hello")
    assert first["generation"]["generated_count"] == 5
    assert first["generation"]["model_state_digest"] == running.receipt.model_state_digest
    assert state_trees_equal(rng, capture_rng_state())
    assert state_digest(model.state_dict()) == identity
    assert [module.training for module in model.modules()] == modes
    assert [parameter.requires_grad for parameter in model.parameters()] == flags
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"[]",
        b'{"prompt":"a","prompt":"b"}',
        b'{"prompt":"a","seed":NaN}',
        b'{"prompt":"a","temperature":1e309}',
        b'{"prompt":"a","unknown":1}',
        b'{"prompt":[]}',
        b'{"prompt":"\\ud800"}',
        b"\xff",
        b"{}",
    ],
)
def test_invalid_json_or_shape_returns_clean_400_and_preserves_state(running, body):
    rng = capture_rng_state()
    identity = state_digest(running.model.state_dict())
    status, value = _request(
        running, "POST", "/generate", body, {"Content-Type": "application/json"}
    )
    assert status == 400 and value == {"error": "invalid_json_request"}
    assert state_trees_equal(rng, capture_rng_state())
    assert state_digest(running.model.state_dict()) == identity


@pytest.mark.parametrize(
    "controls",
    [
        {"prompt": ""},
        {"prompt": "a" * 4097},
        {"prompt": "😀" * 1025},
        {"max_new_tokens": 257},
        {"max_new_tokens": True},
        {"temperature": 0},
        {"top_k": 0},
        {"method": []},
        {"seed": {"nested": 1}},
        {"stop_token_id": 257},
    ],
)
def test_invalid_generation_controls_return_clean_400(running, controls):
    status, value = _post(running, {"prompt": "ok", **controls})
    assert status == 400 and value == {"error": "invalid_generation_request"}


@pytest.mark.parametrize(
    "header,expected",
    [
        (b"", 411),
        (b"Content-Length: -1\r\n", 400),
        (b"Content-Length: 2\r\nContent-Length: 2\r\n", 400),
        (b"Content-Length: invalid\r\n", 400),
        (b"Content-Length: 17000\r\n", 413),
        (b"Content-Length: 2\r\nTransfer-Encoding: chunked\r\n", 400),
    ],
)
def test_invalid_body_framing_rejects_without_reading_body(running, header, expected):
    raw = _raw(
        running,
        b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
        + header
        + b"\r\n",
    )
    assert raw.startswith(f"HTTP/1.1 {expected} ".encode())
    assert b"Traceback" not in raw and b"/Users/" not in raw


def test_partial_body_has_finite_timeout_and_server_remains_available(running):
    raw = _raw(
        running,
        b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{",
    )
    assert raw.startswith(b"HTTP/1.1 408 ")
    assert _request(running, "GET", "/health")[0] == 200


def test_client_disconnect_does_not_stop_server(running):
    with socket.create_connection(running.server_address, timeout=3) as client:
        client.sendall(
            b"POST /generate HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 10\r\n\r\n{"
        )
    assert _request(running, "GET", "/health")[0] == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Content-Type": "text/plain"}, {"Content-Type": "application/json; charset=latin1"}],
)
def test_non_json_utf8_content_type_rejects(running, headers):
    status, value = _request(running, "POST", "/generate", b"{}", headers)
    assert status == 415 and value == {"error": "json_utf8_required"}


def test_unexpected_backend_failure_is_generic_and_does_not_leak(running, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("/private/internal/path credential details")

    monkeypatch.setattr(serving, "generate_tokens", fail)
    status, value = _post(running, {"prompt": "hi"})
    assert status == 500 and value == {"error": "generation_failed"}


@pytest.mark.parametrize(
    "change", ["state", "mode", "trainability", "nonpersistent-buffer", "config"]
)
def test_drift_revokes_readiness_and_generation(running, change):
    if change == "state":
        with torch.no_grad():
            running.model.token_embedding.weight.add_(1)
    elif change == "mode":
        running.model.blocks[0].train()
    elif change == "trainability":
        running.model.token_embedding.weight.requires_grad_(True)
    elif change == "nonpersistent-buffer":
        running.model.register_buffer("extra", torch.ones(1), persistent=False)
    else:
        running.model.config = replace(running.model.config, dropout=0.5)
    assert _request(running, "GET", "/health")[0] == 503
    assert _post(running, {"prompt": "hi"}) == (503, {"error": "not_ready"})


@pytest.mark.parametrize(
    "options",
    [
        {"host": "0.0.0.0"},
        {"host": "localhost"},
        {"port": True},
        {"port": -1},
        {"port": 65536},
        {"request_timeout_seconds": 0},
        {"request_timeout_seconds": float("nan")},
        {"request_timeout_seconds": float("inf")},
        {"request_timeout_seconds": True},
        {"max_request_bytes": 0},
        {"max_request_bytes": True},
        {"max_request_bytes": 65537},
    ],
)
def test_invalid_runtime_options_reject_before_binding(inference, monkeypatch, options):
    monkeypatch.setattr(serving, "_InferenceServer", lambda *args: pytest.fail("bound socket"))
    with pytest.raises(ValueError):
        serving.create_inference_server(*inference, **options)


def test_receipt_identity_or_model_readiness_rejects_before_bind(inference):
    model, receipt = inference
    with pytest.raises(ValueError, match="receipt"):
        serving.create_inference_server(
            model, replace(receipt, model_state_digest="0" * 64), port=0
        )
    model.blocks[0].attention.train()
    with pytest.raises(ValueError, match="eval"):
        serving.create_inference_server(model, receipt, port=0)


def test_unknown_routes_and_unsupported_methods_have_generic_responses(running):
    assert _request(running, "GET", "/private/path")[0] == 404
    assert _request(running, "POST", "/private/path", b"{}")[0] == 404
    assert _request(running, "OPTIONS", "/generate") == (501, {"error": "invalid_http_request"})


@pytest.mark.parametrize("fail", [False, True])
def test_drift_during_generation_never_emits_a_success_receipt(running, monkeypatch, fail):
    original = serving.generate_tokens

    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        running.model.config = replace(running.model.config, dropout=0.5)
        if fail:
            raise RuntimeError("private backend detail")
        return result

    monkeypatch.setattr(serving, "generate_tokens", drift)
    assert _post(running, {"prompt": "ok", "max_new_tokens": 1}) == (503, {"error": "not_ready"})
