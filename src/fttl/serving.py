"""Dependency-free, single-threaded loopback inference prototype, not production serving."""

from __future__ import annotations

import json
import math
import re
import socket
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import chain

import torch

from fttl.generation import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS, generate_tokens
from fttl.inference import InferenceReceiptV1
from fttl.model import TinyTransformer
from fttl.numerical import require_finite_state
from fttl.state import state_digest

MAX_REQUEST_BYTES = 65536
_CONTROLS = {"max_new_tokens", "method", "temperature", "top_k", "seed", "stop_token_id"}


def _model_identity(model: TinyTransformer, receipt: InferenceReceiptV1) -> str:
    if not isinstance(model, TinyTransformer) or model.config.vocab_size != 257:
        raise ValueError("serving requires a byte TinyTransformer")
    if any(module.training for module in model.modules()):
        raise ValueError("serving requires all modules in eval mode")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("serving requires frozen dense CPU FP32 state")
    state = {}
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if (
            tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or tensor.is_nested
            or tensor.requires_grad
            or tensor.is_complex()
            or tensor.is_quantized
            or (tensor.is_floating_point() and tensor.dtype != torch.float32)
        ):
            raise ValueError("serving requires frozen dense CPU FP32 state")
        require_finite_state(tensor, "serving model state")
        state[name] = tensor
    if state_digest(model.state_dict()) != receipt.model_state_digest:
        raise ValueError("inference receipt does not match model state")
    return state_digest({"model_config": asdict(model.config), "runtime_state": state})


def _validate_receipt(receipt: InferenceReceiptV1) -> None:
    if type(receipt) is not InferenceReceiptV1:
        raise ValueError("serving requires an InferenceReceiptV1")
    if type(receipt.schema_version) is not int or receipt.schema_version != 1:
        raise ValueError("unsupported inference receipt schema")
    if receipt.device != "cpu" or receipt.dtype != "torch.float32":
        raise ValueError("serving requires a CPU FP32 inference receipt")
    for field in (
        "config_fingerprint",
        "data_fingerprint",
        "tokenizer_fingerprint",
        "training_run_contract_fingerprint",
        "model_state_digest",
    ):
        value = getattr(receipt, field)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("invalid inference receipt fingerprint")
    for field in ("selected_checkpoint_generation", "completed_training_steps"):
        value = getattr(receipt, field)
        if type(value) is not int or value < 0:
            raise ValueError("invalid inference receipt counter")
    if (
        not isinstance(receipt.limitations, tuple)
        or not receipt.limitations
        or any(not isinstance(item, str) or not item for item in receipt.limitations)
    ):
        raise ValueError("inference receipt limitations must be explicit")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _reject_constant(value: str):
    raise ValueError("non-finite JSON constant")


class _InferenceServer(HTTPServer):
    def ready(self) -> bool:
        try:
            return _model_identity(self.model, self.receipt) == self.runtime_identity
        except Exception:
            return False

    def handle_error(self, request, client_address):
        # Unexpected handler/setup failures must not print private paths or traces.
        pass


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        self.request.settimeout(self.server.request_timeout_seconds)
        super().setup()

    def log_message(self, format, *args):
        pass

    def version_string(self):
        return "fttl-local-prototype"

    def _json(self, status: int, value: object):
        self.close_connection = True
        encoded = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(encoded)
        except (OSError, TimeoutError):
            pass

    def send_error(self, code, message=None, explain=None):
        self._json(code, {"error": "invalid_http_request"})

    def do_GET(self):
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        ready = self.server.ready()
        self._json(
            200 if ready else 503,
            {
                "schema_version": 1,
                "ready": ready,
                "receipt": self.server.receipt.to_dict(),
                "limits": {
                    "max_request_bytes": self.server.max_request_bytes,
                    "max_prompt_tokens": MAX_PROMPT_TOKENS,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
                "limitations": [
                    "Loopback-only, unauthenticated, single-threaded local prototype.",
                    "No public-network, production, concurrency, or language-quality claim.",
                ],
            },
        )

    def do_POST(self):
        if self.path != "/generate":
            self._json(404, {"error": "not_found"})
            return
        if not self.server.ready():
            self._json(503, {"error": "not_ready"})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get_all("Transfer-Encoding"):
            self._json(400, {"error": "unsupported_transfer_encoding"})
            return
        if not lengths:
            self._json(411, {"error": "content_length_required"})
            return
        if len(lengths) != 1 or re.fullmatch(r"[0-9]{1,6}", lengths[0].strip()) is None:
            self._json(400, {"error": "invalid_content_length"})
            return
        length = int(lengths[0])
        if length > self.server.max_request_bytes:
            self._json(413, {"error": "request_too_large"})
            return
        if (
            len(self.headers.get_all("Content-Type", [])) != 1
            or self.headers.get_content_type() != "application/json"
            or self.headers.get_content_charset("utf-8").lower() != "utf-8"
        ):
            self._json(415, {"error": "json_utf8_required"})
            return
        try:
            raw = self.rfile.read(length)
        except socket.timeout:
            self._json(408, {"error": "request_timeout"})
            return
        except OSError:
            self.close_connection = True
            return
        try:
            if len(raw) != length:
                raise ValueError("truncated body")
            payload = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_float=_finite_float,
                parse_constant=_reject_constant,
            )
            if (
                type(payload) is not dict
                or "prompt" not in payload
                or set(payload) - _CONTROLS - {"prompt"}
            ):
                raise ValueError("invalid request fields")
            if not isinstance(payload["prompt"], str):
                raise ValueError("prompt must be UTF-8 text")
            prompt = payload.pop("prompt").encode("utf-8")
        except (ValueError, RecursionError):
            self._json(400, {"error": "invalid_json_request"})
            return
        try:
            result = generate_tokens(self.server.model, prompt, **payload)
        except ValueError:
            if not self.server.ready():
                self._json(503, {"error": "not_ready"})
            else:
                self._json(400, {"error": "invalid_generation_request"})
            return
        except Exception:
            if not self.server.ready():
                self._json(503, {"error": "not_ready"})
            else:
                self._json(500, {"error": "generation_failed"})
            return
        if not self.server.ready():
            self._json(503, {"error": "not_ready"})
            return
        self._json(
            200,
            {
                "schema_version": 1,
                "receipt": self.server.receipt.to_dict(),
                "generation": result.to_dict(),
            },
        )


def create_inference_server(
    model: TinyTransformer,
    receipt: InferenceReceiptV1,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    request_timeout_seconds: float = 5.0,
    max_request_bytes: int = 16384,
) -> HTTPServer:
    """Validate before binding; the caller owns serve_forever/shutdown/server_close.

    The socket timeout bounds idle waits, not a whole request's wall time. Slow
    clients can occupy the one handler. This is an unauthenticated local
    prototype, not isolation against other local users.
    """
    if host != "127.0.0.1":
        raise ValueError("serving binds only 127.0.0.1")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")
    if type(max_request_bytes) is not int or not 1 <= max_request_bytes <= MAX_REQUEST_BYTES:
        raise ValueError("max_request_bytes must be an integer between 1 and 65536")
    if (
        isinstance(request_timeout_seconds, bool)
        or not isinstance(request_timeout_seconds, (int, float))
        or not 0 < request_timeout_seconds <= 30
    ):
        raise ValueError("request_timeout_seconds must be finite and between 0 and 30")
    _validate_receipt(receipt)
    runtime_identity = _model_identity(model, receipt)
    server = _InferenceServer((host, port), _Handler)
    server.model, server.receipt = model, receipt
    server.runtime_identity = runtime_identity
    server.max_request_bytes = max_request_bytes
    server.request_timeout_seconds = float(request_timeout_seconds)
    return server
