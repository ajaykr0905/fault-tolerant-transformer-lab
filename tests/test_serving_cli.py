import http.client
import json
import os
import selectors
import signal
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evaluation_cli import checkpoint_run

from fttl import serving_cli


def _arguments(tmp_path):
    manifest, config, _ = checkpoint_run(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(config.canonical_json())
    return [
        "--config",
        str(config_path),
        "--checkpoint",
        str(tmp_path / "training/checkpoints"),
        "--dataset-manifest",
        str(manifest),
        "--port",
        "0",
    ]


def _request(port, method, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        body = None if payload is None else json.dumps(payload)
        headers = {} if payload is None else {"Content-Type": "application/json"}
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX foreground signal lifecycle evidence")
@pytest.mark.parametrize("installed", [False, True], ids=["module", "installed"])
@pytest.mark.parametrize("termination", [signal.SIGINT, signal.SIGTERM], ids=["sigint", "sigterm"])
def test_foreground_operator_starts_serves_and_reaps_cleanly(tmp_path, installed, termination):
    command = (
        [str(Path(sys.executable).with_name("fttl-serve"))]
        if installed
        else [sys.executable, "-m", "fttl.serving_cli"]
    )
    process = subprocess.Popen(
        [*command, *_arguments(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=20), "bounded startup deadline expired"
            ready = json.loads(process.stdout.readline(16384))
        assert ready["ready"] and ready["host"] == "127.0.0.1"
        assert 1 <= ready["port"] <= 65535
        assert ready["url"] == f"http://127.0.0.1:{ready['port']}"
        status, health = _request(ready["port"], "GET", "/health")
        assert status == 200 and health["ready"]
        assert health["receipt"] == ready["receipt"]
        payload = {"prompt": "Hello", "max_new_tokens": 4, "method": "sample", "seed": 7}
        status, generated = _request(ready["port"], "POST", "/generate", payload)
        assert status == 200 and generated["generation"]["generated_count"] == 4
        assert generated["receipt"] == ready["receipt"]
        assert _request(ready["port"], "POST", "/generate", payload) == (status, generated)
        process.send_signal(termination)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0
        assert not stdout and not stderr
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.parametrize(
    "option,value",
    [
        ("--port", "-1"),
        ("--port", "65536"),
        ("--port", "True"),
        ("--request-timeout-seconds", "0"),
        ("--request-timeout-seconds", "nan"),
        ("--request-timeout-seconds", "inf"),
        ("--request-timeout-seconds", "31"),
        ("--max-request-bytes", "0"),
        ("--max-request-bytes", "65537"),
    ],
)
def test_invalid_operator_flags_fail_before_artifact_loading(monkeypatch, capsys, option, value):
    monkeypatch.setattr(
        serving_cli, "load_inference_checkpoint", lambda *args: pytest.fail("loaded")
    )
    with pytest.raises(SystemExit) as error:
        serving_cli.main(
            [
                "--config",
                "missing",
                "--checkpoint",
                "missing",
                "--dataset-manifest",
                "missing",
                option,
                value,
            ]
        )
    assert error.value.code == 2
    assert not capsys.readouterr().out


def test_missing_checkpoint_exits_before_readiness(tmp_path, capsys):
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--checkpoint") + 1] = str(tmp_path / "missing")
    with pytest.raises(SystemExit) as error:
        serving_cli.main(arguments)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "Traceback" not in captured.err


def test_occupied_port_exits_before_readiness(tmp_path, capsys):
    arguments = _arguments(tmp_path)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        arguments[arguments.index("--port") + 1] = str(occupied.getsockname()[1])
        with pytest.raises(SystemExit) as error:
            serving_cli.main(arguments)
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert not captured.out and "Traceback" not in captured.err


def test_ctrl_c_closes_server_and_restores_signal_handlers(tmp_path, monkeypatch, capsys):
    arguments = _arguments(tmp_path)
    closed = []

    def interrupt():
        raise KeyboardInterrupt

    server = SimpleNamespace(
        server_address=("127.0.0.1", 12345),
        serve_forever=interrupt,
        server_close=lambda: closed.append(True),
    )
    monkeypatch.setattr(serving_cli, "create_inference_server", lambda *args, **kwargs: server)
    before = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    assert serving_cli.main(arguments) == 0
    assert closed == [True]
    assert before == {number: signal.getsignal(number) for number in before}
    assert json.loads(capsys.readouterr().out)["port"] == 12345


def test_unexpected_serving_error_still_closes_server(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path)
    closed = []

    def fail():
        raise RuntimeError("backend defect")

    server = SimpleNamespace(
        server_address=("127.0.0.1", 12345),
        serve_forever=fail,
        server_close=lambda: closed.append(True),
    )
    monkeypatch.setattr(serving_cli, "create_inference_server", lambda *args, **kwargs: server)
    with pytest.raises(RuntimeError, match="backend defect"):
        serving_cli.main(arguments)
    assert closed == [True]
