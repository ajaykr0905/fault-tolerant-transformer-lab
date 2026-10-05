"""Operate the verified, loopback-only CPU inference prototype in the foreground."""

from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path

from fttl.config import ExperimentConfig
from fttl.generation import MAX_NEW_TOKENS, MAX_PROMPT_TOKENS
from fttl.inference import load_inference_checkpoint
from fttl.serving import MAX_REQUEST_BYTES, create_inference_server


def _port(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 0 <= result <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return result


def _request_bytes(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("max-request-bytes must be an integer") from error
    if not 1 <= result <= MAX_REQUEST_BYTES:
        raise argparse.ArgumentTypeError("max-request-bytes must be between 1 and 65536")
    return result


def _idle_timeout(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("request-timeout-seconds must be a number") from error
    if not 0 < result <= 30:
        raise argparse.ArgumentTypeError("request-timeout-seconds must be finite and in (0, 30]")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument(
        "--port", type=_port, default=8080, help="Loopback TCP port; 0 selects a free port"
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=_idle_timeout,
        default=5.0,
        help="Socket idle timeout, not a request wall-time deadline",
    )
    parser.add_argument("--max-request-bytes", type=_request_bytes, default=16384)
    return parser


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig.from_json(args.config.read_text(encoding="utf-8"))
        model, receipt = load_inference_checkpoint(config, args.checkpoint, args.dataset_manifest)
        server = create_inference_server(
            model,
            receipt,
            host="127.0.0.1",
            port=args.port,
            request_timeout_seconds=args.request_timeout_seconds,
            max_request_bytes=args.max_request_bytes,
        )
    except (ValueError, OSError, FloatingPointError) as error:
        parser.error(str(error))
    previous_handlers = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _interrupt)
        host, port = server.server_address[:2]
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ready": True,
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "receipt": receipt.to_dict(),
                    "limits": {
                        "max_request_bytes": args.max_request_bytes,
                        "max_prompt_tokens": MAX_PROMPT_TOKENS,
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "request_idle_timeout_seconds": args.request_timeout_seconds,
                    },
                    "limitations": [
                        "Foreground, loopback-only, unauthenticated local prototype.",
                        "The socket timeout bounds idle waits, not total request duration.",
                    ],
                },
                sort_keys=True,
                allow_nan=False,
            ),
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # shutdown() waits on serve_forever and would deadlock in this same thread.
        server.server_close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
