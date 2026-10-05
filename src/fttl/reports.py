"""Publish completed local JSON evidence without exposing a partial report."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def publish_json(output: Path, value: object) -> None:
    """Link a flushed, same-directory file once; never replace another run's report.

    Requires local filesystem hard-link support. This is not a power-loss proof.
    A process death before cleanup can leave a private temporary, not partial JSON
    under the report's public name.
    """
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.tmp-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
