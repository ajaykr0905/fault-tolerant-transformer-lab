import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fttl.demo import FIXTURE, FIXTURE_SHA256


def test_default_demo_fixture_is_installed_with_package():
    import fttl

    assert FIXTURE.is_absolute()
    assert FIXTURE.is_relative_to(Path(fttl.__file__).parent)
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    assert len(records) == 6
    assert all(item["metadata"]["license"] == "CC0-1.0" for item in records)


@pytest.mark.skipif(os.name != "posix", reason="demo requires POSIX SIGKILL")
def test_installed_demo_runs_outside_checkout_offline(tmp_path):
    output = tmp_path / "proof"
    result = subprocess.run(
        [
            str(Path(sys.executable).with_name("fttl-demo")),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS: 16/16" in result.stdout
    report = json.loads((output / "run/process-recovery-report.json").read_text())
    assert report["interrupted_exitcode"] == -9 and report["exact_equality"]
