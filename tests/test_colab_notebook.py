"""Structural checks only: these tests do not execute or attest to a Colab GPU."""

import ast
import hashlib
import json
import math
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "colab_cuda_recovery.ipynb"


@pytest.fixture
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code(notebook):
    return "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_notebook_is_unexecuted_python_gpu_document(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["metadata"]["kernelspec"]["language"] == "python"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    ast.parse(code(notebook))


def test_bootstrap_is_public_pinned_clean_and_locked(notebook):
    source = code(notebook)
    pins = re.findall(r'IMPLEMENTATION_REVISION = "([0-9a-f]{40})"', source)
    assert len(pins) == 1
    assert "https://github.com/ajaykr0905/fault-tolerant-transformer-lab.git" in source
    assert '"checkout", "--detach", IMPLEMENTATION_REVISION' in source
    assert '"rev-parse", "HEAD"' in source
    assert '"status", "--porcelain"' in source
    assert '"uv==0.9.7"' in source
    assert '"--frozen", "--extra", "test", "--python", "3.12"' in source
    assert 'tempfile.mkdtemp(prefix="fttl-", dir="/content")' in source
    assert 'CUBLAS_WORKSPACE_CONFIG=":4096:8"' in source


def test_subprocesses_are_checked_bounded_and_never_shell_commands(notebook):
    tree = ast.parse(code(notebook))
    runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(runs) == 1  # All commands route through the checked helper.
    keywords = {item.arg: item.value for item in runs[0].keywords}
    assert isinstance(keywords["check"], ast.Constant) and keywords["check"].value is True
    assert keywords["capture_output"].value is True
    assert keywords["text"].value is True
    assert "timeout" in keywords
    assert "shell" not in keywords
    source = code(notebook)
    assert "result.stdout[-8000:]" in source
    assert "result.stderr[-2000:]" in source


def test_gpu_probe_precedes_tests_and_report_requires_real_equality(notebook):
    source = code(notebook)
    assert source.index('"nvidia-smi"') < source.index('"uv==0.9.7"')
    assert source.index("prepare_cuda_execution") < source.index('"pytest"')
    assert '"tests/test_cuda_runtime.py"' in source
    assert '"tests/test_cuda_recovery.py"' in source
    assert "test_real_cuda_rng_checkpoint_is_weights_only_safe_and_cpu_loadable" in source
    assert 'assert "skipped" not in test_output' in source
    assert '"--interruption-step", "2", "--max-eval-tokens", "4096"' in " ".join(source.split())
    assert 'report["execution"]["device"] == "cuda:0"' in source
    assert 'all(report["equality"].values())' in source
    assert 'report["code_revision"] == IMPLEMENTATION_REVISION' in source
    assert 'report["steps"] == 6 and report["tokens_seen"] == 384' in source
    assert 'report["evaluation_device"] == "cpu"' in source
    assert "files.download(str(REPORT_PATH))" in source


@pytest.mark.parametrize(
    "forbidden",
    [
        "/Users/",
        "@gmail.com",
        "drive.mount",
        "getpass",
        "auth.authenticate_user",
        "ngrok",
        "ssh",
        "--force",
    ],
)
def test_notebook_does_not_embed_private_access_or_workarounds(notebook, forbidden):
    assert forbidden not in code(notebook)


def test_measured_report_preserves_export_and_declared_evidence_boundary():
    path = NOTEBOOK.parents[1] / "artifacts/colab-cuda-2026-10-03/cuda-recovery-report.json"
    exported = path.read_bytes()
    assert hashlib.sha256(exported).hexdigest() == (
        "13dca14329edf830df3da6cafdeb69c0834ec7cbc71f1d4daa2daf3330324ffe"
    )
    report = json.loads(exported)
    assert report["schema"] == "CudaRecoveryReportV1"
    assert report["code_revision"] == "4ffaf1da88f9092078d61ed3c801597eefcd0809"
    assert report["execution"]["device"] == "cuda:0"
    assert report["execution"]["gpu_name"] == "Tesla T4"
    assert report["execution"]["precision"] == "float32"
    assert report["exact_equality"] is True
    assert len(report["equality"]) == 11 and all(
        value is True for value in report["equality"].values()
    )
    assert report["steps"] == 6 and report["tokens_seen"] == 384
    assert report["interruption_step"] == report["selected_generation"] == 2
    assert report["evaluation_device"] == "cpu"
    assert report["evaluation"]["evaluated_target_tokens"] == 4096
    assert report["evaluation"]["complete_split"] is False
    assert report["evaluation"]["model_state_digest"] == report["digests"]["model"]
    assert report["memory"]["experiment_peak_allocated_bytes"] > 0
    assert all(math.isfinite(loss) for loss in report["losses"])
    assert all(
        math.isfinite(report[key]) and report[key] > 0
        for key in ("control_seconds", "recovered_seconds", "checkpoint_load_seconds")
    )
    assert "not process death or SIGKILL" in report["limitations"][0]
    assert (
        b"/Users/" not in exported
        and b"/content/" not in exported
        and b"@gmail.com" not in exported
    )
