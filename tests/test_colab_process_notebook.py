"""Notebook contracts only; these tests are not measured GPU evidence."""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/colab_cuda_process_recovery.ipynb"


@pytest.fixture
def notebook_code():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["accelerator"] == "GPU"
    sources = []
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            source = "".join(cell["source"])
            ast.parse(source)
            sources.append(source)
    # Normalize equivalent notebook formatting before inspecting code contracts.
    return ast.unparse(ast.parse("\n".join(sources)))


def test_public_source_is_immutable_clean_and_dependency_locked(notebook_code):
    pins = re.findall(r"IMPLEMENTATION_REVISION = '([0-9a-f]{40})'", notebook_code)
    assert len(pins) == 1 and pins[0] != "0" * 40
    assert "https://github.com/ajaykr0905/fault-tolerant-transformer-lab.git" in notebook_code
    assert "'checkout', '--detach', IMPLEMENTATION_REVISION" in notebook_code
    assert "'status', '--porcelain'" in notebook_code
    assert "'uv==0.9.7'" in notebook_code
    assert "'--frozen', '--extra', 'test', '--python', '3.12'" in notebook_code
    assert "CUBLAS_WORKSPACE_CONFIG=':4096:8'" in notebook_code
    assert notebook_code.index("'nvidia-smi'") < notebook_code.index("'uv==0.9.7'")


def test_commands_are_checked_bounded_and_never_shell_executed(notebook_code):
    tree = ast.parse(notebook_code)
    runs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
        and node.func.attr == "run"
    ]
    assert len(runs) == 1
    keywords = {item.arg: item.value for item in runs[0].keywords}
    assert keywords["check"].value is True and "timeout" in keywords
    assert "shell" not in keywords


def test_gpu_regression_and_process_death_claims_are_gated(notebook_code):
    assert "'tests/test_cuda_process_recovery.py'" in notebook_code
    assert "assert 'skipped' not in output" in notebook_code
    assert "'fttl.cuda_process_recovery'" in notebook_code
    assert "report['code_revision'] == IMPLEMENTATION_REVISION" in notebook_code
    assert "report['interrupted_exitcode'] == -9" in notebook_code
    assert "len(set(report['worker_pids'].values())) == 4" in notebook_code
    assert "all(report['equality'].values())" in notebook_code
    assert "report['steps'] == 6 and report['tokens_seen'] == 384" in notebook_code
    assert "hashlib.sha256(REPORT_PATH.read_bytes())" in notebook_code
    assert "files.download(str(REPORT_PATH))" in notebook_code


@pytest.mark.parametrize(
    "forbidden", ["/Users/", "@gmail.com", "drive.mount", "getpass", "ngrok", "ssh", "--force"]
)
def test_notebook_excludes_private_access_and_quota_workarounds(notebook_code, forbidden):
    assert forbidden not in notebook_code
