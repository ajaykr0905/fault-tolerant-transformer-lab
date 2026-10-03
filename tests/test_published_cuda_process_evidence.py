"""CPU consistency checks for historical exports, not GPU execution or authentication."""

import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from fttl.config import ExperimentConfig
from fttl.state import state_digest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/colab-cuda-process-2026-10-04"


@pytest.mark.parametrize(
    "name,export_sha256,revision,verified_at_utc",
    [
        pytest.param(
            "cuda-process-recovery-report.json",
            "ee77c790103d13d145067570ab77d48c03a2428a048e9564c12552013b4ef8cd",
            "8dd38161ea2563eb3fb82ee960c039141b00cae2",
            "2026-10-03T22:21:44.234457+00:00",
            id="original-8dd3816",
        ),
        pytest.param(
            "cuda-process-recovery-report-65d2808.json",
            "f3cd1f8cddbc4fb26e0446d1d704a02bc3b758e7e00fb03a361d8aa890e96ff3",
            "65d280888e0f34aa9dc7107f8197dcef7f48fa34",
            "2026-10-03T23:28:08.040536+00:00",
            id="corrected-65d2808",
        ),
    ],
)
def test_published_cuda_process_report_preserves_export_and_recovery_contract(
    name: str, export_sha256: str, revision: str, verified_at_utc: str
):
    exported = (EVIDENCE / name).read_bytes()
    assert hashlib.sha256(exported).hexdigest() == export_sha256
    report = json.loads(exported)
    dataset = json.loads(
        (ROOT / "artifacts/peps-recovery-v0.2/dataset-manifest.json").read_text(encoding="utf-8")
    )
    assert report["schema"] == "CudaProcessRecoveryReportV1"
    # These identify the historical run, independently of the current checkout.
    assert report["code_revision"] == revision
    assert report["code_fingerprint"] == (
        "0daa13b08c73f126acc052efc406b04a3adc0201762860b194b0cbb9be45c05b"
    )
    config = ExperimentConfig.from_dict(report["config"])
    assert (
        config.fingerprint()
        == report["config_fingerprint"]
        == ("b0f1cd6f83ada78e17c7783c29d3cbcdd5339989f957903fbc76eac9ec0bbcb5")
    )
    assert (
        report["data_fingerprint"]
        == dataset["dataset_fingerprint"]
        == ("8c0076aec77c84fb8f59b5c5772136131805f7837aae6d4d2af928bfc0be6329")
    )
    assert (
        report["tokenizer_fingerprint"]
        == dataset["tokenizer"]["fingerprint"]
        == ("fbe3aa2431f27491ba3bf813c9b9054782af25af4e318c671a631d3166ed80a3")
    )
    assert dataset["source"]["repository"] == "common-pile/python_enhancement_proposals"
    assert dataset["source"]["revision"] == "f932757e3eba16475c893e1418918c77f14a790d"
    assert dataset["counts"]["documents"] == 656
    assert (
        state_digest(
            {
                "schema": "CudaProcessRecoveryContractV1",
                "config": report["config"],
                "dataset": report["data_fingerprint"],
                "tokenizer": report["tokenizer_fingerprint"],
                "code": report["code_fingerprint"],
                "execution": report["execution"],
                "dtype": "float32",
                "optimizer": "AdamW:foreach=False:fused=False",
            }
        )
        == report["run_contract_fingerprint"]
        == ("87f1466e7bc875dc9ad0135537ade14149e36c40d646fb5d74a4f7fde289e8bf")
    )
    assert report["execution"] == {
        "compute_capability": [7, 5],
        "cublas_workspace_config": ":4096:8",
        "cuda_version": "13.0",
        "cudnn_benchmark": False,
        "cudnn_version": 92000,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "device": "cuda:0",
        "execution": "eager",
        "gpu_name": "Tesla T4",
        "initialization_dtype": "torch.float32",
        "optimizer_policy": "AdamW foreach=False fused=False",
        "precision": "float32",
        "tf32": False,
        "torch_version": "2.13.0+cu130",
        "total_memory_bytes": 15637086208,
    }
    assert report["exact_equality"] is True
    assert set(report["equality"]) == {
        "batch_ids",
        "completion_execution_contract",
        "completion_selected_step",
        "completion_worker_pid",
        "cursor",
        "dataset_contract",
        "expected_completed_state",
        "final_checkpoint_model",
        "final_checkpoint_optimizer",
        "final_checkpoint_rng",
        "generation",
        "independent_worker_pids",
        "logits",
        "losses",
        "model",
        "optimizer",
        "parent_cuda_uninitialized",
        "replay_execution_contract",
        "rng",
        "sample_ids",
        "steps",
        "tokens_seen",
    }
    assert all(value is True for value in report["equality"].values())
    pids = report["worker_pids"]
    assert set(pids) == {"control", "interrupted", "replay", "completion"}
    assert all(type(pid) is int and pid > 0 for pid in pids.values())
    assert len(set(pids.values())) == 4
    assert report["process_start_method"] == "spawn"
    assert report["kill_signal"] == "SIGKILL"
    assert report["interrupted_exitcode"] == -9
    assert report["failure_point"] == "after-optimizer"
    assert report["attempted_step"] == 2
    assert report["durable_step_before_kill"] == report["selected_step"] == 1
    assert report["selected_generation"] == 1
    assert (
        report["failed_batch_id"]
        == report["replayed_batch_id"]
        == ("aeecb2e68ff23f573213c749b0462cf275e7f91e52e404408e897d315dc75797")
    )
    assert report["failed_sample_ids"] == report["replayed_sample_ids"]
    assert len(set(report["failed_sample_ids"])) == config.batch_size == 2
    for sample in report["failed_sample_ids"]:
        document, offset = sample.split(":")
        assert document in dataset["splits"]["train"]["document_ids"]
        assert offset == "000000000000"
    assert config.checkpoint_every == 1 and config.model.dropout == 0.2
    assert config.steps == report["steps"] == 6
    tokens_per_step = config.batch_size * config.model.block_size
    assert report["tokens_seen"] == config.steps * tokens_per_step == 384
    assert report["discarded_compute_tokens"] == tokens_per_step == 64
    assert report["durable_committed_steps_lost"] == report["durable_committed_tokens_lost"] == 0
    assert report["final_state_digest"] == (
        "1e79eb12ad7b8ff72b0c87c32649f532a7d277c92ef1877aad1a67bdb67d0cea"
    )
    assert (
        set(report["timings_seconds"])
        == set(report["timing_boundaries"])
        == {
            "checkpoint_selection_load",
            "completion_spawn_to_exit",
            "control_spawn_to_exit",
            "kill_request_to_exit",
            "resume_spawn_to_replayed_commit_receipt",
        }
    )
    assert all(
        type(seconds) is float and math.isfinite(seconds) and seconds > 0
        for seconds in report["timings_seconds"].values()
    )
    assert (
        "CPU/CUDA RNG restoration and CUDA synchronization"
        in (report["timing_boundaries"]["checkpoint_selection_load"])
    )
    assert (
        "replayed durable save and CUDA synchronization"
        in (report["timing_boundaries"]["resume_spawn_to_replayed_commit_receipt"])
    )
    assert report["verified_at_utc"] == verified_at_utc
    assert datetime.fromisoformat(report["verified_at_utc"]).utcoffset() == timedelta(0)
    limitations = " ".join(report["limitations"])
    for boundary in (
        "parent-paused after-optimizer boundary",
        "not arbitrary asynchronous kill times",
        "not power loss or device failure",
        "No cross-device, mixed-precision, distributed-scale",
        "not benchmarks or service-level objectives",
        "hashes are not authentication",
    ):
        assert boundary in limitations
    for private_marker in (b"/Users/", b"/private/", b"/content/", b"@gmail.com"):
        assert private_marker not in exported
