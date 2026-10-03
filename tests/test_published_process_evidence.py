"""Keep the published CPU proof's declared scope and provenance internally consistent."""

import json
from pathlib import Path

import pytest

from fttl.train import FAILURE_POINTS

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "peps-process-kill-2026-10-03"
CODE_REVISION = "8ecf0ea9c7174a57ba3dd9bc36d04b05854d3a48"
FINAL_STATE = "4ff28fd273874c94dcff5494bde84bf629ba45ad569194a065b2e690bcd5243e"


@pytest.mark.parametrize("point", FAILURE_POINTS)
def test_published_public_data_process_proof(point: str):
    path = EVIDENCE / point / "process-recovery-report.json"
    raw = path.read_text(encoding="utf-8")
    report = json.loads(raw)
    dataset = json.loads(
        (ROOT / "artifacts" / "peps-recovery-v0.2" / "dataset-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema"] == "ProcessRecoveryReportV1"
    assert report["code_revision"] == CODE_REVISION
    assert report["failure_point"] == point
    assert report["data_kind"] == "prepared-dataset"
    assert report["dataset_fingerprint"] == dataset["dataset_fingerprint"]
    assert report["tokenizer_fingerprint"] == dataset["tokenizer"]["fingerprint"]
    assert report["process_start_method"] == "spawn"
    assert report["kill_signal"] == "SIGKILL"
    assert report["interrupted_exitcode"] == -9
    assert len(set(report["worker_pids"].values())) == 4
    assert report["attempted_step"] == 2
    assert report["durable_step_before_kill"] == report["selected_step"] == 1
    assert report["selected_generation"] == 1
    assert report["selected_tokens_seen"] == report["replayed_tokens"] == 64
    assert report["failed_batch_id"] == report["replayed_batch_id"]
    assert report["failed_sample_ids"] == report["replayed_sample_ids"]
    assert report["completed_steps"] == 6
    assert report["tokens_seen"] == 384
    assert report["durable_committed_steps_lost"] == report["durable_committed_tokens_lost"] == 0
    assert report["exact_equality"] is True
    assert len(report["equality"]) == 16
    assert all(value is True for value in report["equality"].values())
    assert report["final_state_digest"] == FINAL_STATE
    assert report["control_run_fingerprint"] == report["recovered_run_fingerprint"]
    assert report["environment"]["device"] == "cpu"
    assert report["environment"]["worker_torch_threads"] == 1
    assert report["staging_cleanup_complete"] is True
    assert bool(report["abandoned_staging_files"]) == (point == "during-checkpoint-write")
    assert all(not Path(item).is_absolute() for item in report["abandoned_staging_files"])
    assert all(value >= 0 for value in report["timings_seconds"].values())
    assert set(report["timings_seconds"]) == set(report["timing_boundaries"])
    assert "/Users/" not in raw and "/private/" not in raw
    assert any("power-loss" in item for item in report["limitations"])
